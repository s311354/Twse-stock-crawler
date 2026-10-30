"""Bounded synthetic feature-storage comparison; no workflow or market I/O.

The old tuple-backed FeatureFrame path is retained as a benchmark reference,
not a second indicator engine. Feature math is identical and precomputed.
Timings describe storage/materialization only, not end-to-end daily speedup.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

from domain.features import FEATURE_PIPELINE_VERSION, FeatureFrame, SharedFeaturePipeline, _source_fingerprint
from domain.les_v7.features import _latest_resilience


def _synthetic_history(rows):
    close = 100. + np.arange(rows) * .05 + np.sin(np.arange(rows) / 5.)
    volume = 100000. + np.arange(rows) % 7 * 1000.
    return pd.DataFrame(dict(Date=pd.bdate_range("2024-01-01", periods=rows),
        Open=close - .1, High=close + 1., Low=close - 1., Close=close,
        Volume=volume, Amount=close * volume, Trades=np.full(rows, 500),
        Bid=close - .01, Ask=close + .01, PE=np.full(rows, 20.),
        **{"TAIEX Close": 10000. + np.arange(rows) * .5}))


def _legacy_store(frame, source, config):
    return FeatureFrame(config=config, pipeline_version=FEATURE_PIPELINE_VERSION,
        source_fingerprint=_source_fingerprint(source), _index=tuple(frame.index.tolist()),
        _columns=tuple(str(column) for column in frame),
        _values=tuple(tuple(frame[column].tolist()) for column in frame))


def _measure(build, count):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    stored = [build() for _ in range(count)]
    storage_seconds = time.perf_counter() - started
    retained_bytes, storage_peak_bytes = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    for item in stored:
        output = item.to_pandas()
        del output
    materialization_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    encoded_bytes = len(pickle.dumps(stored, protocol=pickle.HIGHEST_PROTOCOL))
    return dict(storage_seconds=storage_seconds, materialization_seconds=materialization_seconds,
        traced_retained_storage_bytes=retained_bytes, traced_storage_peak_bytes=storage_peak_bytes,
        traced_storage_and_materialization_peak_bytes=peak_bytes, pickle_bytes=encoded_bytes)


def _legacy_resilience(market, close, rs20):
    result = np.nan
    for i in range(20, len(market)):
        m, stock = market.iloc[i - 20:i + 1], close.iloc[i - 20:i + 1]
        result = np.nan
        if m.notna().all():
            peak = int(np.argmax(m.to_numpy()))
            result = (np.log(stock.iloc[-1] / stock.iloc[peak]) - np.log(m.iloc[-1] / m.iloc[peak])
                      if peak < 20 and m.iloc[-1] < m.iloc[peak] else rs20.iloc[i])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=8)
    parser.add_argument("--rows", type=int, default=700)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.stocks <= 16 or not 21 <= args.rows <= 700 or not 1 <= args.rounds <= 5:
        parser.error("Use 1-16 synthetic stocks, 21-700 rows, and 1-5 rounds")
    source = _synthetic_history(args.rows)
    pipeline = SharedFeaturePipeline()
    frame = pipeline._calculate(source)
    legacy = _legacy_store(frame, source, pipeline.config)
    compact = FeatureFrame.from_pandas(frame, config=pipeline.config, source_data=source)
    pd.testing.assert_frame_equal(legacy.to_pandas(), compact.to_pandas(), check_exact=True)
    for name in frame:
        pd.testing.assert_series_equal(legacy.series(name), compact.series(name), check_exact=True)
    market, close = source["TAIEX Close"].copy(), source["Close"]
    market.iloc[-15:] -= np.arange(15) * 10.
    rs20 = close / close.shift(20) - market / market.shift(20)
    assert _latest_resilience(market, close, rs20) == _legacy_resilience(market, close, rs20)
    measurements = []
    for iteration in range(args.rounds):
        results = {}
        for name, build in (("legacy", lambda: _legacy_store(frame, source, pipeline.config)),
                            ("compact", lambda: FeatureFrame.from_pandas(frame, config=pipeline.config, source_data=source))):
            results[name] = _measure(build, args.stocks)
        for name, calculate in (("legacy", _legacy_resilience), ("compact", _latest_resilience)):
            started = time.perf_counter()
            for _ in range(args.stocks):
                calculate(market, close, rs20)
            results[name]["resilience_seconds"] = time.perf_counter() - started
        measurements.append(dict(iteration=iteration, **results))
    print(json.dumps(dict(schema="numeric-storage-benchmark/v1", data_kind="SYNTHETIC_ONLY",
        scope="STORAGE_AND_LATEST_RESILIENCE_NOT_END_TO_END", exact_feature_parity=True,
        stocks=args.stocks, rows=args.rows, columns=len(frame.columns),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                       for name in ("domain/features.py", "domain/les_v7/features.py", "examples/benchmark_numeric_storage.py")},
        memory_policy="TRACEMALLOC_ALLOCATIONS_NOT_TOTAL_PROCESS_RSS",
        measurements=measurements), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
