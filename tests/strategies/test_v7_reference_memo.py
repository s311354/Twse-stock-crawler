"""Bounded native optimization contracts; synthetic engineering, no model freeze."""
from copy import deepcopy
from dataclasses import asdict
from weakref import ref

import pandas as pd
import pytest

from application.les_v7_dataset import build_training_dataset
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.reference_memo import NativeReferenceMemo, memoize_reference
from domain.les_v7 import training
from tests.test_v7_dataset_builder import builder_inputs


def prepared(number=1):
    return dict(features=dict(close=float(number)),
                feature_audit=dict(profile="SYNTHETIC_TEST_ONLY", status="COMPUTED",
                                   fractal=dict(dfa512=dict(value=.5, window=512, grid=(8, 16)))),
                context="NO_CLEAR_CONTEXT", context_matches=(), plan=None, vector=None,
                feature_gap="REFERENCE_PLAN_NOT_READY", evidence_id=identity(number))


def test_deep_copies_protect_store_and_every_read_and_scope_cleanup():
    original = prepared()
    calls = []
    def calculate():
        calls.append(True)
        return original
    with NativeReferenceMemo(max_entries=2) as memo:
        first = memoize_reference("key", calculate)
        original["features"]["close"] = 800.
        first["features"]["close"] = 900.
        first["feature_audit"]["fractal"]["dfa512"]["value"] = 9.
        second = memoize_reference("key", calculate)
        assert second["features"]["close"] == 1.
        assert second["feature_audit"]["fractal"]["dfa512"]["value"] == .5
        second["feature_audit"]["fractal"]["dfa512"]["value"] = 10.
        assert memoize_reference("key", calculate)["feature_audit"]["fractal"]["dfa512"]["value"] == .5
        assert len(calls) == 1 and memo.stats["hits"] == 2
        assert memo.stats["current_entries"] == 1
    assert memo.stats["current_entries"] == 0
    assert memoize_reference("key", calculate) is original  # No process-global memo.
    assert len(calls) == 2


def test_lru_bound_and_nested_job_contexts_are_isolated():
    with NativeReferenceMemo(max_entries=2) as outer:
        memoize_reference("a", lambda: prepared(1))
        memoize_reference("b", lambda: prepared(2))
        memoize_reference("a", lambda: pytest.fail("LRU read must reuse a"))
        memoize_reference("c", lambda: prepared(3))
        assert outer.stats["evictions"] == 1 and outer.stats["current_entries"] == 2
        assert memoize_reference("b", lambda: prepared(20))["features"]["close"] == 20
        with NativeReferenceMemo(max_entries=1) as inner:
            assert memoize_reference("b", lambda: prepared(200))["features"]["close"] == 200
            assert inner.stats["hits"] == 0 and outer.stats["current_entries"] == 2
        assert inner.stats["current_entries"] == 0
        assert memoize_reference("b", lambda: pytest.fail("Nested scope must restore outer"))["features"]["close"] == 20
    with NativeReferenceMemo(max_entries=1) as next_job:
        assert memoize_reference("b", lambda: prepared(300))["features"]["close"] == 300
        assert next_job.stats["hits"] == 0


def test_failures_none_and_stop_close_without_caching_or_masking():
    class Stopped(BaseException):
        pass
    with NativeReferenceMemo(max_entries=1) as memo:
        for _ in range(2):
            with pytest.raises(ValueError, match="native failure"):
                memoize_reference("bad", lambda: (_ for _ in ()).throw(ValueError("native failure")))
            assert memoize_reference("none", lambda: None) is None
        assert memo.stats["stores"] == 0 and memo.stats["current_entries"] == 0
        assert memo.stats["misses"] == 4
    with pytest.raises(Stopped):
        with NativeReferenceMemo(max_entries=1) as stopped:
            memoize_reference("a", prepared)
            raise Stopped()
    assert stopped.stats["current_entries"] == 0
    with pytest.raises(ValueError, match="exit failed"):
        with NativeReferenceMemo(max_entries=1) as failed:
            memoize_reference("a", prepared)
            raise ValueError("exit failed")
    assert failed.stats["current_entries"] == 0
    with NativeReferenceMemo(max_entries=1) as clean:
        memoize_reference("a", prepared)
        assert clean.stats["hits"] == 0


def test_no_history_factory_or_full_report_retention_and_unsupported_shapes_remain_native():
    source = pd.DataFrame(dict(Date=["2026-01-05"], Close=[1.]))
    source_reference = ref(source)
    with NativeReferenceMemo(max_entries=2) as memo:
        def calculate(history=source):
            return prepared(float(history["Close"].iloc[0]))
        result = memoize_reference("a", calculate)
        assert result == prepared(1.)
        del source, calculate
        assert source_reference() is None
        full = dict(prepared(), reports={"original": "DO_NOT_RETAIN"})
        assert memoize_reference("full", lambda: full) is full
        audit_source = prepared()
        audit_source["feature_audit"]["source"] = pd.DataFrame(dict(Close=[1.]))
        assert memoize_reference("audit-source", lambda: audit_source) is audit_source
        assert memo.stats["stores"] == 1 and memo.stats["current_entries"] == 1
        assert set(memo._entries["a"]) == {"features", "feature_audit", "context", "context_matches",
                                           "plan", "vector", "feature_gap", "evidence_id"}


@pytest.mark.parametrize("bound", [0, -1, 4097, True, 1.5])
def test_invalid_bounds_are_not_silently_expanded(bound):
    with pytest.raises(ValueError, match="REFERENCE_MEMO_ENTRY_BOUND_REQUIRED"):
        NativeReferenceMemo(max_entries=bound)


def test_full_native_builder_labels_rows_diagnostics_and_hashes_equal_with_memo(monkeypatch):
    source, config = builder_inputs(end="2026-01-06")
    original = deepcopy((source, config))
    calculate = training.compute_features
    calls = []
    def counted(bars):
        calls.append(True)
        return calculate(bars)
    monkeypatch.setattr(training, "compute_features", counted)
    selection = dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"])
    baseline = build_training_dataset(source, config)
    baseline_rows = training.prepare_events(baseline["dataset"], selection)
    baseline_calls = len(calls)
    calls.clear()
    with NativeReferenceMemo(max_entries=config["max_events"]) as memo:
        cached = build_training_dataset(source, config)
        cached_rows = training.prepare_events(cached["dataset"], selection)
        assert cached == baseline  # Full dataset, labels, audits and hashes.
        assert cached_rows == baseline_rows
        assert memo.stats["hits"] > 0
        assert len(calls) < baseline_calls
        assert memo.stats["current_entries"] <= config["max_events"]
    assert (source, config) == original and memo.stats["current_entries"] == 0


def test_exact_reference_identity_invalidates_selected_bars_sources_config_symbol_market_and_cutoff(monkeypatch):
    source, build = builder_inputs()
    history = pd.DataFrame(source["payload"]["histories"][0]["bars"])
    config = build["seed"]
    cutoff = "2026-01-05T13:30:00+08:00"
    external = ["SYNTHETIC_SOURCE_NOT_OFFICIAL"]
    calculate = training.compute_features
    calls = []
    def counted(bars):
        calls.append(True)
        return calculate(bars)
    monkeypatch.setattr(training, "compute_features", counted)
    arguments = ("3014", "TWSE", history, cutoff, config, external)
    with NativeReferenceMemo(max_entries=16) as memo:
        base = training.prepare_reference(*arguments)
        assert training.prepare_reference(*arguments) == base and len(calls) == 1
        mutated = history.copy(deep=True)
        row = mutated.index[mutated["Date"] == "2026-01-05"][0]
        mutated.loc[row, "High"] += .1
        changed_bars = training.prepare_reference("3014", "TWSE", mutated, cutoff, config, external)
        assert changed_bars["evidence_id"] != base["evidence_id"]
        changed_source = training.prepare_reference("3014", "TWSE", history, cutoff, config, ["DIFFERENT_SYNTHETIC_SOURCE"])
        assert changed_source["evidence_id"] != base["evidence_id"]
        altered_config = deepcopy(config)
        altered_config["reference_policy"]["version"] += ":REVISION"
        changed_policy = training.prepare_reference("3014", "TWSE", history, cutoff, altered_config, external)
        assert changed_policy["evidence_id"] == base["evidence_id"]
        assert changed_policy["plan"].policy_version != base["plan"].policy_version
        changed_symbol = training.prepare_reference("5471", "TWSE", history, cutoff, config, external)
        changed_market = training.prepare_reference("3014", "TPEX", history, cutoff, config, external)
        changed_cutoff = training.prepare_reference("3014", "TWSE", history, "2026-01-05T14:30:00+08:00", config, external)
        assert all(value["evidence_id"] != base["evidence_id"] for value in (changed_symbol, changed_market, changed_cutoff))
        assert len(calls) == 7 and memo.stats["misses"] == 7 and memo.stats["hits"] == 1
        # A changed invalid raw bar still fails the native selector before reuse.
        mutated.loc[row, "High"] = 1.
        with pytest.raises(ValueError, match="INVALID_OHLCV"):
            training.prepare_reference("3014", "TWSE", mutated, cutoff, config, external)
        assert len(calls) == 7
        base["features"]["close"] = 999.
        assert training.prepare_reference(*arguments)["features"]["close"] != 999.



def _assert_optional_execution_profile_reference_cache(monkeypatch, provider):
    """Actual estimators/audits must stay exact and caller-owned after reuse."""
    source, build = builder_inputs()
    history = pd.DataFrame(source["payload"]["histories"][0]["bars"])
    config = deepcopy(build["seed"])
    base = list(config["feature_names"])
    selected = list(provider.FEATURE_NAMES)
    config.update(
        feature_names=base + selected,
        feature_bounds=config["feature_bounds"] + [list(bounds) for bounds in provider.FEATURE_BOUNDS],
        execution_feature_profile=provider.profile(),
        fill_only_feature_extension=dict(base_feature_names=base,
            profile_id=identity(provider.profile()), selected_features=selected))
    arguments = ("3014", "TWSE", history, "2026-01-05T13:30:00+08:00", config,
                 ["SYNTHETIC_SOURCE_NOT_OFFICIAL"])

    def encoded(result):
        detached = dict(result)
        detached["plan"] = asdict(detached["plan"]) if detached["plan"] is not None else None
        return canonical(detached)

    baseline = training.prepare_reference(*arguments)
    assert baseline["feature_audit"]["execution_feature_audit"]["status"] == "COMPUTED"
    assert baseline["vector"] is not None
    expected = encoded(baseline)
    calculate = training.compute_features
    calls = []

    def counted(bars):
        calls.append(True)
        return calculate(bars)

    monkeypatch.setattr(training, "compute_features", counted)
    with NativeReferenceMemo(max_entries=2) as memo:
        first = training.prepare_reference(*arguments)
        second = training.prepare_reference(*arguments)
        assert encoded(first) == encoded(second) == expected
        assert memo.stats["hits"] == 1 and memo.stats["stores"] == 1 and len(calls) == 1
        first["features"]["close"] = 999.
        first["vector"][0] = 999.
        first["feature_audit"]["execution_feature_audit"]["status"] = "MUTATED_CALLER_COPY"
        assert encoded(second) == expected
        third = training.prepare_reference(*arguments)
        assert encoded(third) == expected and len(calls) == 1 and memo.stats["hits"] == 2
        assert memo.stats["current_entries"] == 1
    assert memo.stats["current_entries"] == 0


def test_actual_r1_execution_audit_reference_is_cached_exactly_and_owned(monkeypatch):
    from domain.les_v7 import execution_features
    _assert_optional_execution_profile_reference_cache(monkeypatch, execution_features)


def test_actual_r2_execution_audit_reference_is_cached_exactly_and_owned(monkeypatch):
    from domain.les_v7 import execution_atom_features
    _assert_optional_execution_profile_reference_cache(monkeypatch, execution_atom_features)
