"""Reuse a pre-refresh screen only for the exact same non-financial inputs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pandas as pd

from application.daily_performance import count_work, measure_stage
from domain.strategies.low_entry_v6_watch import LeadershipWatchConfig, screen_leadership
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.numeric_history import metadata_free_frame, SCREEN_SOURCE_ROWS_ATTR


def watch_input_key(histories, *, session_date, as_of, config):
    digest = hashlib.sha256()
    digest.update(json.dumps({"policy": "watch-cache/v1", "config": asdict(config),
                             "session": str(pd.to_datetime(session_date, errors="coerce")),
                             "cutoff": str(pd.to_datetime(as_of, utc=True, errors="coerce"))},
                            sort_keys=True).encode())
    columns = ("Date", "Open", "High", "Low", "Close", "Volume", "Amount", "TAIEX Close", "Complete")
    for history in histories:
        metadata = {"identity": history.attrs.get("les_v6_daily_identity", {}),
                    "scale": history.attrs.get(PRICE_SCALE_ATTR, 1),
                    "restriction": history.attrs.get("trading_restriction", {}),
                    "historical": bool(history.attrs.get("replay_as_of")),
                    "rows": history.attrs.get(SCREEN_SOURCE_ROWS_ATTR, len(history)),
                    "columns": [name for name in columns if name in history]}
        digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
        # The watch uses only 61 bars, but absence of required columns and exact
        # date/order/quote/complete values are also bound to this signature.
        digest.update(pd.util.hash_pandas_object(metadata_free_frame(history).tail(61).reindex(columns=columns), index=True).to_numpy().tobytes())
    return digest.hexdigest()


def watch_result_key(watch):
    return hashlib.sha256(json.dumps(watch, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def screen_once(histories, *, session_date, as_of, config=None, screener=screen_leadership, reuse=True):
    config = config or LeadershipWatchConfig()
    with measure_stage("leadership_screen"):
        key = watch_input_key(histories, session_date=session_date, as_of=as_of, config=config)
        contexts = [history.attrs.get("les_v6_research_context", {}) for history in histories]
        if reuse and histories and all(context.get("watch_input_sha256") == key and isinstance(context.get("watch"), dict)
                             and context.get("watch_result_sha256") == watch_result_key(context["watch"])
                             for context in contexts):
            count_work("leadership_cache_hits")
            return [deepcopy(context["watch"]) for context in contexts], key
        count_work("leadership_cache_misses")
        count_work("leadership_screen_calls")
        return screener(histories, session_date=session_date, as_of=as_of, config=config), key
