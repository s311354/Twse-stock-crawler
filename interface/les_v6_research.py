"""Offline recovery/base-breakout research. Never sends mail or places orders."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from domain.strategy import StrategyRegistry
from domain.strategies.low_entry_score_v6 import attach_low_entry_v6_context
from domain.strategies.recovery_breakout_research import (
    RESEARCH_COLUMN, RecoveryBreakoutConfig, RecoveryBreakoutResearchStrategy,
)


def analyze_request(payload: dict, *, history: bool = False) -> dict:
    """Use the existing offline v6 bars/context schema; opt-in research only."""
    registry = StrategyRegistry()
    plugin = registry.register(RecoveryBreakoutResearchStrategy(
        RecoveryBreakoutConfig(**payload.get("research_parameters", {}))))
    reports = []
    for request in payload.get("securities", [payload]):
        benchmark = request.get("taiex_bars")
        data = attach_low_entry_v6_context(
            pd.DataFrame(request["bars"]),
            taiex=pd.DataFrame(benchmark) if benchmark is not None else None,
            **request["context"],
        )
        result = plugin.run(data) if history else plugin.run_latest(data)
        reports.extend(result[RESEARCH_COLUMN].tolist())
    return {"model": plugin.name, "research_only": True, "executable": False,
            "evaluation_mode": "CLOSE_TIME_REPLAY" if history else "AS_OF_SNAPSHOT",
            "reports": reports,
            "summary": dict(Counter(row["research_signal"] for row in reports))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSON: bars/context or securities[]; same input as interface.les_v6")
    parser.add_argument("--history", action="store_true", help="Replay each completed close with point-in-time filings")
    args = parser.parse_args(argv)
    try:
        result = analyze_request(json.loads(args.input.read_text(encoding="utf-8")), history=args.history)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
