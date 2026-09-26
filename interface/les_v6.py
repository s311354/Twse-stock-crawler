"""Standalone offline LES v6 JSON runner. No network, email or orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from application.strategy_execution import StrategyExecutionService
from domain.strategies.low_entry_score_v6 import V6_OUTPUT_COLUMNS, attach_low_entry_v6_context, gate_funnel


def analyze_request(payload: dict) -> dict:
    """One security or a same-session universe; input schema is documented in docs/."""
    requests = payload.get("securities", [payload])
    histories = []
    for request in requests:
        benchmark = request.get("taiex_bars")
        histories.append(attach_low_entry_v6_context(
            pd.DataFrame(request["bars"]),
            taiex=pd.DataFrame(benchmark) if benchmark is not None else None,
            **request["context"],
        ))
    service = StrategyExecutionService()
    result = service.evaluate("low_entry_score_v6", V6_OUTPUT_COLUMNS, service.prepare(histories))
    reports = [r for r in result["LES v6 Report"] if isinstance(r, dict)]
    return {"model": "low_entry_score_v6", "reports": reports, "funnel": gate_funnel(reports)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSON: bars, optional taiex_bars, context; or securities[]")
    args = parser.parse_args(argv)
    try:
        result = analyze_request(json.loads(args.input.read_text(encoding="utf-8")))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
