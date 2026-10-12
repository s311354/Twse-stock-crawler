"""Offline LES v6 setup, score and outcome research. No mail or orders."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from application.industry_benchmark import prepare_industry_context
from domain.strategy import StrategyRegistry
from domain.strategies.low_entry_score_v6 import attach_low_entry_v6_context
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, PARAMETER_VERSION
from domain.strategies.low_entry_v6_research import score_audit, feature_ablation_audit
from domain.strategies.low_entry_v6_outcomes import outcome_audit
from application.les_weight_estimation import estimate_weights
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
            **prepare_industry_context(request["context"]),
        )
        result = plugin.run(data) if history else plugin.run_latest(data)
        reports.extend(result[RESEARCH_COLUMN].tolist())
    return {"model": plugin.name, "research_only": True, "executable": False,
            "evaluation_mode": "CLOSE_TIME_REPLAY" if history else "AS_OF_SNAPSHOT",
            "reports": reports,
            "summary": dict(Counter(row["research_signal"] for row in reports))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSON: bars/context, securities[], or an explicit closed-outcome packet")
    parser.add_argument("--history", action="store_true", help="Replay each completed close with point-in-time filings")
    audits = parser.add_mutually_exclusive_group()
    audits.add_argument("--score-audit", action="store_true", help="Read-only v6 contribution ablation and holdout diagnostics; no performance claims")
    audits.add_argument("--feature-audit", action="store_true", help="Native score-channel removal and additive/de-duplicated comparison; not profitable-signal validation")
    audits.add_argument("--outcome-audit", action="store_true", help="Cost-aware closed-episode / purged holdout research; never grants BUY or Kelly authority")
    audits.add_argument("--estimate-weights", action="store_true", help="Offline hierarchical convex fit from frozen Hybrid reports and assessed closed outcomes")
    parser.add_argument("--validation-start", help="YYYY-MM-DD chronological holdout boundary, required for score/feature audits")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if args.outcome_audit or args.estimate_weights:
            if args.history or args.validation_start:
                raise ValueError("OUTCOME_AUDIT_USES_EXPLICIT_PACKET_CUTOFF_AND_SPLIT")
            result = estimate_weights(payload) if args.estimate_weights else outcome_audit(payload)
        elif args.score_audit or args.feature_audit:
            if not args.validation_start:
                raise ValueError("SCORE_AUDIT_REQUIRES_VALIDATION_START")
            result = analyze_scores(payload, validation_start=args.validation_start, feature_audit=args.feature_audit)
        else:
            if args.validation_start:
                raise ValueError("VALIDATION_START_REQUIRES_SCORE_AUDIT")
            result = analyze_request(payload, history=args.history)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


def analyze_scores(payload: dict, *, validation_start: str, feature_audit: bool = False) -> dict:
    """Same native plugin; input digest pins this offline source packet.

    This is score diagnostics, not strict-replay archive/account parity. For
    portfolio/performance research use ResearchJob and a verified dataset.
    """
    import hashlib
    reports = []
    for request in payload.get("securities", [payload]):
        benchmark = request.get("taiex_bars")
        data = attach_low_entry_v6_context(pd.DataFrame(request["bars"]),
            taiex=pd.DataFrame(benchmark) if benchmark is not None else None,
            **prepare_industry_context(request["context"]))
        reports.extend(LowEntryScoreV6Strategy().run(data)["LES v6 Report"].tolist())
    result = (feature_ablation_audit(reports, validation_start=validation_start) if feature_audit else
              score_audit(reports, validation_start=validation_start))
    result["input_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                       allow_nan=False).encode()).hexdigest()
    result["parameter_version"] = PARAMETER_VERSION
    result["source_posture"] = "USER_SOURCE_PACKET_NOT_VERIFIED_REPLAY_DATASET"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
