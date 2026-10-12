"""Standalone offline LES v6 JSON runner. No network, email or orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from application.strategy_execution import StrategyExecutionService
from application.industry_benchmark import prepare_industry_context
from domain.strategies.low_entry_score_v6 import V6_OUTPUT_COLUMNS, attach_low_entry_v6_context, gate_funnel
from domain.strategy import get_strategy


def analyze_request(payload: dict) -> dict:
    """One security or a same-session universe; input schema is documented in docs/."""
    policy = payload.get("score_policy", "unified")
    if policy not in {"unified", "hybrid", "hybrid-legacy"}:
        raise ValueError("UNKNOWN_V6_SCORE_POLICY")
    strategy_name = {"hybrid": "low_entry_score_v6_hybrid", "hybrid-legacy": "low_entry_score_v6_hybrid_legacy",
                     "unified": "low_entry_score_v6"}[policy]
    weights = payload.get("convex_weights")
    if weights is not None and policy == "unified":
        raise ValueError("CONVEX_WEIGHTS_REQUIRE_EXPLICIT_HYBRID_POLICY")
    if weights is not None and policy != "hybrid-legacy":
        raise ValueError("CONVEX_WEIGHTS_REQUIRE_BASELINE_HYBRID_POLICY")
    requests = payload.get("securities", [payload])
    histories = []
    for request in requests:
        benchmark = request.get("taiex_bars")
        context = prepare_industry_context(request["context"])
        if weights is not None:
            if "convex_weights" in context:
                raise ValueError("CONFLICTING_WEIGHT_ARTIFACTS")
            context["convex_weights"] = weights
        histories.append(attach_low_entry_v6_context(
            pd.DataFrame(request["bars"]),
            taiex=pd.DataFrame(benchmark) if benchmark is not None else None,
            **context,
        ))
    service = StrategyExecutionService()
    prepared = service.prepare(histories)
    if weights is None:
        result = service.evaluate(strategy_name, V6_OUTPUT_COLUMNS, prepared)
        reports = [r for r in result["LES v6 Report"] if isinstance(r, dict)]
        return {"model": strategy_name, "reports": reports, "funnel": gate_funnel(reports)}
    # Indicators warm over the exact supplied range; the fitted model is applied
    # only at the explicit cutoff, never retroactively to warmup bars.
    plugin = get_strategy(strategy_name)
    reports = []
    for history, features in zip(prepared.histories, prepared.feature_frames, strict=True):
        reports.extend(plugin.run_with_features(history, features, latest_only=True)["LES v6 Report"].tolist())
    return dict(model=strategy_name, reports=reports, funnel=gate_funnel(reports),
        evaluation_mode="CONVEX_WEIGHT_SNAPSHOT", execution_policy="NATIVE_GATES",
        requires_execution_check=True,
        research_only=any(report.get("evaluation_mode") == "HISTORICAL_REPLAY" for report in reports))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSON: bars, optional taiex_bars, context; or securities[]")
    parser.add_argument("--score-policy", choices=("unified", "hybrid", "hybrid-legacy"),
                        help="Explicit scoring policy; Hybrid weights/thresholds are uncalibrated research candidates")
    parser.add_argument("--weights", type=Path, help="Frozen baseline artifact; native execution gates still apply; requires --score-policy hybrid-legacy")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if args.score_policy is not None:
            payload = dict(payload, score_policy=args.score_policy)
        if args.weights is not None:
            artifact = json.loads(args.weights.read_text(encoding="utf-8"))
            if not isinstance(artifact, dict):
                raise ValueError("WEIGHTS_REQUIRE_OBJECT")
            artifact = artifact.get("artifact") if "artifact" in artifact else artifact
            if artifact is None:
                raise ValueError("NO_ESTIMATED_WEIGHT_ARTIFACT")
            payload = dict(payload, convex_weights=artifact)
        result = analyze_request(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
