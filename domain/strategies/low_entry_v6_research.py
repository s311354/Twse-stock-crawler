"""Read-only score ablation and chronological holdout diagnostics, not trades."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import sqrt
import json

from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_signals import SETUPS


def ablation(evidence: dict) -> dict:
    """Remove one group on the SAME fixed budget; never promote a decision."""
    total = number(evidence.get("total_weight"))
    earned = number(evidence.get("earned_points"))
    if total is None or total <= 0 or earned is None:
        return {}
    return {group: {"removed_points": values["earned_points"],
                    "fixed_budget_score": 100 * (earned - values["earned_points"]) / total}
            for group, values in evidence.get("components", {}).items()}


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    a, b = sum(left) / len(left), sum(right) / len(right)
    x, y = [v - a for v in left], [v - b for v in right]
    denominator = sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(i * j for i, j in zip(x, y)) / denominator if denominator else None


def score_audit(reports: list[dict], *, validation_start: str) -> dict:
    """Require an explicit date split; no tuning or outcomes cross the split."""
    import datetime as dt
    split = dt.date.fromisoformat(validation_start)
    samples = defaultdict(list)
    rows, identities = [], set()
    versions = {json.dumps({key: report.get(key) for key in
                            ("model_version", "parameter_version", "parameters")}, sort_keys=True,
                           allow_nan=False) for report in reports}
    if len(versions) > 1:
        raise ValueError("MIXED_SCORE_RESEARCH_VERSIONS")
    for report in reports:
        day = dt.date.fromisoformat(report["signal_date"])
        identity = (report["symbol"], day)
        if identity in identities:
            raise ValueError("DUPLICATE_SCORE_RESEARCH_ROW")
        identities.add(identity)
        partition = "validation" if day >= split else "discovery"
        for setup in SETUPS:
            evidence = report["setups"][setup]["score_evidence"]
            rows.append({"symbol": report["symbol"], "date": str(day), "setup": setup,
                         "partition": partition, "formal_score": evidence["formal_score"],
                         "estimated_score": evidence["estimated_score"],
                         "coverage_percent": evidence["coverage_percent"],
                         "ablation": ablation(evidence)})
            if evidence["status"] == "COMPLETE":
                samples[(partition, setup)].append(evidence["components"])
    comparisons = []
    for (partition, setup), observations in sorted(samples.items()):
        groups = sorted(observations[0])
        correlations = [{"left": left, "right": right,
                         "pearson": _correlation([o[left]["earned_points"] for o in observations],
                                                 [o[right]["earned_points"] for o in observations])}
                        for left, right in combinations(groups, 2)]
        comparisons.append({"partition": partition, "setup": setup,
                            "complete_samples": len(observations), "correlations": correlations})
    partitions = {row["partition"] for row in rows}
    return {"schema": "les-v6-score-audit/v1", "research_only": True, "executable": False,
            "score_basis": "ADDITIVE_LEDGER_BEFORE_PENALTIES_NOT_FINAL",
            "validation_start": str(split), "rows": rows, "comparisons": comparisons,
            "split_status": "BOTH_PARTITIONS" if len(partitions) == 2 else "INSUFFICIENT_DATE_SPAN",
            "weight_changes_applied": False, "performance_validated": False,
            "limitations": ["CONTRIBUTION_ABLATION_NOT_A_STRATEGY_BACKTEST",
                            "CORRELATION_NOT_CAUSATION_OR_WIN_PROBABILITY",
                            "NO_FILLS_COSTS_OR_CAGR_COMPARISON"]}


def feature_ablation_audit(reports: list[dict], *, validation_start: str) -> dict:
    """Re-run the native decision core with score-only feature interventions.

    Inputs/cutoffs/plans/vetoes stay frozen. Removed evidence is not zero-filled
    in the feature frame, so an EMA ablation cannot delete its hard-veto role.
    Retention measures native BUY/BUY_SMALL labels, NEVER profitable trades.
    """
    from dataclasses import replace
    import datetime as dt
    import hashlib
    from domain.strategies.low_entry_score_v6 import PARAMETER_VERSION
    from domain.strategies.low_entry_v6_correlation import CHANNELS, POOLS, VARIANTS
    from domain.strategies.low_entry_v6_decision import decide_unified
    from domain.strategies.low_entry_v6_quality import QualityResult
    from domain.strategies.low_entry_v6_signals import EXECUTABLE_SIGNALS, LowEntryV6Config

    contribution = score_audit(reports, validation_start=validation_start)
    split = dt.date.fromisoformat(validation_start)
    rows, observations = [], defaultdict(list)
    for report in reports:
        config = LowEntryV6Config(**report["parameters"])
        if (report.get("parameter_version") != PARAMETER_VERSION or report.get("model_version") != "6.0.0"
                or config.decision_policy_version != 2 or config.correlation_policy_version != 2):
            raise ValueError("FEATURE_AUDIT_REQUIRES_CURRENT_DEDUP_REPORTS")
        day = dt.date.fromisoformat(report["signal_date"])
        partition = "validation" if day >= split else "discovery"
        q, data = report["quality"], report["data"]
        quality = QualityResult(status=q["status"], score=q["score"], confidence=data["confidence"],
            integrity=data["integrity"], components=q.get("components", {}), coverage=q.get("coverage", {}),
            metrics=q.get("metrics", {}), reasons=tuple(q.get("reasons", [])))
        kwargs = dict(valuation=report.get("valuation", {}).get("interpretation", "NOT_AVAILABLE"),
                      industry_mode=report.get("industry_profile", {}).get("mode", "UNSUPPORTED"))
        reference = decide_unified(quality, report["technical"], config=config, **kwargs)
        # Compare against the published native result before intervention, not
        # a silently recomputed substitute for inconsistent source reports.
        identity_keys = ("signal", "setup", "raw_score", "penalty", "adjusted_score", "safety_cap",
                         "final_score", "penalties", "safety_caps", "score_evidence", "selection",
                         "correlation_adjustment")
        if any(key not in report["decision"] or report["decision"][key] != reference[key]
               for key in identity_keys) or report["decision"].get("research_only") is True:
            raise ValueError("FEATURE_AUDIT_NATIVE_RESULT_MISMATCH")
        additive_config = replace(config, correlation_policy_version=1)
        additive = decide_unified(quality, report["technical"], config=additive_config, **kwargs)
        variants = {"ADDITIVE_BASELINE": (additive_config, frozenset(), reference),
                    **{name: (config, frozenset(channels), reference) for name, channels in VARIANTS.items()},
                    **{"ADDITIVE_" + name: (additive_config, frozenset(channels), additive)
                       for name, channels in VARIANTS.items()}}
        for name, (parameters, removed, comparator) in variants.items():
            result = decide_unified(quality, report["technical"], config=parameters, score_exclusions=removed, **kwargs)
            source_signal = comparator["signal"] in EXECUTABLE_SIGNALS
            changed_signal = result["signal"] in EXECUTABLE_SIGNALS
            rows.append(dict(symbol=report["symbol"], date=str(day), decision_time=report["decision_time"],
                partition=partition, variant=name, removed_channels=sorted(removed),
                correlation_policy_version=parameters.correlation_policy_version,
                reference=dict(signal=comparator["signal"], setup=comparator["setup"], raw=comparator["raw_score"],
                               final=comparator["final_score"], penalty=comparator["penalty"],
                               correlation_policy=comparator["correlation_adjustment"]["policy_version"]),
                result=dict(signal=result["signal"], setup=result["setup"], raw=result["raw_score"],
                            final=result["final_score"], penalty=result["penalty"], safety_caps=result["safety_caps"],
                            risk_gate=result["risk_gate"], plan=result["plan"]),
                final_delta=result["final_score"] - comparator["final_score"]
                    if result["final_score"] is not None and comparator["final_score"] is not None else None,
                reference_entry_signal=source_signal, retained_entry_signal=source_signal and changed_signal,
                entry_signal_after_intervention=changed_signal, setup_changed=result["setup"] != comparator["setup"]))
        for setup in SETUPS:
            channels = reference["setup_decisions"][setup]["correlation_adjustment"]["channels"]
            for pool, names in POOLS.items():
                active = [name for name in names if CHANNELS[setup][name]]
                if all(channels[name]["complete"] for name in active):
                    observations[(partition, setup, pool)].append({
                        name: channels[name]["retained_points"] for name in active})
    summaries = []
    for partition in ("discovery", "validation"):
        for variant in ("ADDITIVE_BASELINE", *VARIANTS, *("ADDITIVE_" + name for name in VARIANTS)):
            selected = [r for r in rows if r["partition"] == partition and r["variant"] == variant]
            eligible = sum(r["reference_entry_signal"] for r in selected)
            retained = sum(r["retained_entry_signal"] for r in selected)
            summaries.append(dict(partition=partition, variant=variant, samples=len(selected),
                reference_entry_signals=eligible, retained_entry_signals=retained,
                retention_ratio=retained / eligible if eligible else None,
                reference_policy="additive-source-confirmations/v1" if variant.startswith("ADDITIVE_")
                    and variant != "ADDITIVE_BASELINE" else "source-family-max-pooling/v1",
                entry_signals_after_intervention=sum(r["entry_signal_after_intervention"] for r in selected),
                changed_setups=sum(r["setup_changed"] for r in selected)))
    comparisons = []
    for (partition, setup, pool), samples in sorted(observations.items()):
        for left, right in combinations(sorted(samples[0]), 2):
            a, b = [r[left] for r in samples], [r[right] for r in samples]
            comparisons.append(dict(partition=partition, setup=setup, pool=pool, left=left, right=right,
                complete_samples=len(samples), pearson=_correlation(a, b),
                positive_disagreements=sum((x > 0) != (y > 0) for x, y in zip(a, b))))
    return dict(schema="les-v6-feature-ablation/v1", parameter_version=PARAMETER_VERSION,
        reference_parameters=reports[0]["parameters"] if reports else {},
        research_only=True, executable=False, performance_validated=False,
        validation_start=validation_start, split_status=contribution["split_status"],
        input_reports_sha256=hashlib.sha256(json.dumps(reports, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode()).hexdigest(),
        intervention="SCORE_CONFIRMATION_CHANNEL_ONLY_NOT_FULL_INDICATOR_REMOVAL",
        summary=summaries, comparisons=comparisons, rows=rows,
        limitations=["RETENTION_IS_NATIVE_SIGNAL_RETENTION_NOT_PROFITABILITY",
                     "HARD_GATES_PLANS_AND_EXECUTION_REQUIREMENTS_UNCHANGED",
                     "NO_OUTCOME_EVIDENCE_NO_EMPIRICAL_REDUNDANCY_OR_PERFORMANCE_CLAIM",
                     "NO_AUTOMATIC_WEIGHT_TUNING_OR_ENTRY_AUTHORIZATION"])
