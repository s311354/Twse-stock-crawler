"""Forward-only normal-turnover scoring and lower-confidence small entry.

Uncalibrated policy assumptions, not a success probability or bottom guarantee.
Source, Quality, valuation warnings, causal plans and execution limits stay R24.
"""
BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R25"
DECISION_VERSION = "RULE_BASED_NORMAL_TURNOVER_SMALL_R25"
AVAILABLE_AT = "2026-10-29T01:30:00Z"
DAILY_ACTIVATED_AT = "2026-10-29T05:33:12Z"  # Forward only, after the complete 436-stock paired check.


def technical_policy(previous):
    return dict(previous, version="NORMAL_TURNOVER_SMALL_ENTRY_R25",
        buy_minimum=70., small_minimum=55.,
        score_changes="NORMAL_TURNOVER_PARTIAL_CREDIT_FIXED_BUDGET",
        threshold_basis="UNCALIBRATED_SMALL_ONLY_CONCESSION_NOT_WIN_PROBABILITY",
        source_quality_valuation_plan_risk="UNCHANGED_R24",
        missing_policy="NO_IMPUTATION_NO_POSITIVE_SCORE_FLOOR")
