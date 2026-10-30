"""Compare a frozen recorded EREM decision tape, never simulate new accounts."""
from __future__ import annotations

from collections import Counter
import datetime as dt
import math
import re

import pandas as pd

from application.replay_contracts import digest
from domain.strategies.erem_soft_reduction import SoftReductionResearchPolicy, apply_soft_reduction_policy


def _reason_value(reasons, field):
    """Read exact retained audit labels; absence/conflict is not a default."""
    if not isinstance(reasons, list) or any(not isinstance(value, str) for value in reasons):
        return None
    values = {match for reason in reasons for match in re.findall(r"(?:^|；)" + field + r"=([^；]+)", reason)}
    return next(iter(values)) if len(values) == 1 else None


def compare_recorded_erem(run: dict, *, policy=None):
    policy = policy or SoftReductionResearchPolicy()
    if not isinstance(run, dict) or not isinstance(run.get("jobId"), str) or not isinstance(run.get("daily"), list):
        raise ValueError("EREM_RESEARCH_SAVED_RUN_REQUIRED")
    days = run["daily"]
    dates = [day.get("date") for day in days]
    if not dates or any(not isinstance(date, str) for date in dates) or dates != sorted(set(dates)):
        raise ValueError("EREM_RESEARCH_UNIQUE_ORDERED_DATES_REQUIRED")
    for date in dates:
        dt.date.fromisoformat(date)
    prepared, symbols = [], set()
    for day in days:
        snapshot = day.get("openingSnapshot")
        owner = (day.get("research") or {}).get("accountSnapshot")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("holdings"), list) or snapshot != owner:
            raise ValueError("EREM_RESEARCH_ACCOUNT_SNAPSHOT_BINDING_MISSING_OR_CONFLICTING")
        positions = {position["code"]: position for position in snapshot["holdings"]}
        signals = {signal["code"]: signal for signal in day.get("decisions", [])}
        if len(positions) != len(snapshot["holdings"]) or len(signals) != len(day.get("decisions", [])):
            raise ValueError("EREM_RESEARCH_DUPLICATE_SYMBOL")
        symbols.update(positions)
        prepared.append((day, positions, signals))
    rows, keys = [], {}
    for symbol in sorted(symbols):
        commands, entry_les, ages, kinds, graces, source_rows = [], [], [], [], [], []
        for day, positions, signals in prepared:
            position, signal = positions.get(symbol), signals.get(symbol, {})
            reason = signal.get("eremReasons")
            raw_age, raw_grace = _reason_value(reason, "HoldingDays"), _reason_value(reason, "GraceProfile")
            try:
                age = float(raw_age) if raw_age is not None else math.nan
            except ValueError:
                age = math.nan
            commands.append(signal.get("eremDecision"))
            entry_les.append(position.get("entryLES", math.nan) if position else math.nan)
            ages.append(age if position else math.nan)
            kinds.append(position.get("entryDecision") if position else None)
            graces.append(False if raw_grace == "NONE" else True if raw_grace in {
                "BUY_3_SESSION", "BUY_SMALL_2_SESSION", "EARLY_BUY_2_SESSION"} else None)
            source_rows.append((day, position, signal))
        index = pd.Index(dates)
        transformed = apply_soft_reduction_policy(pd.Series(commands, index=index),
            entry_les=pd.Series(entry_les, index=index), holding_days=pd.Series(ages, index=index),
            entry_decision=pd.Series(kinds, index=index), entry_grace=pd.Series(graces, index=index), policy=policy)
        for i, (day, position, signal) in enumerate(source_rows):
            if position is None:
                continue
            value = transformed.iloc[i]
            row = dict(date=day["date"], symbol=symbol, recorded_shares=position.get("shares"),
                recorded_entry_date=position.get("entryDate"), native_decision=commands[i],
                research_decision=value["decision"], research_status=value["status"],
                cooldown_remaining=int(value["cooldown_remaining"]),
                research_account_re_evolved=False, holding_confidence=signal.get("hc"),
                native_hard_financial_exit=signal.get("fundamentalHardExit"),
                external_risk_exit_policy="UNCHANGED_OUTSIDE_THIS_TRANSFORM")
            rows.append(row)
            keys[(day["date"], symbol)] = row
    fills = []
    for day in days:
        for fill in day.get("fills", []):
            if fill.get("side") != "SELL":
                continue
            row = keys.get((day["date"], fill["code"]))
            fills.append(dict(date=day["date"], symbol=fill["code"], actual_reason=fill["reason"],
                actual_shares=fill["shares"], native_decision=row["native_decision"] if row else None,
                research_decision=row["research_decision"] if row else None,
                diagnostic_only=True, counterfactual_fill="NOT_COMPUTED",
                external_risk_exit_unchanged=fill["reason"] != "EREM_REDUCE"))
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    result = dict(schema="erem-frozen-account-decision-comparison/v1", job_id=run["jobId"],
        source_versions=run.get("versions"), source_evidence_policy=run.get("evidencePolicy"),
        source_reference_date=run.get("referenceDate"), policy=policy.record(),
        policy_id=digest(policy.record()), account_policy="RECORDED_ACCOUNT_INPUTS_NOT_COUNTERFACTUAL_ACCOUNTS",
        rows=rows, observed_sell_diagnostics=fills,
        status_counts=dict(Counter(row["research_status"] for row in rows)),
        changed_recorded_position_decisions=sum(row["native_decision"] != row["research_decision"] for row in rows),
        net_return_effect=None, performance_status="NOT_ASSESSABLE_WITHOUT_REEVOLVED_ACCOUNT_AND_COSTS",
        sample_out_of_sample_validated=False, approved_for_live=False, email=False, order_authority=False)
    result["comparison_id"] = digest(result)
    return result
