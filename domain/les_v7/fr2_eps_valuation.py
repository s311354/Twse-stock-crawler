"""Source-bound FR2 price-only valuation, separate from the frozen predictor.

The supplemental projection may restore EPS/comparative facts from exactly the
annual originals already bound to the native report. It cannot replace prices,
Quality, a filing revision, or the causal trade plan. Dividends and share counts
are not imputed; this method estimates scenario price returns, not total returns.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

import pandas as pd

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.native_evidence import _market_source, _receipt
from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt
from domain.les_v7.valuation_input_policy import input_policy
from domain.les_v7.valuation_price import policy as price_policy


INPUT_SCHEMA = "les-v7-fr2-eps-inputs/v1"
ASSESSMENT_SCHEMA = "fr2-verified-eps-price-valuation/v1"
WARNING_REASONS = frozenset({"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN",
                             "BASE_FAIR_PRICE_GT_CURRENT_PRICE"})
_ORIGINAL_KEYS = ("symbol", "market", "source", "scope", "currency", "unit", "period_end",
                  "flow_basis", "available_at", "revision_id", "content_sha256", "filing_schema")
_MONEY_FIELDS = ("revenue", "operating_income")
PROOF_BINDING_VERSION = "FR2_EPS_PRICE_PROOF_BINDING_R1"


def _detached(value):
    return json.loads(canonical(value))


def _original_key(row):
    if not isinstance(row, Mapping):
        raise ValueError("FR2_EPS_ANNUAL_ORIGINAL_MAPPING_REQUIRED")
    # Every compared key was retained by the original slim projection. A
    # matching content digest alone must not hide changed scope or publication.
    return identity({key: row.get(key) for key in _ORIGINAL_KEYS})


def _visible_rows(rows, cutoff):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("FR2_EPS_ANNUAL_ORIGINALS_REQUIRED")
    visible = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FR2_EPS_ANNUAL_ORIGINAL_MAPPING_REQUIRED")
        try:
            available = timestamp(row["available_at"])
            period = pd.Period(row["period_end"], freq="Q")
        except (KeyError, ValueError, TypeError, AttributeError):
            # Retain identical malformed provenance for the source producer's
            # UNKNOWN result; selecting around it would favor older evidence.
            visible.append(row)
            continue
        # Future revisions are never selected, even when later added to a
        # supplement. Their presence cannot improve an earlier assessment.
        if available <= timestamp(cutoff) and period.quarter == 4:
            visible.append(row)
    return visible


def _bind_originals(old_rows, new_rows, cutoff):
    old = _visible_rows(old_rows, cutoff)
    new = _visible_rows(new_rows, cutoff)
    grouped = {}
    for row in old:
        grouped.setdefault(_original_key(row), []).append(row)
    selected = []
    seen = set()
    for row in new:
        key = _original_key(row)
        candidates = grouped.get(key, ())
        if not candidates:
            raise ValueError("FR2_EPS_UNRELATED_OR_SUBSTITUTED_ANNUAL_ORIGINAL")
        # Present native observations and proofs must remain identical. The
        # supplemental EPS and comparative fields add dependencies, not a new
        # revenue/operating-income value from a different revision.
        for original in candidates:
            for field in _MONEY_FIELDS:
                if (canonical(row.get(field)) != canonical(original.get(field))
                        or canonical(row.get("field_evidence", {}).get(field, {}))
                        != canonical(original.get("field_evidence", {}).get(field, {}))):
                    raise ValueError("FR2_EPS_ORIGINAL_FINANCIAL_OPERAND_CHANGED")
                proof_url = original.get("field_evidence", {}).get(field, {}).get("source_url")
                # Missing enrichment is not a conflicting URL. Keep it absent
                # so the native producer returns UNKNOWN; never copy a proof
                # URL into a supposedly verified replacement filing.
                if (proof_url is not None and row.get("source_url") is not None
                        and row.get("source_url") != proof_url):
                    raise ValueError("FR2_EPS_ORIGINAL_SOURCE_URL_CHANGED")
            for field in ("source_url", "revision_number"):
                if field in original and row.get(field) != original[field]:
                    raise ValueError("FR2_EPS_ORIGINAL_SOURCE_VERSION_CHANGED")
        # Duplicate supplements add no information, but conflicting enriched
        # copies are retained for the existing producer's conflict diagnostic.
        if identity(row) not in seen:
            selected.append(row)
            seen.add(identity(row))
    # Omitting the latest inconvenient native version cannot authorize an old
    # favorable original. Keep every cutoff-visible original identity covered.
    if set(grouped) != {_original_key(row) for row in selected}:
        raise ValueError("FR2_EPS_NATIVE_ANNUAL_ORIGINAL_OMITTED")
    return selected


def _proof_metadata_reasons(row, snapshot):
    """Tighten the new supplement only; keep the historical producer unchanged.

    Existing exact-context validation checks hash, period, currency and revision
    identity. This research contract additionally requires the selected EPS
    proof's URL and revision number to agree with its source companion/original.
    An unused basic-EPS fact does not invalidate a verified diluted-EPS method.
    """
    basis = snapshot.get("inputs", {}).get("eps_basis")
    if basis not in {"BASIC", "DILUTED"}:
        return []
    companion = row.get("earnings_source", {})
    revision = row.get("revision_number")
    if (type(revision) is not int or revision < 0
            or type(companion.get("revision_number")) is not int
            or companion.get("revision_number") != revision):
        return ["ANNUAL_EPS_COMPANION_REVISION_NUMBER_UNVERIFIED"]
    field = "basic_eps" if basis == "BASIC" else "diluted_eps"
    reasons = []
    for group in ("field_evidence", "comparative_ytd_field_evidence"):
        proof = companion.get(group, {}).get(field, {})
        if proof.get("source_url") != row.get("source_url"):
            reasons.append("ANNUAL_EPS_PROOF_SOURCE_URL_UNVERIFIED:" + group)
        if type(proof.get("revision_number")) is not int or proof.get("revision_number") != revision:
            reasons.append("ANNUAL_EPS_PROOF_REVISION_NUMBER_UNVERIFIED:" + group)
    return sorted(reasons)


def assess_eps_valuation(original_report, valuation_inputs):
    """Return detached assessment/receipt for an explicitly bound supplement.

    The caller retains responsibility for validating the complete native report
    and independently enforcing Quality, safety, trading restrictions and plan
    gates. Invalid bindings are errors; unavailable financial proof is UNKNOWN.
    """
    if (not isinstance(valuation_inputs, Mapping)
            or valuation_inputs.get("schema") != INPUT_SCHEMA):
        raise ValueError("FR2_EPS_INPUT_SCHEMA_REQUIRED")
    if valuation_inputs.get("original_report_id") != identity(original_report):
        raise ValueError("FR2_EPS_ORIGINAL_REPORT_BINDING_MISMATCH")
    diagnostics, envelope = original_report["diagnostics"], original_report["envelope"]
    base = diagnostics["gate_bindings"]["SAFETY"]
    source = diagnostics["native_evidence"]["source_packet"]
    symbol, market, cutoff = base["symbol"], base["market"], base["cutoff"]
    try:
        valid_cutoff = (timestamp(valuation_inputs["cutoff"]) == timestamp(cutoff)
            == timestamp(envelope["event"]["cutoff_at"]) == timestamp(source["cutoff"]))
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError("FR2_EPS_CUTOFF_BINDING_MISMATCH") from error
    if (not valid_cutoff or valuation_inputs.get("symbol") != symbol
            or valuation_inputs.get("market") != market or source.get("symbol") != symbol
            or source.get("market") != market or envelope["event"].get("security_id") != symbol):
        raise ValueError("FR2_EPS_EVENT_BINDING_MISMATCH")
    annual = _bind_originals(source.get("valuation_filings", ()),
        valuation_inputs.get("valuation_filings", ()), cutoff)
    packet = dict(source, valuation_filings=annual)
    rows = source["market_rows"]
    market_source = _market_source(source, symbol=symbol, market=market, cutoff=cutoff, rows=rows)
    receipt = produce_price_valuation_receipt(packet, symbol=symbol, market=market,
        cutoff=cutoff, features=diagnostics["features"], quality=diagnostics["quality"],
        bars=pd.DataFrame(rows), binding=diagnostics["gate_bindings"]["VALUATION"],
        market_source=market_source, policy=price_policy(), input_policy=input_policy())
    latest = {}
    for row in annual:
        try:
            year = pd.Period(row["period_end"], freq="Q").year
            available = timestamp(row["available_at"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if year not in latest or available > timestamp(latest[year]["available_at"]): latest[year] = row
    snapshot = (receipt or {}).get("valuation_diagnostics", {})
    if receipt and snapshot.get("scenarios") is not None and latest:
        metadata_reasons = _proof_metadata_reasons(latest[max(latest)], snapshot)
        if metadata_reasons:
            snapshot = dict(schema=price_policy()["schema"], policy=price_policy(),
                policy_id=identity(price_policy()), status="UNKNOWN", reasons=metadata_reasons,
                return_basis="PRICE_RETURN_ONLY", input_policy=input_policy(),
                input_policy_id=identity(input_policy()), entry_quality_required=True,
                source_verification_policy=PROOF_BINDING_VERSION)
            receipt = _receipt("VALUATION", "UNKNOWN", binding=diagnostics["gate_bindings"]["VALUATION"],
                source=dict(market_source=market_source, selected_annual_original=latest[max(latest)],
                    valuation_policy=price_policy(), input_policy=input_policy(),
                    source_verification_policy=PROOF_BINDING_VERSION,
                    quality_result_id=identity(diagnostics["quality"])),
                available=receipt["source_available_at"], cutoff=cutoff, reasons=metadata_reasons,
                result=snapshot, producer_version=PROOF_BINDING_VERSION)
            receipt["valuation_diagnostics"] = snapshot
    reasons = sorted(set((receipt or {}).get("reason_codes", ()))) if receipt else ["EPS_PRICE_MARKET_SOURCE_UNVERIFIED"]
    checks = snapshot.get("gate_checks")
    failed = {key for key, passed in checks.items() if passed is False} if isinstance(checks, Mapping) else set()
    warning = bool(receipt and receipt["state"] == "FAIL"
        and snapshot.get("status") == "FAIL" and snapshot.get("scenarios") is not None
        and isinstance(checks, Mapping) and set(checks) == {*WARNING_REASONS, "BEAR_LOSS_LTE_MAXIMUM"}
        and all(type(passed) is bool for passed in checks.values())
        and checks["BEAR_LOSS_LTE_MAXIMUM"] is True
        and failed and failed <= WARNING_REASONS and set(reasons) == failed)
    # A complete producer result selects the newest visible original per year;
    # preserve those source identities alongside its original proof receipt.
    assessment = dict(schema=ASSESSMENT_SCHEMA,
        native_state=(receipt or {}).get("state", "UNKNOWN"), warning_eligible=warning,
        scenario_id=identity(snapshot), receipt_id=identity(receipt), reason_codes=reasons,
        snapshot=snapshot, return_basis="PRICE_RETURN_ONLY",
        selected_annual_ids=[identity(latest[year]) for year in sorted(latest)],
        original_report_id=identity(original_report), valuation_inputs_id=identity(valuation_inputs),
        valuation_policy_id=identity(price_policy()), valuation_input_policy_id=identity(input_policy()),
        input_policy_id=identity(input_policy()),
        missing_market_evidence=market_source is None)
    return _detached(dict(assessment=assessment, receipt=receipt))
