"""Daily v7 signals and account-bound PAE quotas; never broker/order authority."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import json
import pandas as pd

from application.evidence_context import ReadOnlyEvidence
from application.les_v7 import prepare_quality_context
from domain.les_v7.quality_policy import is_source_evidence_policy
from domain.strategy import get_strategy
from domain.les_v7.contracts import ReferencePlan, identity, timestamp, quality_extension_schema
from domain.les_v7.report import validate_report
from domain.numeric_history import metadata_free_frame
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR, PRICE_SCALE_ATTR, SIGNAL_DATE_ATTR


V7_DAILY_COLUMNS = ["LES v7 Raw Score", "LES v7 Penalty", "LES v7 Adjusted Score", "LES v7 Safety Cap",
    "LES v7 Final Score", "LES v7 Score", "LES v7 Decision", "LES v7 Context", "LES v7 Report",
    "LES v7 Required Score", "LES v7 Trigger", "LES v7 Execution Eligible", "Risk Reasons", "Liquidity Status"]
V7_PROJECTED_COLUMNS = set(V7_DAILY_COLUMNS) | {"低點分數", "低點決策", "低點理由", "LES Model Version",
    "LES Parameter Version", "Quality Score", "Quality Status", "Quality Gate", "Data Confidence",
    "No Veto", "Falling Knife Veto", "Falling Knife State", "策略停損", "策略停利", "ATR14",
    "Risk Gate", "Risk Reasons", "RewardRisk", "Position Fraction", "建議股數", "Liquidity Status",
    "Entry Candidate", "Entry Rule", "Entry Trigger Price"}


def plain(value, *, preserve_frozen=False):
    """Normalize scalar adapters without expanding trusted immutable proofs.

    Only the native detached ReadOnlyEvidence boundary is reusable. Ordinary
    mappings/sequences are still materialized, including MappingProxyType over
    mutable caller inputs. Tuple/list have the same canonical JSON semantics.
    Date/numpy scalars retain the existing ISO/native normalization policy.
    """
    if preserve_frozen and isinstance(value, ReadOnlyEvidence):
        if value.json_native:
            return value  # Computed once at the native freeze boundary.
        changed = None
        for key, item in value.items():
            normalized = plain(item, preserve_frozen=True)
            if normalized is not item:
                if changed is None: changed = dict(value)
                changed[key] = normalized
        return ReadOnlyEvidence(changed) if changed is not None else value
    if preserve_frozen and isinstance(value, tuple):
        changed = None
        for index, item in enumerate(value):
            normalized = plain(item, preserve_frozen=True)
            if normalized is not item:
                if changed is None: changed = list(value)
                changed[index] = normalized
        return tuple(changed) if changed is not None else value
    if isinstance(value, Mapping): return {key: plain(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, (datetime, date)): return value.isoformat()
    if hasattr(value, "item"): return value.item()
    return value


def _native_source_packet(history, raw, context):
    """Capture actual source operands, not verdict flags from older models.

    Annual valuation projection is intentionally narrow: unverified TTM,
    arbitrary zero dividends and copied v5 valuation PASS are never inputs.
    Native source times are supplied by the capture owner, not inferred here.
    """
    from domain.les_v7.features import select_bars
    from domain.les_v7.native_evidence import SOURCE_SCHEMA, VALUATION_FIELDS
    from domain.les_v7.rule_baseline import baseline_policy
    price_valuation = bool(context.get("rule_baseline") and baseline_policy(context["market"],
        **({"version": context["rule_baseline_policy_version"]} if "rule_baseline_policy_version" in context else {})).get("valuation_policy"))
    selected = select_bars(raw, context["as_of"])
    rows = selected.astype(object).where(pd.notna(selected), None).to_dict("records")
    for row in rows:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    market_source = plain(history.attrs.get("les_v7_market_evidence", {}))
    def visible_capture(value):
        # A valid future current observation is not evidence at this cutoff.
        # Unverifiable times remain present and fail closed in their validator.
        try:
            return {} if value and timestamp(value["source_available_at"] if "source_available_at" in value
                else value["available_at"]) > timestamp(context["as_of"]) else value
        except (KeyError, ValueError, TypeError, AttributeError):
            return value
    market_source = visible_capture(market_source)
    if market_source:
        # IDs identify the captured numerical source values, not an exchange
        # signature or proof that cached final rows were observed historically.
        bars_id = identity(rows)
        if "bars_sha256" in market_source and market_source["bars_sha256"] != bars_id:
            raise ValueError("V7_NATIVE_CAPTURED_MARKET_HASH_MISMATCH")
        market_source["bars_sha256"] = bars_id
        market_source.setdefault("source_record_ids", [bars_id])
    annual = []
    for filing in context["filings"]:
        try:
            period = pd.Period(filing["period_end"], freq="Q")
            if period.quarter != 4 or timestamp(filing["available_at"]) > timestamp(context["as_of"]):
                continue
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        fields = ("symbol", "market", "source", "scope", "currency", "unit", "period_end",
                  "flow_basis", "available_at", "revision_id", "content_sha256", "filing_schema")
        projected = {**{key: filing.get(key) for key in (*fields, *VALUATION_FIELDS)},
            "field_evidence": {key: filing.get("field_evidence", {}).get(key, {}) for key in VALUATION_FIELDS}}
        if price_valuation:
            projected["source_url"] = filing.get("source_url")
            projected["earnings_source"] = context.get("valuation_earnings", {}).get(filing.get("content_sha256"))
            for group in ("comparative_ytd_fields", "comparative_ytd_field_evidence"):
                projected[group] = {field: filing.get(group, {}).get(field) for field in ("revenue", "operating_income")}
        annual.append(projected)
    restrictions = plain(history.attrs.get("trading_restriction", {}))
    if restrictions.get("scope") != "HISTORICAL_ANNOUNCEMENTS":
        restrictions = visible_capture(restrictions)
    packet = dict(schema=SOURCE_SCHEMA, symbol=context["symbol"], market=context["market"],
        cutoff=timestamp(context["as_of"]).isoformat(), expected_session=str(history.attrs.get("les_v6_analysis_session", ""))[:10],
        historical_replay=bool(history.attrs.get("replay_as_of")), market_rows=plain(rows),
        market_evidence=market_source,
        restriction_evidence=restrictions, valuation_filings=annual)
    from domain.les_v7.rule_baseline import RECOVERY_POLICIES, INDUSTRY_POLICIES
    if context.get("rule_baseline_policy_version") in RECOVERY_POLICIES:
        from application.valuation_peers_r10 import annual_projection, recovery_projection
        packet["valuation_filings"] = annual_projection(dict(les_v6_mops_filings=context["filings"],
            les_v7_context=dict(valuation_earnings=context.get("valuation_earnings", {}))), context["as_of"])
        packet["filings"] = recovery_projection(context["filings"], context["as_of"])
        packet["r10_valuation_context"] = history.attrs.get("les_v7_native_sources", {}).get("r10_valuation_context")
    if context.get("rule_baseline_policy_version") in INDUSTRY_POLICIES:
        packet["industry_input"] = plain(context["industry_input"])
    if attrs_source := history.attrs.get("les_v7_native_sources"):
        from domain.les_v7.policy_r16 import BASELINE_VERSION as R16
        from domain.les_v7.policy_r17 import BASELINE_VERSION as R17
        from domain.les_v7.policy_r18 import BASELINE_VERSION as R18
        from domain.les_v7.policy_r19 import BASELINE_VERSION as R19
        from domain.les_v7.rule_baseline import CONDITIONAL_POLICIES
        if context.get("rule_baseline_policy_version") in CONDITIONAL_POLICIES and attrs_source.get("calendar_evidence") is not None:
            # Retain the existing official raw calendar/proof contract. A list
            # of dates or a caller's PASS flag is not a substitute for it.
            packet["calendar_evidence"] = plain(attrs_source["calendar_evidence"])
        if context.get("rule_baseline_policy_version"):
            from domain.les_v7.rule_baseline import SOURCE_POLICIES
            if context["rule_baseline_policy_version"] in SOURCE_POLICIES:
                packet.update(security_evidence=plain(attrs_source.get("security")),
                    official_pe=plain(attrs_source.get("official_pe", ())),
                    quote_market_rows=context["quote_market_rows"],
                    source_join_diagnostics=context["source_join_diagnostics"])
    if context.get("quote_market_rows") is not None and "quote_market_rows" not in packet:
        packet.update(security_evidence=None, official_pe=[],
            quote_market_rows=context["quote_market_rows"], source_join_diagnostics=context["source_join_diagnostics"])
    if history.attrs.get("les_v7_fundamental_capture_cutoff") is not None:
        packet["financial_capture_cutoff"] = timestamp(history.attrs["les_v7_fundamental_capture_cutoff"]).isoformat()
    return ReadOnlyEvidence(packet)


def research_valuation_inputs(original, context):
    """Detached EPS operands from the same originals as the native report.

    This supplements only the new research valuation component. The frozen
    predictor's original native packet and its old valuation remain intact.
    Missing source companions remain missing; no shares/dividends are imputed.
    """
    from domain.les_v7.native_evidence import VALUATION_FIELDS
    annual = []
    for filing in context["filings"]:
        try:
            period = pd.Period(filing["period_end"], freq="Q")
            if period.quarter != 4 or timestamp(filing["available_at"]) > timestamp(context["as_of"]):
                continue
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        fields = ("symbol", "market", "source", "scope", "currency", "unit", "period_end",
                  "flow_basis", "available_at", "revision_id", "content_sha256", "filing_schema")
        row = {key: filing.get(key) for key in (*fields, *VALUATION_FIELDS)}
        row["field_evidence"] = {key: filing.get("field_evidence", {}).get(key, {}) for key in VALUATION_FIELDS}
        row.update(source_url=filing.get("source_url"), revision_number=filing.get("revision_number"),
            earnings_source=context.get("valuation_earnings", {}).get(filing.get("content_sha256")))
        for group in ("comparative_ytd_fields", "comparative_ytd_field_evidence"):
            row[group] = {field: filing.get(group, {}).get(field) for field in ("revenue", "operating_income")}
        annual.append(row)
    return plain(dict(schema="les-v7-fr2-eps-inputs/v1", original_report_id=identity(original),
        symbol=context["symbol"], market=context["market"], cutoff=timestamp(context["as_of"]).isoformat(),
        valuation_filings=annual))


def _research_bundle_enabled(attrs, *, model, symbol, market):
    raw = attrs.get("les_v7_research_metadata_json")
    if not raw:
        return False
    metadata = json.loads(raw)
    if (not isinstance(metadata, dict) or metadata.get("schema") != "les-v7-research-model-use/v1"
            or model is None or metadata.get("bundle_sha256") != model.get("sha256")
            or metadata.get("research_only") is not True or metadata.get("approved_for_live") is not False
            or metadata.get("execution_action") != "NO_ACTION" or metadata.get("order_authority") is not False
            or not isinstance(metadata.get("symbols"), list) or symbol not in metadata["symbols"] or market != "TWSE"
            or attrs.get("les_v7_native_evidence") is not True or attrs.get("les_v7_rule_baseline") is True
            or attrs.get("les_v7_diagnostics_only") is True or attrs.get("replay_as_of")):
        raise ValueError("V7_RESEARCH_BUNDLE_DAILY_BINDING_MISMATCH")
    return True


def daily_report(history, *, features=None, evidence=None):
    if evidence is not None: evidence.validate(history)
    attrs = history.attrs
    meta = attrs.get("les_v6_daily_identity", {})  # Legacy identity key, not a score source.
    symbol, market = str(meta.get("symbol", "")), attrs.get("market_source", "TWSE")
    cutoff = timestamp(attrs[SIGNAL_DATE_ATTR]).isoformat()
    snapshot = evidence.snapshot if evidence is not None else attrs.get(FUNDAMENTAL_SNAPSHOT_ATTR, {})
    extra = plain(snapshot.get("les_v7_context", {}), preserve_frozen=True)
    for key, expected in (("symbol", symbol), ("market", market), ("as_of", cutoff)):
        if key in extra and (timestamp(extra[key]).isoformat() if key == "as_of" else extra[key]) != expected:
            raise ValueError("V7_DAILY_CONTEXT_BINDING_MISMATCH:" + key)
    model = json.loads(attrs["les_v7_bundle_json"]) if attrs.get("les_v7_bundle_json") else None
    research = _research_bundle_enabled(attrs, model=model, symbol=symbol, market=market)
    context = dict(symbol=symbol, market=market, as_of=cutoff,
        filings=plain(snapshot.get("les_v6_filings", ()), preserve_frozen=True), model_bundle=model,
        quality_policy=model["payload"]["quality_policy"] if model else extra.get("quality_policy"),
        receipts=extra.get("receipts", {}), external_evidence_ids=list(extra.get("external_evidence_ids", ())))
    if research:
        context["valuation_earnings"] = extra.get("valuation_earnings", {})
    if attrs.get("les_v7_strategy_policy_json"):
        context["strategy_policy"] = json.loads(attrs["les_v7_strategy_policy_json"])
    if attrs.get("les_v7_native_evidence") is True and attrs.get("les_v7_diagnostics_only") is True:
        context["diagnostics_only"] = True
    if attrs.get("les_v7_rule_baseline") is True:
        if attrs.get("les_v7_native_evidence") is not True or attrs.get("replay_as_of"):
            raise ValueError("V7_RULE_BASELINE_CURRENT_NATIVE_DAILY_REQUIRED")
        context["rule_baseline"] = True
        context["valuation_earnings"] = extra.get("valuation_earnings", {})
        if attrs.get("les_v7_source_repair") is True:
            from domain.les_v7.rule_baseline import current_source_baseline_version, INDUSTRY_POLICIES
            context["rule_baseline_policy_version"] = current_source_baseline_version(cutoff)
            if context["rule_baseline_policy_version"] in INDUSTRY_POLICIES:
                from domain.industry_benchmark import INDUSTRY_BENCHMARK_ATTR
                context["industry_input"] = dict(symbol=symbol,
                    sector=snapshot.get("sector") or meta.get("sector", ""),
                    benchmark=evidence.industry_benchmark if evidence is not None else attrs.get(INDUSTRY_BENCHMARK_ATTR))
    source_evidence = is_source_evidence_policy(context["quality_policy"]) or (model is None and context["quality_policy"] is None)
    if source_evidence and "les_v6_mops_filings" in snapshot:
        # A batch row's identity/unit cannot masquerade as its nested iXBRL
        # original when adding parent/interest/payment proofs. R2/R3 use the
        # canonical originals consistently; legacy R1 keeps the old adapter.
        originals = plain(snapshot["les_v6_mops_filings"], preserve_frozen=True)
        context["filings"] = tuple(dict(row, market=market) if row.get("market") != market else row
                                   for row in originals)
    prepare_quality_context(context, financial_audits=extra.get("financial_audits", {}),
                            financial_supplements=extra.get("financial_supplements", {}),
                            financial_repairs_r5=extra.get("financial_repairs_r5", {}),
                            signed_financial_audits=extra.get("signed_financial_audits", {}),
                            financial_audits_r7=extra.get("financial_audits_r7", {}))
    raw = metadata_free_frame(history).reindex(columns=["Date", "Open", "High", "Low", "Close", "Volume",
        "Trades", "Amount", "Bid", "Ask", "PE", "TAIEX Close"])
    if attrs.get("les_v7_native_evidence") is True:
        for column in ("Available At", "Complete", "Symbol", "Market"):
            if column in history:
                raw[column] = history[column].to_numpy(copy=True)
    scale = float(attrs.get(PRICE_SCALE_ATTR, 1000.))
    if scale != 1000.: raise ValueError("V7_DAILY_EXPECTS_LEGACY_PRICE_SCALE_1000")
    for column in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce") / scale
    if attrs.get("les_v7_source_repair") is True and context.get("rule_baseline"):
        from domain.les_v7.source_projection import join_official_pe
        if features is not None:
            # Validate the prepared quote input first. Only this controlled
            # source join may invalidate reuse; unrelated stale features must
            # still fail rather than silently rebuilding away a mismatch.
            features.assert_compatible(raw)
        quote = raw.astype(object).where(pd.notna(raw), None).to_dict("records")
        for row in quote:
            row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
        sources = attrs.get("les_v7_native_sources", {})
        from domain.les_v7.rule_baseline import baseline_policy
        join_policy = baseline_policy(market, version=context["rule_baseline_policy_version"])["source_policy"]
        joined, diagnostics = join_official_pe(quote, plain(sources.get("official_pe", ())),
            symbol=symbol, market=market, cutoff=cutoff, source_policy=join_policy)
        joined_frame = pd.DataFrame(joined, index=raw.index)
        if features is not None and not pd.to_numeric(raw["PE"], errors="coerce").astype(float).equals(
                pd.to_numeric(joined_frame["PE"], errors="coerce").astype(float)):
            # P/E participates in both the source fingerprint and rolling
            # feature calculations. Recompute for the validated joined input;
            # never relabel the old fingerprint or weaken assert_compatible.
            from domain.features import SharedFeaturePipeline
            features = SharedFeaturePipeline(config=features.config).build(joined_frame)
        raw = joined_frame
        context.update(quote_market_rows=quote, source_join_diagnostics=diagnostics)
    # Missing/invalid source times cannot be silently dropped to select older filings.
    errors = []
    try:
        for record in context["filings"]: timestamp(record["available_at"])
    except (KeyError, TypeError, ValueError, AttributeError):
        errors.append("INVALID_FINANCIAL_AVAILABILITY")
        context["external_evidence_ids"].append(identity(context["filings"]))
        context["filings"], context["receipts"] = [], {}
    expected_session = attrs.get("les_v6_analysis_session")
    actual_session = str(raw.Date.iloc[-1])[:10] if len(raw) else None
    if expected_session and actual_session != str(expected_session)[:10]:
        errors.append("STALE_MARKET_DATA")
        context["receipts"] = {}
    # Freeze only the small adapter context. Already detached filing/proof nodes
    # are retained, so the native engine's caller-isolation copy is constant
    # depth rather than another expanded financial dict/list graph.
    options = {}
    if attrs.get("les_v7_native_evidence") is True:
        if context["receipts"]:
            # The opt-in producer path does not silently replace an explicitly
            # bound legacy/API receipt set. Such callers retain the old path.
            pass
        else:
            from domain.les_v7.native_evidence import produce_native_receipts
            context["native_evidence_source"] = _native_source_packet(history, raw, context)
            options["receipt_producer"] = produce_native_receipts
    report = get_strategy("low_entry_score_v7").evaluate(raw, context=ReadOnlyEvidence(context), features=features, **options)
    if research:
        from domain.les_v7.fr2_policy import apply_fr2_eps
        # Preserve a fully validated original, including daily diagnostics,
        # before adding the independently identified research-policy overlay.
        _finalize_daily_inputs(report, errors)
        report = apply_fr2_eps(report, model, research_valuation_inputs(report, context))
    account = plain(attrs.get("les_v6_portfolio", {}))
    if attrs.get("replay_as_of") and "as_of" not in account:
        account = {}  # Never treat a current account as a historical snapshot.
    return finalize_daily_report(report, account=account, daily_input_errors=errors)


def finalize_daily_report(report, *, account, daily_input_errors=()):
    """Shared daily/research finalization; never supplies an assumed account.

    The native engine owns calculation. This public adapter retains daily input
    errors, validates the complete native report, derives its account-bound PAE
    candidate quota and validates again. It sends nothing and submits no state.
    """
    errors = list(daily_input_errors)
    if report.get("diagnostics", {}).get("fr2") is not None:
        if report["diagnostics"].get("daily_input_errors", []) != errors:
            raise ValueError("V7_FR2_DAILY_INPUT_ERRORS_BINDING_MISMATCH")
        validate_report(report)
        from domain.les_v7.fr2_execution import assess_execution
        report["fr2_execution"] = assess_execution(report, report["diagnostics"]["fr2"]["admission"], account)
        validate_report(report)
        return report
    _finalize_daily_inputs(report, errors)
    # The immediately preceding call fully validates the complete signal. Use
    # the existing native quota arithmetic for that validated body, retaining
    # the independent full post-authorization check below. Public callers of
    # authorize_candidate still receive their own complete input validation.
    from domain.les_v7.allocation import _authorize_validated
    report["pae_authorization"] = _authorize_validated(report, account)
    validate_report(report)
    return report


def daily_input_outcome(score, decision, *, errors):
    """Pure shared publication semantics, never permission to skip validation.

    Offline score comparisons retain the same daily input-error override as the
    public native finalizer, without publishing a new report or account quota.
    """
    errors = list(errors)
    if any(not isinstance(error, str) or not error for error in errors):
        raise ValueError("V7_BASELINE_DAILY_INPUT_ERRORS_INVALID")
    if errors:
        return dict(score, final_score=None), "NO_BUY" if decision == "NO_BUY" else "DATA_INSUFFICIENT"
    return dict(score), decision


def _finalize_daily_inputs(report, errors):
    """Bind native daily errors before any admission overlay/account authority."""
    score, decision = daily_input_outcome(report["envelope"]["score"],
        report["envelope"]["strategy"]["strategy_decision"], errors=errors)
    report["envelope"]["score"].update(score)
    report["envelope"]["strategy"]["strategy_decision"] = decision
    if errors:
        report["envelope"]["strategy"]["reason_codes"] = sorted(set(report["envelope"]["strategy"]["reason_codes"] + errors))
        from domain.les_v7.policy_r15 import BASELINE_VERSION
        from domain.les_v7.policy_r16 import BASELINE_VERSION as R16
        from domain.les_v7.policy_r17 import BASELINE_VERSION as R17
        from domain.les_v7.policy_r18 import BASELINE_VERSION as R18
        from domain.les_v7.policy_r19 import BASELINE_VERSION as R19
        from domain.les_v7.rule_baseline import TACTICAL_POLICIES
        if (report["diagnostics"].get("rule_baseline") or {}).get("policy", {}).get("policy_version") in TACTICAL_POLICIES:
            report["diagnostics"].pop("reference_entry_route", None)
        if quality_extension_schema(report["diagnostics"]["quality"].get("profile_id")):
            decision = report["envelope"]["strategy"]["strategy_decision"]
            report["diagnostics"]["readiness"]["execution"].update(
                state="READY_FOR_ACCOUNT_CHECK" if decision in {"BUY", "BUY_SMALL"} else "BLOCKED",
                candidate_decision=decision, order_authority=False)
    report["diagnostics"]["daily_input_errors"] = errors
    validate_report(report)
    return report


def project_report(report, *, price_scale=1000., include_report=True):
    """Validate and project native fields, optionally omitting the JSON cell.

    Streaming consumers already hold the verified report and need only its
    detached fields. Omitting the redundant serialization never skips native
    report validation; existing CSV and renderer callers retain their cell.
    """
    if type(include_report) is not bool:
        raise ValueError("V7_PROJECTION_INCLUDE_REPORT_INVALID")
    validate_report(report)
    return _project_validated_report(report, price_scale=price_scale, include_report=include_report)


def _project_validated_report(report, *, price_scale=1000., include_report=True):
    """Internal projection after full native or exact-JSON validation only.

    This is not a public validation switch. Mutable public inputs must use
    project_report(); JSON consumers use read_v7_report_json() first.
    """
    if type(include_report) is not bool:
        raise ValueError("V7_PROJECTION_INCLUDE_REPORT_INVALID")
    from domain.les_v7.plan_r19 import plan_from_record
    e, d = report["envelope"], report["diagnostics"]
    s, q, strategy = e["score"], e["quality"], e["strategy"]
    authorization = report.get("pae_authorization", {})
    authorized = authorization.get("authorized") is True
    plan = d["reference_plan"] or {}
    states = {r["gate_id"]: r["state"] for r in d["gate_receipts"]}
    risk = states.get("STRESS_RISK", "UNKNOWN") if plan else "UNKNOWN"
    if "EXECUTION_FAIL" in strategy["reason_codes"]: risk = "FAIL"
    def scaled(value): return value * price_scale if value is not None else None
    veto_key = "BASELINE_HARD_VETO" if d.get("rule_baseline") is not None else "hard_veto"
    veto = d["safety_caps"].get(veto_key) == 0
    no_veto = False if veto else True if all(states.get(key) == "PASS" for key in ("DATA_INTEGRITY", "SAFETY", "TRADING_STATUS")) else None
    falling_knife_state = "VETO" if veto else "CLEAR" if no_veto else "UNKNOWN"
    from domain.les_v7.rule_baseline import BOUND_SCORE_POLICIES
    if (d.get("rule_baseline") or {}).get("policy", {}).get("policy_version") in BOUND_SCORE_POLICIES:
        veto = bool(d["rule_baseline"]["technical_vetoes"])
        safety_states = [states.get(key) for key in ("DATA_INTEGRITY", "SAFETY", "TRADING_STATUS")]
        no_veto = False if veto or "FAIL" in safety_states else True if all(state == "PASS" for state in safety_states) else None
        falling_knife_state = ("VETO" if veto else "CLEAR" if all(states.get(key) == "PASS"
            for key in ("DATA_INTEGRITY", "SAFETY")) else "UNKNOWN")
    if d.get("rule_baseline") is not None:
        required = 72 if d["confirmed"] is True else 62
        from domain.les_v7.rule_baseline import LIMIT_POLICIES
        if d["rule_baseline"]["policy"]["policy_version"] in LIMIT_POLICIES:
            thresholds = d["rule_baseline"]["policy"]["unified_technical_policy"]
            required = thresholds["buy_minimum"] if strategy["strategy_decision"] == "BUY" else thresholds["small_minimum"]
    elif d.get("fr2") is not None:
        required = 75 if strategy["strategy_decision"] == "BUY" else 65
    else:
        required = 80 if d["confirmed"] is True else 72
    trigger = (None if d["confirmed"] is None and d["starter"] is None else
               d["confirmed"] is True or d["starter"] is True)
    reasons = "; ".join(strategy["reason_codes"]) or "NONE"
    # R2/R3 have explicit diagnostic state contracts. R1's historical compact
    # PASS/FAIL/UNKNOWN projection remains byte-for-byte compatible.
    quality_gate = (q["status"] if quality_extension_schema(d["quality"].get("profile_id")) else
                    "PASS" if q["status"] == "PASS" else "FAIL" if q["status"] == "REJECT" else "UNKNOWN")
    projection = {"LES v7 Raw Score": s["raw_score"], "LES v7 Penalty": s["total_penalty"],
        "LES v7 Adjusted Score": s["adjusted_score"], "LES v7 Safety Cap": s["effective_cap"],
        "LES v7 Final Score": s["final_score"], "LES v7 Score": s["final_score"],
        "LES v7 Decision": strategy["strategy_decision"], "LES v7 Context": e["event"]["context"],
        **({"LES v7 Report": json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)}
           if include_report else {}),
        "LES v7 Required Score": required, "LES v7 Trigger": trigger, "LES v7 Execution Eligible": authorized,
        "低點分數": s["final_score"], "低點決策": strategy["strategy_decision"], "低點理由": reasons,
        "LES Model Version": "7.0.0", "LES Parameter Version": report["parameter_version"],
        "Quality Score": q["quality_score"], "Quality Status": q["status"], "Data Confidence": q["data_confidence"],
        "Quality Gate": quality_gate,
        "No Veto": no_veto, "Falling Knife Veto": veto, "Falling Knife State": falling_knife_state,
        "Risk Gate": risk, "Risk Reasons": reasons,
        "RewardRisk": (plan_from_record(plan).net_rr if d.get("rule_baseline") is not None and plan
                       else d["factor_ledger"].get("rr_net")),
        "策略停損": scaled(plan.get("stop")), "策略停利": scaled(plan.get("target")),
        "ATR14": scaled(d["features"].get("atr")), "Entry Candidate": scaled(plan.get("entry")),
        "Entry Trigger Price": scaled(plan.get("band", [None, None])[1]),
        "Entry Rule": "REFERENCE_H20_NEXT_SESSION_BAND_REQUIRES_EXECUTION_CHECK",
        "Position Fraction": authorization.get("position_fraction", 0.), "建議股數": authorization.get("indicative_shares", 0),
        "Liquidity Status": "PASS" if authorized else "UNKNOWN" if d["features"].get("spread") is None else "NOT_AUTHORIZED"}
    return projection


def build_daily_v7(histories, *, feature_frames=None, evidence=None):
    if feature_frames is not None and len(feature_frames) != len(histories): raise ValueError("V7_FEATURE_COUNT_MISMATCH")
    if evidence is not None and len(evidence) != len(histories): raise ValueError("V7_EVIDENCE_COUNT_MISMATCH")
    return pd.DataFrame([project_report(daily_report(history,
        features=feature_frames[i] if feature_frames is not None else None,
        evidence=evidence[i] if evidence is not None else None)) for i, history in enumerate(histories)])
