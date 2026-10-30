"""SYNTHETIC ONLY: R3/R4 wiring and compatibility, not real evidence repair."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from application.les_v7 import analyze_request, prepare_quality_context
from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from domain.les_v7.contracts import (identity, parameter_version, QUALITY_R2_PROFILE,
                                    QUALITY_R3_PROFILE, QUALITY_R4_PROFILE, quality_extension_schema)
from domain.les_v7.engine import evaluate
from domain.les_v7.quality_policy import default_quality_policy, r2_quality_policy, r3_quality_policy
from domain.les_v7.report import validate_report
from infrastructure.storage.les_v7_repository import LesV7SignalRepository
from tests.strategies.test_daily_v7_pipeline import synthetic_market
from tests.strategies.test_low_entry_score_v7 import POLICY, CUTOFF, bars, bundle, seal
from tests.strategies.test_v7_quality_policy import native_original


def request(policy=None):
    row, _ = native_original()
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row],
                   quality_policy=r3_quality_policy() if policy is None else policy)
    return dict(bars=bars(90).to_dict("records"), context=context)


def test_default_r4_registered_policy_preserves_frozen_r3_identity():
    r3 = r3_quality_policy()
    assert identity(r3) == "ee05f191ee7a3a9ea4be226aacd08d91709ee3dc3b716386e70ce8d34a04d2bc"
    assert r3["policy_sha256"] == "9083b33449f4860c4090c1525eb9a7735bdbbc3a1f7e9413f0f26e3371a51cfa"
    assert r3["source_mapping_sha256"] == "5cc874f10c2aff598c8c5887594ed3b5668001c7d3d4f6943aceaac9233638e0"
    r4 = default_quality_policy()
    assert r4["profile_id"] == QUALITY_R4_PROFILE
    assert r4["mapping_version"] == "QG_MAP_R4"
    assert r4["policy_id"] == "QG_V7_REPORTED_PERIOD_R4_20261013"
    assert r4["policy_sha256"] != r3["policy_sha256"]
    assert r4["source_mapping_sha256"] != r3["source_mapping_sha256"]


def test_implicit_r4_has_bound_reported_summary_and_disabled_execution():
    payload = request()
    payload["context"].pop("quality_policy")
    original = deepcopy(payload)
    report = analyze_request(payload)["reports"][0]
    q = report["diagnostics"]["quality"]
    assert q["profile_id"] == QUALITY_R4_PROFILE
    assert report["envelope"]["schema_version"] == "LES_V7_ENVELOPE_4"
    assert report["quality_extension_schema"] == "v7-quality-evidence/v4"
    assert report["parameter_version"] == parameter_version(QUALITY_R4_PROFILE)
    for key in ("core_reported_pass", "core_inputs_pass", "scoring_basis"):
        assert report["envelope"]["quality"][key] == q[key]
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["execution"]["live_enabled"] is False
    assert payload == original
    validate_report(report)
    assert report == analyze_request(request(default_quality_policy()))["reports"][0]


@pytest.mark.parametrize("damage", ["summary", "profile", "mapping", "coverage", "ledger", "safety"])
def test_r4_report_semantics_cannot_be_bypassed_by_rehashing(damage):
    report = analyze_request(request(default_quality_policy()))["reports"][0]
    q = report["diagnostics"]["quality"]
    if damage == "summary":
        report["envelope"]["quality"]["core_inputs_pass"] = not q["core_inputs_pass"]
    elif damage == "profile":
        q["profile_id"] = QUALITY_R3_PROFILE
    elif damage == "mapping":
        q["mapping_version"] = "QG_MAP_R3"
    elif damage == "coverage":
        q["coverage_total"] = .999
    elif damage == "ledger":
        q["metric_ledger"][0]["earned_points"] += .01
    else:
        q["financial_safety"]["state"] = "PASS"
        q["financial_safety"]["input_hash"] = "SYNTHETIC_ONLY_TAMPER"
    if damage != "summary":
        _rebind_quality_summary(report)
    with pytest.raises(ValueError):
        validate_report(report)


def test_r4_not_meaningful_zero_claim_must_match_bound_original_balance_source():
    from tests.strategies.test_v7_reported_quality import source_records
    payload = request(default_quality_policy())
    payload["context"]["symbol"] = "2330"
    payload["context"]["filings"] = source_records()
    report = analyze_request(payload)["reports"][0]
    q = report["diagnostics"]["quality"]
    assert q["financial_safety_source"]["current_liabilities"] > 0
    metrics = q["reported_result"]["metrics"]
    check = metrics["quality_metric_checks"]["current_ratio"]
    previous_points = check["points"]
    check.update(status="NOT_MEANINGFUL", fraction=0., points=0., available=True)
    metrics["quality_field_checks"][metrics["quality_reported_inputs"]["period"] + ":current_liabilities"]["value"] = 0.
    metrics["current_ratio"] = None
    metrics["quality_raw_score"] -= previous_points
    metrics["quality_final_score"] -= previous_points
    q["reported_result"]["components"]["balance_sheet"] -= previous_points
    q["reported_result"]["score"] -= previous_points
    q["quality_score"] -= previous_points
    q["earned_points"] -= previous_points
    for item in q["metric_ledger"]:
        if item["metric_id"] == "current_ratio":
            item.update(value_status="NOT_MEANINGFUL", mapped_score=0., earned_points=0., available=True)
    q["financial_inputs"] = deepcopy(metrics)
    q["field_checks"] = deepcopy(metrics["quality_field_checks"])
    _rebind_quality_summary(report)
    with pytest.raises(ValueError, match="NOT_MEANINGFUL_SOURCE"):
        validate_report(report)


def test_explicit_r3_is_independent_with_true_diagnostic_ttm_and_null_entry():
    payload = request()
    before = deepcopy(payload)
    report = analyze_request(payload)["reports"][0]
    q, summary = report["diagnostics"]["quality"], report["envelope"]["quality"]
    assert q["profile_id"] == QUALITY_R3_PROFILE
    assert q["mapping_version"] == "QG_MAP_R3"
    assert q["core_ttm_pass"] is False  # One real-format synthetic period cannot prove a TTM.
    assert summary["core_ttm_pass"] is False
    assert summary["core_reported_pass"] == q["core_reported_pass"]
    assert summary["core_inputs_pass"] == q["core_inputs_pass"]
    assert summary["scoring_basis"] == q["scoring_basis"] == "REPORTED_SAME_PERIOD"
    assert report["quality_extension_schema"] == "v7-quality-evidence/v3"
    assert report["envelope"]["schema_version"] == "LES_V7_ENVELOPE_3"
    assert report["parameter_version"].endswith(":" + QUALITY_R3_PROFILE)
    assert report["diagnostics"]["readiness"]["model"]["state"] == "NOT_READY"
    assert {gate["state"] for gate in report["diagnostics"]["gate_receipts"]} == {"UNKNOWN"}
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["execution"]["live_enabled"] is False
    assert project_report(report)["Quality Gate"] == q["status"]
    assert payload == before


@pytest.mark.parametrize("policy,extension", [(POLICY, None), (r2_quality_policy(), "v7-quality-evidence/v2")])
def test_explicit_old_policies_are_not_replaced_by_default_r3(policy, extension):
    payload = request(deepcopy(policy))
    raw = bars(90)
    expected = evaluate(raw, deepcopy(payload["context"]))
    actual = analyze_request(payload)["reports"][0]
    assert actual == expected
    assert actual.get("quality_extension_schema") == extension
    assert actual["envelope"]["schema_version"] == "LES_V7_ENVELOPE_1"
    assert "core_reported_pass" not in actual["envelope"]["quality"]
    assert actual["parameter_version"] != parameter_version(QUALITY_R3_PROFILE)


@pytest.mark.parametrize("policy", [POLICY, r2_quality_policy(), r3_quality_policy()])
def test_frozen_old_bundle_retains_policy_and_rejects_mixed_r3(policy):
    payload = request(deepcopy(policy))
    model = bundle()
    model["payload"]["quality_policy"] = deepcopy(policy)
    model = seal(model)
    payload["context"]["model_bundle"] = model
    result = analyze_request(payload)["reports"][0]
    assert result == evaluate(bars(90), deepcopy(payload["context"]))
    mixed = deepcopy(payload)
    mixed["context"]["quality_policy"] = default_quality_policy()
    with pytest.raises(ValueError, match="QUALITY_POLICY_BUNDLE_MISMATCH"):
        analyze_request(mixed)


def test_r1_r2_r3_r4_can_coexist_and_retry_without_overwriting_results(tmp_path):
    repository = LesV7SignalRepository(tmp_path / "SYNTHETIC_ONLY_RESEARCH")
    reports = [analyze_request(request(deepcopy(policy)))["reports"][0]
               for policy in (POLICY, r2_quality_policy(), r3_quality_policy(), default_quality_policy())]
    receipts = [repository.publish(report) for report in reports]
    assert len({receipt["signal_id"] for receipt in receipts}) == 4
    for report, receipt in zip(reports, receipts):
        assert repository.publish(report) == receipt
        assert repository.load(receipt["signal_id"]) == report


@pytest.mark.parametrize("field", ["core_reported_pass", "core_inputs_pass", "scoring_basis"])
def test_r3_quality_summary_tampering_is_rejected(field):
    report = analyze_request(request())["reports"][0]
    report["envelope"]["quality"][field] = "tampered"
    with pytest.raises(ValueError, match="V7_QUALITY_SUMMARY_MISMATCH"):
        validate_report(report)


def test_r3_rehashed_wrong_basis_is_not_accepted_as_another_reported_policy():
    report = analyze_request(request())["reports"][0]
    q = report["diagnostics"]["quality"]
    q["scoring_basis"] = "ANNUALIZED_OR_UNVERIFIED_TTM"
    report["envelope"]["quality"].update(scoring_basis=q["scoring_basis"], quality_result_id=identity(q))
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_CORE_BINDING_MISMATCH"):
        validate_report(report)


def _rebind_quality_summary(report):
    """Tamper tests deliberately repair hashes; hashes alone are not semantics."""
    q = report["diagnostics"]["quality"]
    summary = report["envelope"]["quality"]
    for field in ("status", "quality_score", "data_confidence", "coverage_total", "core_ttm_pass",
                  "gap_qualifier", "core_reported_pass", "core_inputs_pass", "scoring_basis"):
        summary[field] = q[field]
    summary.update(gap_penalty_points=q["missing_penalty_points"], quality_result_id=identity(q))


@pytest.mark.parametrize("damage", ["score", "confidence", "coverage", "penalty", "core_reported",
    "core_inputs", "native_complete", "native_score", "native_budget", "native_penalty", "native_dc", "metric_ledger"])
def test_r3_semantic_tampering_is_rejected_even_with_new_result_hash(damage):
    report = analyze_request(request())["reports"][0]
    q = report["diagnostics"]["quality"]
    native = q["reported_result"]
    assert native is not None  # Synthetic original has real-shaped native calculation evidence.
    if damage == "score": q["quality_score"] = (q["quality_score"] or 0.) + .1
    elif damage == "confidence": q["data_confidence"] += .1
    elif damage == "coverage": q["coverage_total"] = .999
    elif damage == "penalty": q["missing_penalty_points"] += .1
    elif damage == "core_reported": q["core_reported_pass"] = not q["core_reported_pass"]
    elif damage == "core_inputs": q["core_inputs_pass"] = not q["core_inputs_pass"]
    elif damage == "native_complete": q["native_quality_complete"] = not q["native_quality_complete"]
    elif damage == "native_score": native["score"] = (native["score"] or 0.) + .1
    elif damage == "native_budget": native["metrics"]["quality_budgets"]["balance_sheet"] += .1
    elif damage == "native_penalty": native["metrics"]["quality_penalty"] += .1
    elif damage == "native_dc": native["metrics"]["quality_dc_inputs"]["coverage"] += .1
    else: q["metric_ledger"][0]["earned_points"] += .1
    _rebind_quality_summary(report)
    with pytest.raises(ValueError):
        validate_report(report)


def test_r3_registered_policy_tampering_cannot_be_rehashed_into_valid_result():
    report = analyze_request(request())["reports"][0]
    policy = report["diagnostics"]["quality_policy"]
    policy["source_mapping_id"] = "SYNTHETIC_UNVERIFIED_REPLACEMENT"
    policy["policy_sha256"] = identity({key: value for key, value in policy.items() if key != "policy_sha256"})
    q = report["diagnostics"]["quality"]
    q.update(resolved_policy=deepcopy(policy), quality_policy_id=identity(policy))
    _rebind_quality_summary(report)
    with pytest.raises(ValueError, match="QUALITY_POLICY_HASH_OR_MAPPING_MISMATCH"):
        validate_report(report)


def test_r3_quality_cutoff_cannot_be_rehashed_to_another_evaluation_day():
    report = analyze_request(request())["reports"][0]
    report["diagnostics"]["quality"]["cutoff_at"] = "2026-10-13T13:30:00+08:00"
    _rebind_quality_summary(report)
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_CORE_BINDING_MISMATCH"):
        validate_report(report)


@pytest.mark.parametrize("field,value", [("symbol", "OTHER_SYNTHETIC_ONLY"), ("market", "TPEX")])
def test_r3_source_security_cannot_be_rehashed_into_another_report(field, value):
    report = analyze_request(request())["reports"][0]
    report["diagnostics"]["quality"]["financial_safety_source"][field] = value
    _rebind_quality_summary(report)
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_SECURITY_BINDING_MISMATCH"):
        validate_report(report)


@pytest.mark.parametrize("policy_factory", [r3_quality_policy, default_quality_policy])
def test_reported_daily_public_cli_full_result_parity_and_stale_readiness(tmp_path, capsys, policy_factory):
    from interface.les_v7 import main
    row, _ = native_original()
    market = replace(synthetic_market(), les_v7_bundle_json=None,
        fundamental_snapshots={row["symbol"]: {"sector": "24", "les_v6_filings": [row],
            "les_v6_mops_filings": [row], "les_v7_context": {"quality_policy": policy_factory()}}})
    history = MarketHistoryAssembler().build(market, 0)
    raw = history.reindex(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Trades", "Amount",
                                   "Bid", "Ask", "PE", "TAIEX Close"]).copy()
    raw.attrs = {}
    for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        raw[field] /= 1000.
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], model_bundle=None,
                   external_evidence_ids=[], receipts={}, quality_policy=policy_factory())
    payload = dict(bars=raw.to_dict("records"), context=context)
    expected = analyze_request(payload)
    actual = daily_report(history)
    assert actual.pop("pae_authorization")["authorized"] is False
    assert actual["diagnostics"].pop("daily_input_errors") == []
    assert actual == expected["reports"][0]
    path = tmp_path / "SYNTHETIC_ONLY_R3_request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "SYNTHETIC_ONLY_R3_result.json"
    assert main([str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text()) == expected
    history.attrs["les_v6_analysis_session"] = "2026-10-13"
    stale = daily_report(history)
    assert stale["diagnostics"]["daily_input_errors"] == ["STALE_MARKET_DATA"]
    assert stale["diagnostics"]["readiness"]["execution"]["state"] == "BLOCKED"
    assert stale["diagnostics"]["readiness"]["execution"]["order_authority"] is False
    assert stale["pae_authorization"]["authorized"] is False


def test_r3_source_attachment_preserves_original_and_old_policy_shapes():
    row, _ = native_original()
    proof = row.pop("audit_evidence")
    payload = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], quality_policy=r3_quality_policy())
    prepare_quality_context(payload, financial_audits={row["content_sha256"]: proof})
    assert payload["quality_policy"]["profile_id"] == QUALITY_R3_PROFILE
    assert payload["filings"][0]["audit_evidence"] == proof
    assert "audit_evidence" not in row
    assert quality_extension_schema(QUALITY_R2_PROFILE) == "v7-quality-evidence/v2"
    assert quality_extension_schema(QUALITY_R3_PROFILE) == "v7-quality-evidence/v3"
    assert quality_extension_schema(QUALITY_R4_PROFILE) == "v7-quality-evidence/v4"
    assert quality_extension_schema(None) is None


@pytest.mark.parametrize("policy,bad_version", [(None, "LES_V7_ENVELOPE_1"),
                                               (r2_quality_policy(), "LES_V7_ENVELOPE_3"),
                                               (POLICY, "LES_V7_ENVELOPE_3")])
def test_envelope_schema_version_is_bound_to_quality_profile(policy, bad_version):
    report = analyze_request(request(policy))["reports"][0]
    report["envelope"]["schema_version"] = bad_version
    with pytest.raises(ValueError, match="V7_REPORT_EVENT_BINDING_MISMATCH"):
        validate_report(report)


def test_reported_schema_local_refs_and_native_envelope_shape():
    """Use the project's stdlib shape checks, plus bounded local-ref resolution.

    This is not an alternative production JSON-Schema engine. Cross-field and
    financial semantics are checked by the native report validator above.
    """
    docs = Path(__file__).resolve().parents[1] / "docs"
    old = json.loads((docs / "les-v7-signal-envelope.schema.json").read_text())
    schema = json.loads((docs / "les-v7-reported-envelope.schema.json").read_text())

    def resolve(definition):
        if "$ref" not in definition:
            return definition
        filename, fragment = definition["$ref"].split("#", 1)
        assert filename == "les-v7-signal-envelope.schema.json"
        selected = old
        for key in fragment.removeprefix("/").split("/"):
            selected = selected[int(key)] if isinstance(selected, list) else selected[key]
        return selected

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert isinstance(resolve(value), dict)
            for node in value.values(): check_refs(node)
        elif isinstance(value, list):
            for node in value: check_refs(node)

    check_refs(schema)
    envelope = analyze_request(request())["reports"][0]["envelope"]
    assert set(envelope) == set(schema["required"])
    for section, definition in schema["properties"].items():
        definition = resolve(definition)
        value = envelope[section]
        if "const" in definition: assert value == definition["const"]
        if definition.get("type") != "object": continue
        assert set(value) == set(definition["required"])
        for key, child in definition["properties"].items():
            child = resolve(child)
            if "enum" in child: assert value[key] in child["enum"]
            if "const" in child: assert value[key] == child["const"]
            if value[key] is not None and "minimum" in child: assert value[key] >= child["minimum"]
            if value[key] is not None and "maximum" in child: assert value[key] <= child["maximum"]
    for section in ("spec_id", "model", "event", "localization", "score", "strategy", "execution"):
        assert resolve(schema["properties"][section]) == old["properties"][section]
    assert schema["properties"]["quality"]["properties"]["gap_penalty_points"]["maximum"] == 25
    qualification = schema["allOf"][0]["then"]["properties"]["quality"]["properties"]
    assert qualification["core_inputs_pass"] == {"const": True}
    assert qualification["core_reported_pass"] == {"const": True}
    assert "core_ttm_pass" not in qualification


def test_r4_schema_has_distinct_identity_and_preserves_reported_contract_refs():
    docs = Path(__file__).resolve().parents[1] / "docs"
    r3 = json.loads((docs / "les-v7-reported-envelope.schema.json").read_text())
    r4 = json.loads((docs / "les-v7-reported-r4-envelope.schema.json").read_text())
    envelope = analyze_request(request(default_quality_policy()))["reports"][0]["envelope"]
    assert envelope["schema_version"] == r4["properties"]["schema_version"]["const"] == "LES_V7_ENVELOPE_4"
    assert set(envelope) == set(r4["required"]) == set(r3["required"])
    for section in envelope:
        if section == "schema_version": continue
        assert r4["properties"][section] == {"$ref": "les-v7-reported-envelope.schema.json#/properties/" + section}
        if r3["properties"][section].get("type") == "object":
            assert set(envelope[section]) == set(r3["properties"][section]["required"])
    assert r4["allOf"] == [{"$ref": "les-v7-reported-envelope.schema.json#/allOf/" + str(index)}
                           for index in range(4)]
