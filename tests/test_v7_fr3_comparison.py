"""SYNTHETIC ONLY: complete scope/gaps are not forecasting efficacy evidence."""
from copy import deepcopy
from datetime import datetime
import json
import sqlite3

import pytest

from domain.les_v7.contracts import identity
from infrastructure.les_v7_fr3_comparison import (
    AUDIT_SCHEMA, SCHEMA, _compare_verified_inputs, audit_fr3_scope_sources,
    compare_fr3_four_arms, compare_fr3_predictions, compare_fr3_research,
    inventory_local_safety_sources, retain_fr3_comparison, retain_fr3_source_audit,
)
from tests.test_v7_dataset_scope import scope_inputs, freeze


def test_source_audit_keeps_every_security_without_granting_missing_gates(tmp_path):
    source = scope_inputs(tmp_path, count=3)
    path = freeze(source)
    before = {str(p): p.read_bytes() for p in source["root"].rglob("*") if p.is_file()}
    report = audit_fr3_scope_sources(path, evaluation_cutoff="2026-08-06T13:30:00+08:00",
        inspected_sessions=["2026-08-03"], allow_synthetic=True)
    assert report["schema"] == AUDIT_SCHEMA
    assert report["universe"] == source["symbols"]
    assert report["counts"]["fixed_universe"] == 3
    assert report["counts"]["financial_records"] == 0
    assert report["counts"]["restriction_records"] == 0
    assert [row["symbol"] for row in report["stocks"]] == source["symbols"]
    assert all(row["native_gate_status"] == "NOT_EVALUATED" for row in report["stocks"])
    assert all("HISTORICAL_FINANCIAL_ORIGINALS_NOT_IN_SOURCE_SHARD" in row["reason_codes"]
        for row in report["stocks"])
    assert report["sources_verified"] is True
    assert report["historical_information_set_certified"] is False
    assert report["fresh_confirmatory_status"] == "NOT_ASSESSABLE"
    assert report["last_h20_matured_session"] is None  # No guessed weekday horizon.
    assert report["order_authority"] is False
    assert report["audit_id"] == identity({k: v for k, v in report.items() if k != "audit_id"})
    assert {str(p): p.read_bytes() for p in source["root"].rglob("*") if p.is_file()} == before
    assert report == audit_fr3_scope_sources(path, evaluation_cutoff="2026-08-06T13:30:00+08:00",
        inspected_sessions=["2026-08-03"], allow_synthetic=True)


def test_synthetic_sources_cannot_masquerade_as_real_scope(tmp_path):
    source = scope_inputs(tmp_path, count=1)
    with pytest.raises(ValueError, match="SYNTHETIC_SOURCE_REQUIRES_EXPLICIT_OPT_IN"):
        audit_fr3_scope_sources(freeze(source), evaluation_cutoff="2026-08-06T13:30:00+08:00")


def test_source_audit_cannot_silently_use_changed_scope_bytes(tmp_path):
    source = scope_inputs(tmp_path, count=1)
    path = freeze(source)
    raw = json.loads(path.read_text())
    raw["body"]["universe"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="FREEZE_IDENTITY_INVALID"):
        audit_fr3_scope_sources(path, evaluation_cutoff="2026-08-06T13:30:00+08:00", allow_synthetic=True)


@pytest.mark.parametrize("inspected", [["2026-08-04", "2026-08-03"], ["2026-08-03", "2026-08-03"]])
def test_inspected_dates_are_explicit_not_relabelled_fresh(tmp_path, inspected):
    source = scope_inputs(tmp_path, count=1)
    with pytest.raises(ValueError, match="UNIQUE_ORDERED_INSPECTED_SESSIONS"):
        audit_fr3_scope_sources(freeze(source), evaluation_cutoff="2026-08-06T13:30:00+08:00",
            inspected_sessions=inspected, allow_synthetic=True)


def test_source_audit_uses_aware_cutoff_for_calendar_not_file_creation_time(tmp_path):
    source = scope_inputs(tmp_path, count=1)
    report = audit_fr3_scope_sources(freeze(source), evaluation_cutoff="2026-08-04T09:00:00+08:00",
        allow_synthetic=True)
    assert "2026-08-04" not in report["official_sessions"]
    assert report["evaluation_cutoff"] == "2026-08-04T01:00:00+00:00"


def test_retention_is_content_addressed_idempotent_and_does_not_edit_report(tmp_path):
    report = dict(schema=SCHEMA, status="NOT_ASSESSABLE", research_only=True,
        universe=["SYNTHETIC"], reason_codes=["MISSING_AFTER_COST_ACCOUNTS"])
    before = deepcopy(report)
    folder, completion = retain_fr3_comparison(report, tmp_path)
    bytes_before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert folder.name == identity(report)
    assert completion["status"] == "NOT_ASSESSABLE"
    assert retain_fr3_comparison(report, tmp_path) == (folder, completion)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == bytes_before
    assert report == before
    (folder / "comparison.json").write_text("{}")
    with pytest.raises(ValueError):
        retain_fr3_comparison(report, tmp_path)


def test_retention_requires_explicit_research_schema(tmp_path):
    with pytest.raises(ValueError, match="RESEARCH_COMPARISON_REQUIRED"):
        retain_fr3_comparison(dict(schema=SCHEMA, status="PASS"), tmp_path)


def comparison_inputs():
    """Tiny separately SYNTHETIC archive summaries, not real source clearance."""
    universe = ["1000", "1001", "1002"]
    dataset = dict(universe=[[s, "TWSE"] for s in universe], children=[dict(coverage=[
        dict(symbol=s, market="TWSE", prepared_rows=0 if s == "1001" else 3,
             build_reason_counts={"MISSING_SOURCE": 1} if s == "1001" else {}) for s in universe])])
    outcome = dict(symbols=universe, status="NOT_READY", request_id="SYNTHETIC_REQUEST",
        native_outcome=dict(native_job_id="SYNTHETIC_JOB", dataset_id="SYNTHETIC_DATASET", bundle_sha256=None))
    audit = dict(universe=universe, counts=dict(symbols_without_financial_records=3,
        symbols_without_restriction_records_or_ledger=3), fresh_confirmatory_status="NOT_ASSESSABLE")
    audit["audit_id"] = identity(audit)
    return dict(original=dict(sha256="SYNTHETIC_OLD_BUNDLE"), old_dataset=dataset,
        outcome=outcome, new_dataset=deepcopy(dataset), source_audit=audit)


def synthetic_fr3_report(symbol="1000"):
    body = dict(schema="les-v7-fr3-report/v1", symbol=symbol, market="TWSE",
        cutoff="2026-08-03T05:30:00+00:00", evidence_id="SYNTHETIC_EVIDENCE", model_id="SYNTHETIC_MODEL",
        decision=dict(strategy_decision="BUY_SMALL", reason_codes=[]),
        score=dict(raw_score=70., final_score=70.), plan=None)
    return dict(body, report_id=identity(body))


def test_full_scope_comparison_does_not_call_absent_research_zero_buys():
    inputs = comparison_inputs()
    before = deepcopy(inputs)
    result = _compare_verified_inputs(**inputs)
    assert inputs == before
    assert result["status"] == "NOT_ASSESSABLE"
    assert result["verified_decision_counts"] == dict(FR2=None, FR3=None)
    assert result["fixed_universe_count"] == 3
    assert [r["symbol"] for r in result["stocks"]] == inputs["outcome"]["symbols"]
    assert result["stocks"][1]["new"]["coverage"]["prepared_rows"] == 0
    assert all(row["new"]["decision_status"] == "NOT_EVALUATED" for row in result["stocks"])
    assert result["after_cost_comparison"]["status"] == "NOT_ASSESSABLE"


@pytest.mark.parametrize("damage", ["universe", "drop", "duplicate", "audit_hash"])
def test_full_scope_comparison_rejects_dropped_or_duplicate_scope(damage):
    inputs = comparison_inputs()
    if damage == "universe": inputs["new_dataset"]["universe"].pop()
    elif damage == "drop": inputs["new_dataset"]["children"][0]["coverage"].pop()
    elif damage == "duplicate": inputs["new_dataset"]["children"][0]["coverage"].append(
        inputs["new_dataset"]["children"][0]["coverage"][0])
    else: inputs["source_audit"]["counts"]["symbols_without_financial_records"] = 0
    with pytest.raises(ValueError, match="FR3_"):
        _compare_verified_inputs(**inputs)


def test_hash_only_report_is_not_counted_as_verified_native_buy():
    report = synthetic_fr3_report()
    before = deepcopy(report)
    result = _compare_verified_inputs(**comparison_inputs(), new_reports=[report])
    assert report == before
    assert result["stocks"][0]["new"]["decision_status"] == "UNVERIFIED"
    assert result["verified_decision_counts"]["FR3"] is None
    assert result["stocks"][0]["new"]["reports"][0]["decision"] == "BUY_SMALL"


def test_reexecution_validator_is_required_and_its_failures_are_not_ignored():
    checked = []
    report = synthetic_fr3_report()
    result = _compare_verified_inputs(**comparison_inputs(), new_reports=[report],
        new_report_validator=lambda value: checked.append(value["report_id"]))
    assert checked == [report["report_id"]]
    assert result["verified_decision_counts"]["FR3"] == {"BUY_SMALL": 1}
    assert result["status"] == "NOT_ASSESSABLE"  # Other missing stocks/sources remain.
    def reject(_):
        raise ValueError("SYNTHETIC_ACCOUNT_MISMATCH")
    with pytest.raises(ValueError, match="ACCOUNT_MISMATCH"):
        _compare_verified_inputs(**comparison_inputs(), new_reports=[report], new_report_validator=reject)


def test_changed_native_result_or_duplicate_cutoff_is_rejected_before_aggregation():
    report = synthetic_fr3_report()
    changed = deepcopy(report)
    changed["decision"]["strategy_decision"] = "BUY"
    with pytest.raises(ValueError, match="REPORT_HASH_MISMATCH"):
        _compare_verified_inputs(**comparison_inputs(), new_reports=[changed])
    with pytest.raises(ValueError, match="REPORT_DUPLICATE"):
        _compare_verified_inputs(**comparison_inputs(), new_reports=[report, report])


def four_arm_fr3_input():
    from tests.strategies.test_v7_fr2_comparison import comparison, seal
    body = comparison()["payload"]
    body["schema"] = "les-v7-fr3-comparison-input/v1"
    ref = body["conditions"]["reference_policy_id"]
    for name in ("M0", "M1"):
        model = body["models"][name]["payload"]
        model["reference_policy_id"] = ref
        body["models"][name] = seal(model)
    for name, record in body["arms"].items():
        arm = record["payload"]
        arm["model_id"] = body["models"][name[:2]]["sha256"]
        body["arms"][name] = seal(arm)
    return seal(body), ref


def test_four_arm_accounts_reuse_unchanged_acceptance_and_keep_missing_accounts_unknown():
    record, ref = four_arm_fr3_input()
    before = deepcopy(record)
    report = compare_fr3_four_arms(record, reference_policy_id=ref)
    assert record == before
    assert report["status"] == "NOT_ASSESSABLE"
    assert report["acceptance_kernel"] == "FR_ACCEPT_R2_UNCHANGED"
    assert all(row["status"] == "NOT_ASSESSABLE" for row in report["after_cost_comparisons"].values())
    assert report["model_pair_scope"] == "NEW_FR3_M0_M1_SAME_PLAN_LABELS_NOT_OLD_FR2_PREDICTOR"
    with pytest.raises(ValueError, match="ONE_NEW_REFERENCE_POLICY"):
        compare_fr3_four_arms(record, reference_policy_id="SYNTHETIC_OLD_PLAN")


def test_source_inventory_retention_does_not_publish_model_or_account_acceptance(tmp_path):
    source = scope_inputs(tmp_path, count=1)
    report = audit_fr3_scope_sources(freeze(source), evaluation_cutoff="2026-08-06T13:30:00+08:00",
        allow_synthetic=True)
    folder, receipt = retain_fr3_source_audit(report, tmp_path / "SYNTHETIC-audit")
    assert receipt["status"] == "SOURCE_INVENTORY_COMPLETED_NOT_RESEARCH_OR_PERFORMANCE_ACCEPTANCE"
    assert retain_fr3_source_audit(report, tmp_path / "SYNTHETIC-audit") == (folder, receipt)
    assert (folder / "source-audit.json").exists()
    assert not (folder / "frozen_bundle.json").exists()


def test_local_metadata_inventory_keeps_real_observation_time_and_does_not_claim_gates(tmp_path):
    mops, restrictions = tmp_path / "SYNTHETIC-mops.sqlite3", tmp_path / "SYNTHETIC-restrictions.sqlite3"
    old, future = "2026-08-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"
    with sqlite3.connect(mops) as connection:
        connection.execute("CREATE TABLE filings(symbol TEXT, available_at_epoch INTEGER, fetched_at TEXT, availability_source TEXT)")
        connection.executemany("INSERT INTO filings VALUES(?,?,?,?)", [
            ("1000", int(datetime.fromisoformat(old).timestamp()), old, "FIRST_OBSERVED_DOWNLOAD"),
            # A supposed old availability cannot override a real future fetch.
            ("1001", int(datetime.fromisoformat(old).timestamp()), future, "UNVERIFIED"),
            ("OUT_OF_SCOPE", int(datetime.fromisoformat(old).timestamp()), old, "UNVERIFIED")])
    with sqlite3.connect(restrictions) as connection:
        connection.execute("CREATE TABLE restriction_observations(market TEXT,symbol TEXT,available_at_ns INTEGER)")
        connection.execute("INSERT INTO restriction_observations VALUES(?,?,?)",
            ("TWSE", "1000", int(datetime.fromisoformat(future).timestamp()) * 1_000_000_000))
    before = mops.read_bytes(), restrictions.read_bytes()
    result = inventory_local_safety_sources(["1000", "1001"], mops_cache=mops,
        restriction_cache=restrictions, cutoffs=["2026-09-01T13:30:00+08:00"])
    assert result["financial_record_count"] == 2
    assert result["financial_cutoffs"][0]["observed_available_record_count"] == 1
    assert result["financial_cutoffs"][0]["no_observed_available_records"] == ["1001"]
    assert result["restriction_cutoffs"][0]["observed_record_count"] == 0
    assert result["native_gate_verdicts_emitted"] == result["original_documents_read"] == 0
    assert result["verified_original_publication_claim"] is False
    assert result["inventory_id"] == identity({k: v for k, v in result.items() if k != "inventory_id"})
    assert before == (mops.read_bytes(), restrictions.read_bytes())


def test_missing_local_databases_are_unknown_not_zero_successful_sources(tmp_path):
    result = inventory_local_safety_sources(["1000"], mops_cache=tmp_path / "MISSING-mops",
        restriction_cache=tmp_path / "MISSING-restrictions", cutoffs=["2026-09-01T13:30:00+08:00"])
    assert result["financial_record_count"] is None and result["restriction_metadata"] is None
    assert not result["financial_database_present"] and not result["restriction_database_present"]
    assert list(tmp_path.iterdir()) == []


def test_fr3_prediction_pair_reuses_native_holdout_without_old_plan_relabelling():
    from tests.test_v7_fr2_expansion import case
    from tests.test_v7_fr3_plan import reference
    from domain.les_v7.fr3_plan import FR3ReferencePlan, PLAN_VERSION
    rows, config, m0, m1 = case()
    policy = reference()
    config["reference_policy"] = deepcopy(policy)
    m0["reference_policy"] = deepcopy(policy)
    m1["reference_policy"] = deepcopy(policy)
    for row in rows:
        row["plan"] = FR3ReferencePlan("REFERENCE_H20", PLAN_VERSION, "SYNTHETIC_SOURCE", "REVERSAL",
            100., 95., 110., 2., (99., 100.), 1, .05, .05, signal_close=100., structure_id="a" * 64)
    before = deepcopy((config, m0, m1))
    result = compare_fr3_predictions(rows, config, m0, m1)
    assert result == compare_fr3_predictions(rows, config, m0, m1)
    assert result["native_metrics"]["M0"]["label_sha256"] == result["native_metrics"]["M1"]["label_sha256"]
    assert result["after_cost_account_status"] == "NOT_ASSESSABLE"
    assert result["historical_deployment_claim"] is False
    assert (config, m0, m1) == before
    legacy_rows = case()[0]
    with pytest.raises(ValueError, match="NEW_PLAN_LABELS"):
        compare_fr3_predictions(legacy_rows, config, m0, m1)


def test_public_comparison_verifies_both_native_archives_before_aggregation(tmp_path, monkeypatch):
    from infrastructure import les_v7_fr3_training
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository
    inputs = comparison_inputs()
    config = dict(evaluation_cutoff="2026-08-06T13:30:00+08:00", SYNTHETIC_ONLY=True)
    binding = dict(freeze_id="a"*64, scope_id="b"*64,
        source_evidence_ids=["c"*64], dataset_kind="SYNTHETIC_TEST_ONLY", training_config_id=identity(config))
    inputs["old_dataset"].update(binding)
    inputs["new_dataset"].update(deepcopy(binding))
    request_body = dict(schema=les_v7_fr3_training.REQUEST_SCHEMA, training_config=config)
    request = dict(payload=request_body, sha256=identity(request_body))
    inputs["outcome"]["request_id"] = request["sha256"]
    inputs["outcome"]["native_outcome"]["config_id"] = identity(config)
    inputs["source_audit"].update({key: value for key, value in binding.items() if key != "training_config_id"},
        schema=AUDIT_SCHEMA, sources_verified=True, evaluation_cutoff=config["evaluation_cutoff"])
    inputs["source_audit"]["audit_id"] = identity({key: value for key, value in inputs["source_audit"].items()
                                               if key != "audit_id"})
    old = tmp_path / "SYNTHETIC-old"
    new = tmp_path / "SYNTHETIC-fr3"
    old.mkdir(); (new / "native").mkdir(parents=True)
    (new / "fr3-training-request.json").write_text(json.dumps(request))
    bundle = dict(payload=dict(SYNTHETIC_ONLY=True), sha256=identity(dict(SYNTHETIC_ONLY=True)))
    dataset = dict(payload=inputs["old_dataset"], sha256=identity(inputs["old_dataset"]))
    (old / "frozen_bundle.json").write_text(json.dumps(bundle))
    (old / "dataset.json").write_text(json.dumps(dataset))
    (new / "native" / "scope-training-dataset.json").write_text(json.dumps(dataset))
    outcome = inputs["outcome"]
    outcome["native_root"] = str(new / "native")
    outcome["native_outcome"]["dataset_id"] = dataset["sha256"]
    calls = []
    def old_verify(path):
        calls.append(("old", path))
        return dict(bundle_sha256=bundle["sha256"], dataset_id=dataset["sha256"])
    def new_verify(path):
        calls.append(("new", path))
        return deepcopy(outcome)
    monkeypatch.setattr(LesV7TrainingRepository, "verify", old_verify)
    monkeypatch.setattr(les_v7_fr3_training, "verify_fr3_scope_training", new_verify)
    result = compare_fr3_research(old / "frozen_bundle.json", new, source_audit=inputs["source_audit"])
    assert [name for name, _ in calls] == ["old", "new"]
    assert result["fixed_universe_count"] == 3 and result["status"] == "NOT_ASSESSABLE"
    assert result["verified_decision_counts"] == {"FR2": None, "FR3": None}
    (new / "native" / "scope-training-dataset.json").write_text(json.dumps(dict(payload={}, sha256=dataset["sha256"])))
    with pytest.raises(ValueError, match="NATIVE_ARCHIVE_BINDING_MISMATCH"):
        compare_fr3_research(old / "frozen_bundle.json", new, source_audit=inputs["source_audit"])
