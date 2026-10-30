"""SYNTHETIC source-identity binding only; no model fit or performance claim."""
from copy import deepcopy
import json

import pytest

from domain.les_v7.contracts import identity
from infrastructure import les_v7_fr3_comparison as comparison
from infrastructure import les_v7_fr3_training as training
from infrastructure import les_v7_scope_training as native
from infrastructure.storage.les_v7_training import LesV7TrainingRepository
from tests.test_v7_fr3_comparison import comparison_inputs
from tests.test_v7_fr3_plan import reference
from tests.test_v7_fr3_training import inputs
from tests.test_v7_training_publication import Guard


def seal_audit(audit):
    audit["audit_id"] = identity({key: value for key, value in audit.items() if key != "audit_id"})


def test_existing_v1_native_audit_binds_to_actual_prepared_sources_without_fitting(inputs, monkeypatch):
    monkeypatch.setattr(native, "_train_prepared", lambda *_a, **_k: pytest.fail("no model fit"))
    manifest, build, config, root = inputs
    training.prepare_fr3_scope(manifest, build, config, root, allow_synthetic=True,
        guard_factory=Guard, clock=lambda: "2026-10-23T14:00:00+08:00")
    dataset = json.loads((root / "native" / "scope-training-dataset.json").read_text())["payload"]
    audit = comparison.audit_fr3_scope_sources(manifest,
        evaluation_cutoff=config["evaluation_cutoff"], allow_synthetic=True)
    before = deepcopy((dataset, audit, config))
    assert audit["schema"] == comparison.AUDIT_SCHEMA == "les-v7-fr3-scope-source-audit/v1"
    assert audit["source_evidence_ids"] == dataset["source_evidence_ids"]
    assert audit["source_evidence_ids"] == [child["evidence_id"] for child in dataset["children"]]
    assert comparison._bind_source_audit_to_dataset(audit, dataset, config) is None
    assert (dataset, audit, config) == before


@pytest.fixture
def public_case(tmp_path, monkeypatch):
    """Mock only completed archive verification; real public binding executes."""
    given = comparison_inputs()
    config = dict(evaluation_cutoff="2026-08-06T13:30:00+08:00",
        reference_policy=reference())
    body = dict(schema=training.REQUEST_SCHEMA, training_config=config)
    request = dict(payload=body, sha256=identity(body))
    data = deepcopy(given["new_dataset"])
    data.update(freeze_id="a"*64, scope_id="b"*64, dataset_kind="SYNTHETIC_TEST_ONLY",
        source_evidence_ids=["c"*64, "d"*64], training_config_id=identity(config))
    old = tmp_path / "SYNTHETIC-old"
    new = tmp_path / "SYNTHETIC-new"
    old.mkdir(); (new / "native").mkdir(parents=True)
    bundle = dict(payload=dict(SYNTHETIC_ONLY=True))
    bundle["sha256"] = identity(bundle["payload"])
    record = dict(payload=data, sha256=identity(data))
    (old / "frozen_bundle.json").write_text(json.dumps(bundle))
    (old / "dataset.json").write_text(json.dumps(record))
    (new / "native" / "scope-training-dataset.json").write_text(json.dumps(record))
    (new / "fr3-training-request.json").write_text(json.dumps(request))
    outcome = given["outcome"]
    outcome.update(request_id=request["sha256"], native_root=str(new / "native"))
    outcome["native_outcome"].update(dataset_id=record["sha256"], config_id=identity(config))
    audit = given["source_audit"]
    audit.update(schema=comparison.AUDIT_SCHEMA, freeze_id=data["freeze_id"], scope_id=data["scope_id"],
        dataset_kind=data["dataset_kind"], sources_verified=True,
        source_evidence_ids=list(data["source_evidence_ids"]), evaluation_cutoff=config["evaluation_cutoff"])
    seal_audit(audit)
    calls = []
    def verify_old(_):
        calls.append("old-native")
        return dict(bundle_sha256=bundle["sha256"], dataset_id=record["sha256"])
    def verify_new(_):
        calls.append("new-native")
        return deepcopy(outcome)
    monkeypatch.setattr(LesV7TrainingRepository, "verify", verify_old)
    monkeypatch.setattr(training, "verify_fr3_scope_training", verify_new)
    return dict(old=old, new=new, audit=audit, config=config, outcome=outcome,
        request=request, calls=calls)


def execute(case, **options):
    return comparison.compare_fr3_research(case["old"] / "frozen_bundle.json", case["new"],
        source_audit=case["audit"], **options)


def test_public_entry_accepts_complete_v1_binding_and_keeps_unknowns(public_case):
    before = deepcopy(public_case["audit"])
    result = execute(public_case)
    assert public_case["audit"] == before
    assert public_case["calls"] == ["old-native", "new-native"]
    assert result["status"] == "NOT_ASSESSABLE"
    assert result["verified_decision_counts"] == {"FR2": None, "FR3": None}
    assert result["source_bindings"]["source_audit_id"] == before["audit_id"]
    assert result["fixed_universe_count"] == 3


def test_same_cutoff_in_utc_is_equivalent_not_a_new_time_policy(public_case):
    public_case["audit"]["evaluation_cutoff"] = "2026-08-06T05:30:00+00:00"
    seal_audit(public_case["audit"])
    assert execute(public_case)["status"] == "NOT_ASSESSABLE"


@pytest.mark.parametrize("damage", ["old_freeze_same_universe", "scope", "cutoff", "naive_cutoff",
    "source_order", "source_drop", "source_extra", "source_change", "kind", "not_verified",
    "missing_freeze", "missing_scope", "missing_sources", "missing_cutoff", "unknown_schema",
    "missing_schema", "hash"])
def test_rehashed_old_or_unbound_audit_is_rejected_by_public_entry(public_case, damage):
    audit = public_case["audit"]
    if damage == "old_freeze_same_universe": audit["freeze_id"] = "e"*64
    elif damage == "scope": audit["scope_id"] = "e"*64
    elif damage == "cutoff": audit["evaluation_cutoff"] = "2026-08-07T13:30:00+08:00"
    elif damage == "naive_cutoff": audit["evaluation_cutoff"] = "2026-08-06T13:30:00"
    elif damage == "source_order": audit["source_evidence_ids"].reverse()
    elif damage == "source_drop": audit["source_evidence_ids"].pop()
    elif damage == "source_extra": audit["source_evidence_ids"].append("e"*64)
    elif damage == "source_change": audit["source_evidence_ids"][0] = "e"*64
    elif damage == "kind": audit["dataset_kind"] = "RESEARCH"
    elif damage == "not_verified": audit["sources_verified"] = False
    elif damage == "missing_freeze": audit.pop("freeze_id")
    elif damage == "missing_scope": audit.pop("scope_id")
    elif damage == "missing_sources": audit.pop("source_evidence_ids")
    elif damage == "missing_cutoff": audit.pop("evaluation_cutoff")
    elif damage == "unknown_schema": audit["schema"] = "les-v7-fr3-scope-source-audit/v999"
    elif damage == "missing_schema": audit.pop("schema")
    else: audit["counts"]["symbols_without_financial_records"] = 0
    if damage != "hash": seal_audit(audit)
    assert audit["universe"] == ["1000", "1001", "1002"]
    with pytest.raises(ValueError, match="FR3_SOURCE_AUDIT_"):
        execute(public_case)


@pytest.mark.parametrize("damage", ["hash", "request_identity", "config_identity", "schema", "not_object"])
def test_cutoff_must_come_from_the_exact_verified_training_request(public_case, damage):
    request = public_case["request"]
    if damage in {"hash", "config_identity"}:
        request["payload"]["training_config"]["evaluation_cutoff"] = "2026-08-07T13:30:00+08:00"
    elif damage == "request_identity": request["payload"]["SYNTHETIC_EXTRA"] = True
    elif damage == "schema": request["payload"]["schema"] = "UNKNOWN"
    else: request["payload"] = []
    if damage != "hash": request["sha256"] = identity(request["payload"])
    if damage == "config_identity": public_case["outcome"]["request_id"] = request["sha256"]
    (public_case["new"] / "fr3-training-request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="FR3_COMPARISON_TRAINING_REQUEST_MISMATCH"):
        execute(public_case)


def test_after_cost_callback_uses_same_bound_config_and_cannot_bypass_source_check(public_case, monkeypatch):
    observed = []
    def accounts(value, *, reference_policy_id):
        observed.append(reference_policy_id)
        return dict(status="NOT_ASSESSABLE", SYNTHETIC_ONLY=True)
    monkeypatch.setattr(comparison, "compare_fr3_four_arms", accounts)
    execute(public_case, after_cost_input={"SYNTHETIC_ONLY": "no real account"})
    assert observed == [identity(public_case["config"]["reference_policy"])]
    public_case["audit"]["freeze_id"] = "e"*64
    seal_audit(public_case["audit"])
    with pytest.raises(ValueError, match="FR3_SOURCE_AUDIT_NATIVE_BINDING_MISMATCH"):
        execute(public_case, after_cost_input={})
    assert len(observed) == 1
