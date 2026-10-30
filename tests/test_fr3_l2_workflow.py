"""Synthetic routing/control fixtures only; never market or model evidence."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from domain.les_v7.contracts import canonical, identity
from examples import run_fr3_l2 as workflow


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    from domain.les_v7.fr3_l2 import PLAN_VERSION
    freeze, build, config, original, audit, root = [tmp_path / name for name in
        ("SYNTHETIC-freeze/freezes/fixed/manifest.json", "SYNTHETIC-build.json", "SYNTHETIC-config.json",
         "SYNTHETIC-original/frozen_bundle.json", "SYNTHETIC-audit.json", "SYNTHETIC-output")]
    cfg = dict(reference_policy=dict(version=PLAN_VERSION), evaluation_cutoff="2026-10-23T13:30:00+08:00")
    dataset = dict(freeze_id="d"*64, scope_id="e"*64, dataset_kind="SYNTHETIC_TEST_ONLY",
        source_evidence_ids=["f"*64], universe=[["3010", "TWSE"]])
    report = dict(schema="les-v7-fr3-scope-source-audit/v1", sources_verified=True,
        **{k: dataset[k] for k in ("freeze_id", "scope_id", "dataset_kind", "source_evidence_ids")},
        universe=["3010"], previously_inspected_sessions=[], evaluation_cutoff=cfg["evaluation_cutoff"])
    report["audit_id"] = identity(report)
    for path, value in ((freeze, {"SYNTHETIC_ONLY": True}), (build, {"SYNTHETIC_ONLY": True}),
                        (config, cfg), (original, {"SYNTHETIC_ONLY": True}), (audit, report)):
        save(path, value)
    calls, state = [], dict(sufficient=True, ready=True, m0_error=None, prepare_error=None)

    @dataclass
    class Versions:
        code: str = "SYNTHETIC_ONLY"

    class Guard:
        versions = Versions()
        def retain(self, _): return dict(source="a"*64)
        def check(self): return self.versions

    monkeypatch.setattr(workflow, "ResearchVersionGuard", Guard)

    def prepare(*_a, **_kw):
        calls.append("prepare")
        if state["prepare_error"]:
            raise ValueError(state["prepare_error"])
        save(root / "training/native/scope-training-dataset.json", dict(payload=dataset, sha256=identity(dataset)))
        return dict(status="DATASET_PREPARED")

    def capacity(_):
        calls.append("capacity")
        return dict(capacity_id="1"*64, dataset_id=identity(dataset), count_capacity_allows_fit=state["sufficient"])

    def fit(*_a, **_kw):
        calls.append("fit")
        return dict(status="READY_TO_FREEZE" if state["ready"] else "NOT_READY")

    def verify(_):
        calls.append("verify")
        return dict(status="READY_TO_FREEZE" if state["ready"] else "NOT_READY",
            native_outcome=dict(counts=dict(weight_rows=600, weight_contexts=dict(LEADERSHIP_CONTINUATION=110)),
                bundle_path="models/" + "2"*64 + "/frozen_bundle.json"))

    def m0(*_):
        calls.append("m0")
        if state["m0_error"]:
            raise ValueError(state["m0_error"])
        return dict(model_id="3"*64)

    def study(*_):
        calls.append("study")
        return root / "study", dict(study_id="4"*64, status="NOT_ASSESSABLE")

    from examples import compare_fr3_study
    monkeypatch.setattr(workflow, "prepare_fr3_scope", prepare)
    monkeypatch.setattr(workflow, "audit_fr3_prepared_capacity", capacity)
    monkeypatch.setattr(workflow, "retain_fr3_capacity", lambda *_: None)
    monkeypatch.setattr(workflow, "train_fr3_scope", fit)
    monkeypatch.setattr(workflow, "verify_fr3_scope_training", verify)
    monkeypatch.setattr(workflow, "publish_m0", m0)
    monkeypatch.setattr(compare_fr3_study, "run_fr3_study", study)
    return dict(args=(freeze, build, config, original, audit, root), calls=calls, state=state, root=root)


def test_routes_all_stages_without_claiming_account_or_live_acceptance(case):
    before = Path(case["args"][3]).read_bytes()
    result = workflow.run(*case["args"])
    assert case["calls"] == ["prepare", "capacity", "fit", "verify", "m0", "study"]
    assert result["native_counts"]["weight_contexts"]["LEADERSHIP_CONTINUATION"] == 110
    assert result["after_cost_accounts"] == "NOT_ASSESSABLE"
    assert result["approved_for_live"] is result["mail_sent"] is result["fresh_confirmatory"] is False
    assert Path(case["args"][3]).read_bytes() == before


def test_insufficient_capacity_never_fits_or_manufactures_oof(case):
    case["state"]["sufficient"] = False
    result = workflow.run(*case["args"])
    assert case["calls"] == ["prepare", "capacity"]
    assert result["native_counts"] is result["m0_model_id"] is result["study_id"] is None
    assert result["status"] == "STOPPED_INSUFFICIENT_CAPACITY_NO_FIT"


def test_native_not_ready_never_uses_old_bundle(case):
    case["state"]["ready"] = False
    result = workflow.run(*case["args"])
    assert result["status"] == "STOPPED_NATIVE_NOT_READY"
    assert case["calls"] == ["prepare", "capacity", "fit", "verify"]


def test_known_m0_readiness_gap_is_not_silently_given_m1_weights(case):
    case["state"]["m0_error"] = "FR2_M0_INSUFFICIENT_OWN_PAST_CONTEXT_WEIGHT_ROWS"
    result = workflow.run(*case["args"])
    assert result["m0_model_id"] is None
    assert result["m0_gap"] == case["state"]["m0_error"]
    assert case["calls"][-1] == "study"


def test_corrupt_m0_or_prepare_failure_never_publishes_completion(case):
    case["state"]["m0_error"] = "CORRUPT_REFERENCE"
    with pytest.raises(ValueError, match="CORRUPT_REFERENCE"):
        workflow.run(*case["args"])
    assert not (case["root"] / "completion.json").exists()
    assert case["calls"][-1] == "m0"


def test_interrupted_preparation_can_retry_with_same_identity(case):
    case["state"]["prepare_error"] = "SYNTHETIC_INTERRUPTION"
    with pytest.raises(ValueError, match="SYNTHETIC_INTERRUPTION"):
        workflow.run(*case["args"])
    original = (case["root"] / "request.json").read_bytes()
    case["state"]["prepare_error"] = None
    result = workflow.run(*case["args"])
    assert result["status"] == "RESEARCH_COMPLETE_NOT_PORTFOLIO_ACCEPTANCE"
    assert (case["root"] / "request.json").read_bytes() == original
    assert len(list((case["root"] / "attempts").iterdir())) == 2


def test_changed_input_rejects_retry_before_any_native_work(case):
    workflow.run(*case["args"])
    count = len(case["calls"])
    save(case["args"][1], dict(SYNTHETIC_CHANGED=True))
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        workflow.run(*case["args"])
    assert len(case["calls"]) == count


def test_legacy_settings_are_not_silently_upgraded(case):
    save(case["args"][2], dict(reference_policy=dict(version="FR3_SOURCE_BOUND_LIMIT_H20_R1")))
    with pytest.raises(ValueError, match="FR3_L2_EXPLICIT_SETTINGS_REQUIRED"):
        workflow.run(*case["args"])
    assert not case["calls"]
