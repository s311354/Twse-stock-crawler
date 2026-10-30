"""SYNTHETIC_TEST_ONLY contracts; real fixed archive run is documented separately."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from domain.les_v7.contracts import CONTEXTS, identity
from domain.les_v7.research import fit_factor_weights
from domain.les_v7.training import _oof, prepare_events
from infrastructure.les_v7_fr2_m0 import (_original_binding, load_m0_bundle, publish_m0, weight_rows_from_oof)
from infrastructure.research_objects import ResearchObjects
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset


@pytest.fixture(scope="module")
def case():
    config = synthetic_config()
    rows, _ = prepare_events(synthetic_dataset(config), config)
    predictions, _, _ = _oof(rows, config, ["slope5"], [[-10., 10.]])
    return rows, predictions, config


def test_m0_factor_rows_are_from_own_oof_and_native_solver_is_deterministic(case):
    rows, predictions, config = deepcopy(case)
    before = identity([dict(r, plan=asdict(r["plan"])) for r in rows]), identity(predictions)
    selected, excluded = weight_rows_from_oof(rows, predictions, config)
    assert {r["context"] for r in selected} == set(CONTEXTS)
    assert excluded["NO_STRICTLY_PAST_VALID_CALIBRATOR"] > 0
    own = {p["prediction_id"] for p in predictions}
    assert all(r["prediction_id"] in own for r in selected)
    first = fit_factor_weights(selected, fit_cutoff=config["fit_cutoff"], **config["weight_solver"])
    assert first["status"] == "CONVERGED"
    assert first == fit_factor_weights(selected, fit_cutoff=config["fit_cutoff"], **config["weight_solver"])
    assert before == (identity([dict(r, plan=asdict(r["plan"])) for r in rows]), identity(predictions))


def test_future_labels_are_excluded_not_used_for_weights(case):
    rows, predictions, config = deepcopy(case)
    selected, _ = weight_rows_from_oof(rows, predictions, config)
    target = next(p for p in predictions if p["prediction_id"] == selected[0]["prediction_id"])
    target["label_available_at"] = "2027-01-01T00:00:00Z"
    later, exclusions = weight_rows_from_oof(rows, predictions, config)
    assert target["prediction_id"] not in {r["prediction_id"] for r in later}
    assert exclusions["LABEL_NOT_MATURE_AT_WEIGHT_FIT"] >= 1


@pytest.mark.parametrize("field,value,match", [
    ("head_fit_cutoff", "2027-01-01T00:00:00Z", "STAGE_C_TIME_LEAKAGE"),
    ("calibration_fit_cutoff", "2027-01-01T00:00:00Z", "STAGE_C_CALIBRATION_TIME_LEAKAGE"),
])
def test_past_oof_time_proofs_are_revalidated_by_native_solver(case, field, value, match):
    rows, predictions, config = deepcopy(case)
    selected, _ = weight_rows_from_oof(rows, predictions, config)
    selected[0][field] = value
    with pytest.raises(ValueError, match=match):
        fit_factor_weights(selected, fit_cutoff=config["fit_cutoff"], **config["weight_solver"])


def test_missing_context_rows_not_replaced_with_m1_or_prior(case):
    rows, predictions, config = deepcopy(case)
    predictions = [p for p in predictions if next(r["context"] for r in rows if r["row_id"] == p["row_id"]) != "REVERSAL"]
    with pytest.raises(ValueError, match="INSUFFICIENT_OWN_PAST_CONTEXT"):
        weight_rows_from_oof(rows, predictions, config)


def test_duplicate_predictions_and_source_rows_are_rejected(case):
    rows, predictions, config = deepcopy(case)
    with pytest.raises(ValueError, match="DUPLICATE_OOF_ROW"):
        weight_rows_from_oof(rows, predictions + predictions[:1], config)
    with pytest.raises(ValueError, match="DUPLICATE_PREPARED_ROW"):
        weight_rows_from_oof(rows + rows[:1], predictions, config)


def test_publication_cannot_write_into_original_model_archive(tmp_path):
    archive = tmp_path / "original" / "frozen_bundle.json"
    with pytest.raises(ValueError, match="MUST_NOT_MODIFY_ORIGINAL"):
        publish_m0(archive, archive.parent / "nested")
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_loader_rejects_missing_completion_not_a_bare_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_m0_bundle(tmp_path)


def test_original_model_packet_and_file_bytes_have_distinct_preserved_identities(tmp_path):
    body = dict(example="SYNTHETIC_TEST_ONLY")
    record = dict(payload=body, sha256=identity(body))
    original = tmp_path / "original.json"
    encoded = (json.dumps(record, indent=2) + "\n").encode()
    original.write_bytes(encoded)
    objects = ResearchObjects(tmp_path / "objects")
    binding = _original_binding(original, objects)
    assert binding == dict(original_bundle_sha256=identity(body), original_bundle_packet_id=identity(record),
        original_bundle_file_sha256=hashlib.sha256(encoded).hexdigest())
    assert len(set(binding.values())) == 3
    assert objects.get(binding["original_bundle_file_sha256"]) == encoded
    original.write_text(json.dumps(record, separators=(",", ":")))
    changed = _original_binding(original, objects)
    assert changed["original_bundle_sha256"] == binding["original_bundle_sha256"]
    assert changed["original_bundle_packet_id"] == binding["original_bundle_packet_id"]
    assert changed["original_bundle_file_sha256"] != binding["original_bundle_file_sha256"]


def test_original_model_content_hash_must_match_before_preservation(tmp_path):
    original = tmp_path / "original.json"
    original.write_text(json.dumps(dict(payload=dict(example="SYNTHETIC_TEST_ONLY"), sha256="0" * 64)))
    with pytest.raises(ValueError, match="ORIGINAL_BUNDLE_HASH_MISMATCH"):
        _original_binding(original, ResearchObjects(tmp_path / "objects"))


def completion_case(tmp_path):
    """SYNTHETIC I/O protocol only; never a financial/model-validity fixture."""
    folder, job = tmp_path / "model", tmp_path / "job"
    proposal = dict(schema="SYNTHETIC_TEST_ONLY", model_id="SYNTHETIC_MODEL", request_id="SYNTHETIC_REQUEST",
        artifacts={"SYNTHETIC.json": "SYNTHETIC_ARTIFACT"}, calculation_objects={"source": "SYNTHETIC_SOURCE"},
        completed_at="2026-01-01T01:00:00Z", research_only=True, order_authority=False)
    def verify(target):
        assert target == folder
        receipt = json.loads((folder / "completion.json").read_text())
        return dict(payload=dict(trained_until="2026-01-01T00:30:00Z")), dict(receipt)
    return folder, job, proposal, verify


def test_crash_after_model_receipt_recovers_exact_original_receipt(tmp_path, monkeypatch):
    import infrastructure.les_v7_fr2_m0 as module
    folder, job, proposal, verify = completion_case(tmp_path)
    write = module.write_once
    def interrupted(target, value):
        if target == job / "completion.json":
            raise RuntimeError("SYNTHETIC_PROCESS_EXIT")
        return write(target, value)
    monkeypatch.setattr(module, "write_once", interrupted)
    with pytest.raises(RuntimeError, match="PROCESS_EXIT"):
        module._finish_completion(folder, job, proposal, verify=verify)
    original = (folder / "completion.json").read_bytes()
    assert not (job / "completion.json").exists()
    monkeypatch.setattr(module, "write_once", write)
    later = dict(proposal, completed_at="2026-01-01T02:00:00Z")
    _, metadata = module._finish_completion(folder, job, later, verify=verify)
    assert metadata["completed_at"] == proposal["completed_at"]
    assert (job / "completion.json").read_bytes() == original
    assert (folder / "completion.json").read_bytes() == original
    # A second process exit after the job receipt is already durable is harmless.
    assert module._finish_completion(folder, job, later, verify=verify)[1] == metadata
    assert (job / "completion.json").read_bytes() == original


@pytest.mark.parametrize("field,value", [("request_id", "DIFFERENT"), ("model_id", "DIFFERENT"),
    ("artifacts", {"SYNTHETIC.json": "DIFFERENT"}), ("calculation_objects", {"source": "DIFFERENT"}),
    ("order_authority", True)])
def test_model_completion_recovery_rejects_changed_static_bindings(tmp_path, field, value):
    import infrastructure.les_v7_fr2_m0 as module
    folder, job, proposal, verify = completion_case(tmp_path)
    module.write_once(folder / "completion.json", proposal)
    with pytest.raises(ValueError, match="COMPLETION_BINDING_MISMATCH"):
        module._finish_completion(folder, job, dict(proposal, **{field: value}), verify=verify)
    assert not (job / "completion.json").exists()


def test_corrupt_model_and_source_drift_never_publish_job_receipt(tmp_path):
    import infrastructure.les_v7_fr2_m0 as module
    folder, job, proposal, verify = completion_case(tmp_path)
    def corrupt(_):
        raise ValueError("SYNTHETIC_CORRUPT_MODEL")
    with pytest.raises(ValueError, match="CORRUPT_MODEL"):
        module._finish_completion(folder, job, proposal, verify=corrupt)
    assert not (job / "completion.json").exists()
    def drift():
        raise ValueError("SYNTHETIC_VERSION_DRIFT")
    with pytest.raises(ValueError, match="VERSION_DRIFT"):
        module._finish_completion(folder, job, proposal, verify=verify, check=drift)
    assert not (job / "completion.json").exists()


def test_completed_receipt_cannot_precede_actual_model_publication(tmp_path):
    import infrastructure.les_v7_fr2_m0 as module
    folder, job, proposal, verify = completion_case(tmp_path)
    module.write_once(folder / "completion.json", dict(proposal, completed_at="2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="COMPLETION_TIME_INVALID"):
        module._finish_completion(folder, job, proposal, verify=verify)
    assert not (job / "completion.json").exists()


def completed_job_case(tmp_path):
    """Full SYNTHETIC receipt shape; no market/model authorization is mocked."""
    import infrastructure.les_v7_fr2_m0 as module
    model_id, request_id = "a" * 64, "b" * 64
    output_root, job = tmp_path / "research", tmp_path / "research" / "jobs" / request_id
    folder = output_root / "models" / model_id
    receipt = dict(schema=module.SCHEMA, model_id=model_id, request_id=request_id,
        original_bundle_sha256="c" * 64,
        artifacts={name: "d" * 64 for name in ("frozen_bundle.json", "request.json", "proof.json",
            "native_result.json", "weight_rows.json", "config.json", "dataset.json")},
        calculation_objects=dict(source="e" * 64, environment="f" * 64, parameters="0" * 64),
        completed_at="2026-01-01T01:00:00Z", research_only=True, approved_for_live=False,
        historical_information_set_certified=False, execution_action="NO_ACTION", order_authority=False)
    module.write_once(folder / "completion.json", receipt)
    module.write_once(job / "completion.json", receipt)
    def verify(target):
        assert target == folder
        return dict(sha256=model_id, payload=dict(trained_until="2026-01-01T00:30:00Z")), deepcopy(receipt)
    return output_root, job, receipt, verify


def test_completed_job_retry_requires_entire_model_receipt_and_is_read_only(tmp_path):
    import infrastructure.les_v7_fr2_m0 as module
    output_root, job, receipt, verify = completed_job_case(tmp_path)
    encoded = (job / "completion.json").read_bytes()
    for _ in range(2):
        model, metadata = module._load_completed_job(job, output_root, receipt["request_id"], verify=verify)
        assert model["sha256"] == receipt["model_id"] and metadata == receipt
        assert (job / "completion.json").read_bytes() == encoded


@pytest.mark.parametrize("field,value", [
    ("completed_at", "2026-01-01T02:00:00Z"),
    ("completed_at", "2026-01-01T01:00:00+00:00"),
    ("artifacts", {"config.json": "1" * 64}),
    ("calculation_objects", dict(source="1" * 64)),
    ("order_authority", True), ("research_only", False), ("approved_for_live", True),
    ("execution_action", "BUY"), ("extra_unverified_field", "SYNTHETIC_TAMPERING")])
def test_completed_job_receipt_tampering_never_bypasses_model_verification(tmp_path, field, value):
    import infrastructure.les_v7_fr2_m0 as module
    output_root, job, receipt, verify = completed_job_case(tmp_path)
    changed = dict(receipt, **{field: value})
    (job / "completion.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="COMPLETED_JOB_RECEIPT_MISMATCH"):
        module._load_completed_job(job, output_root, receipt["request_id"], verify=verify)
    assert json.loads((job / "completion.json").read_text()) == changed


@pytest.mark.parametrize("model_id", [None, "", "A" * 64, "a" * 63, "../" + "a" * 64])
def test_completed_job_invalid_model_reference_rejected_before_read(tmp_path, model_id):
    import infrastructure.les_v7_fr2_m0 as module
    output_root, job, receipt, _ = completed_job_case(tmp_path)
    (job / "completion.json").write_text(json.dumps(dict(receipt, model_id=model_id)))
    with pytest.raises(ValueError, match="COMPLETED_JOB_MODEL_ID_INVALID"):
        module._load_completed_job(job, output_root, receipt["request_id"],
            verify=lambda _: pytest.fail("invalid reference reached model loader"))


def test_completed_job_other_request_or_corrupt_model_is_not_a_cache_hit(tmp_path):
    import infrastructure.les_v7_fr2_m0 as module
    output_root, job, receipt, verify = completed_job_case(tmp_path)
    with pytest.raises(ValueError, match="COMPLETED_JOB_RECEIPT_MISMATCH"):
        module._load_completed_job(job, output_root, "1" * 64, verify=verify)
    def corrupt(_):
        raise ValueError("SYNTHETIC_CORRUPT_MODEL")
    with pytest.raises(ValueError, match="CORRUPT_MODEL"):
        module._load_completed_job(job, output_root, receipt["request_id"], verify=corrupt)
