"""SYNTHETIC metadata/native-port contracts; never new real research results."""
from copy import deepcopy
import json

import pytest

from domain.les_v7.contracts import canonical
from infrastructure import les_v7_scope_training as scope
from tests.test_v7_scope_training import inputs, run


def container(schema, size=None):
    """A metadata-only size fixture, NOT a valid model/source artifact."""
    value = dict(schema=schema, research_only=True, approved_for_live=False,
        execution_action="NO_ACTION", synthetic_test_only=True, padding="", archive_path="models/"+"a"*64)
    if size is not None:
        value["padding"] = "x" * (size-len(canonical(value).encode()))
        assert len(canonical(value).encode()) == size
    return value


def test_v2_metadata_budget_never_changes_native_evidence_or_sample_limits():
    assert scope.OUTCOME_SCHEMA == "les-v7-elec-scope-training-outcome/v2"
    assert scope.LEGACY_OUTCOME_SCHEMA == "les-v7-elec-scope-training-outcome/v1"
    assert scope.MAX_OUTCOME_BYTES == 2 * 1024 * 1024
    assert scope.MAX_SYMBOLS == 2048
    assert scope.MAX_PREPARED_ROWS == 100000
    assert scope.MAX_PREPARED_BYTES == 128 * 1024 * 1024
    assert scope.MAX_PREPARED_DECODED_BYTES == 256 * 1024 * 1024


@pytest.mark.parametrize("schema,maximum", [
    ("les-v7-elec-scope-training-outcome/v1", 64*1024),
    ("les-v7-elec-scope-training-outcome/v2", 2*1024*1024)])
def test_schema_specific_exact_metadata_boundary_and_plus_one(schema, maximum):
    assert scope._validate_outcome_metadata(container(schema, maximum)) is None
    with pytest.raises(ValueError, match="OUTCOME_SIZE_LIMIT"):
        scope._validate_outcome_metadata(container(schema, maximum+1))


def test_115_child_metadata_can_be_retained_in_v2_without_dropping_any_reference(tmp_path):
    value = container(scope.OUTCOME_SCHEMA)
    value["build_children"] = [dict(index=i, evidence_id="a"*64, build_job_id="b"*64,
        dataset_id="c"*64, completion_sha256="d"*64, archive_path="builds/"+"e"*64,
        prepared_sha256="f"*64) for i in range(115)]
    value["padding"] = "SYNTHETIC_NON_RESEARCH_METADATA" * 500
    assert 64*1024 < len(canonical(value).encode()) < scope.MAX_OUTCOME_BYTES
    before = deepcopy(value)
    assert scope._validate_outcome_metadata(value) is None and value == before
    # Size acceptance never substitutes for native source/child/model checking.
    with pytest.raises(FileNotFoundError):
        scope.verify_elec_scope_training(tmp_path/"ABSENT_SYNTHETIC_ARCHIVE", value)
    with pytest.raises(ValueError, match="OUTCOME_SIZE_LIMIT"):
        scope._validate_outcome_metadata(dict(value, schema=scope.LEGACY_OUTCOME_SCHEMA))


@pytest.mark.parametrize("damage", ["future_schema", "legacy_size_bypass", "wrong_research", "live", "action"])
def test_unknown_schema_or_changed_authority_never_bypasses_validation(damage):
    value = container(scope.OUTCOME_SCHEMA)
    if damage == "future_schema": value["schema"] = "les-v7-elec-scope-training-outcome/v999"
    elif damage == "legacy_size_bypass": value = container(scope.LEGACY_OUTCOME_SCHEMA, 65537)
    elif damage == "wrong_research": value["research_only"] = False
    elif damage == "live": value["approved_for_live"] = True
    else: value["execution_action"] = "ORDER"
    with pytest.raises(ValueError, match="OUTCOME_(SCOPE_MISMATCH|SIZE_LIMIT)"):
        scope._validate_outcome_metadata(value)


def test_new_native_v2_metadata_retry_preserves_all_children_and_bytes(inputs, monkeypatch):
    outcome = run(inputs)
    assert outcome["schema"] == scope.OUTCOME_SCHEMA
    path = inputs[3]/"scope-training-outcome.json"
    encoded = path.read_bytes()
    assert len(outcome["build_children"]) == outcome["batch_count"] == 2
    assert scope.verify_elec_scope_training(inputs[3], outcome)["status"] == "NOT_READY"
    monkeypatch.setattr(scope, "_train_prepared", lambda *_a: pytest.fail("retry cannot refit"))
    assert run(inputs) == outcome
    assert path.read_bytes() == encoded


def test_legacy_v1_native_verification_and_retry_are_readonly_without_migration(inputs, monkeypatch):
    outcome = run(inputs)
    legacy = dict(outcome, schema=scope.LEGACY_OUTCOME_SCHEMA)
    path = inputs[3]/"scope-training-outcome.json"
    # Only this isolated SYNTHETIC fixture is represented in the older schema.
    path.write_text(canonical(legacy), encoding="utf-8")
    encoded = path.read_bytes()
    assert len(encoded) <= scope.MAX_LEGACY_OUTCOME_BYTES
    assert scope.verify_elec_scope_training(inputs[3], legacy)["status"] == "NOT_READY"
    monkeypatch.setattr(scope, "_train_prepared", lambda *_a: pytest.fail("legacy retry cannot refit"))
    assert run(inputs) == legacy
    assert path.read_bytes() == encoded
    assert json.loads(encoded)["schema"] == scope.LEGACY_OUTCOME_SCHEMA


def test_v2_child_hash_conflict_is_still_rejected_by_native_binding(inputs):
    outcome = run(inputs)
    damaged = deepcopy(outcome)
    damaged["build_children"][0]["dataset_id"] = "0"*64
    with pytest.raises(ValueError, match="OUTCOME_BINDING_MISMATCH"):
        scope.verify_elec_scope_training(inputs[3], damaged)


def test_over_64k_native_v2_completion_reopens_and_retries_without_losing_bindings(inputs, tmp_path, monkeypatch):
    """130 genuine tiny SYNTHETIC child artifacts, not fabricated completions.

    Four bars per stock deliberately produce no eligible training rows. This
    exercises full native persistence/recovery, never real model readiness.
    """
    from tests.test_v7_dataset_scope import scope_inputs, freeze

    source_root = tmp_path / "SYNTHETIC-many-children"
    source_root.mkdir()
    source = scope_inputs(source_root, count=130)
    manifest = freeze(source, batch_size=1)
    isolated = (source, manifest, deepcopy(inputs[2]), tmp_path / "SYNTHETIC-many-output")
    outcome = run(isolated)
    completion = isolated[3] / "scope-training-outcome.json"
    encoded = completion.read_bytes()
    assert 64 * 1024 < len(encoded) <= scope.MAX_OUTCOME_BYTES
    assert outcome["schema"] == scope.OUTCOME_SCHEMA
    assert outcome["symbols"] == source["symbols"]
    assert outcome["symbol_count"] == outcome["batch_count"] == len(outcome["build_children"]) == 130
    assert outcome["status"] == "NOT_READY" and outcome["bundle_path"] is None

    # Reopen actual bytes; every child and every model/report reference must
    # still pass native verification instead of only the size validator.
    reopened = json.loads(encoded)
    receipt = scope.verify_elec_scope_training(isolated[3], reopened)
    assert receipt["status"] == "NOT_READY" and receipt["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert completion.read_bytes() == encoded
    monkeypatch.setattr(scope, "_train_prepared",
                        lambda *_a, **_k: pytest.fail("completed retry cannot refit"))
    monkeypatch.setattr(scope, "_prepare_scope_shards",
                        lambda *_a, **_k: pytest.fail("completed retry cannot rebuild children"))
    assert run(isolated) == reopened
    assert completion.read_bytes() == encoded

    damaged = deepcopy(reopened)
    damaged["build_children"][79]["completion_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="OUTCOME_BINDING_MISMATCH"):
        scope.verify_elec_scope_training(isolated[3], damaged)
    # With zero prepared rows, the genuine fit has no head report receipts.
    # Injecting one must still fail exact native report-reference binding.
    assert reopened["report_references"] == {}
    invented_reference = deepcopy(reopened)
    invented_reference["report_references"] = {"SYNTHETIC_UNTRUSTED_REFERENCE": "0" * 64}
    with pytest.raises(ValueError, match="OUTCOME_BINDING_MISMATCH"):
        scope.verify_elec_scope_training(isolated[3], invented_reference)
    # Only this temporary SYNTHETIC completion is deliberately corrupted.
    # Retry must refuse it and must not overwrite it with a repaired result.
    corrupted = canonical(damaged).encode()
    completion.write_bytes(corrupted)
    with pytest.raises(ValueError, match="OUTCOME_BINDING_MISMATCH"):
        run(isolated)
    assert completion.read_bytes() == corrupted
