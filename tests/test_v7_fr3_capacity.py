"""SYNTHETIC ONLY: prepared capacity is not forecast or performance evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import identity
from infrastructure import les_v7_fr3_capacity as capacity
from infrastructure import les_v7_scope_training as native
from infrastructure.les_v7_fr3_training import REQUEST_SCHEMA, prepare_fr3_scope
from tests.test_v7_fr3_training import inputs, NOW
from tests.test_v7_training_publication import Guard


def row(symbol, day, *, context="REVERSAL", utility=0., label="2026-05-01T05:30:00+00:00", fill=1):
    cutoff = day+"T05:30:00+00:00"
    return dict(symbol=symbol, market="TWSE", context=context, cutoff=cutoff,
        row_id=identity(dict(symbol=symbol, market="TWSE", cutoff=cutoff)),
        label_available_at=label, utility=utility, labels=dict(fill=fill),
        features=[1.], SYNTHETIC_METADATA_ROUTING_ONLY=True)


@pytest.fixture
def prepared(inputs, monkeypatch):
    """Mock native verification; real bounded files/counting run after that port.

    The separate actual-native-verifier test below covers full source/config
    verification. These tiny metadata fixtures are not valid trading evidence.
    """
    freeze_path, build, config, root = inputs
    config = deepcopy(config)
    config.update(min_weight_rows=3, min_context_weight_rows=1, min_holdout_rows=2)
    build = deepcopy(build)
    components = capacity.validate_fr3_training_configs(build, config)
    folder = root / "native"
    folder.mkdir(parents=True)
    body = dict(schema=REQUEST_SCHEMA, dataset_kind="SYNTHETIC_TEST_ONLY", freeze_path=str(freeze_path),
        freeze_id="a"*64, scope_id="b"*64, symbols=["1000", "1001", "1002"],
        availability_policy="SYNTHETIC_ONLY", build_config=build, training_config=config, components=components)
    request = dict(payload=body, sha256=identity(body))
    (root / "fr3-training-request.json").write_text(json.dumps(request))
    for name, value in (("scope-build-config.json", build), ("scope-training-config.json", config)):
        (folder / name).write_text(json.dumps(value))
    first_rows = [
        row("1000", "2026-03-03", utility=0.),
        row("1000", "2026-03-04", context="TREND_PULLBACK", utility=-.5),
        row("1001", "2026-03-05", context="LEADERSHIP_CONTINUATION", utility=.2),
        row("1000", "2026-06-01", utility=1., label="2026-06-02T05:30:00+00:00"),  # mature, outside OOF
        row("1001", "2026-03-06", utility=1., label="2026-07-02T05:30:00+00:00"),  # not fit-mature
        row("1000", "2026-08-03", utility=None, fill=0, label="2026-09-01T05:30:00+00:00"),
        row("1001", "2026-08-04", utility=None, fill=0, label="2026-09-01T05:30:00+00:00"),
    ]
    packets = []
    children = []
    for index, (symbols, rows) in enumerate(((body["symbols"][:2], first_rows), (["1002"], []))):
        packet = dict(schema=native.PREPARED_SCHEMA, dataset_id=str(index)*64,
            config_id=identity(config), feature_columns=["SYNTHETIC_ONLY"], rows=rows, diagnostics=[])
        path = folder / "prepared" / (str(index)+".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(packet))
        packets.append((path, packet))
        children.append(dict(index=index, symbols=symbols, dataset_id=packet["dataset_id"],
            prepared_path="prepared/"+path.name, prepared_sha256=identity(packet), prepared_count=len(rows),
            coverage=[dict(symbol=symbol, market="TWSE", prepared_rows=sum(r["symbol"] == symbol for r in rows),
                           candidate_events=10, build_reason_counts={"SYNTHETIC_ONLY": 1}) for symbol in symbols]))
    manifest = dict(schema="les-v7-sharded-training-dataset/v1", dataset_kind="SYNTHETIC_TEST_ONLY",
        universe=[[s, "TWSE"] for s in body["symbols"]], freeze_path=str(freeze_path),
        freeze_id=body["freeze_id"], scope_id=body["scope_id"], availability_policy=body["availability_policy"],
        children=children, counts=dict(prepared_rows=len(first_rows)))
    record = dict(payload=manifest, sha256=identity(manifest))
    manifest_path = folder / "scope-training-dataset.json"
    manifest_path.write_text(json.dumps(record))
    calls = []
    def verify(value, native_root, *, config):
        calls.append((deepcopy(value), native_root, deepcopy(config)))
        assert value["sha256"] == identity(value["payload"])
        return value["payload"]
    monkeypatch.setattr(capacity, "verify_sharded_training_dataset", verify)
    return dict(root=root, config=config, request=request, manifest=manifest, manifest_path=manifest_path,
                packets=packets, calls=calls)


def reseal(prepared):
    for (path, packet), child in zip(prepared["packets"], prepared["manifest"]["children"]):
        path.write_text(json.dumps(packet))
        child["prepared_sha256"] = identity(packet)
        child["prepared_count"] = len(packet["rows"])
        for covered in child["coverage"]:
            covered["prepared_rows"] = sum(r["symbol"] == covered["symbol"] for r in packet["rows"])
    prepared["manifest"]["counts"]["prepared_rows"] = sum(len(p["rows"]) for _, p in prepared["packets"])
    body = prepared["manifest"]
    prepared["manifest_path"].write_text(json.dumps(dict(payload=body, sha256=identity(body))))


def audit(prepared):
    return capacity.audit_fr3_prepared_capacity(prepared["root"], allow_synthetic=True)


def test_native_sources_are_verified_before_streaming_and_all_zero_stocks_retained(prepared, monkeypatch):
    from domain.les_v7 import training
    monkeypatch.setattr(training, "_oof", lambda *_a, **_k: pytest.fail("capacity ran OOF"))
    monkeypatch.setattr(training, "_fit", lambda *_a, **_k: pytest.fail("capacity fitted heads"))
    result = audit(prepared)
    assert len(prepared["calls"]) == 1
    assert prepared["calls"][0][1] == prepared["root"] / "native"
    assert prepared["calls"][0][2] == prepared["config"]
    assert result["status"] == capacity.SUFFICIENT
    assert result["count_capacity_allows_fit"] is True
    assert result["fit_allowed"] is None and result["ready_to_freeze"] is False
    assert result["necessary_counts_only"] is True and result["model_fit_executed"] is False
    assert result["fixed_universe_count"] == 3
    assert result["zero_prepared_symbols"] == ["1002"]
    assert [s["symbol"] for s in result["stocks"]] == ["1000", "1001", "1002"]
    assert result["stocks"][-1]["native_coverage"]["build_reason_counts"] == {"SYNTHETIC_ONLY": 1}
    assert result["counts"] == dict(prepared_rows=7, utility_known_rows=5, utility_missing_rows=2,
        no_fill_rows=2, fit_mature_rows=4, fit_mature_utility_known_rows=4,
        oof_window_rows=4, oof_window_fit_mature_utility_upper_bound=3, holdout_mature_rows=2)
    assert [result["context_counts"][c]["oof_window_fit_mature_utility_upper_bound"]
            for c in ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION")] == [1, 1, 1]
    assert result["frozen_folds"] == prepared["config"]["folds"]
    assert "NOT_QUALITY_OR_RESTRICTION_PASS" in result["safety_evidence_status"]


def test_none_utility_is_not_zero_and_global_samples_do_not_replace_context_minimum(prepared):
    prepared["packets"][0][1]["rows"][2]["utility"] = None
    reseal(prepared)
    result = audit(prepared)
    assert result["status"] == capacity.INSUFFICIENT and result["fit_allowed"] is False
    assert result["counts"]["utility_known_rows"] == 4
    assert result["context_counts"]["REVERSAL"]["utility_known_rows"] == 3  # zero utility is known
    assert result["context_counts"]["LEADERSHIP_CONTINUATION"]["oof_window_fit_mature_utility_upper_bound"] == 0
    assert "INSUFFICIENT_CONTEXT_WEIGHT_UTILITY_UPPER_BOUND:LEADERSHIP_CONTINUATION" in result["reason_codes"]
    assert "INSUFFICIENT_OOF_WEIGHT_UTILITY_UPPER_BOUND" in result["reason_codes"]


def test_holdout_counts_valid_unfilled_rows_but_does_not_count_dates_outside_window(prepared):
    rows = prepared["packets"][0][1]["rows"]
    rows[-1] = row("1001", "2026-08-06", utility=None, fill=0, label="2026-09-01T05:30:00+00:00")
    reseal(prepared)
    result = audit(prepared)
    assert result["counts"]["no_fill_rows"] == 2
    assert result["counts"]["holdout_mature_rows"] == 1
    assert "INSUFFICIENT_MATURE_HOLDOUT_NECESSARY_COUNT" in result["reason_codes"]
    assert result["fit_allowed"] is False
    assert "bad_bar" not in result["counts"]


def test_oof_windows_and_global_label_maturity_are_independent_necessary_filters(prepared):
    result = audit(prepared)
    assert result["counts"]["fit_mature_utility_known_rows"] == 4
    assert result["counts"]["oof_window_rows"] == 4
    assert result["counts"]["oof_window_fit_mature_utility_upper_bound"] == 3
    first_fold = prepared["config"]["folds"][0]["fold_id"]
    assert result["fold_counts"][first_fold]["oof_window_rows"] == 4
    assert result["fold_counts"][first_fold]["oof_window_fit_mature_utility_upper_bound"] == 3


def test_synthetic_scope_requires_explicit_boolean_opt_in_before_source_work(prepared):
    with pytest.raises(ValueError, match="SYNTHETIC_REQUIRES_EXPLICIT_OPT_IN"):
        capacity.audit_fr3_prepared_capacity(prepared["root"])
    assert not prepared["calls"]
    with pytest.raises(ValueError, match="SYNTHETIC_BOOLEAN"):
        capacity.audit_fr3_prepared_capacity(prepared["root"], allow_synthetic=1)


def test_native_source_or_provenance_failure_stops_before_counting(prepared, monkeypatch):
    def reject(*_a, **_k):
        raise ValueError("SYNTHETIC_SOURCE_PROOF_REJECTED")
    monkeypatch.setattr(capacity, "verify_sharded_training_dataset", reject)
    monkeypatch.setattr(capacity, "inspect_prepared_packet", lambda *_a, **_k: pytest.fail("counted rejected source"))
    with pytest.raises(ValueError, match="SOURCE_PROOF_REJECTED"):
        audit(prepared)


@pytest.mark.parametrize("damage", ["request_hash", "config", "components", "scope", "missing_zero_coverage",
    "duplicate", "utility", "future_label", "path", "changed_packet"])
def test_invalid_inputs_and_concurrent_prepared_edits_cannot_report_capacity(prepared, damage):
    request = deepcopy(prepared["request"])
    if damage == "request_hash":
        request["sha256"] = "0"*64
    elif damage == "config":
        request["payload"]["training_config"]["min_holdout_rows"] = 1
    elif damage == "components":
        request["payload"]["components"]["label_version"] = "OLD_FR2"
    elif damage == "scope":
        request["payload"]["symbols"].pop()
    elif damage == "missing_zero_coverage":
        prepared["manifest"]["children"][-1]["coverage"] = []
    elif damage == "duplicate":
        prepared["packets"][0][1]["rows"].append(deepcopy(prepared["packets"][0][1]["rows"][0]))
    elif damage == "utility":
        prepared["packets"][0][1]["rows"][0]["utility"] = True
    elif damage == "future_label":
        prepared["packets"][0][1]["rows"][0]["label_available_at"] = "2099-01-01T00:00:00Z"
    elif damage == "path":
        prepared["manifest"]["children"][0]["prepared_path"] = "../escaped.json"
    else:
        packet_path = prepared["packets"][0][0]
        packet_path.write_text('{"TAMPERED_AFTER_NATIVE_VERIFICATION":true}')
    if damage != "changed_packet":
        reseal(prepared)
    if damage != "request_hash":
        request["sha256"] = identity(request["payload"])
    (prepared["root"] / "fr3-training-request.json").write_text(json.dumps(request))
    with pytest.raises((ValueError, KeyError)):
        audit(prepared)


def test_capacity_content_identity_and_retention_are_deterministic_and_no_clobber(prepared, tmp_path):
    first = audit(prepared)
    assert audit(prepared) == first
    assert first["capacity_id"] == identity({k: v for k, v in first.items() if k != "capacity_id"})
    folder, completion = capacity.retain_fr3_capacity(first, tmp_path / "retained-capacity")
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert capacity.retain_fr3_capacity(first, tmp_path / "retained-capacity") == (folder, completion)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert completion["necessary_counts_only"] is True and completion["model_fit_executed"] is False
    broken = deepcopy(first); broken["ready_to_freeze"] = True
    broken["capacity_id"] = identity({k: v for k, v in broken.items() if k != "capacity_id"})
    with pytest.raises(ValueError, match="REPORT_HASH_OR_CONTRACT_INVALID"):
        capacity.retain_fr3_capacity(broken, tmp_path / "retained-capacity")
    (folder / "capacity.json").write_text("{}")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        capacity.retain_fr3_capacity(first, tmp_path / "retained-capacity")


def test_actual_native_preparation_and_full_verifier_without_fit_preserve_empty_scope(inputs):
    # Real tiny native dataset/source verification, explicitly synthetic. No
    # monkeypatched source PASS, OOF, calibrator or head fitting is performed.
    prepare_fr3_scope(*inputs, allow_synthetic=True, guard_factory=Guard, clock=lambda: NOW)
    result = capacity.audit_fr3_prepared_capacity(inputs[3], allow_synthetic=True)
    assert result["fixed_universe_count"] == 3 and result["zero_prepared_symbols"] == ["1000", "1001", "1002"]
    assert result["counts"]["prepared_rows"] == 0 and result["fit_allowed"] is False
    assert result["native_sources_verified"] is True and result["model_fit_executed"] is False
    assert not list(inputs[3].rglob("frozen_bundle.json"))
