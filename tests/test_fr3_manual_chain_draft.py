"""SYNTHETIC/mocked routing only. Never launches prepare, fit, HTTP or mail."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import canonical, identity

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_fr3_manual_chain.py"
spec = importlib.util.spec_from_file_location("fr3_manual_chain_draft", SCRIPT)
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


@dataclass
class Versions:
    models: str = "SYNTHETIC_ONLY"
    parameters: str = "SYNTHETIC_ONLY"
    code: str = "SYNTHETIC_ONLY"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    training, output = tmp_path / "SYNTHETIC-training", tmp_path / "SYNTHETIC-chain"
    pinned = dict(source="a"*64, environment="b"*64, parameters="c"*64)
    cfg = dict(evaluation_cutoff="2026-10-23T13:30:00+08:00", SYNTHETIC_ONLY=True)
    build = dict(SYNTHETIC_ONLY=True)
    request_body = dict(schema=chain.REQUEST_SCHEMA, freeze_id="d"*64,
        freeze_path=str(tmp_path / "SYNTHETIC-freeze.json"), training_config=cfg,
        build_config=build, availability_policy="SYNTHETIC_ONLY")
    request = dict(payload=request_body, sha256=identity(request_body))
    native_body = dict(freeze_id=request_body["freeze_id"], pinned_objects=pinned,
        source_root=str(tmp_path / "SYNTHETIC-source"), training_config_id=identity(cfg),
        build_config_id=identity(build), scope_id="8"*64, dataset_kind="SYNTHETIC",
        source_evidence_ids=["5"*64, "6"*64], universe=[["3010", "TWSE"], ["3515", "TWSE"]])
    dataset = dict(payload=native_body, sha256=identity(native_body))
    save(training / "fr3-training-request.json", request)
    save(training / "native/scope-training-dataset.json", dataset)
    native_lock = training / "native/.scope-training.lock"
    native_lock.write_bytes(b"SYNTHETIC-original-lock-metadata")
    original = tmp_path / "SYNTHETIC-original/frozen_bundle.json"
    save(original, {"SYNTHETIC_ONLY": True})
    summary = dict(command="prepare", status="DATASET_PREPARED", content_id=request["sha256"])
    observation = dict(payload=dict(schema=chain.EXIT_SCHEMA, source_session_id="SYNTHETIC_SESSION",
        pid=999999999, request_id=request["sha256"], dataset_id=dataset["sha256"],
        pinned_objects=pinned, observed_at="2026-10-24T13:30:00+08:00",
        exit_result=dict(exit_code=0, output="SYNTHETIC progress\n"+canonical(summary))))
    observation["sha256"] = identity(observation["payload"])
    observed = tmp_path / "SYNTHETIC-exit.json"
    save(observed, observation)
    audit = dict(schema=chain.AUDIT_SCHEMA, sources_verified=True,
        **{k:native_body[k] for k in ("freeze_id", "scope_id", "dataset_kind", "source_evidence_ids")},
        evaluation_cutoff="2026-10-23T05:30:00+00:00", universe=["3010", "3515"],
        previously_inspected_sessions=["2026-09-21", "2026-09-22"])
    audit["audit_id"] = identity(audit)
    audit_path = tmp_path / "SYNTHETIC-source-audit.json"
    save(audit_path, audit)
    calls, state = [], dict(sufficient=True, native_status="READY_TO_FREEZE", m0_fail=False, guard_fail=False)
    class Guard:
        versions = Versions()
        def retain(self, _): return deepcopy(pinned)
        def check(self):
            if state["guard_fail"]: raise ValueError("SYNTHETIC_VERSION_DRIFT")
            return self.versions
    monkeypatch.setattr(chain, "ResearchVersionGuard", Guard)
    monkeypatch.setattr(chain, "source_paths", lambda: [])
    monkeypatch.setattr(chain.os, "kill", lambda *_: (_ for _ in ()).throw(ProcessLookupError()))
    def capacity(_):
        calls.append("capacity")
        return dict(capacity_id="e"*64, count_capacity_allows_fit=state["sufficient"],
            status="SYNTHETIC_ONLY_NECESSARY_COUNTS")
    monkeypatch.setattr(chain, "audit_fr3_prepared_capacity", capacity)
    monkeypatch.setattr(chain, "retain_fr3_capacity", lambda *_: None)
    bundle = training / "native/models" / ("f"*64) / "frozen_bundle.json"
    def outcome():
        return dict(status=state["native_status"], native_outcome=dict(
            bundle_path=str(bundle.relative_to(training / "native")), bundle_sha256="1"*64))
    def fit(*_a, **_k):
        calls.append("fit")
        save(training / "fr3-training-completion.json", outcome())
        save(bundle, {"SYNTHETIC_ONLY": "new-model"})
        return outcome()
    def verify(_):
        calls.append("verify")
        return outcome()
    def m0(*_):
        calls.append("m0")
        if state["m0_fail"]: raise ValueError("INSUFFICIENT_OWN_M0_WEIGHT_ROWS")
        return dict(model_id="2"*64, SYNTHETIC_ONLY=True)
    monkeypatch.setattr(chain, "train_fr3_scope", fit)
    monkeypatch.setattr(chain, "verify_fr3_scope_training", verify)
    monkeypatch.setattr(chain, "publish_m0", m0)
    def study(_original, _training, retained_audit, _output):
        calls.append("study")
        assert retained_audit == audit
        return output / "SYNTHETIC-study", dict(study_id="4"*64, status="NOT_ASSESSABLE")
    monkeypatch.setattr(chain.study, "run_fr3_study", study)
    return dict(args=(training, original, observed, audit_path, output), calls=calls, state=state,
        observation=observation, native_lock=native_lock, bundle=bundle, audit=audit, output=output)


def execute(case): return chain.run_manual_chain(*case["args"])


def test_insufficient_capacity_retains_terminal_and_never_fits(case):
    case["state"]["sufficient"] = False
    first = execute(case)
    assert first["status"] == "STOPPED_INSUFFICIENT_CAPACITY_NO_FIT"
    assert case["calls"] == ["capacity"]
    raw = (case["output"] / "completion.json").read_bytes()
    assert execute(case) == first
    assert (case["output"] / "completion.json").read_bytes() == raw
    assert case["calls"] == ["capacity", "capacity"]


def test_ready_routing_retry_verifies_existing_fit_and_never_refits(case):
    first = execute(case)
    assert case["calls"] == ["capacity", "fit", "verify", "m0", "study"]
    assert first["after_cost_four_account_status"] == "NOT_ASSESSABLE"
    assert first["fresh_confirmatory"] is False and first["mail_sent"] is False
    assert execute(case) == first
    assert case["calls"].count("fit") == 1 and case["calls"].count("verify") == 2
    attempts = sorted((case["output"] / "attempts").iterdir())
    assert len(attempts) == 2
    for folder in attempts:
        completion = json.loads((folder / "completion.json").read_text())
        for reference in completion["stages"]:
            path = folder / reference["file"]
            receipt = json.loads(path.read_text())
            assert identity(receipt) == reference["sha256"]
            assert receipt["exit_code"] == 0
            assert "NOT_SUBPROCESS_EXIT" in receipt["exit_semantics"]
            for stream in ("stdout", "stderr"):
                assert chain.file_id(folder / (receipt["stage"]+"."+stream+".log")) == receipt[stream+"_sha256"]


def test_completed_not_ready_fit_stops_without_m0_or_fake_model(case):
    case["state"]["native_status"] = "NOT_READY"
    result = execute(case)
    assert result["status"] == "STOPPED_NATIVE_NOT_READY"
    assert case["calls"] == ["capacity", "fit", "verify"]
    assert result["published_m0_id"] is None


def test_exit_and_stdout_can_be_retained_from_separate_actual_tool_observations(case):
    case["state"]["sufficient"] = False
    observation = case["observation"]
    body = observation["payload"]
    body["stdout_result"] = dict(body["exit_result"], exit_code=None)
    body["exit_result"] = dict(exit_code=0, output="")
    observation["sha256"] = identity(body)
    save(case["args"][2], observation)
    before = case["args"][2].read_bytes()
    assert execute(case)["status"] == "STOPPED_INSUFFICIENT_CAPACITY_NO_FIT"
    assert case["args"][2].read_bytes() == before


@pytest.mark.parametrize("prior_poll", [False, True])
def test_time_trailer_does_not_hide_unique_original_prepare_summary(case, prior_poll):
    case["state"]["sufficient"] = False
    observation = case["observation"]
    body = observation["payload"]
    trailer = "\n      311.30 real       278.50 user         8.72 sys\n  2000000  maximum resident set size\n"
    if prior_poll:
        body["stdout_result"] = dict(body["exit_result"], exit_code=None)
        body["exit_result"] = dict(exit_code=0, output=trailer)
    else:
        body["exit_result"]["output"] += trailer
    observation["sha256"] = identity(body)
    save(case["args"][2], observation)
    before = case["args"][2].read_bytes()
    assert execute(case)["status"] == "STOPPED_INSUFFICIENT_CAPACITY_NO_FIT"
    assert case["args"][2].read_bytes() == before


@pytest.mark.parametrize("conflict", [False, True])
@pytest.mark.parametrize("separate_poll", [False, True])
def test_duplicate_or_conflicting_prepare_summaries_refuse(case, conflict, separate_poll):
    observation = case["observation"]
    body = observation["payload"]
    second = json.loads(body["exit_result"]["output"].splitlines()[-1])
    if conflict: second["content_id"] = "0"*64
    if separate_poll:
        body["stdout_result"] = dict(body["exit_result"], exit_code=None)
        body["exit_result"]["output"] = canonical(second)
    else:
        body["exit_result"]["output"] += "\n"+canonical(second)
    observation["sha256"] = identity(body)
    save(case["args"][2], observation)
    with pytest.raises(ValueError, match="FINAL_SUMMARY_REQUIRED"): execute(case)
    assert not case["calls"]


@pytest.mark.parametrize("damage", ["exit", "summary", "request", "dataset", "pinned", "checksum"])
def test_partial_or_forged_exit_observation_never_dispatches(case, damage):
    observation = case["observation"]
    body = observation["payload"]
    if damage == "exit": body["exit_result"]["exit_code"] = 1
    elif damage == "summary": body["exit_result"]["output"] = "partial manifest exists"
    elif damage == "request": body["request_id"] = "0"*64
    elif damage == "dataset": body["dataset_id"] = "0"*64
    elif damage == "pinned": body["pinned_objects"] = {}
    else: body["pid"] += 1
    if damage != "checksum": observation["sha256"] = identity(body)
    save(case["args"][2], observation)
    with pytest.raises(ValueError, match="FR3_MANUAL_"): execute(case)
    assert case["calls"] == []


def test_live_or_reused_pid_refuses_without_signalling_or_stopping_it(case, monkeypatch):
    monkeypatch.setattr(chain.os, "kill", lambda *_: None)
    with pytest.raises(ValueError, match="PROCESS_STILL_PRESENT"): execute(case)
    assert case["calls"] == []


def test_original_lock_busy_or_missing_refuses_without_metadata_changes(case):
    raw = case["native_lock"].read_bytes()
    with chain.lock(case["native_lock"], existing=True):
        with pytest.raises(ValueError, match="BUSY"): execute(case)
    assert case["native_lock"].read_bytes() == raw and not case["calls"]
    case["native_lock"].unlink()
    with pytest.raises(FileNotFoundError): execute(case)


def test_stage_failure_has_diagnostic_but_no_completion_then_native_retry_is_safe(case):
    case["state"]["m0_fail"] = True
    with pytest.raises(ValueError, match="INSUFFICIENT_OWN_M0"): execute(case)
    assert not (case["output"] / "completion.json").exists()
    receipt = json.loads((case["output"] / "attempts/000001/m0.json").read_text())
    assert receipt["status"] == "ERROR" and receipt["exit_code"] == 1
    assert "INSUFFICIENT_OWN_M0" in (case["output"] / "attempts/000001/m0.stderr.log").read_text()
    case["state"]["m0_fail"] = False
    execute(case)
    assert case["calls"].count("fit") == 1


def test_version_drift_fails_before_fit_and_never_publishes_completion(case, monkeypatch):
    old = chain.audit_fr3_prepared_capacity
    def drift(*args):
        value = old(*args)
        case["state"]["guard_fail"] = True
        return value
    monkeypatch.setattr(chain, "audit_fr3_prepared_capacity", drift)
    with pytest.raises(ValueError, match="VERSION_DRIFT"): execute(case)
    assert case["calls"] == ["capacity"]
    assert not (case["output"] / "completion.json").exists()


def test_bound_source_file_change_during_stage_is_rejected(case, monkeypatch):
    old = chain.audit_fr3_prepared_capacity
    def drift(*args):
        value = old(*args)
        case["args"][1].write_bytes(b"SYNTHETIC_CHANGED_ORIGINAL")
        return value
    monkeypatch.setattr(chain, "audit_fr3_prepared_capacity", drift)
    with pytest.raises(ValueError, match="BOUND_INPUT_OR_RUNNER_CHANGED"): execute(case)
    assert case["calls"] == ["capacity"]


def test_runner_cannot_be_added_to_the_native_source_identity(case, monkeypatch):
    monkeypatch.setattr(chain, "source_paths", lambda: [Path(chain.__file__)])
    with pytest.raises(ValueError, match="MUST_NOT_CHANGE_NATIVE_SOURCE_IDENTITY"): execute(case)
    assert not case["calls"]


def test_existing_native_input_mutation_cannot_be_called_a_retry(case):
    execute(case)
    raw = (case["output"] / "completion.json").read_bytes()
    case["args"][1].write_bytes(b"SYNTHETIC_CHANGED_ORIGINAL")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"): execute(case)
    assert (case["output"] / "completion.json").read_bytes() == raw
    assert case["calls"].count("fit") == 1


@pytest.mark.parametrize("damage", ["hash", "schema", "freeze_id", "scope_id", "dataset_kind",
    "source_evidence_ids", "sources_verified", "evaluation_cutoff", "universe", "inspected_missing"])
def test_existing_source_audit_must_match_exact_prepared_sources_before_fit(case, damage):
    audit = case["audit"]
    if damage == "hash": audit["universe"] = ["0000"]
    elif damage == "schema": audit["schema"] = "unknown-schema"
    elif damage in {"freeze_id", "scope_id", "dataset_kind"}: audit[damage] = "WRONG"
    elif damage == "source_evidence_ids": audit[damage].reverse()
    elif damage == "sources_verified": audit[damage] = False
    elif damage == "evaluation_cutoff": audit[damage] = "2026-10-22T13:30:00+08:00"
    elif damage == "universe": audit[damage].reverse()
    else: audit.pop("previously_inspected_sessions")
    if damage != "hash": audit["audit_id"] = identity({k:v for k,v in audit.items() if k != "audit_id"})
    save(case["args"][3], audit)
    with pytest.raises(ValueError, match="SOURCE_AUDIT_"): execute(case)
    assert not case["calls"]


def test_existing_audit_bytes_and_known_inspection_are_retained_not_regenerated(case):
    before = case["args"][3].read_bytes()
    result = execute(case)
    assert result["source_audit_id"] == case["audit"]["audit_id"]
    assert case["args"][3].read_bytes() == before
    job = json.loads((case["output"] / "request.json").read_text())
    assert job["source_audit_id"] == case["audit"]["audit_id"]
    assert job["files"][str(case["args"][3])] == chain.file_id(case["args"][3])
    assert job["runtime"]["source_audit"] == chain.file_id(case["args"][3])


def test_audit_mutation_during_stage_rejects_without_fit(case, monkeypatch):
    old = chain.audit_fr3_prepared_capacity
    def drift(*args):
        value = old(*args)
        case["args"][3].write_bytes(b"SYNTHETIC_CHANGED_AUDIT")
        return value
    monkeypatch.setattr(chain, "audit_fr3_prepared_capacity", drift)
    with pytest.raises(ValueError, match="BOUND_INPUT_OR_RUNNER_CHANGED"): execute(case)
    assert case["calls"] == ["capacity"]


def test_audit_mutation_between_parse_and_hash_cannot_rebind_retained_bytes(case, monkeypatch):
    original_bytes = case["args"][3].read_bytes()
    original_sha = chain.file_id(case["args"][3])
    old = chain.require_source_audit
    def drift(*args):
        old(*args)
        case["args"][3].write_bytes(b"SYNTHETIC_CHANGED_AFTER_FIRST_READ")
    monkeypatch.setattr(chain, "require_source_audit", drift)
    with pytest.raises(ValueError, match="BOUND_INPUT_OR_RUNNER_CHANGED"): execute(case)
    assert not case["calls"]
    job = json.loads((case["output"] / "request.json").read_text())
    assert job["files"][str(case["args"][3])] == original_sha
    assert job["runtime"]["source_audit"] == original_sha
    assert (case["output"] / "objects" / original_sha).read_bytes() == original_bytes
    assert job["source_audit_id"] == case["audit"]["audit_id"]


def test_source_audit_raw_read_enforces_a_bound(case, monkeypatch):
    with pytest.raises(ValueError, match="INPUT_SIZE_LIMIT"):
        chain.read_input(case["args"][3], 1)


@pytest.mark.parametrize("input_name", ["request", "dataset", "observation"])
def test_each_parsed_input_uses_initial_bytes_even_if_changed_before_hash(case, monkeypatch, input_name):
    training = case["args"][0]
    path = {"request": training / "fr3-training-request.json",
        "dataset": training / "native/scope-training-dataset.json", "observation": case["args"][2]}[input_name]
    original = path.read_bytes()
    initial_sha = chain.file_id(path)
    old = chain.require_prepare_exit
    def drift(*args):
        old(*args)
        path.write_bytes(b"SYNTHETIC_CHANGED_AFTER_FIRST_PARSE")
    monkeypatch.setattr(chain, "require_prepare_exit", drift)
    with pytest.raises(ValueError, match="BOUND_INPUT_OR_RUNNER_CHANGED"): execute(case)
    assert not case["calls"]
    job = json.loads((case["output"] / "request.json").read_text())
    assert job["files"][str(path)] == initial_sha
    assert job["runtime"]["input_objects"][str(path)] == initial_sha
    assert (case["output"] / "objects" / initial_sha).read_bytes() == original
    assert not (case["output"] / "completion.json").exists()
