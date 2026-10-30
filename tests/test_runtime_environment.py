"""Synthetic build/runtime observations; no fitting or live source collection."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from infrastructure import runtime_environment as environment


def install_probe(monkeypatch, *, free_threaded=False, gil=True, jit=True):
    system = SimpleNamespace(implementation=SimpleNamespace(name="cpython", cache_tag="cpython-315"))
    if gil is not None:
        system._is_gil_enabled = lambda: gil
    if jit is not None:
        system._jit = SimpleNamespace(is_available=lambda: jit, is_enabled=lambda: False)
    config = {"Py_GIL_DISABLED": int(free_threaded),
              "SOABI": "cpython-315t-test" if free_threaded else "cpython-315-test"}
    monkeypatch.setattr(environment, "sys", system)
    monkeypatch.setattr(environment.sysconfig, "get_config_var", config.get)
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.15.0")
    return system, config


def test_same_python_version_regular_and_free_threaded_builds_have_distinct_identities(monkeypatch):
    install_probe(monkeypatch)
    regular = environment.current_runtime_identity()
    install_probe(monkeypatch, free_threaded=True, gil=False)
    threaded = environment.current_runtime_identity()
    assert regular["python_version"] == threaded["python_version"] == "3.15.0"
    assert regular["py_gil_disabled"] == 0 and regular["gil_enabled"] is True
    assert threaded["py_gil_disabled"] == 1 and threaded["gil_enabled"] is False
    assert sha256(json.dumps(regular, sort_keys=True).encode()).hexdigest() != sha256(
        json.dumps(threaded, sort_keys=True).encode()).hexdigest()
    assert json.loads(json.dumps(threaded, allow_nan=False)) == threaded


def test_runtime_probe_is_fresh_and_detects_extension_induced_gil_fallback(monkeypatch):
    system, _ = install_probe(monkeypatch, free_threaded=True, gil=False)
    before = environment.current_runtime_identity()
    system._is_gil_enabled = lambda: True
    after = environment.current_runtime_identity()
    assert before["py_gil_disabled"] == after["py_gil_disabled"] == 1
    assert before["gil_enabled"] is False and after["gil_enabled"] is True
    after["gil_enabled"] = False
    assert environment.current_runtime_identity()["gil_enabled"] is True


def test_unavailable_observation_apis_are_unknown_not_success(monkeypatch):
    _, config = install_probe(monkeypatch, gil=None, jit=None)
    config["Py_GIL_DISABLED"] = None
    value = environment.current_runtime_identity()
    assert value["py_gil_disabled"] == 0
    assert value["gil_enabled"] is None
    assert value["jit_available"] is value["jit_enabled"] is None


@pytest.mark.parametrize("name", ["gil", "jit_available", "jit_enabled", "build"])
def test_invalid_observation_is_not_silently_interpreted_as_disabled(monkeypatch, name):
    system, config = install_probe(monkeypatch)
    if name == "gil":
        system._is_gil_enabled = lambda: 0
    elif name == "jit_available":
        system._jit.is_available = lambda: "false"
    elif name == "jit_enabled":
        system._jit.is_enabled = lambda: None
    else:
        config["Py_GIL_DISABLED"] = "0"
    with pytest.raises(ValueError, match="RUNTIME_IDENTITY"):
        environment.current_runtime_identity()


def identity(*, free=False, gil=True):
    return dict(schema="python-runtime-identity/v1", implementation="cpython",
                python_version="3.15.0", cache_tag="cpython-315", soabi="SYNTHETIC_ABI",
                py_gil_disabled=int(free), gil_enabled=gil, jit_available=True, jit_enabled=False)


def lightweight_guard(monkeypatch, state):
    from application.replay_contracts import ReplayVersions, digest
    from infrastructure import research_objects as objects
    parameters = dict(schema="SYNTHETIC_FROZEN_PARAMETERS")
    versions = ReplayVersions("SYNTHETIC", digest(parameters), "SYNTHETIC_CODE")
    # Freeze the existing parameter/environment scans so this proves the new
    # runtime observation itself rejects a transition, independently of them.
    monkeypatch.setattr(objects.runtime, "current_versions", lambda: versions)
    monkeypatch.setattr(objects.runtime, "parameter_manifest", lambda: deepcopy(parameters))
    monkeypatch.setattr(objects, "current_runtime_identity", lambda: deepcopy(state))
    monkeypatch.setattr(objects, "source_paths", lambda: [])
    monkeypatch.setattr(objects, "distributions", lambda: [])
    return objects.ResearchVersionGuard()


def test_native_parameter_manifest_binds_runtime_mode_without_changing_model_parameters(monkeypatch):
    from infrastructure import replay_runtime
    state = identity()
    monkeypatch.setattr(replay_runtime, "current_runtime_identity", lambda: deepcopy(state))
    regular = replay_runtime.parameter_manifest()
    state.update(py_gil_disabled=1, gil_enabled=False)
    threaded = replay_runtime.parameter_manifest()
    assert regular["runtime"] != threaded["runtime"]
    assert {k: v for k, v in regular.items() if k != "runtime"} == {
        k: v for k, v in threaded.items() if k != "runtime"}


@pytest.mark.parametrize("field,new", [("py_gil_disabled", 0), ("gil_enabled", True),
    ("soabi", "CHANGED_ABI"), ("cache_tag", "CHANGED_TAG"),
    ("implementation", "CHANGED_IMPLEMENTATION"), ("jit_enabled", True)])
def test_guard_rejects_same_version_runtime_build_or_effective_gil_change(monkeypatch, field, new):
    state = identity(free=True, gil=False)
    guard = lightweight_guard(monkeypatch, state)
    assert guard.check() == guard.versions
    assert guard.environment["runtime"] == state
    state[field] = new
    with pytest.raises(ValueError, match="REPLAY_VERSION_CHANGED_DURING_JOB"):
        guard.check()


def test_legacy_stored_environment_is_read_verbatim_without_runtime_rewrite(tmp_path):
    from infrastructure.research_objects import ResearchObjects
    legacy = dict(python="3.14.7", platform="SYNTHETIC_OLD_PLATFORM", dependencies=[["numpy", "old"]])
    store = ResearchObjects(tmp_path / "objects")
    reference = store.put_json(legacy)
    before = store.get(reference)
    assert store.get_json(reference) == legacy
    assert "runtime" not in store.get_json(reference)
    assert store.get(reference) == before
