"""Fresh, lightweight interpreter identity for native work provenance.

Build support and the effective GIL are separate facts: an extension import can
enable the GIL in a free-threaded build. Never infer a disabled GIL from a version
string, an environment variable, or a configured worker count.
"""
from __future__ import annotations

import platform
import sys
import sysconfig


def _optional_bool_probe(owner, name):
    probe = getattr(owner, name, None)
    if probe is None:
        return None
    value = probe()
    if type(value) is not bool:
        raise ValueError("RUNTIME_IDENTITY_PROBE_INVALID:" + name)
    return value


def current_runtime_identity() -> dict:
    """Return detached JSON primitives, including state after prior imports.

    This function deliberately has no memo. Older interpreters without a probe
    report null rather than claiming a successful free-threading/JIT check. JIT
    availability/enabling are stable configuration facts; is_active() is not
    sampled because it describes the currently executing observation itself.
    """
    build = sysconfig.get_config_var("Py_GIL_DISABLED")
    if build not in (None, 0, 1):
        raise ValueError("RUNTIME_IDENTITY_BUILD_FLAG_INVALID")
    jit = getattr(sys, "_jit", None)
    return dict(
        schema="python-runtime-identity/v1",
        implementation=sys.implementation.name,
        python_version=platform.python_version(),
        cache_tag=sys.implementation.cache_tag,
        soabi=sysconfig.get_config_var("SOABI"),
        py_gil_disabled=int(build or 0),
        gil_enabled=_optional_bool_probe(sys, "_is_gil_enabled"),
        jit_available=_optional_bool_probe(jit, "is_available"),
        jit_enabled=_optional_bool_probe(jit, "is_enabled"),
    )
