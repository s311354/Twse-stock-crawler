"""Strict opt-in metadata for fill-only execution features.

The original base feature/plan contract remains unchanged.  This module only
describes a frozen extension to the fill head's input coordinates; it does not
select a profile, fit a model or grant entry permission.
"""
from __future__ import annotations

from domain.les_v7.contracts import identity
from domain.les_v7 import execution_features, execution_atom_features, execution_combined_features

_PROVIDERS = {module.EXECUTION_FEATURE_PROFILE: module
              for module in (execution_features, execution_atom_features, execution_combined_features)}
_RESERVED_NAMES = frozenset(name for module in _PROVIDERS.values() for name in module.FEATURE_NAMES)


def validate_execution_profile(config_or_model: dict) -> tuple[list[str], list[str]] | None:
    """Return declared base/extension names, or ``None`` on the legacy path.

    A profile declaration cannot be partially present, relabel another formula
    or reorder coordinates. Bounds are fixed before fitting. Each exact
    version supports its ordered nonfractal inputs or its full declared inputs.
    """
    fields = ("execution_feature_profile", "fill_only_feature_extension")
    present = [name in config_or_model for name in fields]
    if not any(present):
        if any(name in _RESERVED_NAMES for name in config_or_model.get("feature_names", [])):
            raise ValueError("EXECUTION_PROFILE_DECLARATION_REQUIRED")
        return None
    if not all(present):
        raise ValueError("EXECUTION_PROFILE_DECLARATION_INCOMPLETE")
    declared, extension = (config_or_model[name] for name in fields)
    provider = _PROVIDERS.get(declared.get("version")) if isinstance(declared, dict) else None
    if provider is None:
        raise ValueError("EXECUTION_FEATURE_PROFILE_MISMATCH")
    expected = provider.profile()
    if declared != expected:
        raise ValueError("EXECUTION_FEATURE_PROFILE_MISMATCH")
    if (not isinstance(extension, dict)
            or set(extension) != {"base_feature_names", "profile_id", "selected_features"}
            or extension["profile_id"] != identity(expected)):
        raise ValueError("FILL_ONLY_FEATURE_EXTENSION_MISMATCH")
    base, selected = extension["base_feature_names"], extension["selected_features"]
    ordinary = [name for name in provider.FEATURE_NAMES if name not in provider.FRACTAL_DERIVED_NAMES]
    if (not isinstance(base, list) or not base
            or any(not isinstance(name, str) or not name or name in _RESERVED_NAMES for name in base)
            or len(base) != len(set(base))
            or selected not in (ordinary, list(provider.FEATURE_NAMES))):
        raise ValueError("FILL_ONLY_FEATURE_PARTITION_INVALID")
    names, bounds = config_or_model.get("feature_names"), config_or_model.get("feature_bounds")
    selected_bounds = [list(provider.FEATURE_BOUNDS[provider.FEATURE_NAMES.index(name)]) for name in selected]
    if (names != base + selected or not isinstance(bounds, list)
            or len(bounds) != len(names) or bounds[len(base):] != selected_bounds):
        raise ValueError("EXECUTION_FEATURE_COORDINATES_MISMATCH")
    return list(base), list(selected)


def execution_feature_provider(config_or_model: dict):
    """Return the exact validated version's pure feature module, or ``None``."""
    if validate_execution_profile(config_or_model) is None:
        return None
    return _PROVIDERS[config_or_model["execution_feature_profile"]["version"]]


def execution_base_count(config: dict, current_names: list[str]) -> int | None:
    """Resolve the protected prefix after native same-row fractal ablation.

    Ablation can remove base fractals and derived interactions.  It cannot move
    an extension into the protected prefix or introduce undeclared coordinates.
    """
    partition = validate_execution_profile(config)
    if partition is None:
        return None
    if not isinstance(current_names, list):
        raise ValueError("FILL_ONLY_ACTIVE_FEATURE_PARTITION_INVALID")
    base, selected = partition
    active_base = [name for name in base if name in current_names]
    active_selected = [name for name in selected if name in current_names]
    if not active_base or current_names != active_base + active_selected or not active_selected:
        raise ValueError("FILL_ONLY_ACTIVE_FEATURE_PARTITION_INVALID")
    return len(active_base)
