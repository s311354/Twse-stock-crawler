"""Exact union of the two existing causal execution coordinate systems.

No new estimator, target, reference band, price grid or trading rule is added.
Keeping both empirical gap statistics and atom-aware legal-band coordinates
lets an offline fill head test their complementary information.
"""
from __future__ import annotations

from domain.les_v7 import execution_features as gaps, execution_atom_features as atoms
from domain.les_v7.contracts import identity
from domain.les_v7.features import select_bars

EXECUTION_FEATURE_PROFILE = "V7_EXECUTION_GAP_ATOM_UNION_R3"
FRACTAL_DERIVED_NAMES = (*gaps.FRACTAL_DERIVED_NAMES, *atoms.FRACTAL_DERIVED_NAMES)
FEATURE_NAMES = (*(name for module in (gaps, atoms) for name in module.FEATURE_NAMES
                   if name not in module.FRACTAL_DERIVED_NAMES), *FRACTAL_DERIVED_NAMES)
_BOUNDS = {name: bound for module in (gaps, atoms) for name, bound in zip(module.FEATURE_NAMES, module.FEATURE_BOUNDS)}
FEATURE_BOUNDS = tuple(_BOUNDS[name] for name in FEATURE_NAMES)
REQUIRED_CLOSE_COUNT = max(gaps.REQUIRED_CLOSE_COUNT, atoms.REQUIRED_CLOSE_COUNT)


def profile():
    return dict(version=EXECUTION_FEATURE_PROFILE, feature_names=list(FEATURE_NAMES),
        feature_bounds=[list(bound) for bound in FEATURE_BOUNDS], required_close_count=REQUIRED_CLOSE_COUNT,
        component_profiles=[gaps.profile(), atoms.profile()],
        component_profile_ids=[identity(gaps.profile()), identity(atoms.profile())],
        composition="EXACT_ORDERED_UNION_NO_AVERAGING_OR_FEATURE_REFIT",
        missing_policy="NOT_READY_IF_EITHER_COMPONENT_NOT_READY_NO_IMPUTATION",
        support_policy="OUT_OF_SUPPORT_NO_CLIPPING", fractal_interactions={
            **gaps.profile()['fractal_interactions'], **atoms.profile()['fractal_interactions']},
        interpretation="CAUSAL_EXECUTION_FEATURES_NOT_FITTED_FILL_PROBABILITIES")


def compute_execution_features(data, *, cutoff, base_features, plan):
    return _from_selected_bars(select_bars(data, cutoff), base_features=base_features, plan=plan)


def _from_selected_bars(data, *, base_features, plan):
    first, first_audit = gaps._from_selected_bars(data, base_features=base_features, plan=plan)
    second, second_audit = atoms._from_selected_bars(data, base_features=base_features, plan=plan)
    values = dict(first, **second)
    ready = first_audit['status'] == second_audit['status'] == 'COMPUTED'
    status = 'COMPUTED' if ready else ('OOD' if 'OOD' in (first_audit['status'], second_audit['status']) else 'NOT_READY')
    audit = dict(profile=EXECUTION_FEATURE_PROFILE, status=status,
        available_close_count=len(data), required_close_count=REQUIRED_CLOSE_COUNT,
        cutoff_session=first_audit['cutoff_session'], component_audits=[first_audit, second_audit],
        reasons=sorted(set(first_audit.get('reasons', []) + second_audit.get('reasons', []))))
    return {name: values[name] for name in FEATURE_NAMES}, audit
