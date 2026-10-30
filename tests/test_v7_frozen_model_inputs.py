"""SYNTHETIC frozen-input compatibility, not trained forecasting evidence.

Exercise the native immutable daily context without accessing real archives,
market data, SMTP, account state or fitting any model.
"""
from copy import deepcopy

import numpy as np
import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7 import execution_combined_features as combined
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.engine import evaluate
from domain.les_v7.localization import FrozenBundle
from tests.strategies.test_low_entry_score_v7 import bars, bundle, CUTOFF, filings, POLICY
from tests.strategies.test_v7_execution_runtime import extended_bundle as gap_bundle
from tests.strategies.test_v7_execution_atom_runtime import extended_bundle as atom_bundle


def combined_bundle():
    """Compose existing synthetic base heads with the registered union profile."""
    record = bundle()
    model = record["payload"]
    selected = [name for name in combined.FEATURE_NAMES if name not in combined.FRACTAL_DERIVED_NAMES]
    bounds = dict(zip(combined.FEATURE_NAMES, combined.FEATURE_BOUNDS))
    definition = combined.profile()
    model.update(execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=["close"], profile_id=identity(definition),
            selected_features=selected), feature_names=["close"] + selected,
        feature_bounds=[[1., 200.]] + [list(bounds[name]) for name in selected],
        imputation=[None] * (len(selected) + 1), supported_masks=[[True] * (len(selected) + 1)])
    for name, value in model["heads"].items():
        matrix = np.asarray(value)
        model["heads"][name] = np.pad(matrix,
            [(0, 0)] * (matrix.ndim - 1) + [(0, len(selected))]).tolist()
    model["heads"]["fill_only_base_count"] = 1
    model["heads"]["fill"][-1] = .25
    record["sha256"] = identity(model)
    return record


FACTORIES = [bundle, gap_bundle, atom_bundle, combined_bundle]


@pytest.mark.parametrize("factory", FACTORIES, ids=["legacy", "gap-r1", "atom-r2", "union-r3"])
def test_plain_and_immutable_frozen_bundle_load_identically_without_source_mutation(factory):
    record = factory()
    frozen = ReadOnlyEvidence(record)
    before = canonical(record)
    plain_model, frozen_model = FrozenBundle.load(record), FrozenBundle.load(frozen)
    assert plain_model == frozen_model
    assert plain_model.record() == record["payload"]
    assert canonical(record) == canonical(frozen) == before
    changed = frozen_model.record()
    changed["feature_names"].append("MUTATED_DETACHED_COPY")
    changed["weights"]["GLOBAL"][0] = 999
    assert frozen_model.record() == record["payload"]
    with pytest.raises(TypeError, match="READ_ONLY_EVIDENCE"):
        frozen["payload"]["bundle_id"] = "MUTATE_FROZEN_SOURCE"


@pytest.mark.parametrize("factory", FACTORIES, ids=["legacy", "gap-r1", "atom-r2", "union-r3"])
def test_complete_engine_result_is_equal_for_plain_and_immutable_context(factory):
    context = dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
        filings=filings(), quality_policy=deepcopy(POLICY), model_bundle=factory(), receipts={})
    before = canonical(context)
    # The no-fractal fixtures need only the execution profile's bounded
    # causal window; do not make this container regression a DFA benchmark.
    data = bars(150)
    expected = evaluate(data, context)
    actual = evaluate(data, ReadOnlyEvidence(context))
    assert actual == expected
    assert canonical(actual) == canonical(expected)
    assert canonical(context) == before


def test_nested_policy_arrays_keep_canonical_binding_across_immutable_context():
    policy = dict(deepcopy(POLICY), synthetic_nested_policy={"accepted": ["SYNTHETIC_ONLY"]})
    record = gap_bundle()
    record["payload"]["quality_policy"] = deepcopy(policy)
    record["sha256"] = identity(record["payload"])
    context = dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
        filings=filings(), quality_policy=policy, model_bundle=record, receipts={})
    expected = evaluate(bars(100), context)
    assert evaluate(bars(100), ReadOnlyEvidence(context)) == expected
    changed = deepcopy(context)
    changed["quality_policy"]["synthetic_nested_policy"]["accepted"].append("SYNTHETIC_DIFFERENT_POLICY")
    for supplied in (changed, ReadOnlyEvidence(changed)):
        with pytest.raises(ValueError, match="QUALITY_POLICY_BUNDLE_MISMATCH"):
            evaluate(bars(100), supplied)


@pytest.mark.parametrize("factory", [gap_bundle, atom_bundle, combined_bundle], ids=["gap-r1", "atom-r2", "union-r3"])
@pytest.mark.parametrize("damage,reason", [
    ("hash", "MODEL_BUNDLE_HASH_MISMATCH"),
    ("profile", "EXECUTION_FEATURE_PROFILE_MISMATCH"),
    ("bounds", "EXECUTION_FEATURE_COORDINATES_MISMATCH"),
    ("protected", "FILL_ONLY_PROTECTED_HEAD_COORDINATES_MISMATCH"),
])
def test_immutable_model_validation_still_rejects_invalid_content(factory, damage, reason):
    record = factory()
    model = record["payload"]
    if damage == "hash":
        model["bundle_id"] = "SYNTHETIC_CHANGED_WITHOUT_REHASH"
    else:
        if damage == "profile":
            model["execution_feature_profile"]["required_close_count"] += 1
        elif damage == "bounds":
            model["feature_bounds"][-1][-1] += .1
        elif damage == "protected":
            model["heads"]["price"][0][-1] = .1
        record["sha256"] = identity(model)
    frozen = ReadOnlyEvidence(record)
    before = canonical(frozen)
    with pytest.raises(ValueError, match=reason):
        FrozenBundle.load(frozen)
    assert canonical(frozen) == canonical(record) == before
