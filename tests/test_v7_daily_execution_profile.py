"""SYNTHETIC_ONLY daily adapter regressions; no model skill/approval claim.

The real public daily adapter, immutable evidence context, native v7 kernel,
native report verification and optional EPS/FR2 overlay all execute. Fixtures
provide synthetic prices/evidence only; no network, SMTP or account store runs.
"""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from application import low_entry_v7_daily as daily
from application.evidence_context import ReadOnlyEvidence
from application.market_history import MarketHistoryAssembler
from domain.les_v7 import execution_features, execution_atom_features, execution_combined_features
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr2_policy import restore_original
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.report import validate_report
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from tests.strategies.test_daily_v7_pipeline import synthetic_market
from tests.strategies.test_low_entry_score_v7 import bars
from tests.test_v7_fr2_eps_daily import research_history


PROVIDERS = (execution_features, execution_atom_features, execution_combined_features)


def extended_model(record, provider):
    """Explicit synthetic fill extension; other four heads are unchanged."""
    record = deepcopy(record)
    payload = record["payload"]
    base = list(payload["feature_names"])
    selected = [name for name in provider.FEATURE_NAMES if name not in provider.FRACTAL_DERIVED_NAMES]
    definition = provider.profile()
    payload.update(execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=base, profile_id=identity(definition),
            selected_features=selected),
        feature_names=base + selected,
        feature_bounds=payload["feature_bounds"] + [list(provider.FEATURE_BOUNDS[
            provider.FEATURE_NAMES.index(name)]) for name in selected],
        imputation=payload["imputation"] + [None] * len(selected),
        supported_masks=[mask + [True] * len(selected) for mask in payload["supported_masks"]])
    for name in ("price", "timing", "fill", "rebound", "joint"):
        matrix = np.asarray(payload["heads"][name])
        payload["heads"][name] = np.pad(matrix,
            [(0, 0)] * (matrix.ndim - 1) + [(0, len(selected))]).tolist()
    payload["heads"].update(fill_only_base_count=len(base))
    payload["heads"]["fill"][-1] = 2.
    record["sha256"] = identity(payload)
    return record


def history_case(provider, *, research):
    source = research_history() if research else MarketHistoryAssembler().build(synthetic_market(), 0)
    # Native daily prices use the legacy x1000 scale. Give both modes the
    # same synthetic plan-ready path, aligned to their actual declared signal
    # session; no real market/data gap is repaired or assumed away here.
    history = bars(600)
    history["Date"] = pd.bdate_range(end=str(source.Date.iloc[-1])[:10],
        periods=len(history)).strftime("%Y-%m-%d").tolist()
    for name in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        history[name] *= 1000.
    history.attrs = deepcopy(source.attrs)
    model = extended_model(json.loads(history.attrs["les_v7_bundle_json"]), provider)
    history.attrs["les_v7_bundle_json"] = json.dumps(model)
    if research:
        metadata = json.loads(history.attrs["les_v7_research_metadata_json"])
        metadata["bundle_sha256"] = model["sha256"]
        history.attrs["les_v7_research_metadata_json"] = json.dumps(metadata)
    return history, model


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda module: module.EXECUTION_FEATURE_PROFILE)
@pytest.mark.parametrize("research", [False, True], ids=["daily-configured", "research-eps-fr2"])
def test_daily_immutable_model_profiles_match_json_containers_end_to_end(monkeypatch, provider, research):
    history, model = history_case(provider, research=research)
    before_frame = history.copy(deep=True)
    before_snapshot = canonical(history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR])
    before_bundle = history.attrs["les_v7_bundle_json"]
    real_load = FrozenBundle.load
    frozen_inputs = []

    def observe_load(cls, record):
        if isinstance(record, ReadOnlyEvidence):
            frozen_inputs.append(record)
            assert isinstance(record["payload"]["execution_feature_profile"]["feature_names"], tuple)
        return real_load(record)

    monkeypatch.setattr(FrozenBundle, "load", classmethod(observe_load))
    actual = daily.daily_report(history)
    assert frozen_inputs  # Exercise the exact adapter -> frozen model boundary.
    validate_report(actual)
    diagnostics = actual["diagnostics"]
    assert actual["envelope"]["model"]["bundle_sha256"] == model["sha256"]
    assert diagnostics["execution_feature_audit"]["profile"] == provider.EXECUTION_FEATURE_PROFILE
    assert diagnostics["execution_feature_audit"]["status"] == "COMPUTED"
    assert diagnostics["localization"]["status"] == "VALID"
    assert set(diagnostics["execution_features"]) == set(model["payload"]["fill_only_feature_extension"]["selected_features"])

    # Public JSON semantics are the reference, not a regenerated expected
    # forecast. Only container representation changes; all actual validators
    # and the native engine continue to execute on this comparison path.
    class PlainJsonEvidence(dict):
        def __init__(self, value):
            super().__init__(json.loads(canonical(value)))

    monkeypatch.setattr(daily, "ReadOnlyEvidence", PlainJsonEvidence)
    expected = daily.daily_report(history)
    assert actual == expected  # Full signals, gates, forecasts, scores and allocation.
    pd.testing.assert_frame_equal(history, before_frame)
    assert canonical(history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR]) == before_snapshot
    assert history.attrs["les_v7_bundle_json"] == before_bundle
    if research:
        original = restore_original(actual)
        assert diagnostics["fr2"]["schema"] == "les-v7-fr2-eps-overlay/v1"
        for name in ("features", "reference_plan", "localization", "execution_features", "execution_feature_audit"):
            assert diagnostics[name] == original["diagnostics"][name]
        assert actual["fr2_execution"]["live_order_authorized"] is False
        assert actual["fr2_execution"]["filled_shares"] == 0
        assert "pae_authorization" not in actual
    else:
        assert "fr2" not in diagnostics
        assert actual["pae_authorization"]["authorized"] is False


def test_daily_frozen_model_still_rejects_rehashed_profile_formula_change():
    history, model = history_case(execution_combined_features, research=False)
    model["payload"]["execution_feature_profile"]["component_profiles"][1]["jeffreys_alpha"] = 1.
    model["sha256"] = identity(model["payload"])
    history.attrs["les_v7_bundle_json"] = json.dumps(model)
    with pytest.raises(ValueError, match="EXECUTION_FEATURE_PROFILE_MISMATCH"):
        daily.daily_report(history)
