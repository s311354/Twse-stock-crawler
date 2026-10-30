"""SYNTHETIC contracts only; never evidence of market forecasting skill."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from domain.les_v7.contracts import ReferencePlan, identity, timestamp
from infrastructure.les_v7_fr2_expansion import compare_prepared
from tests.strategies.test_low_entry_score_v7 import bundle


def case():
    m0 = bundle()["payload"]
    m0["feature_names"] = ["slope5"]
    m0["feature_bounds"] = [[-10, 10]]
    m1 = deepcopy(m0)
    m1.pop("fractal_mode")
    m1["feature_names"] = ["slope5", "dfa512"]
    m1["feature_bounds"] = [[-10, 10], [0, 3]]
    for key, head in m1["heads"].items():
        m1["heads"][key] = [v + [0.] for v in head] if isinstance(head[0], list) else head + [0.]
    config = dict(feature_names=["slope5", "dfa512"], feature_bounds=[[-10, 10], [0, 3]],
                  fit_cutoff=m0["fit_cutoff"], evaluation_cutoff="2026-03-01T13:30:00+08:00")
    rows = []
    for i, day in enumerate(("2026-01-05", "2026-01-06", "2026-01-07")):
        cutoff = timestamp(day + "T13:30:00+08:00").isoformat()
        symbol = "SYNTHETIC_" + str(i)
        rows.append(dict(row_id=identity(dict(symbol=symbol, market="TWSE", cutoff=cutoff)),
            symbol=symbol, market="TWSE", cutoff=cutoff, label_available_at="2026-02-20T13:30:00+08:00",
            context="REVERSAL", features=dict(close=100., low=99., slope5=.2, dfa512=1.),
            plan=ReferencePlan("REFERENCE_H20", "SYNTHETIC", "SYNTHETIC_SOURCE", "REVERSAL",
                100., 95., 110., 2., (100., 100.5), 1, .05, .05),
            bottom=98., observation_weight=1., labels=dict(fill=1, rebound=1, joint=0, timing=2)))
    return rows, config, m0, m1


def row_digest(rows):
    return identity([dict(row, plan=asdict(row["plan"])) for row in rows])


def test_expansion_uses_identical_native_labels_preserves_publication_and_inputs():
    rows, config, m0, m1 = case()
    # Deliberately later than evaluation. This port is explicitly retrospective,
    # not an attempt to make the ordinary historical runtime accept this model.
    m0["trained_until"] = m1["trained_until"] = "2026-10-18T13:30:00+08:00"
    before = row_digest(rows), identity(config), identity(m0), identity(m1)
    first = compare_prepared(rows, config, m0, m1)
    assert first == compare_prepared(rows, config, m0, m1)
    assert first["events"]["M0"] == first["events"]["M1"]
    assert first["native_metrics"]["M0"]["label_sha256"] == first["native_metrics"]["M1"]["label_sha256"]
    assert all(c["status"] == "NOT_ASSESSABLE" for c in first["comparison"]["criteria"].values())
    assert before == (row_digest(rows), identity(config), identity(m0), identity(m1))
    first["events"]["M0"][0]["forecast"]["p_fill"] = 0
    first["events"]["M0"][0]["labels"]["fill"] = 0
    assert before == (row_digest(rows), identity(config), identity(m0), identity(m1))


@pytest.mark.parametrize("change", ["training_event", "future_label", "duplicate", "future_fit", "wrong_features", "wrong_bounds"])
def test_expansion_rejects_leakage_unpaired_models_and_duplicates(change):
    rows, config, m0, m1 = case()
    if change == "training_event":
        rows[0]["cutoff"] = config["fit_cutoff"]
    elif change == "future_label":
        rows[0]["label_available_at"] = "2027-01-01T00:00:00Z"
    elif change == "duplicate":
        rows += rows[:1]
    elif change == "future_fit":
        m0["last_label_available_at"] = "2027-01-01T00:00:00Z"
    elif change == "wrong_features":
        m0["feature_names"] = ["dfa512"]
    else:
        m1["feature_bounds"] = [[-10, 11], [0, 3]]
    with pytest.raises(ValueError, match="FR2_EXPANSION"):
        compare_prepared(rows, config, m0, m1)


def test_entry_weight_changes_do_not_rewrite_prediction_heads_or_probabilities():
    rows, config, m0, m1 = case()
    before = compare_prepared(rows, config, m0, m1)
    m0["weights"]["REVERSAL"] = [1., 0., 0., 0., 0., 0.]
    after = compare_prepared(rows, config, m0, m1)
    assert after == before
