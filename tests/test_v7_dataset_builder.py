"""Synthetic engineering coverage of causal, source-bound dataset construction.

Every invented bar, calendar, cost and coordinate assertion is explicitly
SYNTHETIC_TEST_ONLY. None is a real historical source or performance claim.
"""
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from application.les_v7_dataset import build_training_dataset, validate_build_config
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.training import prepare_events, prepare_reference
from tests.strategies.test_v7_offline_training import synthetic_config
from tests.test_v7_reference_path import fixture as reference_fixture


def seal(payload):
    return dict(payload=payload, sha256=identity(payload))


def builder_inputs(*, end="2026-01-05", outcome="timeout"):
    """One synthetic REVERSAL history with 540 causal closes and H20 paths."""
    _, _, path_policy, _ = reference_fixture()
    seed = {key: deepcopy(synthetic_config()[key]) for key in
            ("feature_names", "feature_bounds", "context_rules", "reference_policy")}
    seed["reference_policy"].update(reference_shares=1000,
                                   cost_path_policy=deepcopy(path_policy))
    dates = pd.bdate_range(end="2026-01-05", periods=540)
    future = pd.bdate_range(start="2026-01-06", periods=22)
    index = np.arange(len(dates))
    close = 60. + .04 * index + .6 * np.sin(index / 4)
    close[-40:] = np.r_[np.linspace(close[-41], 70., 30),
                      [70.3, 70.6, 70.9, 71.2, 71.1, 71.4, 71.8, 72.2, 72.5, 72.8]]
    close = np.round(close, 1)
    coordinate = "SYNTHETIC_TEST_ONLY:RAW_TWD:3014"

    def bar(day, price):
        price = float(price)
        return dict(Date=day, Open=round(price - .1, 1), High=round(price + .8, 1),
                    Low=round(price - .8, 1), Close=price, Volume=100000., Amount=price * 100000,
                    Bid=round(price - .1, 1), Ask=round(price + .1, 1),
                    **{"TAIEX Close": 10000., "Available At": day + "T13:30:00+08:00",
                       "Complete": True, "Corporate Action State": "NO_ACTION"})

    bars = [bar(d.date().isoformat(), c) for d, c in zip(dates, close)]
    for d in future:
        row = bar(d.date().isoformat(), close[-1])
        row["Open"] = float(close[-1])
        bars.append(row)
    if outcome == "no_fill":
        bars[540].update(Open=74., High=74.5, Low=72.5, Close=72.8)
    elif outcome == "ambiguous":
        bars[541].update(High=90., Low=60.)
    elif outcome != "timeout":
        raise ValueError("UNKNOWN_SYNTHETIC_OUTCOME")
    sessions = [row["Date"] for row in bars]
    first, last = date.fromisoformat(sessions[0]), date.fromisoformat(sessions[-1])
    days = {(first + timedelta(days=i)).isoformat(): "CLOSED"
            for i in range((last - first).days + 1)}
    days.update({s: "OPEN" for s in sessions})
    calendar = dict(source="SYNTHETIC_TEST_ONLY", sessions=sessions, days=days,
                    available_at="2020-01-01T00:00:00+08:00",
                    availability_by_date={s: "2020-01-01T00:00:00+08:00" for s in days},
                    evidence_ids=["SYNTHETIC_FROZEN_CALENDAR"])
    history = dict(symbol="3014", market="TWSE", coordinate_id=coordinate, bars=bars,
                   bar_lineage=[dict(date=s, source_row_ids=["SYNTHETIC_BAR:" + s]) for s in sessions])
    coverage = dict(schema="les-v7-coordinate-coverage/v1", policy=path_policy["coordinate_policy"],
                    coordinate_id=coordinate, market="TWSE", start_session="2026-01-05",
                    end_session=sessions[-1], status="VERIFIED_NO_ACTIONS",
                    evidence_ids=["SYNTHETIC_NO_ACTION_COVERAGE"],
                    available_at=sessions[-1] + "T13:30:00+08:00")
    config = dict(schema="les-v7-dataset-build-config/v1", event_start="2026-01-05", event_end=end,
                  evaluation_cutoff="2026-02-10T13:30:00+08:00", max_events=3,
                  event_sampling="ALL_FIXED_UNIVERSE_OFFICIAL_SESSIONS", seed=seed,
                  path_policy=path_policy, coordinate_coverages=[coverage])
    source = dict(schema="les-v7-training-source/v1", dataset_kind="SYNTHETIC_TEST_ONLY",
                  universe=[["3014", "TWSE"]], histories=[history], calendars=dict(TWSE=calendar),
                  price_unit="TWD", volume_unit="SHARES", amount_unit="TWD",
                  source_evidence_ids=["SYNTHETIC_SOURCE_NOT_OFFICIAL"],
                  universe_policy="SYNTHETIC_FIXED_UNIVERSE_WITHOUT_WINNER_FILTER",
                  availability_policy="SYNTHETIC_SESSION_CLOSE_FIXTURE",
                  historical_information_set_certified=False,
                  lineage=dict(evaluation_cutoff=config["evaluation_cutoff"]), diagnostics=[])
    return seal(source), config


def reseal(source):
    source["sha256"] = identity(source["payload"])
    return source


def test_closed_calendar_publication_is_part_of_label_maturity():
    source, config = builder_inputs()
    calendar = source["payload"]["calendars"]["TWSE"]
    calendar["days"]["2026-01-07"] = "CLOSED"
    calendar["sessions"].remove("2026-01-07")
    calendar["availability_by_date"]["2026-01-07"] = "2026-02-08T00:00:00+08:00"
    history = source["payload"]["histories"][0]
    history["bars"] = [b for b in history["bars"] if b["Date"] != "2026-01-07"]
    history["bar_lineage"] = [r for r in history["bar_lineage"] if r["date"] != "2026-01-07"]
    reseal(source)
    built = build_training_dataset(source, config)
    rows, _ = prepare_events(built["dataset"], dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"]))
    assert len(rows) == 1
    assert timestamp(rows[0]["label_available_at"]) == timestamp("2026-02-08T00:00:00+08:00")


def test_synthetic_adapter_calendar_is_explicitly_marked_in_training_export():
    source, config = builder_inputs()
    source["payload"]["calendars"]["TWSE"]["source"] = "TWSE_OFFICIAL"
    reseal(source)
    built = build_training_dataset(source, config)
    cal = built["dataset"]["payload"]["calendars"]["TWSE"]
    assert cal["source"] == "SYNTHETIC_TEST_ONLY"
    assert cal["source_archive_calendar_source"] == "TWSE_OFFICIAL"
    assert built["result"]["status"] == "EVIDENCE_READY"


def test_rehashed_fabricated_execution_is_rejected_by_native_re_evaluation():
    source, config = builder_inputs()
    built = build_training_dataset(source, config)
    altered = deepcopy(built["dataset"])
    event = altered["payload"]["events"][0]
    event["execution"]["realized_net_pnl"] = 999999.
    event["execution_sha256"] = identity(event["execution"])
    altered["sha256"] = identity(altered["payload"])
    with pytest.raises(ValueError, match="SIMULATED_EXECUTION_RECOMPUTATION_MISMATCH"):
        prepare_events(altered, dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"]))


def test_native_build_is_deterministic_no_mutation_and_preserves_actual_reference_labels():
    source, config = builder_inputs()
    original = deepcopy((source, config))
    first = build_training_dataset(source, config)
    second = build_training_dataset(source, config)
    assert first == second and (source, config) == original
    assert first["result"]["status"] == "EVIDENCE_READY"
    assert first["result"]["counts"] == dict(candidate_events=1, path_events=1, eligible_events=1, labels={1: 1})
    assert first["dataset"]["payload"]["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert first["result"]["research_only"] and not first["result"]["approved_for_live"]
    assert first["dataset"]["sha256"] == identity(first["dataset"]["payload"])
    event = first["dataset"]["payload"]["events"][0]
    assert event["execution_sha256"] == identity(event["execution"])
    assert event["execution"]["exit_reason"] == "H20_CLOSE"
    assert event["execution"]["historical_deployment_claim"] is False


def test_builder_plan_ids_and_labels_exactly_match_native_training_selection():
    source, config = builder_inputs()
    built = build_training_dataset(source, config)
    rows, diagnostics = prepare_events(built["dataset"], dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"]))
    event = built["dataset"]["payload"]["events"][0]
    assert rows[0]["plan"].plan_id == event["execution"]["plan_id"]
    assert rows[0]["labels"]["fill"] == 1
    assert rows[0]["labels"]["rebound"] == 0 and rows[0]["labels"]["joint"] == 0
    assert diagnostics[0]["plan"] == built["result"]["diagnostics"][0]["plan"]
    assert built["result"]["label_diagnostics"] == diagnostics


def test_future_price_changes_may_change_labels_but_not_causal_numeric_features_or_plan():
    source, config = builder_inputs()
    altered = deepcopy(source)
    altered["payload"]["histories"][0]["bars"][541].update(Open=72.8, High=90., Low=72.0, Close=80.)
    reseal(altered)
    before, after = (build_training_dataset(item, config) for item in (source, altered))
    assert before["dataset"]["sha256"] != after["dataset"]["sha256"]
    assert before["dataset"]["payload"]["events"][0]["execution"]["exit_reason"] == "H20_CLOSE"
    assert after["dataset"]["payload"]["events"][0]["execution"]["exit_reason"] == "NET_2R"
    seeds = [prepare_reference("3014", "TWSE", pd.DataFrame(item["payload"]["histories"][0]["bars"]),
               "2026-01-05T13:30:00+08:00", config["seed"], [*item["payload"]["source_evidence_ids"], item["sha256"]])
             for item in (source, altered)]
    assert seeds[0]["features"] == seeds[1]["features"]
    plans = [dict(d["plan"]) for d in (before["result"]["diagnostics"][0], after["result"]["diagnostics"][0])]
    assert plans[0].pop("evidence_id") != plans[1].pop("evidence_id")
    assert plans[0] == plans[1]


def test_ambiguous_daily_bar_path_is_retained_but_never_a_favorable_training_label():
    source, config = builder_inputs(outcome="ambiguous")
    built = build_training_dataset(source, config)
    assert built["result"]["status"] == "PARTIAL"
    assert built["result"]["counts"]["eligible_events"] == 0
    assert built["result"]["counts"]["path_events"] == 1
    assert built["dataset"]["payload"]["events"][0]["execution"]["path_order_verified"] is False
    assert built["result"]["reason_counts"]["PATH_AMBIGUOUS"] >= 1


def test_verified_no_fill_requires_complete_source_path_and_keeps_conditional_labels_null():
    source, config = builder_inputs(outcome="no_fill")
    built = build_training_dataset(source, config)
    assert built["result"]["status"] == "EVIDENCE_READY"
    rows, _ = prepare_events(built["dataset"], dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"]))
    assert rows[0]["labels"]["fill"] == 0
    assert rows[0]["labels"]["rebound"] is None and rows[0]["labels"]["joint"] is None
    assert rows[0]["utility"] is None
    missing = deepcopy(source)
    missing["payload"]["histories"][0]["bars"].pop(540)
    reseal(missing)
    gap = build_training_dataset(missing, config)
    assert gap["result"]["status"] == "PARTIAL"
    assert gap["result"]["counts"]["path_events"] == 0
    assert gap["result"]["counts"]["eligible_events"] == 0
    assert gap["dataset"]["payload"]["events"] == []


@pytest.mark.parametrize("problem", ["calendar", "publication", "coordinate", "lineage", "zero_volume", "warmup"])
def test_real_evidence_gaps_remain_partial_without_manufacturing_executable_labels(problem):
    source, config = builder_inputs()
    history = source["payload"]["histories"][0]
    calendar = source["payload"]["calendars"]["TWSE"]
    if problem == "calendar":
        calendar["days"][history["bars"][542]["Date"]] = "UNKNOWN"
    elif problem == "publication":
        calendar["availability_by_date"].pop(history["bars"][542]["Date"])
    elif problem == "coordinate":
        config["coordinate_coverages"] = []
    elif problem == "lineage":
        history["bar_lineage"][542]["source_row_ids"] = []
    elif problem == "zero_volume":
        history["bars"][542]["Volume"] = 0
    else:
        history["bars"] = history["bars"][-100:]
    reseal(source)
    built = build_training_dataset(source, config)
    assert built["result"]["status"] == "PARTIAL"
    assert built["result"]["counts"]["eligible_events"] == 0
    assert built["result"]["reason_counts"]


def test_all_three_fixed_universe_dates_are_retained_without_buy_filter():
    source, config = builder_inputs(end="2026-01-07")
    built = build_training_dataset(source, config)
    assert built["result"]["counts"]["candidate_events"] == 3
    assert [d["cutoff"][:10] for d in built["result"]["diagnostics"]] == ["2026-01-05", "2026-01-06", "2026-01-07"]
    assert built["dataset"]["payload"]["universe"] == source["payload"]["universe"]
    config["max_events"] = 2
    with pytest.raises(ValueError, match="COUNT_EXCEEDS_DECLARED_BOUND"):
        build_training_dataset(source, config)


@pytest.mark.parametrize("problem", ["hash", "schema", "units", "universe", "cutoff", "filter", "cost_policy"])
def test_inconsistent_inputs_fail_closed_before_a_dataset_is_returned(problem):
    source, config = builder_inputs()
    if problem == "hash":
        source["sha256"] = "0" * 64
    elif problem == "schema":
        source["payload"]["schema"] = "UNRECOGNIZED_SOURCE"
    elif problem == "units":
        source["payload"]["price_unit"] = "UNKNOWN"
    elif problem == "universe":
        source["payload"]["universe"] = [["6834", "TWSE"]]
    elif problem == "cutoff":
        source["payload"]["lineage"]["evaluation_cutoff"] = "2026-02-11T13:30:00+08:00"
    elif problem == "filter":
        config["event_sampling"] = "BUY_SIGNALS_ONLY"
    else:
        config["seed"]["reference_policy"]["cost_path_policy"]["cost_policy"]["buy_rate"] = ".002"
    if problem != "hash":
        reseal(source)
    with pytest.raises(ValueError):
        build_training_dataset(source, config)


def test_references_cannot_be_reinterpreted_under_another_training_seed():
    source, config = builder_inputs()
    built = build_training_dataset(source, config)
    changed = dict(deepcopy(config["seed"]), evaluation_cutoff=config["evaluation_cutoff"])
    changed["reference_policy"]["entry_band_atr"] = [0., .5]
    with pytest.raises(ValueError, match="REFERENCE_CONFIG_MISMATCH"):
        prepare_events(built["dataset"], changed)


def test_builder_configuration_rejects_invalid_bounds_and_hidden_sample_filters():
    _, config = builder_inputs()
    validate_build_config(config)
    config["event_start"] = "2026-01-06"
    with pytest.raises(ValueError, match="BOUNDED_DATASET_EVENT_RANGE"):
        validate_build_config(config)
