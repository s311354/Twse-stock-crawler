"""SYNTHETIC integration fixtures, not authenticated historical market proof.

Transport documents, calendars, prices and paths originate in the existing
explicitly synthetic helpers. Tests never contact TWSE or change real data.
"""
from copy import deepcopy
import hashlib
import json
from urllib.parse import urlencode

import pytest

from application.les_v7_dataset import build_training_dataset
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.reference_path import evaluate_reference_path
from domain.les_v7.training import prepare_events, validate_training_config
from infrastructure.les_v7_actions_final import (
    FEEDS, _packet_id, build_final_coordinate_coverages, parse_actions_response, verify_packet,
)
from infrastructure.les_v7_training_source import FINAL_CLOSE, FINAL_REFERENCE, load_training_source
from tests.test_v7_actions_final import packet as actions_fixture
from tests.test_v7_calendar_final import packet as calendar_fixture
from tests.test_v7_dataset_builder import builder_inputs, reseal
from tests.test_v7_reference_path import fixture as path_fixture
from tests.test_v7_training_source import CUTOFF, fixture as archive_fixture, save
from tests.strategies.test_v7_offline_training import synthetic_config

FINAL_COORDINATE = "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1"


def action_packet(*, start="2026-01-05", end="2026-02-10", action_day=None):
    """Rescope the synthetic official-shape fixture; rebind every original byte."""
    value = actions_fixture()
    value.update(start=start, end=end)
    for row in value["feeds"]:
        feed = row["feed"]
        payload = json.loads(row["document"]["body_utf8"])
        first, last = start.replace("-", ""), end.replace("-", "")
        if feed == "TWTB8U":
            payload["params"].update(startDate=first, endDate=last)
        else:
            payload.update(strDate=first, endDate=last)
        if action_day and feed == "TWT49U":
            year, month, day = map(int, action_day.split("-"))
            payload["data"] = [[f"{year-1911}年{month}月{day}日", "3014", "SYNTHETIC_TEST_ONLY",
                                  "SYNTHETIC_TEST_ONLY", "SYNTHETIC_TEST_ONLY", "SYNTHETIC_TEST_ONLY", "SYNTHETIC_TEST_ONLY"]]
        body = json.dumps(payload, ensure_ascii=False).encode()
        query = dict(startDate=first, endDate=last, response="json")
        row.update(query=query, parsed=parse_actions_response(feed, body, start=start, end=end))
        row["document"].update(
            url="https://www.twse.com.tw/rwd/zh" + FEEDS[feed][0] + "?" + urlencode(query),
            sha256=hashlib.sha256(body).hexdigest(), body_utf8=body.decode())
    value["evidence_id"] = _packet_id(value)
    verify_packet(value)
    return value


def final_builder(*, action_day=None):
    source, config = builder_inputs()
    config["path_policy"]["coordinate_policy"] = FINAL_COORDINATE
    config["seed"]["reference_policy"]["cost_path_policy"] = deepcopy(config["path_policy"])
    config["reference_evidence_policy"] = FINAL_REFERENCE
    config["coordinate_coverages"] = []
    source["payload"]["reference_evidence_policy"] = FINAL_REFERENCE
    packet = action_packet(action_day=action_day)
    source["payload"]["coordinate_coverages"] = build_final_coordinate_coverages(
        source["payload"]["histories"], packet)
    return reseal(source), config


def final_archive(tmp_path, *, actions=True):
    original = archive_fixture()
    calendar, _ = calendar_fixture(tmp_path)
    original["calendar"] = calendar
    if actions:
        original["corporate_actions_final"] = actions_fixture()
    original["reference_evidence_policy"] = FINAL_REFERENCE
    return save(tmp_path, original), original


def test_reference_final_calendar_requires_separate_explicit_bar_opt_in(tmp_path):
    path, _ = final_archive(tmp_path)
    with pytest.raises(ValueError, match="REQUIRES_EXPLICIT_FINAL_DAILY"):
        load_training_source(path, evaluation_cutoff=CUTOFF, reference_evidence_policy=FINAL_REFERENCE)
    with pytest.raises(ValueError, match="INVALID_REFERENCE_EVIDENCE_POLICY"):
        load_training_source(path, evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE,
                             reference_evidence_policy="AUTO")


def test_default_does_not_use_retrospective_calendar_as_historical_publication(tmp_path):
    path, _ = final_archive(tmp_path)
    default = load_training_source(path, evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE)
    final = load_training_source(path, evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE,
                                 reference_evidence_policy=FINAL_REFERENCE)
    assert set(default["payload"]["calendars"]["TWSE"]["days"].values()) == {"UNKNOWN"}
    assert final["payload"]["calendars"]["TWSE"]["sessions"] == ["2026-08-03", "2026-08-04", "2026-08-05"]
    assert default["sha256"] != final["sha256"]
    assert not final["payload"]["historical_information_set_certified"]
    assert "reference_evidence_policy" not in default["payload"]
    assert "coordinate_coverages" not in default["payload"]
    assert "final_reference_evidence_policy" not in default["payload"]["lineage"]
    assert default == load_training_source(path, evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE)


def test_adapter_retains_late_fetch_and_null_original_time_with_close_convention(tmp_path):
    path, original = final_archive(tmp_path)
    before = deepcopy(original)
    result = load_training_source(path, evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE,
                                  reference_evidence_policy=FINAL_REFERENCE)["payload"]
    assert original == before
    bar = result["histories"][0]["bars"][0]
    assert bar["Available At"] == "2026-08-03T13:30:00+08:00"
    assert result["histories"][0]["bar_lineage"][0]["original_observed_at"] is None
    coverage = result["coordinate_coverages"][0]
    assert timestamp(coverage["fetched_at"]) > timestamp(CUTOFF)
    assert coverage["available_at"] == coverage["fetched_at"]
    assert coverage["historical_available_at"] is None
    assert coverage["actions_packet_id"] == original["corporate_actions_final"]["evidence_id"]
    assert original["calendar"]["documents"]["2026"]["document"]["retrieved_at"] == "2026-10-13T00:00:00+00:00"


def test_empty_complete_action_feeds_and_absent_evidence_are_not_equivalent(tmp_path):
    complete_path, _ = final_archive(tmp_path / "complete")
    missing_path, _ = final_archive(tmp_path / "missing", actions=False)
    options = dict(evaluation_cutoff=CUTOFF, availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)
    complete = load_training_source(complete_path, **options)["payload"]
    missing = load_training_source(missing_path, **options)["payload"]
    assert complete["coordinate_coverages"]
    assert missing["coordinate_coverages"] == []
    assert "FINAL_ACTIONS_SOURCE_NOT_COLLECTED" in missing["diagnostics"]["preparation_gaps"]
    assert all(c["historical_information_set_certified"] is False for c in complete["coordinate_coverages"])


@pytest.mark.parametrize("marker", ["X", " <p>X</p> "])
def test_unresolved_raw_action_marker_not_hidden_by_empty_feed_or_display_wrapper(tmp_path, marker):
    _, original = final_archive(tmp_path)
    original["bars"][0]["data"][0][9] = marker
    result = load_training_source(save(tmp_path, original), evaluation_cutoff=CUTOFF,
        availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)["payload"]
    segments = [c for c in result["coordinate_coverages"] if c["symbol"] == "3014"]
    assert all(not c["start_session"] <= "2026-08-03" <= c["end_session"] for c in segments)
    assert original["bars"][0]["data"][0][9] == marker


@pytest.mark.parametrize("change", ["document", "fetched_time", "missing_feed", "false_historical_time"])
def test_adapter_rechecks_action_originals_and_time_binding(tmp_path, change):
    _, original = final_archive(tmp_path)
    packet = original["corporate_actions_final"]
    if change == "document":
        packet["feeds"][0]["document"]["body_utf8"] += " "
    elif change == "fetched_time":
        packet["feeds"][0]["fetched_at"] = "2026-10-14T10:00:00+08:00"
    elif change == "missing_feed":
        packet["feeds"].pop()
    else:
        packet["historical_available_at"] = "2026-01-01T00:00:00+08:00"
    with pytest.raises(ValueError):
        load_training_source(save(tmp_path, original), evaluation_cutoff=CUTOFF,
                             availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)


def test_builder_consumes_source_bound_final_coverages_and_native_selection_matches():
    source, config = final_builder()
    before = deepcopy((source, config))
    built = build_training_dataset(source, config)
    assert (source, config) == before
    assert built["result"]["counts"]["eligible_events"] == 1
    assert built["dataset"]["payload"]["builder_manifest"]["coordinate_coverages"] == source["payload"]["coordinate_coverages"]
    selection = dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"], reference_evidence_policy=FINAL_REFERENCE)
    rows, _ = prepare_events(built["dataset"], selection)
    assert len(rows) == 1 and rows[0]["labels"]["fill"] == 1
    event = built["dataset"]["payload"]["events"][0]
    assert event["execution"]["source_observed_at"] == timestamp("2026-10-13T10:00:00+08:00").isoformat()
    assert event["execution"]["original_historical_available_at"] is None
    assert timestamp(rows[0]["label_available_at"]) < timestamp(config["evaluation_cutoff"])
    assert built["result"]["research_only"] and not built["result"]["approved_for_live"]


def test_builder_and_trainer_reject_policy_omission_or_rehashed_policy_tamper():
    source, config = final_builder()
    old = deepcopy(config)
    old.pop("reference_evidence_policy")
    with pytest.raises(ValueError, match="REFERENCE.*POLICY_MISMATCH"):
        build_training_dataset(source, old)
    built = build_training_dataset(source, config)
    selection = dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"])
    with pytest.raises(ValueError, match="REFERENCE.*POLICY_MISMATCH"):
        prepare_events(built["dataset"], selection)
    changed = deepcopy(built["dataset"])
    changed["payload"]["reference_evidence_policy"] = "AUTO"
    changed["sha256"] = identity(changed["payload"])
    selection["reference_evidence_policy"] = FINAL_REFERENCE
    with pytest.raises(ValueError, match="REFERENCE.*POLICY"):
        prepare_events(changed, selection)


def test_native_selection_rejects_rehashed_matching_but_unknown_reference_policy():
    source, config = final_builder()
    built = build_training_dataset(source, config)
    changed = deepcopy(built["dataset"])
    changed["payload"]["reference_evidence_policy"] = "AUTO"
    changed["sha256"] = identity(changed["payload"])
    selection = dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"], reference_evidence_policy="AUTO")
    with pytest.raises(ValueError, match="REFERENCE_EVIDENCE_POLICY"):
        prepare_events(changed, selection)


@pytest.mark.parametrize("change", ["unknown", "coordinate_mismatch"])
def test_full_training_config_validates_reference_policy_not_just_equality(change):
    source, build = final_builder()
    config = synthetic_config()
    config.update(deepcopy(build["seed"]), reference_evidence_policy=FINAL_REFERENCE)
    if change == "unknown":
        config["reference_evidence_policy"] = "AUTO"
    else:
        config["reference_policy"]["cost_path_policy"]["coordinate_policy"] = "RAW_UNADJUSTED_NO_ACTIONS_COVERED"
    with pytest.raises(ValueError, match="REFERENCE|COORDINATE"):
        validate_training_config(config)


def test_action_inside_h20_window_is_excluded_not_adjusted_or_filled():
    source, config = final_builder(action_day="2026-01-15")
    built = build_training_dataset(source, config)
    assert not built["dataset"]["payload"]["events"]
    assert built["result"]["counts"]["eligible_events"] == 0
    assert built["result"]["reason_counts"]["CORPORATE_ACTION_OR_COORDINATE_COVERAGE_UNVERIFIED"] == 1


def test_unknown_calendar_day_stays_unknown_under_final_reference_policy():
    source, config = final_builder()
    source["payload"]["calendars"]["TWSE"]["days"]["2026-01-10"] = "UNKNOWN"
    reseal(source)
    built = build_training_dataset(source, config)
    assert built["result"]["counts"]["eligible_events"] == 0
    assert built["result"]["reason_counts"]["CALENDAR_WINDOW_CONTAINS_UNKNOWN_DATES"] == 1
    assert built["dataset"]["payload"]["calendars"]["TWSE"]["days"]["2026-01-10"] == "UNKNOWN"


def test_wide_final_coverage_preserves_receipt_but_labels_at_h20_window_close():
    plan, window, policy, kwargs = path_fixture()
    policy["coordinate_policy"] = FINAL_COORDINATE
    history = dict(symbol="3014", market="TWSE", coordinate_id=kwargs["coordinate_id"], bars=[])
    kwargs["corporate_action_coverage"] = build_final_coordinate_coverages([history], action_packet())[0]
    result = evaluate_reference_path(plan, window, policy, **kwargs)
    assert result["status"] == "VERIFIED_FILLED"
    assert result["execution"]["path_available_at"] == timestamp(window[-1]["available_at"]).isoformat()
    assert result["execution"]["source_observed_at"] == timestamp("2026-10-13T10:00:00+08:00").isoformat()
    assert result["execution"]["label_availability_basis"] == "RETROSPECTIVE_H20_WINDOW_CLOSE_NOT_ORIGINAL_PUBLICATION"
    assert kwargs["corporate_action_coverage"]["end_session"] > window[-1]["session"]


def test_old_builder_export_does_not_add_null_reconstruction_metadata():
    source, config = builder_inputs()
    first = build_training_dataset(source, config)
    again = build_training_dataset(source, config)
    assert first == again
    assert "reference_evidence_policy" not in first["dataset"]["payload"]
    assert "source_observed_at" not in first["dataset"]["payload"]["events"][0]["execution"]
    rows, _ = prepare_events(first["dataset"], dict(config["seed"], evaluation_cutoff=config["evaluation_cutoff"]))
    assert len(rows) == 1
