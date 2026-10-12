"""Synthetic outcome contracts only; not real fills, calibration or performance."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from domain.strategies.low_entry_v6_outcomes import outcome_audit, wilson_interval
from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_v6_report import report_failure
from interface.les_v6_research import main
from tests.strategies.test_low_entry_score_v6 import execution_report


VERSIONS = dict(model="6.0.0", parameters="synthetic-parameters", code="synthetic-code",
                execution_policy="synthetic-one-entry-exit-v1")
TAIPEI = timezone(timedelta(hours=8))


def observation(index, *, validation=False, win=True, **changes):
    day = datetime(2026, 9, 1, 13, 30, tzinfo=TAIPEI) if validation else datetime(2026, 3, 1, 13, 30, tzinfo=TAIPEI)
    signal = day + timedelta(days=index)
    entered = (signal + timedelta(days=1)).replace(hour=9, minute=5)
    closed = signal + timedelta(days=3)
    row = dict(episode_id=f"{'v' if validation else 't'}-{index}", symbol=str(4000 + index),
               market="TWSE", sector="ELEC", market_regime="NEUTRAL", setup="REVERSAL",
               entry_decision="BUY_SMALL", versions=dict(VERSIONS),
               formal_score=80., score_coverage=100., signal_at=signal.isoformat(),
               entry_at=entered.isoformat(), exit_at=closed.isoformat(), available_at=closed.isoformat(),
               holding_sessions=3, quantity=100, sold_quantity=100, fully_closed=True,
               cash_unit="TWD", price_unit="TWD_PER_SHARE", quantity_unit="SHARES",
               costs_complete=True, entry_notional=10000., entry_cost=15.,
               exit_notional=12000. if win else 9000., exit_cost=45., net_cash_dividends=0.,
               post_fill_min_price=99. if win else 90., excursion_basis="AFTER_FILL_UNTIL_FULL_EXIT",
               source_id="synthetic-ledger", source_sha256="a" * 64, evidence_id="synthetic-evidence",
               research_result_sha256="b" * 64, account_id="synthetic-account")
    row.update(changes)
    return row


def packet(train=40, validation=25):
    return dict(schema="les-v6-closed-outcomes/v1", source_kind="SYNTHETIC_TEST_ONLY",
                inclusion_policy="ALL_CLOSED_EPISODES_NO_WINNER_FILTER", versions=dict(VERSIONS),
                as_of="2026-10-01T13:30:00+08:00", validation_start="2026-09-01T13:30:00+08:00",
                definition=dict(success="POSITIVE_NET_CASH_RETURN", max_holding_sessions=20,
                                max_adverse_fraction=.04, position_policy="ONE_ENTRY_FULL_EXIT_NO_ADDS"),
                outcomes=[observation(i, win=i % 5 < 3) for i in range(train)] +
                         [observation(i, validation=True, win=i % 2 == 0) for i in range(validation)])


def test_cash_math_sample_counts_and_explicit_research_boundaries():
    source = packet()
    before = deepcopy(source)
    result = outcome_audit(source)
    assert source == before
    assert result == outcome_audit(source)
    json.dumps(result, allow_nan=False)
    cohort = result["cohorts"][0]
    assert cohort["training"]["sample_count"] == 40
    assert cohort["training"]["wins"] == 24
    assert cohort["estimated_win_rate"] == .6
    assert cohort["estimated_low_capture_rate"] == .6
    assert cohort["holdout_status"] == "DIAGNOSTICS_AVAILABLE_NOT_VALIDATED"
    expected_brier = (13 * .4 ** 2 + 12 * .6 ** 2) / 25
    assert cohort["holdout_brier"] == pytest.approx(expected_brier)
    assert cohort["holdout_baseline_brier"] == pytest.approx(expected_brier)
    assert result["validation_prediction_coverage"] == 1
    first = result["rows"][0]
    assert first["net_return"] == pytest.approx((12000 - 45) / (10000 + 15) - 1)
    assert first["mae_fraction"] == pytest.approx(-.01)
    assert result["source_posture"] == "DECLARED_OUTCOMES_NOT_INDEPENDENTLY_VERIFIED"
    assert result["executable"] is result["kelly_calibration_eligible"] is False
    assert result["bottom_guaranteed"] is result["performance_validated"] is False


def test_probability_is_not_les_score_and_estimate_is_separate_from_holdout_labels():
    source = packet()
    first = outcome_audit(source)
    for row in source["outcomes"][40:]:
        row["exit_notional"] = 1000000.
    second = outcome_audit(source)
    assert first["cohorts"][0]["estimated_win_rate"] == second["cohorts"][0]["estimated_win_rate"] == .6
    assert second["cohorts"][0]["holdout_brier"] != first["cohorts"][0]["holdout_brier"]
    assert first["cohorts"][0]["estimated_win_rate"] != .8


def test_high_win_rate_can_have_negative_net_returns_and_never_grants_entry():
    source = packet()
    for i, row in enumerate(source["outcomes"][:40]):
        row["exit_notional"] = 10180. if i < 38 else 100.
    cohort = outcome_audit(source)["cohorts"][0]
    assert cohort["estimated_win_rate"] == .95
    assert cohort["training"]["mean_net_return"] < 0
    assert outcome_audit(source)["executable"] is False


def test_future_closed_outcomes_do_not_change_training_or_earlier_predictions():
    source = packet()
    source["as_of"] = "2026-09-10T13:30+08:00"
    earlier = outcome_audit(source)
    source["as_of"] = "2026-10-01T13:30+08:00"
    later = outcome_audit(source)
    assert earlier["cohorts"][0]["training"] == later["cohorts"][0]["training"]
    assert earlier["cohorts"][0]["estimated_win_rate"] == later["cohorts"][0]["estimated_win_rate"]
    assert earlier["predictions"] == later["predictions"][:len(earlier["predictions"])]
    assert earlier["cohorts"][0]["holdout_status"] == "NOT_ASSESSABLE"
    assert earlier["exclusion_counts"]["OUTCOME_UNAVAILABLE_AT_CUTOFF"] == 18


def test_appending_future_mixed_version_does_not_contaminate_selected_identity():
    source = packet()
    first = outcome_audit(source)
    source["outcomes"].append(observation(99, validation=True, versions=dict(VERSIONS, code="future-code")))
    second = outcome_audit(source)
    assert first["input_sha256"] != second["input_sha256"]
    assert first["selected_outcomes_sha256"] == second["selected_outcomes_sha256"]
    assert first["study_id"] == second["study_id"]
    assert first["cohorts"] == second["cohorts"]
    assert first["predictions"] == second["predictions"]


def test_labels_crossing_split_are_purged_not_trained_on_later_results():
    source = packet()
    source["outcomes"].append(observation(999, signal_at="2026-08-31T13:30+08:00",
                                         entry_at="2026-09-01T09:05+08:00",
                                         exit_at="2026-09-02T13:30+08:00",
                                         available_at="2026-09-02T14:00+08:00"))
    result = outcome_audit(source)
    assert result["exclusion_counts"] == {"PURGED_LABEL_CROSSES_VALIDATION_BOUNDARY": 1}
    assert result["cohorts"][0]["training"]["sample_count"] == 40


@pytest.mark.parametrize("train,validation", [(0, 0), (1, 1), (29, 20)])
def test_insufficient_training_keeps_forecast_unknown(train, validation):
    result = outcome_audit(packet(train, validation))
    assert result["forecast_status"] == "UNKNOWN"
    assert all(c["estimated_win_rate"] is None for c in result["cohorts"])
    assert all(p["brier_loss"] is None for p in result["predictions"])


def test_single_class_training_is_not_a_100_percent_safety_claim():
    source = packet()
    for row in source["outcomes"]:
        row["exit_notional"] = 12000
    cohort = outcome_audit(source)["cohorts"][0]
    assert cohort["training"]["observed_win_rate"] == 1
    assert cohort["estimated_win_rate"] is None
    assert "SINGLE_CLASS_TRAINING_OUTCOMES" in cohort["reasons"]


def test_missing_or_wrong_post_fill_excursions_do_not_manufacture_low_capture():
    source = packet()
    for row in source["outcomes"]:
        row["excursion_basis"] = "SIGNAL_DAY_OHLC_INCLUDES_PRE_FILL_TIME"
    cohort = outcome_audit(source)["cohorts"][0]
    assert cohort["estimated_win_rate"] == .6
    assert cohort["low_capture_status"] == "UNKNOWN"
    assert cohort["estimated_low_capture_rate"] is None
    assert cohort["training"]["low_capture_sample_count"] == 0
    assert cohort["training"]["low_capture_missing_count"] == 40


def test_one_missing_mae_does_not_promote_a_biased_subset_to_a_low_capture_estimate():
    source = packet()
    source["outcomes"][3].pop("post_fill_min_price")
    cohort = outcome_audit(source)["cohorts"][0]
    assert cohort["training"]["low_capture_sample_count"] == 39
    assert cohort["estimated_low_capture_rate"] is None
    assert "INCOMPLETE_POST_FILL_COVERAGE" in cohort["low_capture_reasons"]
    assert cohort["estimated_win_rate"] == .6


@pytest.mark.parametrize("change,reason", [
    ({"score_coverage": 94}, "INCOMPLETE_OR_INVALID_FORMAL_SCORE"),
    ({"formal_score": 101}, "INCOMPLETE_OR_INVALID_FORMAL_SCORE"),
    ({"entry_decision": "WATCH"}, "NOT_A_FORMAL_ENTRY_EPISODE"),
    ({"fully_closed": False}, "INCOMPLETE_EPISODE_OR_COSTS"),
    ({"costs_complete": False}, "INCOMPLETE_EPISODE_OR_COSTS"),
    ({"entry_cost": None}, "INVALID_OR_MISSING_CASH_FLOWS"),
    ({"net_cash_dividends": None}, "INVALID_OR_MISSING_CASH_FLOWS"),
    ({"entry_cost": -1}, "INVALID_OR_MISSING_CASH_FLOWS"),
    ({"holding_sessions": 21}, "HOLDING_POLICY_MISMATCH"),
    ({"market_regime": None}, "UNSUPPORTED_OUTCOME_COHORT"),
    ({"sector": None}, "UNSUPPORTED_OUTCOME_COHORT"),
    ({"cash_unit": "USD"}, "INCOMPATIBLE_OR_MISSING_OUTCOME_UNITS"),
    ({"cash_unit": "TWD_THOUSAND"}, "INCOMPATIBLE_OR_MISSING_OUTCOME_UNITS"),
    ({"quantity_unit": "LOTS"}, "INCOMPATIBLE_OR_MISSING_OUTCOME_UNITS"),
    ({"sold_quantity": True, "quantity": 1}, "INCOMPLETE_POSITION_QUANTITIES"),
    ({"source_id": ""}, "MISSING_OUTCOME_PROVENANCE"),
    ({"research_result_sha256": "unknown"}, "INVALID_OUTCOME_PROVENANCE_HASH"),
    ({"available_at": "2026-03-04"}, "INVALID_OR_MISSING_OUTCOME_TIMESTAMPS"),
    ({"available_at": "2026-03-01T13:30+08:00"}, "INVALID_OUTCOME_TIME_ORDER"),
])
def test_invalid_or_incomplete_observations_are_diagnosed(change, reason):
    source = packet(0, 0)
    source["outcomes"] = [observation(0, **change)]
    result = outcome_audit(source)
    assert result["rows"] == []
    assert result["forecast_status"] == "UNKNOWN"
    assert result["exclusion_counts"] == {reason: 1}


def test_exact_net_break_even_is_not_a_win():
    source = packet(0, 0)
    source["outcomes"] = [observation(0, exit_notional=10060., post_fill_min_price=100.)]
    row = outcome_audit(source)["rows"][0]
    assert row["net_return"] == 0 and row["success"] is False


def test_setups_entry_sizes_and_score_bands_are_not_pooled_into_one_scale():
    source = packet()
    source["outcomes"][0]["setup"] = "LEADERSHIP_CONTINUATION"
    source["outcomes"][1]["entry_decision"] = "BUY"
    source["outcomes"][2]["formal_score"] = 100.
    result = outcome_audit(source)
    assert len(result["cohorts"]) == 4
    small = [c for c in result["cohorts"] if c["training"]["sample_count"] == 1]
    assert len(small) == 3 and all(c["estimated_win_rate"] is None for c in small)
    assert next(c for c in small if c["score_band"] == [90, 100])["upper_inclusive"] is True


def test_sectors_and_market_regimes_are_not_pooled_into_one_forecast():
    source = packet()
    source["outcomes"][0]["sector"] = "SEMI"
    source["outcomes"][1]["market_regime"] = "RISK_OFF"
    result = outcome_audit(source)
    assert len(result["cohorts"]) == 3
    assert all(c["estimated_win_rate"] is None for c in result["cohorts"]
               if c["sector"] == "SEMI" or c["market_regime"] == "RISK_OFF")


def test_arithmetic_overflow_and_unsafe_quantities_do_not_create_a_valid_outcome():
    source = packet(0, 0)
    source["outcomes"] = [observation(0, entry_notional=1e308, entry_cost=1e308)]
    assert outcome_audit(source)["exclusion_counts"] == {"INVALID_NET_PROCEEDS": 1}
    source["outcomes"] = [observation(0, quantity=2 ** 100, sold_quantity=2 ** 100)]
    assert outcome_audit(source)["exclusion_counts"] == {"INCOMPLETE_POSITION_QUANTITIES": 1}
    source["outcomes"] = [observation(0, entry_notional=10 ** 400)]
    assert outcome_audit(source)["exclusion_counts"] == {"INVALID_OR_MISSING_CASH_FLOWS": 1}
    source["outcomes"] = [observation(0, entry_notional=1e-320, quantity=1000000, sold_quantity=1000000)]
    assert outcome_audit(source)["exclusion_counts"] == {"INVALID_NOTIONAL_QUANTITY_PRICE": 1}


def test_effective_identity_pins_policy_and_original_selected_outcomes_not_only_rates():
    source = packet()
    first = outcome_audit(source)
    source["definition"]["max_adverse_fraction"] = .03
    changed_policy = outcome_audit(source)
    assert first["study_id"] != changed_policy["study_id"]
    assert first["selected_outcomes_sha256"] == changed_policy["selected_outcomes_sha256"]
    source["outcomes"][0]["source_id"] = "other-declared-source"
    assert outcome_audit(source)["selected_outcomes_sha256"] != first["selected_outcomes_sha256"]


def test_returned_metadata_does_not_expose_mutable_input_references():
    source = packet()
    source["definition"]["notes"] = {"purpose": "synthetic isolation test"}
    result = outcome_audit(source)
    result["definition"]["notes"]["purpose"] = "changed"
    result["rows"][0]["net_return"] = 999
    assert source["definition"]["notes"]["purpose"] == "synthetic isolation test"
    assert "net_return" not in source["outcomes"][0]


def test_duplicate_cross_account_and_overlapping_episodes_are_rejected():
    source = packet()
    source["outcomes"].append(deepcopy(source["outcomes"][0]))
    with pytest.raises(ValueError, match="DUPLICATE"):
        outcome_audit(source)
    source["outcomes"][-1].update(episode_id="other-account", account_id="other")
    with pytest.raises(ValueError, match="DUPLICATE"):
        outcome_audit(source)
    source = packet(0, 0)
    source["outcomes"] = [observation(0), observation(1, symbol="4000")]
    with pytest.raises(ValueError, match="OVERLAPPING"):
        outcome_audit(source)


@pytest.mark.parametrize("key", ["model", "parameters", "code", "execution_policy"])
def test_eligible_mixed_versions_are_rejected(key):
    source = packet()
    source["outcomes"][0]["versions"][key] = "different"
    with pytest.raises(ValueError, match="MIXED_OUTCOME_VERSIONS"):
        outcome_audit(source)


@pytest.mark.parametrize("change", [
    {"schema": "wrong"}, {"source_kind": None}, {"inclusion_policy": "WINNERS_ONLY"},
    {"validation_start": "2027-01-01T13:30+08:00"}, {"as_of": "2026-10-01"},
    {"definition": {}}, {"definition": None}, {"outcomes": None}, {"versions": {}},
])
def test_packet_requires_explicit_versions_cost_policy_cutoff_and_split(change):
    source = packet()
    source.update(change)
    with pytest.raises((ValueError, KeyError)):
        outcome_audit(source)


def test_wilson_bounds_and_missing_samples():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(5, 10) == pytest.approx([.236593090512564, .763406909487436])
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    for wins, count in [(1, 0), (-1, 5), (True, 5)]:
        with pytest.raises(ValueError):
            wilson_interval(wins, count)


def test_research_cannot_change_native_gates_or_payoff_probability():
    native = execution_report()
    before = deepcopy(native)
    assert report_failure(native) is None
    outcome_audit(packet())
    assert native == before
    assert report_failure(native) is None
    payoff = payoff_scenarios(native["decision"]["plan"], buy_cost_rate=.001425, sell_cost_rate=.004425)
    assert payoff["estimated_win_rate"] is None
    assert payoff["forecast_status"] == "NOT_CALIBRATED"


def test_cli_is_opt_in_deterministic_and_rejects_ambiguous_modes(tmp_path, capsys):
    path = tmp_path / "synthetic-outcomes.json"
    path.write_text(json.dumps(packet()))
    assert main([str(path), "--outcome-audit"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == outcome_audit(packet())
    for flags in [["--outcome-audit", "--history"], ["--outcome-audit", "--validation-start", "2026-09-01"],
                  ["--outcome-audit", "--score-audit"]]:
        with pytest.raises(SystemExit):
            main([str(path)] + flags)
