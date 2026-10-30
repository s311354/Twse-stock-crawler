"""Synthetic engineering fixtures only, never market/source certification."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext

import pytest

from domain.les_v7.contracts import ReferencePlan, identity, timestamp
from domain.les_v7.reference_path import (
    COORDINATE_POLICY, GRID_VERSION, PATH_SCHEMA, PATH_VERSION, PRECISION,
    estimate_reference_costs, evaluate_reference_path, path_window, validate_reference_path_policy,
    _net_2r_price, _round_price,
)
from domain.les_v7.research import label_event


def fixture():
    sessions = []
    day = date(2026, 1, 5)
    while len(sessions) < 21:
        # Explicit synthetic weekdays; not an official calendar claim.
        if day.weekday() < 5:
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    coordinate = "SYNTHETIC_TEST_ONLY:RAW_TWD:3014"
    window = [dict(session=s, open=100., high=100.5, low=99.5, close=100., volume=100000,
                   complete=True, available_at=s + "T13:30:00+08:00", coordinate_id=coordinate,
                   evidence_ids=["SYNTHETIC_TEST_ONLY:bar:" + s], corporate_action_state="NO_ACTION")
              for s in sessions]
    policy = dict(schema=PATH_SCHEMA, version=PATH_VERSION, market="TWSE", instrument_type="ORDINARY_STOCK",
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", exit_rule="FIRST_STOP_OR_NET_2R_ELSE_H20_CLOSE",
        exit_activation="NEXT_SESSION_AFTER_FILL", shares_rule="NATIVE_REFERENCE_SHARES",
        price_grid=GRID_VERSION, coordinate_policy=COORDINATE_POLICY, slippage_bps="0",
        available_at="2026-10-13T10:00:00+08:00", source_evidence_ids=["SYNTHETIC_POLICY_R1"],
        cost_policy=dict(version="SYNTHETIC_TWD_CENT_COST_R1", buy_rate="0.001425", sell_rate="0.004425",
                         minimum_buy_fee="0", minimum_sell_fee="0", currency_quantum="0.01",
                         rounding="ROUND_HALF_UP", available_at="2026-10-13T10:00:00+08:00",
                         effective_start="2026-01-01", effective_end="2026-12-31",
                         evidence_ids=["SYNTHETIC_DECLARED_COST_ASSUMPTION"]))
    coverage = dict(schema="les-v7-coordinate-coverage/v1", policy=COORDINATE_POLICY,
                    coordinate_id=coordinate, market="TWSE", start_session=sessions[0], end_session=sessions[-1],
                    status="VERIFIED_NO_ACTIONS", evidence_ids=["SYNTHETIC_NO_ACTION_COVERAGE"],
                    available_at=sessions[-1] + "T13:30:00+08:00")
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_NATIVE_R1", "SYNTHETIC_PAST_BARS", "REVERSAL",
                         100., 95.25, 105., 2., (99., 101.), 1000, .02, .02)
    kwargs = dict(expected_sessions=sessions, cutoff=sessions[0] + "T13:30:00+08:00",
                  coordinate_id=coordinate, source_evidence_ids=["SYNTHETIC_RAW_SOURCE"],
                  corporate_action_coverage=coverage)
    return plan, window, policy, kwargs


def run(items):
    plan, window, policy, kwargs = items
    return evaluate_reference_path(plan, window, policy, **kwargs)


def labels(items, result):
    plan, window, _, kwargs = items
    return label_event(window, cutoff=kwargs["cutoff"], expected_sessions=kwargs["expected_sessions"],
                       plan=plan, fill=result["execution"])


def test_exact_timeout_costs_source_binding_no_mutation_and_native_plan_unchanged():
    items = fixture()
    original = deepcopy(items)
    result = run(items)
    execution = result["execution"]
    assert result["status"] == "VERIFIED_FILLED"
    assert execution["exit_reason"] == "H20_CLOSE"
    assert execution["realized_net_pnl"] == -585.
    assert execution["cost_arithmetic"]["buy_fee"] == "142.50"
    assert execution["cost_arithmetic"]["sell_fee"] == "442.50"
    assert execution["cost_arithmetic"]["native_gross_reference_risk"] == "4750.00"
    assert execution["initial_stop"] == items[0].stop == 95.25
    assert execution["executable_stop"] == "95.2"
    assert execution["native_resistance_target"] == 105.
    assert execution["shares"] == 1000
    assert execution["cost_policy_id"] == identity(items[2]["cost_policy"])
    assert execution["path_policy_id"] == identity(items[2])
    assert result["diagnostic"]["raw_window_sha256"] == identity(items[1])
    assert execution["path_available_at"] == timestamp(items[1][-1]["available_at"]).isoformat()
    assert execution["historical_deployment_claim"] is False
    assert items == original
    label = labels(items, result)
    assert label["status"] == "TIMEOUT" and label["fill"] == 1 and label["rebound"] == 0
    assert label["outcome_r"] == pytest.approx(-585 / 4750)


def test_net2r_costs_use_native_gross_risk_and_smallest_legal_target():
    result = run(fixture())
    target = Decimal(result["execution"]["executable_net_2r"])
    assert target == Decimal("110.5")
    cost = result["execution"]["cost_arithmetic"]
    debit = Decimal(cost["buy_debit"])
    target_fee = (target * 1000 * Decimal(".004425")).quantize(Decimal(".01"))
    previous_fee = ((target - Decimal(".5")) * 1000 * Decimal(".004425")).quantize(Decimal(".01"))
    assert target * 1000 - target_fee - debit >= Decimal("9500")
    assert (target - Decimal(".5")) * 1000 - previous_fee - debit < Decimal("9500")


def _independent_sell_cash(price, shares, cost):
    notional = price * shares
    fee = max(notional * Decimal(cost['sell_rate']), Decimal(cost['minimum_sell_fee']))
    return notional - fee.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)


def test_real_2342_20260325_fee_step_regression_does_not_underpay_net2r():
    """Numbers from the failed 773afaba... job; no new source/vintage claim.

    2342 茂矽 cutoff 2026-03-25 had next open31.20 and native stop below.
    Continuous cash39.544585... withfee.17 meets the requirement, but upward
    price rounding39.55 changesfee to.18 and misses by.004585765157036.
    """
    cost = fixture()[2]['cost_policy']
    with localcontext() as ctx:
        ctx.prec = 50
        entry, stop = Decimal('31.2'), Decimal('27.132707117421482')
        target, debit, risk, stop_loss = _net_2r_price(entry, stop, Decimal('27.10'), 1, cost)
        required = debit + 2 * risk
        assert required == Decimal('39.374585765157036')
        assert _independent_sell_cash(Decimal('39.544585765157036'), 1, cost) == required
        assert _independent_sell_cash(Decimal('39.55'), 1, cost) < required
        assert target == Decimal('39.60')
        assert _independent_sell_cash(target, 1, cost) == Decimal('39.42') >= required
        assert debit == Decimal('31.24') and stop_loss == Decimal('4.26')


def test_real_triggered_plan_numbers_complete_path_and_native_labels():
    # Synthetic OHLC/receipts exercise the actual failing plan numbers; this
    # does not replay or replace the original official H20 source archive.
    items = list(fixture())
    items[0] = ReferencePlan('REFERENCE_H20', 'SYNTHETIC_NATIVE_R1', 'SYNTHETIC_REGRESSION_2342',
        'REVERSAL', 31.15, 27.132707117421482, 34.65, 1.7243096085950589,
        (31.15, 31.581077402148765), 1, .19, .192707117421482)
    for row in items[1]:
        row.update(open=31.2, high=31.25, low=31.15, close=31.2)
    items[1][2].update(high=39.6, close=39.6)
    result = run(items)
    execution = result['execution']
    assert result['status'] == 'VERIFIED_FILLED'
    assert execution['executable_net_2r'] == '39.60'
    assert execution['exit_reason'] == 'NET_2R' and execution['exit_price'] == 39.6
    assert execution['realized_net_pnl'] == 8.18
    label = labels(items, result)
    assert label['rebound'] == 1 and label['outcome_r'] >= 2


@pytest.mark.parametrize('entry,stop,shares,minimum', [
    ('9.99', '9.1', 1, '0'), ('49.95', '47.3', 1, '0'),
    ('99.9', '95.25', 1, '0'), ('499.5', '476.3', 1, '0'),
    ('999', '960.2', 1, '0'), ('31.2', '27.132707117421482', 1000, '20'),
    ('1', '.5', 1, '500'),
])
def test_net2r_legal_grid_minimality_across_bands_and_minimum_fees(entry, stop, shares, minimum):
    cost = deepcopy(fixture()[2]['cost_policy'])
    cost['minimum_sell_fee'] = minimum
    with localcontext() as ctx:
        ctx.prec = 50
        entry, stop = Decimal(entry), Decimal(stop)
        target, debit, risk, _ = _net_2r_price(entry, stop, _round_price(stop, 'down'), shares, cost)
        required = debit + 2 * risk
        # Independent cent enumeration using the published ordinary-stock
        # bands, checking all legal prices until the first actual cash pass.
        candidate = entry
        while True:
            tick = (Decimal('.01') if candidate < 10 else Decimal('.05') if candidate < 50
                    else Decimal('.1') if candidate < 100 else Decimal('.5') if candidate < 500
                    else Decimal('1') if candidate < 1000 else Decimal('5'))
            if candidate % tick == 0 and _independent_sell_cash(candidate, shares, cost) >= required:
                break
            candidate += Decimal('.01')
        assert target == candidate
        assert _independent_sell_cash(target, shares, cost) >= required


def test_zero_fee_exact_threshold_and_minimum_commission_are_explicit():
    items = fixture()
    items[2]["cost_policy"].update(buy_rate="0", sell_rate="0")
    result = run(items)
    assert result["execution"]["executable_net_2r"] == "109.5"
    items[2]["cost_policy"].update(minimum_buy_fee="20", minimum_sell_fee="20")
    result = run(items)
    assert result["execution"]["executable_net_2r"] == "110.0"
    assert result["execution"]["cost_arithmetic"]["buy_fee"] == "20.00"


@pytest.mark.parametrize("open_price", [98., 102.])
def test_open_outside_native_band_is_verified_no_fill_never_later_intraday_fill(open_price):
    items = fixture()
    items[1][1].update(open=open_price, high=103., low=97.)
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["filled"] is False
    assert result["execution"]["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
    label = labels(items, result)
    assert label["fill"] == 0 and label["rebound"] is None and label["joint"] is None


@pytest.mark.parametrize("side", ["low", "high"])
def test_native_band_edges_are_inclusive(side):
    items = fixture()
    entry = items[0].band[0 if side == "low" else 1]
    items[1][1].update(open=entry, low=min(entry, 99.5), high=max(entry, 100.5))
    assert run(items)["execution"]["entry"] == entry


def test_range_target_hit_has_explicit_session_bound_not_fake_intraday_time():
    items = fixture()
    items[1][2].update(high=111., low=99.)
    result = run(items)
    execution = result["execution"]
    assert execution["exit_reason"] == "NET_2R"
    marker = execution["hit_markers"]["NET_2R"]
    assert marker["time_precision"] == PRECISION
    assert marker["at"] == timestamp(items[1][2]["available_at"]).isoformat()
    assert marker["interval_start"] != marker["interval_end"]
    assert execution["realized_net_pnl"] >= 9500
    label = labels(items, result)
    assert label["status"] == "MATURED" and label["rebound"] == 1


def test_range_stop_only_is_verified_and_same_bar_opposing_hits_are_ambiguous():
    items = fixture()
    items[1][2].update(low=94., high=101.)
    stopped = run(items)
    assert stopped["execution"]["exit_reason"] == "STOP"
    assert stopped["execution"]["exit_price"] == 95.2
    assert stopped["execution"]["hit_markers"]["STOP"]["time_precision"] == PRECISION
    items[1][2].update(high=111.)
    ambiguous = run(items)
    assert ambiguous["status"] == "PATH_AMBIGUOUS"
    assert ambiguous["execution"]["filled"] is True
    assert ambiguous["execution"]["path_order_verified"] is False
    assert ambiguous["execution"]["realized_net_pnl"] is None
    assert labels(items, ambiguous)["status"] == "PATH_AMBIGUOUS"


@pytest.mark.parametrize("opening,reason", [(94., "STOP"), (111., "NET_2R")])
def test_observed_open_gap_proves_order_even_when_range_crosses_both(opening, reason):
    items = fixture()
    items[1][2].update(open=opening, high=112., low=93.)
    result = run(items)
    execution = result["execution"]
    assert result["status"] == "VERIFIED_FILLED"
    assert execution["exit_reason"] == reason
    assert execution["exit_price"] == opening
    assert execution["hit_markers"][reason]["time_precision"] == "OBSERVED_SESSION_OPEN"
    assert labels(items, result)["rebound"] == int(reason == "NET_2R")


def test_no_day_trade_policy_does_not_exit_on_entry_day_even_after_both_crossed():
    items = fixture()
    items[1][1].update(high=112., low=94.)
    result = run(items)
    assert result["execution"]["exit_reason"] == "H20_CLOSE"
    assert result["diagnostic"]["ignored_entry_session_range"] is True
    assert result["execution"]["exit_activation"] == "NEXT_SESSION_AFTER_FILL"


@pytest.mark.parametrize("problem", ["missing", "incomplete", "conflicting", "illegal", "zero_volume", "future_coordinate", "action"])
def test_missing_or_inconsistent_market_data_never_becomes_verified_no_fill(problem):
    items = fixture()
    if problem == "missing":
        items[1].pop(2)
    elif problem == "incomplete":
        items[1][2]["complete"] = False
    elif problem == "conflicting":
        items[1][2]["high"] = 99.
    elif problem == "illegal":
        items[1][2]["open"] = 100.01
    elif problem == "zero_volume":
        items[1][2]["volume"] = 0
    elif problem == "future_coordinate":
        items[1][2]["coordinate_id"] = "UNVERIFIED_SPLIT"
    else:
        items[1][2]["corporate_action_state"] = "SPLIT"
    result = run(items)
    assert result["status"] == "DATA_GAP" and result["execution"] is None


@pytest.mark.parametrize("change", ["absent", "status", "period", "market", "source", "early"])
def test_missing_or_incompatible_action_coverage_stays_gap(change):
    items = fixture()
    coverage = items[3]["corporate_action_coverage"]
    if change == "absent":
        items[3]["corporate_action_coverage"] = None
    elif change == "status":
        coverage["status"] = "UNKNOWN"
    elif change == "period":
        coverage["start_session"] = items[1][1]["session"]
    elif change == "market":
        coverage["market"] = "TPEX"
    elif change == "source":
        coverage["evidence_ids"] = []
    else:
        coverage["available_at"] = items[1][0]["available_at"]
    assert run(items)["status"] == "DATA_GAP"


@pytest.mark.parametrize("change", ["cost_missing", "slippage", "activation", "rounding", "date_range", "rate"])
def test_unbound_or_unsupported_research_policy_rejects_without_default(change):
    items = fixture()
    policy = items[2]
    if change == "cost_missing":
        policy["cost_policy"]["evidence_ids"] = []
    elif change == "slippage":
        policy["slippage_bps"] = "5"
    elif change == "activation":
        policy["exit_activation"] = "SAME_SESSION"
    elif change == "rounding":
        policy["cost_policy"]["rounding"] = "ROUND_DOWN"
    elif change == "date_range":
        policy["cost_policy"]["effective_start"] = "2026-02-01"
    else:
        policy["cost_policy"]["buy_rate"] = "NaN"
    with pytest.raises(ValueError):
        run(items)


def test_late_actual_coverage_delays_label_maturity_but_not_policy_creation_time():
    items = fixture()
    items[3]["corporate_action_coverage"]["available_at"] = "2026-02-20T10:00:00+08:00"
    result = run(items)
    assert result["execution"]["path_available_at"] == timestamp("2026-02-20T10:00:00+08:00").isoformat()
    assert labels(items, result)["label_available_at"] == result["execution"]["path_available_at"]


def test_cutoff_and_calendar_must_be_explicit_canonical_and_close_bound():
    items = fixture()
    items[3]["cutoff"] = items[1][0]["session"] + "T09:00:00+08:00"
    with pytest.raises(ValueError, match="CLOSE_CUTOFF"):
        run(items)
    items = fixture()
    items[3]["expected_sessions"] = items[3]["expected_sessions"][:-1]
    with pytest.raises(ValueError, match="H20_SESSION"):
        run(items)


def test_policy_and_every_source_or_bar_change_have_distinct_binding():
    items = fixture()
    first = run(items)
    items[2]["cost_policy"]["sell_rate"] = "0.003425"
    second = run(items)
    assert first["execution"]["cost_policy_id"] != second["execution"]["cost_policy_id"]
    assert first["execution"]["path_policy_id"] != second["execution"]["path_policy_id"]
    items[1][-1]["evidence_ids"].append("SYNTHETIC_REVISION")
    third = run(items)
    assert second["diagnostic"]["raw_window_sha256"] != third["diagnostic"]["raw_window_sha256"]
    assert "SYNTHETIC_REVISION" in third["execution"]["path_evidence_ids"]


def test_policy_validation_returns_detached_copy():
    policy = fixture()[2]
    copied = validate_reference_path_policy(policy)
    copied["cost_policy"]["evidence_ids"].append("SYNTHETIC_MUTATION")
    assert copied != policy


def test_source_bound_cutoff_cost_estimates_include_adverse_stop_rounding_only():
    items = fixture()
    before = deepcopy(items[2])
    result = estimate_reference_costs(100., 95.25, 105., 1000, items[2])
    assert result["cost_up_per_share"] == .60713
    assert result["cost_stop_per_share"] == .61376
    assert result["detail"]["stop_rounding_loss"] == "50.00"
    assert result["detail"]["buy_fee"] == "142.50"
    assert result["detail"]["stop_sell_fee"] == "421.26"
    assert result["detail"]["target_sell_fee"] == "464.63"
    assert result["cost_policy_id"] == identity(items[2]["cost_policy"])
    assert items[2] == before


def test_missing_target_does_not_manufacture_resistance_or_use_future_outcome():
    result = estimate_reference_costs(100., 95.25, None, 1000, fixture()[2])
    assert result["detail"]["native_target"] is None
    assert result["detail"]["target_present"] is False
    assert result["detail"]["executable_target"] == "100.0"
    assert result["cost_up_per_share"] == .585


@pytest.mark.parametrize("entry,stop,target,shares", [
    (100.01, 95., 105., 1000), (100., 100., 105., 1000),
    (100., 95., 100., 1000), (100., 95., 105., 0), (100., 95., 105., True),
])
def test_invalid_cutoff_cost_inputs_reject(entry, stop, target, shares):
    with pytest.raises(ValueError):
        estimate_reference_costs(entry, stop, target, shares, fixture()[2])


@pytest.mark.parametrize("key", ["open", "high", "low", "close"])
def test_every_observed_ohlc_price_is_checked_not_rounded_into_evidence(key):
    items = fixture()
    items[1][2][key] += .01
    assert run(items)["status"] == "DATA_GAP"


def test_missing_bar_cannot_be_hidden_by_known_open_outside_band():
    items = fixture()
    items[1][1].update(open=102., high=103., low=99.)
    items[1].pop(3)
    assert run(items)["status"] == "DATA_GAP"


def test_broader_verified_no_action_coverage_can_be_shared_without_narrowing_source():
    items = fixture()
    coverage = items[3]["corporate_action_coverage"]
    coverage.update(start_session="2025-12-01", end_session="2026-02-20",
                    available_at="2026-02-20T13:30:00+08:00")
    before = deepcopy(coverage)
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED"
    assert result["execution"]["coordinate_coverage_sha256"] == identity(coverage)
    assert coverage == before


@pytest.mark.parametrize("exit_day", [None, 2, -1])
def test_daily_total_volume_is_necessary_not_sufficient_for_reference_size(exit_day):
    items = fixture()
    if exit_day is None:
        items[1][1]["volume"] = 999
        reason = "DAILY_VOLUME_BELOW_REFERENCE_ENTRY_SIZE"
    else:
        items[1][exit_day]["volume"] = 999
        if exit_day == 2:
            items[1][exit_day]["high"] = 111.
        reason = "DAILY_VOLUME_BELOW_REFERENCE_EXIT_SIZE"
    result = run(items)
    assert result["status"] == "DATA_GAP"
    assert result["execution"] is None and result["diagnostic"]["reason_codes"] == [reason]
    valid = run(fixture())["execution"]
    assert valid["liquidity_assumption"] == "DAILY_OPEN_REFERENCE_NO_DEPTH_PROOF"
    assert valid["capacity_guarantee"] is False and valid["partial_fills_modeled"] is False


def source_packet(items):
    return dict(coordinate_id=items[3]["coordinate_id"],
                bars=[dict(Date=row["session"], Open=row["open"], High=row["high"], Low=row["low"],
                           Close=row["close"], Volume=row["volume"], Complete=row["complete"],
                           **{"Available At": row["available_at"]}) for row in items[1]],
                bar_lineage=[dict(date=row["session"], source_row_ids=row["evidence_ids"])
                             for row in items[1]])


def test_path_window_source_projection_detaches_ids_and_preserves_missing_sessions():
    items = fixture()
    packet = source_packet(items)
    before = deepcopy(packet)
    projected = path_window(packet, items[3]["expected_sessions"])
    assert projected[0]["open"] == 100.
    assert projected[0]["evidence_ids"] == packet["bar_lineage"][0]["source_row_ids"]
    assert run((items[0], projected, items[2], items[3]))["status"] == "VERIFIED_FILLED"
    projected[0]["evidence_ids"].append("SYNTHETIC_MUTATION")
    assert packet == before
    packet["bars"].pop(2)
    projected = path_window(packet, items[3]["expected_sessions"])
    assert len(projected) == 21 and projected[2]["complete"] is False
    assert run((items[0], projected, items[2], items[3]))["status"] == "DATA_GAP"


@pytest.mark.parametrize("field", ["bars", "bar_lineage"])
def test_path_window_rejects_duplicates_even_after_selected_window(field):
    items = fixture()
    packet = source_packet(items)
    later = deepcopy(packet[field][-1])
    later["Date" if field == "bars" else "date"] = "2026-02-10"
    packet[field].extend([later, deepcopy(later)])
    with pytest.raises(ValueError, match="DUPLICATE_PATH"):
        path_window(packet, items[3]["expected_sessions"])


def test_path_window_missing_lineage_never_synthesizes_source_id():
    items = fixture()
    packet = source_packet(items)
    packet["bar_lineage"].pop(2)
    projected = path_window(packet, items[3]["expected_sessions"])
    assert projected[2]["evidence_ids"] == []
    assert run((items[0], projected, items[2], items[3]))["status"] == "DATA_GAP"
