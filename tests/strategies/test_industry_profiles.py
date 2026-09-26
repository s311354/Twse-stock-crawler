"""Explicitly enabled production profiles preserve native model and risk gates."""
from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import json

import pandas as pd
import pytest

from domain.fundamentals import GateStatus, _score_quality_metrics, evaluate_quality_gate
from domain.industry_admission import industry_entry_signal
from domain.industry_profiles import ELEC_QUALITY_PROFILE_VERSION, ELECCOMP_QUALITY_PROFILE_VERSION, IndustryProfileMode, IndustryProfileRegistry, MetricRule
from domain.models import Stocktype
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Config, LowEntryScoreV41Strategy
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Strategy
from domain.strategies.low_entry_v6_report import report_failure
from domain.strategies.low_entry_v6_signals import LowEntryV6Config
from infrastructure.report.html_renderer import RendererFactory
from tests.strategies.test_daily_v6_pipeline import build, daily_market
from tests.strategies.test_low_entry_score_v4 import (
    annual_fundamentals, fundamental_snapshot, buy_candidate_history,
)
from tests.strategies.test_low_entry_score_v6 import execution_report
from tests.strategies.test_low_entry_v5_elec_pullback import _history


SECTORS = {"VEH": "12", "ELEC": "13", "BUI": "14", "AIR": "15", "CHEMI": "21",
           "BIO": "22", "SEMI": "24", "COMP": "25", "COMM": "27", "ELECCOMP": "28"}


@pytest.mark.parametrize("name,code", SECTORS.items())
def test_each_stocktype_resolves_to_immutable_versioned_active_profile(name, code):
    profile = IndustryProfileRegistry.resolve(code)
    assert profile is IndustryProfileRegistry.resolve(int(code))
    assert profile is IndustryProfileRegistry.resolve(name)
    assert profile is IndustryProfileRegistry.resolve(Stocktype[name])
    assert profile is IndustryProfileRegistry.resolve(f"Stocktype.{name}")
    assert profile is IndustryProfileRegistry.get(f"{name}_PROFILE")
    assert profile.name == f"{name}_PROFILE" and code in profile.sector_codes
    expected_version = {"ELEC": ELEC_QUALITY_PROFILE_VERSION, "ELECCOMP": ELECCOMP_QUALITY_PROFILE_VERSION}
    assert profile.version == expected_version.get(name, f"{name.lower()}-quality-2026-09-25-active-v2")
    assert profile.mode is IndustryProfileMode.ACTIVE
    assert profile.calibration_status == "UNCALIBRATED" and profile.baseline_version
    assert profile.allow_buy and profile.allow_buy_small
    assert sum(rule.maximum for rule in profile.metric_rules) == 100
    with pytest.raises(FrozenInstanceError):
        profile.pass_score = 0


def test_codes_are_unique_and_specific_company_sector_beats_aggregate_electronics():
    profiles = IndustryProfileRegistry.profiles()
    assert len(profiles) == len(SECTORS) == len({p.version for p in profiles})
    codes = [code for profile in profiles for code in profile.sector_codes]
    assert len(codes) == len(set(codes))
    for code in ("26", "29", "30", "31"):
        assert IndustryProfileRegistry.resolve(code).name == "ELEC_PROFILE"
    for unknown in (None, "", "UNKNOWN", "17", "FINANCIAL", "999"):
        assert IndustryProfileRegistry.resolve(unknown) is None


def test_quality_rule_changes_do_not_mutate_other_sectors_or_entry_risk():
    original = IndustryProfileRegistry.resolve("24")
    unrelated = IndustryProfileRegistry.resolve("25")
    before_v41, before_v6 = LowEntryScoreV41Config(), LowEntryV6Config()
    test_profile = replace(original, metric_rules=(MetricRule("roic_ttm", "A", 100, lambda x: 100 if x >= .2 else 0),))
    assert _score_quality_metrics({"roic_ttm": .19}, test_profile)[0].earned == 0
    assert _score_quality_metrics({"roic_ttm": .2}, test_profile)[0].earned == 100
    assert not _score_quality_metrics({}, test_profile)[0].available
    assert IndustryProfileRegistry.resolve("24") is original
    assert IndustryProfileRegistry.resolve("25") is unrelated
    assert LowEntryScoreV41Config() == before_v41
    assert LowEntryV6Config() == before_v6


@pytest.mark.parametrize("name,code", SECTORS.items())
def test_active_quality_metrics_keep_scores_and_hard_rejects(name, code):
    inputs = annual_fundamentals(), fundamental_snapshot(sector=code)
    result = evaluate_quality_gate(*inputs, "2026-07-21T05:30:00Z")
    assert result.industry_profile == f"{name}_PROFILE"
    assert result.industry_profile_mode == "ACTIVE"
    assert result.score is not None and 0 <= result.score <= 100
    assert "INDUSTRY_PROFILE_SHADOW_MODE" not in result.qa_flags
    rejected = evaluate_quality_gate(inputs[0], dict(inputs[1], going_concern=True), "2026-07-21T05:30:00Z")
    assert rejected.status is GateStatus.REJECT
    assert "GOING_CONCERN_MATERIAL_UNCERTAINTY" in rejected.hard_red_flags
    future = evaluate_quality_gate(inputs[0], dict(inputs[1], available_at="2027-01-01T00:00:00Z"), "2026-07-21T05:30:00Z")
    assert future.status is GateStatus.UNKNOWN


@pytest.mark.parametrize("signal", ["BUY", "EARLY_BUY", "BUY_SMALL", "ARMED", "WATCH"])
def test_active_keeps_native_signals_and_explicit_shadow_still_caps_entry(signal):
    for profile in IndustryProfileRegistry.profiles():
        assert industry_entry_signal(signal, profile) == signal
        assert industry_entry_signal(signal, replace(profile, mode=IndustryProfileMode.SHADOW)) == "WATCH_ONLY"
        for hard in ("NO_BUY", "NO_MODEL", "REJECT", "DATA_INSUFFICIENT", "PANIC_WATCH", "AVOID"):
            assert industry_entry_signal(hard, profile) == hard
    assert industry_entry_signal(signal, None) == "NO_MODEL"


@pytest.mark.parametrize("name", SECTORS)
def test_explicit_signal_permissions_still_apply_in_active_mode(name):
    profile = IndustryProfileRegistry.resolve(name)
    no_buy = replace(profile, allow_buy=False)
    no_small = replace(profile, allow_buy_small=False)
    assert industry_entry_signal("BUY", no_buy) == "WATCH_ONLY"
    assert industry_entry_signal("EARLY_BUY", no_buy) == "WATCH_ONLY"
    assert industry_entry_signal("BUY_SMALL", no_buy) == "BUY_SMALL"
    assert industry_entry_signal("BUY_SMALL", no_small) == "WATCH_ONLY"
    assert industry_entry_signal("BUY", no_small) == "BUY"


def test_v41_active_keeps_an_otherwise_complete_entry():
    data = attach_low_entry_v4_context(
        buy_candidate_history(), annual_fundamentals=annual_fundamentals(),
        fundamental_snapshot=fundamental_snapshot(), signal_date="2026-07-21T05:30:00Z",
    )
    result = LowEntryScoreV41Strategy().run(data).iloc[-1]
    assert result["Quality Gate"] == "PASS"
    assert result["Industry Quality Profile"] == "SEMI_PROFILE"
    assert result["低點決策"] == "BUY" and result["建議股數"] > 0


def test_v5_active_preserves_raw_candidate_risk_and_sizing():
    data = _history()
    before = deepcopy(data)
    result = LowEntryScoreV5Strategy().run(data).iloc[-1]
    assert result["Industry Shadow Decision"] == "BUY_SMALL"
    assert result["低點決策"] == "BUY_SMALL"
    assert result["Industry Quality Profile"] == "SEMI_PROFILE"
    assert result["Risk Gate"] == "PASS" and result["RewardRisk"] >= 1.7
    assert result["Position Fraction"] > 0 and result["建議股數"] > 0
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("name,code", SECTORS.items())
def test_daily_v6_company_sector_keeps_native_buy_through_render_and_validation(name, code):
    market = daily_market()
    snapshots = deepcopy(market.fundamental_snapshots)
    snapshots["2330"]["sector"] = code
    _, decisions = build(replace(market, fundamental_snapshots=snapshots))
    decision = decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    assert report["industry_profile"]["name"] == f"{name}_PROFILE"
    assert report["industry_profile"]["shadow_signal"] == "BUY"
    assert report["quality"]["status"] == "PASS"
    assert report["decision"]["signal"] == decision.value("低點決策") == "BUY"
    assert report["sizing"]["indicative_shares"] == decision.value("建議股數") > 0
    assert decision.value("Position Fraction") > 0
    assert report_failure(report) is None
    assert decision.value("Industry Quality Profile") == f"{name}_PROFILE"
    assert decision.value("Industry Shadow Decision") == report["industry_profile"]["shadow_signal"]
    assert decision.field("Industry Quality Profile").model_version == "6.0.0"
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert f"{name}_PROFILE" in html and "ACTIVE" in html and "進場已啟用" in html
    assert "SHADOW" not in html and "本次沒有 BUY / BUY_SMALL" not in html


@pytest.mark.parametrize("metadata", [None, {}, {"name": "SEMI_PROFILE", "mode": "SHADOW"}])
def test_shadow_report_cannot_be_relabelled_buy_for_execution(metadata):
    report = execution_report()
    report["industry_profile"] = metadata
    assert report_failure(report) == "INDUSTRY_PROFILE_NOT_ACTIVE"
