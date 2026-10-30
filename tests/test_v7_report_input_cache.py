"""SYNTHETIC ONLY: exact report validation reuse, not market/fill evidence."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from unittest.mock import patch

import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.strategy_contracts import StrategyDecisionSchema
from application.v7_report_input import read_v7_report_json
from domain.les_v7.report import validate_report


@pytest.fixture(scope="module")
def native_report():
    from tests.test_v7_r18_pipeline import native
    return native()


@pytest.fixture(scope="module")
def decision():
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.valuation_peers_r10 import capture_comparison_contexts
    from domain.les_v7.policy_r18 import BASELINE_VERSION
    from tests.test_v7_r18_pipeline import neutral_market
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        return AnalysisDatasetBuilder().build_decisions(capture_comparison_contexts(neutral_market()),
            [[1, 1, 1, 1, .01]])[0]


def encoded(report):
    return json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)


def counter(monkeypatch):
    calls = []
    def checked(report):
        calls.append(report["symbol"])
        validate_report(report)
    monkeypatch.setattr("domain.les_v7.report.validate_report", checked)
    return calls


def changed_field(decision, name, value=None, *, future=False):
    return replace(decision, fields=tuple(replace(field,
        **({"available_at": decision.evaluated_at + timedelta(seconds=1)} if future else {"value": value}))
        if field.name == name else field for field in decision.fields))


def test_exact_bytes_validate_once_but_returns_are_always_detached(native_report, monkeypatch):
    calls = counter(monkeypatch)
    raw = encoded(native_report); probe = DailyPerformance()
    with probe.activate(), DailyWorkContext("SYNTHETIC-v1").activate():
        first = read_v7_report_json(raw)
        first["diagnostics"]["features"]["close"] = -999.
        with pytest.raises(ValueError):
            validate_report(first)  # Mutable public calls never use this token.
        second = read_v7_report_json(raw)
        assert second == native_report
        assert read_v7_report_json(raw) == native_report
    assert calls == [native_report["symbol"]]
    assert probe.counters["v7_report_full_validations"] == 1
    assert probe.counters["v7_report_validation_cache_hits"] == 2
    assert probe.counters["v7_report_json_decodes"] == 3


@pytest.mark.parametrize("mode", ["no_work", "disabled", "bytes_zero", "entries_zero"])
def test_no_scope_disabled_or_zero_capacity_never_skips_validation(native_report, monkeypatch, mode):
    from contextlib import nullcontext
    calls = counter(monkeypatch); raw = encoded(native_report)
    options = {"enabled": mode != "disabled",
        "validation_cache_bytes": 0 if mode == "bytes_zero" else 1024 * 1024,
        "validation_cache_entries": 0 if mode == "entries_zero" else 4096}
    scope = nullcontext() if mode == "no_work" else DailyWorkContext("SYNTHETIC", **options).activate()
    with scope:
        assert read_v7_report_json(raw) == read_v7_report_json(raw) == native_report
    assert len(calls) == 2


def test_exact_source_bytes_and_validator_policy_changes_are_misses(native_report, monkeypatch):
    import application.v7_report_input as module
    calls = counter(monkeypatch); raw = encoded(native_report)
    with DailyWorkContext("SYNTHETIC-v1").activate():
        read_v7_report_json(raw)
        read_v7_report_json(raw + "\n")
        monkeypatch.setattr(module, "VALIDATION_CACHE_VERSION", "SYNTHETIC-other-validator")
        read_v7_report_json(raw)
    with DailyWorkContext("SYNTHETIC-v2").activate():
        read_v7_report_json(raw)
    assert len(calls) == 4


@pytest.mark.parametrize("damage", ["score", "future", "restriction", "policy"])
def test_mutation_or_forged_hash_cannot_reuse_success_and_failure_is_not_cached(
        native_report, monkeypatch, damage):
    calls = counter(monkeypatch); raw = encoded(native_report)
    bad = deepcopy(native_report)
    if damage == "score": bad["envelope"]["score"]["final_score"] += 1
    elif damage == "future": bad["envelope"]["event"]["cutoff_at"] = "2020-01-01T13:30:00+08:00"
    elif damage == "restriction": bad["diagnostics"]["gate_receipts"][1]["state"] = "UNKNOWN"
    else: bad["diagnostics"]["rule_baseline"]["policy"]["native_risk_limits"]["max_stop_atr"] = 99.
    with DailyWorkContext("SYNTHETIC").activate():
        read_v7_report_json(raw)
        for _ in range(2):
            with pytest.raises((KeyError, ValueError)):
                read_v7_report_json(encoded(bad))
        assert read_v7_report_json(raw) == native_report
    assert len(calls) == 3


def test_token_budget_is_independent_bounded_and_cleared_on_exception_and_close(native_report, monkeypatch):
    calls = counter(monkeypatch); raw = encoded(native_report)
    work = DailyWorkContext("SYNTHETIC", max_bytes=512, validation_cache_entries=1)
    with work.activate():
        read_v7_report_json(raw)
        for i in range(5):
            work.memo_immutable("heavy", str(i), lambda: "x" * 100, len)
        read_v7_report_json(raw)
        assert len(calls) == 1
        read_v7_report_json(raw + "\n")
        read_v7_report_json(raw)
        assert len(calls) == 3
        assert work.stats["validation_entries"] == 1
        assert work.stats["validation_bytes"] <= work.validation_cache_bytes
    with pytest.raises(RuntimeError), work.activate():
        raise RuntimeError("SYNTHETIC_INTERRUPT")
    assert work.stats["validation_entries"] == 0
    with work.activate(): read_v7_report_json(raw)
    assert len(calls) == 4
    work.close()
    assert work.stats["validation_bytes"] == 0
    with pytest.raises(ValueError, match="CLOSED"):
        work.validate_once("test", "key", lambda: None)


def test_pae_renderer_complete_outputs_match_and_account_projection_checks_still_run(decision, monkeypatch):
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from infrastructure.report.html_renderer import RendererFactory
    calls = counter(monkeypatch)
    original = json.loads(decision.value("LES v7 Report"))
    account = original["pae_authorization"]["account"]
    context = PortfolioRuntimeContext(account_equity=account["value"], current_cash=account["available_cash"])
    outputs, counts = [], []
    for enabled in (False, True):
        calls.clear()
        with DailyWorkContext("SYNTHETIC", enabled=enabled).activate():
            StrategyDecisionSchema().validate(decision)
            result = PortfolioAllocationV2Pipeline().enrich([decision], context)
            markup = RendererFactory.get_renderer("daily_strategy_email").render(result)
            outputs.append((result, markup))
            counts.append(len(calls))
    assert outputs[0] == outputs[1]
    assert counts[0] >= 5 and counts[1] == 1
    with DailyWorkContext("SYNTHETIC").activate():
        StrategyDecisionSchema().validate(decision)
        for name, value in (("低點分數", 1000.), ("日期", "2020-01-01"), ("證券代號", "OTHER"),
                            ("收盤", -1000.)):
            with pytest.raises(ValueError):
                StrategyDecisionSchema().validate(changed_field(decision, name, value))
        with pytest.raises(ValueError, match="future"):
            StrategyDecisionSchema().validate(changed_field(decision, "收盤", future=True))
        with pytest.raises(ValueError, match="ACCOUNT_BINDING"):
            PortfolioAllocationV2Pipeline._validate_v7_account(decision, account["value"] * 2,
                account["available_cash"], ())


def test_renderer_still_verifies_external_source_on_every_cache_hit(decision, monkeypatch):
    from infrastructure.report import low_entry_v7_renderer as renderer
    calls = counter(monkeypatch); checks = []
    def verify(report, *, verifier=None):
        checks.append(report["symbol"])
        if len(checks) == 2: raise ValueError("SYNTHETIC_SOURCE_CORRUPTION")
    monkeypatch.setattr(renderer, "verify_report_valuation_sources", verify)
    with DailyWorkContext("SYNTHETIC").activate():
        renderer.report_for(decision, source_verifier=lambda _: None)
        with pytest.raises(ValueError, match="SOURCE_CORRUPTION"):
            renderer.report_for(decision, source_verifier=lambda _: None)
    assert len(calls) == 1 and len(checks) == 2
