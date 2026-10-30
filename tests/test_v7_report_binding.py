"""Exact-byte immutable schema operands; all caller bindings remain checked."""
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.v7_report_input import v7_report_binding
from application.strategy_contracts import StrategyDecisionSchema
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from tests.test_v7_report_input_cache import native_report,decision,counter,encoded,changed_field


def test_repeat_schema_checks_decode_once_and_do_not_retain_report_graph(decision):
    probe=DailyPerformance();work=DailyWorkContext('SYNTHETIC_BINDING')
    with work.activate(),probe.activate():
        for _ in range(3):StrategyDecisionSchema().validate(decision)
        binding=v7_report_binding(decision.value('LES v7 Report'))
        assert probe.counters['v7_report_json_decodes']==1
        assert probe.counters['v7_report_binding_builds']==1
        assert binding.symbol==decision.value('證券代號')
        assert 'LES v7 Report' not in dict(binding.fields)
        assert all(type(value) in (str,int,float,bool,type(None)) for _,value in binding.fields)
        with pytest.raises(FrozenInstanceError):binding.symbol='MUTATED'
        for field,value in [('低點分數',1000.),('日期','2020-01-01'),('證券代號','OTHER'),('收盤',-1.)]:
            with pytest.raises(ValueError):StrategyDecisionSchema().validate(changed_field(decision,field,value))
    work.close();assert not work._values and not work._validation_tokens


def test_header_never_reuses_after_validation_proof_invalidation(native_report,monkeypatch):
    calls=counter(monkeypatch);raw=encoded(native_report);work=DailyWorkContext('SYNTHETIC_BINDING')
    with work.activate():v7_report_binding(raw)
    assert len(calls)==1
    with pytest.raises(RuntimeError),work.activate():raise RuntimeError('SYNTHETIC_INTERRUPT')
    with work.activate():v7_report_binding(raw)
    assert len(calls)==2
    work._validation_tokens.clear();work._validation_bytes=0
    with patch('domain.les_v7.report.validate_report',side_effect=ValueError('SYNTHETIC_REJECT')):
        with work.activate():
            for _ in range(2):
                with pytest.raises(ValueError,match='SYNTHETIC_REJECT'):v7_report_binding(raw)


def test_current_account_is_checked_on_every_immutable_header_hit(decision):
    raw=decision.value('LES v7 Report')
    with DailyWorkContext('SYNTHETIC_BINDING').activate():
        binding=v7_report_binding(raw)
        PortfolioAllocationV2Pipeline._validate_v7_account(decision,binding.account_value,binding.account_cash,())
        with pytest.raises(ValueError,match='ACCOUNT_BINDING'):
            PortfolioAllocationV2Pipeline._validate_v7_account(decision,binding.account_value,binding.account_cash-1,())


def test_canonical_text_and_changed_validator_identity_are_not_interchangeable(native_report,monkeypatch):
    import application.v7_report_input as module
    raw=encoded(native_report);calls=counter(monkeypatch)
    with DailyWorkContext('SYNTHETIC_BINDING').activate():
        v7_report_binding(raw)
        with pytest.raises(ValueError,match='projection mismatch'):v7_report_binding(raw+'\n')
        monkeypatch.setattr(module,'VALIDATION_CACHE_VERSION','SYNTHETIC_NEW_VALIDATOR')
        v7_report_binding(raw)
    assert len(calls)==3


@pytest.mark.parametrize('options',[{'enabled':False},{'max_bytes':0},{'validation_cache_entries':0}])
def test_disabled_evicted_or_no_header_capacity_keeps_full_checks(native_report,options):
    probe=DailyPerformance()
    with DailyWorkContext('SYNTHETIC_BINDING',**options).activate(),probe.activate():
        assert v7_report_binding(encoded(native_report))==v7_report_binding(encoded(native_report))
    assert probe.counters['v7_report_json_decodes']==2
