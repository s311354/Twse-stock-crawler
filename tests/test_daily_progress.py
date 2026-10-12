"""Public, observational progress preserves outputs and exception semantics."""
from types import SimpleNamespace
import pytest
from application.daily_performance import DailyPerformance, measure_stage
from application.stock_service import StockAnalysisService


def test_nested_events_preserve_monotonic_paths_and_accumulation():
    clock=[0.];events=[]
    probe=DailyPerformance(clock=lambda:clock[0],stage_callback=events.append)
    with probe.activate(),measure_stage('total'):
        clock[0]=1
        with measure_stage('filing_parse'):clock[0]=4
        with measure_stage('filing_parse'):clock[0]=6
    assert [(e['event'],e['name']) for e in events]==[
        ('started','total'),('started','filing_parse'),('finished','filing_parse'),
        ('started','filing_parse'),('finished','filing_parse'),('finished','total')]
    assert events[1]['path']==(('total',0.),('filing_parse',1))
    assert events[-2]['path']==(('total',0.),)
    assert events[-2]['seconds']==5 and events[-2]['calls']==2
    assert events[-1]['path']==() and events[-1]['status']=='complete'
    assert probe.snapshot()['stages']['total']['seconds']==6


def test_callback_failure_and_interruption_do_not_change_exception_or_scope():
    def broken(_):raise OSError('telemetry only')
    probe=DailyPerformance(clock=lambda:0,stage_callback=broken)
    with pytest.raises(KeyboardInterrupt),probe.activate(),measure_stage('outer'):
        with measure_stage('inner'):raise KeyboardInterrupt()
    assert probe.counters['performance_callback_errors']==4
    assert probe.calls['inner']==probe.calls['outer']==1
    events=[]
    with DailyPerformance(stage_callback=events.append).activate(),measure_stage('fresh'):pass
    assert events[0]['path'][0][0]=='fresh' and len(events[0]['path'])==1


@pytest.mark.parametrize('failing',[False,True])
def test_service_public_stage_callback_preserves_native_sequence(failing):
    def execute_with(callback):
        calls=[]
        def use_case(name,result=None):
            def execute(*args):
                calls.append(name)
                if failing and name=='strategies':raise ValueError('model failure')
                return result
            return SimpleNamespace(execute=execute)
        service=StockAnalysisService(fetch_market_data=use_case('market','context'),
            refresh_fundamentals=use_case('fundamentals'),evaluate_strategies=use_case('strategies','evaluation'),
            generate_report=use_case('report'),send_notification=use_case('notification'),line_chart=None,
            performance_stage_callback=callback)
        request=SimpleNamespace(linechart=False,notification_enabled=False)
        if failing:
            with pytest.raises(ValueError,match='model failure'):service.run(request)
        else:assert service.run(request) is True
        return calls,service.last_performance
    expected,_=execute_with(None);events=[];actual,metrics=execute_with(events.append)
    assert actual==expected
    assert [e['name'] for e in events if e['event']=='started']==['total','market_data','fundamentals','strategies']+([] if failing else ['report'])
    assert events[-1]['status']==('failed' if failing else 'complete')
    assert metrics['status']==('FAILED' if failing else 'SUCCESS')
