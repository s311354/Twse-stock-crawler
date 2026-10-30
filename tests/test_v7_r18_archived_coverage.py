"""Synthetic archived-vintage fixtures; no new policy or market results."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from domain.les_v7 import rule_baseline
from domain.les_v7.report import validate_report
from tests.test_v7_r18_pipeline import native


@pytest.fixture(scope='module')
def archived():
    score = rule_baseline.score_baseline
    def old(*args, **kwargs):
        return score(*args, **dict(kwargs, legacy_resilience=True))
    with patch('domain.les_v7.engine.score_baseline', side_effect=old):
        return native()


def test_archived_r18_still_fully_recomputes_old_coverage(archived):
    before=deepcopy(archived)
    validate_report(archived)
    assert archived==before
    row=next(c for c in archived['diagnostics']['rule_baseline']['score_evidence']['conditions']
        if c['id']=='relative_strength:correction_resilience')
    assert row['dependencies']==['market_correction20','downside_capture20','correction_support','low']


@pytest.mark.parametrize('change',['points','coverage','decision','gate'])
def test_legacy_compatibility_never_accepts_tampered_score_or_safety(archived,change):
    report=deepcopy(archived)
    if change=='points':report['envelope']['score']['final_score']+=1
    elif change=='coverage':report['diagnostics']['rule_baseline']['score_evidence']['available_weight']+=1
    elif change=='decision':report['envelope']['strategy']['strategy_decision']='BUY'
    else:report['diagnostics']['gate_receipts'][0]['state']='UNKNOWN'
    with pytest.raises(ValueError):validate_report(report)


def test_new_reports_keep_corrected_resilience_semantics():
    report=native()
    validate_report(report)
    row=next(c for c in report['diagnostics']['rule_baseline']['score_evidence']['conditions']
        if c['id']=='relative_strength:correction_resilience')
    assert row['dependencies'][:2]==['market_ready','market_correction20']
