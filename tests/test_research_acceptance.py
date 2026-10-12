"""SYNTHETIC acceptance diagnostics. No sources here are real publication proof."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import json

import pytest

from application.replay_contracts import digest, calendar_window
from application.replay_inputs import PinnedReplayEvidence
from application.research_audit import coverage_audit, summarize
from application.research_acceptance import acceptance_checks, decision_stages
from tests.test_strict_replay import payload_fixture, request


def selected_inputs(tmp_path):
    payload = payload_fixture()
    req = request(tmp_path, payload)
    return req, PinnedReplayEvidence(digest(payload), payload).select(req)


@pytest.mark.parametrize('days', [1, 3, 700])
def test_coverage_uses_explicit_inclusive_range_not_default_700(tmp_path, days):
    payload = payload_fixture()
    req = request(tmp_path, payload)
    req = replace(req, history_start=req.history_end - dt.timedelta(days=days-1))
    # The 700-day request extends the synthetic source, never pads market bars.
    if days == 700:
        payload['start'] = req.history_start.isoformat()
        for i in range(days):
            day = req.history_start + dt.timedelta(days=i)
            payload['calendar']['days'].setdefault(str(day), 'OPEN' if day.weekday() < 5 else 'CLOSED')
        req = replace(req, evidence_id=digest(payload))
        assert (req.history_start, req.history_end) == calendar_window(req.history_end)
    selected = PinnedReplayEvidence(digest(payload), payload).select(req)
    row = coverage_audit(req, selected)['rows'][0]
    assert row['inclusive_calendar_days'] == days
    assert row['window_start'] == str(req.history_start)
    assert 'NOT_700_CALENDAR_DAY_WINDOW' not in row['coverage_problems']
    assert row['known_open_sessions'] == sum(str(req.history_start) <= d <= str(req.history_end)
                                             for d in selected.sessions)


def test_pending_proof_is_not_the_same_as_later_verified_publication(tmp_path):
    req, selected = selected_inputs(tmp_path)
    selected.diagnostics['excluded_evidence']['2330'] = [
        dict(fiscal_year=2026, quarter=2, availability_source=source, reason='AFTER_CUTOFF')
        for source in ('FIRST_OBSERVED_DOWNLOAD', 'OFFICIAL_FILING_TIMESTAMP')]
    before = deepcopy(selected.diagnostics)
    report = coverage_audit(req, selected)
    check = report['rows'][0]['acceptance_checks']['financial_availability']
    assert [r['gap_status'] for r in check['excluded_versions']] == [
        'DOCUMENT_PENDING_PUBLICATION_PROOF', 'PUBLISHED_AFTER_CUTOFF']
    assert check['pending_proof_count'] == 1
    assert 'PUBLICATION_PROOF_PENDING' in report['rows'][0]['coverage_problems']
    assert selected.diagnostics == before


def test_publication_pass_does_not_certify_financial_comparability(tmp_path):
    req, selected = selected_inputs(tmp_path)
    selected.diagnostics['financials']['2330'].update(
        missing_periods=['2024Q1'], unresolved_fields=['2024Q4:basic_eps:PERIOD_BASIS_REVIEW'])
    checks = acceptance_checks(req, selected, selected.diagnostics, '2330')
    assert checks['financial_availability']['status'] == 'PASS'
    assert checks['financial_comparability']['status'] == 'UNKNOWN'
    assert checks['financial_comparability']['unresolved_fields'] == ['2024Q4:basic_eps:PERIOD_BASIS_REVIEW']


@pytest.mark.parametrize('status,expected', [('CLEAR', 'PASS'), ('RESTRICTED', 'FAIL'), ('UNKNOWN', 'UNKNOWN')])
def test_restrictions_preserve_validated_clear_restricted_and_missing(tmp_path, status, expected):
    req, selected = selected_inputs(tmp_path)
    selected.diagnostics['restrictions']['2330'] = dict(status=status, reason='SYNTHETIC')
    check = acceptance_checks(req, selected, selected.diagnostics, '2330')['historical_restrictions']
    assert check['status'] == expected
    assert check['check']['reason'] == 'SYNTHETIC'


@pytest.mark.parametrize('missing', ['price', 'calendar', 'benchmark', 'breadth', 'kelly'])
def test_market_or_calibration_gap_remains_unknown(tmp_path, missing):
    req, selected = selected_inputs(tmp_path)
    diag = selected.diagnostics
    diag['kelly']['2330'] = {'status': 'PASS'}
    diag['missing_taiex_dates'] = []
    diag['missing_breadth_dates'] = []
    diag['missing_bars']['2330'] = []
    diag['calendar_unknown_dates'] = []
    if missing == 'price': diag['missing_bars']['2330'] = [str(req.history_end)]
    if missing == 'calendar': diag['calendar_unknown_dates'] = [str(req.history_end)]
    if missing == 'benchmark': diag['missing_taiex_dates'] = [str(req.history_end)]
    if missing == 'breadth': diag['missing_breadth_dates'] = [str(req.history_end)]
    if missing == 'kelly': diag['kelly']['2330'] = {'status': 'UNKNOWN', 'reason': 'NO_CALIBRATION'}
    check = acceptance_checks(req, selected, diag, '2330')['market_and_calibration']
    assert check['status'] == 'UNKNOWN'
    assert check['prices']['latest_bar'] == str(req.history_end)


def test_raw_buy_and_high_score_are_not_eligibility_allocation_or_fills(tmp_path):
    req, selected = selected_inputs(tmp_path)
    model = dict(symbol='2330', raw=dict(les='BUY', les_score=99, erem='HOLD', pae='NO_ACTION'),
                 hard_gates=dict(quality='PASS', restrictions='UNKNOWN', kelly='UNKNOWN', allocation='FAIL'),
                 blocking_reasons=['restrictions:UNKNOWN', 'kelly:UNKNOWN', 'allocation:FAIL'],
                 research_entry_eligible=False)
    stages = decision_stages(model)
    assert stages['raw']['les'] == 'BUY'
    assert stages['blockers']['native_model'] == {}
    assert stages['blockers']['historical_evidence'] == {'restrictions': 'UNKNOWN', 'kelly': 'UNKNOWN'}
    assert stages['blockers']['account_allocation'] == {'allocation': 'FAIL'}
    assert not stages['research_entry_eligible'] and not stages['is_fill']
    audit = coverage_audit(req, selected, dict(decisions=[model]))
    text = summarize(audit)
    assert 'Four Evidence Checks' in text and 'BUY / HOLD / NO_ACTION' in text
    assert 'historical_evidence: restrictions=UNKNOWN, kelly=UNKNOWN' in text


def test_explicit_range_audit_cli_matches_shared_api_without_state_or_network(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from infrastructure.replay_runtime import build_replay_engine, current_versions
    from infrastructure.research_archive import create_dataset
    from interface.research import main
    from tests.test_research_archive import raw_fixture
    payload = raw_fixture()
    evidence = create_dataset(payload, tmp_path/'datasets')
    output = tmp_path/'audit'
    holdings = tmp_path/'empty.json'
    holdings.write_text('[]')
    monkeypatch.setattr('requests.sessions.Session.request', Mock(side_effect=AssertionError('network')))
    args = ['audit', '--evidence', str(evidence), '--start', '2026-09-11', '--date', '2026-09-11',
            '--holdings', str(holdings), '--cash', '1000000', '--equity', '1000000',
            '--run-id', 'audit', '--output', str(output)]
    assert main(args) == 0
    engine, eid = build_replay_engine(evidence)
    req = replace(request(tmp_path, payload, replay_id='audit'),
                  versions=current_versions(), evidence_id=eid, history_start=dt.date(2026, 9, 11))
    shared = coverage_audit(req, engine.evidence.select(req))
    assert json.loads((output/'coverage.json').read_text()) == json.loads(json.dumps(shared))
    assert (output/'coverage.md').read_text() == summarize(shared)
    assert not list(output.glob('*.sqlite3'))
    assert main(args) == 0  # Same audited input is idempotent, never rebases.
    args[args.index('--start')+1] = '2026-09-12'
    with pytest.raises(SystemExit):
        main(args)
