"""SYNTHETIC cache only: native query/parser audit and write-once contracts."""
from dataclasses import replace
from hashlib import sha256
from html import escape
import json

import pytest

from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.ttm_audit import audit_ttm, main
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.strategies.test_low_entry_v6_comparative_ttm import CUTOFF, originals


def document(filing):
    contexts = {}
    values = []
    for fact in filing.facts:
        key = fact.context_id
        contexts[key] = (fact.start_date, fact.end_date, fact.instant)
        if fact.numeric_value is None:
            values.append(f'<ix:nonNumeric name="{fact.concept}" contextRef="{key}">{escape(fact.text_value or "")}</ix:nonNumeric>')
        else:
            values.append(f'<ix:nonFraction name="{fact.concept}" contextRef="{key}" unitRef="money">{fact.numeric_value}</ix:nonFraction>')
    definitions = []
    for key, (start, end, instant) in contexts.items():
        period = f'<xbrli:instant>{instant}</xbrli:instant>' if instant else f'<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>'
        definitions.append(f'<xbrli:context id="{key}"><xbrli:period>{period}</xbrli:period></xbrli:context>')
    return ('<html xmlns:currency="http://www.xbrl.org/2003/iso4217" xmlns:xbrli="http://www.xbrl.org/2003/instance"><body>'
            '<xbrli:unit id="money"><xbrli:measure>currency:TWD</xbrli:measure></xbrli:unit>'
            + ''.join(definitions + values) + '</body></html>').encode()


def cache(root):
    repo = MopsRepository(root)
    for filing in originals()[1]:
        content = document(filing)
        candidate = replace(_candidate(filing.fiscal_year, filing.quarter), period_end=filing.period_end,
                            source_url="https://example.test/SYNTHETIC")
        repo.save_revision(candidate, content, parse_ixbrl(content), available_at=filing.available_at,
                           fetched_at=filing.available_at)
    return repo


def test_readonly_audit_recovers_ttm_keeps_missing_eps_and_has_stable_identity(tmp_path):
    root = tmp_path / 'mops'
    cache(root)
    before = sha256((root / 'mops.sqlite3').read_bytes()).hexdigest()
    result = audit_ttm(root, ('2330',), CUTOFF)
    assert result['summary']['required_ttm_complete_before'] == 0
    assert result['summary']['required_ttm_complete_after'] == 1
    after = result['rows'][0]['after']
    assert after['profitability_basis'] == 'REPORTED_SAME_PERIOD_MARGIN'
    assert result['summary']['quality_scoring_required'] is False
    for key in ('score', 'status', 'confidence', 'coverage', 'verification'):
        assert result['rows'][0]['before'][key] == after[key]
    assert after['coverage']['growth'] == 100  # Eight-quarter EPS is diagnostic only.
    assert after['diagnostic_checks']['positive_eps_quarters']['value'] is None
    assert after['diagnostic_checks']['positive_eps_quarters']['scored'] is False
    assert after['field_checks'] and after['verification']['total_checks'] > 0
    assert result['rows'][0]['after']['ttm']['basic_eps_ttm'] is None
    assert result == audit_ttm(root, ('2330',), CUTOFF)
    assert before == sha256((root / 'mops.sqlite3').read_bytes()).hexdigest()


def test_cli_explicit_universe_and_immutable_publication(tmp_path, capsys):
    root, output = tmp_path / 'mops', tmp_path / 'audit.json'
    cache(root)
    args = ['--cache', str(root), '--as-of', CUTOFF, '--output', str(output), '2330']
    assert main(args) == 0
    first = output.read_bytes()
    assert main(args) == 0 and output.read_bytes() == first
    assert json.loads(first)['symbols'] == ['2330']
    with pytest.raises(ValueError, match='MUST_NOT_MODIFY'):
        main(['--cache', str(root), '--as-of', CUTOFF, '--output', str(root/'audit.json'), '2330'])


def test_report_universe_includes_stale_without_selecting_winners(tmp_path):
    root, output, report = tmp_path / 'mops', tmp_path / 'audit.json', tmp_path / 'report.json'
    cache(root)
    # Explicit synthetic saved report; missing-company evidence stays missing.
    report.write_text(json.dumps(dict(reports=[dict(symbol='2330')], stale_reports=[dict(symbol='9999')], signal_date='2026-10-02')))
    assert main(['--cache', str(root), '--as-of', CUTOFF, '--output', str(output), '--report-universe', str(report)]) == 0
    result = json.loads(output.read_text())
    assert result['symbols'] == ['2330', '9999']
    assert result['rows'][1]['after']['status'] == 'DATA_INSUFFICIENT'
