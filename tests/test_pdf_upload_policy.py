"""SYNTHETIC policy examples; never represent historical version proof."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from application.pdf_upload_policy import (PDF_UPLOAD_SOURCE, apply_pdf_upload_policy,
                                           verify_pdf_upload_policy)
from application.filing_publication import audit_filings
from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import filing_binding, materialize_filings, verify_publication
from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.publication_repair import publish_repair
from infrastructure.research_archive import create_dataset
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.storage.research_cache import SQLiteFilingHistory
from interface.research import main
from tests.test_research_archive import raw_fixture, add_raw
from tests.test_strict_replay import request


def fixture(upload="115/08/01 12:00:00"):
    data = raw_fixture()
    item = add_raw(data, proof_present=False)
    body = f'''<html><body>SYNTHETIC TEST ONLY<table><tr>
        <th>證券代號</th><th>資料年度</th><th>資料細節說明</th>
        <th>電子檔案</th><th>上傳日期</th><th>財務報告更(補)正</th></tr>
        <tr><td>2330</td><td>115 年 第二季</td><td>IFRSs合併財報</td>
        <td>202602_2330_AI1.pdf</td><td>{upload}</td><td>無</td></tr></table></body></html>'''
    doc = dict(url='https://doc.twse.com.tw/server-java/t57sb01?co_id=2330&year=115',
               retrieved_at='2026-09-20T08:00:00+08:00', body_utf8=body,
               sha256=hashlib.sha256(body.encode()).hexdigest())
    review = dict(schema='filing-pdf-upload-review/v1', parent_dataset_id=digest(data),
                  accepted_by='SYNTHETIC_TEST_USER', accepted_at='2026-09-30T00:00:00+08:00',
                  reviews=[dict(binding=filing_binding('2330', item),
                                pdf_filename='202602_2330_AI1.pdf', document=doc)])
    return data, review


def audit(data, cutoff='2026-08-04T13:30:00+08:00'):
    return audit_filings(data, ('2330',), cutoff, parse_filing=parse_ixbrl)


def test_user_policy_changes_selection_not_original_metadata_or_proof():
    data, review = fixture()
    original = deepcopy(data)
    adopted, _ = apply_pdf_upload_policy(data, review)
    item = adopted['filings']['2330'][0]
    assert data == original
    assert {k:v for k,v in item.items() if k != 'publication_time_policy'} == original['filings']['2330'][0]
    assert audit(original)['rows'][0]['status'] == 'DOCUMENT_PENDING_PUBLICATION_PROOF'
    a = audit(adopted)
    assert a['rows'][0]['status'] == 'POLICY_AVAILABLE'
    assert a['rows'][0]['eligible_at_cutoff'] and a['rows'][0]['selected']
    assert a['rows'][0]['version_binding_status'] == 'ASSUMED_NOT_VERIFIED'
    assert a['status'] == 'PARTIAL'
    assert a['rows'][0]['published_at'] is None
    rows, errors = materialize_filings(adopted, parse_filing=parse_ixbrl)
    assert not errors['2330']
    assert rows['2330'][0]['availability_source'] == PDF_UPLOAD_SOURCE
    assert PDF_UPLOAD_SOURCE not in TRUSTED_AVAILABILITY_SOURCES
    with pytest.raises(ValueError, match='NOT_VERIFIED'):
        verify_publication(item['publication_time_policy'], binding=review['reviews'][0]['binding'])


@pytest.mark.parametrize('upload,before,boundary', [
    ('115/08/04 13:30:00','2026-08-04T13:29:59+08:00','2026-08-04T05:30:00Z'),
    ('115/08/04','2026-08-04T23:59:59+08:00','2026-08-05T00:00:00+08:00'),
])
def test_source_timestamp_precision_timezone_and_cutoff(upload, before, boundary):
    data, review = fixture(upload)
    adopted, _ = apply_pdf_upload_policy(data, review)
    assert audit(adopted,before)['rows'][0]['status'] == 'POLICY_AFTER_CUTOFF'
    assert audit(adopted,boundary)['rows'][0]['status'] == 'POLICY_AVAILABLE'


@pytest.mark.parametrize('field,value', [('symbol','3014'),('quarter',3),('statement_scope','individual'),
                                       ('content_sha256','a'*64)])
def test_wrong_binding_rejected(field, value):
    data, review = fixture()
    review['reviews'][0]['binding'][field] = value
    with pytest.raises(ValueError,match='EXACT_EXISTING'):
        apply_pdf_upload_policy(data,review)


@pytest.mark.parametrize('damage', ['company','period','scope','correction','duplicate','hash','host','time','acceptance'])
def test_official_source_must_support_the_row_even_for_an_assumption(damage):
    data, review = fixture()
    doc = review['reviews'][0]['document']
    if damage=='company': doc['body_utf8']=doc['body_utf8'].replace('<td>2330</td>','<td>3014</td>')
    if damage=='period': doc['body_utf8']=doc['body_utf8'].replace('第二季','第三季')
    if damage=='scope': doc['body_utf8']=doc['body_utf8'].replace('合併','個體')
    if damage=='correction': doc['body_utf8']=doc['body_utf8'].replace('<td>無</td>','<td>更正</td>')
    if damage=='duplicate': doc['body_utf8']+=doc['body_utf8']
    if damage=='time': doc['body_utf8']=doc['body_utf8'].replace('115/08/01 12:00:00','115/13/01 12:00:00')
    if damage=='host': doc['url']=doc['url'].replace('doc.twse.com.tw','doc.twse.com.tw.example.com')
    if damage=='acceptance': review['accepted_at']='2026-09-19T00:00:00+08:00'
    doc['sha256']='a'*64 if damage=='hash' else hashlib.sha256(doc['body_utf8'].encode()).hexdigest()
    with pytest.raises(ValueError): apply_pdf_upload_policy(data,review)


def test_known_revisions_conflicts_and_proofs_cannot_borrow_pdf_original_date():
    data, review = fixture()
    data['filings']['2330'][0]['revision_number']=2
    with pytest.raises(ValueError, match='KNOWN_REVISION'): apply_pdf_upload_policy(data,review)
    data, review = fixture()
    data['filings']['2330'][0]['publication_proof']={'status':'CONFLICT'}
    with pytest.raises(ValueError, match='OVERRIDE_PROOF'): apply_pdf_upload_policy(data,review)
    data, review = fixture()
    data['filings']['2330'].append(dict(data['filings']['2330'][0],content_sha256='b'*64))
    with pytest.raises(ValueError, match='MULTIPLE_VERSIONS'): apply_pdf_upload_policy(data,review)


def test_big5_and_untrusted_extracted_text():
    import base64
    data,review=fixture()
    doc=review['reviews'][0]['document']
    body=doc.pop('body_utf8').encode('big5')
    doc.update(body_base64=base64.b64encode(body).decode(),sha256=hashlib.sha256(body).hexdigest(),
               extracted_text='False upload time 100/01/01')
    adopted,_=apply_pdf_upload_policy(data,review)
    result=verify_pdf_upload_policy('2330',adopted['filings']['2330'][0])
    assert result['available_at']=='2026-08-01T12:00:00+08:00'


def test_parse_reuse_and_new_availability_index(tmp_path):
    data,review=fixture()
    calls=[]
    def parse(content): calls.append(True); return parse_ixbrl(content)
    cutoff=aware_timestamp('2026-08-04T13:30:00+08:00')
    def index(payload): return SQLiteFilingHistory(tmp_path/'parse.sqlite3',evidence_id=digest(payload),version='test',parse_filing=parse)
    assert not index(data).load(data,('2330',),cutoff)[0]['2330'][0]['facts']
    adopted,_=apply_pdf_upload_policy(data,review)
    rows,errors=index(adopted).load(adopted,('2330',),cutoff)
    assert rows['2330'][0]['facts'] and not errors['2330'] and len(calls)==1
    assert rows==json.loads(json.dumps(materialize_filings(adopted,parse_filing=parse_ixbrl)[0]))


@pytest.mark.parametrize('published,expected', [('2026-08-03', 'VERIFIED_AVAILABLE'),
                                               ('2026-08-05', 'PUBLISHED_AFTER_CUTOFF')])
def test_formal_proof_supersedes_pdf_policy_without_rewriting_original(published, expected, tmp_path):
    from application.filing_publication import apply_reviews
    from tests.test_filing_publication_repair import reviewed
    data, review = fixture()
    adopted, _ = apply_pdf_upload_policy(data, review)
    original = deepcopy(adopted)
    item = adopted['filings']['2330'][0]
    calls = []
    def parse(content):
        calls.append(True)
        return parse_ixbrl(content)
    cutoff = aware_timestamp('2026-08-04T13:30:00+08:00')
    def index(payload):
        return SQLiteFilingHistory(tmp_path/'proof.sqlite3', evidence_id=digest(payload),
                                  version='test', parse_filing=parse)
    assert index(adopted).load(adopted, ('2330',), cutoff)[0]['2330'][0]['facts']
    fixed, _ = apply_reviews(adopted, [reviewed(adopted, item, published)])
    audited = audit(fixed)['rows'][0]
    assert audited['status'] == expected
    assert audited['version_binding_status'] == 'VERIFIED'
    assert adopted == original
    assert fixed['filings']['2330'][0]['publication_time_policy'] == item['publication_time_policy']
    rows, errors = index(fixed).load(fixed, ('2330',), cutoff)
    row = rows['2330'][0]
    assert not errors['2330'] and len(calls) == 1  # Reuse parsing, not old selection.
    assert row['availability_source'] == 'OFFICIAL_FILING_TIMESTAMP'
    assert bool(row['facts']) is (expected == 'VERIFIED_AVAILABLE')
    assert 'publication_policy' not in row['receipt']
    assert row['receipt']['publication_verification']['status'] == 'VERIFIED'
    assert row['receipt']['original_availability']['available_at'] == item['available_at']


@pytest.mark.parametrize('damage', ['conflict', 'binding', 'chain', 'empty'])
def test_invalid_proof_cannot_fall_back_to_earlier_assumed_pdf_time(damage):
    from tests.test_filing_publication_repair import reviewed
    data, review = fixture()
    adopted, _ = apply_pdf_upload_policy(data, review)
    item = adopted['filings']['2330'][0]
    p = reviewed(adopted, item)
    if damage == 'conflict': p = {'status': 'CONFLICT'}
    if damage == 'binding': p['binding']['symbol'] = '3014'
    if damage == 'chain': p['version_chain'][-1]['to_id'] = '0'*64
    if damage == 'empty': p = {}
    item['publication_proof'] = p
    row = audit(adopted)['rows'][0]
    assert not row['eligible_at_cutoff'] and not row['selected']
    assert row['status'] in {'VERSION_CONFLICT', 'PUBLICATION_VERSION_UNVERIFIED'}
    rows, errors = materialize_filings(adopted, parse_filing=parse_ixbrl)
    assert not rows['2330'] and errors['2330']


def test_publish_explicit_policy_parent_repeat_cli_and_strict_research(tmp_path):
    import base64
    data,review=fixture()
    source=create_dataset(data,tmp_path/'datasets')
    old_bytes=source.read_bytes()
    review['parent_dataset_id']=source.parent.name
    doc=review['reviews'][0]['document']
    doc['body_base64']=base64.b64encode(doc.pop('body_utf8').encode()).decode()
    review_file=tmp_path/'review.json'
    review_file.write_text(json.dumps(review))
    kwargs=dict(as_of='2026-08-04T13:30:00+08:00',symbols=('2330',))
    with pytest.raises(ValueError,match='SCHEMA_MISMATCH'):
        publish_repair(source,review_file,tmp_path/'datasets',**kwargs)
    path,report=publish_repair(source,review_file,tmp_path/'datasets',accept_pdf_upload_policy=True,**kwargs)
    assert path.parent.name!=source.parent.name and source.read_bytes()==old_bytes
    assert report['verified_reviews']==0 and report['policy_assumptions']==1
    assert report['status']=='USER_ACCEPTED_PDF_UPLOAD_POLICY'
    assert json.loads((path.parent/'manifest.json').read_text())['parent_dataset_id']==source.parent.name
    assert b'body_base64' not in path.read_bytes()
    document=json.loads(path.read_text())['payload']['filings']['2330'][0]['publication_time_policy']['document']
    assert 'body_ref' in document and 'body_utf8' not in document
    assert publish_repair(source,review_file,tmp_path/'datasets',accept_pdf_upload_policy=True,**kwargs)[0]==path
    args=['filing-pdf-policy','--evidence',str(source),'--review',str(review_file),
          '--store',str(tmp_path/'datasets'),'--output',str(tmp_path/'cli.json'),
          '--as-of',kwargs['as_of'],'--accept-pdf-upload-assumption','2330']
    assert main(args)==0
    assert json.loads((tmp_path/'cli.json').read_text())['dataset_id']==path.parent.name
    engine,_=build_replay_engine(path)
    req=replace(request(tmp_path,data),versions=current_versions(),evidence_id=path.parent.name)
    result,coverage=engine.run_with_coverage(req)
    assert result['diagnostics']['publication_time_assumptions']['2330']
    decision=result['decisions'][0]
    assert 'publication_time' not in decision['hard_gates']
    assert decision['publication_time_status']=='ASSUMED_NOT_VERIFIED'
    assert decision['evidence_warnings']==['ASSUMED_NOT_VERIFIED:OFFICIAL_PDF_UPLOAD_TIME']
    assert not any('ASSUMED' in reason for reason in decision['blocking_reasons'])
    assert not decision['research_entry_eligible']
    assert 'ASSUMED_PUBLICATION_TIME' in coverage['rows'][0]['coverage_problems']
    assert 'PDF_UPLOAD_TIME_ASSUMPTION' in result['model_policy']


@pytest.mark.parametrize('blocker', [None, 'restrictions', 'kelly', 'market', 'conflict', 'risk'])
def test_declared_assumption_alone_does_not_block_research_or_pae(tmp_path, monkeypatch, blocker):
    """SYNTHETIC all-PASS adapter tests orchestration, never proves a real BUY."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from application.replay_inputs import PinnedReplayEvidence
    from application.replay_service import StrictReplayEngine
    from infrastructure.storage.replay_state import SQLiteReplayState
    from tests.test_strict_replay import payload_fixture, VERSIONS

    data = payload_fixture()
    req = request(tmp_path, data)
    selected = PinnedReplayEvidence(digest(data), data).select(req)
    selected.diagnostics['publication_time_assumptions'] = {'2330': [{'policy': 'SYNTHETIC'}]}
    if blocker == 'restrictions': selected.diagnostics['restrictions']['2330']['status'] = 'UNKNOWN'
    if blocker == 'market': selected.diagnostics['missing_bars']['2330'] = ['2026-09-11']
    if blocker == 'conflict':
        selected.diagnostics['excluded_evidence']['2330'] = [{'reason': 'CONFLICTING_FILING_VERSION'}]
    values = {'證券代號': '2330', '日期': '2026-09-11', '低點決策': 'BUY', '低點分數': 99,
              'EREM 決策訊號': 'HOLD', 'EREM 分數': 80, 'PAE Binding': 'NON_BINDING',
              'PAE Decision': 'NO_ACTION', 'Half Kelly Weight': None if blocker == 'kelly' else .1,
              'LES v6 Report': json.dumps({'quality': {}, 'data': {}})}
    decision = SimpleNamespace(fields=[SimpleNamespace(name=k, value=v) for k, v in values.items()],
                               value=values.get)
    monkeypatch.setattr('application.replay_service.asdict', lambda obj: vars(obj))
    monkeypatch.setattr('application.replay_service.entry_gate_checks',
                        lambda report: {'quality': 'PASS', 'risk': 'FAIL' if blocker == 'risk' else 'PASS'})
    monkeypatch.setattr('application.replay_service.report_failure', lambda *a, **kw: None)
    allocator = Mock()
    allocator.enrich.return_value = (decision,)
    monkeypatch.setattr('application.replay_service.PortfolioAllocationV2Pipeline', lambda **kw: allocator)
    engine = StrictReplayEngine(SimpleNamespace(select=lambda request: selected), SQLiteReplayState(),
                                lambda: VERSIONS, builder=SimpleNamespace(build_decisions=lambda *a: (decision,)))
    result, coverage = engine.run_with_coverage(req)
    row = result['decisions'][0]
    assert row['research_entry_eligible'] is (blocker is None)
    assert row['publication_time_status'] == 'ASSUMED_NOT_VERIFIED'
    assert not row['live_order_authorized'] and not row['is_fill']
    assert coverage['rows'][0]['coverage_status'] == 'PARTIAL'
    assert 'publication_time' not in coverage['rows'][0]['insufficient_gates']
    if blocker in {'market', 'conflict'}:
        assert allocator.enrich.call_args.kwargs['blocked_reason']
    else:
        assert 'blocked_reason' not in allocator.enrich.call_args.kwargs
