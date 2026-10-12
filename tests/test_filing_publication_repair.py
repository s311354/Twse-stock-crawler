"""SYNTHETIC proofs only; production collection never auto-verifies these."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from application.filing_publication import (PROOF_SCHEMA, REVIEW_CHECKS, audit_filings,
                                            verify_filing_publication, apply_reviews)
from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import filing_binding, materialize_filings, verified_document
from domain.errors import NetworkError
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.filing_evidence_collection import collect_sources, collection_plan, classify_source
from infrastructure.publication_repair import (audit_dataset, publish_repair, review_repair, compare_datasets)
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.research_archive import create_dataset
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.storage.research_cache import SQLiteFilingHistory
from tests.test_research_archive import raw_fixture, add_raw, proof
from tests.test_strict_replay import request
from tests.test_research_sqlite_cache import history
from interface.research import main


def reviewed(payload, item, published="2026-08-01", precision="DATE"):
    result = proof(filing_binding("2330", item), publication=published, precision=precision)
    result.update(schema=PROOF_SCHEMA, review_checks={k: True for k in REVIEW_CHECKS},
                  version_identity=dict(official_document_id="SYNTHETIC-doc", revision_id=f"SYNTHETIC-r{item['revision_number']}",
                                        revision_number=item["revision_number"], kind="ORIGINAL"))
    doc_id, revision = "SYNTHETIC-doc", result["version_identity"]["revision_id"]
    result["version_chain"] = [dict(from_id=doc_id, to_id=revision, document=result["document"], quotes=result["quotes"]),
        dict(from_id=revision, to_id=item["content_sha256"],
             document=deepcopy(payload["raw_documents"][item["content_sha256"]]), quotes=["ifrs-full:Revenue"])]
    return result


def fixture():
    data = raw_fixture()
    item = add_raw(data, proof_present=False)
    return data, item


def audit(data, cutoff="2026-08-04T13:30:00+08:00"):
    return audit_filings(data, ("2330",), cutoff, parse_filing=parse_ixbrl)


def test_late_download_early_publication_preserves_all_original_times():
    data, item = fixture()
    original = deepcopy(data)
    assert audit(data)["rows"][0]["status"] == "DOCUMENT_PENDING_PUBLICATION_PROOF"
    fixed, _ = apply_reviews(data, [reviewed(data, item)])
    row = audit(fixed)["rows"][0]
    assert row["status"] == "VERIFIED_AVAILABLE" and row["selected"]
    assert row["effective_available_at"] == "2026-08-02T00:00:00+08:00"
    assert row["fetched_at"] == row["stored_available_at"] == item["receipt"]["fetched_at"]
    assert data == original
    assert fixed["raw_documents"] == data["raw_documents"]
    assert row["comparability"]["status"] == "FIELDS_MISSING_OR_INCOMPARABLE"
    assert audit(fixed)["status"] == "PARTIAL"  # Proof != financial/model completeness.


@pytest.mark.parametrize("field,value", [("symbol","3014"), ("quarter",3), ("fiscal_year",2025),
    ("statement_scope","individual"), ("content_sha256","a"*64), ("report_id","A"), ("source_version","made-up")])
def test_wrong_exact_binding_rejected(field, value):
    data, item = fixture()
    p = reviewed(data, item); p["binding"][field] = value
    with pytest.raises(ValueError, match="EXACT_EXISTING"):
        apply_reviews(data, [p])


@pytest.mark.parametrize("damage", ["broken_chain", "content", "review", "revision", "source_quote", "review_time"])
def test_unsupported_version_chain_cannot_backdate(damage):
    data, item = fixture()
    p = reviewed(data, item)
    if damage == "broken_chain": p["version_chain"][1]["from_id"] = "wrong"
    if damage == "content": p["version_chain"][-1]["to_id"] = "a"*64
    if damage == "review": p["review_checks"]["revision_history"] = False
    if damage == "revision": p["version_identity"]["revision_id"] = "wrong"
    if damage == "source_quote": p["version_chain"][0]["quotes"] = ["not in source"]
    if damage == "review_time": p["verified_at"] = "2026-09-01T10:00:00+08:00"
    with pytest.raises(ValueError): apply_reviews(data, [p])


def test_dates_timezone_cutoff_and_no_proof_never_backfill():
    data, item = fixture()
    item["publication_proof"] = reviewed(data, item, "2026-08-04")
    assert audit(data)["rows"][0]["status"] == "PUBLISHED_AFTER_CUTOFF"
    assert audit(data,"2026-08-05T00:00:00+08:00")["rows"][0]["eligible_at_cutoff"]
    item["publication_proof"] = reviewed(data, item, "2026-08-04T05:30:00Z", "TIMESTAMP")
    assert audit(data)["rows"][0]["eligible_at_cutoff"]
    assert not audit(data,"2026-08-04T13:29:59.999999+08:00")["rows"][0]["eligible_at_cutoff"]
    with pytest.raises(ValueError, match="TIMEZONE"): audit(data,"2026-08-04T13:30:00")
    item.pop("publication_proof")
    assert not audit(data)["rows"][0]["eligible_at_cutoff"]


def revised_history():
    data = history()
    first, second = data["filings"]["2330"]
    first["publication_proof"] = reviewed(data, first)
    p = reviewed(data, second,"2026-09-11T13:30:00+08:00","TIMESTAMP")
    p["version_identity"].update(kind="REVISION", supersedes_sha256=first["content_sha256"])
    p.update(predecessor=deepcopy(first["publication_proof"]), original_published_at="2026-08-01")
    second["publication_proof"] = p
    return data


def test_revision_chronology_and_old_content_selection():
    data = revised_history()
    first, second = data["filings"]["2330"]
    assert audit(data,"2026-09-10T13:30:00+08:00")["rows"][0]["selected"]
    assert audit(data,"2026-09-11T13:30:00+08:00")["rows"][1]["selected"]
    second["publication_proof"]["published_at"] = "2026-08-01T00:00:00+08:00"
    rows, errors = materialize_filings(data, parse_filing=parse_ixbrl)
    assert len(rows["2330"]) == 1
    assert errors["2330"][0]["reason"] == "REVISION_PUBLICATION_ORDER_UNVERIFIED"
    second["publication_proof"]["version_identity"]["kind"] = "ORIGINAL"
    with pytest.raises(ValueError,match="BORROW"): verify_filing_publication("2330",second)


def test_conflicting_proofs_stay_unknown_not_choose_earliest():
    data, item = fixture()
    first, _ = apply_reviews(data, [reviewed(data,item)])
    conflict, _ = apply_reviews(first, [reviewed(data,item,"2026-08-03")])
    row = audit(conflict)["rows"][0]
    assert row["status"] == "VERSION_CONFLICT" and not row["eligible_at_cutoff"]
    assert materialize_filings(conflict,parse_filing=parse_ixbrl)[1]["2330"]


def test_conflict_is_research_blocker_independent_of_native_hard_gate_scores(tmp_path, monkeypatch):
    data,item=fixture()
    fixed,_=apply_reviews(data,[reviewed(data,item)])
    conflict,_=apply_reviews(fixed,[reviewed(data,item,'2026-08-03')])
    path=create_dataset(conflict,tmp_path/'datasets')
    req=replace(request(tmp_path,data),versions=current_versions(),evidence_id=path.parent.name)
    engine,_=build_replay_engine(path)
    # Force native gate diagnostics to PASS: publication integrity still cannot
    # be overridden. No actual scoring threshold or model formula is changed.
    monkeypatch.setattr('application.replay_service.entry_gate_checks',lambda _: {'quality':'PASS'})
    result=engine.run(req)
    decision=result['decisions'][0]
    assert decision['hard_gates']['quality']=='PASS'
    assert 'PUBLICATION_VERSION_CONFLICT:UNKNOWN' in decision['blocking_reasons']
    assert not decision['research_entry_eligible']


def test_original_missing_parse_failure_and_units_are_separate():
    data,item = fixture()
    data["raw_documents"].clear()
    assert audit(data)["rows"][0]["status"] == "ORIGINAL_NOT_ACQUIRED"
    data,item = fixture()
    broken = audit_filings(data,("2330",),"2026-08-04T13:30:00Z",parse_filing=lambda _: [])
    assert broken["rows"][0]["status"] == "PARSE_FAILED"
    from dataclasses import replace as evolve
    def wrong_units(content): return [evolve(f,unit="iso4217:USD") for f in parse_ixbrl(content)]
    bad = audit_filings(data,("2330",),"2026-08-04T13:30:00Z",parse_filing=wrong_units)
    comp = bad["rows"][0]["comparability"]
    assert comp["status"] == "FIELDS_MISSING_OR_INCOMPARABLE" and "revenue" in comp["missing_fields"]
    assert comp["contexts"][0]["unit"] == "iso4217:USD"


def test_scope_period_basis_and_context_problems_do_not_become_comparable():
    import base64
    from domain.strategies.low_entry_v6_quality import normalize_quarters
    import pandas as pd
    data, item = fixture()
    old = deepcopy(item)
    old.update(quarter=1, period_end='2026-03-31', statement_scope='individual')
    body = base64.b64decode(data['raw_documents'][item['content_sha256']]['body_base64']).replace(b'2026-06-30',b'2026-03-31')
    sha = hashlib.sha256(body).hexdigest()
    old.update(content_sha256=sha,source_version='test-quarter-one')
    data['raw_documents'][sha] = dict(data['raw_documents'][item['content_sha256']],sha256=sha,body_base64=base64.b64encode(body).decode())
    data['filings']['2330'].append(old)
    report=audit(data,'2026-09-21T13:30:00+08:00')
    # The excluded older scope remains auditable but is no longer a blanket
    # dependency of current-original reported-period Quality scoring.
    assert 'INCOMPARABLE_FINANCIAL_STATEMENTS' in report['financials']['2330']['diagnostic_unresolved_fields']
    assert report['native_quality']['2330']['status'] != 'PASS'
    # Preserve the native refusal to combine QUARTER with YTD or subtract EPS.
    periods={pd.Period('2026Q1'):dict(flow_basis='QUARTER',revenue=5,basic_eps=1),
             pd.Period('2026Q2'):dict(flow_basis='YTD',revenue=12,basic_eps=2)}
    normalized=normalize_quarters(periods)
    assert normalized[pd.Period('2026Q2')]['revenue'] is None
    assert normalized[pd.Period('2026Q2')]['basic_eps'] is None


def test_future_revision_does_not_change_selected_native_inputs(tmp_path):
    data=revised_history()
    original=deepcopy(data); original['filings']['2330']=original['filings']['2330'][:1]
    from tests.test_research_archive import PinnedReplayEvidence
    req=request(tmp_path,data,day='2026-09-10')
    a=PinnedReplayEvidence(digest(data),data).select(req)
    b=PinnedReplayEvidence(digest(original),original).select(replace(req,evidence_id=digest(original)))
    import pandas as pd
    pd.testing.assert_frame_equal(a.market.annual_fundamentals['2330'],b.market.annual_fundamentals['2330'])
    assert a.market.fundamental_snapshots==b.market.fundamental_snapshots
    assert a.evidence['filings']['2330']==b.evidence['filings']['2330']


def test_proof_changes_invalidate_index_reuse_only_parses(tmp_path):
    data,item = fixture()
    calls=[]
    def parse(content): calls.append(True); return parse_ixbrl(content)
    cutoff=aware_timestamp("2026-08-04T13:30:00+08:00")
    def cached(d): return SQLiteFilingHistory(tmp_path/'cache.sqlite3',evidence_id=digest(d),version='test',parse_filing=parse)
    assert cached(data).load(data,('2330',),cutoff)[0]['2330'][0]['facts'] == []
    fixed,_=apply_reviews(data,[reviewed(data,item)])
    fresh=cached(fixed)
    assert fresh.load(fixed,('2330',),cutoff)[0]['2330'][0]['facts']
    assert len(calls)==1 and fresh.stats['sqlite_ixbrl_hit']==1


def test_publish_idempotent_original_unchanged_and_no_verified_repair_no_new_id(tmp_path, monkeypatch):
    data,item=fixture()
    source=create_dataset(data,tmp_path/'datasets')
    original=source.read_bytes()
    review=tmp_path/'review.json'
    review.write_text(json.dumps(dict(schema='filing-repair/v1',parent_dataset_id=source.parent.name,reviews=[])))
    kwargs=dict(as_of='2026-08-04T13:30:00+08:00',symbols=('2330',))
    assert publish_repair(source,review,tmp_path/'datasets',**kwargs)[0] is None
    review.write_text(json.dumps(dict(schema='filing-repair/v1',parent_dataset_id=source.parent.name,
                                      reviews=[reviewed(data,item)])))
    import infrastructure.publication_repair as module
    write=module.write_once
    def interrupt(path,value):
        if path.name=='publication-repair.json': raise RuntimeError('interrupted publication')
        return write(path,value)
    monkeypatch.setattr(module,'write_once',interrupt)
    with pytest.raises(RuntimeError): publish_repair(source,review,tmp_path/'datasets',**kwargs)
    monkeypatch.setattr(module,'write_once',write)
    path,report=publish_repair(source,review,tmp_path/'datasets',**kwargs)
    assert publish_repair(source,review,tmp_path/'datasets',**kwargs)[0]==path
    assert report['difference']['improved_count']==1
    assert json.loads((path.parent/'manifest.json').read_text())['parent_dataset_id']==source.parent.name
    assert source.read_bytes()==original and b'body_base64' not in path.read_bytes()
    assert main(['filing-audit','--evidence',str(path),'--as-of',kwargs['as_of'],'--output',str(tmp_path/'audit.json'),'2330'])==0
    assert json.loads((tmp_path/'audit.json').read_text())['status']=='PARTIAL'


def test_collection_resume_empty_and_failures_are_not_absence(tmp_path):
    data,_=fixture(); plan=collection_plan(data,audit(data))
    class Client:
        def __init__(self): self.calls=0
        def fetch_official_source(self,method,url,**kwargs):
            self.calls+=1
            return SimpleNamespace(content='查無資料'.encode(),status_code=200,attempts=1),url
    client=Client()
    result=collect_sources(plan,tmp_path,client=client,max_requests=1,now=lambda:'2026-09-29T10:00:00Z')
    assert client.calls==1 and result['results'][0]['status']=='NO_MATCH_UNPROVEN'
    result=collect_sources(plan,tmp_path,client=client,max_requests=100,now=lambda:'2026-09-29T10:00:00Z')
    assert client.calls==len(plan)
    assert collect_sources(plan,tmp_path,client=client,max_requests=100)==result
    assert client.calls==len(plan)
    class Offline:
        def fetch_official_source(self,*a,**k): raise NetworkError('offline')
    failed=collect_sources(plan,tmp_path/'offline',client=Offline(),max_requests=100)
    assert sum(r['status']=='NETWORK_ERROR' for r in failed['results'])==2
    assert all(r['status'] in {'NETWORK_ERROR','PENDING'} for r in failed['results'])


def test_recognized_empty_corrections_and_changed_original_stay_unproven():
    task=dict(kind='CORRECTIONS',symbols=['2330'])
    html='<table><tr><th>公司代號</th><th>公告日期</th></tr><tr><td>3014</td><td>20260929</td></tr></table>'.encode()
    assert classify_source(task,html,'https://mops.twse.com.tw/source')['status']=='NO_MATCH_UNPROVEN'
    assert classify_source(dict(kind='ORIGINAL',expected_sha256='a'*64),b'changed','https://mops.twse.com.tw/source')['status']=='ORIGINAL_VERSION_UNRECOVERABLE'


def test_official_ebook_source_is_captured_without_becoming_publication_proof(tmp_path):
    body = b'SYNTHETIC official ebook listing, no exact filing version binding'
    url = 'https://doc.twse.com.tw/server-java/t57sb01?co_id=3014&year=113'
    task = dict(kind='EBOOK_LISTING', symbols=['3014'], method='GET', url=url, data=None)
    client = SimpleNamespace(fetch_official_source=lambda *a, **k:
        (SimpleNamespace(content=body, status_code=200, attempts=1), url))
    result = collect_sources([dict(request_id=digest(task), **task)], tmp_path,
                             client=client)['results'][0]
    document = result['document']
    reader = EvidenceDocuments(tmp_path / 'objects').read
    assert verified_document(document, document_reader=reader) == body
    assert result['status'] == 'SOURCE_UNSUPPORTED'
    assert 'publication_proof' not in result
    for bad in ('http://doc.twse.com.tw/source', 'https://doc.twse.com.tw.example.com/source',
                'https://user@doc.twse.com.tw/source'):
        with pytest.raises(ValueError, match='OFFICIAL_EVIDENCE_SOURCE_REQUIRED'):
            verified_document(dict(document, url=bad), document_reader=reader)


def test_collection_keeps_actual_official_url_after_session_reset(tmp_path, monkeypatch):
    import infrastructure.filing_evidence_collection as module
    class Client:
        def __init__(self, http_client, **kwargs): self.http_client=http_client
        def fetch_official_source(self,method,url,**kwargs):
            kwargs['before_attempt']()
            self.http_client.reset_session()
            kwargs['before_attempt']()
            response=SimpleNamespace(url='https://mops.twse.com.tw/actual-final',is_redirect=False)
            for hook in self.http_client.session.hooks['response']: hook(response)
            return SimpleNamespace(content=b'original',status_code=200,attempts=2),url
    monkeypatch.setattr(module,'MopsClient',Client)
    task=dict(kind='ORIGINAL',symbols=['2330'],method='GET',data=None,url='https://mopsov.twse.com.tw/requested',
              expected_sha256=hashlib.sha256(b'original').hexdigest())
    result=collect_sources([dict(request_id=digest(task),**task)],tmp_path)
    assert result['results'][0]['document']['url']=='https://mops.twse.com.tw/actual-final'


def test_recovery_copy_does_not_overwrite_missing_old_dataset_object(tmp_path):
    data,item=fixture(); source=create_dataset(data,tmp_path/'source')
    sha=item['content_sha256']
    missing=tmp_path/'source'/'objects'/sha
    content=missing.read_bytes(); missing.unlink()
    task=dict(kind='ORIGINAL',symbols=['2330'],method='GET',url=item['source_url'],data=None,expected_sha256=sha)
    task=dict(request_id=digest(task),**task)
    client=SimpleNamespace(fetch_official_source=lambda *a,**k:(SimpleNamespace(content=content,status_code=200,attempts=1),item['source_url']))
    collected=collect_sources([task],tmp_path/'collection',client=client)
    receipt=tmp_path/'collection'/'reports'/f'{digest(collected)}.json'
    review=tmp_path/'review.json'
    review.write_text(json.dumps(dict(schema='filing-repair/v1',parent_dataset_id=source.parent.name,reviews=[])))
    path,report=publish_repair(source,review,tmp_path/'new',as_of='2026-08-04T13:30:00+08:00',symbols=('2330',),collection_file=receipt)
    assert path and report['recovered_documents']==1
    assert not missing.exists() and (tmp_path/'new'/'objects'/sha).read_bytes()==content
    assert report['after']['rows'][0]['status']=='DOCUMENT_PENDING_PUBLICATION_PROOF'


def test_native_repair_cached_parity_future_and_cutoff_scope(tmp_path):
    data,item=fixture()
    old=create_dataset(data,tmp_path/'datasets')
    fixed,_=apply_reviews(data,[reviewed(data,item,'2026-09-11T13:30:00+08:00','TIMESTAMP')])
    new=create_dataset(fixed,tmp_path/'datasets',parent=old)
    for day in ('2026-09-10','2026-09-11'):
        req=replace(request(tmp_path,data,day=day),evidence_id=new.parent.name,versions=current_versions())
        plain,_=build_replay_engine(new)
        cached,_=build_replay_engine(new,parse_cache_path=tmp_path/'cache.sqlite3')
        a,ca=plain.run_with_coverage(replace(req,state_path=tmp_path/f'plain-{day}.sqlite3'))
        b,cb=cached.run_with_coverage(replace(req,state_path=tmp_path/f'cached-{day}.sqlite3'))
        assert a==b and ca==cb
        before,_=build_replay_engine(old)
        previous=before.run(replace(req,evidence_id=old.parent.name,state_path=tmp_path/f'old-{day}.sqlite3'))
        if day=='2026-09-10':
            assert a['decisions']==previous['decisions']
            assert not a['evidence']['filings']['2330']
        else:
            assert a['evidence']['filings']['2330']
    req=replace(request(tmp_path,data),evidence_id=old.parent.name,versions=current_versions())
    compared=compare_datasets(old,new,[req.inputs()],tmp_path/'comparison',initial_state=True)
    assert len(compared['days'])==1 and compared['live_order_authorized'] is False
