"""Synthetic PDF/fact mappings, never historical proof or production evidence."""
from copy import deepcopy
import hashlib
import json
import shutil

import pytest

from application.filing_pdf_comparison import COMPARISON_VERSION, compare_pdf_facts
from application.pdf_upload_policy import apply_pdf_upload_policy
from application.replay_contracts import digest
from infrastructure.filing_pdf_comparison import compare_filing_pdf, extract_pdf_pages
from infrastructure.research_archive import create_dataset
from infrastructure.research_documents import EvidenceDocuments
from interface.research import main
from tests.test_pdf_upload_policy import fixture


def example():
    fact = dict(concept='ifrs-full:Revenue', context_id='YTD', numeric_value=1_234_000.,
                text_value=None, unit='iso4217:TWD', start_date='2026-01-01',
                end_date='2026-06-30', instant=None, dimensions=[])
    page = ('SYNTHETIC 2330 合併財報\n2026/01/01-2026/06/30\n新臺幣仟元\n'
            '4000 營 業 收 入  $ 1,234  100  900  100\n')
    context = {k: fact[k] for k in ('unit', 'start_date', 'end_date', 'instant', 'dimensions')}
    mapping = dict(schema=COMPARISON_VERSION, reviewed_by='SYNTHETIC_TEST', statement_scope='consolidated',
        identity_evidence={key: dict(page=1, quote=quote) for key, quote in
                           [('company', '2330'), ('scope', '合併'), ('period', '2026/06/30')]},
        fields=[dict(fact_id=digest(fact), context=context, page=1, row_label='4000營業收入',
                     number_index=0, scale=1000,
                     period_evidence=dict(page=1, quote='2026/01/01-2026/06/30'),
                     unit_evidence=dict(page=1, quote='新臺幣仟元'))])
    return fact, page, mapping


def compare(fact, page, mapping):
    return compare_pdf_facts([fact], (page,), mapping, statement_scope='consolidated')


def test_exact_comparison_retains_fact_context_and_source_row():
    fact, page, mapping = example()
    original = deepcopy(mapping)
    report = compare(fact, page, mapping)
    assert report['status'] == 'ALL_MAPPED_VALUES_MATCH'
    assert report['rows'][0]['delta'] == '0.0'
    assert report['rows'][0]['pdf_row'].startswith('4000')
    assert not report['publication_version_verified']
    assert compare(fact, page, mapping) == report and mapping == original


@pytest.mark.parametrize('value,expected', [('1,235', 'MISMATCH'), ('(1,234)', 'MISMATCH'),
                                         ('--', 'UNRESOLVED'), ('NaN', 'UNRESOLVED')])
def test_numeric_difference_missing_nonfinite_and_parentheses(value, expected):
    fact, page, mapping = example()
    result = compare(fact, page.replace('1,234', value), mapping)
    assert result['rows'][0]['status'] == expected
    if value == '(1,234)': assert result['rows'][0]['pdf_value'] == '-1234000'


@pytest.mark.parametrize('key,value', [('unit', 'iso4217:USD'), ('start_date', '2026-04-01'),
    ('end_date', '2025-06-30'), ('instant', '2026-06-30'), ('dimensions', [['segment', 'x']])])
def test_single_quarter_ytd_units_dates_and_dimensions_never_mixed(key, value):
    fact, page, mapping = example()
    mapping['fields'][0]['context'][key] = value
    assert compare(fact, page, mapping)['rows'][0]['reason'] == 'PDF_IXBRL_CONTEXT_MISMATCH'


def test_scope_duplicate_rows_unmapped_and_text_cannot_be_automatic_pass():
    fact, page, mapping = example()
    mapping['statement_scope'] = 'individual'
    with pytest.raises(ValueError, match='SCOPE_MISMATCH'): compare(fact, page, mapping)
    mapping['statement_scope'] = 'consolidated'
    assert compare(fact, page + page, mapping)['rows'][0]['reason'] == 'PDF_ROW_MISSING_OR_AMBIGUOUS'
    mapping['fields'] = []
    assert compare(fact, page, mapping)['counts'] == {'NOT_MAPPED': 1}
    fact, page, mapping = example()
    fact.update(numeric_value=None, text_value='auditor opinion')
    mapping['fields'][0]['fact_id'] = digest(fact)
    assert compare(fact, page, mapping)['counts'] == {'TEXT_REVIEW_REQUIRED': 1}


def test_bad_column_quote_scale_and_mapping_do_not_match():
    fact, page, mapping = example()
    for key, value in [('number_index', -1), ('scale', 'Infinity'),
                       ('period_evidence', {'page': 1, 'quote': 'not in PDF'}), ('page', 999)]:
        bad = deepcopy(mapping)
        bad['fields'][0][key] = value
        assert compare(fact, page, bad)['rows'][0]['status'] == 'UNRESOLVED'
    mapping['fields'] *= 2
    with pytest.raises(ValueError, match='DUPLICATE'): compare(fact, page, mapping)


def test_full_text_comparison_normalizes_cjk_but_partial_quote_never_passes():
    fact, page, mapping = example()
    fact.update(numeric_value=None, text_value='非重要\n子公司未經核閱。')
    mapping['fields'][0].update(fact_id=digest(fact), text_evidence=[{'page': 1, 'quote': '非重要 子公司未經核閱。'}])
    page += '非重要 子公司未經核閱。\n'
    assert compare(fact, page, mapping)['counts'] == {'MATCH': 1}
    mapping['fields'][0]['text_evidence'][0]['quote'] = '非重要 子公司'
    assert compare(fact, page, mapping)['counts'] == {'TEXT_REVIEW_REQUIRED': 1}


def synthetic_pdf(text):
    """Minimal ASCII-only synthetic PDF to exercise the real Poppler adapter."""
    stream = f'BT /F1 12 Tf 20 750 Td ({text}) Tj ET'.encode()
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    content, offsets = b'%PDF-1.4\n', [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'
    position = len(content)
    content += b'xref\n0 6\n0000000000 65535 f \n'
    content += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    return content + f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{position}\n%%EOF'.encode()


def test_missing_extractor_html_and_scanned_errors(monkeypatch):
    with pytest.raises(ValueError, match='NOT_HTML'): extract_pdf_pages(b'<html>blocked</html>')
    monkeypatch.setattr('infrastructure.filing_pdf_comparison.shutil.which', lambda name: None)
    with pytest.raises(ValueError, match='INSTALL_POPPLER'): extract_pdf_pages(b'%PDF-1.4')


@pytest.mark.skipif(shutil.which('pdftotext') is None, reason='optional system Poppler')
def test_real_pdf_parser_cli_offline_hashes_and_forged_extracted_text(tmp_path, monkeypatch):
    data, review = fixture()
    adopted, _ = apply_pdf_upload_policy(data, review)
    path = create_dataset(adopted, tmp_path/'datasets')
    original = path.read_bytes()
    content = synthetic_pdf('SYNTHETIC TEST ONLY 2330 revenue 5000')
    pages, version = extract_pdf_pages(content)
    assert '2330 revenue 5000' in pages[0] and 'pdftotext' in version
    pdf = dict(url='https://doc.twse.com.tw/pdf/202602_2330_AI1.pdf',
               sha256=hashlib.sha256(content).hexdigest(), retrieved_at='2026-09-30T08:00:00+08:00',
               body_ref=EvidenceDocuments(tmp_path/'objects').store(content), extracted_text='FORGED_TEXT')
    receipt = tmp_path/'pdf.json'
    receipt.write_text(json.dumps(pdf))
    sha = adopted['filings']['2330'][0]['content_sha256']
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **kw: pytest.fail('No network'))
    result = compare_filing_pdf(path, symbol='2330', filing_sha256=sha, pdf_document=receipt)
    assert result['status'] == 'PARTIAL'
    assert result['counts']['NOT_MAPPED'] > 0
    assert all('FORGED_TEXT' not in p['text'] for p in result['pages'])
    assert path.read_bytes() == original and not result['publication_version_verified']
    args = ['filing-pdf-compare', '--evidence', str(path), '--symbol', '2330', '--filing-sha256', sha,
            '--pdf-document', str(receipt), '--output', str(tmp_path/'comparison.json')]
    assert main(args) == 0
    assert json.loads((tmp_path/'comparison.json').read_text()) == result
    changed = json.loads(json.dumps(adopted))
    changed['raw_documents'][sha]['sha256'] = 'b' * 64
    monkeypatch.setattr('infrastructure.filing_pdf_comparison.open_dataset',
                        lambda path: ('synthetic-invalid', changed, EvidenceDocuments(tmp_path/'objects')))
    with pytest.raises(ValueError, match='FILING_ORIGINAL_HASH_MISMATCH'):
        compare_filing_pdf(path, symbol='2330', filing_sha256=sha, pdf_document=receipt)
    pdf['url'] = pdf['url'].replace('2330', '3014')
    receipt.write_text(json.dumps(pdf))
    with pytest.raises(ValueError, match='FILENAME'): compare_filing_pdf(path, symbol='2330', filing_sha256=sha, pdf_document=receipt)


def test_download_chain_matches_visible_name_and_exact_official_target():
    from infrastructure.filing_evidence_collection import pdf_download_links, classify_source
    url = 'https://doc.twse.com.tw/server-java/t57sb01'
    content = b'<a href="/pdf/202402_3014_AI1_20260930.pdf">202402_3014_AI1.pdf</a>'
    assert pdf_download_links(content, url) == [dict(url='https://doc.twse.com.tw/pdf/202402_3014_AI1_20260930.pdf',
                                                   filename='202402_3014_AI1.pdf')]
    assert pdf_download_links(content.replace(b'/pdf/', b'https://evil.example/'), url) == []
    assert classify_source({'kind': 'PDF_DOWNLOAD_PAGE'}, content, url)['status'] == 'SOURCE_CAPTURED_REVIEW_REQUIRED'
    assert classify_source({'kind': 'PDF_ORIGINAL'}, b'%PDF-1.4', url)['reason'] == 'PDF_CONTENT_NOT_HISTORICAL_VERSION_PROOF'
    assert classify_source({'kind': 'PDF_ORIGINAL'}, b'<html>not a PDF</html>', url)['status'] == 'SOURCE_UNSUPPORTED'


def test_bounded_fetch_preserves_link_receipts_and_is_resumable(tmp_path, monkeypatch):
    from infrastructure.filing_pdf_comparison import collect_filing_pdf
    from infrastructure.filing_evidence_collection import collect_sources as real_collect
    from types import SimpleNamespace
    data, review = fixture()
    adopted, _ = apply_pdf_upload_policy(data, review)
    path = create_dataset(adopted, tmp_path/'datasets')
    old = path.read_bytes()
    pdf = synthetic_pdf('SYNTHETIC TEST ONLY')
    calls = []
    def fetch(method, url, **kwargs):
        calls.append((method, url))
        content = (b'<a href="/pdf/202602_2330_AI1_temporary.pdf">202602_2330_AI1.pdf</a>'
                   if method == 'POST' else pdf)
        return SimpleNamespace(content=content, status_code=200, attempts=1), url
    client = SimpleNamespace(fetch_official_source=fetch)
    monkeypatch.setattr('infrastructure.filing_pdf_comparison.collect_sources',
        lambda plan, root, **kwargs: real_collect(plan, root, client=client, **kwargs))
    kwargs = dict(symbol='2330', filing_sha256=adopted['filings']['2330'][0]['content_sha256'], output=tmp_path/'capture')
    result = collect_filing_pdf(path, **kwargs)
    assert result['status'] == 'PDF_CAPTURED_NOT_VERSION_VERIFIED'
    assert len(calls) == 2 and collect_filing_pdf(path, **kwargs) == result and len(calls) == 2
    receipt = json.loads((tmp_path/'capture/pdf.json').read_text())
    assert receipt['download_page']['body_ref'] and receipt['body_ref']
    assert path.read_bytes() == old
    if shutil.which('pdftotext'):
        compared = compare_filing_pdf(path, symbol='2330', filing_sha256=kwargs['filing_sha256'],
                                      pdf_document=tmp_path/'capture/pdf.json')
        assert compared['status'] == 'PARTIAL'
