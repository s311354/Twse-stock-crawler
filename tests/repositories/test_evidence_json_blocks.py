"""JSON block codec preserves logical evidence and all decoding limits."""
import hashlib
import json
import sqlite3
import zlib
from copy import deepcopy

import pytest

from application.daily_performance import DailyPerformance
from application.replay_contracts import canonical_json, digest
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter, MAX_JSON_BLOCK_BYTES
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter


def encode(root, value, size):
    probe = DailyPerformance()
    with probe.activate(), EvidenceGraphWriter(root/'evidence.sqlite3', json_block_bytes=size) as writer:
        ref = writer.retain(value)
    with EvidenceGraphReader(root) as reader:
        decoded = reader.resolve(ref)
        portable = list(reader.reachable_nodes(ref['root'], ref['store']))
    return ref, decoded, portable, probe


def test_realistic_blocks_reduce_nodes_without_changing_evidence(tmp_path):
    value = dict(financial=[dict(period=str(i), fields=[dict(name=str(j), value=j,
        source='SYNTHETIC_ONLY-'+str(i)+'-'+str(j)+'x'*700) for j in range(10)]) for i in range(40)],
        values=[-0.0, 0.0, True, False, 1, 1.0, None, '中文', ('tuple',)])
    before = deepcopy(value)
    old, legacy, _, old_probe = encode(tmp_path/'old', value, 0)
    ref, actual, nodes, probe = encode(tmp_path/'new', value, 16*1024)
    assert old['value_sha256'] == ref['value_sha256'] == digest(value)
    assert canonical_json(actual) == canonical_json(legacy) == canonical_json(value)
    assert value == before and all(len(body) < 8*1024*1024 for _, body in nodes)
    assert probe.counters['evidence_nodes_written'] < old_probe.counters['evidence_nodes_written']/5
    assert probe.counters['evidence_json_blocks'] > 0


def test_old_and_packed_roots_share_store_and_detach_each_resolve(tmp_path):
    value = dict(symbol='TEST', signal_date='2026-10-01', model_version='7.0.0',
                 parameter_version='SYNTHETIC_ONLY', proof={'x': [1, -0.0, '資料'*400]})
    with ReportEvidenceWriter(tmp_path/'report_evidence.sqlite3') as writer:
        old = writer.retain(value)
    with ReportEvidenceWriter(tmp_path/'report_evidence.sqlite3', json_block_bytes=4096) as writer:
        new = writer.retain(value)
    assert old['report_sha256'] == new['report_sha256']
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(old) == reader.resolve(new) == value
        changed = reader.resolve(new); changed['proof']['x'][0] = 'changed'
        assert reader.resolve(new) == value


def test_mutating_writer_input_does_not_hide_behind_block_memo(tmp_path):
    value = dict(proof=['x'*900, {'source': 'SYNTHETIC_ONLY'}])
    with EvidenceGraphWriter(tmp_path/'evidence.sqlite3', json_block_bytes=4096) as writer:
        first = writer.retain(value)
        value['proof'][1]['source'] = 'mutated'
        second = writer.retain(value)
    assert first['value_sha256'] != second['value_sha256']
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(first)['proof'][1]['source'] == 'SYNTHETIC_ONLY'
        assert reader.resolve(second) == value


def test_budget_is_exactly_the_legacy_logical_expansion_budget(tmp_path):
    value = dict(x=[{'x': i} for i in range(300)])
    old, _, _, _ = encode(tmp_path/'old', value, 0)
    new, _, _, _ = encode(tmp_path/'new', value, 16384)
    def charge(root, ref):
        with EvidenceGraphReader(root) as reader:
            budget = [0]
            reader.check_store(ref['store'])
            reader._decode(ref['root'], ref['store'], budget)
            return budget[0]
    assert charge(tmp_path/'new', new) == charge(tmp_path/'old', old)
    size = charge(tmp_path/'new', new)
    for root, ref in [(tmp_path/'old', old), (tmp_path/'new', new)]:
        with EvidenceGraphReader(root, max_value_bytes=size-1) as reader:
            with pytest.raises(ValueError, match='EXPANSION_TOO_LARGE'):
                reader.resolve(ref)
        with EvidenceGraphReader(root, max_value_bytes=size) as reader:
            assert reader.resolve(ref) == value


@pytest.mark.parametrize('value', [{1: 'bad'}, {'nested': {2:'bad'}}, {'x': float('nan')}, {'x': float('inf')}])
def test_invalid_input_is_never_packed(tmp_path, value):
    with pytest.raises((ValueError, TypeError)), EvidenceGraphWriter(tmp_path/'evidence.sqlite3', json_block_bytes=4096) as writer:
        writer.retain(value)


def test_depth_limit_applies_across_graph_and_block_boundaries(tmp_path):
    value = {'x':'synthetic'}
    for _ in range(129): value = [value]
    with pytest.raises(ValueError, match='TOO_DEEP'), EvidenceGraphWriter(tmp_path/'evidence.sqlite3', json_block_bytes=4096) as writer:
        writer.retain(value)


@pytest.mark.parametrize('body', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '42', '[] trailing', '['])
def test_malformed_packed_payload_fails_closed(tmp_path, body):
    node = canonical_json(['json', body]).encode()
    sha = hashlib.sha256(node).hexdigest()
    with sqlite3.connect(tmp_path/'evidence.sqlite3') as conn:
        conn.execute('CREATE TABLE report_objects (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)')
        conn.execute('INSERT INTO report_objects VALUES (?,?)', (sha,zlib.compress(node)))
    ref = dict(schema='shared-evidence-reference/v1', store='evidence.sqlite3',root=['ref',sha],value_sha256=digest({}))
    with EvidenceGraphReader(tmp_path) as reader:
        with pytest.raises(ValueError, match='JSON_BLOCK'):
            reader.resolve(ref)


def test_warm_reader_rejects_tampering_and_export_treats_json_as_terminal(tmp_path):
    value = {'data':['ref', 'a'*64], 'proof':'x'*900}
    ref, _, nodes, _ = encode(tmp_path, value, 4096)
    assert len(nodes) == 1
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(ref) == value
        with sqlite3.connect(tmp_path/'evidence.sqlite3') as conn:
            conn.execute('UPDATE report_objects SET body=?', (b'corrupt',))
        with pytest.raises(ValueError, match='HASH_MISMATCH'):
            reader.resolve(ref)


@pytest.mark.parametrize('size', [-1, True, MAX_JSON_BLOCK_BYTES+1])
def test_block_capacity_is_bounded(tmp_path, size):
    with pytest.raises(ValueError, match='JSON_BLOCK_BYTES'):
        EvidenceGraphWriter(tmp_path/'evidence.sqlite3', json_block_bytes=size)


@pytest.mark.parametrize('field', ['key', 'value'])
def test_packed_payload_capacity_counts_utf8_bytes_not_codepoints(tmp_path, field):
    text = '字' * 90000
    value = {text: 0} if field == 'key' else {'x': text}
    payload = canonical_json(value)
    assert len(payload) < MAX_JSON_BLOCK_BYTES < len(payload.encode('utf8'))
    reference = dict(schema='shared-evidence-reference/v1', store='inline.sqlite3',
        root=['inline', ['json', payload]], value_sha256=digest(value))
    with EvidenceGraphReader(tmp_path) as reader:
        with pytest.raises(ValueError, match='JSON_BLOCK'):
            reader.resolve(reference)


def test_packed_payload_accepts_exact_byte_boundary_and_rejects_one_more(tmp_path):
    value = ['x' * (MAX_JSON_BLOCK_BYTES - 4)]
    payload = canonical_json(value)
    assert len(payload.encode()) == MAX_JSON_BLOCK_BYTES
    reference = dict(schema='shared-evidence-reference/v1', store='inline.sqlite3',
        root=['inline', ['json', payload]], value_sha256=digest(value))
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(reference) == value
        larger = [value[0] + 'x']
        reference = dict(reference, root=['inline', ['json', canonical_json(larger)]],
            value_sha256=digest(larger))
        with pytest.raises(ValueError, match='JSON_BLOCK'):
            reader.resolve(reference)


@pytest.mark.parametrize('outer,inner,allowed', [(64, 64, True), (65, 64, False)])
def test_decoder_combines_graph_and_packed_json_depth(tmp_path, outer, inner, allowed):
    value = 0
    for _ in range(inner):
        value = [value]
    descriptor = ['inline', ['json', canonical_json(value)]]
    for _ in range(outer):
        descriptor = ['inline', ['list', [descriptor]]]
        value = [value]
    reference = dict(schema='shared-evidence-reference/v1', store='inline.sqlite3',
        root=descriptor, value_sha256=digest(value))
    with EvidenceGraphReader(tmp_path) as reader:
        if allowed:
            assert canonical_json(reader.resolve(reference)) == canonical_json(value)
        else:
            with pytest.raises(ValueError, match='JSON_BLOCK'):
                reader.resolve(reference)


def test_packed_leaf_portable_export_restores_without_sqlite(tmp_path):
    value = {'data': ['ref', 'a' * 64], 'source': '資料' * 400}
    reference, _, nodes, _ = encode(tmp_path/'source', value, 4096)
    assert len(nodes) == 1
    portable = tmp_path/'portable'
    node_root = portable/(reference['store'] + '.nodes')
    node_root.mkdir(parents=True)
    for sha, encoded in nodes:
        (node_root/sha).write_bytes(encoded)
    assert not (portable/reference['store']).exists()
    with EvidenceGraphReader(portable) as reader:
        assert reader.resolve(reference) == value
        changed = reader.resolve(reference)
        changed['data'][0] = 'changed'
        assert reader.resolve(reference) == value
    # Portable bodies are still verified rather than accepted as cached bytes.
    (node_root/nodes[0][0]).write_bytes(b'corrupt')
    with EvidenceGraphReader(portable) as reader:
        with pytest.raises(ValueError, match='HASH_MISMATCH'):
            reader.resolve(reference)


def test_repeated_packed_reference_charges_each_logical_expansion(tmp_path):
    leaf = {'x': [0] * 800, 'text': '資料' * 150}
    reference, _, _, _ = encode(tmp_path/'packed', leaf, 4096)
    assert reference['root'][0] == 'ref'
    value = [leaf] * 10
    repeated = dict(reference, root=['inline', ['list', [reference['root']] * 10]],
        value_sha256=digest(value))
    old, _, _, _ = encode(tmp_path/'legacy', value, 0)
    with EvidenceGraphReader(tmp_path/'legacy') as reader:
        reader.check_store(old['store'])
        expected = [0]
        reader._decode(old['root'], old['store'], expected)
    with EvidenceGraphReader(tmp_path/'packed') as reader:
        reader.check_store(repeated['store'])
        actual = [0]
        assert reader._decode(repeated['root'], repeated['store'], actual) == value
    assert actual == expected
    with EvidenceGraphReader(tmp_path/'packed', max_value_bytes=expected[0]-1) as reader:
        with pytest.raises(ValueError, match='EXPANSION_TOO_LARGE'):
            reader.resolve(repeated)
    with EvidenceGraphReader(tmp_path/'packed', max_value_bytes=expected[0]) as reader:
        assert reader.resolve(repeated) == value
