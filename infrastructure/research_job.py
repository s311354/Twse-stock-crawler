"""Public offline research job: one pinned dataset, one isolated evolving state.

No model implementation is duplicated here. StrictReplayEngine remains the
only research engine. Legacy daily-research/v1 archives are left untouched.
"""
from dataclasses import asdict, dataclass, replace
import fcntl
import json
from pathlib import Path
import tempfile
import zipfile

from application.replay_cache import ReplayCache
from application.replay_contracts import digest
from application.replay_inputs import PinnedReplayEvidence
from application.replay_service import StrictReplayEngine
from application.research_audit import compare_results
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from infrastructure import replay_runtime as runtime
from infrastructure.daily_research import request_from_document
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_archive import write_once, utc_now
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.storage.research_state import IncrementalResearchState
from infrastructure.storage.research_cache import SQLiteFilingHistory


@dataclass(frozen=True)
class ResearchDay:
    result: dict
    coverage: dict
    receipt: dict
    state_id: str
    folder: Path


class ResearchJob:
    def __init__(self, evidence_file, root, *, cache_enabled=True, cache_bytes=32*1024*1024, object_root=None,
                 parse_cache_path=None):
        self.root, self.evidence_file = Path(root).resolve(), Path(evidence_file).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root/'.lock').open('a')
        self.closed = False
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.guard = ResearchVersionGuard()
            self.cache = ReplayCache(cache_bytes, enabled=cache_enabled)
            self.parse_cache = None
            def filing_index(eid):
                self.parse_cache = SQLiteFilingHistory(
                    parse_cache_path if parse_cache_path is not None else self.root/'parse-cache.sqlite3',
                    evidence_id=eid, version=digest(asdict(self.guard.versions)), parse_filing=runtime.parse_ixbrl,
                    document_reader=self.documents.read)
                return self.parse_cache
            def load_pinned(path):
                eid, payload = load_evidence(path)
                self.documents = EvidenceDocuments.for_evidence(path, eid)
                self.documents.verify(payload)
                return eid, payload
            self.evidence = PinnedReplayEvidence.from_file(self.evidence_file, loader=load_pinned,
                parse_filing=runtime.parse_ixbrl,
                annual_calendar=lambda data: runtime.TwseCalendar(fetcher=lambda: data),
                cache=self.cache, version=digest(asdict(self.guard.versions)),
                document_reader=lambda ref: self.documents.read(ref),
                document_check=lambda: self.documents.check(),
                filing_index_factory=filing_index if cache_enabled else None)
            if self.evidence._payload.get('schema') != RAW_EVIDENCE_SCHEMA:
                raise ValueError('DAILY_RETENTION_REQUIRES_RAW_V2_DATASET')
            self.evidence_stamp = self._evidence_stamp()
            self.objects = ResearchObjects(Path(object_root).resolve() if object_root else self.root/'objects')
            self.store = IncrementalResearchState(result_store=self.objects)
            self.engine = StrictReplayEngine(self.evidence, self.store, self._versions)
            self.assets = self.guard.retain(self.objects)
            self._recovered = False
            self.head = None
            self.latest = None
            self.request_template = None
        except BaseException:
            self.close()
            raise

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def close(self):
        if not self.closed:
            if hasattr(self, 'evidence'): self.evidence.close()
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self.closed = True

    @property
    def cache_stats(self):
        return dict(self.cache.stats, **(self.parse_cache.stats if self.parse_cache is not None else {}))

    def _evidence_stamp(self):
        stat = self.evidence_file.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _versions(self):
        if self._evidence_stamp() != self.evidence_stamp:
            raise ValueError('RESEARCH_DATASET_CHANGED')
        self.documents.check()
        return self.guard.check()

    def _read(self, folder):
        receipt = json.loads((folder/'complete.json').read_text())
        if receipt.get('schema') != 'daily-research/v2' or receipt['dataset_id'] != self.evidence.evidence_id:
            raise ValueError('DAILY_DATASET_MISMATCH')
        if any(receipt['objects'].get(k) != v for k,v in self.assets.items()):
            raise ValueError('DAILY_VERSION_OBJECT_MISMATCH')
        for key, sha in receipt['objects'].items():
            self.objects.get(sha)
        delta = json.loads((folder/'delta.json').read_text())
        if digest(delta) != receipt['objects']['delta'] or delta['state_id'] != receipt['poststate_id']:
            raise ValueError('DAILY_STATE_RECEIPT_MISMATCH')
        inputs = json.loads((folder/'request.json').read_text())
        if digest(inputs) != receipt['request_sha256'] or receipt['objects']['request'] != digest(inputs):
            raise ValueError('DAILY_REQUEST_HASH_MISMATCH')
        result = self.objects.get_json(receipt['objects']['result'])
        coverage = self.objects.get_json(receipt['objects']['coverage'])
        if (digest(result) != receipt['result_sha256'] or delta['session'][2] != digest(result) or
                delta['session'][1] != digest(inputs) or coverage['dataset_id'] != inputs['evidence_id'] or
                coverage['as_of'] != inputs['as_of']):
            raise ValueError('DAILY_RESEARCH_LINK_MISMATCH')
        return ResearchDay(result, coverage, receipt, receipt['poststate_id'], folder)

    def _recover(self, request):
        base = self.store.base(request)
        if (self.root/'base.json').exists():
            base = json.loads((self.root/'base.json').read_text())
            body = {k:v for k,v in base.items() if k != 'state_id'}
            if (base.get('schema') != 'research-state-base/v2' or digest(body) != base.get('state_id') or
                base.get('identity') != request.identity() or base.get('replay_id') != request.replay_id):
                raise ValueError('REPLAY_PREDECESSOR_IDENTITY_MISMATCH')
        write_once(self.root/'base.json', base)
        manifest = dict(schema='research-job/v2', identity=request.identity(), replay_id=request.replay_id,
                        base_state_id=base['state_id'], objects=self.assets)
        write_once(self.root/'job.json', manifest)
        sessions, observations = base.get('sessions', []), base.get('observations', [])
        for row in sessions:
            if digest(self.store._result(row[3])) != row[2]: raise ValueError('RESEARCH_STATE_RESULT_HASH_MISMATCH')
        head, last = base['state_id'], sessions[-1][0] if sessions else None
        self.pending = None
        for folder in sorted((self.root/'days').glob('*')):
            if self.pending is not None:
                raise ValueError('RESEARCH_INCOMPLETE_CHAIN')
            inputs = json.loads((folder/'request.json').read_text())
            req = request_from_document(inputs, request.state_path)
            if folder.name != str(req.history_end): raise ValueError('RESEARCH_DAY_PATH_MISMATCH')
            if req.identity() != request.identity() or req.replay_id != request.replay_id:
                raise ValueError('REPLAY_PREDECESSOR_IDENTITY_MISMATCH')
            if (folder/'complete.json').exists() and not (folder/'delta.json').exists():
                raise ValueError('DAILY_COMPLETION_WITHOUT_STATE')
            if (folder/'delta.json').exists():
                delta = json.loads((folder/'delta.json').read_text())
                result = self.store.validate_delta(request, delta, head, last)
                if delta['session'][1] != digest(inputs) or result['pae_state']['previous_result_sha256'] != (sessions[-1][2] if sessions else None):
                    raise ValueError('DAILY_REQUEST_OR_PREDECESSOR_MISMATCH')
                sessions.append(delta['session']); observations.extend(delta['observations'])
                head, last = delta['state_id'], delta['session'][0]
                if (folder/'complete.json').exists(): self._read(folder)
            elif last and inputs['as_of'] <= last:
                raise ValueError('RESEARCH_INCOMPLETE_CHAIN')
            if not (folder/'complete.json').exists(): self.pending = inputs['as_of']
        stored, obs = self.store.read_rows(request)
        if not request.state_path.exists():
            self.store.restore_rows(request, sessions, observations)
        elif stored != sessions or obs != observations:
            # Exactly one native commit may precede delta publication. Verify
            # its immutable intent and result; do not synthesize completion.
            if stored[:len(sessions)] != sessions or len(stored) != len(sessions)+1:
                raise ValueError('RESEARCH_STATE_INDEX_MISMATCH')
            pending = stored[-1]
            folder = self.root/'days'/pending[0][:10]
            intent = json.loads((folder/'intent.json').read_text())
            if intent['predecessor_state_id'] != head or intent['request_sha256'] != pending[1]:
                raise ValueError('RESEARCH_PENDING_COMMIT_MISMATCH')
            pending_req = request_from_document(json.loads((folder/'request.json').read_text()), request.state_path)
            delta = self.store.delta(pending_req, head)
            result = self.store.validate_delta(request, delta, head, last)
            if result['pae_state']['previous_result_sha256'] != (sessions[-1][2] if sessions else None):
                raise ValueError('RESEARCH_PENDING_PREDECESSOR_MISMATCH')
            if obs != observations + delta['observations']:
                raise ValueError('RESEARCH_STATE_INDEX_MISMATCH')
        self.head, self.latest = head, last
        self._recovered = True

    def run_day(self, request, *, predecessor_state_id=None):
        if self.closed: raise ValueError('RESEARCH_JOB_CLOSED')
        self._versions()
        if request.evidence_id != self.evidence.evidence_id or request.versions != self.guard.versions:
            raise ValueError('REPLAY_MODEL_OR_EVIDENCE_MISMATCH')
        local = replace(request, state_path=self.root/'state.sqlite3')
        if not self._recovered: self._recover(local)
        if self.request_template is not None and (local.identity() != self.request_template.identity() or local.replay_id != self.request_template.replay_id):
            raise ValueError('REPLAY_PREDECESSOR_IDENTITY_MISMATCH')
        self.request_template = local
        day, cutoff = str(local.history_end), local.as_of.isoformat()
        if self.pending is not None and self.pending != cutoff:
            raise ValueError('RESEARCH_PENDING_DAY_MUST_BE_RECOVERED_FIRST')
        if self.latest and cutoff < self.latest: raise ValueError('REPLAY_OUT_OF_ORDER')
        folder = self.root/'days'/day
        if (folder/'complete.json').exists():
            value = self._read(folder)
            if value.receipt['request_sha256'] != digest(local.inputs()):
                raise ValueError('REPLAY_PREDECESSOR_INPUT_CHANGED')
            if predecessor_state_id is not None and predecessor_state_id != value.receipt['predecessor_state_id']:
                raise ValueError('REPLAY_PREDECESSOR_STATE_CHANGED')
            self.head, self.latest, self.pending = value.state_id, cutoff, None
            return value  # Intent/account already validated in original execution.
        intent_file = folder/'intent.json'
        if intent_file.exists():
            intent = json.loads(intent_file.read_text())
            previous = intent['predecessor_state_id']
            if intent['request_sha256'] != digest(local.inputs()): raise ValueError('REPLAY_PREDECESSOR_INPUT_CHANGED')
        else:
            previous = self.head
            intent = dict(schema='research-intent/v2', request_sha256=digest(local.inputs()),
                          predecessor_state_id=previous, started_at=utc_now())
        if predecessor_state_id is not None and predecessor_state_id != previous:
            raise ValueError('REPLAY_PREDECESSOR_STATE_CHANGED')
        write_once(folder/'request.json', local.inputs())
        write_once(intent_file, intent)
        self.pending = cutoff
        result, coverage = self.engine.run_with_coverage(local)
        delta = self.store.delta(local, previous)
        # Publish delta then receipt. Native commit is recovered by _recover if
        # interruption occurs before either file. No repeated observations.
        write_once(folder/'delta.json', delta)
        refs = dict(self.assets, request=self.objects.put_json(local.inputs()),
                    result=delta['session'][2], coverage=self.objects.put_json(coverage),
                    delta=self.objects.put_json(delta))
        self._versions()
        receipt = dict(schema='daily-research/v2', dataset_id=local.evidence_id,
            research_run_id=digest(dict(request=local.inputs(), predecessor_state_id=previous)),
            request_sha256=digest(local.inputs()), result_sha256=digest(result),
            predecessor_state_id=previous, poststate_id=delta['state_id'], objects=refs,
            completed_at=utc_now(), status='COMPLETED', coverage_status=coverage['status'], live_order_authorized=False)
        write_once(folder/'complete.json', receipt)
        self.head, self.latest = delta['state_id'], cutoff
        self.pending = None
        return ResearchDay(result, coverage, receipt, self.head, folder)

    def export_snapshot(self):
        """Explicit expensive v1 logical export; NOT used by daily execution."""
        if self.request_template is None: raise ValueError('RESEARCH_REQUEST_REQUIRED')
        return self.store.export_snapshot(self.request_template)

    def initialize_from_snapshot(self, request, snapshot):
        """Import an explicit v1 predecessor once, without rebasing its identity.

        The source snapshot is never changed. Future daily state uses references.
        """
        from infrastructure.storage.replay_state import SQLiteReplayState
        from application.replay_contracts import canonical_json
        if self._recovered or (self.root/'job.json').exists():
            raise ValueError('RESTORE_REQUIRES_EMPTY_RESEARCH_DESTINATION')
        self._versions()
        if request.versions != self.guard.versions or request.evidence_id != self.evidence.evidence_id:
            raise ValueError('REPLAY_MODEL_OR_EVIDENCE_MISMATCH')
        with tempfile.TemporaryDirectory(prefix='twse-state-import-') as temporary:
            target = replace(request, state_path=Path(temporary)/'validate.sqlite3')
            SQLiteReplayState().restore_snapshot(target, snapshot)
        sessions = []
        for row in snapshot['sessions']:
            ref = self.objects.put_json(json.loads(row[3]))
            sessions.append([*row[:3], canonical_json(dict(schema='replay-result-ref/v1', sha256=ref))])
        body = dict(schema='research-state-base/v2', identity=request.identity(), replay_id=request.replay_id,
                    imported_snapshot_id=snapshot['state_id'], sessions=sessions, observations=snapshot['observations'])
        write_once(self.root/'base.json', dict(body, state_id=digest(body)))

    def _verify_archive(self):
        folders = sorted((self.root/'days').glob('*'))
        if not folders: raise ValueError('NO_COMPLETED_RESEARCH_DAYS')
        if not self._recovered:
            req = request_from_document(json.loads((folders[0]/'request.json').read_text()), self.root/'state.sqlite3')
            self._recover(req)
            self.request_template = req
        if self.pending is not None: raise ValueError('RESEARCH_INCOMPLETE_CHAIN')

    def reproduce(self):
        """Independent native rerun of all completed days, not saved-result comparison."""
        self._verify_archive()
        with tempfile.TemporaryDirectory(prefix='twse-job-reproduce-') as directory:
            checks = []
            with ResearchJob(self.evidence_file, Path(directory), cache_enabled=self.cache.enabled) as other:
                base = json.loads((self.root/'base.json').read_text())
                for row in base.get('sessions', []):
                    sha = json.loads(row[3])['sha256']
                    other.objects.put(self.objects.get(sha))
                write_once(other.root/'base.json', base)
                for folder in sorted((self.root/'days').glob('*')):
                    expected = self._read(folder)
                    req = request_from_document(json.loads((folder/'request.json').read_text()), Path(directory)/'unused')
                    actual = other.run_day(req)
                    checks.append(dict(date=str(req.history_end), result=compare_results(expected.result, actual.result),
                                       coverage=compare_results(expected.coverage, actual.coverage),
                                       state_equal=expected.state_id == actual.state_id))
            return dict(reexecuted=True, days=checks, equal=all(
                c['result']['equal'] and c['coverage']['equal'] and c['state_equal'] for c in checks))

    def export_bundle(self, destination):
        """Portable checkpoint: immutable chain + reachable objects + dataset.

        SQLite is deliberately excluded. Opening an extracted bundle rebuilds
        the index once from its verified chain, not a copy of an open database.
        """
        self._versions()
        self._verify_archive()
        refs = set(self.assets.values())
        files = [self.root/'job.json', self.root/'base.json']
        base = json.loads((self.root/'base.json').read_text())
        refs.update(json.loads(row[3])['sha256'] for row in base.get('sessions', []))
        for folder in sorted((self.root/'days').glob('*')):
            value = self._read(folder)  # Incomplete days cannot be exported as complete.
            refs.update(value.receipt['objects'].values())
            files.extend(folder/name for name in ('request.json', 'intent.json', 'delta.json', 'complete.json'))
        # Explicit export may revalidate the source file, unlike each daily run.
        eid, payload = load_evidence(self.evidence_file)
        if eid != self.evidence.evidence_id: raise ValueError('RESEARCH_DATASET_CHANGED')
        self.documents.verify(payload)
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.evidence_file, 'evidence.json')
            for file in files: archive.write(file, str(file.relative_to(self.root)))
            exported = set()
            for name, content in self.objects.export_entries(refs):
                archive.writestr('objects/'+name, content)
                exported.add(name)
            # Logical JSON IDs may coincide with original document bytes.
            # A graph root alone cannot replace a raw document path/stamp.
            for sha in sorted(self.documents.references.keys() - exported):
                archive.writestr('objects/'+sha, self.documents.read(self.documents.references[sha]))
            for file in (self.root/'source-index').glob('*.json'):
                archive.write(file, str(file.relative_to(self.root)))
