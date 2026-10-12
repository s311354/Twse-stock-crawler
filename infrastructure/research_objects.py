"""Content-addressed immutable archive objects and job-scoped version guard."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile
from importlib.metadata import distributions

from application.replay_contracts import canonical_json, digest
from infrastructure import replay_runtime as runtime
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter

GRAPH_STORE_NAME = "evidence.sqlite3"


class ResearchObjects:
    def __init__(self, root):
        self.root = Path(root)
        self.stats = dict(json_roots_written=0, json_roots_reused=0,
                          nodes_written=0, nodes_reused=0, node_bytes_written=0)

    def path(self, sha):
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('INVALID_RESEARCH_OBJECT_ID')
        return self.root / sha

    def put(self, content):
        sha = hashlib.sha256(content).hexdigest()
        if self.path(sha).exists():
            if self.path(sha).read_bytes() != content:
                raise ValueError(f'IMMUTABLE_ARTIFACT_CONFLICT:{sha}')
        else:
            write_bytes_once(self.path(sha), content)
        return sha

    def get(self, sha):
        path = self.path(sha)
        if path.exists():
            content = path.read_bytes()
        else:
            ref = self._reference(sha)
            # A whole-universe result is not a single daily report. Preserve
            # legacy result sizes; encoded-node and cache bounds still apply.
            with EvidenceGraphReader(self.root, max_value_bytes=None) as reader:
                content = canonical_json(reader.resolve(ref)).encode()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError('RESEARCH_OBJECT_HASH_MISMATCH')
        return content

    def _index_path(self, sha):
        self.path(sha)
        return self.root / 'json-index' / (sha + '.json')

    def _reference(self, sha):
        ref = json.loads(self._index_path(sha).read_bytes())
        if (not isinstance(ref, dict) or ref.get('value_sha256') != sha
                or ref.get('store') != GRAPH_STORE_NAME):
            raise ValueError('RESEARCH_OBJECT_HASH_MISMATCH')
        return ref

    def put_json(self, value):
        sha = digest(value)
        if self.path(sha).exists() or self._index_path(sha).exists():
            if self.path(sha).exists():
                self.put(canonical_json(value).encode())
            else:
                self.get(sha)
            self.stats['json_roots_reused'] += 1
            return sha
        # Commit immutable nodes before publishing a durable root reference.
        # A crash may leave unreferenced nodes, never a half-published root.
        try:
            with EvidenceGraphWriter(self.root / GRAPH_STORE_NAME) as writer:
                reference = writer.retain(value)
        except ValueError as error:
            if str(error) not in {'INVALID_EVIDENCE_KEY', 'EVIDENCE_NODE_TOO_LARGE', 'EVIDENCE_TOO_DEEP'}:
                raise
            # Retain the old byte format for JSON outside the bounded graph
            # shape, including numeric-key ordering. Never hide corruption.
            return self.put(canonical_json(value).encode())
        write_once(self._index_path(sha), reference)
        self.stats['json_roots_written'] += 1
        for key, count in writer.stats.items():
            self.stats[key] += count
        return sha

    def get_json(self, sha):
        return json.loads(self.get(sha))

    def export_entries(self, references):
        """Portable, reachable graph only; no open SQLite database is copied."""
        emitted = set()
        with EvidenceGraphReader(self.root, max_value_bytes=None) as reader:
            for sha in sorted(set(references)):
                content = self.get(sha)  # Verify logical identity before export.
                if self.path(sha).exists():
                    yield sha, content
                    continue
                ref = self._reference(sha)
                yield 'json-index/' + sha + '.json', canonical_json(ref).encode()
                for node_id, encoded in reader.reachable_nodes(ref['root'], ref['store']):
                    if node_id not in emitted:
                        emitted.add(node_id)
                        yield GRAPH_STORE_NAME + '.nodes/' + node_id, encoded


def source_paths():
    paths = [runtime.ROOT / 'main.py', runtime.ROOT / 'requirements.txt']
    for folder in ('application', 'domain', 'infrastructure', 'interface', 'config', 'twse'):
        paths.extend(sorted((runtime.ROOT / folder).rglob('*.py')))
    return [p for p in paths if p.is_file()]


def stamps(paths):
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths)


def source_zip():
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            entry = zipfile.ZipInfo(str(path.relative_to(runtime.ROOT)))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    return output.getvalue()


class ResearchVersionGuard:
    """Pin full content at open; cheap change detection before/after each day.

    Dirty source is included, not Git HEAD. A changed file/stat, added source,
    package metadata or effective parameter rejects the job, even if restored
    to the same version string. New contexts rehash all content.
    """
    def __init__(self):
        self.versions = runtime.current_versions()
        self.parameters = runtime.parameter_manifest()
        self.source_stamps = stamps(source_paths())
        deps = list(distributions())
        self.environment = dict(python=platform.python_version(), platform=platform.platform(),
            dependencies=sorted({(d.metadata['Name'], d.version) for d in deps if d.metadata['Name']}))
        self.metadata_paths = sorted({Path(d._path) for d in deps if hasattr(d, '_path') and Path(d._path).exists()})
        self.metadata_files = [p for d in self.metadata_paths for p in d.rglob('*') if p.is_file()]
        self.environment_stamps = stamps(sorted(set(self.metadata_paths + self.metadata_files +
            [p.parent for p in self.metadata_paths])))

    def check(self):
        if (stamps(source_paths()) != self.source_stamps or
            digest(runtime.parameter_manifest()) != self.versions.parameters or
            stamps(sorted(set(self.metadata_paths + self.metadata_files + [p.parent for p in self.metadata_paths])))
                != self.environment_stamps):
            raise ValueError('REPLAY_VERSION_CHANGED_DURING_JOB')
        return self.versions

    def retain(self, objects):
        # A content index allows subsequent jobs on the same object store to
        # verify and reuse the original compressed source instead of zipping it.
        key = digest(dict(source_files=[
            [str(p.relative_to(runtime.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in source_paths()]))
        index = objects.root.parent / 'source-index' / (key + '.json')
        if index.exists():
            ref = json.loads(index.read_text())['source']
            objects.get(ref)
        else:
            ref = objects.put(source_zip())
            write_bytes_once(index, canonical_json(dict(source=ref)).encode())
        result = dict(source=ref, environment=objects.put_json(self.environment),
                      parameters=objects.put_json(self.parameters))
        self.check()
        return result
