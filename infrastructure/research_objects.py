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
from application.work_version import WorkVersionChanged
from infrastructure import replay_runtime as runtime
from infrastructure.runtime_environment import current_runtime_identity
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
    result = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            # A removed metadata file is version drift, not an unrelated
            # FileNotFoundError that hides the pinned-job failure contract.
            result.append((str(path), None, None, None))
        else:
            result.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(result)


def _distribution_bindings(deps):
    return tuple(sorted((d.metadata['Name'] or '', d.version,
                         str(d._path) if hasattr(d, '_path') else '') for d in deps))


def _parent_members(parents):
    members = []
    for parent in parents:
        try:
            members.extend(p for p in parent.iterdir() if p.name not in {'__pycache__', '.DS_Store'})
        except FileNotFoundError:
            continue
    return sorted(members)


def _path_changes(before, after):
    old = {s[0]: s[1:] for s in before if s[1] is not None}
    new = {s[0]: s[1:] for s in after if s[1] is not None}

    def labels(paths):
        result = []
        for path in sorted(paths):
            try:
                result.append(str(Path(path).relative_to(runtime.ROOT)))
            except ValueError:
                result.append(path)
        return {"count": len(result), "paths": result[:20]}

    return dict(added=labels(new.keys() - old.keys()),
                removed=labels(old.keys() - new.keys()),
                modified=labels(k for k in old.keys() & new.keys() if old[k] != new[k]))


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
    to the same version string. Parent-directory timestamps only trigger a
    distribution/member inventory check: a timestamp-only update or known
    non-runtime cache is not a package change. New importable files and path
    configuration still reject the job. New contexts rehash all source content.
    """
    def __init__(self):
        self.runtime_identity = current_runtime_identity()
        # Take the change-detection baseline before reading full source hashes,
        # not after: an edit during the initial scan must also be rejected.
        self.source_stamps = stamps(source_paths())
        self.versions = runtime.current_versions()
        self.parameters = runtime.parameter_manifest()
        deps = list(distributions())
        self.distribution_bindings = _distribution_bindings(deps)
        self.environment = dict(python=platform.python_version(), platform=platform.platform(),
            runtime=dict(self.runtime_identity),
            dependencies=sorted({(d.metadata['Name'], d.version) for d in deps if d.metadata['Name']}))
        self.metadata_paths = sorted({Path(d._path) for d in deps if hasattr(d, '_path') and Path(d._path).exists()})
        # Include nested directories so adding a new metadata member is caught
        # without reading every installed distribution on every checkpoint.
        self.metadata_files = [p for d in self.metadata_paths for p in d.rglob('*')
                               if p.is_file() or p.is_dir()]
        self.metadata_parents = sorted({p.parent for p in self.metadata_paths})
        self.parent_stamps = stamps(self.metadata_parents)
        self.parent_members = _parent_members(self.metadata_parents)
        # Import configuration and root-level modules may change without
        # changing a distribution version or its metadata directory.
        self.import_files = [p for p in self.parent_members if p.is_file()
                             and p.suffix in {'.pth', '.egg-link', '.py', '.pyc', '.so', '.pyd'}]
        self.environment_stamps = stamps(sorted(set(
            self.metadata_paths + self.metadata_files + self.import_files)))
        self.check()

    def check(self):
        changes = {}
        current_runtime = current_runtime_identity()
        if current_runtime != self.runtime_identity:
            changes['runtime'] = sorted(k for k in self.runtime_identity.keys() | current_runtime.keys()
                                        if self.runtime_identity.get(k) != current_runtime.get(k))
        source = stamps(source_paths())
        if source != self.source_stamps:
            changes['source'] = _path_changes(self.source_stamps, source)
        parameters = runtime.parameter_manifest()
        parameter_hash = digest(parameters)
        if parameter_hash != self.versions.parameters:
            changes['parameters'] = dict(expected=self.versions.parameters, actual=parameter_hash,
                keys=sorted(k for k in self.parameters.keys() | parameters.keys()
                            if self.parameters.get(k) != parameters.get(k)))
        environment = stamps(sorted(set(self.metadata_paths + self.metadata_files + self.import_files)))
        if environment != self.environment_stamps:
            changes['environment_metadata'] = _path_changes(self.environment_stamps, environment)
        parents = stamps(self.metadata_parents)
        if parents != self.parent_stamps:
            members = _parent_members(self.metadata_parents)
            if members != self.parent_members:
                changes['environment_inventory'] = _path_changes(
                    [(str(p), 0, 0, 0) for p in self.parent_members],
                    [(str(p), 0, 0, 0) for p in members])
            bindings = _distribution_bindings(list(distributions()))
            if bindings != self.distribution_bindings:
                old, new = set(self.distribution_bindings), set(bindings)
                changes['distributions'] = dict(removed=sorted({b[0] for b in old - new})[:20],
                                                added=sorted({b[0] for b in new - old})[:20])
            # Do not accept an unstable inventory observed during installation.
            if stamps(self.metadata_parents) != parents:
                changes['environment_scan'] = 'CHANGED_DURING_CHECK'
        if changes:
            raise WorkVersionChanged(changes)
        self.parent_stamps = parents
        return self.versions

    def retain(self, objects):
        # A content index allows subsequent jobs on the same object store to
        # verify and reuse the original compressed source instead of zipping it.
        paths = source_paths()
        manifest = [[str(p.relative_to(runtime.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()]
                    for p in paths]
        key = digest(dict(source_files=manifest))
        index = objects.root.parent / 'source-index' / (key + '.json')
        if index.exists():
            ref = json.loads(index.read_text())['source']
            content = objects.get(ref)
            # An object's own hash does not bind a mutable source-index entry
            # to this source version. Check every member against the live
            # manifest before accepting a reused immutable archive.
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    members = archive.infolist()
                    if ([m.filename for m in members] != [name for name, _ in manifest]
                            or [m.file_size for m in members] != [p.stat().st_size for p in paths]
                            or any(hashlib.sha256(archive.read(name)).hexdigest() != sha
                                   for name, sha in manifest)):
                        raise ValueError('RESEARCH_SOURCE_INDEX_BINDING_MISMATCH')
            except (zipfile.BadZipFile, RuntimeError, OSError) as error:
                raise ValueError('RESEARCH_SOURCE_INDEX_BINDING_MISMATCH') from error
        else:
            ref = objects.put(source_zip())
            write_bytes_once(index, canonical_json(dict(source=ref)).encode())
        result = dict(source=ref, environment=objects.put_json(self.environment),
                      parameters=objects.put_json(self.parameters))
        self.check()
        return result
