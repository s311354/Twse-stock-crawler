"""Offline bounded performance probe; explicit audited accounts, no fills/mail.

Run as a module from the repository root. Timings are inclusive (nested stages
overlap); counters and full result hashes are the stable acceptance metrics.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import datetime as dt
import json
import io
from pathlib import Path
import resource
import subprocess
import sys
import time
from copy import deepcopy
from unittest.mock import patch

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayRequest, ReplayAccount, ReplayHolding, calendar_window
from infrastructure.replay_runtime import current_versions
from infrastructure.storage.replay_state import SQLiteReplayState
from infrastructure.research_archive import write_once


class Probe:
    def __init__(self):
        self.stages = {}

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                value = function(*args, **kwargs)
                return list(value) if name == 'environment_scan' else value
            finally:
                row = self.stages.setdefault(name, dict(calls=0, seconds=0.0))
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start
        return measured

    def install(self, stack):
        import application.replay_inputs as inputs
        import application.research_evidence as raw
        import application.replay_service as service
        import domain.market_breadth as breadth
        import infrastructure.daily_research as daily
        import infrastructure.replay_runtime as runtime
        import infrastructure.replay_evidence as evidence
        import infrastructure.research_job as job
        import infrastructure.research_objects as objects
        from infrastructure.storage.research_state import IncrementalResearchState
        from infrastructure.research_objects import ResearchVersionGuard, ResearchObjects
        from infrastructure.storage.replay_state import ReplaySession
        # Time the complete legacy inline ZIP block too, not only ZipFile
        # construction (metadata libraries also open read-only ZIPs).
        original_init = daily.zipfile.ZipFile.__init__
        original_close = daily.zipfile.ZipFile.close
        def zip_init(archive, file, *args, **kwargs):
            original_init(archive, file, *args, **kwargs)
            if archive.mode == 'w' and isinstance(file, io.BytesIO):
                archive._benchmark_start = time.perf_counter()
        def zip_close(archive):
            try:
                return original_close(archive)
            finally:
                if hasattr(archive, '_benchmark_start'):
                    row = self.stages.setdefault('source_zip_write', dict(calls=0, seconds=0.0))
                    row['calls'] += 1
                    row['seconds'] += time.perf_counter()-archive._benchmark_start
                    del archive._benchmark_start
        stack.enter_context(patch.object(daily.zipfile.ZipFile, '__init__', zip_init))
        stack.enter_context(patch.object(daily.zipfile.ZipFile, 'close', zip_close))
        from application.analysis_dataset import AnalysisDatasetBuilder
        from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
        targets = [(evidence, 'load_evidence', 'load_verify'), (runtime, 'load_evidence', 'load_verify'),
            (daily, 'load_evidence', 'load_verify'), (job, 'load_evidence', 'load_verify'),
            (inputs.PinnedReplayEvidence, 'select', 'select'),
            (runtime, 'parse_ixbrl', 'ixbrl_parse'), (raw, 'verify_publication', 'publication_verify'),
            (breadth, 'derive_twse_market_breadth', 'breadth'),
            (AnalysisDatasetBuilder, 'build_decisions', 'les_erem'),
            (PortfolioAllocationV2Pipeline, 'enrich', 'pae'),
            (daily, 'coverage_audit', 'coverage'), (service, 'coverage_audit', 'coverage'),
            (daily, 'current_versions', 'version_scan'),
            (runtime, 'current_versions', 'version_scan'), (daily, 'distributions', 'environment_scan'),
            (objects, 'distributions', 'environment_scan'), (objects, 'source_zip', 'source_zip_build'),
            (daily, 'write_once', 'legacy_artifact_write'), (ReplaySession, 'finish', 'state_result_save'),
            (ResearchVersionGuard, 'retain', 'source_environment_retention'),
            (ResearchObjects, 'put_json', 'object_json_retention'),
            (SQLiteReplayState, 'export_snapshot', 'full_state_export'),
            (SQLiteReplayState, 'restore_snapshot', 'full_state_restore'),
            (IncrementalResearchState, 'read_rows', 'state_read'),
            (IncrementalResearchState, 'restore_rows', 'state_restore'),
            (IncrementalResearchState, 'delta', 'state_delta'),
            (service.StrictReplayEngine, 'run_with_coverage', 'engine')]
        for owner, attr, name in targets:
            stack.enter_context(patch.object(owner, attr, self.wrap(name, getattr(owner, attr))))


def disk_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--case-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['legacy', 'job', 'uncached'], default='legacy')
    parser.add_argument('--days', type=int, choices=[3, 4, 5], default=3)
    parser.add_argument('--resume-only', action='store_true', help='Fresh-process recovery probe of an existing benchmark')
    args = parser.parse_args()
    if args.resume_only:
        from infrastructure.daily_research import request_from_document, retain_research
        probe = Probe(); started = time.perf_counter()
        with ExitStack() as stack:
            probe.install(stack)
            if args.mode == 'legacy':
                folders = list((args.output/'archives').glob('*'))
                folder = max(folders, key=lambda p: json.loads((p/'request.json').read_text())['as_of'])
                req = request_from_document(json.loads((folder/'request.json').read_text()), args.output/'UNUSED')
                retain_research(args.evidence, req, args.output/'archives',
                                predecessor=json.loads((folder/'predecessor.json').read_text()))
            else:
                from infrastructure.research_job import ResearchJob
                folder = sorted((args.output/'job'/'days').glob('*'))[-1]
                req = request_from_document(json.loads((folder/'request.json').read_text()), args.output/'UNUSED')
                with ResearchJob(args.evidence, args.output/'job', cache_enabled=args.mode == 'job') as job:
                    job.run_day(req)
        report = dict(schema='replay-benchmark-resume/v1', seconds=time.perf_counter()-started,
                      stages=probe.stages, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_once(args.output/'resume-benchmark.json', report)
        print(json.dumps(report))
        return
    if args.output.exists():
        raise ValueError('FRESH_BENCHMARK_OUTPUT_REQUIRED')
    args.output.mkdir(parents=True)
    # Only inspect the small manifest here: measured evidence loading happens
    # inside the selected public entry point, not as benchmark prewarming.
    manifest = json.loads((args.evidence.parent / 'manifest.json').read_text())
    eid = manifest['dataset_id']
    source = json.loads(args.case_audit.read_text())
    days = [d for d in source['days'] if '2026-08-03' <= d['as_of'][:10] <= '2026-08-07'][:args.days]
    if len(days) != args.days:
        raise ValueError('EXPLICIT_DAILY_ACCOUNTS_REQUIRED')
    versions = current_versions()
    requests = []
    for row in days:
        day = dt.date.fromisoformat(row['as_of'][:10])
        account = row['account']
        requests.append(ReplayRequest('performance-pilot', replay_close_cutoff(str(day)), *calendar_window(day),
            ('3014', '6834'), 'TWSE', '13', ReplayAccount(account['cash'], account['account_equity'],
                tuple(ReplayHolding(**h) for h in account['holdings'])), versions, eid,
            args.output / 'NEVER_WRITTEN.sqlite3'))
    probe, measurements = Probe(), []
    start = time.perf_counter()
    initialization_seconds, restart_seconds = 0.0, None
    with ExitStack() as stack:
        probe.install(stack)
        from infrastructure.daily_research import retain_research
        if args.mode == 'legacy':
            pre = SQLiteReplayState.empty_snapshot(requests[0])
            for req in requests:
                before = disk_bytes(args.output); tick = time.perf_counter()
                folder = retain_research(args.evidence, req, args.output / 'archives', predecessor=pre)
                measurements.append(dict(day=str(req.history_end), seconds=time.perf_counter()-tick,
                    added_bytes=disk_bytes(args.output)-before, archive=str(folder)))
                last_pre = pre
                pre = json.loads((folder / 'poststate.json').read_text())
            tick = time.perf_counter()
            retain_research(args.evidence, requests[-1], args.output/'archives', predecessor=last_pre)
            retry = time.perf_counter()-tick
            cache = None
            primary_stages = deepcopy(probe.stages)
            primary_seconds = time.perf_counter()-start
        else:
            from infrastructure.research_job import ResearchJob
            tick = time.perf_counter()
            instance = ResearchJob(args.evidence, args.output/'job', cache_enabled=args.mode == 'job')
            initialization_seconds = time.perf_counter()-tick
            with instance as job:
                for req in requests:
                    before = disk_bytes(args.output); tick = time.perf_counter()
                    day = job.run_day(req)
                    measurements.append(dict(day=str(req.history_end), seconds=time.perf_counter()-tick,
                        added_bytes=disk_bytes(args.output)-before, archive=str(day.folder)))
                tick = time.perf_counter(); job.run_day(requests[-1]); retry = time.perf_counter()-tick
                cache = job.cache_stats
                primary_stages = deepcopy(probe.stages)
                primary_seconds = time.perf_counter()-start
        primary_output_bytes = disk_bytes(args.output)
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-m', 'examples.research.benchmark_replay', '--resume-only',
            '--evidence', str(args.evidence), '--case-audit', str(args.case_audit), '--output', str(args.output),
            '--mode', args.mode, '--days', str(args.days)], check=True, capture_output=True, text=True)
        restart_seconds = time.perf_counter()-tick
    source_copies = len(list(args.output.rglob('source.zip'))) if args.mode == 'legacy' else primary_stages.get('source_zip_build', {}).get('calls', 0)
    report = dict(schema='replay-benchmark/v1', mode=args.mode, evidence_id=eid, versions=versions.__dict__,
        symbols=['3014','6834'], accounts_source=str(args.case_audit), measurements=measurements,
        same_day_retry_seconds=retry, total_seconds=time.perf_counter()-start, stages=probe.stages,
        initialization_seconds=initialization_seconds, restart_and_retry_seconds=restart_seconds,
        primary_stages=primary_stages, primary_seconds=primary_seconds,
        primary_output_bytes=primary_output_bytes,
        restart_report=json.loads((args.output/'resume-benchmark.json').read_text()),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_bytes=disk_bytes(args.output), cache=cache, timing_policy='inclusive nested stages; macOS RSS bytes')
    report['source_archives_created_primary'] = source_copies
    report['cold_open_plus_first_day_seconds'] = initialization_seconds + measurements[0]['seconds']
    write_once(args.output/'benchmark.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
