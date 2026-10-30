"""Manual expanded FR2 frozen-fit diagnostics; no fitting, mail or trading."""
import argparse
from dataclasses import asdict
from pathlib import Path

from domain.les_v7.contracts import canonical, identity
from infrastructure.les_v7_fr2_expansion import expand_archive
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from interface.les_v7_fr2_compare import summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--m0", required=True, type=Path)
    parser.add_argument("--scope-job", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        for source in (args.archive, args.m0, args.scope_job):
            original = source.expanduser().resolve()
            original = original.parent if original.is_file() else original
            if output == original or output.is_relative_to(original):
                raise ValueError("FR2_EXPANSION_OUTPUT_MUST_NOT_MODIFY_INPUTS")
        guard = ResearchVersionGuard()
        pinned = guard.retain(ResearchObjects(output / "objects"))
        def progress(done, total, rows):
            print(canonical(dict(event="fr2_expansion_shard_prepared", completed=done, total=total, rows=rows)), flush=True)
        report, forecasts = expand_archive(args.archive, args.m0, args.scope_job, progress=progress)
        guard.check()
        report = dict(report, calculation_versions=asdict(guard.versions), calculation_objects=pinned,
                      calculation_environment_id=identity(guard.environment))
        report_id = identity(report)
        folder = output / report_id
        write_once(folder / "forecasts.json", dict(payload=forecasts, sha256=identity(forecasts)))
        write_once(folder / "comparison.json", dict(payload=report, sha256=report_id))
        write_bytes_once(folder / "comparison_support_note.md", summary(report).encode())
        receipt = dict(schema="les-v7-fr2-expansion-completion/v1", report_id=report_id,
            forecasts_id=identity(forecasts), status=report["status"], research_only=True,
            approved_for_live=False, execution_action="NO_ACTION", order_authority=False)
        guard.check()
        write_once(folder / "completion.json", receipt)
        print(canonical(dict(receipt, output=str(folder))))
        return 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
