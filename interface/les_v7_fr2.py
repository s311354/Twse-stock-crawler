"""Manual FR2 research; no network, mail, scheduler, broker or formal account writes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Explicit cutoff bars/native sources/account request")
    parser.add_argument("--bundle", type=Path, help="Original native frozen model archive (never rewritten)")
    parser.add_argument("--output", type=Path, required=True, help="New immutable research result JSON")
    parser.add_argument("--html", type=Path, help="New local resender-layout preview; NEVER sends mail")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.html is not None and args.html.exists():
            raise ValueError("FR2_NEW_OUTPUT_REQUIRED")
        if args.input.stat().st_size > 64 * 1024 * 1024: raise ValueError("FR2_BOUNDED_REQUEST_REQUIRED")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        model_use = None
        if args.bundle:
            from infrastructure.les_v7_research_bundle import load_research_bundle
            payload["model_bundle"], model_use = load_research_bundle(args.bundle)
        from application.fr2_research import analyze_fr2_request
        from infrastructure.research_archive import write_once, write_bytes_once
        from infrastructure.research_objects import ResearchVersionGuard, ResearchObjects
        from domain.les_v7.report import validate_report
        guard = ResearchVersionGuard()
        objects = ResearchObjects(args.output.parent / "fr2-objects")
        retained = guard.retain(objects)
        retained["request"] = objects.put_json(payload)
        result = analyze_fr2_request(payload)
        result["runtime"] = dict(source_guard_versions=asdict(guard.versions), objects=retained,
            object_store="fr2-objects", environment_id=retained["environment"],
            scope="SOURCE_AND_ENVIRONMENT_RETAINED_NOT_HISTORICAL_DEPLOYMENT")
        if model_use is not None: result["model_use"] = model_use
        for report in result["reports"]: validate_report(report)
        # Rebind the public result after adding verified archive provenance.
        from domain.les_v7.contracts import identity
        result.pop("result_id", None)
        result["result_id"] = identity(result)
        guard.check()
        write_once(args.output, result)
        html_id = None
        if args.html is not None:
            from infrastructure.report.low_entry_v7_renderer import render_reports
            html = render_reports(result["reports"])
            write_bytes_once(args.html, html.encode("utf-8"))
            import hashlib
            html_id = hashlib.sha256(html.encode("utf-8")).hexdigest()
        guard.check()
        receipt = dict(schema="les-v7-fr2-completion/v1", result_id=result["result_id"],
            request_id=result["request_id"], html_sha256=html_id, research_only=True,
            mail_sent=False, order_authority=False, execution_action="NO_ACTION")
        write_once(args.output.with_suffix(args.output.suffix + ".complete.json"), receipt)
        print(json.dumps(dict(output=str(args.output), result_id=result["result_id"],
            reports=len(result["reports"]), research_only=True, mail_sent=False), ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
