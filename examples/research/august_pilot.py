"""Bounded REAL-data verification, not synthetic fixtures or a trading simulator.

--prepare fetches three named official closure PDFs and the existing annual
calendar endpoint, once. --run is offline and uses the audited case's explicit
account for each day; it does not simulate fills or touch BankEmulator state.
"""
import argparse
import base64
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

import requests

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayRequest, ReplayAccount, ReplayHolding, calendar_window, digest
from application.research_audit import compare_results, summarize
from infrastructure.calendar.twse_calendar import TwseCalendar, TWSE_CALENDAR_URL
from infrastructure.daily_research import retain_research, reproduce_research
from infrastructure.replay_evidence import collect_evidence, load_evidence
from infrastructure.replay_runtime import current_versions
from infrastructure.research_archive import create_dataset, write_once, write_bytes_once, utc_now
from infrastructure.storage.replay_state import SQLiteReplayState

SOURCES = (
    ("2024-10-02", "2024-10-01", "中華民國 113 年 10 月 1 日", "113年10月2日休市一天",
     "https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e301924806d5b4004e.pdf"),
    ("2024-10-03", "2024-10-02", "中華民國 113 年 10 月 2 日", "113年10月3日休市一天",
     "https://investoredu.twse.com.tw/Mobile_pages/..%2FFileSystem%2FFileUpload%2F2c6e50d4-a390-439c-a2c4-823937dee4b2.pdf"),
    ("2024-10-31", "2024-10-30", "中華民國 113 年 10 月 30 日", "113年10月31日休市一天",
     "https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e30192dd5179bc0327.pdf"),
)


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    start, end = calendar_window(dt.date(2026, 8, 3))[0], dt.date(2026, 8, 7)
    annual_path = root / "annual-sources.json"
    if annual_path.exists():
        annual = json.loads(annual_path.read_text())
    else:
        annual = {}
        for year in range(start.year, end.year + 1):
            raw = TwseCalendar()._fetch_year_json(year)  # Existing source client; no new endpoint.
            annual[str(year)] = dict(payload=raw, source_url=TWSE_CALENDAR_URL,
                retrieved_at=utc_now(), availability_policy="ORIGINAL_PUBLICATION_UNVERIFIED")
        write_once(annual_path, annual)
    days = {(start + dt.timedelta(days=i)).isoformat(): "UNKNOWN" for i in range((end-start).days+1)}
    calendar = dict(source="TWSE_OFFICIAL", days=days, documents=annual, announcements=[])
    for day, published, date_quote, closure_quote, url in SOURCES:
        path = root / f"closure-{day}.json"
        if path.exists():
            event = json.loads(path.read_text())
        else:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.content
            if not content.startswith(b"%PDF-"):
                raise ValueError("OFFICIAL_SOURCE_NOT_PDF")
            text = subprocess.run(["pdftotext", "-layout", "-", "-"], input=content,
                                  capture_output=True, check=True).stdout.decode()
            # These specific original announcements were manually reviewed.
            # A changed PDF is not silently accepted on title/date assumptions.
            if date_quote not in text or closure_quote not in text:
                raise ValueError(f"REVIEW_SOURCE_TEXT_REQUIRED:{day}:{text}")
            event = dict(date=day, status="CLOSED", market="TWSE")
            event["proof"] = dict(status="VERIFIED", verified_by="2026-09-28 bounded source review",
                verified_at=utc_now(), verification_method="Official original PDF date and full-market closure text reviewed",
                binding=dict(kind="CALENDAR_EXCEPTION", **event), publication_precision="DATE",
                published_at=published, extraction_reviewed=True, quotes=[date_quote, closure_quote],
                document=dict(url=url, retrieved_at=utc_now(), sha256=hashlib.sha256(content).hexdigest(),
                    body_base64=base64.b64encode(content).decode(), extracted_text=text))
            write_once(path, event)
        calendar["announcements"].append(event)
    write_once(root / "calendar-repaired.json", calendar)
    baseline = deepcopy(calendar); baseline["announcements"] = []
    write_once(root / "calendar-before.json", baseline)
    payload = collect_evidence(market_cache=Path("data/market_history.sqlite3"),
        mops_cache=Path("data/mops/mops.sqlite3"), restriction_cache=Path("data/mops/trading_restrictions.sqlite3"),
        restriction_file=None, calendar=baseline, symbols=("3014", "6834"), market="TWSE", sector="13",
        start=start, end=end, raw=True)
    payload["collection_scope"] = "BOUNDED_3014_6834_AUDIT_NOT_FULL_UNIVERSE_PERFORMANCE"
    payload["batch_policy"] = "SOURCE_NORMALIZED_TWSE_SNAPSHOT_ORIGINAL_RESPONSE_NOT_RETAINED_BY_LEGACY_CACHE"
    before = create_dataset(payload, root / "datasets")
    payload["calendar"] = calendar
    after = create_dataset(payload, root / "datasets", parent=before)
    write_once(root / "datasets.json", dict(before=str(before.resolve()), after=str(after.resolve())))
    print(json.dumps(dict(before=str(before), after=str(after))))


def run(root: Path, case_path: Path):
    datasets = json.loads((root / "datasets.json").read_text())
    case = json.loads(case_path.read_text())
    source_days = [day for day in case["days"] if "2026-08-03" <= day["as_of"][:10] <= "2026-08-07"]
    if len(source_days) != 5:
        raise ValueError("FIVE_EXPLICIT_SOURCE_ACCOUNTS_REQUIRED")
    output = root / "runs" / current_versions().code
    records, stored = [], {}
    for label, evidence_path in datasets.items():
        evidence = Path(evidence_path); eid, payload = load_evidence(evidence)
        pre = None
        for source in source_days:
            day = dt.date.fromisoformat(source["as_of"][:10])
            account = source["account"]
            holdings = tuple(ReplayHolding(**h) for h in account["holdings"])
            start, end = calendar_window(day)
            request = ReplayRequest("august-evidence-pilot", replay_close_cutoff(day.isoformat()), start, end,
                tuple(payload["symbols"]), "TWSE", "13", ReplayAccount(account["cash"], account["account_equity"], holdings),
                current_versions(), eid, output / "NEVER_WRITTEN.sqlite3")
            if pre is None: pre = SQLiteReplayState.empty_snapshot(request)
            folder = retain_research(evidence, request, output / label, predecessor=pre)
            comparison = reproduce_research(folder, evidence)
            write_once(folder / "verification.json", {k:v for k,v in comparison.items() if k != "reproduced_at"})
            if not comparison["equal"]: raise ValueError("PILOT_REPRODUCTION_MISMATCH")
            pre = json.loads((folder / "poststate.json").read_text())
            audit = json.loads((folder / "coverage.json").read_text())
            result = json.loads((folder / "result.json").read_text())
            stored[label, day.isoformat()] = result
            write_bytes_once(folder / "coverage.md", summarize(audit).encode())
            records.append(dict(variant=label, day=day.isoformat(), archive=str(folder.resolve()),
                dataset_id=eid, reproduction_equal=comparison["equal"], coverage_status=audit["status"],
                account_source_report=source["path"], account_source_sha256=source["file_sha256"],
                calendar_unknown_count=len(audit["calendar_unknown_dates"]),
                stocks=[dict(symbol=r["symbol"], valid_bar_days=r["valid_bar_days"],
                    expected_market_sessions=r["expected_market_sessions"], missing_quarters=r["financial"].get("missing_periods"),
                    quality=r["quality_status"], quality_coverage=r["quality_coverage"],
                    restrictions=r["restrictions"]["status"], kelly=r["kelly"]["status"],
                    final_les=r["final_les"], coverage_problems=r["coverage_problems"]) for r in audit["rows"]]))
            print(label, day, audit["status"], "reexecuted equal", flush=True)
    changes = {day["as_of"][:10]: compare_results(stored["before",day["as_of"][:10]],
                  stored["after",day["as_of"][:10]]) for day in source_days}
    summary = dict(schema="bounded-real-pilot/v1", source_case=str(case_path.resolve()),
        source_universe_count=case["original_universe_count"], scope=["3014","6834"], days=records,
        before_after_full_differences=changes, status="PARTIAL", all_reproductions_equal=True,
        exclusions="Not an investment-return test, no simulated fills or whole-universe validation")
    write_once(output / "summary.json", summary)
    print(output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case-audit", type=Path)
    args = parser.parse_args()
    if args.prepare: prepare(args.root)
    else:
        if args.case_audit is None: parser.error("--case-audit is required with --run")
        run(args.root, args.case_audit)
