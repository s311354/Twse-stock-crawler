"""Bounded, explicit current restriction capture for an isolated repair study.

Use existing endpoint, schema and completeness checks. Retain raw HTTPS bodies
and actual clocks; this is not historical restriction evidence or a live order.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.trading_restrictions import restriction_check
from infrastructure.crawler.trading_restrictions import TradingRestrictionClient
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig, TwseHttpResponse
from infrastructure.research_archive import write_bytes_once, write_once


def capture(symbols, output_dir, *, transport=None, clock=None):
    """Exactly one complete TWSE list set, never one request per stock."""
    symbols = list(symbols)
    if not symbols or len(symbols) != len(set(symbols)) or any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("EXPLICIT_UNIQUE_CAPTURE_SCOPE_REQUIRED")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=False)
    clock = clock or (lambda: datetime.now(timezone.utc))
    client = transport or TwseHttpClient(TwseHttpConfig(max_attempts=1,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    started = clock(); documents = []

    class RetainedClient:
        def get_json(self, url, *, context):
            response = client.request_bytes("GET", url, context=context)
            raw = response.content
            payload = json.loads(raw)
            content_hash = sha256(raw).hexdigest()
            path = "originals/" + content_hash + ".json"
            write_bytes_once(root / path, raw)
            documents.append(dict(url=url, path=path, content_sha256=content_hash,
                fetched_at=clock().isoformat(), status_code=response.status_code,
                attempts=response.attempts, payload_sha256=digest(payload)))
            return TwseHttpResponse(payload, response.status_code, response.attempts)

    day = started.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    values = TradingRestrictionClient(RetainedClient()).load(symbols, session_date=day, observed_at=started)
    completed = clock()
    if completed < started or completed.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() != day:
        raise ValueError("CAPTURE_CLOCK_OR_SESSION_CHANGED")
    for symbol, value in values.items():
        value["available_at"] = completed.isoformat()
        check = restriction_check(value, market="TWSE", market_data_date=day, decision_time=completed, symbol=symbol)
        if check["status"] == "UNKNOWN":
            raise ValueError("INCOMPLETE_CURRENT_RESTRICTION_CAPTURE:" + check["reason"])
    body = dict(schema="les-r24-current-source-overrides/v1", captured_at=completed.isoformat(),
        by_symbol={s: dict(restriction_evidence=values[s]) for s in symbols},
        metadata=dict(schema="les-r24-retained-restriction-originals/v1", market="TWSE",
            started_at=started.isoformat(), completed_at=completed.isoformat(), session_date=day,
            symbols=symbols, documents=documents, original_directory=".",
            projection_policy="EXISTING_TRADING_RESTRICTION_CLIENT_AVAILABLE_AT_CAPTURE_COMPLETION",
            availability_basis="ACTUAL_CURRENT_OBSERVATION_NOT_HISTORICAL_BACKFILL",
            historical_information_set_certified=False, no_mail_orders_or_accounts=True))
    result = dict(body, content_sha256=digest(body))
    write_once(root / "source-overrides.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--capture", action="store_true", help="Explicit HTTPS requests to the three existing official lists")
    args = p.parse_args(argv)
    if not args.capture: raise ValueError("EXPLICIT_CAPTURE_REQUIRED")
    index = json.loads((args.comparison_root / "comparison-index.json").read_text())
    complete = json.loads((args.comparison_root / "complete-index.json").read_text())
    if digest(index) != complete["index_sha256"]: raise ValueError("FIXED_SCOPE_INDEX_HASH_MISMATCH")
    result = capture([e["symbol"] for e in index["records"]], args.output_dir)
    print(json.dumps(dict(count=len(result["by_symbol"]), captured_at=result["captured_at"],
        content_sha256=result["content_sha256"], output=str(args.output_dir / "source-overrides.json"))))
    return 0


if __name__ == "__main__": raise SystemExit(main())
