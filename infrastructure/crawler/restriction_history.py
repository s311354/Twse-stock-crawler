"""Bounded, opt-in collection of TWSE historical query source documents.

Queries are useful evidence but cannot automatically certify original release
times, carry-in events or superseded versions. Collection remains UNVERIFIED.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from urllib.parse import urlencode

from domain.restriction_announcements import BUNDLE_SCHEMA, CATEGORIES
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseSchemaChangedError


class HistoricalRestrictionCollector:
    def __init__(self, http: TwseHttpClient, *, clock=None):
        self.http = http
        self.clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))

    def collect(self, *, start: dt.date, end: dt.date) -> dict:
        if start > end:
            raise ValueError("HISTORICAL_QUERY_RANGE_INVALID")
        ymd = lambda day: day.strftime("%Y%m%d")
        # Full halt history is needed to retain long-running pre-window halts.
        # The altered-trading endpoint is a dated snapshot, NOT a publication
        # ledger; keeping it UNVERIFIED prevents treating it as one.
        queries = [
            (CATEGORIES[0], "afterTrading/TWTAWU", {"startDate": "20111003", "endDate": ymd(end), "querytype": "3"}),
            (CATEGORIES[1], "fullDelivery/TWT85U", {"date": ymd(end)}),
            (CATEGORIES[2], "announcement/punish", {"startDate": ymd(start), "endDate": ymd(end),
                                                   "querytype": "3", "sortKind": "DATE"}),
        ]
        bundle = {"schema_version": BUNDLE_SCHEMA, "market": "TWSE", "documents": {},
                  "coverage": [], "announcements": [], "collection_status": "UNVERIFIED"}
        for category, endpoint, params in queries:
            url = "https://www.twse.com.tw/rwd/zh/" + endpoint + "?" + urlencode(dict(params, response="json"))
            # Shared transport enforces throttling and stops on access blocking;
            # no per-symbol requests and no catch-and-continue after a failure.
            response = self.http.request_bytes("GET", url, context=f"TWSE historical {category}")
            body = response.content.decode("utf-8")
            try:
                payload = json.loads(body.lstrip("\ufeff"))
            except (ValueError, TypeError) as error:
                raise TwseSchemaChangedError("Historical endpoint did not return JSON") from error
            if not isinstance(payload, dict) or payload.get("stat") != "OK" or not isinstance(payload.get("data"), list):
                raise TwseSchemaChangedError("Historical endpoint lacks a complete successful data response")
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            bundle["documents"][category] = {"url": url, "retrieved_at": self.clock().isoformat(),
                                              "sha256": digest, "body_utf8": body}
            bundle["coverage"].append({
                "category": category, "start_at": f"{start.isoformat()}T00:00:00+08:00",
                "end_at": f"{end.isoformat()}T23:59:59.999999+08:00", "status": "UNVERIFIED",
                "complete": False, "includes_carry_in": False, "includes_all_revisions": False,
                "universe_scope": "UNVERIFIED",
                "proofs": [], "query_rows": len(payload["data"]),
                "reason": "ORIGINAL_PUBLICATION_AND_COMPLETE_REVISION_COVERAGE_REQUIRE_VERIFICATION",
            })
        return bundle
