"""Manual public-fee source capture and fixed-fill sensitivity, no ledger/mail."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from application.exit_fee_sensitivity import compare_public_fees
from application.replay_contracts import digest
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import write_once, utc_now
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard


SOURCES = {
    "yuanta_odd_page": ("https://www.yuanta.com.tw/file-repository/content/odd/index.html", ["qa-data.js"]),
    "yuanta_odd": ("https://www.yuanta.com.tw/file-repository/content/odd/js/qa-data.js", ["0.001425", "1元"]),
    "yuanta_accounts": ("https://www.yuanta.com.tw/eYuanta/securities/aporder/Instructions/836878aa-5e5f-4dc8-9d18-984e9bf5c1cd?TargetId=8bc55154-c7ef-4f09-9e6d-e65a8ceb0f39&TargetMode=1", ["無條件捨去至整數位"]),
    "banktaiwan_faq": ("https://www.twfhcsec.com.tw/Home/faq", ["0.1425", "20元", "無條件捨去"]),
    "banktaiwan_business": ("https://www.twfhcsec.com.tw/Activity/BusinessIntro", ["0.0498", "同一委託書相同成交價格"]),
    "stock_tax": ("https://www.etax.nat.gov.tw/etwmain/tax-info/understanding/tax-q-and-a/national/securities-transaction-tax/filing/mM8n39b", ["千分之3"]),
    "tpex_odd_session": ("https://www.tpex.org.tw/zh-tw/mainboard/trading/rules/odd-lot.html", ["9:10"]),
}


class SecureSystemCurl:
    """Explicit system trust-store alternative; never --insecure/-k."""
    transport = "SYSTEM_CURL_TLS_VALIDATION"

    def request_bytes(self, method, url, *, context, content_validator):
        if method != "GET" or url not in {row[0] for row in SOURCES.values()}:
            raise ValueError("FEE_SOURCE_ALLOWLISTED_GET_REQUIRED")
        time.sleep(1)
        response = subprocess.run(["/usr/bin/curl", "--disable", "--fail", "--silent", "--show-error",
            "--location", "--max-redirs", "3", "--proto", "=https", "--proto-redir", "=https",
            "--connect-timeout", "10", "--max-time", "20", "--max-filesize", str(16 * 1024 * 1024), url],
            capture_output=True, check=True, timeout=25)
        content_validator(response.stdout)
        return SimpleNamespace(content=response.stdout)


def validate_source(key, content):
    if not isinstance(content, bytes) or not 0 < len(content) <= 16 * 1024 * 1024:
        raise ValueError("FEE_SOURCE_SIZE_INVALID:" + key)
    text = unescape(re.sub(r"<[^>]+>", "", content.decode("utf-8-sig")))
    if key == "yuanta_odd_page":
        text += content.decode("utf-8-sig")  # retain the observed source script link
    compact = re.sub(r"\s+", "", text)
    if any(marker in compact.lower() for marker in ("accessdenied", "accessblocked", "因為安全性考量")):
        raise ValueError("FEE_SOURCE_BLOCKED:" + key)
    if any(marker not in compact for marker in SOURCES[key][1]):
        raise ValueError("FEE_SOURCE_CLAIM_NOT_FOUND:" + key)


def capture(folder, *, client=None):
    folder = Path(folder)
    path = folder / "sources.json"
    if path.exists():
        return load_sources(path)
    client = client or TwseHttpClient(TwseHttpConfig(max_attempts=1, timeout_seconds=20,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    if isinstance(client, TwseHttpClient) and (client.config.allow_insecure_ssl_fallback
            or client.config.max_attempts != 1 or client.config.min_request_interval_seconds < 1):
        raise ValueError("FEE_SOURCE_SECURE_BOUNDED_TRANSPORT_REQUIRED")
    documents, rows = EvidenceDocuments(folder / "objects"), []
    for key, (url, _) in SOURCES.items():
        response = client.request_bytes("GET", url, context="Public fee source " + key,
            content_validator=lambda value, key=key: validate_source(key, value))
        content = response.content
        rows.append(dict(key=key, url=url, fetched_at=utc_now(), historical_available_at=None,
            transport=getattr(client, "transport", "REQUESTS_TLS_VALIDATION"),
            original_historical_version_verified=False, body_ref=documents.store(content)))
    packet = dict(schema="public-fee-source-capture/v1", sources=rows,
        purpose="PUBLIC_RATE_SCENARIOS_NOT_PRIVATE_OR_HISTORICAL_BROKER_CONTRACT")
    packet["source_id"] = digest(packet)
    write_once(path, packet)
    return load_sources(path)


def load_sources(path):
    path = Path(path)
    packet = json.loads(path.read_bytes())
    if (packet.get("schema") != "public-fee-source-capture/v1"
            or packet.get("source_id") != digest({k: v for k, v in packet.items() if k != "source_id"})
            or [row.get("key") for row in packet.get("sources", [])] != list(SOURCES)):
        raise ValueError("FEE_SOURCE_PACKET_MISMATCH")
    documents = EvidenceDocuments(path.parent / "objects")
    for row in packet["sources"]:
        if (row.get("url") != SOURCES[row["key"]][0] or row.get("historical_available_at") is not None
                or row.get("original_historical_version_verified") is not False):
            raise ValueError("FEE_SOURCE_BINDING_MISMATCH")
        validate_source(row["key"], documents.read(row["body_ref"]))
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("--directory", type=Path, required=True)
    collect.add_argument("--transport", choices=["requests", "system-curl"], default="requests")
    compare = commands.add_parser("compare")
    compare.add_argument("--bank-run", type=Path, required=True)
    compare.add_argument("--sources", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        packet = capture(args.directory, client=SecureSystemCurl() if args.transport == "system-curl" else None)
        print(json.dumps(dict(source_id=packet["source_id"], status="PUBLIC_ORIGINALS_CAPTURED")))
        return 0
    if args.output.resolve() in {args.bank_run.resolve(), args.sources.resolve()}:
        parser.error("output must not replace source")
    guard = ResearchVersionGuard()
    sources = load_sources(args.sources)
    source_bytes = args.bank_run.read_bytes()
    run = json.loads(gzip.decompress(source_bytes) if args.bank_run.suffix == ".gz" else source_bytes)
    result = compare_public_fees(run)
    result.update(public_sources=sources, source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        execution_versions=asdict(guard.versions))
    if args.bank_run.read_bytes() != source_bytes:
        raise ValueError("FEE_RESEARCH_SOURCE_CHANGED")
    if load_sources(args.sources) != sources:
        raise ValueError("FEE_RESEARCH_PUBLIC_SOURCES_CHANGED")
    guard.check()
    result["execution_objects"] = guard.retain(ResearchObjects(args.output.parent / "objects"))
    result["comparison_id"] = digest({k: v for k, v in result.items() if k != "comparison_id"})
    write_once(args.output, result)
    print(json.dumps(dict(comparison_id=result["comparison_id"], promotion=result["policy_promotion"],
        scenarios=[{k: row[k] for k in ("key", "commission_rounding", "total_fee_twd", "conditional_pnl_bridge_twd")}
                   for row in result["scenarios"]]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
