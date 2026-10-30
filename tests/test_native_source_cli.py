"""SYNTHETIC ONLY: cache-only immutable source audit, no network or orders."""
import json

import pytest

from domain.les_v7.contracts import identity
from infrastructure.crawler.official_pe import OfficialPeRepository
from infrastructure.crawler.security_master import collect_security_master
from interface.native_sources import main
from tests.test_official_pe import snapshot
from tests.test_security_master import SyntheticClient, OBSERVED, CUTOFF


def retained(tmp_path):
    root = tmp_path / "SYNTHETIC_ONLY"
    master = collect_security_master(root / "security-master", "TWSE",
        client=SyntheticClient(), now=lambda: OBSERVED)
    OfficialPeRepository(root / "official-pe.sqlite3").record(snapshot(symbol="2330"))
    output = root / "audit.json"
    return root, master, output, ["--root", str(root), "--market", "TWSE", "--session", "2026-10-13",
        "--symbols", "2330", "8112A", "--master-manifest", str(master), "--cutoff", CUTOFF,
        "--output", str(output)]


def test_cache_only_audit_verifies_raw_sources_and_never_uses_network(tmp_path, monkeypatch):
    root, master, output, argv = retained(tmp_path)
    original = master.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("cache-only audit tried network")
    monkeypatch.setattr("infrastructure.crawler.twse_http_client.TwseHttpClient.request_bytes", forbidden)
    assert main(argv) == 0
    result = json.loads(output.read_text())
    body = {key: value for key, value in result.items() if key != "audit_id"}
    assert result["audit_id"] == identity(body)
    assert result["summary"]["security_status_counts"] == {"VERIFIED": 1, "OUT_OF_SCOPE": 1}
    assert result["rows"]["2330"]["official_pe"][0]["status"] == "REPORTED"
    assert result["no_accounts_or_mail"] and not result["historical_information_set_certified"]
    assert master.read_bytes() == original
    first = output.read_bytes()
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        main(argv)
    assert output.read_bytes() == first


def test_cache_only_requires_explicit_master_and_preserves_missing_pe(tmp_path):
    root, master, output, argv = retained(tmp_path)
    at = argv.index("--master-manifest")
    with pytest.raises(ValueError, match="CACHE_ONLY_MASTER"):
        main(argv[:at] + argv[at + 2:])
    # A different date is absence of retained observation, not an empty
    # official table or a positive ratio carried forward from the next day.
    argv[argv.index("--session") + 1] = "2026-10-12"
    assert main(argv) == 0
    assert json.loads(output.read_text())["rows"]["2330"]["official_pe"][0]["status"] == "MISSING"


def test_audit_cannot_write_inside_original_manifest_namespace(tmp_path):
    root, master, output, argv = retained(tmp_path)
    argv[argv.index("--output") + 1] = str(master.parent / "new.json")
    with pytest.raises(ValueError, match="OVERWRITE_ORIGINALS"):
        main(argv)


def test_cache_only_rejects_corrupt_master_not_silently_recaptures(tmp_path, monkeypatch):
    root, master, output, argv = retained(tmp_path)
    raw = json.loads(master.read_text())
    raw["market"] = "TPEX"
    master.write_text(json.dumps(raw))  # Deliberately corrupt synthetic fixture.
    def forbidden(*args, **kwargs):
        raise AssertionError("corruption triggered network fallback")
    monkeypatch.setattr("infrastructure.crawler.twse_http_client.TwseHttpClient.request_bytes", forbidden)
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()
