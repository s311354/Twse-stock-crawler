"""Declared units and local-only migration of pre-v6 EPS unit references."""

from dataclasses import replace
import hashlib

import pandas as pd
import pytest

import application.fundamental_deriver as deriver_module
from application.fundamental_deriver import FundamentalDeriver
from application.mops_refresh_control import MopsRefreshPolicy
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_filings import ixbrl_record
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import _FakeMopsClient, _candidate
from tests.fundamentals.test_mops_refresh_control import Clock, SIGNAL, service


def document(*, currency="TWD", eps_unit="EarningsPerShare", declarations=True):
    units = f"""
      <xbrli:unit id="money"><xbrli:measure>currency:{currency}</xbrli:measure></xbrli:unit>
      <xbrli:unit id="{eps_unit}"><xbrli:divide>
        <xbrli:unitNumerator><xbrli:measure>currency:{currency}</xbrli:measure></xbrli:unitNumerator>
        <xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unitDenominator>
      </xbrli:divide></xbrli:unit>
    """ if declarations else ""
    return f"""<html xmlns:xbrli="http://www.xbrl.org/2003/instance"
      xmlns:currency="http://www.xbrl.org/2003/iso4217"><body>
      <xbrli:context id="YTD"><xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate>
        <xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
      {units}
      <ix:nonFraction name="ifrs-full:Revenue" contextRef="YTD" unitRef="money" scale="3">5,000</ix:nonFraction>
      <ix:nonFraction name="ifrs-full:BasicEarningsLossPerShare" contextRef="YTD"
        unitRef="{eps_unit}" scale="0" sign="-">0.14</ix:nonFraction>
    </body></html>""".encode()


def filed(facts):
    return FiledFinancials(2026, 2, "2026-08-10T05:00:00Z", 1, tuple(
        FiledFact(concept=f.concept, numeric_value=f.numeric_value, text_value=f.text_value,
                  start_date=f.start_date, end_date=f.end_date, instant=f.instant,
                  unit=f.unit, dimensions=f.dimensions, context_id=f.context_id)
        for f in facts
    ))


@pytest.mark.parametrize("unit_id", ["EarningsPerShare", "u123", "TWDPerShare"])
def test_parser_resolves_arbitrary_eps_unit_ids_and_keeps_scale_and_sign(unit_id):
    facts = parse_ixbrl(document(eps_unit=unit_id))
    assert facts[0].unit == "iso4217:TWD"
    assert facts[1].unit == "iso4217:TWD/xbrli:shares"
    row = ixbrl_record("3016", filed(facts))
    assert row["revenue"] == 5000.
    assert row["basic_eps"] == -.14  # EPS is not divided by 1,000.
    assert row["available_at"] == "2026-08-10T05:00:00Z"


@pytest.mark.parametrize("damage", ["usd", "undeclared", "unknown_namespace", "conflicting", "bad_denominator"])
def test_unit_labels_cannot_supply_missing_or_conflicting_currency_evidence(damage):
    content = document(eps_unit="TWDPerShare")
    if damage == "usd":
        content = document(currency="USD", eps_unit="TWDPerShare")
    elif damage == "undeclared":
        content = document(declarations=False, eps_unit="TWDPerShare")
    elif damage == "unknown_namespace":
        content = content.replace(b"http://www.xbrl.org/2003/iso4217", b"https://example.test/not-currency")
    elif damage == "conflicting":
        content = content.replace(b"</body>", b'<xbrli:unit id="TWDPerShare"><xbrli:measure>xbrli:pure</xbrli:measure></xbrli:unit></body>')
    else:
        content = content.replace(b"xbrli:shares</xbrli:measure>", b"xbrli:pure</xbrli:measure>")
    assert ixbrl_record("3016", filed(parse_ixbrl(content)))["basic_eps"] is None


def test_money_and_per_share_units_cannot_be_interchanged():
    original = filed(parse_ixbrl(document()))
    corrupted = replace(original, facts=tuple(replace(f, unit=("iso4217:TWD" if
        f.concept.endswith("PerShare") else "iso4217:TWD/xbrli:shares")) for f in original.facts))
    row = ixbrl_record("3016", corrupted)
    assert row["revenue"] is None and row["basic_eps"] is None


def test_direct_adapter_cannot_treat_legacy_unit_ids_as_measure_evidence():
    source = filed(parse_ixbrl(document()))
    legacy = replace(source, facts=tuple(replace(f, unit=("TWDPerShare" if
        f.concept.endswith("PerShare") else "TWD")) for f in source.facts))
    row = ixbrl_record("3016", legacy)
    assert row["revenue"] is None and row["basic_eps"] is None


def seed_legacy(repo):
    content = document()
    facts = parse_ixbrl(content)
    # Old parser saved unitRef names, not the declared semantic measures.
    legacy = [replace(facts[0], unit="money"), replace(facts[1], unit="EarningsPerShare")]
    candidate = _candidate(2026, 2, symbol="3016")
    stored = repo.save_revision(candidate, content, legacy,
                               available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    return stored, legacy


def test_legacy_unit_repair_uses_one_join_and_preserves_original_provenance(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    stored, legacy = seed_legacy(repo)
    statements = []
    original = repo._connect

    def connect():
        connection = original()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repo, "_connect", connect)
    records = repo.load_filed_financials("3016", SIGNAL)
    assert len([q for q in statements if "SELECT" in q.upper()]) == 1
    assert "LEFT JOIN facts" in statements[0]
    assert ixbrl_record("3016", records[0])["basic_eps"] == -.14
    assert records[0].available_at == stored.available_at
    assert records[0].content_sha256 == hashlib.sha256(document()).hexdigest()
    assert records[0].revision_number == stored.revision_number == 1
    # Repair the read model only; never rewrite original facts or filing dates.
    assert repo.load_facts(stored.filing_id) == legacy
    assert repo.load_filed_financials("3016", "2026-08-09T23:59:59Z") == []


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_raw_file_changes_invalidate_unit_cache_and_cannot_upgrade_eps(tmp_path, damage):
    repo = MopsRepository(tmp_path / "mops")
    stored, _ = seed_legacy(repo)
    assert ixbrl_record("3016", repo.load_filed_financials("3016", SIGNAL)[0])["basic_eps"] == -.14
    raw = repo.raw_path(stored)
    if damage == "changed":
        raw.write_bytes(document(currency="USD"))
    else:
        raw.unlink()
    assert ixbrl_record("3016", repo.load_filed_financials("3016", SIGNAL)[0])["basic_eps"] is None


def test_local_migration_invalidates_old_snapshot_without_remote_requests(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    _, legacy = seed_legacy(repo)
    old_version = "2026-09-25-qs-dc-v10-audit-period-evidence"
    with monkeypatch.context() as patch:
        patch.setattr(deriver_module, "FUNDAMENTAL_DERIVATION_VERSION", old_version)
        with repo.unit_of_work() as uow:
            old, _ = FundamentalDeriver(repo).load_or_derive(
                "3016", parsed_signal=pd.Timestamp(SIGNAL), resolved_sector="24",
                filed_financials=[replace(repo.load_filed_financials("3016", SIGNAL)[0], facts=filed(legacy).facts)],
                uow=uow,
            )
    assert old.snapshot["les_v6_mops_filings"][0]["basic_eps"] is None
    client = _FakeMopsClient(repo, [], [])
    svc = service(repo, client, Clock(), MopsRefreshPolicy(max_requests=0))
    assert svc.migrate_local_cache(["3016"], signal_date=SIGNAL, sector="24") == 1
    cached = repo.load_cached_bundles({"3016": "24"}, as_of=SIGNAL,
                                     derivation_version=deriver_module.FUNDAMENTAL_DERIVATION_VERSION)["3016"]
    assert cached.bundle.snapshot["les_v6_mops_filings"][0]["basic_eps"] == -.14
    assert client.discovery_calls == 0 and client.downloaded_periods == []
    assert svc.migrate_local_cache(["3016"], signal_date=SIGNAL, sector="24") == 0
