"""Offline fixed-source benchmark inputs; synthetic documents stay labelled."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from domain.fundamental_models import FilingCandidate
from domain.les_v7.contracts import identity
from examples import daily_fixed_fixture as fixture
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository


CUTOFF = "2026-10-16T01:28:30.674655+00:00"
EVALUATION = "2026-10-16T15:16:04+00:00"


def source(tmp_path, monkeypatch, *, symbol="3014", issuer=None):
    root = tmp_path / "source"
    repository = MopsRepository(root, connect_attempts=1)
    body = b"<html><body>SYNTHETIC_TEST_ONLY</body></html>"
    filing = repository.save_revision(FilingCandidate(issuer or symbol, 2026, 2, "C", "https://example.invalid/test"),
        body, parse_ixbrl(body), available_at="2026-08-10T05:00:00+00:00", fetched_at="2026-08-10T05:01:00+00:00")
    bundle = tmp_path / "source-bundle.json"
    bundle.write_text('{"schema":"SYNTHETIC_TEST_ONLY"}')
    account = dict(as_of=CUTOFF, value=1000000., available_cash=1000000.,
                   positions=[], intended_weight=.25, capacity_weight=.25)
    packet = dict(symbol=symbol, market="TWSE", cutoff=CUTOFF,
        market_rows=[dict(Date="2026-10-15", Close=100., Open=99., High=101., Low=98., Volume=1000000)],
        filings=[dict(symbol=issuer or symbol, available_at="2026-08-10T05:00:00+00:00")],
        valuation_filings=[], expected_session="2026-10-15", historical_replay=False,
        restriction_evidence=dict(status="UNKNOWN"), calendar_evidence=None)
    report = dict(symbol=symbol, signal_date="2026-10-15", score=999,
        diagnostics=dict(native_evidence=dict(source_packet=packet),
            evidence_binding=dict(market="TWSE", cutoff=CUTOFF, bars_hash=identity(packet["market_rows"])),
            quality=dict(status="PASS"), factor_ledger=dict(fake=999)),
        pae_authorization=dict(account=account, account_sha256=identity(account)))
    reports = [report]
    def stream(path, *, context):
        context._started = True
        for item in reports:
            yield deepcopy(item)
        context.complete = True
    monkeypatch.setattr(fixture, "iter_report_bundle", stream)
    return repository, bundle, reports, filing


def freeze(tmp_path, repository, bundle, **options):
    return fixture.freeze_fixture(bundle, repository.database_path, tmp_path / "frozen",
        expected_bundle_sha256=fixture.file_hash(bundle), evaluation_cutoff=EVALUATION, **options)


def test_fixture_retains_only_raw_operands_and_explicit_account(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    before = fixture.file_hash(repository.database_path), repository.database_path.stat().st_mtime_ns
    manifest = freeze(tmp_path, repository, bundle)
    assert manifest["symbols"] == ["3014"]
    assert manifest["account"]["positions"] == []
    assert manifest["account"]["available_cash"] == 1000000
    assert manifest["source_cutoffs"] == [CUTOFF]
    assert manifest["pae_state"]["prestate"] == "EMPTY"
    assert manifest["pae_state"]["live_order_authority"] is False
    assert manifest["gaps"] == []
    assert before == (fixture.file_hash(repository.database_path), repository.database_path.stat().st_mtime_ns)
    assert fixture.validate_fixture(tmp_path / "frozen") == manifest
    shard = fixture.load_symbol(tmp_path / "frozen", manifest, manifest["stocks"][0])
    assert set(shard) == {"schema", "symbol", "name", "market", "source_issuer", "source_cutoff",
                          "evaluation_cutoff", "account", "source_packet"}
    assert shard["source_packet"] == reports[0]["diagnostics"]["native_evidence"]["source_packet"]
    assert "quality" not in shard and "score" not in shard and "factor_ledger" not in shard
    target = tmp_path / "frozen" / manifest["mops"]["path"]
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM fundamental_snapshots").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM refresh_state").fetchone()[0] == 0
    assert (target.parent / filing.raw_path).read_bytes().startswith(b"<html>")
    assert all(not Path(entry["path"]).is_absolute() for entry in manifest["files"])


def test_later_metadata_and_exact_subsecond_cutoff_are_excluded(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    after = b"<html><body>SYNTHETIC_FUTURE</body></html>"
    repository.save_revision(FilingCandidate("3014", 2026, 3, "C", "https://example.invalid/future"),
        after, parse_ixbrl(after), available_at="2026-10-16T01:28:30.700000+00:00",
        fetched_at="2026-10-16T01:28:30.700000+00:00")
    with repository.unit_of_work() as unit:
        for number, observed in enumerate(("2026-10-16T01:28:30.600000+00:00", "2026-10-16T01:28:30.700000+00:00"), 1):
            unit.connection.execute("INSERT INTO financial_batch_snapshots "
                "(id,symbol,fiscal_year,quarter,content_sha256,observed_at,observed_at_epoch,snapshot_json) "
                "VALUES (?,?,?,?,?,?,?,?)", (number, "3014", 2026, 2, str(number) * 64,
                observed, int(fixture.timestamp(observed).timestamp()), "{}"))
    manifest = freeze(tmp_path, repository, bundle)
    with sqlite3.connect(tmp_path / "frozen" / manifest["mops"]["path"]) as connection:
        assert connection.execute("SELECT fiscal_year,quarter FROM filings").fetchall() == [(2026, 2)]
        assert connection.execute("SELECT id FROM financial_batch_snapshots").fetchall() == [(1,)]


def test_all_eligible_revisions_are_copied_and_not_latest_only(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    revised = b"<html><body>SYNTHETIC_REVISED</body></html>"
    repository.save_revision(FilingCandidate("3014", 2026, 2, "C", "https://example.invalid/revised"),
        revised, parse_ixbrl(revised), available_at="2026-09-01T01:00:00+00:00", fetched_at="2026-09-01T01:00:00+00:00")
    manifest = freeze(tmp_path, repository, bundle)
    assert manifest["counts"]["filing_rows"] == 2
    with sqlite3.connect(tmp_path / "frozen" / manifest["mops"]["path"]) as connection:
        assert connection.execute("SELECT revision_number FROM filings ORDER BY revision_number").fetchall() == [(1,), (2,)]


def test_missing_original_is_retained_as_a_gap_not_imputed(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    (repository.root / filing.raw_path).unlink()
    manifest = freeze(tmp_path, repository, bundle)
    assert manifest["counts"]["filing_rows"] == 1
    assert manifest["gaps"][0]["reason"] == "LOCAL_ORIGINAL_MISSING_NOT_REPAIRED"
    assert not (tmp_path / "frozen" / manifest["gaps"][0]["path"]).exists()
    fixture.validate_fixture(tmp_path / "frozen")


def test_corrupt_original_and_capacity_limits_fail_without_complete_manifest(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="TOTAL_RAW_CAPACITY"):
        freeze(tmp_path, repository, bundle, max_total_raw_bytes=1)
    assert not (tmp_path / "frozen" / "manifest.json").exists()
    (repository.root / filing.raw_path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SOURCE_CONTENT_HASH"):
        fixture.freeze_fixture(bundle, repository.database_path, tmp_path / "corrupt",
            expected_bundle_sha256=fixture.file_hash(bundle), evaluation_cutoff=EVALUATION)
    assert not (tmp_path / "corrupt" / "manifest.json").exists()


def test_filing_issuer_must_be_explicit_and_no_suffix_inference(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch, symbol="3036A", issuer="3036")
    manifest = freeze(tmp_path, repository, bundle)
    assert manifest["stocks"][0]["source_issuer"] == "3036"
    assert manifest["stocks"][0]["symbol"] == "3036A"
    assert manifest["counts"]["filing_rows"] == 1
    reports[0]["diagnostics"]["native_evidence"]["source_packet"]["filings"].append(
        dict(symbol="3014", available_at="2026-08-01T01:00:00+00:00"))
    with pytest.raises(ValueError, match="CONFLICTING_DECLARED_ISSUERS"):
        fixture.freeze_fixture(bundle, repository.database_path, tmp_path / "conflict",
            expected_bundle_sha256=fixture.file_hash(bundle), evaluation_cutoff=EVALUATION)


@pytest.mark.parametrize("failure", ["missing_account", "account_hash", "account_time", "future_clock", "market_hash"])
def test_invalid_bound_inputs_are_not_replaced_by_defaults(tmp_path, monkeypatch, failure):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    r = reports[0]
    expected = EVALUATION
    if failure == "missing_account":
        r["pae_authorization"]["account"].pop("positions")
    elif failure == "account_hash":
        r["pae_authorization"]["account_sha256"] = "0" * 64
    elif failure == "account_time":
        r["pae_authorization"]["account"]["as_of"] = EVALUATION
    elif failure == "future_clock":
        expected = "2026-10-15T00:00:00+00:00"
    else:
        r["diagnostics"]["evidence_binding"]["bars_hash"] = "0" * 64
    with pytest.raises(ValueError):
        fixture.freeze_fixture(bundle, repository.database_path, tmp_path / "frozen",
            expected_bundle_sha256=fixture.file_hash(bundle), evaluation_cutoff=expected)
    assert not (tmp_path / "frozen" / "manifest.json").exists()


def test_full_file_validation_and_shard_read_detect_tampering(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    manifest = freeze(tmp_path, repository, bundle)
    shard = tmp_path / "frozen" / manifest["stocks"][0]["source_packet_ref"]["path"]
    value = json.loads(shard.read_bytes()); value["account"]["available_cash"] = 5
    shard.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="FILE_CONTENT_MISMATCH"):
        fixture.validate_fixture(tmp_path / "frozen")
    with pytest.raises(ValueError, match="STOCK_SHARD_CHANGED"):
        fixture.load_symbol(tmp_path / "frozen", manifest, manifest["stocks"][0])


def test_business_id_stable_across_created_times_and_new_destination(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    first = freeze(tmp_path, repository, bundle)
    second = fixture.freeze_fixture(bundle, repository.database_path, tmp_path / "other",
        expected_bundle_sha256=fixture.file_hash(bundle), evaluation_cutoff=EVALUATION)
    assert first["created_at"] != second["created_at"]
    assert first["input_id"] == second["input_id"]
    with pytest.raises(ValueError, match="DESTINATION_ALREADY_EXISTS"):
        freeze(tmp_path, repository, bundle)


def test_requested_symbols_cannot_silently_disappear(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="REQUESTED_SCOPE_NOT_COMPLETE"):
        freeze(tmp_path, repository, bundle, symbols=["3014", "6834"])
    assert not (tmp_path / "frozen" / "manifest.json").exists()


def test_portable_monthly_loader_uses_retained_alias_not_original_path(tmp_path, monkeypatch):
    relative = "valuation-history/" + "a" * 64 + "/manifests/" + "a" * 64 + ".json"
    seen = []
    import infrastructure.crawler.valuation_history as monthly
    monkeypatch.setattr(monthly, "load_valuation_history", lambda path, root: seen.append((path, root)) or {"verified": True})
    manifest = dict(valuation_aliases={"/nonexistent/old/manifests/abc.json": relative})
    loader = fixture.valuation_capture_loader(tmp_path, manifest)
    assert loader(Path("/nonexistent/old/manifests/abc.json")) == {"verified": True}
    assert seen == [(tmp_path / relative, (tmp_path / relative).parent.parent)]
    with pytest.raises(ValueError, match="MONTHLY_ALIAS_NOT_FROZEN"):
        loader(Path("/unfrozen/manifest.json"))


def test_source_schema_column_order_is_not_a_requirement(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    original = fixture._table_columns
    def legacy_order(connection, table):
        result = original(connection, table)
        # Named insertion remains valid when a migrated source appends a column.
        return result[1:] + result[:1] if table == "filings" else result
    monkeypatch.setattr(fixture, "_table_columns", legacy_order)
    manifest = freeze(tmp_path, repository, bundle)
    assert manifest["counts"]["filing_rows"] == 1


def test_alias_paths_must_not_escape_fixture(tmp_path):
    loader = fixture.valuation_capture_loader(tmp_path, dict(valuation_aliases={"/old.json": "../outside.json"}))
    with pytest.raises(ValueError, match="RELATIVE_PATH_REQUIRED"):
        loader(Path("/old.json"))


def test_period_window_retains_five_annuals_and_nine_calendar_quarters():
    rows = [dict(fiscal_year=year, quarter=quarter) for year in range(2020, 2027)
            for quarter in range(1, 5) if (year, quarter) <= (2026, 2)]
    selected = fixture.selected_periods(rows)
    assert [(year, 4) for year in range(2021, 2026)] == [p for p in selected if p[1] == 4]
    assert (2024, 2) in selected and (2026, 2) in selected
    assert (2024, 1) not in selected and (2020, 4) not in selected
    assert len(selected) <= 14
    assert fixture.selected_periods(rows, 2) == [(2026, 1), (2026, 2)]


def test_missing_recent_quarter_is_not_replaced_by_older_visible_quarter():
    rows = [dict(fiscal_year=2026, quarter=2), dict(fiscal_year=2023, quarter=3)]
    selected = fixture.selected_periods(rows)
    assert (2026, 1) in selected and (2023, 3) not in selected


def test_unlisted_signed_binding_cannot_change_a_validated_fixture(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    freeze(tmp_path, repository, bundle)
    path = tmp_path / "frozen" / "mops" / "signed-audit" / "bindings" / "unlisted.json"
    path.parent.mkdir(parents=True); path.write_text("{}")
    with pytest.raises(ValueError, match="UNLISTED_OR_MISSING_FILES"):
        fixture.validate_fixture(tmp_path / "frozen")


def test_internal_symlink_cannot_bypass_immutable_file_inventory(tmp_path, monkeypatch):
    repository, bundle, reports, filing = source(tmp_path, monkeypatch)
    manifest = freeze(tmp_path, repository, bundle)
    root = tmp_path / "frozen"
    target = root / manifest["stocks"][0]["source_packet_ref"]["path"]
    original = root / "retained-shard.json"
    target.rename(original); target.symlink_to(original)
    with pytest.raises(ValueError, match="SYMLINK_NOT_ALLOWED"):
        fixture.validate_fixture(root)
