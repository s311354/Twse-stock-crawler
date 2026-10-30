"""SYNTHETIC source bytes exercise capture safety, not real membership evidence."""
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from application.industry_benchmark import load_industry_benchmarks
from application.replay_contracts import digest
from infrastructure.crawler.industry_index import (
    ENDPOINT, FIELDS, INDEX_COLUMNS, SCHEMA, PARSER_VERSION, SECTOR_SCHEMA, SECTOR_PARSER_VERSION,
    capture_index, capture_indices, months_in_range, parse_month, parse_sector_month, verify_capture,
)
from interface.industry_evidence import main


def payload():
    return dict(stat="OK", title="115年10月 電子類指數及金融保險類指數", date="20261001", fields=FIELDS[:],
        data=[["115/10/14", "3,050.05", *["1.00"] * 9], ["115/10/15", "2,933.03", *["1.00"] * 9]], total=2)


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class Client:
    def __init__(self):
        self.calls = []

    def request_bytes(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return SimpleNamespace(content=encode(payload()))


def capture(tmp_path, *, end="2026-10-14", clock="2026-10-15T23:00:00+08:00", client=None):
    return capture_index(tmp_path, start="2026-10-01", end=end,
        client=client or Client(), clock=lambda: datetime.fromisoformat(clock))


def test_exact_price_schema_and_monthly_rows_are_not_capped_by_request_date():
    rows = parse_month(encode(payload()), month="2026-10-01")
    assert rows == [dict(trade_date="2026-10-14", close=3050.05), dict(trade_date="2026-10-15", close=2933.03)]


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(stat="NO_DATA"),
    lambda p: p.update(title="115年10月 電子類報酬指數"),
    lambda p: p.update(date="20261014"),
    lambda p: p.update(total=3),
    lambda p: p["fields"].reverse(),
    lambda p: p["data"].append(p["data"][0]) or p.update(total=3),
    lambda p: p["data"][0].__setitem__(0, "115/09/14"),
    lambda p: p["data"][0].__setitem__(1, "30,50.05"),
    lambda p: p["data"][0].__setitem__(1, "NaN"),
    lambda p: p["data"][0].__setitem__(1, "--"),
    lambda p: p["data"][0].__setitem__(1, "0.00"),
])
def test_changed_or_conflicting_schema_is_not_silently_repaired(mutate):
    value = payload()
    mutate(value)
    with pytest.raises(ValueError):
        parse_month(encode(value), month="2026-10-01")


def test_budget_is_checked_before_network_and_rolls_over_year():
    assert months_in_range("2025-12-25", "2026-02-01") == ["2025-12-01", "2026-01-01", "2026-02-01"]
    client = Client()
    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        capture_index("unused", start="2026-01-01", end="2026-10-14", client=client)
    assert client.calls == []


def test_capture_filters_future_rows_keeps_original_bytes_and_never_certifies_membership(tmp_path):
    client = Client()
    path = capture(tmp_path, client=client)
    result = verify_capture(path, tmp_path)
    assert len(client.calls) == 1 and client.calls[0][:2] == ("GET", ENDPOINT)
    assert result["status"] == "PARTIAL" and result["eligible_for_industry_scoring"] is False
    assert result["membership_evidence"] == [] and result["historical_information_set_certified"] is False
    assert len(result["rows"]) == 1 and result["rows"][0]["close"] == 3050.05
    row = result["rows"][0]
    assert row["available_at"] == "2026-10-15T23:00:00+08:00"
    assert row["original_publication_time"] is None and row["original_revision"] is None
    assert result["excluded_rows"][0]["trade_date"] == "2026-10-15"
    assert result["excluded_rows"][0]["reason"] == "OUTSIDE_REQUESTED_DATE_RANGE"
    original = tmp_path / "objects" / (result["documents"][0]["sha256"] + ".json")
    assert original.read_bytes() == encode(payload())
    # The capture is deliberately NOT accepted as an R11 raw-source mapping.
    with pytest.raises(ValueError):
        load_industry_benchmarks(path)
    assert capture(tmp_path) == path  # Same observation is idempotent, not overwritten.
    assert len(list((tmp_path / "captures").iterdir())) == 1


def test_intraday_cutoff_cannot_use_same_day_close_even_if_endpoint_returns_it(tmp_path):
    path = capture(tmp_path, end="2026-10-15", clock="2026-10-15T12:00:00+08:00")
    result = verify_capture(path, tmp_path)
    assert [r["trade_date"] for r in result["rows"]] == ["2026-10-14"]
    assert result["excluded_rows"][0]["reason"] == "AFTER_CUTOFF_SESSION_CLOSE"


def test_work_crossing_close_does_not_promote_intraday_observation_to_final(tmp_path):
    values = iter([datetime.fromisoformat("2026-10-15T12:00:00+08:00"),
                   datetime.fromisoformat("2026-10-15T14:00:00+08:00")])
    path = capture_index(tmp_path, start="2026-10-01", end="2026-10-15", client=Client(), clock=lambda: next(values))
    result = verify_capture(path, tmp_path)
    assert [r["trade_date"] for r in result["rows"]] == ["2026-10-14"]
    assert result["excluded_rows"][0]["reason"] == "DOCUMENT_OBSERVED_BEFORE_SESSION_CLOSE"


def test_empty_month_stays_a_gap_and_never_means_market_closed(tmp_path):
    class Empty(Client):
        def request_bytes(self, *args, **kwargs):
            return SimpleNamespace(content=encode(dict(payload(), data=[], total=0)))
    result = verify_capture(capture(tmp_path, client=Empty()), tmp_path)
    assert result["rows"] == []
    assert "EMPTY_INDEX_MONTH:2026-10-01" in result["gaps"]
    assert "NO_INDEX_ROWS_IN_REQUESTED_RANGE" in result["gaps"]
    assert result["missing_rows_policy"] == "NO_MARKET_CLOSURE_INFERENCE"


def test_rehashed_cutoff_before_download_excludes_all_observations(tmp_path):
    path = capture(tmp_path)
    packet = json.loads(path.read_text())
    packet["cutoff"] = "2026-10-14T23:00:00+08:00"
    packet["capture_id"] = digest({k: v for k, v in packet.items() if k != "capture_id"})
    other = tmp_path / "edited.json"
    other.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="EXTRACTION_MISMATCH"):
        verify_capture(other, tmp_path)


def test_source_tamper_new_capture_and_offline_cli(tmp_path, capsys):
    path = capture(tmp_path)
    second = capture(tmp_path, clock="2026-10-16T10:00:00+08:00")
    assert second != path and path.exists()
    assert len(list((tmp_path / "objects").iterdir())) == 1
    assert main(["--root", str(tmp_path), "--verify", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requests"] == 0 and output["eligible_for_industry_scoring"] is False
    checked = verify_capture(path, tmp_path)
    original = tmp_path / "objects" / (checked["documents"][0]["sha256"] + ".json")
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match="ORIGINAL_HASH_MISMATCH"):
        verify_capture(path, tmp_path)


@pytest.mark.parametrize("field,value", [("sector", "24"), ("market", "TPEX"),
                                         ("index_type", "TOTAL_RETURN"), ("eligible_for_industry_scoring", True)])
def test_rehashed_claims_cannot_relabel_index_or_certify_membership(tmp_path, field, value):
    path = capture(tmp_path)
    packet = json.loads(path.read_text())
    packet[field] = value
    packet["capture_id"] = digest({k: v for k, v in packet.items() if k != "capture_id"})
    other = tmp_path / "edited.json"
    other.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        verify_capture(other, tmp_path)


def named_payload():
    value = payload()
    for row in value["data"]:
        for column in range(1, len(FIELDS)):
            row[column] = f"{column * 100}.25"
    return value


@pytest.mark.parametrize("sector,column", [("13", 1), ("24", 2), ("25", 3), ("26", 4),
    ("27", 5), ("28", 6), ("29", 7), ("30", 8), ("31", 9), ("SEMI", 2),
    ("ELECCOMP", 6), ("通信網路類指數", 5)])
def test_named_sector_parser_uses_exact_official_column_not_electronics_default(sector, column):
    rows = parse_sector_month(encode(named_payload()), month="2026-10-01", sector=sector)
    assert rows == [dict(trade_date=day, close=column * 100 + .25)
                    for day in ("2026-10-14", "2026-10-15")]
    assert parse_month(encode(named_payload()), month="2026-10-01") == parse_sector_month(
        encode(named_payload()), month="2026-10-01", sector="13")


@pytest.mark.parametrize("sector", [None, True, "", "999", "金融保險類指數", "guess-28"])
def test_unsupported_sector_rejected_without_network(tmp_path, sector):
    client = Client()
    with pytest.raises(ValueError, match="UNSUPPORTED_OFFICIAL_INDUSTRY_SECTOR"):
        capture_indices(tmp_path, sectors=[sector], start="2026-10-01", end="2026-10-14", client=client)
    assert client.calls == []


@pytest.mark.parametrize("sectors,error", [([], "EXPLICIT"), ("24", "EXPLICIT"),
    (["24", "SEMI"], "DUPLICATE"), (["13", "電子類指數"], "DUPLICATE")])
def test_sector_scope_is_explicit_unique_and_checked_before_network(tmp_path, sectors, error):
    client = Client()
    with pytest.raises(ValueError, match=error):
        capture_indices(tmp_path, sectors=sectors, start="2026-10-01", end="2026-10-14", client=client)
    assert client.calls == []


@pytest.mark.parametrize("bad", ["--", "NaN", "30,50.05", "0.00", 300.])
def test_bad_requested_subindex_is_not_replaced_by_valid_broad_index(bad):
    value = named_payload(); value["data"][0][6] = bad
    assert parse_month(encode(value), month="2026-10-01")
    with pytest.raises(ValueError, match="INVALID_INDUSTRY_PRICE_VALUE"):
        parse_sector_month(encode(value), month="2026-10-01", sector="28")


def test_multiple_subindices_share_one_download_actual_observation_and_original(tmp_path):
    class NamedClient(Client):
        def request_bytes(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return SimpleNamespace(content=encode(named_payload()))
    client = NamedClient()
    clock = lambda: datetime.fromisoformat("2026-10-28T10:00:00+08:00")
    paths = capture_indices(tmp_path, sectors=["24", "27", "28"], start="2026-10-01", end="2026-10-14",
        client=client, clock=clock)
    assert len(client.calls) == 1 and len(list((tmp_path / "objects").iterdir())) == 1
    assert set(paths) == {"24", "27", "28"} and len(set(paths.values())) == 3
    records = [verify_capture(path, tmp_path) for path in paths.values()]
    assert all(r["schema"] == SECTOR_SCHEMA and r["parser_version"] == SECTOR_PARSER_VERSION for r in records)
    assert [r["rows"][0]["close"] for r in records] == [200.25, 500.25, 600.25]
    assert all(r["status"] == "PARTIAL" and r["membership_evidence"] == [] and
        r["eligible_for_industry_scoring"] is False for r in records)
    assert all(r["rows"][0]["available_at"] == clock().isoformat() for r in records)
    assert all(r["rows"][0]["original_publication_time"] is None for r in records)
    assert records[0]["documents"] == records[1]["documents"] == records[2]["documents"]
    assert all(r["index_name"] == INDEX_COLUMNS[r["sector"]] for r in records)
    # Explicit v2 capture is idempotent; default capture still emits exact v1.
    assert paths == capture_indices(tmp_path, sectors=["24", "27", "28"], start="2026-10-01", end="2026-10-14",
        client=NamedClient(), clock=clock)
    legacy = verify_capture(capture_index(tmp_path, start="2026-10-01", end="2026-10-14",
        client=NamedClient(), clock=clock), tmp_path)
    assert legacy["schema"] == SCHEMA and legacy["parser_version"] == PARSER_VERSION
    assert legacy["sector"] == "13" and "index_selection" not in legacy
    assert legacy["rows"][0]["close"] == 100.25
    with pytest.raises(ValueError):
        load_industry_benchmarks(paths["24"])


@pytest.mark.parametrize("field,value", [("sector", "27"), ("index_name", "電子類指數"),
    ("index_selection", {}), ("eligible_for_industry_scoring", True), ("parser_version", PARSER_VERSION)])
def test_rehashed_named_capture_cannot_switch_column_or_certify_membership(tmp_path, field, value):
    path = capture_index(tmp_path, start="2026-10-01", end="2026-10-14", sector="24",
        client=Client(), clock=lambda: datetime.fromisoformat("2026-10-15T23:00:00+08:00"))
    packet = json.loads(path.read_text()); packet[field] = value
    packet["capture_id"] = digest({k:v for k,v in packet.items() if k != "capture_id"})
    edited = tmp_path / "edited.json"; edited.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        verify_capture(edited, tmp_path)


def test_cli_can_explicitly_capture_shared_subindices_and_verify_without_relabeling(tmp_path, monkeypatch, capsys):
    from interface import industry_evidence
    real_capture = capture_indices
    calls = []
    def captured(root, **kwargs):
        calls.append(kwargs)
        return real_capture(root, **kwargs, client=Client(),
            clock=lambda: datetime.fromisoformat("2026-10-15T23:00:00+08:00"))
    monkeypatch.setattr(industry_evidence, "capture_indices", captured)
    assert main(["--capture", "--root", str(tmp_path), "--start", "2026-10-01", "--end", "2026-10-14",
        "--sectors", "24", "27", "28"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requests"] == 1 and [r["sector"] for r in output["captures"]] == ["24", "27", "28"]
    manifest = output["captures"][0]["manifest"]
    assert main(["--verify", manifest, "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["sector"] == "24"
    with pytest.raises(SystemExit):
        main(["--verify", manifest, "--root", str(tmp_path), "--sectors", "13"])


def test_multiple_sectors_obey_month_request_budget_not_sector_times_months(tmp_path):
    class Monthly(Client):
        def request_bytes(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            requested = kwargs["params"]["date"]
            year, month = int(requested[:4]), int(requested[4:6])
            value = named_payload()
            value.update(date=requested, title=f"{year-1911}年{month:02d}月 電子類指數及金融保險類指數")
            for day, row in enumerate(value["data"], 1):
                row[0] = f"{year-1911:03d}/{month:02d}/{day:02d}"
            return SimpleNamespace(content=encode(value))
    client = Monthly()
    paths = capture_indices(tmp_path, sectors=["24", "27", "28"], start="2026-07-01", end="2026-10-27",
        max_months=4, client=client, clock=lambda: datetime.fromisoformat("2026-10-28T10:00:00+08:00"))
    assert len(client.calls) == 4
    assert len(list((tmp_path / "objects").iterdir())) == 4
    for path in paths.values():
        value = verify_capture(path, tmp_path)
        assert len(value["documents"]) == 4 and len(value["rows"]) == 8
    blocked = Monthly()
    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        capture_indices(tmp_path, sectors=["24", "27", "28"], start="2026-07-01", end="2026-10-27",
            max_months=3, client=blocked)
    assert blocked.calls == []


def test_bad_requested_sector_stops_before_publishing_any_partial_capture(tmp_path):
    class Bad(Client):
        def request_bytes(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            value = named_payload(); value["data"][0][6] = "--"
            return SimpleNamespace(content=encode(value))
    client = Bad()
    with pytest.raises(ValueError, match="INVALID_INDUSTRY_PRICE_VALUE"):
        capture_indices(tmp_path, sectors=["24", "28"], start="2026-10-01", end="2026-10-14", client=client)
    assert len(client.calls) == 1
    assert not list((tmp_path / "captures").glob("*.json"))
