"""Synthetic fee controls; no proof of real fills or strategy efficacy."""
from copy import deepcopy
from decimal import Decimal
import gzip
import json
from types import SimpleNamespace

import pytest

from application.exit_fee_sensitivity import compare_public_fees, quote_order
from interface.exit_fee_sensitivity import SOURCES, capture, load_sources, main, validate_source


def test_minimum_is_per_order_not_per_partial_fill_and_unfilled_is_free():
    fills = [dict(shares=1, price="100.60"), dict(shares=1, price="100.60")]
    q = quote_order("BUY", fills, rate=".001425", minimum="20")
    assert q["settlement_gross"] == "201" and q["commission"] == "20"
    assert q["cash_delta"] == "-221" and q["price_groups"] == 1
    assert quote_order("BUY", [], rate=".001425", minimum="20")["fees"] == "0"


def test_settlement_groups_identical_price_not_all_prices_or_each_fill():
    fills = [dict(shares=1, price="26.65"), dict(shares=229, price="26.65"),
             dict(shares=75, price="26.75"), dict(shares=295, price="26.85")]
    q = quote_order("BUY", fills, rate=".000498", minimum="20")
    assert q["gross_raw"] == "16056.50" and q["settlement_gross"] == "16055"
    assert q["price_groups"] == 3
    assert q["commission"] == "20" and q["cash_delta"] == "-16075"


def test_tax_and_commission_are_separate_and_rounding_is_explicit():
    f = [dict(shares=20, price=100)]
    floor = quote_order("SELL", f, rate=".001425", minimum="1")
    half = quote_order("SELL", f, rate=".001425", minimum="1", commission_rounding="HALF_UP")
    assert floor["commission"] == "2" and half["commission"] == "3"
    assert floor["sell_tax"] == half["sell_tax"] == "6"
    assert floor["cash_delta"] == "1992" and half["cash_delta"] == "1991"


@pytest.mark.parametrize("qty,price", [(True, 10), (0, 10), (-1, 10), (1.5, 10),
    (1, None), (1, "NaN"), (1, "Infinity"), (1, -1), (1, "10.005")])
def test_unknown_or_invalid_inputs_are_not_zero_cost(qty, price):
    with pytest.raises(ValueError, match="FEE_RESEARCH"):
        quote_order("BUY", [dict(shares=qty, price=price)], rate=".001425", minimum="1")


def run():
    return dict(jobId="SYNTHETIC_ONLY", initialCapital=1000000, summary=dict(nav=1000000, fees=14.55),
        trades=[dict(id=1, date="2026-09-01", code="9999", shares=20, price=100, amount=2000,
                     side="BUY", fee=2.85, priceBasis="OPEN"),
                dict(id=2, date="2026-09-02", code="9999", shares=20, price=100, amount=2000,
                     side="SELL", fee=8.85, priceBasis="OPEN"),
                dict(id=3, date="2026-09-03", code="9999", shares=20, price=100, amount=2000,
                     side="BUY", fee=2.85, priceBasis="OPEN")])


def test_original_account_is_not_replaced_by_fixed_fill_cost_sensitivity():
    source = run()
    before = deepcopy(source)
    r = compare_public_fees(source)
    assert source == before and len(r["scenarios"]) == 6
    assert not r["full_account_comparison_complete"] and not r["approved_for_live"]
    assert r["execution_diagnostic"]["odd_lot_sized_rows"] == 3
    assert not r["execution_diagnostic"]["order_ids_available"]
    low, high = r["scenarios"][0], r["scenarios"][2]
    assert low["total_fee_twd"] == "12" and high["total_fee_twd"] == "66"
    assert Decimal(low["conditional_pnl_bridge_twd"]) == Decimal("2.55")
    assert Decimal(high["conditional_pnl_bridge_twd"]) == Decimal("-51.45")
    assert "NOT_REEVOLVED" in high["result_type"]
    assert not r["execution_diagnostic"]["corporate_action_accounting_verified"]
    assert compare_public_fees(source) == r


@pytest.mark.parametrize("mutate", [
    lambda r: r["trades"][1].update(date="2026-09-01"),
    lambda r: r["trades"][1].update(id=1),
    lambda r: r["trades"][0].update(amount=1999),
    lambda r: r["summary"].update(fees=1),
    lambda r: r["trades"][0].update(fee=-1),
])
def test_unsupported_or_conflicting_inputs_are_rejected(mutate):
    source = run()
    mutate(source)
    with pytest.raises(ValueError, match="FEE_RESEARCH"):
        compare_public_fees(source)


class SyntheticClient:
    transport = "SYNTHETIC_ONLY_NOT_FOR_ACTUAL_RESULTS"

    def __init__(self):
        self.calls = 0

    def request_bytes(self, method, url, **options):
        self.calls += 1
        markers = next(row[1] for row in SOURCES.values() if row[0] == url)
        content = ("<html>" + " ".join(markers) + "</html>").encode()
        options["content_validator"](content)
        return SimpleNamespace(content=content)


def test_public_source_capture_is_immutable_and_warm_read_verifies_originals(tmp_path):
    client = SyntheticClient()
    first = capture(tmp_path, client=client)
    assert client.calls == len(SOURCES)
    assert capture(tmp_path, client=client) == first
    assert client.calls == len(SOURCES)
    assert all(row["historical_available_at"] is None for row in first["sources"])
    sha = first["sources"][0]["body_ref"]["sha256"]
    (tmp_path / "objects" / sha).write_bytes(b"CORRUPTION")
    with pytest.raises(ValueError):
        load_sources(tmp_path / "sources.json")


def test_missing_public_claim_and_security_block_are_not_verified_sources():
    with pytest.raises(ValueError, match="CLAIM_NOT_FOUND"):
        validate_source("yuanta_odd", b"<html>maintenance</html>")
    with pytest.raises(ValueError, match="SOURCE_BLOCKED"):
        validate_source("yuanta_odd", b"Access Denied")


def test_cli_is_write_once_and_fixed_source_and_result_are_distinct(tmp_path, capsys):
    capture(tmp_path / "public", client=SyntheticClient())
    source = tmp_path / "run.json.gz"
    source.write_bytes(gzip.compress(json.dumps(run()).encode()))
    original = source.read_bytes()
    target = tmp_path / "result.json"
    args = ["compare", "--bank-run", str(source), "--sources", str(tmp_path / "public/sources.json"),
            "--output", str(target)]
    assert main(args) == 0 and main(args) == 0
    assert source.read_bytes() == original
    result = json.loads(target.read_bytes())
    assert result["public_sources"]["source_id"]
    assert result["execution_objects"]["source"]
    assert not result["full_account_comparison_complete"]
    with pytest.raises(SystemExit):
        main(args[:-1] + [str(source)])
    target.write_text("{}")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        main(args)
