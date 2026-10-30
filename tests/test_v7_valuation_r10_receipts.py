"""SYNTHETIC ONLY: retained-source bindings and R10 fallback, no market repair."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity
from domain.les_v7.valuation_earnings import valuation_earnings
from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt, RECOVERY_PRODUCER
from domain.les_v7.valuation_recovery_r10 import policy, select_recovery, validate_comparables
from domain.strategies.low_entry_v6_filings import ixbrl_record
from infrastructure.crawler.security_master import capture_security_master, select_security_master
from tests.test_security_master import synthetic_instruments, synthetic_issuers
from tests.test_v7_eps_price_receipts import item

CUTOFF = "2026-10-16T13:30:00+08:00"
SYMBOLS = [str(1000 + index) for index in range(11)]


def annual(symbol, year):
    facts = tuple(FiledFact(concept, value, None, f"{yr}-01-01", f"{yr}-12-31", None,
        unit="iso4217:TWD/xbrli:shares" if "Share" in concept else "iso4217:TWD",
        context_id=f"SYNTHETIC_{yr}") for yr in (year, year - 1) for concept, value in (
            ("ifrs-full:BasicEarningsLossPerShare", 12.), ("ifrs-full:DilutedEarningsLossPerShare", 12.),
            ("ifrs-full:Revenue", 120000000.), ("ifrs-full:ProfitLossFromOperatingActivities", 30000000.)))
    filing = FiledFinancials(year, 4, f"{year + 1}-03-10T05:00:00Z", 0, facts,
        content_sha256=identity(dict(SYNTHETIC_ONLY=True, symbol=symbol, year=year)),
        source_url="https://mopsov.twse.com.tw/SYNTHETIC_ONLY.html", period_end=f"{year}-12-31")
    row = dict(ixbrl_record(symbol, filing), market="TWSE")
    row["earnings_source"] = valuation_earnings(filing, symbol=symbol, original_verified=True, as_of=CUTOFF)
    return row


def context(*, cutoff=CUTOFF, observed_at="2026-10-15T06:00:00Z"):
    rows = [("股票", [f"{symbol}　合成普通股", f"TW000{symbol}008", "1994/09/05", "上市", "半導體業", "ESVUFR", ""])
            for symbol in SYMBOLS]
    issuers = [dict(公司代號=symbol, 公司簡稱="合成普通股", 公司名稱="合成股份有限公司", 營利事業統一編號="12345678", 上市日期="19940905")
               for symbol in SYMBOLS]
    master = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(rows=rows),
        issuer_content=synthetic_issuers(rows=issuers), fetched_at=observed_at)
    blocks = []
    for index, symbol in enumerate(SYMBOLS):
        dates = pd.date_range("2023-10-01", "2026-09-01", freq="MS") + pd.Timedelta(days=20)
        prices = [dict(Date=day.date().isoformat(), Close=120.) for day in dates] if index == 0 else []
        prices.append(dict(Date="2026-10-15", Close=120.))
        blocks.append(dict(symbol=symbol, security_evidence=select_security_master(master, symbol, cutoff),
            market_rows=prices, market_evidence=dict(symbol=symbol, market="TWSE", session_date="2026-10-15",
                source="TWSE", price_unit="TWD", volume_unit="shares", amount_unit="TWD", source_final=True,
                bars_sha256=identity(prices), source_available_at=observed_at, source_record_ids=[identity(prices)]),
            annual_filings=[annual(symbol, year) for year in range(2022, 2026)]))
    return dict(schema="les-v7-recovery-context/v1", symbol=SYMBOLS[0], market="TWSE", cutoff=cutoff,
                issuer=blocks[0], peers=blocks[1:], universe_symbols=SYMBOLS, exclusions=[])


def validate(data):
    original = next(row for row in data["issuer"]["annual_filings"]
                    if row["period_end"] == "2025-12-31" and row["available_at"] < CUTOFF)
    return validate_comparables(data, symbol=SYMBOLS[0], market="TWSE", cutoff=CUTOFF,
        eps_basis="DILUTED", fiscal_year=2025, scope="consolidated", annual_source=original)


def test_native_source_complete_comparables_derive_medians_and_preserve_input():
    data = context(); before = deepcopy(data)
    result = validate(data)
    assert result["status"] == "VERIFIED"
    assert result["peer_count"] == 10 and result["issuer_month_count"] == 36
    assert result["peer_pe_median"] == result["issuer_pe_median"] == 10.
    assert all(row["annual"]["year"] <= int(row["quote_date"][:4]) for row in result["monthly_sources"])
    assert data == before
    assert validate(ReadOnlyEvidence(data)) == result


def test_annual_eps_unavailable_at_month_is_not_backfilled_from_latest_year():
    data = context()
    data["issuer"]["annual_filings"] = data["issuer"]["annual_filings"][-1:]
    result = validate(data)
    assert result["status"] == "INSUFFICIENT" and result["issuer_month_count"] < 24


@pytest.mark.parametrize("damage", ["cutoff", "hash", "market", "duplicates", "future_quote", "security", "annual_conflict"])
def test_sourced_conflicts_remain_unknown_not_favorable_fallback(damage):
    data = context()
    if damage == "cutoff": data["cutoff"] = "2027-01-01T00:00:00Z"
    elif damage == "hash": data["issuer"]["market_rows"][-1]["Close"] = 999.
    elif damage == "market": data["issuer"]["market_evidence"]["market"] = "TPEX"
    elif damage == "duplicates": data["peers"].append(deepcopy(data["peers"][0]))
    elif damage == "future_quote":
        block = data["issuer"]; block["market_rows"][-1]["Date"] = "2027-01-01"
        block["market_evidence"].update(session_date="2027-01-01", bars_sha256=identity(block["market_rows"]), source_record_ids=[identity(block["market_rows"])])
    elif damage == "security": data["issuer"]["security_evidence"]["instrument"]["sector"] = "電子通路業"
    else:
        conflict = deepcopy(data["issuer"]["annual_filings"][-1]); conflict["revision_id"] = "SYNTHETIC_CONFLICT"
        data["issuer"]["annual_filings"].append(conflict)
    assert validate(data)["status"] == "UNKNOWN"


def test_future_revision_cannot_change_existing_monthly_or_current_comparables():
    data = context(); before = validate(data)
    future = deepcopy(data["issuer"]["annual_filings"][-1]); future.update(available_at="2027-01-01T00:00:00Z", revision_id="FUTURE")
    data["issuer"]["annual_filings"].append(future)
    # Context content identity changes; verified business operands and monthly
    # source selection do not. Latest future revision must not win at cutoff.
    after = validate(data)
    assert after["status"] == "VERIFIED"
    for key in ("peer_pe_median", "issuer_pe_median", "monthly_sources", "peer_sources"):
        assert after[key] == before[key]


@pytest.mark.parametrize("damage", ["companion_hash", "eps_proof_hash", "eps_context_unit"])
def test_present_conflicting_peer_eps_proof_is_not_dropped_for_more_favorable_peers(damage):
    data = context(); row = data["peers"][0]["annual_filings"][-1]
    if damage == "companion_hash": row["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "eps_proof_hash": row["earnings_source"]["field_evidence"]["diluted_eps"]["content_sha256"] = "0" * 64
    else: row["earnings_source"]["field_evidence"]["diluted_eps"]["contexts"][0]["unit"] = "iso4217:TWD"
    assert validate(data)["status"] == "UNKNOWN"


def test_known_nonpositive_peer_earnings_is_incomparable_not_fabricated_pe():
    data = context(); row = data["peers"][0]["annual_filings"][-1]
    row["earnings_source"]["fields"]["diluted_eps"] = -1.
    proof = row["earnings_source"]["field_evidence"]["diluted_eps"]
    proof["contexts"][0]["numeric_value"] = -1.
    result = validate(data)
    assert result["status"] == "INSUFFICIENT" and result["peer_count"] == 9
    assert result["peer_exclusions"][0]["reason"] == "INCOMPARABLE_ANNUAL_OR_QUOTE_SESSION"


def test_missing_peer_coverage_explicitly_uses_old_method_receipt_and_does_not_relax_unknown():
    data = item(); data.update(cutoff=CUTOFF, policy=policy())
    result = produce_price_valuation_receipt(**data)
    assert result["gate_version"] == RECOVERY_PRODUCER
    diagnostic = result["valuation_diagnostics"]
    assert diagnostic["selected_method"] == "ORIGINAL_R1_FALLBACK"
    assert diagnostic["scenarios"] == diagnostic["fallback_result"]["scenarios"]
    data["bars"]["PE"] = None
    assert produce_price_valuation_receipt(**data)["state"] == "UNKNOWN"


def recent(*, cfo=10., revenue=130., income=25.):
    facts = tuple(FiledFact(concept, value * 1000, None, f"{year}-01-01", f"{year}-06-30", None,
        unit="iso4217:TWD", context_id=f"SYNTHETIC_{year}") for year in (2026, 2025) for concept, value in (
            ("ifrs-full:Revenue", revenue if year == 2026 else 100.),
            ("ifrs-full:ProfitLossFromOperatingActivities", income if year == 2026 else 20.),
            ("ifrs-full:CashFlowsFromUsedInOperatingActivities", cfo)))
    filing = FiledFinancials(2026, 2, "2026-08-14T05:00:00Z", 0, facts,
        content_sha256=identity(dict(SYNTHETIC_ONLY=True, cfo=cfo, revenue=revenue, income=income)),
        source_url="https://mopsov.twse.com.tw/SYNTHETIC_YTD.html", period_end="2026-06-30")
    return dict(ixbrl_record(SYMBOLS[0], filing), market="TWSE")


def recovery(data):
    return select_recovery(data, symbol=SYMBOLS[0], market="TWSE", cutoff=CUTOFF, scope="consolidated")


def test_same_original_ytd_recovery_resolves_real_native_fields_without_eps_annualization():
    row = recent(); result = recovery([row])
    assert result["status"] == "VERIFIED"
    assert result["values"] == dict(current_revenue=130., prior_revenue=100., current_operating_income=25., prior_operating_income=20., current_cfo=10.)
    assert "eps" not in result["values"]


@pytest.mark.parametrize("changes", [dict(cfo=-1.), dict(cfo=0.), dict(revenue=95.), dict(income=18.)])
def test_verified_failed_recovery_conditions_earn_no_credit_not_missing(changes):
    assert recovery([recent(**changes)])["status"] == "NO_RECOVERY"


def test_missing_latest_ytd_does_not_use_older_favorable_original_or_future_source():
    row = recent(); missing = deepcopy(row); missing["available_at"] = "2026-10-15T05:00:00Z"
    missing["field_evidence"].pop("cfo")
    assert recovery([row, missing])["status"] == "UNKNOWN"  # Other retained proofs still bind the old revision time.
    missing["available_at"] = "2027-01-01T00:00:00Z"
    assert recovery([row, missing]) == recovery([row])


def test_unknown_recovery_timestamp_is_not_assumed_historical():
    row = recent(); row["available_at"] = "bad"
    assert recovery([row])["status"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["comparative_hash", "scope", "cfo_unit"])
def test_present_ytd_conflict_is_not_optional_missing_or_recovery_credit(damage):
    row = recent()
    if damage == "comparative_hash": row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
    elif damage == "scope": row["scope"] = "individual"
    else: row["field_evidence"]["cfo"]["contexts"][0]["unit"] = "xbrli:shares"
    assert recovery([row])["status"] == "UNKNOWN"


def test_unknown_or_backdated_r10_policy_rejected():
    data = item(); data.update(cutoff="2026-10-14T05:30:00Z", policy=policy())
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        produce_price_valuation_receipt(**data)
    data["cutoff"] = CUTOFF; data["policy"]["base"]["years"] = 1
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        produce_price_valuation_receipt(**data)


def test_same_date_final_quote_cannot_be_used_before_its_completed_close():
    data = context(); early = "2026-10-15T10:00:00+08:00"
    # Keep compact security proofs valid at the original cutoff; observation
    # too early is independently invalid even when the date isn't future.
    block = data["issuer"]
    block["market_evidence"]["source_available_at"] = early
    assert validate(data)["status"] == "UNKNOWN"
    from domain.les_v7.valuation_recovery_r10 import _block
    data = context(cutoff=early, observed_at="2026-10-15T01:00:00Z")
    # Valid prior-to-cutoff synthetic identity/capture alone cannot authorize
    # today's not-yet-completed final quote at an early cutoff.
    with pytest.raises(ValueError, match="QUOTE_CLOSE_NOT_AVAILABLE"):
        _block(data["issuer"], market="TWSE", cutoff=early)


@pytest.mark.parametrize("price,session", [(121., "2026-10-15"), (120., "2026-10-14")])
def test_valid_comparison_sources_cannot_substitute_a_different_strategy_quote(price, session):
    data = context()
    result = validate_comparables(data, symbol=SYMBOLS[0], market="TWSE", cutoff=CUTOFF,
        eps_basis="DILUTED", fiscal_year=2025, scope="consolidated", current_price=price,
        expected_session=session)
    assert result["status"] == "UNKNOWN"
    assert result["reasons"] == ["R10_ISSUER_QUOTE_STRATEGY_BINDING_CONFLICT"]


def test_receipt_passes_actual_strategy_quote_binding_to_comparison_validator():
    ctx = context(); row = ctx["issuer"]["annual_filings"][-1]
    data = item(row); data.update(symbol=SYMBOLS[0], cutoff=CUTOFF, policy=policy(), features=dict(close=120.))
    data["bars"]["Date"] = "2026-10-15"
    data["packet"]["r10_valuation_context"] = ctx
    assert produce_price_valuation_receipt(**data)["valuation_diagnostics"]["selected_method"] == "RECOVERY_INDUSTRY_R10"
    data["features"]["close"] = 121.
    result = produce_price_valuation_receipt(**data)
    assert result["state"] == "UNKNOWN" and result["valuation_diagnostics"]["scenarios"] is None
