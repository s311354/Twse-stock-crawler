"""SYNTHETIC ONLY: exact-context R2 source mapping, not repaired official data."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from domain.fundamental_derivation import FiledFact
from domain.financial_taxonomy import ixbrl_schema_refs
from domain.les_v7.financial_sources import (EXTRA_FIELDS, FINANCIAL_MAPPING, PAYMENT_POLICY,
    audit_proof, financial_supplement, ixbrl_record_r2, payment_outflow)
from domain.strategies.low_entry_v6_filings import ixbrl_record
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from tests.fundamentals.test_ixbrl_units import filed
from tests.fundamentals.test_taxonomy_cash_debt import cash_document


CUTOFF = "2026-10-12T13:30:00+08:00"


def original():
    body = cash_document()
    base = replace(filed(parse_ixbrl(body)), content_sha256=hashlib.sha256(body).hexdigest(),
        source_url="https://mopsov.twse.com.tw/SYNTHETIC_ONLY.html", period_end="2026-06-30",
        taxonomy_schema_refs=ixbrl_schema_refs(body))
    additions = []
    for field, concepts in EXTRA_FIELDS.items():
        instant = field in {"parent_equity", "accounts_receivable", "inventory"}
        amount = -200000. if field in {"intangible_capex", "dividends", "share_repurchases"} else 300000.
        additions.append(FiledFact(concepts[0], amount, None,
            None if instant else "2026-01-01", None if instant else "2026-06-30",
            "2026-06-30" if instant else None, unit="iso4217:TWD", context_id="SYNTHETIC_ONLY"))
    additions.append(FiledFact("tifrs-ar:UnqualifiedOpinionAbstract", None, "Y", None,
                              "2026-06-30", None, context_id="SYNTHETIC_ONLY_AUDIT"))
    return replace(base, facts=(*base.facts, *additions))


def test_r2_maps_new_exact_fields_without_changing_legacy_contract():
    source = original()
    before = ixbrl_record("2330", source)
    row = ixbrl_record_r2("2330", source, original_verified=True)
    assert not set(EXTRA_FIELDS) & set(before)
    assert ixbrl_record("2330", source) == before
    assert row["parent_equity"] == row["parent_net_income"] == row["interest_expense"] == 300.
    assert row["intangible_capex"] == -200.
    assert row["financial_mapping_version"] == FINANCIAL_MAPPING
    assert row["audit_evidence"]["status"] == "VERIFIED_REPORTED"
    assert row["receivable_basis"] == "REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES"


def test_source_mapping_cannot_be_mutated_in_a_running_job():
    with pytest.raises(TypeError): EXTRA_FIELDS["interest_expense"] = ("ifrs-full:FinanceCosts",)


def test_finance_costs_cannot_substitute_reported_interest_expense():
    source = original()
    facts = tuple(replace(f, concept="ifrs-full:FinanceCosts") if f.concept == "tifrs-notes:InterestExpense_n" else f for f in source.facts)
    row = ixbrl_record_r2("2330", replace(source, facts=facts), original_verified=True)
    assert row["interest_expense"] is None
    assert row["field_evidence"]["interest_expense"]["reason"] == "CONCEPT_NOT_REPORTED"


def test_missing_related_receivables_are_not_assumed_zero_or_renamed_total():
    source = original()
    related = FiledFact("tifrs-bsci-ci:AccountsReceivableDuefromRelatedPartiesNet", 99000., None,
                        None, None, "2026-06-30", unit="iso4217:TWD")
    row = ixbrl_record_r2("2330", replace(source, facts=(*source.facts, related)), original_verified=True)
    assert row["accounts_receivable"] == 300.  # explicitly ordinary net, not 399/total
    assert row["field_evidence"]["accounts_receivable"]["contexts"][0]["concept"] == "tifrs-bsci-ci:AccountsReceivableNet"


@pytest.mark.parametrize("damage", ["not_verified", "prior_period", "undated"])
def test_audit_safety_requires_verified_current_original(damage):
    source = original()
    if damage != "not_verified":
        facts = tuple(replace(f, end_date="2025-06-30" if damage == "prior_period" else None)
                      if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
        source = replace(source, facts=facts)
    proof = audit_proof(source, symbol="2330", original_verified=damage != "not_verified")
    assert proof["status"] == "PENDING_VERIFICATION"


@pytest.mark.parametrize("body", [
    "會計師核閱報告：民國115年及114年6月30日之合併資產負債表。",
    "會計師核閱報告：民國一一五年六月三十日之合併財務報表。",
    "民國一一五年六月三十日及民國一一四年六月三十日之合併資產負債表。",
    "Review report: consolidated financial statements as of June 30, 2026.",
    "We reviewed consolidated balance sheets of ASUSTEK COMPUTER INC. and subsidiaries as of June 30, 2026 and 2025.",
])
def test_signing_date_binds_reported_financial_period_without_rewriting_context(body):
    source = original()
    retained = tuple(replace(f, end_date="2026-08-09") if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    report = FiledFact("tifrs-ar:AccountantsReportBody", None, body, None, "2026-08-09", None,
                       context_id="SYNTHETIC_SIGNING_DATE")
    proof = audit_proof(replace(source, facts=(*retained, report)), symbol="2330", original_verified=True)
    assert proof["status"] == "VERIFIED_REPORTED"
    assert proof["period_end"] == "2026-06-30"
    assert proof["period_binding"]["auditor_report_date"] == "2026-08-09"
    assert all((node["end_date"] or node["instant"]) == "2026-08-09" for node in proof["contexts"])
    assert proof["available_at"] == source.available_at  # never moved to signing day


@pytest.mark.parametrize("damage", ["wrong_period", "wrong_scope", "future_signature", "invalid_date", "no_body", "ambiguous_date"])
def test_signing_date_without_verified_document_period_binding_stays_pending(damage):
    source = original()
    signed = "2027-01-01" if damage == "future_signature" else "2026-08-zz" if damage == "invalid_date" else "2026-08-09"
    retained = tuple(replace(f, end_date=signed) if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    body = "民國115年6月30日之合併財務報表。"
    if damage == "wrong_period": body = "民國114年6月30日之合併財務報表。"
    if damage == "wrong_scope": body = "民國115年6月30日之個體財務報表。"
    extra = () if damage == "no_body" else (FiledFact("tifrs-ar:AccountantsReportBody", None, body, None, signed, None),)
    if damage == "ambiguous_date": extra += (FiledFact("tifrs-ar:UnqualifiedOpinionAbstract", None, "Y", None, "2026-08-08", None),)
    proof = audit_proof(replace(source, facts=(*retained, *extra)), symbol="2330", original_verified=True)
    assert proof["status"] == "PENDING_VERIFICATION"


@pytest.mark.parametrize("context_date", ["2026-06-30", "2026-08-09"])
@pytest.mark.parametrize("body,scope", [
    ("民國115年6月30日之個體財務報表。本公司另編製合併財務報表。", "consolidated"),
    ("民國115年6月30日之合併財務報表。本公司另編製個體財務報表。", "individual"),
    ("Separate financial statements as of June 30, 2026. A consolidated report is also prepared.", "consolidated"),
    ("民國115年6月30日之個體財務報表。另附民國115年6月30日之合併財務報表。", "consolidated"),
    ("民國115年6月30日之合併財務報表。民國一一五年六月三十日之個體財務報表。", "consolidated"),
    ("民國115年及114年6月30日之合併資產負債表。民國115年6月30日之個體財務報表。", "consolidated"),
    ("Separate financial statements as of June 30, 2026. Consolidated financial statements as of June 30, 2026.", "consolidated"),
    ("民國1115年6月30日之合併財務報表。", "consolidated"),
    ("民國一一一五年六月三十日之合併財務報表。", "consolidated"),
    ("12026年6月30日之合併財務報表。", "consolidated"),
])
def test_audit_body_period_and_scope_must_bind_the_same_description(context_date, body, scope):
    source = original()
    retained = tuple(replace(f, end_date=context_date) if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    report = FiledFact("tifrs-ar:AccountantsReportBody", None, body, None, context_date, None)
    proof = audit_proof(replace(source, facts=(*retained, report), statement_scope=scope),
                        symbol="2330", original_verified=True)
    assert proof["status"] == "PENDING_VERIFICATION"


def test_conflicting_report_bodies_cannot_be_ignored_in_favor_of_matching_body():
    source = original()
    retained = tuple(replace(f, end_date="2026-08-09") if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    reports = tuple(FiledFact("tifrs-ar:AccountantsReportBody", None, text, None, "2026-08-09", None)
                    for text in ("民國115年6月30日之合併財務報表。", "民國115年6月30日之個體財務報表。"))
    proof = audit_proof(replace(source, facts=(*retained, *reports)), symbol="2330", original_verified=True)
    assert proof["status"] == "PENDING_VERIFICATION"


def test_matching_scopes_across_numeric_and_spelled_dates_remain_source_bound():
    source = original()
    retained = tuple(replace(f, end_date="2026-08-09") if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    report = FiledFact("tifrs-ar:AccountantsReportBody", None,
        "民國115年6月30日之合併財務報表。民國一一五年六月三十日之合併財務報表。",
        None, "2026-08-09", None)
    proof = audit_proof(replace(source, facts=(*retained, report)), symbol="2330", original_verified=True)
    assert proof["status"] == "VERIFIED_REPORTED"
    assert proof["period_binding"]["body_bindings"][0]["financial_period_quotes"] == [
        "115年6月30日", "一一五年六月三十日"]


def test_r2_additive_note_ledger_is_exact_and_legacy_ledger_stays_unchanged():
    source = original()
    r1 = ixbrl_record("2330", source)
    r2 = ixbrl_record_r2("2330", source, original_verified=True)
    assert "tifrs-notes:InterestExpense_n" not in r1["statement_ledger"]
    assert r2["statement_ledger"]["tifrs-notes:InterestExpense_n"] == 300.
    assert financial_supplement(source, symbol="2330", original_verified=True)["statement_ledger"]["tifrs-notes:InterestExpense_n"] == 300.


def test_mixed_financial_and_signing_opinion_contexts_are_not_silently_prioritized():
    source = original()
    conflicting = FiledFact("tifrs-ar:AdverseOpinionAbstract", None, "Y", None, "2026-08-09", None)
    proof = audit_proof(replace(source, facts=(*source.facts, conflicting)), symbol="2330", original_verified=True)
    assert proof["status"] == "PENDING_VERIFICATION"


def test_thin_supplement_contains_only_added_fields_and_exact_source_identity():
    source = original()
    result = financial_supplement(source, symbol="2330", market="TPEX", original_verified=True)
    assert result["schema"] == "financial-supplement/v1" and result["status"] == "VERIFIED_ORIGINAL"
    assert result["market"] == "TPEX" and result["content_sha256"] == source.content_sha256
    assert set(result["fields"]) == set(EXTRA_FIELDS)
    assert "cfo" not in result["field_evidence"] and "revenue" not in result["fields"]
    assert result["payment_policy"] == PAYMENT_POLICY


def test_verified_payment_magnitude_preserves_raw_value():
    row = ixbrl_record_r2("2330", original(), original_verified=True)
    trace = dict(status="VERIFIED_REPORTED", source_ids=[row["content_sha256"]],
                 available_at=row["available_at"], value=-200., paths=[dict(value=-200.)])
    before = deepcopy(row)
    value, proof = payment_outflow(-200., field="intangible_capex", periods={"P": row}, trace=trace, cutoff=CUTOFF)
    assert value == 200. and proof["raw_value"] == -200.
    assert proof["status"] == "VERIFIED_DERIVED" and row == before


@pytest.mark.parametrize("damage", ["mapping", "concept", "future", "mixed_sign", "source_id"])
def test_negative_value_without_exact_payment_source_is_never_abs_repaired(damage):
    row = ixbrl_record_r2("2330", original(), original_verified=True)
    trace = dict(status="VERIFIED_REPORTED", source_ids=[row["content_sha256"]], available_at=row["available_at"],
                 value=-200., paths=[dict(value=-200.)])
    if damage == "mapping": row.pop("financial_mapping_version")
    elif damage == "concept": row["field_evidence"]["intangible_capex"]["contexts"][0]["concept"] = "ifrs-full:IntangibleAssetsOtherThanGoodwill"
    elif damage == "future": row["available_at"] = "2027-01-01T00:00:00Z"
    elif damage == "source_id": trace["source_ids"] = ["unknown"]
    else:
        proof = deepcopy(row["field_evidence"]["intangible_capex"])
        proof.update(start_date="2025-01-01", end_date="2025-06-30")
        proof["contexts"][0].update(start_date="2025-01-01", end_date="2025-06-30", numeric_value=200000.)
        row["comparative_ytd_field_evidence"]["intangible_capex"] = proof
    value, proof = payment_outflow(-200., field="intangible_capex", periods={"P": row}, trace=trace, cutoff=CUTOFF)
    assert value is None and proof["status"] == "UNKNOWN"


def test_repository_source_companion_is_cutoff_bound_and_rejects_corrupt_original(tmp_path):
    from infrastructure.fundamentals.mops_repository import MopsRepository
    from tests.fundamentals.test_filing_refresh_planner import _candidate
    root = tmp_path / "mops"
    repository = MopsRepository(root)
    body = cash_document().replace(b"</body>", b'<ix:nonNumeric name="tifrs-ar:UnqualifiedOpinionAbstract" '
        b'contextRef="YTD">Y</ix:nonNumeric></body>')
    first = repository.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:01:00Z")
    future = body.replace(b"</body>", b"<!-- SYNTHETIC FUTURE REVISION --></body>")
    second = repository.save_revision(_candidate(2026, 2), future, parse_ixbrl(future),
        available_at="2026-11-01T00:00:00Z", fetched_at="2026-11-01T00:01:00Z")
    sources = repository.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),))
    assert set(sources["financial_audits"]) == {first.content_sha256}
    assert second.content_sha256 not in sources["financial_supplements"]
    assert sources["financial_audits"][first.content_sha256]["status"] == "VERIFIED_REPORTED"
    (root / first.raw_path).write_bytes(b"SYNTHETIC CORRUPT ORIGINAL")
    assert repository.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),)) == dict(
        financial_audits={}, financial_supplements={})
