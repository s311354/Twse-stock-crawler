"""SYNTHETIC ONLY: reported annual EPS proofs, not official repaired coverage."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.valuation_earnings import EPS_CONCEPTS, EPS_UNIT, MAPPING_VERSION, SCHEMA, valuation_earnings


CUTOFF = "2026-10-14T13:30:00+08:00"


def eps(concept, value, year, **changes):
    return replace(FiledFact(concept, value, None, f"{year}-01-01", f"{year}-12-31", None,
        unit=EPS_UNIT, context_id=f"SYNTHETIC_ANNUAL_{year}"), **changes)


def original():
    facts = tuple(eps(concept, value, year) for year, basic, diluted in ((2025, 2.5, 2.4), (2024, 1.8, 1.7))
        for concept, value in (("ifrs-full:BasicEarningsLossPerShare", basic),
                               ("ifrs-full:DilutedEarningsLossPerShare", diluted)))
    return FiledFinancials(2025, 4, "2026-03-10T05:00:00Z", 2, facts,
        content_sha256=hashlib.sha256(b"SYNTHETIC_ONLY_EPS_ORIGINAL").hexdigest(),
        source_url="https://mopsov.twse.com.tw/SYNTHETIC_ONLY_EPS.html", period_end="2025-12-31")


def mapped(filing=None, **options):
    return valuation_earnings(filing or original(), symbol="SYNTHETIC", original_verified=True,
        as_of=CUTOFF, **options)


def test_current_and_prior_total_eps_preserve_exact_original_units_and_availability():
    source = original()
    before = deepcopy(source)
    result = mapped(source)
    assert result["schema"] == SCHEMA and result["mapping_version"] == MAPPING_VERSION
    assert result["status"] == "VERIFIED_ORIGINAL"
    assert result["fields"] == dict(basic_eps=2.5, diluted_eps=2.4)
    assert result["comparative_ytd_fields"] == dict(basic_eps=1.8, diluted_eps=1.7)
    for group in ("field_evidence", "comparative_ytd_field_evidence"):
        for proof in result[group].values():
            assert proof["status"] == "RESOLVED" and proof["basis"] == "ANNUAL"
            assert proof["unit"] == "TWD/share"
            assert proof["content_sha256"] == proof["revision_id"] == source.content_sha256
            assert proof["available_at"] == source.available_at
            assert proof["contexts"][0]["unit"] == EPS_UNIT
    assert result["comparative_ytd_field_evidence"]["diluted_eps"]["end_date"] == "2024-12-31"
    assert source == before


@pytest.mark.parametrize("amount", [0., -1.09])
def test_known_zero_or_negative_eps_is_reported_not_missing(amount):
    source = original()
    facts = tuple(replace(fact, numeric_value=amount) for fact in source.facts)
    result = mapped(replace(source, facts=facts))
    assert result["fields"] == dict(basic_eps=amount, diluted_eps=amount)
    assert all(p["status"] == "RESOLVED" for p in result["field_evidence"].values())


@pytest.mark.parametrize("damage,reason", [
    ("unit", "EPS_UNIT_MISSING_OR_INCOMPATIBLE"),
    ("dimensions", "NO_EXACT_AGGREGATE_ANNUAL_DURATION"),
    ("quarter_duration", "NO_EXACT_AGGREGATE_ANNUAL_DURATION"),
    ("instant", "NO_EXACT_AGGREGATE_ANNUAL_DURATION"),
    ("conflict", "CONFLICTING_AGGREGATE_EPS_VALUES"),
    ("nil", "NIL_OR_NONFINITE_EPS_VALUE"),
])
def test_invalid_exact_context_remains_unknown_without_other_fact_fallback(damage, reason):
    source = original()
    current = source.facts[0]
    options = dict(unit="iso4217:TWD") if damage == "unit" else (
        dict(dimensions=(("SYNTHETIC_AXIS", "SUBSIDIARY"),)) if damage == "dimensions" else
        dict(start_date="2025-10-01") if damage == "quarter_duration" else
        dict(instant="2025-12-31") if damage == "instant" else
        dict(numeric_value=None) if damage == "nil" else {})
    facts = (replace(current, **options), *source.facts[1:])
    if damage == "conflict": facts += (replace(current, numeric_value=999.),)
    result = mapped(replace(source, facts=facts))
    assert result["fields"]["basic_eps"] is None
    assert result["field_evidence"]["basic_eps"]["status"] == "UNKNOWN"
    assert result["field_evidence"]["basic_eps"]["reason"] == reason
    assert result["fields"]["diluted_eps"] == 2.4


@pytest.mark.parametrize("variant", ["FromContinuingOperations", "FromDiscontinuedOperations"])
def test_component_eps_is_never_an_alias_for_total_reported_eps(variant):
    source = original()
    facts = tuple(replace(fact, concept=fact.concept + variant) for fact in source.facts)
    result = mapped(replace(source, facts=facts))
    assert result["fields"] == dict(basic_eps=None, diluted_eps=None)
    assert all(p["reason"] == "TOTAL_EPS_CONCEPT_NOT_REPORTED" for p in result["field_evidence"].values())


@pytest.mark.parametrize("damage,reason", [
    ("future", "FINANCIAL_AVAILABILITY_AFTER_CUTOFF"),
    ("naive_time", "FINANCIAL_AVAILABILITY_UNVERIFIED"),
    ("wrong_period", "FILING_PERIOD_END_MISMATCH"),
    ("hash", "ORIGINAL_IDENTITY_UNVERIFIED"),
    ("scope", "ORIGINAL_IDENTITY_UNVERIFIED"),
])
def test_unverified_source_metadata_cannot_promote_eps(damage, reason):
    changes = dict(available_at="2027-01-01T00:00:00Z") if damage == "future" else (
        dict(available_at="2026-03-10T05:00:00") if damage == "naive_time" else
        dict(period_end="2025-09-30") if damage == "wrong_period" else
        dict(content_sha256="unverified") if damage == "hash" else dict(statement_scope="mixed"))
    result = mapped(replace(original(), **changes))
    assert result["status"] == "PENDING_VERIFICATION"
    assert result["fields"] == result["comparative_ytd_fields"] == dict(basic_eps=None, diluted_eps=None)
    assert all(p["reason"] == reason for p in result["field_evidence"].values())


def test_unverified_original_cannot_produce_effective_values():
    result = valuation_earnings(original(), symbol="SYNTHETIC", as_of=CUTOFF)
    assert result["fields"] == dict(basic_eps=None, diluted_eps=None)
    assert result["status"] == "PENDING_VERIFICATION"


def test_missing_total_diluted_eps_does_not_substitute_basic_or_infer_shares():
    source = original()
    result = mapped(replace(source, facts=tuple(f for f in source.facts if "Diluted" not in f.concept)))
    assert result["fields"] == dict(basic_eps=2.5, diluted_eps=None)
    assert "diluted_shares" not in result and "diluted_shares" not in result["fields"]


def test_nonannual_filing_is_not_mapped_as_annual_eps():
    with pytest.raises(ValueError, match="ANNUAL_FILING_REQUIRED"):
        mapped(replace(original(), quarter=2))


def test_total_eps_alias_policy_cannot_be_mutated_in_a_running_job():
    with pytest.raises(TypeError):
        EPS_CONCEPTS["diluted_eps"] = "ifrs-full:BasicEarningsLossPerShare"


def test_individual_original_and_market_remain_explicit_without_scope_mixing():
    result = mapped(replace(original(), statement_scope="individual"), market="TPEX")
    assert result["market"] == "TPEX" and result["scope"] == "individual"
    assert all(p["scope"] == "individual" for p in result["field_evidence"].values())


def test_source_repository_adds_annual_eps_without_additional_reads_or_changed_old_groups(monkeypatch):
    from infrastructure.fundamentals.mops_queries import MopsQueryRepository
    from types import SimpleNamespace
    import infrastructure.fundamentals.mops_queries as module
    source = original()
    calls, queries = [], []
    stored = SimpleNamespace(fiscal_year=2025, quarter=4, raw_path="raw/SYNTHETIC/2025Q4.html",
        content_sha256=source.content_sha256, available_at=source.available_at, revision_number=2,
        correction_description=None, availability_source="ORIGINAL_OBSERVATION", filing_id=1,
        source_url=source.source_url, period_end=source.period_end, statement_scope=source.statement_scope,
        audit_status=None)
    class FakePath:
        def __truediv__(self, _): return self
        def resolve(self): return self
        def is_relative_to(self, _): return True
        def stat(self): return SimpleNamespace(st_mtime_ns=1, st_ctime_ns=2, st_size=3)
        @property
        def parent(self): return self
    def load_filings_many(*args, **kwargs):
        queries.append((args, kwargs))
        return {"SYNTHETIC": [stored]}
    repo = SimpleNamespace(_raw_root=FakePath(), load_filings_many=load_filings_many)
    monkeypatch.setattr(module, "_raw_normalized_facts", lambda *args: (
        calls.append(args) or (source.facts, source.taxonomy_schema_refs)))
    result = MopsQueryRepository.load_v7_financial_sources(repo, "SYNTHETIC", CUTOFF, periods=((2025, 4),))
    from domain.les_v7.financial_sources import audit_proof, financial_supplement
    assert len(calls) == len(queries) == 1
    assert result["financial_audits"] == {source.content_sha256: audit_proof(source, symbol="SYNTHETIC", original_verified=True)}
    assert result["financial_supplements"] == {source.content_sha256: financial_supplement(source, symbol="SYNTHETIC", original_verified=True)}
    assert result["valuation_earnings"][source.content_sha256]["fields"] == dict(basic_eps=2.5, diluted_eps=2.4)


def test_real_repository_boundary_excludes_future_annual_revision_and_corrupt_original(tmp_path):
    from infrastructure.fundamentals.mops_repository import MopsRepository
    from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
    from tests.fundamentals.test_filing_refresh_planner import _candidate
    body = b'''<html xmlns:xbrli="http://www.xbrl.org/2003/instance"
      xmlns:iso4217="http://www.xbrl.org/2003/iso4217"><body>
      <xbrli:unit id="EPS"><xbrli:divide><xbrli:unitNumerator>
        <xbrli:measure>iso4217:TWD</xbrli:measure></xbrli:unitNumerator>
        <xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure>
        </xbrli:unitDenominator></xbrli:divide></xbrli:unit>
      <xbrli:context id="ANNUAL"><xbrli:period>
        <xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate>
        </xbrli:period></xbrli:context>
      <ix:nonFraction name="ifrs-full:BasicEarningsLossPerShare" contextRef="ANNUAL" unitRef="EPS">2.5</ix:nonFraction>
      <ix:nonFraction name="ifrs-full:DilutedEarningsLossPerShare" contextRef="ANNUAL" unitRef="EPS">2.4</ix:nonFraction>
    </body></html>'''
    root = tmp_path / "SYNTHETIC_ONLY_mops"
    repository = MopsRepository(root)
    candidate = replace(_candidate(2025, 4), period_end="2025-12-31")
    first = repository.save_revision(candidate, body, parse_ixbrl(body),
        available_at="2026-03-10T05:00:00Z", fetched_at="2026-03-10T05:01:00Z")
    future = body.replace(b">2.4<", b">999.0<")
    second = repository.save_revision(candidate, future, parse_ixbrl(future),
        available_at="2027-03-10T05:00:00Z", fetched_at="2027-03-10T05:01:00Z")
    result = repository.load_v7_financial_sources("2330", CUTOFF, periods=((2025, 4),))
    assert set(result["valuation_earnings"]) == {first.content_sha256}
    assert second.content_sha256 not in result["valuation_earnings"]
    assert result["valuation_earnings"][first.content_sha256]["fields"] == dict(basic_eps=2.5, diluted_eps=2.4)
    (root / first.raw_path).write_bytes(b"SYNTHETIC_ONLY_CORRUPT_ORIGINAL")
    assert repository.load_v7_financial_sources("2330", CUTOFF, periods=((2025, 4),)) == dict(
        financial_audits={}, financial_supplements={})
