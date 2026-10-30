"""SYNTHETIC: isolate conflicting P/E, never certify fabricated market evidence."""
from copy import deepcopy
from datetime import timedelta
import json

import pandas as pd
import pytest

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.features import compute_features
from domain.les_v7.native_evidence import produce_native_receipts, SCOPED_SOURCE_PRODUCER_VERSION
from domain.les_v7.official_pe import OfficialPeIndex, build_snapshot
from domain.les_v7.policy_r23 import AVAILABLE_AT
from domain.les_v7.source_policy import VERSION, SIGNED_VERSION, SCOPED_VERSION, source_policy
from domain.les_v7.source_projection import join_official_pe, validate_source_projection, scoped_pe_inputs
from tests.test_official_pe import payload
from tests.test_v7_native_evidence import inputs

AT = "2026-10-28T07:20:00Z"


def pe_selection(*, day="2026-10-13", close=100., pe="12.50"):
    body=payload(symbol="SYN1",pe=pe)
    body["date"]=day.replace("-","");body["data"][0][2]=str(close)
    snapshot=build_snapshot(json.dumps(body).encode(),market="TWSE",trade_date=day,fetched_at=AT)
    return OfficialPeIndex([snapshot]).select(market="TWSE",symbol="SYN1",trade_date=day,cutoff=AT)


def join(rows, selection, version=SCOPED_VERSION, *, cutoff=AT):
    return join_official_pe(rows,[selection],symbol="SYN1",market="TWSE",cutoff=cutoff,
        source_policy=source_policy(version=version) if version else None)


@pytest.mark.parametrize("version", [None,VERSION,SIGNED_VERSION])
def test_legacy_join_retains_exact_conservative_output(version):
    rows=[dict(Date="2026-10-13",Close=100.,PE=12.49)]
    before=deepcopy(rows)
    result,diagnostics=join(rows,pe_selection(),version)
    assert result==[dict(rows[0],PE=None)]
    assert diagnostics==[dict(session_date="2026-10-13",reason="OFFICIAL_PE_QUOTE_CONFLICT")]
    assert rows==before


def test_scoped_ratio_conflict_retains_both_values_and_unknown_no_substitution():
    selection=pe_selection();rows=[dict(Date="2026-10-13",Close=100.,PE=12.49)]
    result,diagnostics=join(rows,selection)
    assert result[0]["PE"] is None and result[0]["Close"]==100.
    d=diagnostics[0]
    assert d["reason"]=="OFFICIAL_PE_RATIO_CONFLICT" and d["state"]=="UNKNOWN"
    assert d["conflict_fields"]==["PE"] and d["blocks_data_integrity"] is False
    assert d["quote_values"]==dict(Close=100.,PE=12.49)
    assert d["official_values"]==dict(Close=100.,PE=12.5,status="REPORTED")
    assert d["source_record_ids"]==selection["source_record_ids"]
    assert d["quarantine"]=="PE_NONE_NO_SELECTION_OR_IMPUTATION"


@pytest.mark.parametrize("quote_pe,expected", [(12.5,["Close"]),(12.49,["Close","PE"])])
def test_real_price_conflict_stays_blocking_even_with_ratio_partition(quote_pe,expected):
    rows=[dict(Date="2026-10-13",Close=101.,PE=quote_pe)]
    result,diagnostics=join(rows,pe_selection())
    assert result[0]["PE"] is None and result[0]["Close"]==101.
    assert diagnostics[0]["reason"]=="OFFICIAL_CLOSE_QUOTE_CONFLICT"
    assert diagnostics[0]["conflict_fields"]==expected
    assert diagnostics[0]["blocks_data_integrity"] is True


def test_current_ratio_conflict_is_not_accepted_as_positive_or_counted_history():
    rows=[dict(Date="2026-10-12",Close=99.,PE=12.),dict(Date="2026-10-13",Close=100.,PE=12.49)]
    result,_=join(rows,pe_selection())
    assert result[-1]["PE"] is None
    assert sum(r["PE"] is not None and r["PE"]>0 for r in result)==1


def test_conflict_scope_tampering_unknown_policy_and_future_policy_are_rejected():
    rows=[dict(Date="2026-10-13",Close=100.,PE=12.49)];selected=pe_selection()
    changed=source_policy(version=SCOPED_VERSION)
    changed["pe_conflict_scope"]["blocking_data_integrity_fields"]=[]
    with pytest.raises(ValueError,match="POLICY_MISMATCH"):
        join_official_pe(rows,[selected],symbol="SYN1",market="TWSE",cutoff=AT,source_policy=changed)
    with pytest.raises(ValueError,match="POLICY_VERSION_UNKNOWN"):
        source_policy(version="ALLOW_ALL")
    before=(timestamp(AVAILABLE_AT)-timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError,match="AFTER_CUTOFF"):
        join(rows,selected,cutoff=before)


def native_case(monkeypatch, *, historical=True, price_conflict=False, version=SCOPED_VERSION,
                positive_count=None):
    item=inputs();day_index=-2 if historical else -1
    item["symbol"]="SYN1";item["cutoff"]=timestamp(AT).isoformat();item["plan"]=None
    packet=deepcopy(item["source_packet"])
    packet.update(symbol="SYN1",cutoff=item["cutoff"])
    packet["market_evidence"]["symbol"]="SYN1"
    quote=deepcopy(packet["market_rows"])
    if positive_count is not None:
        for n,row in enumerate(quote):row["PE"]=12. if n>=len(quote)-positive_count else None
    day=quote[day_index]["Date"];price=quote[day_index]["Close"]
    selected=pe_selection(day=day,close=price+1. if price_conflict else price)
    quote[day_index]["PE"]=12.49
    joined,diagnostics=join(quote,selected,version)
    packet.update(quote_market_rows=quote,market_rows=joined,official_pe=[selected],source_join_diagnostics=diagnostics)
    packet["market_evidence"].update(bars_sha256=identity(joined),source_record_ids=[identity(joined)])
    item["source_packet"]=packet;item["bars"]=pd.DataFrame(joined)
    item["features"],_=compute_features(item["bars"])
    from domain.les_v7.decision import vetoes
    item["risks"]=vetoes(item["features"])
    item["bindings"]=deepcopy(item["bindings"])
    for binding in item["bindings"].values():
        binding.update(symbol="SYN1",cutoff=item["cutoff"],bars_hash=identity(joined))
    item["bindings"]["STRESS_RISK"]["plan"]=None
    item["source_policy"]=source_policy(version=version)
    # Identity is a separate fully tested port. This focused synthetic test
    # supplies its verified result, never a production source or stored PASS.
    monkeypatch.setattr("domain.les_v7.source_projection.security_state",
        lambda *a,**k:dict(status="VERIFIED",issuer_symbol="SYN1",reason="SYNTHETIC_IDENTITY_ONLY",
            master=dict(available_at=AT)))
    return item


@pytest.mark.parametrize("historical", [True,False])
def test_ratio_conflict_is_quarantined_without_invalidating_matching_ohlc_identity(monkeypatch,historical):
    item=native_case(monkeypatch,historical=historical)
    receipts=produce_native_receipts(**item);receipt=receipts["DATA_INTEGRITY"]
    assert receipt["state"]=="PASS" and receipt["reason_codes"]==[]
    assert receipt["gate_version"]==SCOPED_SOURCE_PRODUCER_VERSION
    assert item["bars"].iloc[-2 if historical else -1]["PE"] is None or pd.isna(item["bars"].iloc[-2 if historical else -1]["PE"])
    if not historical:
        assert receipts["VALUATION"]["state"]=="UNKNOWN"
        assert "CURRENT_PE_UNAVAILABLE_OR_NON_POSITIVE" in receipts["VALUATION"]["reason_codes"]
        assert receipts["VALUATION"]["valuation_diagnostics"]["pe_input_evidence"]["current_pe"] is None


@pytest.mark.parametrize("historical,positive_count", [(False,None),(True,10)])
def test_eps_result_cannot_compensate_missing_current_or_positive_history(monkeypatch,historical,positive_count):
    item=native_case(monkeypatch,historical=historical,positive_count=positive_count)
    # Explicit synthetic stub proves only gate precedence, not a real
    # valuation PASS or certified annual source. Its full receipt is retained.
    raw=dict(state="PASS",source_available_at=AT,valuation_diagnostics=dict(synthetic_only=True,status="COMPLETE"))
    monkeypatch.setattr("domain.les_v7.native_evidence._valuation",lambda *a,**k:deepcopy(raw))
    result=produce_native_receipts(**item)
    assert result["DATA_INTEGRITY"]["state"]=="PASS"
    v=result["VALUATION"]
    assert v["state"]=="UNKNOWN" and v["gate_version"]==SCOPED_SOURCE_PRODUCER_VERSION
    assert v["valuation_diagnostics"]["raw_valuation_receipt"]==raw
    assert v["valuation_diagnostics"]["raw_valuation_diagnostics_id"]==identity(raw["valuation_diagnostics"])
    assert v["valuation_diagnostics"]["raw_valuation_receipt"]["valuation_diagnostics"]["synthetic_only"] is True
    assert v["valuation_diagnostics"]["pe_input_evidence"]["imputed_values"] is False


def test_a_historical_ratio_conflict_does_not_block_sufficient_current_history(monkeypatch):
    item=native_case(monkeypatch)
    raw=dict(state="PASS",source_available_at=AT,valuation_diagnostics=dict(synthetic_only=True))
    monkeypatch.setattr("domain.les_v7.native_evidence._valuation",lambda *a,**k:deepcopy(raw))
    result=produce_native_receipts(**item)
    assert scoped_pe_inputs(item["source_packet"]["market_rows"])["state"]=="PASS"
    assert result["VALUATION"]==raw


@pytest.mark.parametrize("native_state", ["PASS", "FAIL", "UNKNOWN"])
def test_independent_eps_gate_uses_its_own_sources_not_an_extra_global_pe_gate(monkeypatch,native_state):
    item=native_case(monkeypatch,historical=False,positive_count=10)
    # Explicit synthetic valuation port; not evidence of a real financial
    # PASS. Its state is retained exactly, including a genuinely UNKNOWN EPS.
    raw=dict(state=native_state,source_available_at=AT,valuation_diagnostics=dict(synthetic_only=True))
    monkeypatch.setattr("domain.les_v7.valuation_eps_receipt.produce_price_valuation_receipt",
        lambda *a,**k:deepcopy(raw))
    item["valuation_policy"]={"synthetic_test_port":True}
    result=produce_native_receipts(**item)["VALUATION"]
    assert result["state"]==native_state
    component=result["valuation_diagnostics"]["quote_pe_component"]
    assert component["state"]=="UNKNOWN" and component["current_pe"] is None
    assert component["positive_observations"]<20 and not component["imputed_values"]
    assert component["affects_independent_eps_gate"] is False


def test_price_conflict_still_produces_unknown_data_integrity(monkeypatch):
    item=native_case(monkeypatch,price_conflict=True)
    receipt=produce_native_receipts(**item)["DATA_INTEGRITY"]
    assert receipt["state"]=="UNKNOWN"
    assert receipt["reason_codes"]==["OFFICIAL_CLOSE_QUOTE_CONFLICT"]


def test_old_source_policy_keeps_ratio_conflict_global_unknown(monkeypatch):
    item=native_case(monkeypatch,version=SIGNED_VERSION)
    receipt=produce_native_receipts(**item)["DATA_INTEGRITY"]
    assert receipt["state"]=="UNKNOWN" and receipt["gate_version"]=="LES_V7_NATIVE_SOURCE_GATES_R2"
    assert receipt["reason_codes"]==["OFFICIAL_PE_QUOTE_CONFLICT"]


def test_projection_cannot_drop_or_weaken_quarantined_conflict_diagnostics(monkeypatch):
    item=native_case(monkeypatch)
    packet=deepcopy(item["source_packet"])
    packet["source_join_diagnostics"][0]["blocks_data_integrity"]=True
    with pytest.raises(ValueError,match="SOURCE_PROJECTION_RECOMPUTATION_MISMATCH"):
        validate_source_projection(packet,symbol="SYN1",market="TWSE",cutoff=AT,source_policy=item["source_policy"])


def test_retained_r22_policy_content_hash_does_not_change():
    from domain.les_v7.rule_baseline import baseline_policy
    assert identity(baseline_policy(version="V7_RULE_BASED_BASELINE_R22"))=="8f1cc12ad81530a5a1c22819f84f5e8ba6ff9a8381cfc88b10bc07183cb6646a"
