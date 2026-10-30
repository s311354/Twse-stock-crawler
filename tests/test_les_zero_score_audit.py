"""Synthetic algebra/source attribution, never manufactured trading evidence."""
from copy import deepcopy

import pytest

from examples.audit_les_zero_scores import explain_zero


def sample():
    ledger=dict(conditions=[dict(earned_points=0.,available_weight=70.),
        dict(earned_points=10.,available_weight=15.),dict(earned_points=0.,available_weight=0.)],
        components=dict(structure=dict(total_weight=100,earned_points=10.,available_weight=85.)),
        missing_fields=["industry_rs20","@reference_plan"])
    score=dict(raw_score=10.,total_penalty=14.,final_score=0.)
    gates=[dict(gate_id=k,state=s,reasons=r) for k,s,r in
        (("DATA_INTEGRITY","PASS",[]),("TRADING_STATUS","PASS",[]),
         ("SAFETY","FAIL",["SUPPORT_FAILURE"]),("VALUATION","FAIL",["PRICE_WARNING"]),
         ("STRESS_RISK","UNKNOWN",["REFERENCE_PLAN_MISSING"]))]
    return ledger,score,dict(plan=None,features=dict(history_ready=True,market_ready=True,
        support_intact=False,no_new_low2=False),gates=gates,quality=dict(status="PASS",quality_score=80.))


def test_nonzero_technical_score_and_known_failure_are_not_missing_data():
    ledger,score,kw=sample();before=deepcopy((ledger,score,kw))
    r=explain_zero(ledger,score,**kw)
    assert (ledger,score,kw)==before
    assert r["raw"]==10 and r["signed_after_penalty"]==-4 and r["final"]==0
    assert r["cause"]=="EXPLICIT_PENALTY_FLOOR_NOT_ZERO_RAW"
    assert r["known_zero_weight"]==70 and r["unavailable_weight"]==15
    assert r["quality_status"]=="PASS" and r["gate_states"]["SAFETY"]=="FAIL"
    assert "TECHNICAL" in r["safety_scope"] and r["stress_scope"]=="NO_PLAN_TO_EVALUATE"
    assert r["mathematical_ceiling_if_all_unavailable_full"]==11
    assert "NOT_IMPUTED" in r["ceiling_role"]
    assert not r["decision_recomputed"] and not r["order_authority"]


@pytest.mark.parametrize("key,value",[("raw_score",20.),("final_score",1.)])
def test_cannot_explain_a_tampered_or_implicitly_boosted_score(key,value):
    ledger,score,kw=sample();score[key]=value
    with pytest.raises(ValueError,match="MISMATCH"):explain_zero(ledger,score,**kw)
