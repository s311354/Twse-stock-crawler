"""Isolated FR2 native research composition; no daily configuration or state writes."""
from copy import deepcopy
import pandas as pd

from application.les_v7 import prepare_quality_context
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.fr2_policy import apply_fr2
from domain.les_v7.fr2_execution import _assess_validated
from domain.les_v7.native_evidence import produce_native_receipts


def analyze_fr2_request(payload):
    """Recompute frozen predictors first; admission and sizing have separate identities.

    Explicit accounts are mandatory. Missing existing-position risk evidence is
    retained as allocation UNKNOWN, never substituted with an empty account.
    """
    if not isinstance(payload, dict) or payload.get("schema") != "les-v7-fr2-request/v1":
        raise ValueError("FR2_EXPLICIT_REQUEST_SCHEMA_REQUIRED")
    requests = payload.get("securities")
    if not isinstance(requests, list) or not 1 <= len(requests) <= 2048:
        raise ValueError("FR2_BOUNDED_SECURITY_REQUESTS_REQUIRED")
    reports, seen = [], set()
    for item in requests:
        request = deepcopy(item)
        if not isinstance(request.get("account"), dict): raise ValueError("FR2_EXPLICIT_ACCOUNT_REQUIRED")
        context = request["context"]
        key = context["symbol"], context["market"], timestamp(context["as_of"]).isoformat()
        if key in seen: raise ValueError("FR2_DUPLICATE_SECURITY_CUTOFF")
        seen.add(key)
        if payload.get("model_bundle") is not None:
            if context.get("model_bundle") is not None and context["model_bundle"] != payload["model_bundle"]:
                raise ValueError("FR2_CONFLICTING_MODEL_BUNDLES")
            context["model_bundle"] = deepcopy(payload["model_bundle"])
        if not context.get("model_bundle") or not context.get("native_evidence_source") or context.get("receipts"):
            raise ValueError("FR2_FROZEN_MODEL_AND_NATIVE_SOURCE_REQUIRED_NO_PREBOUND_GATES")
        prepare_quality_context(context)
        original = evaluate(pd.DataFrame(request["bars"]), context, receipt_producer=produce_native_receipts)
        report = apply_fr2(original, context["model_bundle"])
        report["fr2_execution"] = _assess_validated(report, report["diagnostics"]["fr2"]["admission"],
            request["account"], quote=request.get("quote"))
        reports.append(report)
    result = dict(schema="les-v7-fr2-research-result/v1", request_id=identity(payload), reports=reports,
        allocation_scope="INDEPENDENT_CANDIDATE_QUOTAS_NOT_SIMULTANEOUS_PORTFOLIO_APPROVAL",
        research_only=True, execution_enabled=False, approved_for_live=False,
        evaluation_scope="COUNTERFACTUAL_FROZEN_CUTOFF_NOT_HISTORICAL_DEPLOYMENT")
    return dict(result, result_id=identity(result))
