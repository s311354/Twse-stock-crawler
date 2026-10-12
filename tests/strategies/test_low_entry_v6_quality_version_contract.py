"""Version contracts only; synthetic checks are not financial-data completion."""
from copy import deepcopy
from pathlib import Path

import pytest

from application.low_entry_v6_daily import DAILY_V6_PARAMETER_VERSION
from application.replay_contracts import digest
from domain.strategies import low_entry_v6_weights as weights
from domain.strategies.low_entry_score_v6 import PARAMETER_VERSION as LEGACY_PARAMETERS
from domain.strategies.low_entry_v6_hybrid import PARAMETER_VERSION as HYBRID_PARAMETERS
from domain.strategies.low_entry_v6_hybrid import RELAXED_PARAMETER_VERSION
from domain.strategies.low_entry_v6_quality import (
    QUALITY_DC_WEIGHTS,
    QUALITY_POLICY_VERSION,
    QUALITY_PROFITABILITY_BASIS,
    QUALITY_PROFITABILITY_WINDOW,
    QUALITY_REQUIRED_REPORTED_FIELDS,
    DIAGNOSTIC_TTM_FIELDS,
    QUALITY_SCORING_POLICY,
    QUALITY_SOURCE_SELECTION_POLICY,
)
from domain.strategies.low_entry_v6_quality_evidence import QUALITY_EVIDENCE_SCHEMA
from domain.strategies.low_entry_v6_ttm import TTM_BRIDGE_POLICY_VERSION
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
from domain.financial_taxonomy import (CASH_RECONCILIATION_POLICY, TAXONOMY_MAPPING_VERSION,
                                       OPERATING_COST_RECONCILIATION_POLICY)
from infrastructure.replay_runtime import parameter_manifest


def test_replay_manifest_pins_shared_quality_policy_and_new_semantics():
    manifest = parameter_manifest()
    assert manifest["quality_policy"] == QUALITY_POLICY_VERSION
    assert manifest["quality_evidence_schema"] == QUALITY_EVIDENCE_SCHEMA
    assert manifest["quality_ttm_bridge_policy"] == TTM_BRIDGE_POLICY_VERSION
    assert manifest["quality_filing_schema"] == V6_FILING_SCHEMA_VERSION
    assert manifest["quality_source_selection_policy"] == QUALITY_SOURCE_SELECTION_POLICY
    assert manifest["quality_cash_reconciliation_policy"] == CASH_RECONCILIATION_POLICY
    assert manifest["quality_operating_cost_reconciliation_policy"] == OPERATING_COST_RECONCILIATION_POLICY
    assert manifest["financial_taxonomy_mapping_version"] == TAXONOMY_MAPPING_VERSION
    assert manifest["quality_policy"] == "2026-10-04-v6-reported-period-quality-v7"
    assert manifest["quality_profitability_basis"] == QUALITY_PROFITABILITY_BASIS == "REPORTED_SAME_PERIOD_MARGIN"
    assert manifest["quality_profitability_window"] == QUALITY_PROFITABILITY_WINDOW == 2
    assert manifest["quality_required_reported_fields"] == list(QUALITY_REQUIRED_REPORTED_FIELDS)
    assert manifest["quality_diagnostic_ttm_fields"] == list(DIAGNOSTIC_TTM_FIELDS)
    assert QUALITY_REQUIRED_REPORTED_FIELDS == DIAGNOSTIC_TTM_FIELDS == (
        "revenue", "cost_of_revenue", "operating_income", "net_income", "cfo",
    )
    assert "quality_required_ttm_fields" not in manifest
    assert manifest["quality_dc_weights"] == QUALITY_DC_WEIGHTS
    assert manifest["quality_scoring_policy"] == QUALITY_SCORING_POLICY
    assert manifest["quality_scoring_policy"]["flow_basis"] == "UNIFORM_REPORTED_YTD_ANNUAL_OR_EXPLICIT_QUARTER"
    assert manifest["quality_scoring_policy"]["comparison_basis"] == "SELECTED_ORIGINAL_SAME_DURATION"
    assert manifest["quality_minimum_score"] == 65
    assert manifest["quality_minimum_confidence"] == 65
    assert manifest["quality_minimum_available_weight"] == 75
    assert manifest["les_version"] == DAILY_V6_PARAMETER_VERSION == RELAXED_PARAMETER_VERSION
    assert manifest["les"]["decision_policy_version"] == 4
    assert manifest["les"]["buy_score_minimum"] == 72
    assert LEGACY_PARAMETERS == "2026-10-04-v6-pit-reported-period-quality-v8"
    assert HYBRID_PARAMETERS == "2026-10-04-v6-hybrid-reported-period-quality-v8"
    assert RELAXED_PARAMETER_VERSION == "2026-10-07-v6-hybrid-72-62-reported-period-quality-v8"
    for key in (
        "quality_evidence_schema", "quality_ttm_bridge_policy", "quality_filing_schema",
        "quality_profitability_basis", "quality_profitability_window", "quality_dc_weights",
        "quality_scoring_policy", "quality_required_reported_fields", "quality_diagnostic_ttm_fields",
        "quality_cash_reconciliation_policy", "quality_operating_cost_reconciliation_policy", "financial_taxonomy_mapping_version",
        "quality_source_selection_policy",
    ):
        changed = deepcopy(manifest)
        changed[key] = {"changed": True}
        assert digest(changed) != digest(manifest), key


@pytest.mark.parametrize("source", ["low_entry_v6_quality_evidence.py", "low_entry_v6_reported_period.py", "financial_taxonomy.py"])
def test_frozen_weight_signature_includes_reported_period_evidence_implementation(monkeypatch, source):
    weights.implementation_signature.cache_clear()
    original = Path.read_bytes
    baseline = weights.implementation_signature()
    touched = []

    def changed_read(path):
        raw = original(path)
        if path.name == source:
            touched.append(path.name)
            return raw + b"\n# SYNTHETIC_TEST_ONLY changed verification formula\n"
        return raw

    try:
        monkeypatch.setattr(Path, "read_bytes", changed_read)
        weights.implementation_signature.cache_clear()
        assert weights.implementation_signature() != baseline
        assert touched == [source]
    finally:
        weights.implementation_signature.cache_clear()
