"""Public composition root for strict replay consumers, including BankEmulator."""
from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from application.replay_contracts import ReplayVersions, digest
from application.replay_inputs import PinnedReplayEvidence
from application.replay_service import StrictReplayEngine, json_value
from domain.industry_profiles import IndustryProfileRegistry
from domain.strategies.expected_return_exit_b import DAILY_EREM_MODEL_VERSION as EREM_V3_MODEL_VERSION
from domain.strategies.erem_soft_reduction import SoftReductionFormalPolicy
from application.low_entry_v6_daily import DAILY_V6_CONFIG, DAILY_V6_PARAMETER_VERSION, DAILY_V6_STRATEGY
from domain.strategies.low_entry_v6_hybrid import model_spec
from domain.strategies.low_entry_v6_quality import (QUALITY_POLICY_VERSION, QUALITY_BUDGETS,
    QUALITY_COVERAGE_MINIMA, QUALITY_MINIMUM_AVAILABLE_WEIGHT, QUALITY_MISSING_PENALTY_RATE,
    QUALITY_REQUIRED_REPORTED_FIELDS, DIAGNOSTIC_TTM_FIELDS, QUALITY_MINIMUM_CONFIDENCE, MINIMUM_ENTRY_QUALITY,
    QUALITY_PROFITABILITY_BASIS, QUALITY_PROFITABILITY_WINDOW, QUALITY_DC_WEIGHTS,
    QUALITY_SCORING_POLICY, QUALITY_SOURCE_SELECTION_POLICY)
from domain.strategies.low_entry_v6_quality_evidence import QUALITY_EVIDENCE_SCHEMA
from domain.strategies.low_entry_v6_ttm import TTM_BRIDGE_POLICY_VERSION
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
from domain.strategies.low_entry_score_v5 import LES_V5_MODEL_VERSION, LES_V5_PARAMETER_VERSION
from domain.strategies.portfolio_allocation_v2 import PAE_V2_MODEL_VERSION, PortfolioAllocationV2Config
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.financial_taxonomy import (CASH_RECONCILIATION_POLICY, TAXONOMY_MAPPING_VERSION,
                                       OPERATING_COST_RECONCILIATION_POLICY)
from infrastructure.replay_evidence import load_evidence
from infrastructure.runtime_environment import current_runtime_identity
from infrastructure.storage.replay_state import SQLiteReplayState
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl

ROOT = Path(__file__).resolve().parents[1]


def current_versions() -> ReplayVersions:
    signature = hashlib.sha256()
    for folder in ("application", "domain", "infrastructure", "interface", "config", "twse"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            signature.update(str(path.relative_to(ROOT)).encode())
            signature.update(path.read_bytes())
    return ReplayVersions(f"LES=6.0.0;legacy_LES={LES_V5_MODEL_VERSION};EREM={EREM_V3_MODEL_VERSION};PAE={PAE_V2_MODEL_VERSION}",
                          digest(parameter_manifest()), signature.hexdigest())


def parameter_manifest() -> dict:
    """The complete effective parameter/environment input to the version hash."""
    parameters = dict(les=asdict(DAILY_V6_CONFIG), les_version=DAILY_V6_PARAMETER_VERSION,
                      erem_exit_policy=SoftReductionFormalPolicy().record(),
                      les_strategy=DAILY_V6_STRATEGY,
                      les_score_model=model_spec(DAILY_V6_CONFIG),
                      quality_policy=QUALITY_POLICY_VERSION, quality_budgets=QUALITY_BUDGETS,
                      quality_coverage_minima=QUALITY_COVERAGE_MINIMA,
                      quality_minimum_available_weight=QUALITY_MINIMUM_AVAILABLE_WEIGHT,
                      quality_minimum_confidence=QUALITY_MINIMUM_CONFIDENCE,
                      quality_minimum_score=MINIMUM_ENTRY_QUALITY,
                      quality_missing_penalty_rate=QUALITY_MISSING_PENALTY_RATE,
                      quality_required_reported_fields=QUALITY_REQUIRED_REPORTED_FIELDS,
                      quality_diagnostic_ttm_fields=DIAGNOSTIC_TTM_FIELDS,
                      quality_evidence_schema=QUALITY_EVIDENCE_SCHEMA,
                      quality_ttm_bridge_policy=TTM_BRIDGE_POLICY_VERSION,
                      quality_filing_schema=V6_FILING_SCHEMA_VERSION,
                      quality_source_selection_policy=QUALITY_SOURCE_SELECTION_POLICY,
                      quality_cash_reconciliation_policy=CASH_RECONCILIATION_POLICY,
                      quality_operating_cost_reconciliation_policy=OPERATING_COST_RECONCILIATION_POLICY,
                      financial_taxonomy_mapping_version=TAXONOMY_MAPPING_VERSION,
                      quality_profitability_basis=QUALITY_PROFITABILITY_BASIS,
                      quality_profitability_window=QUALITY_PROFITABILITY_WINDOW,
                      quality_dc_weights=QUALITY_DC_WEIGHTS,
                      quality_scoring_policy=QUALITY_SCORING_POLICY,
                      legacy_les_version=LES_V5_PARAMETER_VERSION, pae=asdict(PortfolioAllocationV2Config()),
                      derivation=FUNDAMENTAL_DERIVATION_VERSION,
                      profiles=[asdict(p) for p in IndustryProfileRegistry.profiles()],
                      python=platform.python_version(),
                      runtime=current_runtime_identity(),
                      dependencies={name:version(name) for name in ("numpy", "pandas")})
    return json_value(parameters)


def build_replay_engine(evidence_file: str | Path, *, parse_cache_path: str | Path | None = None) -> tuple[StrictReplayEngine, str]:
    from infrastructure.research_documents import EvidenceDocuments
    from infrastructure.storage.research_cache import SQLiteFilingHistory
    evidence_id, payload = load_evidence(evidence_file)
    documents = EvidenceDocuments.for_evidence(evidence_file, evidence_id)
    documents.verify(payload)
    indexed_versions = current_versions() if parse_cache_path is not None else None
    index = (SQLiteFilingHistory(Path(parse_cache_path), evidence_id=evidence_id,
             version=digest(asdict(indexed_versions)), parse_filing=parse_ixbrl, document_reader=documents.read)
             if parse_cache_path is not None else None)
    evidence = PinnedReplayEvidence(evidence_id, payload, parse_filing=parse_ixbrl,
                                   annual_calendar=lambda data: TwseCalendar(fetcher=lambda: data),
                                   document_reader=documents.read, document_check=documents.check, filing_index=index)
    def versions():
        documents.check()
        current = current_versions()
        if indexed_versions is not None and current != indexed_versions:
            raise ValueError("REPLAY_VERSION_CHANGED_DURING_JOB")
        return current
    return StrictReplayEngine(evidence, SQLiteReplayState(), versions), evidence_id
