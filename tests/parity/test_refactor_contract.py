"""Executable compatibility contract for behavior-preserving refactors."""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import re
import sqlite3
from pathlib import Path

from application.analysis_dataset import ANALYSIS_COLUMNS
from application.market_history_cache import CachedMarketDay
from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import STRATEGY_DECISION_SCHEMA_VERSION
from domain.model_contracts import StrategyDecision
from domain.models import Stocktype
from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategy import _BUILTIN_STRATEGIES
from domain.strategy import get_strategy
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache
from infrastructure.storage.portfolio_persistence import SQLitePortfolioPersistence
from interface.cli import create_parser


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "refactor_contract_v1.json"
)
EVALUATED_AT = datetime.datetime(
    2026,
    8,
    22,
    13,
    30,
    tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cli_actions() -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for action in create_parser()._actions:
        if action.dest == "help":
            continue
        choices = (
            None
            if action.choices is None
            else [getattr(value, "name", value) for value in action.choices]
        )
        actions.append(
            {
                "dest": action.dest,
                "options": list(action.option_strings),
                "nargs": action.nargs,
                "choices": choices,
            }
        )
    return actions


def _daily_decision() -> StrategyDecision:
    values: dict[str, object] = {
        "日期": "2026-08-22",
        "證券代號": "2330",
        "證券名稱": "台積電",
        "收盤": 1_000_000.0,
        "低點分數": 82.0,
        "低點決策": "BUY",
        "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
        "LES v5 Final Quality Score": 88.0,
        "Valuation Gate": "PASS",
        "Valuation Model Status": "COMPLETE",
        "Base Case Expected CAGR": 0.18,
        "Falling Knife State": "CLEAR",
        "LES v5 Trigger": True,
        "Risk Gate": "PASS",
        "RewardRisk": 2.1,
        "Position Fraction": 1 / 3,
        "策略停損": 960_000.0,
        "策略停利": 1_120_000.0,
        "Entry Rule": "NEXT_SESSION_BREAKOUT",
        "低點理由": "Quality=PASS；Valuation=PASS；Trigger=YES；Risk=PASS",
        "EREM 分數": 78.0,
        "EREM Holding Confidence": 80.0,
        "EREM 決策訊號": "HOLD",
        "EREM 理由": "趨勢完整；決策=HOLD",
        "PAE Status": "SUCCESS",
        "PAE Mode": "COLD_START",
        "PAE Portfolio Input": "COLD_START_ALL_CASH",
        "PAE Role": "CANDIDATE",
        "PAE CER": 0.16,
        "PAE Quality Score": 88.0,
        "PAE Required Return": 0.10,
        "PAE Opportunity Edge": 0.06,
        "PAE Absolute Strategic Candidate": True,
        "PAE Strategic Candidate": True,
        "PAE Executable Candidate": True,
        "PAE Dual Dominance Candidate": None,
        "PAE Capital Claim": 0.04,
        "PAE Best Strategic Alternative": "2330",
        "PAE Best Executable Alternative": "2330",
        "PAE Current Weight": 0.0,
        "PAE Target Weight": 0.3,
        "PAE Weight Change": 0.3,
        "PAE Target Cash Weight": 0.7,
        "PAE Portfolio Expected CER": 0.16,
        "PAE Decision": "ADD",
        "PAE Reason": "COLD_START_EXECUTABLE",
        "PAE Model Version": "1.2.1",
    }
    fields = tuple(
        DecisionField(
            name=name,
            value=value,
            source=FieldSource.LES_V5,
            available_at=EVALUATED_AT,
            confidence=100,
            unit=FieldUnit.TEXT,
            model_version="parity-v1",
        )
        for name, value in values.items()
    )
    return StrategyDecision(
        strategy_name="daily_analysis_v1",
        evaluated_at=EVALUATED_AT,
        fields=fields,
    )


def _pae_decision() -> dict[str, object]:
    candidate = AllocationSecurity(
        symbol="2330",
        sector="24",
        correlation_cluster="SEMI",
        quality_status="QUALITY_CONFIRMED",
        quality_score=90,
        data_confidence=90,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=0.12,
        base_cagr=0.18,
        bull_cagr=0.24,
        valuation_confidence=90,
        les_score=82,
        les_decision="BUY",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        executable_shares=100,
        current_price=1_000,
        maximum_weight=1 / 3,
    )
    decision = PortfolioAllocationEngine().allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(candidate,),
        )
    )
    return {
        "mode": decision.mode.value,
        "model_version": decision.model_version,
        "required_return": round(decision.required_return, 12),
        "best_strategic_alternative": decision.best_strategic_alternative,
        "best_executable_alternative": decision.best_executable_alternative,
        "target_cash_weight": round(decision.target_cash_weight, 12),
        "portfolio_expected_cer": round(
            float(decision.portfolio_expected_cer or 0.0),
            12,
        ),
        "reason": decision.reason,
        "lines": [
            {
                "symbol": line.symbol,
                "role": line.role,
                "action": line.action.value,
                "target_weight": round(line.target_weight, 12),
                "reason": line.reason,
                "strategic": line.evaluation.strategic_candidate,
                "executable": line.evaluation.executable_candidate,
                "cer": round(float(line.evaluation.cer or 0.0), 12),
            }
            for line in decision.lines
        ],
    }


def _sqlite_contract(database_path: Path) -> dict[str, object]:
    with sqlite3.connect(database_path) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        indexes = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        return {
            "tables": {
                table: [
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info({})".format(table)
                    )
                ]
                for table in tables
            },
            "indexes": indexes,
        }


def test_cli_contract_fixture_digest_matches_actions() -> None:
    expected_cli = _fixture()["cli"]
    assert _digest(expected_cli["actions"]) == expected_cli["sha256"], (
        "CLI fixture actions and reviewed SHA-256 must be updated together"
    )


def test_cli_accepts_eleccomp_profile() -> None:
    args = create_parser().parse_args(["-t", "ELECCOMP", "stocklist_eleccomp"])

    assert args.type is Stocktype.ELECCOMP
    assert args.stocklist == ["stocklist_eleccomp"]


def test_cli_strategy_dataset_and_renderer_contracts_are_stable() -> None:
    fixture = _fixture()
    cli_actions = _cli_actions()
    expected_cli = fixture["cli"]
    assert isinstance(expected_cli, dict)
    # Explicitly reviewed additive replay options. Preserve the legacy actions
    # and hash rather than regenerating the golden contract to hide a change.
    replay_options = [
        {"dest": "replay_mail", "options": ["--replay-mail"], "nargs": 0, "choices": None},
        {"dest": "replay_restrictions_file", "options": ["--replay-restrictions-file"], "nargs": None, "choices": None},
        {"dest": "replay_id", "options": ["--replay-id"], "nargs": None, "choices": None},
        {"dest": "replay_evidence_file", "options": ["--replay-evidence-file"], "nargs": None, "choices": None},
        {"dest": "replay_state_path", "options": ["--replay-state-path"], "nargs": None, "choices": None},
        {"dest": "replay_output_dir", "options": ["--replay-output-dir"], "nargs": None, "choices": None},
        {"dest": "replay_parse_cache", "options": ["--replay-parse-cache"], "nargs": None, "choices": None},
    ]
    replay_names = {action["dest"] for action in replay_options}
    assert [action for action in cli_actions if action["dest"] in replay_names] == replay_options
    legacy_actions = [action for action in cli_actions if action["dest"] not in replay_names]
    assert legacy_actions == expected_cli["actions"]
    assert _digest(legacy_actions) == expected_cli["sha256"]

    expected_columns = fixture["analysis_columns"]
    assert isinstance(expected_columns, dict)
    # Explicitly reviewed additive fields, not a regenerated golden schema.
    # Formal score, estimate and coverage must not share a misleading column.
    added = {"Industry Quality Profile", "Industry Profile Version", "Industry Profile Mode", "Industry Shadow Decision",
             "LES v6 Formal Score", "LES v6 Estimated Score", "LES v6 Score Coverage",
             "LES v6 Raw Score", "LES v6 Penalty", "LES v6 Adjusted Score", "LES v6 Safety Cap", "LES v6 Final Score"}
    assert added.issubset(ANALYSIS_COLUMNS)
    original_columns = [name for name in ANALYSIS_COLUMNS if name not in added]
    assert len(original_columns) == expected_columns["count"]
    assert _digest(original_columns) == expected_columns["sha256"]

    expected_strategies = fixture["strategies"]
    assert isinstance(expected_strategies, dict)
    # Explicit opt-in addition; all earlier plugin versions and renderers stay pinned.
    hybrid = {"low_entry_score_v6_hybrid", "low_entry_score_v6_hybrid_legacy"}
    assert hybrid.issubset(_BUILTIN_STRATEGIES)
    assert get_strategy("low_entry_score_v6_hybrid").version == "6.0.0"
    assert sorted(set(_BUILTIN_STRATEGIES) - hybrid) == sorted(expected_strategies)
    assert {
        name: get_strategy(name).version
        for name in sorted(expected_strategies)
    } == expected_strategies
    assert hybrid.issubset(RendererFactory._renderers)
    assert sorted(set(RendererFactory._renderers) - hybrid) == fixture["renderers"]
    assert STRATEGY_DECISION_SCHEMA_VERSION == fixture["decision_schema_version"]


def test_public_import_paths_remain_available() -> None:
    public_imports = _fixture()["public_imports"]
    assert isinstance(public_imports, dict)
    for module_name, names in public_imports.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), "{}.{} is missing".format(
                module_name,
                name,
            )


def test_pae_decision_contract_is_stable() -> None:
    assert _pae_decision() == _fixture()["pae_decision"]


def test_daily_email_normalized_contract_is_stable() -> None:
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(
        (_daily_decision(),)
    )
    normalized = re.sub(r"\s+", " ", rendered).strip()
    expected = _fixture()["daily_email"]
    assert isinstance(expected, dict)
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        expected["normalized_sha256"]
    )
    assert re.findall(r'data-section="([^"]+)"', normalized) == (
        expected["sections"]
    )
    assert re.findall(r"<h[23][^>]*>(.*?)</h[23]>", normalized) == (
        expected["headings"]
    )


def test_sqlite_storage_contracts_are_stable(tmp_path: Path) -> None:
    market_path = tmp_path / "market.sqlite3"
    market = SQLiteMarketHistoryCache(market_path)
    market.upsert_day(
        CachedMarketDay(
            trade_date="20260822",
            stock_rows=[["2330", "TSMC"]],
            taiex_close=10_000.0,
        ),
        market_type="ALL",
    )

    portfolio_path = tmp_path / "portfolio.sqlite3"
    SQLitePortfolioPersistence(portfolio_path)

    mops = MopsRepository(tmp_path / "mops", connect_attempts=1)
    with mops.unit_of_work():
        pass

    observed = {
        "market_history": _sqlite_contract(market_path),
        "portfolio": _sqlite_contract(portfolio_path),
        "mops": _sqlite_contract(mops.database_path),
    }
    expected = _fixture()["sqlite"]
    # Reviewed additive header metadata/index only; keep the legacy golden fixture and
    # compare every other table/column/index unchanged, rather than regenerate it.
    snapshot_columns = observed["mops"]["tables"]["fundamental_snapshots"]
    assert snapshot_columns == expected["mops"]["tables"]["fundamental_snapshots"] + [
        "filing_schema_version", "has_mops_filings",
    ]
    observed["mops"]["tables"]["fundamental_snapshots"] = snapshot_columns[:-2]
    assert "fundamental_snapshots_header" in observed["mops"]["indexes"]
    observed["mops"]["indexes"].remove("fundamental_snapshots_header")
    assert observed == expected
