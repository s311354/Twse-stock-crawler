"""Executable dependency rules for the clean architecture boundaries."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORT_ROOTS = {
    "domain": {
        "application",
        "config",
        "infrastructure",
        "interface",
        "stockanalysis",
        "twse",
        "twstockcrawler",
    },
    "application": {
        "config",
        "infrastructure",
        "interface",
        "stockanalysis",
        "twse",
        "twstockcrawler",
    },
}
PRODUCTION_PAE_FILES = (
    "application/portfolio_allocation_v2.py",
    "infrastructure/stock_analysis_adapters.py",
    "infrastructure/runtime/strategy_adapter.py",
    "infrastructure/storage/portfolio_persistence.py",
)
PAE_V2_FACADE = PROJECT_ROOT / "domain/strategies/portfolio_allocation_v2.py"
LES_V5_FACADE = PROJECT_ROOT / "domain/strategies/low_entry_score_v5.py"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def _import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_IMPORT_ROOTS))
def test_inner_layers_do_not_import_outer_layers(layer: str) -> None:
    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / layer).rglob("*.py")):
        forbidden = _import_roots(path) & FORBIDDEN_IMPORT_ROOTS[layer]
        if forbidden:
            relative_path = path.relative_to(PROJECT_ROOT)
            violations.append(
                f"{relative_path}: {', '.join(sorted(forbidden))}"
            )

    assert not violations, "Dependency direction violations:\n" + "\n".join(
        violations
    )


def test_production_pae_does_not_depend_on_v1_application_pipeline() -> None:
    violations = [
        relative_path
        for relative_path in PRODUCTION_PAE_FILES
        if "application.portfolio_allocation"
        in _import_modules(PROJECT_ROOT / relative_path)
    ]

    assert not violations, (
        "Production PAE imports the v1 application pipeline: "
        + ", ".join(violations)
    )


def test_v1_portfolio_module_reexports_shared_contract_identity() -> None:
    legacy = importlib.import_module("application.portfolio_allocation")
    contracts = importlib.import_module("application.portfolio_contracts")

    for name in (
        "PersistenceObservation",
        "PortfolioAllocationInputError",
        "PortfolioPersistenceError",
        "PortfolioRuntimeContext",
    ):
        assert getattr(legacy, name) is getattr(contracts, name)

    for name in (
        "PAE_INPUT_COLD_START",
        "PAE_INPUT_EXPLICIT",
        "PAE_INPUT_MIXED",
        "PAE_INPUT_SYMBOL_ONLY",
    ):
        assert getattr(legacy, name) == getattr(contracts, name)


def test_pae_v2_facade_does_not_reabsorb_policy_implementations() -> None:
    tree = ast.parse(
        PAE_V2_FACADE.read_text(encoding="utf-8"),
        filename=str(PAE_V2_FACADE),
    )
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PortfolioAllocationEngineV2"
    )
    method_names = {
        node.name
        for node in engine.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden = {
        "_breach_lines",
        "_direct_selection_lines",
        "_fast_path",
        "_full_portfolio_lines",
        "_holding_line",
        "_open_slot_lines",
        "_portfolio_improves",
        "_replacement_comparison",
        "_switch_hurdle",
        "_target_weight",
    }

    assert method_names.isdisjoint(forbidden)


def test_les_v5_strategy_composes_without_eager_rule_engine() -> None:
    tree = ast.parse(
        LES_V5_FACADE.read_text(encoding="utf-8"),
        filename=str(LES_V5_FACADE),
    )
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    compatibility = classes["LowEntryV5RuleEngine"]
    compatibility_methods = {
        node.name
        for node in compatibility.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    strategy = classes["LowEntryScoreV5Strategy"]
    initializer = next(
        node
        for node in strategy.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    called_names = {
        node.func.id
        for node in ast.walk(initializer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert compatibility_methods == {"__init__"}
    assert "LowEntryV5CompositionRoot" in called_names
    assert "LowEntryV5RuleEngine" not in called_names
