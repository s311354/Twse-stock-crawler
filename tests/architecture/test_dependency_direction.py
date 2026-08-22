"""Executable dependency rules for the clean architecture boundaries."""

from __future__ import annotations

import ast
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


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


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
