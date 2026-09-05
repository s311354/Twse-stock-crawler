"""Contract tests for the AIStock package integration boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from aistock_twse_integration import CONTRACT_NAME
from aistock_twse_integration import CONTRACT_VERSION
from aistock_twse_integration import PACKAGE_VERSION
from aistock_twse_integration import package_contract
from aistock_twse_integration.owner_research import select_output_lines


def test_package_contract_is_explicit_and_versioned() -> None:
    assert package_contract() == {
        "name": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "packageVersion": PACKAGE_VERSION,
    }
    assert package_contract() == {
        "name": "aistock.twse.owner-research",
        "version": 1,
        "packageVersion": "1.0.0",
    }
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["project"]["version"] == PACKAGE_VERSION


def test_owner_research_selection_preserves_entry_and_pae_signals() -> None:
    lines = [
        {"code": "2330", "role": "HOLDING", "lowEntry": {"decision": "AVOID"}},
        *[
            {
                "code": str(3000 + index),
                "role": "CANDIDATE",
                "lowEntry": {"decision": "NO_BUY"},
            }
            for index in range(30)
        ],
        {"code": "2441", "role": "CANDIDATE", "lowEntry": {"decision": "BUY_SMALL"}},
        {"code": "3034", "role": "CANDIDATE", "lowEntry": {"decision": "EARLY_BUY"}},
        {"code": "6799", "role": "CANDIDATE", "lowEntry": {"decision": "BUY_SMALL"}},
        {
            "action": "ADD",
            "code": "7777",
            "role": "CANDIDATE",
            "lowEntry": {"decision": "WATCH"},
        },
    ]
    selected = select_output_lines(lines)
    codes = [item["code"] for item in selected]
    assert len(codes) == 24
    assert codes[0] == "2330"
    assert {"2441", "3034", "6799", "7777"}.issubset(codes)


def test_installed_package_accepts_configured_project_and_data_paths(tmp_path) -> None:
    project = tmp_path / "crawler-runtime"
    data = tmp_path / "crawler-data"
    environment = {
        **os.environ,
        "TWSE_DATA_DIR": str(data),
        "TWSE_PROJECT_ROOT": str(project),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from config.settings import DATA_DIR, PROJECT_ROOT; "
                "print(json.dumps([str(PROJECT_ROOT), str(DATA_DIR)]))"
            ),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.stdout.strip() == f'["{project}", "{data}"]'
