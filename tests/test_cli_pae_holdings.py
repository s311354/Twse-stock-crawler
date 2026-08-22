from pathlib import Path

import pytest

from config.settings import SettingsError
from interface.cli import build_run_configuration
from interface.cli import load_pae_holdings_file


def test_load_pae_holdings_file_accepts_symbol_only_list(tmp_path: Path) -> None:
    path = tmp_path / "holdings.txt"
    path.write_text("(2449,3008,3016,6271)\n", encoding="utf-8")

    positions = load_pae_holdings_file(path)

    assert [position.symbol for position in positions] == [
        "2449",
        "3008",
        "3016",
        "6271",
    ]
    assert all(position.shares is None for position in positions)


def test_load_pae_holdings_file_accepts_explicit_shares_and_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "holdings.txt"
    path.write_text(
        "2449:1000; 3008:20 # broker position\n",
        encoding="utf-8",
    )

    positions = load_pae_holdings_file(path)

    assert [(position.symbol, position.shares) for position in positions] == [
        ("2449", 1000.0),
        ("3008", 20.0),
    ]


def test_load_pae_holdings_file_accepts_entry_date_les_and_empty_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "holdings.txt"
    path.write_text(
        "2449:125:2026-08-05:82; 3008::2026-07-01:; "
        "3016:342:2026-08-05:\n",
        encoding="utf-8",
    )

    positions = load_pae_holdings_file(path)

    assert positions[0].shares == 125
    assert positions[0].entry_date.isoformat() == "2026-08-05"
    assert positions[0].entry_les == 82
    assert positions[1].shares is None
    assert positions[1].entry_date.isoformat() == "2026-07-01"
    assert positions[1].entry_les is None
    assert positions[2].shares == 342
    assert positions[2].entry_date.isoformat() == "2026-08-05"
    assert positions[2].entry_les is None


@pytest.mark.parametrize(
    "content",
    (
        "2449,2449",
        "2449,3008,3016,6271,2330",
        "bad-symbol",
        "2449:zero",
        "2449:100:2026/08/05:82",
        "2449:100:2026-08-05:bad",
    ),
)
def test_load_pae_holdings_file_rejects_invalid_input(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "holdings.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SettingsError):
        load_pae_holdings_file(path)


def test_cli_appends_holdings_to_analysis_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stocklist = tmp_path / "stocklist"
    stocklist.write_text("2317;", encoding="utf-8")
    holdings = tmp_path / "holdings.txt"
    holdings.write_text("2449,3008", encoding="utf-8")
    monkeypatch.setenv(
        "TWSE_MARKET_HISTORY_CACHE_PATH",
        str(tmp_path / "market.sqlite3"),
    )
    monkeypatch.setenv(
        "TWSE_PAE_PERSISTENCE_PATH",
        str(tmp_path / "pae.sqlite3"),
    )

    request, settings = build_run_configuration(
        [
            "--skip-mops",
            "--pae-holdings-file",
            str(holdings),
            "-e",
            "10",
            "-b",
            "0",
            str(stocklist),
        ]
    )

    assert request.stocklist == ["2317", "2449", "3008"]
    assert [position.symbol for position in settings.pae_positions] == [
        "2449",
        "3008",
    ]
