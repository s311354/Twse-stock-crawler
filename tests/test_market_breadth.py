import pytest

from domain.market_breadth import derive_twse_market_breadth


def _row(
    stock_no: str,
    *,
    close: object,
    sign: object,
    change: object,
) -> list[object]:
    return [
        stock_no,
        "測試",
        "1000",
        "10",
        "100000",
        close,
        close,
        close,
        close,
        sign,
        change,
        close,
        "1",
        close,
        "1",
        "10",
    ]


def test_market_breadth_derives_returns_and_deduplicates_symbols() -> None:
    snapshot = derive_twse_market_breadth(
        [
            _row("2330", close="110", sign="+", change="10"),
            _row("2303", close="90", sign="-", change="10"),
            _row("2454", close="100", sign="", change="0"),
            _row("2330", close="90", sign="-", change="10"),
        ],
        as_of="2026-07-31",
        universe="ELEC",
        source_final=True,
        taiex_return=0.025,
    )

    assert snapshot.eligible_count == 3
    assert snapshot.valid_return_count == 3
    assert snapshot.advancers == 1
    assert snapshot.decliners == 1
    assert snapshot.unchanged == 1
    assert snapshot.advancer_ratio == pytest.approx(1 / 3)
    assert snapshot.median_return == 0
    assert snapshot.coverage == 1
    assert snapshot.taiex_return == pytest.approx(0.025)


def test_market_breadth_accepts_html_signs_and_excludes_non_common_codes() -> None:
    snapshot = derive_twse_market_breadth(
        [
            _row(
                "2330",
                close="110",
                sign="<p style= color:red>+</p>",
                change="10",
            ),
            _row(
                "2303",
                close="90",
                sign="<p style= color:green>-</p>",
                change="10",
            ),
            _row("0050", close="100", sign="+", change="1"),
            _row("1234A", close="100", sign="+", change="1"),
        ],
        as_of="2026-07-31",
        universe="ELEC",
        source_final=True,
    )

    assert snapshot.eligible_count == 2
    assert snapshot.valid_return_count == 2
    assert snapshot.advancers == 1
    assert snapshot.decliners == 1


def test_invalid_return_evidence_reduces_breadth_coverage() -> None:
    snapshot = derive_twse_market_breadth(
        [
            _row("2330", close="110", sign="+", change="10"),
            _row("2303", close="--", sign="-", change="10"),
            _row("2454", close="100", sign="X", change="5"),
        ],
        as_of="2026-07-31",
        universe="ELEC",
        source_final=False,
    )

    assert snapshot.eligible_count == 3
    assert snapshot.valid_return_count == 1
    assert snapshot.coverage == pytest.approx(1 / 3)
    assert snapshot.source_final is False
