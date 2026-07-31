import datetime

import pytest

from domain.trading_days import build_backtrack_dates


def test_build_backtrack_dates_uses_inclusive_end_and_exclusive_begin() -> None:
    reference_date = datetime.date(2026, 7, 20)

    assert build_backtrack_dates(reference_date, begin_backtrack=0, end_backtrack=5) == [
        datetime.date(2026, 7, 15),
        datetime.date(2026, 7, 16),
        datetime.date(2026, 7, 17),
        datetime.date(2026, 7, 18),
        datetime.date(2026, 7, 19),
    ]
    assert build_backtrack_dates(reference_date, begin_backtrack=2, end_backtrack=5) == [
        datetime.date(2026, 7, 15),
        datetime.date(2026, 7, 16),
        datetime.date(2026, 7, 17),
    ]


def test_build_backtrack_dates_can_include_reference_date_explicitly() -> None:
    reference_date = datetime.date(2026, 7, 20)

    assert build_backtrack_dates(
        reference_date,
        begin_backtrack=0,
        end_backtrack=5,
        include_reference=True,
    ) == [
        datetime.date(2026, 7, 15),
        datetime.date(2026, 7, 16),
        datetime.date(2026, 7, 17),
        datetime.date(2026, 7, 18),
        datetime.date(2026, 7, 19),
        datetime.date(2026, 7, 20),
    ]


def test_include_reference_does_not_override_nonzero_begin_boundary() -> None:
    reference_date = datetime.date(2026, 7, 20)

    assert build_backtrack_dates(
        reference_date,
        begin_backtrack=2,
        end_backtrack=5,
        include_reference=True,
    ) == [
        datetime.date(2026, 7, 15),
        datetime.date(2026, 7, 16),
        datetime.date(2026, 7, 17),
    ]


@pytest.mark.parametrize(
    ("begin_backtrack", "end_backtrack"),
    [(-1, 5), (0, -1), (5, 5), (6, 5)],
)
def test_build_backtrack_dates_rejects_invalid_boundaries(
    begin_backtrack: int,
    end_backtrack: int,
) -> None:
    with pytest.raises(ValueError):
        build_backtrack_dates(
            datetime.date(2026, 7, 20),
            begin_backtrack=begin_backtrack,
            end_backtrack=end_backtrack,
        )
