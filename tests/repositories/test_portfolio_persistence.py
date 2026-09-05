import datetime

from application.portfolio_allocation import PersistenceObservation
from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
from infrastructure.storage.portfolio_persistence import (
    SQLitePortfolioPersistence,
)


def test_persistence_upserts_same_day_and_loads_latest_three_sessions(
    tmp_path,
) -> None:
    repository = SQLitePortfolioPersistence(tmp_path / "pae.sqlite3")
    for day, excess in (
        (1, 0.01),
        (2, -0.01),
        (3, 0.02),
        (4, 0.03),
    ):
        repository.record_session(
            datetime.date(2026, 8, day),
            (PersistenceObservation("2330", "2317", excess),),
        )
    repository.record_session(
        datetime.date(2026, 8, 4),
        (PersistenceObservation("2330", "2317", 0.04),),
    )

    evidence = repository.load_recent(
        ("2330",),
        ("2317",),
        through_date=datetime.date(2026, 8, 4),
        limit_per_pair=3,
    )

    assert len(evidence) == 1
    assert evidence[0].recent_excess_opportunities == (-0.01, 0.02, 0.04)


def test_portfolio_persistence_isolates_v2_boolean_evidence(tmp_path) -> None:
    repository = SQLitePortfolioPersistence(tmp_path / "pae-v2.sqlite3")
    repository.record_session_v2(
        datetime.date(2026, 8, 27),
        (PersistenceObservationV2("2330", "2454", True),),
    )
    repository.record_session_v2(
        datetime.date(2026, 8, 28),
        (PersistenceObservationV2("2330", "2454", False),),
    )

    evidence = repository.load_recent_v2(
        ("2330",),
        ("2454",),
        through_date=datetime.date(2026, 8, 28),
        limit_per_pair=3,
    )

    assert len(evidence) == 1
    assert evidence[0].recent_passes == (True, False)
    assert repository.load_recent(
        ("2330",),
        ("2454",),
        through_date=datetime.date(2026, 8, 28),
        limit_per_pair=3,
    ) == ()
