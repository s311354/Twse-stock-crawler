"""SYNTHETIC research state tests; no real account, market, mail or orders."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PersistenceObservation, PortfolioPersistenceError
from application.replay_contracts import digest
from domain.strategies.portfolio_allocation_v2 import (
    PersistenceObservationV2, PortfolioActionV2,
)
from infrastructure.storage.portfolio_persistence import SQLitePortfolioPersistence
from infrastructure.storage.research_portfolio import (
    APPLICATION_ID, ResearchStateError, SQLiteResearchPortfolioPersistence,
)
from tests.strategies.test_portfolio_allocation_v2 import request, security


IDENTITY = dict(schema="SYNTHETIC_RESEARCH_ONLY", research_id="test-research",
    model="SYNTHETIC_MODEL_V1", versions=dict(code="SYNTHETIC_CODE_V1"),
    symbols=["1000", "1001", "1002", "1003", "2330"])


def day(number):
    return dt.date(2026, 8, number)


def native_request(number=1, *, hour=13, cash=200000):
    holdings = tuple(security(str(1000 + i), current_weight=.20, quality_score=80,
        bear_cagr=.05, base_cagr=.08, bull_cagr=.11) for i in range(4))
    candidate = security("2330", quality_score=84, data_confidence=80,
        bear_cagr=.18, base_cagr=.25, bull_cagr=.32)
    return replace(request(holdings=holdings, candidates=(candidate,), cash_weight=.20),
        decision_time=dt.datetime(2026, 8, number, hour, tzinfo=dt.timezone.utc),
        available_cash=cash)


def pipeline(path, *, identity=None, check_version=None):
    return PortfolioAllocationV2Pipeline(persistence=SQLiteResearchPortfolioPersistence(
        path, deepcopy(IDENTITY) if identity is None else identity, check_version=check_version))


def rows(path):
    with sqlite3.connect(path) as connection:
        sessions = connection.execute("SELECT * FROM research_sessions ORDER BY trade_date").fetchall()
        observations = connection.execute("SELECT * FROM pae_opportunity_evidence "
            "ORDER BY trade_date,incumbent_symbol,challenger_symbol,model_version").fetchall()
        return sessions, observations


def test_native_engine_full_results_match_ordinary_port_across_restarts(tmp_path):
    normal = PortfolioAllocationV2Pipeline(persistence=SQLitePortfolioPersistence(tmp_path / "normal.sqlite3"))
    path = tmp_path / "research.sqlite3"
    for number in (1, 2, 5, 8):
        current = native_request(number)
        expected = normal._evaluate_with_persistence(current, day(number))
        actual = pipeline(path)._evaluate_with_persistence(current, day(number))
        assert actual == expected  # Every business/audit/account field, not only the action.
    assert len(rows(path)[0]) == 4 and len(rows(path)[1]) == 16


def test_same_day_later_cutoff_is_idempotent_and_cannot_fake_two_of_three(tmp_path):
    path = tmp_path / "research.sqlite3"
    first = pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    assert next(line for line in first.lines if line.stock_id == "2330").action is PortfolioActionV2.NO_ACTION
    for hour in (14, 15, 16, 17):
        result = pipeline(path)._evaluate_with_persistence(native_request(hour=hour), day(1))
        assert next(line for line in result.lines if line.stock_id == "2330").action is PortfolioActionV2.NO_ACTION
    assert len(rows(path)[0]) == 1 and len(rows(path)[1]) == 4
    second = pipeline(path)._evaluate_with_persistence(native_request(2), day(2))
    candidate = next(line for line in second.lines if line.stock_id == "2330")
    assert candidate.persistence_pass is True and candidate.action is PortfolioActionV2.ROTATE_75
    assert len(rows(path)[0]) == 2 and len(rows(path)[1]) == 8


def test_same_day_account_dependent_allocation_is_recomputed_not_cached(tmp_path):
    path = tmp_path / "research.sqlite3"
    repo = SQLiteResearchPortfolioPersistence(path, IDENTITY)
    calls = []
    def allocate(cash):
        def run(session):
            calls.append(cash)
            session.record_session_v2(day(1), [PersistenceObservationV2("1000", "2330", True)])
            return dict(actual_cash=cash, history=session.load_recent_v2(
                ["1000"], ["2330"], through_date=day(1), limit_per_pair=3))
        return repo.evaluate_session_v2(replace(native_request(), available_cash=cash), day(1), evaluate=run)
    assert allocate(900000)["actual_cash"] == 900000
    assert allocate(123456)["actual_cash"] == 123456
    assert calls == [900000, 123456]
    assert len(rows(path)[0]) == 1 and len(rows(path)[1]) == 1


@pytest.mark.parametrize("change", ["passed", "pair", "removed", "added"])
def test_same_day_changed_observation_set_is_rejected_and_preserves_state(tmp_path, change):
    path = tmp_path / "research.sqlite3"
    repo = SQLiteResearchPortfolioPersistence(path, IDENTITY)
    original = [PersistenceObservationV2("1000", "2330", True)]
    def call(observations):
        def evaluate(session):
            session.record_session_v2(day(1), observations)
            return "allocated"
        return repo.evaluate_session_v2(None, day(1), evaluate=evaluate)
    assert call(original) == "allocated"
    before = rows(path)
    changed = {"passed": [PersistenceObservationV2("1000", "2330", False)],
        "pair": [PersistenceObservationV2("1001", "2330", True)], "removed": [],
        "added": original + [PersistenceObservationV2("1001", "2330", True)]}[change]
    with pytest.raises(ResearchStateError, match="SAME_DAY_OBSERVATIONS_CHANGED"):
        call(changed)
    assert rows(path) == before


def test_same_day_new_empty_candidate_set_fails_through_public_pipeline(tmp_path, monkeypatch):
    path = tmp_path / "research.sqlite3"
    current = pipeline(path)
    current._evaluate_with_persistence(native_request(), day(1))
    before = rows(path)
    changed = replace(native_request(hour=15), candidates=())
    monkeypatch.setattr(current, "_build_request", lambda *_: (changed, day(1), "SYNTHETIC"))
    with pytest.raises(ResearchStateError, match="SAME_DAY_OBSERVATIONS_CHANGED"):
        current.enrich((SimpleNamespace(evaluated_at=changed.decision_time),), SimpleNamespace())
    assert rows(path) == before


def test_empty_portfolio_and_no_candidates_still_seal_monotonic_headers(tmp_path):
    path = tmp_path / "empty.sqlite3"
    empty = replace(native_request(2), holdings=(), candidates=(), available_cash=1_000_000)
    pipeline(path)._evaluate_with_persistence(empty, day(2))
    sessions, observations = rows(path)
    assert sessions == [(day(2).isoformat(), digest([]), 0)] and observations == []
    assert pipeline(path)._evaluate_with_persistence(empty, day(2)) == pipeline(path)._evaluate_with_persistence(empty, day(2))
    with pytest.raises(ResearchStateError, match="OUT_OF_ORDER"):
        pipeline(path)._evaluate_with_persistence(replace(empty, decision_time=native_request().decision_time), day(1))
    with pytest.raises(ResearchStateError, match="SAME_DAY_OBSERVATIONS_CHANGED"):
        pipeline(path)._evaluate_with_persistence(native_request(2), day(2))
    assert rows(path) == (sessions, observations)


@pytest.mark.parametrize("field,value", [("model", "SYNTHETIC_V2"),
    ("versions", {"code": "SYNTHETIC_NEW_CODE"}), ("symbols", ["2330"])])
def test_versions_model_or_universe_change_rejected_before_mutation(tmp_path, field, value):
    path = tmp_path / "research.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    before = path.read_bytes()
    identity = deepcopy(IDENTITY)
    identity[field] = value
    with pytest.raises(ResearchStateError, match="VERSION_MODEL_OR_UNIVERSE_CHANGED"):
        SQLiteResearchPortfolioPersistence(path, identity)
    assert path.read_bytes() == before


def test_constructor_detaches_caller_identity_and_rechecks_before_each_session(tmp_path):
    path = tmp_path / "research.sqlite3"
    identity = deepcopy(IDENTITY)
    repo = SQLiteResearchPortfolioPersistence(path, identity)
    identity["symbols"].append("MUTATED_CALLER_ONLY")
    repo.evaluate_session_v2(None, day(1), evaluate=lambda session: "no-account-cache")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE research_identity SET identity_sha256='tampered'")
    with pytest.raises(ResearchStateError, match="VERSION_MODEL_OR_UNIVERSE_CHANGED"):
        repo.evaluate_session_v2(None, day(2), evaluate=lambda session: pytest.fail("identity bypassed"))
    assert len(rows(path)[0]) == 1


@pytest.mark.parametrize("foreign", ["ordinary", "other", "future-schema"])
def test_foreign_and_ordinary_database_rejected_without_changing_bytes(tmp_path, foreign):
    path = tmp_path / "existing.sqlite3"
    if foreign == "ordinary":
        SQLitePortfolioPersistence(path).record_session_v2(day(1), [PersistenceObservationV2("1000", "2330", True)])
    elif foreign == "other":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE actual_account_ledger (amount REAL)")
            connection.execute("INSERT INTO actual_account_ledger VALUES (1000000)")
    else:
        pipeline(path)._evaluate_with_persistence(native_request(), day(1))
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA user_version=999")
    before = path.read_bytes()
    with pytest.raises(ResearchStateError, match="LIVE_OR_FOREIGN_DATABASE"):
        SQLiteResearchPortfolioPersistence(path, IDENTITY)
    assert path.read_bytes() == before


def test_ordinary_port_refuses_research_database_without_changing_bytes(tmp_path):
    path = tmp_path / "research.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    before = path.read_bytes()
    with pytest.raises(PortfolioPersistenceError, match="RESEARCH_OR_FOREIGN_DATABASE_FORBIDDEN"):
        SQLitePortfolioPersistence(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["read-v1", "write-v1", "read-v2", "write-v2"])
@pytest.mark.parametrize("redirect", ["replace-file", "retarget-object"])
def test_existing_ordinary_port_rechecks_research_isolation_on_each_connection(tmp_path, operation, redirect):
    research = tmp_path / "research.sqlite3"
    pipeline(research)._evaluate_with_persistence(native_request(), day(1))
    ordinary = tmp_path / "ordinary.sqlite3"
    repo = SQLitePortfolioPersistence(ordinary)
    if redirect == "replace-file":
        ordinary.write_bytes(research.read_bytes())
        target = ordinary
    else:
        repo.database_path = research
        target = research
    before = target.read_bytes()
    operations = {
        "read-v1": lambda: repo.load_recent(["1000"], ["2330"], through_date=day(1), limit_per_pair=3),
        "write-v1": lambda: repo.record_session(day(2), [PersistenceObservation("1000", "2330", 1.0)]),
        "read-v2": lambda: repo.load_recent_v2(["1000"], ["2330"], through_date=day(1), limit_per_pair=3),
        "write-v2": lambda: repo.record_session_v2(day(2), [PersistenceObservationV2("1000", "2330", True)]),
    }
    with pytest.raises(PortfolioPersistenceError, match="RESEARCH_OR_FOREIGN_DATABASE_FORBIDDEN"):
        operations[operation]()
    assert target.read_bytes() == before


def test_different_research_paths_evolve_independently(tmp_path):
    first, second = tmp_path / "first.sqlite3", tmp_path / "second.sqlite3"
    pipeline(first)._evaluate_with_persistence(native_request(), day(1))
    pipeline(first)._evaluate_with_persistence(native_request(2), day(2))
    identity = dict(IDENTITY, research_id="SYNTHETIC_OTHER_RESEARCH")
    result = pipeline(second, identity=identity)._evaluate_with_persistence(native_request(2), day(2))
    assert next(line for line in result.lines if line.stock_id == "2330").action is PortfolioActionV2.NO_ACTION
    assert len(rows(first)[1]) == 8 and len(rows(second)[1]) == 4


@pytest.mark.parametrize("stage", ["allocation", "post-version-check"])
def test_atomic_failure_rolls_back_observations_and_header_then_retry_succeeds(tmp_path, stage):
    path = tmp_path / "research.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    before = rows(path)
    checked = []
    def check():
        checked.append(1)
        if stage == "post-version-check" and len(checked) == 2:
            raise ValueError("SYNTHETIC_SOURCE_DRIFT")
    repo = SQLiteResearchPortfolioPersistence(path, IDENTITY, check_version=check)
    def interrupt(session):
        session.record_session_v2(day(2), [PersistenceObservationV2("1000", "2330", True)])
        if stage == "allocation":
            raise RuntimeError("SYNTHETIC_ALLOCATION_INTERRUPTED")
        return "allocated"
    with pytest.raises(RuntimeError, match="SYNTHETIC_ALLOCATION_INTERRUPTED|VERSION_CHANGED_DURING_SESSION"):
        repo.evaluate_session_v2(None, day(2), evaluate=interrupt)
    assert rows(path) == before
    pipeline(path)._evaluate_with_persistence(native_request(2), day(2))
    assert len(rows(path)[0]) == 2 and len(rows(path)[1]) == 8


def test_failed_first_session_is_retryable_and_never_leaves_complete_header(tmp_path):
    path = tmp_path / "new.sqlite3"
    repo = SQLiteResearchPortfolioPersistence(path, IDENTITY)
    def interrupt(session):
        session.record_session_v2(day(1), [PersistenceObservationV2("1000", "2330", True)])
        raise RuntimeError("SYNTHETIC_FIRST_INTERRUPTED")
    with pytest.raises(RuntimeError, match="FIRST_INTERRUPTED"):
        repo.evaluate_session_v2(None, day(1), evaluate=interrupt)
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    assert len(rows(path)[0]) == 1 and len(rows(path)[1]) == 4


def test_same_day_parallel_retries_serialize_and_record_once(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    def run(_):
        return pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert results[0] == results[1]
    assert len(rows(path)[0]) == 1 and len(rows(path)[1]) == 4


def test_corrupt_prior_observation_cannot_be_used_as_persistence(tmp_path):
    path = tmp_path / "research.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE pae_opportunity_evidence SET excess_opportunity=0")
    with pytest.raises(ResearchStateError, match="SESSION_HASH_MISMATCH"):
        pipeline(path)._evaluate_with_persistence(native_request(2), day(2))
    assert len(rows(path)[0]) == 1 and len(rows(path)[1]) == 4


def test_store_is_research_identified_and_pair_primary_key_stays_native(tmp_path):
    path = tmp_path / "research.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        columns = connection.execute("PRAGMA table_info(pae_opportunity_evidence)").fetchall()
        assert [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]] == [
            "trade_date", "incumbent_symbol", "challenger_symbol", "model_version"]


@pytest.mark.parametrize("role", ["holding", "candidate"])
def test_public_port_rejects_request_outside_frozen_scope_and_rolls_back(tmp_path, role):
    path = tmp_path / "scope.sqlite3"
    current = pipeline(path)
    current._evaluate_with_persistence(native_request(), day(1))
    before = rows(path)
    request = native_request(2)
    if role == "holding":
        request = replace(request, holdings=(security("9999", current_weight=.20),))
    else:
        request = replace(request, candidates=(security("9999"),))
    with pytest.raises(ResearchStateError, match="REQUEST_OUTSIDE_FROZEN_UNIVERSE"):
        current._evaluate_with_persistence(request, day(2))
    assert rows(path) == before


def test_public_observation_port_rejects_frozen_scope_bypass(tmp_path):
    path = tmp_path / "scope.sqlite3"
    pipeline(path)._evaluate_with_persistence(native_request(), day(1))
    before = rows(path)
    def bypass(session):
        session.record_session_v2(day(2), [PersistenceObservationV2("1000", "9999", True)])
        pytest.fail("outside observation accepted")
    with pytest.raises(ResearchStateError, match="OBSERVATION_OUTSIDE_FROZEN_UNIVERSE"):
        SQLiteResearchPortfolioPersistence(path, IDENTITY).evaluate_session_v2(None, day(2), evaluate=bypass)
    assert rows(path) == before
