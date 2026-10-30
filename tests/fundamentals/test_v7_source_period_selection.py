"""SYNTHETIC ONLY: bounded cutoff-bound source selection, not financial repair."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from application.fundamental_service import MopsFundamentalService


CUTOFF = "2026-10-14T13:30:00+08:00"


def row(period, *, available_at="2026-07-21T00:00:00Z", source="MOPS_IXBRL", revision="1"):
    return dict(source=source, period_end=pd.Period(period, freq="Q").end_time.date().isoformat(),
        available_at=available_at,
        content_sha256=hashlib.sha256(f"SYNTHETIC:{period}:{revision}".encode()).hexdigest())


class SourcesPort:
    def __init__(self, values=None):
        self.values = values or dict(financial_audits={}, financial_supplements={})
        self.calls = []
        self.uow_calls = 0
        self.uow = object()

    @contextmanager
    def unit_of_work(self):
        self.uow_calls += 1
        yield self.uow

    def load_v7_financial_sources(self, symbol, as_of, *, periods, uow):
        assert uow is self.uow
        self.calls.append(dict(symbol=symbol, as_of=as_of, periods=periods))
        return self.values


def load(repository, records, *, cutoff=CUTOFF, market="TWSE"):
    service = SimpleNamespace(repository=repository)
    return MopsFundamentalService.load_v7_financial_sources(service, "SYNTHETIC", as_of=cutoff,
        records=records, market=market)


def test_union_includes_older_annual_origins_without_duplicate_quarters():
    records = [row("2026Q2"), *(row(f"{year}Q4") for year in range(2018, 2026))]
    before = deepcopy(records)
    repository = SourcesPort()
    load(repository, records)
    periods = repository.calls[0]["periods"]
    recent = tuple(((pd.Period("2026Q2") - i).year, (pd.Period("2026Q2") - i).quarter) for i in range(9))
    assert periods[:9] == recent
    assert set(periods) == set(recent) | {(year, 4) for year in range(2021, 2026)}
    assert len(periods) == len(set(periods)) == 12
    assert repository.uow_calls == len(repository.calls) == 1
    assert records == before


def test_sparse_annual_history_is_bounded_by_actual_latest_five_origins():
    records = [row("2026Q2"), *(row(f"{year}Q4") for year in range(2000, 2021))]
    repository = SourcesPort()
    load(repository, records)
    periods = repository.calls[0]["periods"]
    assert len(periods) == 14
    assert {(year, 4) for year in range(2016, 2021)} <= set(periods)
    assert (2015, 4) not in periods


def test_future_annual_and_noncanonical_sources_do_not_change_visible_periods():
    records = [row("2026Q2"), row("2021Q4"),
        row("2026Q4", available_at="2027-01-01T00:00:00Z"),
        row("2025Q4", source="TWSE_OPENAPI"),
        row("2024Q4", available_at="unverified")]
    repository = SourcesPort()
    load(repository, records)
    periods = repository.calls[0]["periods"]
    assert periods[0] == (2026, 2)
    assert (2021, 4) in periods
    assert (2026, 4) not in periods
    assert len(periods) == 10  # latest nine plus the actual visible old annual


def test_returned_proofs_remain_bound_to_cutoff_visible_source_hashes_and_market():
    visible = row("2021Q4")
    future = row("2021Q4", available_at="2026-11-01T00:00:00Z", revision="2")
    ignored = row("2025Q4", source="TWSE_OPENAPI")
    proof = dict(market="TWSE", available_at=visible["available_at"],
                 content_sha256=visible["content_sha256"], revision_id="SYNTHETIC_ORIGINAL")
    values = {group: {visible["content_sha256"]: deepcopy(proof),
                      future["content_sha256"]: {}, ignored["content_sha256"]: {}, "unselected": {}}
              for group in ("financial_audits", "financial_supplements")}
    before = deepcopy(values)
    repository = SourcesPort(values)
    result = load(repository, [visible, future, ignored], market="TPEX")
    assert repository.calls[0]["as_of"] == "2026-10-14T05:30:00+00:00"
    for group in result:
        assert set(result[group]) == {visible["content_sha256"]}
        assert result[group][visible["content_sha256"]] == dict(proof, market="TPEX")
    assert values == before


def test_each_cutoff_reselects_originals_and_does_not_reuse_future_revision():
    old = row("2021Q4")
    revision = row("2026Q4", available_at="2027-03-01T00:00:00Z", revision="2")
    repository = SourcesPort()
    load(repository, [old, revision])
    load(repository, [old, revision], cutoff="2027-04-01T13:30:00+08:00")
    assert repository.calls[0]["periods"][0] == (2021, 4)
    assert repository.calls[1]["periods"][0] == (2026, 4)
    assert repository.uow_calls == 2


@pytest.mark.parametrize("records", [[], [row("2025Q4", source="TWSE_OPENAPI")],
    [row("2025Q4", available_at="2027-01-01T00:00:00Z")]])
def test_empty_eligible_input_does_not_open_storage(records):
    repository = SourcesPort()
    assert load(repository, records) == dict(financial_audits={}, financial_supplements={})
    assert repository.calls == [] and repository.uow_calls == 0


def test_legacy_repository_without_public_source_port_keeps_explicit_gaps():
    assert load(SimpleNamespace(), [row("2025Q4")]) == dict(
        financial_audits={}, financial_supplements={})
