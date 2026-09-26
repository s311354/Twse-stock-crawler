"""Date -b/-e syntax activates the existing PIT replay, never live execution."""
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.contracts import AnalysisContext, MarketStatus
from application.use_cases import FetchMarketData
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from domain.trading_days import build_backtrack_dates
from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
from interface.cli import build_run_configuration, build_preflight_summary, create_parser, main
from tests.test_stockanalysis import CapturingTwseCrawler
from tests.strategies.test_daily_v6_pipeline import daily_market, build


ROOT = Path(__file__).resolve().parents[1]


def date_args(tmp_path, start="2024-10-25", end="2026-09-16"):
    holdings = tmp_path / "historical.txt"
    holdings.write_text("# explicit historical all-cash account\n")
    return ["2330", "-b", start, "-e", end, "--pae-holdings-file", str(holdings),
            "--pae-current-cash", "900000", "--account-equity", "1000000"]


class Calendar:
    def __init__(self, closed=()):
        self.closed = set(closed)

    def is_trading_day(self, day):
        return day.weekday() < 5 and day not in self.closed

    def trading_day_status(self, day):
        return TradingDayStatus.OPEN if self.is_trading_day(day) else TradingDayStatus.CLOSED


def fetch(req, calendar=None):
    crawler = CapturingTwseCrawler(1)
    crawler.stocknumbers = ["2330"]
    crawler.get_twse_daily_stocks = Mock()
    adapter = CrawlerMarketDataAdapter(lambda count: crawler)
    clock = Mock(side_effect=AssertionError("replay must not consult today's clock"))
    context = FetchMarketData(adapter, calendar or Calendar(), clock, clock).execute(req)
    return context, crawler


def test_date_pair_infers_close_and_preserves_inclusive_start(tmp_path, monkeypatch):
    monkeypatch.setenv("TWSE_PAE_HOLDINGS_JSON", '[{"symbol":"3008","shares":10,"entry_date":"2026-09-25"}]')
    req, settings = build_run_configuration(date_args(tmp_path))
    assert req.as_of.isoformat() == "2026-09-16T13:30:00+08:00"
    assert req.history_start_date == dt.date(2024, 10, 25)
    assert req.history_end_date == dt.date(2026, 9, 16)
    assert type(req.beginbacktrack) is int and type(req.endbacktrack) is int
    assert req.beginbacktrack == 0
    assert req.endbacktrack == (dt.date(2026, 9, 16) - dt.date(2024, 10, 25)).days
    assert settings.pae_positions == () and not req.notification_enabled
    summary = build_preflight_summary(req, settings)
    assert summary["evaluation_mode"] == "HISTORICAL_REPLAY"
    assert summary["date_range"] == {"start": "2024-10-25", "end": "2026-09-16", "inclusive": True}
    context, crawler = fetch(req)
    assert context.as_of == req.as_of
    assert crawler.iso_scheduled_times[0] == "2024-10-25"
    assert crawler.iso_scheduled_times[-1] == "2026-09-16"
    assert crawler.low_entry_v4_signal_dates_by_stock["2330"] == req.as_of.isoformat()


@pytest.mark.parametrize("as_of", [
    "2026-09-16",
    "2026-09-16T09:00:00+08:00", "2026-09-16T01:00:00Z",
    "2026-09-16T13:29:59+08:00", "2026-09-16T13:30:00+08:00",
    "2026-09-16T15:00:00+08:00", "2026-09-16 13:30:00+08:00",
    "2026-09-16T05:30:00Z",
])
@pytest.mark.parametrize("integer_offsets", [False, True])
def test_removed_as_of_rejected_before_analysis(tmp_path, as_of, integer_offsets, monkeypatch):
    args = date_args(tmp_path, start="0", end="792") if integer_offsets else date_args(tmp_path)
    service = Mock(side_effect=AssertionError("removed option must not start analysis"))
    monkeypatch.setattr("interface.cli.build_stock_analysis_service", service)
    with pytest.raises(SystemExit) as error:
        main(args + ["--as-of", as_of])
    assert error.value.code == 2
    service.assert_not_called()


def test_removed_as_of_is_not_exposed_in_help_or_parser_actions():
    parser = create_parser()
    assert "--as-of" not in parser.format_help()
    assert all(action.dest != "as_of" for action in parser._actions)


def test_date_pair_retains_same_internal_cutoff_and_report_identity(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path))
    previous_cutoff = dt.datetime.fromisoformat("2026-09-16T13:30:00+08:00")
    assert req.as_of == previous_cutoff
    assert req.report_file_prefix == replace(req, as_of=previous_cutoff).report_file_prefix


def test_calendar_holidays_and_weekends_are_respected(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path, end="2026-09-20"))
    _, crawler = fetch(req)
    assert crawler.iso_scheduled_times[-1] == "2026-09-18"
    req, _ = build_run_configuration(date_args(tmp_path, end="2026-09-14"))
    _, crawler = fetch(req, Calendar([dt.date(2026, 9, 14)]))
    assert crawler.iso_scheduled_times[-1] == "2026-09-11"


def test_single_day_range_does_not_add_earlier_history(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path, start="2026-09-16"))
    _, crawler = fetch(req)
    assert crawler.iso_scheduled_times == ["2026-09-16"]


def test_old_end_cannot_stand_in_for_later_close(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path, end="2026-09-15"))
    # Defensive internal contract: manually inconsistent requests still fail.
    req = replace(req, as_of=dt.datetime.fromisoformat("2026-09-16T13:30:00+08:00"))
    with pytest.raises(ValueError, match="REPLAY_RANGE_MISSES_CUTOFF_SESSION"):
        fetch(req)


@pytest.mark.parametrize("start,end,extra", [
    ("0", "2026-09-16", []), ("2024-10-25", "792", []),
    ("2026-09-17", "2026-09-16", []), ("2026-02-30", "2026-09-16", []),
    ("2024-10-25", "2026-9-16", []), ("2024-10-25", "2999-01-01", []),
    ("2024-10-25", "2026-09-16T13:30:00+08:00", []),
    ("2024-10-25T00:00:00+08:00", "2026-09-16", []),
    ("2024-10-25", "2026-09-16", ["-l"]),
    ("2024-10-25", "2026-09-16", ["-m"]),
])
def test_invalid_or_ambiguous_ranges_rejected_before_running(tmp_path, start, end, extra):
    with pytest.raises(SystemExit):
        build_run_configuration(date_args(tmp_path, start=start, end=end) + extra)


def test_date_replay_still_requires_explicit_historical_portfolio():
    with pytest.raises(SystemExit):
        build_run_configuration(["2330", "-b", "2024-10-25", "-e", "2026-09-16"])


@pytest.mark.parametrize("begin,end", [(0, 792), (9, 792)])
def test_integer_mode_unchanged(begin, end):
    live, settings = build_run_configuration(["2330", "-b", str(begin), "-e", str(end)])
    assert (live.beginbacktrack, live.endbacktrack, live.as_of) == (begin, end, None)
    assert live.history_start_date is None and live.history_end_date is None
    assert build_preflight_summary(live, settings)["evaluation_mode"] == "LIVE"


def test_date_preflight_never_runs_service_or_sends_mail(tmp_path, monkeypatch, capsys):
    service = Mock(side_effect=AssertionError("preflight must not start analysis"))
    monkeypatch.setattr("interface.cli.build_stock_analysis_service", service)
    assert main(date_args(tmp_path) + ["--check-config"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["as_of"] == "2026-09-16T13:30:00+08:00"
    assert summary["mail_enabled"] is False and summary["date_range"]["start"] == "2024-10-25"
    service.assert_not_called()


@pytest.mark.parametrize("mail", [False, True])
def test_shell_date_pair_no_live_mail_or_current_holdings_fallback(tmp_path, mail):
    args = date_args(tmp_path)
    capture = tmp_path / "argv.txt"
    fake = tmp_path / "python"
    fake.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake.chmod(0o755)
    result = subprocess.run(["./stockdataanalysis.sh", "--profile", "elec", *args[1:], "--check"]
                            + (["--replay-mail"] if mail else []), cwd=ROOT,
                            env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake), "CAPTURED_ARGS": str(capture)},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    sent = capture.read_text().splitlines()
    assert sent[sent.index("-b") + 1] == "2024-10-25"
    assert sent[sent.index("-e") + 1] == "2026-09-16"
    assert "-m" not in sent and "--as-of" not in sent
    assert ("--replay-mail" in sent) is mail
    assert sent[sent.index("--pae-holdings-file") + 1] == str(tmp_path / "historical.txt")


@pytest.mark.parametrize("bounds", [["-b", "0", "-e", "2026-09-16"],
                                    ["-b", "2024-10-25", "-e", "792"],
                                    ["-b", "2024-10-25", "-e", "2026-09-16"]])
def test_shell_date_range_rejects_mixed_types_and_missing_snapshot(bounds):
    result = subprocess.run(["./stockdataanalysis.sh", *bounds, "--check"], cwd=ROOT,
                            env={"PATH": "/usr/bin:/bin"}, text=True, capture_output=True)
    assert result.returncode == 2


@pytest.mark.parametrize("value", ["2026-09-16", "2026-09-16T09:00:00+08:00", "2026-09-16T13:30:00+08:00",
                                 "2026-09-16T05:30:00Z", "2026-09-16T15:00:00+08:00", ""])
def test_shell_rejects_removed_as_of_before_python_or_current_portfolio_fallback(tmp_path, value):
    marker = tmp_path / "python_called"
    fake = tmp_path / "python"
    fake.write_text('#!/bin/sh\nprintf called > "$MARKER"\n')
    fake.chmod(0o755)
    result = subprocess.run(["./stockdataanalysis.sh", "--as-of", value, "--check"], cwd=ROOT,
                            env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake), "MARKER": str(marker)},
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "Unknown argument: --as-of" in result.stderr
    assert not marker.exists()


@pytest.mark.usefixtures("active_industry_profiles")
def test_date_cutoff_propagates_to_les_erem_pae_and_execution_isolation(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path, end="2026-09-11"))
    market = daily_market()
    evidence = dict(market.trading_restrictions["2330"], available_at=req.as_of.isoformat())
    market = replace(market, replay_as_of=req.as_of.isoformat(),
                     signal_dates={"2330": req.as_of.isoformat()}, trading_restrictions={"2330": evidence})
    _, decisions = build(market)
    persistence = Mock()
    crawler = SimpleNamespace(cal_max_profit=Mock(return_value=[]), build_strategy_decisions=Mock(return_value=decisions))
    context = AnalysisContext(crawler, req.as_of.date(), MarketStatus.OPEN, req.as_of)
    evaluation = CrawlerStrategyAdapter(portfolio_allocator=PortfolioAllocationV2Pipeline(persistence=persistence),
                                        portfolio_current_cash=900000).evaluate(context)
    decision = evaluation.decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    assert decision.value("EREM 分數") is not None and decision.value("PAE Status") is not None
    assert decision.evaluated_at == req.as_of
    assert report["replay_as_of"] == req.as_of.isoformat()
    assert report_failure(report, allow_replay=True) is None
    assert assess_execution(report, {}, now=req.as_of, trading_sessions=[], portfolio={})["reason"] == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
    persistence.load_recent_v2.assert_not_called()
    persistence.record_session_v2.assert_not_called()


@pytest.mark.usefixtures("active_industry_profiles")
def test_date_derived_cutoff_matches_explicit_internal_cutoff_decisions(tmp_path):
    req, _ = build_run_configuration(date_args(tmp_path, end="2026-09-11"))
    # Same timestamp the removed option used to produce, independent of the
    # new parser. Original source timestamps and all other inputs stay fixed.
    baseline_cutoff = dt.datetime.fromisoformat("2026-09-11T13:30:00+08:00")
    market = daily_market()
    evidence = dict(market.trading_restrictions["2330"], available_at=baseline_cutoff.isoformat())
    market = replace(market, trading_restrictions={"2330": evidence})

    def evaluate(cutoff):
        builder, decisions = build(replace(market, signal_dates={"2330": cutoff.isoformat()},
                                           replay_as_of=cutoff.isoformat()))
        persistence = Mock()
        crawler = SimpleNamespace(cal_max_profit=Mock(return_value=[]), build_strategy_decisions=Mock(return_value=decisions))
        context = AnalysisContext(crawler, cutoff.date(), MarketStatus.OPEN, cutoff)
        evaluation = CrawlerStrategyAdapter(portfolio_allocator=PortfolioAllocationV2Pipeline(persistence=persistence),
                                            portfolio_current_cash=900000).evaluate(context)
        persistence.load_recent_v2.assert_not_called()
        persistence.record_session_v2.assert_not_called()
        decision = evaluation.decisions[0]
        assert decision.value("低點決策") == "BUY"
        assert decision.value("EREM 分數") is not None
        assert decision.value("PAE Status") is not None
        return builder.project_decisions(evaluation.decisions)

    pd.testing.assert_frame_equal(evaluate(req.as_of), evaluate(baseline_cutoff))


def test_domain_date_bounds_are_inclusive_and_require_both_endpoints():
    day = dt.date(2026, 9, 16)
    assert build_backtrack_dates(day, 0, 1, include_reference=True, start_date=day, end_date=day) == [day]
    assert build_backtrack_dates(day, 0, 1, start_date=day, end_date=day) == []
    with pytest.raises(ValueError):
        build_backtrack_dates(day, 0, 1, start_date=day)
