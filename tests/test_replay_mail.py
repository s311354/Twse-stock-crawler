"""Opt-in historical email; all transports are mocked, no real mail is sent."""
import datetime as dt
import json
from dataclasses import replace
from email import message_from_string
from email.header import decode_header, make_header
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import interface.cli as cli
from application.contracts import AnalysisContext, MarketStatus, StrategyEvaluation
from application.use_cases import SendNotification
from infrastructure.runtime.output_adapters import CrawlerNotificationAdapter
from tests.test_historical_replay import close_replay_market, replay_args, request
from tests.strategies.test_daily_v6_pipeline import build


@pytest.fixture
def smtp_env(monkeypatch):
    for name, value in {"EMAIL": "sender@example.invalid", "PASSWORD": "test-only",
                        "RECEIVERS": "recipient@example.invalid", "HOST": "smtp.example.invalid"}.items():
        monkeypatch.setenv(f"TWSE_SMTP_{name}", value)


def test_cli_explicit_replay_mail_enables_existing_smtp(tmp_path, smtp_env):
    req, _ = cli.build_run_configuration(replay_args(tmp_path) + ["--replay-mail"])
    assert req.mail and req.replay_mail and req.notification_enabled
    assert req.report_file_prefix.endswith("_replay_20260916T133000+0800")
    normal, _ = cli.build_run_configuration(["2330", "-m"])
    assert normal.mail and not normal.replay_mail and normal.notification_enabled


@pytest.mark.parametrize("args", [["2330", "--replay-mail"], ["2330", "--replay-mail", "-m"]])
def test_replay_mail_without_cutoff_rejected(args):
    with pytest.raises(SystemExit):
        cli.build_run_configuration(args)


def test_replay_mail_requires_smtp_and_plain_mail_still_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TWSE_SMTP_PASSWORD", "")
    with pytest.raises(SystemExit):
        cli.build_run_configuration(replay_args(tmp_path) + ["--replay-mail"])
    with pytest.raises(SystemExit):
        cli.build_run_configuration(replay_args(tmp_path) + ["-m"])


def test_check_config_validates_mail_but_never_runs_service(tmp_path, smtp_env, monkeypatch, capsys):
    service = Mock(side_effect=AssertionError("analysis or email must not run"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    assert cli.main(replay_args(tmp_path) + ["--replay-mail", "--check-config"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mail_enabled"] is True and summary["replay_mail_enabled"] is True
    service.assert_not_called()


@pytest.mark.parametrize("mail,enabled,expected", [(False, False, 0), (True, False, 0), (True, True, 1)])
def test_notification_dispatch_requires_explicit_consent(mail, enabled, expected):
    req = request(mail=mail, replay_mail=enabled)
    port = Mock()
    SendNotification(port).execute(None, req, None)
    assert port.send.call_count == expected


@pytest.mark.parametrize("failure", ["consent", "context", "mode", "cutoff", "decision_time", "empty"])
def test_adapter_rejects_unauthorized_or_mixed_reports_before_smtp(failure):
    req = request(mail=True, replay_mail=failure != "consent")
    crawler = SimpleNamespace(smtp_strategy_email=Mock())
    context = AnalysisContext(crawler, req.as_of.date(), MarketStatus.OPEN,
                              None if failure == "context" else req.as_of)
    report = dict(evaluation_mode="HISTORICAL_REPLAY", replay_as_of=req.as_of.isoformat(),
                  decision_time=req.as_of.isoformat())
    if failure == "mode":
        report["evaluation_mode"] = "LIVE"
    if failure == "cutoff":
        report["replay_as_of"] = "2026-09-23T13:30:00+08:00"
    if failure == "decision_time":
        report["decision_time"] = "2026-09-23T13:30:00+08:00"
    decisions = () if failure == "empty" else (SimpleNamespace(value=lambda key: json.dumps(report)),)
    with pytest.raises(ValueError, match="REPLAY_MAIL"):
        CrawlerNotificationAdapter().send(context, req, StrategyEvaluation([], decisions))
    crawler.smtp_strategy_email.assert_not_called()


def test_real_mime_contains_replay_subject_cutoff_and_body_notice(monkeypatch):
    from textmewhenitsdone import TextMeWhenItsDone
    from tests.test_stockanalysis import CapturingTwseCrawler
    from domain.strategies.low_entry_v6_report import report_failure
    market = close_replay_market()
    cutoff = market.signal_dates["2330"]
    _, decisions = build(replace(market, replay_as_of=cutoff))
    req = replace(request(mail=True, replay_mail=True), as_of=dt.datetime.fromisoformat(cutoff))
    transport = object.__new__(TextMeWhenItsDone)
    transport.server = Mock()

    class Mail:
        def __init__(self, subject, *args, **kwargs):
            self.subject = subject
        def smtpauthentication(self):
            pass
        def textstockprofittable(self, **kwargs):
            transport.textstockprofittableme(subject=self.subject, email="sender@example.invalid",
                                             password="unused", receiver="recipient@example.invalid", **kwargs)

    monkeypatch.setattr("infrastructure.legacy_crawler_facade.SMTPEmail", Mail)
    crawler = CapturingTwseCrawler(1)
    crawler.iso_scheduled_times = ["2026-09-11"]
    context = AnalysisContext(crawler, req.as_of.date(), MarketStatus.OPEN, req.as_of)
    CrawlerNotificationAdapter().send(context, req, StrategyEvaluation([], decisions))
    transport.server.sendmail.assert_called_once()
    message = message_from_string(transport.server.sendmail.call_args.kwargs["msg"])
    subject = str(make_header(decode_header(message["Subject"])))
    assert "HISTORICAL_REPLAY" in subject and "不可下單" in subject and cutoff in subject
    body = message.get_payload()[0].get_payload(decode=True).decode("utf-8")
    assert "歷史重播／不可下單" in body and cutoff in body
    assert "不是今日可執行訊號" in body
    assert report_failure(json.loads(decisions[0].value("LES v6 Report"))) == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
