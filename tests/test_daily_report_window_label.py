"""Report scope is a calendar lookback, never a fabricated daily backtest.

Date-only fixtures and a mocked mail transport; no official data or real mail.
"""
from email import message_from_string
from unittest.mock import Mock

import pytest

from stockanalysis import TwseCrawker
from textmewhenitsdone import StockhtmlMessage, TextMeWhenItsDone


@pytest.mark.parametrize("start,end,expected", [
    ("2024-11-15", "2026-10-15", 700),
    ("2026-10-15", "2024-11-15", 700),
    ("2024-11-14", "2026-10-15", 701),
    ("2024-01-31", "2024-02-01", 2),
    ("2024-02-28", "2024-03-01", 3),
    ("2025-02-28", "2025-03-01", 2),
    ("2024-12-31", "2025-01-01", 2),
    ("2000-02-28", "2000-03-01", 3),
    ("2100-02-28", "2100-03-01", 2),
    ("2026-10-15", "2026-10-15", 1),
])
def test_calendar_span_uses_real_dates_including_leap_boundaries(start, end, expected):
    assert TwseCrawker.days_between_isodates(None, start, end) == expected


@pytest.mark.parametrize("invalid", ["2025-02-29", "2024-02-30", "2026-13-01"])
def test_invalid_calendar_dates_are_not_silently_counted(invalid):
    with pytest.raises(ValueError):
        TwseCrawker.days_between_isodates(None, invalid, "2026-10-15")


def test_email_template_distinguishes_input_window_from_backtest():
    body = StockhtmlMessage.format(days=700, begin="2024-11-15", end="2026-10-15",
                                   entryanalysis="<section>native signals</section>")
    assert "Input lookback: 700 calendar days" in body
    assert "Single-cutoff signals" in body
    assert "not a day-by-day historical backtest" in body
    assert "trading days" not in body
    assert "<section>native signals</section>" in body


@pytest.mark.parametrize("version", ["5.0.0", "6.0.0", "7.0.0"])
def test_existing_mail_contract_renders_correct_scope_without_network(version):
    transport = TextMeWhenItsDone.__new__(TextMeWhenItsDone)
    transport.server = Mock()
    transport.textstockprofittableme(subject="test", email="sender@example.invalid",
        password="unused", iso_scheduled_times=["2024-11-15", "2026-10-15"],
        transactiondays=700, stocktype="ELEC", stockprofittable="",
        entryanalysis="<section>native signals</section>",
        receiver="recipient@example.invalid", les_model_version=version)
    transport.server.sendmail.assert_called_once()
    message = message_from_string(transport.server.sendmail.call_args.kwargs["msg"])
    body = message.get_payload()[0].get_payload(decode=True).decode("utf-8")
    assert f"LES v{version[0]} / EREM v3" in message["Subject"]
    assert "Input lookback: 700 calendar days (2024-11-15 ~ 2026-10-15)" in body
    assert "Single-cutoff signals; this is not a day-by-day historical backtest" in body
    assert "trading days" not in body
