"""Synthetic projection parity; no fitted model or real research claim."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from tests.strategies.test_daily_v7_rule_baseline import rule_market
from tests.strategies.test_daily_v7_pipeline import synthetic_market


@pytest.fixture(scope="module", params=["baseline", "model"])
def report(request):
    market = rule_market() if request.param == "baseline" else synthetic_market()
    return daily_report(MarketHistoryAssembler().build(market, 0))


@pytest.mark.parametrize("price_scale", [1., 1000.])
def test_field_projection_is_identical_and_preserves_report(report, price_scale):
    before = deepcopy(report)
    complete = project_report(report, price_scale=price_scale)
    fields = project_report(report, price_scale=price_scale, include_report=False)
    assert json.loads(complete.pop("LES v7 Report")) == before
    assert fields == complete
    assert list(fields) == list(complete)
    assert report == before


def test_fields_do_not_serialize_report_but_still_validate_every_call(report, monkeypatch):
    import application.low_entry_v7_daily as module
    def forbidden(*args, **kwargs):
        raise AssertionError("full JSON cell must not be serialized")
    monkeypatch.setattr(module, "json", SimpleNamespace(dumps=forbidden))
    assert "LES v7 Report" not in project_report(report, include_report=False)
    tampered = deepcopy(report)
    tampered["envelope"]["event"]["security_id"] = "OTHER_SYNTHETIC_ONLY"
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        project_report(tampered, include_report=False)


@pytest.mark.parametrize("include_report", [None, 0, "false"])
def test_serialization_option_cannot_be_truthy_or_implicit(report, include_report):
    with pytest.raises(ValueError, match="V7_PROJECTION_INCLUDE_REPORT_INVALID"):
        project_report(report, include_report=include_report)
