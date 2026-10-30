"""Synthetic display checks: raw FAIL is never called an overall PASS."""
from infrastructure.report.low_entry_v7_renderer import valuation_note


def test_bear_warning_names_small_only_not_overall_valuation_pass():
    value = dict(band="PASS", state="ALLOWED_WITH_WARNING", native_state="FAIL",
                 bear_warning=True, points=0., multiplier=.5, small_only=True)
    rendered = valuation_note(value)
    assert "價格條件通過" in rendered and "情境警示：僅 BUY_SMALL" in rendered
    assert "部位倍率 ×0.5" in rendered and "估值 PASS" not in rendered
    assert value["native_state"] == "FAIL"


def test_price_and_bear_warnings_keep_penalty_and_combined_multiplier_visible():
    rendered = valuation_note(dict(band="MILD", state="ALLOWED_WITH_WARNING",
        bear_warning=True, points=5., multiplier=.375))
    assert "價格 MILD · 扣分 5.00" in rendered
    assert "僅 BUY_SMALL" in rendered and "倍率 ×0.375" in rendered


def test_previous_non_bear_warning_presentation_remains_exact():
    value = dict(band="MILD", state="ALLOWED_WITH_WARNING", points=5., multiplier=.75)
    assert valuation_note(value) == "估值 MILD · 扣分 5.00<br>部位倍率 ×0.75"
