"""Explicit ACTIVE policy doubles for pre-existing entry/execution parity tests.

Not autouse: production policy tests must exercise the real registry.
This changes no quality metric, technical score, risk limit, or golden fixture.
"""
from dataclasses import replace

import pytest

from domain.industry_profiles import IndustryProfileMode, IndustryProfileRegistry


@pytest.fixture
def active_industry_profiles(monkeypatch):
    resolve = IndustryProfileRegistry.resolve
    get = IndustryProfileRegistry.get
    active = {
        profile.name: replace(profile, mode=IndustryProfileMode.ACTIVE,
                              allow_buy=True, allow_buy_small=True,
                              calibration_status="TEST_ONLY")
        for profile in IndustryProfileRegistry.profiles()
    }

    def active_profile(profile):
        return active.get(profile.name, profile) if profile is not None else None

    monkeypatch.setattr(IndustryProfileRegistry, "resolve", classmethod(
        lambda cls, value: active_profile(resolve(value))
    ))
    monkeypatch.setattr(IndustryProfileRegistry, "get", classmethod(
        lambda cls, value: active_profile(get(value))
    ))
