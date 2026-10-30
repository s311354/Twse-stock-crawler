"""SYNTHETIC composition checks; build the real runtime but never run it.

Fake source/calendar ports prohibit fetching. Real CLI identity, study manifest,
portfolio adapters and PAE constructor wiring operate only below tmp_path.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from infrastructure import bootstrap
from infrastructure.research_series import ensure_study, metadata_identity, ResearchSeriesError
from infrastructure.storage.portfolio_persistence import SQLitePortfolioPersistence
from infrastructure.storage.research_portfolio import SQLiteResearchPortfolioPersistence
from interface import cli
from tests.test_v7_research_id_cli import named_research, metadata


def build_runtime(monkeypatch, settings):
    """The public composition root creates adapters, without invoking ports."""
    real_pipeline = bootstrap.PortfolioAllocationV2Pipeline
    pipeline = Mock(side_effect=real_pipeline)
    crawler = Mock(side_effect=AssertionError("SYNTHETIC bootstrap executed crawler"))
    http = SimpleNamespace(request=Mock(side_effect=AssertionError("SYNTHETIC bootstrap used HTTP")))
    monkeypatch.setattr(bootstrap, "PortfolioAllocationV2Pipeline", pipeline)
    # No unrelated restriction database is needed to test this dependency edge.
    monkeypatch.setattr(bootstrap, "TradingRestrictionRepository", lambda _: SimpleNamespace())
    runtime = bootstrap.build_stock_analysis_runtime(
        settings=settings, crawler_factory=crawler, market_calendar=SimpleNamespace(),
        fundamental_service=SimpleNamespace(), http_client=http,
        run_id_provider=lambda: "SYNTHETIC_BOOTSTRAP_NO_EXECUTION")
    assert runtime.settings is settings and runtime.service is not None
    crawler.assert_not_called()
    http.request.assert_not_called()
    pipeline.assert_called_once()
    return pipeline.call_args.kwargs


def configured_named(named_research):
    args, versions, live = named_research
    request, settings = cli.build_run_configuration([*args, "--check-config"])
    return request, settings, versions, live


def test_fixed_research_bootstrap_uses_real_identity_bound_adapter_and_portfolio_id(
        named_research, monkeypatch):
    _, settings, versions, live = configured_named(named_research)
    info = metadata(settings)
    identity = metadata_identity(info)
    root = settings.pae_persistence_path.parent
    ensure_study(root, identity)
    before = live.read_bytes(), (root / "study.json").read_bytes()
    options = build_runtime(monkeypatch, settings)
    adapter = options["persistence"]
    assert isinstance(adapter, SQLiteResearchPortfolioPersistence)
    assert not isinstance(adapter, SQLitePortfolioPersistence)
    assert adapter.database_path == settings.pae_persistence_path
    assert options["portfolio_id"] == "research:" + info["research_id"]
    assert json.loads(adapter.identity_json) == identity
    assert not settings.pae_persistence_path.exists()  # Building is not committing a session.
    adapter.check_version()
    versions["current"] = replace(versions["current"], code="f" * 64)
    with pytest.raises(ResearchSeriesError, match="VERSIONS_CHANGED"):
        adapter.check_version()
    assert (live.read_bytes(), (root / "study.json").read_bytes()) == before


@pytest.mark.parametrize("mode", ["ordinary", "anonymous_research"])
def test_bootstrap_retains_ordinary_adapter_for_non_series_paths(named_research, monkeypatch, tmp_path, mode):
    args, _, live = named_research
    if mode == "ordinary":
        _, settings = cli.build_run_configuration(["--check-config", "--skip-mops", "1"])
        settings = replace(settings, pae_persistence_path=tmp_path / "SYNTHETIC_ordinary.sqlite3")
        assert settings.les_v7_research_metadata_json is None
    else:
        args = list(args)
        offset = args.index("--research-id")
        del args[offset:offset + 2]
        _, settings = cli.build_run_configuration([*args, "--check-config"])
        assert metadata(settings).get("research_id") is None
    before = live.read_bytes()
    options = build_runtime(monkeypatch, settings)
    assert isinstance(options["persistence"], SQLitePortfolioPersistence)
    assert not isinstance(options["persistence"], SQLiteResearchPortfolioPersistence)
    assert options["persistence"].database_path == settings.pae_persistence_path
    from domain.strategies.erem_soft_reduction import FORMAL_VERSION
    assert options["portfolio_id"] == "default:" + FORMAL_VERSION
    assert live.read_bytes() == before


def test_fixed_research_bootstrap_refuses_missing_study_instead_of_falling_back(
        named_research, monkeypatch):
    _, settings, _, live = configured_named(named_research)
    before = live.read_bytes()
    pipeline = Mock(side_effect=AssertionError("missing study reached allocation"))
    monkeypatch.setattr(bootstrap, "PortfolioAllocationV2Pipeline", pipeline)
    monkeypatch.setattr(bootstrap, "TradingRestrictionRepository", lambda _: SimpleNamespace())
    with pytest.raises(ResearchSeriesError, match="STUDY_NOT_INITIALIZED"):
        bootstrap.build_stock_analysis_service(settings=settings,
            crawler_factory=Mock(side_effect=AssertionError("crawler used")),
            market_calendar=SimpleNamespace(), fundamental_service=SimpleNamespace(), http_client=SimpleNamespace())
    pipeline.assert_not_called()
    assert not settings.pae_persistence_path.exists()
    assert not (settings.pae_persistence_path.parent / "study.json").exists()
    assert live.read_bytes() == before


@pytest.mark.parametrize("damage", ["identity_hash", "research_id", "bundle_sha256", "dataset_id",
    "config_id", "native_job_id", "symbols", "symbol_order", "admission_component",
    "execution_component", "valuation_policy", "decision_thresholds"])
def test_fixed_bootstrap_rejects_metadata_identity_mismatch_before_allocation(
        named_research, monkeypatch, damage):
    _, settings, _, live = configured_named(named_research)
    info = metadata(settings)
    root = settings.pae_persistence_path.parent
    ensure_study(root, metadata_identity(info))
    before = live.read_bytes(), (root / "study.json").read_bytes()
    changed = deepcopy(info)
    if damage == "identity_hash":
        changed["research_identity_id"] = "f" * 64
    elif damage == "research_id":
        changed["research_id"] = "SYNTHETIC_OTHER_STUDY"
    elif damage == "symbols":
        changed["symbols"].append("2330")
    elif damage == "symbol_order":
        changed["symbols"].reverse()
    elif damage in {"admission_component", "execution_component", "valuation_policy", "decision_thresholds"}:
        changed[damage] = dict(changed[damage], SYNTHETIC_TAMPER=True)
    else:
        changed[damage] = "f" * 64
    settings = replace(settings, les_v7_research_metadata_json=json.dumps(changed))
    pipeline = Mock(side_effect=AssertionError("metadata mismatch reached allocation"))
    monkeypatch.setattr(bootstrap, "PortfolioAllocationV2Pipeline", pipeline)
    monkeypatch.setattr(bootstrap, "TradingRestrictionRepository", lambda _: SimpleNamespace())
    with pytest.raises(ResearchSeriesError, match="METADATA_IDENTITY_MISMATCH"):
        bootstrap.build_stock_analysis_service(settings=settings,
            crawler_factory=Mock(side_effect=AssertionError("crawler used")),
            market_calendar=SimpleNamespace(), fundamental_service=SimpleNamespace(), http_client=SimpleNamespace())
    pipeline.assert_not_called()
    assert not settings.pae_persistence_path.exists()
    assert (live.read_bytes(), (root / "study.json").read_bytes()) == before


def test_fixed_bootstrap_rejects_versions_changed_after_preflight(named_research, monkeypatch):
    _, settings, versions, live = configured_named(named_research)
    root = settings.pae_persistence_path.parent
    ensure_study(root, metadata_identity(metadata(settings)))
    before = live.read_bytes(), (root / "study.json").read_bytes()
    versions["current"] = replace(versions["current"], parameters="f" * 64)
    pipeline = Mock(side_effect=AssertionError("changed version reached allocation"))
    monkeypatch.setattr(bootstrap, "PortfolioAllocationV2Pipeline", pipeline)
    monkeypatch.setattr(bootstrap, "TradingRestrictionRepository", lambda _: SimpleNamespace())
    with pytest.raises(ResearchSeriesError, match="VERSIONS_CHANGED"):
        bootstrap.build_stock_analysis_service(settings=settings,
            crawler_factory=Mock(side_effect=AssertionError("crawler used")),
            market_calendar=SimpleNamespace(), fundamental_service=SimpleNamespace(), http_client=SimpleNamespace())
    pipeline.assert_not_called()
    assert not settings.pae_persistence_path.exists()
    assert (live.read_bytes(), (root / "study.json").read_bytes()) == before
