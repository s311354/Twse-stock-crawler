"""Tests for the --list-strategies CLI feature."""

from interface.cli import build_request
from interface.cli import list_strategies
from interface.cli import main
from domain.strategy import available_strategies
from domain.strategy import strategy_registry


def test_list_strategies_displays_all_registered_strategies(capsys):
    """Verify list_strategies() outputs all strategies with metadata."""
    list_strategies()
    captured = capsys.readouterr()
    
    # Check header is present
    assert "Available Trading Strategies" in captured.out
    assert "Name" in captured.out
    assert "Version" in captured.out
    assert "Description" in captured.out
    
    # Check that key strategies are listed
    assert "low_entry_score_v5" in captured.out
    assert "expected_return_exit_v3" in captured.out
    assert "low_entry_score_v4_1" in captured.out
    assert "portfolio_allocation_v1" in captured.out


def test_list_strategies_shows_version_and_description(capsys):
    """Verify that version and description metadata are displayed."""
    list_strategies()
    captured = capsys.readouterr()
    
    # Check LES v5 metadata
    assert "5.0.0" in captured.out
    assert "Quality-first panic reversal" in captured.out
    
    # Check EREM v3 metadata
    assert "3.2.4" in captured.out
    assert "decaying Entry LES memory" in captured.out


def test_cli_list_strategies_flag_exits_without_requiring_stocklist(capsys):
    """Verify --list-strategies works without providing a stocklist."""
    exit_code = main(["--list-strategies"])
    captured = capsys.readouterr()
    
    assert exit_code == 0
    assert "Available Trading Strategies" in captured.out
    assert "low_entry_score_v5" in captured.out


def test_available_strategies_returns_all_builtin_names():
    """Verify available_strategies() loads and returns all builtin strategy names."""
    names = available_strategies()
    
    assert isinstance(names, list)
    assert len(names) == 11
    assert "low_entry_score_v5" in names
    assert "expected_return_exit_v3" in names
    assert "low_entry_score_v4_1" in names
    assert "portfolio_allocation_v1" in names
    assert "portfolio_allocation_v2" in names


def test_strategy_registry_all_method_returns_strategy_objects():
    """Verify registry.all() returns actual Strategy instances."""
    available_strategies()  # Ensure strategies are loaded
    strategies = strategy_registry.all()
    
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    
    # Check that each item is a Strategy with required attributes
    for strategy in strategies:
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "run")
        assert hasattr(strategy, "description")
        assert hasattr(strategy, "version")


def test_strategy_metadata_fields_have_correct_types():
    """Verify Strategy base class metadata fields are strings."""
    available_strategies()
    strategies = strategy_registry.all()
    
    for strategy in strategies:
        assert isinstance(strategy.name, str)
        assert isinstance(strategy.description, str)
        assert isinstance(strategy.version, str)
        # Name should never be empty
        assert len(strategy.name) > 0
