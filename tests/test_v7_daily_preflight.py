"""SYNTHETIC configuration checks, not real model, evidence or entry readiness."""
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from application.les_v7_preflight import daily_v7_preflight
from config.settings import AppSettings, SettingsError
from domain.les_v7.contracts import identity
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.quality_policy import default_quality_policy
from domain.les_v7.strategy_policy import StrategyPolicy
from interface import cli
from tests.strategies.test_low_entry_score_v7 import bundle, seal


CHECKED_AT = "2026-10-13T18:00:00+08:00"


def model_record():
    """Complete synthetic artifact: exercise real validation, no archive stub."""
    record = bundle()
    record["payload"]["quality_policy"] = default_quality_policy()
    return seal(record)


def standalone_policy(record=None):
    model = FrozenBundle.load(record or model_record())
    policy = StrategyPolicy.from_bundle(model).export()
    del policy["payload"]["source_bundle_sha256"]
    policy["payload"]["policy_version"] = "SYNTHETIC_STANDALONE_ONLY"
    policy["sha256"] = identity(policy["payload"])
    return policy


def settings(**changes):
    return replace(AppSettings(daily_entry_model="v7", les_v7_native_evidence=True), **changes)


def preflight(configuration):
    return daily_v7_preflight(configuration, checked_at=CHECKED_AT)


def test_full_daily_requires_model_instead_of_silently_running_all_insufficient():
    result = preflight(settings())
    assert result["state"] == "BLOCKED" and result["mode"] == "FULL_V7"
    assert result["reason_codes"] == ["V7_DAILY_MODEL_REQUIRED_USE_EXPLICIT_RESEARCH_OR_DIAGNOSTIC_MODE"]
    assert not result["formal_entry_enabled"] and not result["order_authority"]


def test_diagnostics_requires_explicit_fixed_policy_without_inventing_defaults():
    result = preflight(settings(les_v7_diagnostics_only=True))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_STRATEGY_POLICY_REQUIRED"]


def test_explicit_standalone_policy_enables_diagnostics_not_formal_entry():
    record = standalone_policy()
    result = preflight(settings(les_v7_diagnostics_only=True,
        les_v7_strategy_policy_json=json.dumps(record)))
    assert result["state"] == "READY" and result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["strategy_policy_sha256"] == record["sha256"]
    assert result["model_sha256"] is None
    assert not result["formal_entry_enabled"] and not result["order_authority"]
    assert result["cutoff_policy"] == "AFTER_COLLECTION_CURRENT_DAILY_ONLY"


def test_complete_synthetic_model_retains_hash_and_native_evidence_requirement():
    record = model_record()
    result = preflight(settings(les_v7_bundle_json=json.dumps(record)))
    assert result["state"] == "READY" and result["formal_entry_enabled"]
    assert result["model_sha256"] == record["sha256"]
    assert result["strategy_policy_sha256"] == StrategyPolicy.from_bundle(FrozenBundle.load(record)).sha256
    assert not result["order_authority"]


def test_preflight_exposes_active_fractal_warmup_without_claiming_data_completeness():
    from tests.strategies.test_v7_fractal_requirements import _delta_bundle
    record = _delta_bundle(validated=True)
    record["payload"]["quality_policy"] = default_quality_policy()
    result = preflight(settings(les_v7_bundle_json=json.dumps(seal(record))))
    assert result["state"] == "READY"
    assert result["warmup"]["required_valid_close_count"] == 518
    assert result["warmup"]["active_feature_requirements"] == {"delta_dfa512_5": 518}
    assert result["warmup"]["policy"] == "CALENDAR_WINDOW_IS_NOT_PROOF_OF_VALID_CLOSES"
    assert "SUFFICIENT_HISTORY" not in result["warmup"].values()


def test_supplied_model_does_not_turn_diagnostics_into_formal_entry():
    result = preflight(settings(les_v7_diagnostics_only=True,
        les_v7_bundle_json=json.dumps(model_record())))
    assert result["state"] == "READY" and not result["formal_entry_enabled"]


def test_disabled_native_receipts_cannot_be_declared_ready():
    result = preflight(settings(les_v7_bundle_json=json.dumps(model_record()), les_v7_native_evidence=False))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_DAILY_NATIVE_EVIDENCE_REQUIRED"]


def test_legacy_quality_model_cannot_enter_cyclic_native_receipt_path():
    result = preflight(settings(les_v7_bundle_json=json.dumps(bundle())))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_NATIVE_REPORTED_QUALITY_POLICY_REQUIRED"]


def test_matching_model_and_policy_do_not_bypass_registered_quality_validation():
    record = model_record()
    record["payload"]["quality_policy"]["freshness_max_age_days"] = 9999
    record = seal(record)
    policy = standalone_policy(record)
    result = preflight(settings(les_v7_bundle_json=json.dumps(record),
        les_v7_strategy_policy_json=json.dumps(policy)))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["QUALITY_POLICY_HASH_OR_MAPPING_MISMATCH:freshness_max_age_days"]


def test_future_model_publication_is_not_backdated_to_current_daily_clock():
    record = model_record()
    record["payload"]["trained_until"] = "2027-01-01T00:00:00Z"
    result = preflight(settings(les_v7_bundle_json=json.dumps(seal(record))))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["MODEL_NOT_AVAILABLE_AT_PREFLIGHT"]


def test_future_standalone_policy_publication_is_rejected():
    record = standalone_policy()
    record["payload"]["available_at"] = "2027-01-01T00:00:00Z"
    record["sha256"] = identity(record["payload"])
    result = preflight(settings(les_v7_diagnostics_only=True, les_v7_strategy_policy_json=json.dumps(record)))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["STRATEGY_POLICY_NOT_AVAILABLE_AT_CUTOFF"]


@pytest.mark.parametrize("block", ["context_rules", "trigger_rules", "reference_policy", "quality_policy"])
def test_all_four_strategy_blocks_are_bound_to_fitted_model(block):
    record = model_record()
    policy = standalone_policy(record)
    if block == "context_rules":
        policy["payload"][block]["REVERSAL"]["value"] = 1.
    elif block == "trigger_rules":
        policy["payload"][block]["near_pullback_support"]["value"] = 1.
    elif block == "reference_policy":
        policy["payload"][block]["cost_up_per_share"] = .4
    else:
        policy["payload"][block]["minimum_quality_score"] = 99.
    policy["sha256"] = identity(policy["payload"])
    result = preflight(settings(les_v7_bundle_json=json.dumps(record),
        les_v7_strategy_policy_json=json.dumps(policy)))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["STRATEGY_POLICY_MODEL_BINDING_MISMATCH"]


def test_matching_explicit_policy_and_model_remain_ready_without_mutating_records():
    record, policy = model_record(), standalone_policy()
    before = deepcopy((record, policy))
    result = preflight(settings(les_v7_bundle_json=json.dumps(record),
        les_v7_strategy_policy_json=json.dumps(policy)))
    assert result["state"] == "READY"
    assert (record, policy) == before


def test_legacy_daily_entry_does_not_gain_new_v7_configuration_dependency():
    result = preflight(AppSettings(daily_entry_model="v6"))
    assert result == dict(state="NOT_APPLICABLE", mode="LEGACY", reason_codes=[])


def isolated_env(tmp_path):
    return dict(TWSE_DAILY_ENTRY_MODEL="v7",
        TWSE_MARKET_HISTORY_CACHE_PATH=str(tmp_path / "market.sqlite3"),
        TWSE_RUN_LOCK_PATH=str(tmp_path / "run.lock"), MOPS_CACHE_DIR=str(tmp_path / "mops"),
        TWSE_PAE_PERSISTENCE_PATH=str(tmp_path / "pae.sqlite3"))


def test_settings_load_standalone_policy_via_cli_path_and_environment(tmp_path):
    record = standalone_policy()
    path = tmp_path / "SYNTHETIC_POLICY.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    configured = AppSettings.from_env(isolated_env(tmp_path), validate_paths=False,
        les_v7_strategy_policy=path, les_v7_diagnostics_only=True)
    assert json.loads(configured.les_v7_strategy_policy_json) == record
    assert configured.les_v7_diagnostics_only and configured.les_v7_native_evidence
    environment = {**isolated_env(tmp_path), "TWSE_LES_V7_STRATEGY_POLICY": str(path),
        "TWSE_LES_V7_DIAGNOSTICS_ONLY": "true"}
    assert AppSettings.from_env(environment, validate_paths=False) == configured


def test_settings_reject_policy_hash_tampering(tmp_path):
    record = standalone_policy()
    record["payload"]["policy_version"] = "TAMPERED_SYNTHETIC_ONLY"
    path = tmp_path / "SYNTHETIC_TAMPER.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SettingsError, match="STRATEGY_POLICY_HASH_MISMATCH"):
        AppSettings.from_env(isolated_env(tmp_path), validate_paths=False, les_v7_strategy_policy=path)


def test_settings_reject_unbounded_policy_file(tmp_path):
    path = tmp_path / "SYNTHETIC_OVERSIZED.json"
    path.write_text(" " * (256 * 1024 + 1), encoding="utf-8")
    with pytest.raises(SettingsError, match="BOUNDED_STRATEGY_POLICY_REQUIRED"):
        AppSettings.from_env(isolated_env(tmp_path), validate_paths=False, les_v7_strategy_policy=path)


def configure_cli_without_inherited_credentials(monkeypatch, tmp_path, extra_environment=None):
    environment = {**isolated_env(tmp_path), **(extra_environment or {})}
    real_from_env = AppSettings.from_env
    monkeypatch.setattr(AppSettings, "from_env", lambda **kwargs: real_from_env(
        environ=environment, **{key: value for key, value in kwargs.items() if key != "environ"}))
    service = Mock(side_effect=AssertionError("configuration check started analysis or notification"))
    lock = Mock(side_effect=AssertionError("blocked readiness acquired a runtime lock"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    monkeypatch.setattr(cli, "ProcessRunLock", lock)
    return service, lock


def test_cli_blocked_check_reports_exit_two_without_analysis_email_or_lock(monkeypatch, tmp_path, capsys):
    service, lock = configure_cli_without_inherited_credentials(monkeypatch, tmp_path,
        {"TWSE_LES_V7_RULE_BASELINE": "false"})
    assert cli.main(["0", "--check-config", "--skip-mops"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "CONFIG_NOT_READY"
    assert result["les_v7_readiness"]["state"] == "BLOCKED"
    assert not result["mail_enabled"] and not result["smtp_configured"]
    service.assert_not_called(); lock.assert_not_called()
    assert not (tmp_path / "run.lock").exists()


def test_cli_missing_model_fails_before_analysis_network_or_notification(monkeypatch, tmp_path, capsys):
    service, lock = configure_cli_without_inherited_credentials(monkeypatch, tmp_path,
        {"TWSE_LES_V7_RULE_BASELINE": "false"})
    with pytest.raises(SystemExit) as error:
        cli.main(["0", "--skip-mops"])
    assert error.value.code == 2
    assert "LES_V7_CONFIGURATION_NOT_READY" in capsys.readouterr().err
    service.assert_not_called(); lock.assert_not_called()


def test_cli_explicit_diagnostics_check_uses_native_registered_policy(monkeypatch, tmp_path, capsys):
    path = tmp_path / "SYNTHETIC_POLICY.json"
    policy = standalone_policy()
    path.write_text(json.dumps(policy), encoding="utf-8")
    service, lock = configure_cli_without_inherited_credentials(monkeypatch, tmp_path)
    assert cli.main(["0", "--check-config", "--skip-mops", "--les-v7-diagnostics-only",
        "--les-v7-strategy-policy", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "OK"
    assert result["les_v7_readiness"]["mode"] == "DIAGNOSTIC_ONLY"
    assert not result["les_v7_readiness"]["formal_entry_enabled"]
    assert result["les_v7_quality_policy"]["profile_id"] == policy["payload"]["quality_policy"]["profile_id"]
    assert result["les_v7_readiness"]["strategy_policy_sha256"] == policy["sha256"]
    service.assert_not_called(); lock.assert_not_called()
