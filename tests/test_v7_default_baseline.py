"""SYNTHETIC default-rule wiring checks; not calibrated forecasts or live trades."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from application.les_v7_preflight import daily_v7_preflight
from config.settings import AppSettings, SettingsError
from domain.les_v7.contracts import identity
from interface import cli
from tests.test_v7_daily_preflight import (
    configure_cli_without_inherited_credentials, isolated_env, model_record, standalone_policy,
)


AFTER_PUBLICATION = "2026-10-14T15:30:00+08:00"
BEFORE_PUBLICATION = "2026-10-13T14:23:15Z"


def preflight(configuration, checked_at=AFTER_PUBLICATION):
    return daily_v7_preflight(configuration, checked_at=checked_at)


def configured(tmp_path, **values):
    return AppSettings.from_env({**isolated_env(tmp_path), **values}, validate_paths=False)


def test_default_environment_without_model_enables_explicitly_versioned_rule_baseline(tmp_path):
    settings = configured(tmp_path)
    assert settings.les_v7_rule_baseline and settings.les_v7_native_evidence
    assert settings.les_v7_bundle_json is None and settings.les_v7_strategy_policy_json is None
    result = preflight(settings)
    assert result["state"] == "READY" and result["mode"] == "RULE_BASED_BASELINE"
    assert result["model_sha256"] is None
    assert result["formal_entry_enabled"] and not result["order_authority"]
    assert result["model_status"] == "NOT_REQUIRED_RULE_BASED_BASELINE"
    assert result["prediction_status"] == result["calibration_status"] == "NOT_APPLICABLE"
    assert result["probability_defaults"] is None


def test_source_repair_default_is_new_version_and_explicit_opt_out_keeps_old_policy(tmp_path):
    current = preflight(configured(tmp_path))
    old = preflight(configured(tmp_path, TWSE_LES_V7_SOURCE_REPAIR="false"))
    assert current["rule_baseline_policy"]["policy_version"] == "V7_RULE_BASED_BASELINE_R6"
    assert current["rule_baseline_policy"]["quality_policy"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R5"
    assert old["rule_baseline_policy"]["policy_version"] == "V7_RULE_BASED_BASELINE_R5"
    assert old["rule_baseline_policy_id"] == "2401d1f1185716e7f7b5b456b310c73d8136ac73b8860963f3dd017755c4fd2e"
    assert preflight(configured(tmp_path), "2026-10-14T06:37:59Z")["state"] == "BLOCKED"


def test_manual_legacy_api_settings_do_not_silently_gain_baseline_semantics():
    settings = AppSettings(daily_entry_model="v7", les_v7_native_evidence=True)
    assert not settings.les_v7_rule_baseline
    result = preflight(settings)
    assert result["state"] == "BLOCKED" and result["mode"] == "FULL_V7"


@pytest.mark.parametrize("changes", [dict(les_v7_diagnostics_only=True),
    dict(les_v7_bundle_json=json.dumps(model_record())),
    dict(les_v7_strategy_policy_json=json.dumps(standalone_policy()))])
def test_manual_conflicting_baseline_flags_cannot_override_requested_mode(tmp_path, changes):
    result = preflight(replace(configured(tmp_path), **changes))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_RULE_BASELINE_CANNOT_MIX_MODEL_OR_CUSTOM_POLICY"]


def test_baseline_opt_out_keeps_missing_model_rejection(tmp_path):
    settings = configured(tmp_path, TWSE_LES_V7_RULE_BASELINE="false")
    assert not settings.les_v7_rule_baseline
    result = preflight(settings)
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_DAILY_MODEL_REQUIRED_USE_EXPLICIT_RESEARCH_OR_DIAGNOSTIC_MODE"]


def test_rule_baseline_publication_is_not_backdated(tmp_path):
    result = preflight(configured(tmp_path), checked_at=BEFORE_PUBLICATION)
    assert result["state"] == "BLOCKED"
    assert not result["formal_entry_enabled"] and result["reason_codes"]
    assert "AVAILABLE" in result["reason_codes"][0]


def test_baseline_still_requires_native_evidence_capture_configuration(tmp_path):
    result = preflight(replace(configured(tmp_path), les_v7_native_evidence=False))
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_DAILY_NATIVE_EVIDENCE_REQUIRED"]


def test_deployed_model_keeps_fitted_path_instead_of_default_rule_fallback(tmp_path):
    record = model_record()
    path = tmp_path / "SYNTHETIC_MODEL.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    settings = configured(tmp_path, TWSE_LES_V7_BUNDLE=str(path))
    assert settings.les_v7_bundle_json is not None and not settings.les_v7_rule_baseline
    result = preflight(settings)
    assert result["state"] == "READY" and result["mode"] == "FULL_V7"
    assert result["model_sha256"] == record["sha256"]


def test_explicit_standalone_policy_requires_its_own_mode_not_baseline_substitution(tmp_path):
    path = tmp_path / "SYNTHETIC_POLICY.json"
    path.write_text(json.dumps(standalone_policy()), encoding="utf-8")
    settings = configured(tmp_path, TWSE_LES_V7_STRATEGY_POLICY=str(path))
    assert settings.les_v7_strategy_policy_json is not None and not settings.les_v7_rule_baseline
    assert preflight(settings)["state"] == "BLOCKED"
    diagnostic = configured(tmp_path, TWSE_LES_V7_STRATEGY_POLICY=str(path), TWSE_LES_V7_DIAGNOSTICS_ONLY="true")
    assert not diagnostic.les_v7_rule_baseline
    result = preflight(diagnostic)
    assert result["state"] == "READY" and result["mode"] == "DIAGNOSTIC_ONLY"
    assert not result["formal_entry_enabled"]


def test_diagnostic_flag_without_policy_does_not_receive_default_baseline_policy(tmp_path):
    settings = configured(tmp_path, TWSE_LES_V7_DIAGNOSTICS_ONLY="true")
    assert not settings.les_v7_rule_baseline
    result = preflight(settings)
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["V7_STRATEGY_POLICY_REQUIRED"]


@pytest.mark.parametrize("record", [{}, {"payload": {}, "sha256": "0" * 64}])
def test_invalid_configured_model_is_rejected_not_replaced_by_rules(tmp_path, record):
    path = tmp_path / "SYNTHETIC_INVALID_MODEL.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SettingsError, match="TWSE_LES_V7_BUNDLE is invalid"):
        configured(tmp_path, TWSE_LES_V7_BUNDLE=str(path))


def test_missing_configured_model_file_is_rejected_not_replaced_by_rules(tmp_path):
    with pytest.raises(SettingsError, match="TWSE_LES_V7_BUNDLE is invalid"):
        configured(tmp_path, TWSE_LES_V7_BUNDLE=str(tmp_path / "MISSING_MODEL.json"))


def test_invalid_baseline_environment_switch_is_rejected(tmp_path):
    with pytest.raises(SettingsError, match="TWSE_LES_V7_RULE_BASELINE must be"):
        configured(tmp_path, TWSE_LES_V7_RULE_BASELINE="maybe")


def test_legacy_v6_does_not_use_new_rule_baseline(tmp_path):
    settings = configured(tmp_path, TWSE_DAILY_ENTRY_MODEL="v6")
    assert not settings.les_v7_rule_baseline
    assert preflight(settings)["state"] == "NOT_APPLICABLE"


def test_baseline_is_fixed_rule_configuration_not_fitted_or_calibrated_probabilities():
    from domain.les_v7.rule_baseline import baseline_policy, localization_record
    policy = baseline_policy("TWSE")
    assert policy["weights_kind"] == "FIXED_RULE_BUDGETS_NOT_FITTED"
    assert not policy["performance_validated"]
    assert policy["prediction_status"] == policy["calibration_status"] == "NOT_APPLICABLE"
    assert not {"heads", "calibration", "training_manifest"}.intersection(policy)
    assert sum(policy["model_spec"]["factor_budgets"].values()) == 100
    prediction = localization_record()
    assert prediction["status"] == "NOT_APPLICABLE"
    assert all(prediction[key] is None for key in ("quantiles", "timing", "p_fill", "p_rebound", "p_entry",
        "raw_probabilities", "calibration_version", "p_bottom_soon", "p_bottom_now"))


def test_baseline_market_binding_changes_identity_and_rejects_unknown_market():
    from domain.les_v7.rule_baseline import baseline_policy, require_baseline_policy
    twse, tpex = baseline_policy("TWSE"), baseline_policy("TPEX")
    assert identity(twse) != identity(tpex)
    assert tpex["reference_policy"]["cost_path_policy"]["market"] == "TPEX"
    with pytest.raises(ValueError, match="V7_BASELINE_POLICY_MISMATCH"):
        require_baseline_policy(twse, market="TPEX", cutoff=AFTER_PUBLICATION)
    with pytest.raises(ValueError, match="BASELINE_MARKET_REQUIRED"):
        baseline_policy("UNKNOWN")


def test_returned_baseline_policy_cannot_mutate_shared_configuration():
    from domain.les_v7.rule_baseline import baseline_policy
    record = baseline_policy("TWSE")
    original_id = identity(record)
    record["config"]["buy_fee_rate"] = 999.
    record["model_spec"]["factor_budgets"][next(iter(record["model_spec"]["factor_budgets"]))] = 999.
    record["quality_policy"]["freshness_max_age_days"] = 999.
    assert identity(baseline_policy("TWSE")) == original_id


def test_default_cli_check_ready_without_service_network_mail_or_lock(monkeypatch, tmp_path, capsys):
    service, lock = configure_cli_without_inherited_credentials(monkeypatch, tmp_path)
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(AFTER_PUBLICATION)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)
    monkeypatch.setattr(cli, "dt", SimpleNamespace(datetime=FixedDateTime, timezone=timezone, date=date, timedelta=timedelta))
    assert cli.main(["0", "--check-config", "--skip-mops"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "OK" and summary["les_v7_readiness"]["state"] == "READY"
    assert summary["les_v7_readiness"]["mode"] == "RULE_BASED_BASELINE"
    assert not summary["mail_enabled"] and not summary["smtp_configured"]
    assert summary["les_v7_bundle_configured"] is False
    service.assert_not_called(); lock.assert_not_called()
    assert not (tmp_path / "run.lock").exists()
