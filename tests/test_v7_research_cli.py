"""SYNTHETIC engineering checks only; no real trained model/readiness claim."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.settings import AppSettings
from domain.les_v7.quality_policy import default_quality_policy
from infrastructure import les_v7_research_bundle as native
from interface import cli
from tests.test_v7_training_publication import run, fitted
from tests.strategies.test_low_entry_score_v7 import bundle, seal


def test_synthetic_native_publication_cannot_enter_real_research(tmp_path):
    receipt = run(tmp_path / "models")
    with pytest.raises(ValueError, match="READY_REAL_RESEARCH_MODEL_REQUIRED"):
        native.load_research_bundle(Path(receipt["output"]) / "frozen_bundle.json")


def test_full_native_archive_verification_catches_oof_tampering(tmp_path):
    receipt = run(tmp_path / "models")
    folder = Path(receipt["output"])
    (folder / "oof_predictions.json").write_text(json.dumps(dict(predictions=["SYNTHETIC_TAMPER"])))
    with pytest.raises(ValueError, match="TRAINING_ARTIFACT_HASH_MISMATCH"):
        native.load_research_bundle(folder / "frozen_bundle.json")


def test_not_ready_archive_cannot_be_treated_as_a_published_model(tmp_path):
    receipt = run(tmp_path / "models", trainer=lambda d, c: fitted(d, c, ready=False))
    with pytest.raises(ValueError, match="READY_REAL_RESEARCH_MODEL_REQUIRED"):
        native.load_research_bundle(Path(receipt["output"]) / "frozen_bundle.json")


def test_standalone_download_and_symlink_do_not_replace_archive(tmp_path):
    standalone = tmp_path / "frozen_bundle.json"
    standalone.write_text("{}")
    with pytest.raises(ValueError, match="NATIVE_TRAINING_ARCHIVE_REQUIRED"):
        native.load_research_bundle(standalone)
    link = tmp_path / "link" / "frozen_bundle.json"
    link.parent.mkdir()
    link.symlink_to(standalone)
    with pytest.raises(ValueError, match="NATIVE_FROZEN_BUNDLE_PATH_REQUIRED"):
        native.load_research_bundle(link)


def synthetic_verified_model(monkeypatch, tmp_path):
    """Stub public archive verification to test CLI orchestration, not readiness."""
    # Complete synthetic model contract; archive verification remains stubbed
    # here, while readiness validates the same shape as the public v7 kernel.
    record = bundle()
    record["payload"]["quality_policy"] = default_quality_policy()
    seal(record)
    metadata = dict(schema="les-v7-research-model-use/v1", native_job_id="b" * 64,
        bundle_sha256=record["sha256"], dataset_id="c" * 64, config_id="d" * 64,
        archive_path=str(tmp_path / "SYNTHETIC_ONLY"), symbols=["3019", "5471"],
        required_valid_close_count=513, research_only=True, approved_for_live=False,
        execution_action="NO_ACTION", order_authority=False)
    monkeypatch.setattr(native, "load_research_bundle", lambda *_: (deepcopy(record), deepcopy(metadata)))
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("TWSE_MARKET_HISTORY_CACHE_PATH", str(tmp_path / "market.sqlite3"))
    monkeypatch.setenv("TWSE_RUN_LOCK_PATH", str(tmp_path / "run.lock"))
    monkeypatch.setenv("MOPS_CACHE_DIR", str(tmp_path / "mops"))
    monkeypatch.setenv("TWSE_DAILY_ENTRY_MODEL", "v6")
    monkeypatch.setenv("TWSE_LES_V7_BUNDLE", str(tmp_path / "unrelated-unapproved-model.json"))
    monkeypatch.setenv("TWSE_PAE_PERSISTENCE_PATH", str(tmp_path / "live-pae.sqlite3"))
    monkeypatch.setenv("TWSE_SMTP_EMAIL", "research@example.com")
    monkeypatch.setenv("TWSE_SMTP_PASSWORD", "SYNTHETIC_ONLY_NOT_A_SECRET")
    monkeypatch.setenv("TWSE_SMTP_RECEIVERS", "reader@example.com")
    monkeypatch.setenv("TWSE_SMTP_HOST", "smtp.example.invalid")
    monkeypatch.delenv("TWSE_PAE_HOLDINGS_JSON", raising=False)
    monkeypatch.delenv("TWSE_PAE_CURRENT_CASH", raising=False)
    return ["--les-v7-research-bundle", str(tmp_path / "SYNTHETIC_ONLY.json")]


def test_explicit_research_uses_fixed_symbols_and_isolated_state_without_changing_defaults(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    request, settings = cli.build_run_configuration([*args, "--skip-mops", "1", "2"])
    assert settings.daily_entry_model == "v7" and settings.les_v7_bundle_json
    assert request.stocklist == ["3019", "5471"]
    assert request.explicit_symbols == ("3019", "5471")
    assert request.endbacktrack == 1100 and request.beginbacktrack == 0
    assert request.mail is True and request.notification_enabled is True
    assert "LES_V7_RESEARCH" in request.subject and "不可下單" in request.subject
    assert settings.pae_persistence_path != tmp_path / "live-pae.sqlite3"
    assert settings.pae_persistence_path.is_relative_to(tmp_path / "data" / "les-v7-research")
    model_hash = json.loads(settings.les_v7_bundle_json)["sha256"]
    assert "LESv7_RESEARCH_" + model_hash[:12] in request.report_file_prefix
    summary = cli.build_preflight_summary(request, settings)
    assert summary["evaluation_mode"] == "OFFLINE_MODEL_RESEARCH_ON_CURRENT_DATA"
    assert summary["les_v7_research_model"]["approved_for_live"] is False
    assert summary["les_v7_research_model"]["required_valid_close_count"] == 513
    # A calendar window is not proof of sufficient observations. Native gates
    # still reject insufficient closes, even when a shorter window is explicit.
    explicit, _ = cli.build_run_configuration([*args, "--skip-mops", "-e", "700"])
    assert explicit.endbacktrack == 700
    attached, _ = cli.build_run_configuration([*args, "--skip-mops", "-e2000"])
    assert attached.endbacktrack == 2000
    assert AppSettings.from_env({}, validate_paths=False).daily_entry_model == "v6"


def test_research_no_mail_is_explicit_and_does_not_require_smtp(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    for key in ("TWSE_SMTP_EMAIL", "TWSE_SMTP_PASSWORD", "TWSE_SMTP_RECEIVERS", "TWSE_SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    request, settings = cli.build_run_configuration([*args, "--no-mail", "--skip-mops"])
    assert request.mail is False
    assert request.notification_enabled is False
    assert settings.les_v7_research_metadata_json is not None


def test_no_mail_cannot_be_combined_with_explicit_delivery(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.build_run_configuration(["--no-mail", "--mail", "--skip-mops", "0"])
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [["--replay-mail"], ["--linechart"],
    ["-b", "2024-01-01", "-e", "2026-10-12"], ["-t", "SEMI"]])
def test_research_rejects_replay_mail_historical_backdating_and_other_profiles(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        cli.build_run_configuration(["--les-v7-research-bundle", str(tmp_path / "missing"), *extra])
    assert error.value.code == 2


def test_research_cannot_expand_frozen_scope(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("2330")
    with pytest.raises(SystemExit):
        cli.build_run_configuration([*args, "--include-symbols-file", str(symbols)])
    holdings = tmp_path / "holdings.txt"
    holdings.write_text("2330:10")
    with pytest.raises(SystemExit):
        cli.build_run_configuration([*args, "--pae-holdings-file", str(holdings)])


def test_research_preflight_does_not_start_analysis_or_create_completion(monkeypatch, tmp_path, capsys):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    for key in ("TWSE_SMTP_EMAIL", "TWSE_SMTP_PASSWORD", "TWSE_SMTP_RECEIVERS", "TWSE_SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    service = Mock(side_effect=AssertionError("preflight started analysis"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    assert cli.main([*args, "--check-config", "--skip-mops"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mail_enabled"] is False and summary["les_v7_research_model"]["execution_action"] == "NO_ACTION"
    service.assert_not_called()
    assert not list((tmp_path / "data").rglob("completion.json"))


def test_research_analysis_provenance_is_completion_last(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    captured = []
    monkeypatch.setattr(cli, "build_stock_analysis_service", lambda **kw: SimpleNamespace(
        run=lambda request: captured.append((request, kw["settings"]))))
    assert cli.main([*args, "--skip-mops"]) == 0
    request, settings = captured[0]
    provenance = json.loads((settings.pae_persistence_path.parent / "completion.json").read_text())
    assert provenance["report_file_prefix"] == request.report_file_prefix
    assert provenance["mail_enabled"] is True
    assert provenance["approved_for_live"] is False and provenance["execution_action"] == "NO_ACTION"
    assert not (tmp_path / "live-pae.sqlite3").exists()


def test_failed_research_keeps_request_without_a_completion_receipt(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    def fail(request):
        raise RuntimeError("SYNTHETIC_ANALYSIS_FAILURE")
    monkeypatch.setattr(cli, "build_stock_analysis_service", lambda **kw: SimpleNamespace(run=fail))
    with pytest.raises(RuntimeError, match="SYNTHETIC_ANALYSIS_FAILURE"):
        cli.main([*args, "--skip-mops"])
    assert len(list((tmp_path / "data").rglob("request.json"))) == 1
    assert not list((tmp_path / "data").rglob("completion.json"))


def test_explicit_mail_is_compatible_with_default_research_delivery(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    request, _ = cli.build_run_configuration([*args, "--skip-mops", "--mail"])
    assert request.mail and request.notification_enabled


def test_research_without_smtp_configuration_fails_before_analysis(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    monkeypatch.delenv("TWSE_SMTP_PASSWORD", raising=False)
    service = Mock(side_effect=AssertionError("analysis began without SMTP validation"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--skip-mops"])
    assert error.value.code == 2
    service.assert_not_called()
    assert not list((tmp_path / "data").rglob("request.json"))


def test_labelled_research_report_uses_existing_notification_port(monkeypatch, tmp_path):
    from application.contracts import AnalysisContext, MarketStatus, StrategyEvaluation
    from infrastructure.runtime.output_adapters import CrawlerNotificationAdapter
    import datetime
    args = synthetic_verified_model(monkeypatch, tmp_path)
    request, _ = cli.build_run_configuration([*args, "--skip-mops"])
    crawler = SimpleNamespace(smtp_strategy_email=Mock())
    context = AnalysisContext(crawler, datetime.date(2026, 10, 13), MarketStatus.OPEN)
    # The existing notification port receives typed results; this test never
    # connects to SMTP or claims a real model's eligibility was verified.
    CrawlerNotificationAdapter().send(context, request, StrategyEvaluation([], (SimpleNamespace(),)))
    crawler.smtp_strategy_email.assert_called_once()
    assert "LES_V7_RESEARCH" in crawler.smtp_strategy_email.call_args.kwargs["subject"]
    assert "不可下單" in crawler.smtp_strategy_email.call_args.kwargs["subject"]
