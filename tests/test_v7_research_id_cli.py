"""SYNTHETIC engineering contracts for named, isolated daily research series.

Native archive readiness is stubbed just as in the original CLI tests. No real
model, market refresh, SMTP delivery, account evolution or PAE ledger is run.
The real series manifest and CLI receipt boundaries are exercised on tmp_path.
"""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.replay_contracts import ReplayVersions
from infrastructure import les_v7_research_bundle as native
from interface import cli
from tests.strategies.test_low_entry_score_v7 import seal
from tests.test_v7_research_cli import synthetic_verified_model


RESEARCH_ID = "elec-study_Alpha-1"


@pytest.fixture
def named_research(monkeypatch, tmp_path):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    series = importlib.import_module("infrastructure.research_series")
    versions = {"current": ReplayVersions("SYNTHETIC_MODELS", "a" * 64, "b" * 64)}
    monkeypatch.setattr(series, "current_versions", lambda: versions["current"])
    # A sentinel is deliberately not a database. Research must not read or
    # initialize it merely because a normal daily path is configured.
    live = tmp_path / "live-pae.sqlite3"
    live.write_bytes(b"SYNTHETIC_FORMAL_STATE_DO_NOT_TOUCH")
    return [*args, "--research-id", RESEARCH_ID, "--skip-mops"], versions, live


def capture_service(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "build_stock_analysis_service", lambda **kw: SimpleNamespace(
        run=lambda request: captured.append((request, kw["settings"]))))
    return captured


def metadata(settings):
    return json.loads(settings.les_v7_research_metadata_json)


def study_root(tmp_path):
    return tmp_path / "data" / "les-v7-research" / RESEARCH_ID


def freeze_cli_clock(monkeypatch, day):
    current = dt.datetime.combine(day, dt.time(16), tzinfo=dt.timezone.utc)

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(cli, "dt", SimpleNamespace(datetime=Clock, date=dt.date,
        timezone=dt.timezone, timedelta=dt.timedelta))


def test_fixed_id_reuses_only_state_path_and_keeps_each_attempt_distinct(named_research, monkeypatch, tmp_path):
    args, _, live = named_research
    captured = capture_service(monkeypatch)
    before = live.read_bytes()
    freeze_cli_clock(monkeypatch, dt.date(2026, 10, 22))
    assert cli.main(args) == 0
    first_request, first_settings = captured[-1]
    root = study_root(tmp_path)
    original_study = (root / "study.json").read_bytes()
    freeze_cli_clock(monkeypatch, dt.date(2026, 10, 23))
    assert cli.main(args) == 0
    second_request, second_settings = captured[-1]
    first, second = metadata(first_settings), metadata(second_settings)
    assert first["research_id"] == second["research_id"] == RESEARCH_ID
    assert first["research_run_id"] != second["research_run_id"]
    assert first_settings.pae_persistence_path == second_settings.pae_persistence_path == root / "portfolio.sqlite3"
    assert first_request.report_file_prefix != second_request.report_file_prefix
    assert first_request.subject == second_request.subject
    assert first_request.stocklist == second_request.stocklist == ["3019", "5471"]
    assert (root / "study.json").read_bytes() == original_study
    assert not (root / "request.json").exists() and not (root / "completion.json").exists()
    for request, settings in captured:
        attempt = root / "runs" / metadata(settings)["research_run_id"]
        requested = json.loads((attempt / "request.json").read_bytes())
        completed = json.loads((attempt / "completion.json").read_bytes())
        assert requested["report_file_prefix"] == completed["report_file_prefix"] == request.report_file_prefix
        assert requested["model_use"]["research_id"] == completed["model_use"]["research_id"] == RESEARCH_ID
        assert completed["status"] == "COMPLETE"
        assert completed["execution_action"] == "NO_ACTION" and completed["approved_for_live"] is False
    assert live.read_bytes() == before


@pytest.mark.parametrize("research_id", ["", ".", "..", "../live", "study/child", "study\\child", "with space", "研究", "a" * 81])
def test_invalid_research_id_is_rejected_before_analysis(monkeypatch, tmp_path, research_id):
    args = synthetic_verified_model(monkeypatch, tmp_path)
    service = Mock(side_effect=AssertionError("invalid research id reached analysis"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--research-id", research_id, "--skip-mops"])
    assert error.value.code == 2
    service.assert_not_called()
    assert not list((tmp_path / "data").rglob("study.json"))
    assert not list((tmp_path / "data").rglob("request.json"))


def test_fixed_id_requires_explicit_research_bundle(monkeypatch, tmp_path):
    synthetic_verified_model(monkeypatch, tmp_path)
    service = Mock(side_effect=AssertionError("named ordinary daily analysis began"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    with pytest.raises(SystemExit) as error:
        cli.main(["--research-id", RESEARCH_ID, "--check-config", "--skip-mops", "1"])
    assert error.value.code == 2
    service.assert_not_called()
    assert not list((tmp_path / "data").rglob("study.json"))


@pytest.mark.parametrize("research_id", ["a", "a" * 80, "UPPER_lower-012"])
def test_valid_id_boundaries_are_not_replaced_with_random_run_id(named_research, research_id):
    args, _, _ = named_research
    args = list(args)
    args[args.index("--research-id") + 1] = research_id
    request, settings = cli.build_run_configuration([*args, "--check-config"])
    data = metadata(settings)
    assert data["research_id"] == research_id
    assert settings.pae_persistence_path.parent.name == research_id
    assert data["research_run_id"] != research_id
    assert request.mail is False


def test_named_preflight_is_read_only_even_without_smtp(named_research, monkeypatch, tmp_path, capsys):
    args, _, live = named_research
    before = live.read_bytes()
    for key in ("TWSE_SMTP_EMAIL", "TWSE_SMTP_PASSWORD", "TWSE_SMTP_RECEIVERS", "TWSE_SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    service = Mock(side_effect=AssertionError("preflight started analysis"))
    monkeypatch.setattr(cli, "build_stock_analysis_service", service)
    assert cli.main([*args, "--check-config"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["les_v7_research_model"]["research_id"] == RESEARCH_ID
    assert summary["mail_enabled"] is False
    assert not (study_root(tmp_path) / "study.json").exists()
    assert not (study_root(tmp_path) / "portfolio.sqlite3").exists()
    assert not list((tmp_path / "data").rglob("request.json"))
    assert not list((tmp_path / "data").rglob("completion.json"))
    assert live.read_bytes() == before
    service.assert_not_called()


@pytest.mark.parametrize("change", ["symbols", "symbol_order", "bundle", "dataset", "config", "code", "parameters"])
def test_existing_study_rejects_changed_identity_without_touching_manifest_or_live_state(
        named_research, monkeypatch, tmp_path, change):
    args, versions, live = named_research
    captured = capture_service(monkeypatch)
    assert cli.main(args) == 0
    manifest = study_root(tmp_path) / "study.json"
    before, live_before = manifest.read_bytes(), live.read_bytes()
    if change in {"code", "parameters"}:
        versions["current"] = replace(versions["current"], **{change: "f" * 64})
    else:
        loader = native.load_research_bundle

        def changed(path):
            record, data = loader(path)
            record, data = deepcopy(record), deepcopy(data)
            if change == "symbols":
                data["symbols"].append("2330")
            elif change == "symbol_order":
                data["symbols"].reverse()
            elif change == "bundle":
                record["payload"]["bundle_id"] = "SYNTHETIC_OTHER_MODEL"
                seal(record)
                data["bundle_sha256"] = record["sha256"]
            else:
                data[change + "_id"] = "e" * 64
            return record, data

        monkeypatch.setattr(native, "load_research_bundle", changed)
    with pytest.raises((ValueError, SystemExit)):
        cli.main(args)
    assert len(captured) == 1
    assert manifest.read_bytes() == before
    assert live.read_bytes() == live_before
    assert len(list((study_root(tmp_path) / "runs").glob("*/request.json"))) == 1


def test_existing_study_is_checked_but_not_rewritten_by_preflight(named_research, monkeypatch, tmp_path):
    args, versions, _ = named_research
    captured = capture_service(monkeypatch)
    assert cli.main(args) == 0
    manifest = study_root(tmp_path) / "study.json"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    assert cli.main([*args, "--check-config"]) == 0
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert len(captured) == 1
    versions["current"] = replace(versions["current"], code="f" * 64)
    with pytest.raises((ValueError, SystemExit)):
        cli.main([*args, "--check-config"])
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert len(captured) == 1


def test_account_values_can_evolve_without_rebasing_study_identity(named_research, monkeypatch, tmp_path):
    args, _, live = named_research
    captured = capture_service(monkeypatch)
    assert cli.main([*args, "--pae-current-cash", "1000000", "--account-equity", "1000000"]) == 0
    manifest = study_root(tmp_path) / "study.json"
    before = manifest.read_bytes()
    holdings = tmp_path / "SYNTHETIC_RESEARCH_HOLDINGS.txt"
    holdings.write_text("3019:10:2026-10-19:75:BUY_SMALL\n")
    assert cli.main([*args, "--pae-holdings-file", str(holdings),
        "--pae-current-cash", "900000", "--account-equity", "1100000"]) == 0
    assert len(captured) == 2
    assert captured[0][1].pae_current_cash == 1_000_000
    assert captured[1][1].pae_current_cash == 900_000
    assert captured[1][1].account_equity == 1_100_000
    assert captured[1][1].pae_positions[0].symbol == "3019"
    assert (study_root(tmp_path) / "study.json").read_bytes() == before
    assert live.read_bytes() == b"SYNTHETIC_FORMAL_STATE_DO_NOT_TOUCH"


@pytest.mark.parametrize("target_name", ["study.json", "portfolio.sqlite3", "runs"])
def test_named_study_rejects_symlink_redirection_before_service_or_target_write(
        named_research, monkeypatch, tmp_path, target_name):
    args, _, live = named_research
    captured = capture_service(monkeypatch)
    assert cli.main(args) == 0
    root = study_root(tmp_path)
    target = root / target_name
    if target.exists():
        target.rename(root / ("SYNTHETIC_BACKUP_" + target_name))
    before = live.read_bytes()
    target.symlink_to(live)
    with pytest.raises((ValueError, SystemExit)):
        cli.main([*args, "--check-config"])
    assert len(captured) == 1
    assert live.read_bytes() == before


def test_failed_named_attempt_keeps_request_and_next_attempt_does_not_overwrite_it(
        named_research, monkeypatch, tmp_path):
    args, _, _ = named_research

    def fail(request):
        raise RuntimeError("SYNTHETIC_RESEARCH_ATTEMPT_FAILURE")

    monkeypatch.setattr(cli, "build_stock_analysis_service", lambda **kw: SimpleNamespace(run=fail))
    with pytest.raises(RuntimeError, match="SYNTHETIC_RESEARCH_ATTEMPT_FAILURE"):
        cli.main(args)
    root = study_root(tmp_path)
    requests = list((root / "runs").glob("*/request.json"))
    assert len(requests) == 1
    failed_request, failed_body = requests[0], requests[0].read_bytes()
    assert not list((root / "runs").glob("*/completion.json"))
    captured = capture_service(monkeypatch)
    assert cli.main(args) == 0 and len(captured) == 1
    assert len(list((root / "runs").glob("*/request.json"))) == 2
    assert len(list((root / "runs").glob("*/completion.json"))) == 1
    assert failed_request.read_bytes() == failed_body
    assert not failed_request.with_name("completion.json").exists()


def test_omitting_id_retains_existing_one_shot_layout_and_normal_daily_setting(named_research, monkeypatch, tmp_path):
    args, _, live = named_research
    args = list(args)
    position = args.index("--research-id")
    del args[position:position + 2]
    captured = capture_service(monkeypatch)
    assert cli.main(args) == 0
    request, settings = captured[0]
    data = metadata(settings)
    root = settings.pae_persistence_path.parent
    assert root.name == data["research_run_id"]
    assert (root / "request.json").exists() and (root / "completion.json").exists()
    assert not (root / "study.json").exists()
    _, ordinary = cli.build_run_configuration(["--check-config", "--skip-mops", "1"])
    assert ordinary.pae_persistence_path == live
    assert ordinary.les_v7_research_metadata_json is None
    assert request.report_file_prefix.startswith("SHIRONG_LESv7_RESEARCH_")
    assert live.read_bytes() == b"SYNTHETIC_FORMAL_STATE_DO_NOT_TOUCH"
