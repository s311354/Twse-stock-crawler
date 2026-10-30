"""Synthetic, offline all-universe orchestration; never market certification.

These tiny real SQLite fixtures test scope/batch/publication boundaries. They
have no real prices, original observation receipts, corporate-action proof or
trained model. No production database, account, replay or Bank state is used.
"""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sqlite3

import pytest

from application.les_v7_dataset import build_training_dataset
from application.replay_contracts import canonical_json, digest
from infrastructure.les_v7_dataset_scope import (
    freeze_elec_scope, run_elec_scope, verify_elec_scope,
)
from infrastructure.les_v7_training_source import FINAL_CLOSE, load_training_source
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository
from tests.test_v7_dataset_builder import builder_inputs
from tests.test_v7_training_publication import Guard
from tests.test_v7_training_source import fixture as source_fixture


NOW = "2026-10-13T15:00:00+08:00"
START = date(2026, 7, 30)
END = date(2026, 8, 5)


def scope_inputs(tmp_path, *, count=65, max_events=4096, absent_on_event=()):
    """Official-shape SYNTHETIC source with all names fixed before building."""
    parent = source_fixture()
    original = canonical_json(dict(evidence_id=digest(parent), payload=parent)).encode()
    source = tmp_path / "SYNTHETIC-calendar-evidence.json"
    source.write_bytes(original)
    symbols = [str(1000 + index) for index in range(count)]
    template = parent["bars"][0]["data"][0]
    rows = []
    for symbol in symbols:
        row = deepcopy(template)
        row[0], row[1] = symbol, "SYNTHETIC_ONLY:" + symbol
        rows.append(row)
    cache = tmp_path / "SYNTHETIC-market.sqlite3"
    with sqlite3.connect(cache) as connection:
        connection.execute("""CREATE TABLE market_history_cache (
            trade_date TEXT, dataset TEXT, market_type TEXT,
            schema_version INTEGER, payload_json TEXT, cached_at_epoch INTEGER,
            PRIMARY KEY(trade_date,dataset,market_type,schema_version))""")
        for day in ("20260730", "20260803", "20260804", "20260805"):
            selected = [row for row in rows if day != "20260803" or row[0] not in absent_on_event]
            connection.execute("INSERT INTO market_history_cache VALUES (?,?,?,?,?,?)",
                               (day, "STOCK_TABLE", "13", 1, json.dumps(selected), 1))
            connection.execute("INSERT INTO market_history_cache VALUES (?,?,?,?,?,?)",
                               (day, "TAIEX", "IND", 1, json.dumps("22,000"), 1))
    _, config = builder_inputs()
    config.update(event_start="2026-08-03", event_end="2026-08-05",
                  evaluation_cutoff="2026-08-06T13:30:00+08:00",
                  coordinate_coverages=[], max_events=max_events)
    return dict(source=source, source_bytes=original, cache=cache,
                symbols=symbols, config=config, root=tmp_path / "SYNTHETIC-scope")


def freeze(inputs, **options):
    return freeze_elec_scope(inputs["source"], market_cache=inputs["cache"],
        history_start=START, history_end=END, universe_date=END,
        config=inputs["config"], root=inputs["root"], allow_synthetic=True, **options)


def native_repository(root, **options):
    return LesV7DatasetRepository(root, **options)


def execute(inputs, freeze_path, **options):
    return run_elec_scope(freeze_path, inputs["config"], root=inputs["root"],
        allow_final_daily_reconstruction=True, allow_synthetic=True,
        guard_factory=Guard, repository_factory=native_repository,
        clock=lambda: NOW, **options)


def frozen_body(freeze_path):
    record = json.loads(Path(freeze_path).read_text())
    assert record["freeze_id"] == digest(record["body"])
    assert record["body"]["schema"] == "les-v7-elec-scope-freeze/v1"
    return record["body"]


def evidence_for(inputs, chunk):
    matches = [path for path in inputs["root"].rglob("evidence.json")
               if path.parent.name == chunk["evidence_id"]]
    assert len(matches) == 1
    return matches[0]


def test_fixed_scope_partitions_every_symbol_once_and_preserves_original_source(tmp_path):
    inputs = scope_inputs(tmp_path)
    cache_bytes = inputs["cache"].read_bytes()
    manifest = freeze(inputs)
    body = frozen_body(manifest)
    assert body["universe"] == inputs["symbols"]
    assert [len(chunk["symbols"]) for chunk in body["chunks"]] == [32, 32, 1]
    flattened = [symbol for chunk in body["chunks"] for symbol in chunk["symbols"]]
    assert flattened == body["universe"] and len(flattened) == len(set(flattened))
    assert inputs["source"].read_bytes() == inputs["source_bytes"]
    assert inputs["cache"].read_bytes() == cache_bytes
    universe_record = body["universe_record"]
    assert universe_record["source_day"] == END.isoformat()
    assert universe_record["market_type"] == "13"
    assert universe_record["raw_sha256"]
    assert [row["symbol"] for row in universe_record["rows"]] == inputs["symbols"]


def test_identical_freeze_is_idempotent_and_does_not_overwrite_timestamps(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    first = freeze(inputs)
    before = {str(path.relative_to(inputs["root"])): path.read_bytes()
              for path in inputs["root"].rglob("*") if path.is_file()}
    assert freeze(inputs) == first
    after = {str(path.relative_to(inputs["root"])): path.read_bytes()
             for path in inputs["root"].rglob("*") if path.is_file()}
    assert after == before


def test_missing_event_bar_remains_in_fixed_scope_and_native_gap_diagnostics(tmp_path):
    inputs = scope_inputs(tmp_path, count=3, absent_on_event=("1001",))
    manifest = freeze(inputs)
    body = frozen_body(manifest)
    chunk = body["chunks"][0]
    source = load_training_source(evidence_for(inputs, chunk),
        evaluation_cutoff=inputs["config"]["evaluation_cutoff"], availability_policy=FINAL_CLOSE)
    assert source["payload"]["universe"] == [[symbol, "TWSE"] for symbol in inputs["symbols"]]
    history = next(row for row in source["payload"]["histories"] if row["symbol"] == "1001")
    assert "2026-08-03" not in [bar["Date"] for bar in history["bars"]]
    built = build_training_dataset(source, inputs["config"])
    gap = next(row for row in built["result"]["diagnostics"]
               if row["symbol"] == "1001" and row["cutoff"].startswith("2026-08-03"))
    assert "NO_COMPLETE_BAR_FOR_EVENT_SESSION" in gap["reasons"]
    assert built["result"]["status"] == "PARTIAL"
    assert built["dataset"]["payload"]["universe"] == source["payload"]["universe"]


def test_each_chunk_respects_event_budget_without_dropping_symbols_or_dates(tmp_path):
    inputs = scope_inputs(tmp_path, count=9, max_events=6)
    body = frozen_body(freeze(inputs))
    assert [len(chunk["symbols"]) for chunk in body["chunks"]] == [2, 2, 2, 2, 1]
    assert [symbol for chunk in body["chunks"] for symbol in chunk["symbols"]] == inputs["symbols"]
    for chunk in body["chunks"]:
        source = load_training_source(evidence_for(inputs, chunk),
            evaluation_cutoff=inputs["config"]["evaluation_cutoff"], availability_policy=FINAL_CLOSE)
        built = build_training_dataset(source, inputs["config"])
        assert built["result"]["counts"]["candidate_events"] == len(chunk["symbols"]) * 3


def test_event_budget_smaller_than_one_stock_window_rejects_before_publication(tmp_path):
    inputs = scope_inputs(tmp_path, count=3, max_events=2)
    with pytest.raises(ValueError):
        freeze(inputs)
    assert not list(inputs["root"].rglob("manifest.json"))


@pytest.mark.parametrize("batch_size", [0, -1, 33, True, 1.5])
def test_invalid_batch_bound_is_rejected_without_partial_freeze(tmp_path, batch_size):
    inputs = scope_inputs(tmp_path, count=3)
    with pytest.raises(ValueError):
        freeze(inputs, batch_size=batch_size)
    assert not list(inputs["root"].rglob("manifest.json"))


def test_synthetic_source_requires_explicit_opt_in_before_publication(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    with pytest.raises(ValueError):
        freeze_elec_scope(inputs["source"], market_cache=inputs["cache"],
            history_start=START, history_end=END, universe_date=END,
            config=inputs["config"], root=inputs["root"])
    assert not list(inputs["root"].rglob("manifest.json"))


def test_same_scope_native_retry_is_immutable_and_never_rebuilds_completed_batches(tmp_path):
    inputs = scope_inputs(tmp_path, count=5)
    manifest = freeze(inputs, batch_size=2)
    calls = []

    def counting_repository(root, **options):
        class CountingRepository(LesV7DatasetRepository):
            def run(self, source, config, *, allow_synthetic=False, **kwargs):
                def builder(selected, settings):
                    calls.append(deepcopy(selected["payload"]["universe"]))
                    return build_training_dataset(selected, settings)
                return super().run(source, config, allow_synthetic=allow_synthetic, builder=builder)
        return CountingRepository(root, **options)

    first = run_elec_scope(manifest, inputs["config"], root=inputs["root"],
        allow_final_daily_reconstruction=True, allow_synthetic=True,
        guard_factory=Guard, repository_factory=counting_repository, clock=lambda: NOW)
    before = {str(path.relative_to(inputs["root"])): path.read_bytes()
              for path in inputs["root"].rglob("*") if path.is_file()}
    second = run_elec_scope(manifest, inputs["config"], root=inputs["root"],
        allow_final_daily_reconstruction=True, allow_synthetic=True,
        guard_factory=Guard, repository_factory=counting_repository, clock=lambda: NOW)
    assert first == second == verify_elec_scope(first["output"], root=inputs["root"])
    assert len(calls) == 3
    assert first["status"] == "PARTIAL" and first["symbol_count"] == 5 and first["batch_count"] == 3
    assert {str(path.relative_to(inputs["root"])): path.read_bytes()
            for path in inputs["root"].rglob("*") if path.is_file()} == before


def test_after_batch_interruption_recovers_same_native_results_without_duplicate_events(tmp_path):
    inputs = scope_inputs(tmp_path, count=5)
    manifest = freeze(inputs, batch_size=2)
    callbacks = []

    def interrupt(index, receipt):
        callbacks.append((index, receipt["dataset_id"]))
        if index == 0:
            raise RuntimeError("SYNTHETIC_SCOPE_INTERRUPTION")

    with pytest.raises(RuntimeError, match="SYNTHETIC_SCOPE_INTERRUPTION"):
        execute(inputs, manifest, after_batch=interrupt)
    assert not list((inputs["root"] / "jobs").rglob("completion.json"))
    first_native = [path for path in inputs["root"].rglob("completion.json")
                    if json.loads(path.read_text()).get("schema") == "les-v7-dataset-build-completion/v1"]
    assert len(first_native) == 1
    previous = first_native[0].read_bytes()
    complete = execute(inputs, manifest)
    assert first_native[0].read_bytes() == previous
    assert complete == verify_elec_scope(complete["output"], root=inputs["root"])
    native = [path for path in inputs["root"].rglob("completion.json")
              if json.loads(path.read_text()).get("schema") == "les-v7-dataset-build-completion/v1"]
    assert len(native) == 3


def test_changed_config_creates_new_research_job_without_rewriting_old_job_or_evidence(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    previous = execute(inputs, manifest)
    before = {path.name: path.read_bytes() for path in Path(previous["output"]).iterdir() if path.is_file()}
    changed = deepcopy(inputs["config"])
    changed["seed"]["reference_policy"]["reference_shares"] += 1
    current = run_elec_scope(manifest, changed, root=inputs["root"],
        allow_final_daily_reconstruction=True, allow_synthetic=True,
        guard_factory=Guard, repository_factory=native_repository, clock=lambda: NOW)
    assert current["freeze_id"] == previous["freeze_id"]
    assert current["job_id"] != previous["job_id"] and current["config_id"] != previous["config_id"]
    assert {path.name: path.read_bytes() for path in Path(previous["output"]).iterdir() if path.is_file()} == before
    assert verify_elec_scope(previous["output"], root=inputs["root"]) == previous


def test_corrupt_frozen_evidence_is_rejected_not_overwritten_or_marked_complete(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    chunk = frozen_body(manifest)["chunks"][0]
    evidence = evidence_for(inputs, chunk)
    damaged = b"SYNTHETIC CORRUPT EVIDENCE MUST NOT BE OVERWRITTEN"
    evidence.write_bytes(damaged)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        execute(inputs, manifest)
    assert evidence.read_bytes() == damaged
    assert not list(inputs["root"].rglob("completion.json"))


def test_source_version_drift_cannot_publish_scope_completion(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    guard = Guard()
    guard.drift_after = 2
    with pytest.raises(ValueError, match="VERSION_CHANGED_DURING_JOB"):
        run_elec_scope(manifest, inputs["config"], root=inputs["root"],
            allow_final_daily_reconstruction=True, allow_synthetic=True,
            guard_factory=lambda: guard, repository_factory=native_repository, clock=lambda: NOW)
    assert not list((inputs["root"] / "jobs").rglob("completion.json"))


def test_completed_scope_detects_child_corruption_without_rebuilding(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    receipt = execute(inputs, manifest)
    native = next(path for path in inputs["root"].rglob("completion.json")
                  if json.loads(path.read_text()).get("schema") == "les-v7-dataset-build-completion/v1")
    artifact = native.parent / "build_result.json"
    damaged = b"SYNTHETIC CORRUPT RESULT"
    artifact.write_bytes(damaged)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify_elec_scope(receipt["output"], root=inputs["root"])
    with pytest.raises((ValueError, json.JSONDecodeError)):
        execute(inputs, manifest)
    assert artifact.read_bytes() == damaged


def test_duplicate_source_membership_is_rejected_instead_of_silently_set_collapsed(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    with sqlite3.connect(inputs["cache"]) as connection:
        serialized, = connection.execute("""SELECT payload_json FROM market_history_cache
            WHERE trade_date='20260805' AND dataset='STOCK_TABLE' AND market_type='13'""").fetchone()
        rows = json.loads(serialized)
        rows.append(deepcopy(rows[0]))
        connection.execute("""UPDATE market_history_cache SET payload_json=?
            WHERE trade_date='20260805' AND dataset='STOCK_TABLE' AND market_type='13'""", (json.dumps(rows),))
    with pytest.raises(ValueError, match="DUPLICATE_SYMBOL"):
        freeze(inputs)
    assert not list(inputs["root"].rglob("manifest.json"))


def test_nonordinary_member_is_preserved_but_never_becomes_eligible_reference_event(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    with sqlite3.connect(inputs["cache"]) as connection:
        records = connection.execute("""SELECT trade_date,payload_json FROM market_history_cache
            WHERE dataset='STOCK_TABLE' AND market_type='13'""").fetchall()
        for day, serialized in records:
            rows = json.loads(serialized)
            rows[1][0], rows[1][1] = "1001A", "SYNTHETIC_TEST_ONLY 特別股"
            connection.execute("""UPDATE market_history_cache SET payload_json=?
                WHERE trade_date=? AND dataset='STOCK_TABLE' AND market_type='13'""", (json.dumps(rows), day))
    manifest = freeze(inputs)
    body = frozen_body(manifest)
    assert body["universe"] == ["1000", "1001A", "1002"]
    source = load_training_source(evidence_for(inputs, body["chunks"][0]),
        evaluation_cutoff=inputs["config"]["evaluation_cutoff"], availability_policy=FINAL_CLOSE)
    history = next(row for row in source["payload"]["histories"] if row["symbol"] == "1001A")
    assert history["security_type"] == "PREFERRED"
    built = build_training_dataset(source, inputs["config"])
    assert not any(event["symbol"] == "1001A" for event in built["dataset"]["payload"]["events"])
    assert all(row["status"] == "DATA_GAP" for row in built["result"]["diagnostics"] if row["symbol"] == "1001A")
    receipt = execute(inputs, manifest)
    assert receipt["symbol_count"] == 3
    assert receipt["summary"]["unsupported_instruments"] == ["1001A"]


def test_resume_uses_frozen_source_not_later_market_cache_changes(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    body = frozen_body(manifest)
    evidence = evidence_for(inputs, body["chunks"][0])
    frozen_bytes = evidence.read_bytes()
    first = execute(inputs, manifest)
    with sqlite3.connect(inputs["cache"]) as connection:
        serialized, = connection.execute("""SELECT payload_json FROM market_history_cache
            WHERE trade_date='20260805' AND dataset='STOCK_TABLE' AND market_type='13'""").fetchone()
        rows = json.loads(serialized)
        rows[0][8] = "101.50"
        connection.execute("""UPDATE market_history_cache SET payload_json=?
            WHERE trade_date='20260805' AND dataset='STOCK_TABLE' AND market_type='13'""", (json.dumps(rows),))
    assert execute(inputs, manifest) == first
    assert evidence.read_bytes() == frozen_bytes
    second_freeze = freeze(inputs)
    assert second_freeze != manifest
    assert frozen_body(second_freeze)["universe"] == body["universe"]
    assert evidence.read_bytes() == frozen_bytes


def test_cli_resume_preserves_explicit_policy_and_never_reopens_current_cache(tmp_path, monkeypatch, capsys):
    import interface.les_v7_dataset_scope as cli

    inputs = scope_inputs(tmp_path, count=3)
    config_path = tmp_path / "SYNTHETIC-config.json"
    config_path.write_text(json.dumps(inputs["config"]))
    manifest = tmp_path / "SYNTHETIC-frozen-manifest.json"
    recorded = []
    receipt = dict(status="PARTIAL", symbol_count=3, batch_count=1,
                   research_only=True, approved_for_live=False, execution_action="NO_ACTION")

    def run(path, config, **options):
        recorded.append((path, config, options))
        return receipt

    monkeypatch.setattr(cli, "freeze_elec_scope", lambda *_, **__: pytest.fail("resume reopened current market cache"))
    monkeypatch.setattr(cli, "run_elec_scope", run)
    assert cli.main(["--config", str(config_path), "--output-dir", str(inputs["root"]),
        "--resume-freeze", str(manifest), "--allow-final-daily-reconstruction", "--allow-synthetic"]) == 2
    assert len(recorded) == 1
    path, config, options = recorded[0]
    assert path == manifest and config == inputs["config"]
    assert options["root"] == inputs["root"] and options["allow_synthetic"] is True
    assert options["allow_final_daily_reconstruction"] is True
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[-1] == receipt
    assert not inputs["root"].exists()


def test_cli_new_freeze_rejects_incomplete_source_flags_before_any_publication(tmp_path, monkeypatch):
    import interface.les_v7_dataset_scope as cli

    inputs = scope_inputs(tmp_path, count=3)
    config_path = tmp_path / "SYNTHETIC-config.json"
    config_path.write_text(json.dumps(inputs["config"]))
    monkeypatch.setattr(cli, "freeze_elec_scope", lambda *_, **__: pytest.fail("incomplete flags froze evidence"))
    monkeypatch.setattr(cli, "run_elec_scope", lambda *_, **__: pytest.fail("incomplete flags ran research"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config_path), "--output-dir", str(inputs["root"]),
                  "--calendar-evidence", str(inputs["source"]), "--market-cache", str(inputs["cache"])])
    assert error.value.code == 2
    assert not inputs["root"].exists()


def test_cli_verify_never_builds_or_reads_missing_configuration(tmp_path, monkeypatch, capsys):
    import interface.les_v7_dataset_scope as cli

    root, job = tmp_path / "SYNTHETIC-root", tmp_path / "SYNTHETIC-job"
    recorded = []
    receipt = dict(status="PARTIAL", execution_action="NO_ACTION")
    monkeypatch.setattr(cli, "verify_elec_scope", lambda path, **options: recorded.append((path, options)) or receipt)
    monkeypatch.setattr(cli, "run_elec_scope", lambda *_, **__: pytest.fail("verify ran research"))
    monkeypatch.setattr(cli, "freeze_elec_scope", lambda *_, **__: pytest.fail("verify read current source"))
    assert cli.main(["--config", str(tmp_path / "SYNTHETIC-missing-config.json"),
                     "--output-dir", str(root), "--verify-job", str(job)]) == 2
    assert recorded == [(job, dict(root=root))]
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.parametrize("field,altered", [
    ("symbols_with_513_closes", 3),
    ("symbols_with_no_bars", 3),
    ("unsupported_instruments", ["1001"]),
])
def test_completed_summary_cannot_disagree_with_native_coverage(tmp_path, field, altered):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    receipt = execute(inputs, manifest)
    path = Path(receipt["output"]) / "completion.json"
    damaged = json.loads(path.read_text())
    assert damaged["summary"][field] != altered
    damaged["summary"][field] = altered
    path.write_text(canonical_json(damaged))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        verify_elec_scope(receipt["output"], root=inputs["root"])
    with pytest.raises(ValueError):
        execute(inputs, manifest)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,altered", [
    ("research_only", False),
    ("approved_for_live", True),
    ("dataset_kind", "RESEARCH"),
])
def test_completion_safety_and_source_kind_flags_cannot_be_reinterpreted(tmp_path, field, altered):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    receipt = execute(inputs, manifest)
    path = Path(receipt["output"]) / "completion.json"
    damaged = json.loads(path.read_text())
    damaged[field] = altered
    path.write_text(canonical_json(damaged))
    with pytest.raises(ValueError):
        verify_elec_scope(receipt["output"], root=inputs["root"])


def test_rehashed_invalid_coverage_schema_cannot_be_accepted(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    receipt = execute(inputs, freeze(inputs))
    folder = Path(receipt["output"])
    coverage_path, completion_path = folder / "coverage.json", folder / "completion.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["schema"] = "SYNTHETIC_UNSUPPORTED_COVERAGE_SCHEMA"
    coverage_path.write_text(canonical_json(coverage))
    completion = json.loads(completion_path.read_text())
    completion["coverage_sha256"] = digest(coverage)
    completion_path.write_text(canonical_json(completion))
    with pytest.raises(ValueError):
        verify_elec_scope(folder, root=inputs["root"])


def test_saved_freeze_reference_is_verified_not_ignored(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    receipt = execute(inputs, freeze(inputs))
    path = Path(receipt["output"]) / "freeze_reference.json"
    path.write_text(canonical_json(dict(freeze_id="0" * 64)))
    with pytest.raises(ValueError):
        verify_elec_scope(receipt["output"], root=inputs["root"])


@pytest.mark.parametrize("mutation", ["calendar", "market", "batch_size", "classification"])
def test_new_manifest_hash_does_not_excuse_inconsistent_existing_shards(tmp_path, mutation):
    inputs = scope_inputs(tmp_path, count=3)
    manifest = freeze(inputs)
    record = json.loads(Path(manifest).read_text())
    body = record["body"]
    if mutation == "calendar":
        body["calendar_sha256"] = "0" * 64
    elif mutation == "market":
        body["market"] = "TPEX"
    elif mutation == "batch_size":
        body["batch_size"] = 1
    else:
        body["universe_record"]["rows"][0]["security_type"] = "PREFERRED"
    record["freeze_id"] = digest(body)
    forged = inputs["root"] / "freezes" / record["freeze_id"] / "manifest.json"
    forged.parent.mkdir(parents=True)
    forged.write_text(canonical_json(record))
    before = forged.read_bytes()
    with pytest.raises(ValueError):
        execute(inputs, forged)
    assert forged.read_bytes() == before
    assert not list((inputs["root"] / "jobs").rglob("completion.json"))


def test_checkpoint_resume_does_not_reselect_or_rebuild_completed_first_batch(tmp_path, monkeypatch):
    import infrastructure.les_v7_dataset_scope as scope

    inputs = scope_inputs(tmp_path, count=5)
    manifest = freeze(inputs, batch_size=2)
    first_chunk = frozen_body(manifest)["chunks"][0]

    def interrupt(index, receipt):
        if index == 0:
            raise RuntimeError("SYNTHETIC_INTERRUPT_AFTER_SAVED_BATCH_CHECKPOINT")

    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPT"):
        execute(inputs, manifest, after_batch=interrupt)
    load_source, selections = scope.load_training_source, []

    def selected(path, **options):
        assert Path(path).parent.name != first_chunk["evidence_id"], "checkpointed batch was selected again"
        selections.append(Path(path).parent.name)
        return load_source(path, **options)

    monkeypatch.setattr(scope, "load_training_source", selected)
    receipt = execute(inputs, manifest)
    assert len(selections) == 2
    assert receipt["batch_count"] == 3 and receipt["symbol_count"] == 5
    assert verify_elec_scope(receipt["output"], root=inputs["root"]) == receipt


def test_completed_scope_retry_does_not_select_any_sources(tmp_path, monkeypatch):
    import infrastructure.les_v7_dataset_scope as scope

    inputs = scope_inputs(tmp_path, count=5)
    manifest = freeze(inputs, batch_size=2)
    receipt = execute(inputs, manifest)
    monkeypatch.setattr(scope, "load_training_source", lambda *_, **__: pytest.fail("completed scope selected source again"))
    assert execute(inputs, manifest) == receipt
