"""SYNTHETIC ONLY: L2 public settings/native wiring, not a market fit or gain."""
from copy import deepcopy
import json

import pytest

from domain.les_v7 import fr3_plan, fr3_l2
from domain.les_v7.contracts import ReferencePlan, canonical, identity
from infrastructure import les_v7_fr3_training as port
from interface import les_v7_fr3 as cli
from tests.test_v7_fr3_training import inputs as r1_inputs
from tests.test_v7_training_publication import Guard

CREATED = "2026-10-24T10:00:00+08:00"


def nonempty_native_source(*, fill=True):
    """Invented normalized production-shape source, never official evidence.

    RESEARCH selects the native normalized-calendar schema under test. Every
    upstream evidence/coordinate ID remains explicitly SYNTHETIC; no source
    authenticity, real training result or historical availability is claimed.
    """
    import numpy as np
    import pandas as pd
    from tests.test_v7_dataset_builder import builder_inputs, reseal
    from tests.test_v7_plan_r18 import neutral_bars
    source, build = builder_inputs()
    trailing = neutral_bars()
    trailing.loc[trailing.index[-10:], "Close"] = [99.5, 99.5, 100., 100., 100., 100.5, 100.5, 100.5, 100.5, 100.5]
    trailing.loc[trailing.index[-10:], "Open"] = trailing.loc[trailing.index[-10:], "Close"]
    prefix = trailing.iloc[:1].loc[trailing.index[:1].repeat(390)].copy().reset_index(drop=True)
    prices = 100. + .1 * np.sin(np.arange(390) / 5.)
    prefix["Open"] = prefix["Close"] = prices
    frame = pd.concat((prefix, trailing), ignore_index=True)
    frame["Date"] = pd.bdate_range(end="2026-10-20", periods=540).strftime("%Y-%m-%d")
    future = frame.iloc[-1:].loc[frame.index[-1:].repeat(22)].copy().reset_index(drop=True)
    future["Date"] = pd.bdate_range(start="2026-10-21", periods=22).strftime("%Y-%m-%d")
    future[["Open", "High", "Low", "Close"]] = [101.5, 102., 99., 101.5]
    if fill:
        future.loc[1, "Open"] = 100.  # Explicit invented second-open event.
    frame = pd.concat((frame, future), ignore_index=True)
    frame["Amount"] = frame["Close"] * frame["Volume"]
    frame["Bid"] = frame["Close"] - .1
    frame["Ask"] = frame["Close"]
    frame["Available At"] = frame["Date"] + "T13:30:00+08:00"
    frame["Corporate Action State"] = "NO_ACTION"
    history = source["payload"]["histories"][0]
    history["bars"] = frame.to_dict("records")
    history["bar_lineage"] = [dict(date=d, source_row_ids=["SYNTHETIC_BAR:" + d]) for d in frame["Date"]]
    sessions = list(frame["Date"])
    days = {str(d.date()): "CLOSED" for d in pd.date_range(sessions[0], sessions[-1])}
    days.update({d: "OPEN" for d in sessions})
    calendar = source["payload"]["calendars"]["TWSE"]
    calendar.update(source="TWSE_OFFICIAL", sessions=sessions, days=days,
        availability_by_date={d:"2020-01-01T00:00:00+08:00" for d in days},
        publication_events=["SYNTHETIC_NORMALIZED_CALENDAR_NOT_CERTIFIED"])
    source["payload"].update(dataset_kind="RESEARCH", purpose="SYNTHETIC_TEST_ONLY_NOT_OFFICIAL_EVIDENCE")
    source["payload"]["lineage"]["evaluation_cutoff"] = "2026-11-30T13:30:00+08:00"
    build.update(event_start="2026-10-20", event_end="2026-10-20", evaluation_cutoff="2026-11-30T13:30:00+08:00")
    build["coordinate_coverages"][0].update(start_session="2026-10-20", end_session=sessions[-1],
        available_at=sessions[-1] + "T13:30:00+08:00")
    seed = build["seed"]
    seed["context_rules"] = {context: dict(op="gt" if context == "LEADERSHIP_CONTINUATION" else "lt",
        field="close", value=0.) for context in seed["context_rules"]}
    seed["reference_policy"] = fr3_plan.policy(seed["reference_policy"], available_at=CREATED,
        version=fr3_l2.PLAN_VERSION)
    build["path_policy"] = deepcopy(seed["reference_policy"]["cost_path_policy"])
    reseal(source)
    return source, build


def l2_configs(build, training):
    build, training = deepcopy(build), deepcopy(training)
    reference = fr3_plan.policy(training["reference_policy"], available_at=CREATED,
        version=fr3_l2.PLAN_VERSION)
    training["reference_policy"] = reference
    build["seed"]["reference_policy"] = deepcopy(reference)
    build["path_policy"] = deepcopy(reference["cost_path_policy"])
    return build, training


def prepare_parent(inputs):
    return port.prepare_fr3_scope(*inputs, allow_synthetic=True, guard_factory=Guard, clock=lambda: CREATED)


def test_default_r1_identity_stays_exact_and_l2_is_explicit(r1_inputs):
    _, build, training, _ = r1_inputs
    original = deepcopy((build, training))
    r1_components = port.validate_fr3_training_configs(build, training)
    l2_build, l2_training = l2_configs(build, training)
    components = port.validate_fr3_training_configs(l2_build, l2_training)
    assert components["plan_version"] == fr3_l2.PLAN_VERSION
    assert components["path_version"] == fr3_l2.PATH_VERSION
    assert components["label_version"] == fr3_l2.LABEL_VERSION
    assert r1_components["plan_version"] == fr3_plan.PLAN_VERSION
    assert (build, training) == original
    assert components["reference_config_id"] != r1_components["reference_config_id"]
    assert l2_training["quality_policy"] == training["quality_policy"]
    assert l2_training["reference_policy"]["cost_path_policy"]["cost_policy"] == training["reference_policy"]["cost_path_policy"]["cost_policy"]


@pytest.mark.parametrize("damage", ["old_path", "old_label", "wrong_version", "seed", "folds"])
def test_mixed_plan_path_label_or_native_seed_is_rejected(r1_inputs, damage):
    _, old_build, old_training, _ = r1_inputs
    build, training = l2_configs(old_build, old_training)
    if damage == "old_path": training["reference_policy"]["cost_path_policy"] = old_build["path_policy"]
    elif damage == "old_label": training["reference_policy"]["cost_path_policy"]["label_version"] = fr3_plan.LABEL_VERSION
    elif damage == "wrong_version": training["reference_policy"]["version"] = "FR3_UNREGISTERED"
    elif damage == "seed": build["seed"]["reference_policy"] = old_build["seed"]["reference_policy"]
    else:
        training["folds"][0]["prediction_start"] = training["folds"][0]["fit_cutoff"]
    with pytest.raises(ValueError): port.validate_fr3_training_configs(build, training)


def test_freeze_l2_preserves_all_parent_settings_sources_and_clock_on_retry(r1_inputs, tmp_path):
    prepare_parent(r1_inputs)
    parent = r1_inputs[3]
    originals = {path: path.read_bytes() for path in parent.rglob("*.json")}
    output = tmp_path / "L2-settings"
    record = cli.freeze_fr3_l2_configs(parent, output, available_at=CREATED)
    assert cli.freeze_fr3_l2_configs(parent, output, available_at=CREATED) == record
    assert record["sha256"] == identity(record["payload"])
    assert record["payload"]["parent_prepared_rows_reused"] is False
    assert record["payload"]["parent_model_reused"] is False
    assert record["payload"]["symbols"] == ["1000", "1001", "1002"]
    build = json.loads((output / "build-config.json").read_text())
    training = json.loads((output / "training-config.json").read_text())
    old_build, old_training = deepcopy(r1_inputs[1]), deepcopy(r1_inputs[2])
    training.pop("reference_policy"); old_training.pop("reference_policy")
    assert training == old_training  # Entire split/feature/solver/economics contract.
    for item in (build, old_build):
        item.pop("path_policy")
        item["seed"].pop("reference_policy")
    assert build == old_build
    assert {path: path.read_bytes() for path in originals} == originals
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        cli.freeze_fr3_l2_configs(parent, output, available_at="2026-10-24T10:00:01+08:00")


@pytest.mark.parametrize("damage", ["request_hash", "native_config", "manifest_hash", "scope", "symbols"])
def test_l2_freeze_rejects_parent_metadata_drift_before_publishing(r1_inputs, tmp_path, damage):
    prepare_parent(r1_inputs)
    parent = r1_inputs[3]
    if damage == "native_config":
        path = parent / "native" / "scope-training-config.json"
        value = json.loads(path.read_text()); value["head_iterations"] += 1
    elif damage in {"manifest_hash", "scope"}:
        path = parent / "native" / "scope-training-dataset.json"
        value = json.loads(path.read_text())
        if damage == "manifest_hash": value["sha256"] = "0" * 64
        else:
            value["payload"]["scope_id"] = "0" * 64
            value["sha256"] = identity(value["payload"])
    else:
        path = parent / "fr3-training-request.json"
        value = json.loads(path.read_text())
        if damage == "request_hash": value["sha256"] = "0" * 64
        else:
            value["payload"]["symbols"].pop()
            value["sha256"] = identity(value["payload"])
    path.write_text(canonical(value))  # Explicit synthetic corruption only.
    output = tmp_path / "new-settings"
    with pytest.raises(ValueError): cli.freeze_fr3_l2_configs(parent, output, available_at=CREATED)
    assert not output.exists()


def test_public_cli_requires_explicit_aware_creation_time_and_new_directory(r1_inputs, tmp_path, capsys):
    prepare_parent(r1_inputs)
    parent = r1_inputs[3]
    args = ["freeze-l2-configs", "--parent-training-root", str(parent),
        "--output-dir", str(tmp_path / "settings")]
    with pytest.raises(SystemExit): cli.main(args)
    assert not (tmp_path / "settings").exists()
    assert cli.main(args + ["--available-at", CREATED]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == "freeze-l2-configs" and result["mail_sent"] is False
    assert result["approved_for_live"] is False
    with pytest.raises(ValueError, match="NEW_OUTPUT"):
        cli.freeze_fr3_l2_configs(parent, parent / "settings", available_at=CREATED)
    with pytest.raises(ValueError, match="TIMEZONE_REQUIRED"):
        cli.freeze_fr3_l2_configs(parent, tmp_path / "naive", available_at="2026-10-24T10:00:00")
    assert not (tmp_path / "naive").exists()


def test_l2_native_preparation_rebuilds_all_frozen_symbols_without_fitting(r1_inputs, tmp_path, monkeypatch):
    from infrastructure import les_v7_scope_training as native
    prepare_parent(r1_inputs)
    build, training = l2_configs(r1_inputs[1], r1_inputs[2])
    monkeypatch.setattr(native, "_train_prepared", lambda *_a, **_kw: pytest.fail("L2 preparation must not fit"))
    root = tmp_path / "new-L2-preparation"
    result = port.prepare_fr3_scope(r1_inputs[0], build, training, root,
        allow_synthetic=True, guard_factory=Guard, clock=lambda: CREATED)
    assert result["status"] == "DATASET_PREPARED"
    assert result["model_fitted"] is False
    request = json.loads((root / "fr3-training-request.json").read_text())["payload"]
    assert request["symbols"] == ["1000", "1001", "1002"]
    assert request["components"]["label_version"] == fr3_l2.LABEL_VERSION
    assert request["prepared_policy"].startswith("REBUILD_FROM_FROZEN_RAW")
    assert not (root / "fr3-training-completion.json").exists()
    new_manifest = json.loads((root / "native" / "scope-training-dataset.json").read_text())
    old_manifest = json.loads((r1_inputs[3] / "native" / "scope-training-dataset.json").read_text())
    assert new_manifest["sha256"] != old_manifest["sha256"]
    assert [s for child in new_manifest["payload"]["children"] for s in child["symbols"]] == request["symbols"]
    assert set(child["dataset_id"] for child in new_manifest["payload"]["children"]).isdisjoint(
        child["dataset_id"] for child in old_manifest["payload"]["children"])
    assert result == port.prepare_fr3_scope(r1_inputs[0], build, training, root,
        allow_synthetic=True, guard_factory=Guard, clock=lambda: CREATED)


@pytest.mark.parametrize("filled", [False, True])
def test_nonempty_native_builder_and_preparation_bind_same_complete_l2_label_window(filled):
    from application.les_v7_dataset import build_training_dataset
    from domain.les_v7.reference_path import path_window
    from domain.les_v7.research import label_event
    from domain.les_v7.training import prepare_events
    source, build = nonempty_native_source(fill=filled)
    original = deepcopy((source, build))
    built = build_training_dataset(source, build)
    assert built["result"]["counts"] == dict(candidate_events=1, path_events=1,
        eligible_events=1, labels={int(filled): 1})
    selection = dict(build["seed"], evaluation_cutoff=build["evaluation_cutoff"])
    rows, diagnostics = prepare_events(built["dataset"], selection)
    assert len(rows) == 1 and diagnostics == built["result"]["label_diagnostics"]
    row, event = rows[0], built["dataset"]["payload"]["events"][0]
    assert row["context"] == "LEADERSHIP_CONTINUATION"
    assert row["plan"].policy_version == fr3_l2.PLAN_VERSION
    assert row["labels"]["fill"] == int(filled)
    sessions = source["payload"]["calendars"]["TWSE"]["sessions"][-23:-2]
    history = built["dataset"]["payload"]["histories"][0]
    full_window = path_window(history, sessions)
    minimal_window = [{key:value for key,value in item.items()
        if key in {"session", "low", "complete", "coordinate_id", "available_at"}} for item in full_window]
    assert event["execution"]["raw_window_sha256"] == identity(full_window)
    assert identity(minimal_window) != identity(full_window)
    label = label_event(full_window, cutoff=event["cutoff"], expected_sessions=sessions,
        plan=row["plan"], fill=event["execution"])
    assert label["label_version"] == fr3_l2.LABEL_VERSION
    assert label["label_source_id"]
    if filled:
        assert event["execution"]["conditional_entry"]["filled_opportunity"] == 2
    else:
        assert row["labels"]["rebound"] is None and row["utility"] is None
    # Retain the complete-window invariant instead of accepting a rehashed
    # low-only projection. This is the real preview's former failure mode.
    with pytest.raises(ValueError, match="L2_LABEL_WINDOW_BINDING_REQUIRED"):
        label_event(minimal_window, cutoff=event["cutoff"], expected_sessions=sessions,
            plan=row["plan"], fill=event["execution"])
    changed = deepcopy(built["dataset"])
    altered = changed["payload"]["events"][0]
    altered["execution"]["raw_window_sha256"] = identity(minimal_window)
    altered["execution_sha256"] = identity(altered["execution"])
    changed["sha256"] = identity(changed["payload"])
    with pytest.raises(ValueError, match="EXECUTION_RECOMPUTATION_MISMATCH"):
        prepare_events(changed, selection)
    assert (source, build) == original


def test_l2_typed_plan_memo_is_bounded_detached_and_roundtrips_prepared_shards(monkeypatch):
    from domain.les_v7 import training
    from domain.les_v7.reference_memo import NativeReferenceMemo
    from infrastructure.les_v7_scope_training import _pack_row, _unpack_row
    from tests.test_v7_fr3_plan import reference, CUTOFF
    from tests.test_v7_plan_r18 import neutral_bars
    from tests.test_v7_plan_r16 import synthetic_calendar
    from tests.strategies.test_low_entry_score_v7 import bundle
    data = neutral_bars()
    config = bundle()["payload"]
    config["reference_policy"] = fr3_plan.policy(reference(), available_at=CREATED, version=fr3_l2.PLAN_VERSION)
    monkeypatch.setattr(training, "select_context", lambda _f, _r:("LEADERSHIP_CONTINUATION", ("LEADERSHIP_CONTINUATION",)))
    calendar = synthetic_calendar(data)
    with NativeReferenceMemo(max_entries=2) as memo:
        a = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a" * 64], session_evidence=calendar)
        assert isinstance(a["plan"], fr3_l2.FR3L2ReferencePlan), a
        assert a["plan"].expiry == "THIRD_OFFICIAL_SESSION_CLOSE"
        frozen = deepcopy(a)
        a["features"]["close"] = -99.
        again = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a" * 64], session_evidence=calendar)
        assert again == frozen and memo.stats["hits"] == 1
        assert memo.stats["current_entries"] <= 2
        row = dict(plan=again["plan"], features={"close":100., "atr":again["features"]["atr"]})
        assert _unpack_row(_pack_row(row)) == row
        missing = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a" * 64])
        assert missing["plan"] is None and missing["evidence_id"] != again["evidence_id"]


@pytest.mark.parametrize("version,context,expiry,valid", [
    (fr3_l2.PLAN_VERSION, "LEADERSHIP_CONTINUATION", "THIRD_OFFICIAL_SESSION_CLOSE", True),
    (fr3_l2.PLAN_VERSION, "LEADERSHIP_CONTINUATION", "NEXT_SESSION_CLOSE", False),
    (fr3_l2.PLAN_VERSION, "REVERSAL", "THIRD_OFFICIAL_SESSION_CLOSE", False),
    (fr3_plan.PLAN_VERSION, "LEADERSHIP_CONTINUATION", "THIRD_OFFICIAL_SESSION_CLOSE", False),
    ("UNREGISTERED", "LEADERSHIP_CONTINUATION", "THIRD_OFFICIAL_SESSION_CLOSE", False),
])
def test_three_session_expiry_is_exclusive_to_l2_leader(version, context, expiry, valid):
    args = dict(policy_id="REFERENCE_H20", policy_version=version, evidence_id="SYNTHETIC", context=context,
        entry=100., stop=98., target=106., atr=1., band=(99.,100.), reference_shares=1,
        cost_up=.5, cost_stop=.5, expiry=expiry)
    if valid: assert ReferencePlan(**args).expiry == expiry
    else:
        with pytest.raises(ValueError): ReferencePlan(**args)


def test_l2_frozen_bundle_must_bind_exact_plan_path_label_and_cannot_enter_daily(tmp_path, monkeypatch):
    from domain.les_v7.localization import FrozenBundle
    from tests.strategies.test_low_entry_score_v7 import bundle, seal
    from tests.test_v7_fr3_plan import reference
    import infrastructure.les_v7_research_bundle as loader
    model = bundle()
    model["payload"]["reference_policy"] = fr3_plan.policy(reference(), available_at=CREATED, version=fr3_l2.PLAN_VERSION)
    model["payload"]["trained_until"] = "2026-10-24T03:00:00Z"
    seal(model)
    assert FrozenBundle.load(model).sha256 == model["sha256"]
    wrong = deepcopy(model)
    wrong["payload"]["reference_policy"]["cost_path_policy"]["label_version"] = fr3_plan.LABEL_VERSION
    seal(wrong)
    with pytest.raises(ValueError): FrozenBundle.load(wrong)
    folder = tmp_path / ("b" * 64); folder.mkdir()
    archive = folder / "frozen_bundle.json"; archive.write_text(canonical(model))
    monkeypatch.setattr(loader.LesV7TrainingRepository, "verify", lambda _: dict(status="READY_TO_FREEZE",
        dataset_kind="RESEARCH", research_only=True, approved_for_live=False, execution_action="NO_ACTION"))
    with pytest.raises(ValueError, match="INDEPENDENT_RESEARCH_ENTRY_REQUIRED"):
        loader.load_research_bundle(archive)
