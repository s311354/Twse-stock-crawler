"""SYNTHETIC routing only: no real fit, accuracy, account or profit claim."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import identity
from examples import compare_fr3_study as study
from tests.test_v7_fr3_plan import reference


@dataclass
class Versions:
    models: str = "SYNTHETIC_ONLY"
    parameters: str = "SYNTHETIC_ONLY"
    code: str = "SYNTHETIC_ONLY"


class Guard:
    versions = Versions()

    def retain(self, objects):
        return dict(source=objects.put(b"SYNTHETIC_ONLY native source"),
            environment=objects.put_json({"SYNTHETIC_ONLY": "environment"}),
            parameters=objects.put_json({"SYNTHETIC_ONLY": "parameters"}))

    def check(self):
        return self.versions


@pytest.fixture
def routed(tmp_path, monkeypatch):
    root = tmp_path / "new-training"
    root.mkdir()
    config = dict(reference_policy=reference(), SYNTHETIC_ONLY=True)
    request_body = dict(training_config=config)
    request = dict(payload=request_body, sha256=identity(request_body))
    (root / "fr3-training-request.json").write_text(json.dumps(request))
    symbols = ["1000", "1001", "1002"]
    outcome = dict(status="READY_TO_FREEZE", request_id=request["sha256"],
        symbols=symbols, native_root=str(root / "native"), native_outcome=dict(
            dataset_id="a"*64, config_id=identity(config), bundle_sha256="b"*64,
            bundle_path="models/"+"c"*64+"/frozen_bundle.json", reason_codes=[]))
    binding = dict(dataset_id="a"*64, config_id=identity(config), bundle_sha256="b"*64,
        native_fit_report_id="SYNTHETIC_FIT")
    left = [dict(row_id="r"+str(i), symbol=symbols[i], market="TWSE",
        cutoff="2026-09-15T05:30:00+00:00", labels=dict(fill=1-i),
        forecast=dict(SYNTHETIC_ONLY="M0")) for i in range(2)]
    right = deepcopy(left)
    for item in right:
        item["forecast"] = dict(SYNTHETIC_ONLY="M1")
    rows = [dict(row_id="r0", filled=True, realized_net_pnl="0.94"),
            dict(row_id="r1", filled=False)]
    costs = dict(completed_reference_paths=1, unfilled_reference_paths=1,
        realized_net_pnl_twd_sum="0.94", execution_rows_id=identity(rows),
        cost_path_policy=config["reference_policy"]["cost_path_policy"],
        warning="SYNTHETIC_ONLY_NOT_FOUR_ACCOUNTS")
    metrics = dict(baseline={"SYNTHETIC_ONLY": 1.}, fractal={"SYNTHETIC_ONLY": .9},
        criteria=dict(SYNTHETIC_ONLY={"status": "NOT_ASSESSABLE", "confidence_interval": None}))
    reconstruction = dict(source_bindings=binding,
        forecast_reproduction=dict(baseline_complete_metric_and_hash_equality=True,
            fractal_complete_metric_and_hash_equality=True, new_head_fits=0),
        counts=dict(paired_forecast_events=2, distinct_symbols=2, sessions=1),
        sessions=["2026-09-15"], forecast_comparisons=dict(P0=metrics, P1=deepcopy(metrics)),
        reference_path_diagnostic=costs,
        arms={"M0P0": {"entry_policy": "LEGACY_FR2_MUST_NOT_BECOME_FR3"}},
        reason_codes=["LEGACY_FR2_GAP_MUST_NOT_BECOME_FR3_DIAGNOSIS"])
    artifact = dict(source_bindings=deepcopy(binding), M0=left, M1=right, reference_cost_rows=rows)
    calls = []

    def reconstruct(path):
        calls.append(("reconstruct", path))
        return deepcopy(reconstruction), deepcopy(artifact)

    def compare(original, training_root, **options):
        calls.append(("compare", options))
        return dict(status="NOT_ASSESSABLE", universe=list(symbols), fixed_universe_count=3,
            stocks=[dict(symbol=s, native_signal="NOT_EVALUATED") for s in symbols],
            verified_decision_counts={"FR2": None, "FR3": None},
            reason_codes=["FOUR_SOURCE_BOUND_EXECUTED_AFTER_COST_ACCOUNT_PATHS_REQUIRED"])

    monkeypatch.setattr(study, "ResearchVersionGuard", Guard)
    monkeypatch.setattr(study, "verify_fr3_scope_training", lambda _: deepcopy(outcome))
    monkeypatch.setattr(study, "reconstruct_archive", reconstruct)
    monkeypatch.setattr(study, "compare_fr3_research", compare)
    audit = {"SYNTHETIC_ONLY": "source audit retains missing financial/restriction evidence"}
    args = (tmp_path / "original" / "frozen_bundle.json", root, audit, tmp_path / "study")
    return dict(args=args, config=config, outcome=outcome, report=reconstruction,
                artifact=artifact, calls=calls, symbols=symbols)


def load_study(result):
    return json.loads((result[0] / "study.json").read_text())


def test_examples_and_tests_are_excluded_from_native_source_guard():
    paths = study.source_paths()
    root = Path(study.__file__).resolve().parents[1]
    assert Path(study.__file__).resolve() not in paths
    assert not any(path.is_relative_to(root / "examples") or path.is_relative_to(root / "tests") for path in paths)


def test_runner_recomputes_both_new_models_costs_and_retains_all_fixed_symbols(routed):
    folder, receipt = study.run_fr3_study(*routed["args"])
    result = load_study((folder, receipt))
    assert [name for name, _ in routed["calls"]] == ["reconstruct", "compare"]
    bundle = routed["calls"][0][1]
    assert bundle.is_relative_to(routed["args"][1] / "native" / "models")
    options = routed["calls"][1][1]
    assert options["forecast_comparison"]["counts"]["paired_forecast_events"] == 2
    assert options["reference_costs"]["realized_net_pnl_twd_sum"] == "0.94"
    assert "old_reports" not in options and "new_reports" not in options and "after_cost_input" not in options
    assert result["fixed_universe_count"] == 3
    assert result["full_frozen_universe"] == routed["symbols"]
    assert len(result["comparison"]["stocks"]) == 3
    assert result["comparison"]["verified_decision_counts"] == {"FR2": None, "FR3": None}
    assert result["four_account_cost_comparison_status"] == result["status"] == "NOT_ASSESSABLE"
    assert result["fresh_confirmatory"] is False and result["mail_sent"] is False
    assert result["holdout_interpretation"] == study.HOLDOUT_POSTURE
    assert "NOT_ALLOCATION_OR_ACCOUNT" in result["reference_path_cost_diagnostic"]["scope"]
    assert result["matched_forecast_recomputation"]["metrics"] == routed["report"]["forecast_comparisons"]["P0"]
    assert result["matched_forecast_recomputation"]["complete_native_reproduction"]["new_head_fits"] == 0
    forecasts = json.loads((folder / "recomputed-forecasts.json").read_text())
    assert forecasts["label_version"] == study.LABEL_VERSION
    assert forecasts["M0"] == routed["artifact"]["M0"]
    assert forecasts["M1"] == routed["artifact"]["M1"]
    assert "arms" not in forecasts and "arms" not in result["matched_forecast_recomputation"]
    assert "LEGACY_FR2_GAP" not in json.dumps(result)
    assert receipt["study_id"] == identity({k: v for k, v in result.items() if k != "study_id"})
    assert receipt["runner_sha256"] == hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()
    runtime = result["runtime"]
    objects = study.ResearchObjects(routed["args"][3] / "objects")
    assert objects.get(runtime["runner"]["object_id"]) == Path(study.__file__).read_bytes()
    assert objects.get_json(runtime["objects"]["recomputed_forecasts"]) == forecasts


def test_not_ready_keeps_actual_fit_gap_and_never_uses_original_model(routed):
    routed["outcome"]["status"] = "NOT_READY"
    routed["outcome"]["native_outcome"].update(bundle_path=None, bundle_sha256=None,
        reason_codes=["SYNTHETIC_NATIVE_INSUFFICIENT_POSITIVE_LABELS"])
    result = load_study(study.run_fr3_study(*routed["args"]))
    assert [name for name, _ in routed["calls"]] == ["compare"]
    assert result["status"] == "NOT_READY"
    assert result["matched_forecast_recomputation"] is None
    assert result["reference_path_cost_diagnostic"] is None
    assert result["native_training_reason_codes"] == ["SYNTHETIC_NATIVE_INSUFFICIENT_POSITIVE_LABELS"]
    assert routed["calls"][0][1]["forecast_comparison"] is None
    assert result["full_frozen_universe"] == routed["symbols"]


@pytest.mark.parametrize("damage", ["baseline_equality", "full_equality", "head_refit", "labels",
    "binding", "metrics", "cost_policy", "cost_rows", "cost_sum"])
def test_incomplete_or_mismatched_recomputation_cannot_publish_completion(routed, damage):
    report, artifact = routed["report"], routed["artifact"]
    if damage == "baseline_equality": report["forecast_reproduction"]["baseline_complete_metric_and_hash_equality"] = False
    elif damage == "full_equality": report["forecast_reproduction"]["fractal_complete_metric_and_hash_equality"] = False
    elif damage == "head_refit": report["forecast_reproduction"]["new_head_fits"] = 1
    elif damage == "labels": artifact["M1"][0]["labels"]["fill"] = 0
    elif damage == "binding": artifact["source_bindings"]["bundle_sha256"] = "0"*64
    elif damage == "metrics": report["forecast_comparisons"]["P1"]["baseline"]["SYNTHETIC_ONLY"] = 2.
    elif damage == "cost_policy": report["reference_path_diagnostic"]["cost_path_policy"] = {"version": "OLD_FR2"}
    elif damage == "cost_rows": artifact["reference_cost_rows"].pop()
    else: report["reference_path_diagnostic"]["realized_net_pnl_twd_sum"] = "10.94"
    with pytest.raises(ValueError, match="FR3_STUDY_"):
        study.run_fr3_study(*routed["args"])
    assert not list(routed["args"][3].glob("*/completion.json"))


def test_ready_requires_new_model_and_path_cannot_escape_native_root(routed):
    routed["outcome"]["native_outcome"]["bundle_path"] = "../../original/frozen_bundle.json"
    with pytest.raises(ValueError, match="NATIVE_BUNDLE_PATH_INVALID"):
        study.run_fr3_study(*routed["args"])
    assert not routed["calls"]


def test_cannot_report_full_scope_after_comparator_drops_a_stock(routed, monkeypatch):
    monkeypatch.setattr(study, "compare_fr3_research", lambda *_a, **_k:
        dict(status="NOT_ASSESSABLE", universe=["1000"], fixed_universe_count=1))
    with pytest.raises(ValueError, match="FULL_FROZEN_SCOPE"):
        study.run_fr3_study(*routed["args"])


def test_missing_account_paths_cannot_be_relabelled_pass(routed, monkeypatch):
    monkeypatch.setattr(study, "compare_fr3_research", lambda *_a, **_k:
        dict(status="PASS", universe=routed["symbols"], fixed_universe_count=3))
    with pytest.raises(ValueError, match="MISSING_FOUR_ACCOUNT_PATHS_CANNOT_PASS"):
        study.run_fr3_study(*routed["args"])


def test_rerun_reexecutes_predictions_but_publishes_identical_write_once_results(routed):
    first = study.run_fr3_study(*routed["args"])
    before = {path.name: path.read_bytes() for path in first[0].iterdir()}
    assert study.run_fr3_study(*routed["args"]) == first
    assert {path.name: path.read_bytes() for path in first[0].iterdir()} == before
    assert [name for name, _ in routed["calls"]].count("reconstruct") == 2


def test_interruption_before_completion_is_not_complete_and_retry_recovers(routed, monkeypatch):
    original = study.write_once
    crash = [True]
    def publish(path, value):
        if path.name == "completion.json" and crash[0]:
            crash[0] = False
            raise OSError("SYNTHETIC_COMPLETION_INTERRUPT")
        return original(path, value)
    monkeypatch.setattr(study, "write_once", publish)
    with pytest.raises(OSError, match="COMPLETION_INTERRUPT"):
        study.run_fr3_study(*routed["args"])
    assert not list(routed["args"][3].glob("*/completion.json"))
    folder, completion = study.run_fr3_study(*routed["args"])
    assert (folder / "completion.json").is_file() and completion["status"] == "NOT_ASSESSABLE"


def test_original_and_training_archives_cannot_be_output_targets(routed):
    old, root, audit, _ = routed["args"]
    for output in (old.parent, old.parent / "child", root, root / "child"):
        with pytest.raises(ValueError, match="SEPARATE_OUTPUT_REQUIRED"):
            study.run_fr3_study(old, root, audit, output)
    assert not routed["calls"]


def test_cli_retains_not_assessable_and_never_claims_four_account_success(tmp_path, monkeypatch, capsys):
    calls = []
    def run(*args):
        calls.append(args)
        return tmp_path / "retained", dict(status="NOT_ASSESSABLE", study_id="a"*64,
            research_only=True, approved_for_live=False, mail_sent=False)
    monkeypatch.setattr(study, "run_fr3_study", run)
    code = study.main(["--archive", str(tmp_path / "parent.json"),
        "--training-root", str(tmp_path / "training"), "--source-audit", str(tmp_path / "audit.json"),
        "--output-dir", str(tmp_path / "study")])
    assert len(calls) == 1 and code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "NOT_ASSESSABLE" and summary["mail_sent"] is False
