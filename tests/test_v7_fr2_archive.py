"""SYNTHETIC archive engineering tests; no real trading or accuracy claim."""
from copy import deepcopy
import json

import pytest

from domain.les_v7.contracts import ReferencePlan, identity
from domain.les_v7 import training
from infrastructure.les_v7_fr2_archive import _detached_results, _reference_costs, _saved_head_provider
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset


def test_native_oof_archive_reuse_is_complete_result_identical_without_refit(monkeypatch):
    config = synthetic_config()
    rows, _ = training.prepare_events(synthetic_dataset(config), config)
    names, bounds = config["feature_names"], config["feature_bounds"]
    original_fit = training._fit
    saved = {}
    def fit(training_rows, options, cutoff, feature_names, feature_bounds, *, reports):
        heads = original_fit(training_rows, options, cutoff, feature_names, feature_bounds, reports=reports)
        saved[cutoff] = deepcopy(heads)
        return heads
    monkeypatch.setattr(training, "_fit", fit)
    baseline = training._oof(rows, config, names, bounds)
    monkeypatch.setattr(training, "_fit", lambda *_a, **_k: pytest.fail("archive reuse refit heads"))
    def provider(training_rows, options, cutoff, feature_names, feature_bounds, reports):
        assert options == config and feature_names == names and feature_bounds == bounds
        return deepcopy(saved[cutoff])
    assert training._oof(rows, config, names, bounds, head_provider=provider) == baseline


def head_report():
    body = dict(schema="v7-standardized-convex-head-fit/v2", fit_cutoff="2026-01-01T00:00:00Z",
        feature_count=1, converged=True, training_row_count=1,
        ordered_training_row_ids_sha256=identity(["SYNTHETIC_ROW"]),
        raw_heads=dict(SYNTHETIC_HEAD=[1]), heads_sha256=identity(dict(SYNTHETIC_HEAD=[1])),
        last_label_available_at="2025-12-31T00:00:00Z")
    return body


def call_provider(report, *, training_rows=None, bounds=None, report_id=None):
    provider = _saved_head_provider(dict(test="SYNTHETIC"), ["close"], [[0, 100]],
        {report_id or identity(report): report})
    return provider(training_rows or [dict(row_id="SYNTHETIC_ROW")], dict(test="SYNTHETIC"),
        "2026-01-01T00:00:00Z", ["close"], bounds or [[0, 100]], {})


def test_saved_heads_require_exact_content_identity_training_order_and_support():
    report = head_report()
    assert call_provider(report) == report["raw_heads"]
    with pytest.raises(ValueError, match="REQUEST_CHANGED"):
        call_provider(report, bounds=[[0, 101]])
    with pytest.raises(ValueError, match="TRAINING_BINDING"):
        call_provider(report, training_rows=[dict(row_id="DIFFERENT")])
    with pytest.raises(ValueError, match="TRAINING_BINDING"):
        call_provider(report, report_id="0" * 64)


@pytest.mark.parametrize("field,value", [("converged", False),
    ("last_label_available_at", "2026-02-01T00:00:00Z"), ("heads_sha256", "0" * 64)])
def test_future_or_unverified_archived_head_never_reused(field, value):
    report = head_report()
    report[field] = value
    with pytest.raises(ValueError, match="TRAINING_BINDING"):
        call_provider(report)


def reference_cost_fixture(tmp_path):
    cutoff = "2026-01-05T05:30:00+00:00"
    row_id = identity(dict(symbol="SYNTHETIC", market="TWSE", cutoff=cutoff))
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC", "SYNTHETIC_EVIDENCE", "REVERSAL",
        10, 9, 12, 1, (10, 10.25), 1, .05, .05)
    cost = dict(buy_rate="0.001425", sell_rate="0.004425", minimum_buy_fee="0", minimum_sell_fee="0")
    execution = dict(filled=True, plan_id=plan.plan_id, path_order_verified=True,
        path_evidence_ids=["SYNTHETIC_PATH"], path_available_at="2026-02-20T05:30:00Z",
        shares=1, entry=10, exit_price=11, realized_net_pnl=.94,
        filled_at="2026-01-06T01:00:00Z", realized_at="2026-01-07T05:30:00Z",
        exit_reason="SYNTHETIC", cost_policy_id=identity(cost),
        cost_arithmetic=dict(buy_debit="10.01", sell_proceeds="10.95", realized_net_pnl="0.94"))
    event = dict(symbol="SYNTHETIC", market="TWSE", cutoff=cutoff, execution=execution,
        execution_sha256=identity(execution))
    payload = dict(events=[event])
    packet = dict(payload=payload, sha256=identity(payload))
    folder = tmp_path / "builds" / "SYNTHETIC"
    folder.mkdir(parents=True)
    path = folder / "dataset.json"
    path.write_text(json.dumps(packet))
    dataset = dict(payload=dict(children=[dict(archive_path="builds/SYNTHETIC", dataset_id=packet["sha256"])]))
    config = dict(evaluation_cutoff="2026-03-01T00:00:00Z", reference_policy=dict(cost_path_policy=dict(cost_policy=cost)))
    rows = [dict(row_id=row_id, plan=plan, labels=dict(fill=1))]
    return dataset, config, rows, packet, path


def test_reference_cost_diagnostic_is_not_four_account_profit(tmp_path):
    dataset, config, rows, _, _ = reference_cost_fixture(tmp_path)
    report, executions = _reference_costs(tmp_path, dataset, config, rows)
    assert report["completed_reference_paths"] == 1
    assert report["realized_net_pnl_twd_sum"] == "0.94"
    assert "NOT_ACCOUNT" in report["scope"]
    assert "NOT_FOUR_ALLOCATED_ACCOUNT_PATHS" in report["warning"]
    assert len(executions) == 1


def test_reference_cost_tampering_rejected_even_if_outer_hash_resealed(tmp_path):
    dataset, config, rows, packet, path = reference_cost_fixture(tmp_path)
    execution = packet["payload"]["events"][0]["execution"]
    execution["realized_net_pnl"] = 1.94
    packet["payload"]["events"][0]["execution_sha256"] = identity(execution)
    packet["sha256"] = identity(packet["payload"])
    dataset["payload"]["children"][0]["dataset_id"] = packet["sha256"]
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="NET_PNL_ARITHMETIC_MISMATCH"):
        _reference_costs(tmp_path, dataset, config, rows)


def test_archive_outputs_are_canonical_identical_but_not_shared_with_sources():
    source = dict(dataset_id="SYNTHETIC_SOURCE")
    policy = dict(minimum_paired_events=2000)
    report = dict(source_bindings=source, acceptance_policy=policy)
    forecasts = dict(source_bindings=source, M0=[dict(value=.5)])
    left, right = _detached_results(report, forecasts)
    assert identity(left) == identity(report) and identity(right) == identity(forecasts)
    left["source_bindings"]["dataset_id"] = "CALLER_MUTATION"
    left["acceptance_policy"]["minimum_paired_events"] = 0
    right["M0"][0]["value"] = 99
    assert source == dict(dataset_id="SYNTHETIC_SOURCE")
    assert policy == dict(minimum_paired_events=2000)
    assert right["source_bindings"] == source
    assert forecasts["M0"][0]["value"] == .5
