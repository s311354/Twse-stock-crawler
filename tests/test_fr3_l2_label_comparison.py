"""Synthetic census fixtures, not market samples or OOF evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from domain.les_v7.contracts import canonical, identity, timestamp
from examples import compare_fr3_l2_labels as audit


@pytest.fixture
def case(tmp_path, monkeypatch):
    roots = [tmp_path / "SYNTHETIC-R1", tmp_path / "SYNTHETIC-L2"]
    manifests, packets = {}, {}
    config = dict(fit_cutoff="2026-07-01T13:30:00+08:00", validation_start="2026-08-01T13:30:00+08:00",
        validation_end="2026-08-31T13:30:00+08:00", evaluation_cutoff="2026-10-23T13:30:00+08:00",
        folds=[dict(prediction_start="2026-01-01T13:30:00+08:00", prediction_end="2026-02-01T13:30:00+08:00")])
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(value))
    for index, root in enumerate(roots):
        rows, events = [], []
        for day in ("2026-01-05", "2026-01-06"):
            cutoff = timestamp(day + "T13:30:00+08:00").isoformat()
            rid = identity(dict(symbol="3010", market="TWSE", cutoff=cutoff))
            rows.append(dict(row_id=rid, symbol="3010", market="TWSE", context="LEADERSHIP_CONTINUATION",
                cutoff=cutoff, label_available_at="2026-02-10T13:30:00+08:00",
                labels=dict(fill=index), utility=-.2 if index else None))
            execution = dict(filled=bool(index), filled_at="2026-01-07T09:00:00+08:00")
            events.append(dict(symbol="3010", market="TWSE", cutoff=cutoff,
                execution=execution, execution_sha256=identity(execution)))
        packet = dict(rows=rows)
        dataset = dict(payload=dict(events=events))
        dataset["sha256"] = identity(dataset["payload"])
        child = dict(symbols=["3010", "3515"], prepared_path="prepared/0.json.gz",
            prepared_sha256=identity(packet), archive_path="builds/SYNTHETIC", dataset_id=dataset["sha256"])
        body = dict(universe=[["3010", "TWSE"], ["3515", "TWSE"]], freeze_id="1"*64,
            scope_id="2"*64, source_evidence_ids=["3"*64], children=[child])
        manifests[str(root / "native")] = body
        packets[str(root / "native/prepared/0.json.gz")] = packet
        save(root / "native/scope-training-config.json", config)
        save(root / "native/scope-training-dataset.json", dict(payload=body, sha256=identity(body)))
        save(root / "native/builds/SYNTHETIC/dataset.json", dataset)
    monkeypatch.setattr(audit, "verify_sharded_training_dataset", lambda _r, root, **_k: manifests[str(root)])
    monkeypatch.setattr(audit, "inspect_prepared_packet", lambda p: SimpleNamespace(value=deepcopy(packets[str(p)])))
    return roots, tmp_path / "SYNTHETIC-output", manifests


def test_census_keeps_zero_rows_and_losing_known_utility_without_calling_it_valid_oof(case):
    roots, output, _ = case
    folder, report = audit.compare_labels(*roots, output)
    assert folder.is_dir()
    assert report["fixed_universe_count"] == 2
    assert report["counts"]["R1"]["filled"] == 0
    assert report["counts"]["L2"]["filled"] == 2
    assert report["counts"]["L2"]["known_utility"] == 2
    assert report["leadership_mature_oof_utility_entry_episodes"] == 1
    assert report["leadership_episode_stock_count"] == 1
    assert report["valid_calibrated_oof_count"] is None
    assert report["forecasts_recomputed"] is False
    assert report["stocks"][1]["symbol"] == "3515" and not report["stocks"][1]["L2"]


def test_different_universe_cannot_be_presented_as_matched_comparison(case):
    roots, output, manifests = case
    manifests[str(roots[1] / "native")]["universe"].pop()
    with pytest.raises(ValueError, match="IDENTICAL_RAW_SCOPE"):
        audit.compare_labels(*roots, output)


def test_changed_splits_cannot_be_presented_as_matched_comparison(case):
    roots, output, _ = case
    path = roots[1] / "native/scope-training-config.json"
    import json
    cfg = json.loads(path.read_text())
    cfg["fit_cutoff"] = "2026-07-02T13:30:00+08:00"
    path.write_text(canonical(cfg))
    with pytest.raises(ValueError, match="IDENTICAL_SPLITS"):
        audit.compare_labels(*roots, output)
