"""SYNTHETIC_ONLY archive format and optional predictor composition checks."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from domain.les_v7.contracts import ReferencePlan, canonical, identity
from domain.les_v7.execution_features import FEATURE_NAMES, FEATURE_BOUNDS, profile
from domain.les_v7.fr2_policy import apply_fr2
from domain.les_v7.report import validate_report
from infrastructure import les_v7_fr2_archive as archive_reader
from infrastructure import les_v7_fr2_m0 as m0_reader
from infrastructure.les_v7_scope_training import LEGACY_PREPARED_SCHEMA, _prepared_packet, _pack_row
from infrastructure.storage.les_v7_prepared import write_prepared_packet
from tests.test_v7_fr2_policy import request, native
from tests.strategies.test_low_entry_score_v7 import bars, seal


def prepared_case(tmp_path, *, columnar, compressed=False):
    """Only a serialization fixture; never claimed as a verified native archive."""
    config = dict(schema="SYNTHETIC_ONLY")
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "SYNTHETIC_EVIDENCE", "REVERSAL",
        10., 9., 12., 1., (10., 10.25), 1, .01, .01)
    row = dict(row_id="SYNTHETIC_ONLY", symbol="SYNTHETIC_ONLY", market="TWSE",
        cutoff="2026-01-05T05:30:00+00:00", features=dict(close=10., dfa512=.5), plan=plan)
    packet = _prepared_packet("SYNTHETIC_DATASET", identity(config), [row], [])
    if not columnar:
        packet = dict(schema=LEGACY_PREPARED_SCHEMA, dataset_id="SYNTHETIC_DATASET", config_id=identity(config),
            rows=[_pack_row(row)], diagnostics=[])
    root = tmp_path / "scope"; folder = root / "models" / "SYNTHETIC_ONLY"
    folder.mkdir(parents=True); (root / "prepared").mkdir()
    relative = "prepared/0.json.gz" if compressed else "prepared/0.json"
    write_prepared_packet(root / relative, packet)
    child = dict(prepared_path=relative, prepared_sha256=identity(packet), dataset_id=packet["dataset_id"])
    dataset = dict(payload=dict(schema="les-v7-sharded-training-dataset/v1", children=[child]))
    return root, folder, config, dataset, packet, row


@pytest.mark.parametrize("columnar", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_m0_reader_decodes_legacy_and_columnar_prepared_rows(tmp_path, columnar, compressed):
    _, folder, config, dataset, _, row = prepared_case(tmp_path, columnar=columnar, compressed=compressed)
    assert m0_reader._prepared(folder, config, dataset) == [row]


def test_m0_reader_preserves_separate_decoded_and_row_bounds(tmp_path, monkeypatch):
    _, folder, config, dataset, _, _ = prepared_case(tmp_path, columnar=True, compressed=True)
    monkeypatch.setattr(m0_reader, 'MAX_PREPARED_DECODED_BYTES', 1)
    with pytest.raises(ValueError, match='PREPARED_BYTE_BUDGET'):
        m0_reader._prepared(folder, config, dataset)
    monkeypatch.setattr(m0_reader, 'MAX_PREPARED_DECODED_BYTES', 256 * 1024 * 1024)
    monkeypatch.setattr(m0_reader, 'MAX_PREPARED_ROWS', 0)
    with pytest.raises(ValueError, match='PREPARED_ROW_BUDGET'):
        m0_reader._prepared(folder, config, dataset)


@pytest.mark.parametrize("columnar", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_archive_reader_reaches_next_verified_stage_after_decoding_both_formats(tmp_path, monkeypatch, columnar, compressed):
    root, folder, config, dataset, packet, _ = prepared_case(tmp_path, columnar=columnar, compressed=compressed)
    bundle = dict(payload=dict(SYNTHETIC_ONLY=True)); bundle["sha256"] = identity(bundle["payload"])
    class NextStageReached(Exception): pass
    def read(path):
        if path.name == "fit_result.json": raise NextStageReached
        values = {"frozen_bundle.json": bundle, "config.json": config, "dataset.json": dataset, "0.json": packet}
        return values[path.name]
    # Gate stubs isolate the serialization boundary. This test does not bypass
    # any production verification or claim the fixture is a publishable model.
    monkeypatch.setattr(archive_reader.LesV7TrainingRepository, "verify", lambda *_: dict(
        dataset_kind="RESEARCH", status="READY_TO_FREEZE", bundle_sha256=bundle["sha256"]))
    monkeypatch.setattr(archive_reader, "_read", read)
    with pytest.raises(NextStageReached): archive_reader.reconstruct_archive(folder / "frozen_bundle.json")


def extended_request():
    item = request(); model = item["context"]["model_bundle"]; payload = model["payload"]
    selected = list(FEATURE_NAMES[:13]); definition = profile()
    payload.update(execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=["close"], profile_id=identity(definition), selected_features=selected),
        feature_names=["close"] + selected, feature_bounds=[[1., 200.]] + [list(x) for x in FEATURE_BOUNDS[:13]],
        imputation=[None] * 14, supported_masks=[[True] * 14])
    for name, coefficients in payload["heads"].items():
        array = np.asarray(coefficients)
        payload["heads"][name] = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, 13)]).tolist()
    payload["heads"].update(fill_only_base_count=1)
    payload["heads"]["fill"][-1] = 2.
    seal(model)
    data = bars(600); data["PE"] = 12.
    rows = data.astype(object).where(data.notna(), None).to_dict("records")
    for row in rows: row["Date"] = str(row["Date"])[:10]
    item["bars"] = rows
    source = item["context"]["native_evidence_source"]
    source["market_rows"] = rows
    source["market_evidence"].update(bars_sha256=identity(rows), source_record_ids=[identity(rows)])
    return item


def test_fr2_recomputes_optional_predictor_with_native_execution_inputs_but_keeps_base_gates():
    item = extended_request(); original = native(item); before = canonical(original)
    assert original["diagnostics"]["localization"]["status"] == "VALID"
    result = apply_fr2(original, item["context"]["model_bundle"])
    validate_report(result)
    assert canonical(original) == before
    for key in ("features", "reference_plan", "gate_bindings", "execution_features", "localization"):
        assert result["diagnostics"][key] == original["diagnostics"][key]
    damaged = deepcopy(original)
    damaged["diagnostics"]["execution_features"]["execution_gap_mean5"] += .1
    with pytest.raises(ValueError, match="EXECUTION_FEATURE_RECOMPUTATION"):
        apply_fr2(damaged, item["context"]["model_bundle"])
