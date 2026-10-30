"""Manual FR2 four-arm comparison or read-only native holdout reconstruction."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr2_comparison import compare
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_training import MAX_DATASET_BYTES, read_json_bounded


def summary(report):
    lines = ["# FR2 研究比較", "", "結論：" + report["status"], "",
             "四組比較不以 BUY 數量或事後獲利個股作驗收。原 bundle 與正式每日政策未變更。", "",
             "| 組別 | 模型 | 進場政策 | 研究／成本後狀態 |", "| --- | --- | --- | --- |"]
    for name, arm in report.get("arms", {}).items():
        lines.append("| " + " | ".join((name, arm.get("model_kind", arm.get("model_id", "")),
            arm.get("entry_policy", arm.get("entry_policy_id", "")), arm["status"])) + " |")
    lines += ["", "## 資料與限制", ""]
    lines += ["- " + reason for reason in report.get("reason_codes", [])]
    for name, comparison in report.get("forecast_comparisons", {}).items():
        lines += ["", "## " + name + "：同條件預測比較", "",
            "| 指標（越低越好） | 無碎形 | 碎形 | 相對改善 | 驗收 |", "| --- | ---: | ---: | ---: | --- |"]
        for metric, c in comparison["criteria"].items():
            def number(value):
                return "N/A" if value is None else f"{value:.6f}"
            lines.append(f"| {metric} | {number(c.get('baseline'))} | {number(c.get('fractal'))} | {number(c.get('relative_improvement'))} | {c['status']} |")
    lines += ["", "研究用途；完整性雜湊不等於歷史來源認證，不代表可部署或保證低點。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Sealed four-arm comparison input; all accounts explicit")
    inputs.add_argument("--archive", type=Path, help="Preserved native frozen_bundle.json; reconstruct development holdout only")
    parser.add_argument("--output-dir", required=True, type=Path, help="New immutable comparison artifacts; never model archive")
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        if args.archive:
            original = args.archive.expanduser().resolve().parent
            if output == original or output.is_relative_to(original):
                raise ValueError("FR2_OUTPUT_MUST_NOT_MODIFY_ORIGINAL_ARCHIVE")
        guard = ResearchVersionGuard()
        objects = ResearchObjects(output / "objects")
        pinned = guard.retain(objects)
        forecasts = None
        if args.archive:
            from infrastructure.les_v7_fr2_archive import reconstruct_archive
            report, forecasts = reconstruct_archive(args.archive)
        else:
            report = compare(read_json_bounded(args.input, MAX_DATASET_BYTES))
        guard.check()
        report = dict(report, calculation_versions=asdict(guard.versions),
            calculation_objects=pinned, calculation_environment_id=identity(guard.environment),
            reproduction_scope="REEXECUTED_NATIVE_CURRENT_CODE_NOT_HISTORICAL_DEPLOYMENT")
        report_id = identity(report)
        folder = args.output_dir.expanduser().resolve() / report_id
        if forecasts is not None:
            write_once(folder / "forecasts.json", dict(payload=forecasts, sha256=identity(forecasts)))
        write_once(folder / "comparison.json", dict(payload=report, sha256=report_id))
        write_bytes_once(folder / "comparison_support_note.md", summary(report).encode())
        receipt = dict(schema="les-v7-fr2-comparison-completion/v1", report_id=report_id,
            status=report["status"], forecasts_id=identity(forecasts) if forecasts else None,
            research_only=True, approved_for_live=False, execution_action="NO_ACTION", order_authority=False)
        write_once(folder / "completion.json", receipt)
        print(canonical(dict(receipt, output=str(folder))))
        return 0 if report["status"] == "PASS" else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
