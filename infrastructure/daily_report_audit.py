"""Read-only report intake and immutable, content-addressed audit publication."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from application.daily_report_audit import SCHEMA, audit_daily_report
from application.replay_contracts import canonical_json, digest
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.research_objects import ResearchObjects
from infrastructure.storage.report_evidence import ReportEvidenceReader


def _cell(value: object) -> str:
    return str(value if value is not None else "N/A").replace("|", "\\|").replace("\n", " ")


def render_audit(audit: dict) -> str:
    summary = audit["summary"]
    lines = ["# ELEC 原報告全名單稽核", "", f"Baseline ID: `{audit['baseline_id']}`",
        f"稽核 {summary['audited_symbols']} 檔；JSON 有效區 {summary['report_buckets']['reports']} 檔，"
        f"stale_reports 區 {summary['report_buckets']['stale_reports']} 檔。兩區皆納入，不把 stale 全視為價格過期。",
        "", "這是已保存輸出的稽核，不是模型重算，也不是歷史證據已完整的資料集。",
        "", "## 主要結果", "",
        f"- LES 內嵌 JSON 與映射欄位一致性：{summary['export_parity']}；不代表當日／重播路徑一致。",
        "- EREM／PAE 只保留 CSV 輸出，沒有另一份完整輸入／輸出供獨立一致性驗證。",
        f"- 原始 LES：{summary['original_les_decisions']}",
        f"- 模型：{summary['model_versions']}；參數：{summary['parameter_versions']}",
        f"- 行情日期：{summary['signal_dates']}；決策日期（臺北）：{summary['decision_dates_taipei']}",
        "- 同輸入重播、歷史重建差異比較、發布研究資料集及成交模擬均未執行。",
        "- 帳戶標籤及 indicative_shares 不是完整帳戶快照；不得據此反推現金、淨值或成交。",
        "- stocklist 若為 legacy-indices，只核對數量，不宣稱已還原當次索引對應。",
        "", "## 原報告進場樣本", "",
        "| 股票 | 行情日 | LES | 分數 | QS | EREM | PAE | 配置阻擋 |",
        "|---|---|---|---:|---:|---|---|---|"]
    for row in audit["rows"]:
        if row["symbol"] in audit["entry_anchor_symbols"]:
            fields = [row["symbol"] + " " + (row["name"] or ""), row["signal_date"],
                row["original_les"].get("signal"), row["original_les"].get("score"),
                row["original_quality"].get("score"), row["original_erem"].get("EREM 決策訊號"),
                row["original_pae"].get("PAE Decision"), row["original_pae"].get("PAE Blocking Reasons")]
            lines.append("| " + " | ".join(map(_cell, fields)) + " |")
    lines += ["", "## 缺口分類", "", "| 診斷 | 股票數 |", "|---|---:|"]
    for key, count in summary["issue_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "`latest_available_at` 晚於行情日只表示報告所載時間跨界；"
        "不能推論每份財報均晚公布，也不能把它當作已驗證公布時間。",
        "", "### 財報欄位與期間", "", "| 原報告缺口 | 股票數 |", "|---|---:|"]
    for key, count in (summary["missing_period_counts"] | summary["unresolved_field_counts"]).items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "公布證明／選版無原始完整快照可供獨立驗證。上述為原報告診斷，"
        "缺季度、EPS 單季／累計問題不可用補公布日期代替。空缺口清單也不等於已驗證完整。",
        "", "## 同輸入重播尚缺", ""]
    lines += [f"- `{key}`: {value}" for key, value in audit["missing_original_inputs"].items()]
    lines += ["", "## 全名單", "", "| 股票 | 行情日 | 原始 LES | Quality | 匯出一致 | 缺季度 | 欄位待核實 |",
        "|---|---|---|---|---|---|---|"]
    for row in audit["rows"]:
        fields = [row["symbol"] + " " + (row["name"] or ""), row["signal_date"],
            row["original_les"].get("signal"), row["original_quality"].get("status"),
            row["export_parity"]["status"], ", ".join(row["financial_verification"].get("missing_periods", [])),
            ", ".join(row["financial_verification"].get("unresolved_fields", []))]
        lines.append("| " + " | ".join(map(_cell, fields)) + " |")
    lines += ["", "## 下一步", "",
        "1. 下次當日計算須同時保存完整輸入、固定研究時鐘、帳戶、模型／參數／程式及 PAE 前序快照，再做同輸入路徑比較。",
        "2. 對本次全名單另建歷史重建資料，逐版核實財報公布鏈、可比較性及歷史限制；不可用今天 cache 偽裝本次輸入。",
        "3. 歷史證明不足的假設情境另立政策與資料集，明示假設；有效補證才發布修復 ID。",
        "4. BankEmulator 只能用完整研究資料集建立獨立 run；此 baseline_id 不是 evidence dataset ID。",
        "", "## 人工確認", "",
        "- 原 CSV、JSON、stocklist 與 objects 雜湊一致，原檔未改。",
        "- reports 與 stale_reports 的全部代號皆出現在 audit.json／audit.csv。",
        "- 原始 BUY_SMALL 與 PAE NO_ACTION、研究資格及成交分开。",
        "- 缺少輸入保持 NOT_RETAINED／NOT_EVALUATED，不改寫為 PASS。", ""]
    return "\n".join(lines)


def retain_daily_report_audit(analysis_csv: Path, reports: Path, stocklist: Path,
                              output: Path, *, stocklist_format: str) -> Path:
    sources = {"analysis_csv": Path(analysis_csv), "reports": Path(reports), "stocklist": Path(stocklist)}
    content = {key: path.read_bytes() for key, path in sources.items()}
    reader = csv.DictReader(io.StringIO(content["analysis_csv"].decode("utf-8-sig")))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV_HEADERS_MISSING_OR_DUPLICATE")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("CSV_ROW_WIDTH_MISMATCH")
    envelope = json.loads(content["reports"])
    with ReportEvidenceReader(Path(reports).parent) as evidence:
        for bucket in ("reports", "stale_reports"):
            envelope[bucket] = [evidence.resolve(r) for r in envelope.get(bucket, [])]
        for row in rows:
            if row.get("LES v6 Report"):
                row["LES v6 Report"] = canonical_json(evidence.resolve(json.loads(row["LES v6 Report"])))
    # Retain the resolved native JSON too: the audit remains portable even if
    # the live append-only object store later moves. The original exports remain
    # unchanged and have their own intake hashes.
    if json.loads(content["reports"]).get("storage_schema") == "daily-reports/v2":
        content["resolved_reports"] = canonical_json(envelope).encode()
    audit = audit_daily_report(rows, envelope, content["stocklist"].decode("utf-8-sig"),
                               stocklist_format=stocklist_format)
    refs = {key: dict(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)) for key, body in content.items()}
    baseline_id = digest(dict(schema=SCHEMA, sources=refs, stocklist_format=stocklist_format))
    folder = Path(output) / baseline_id
    objects = ResearchObjects(Path(output) / "objects")
    # Fail before publication if a live writer replaced any intake artifact.
    if any(path.read_bytes() != content[key] for key, path in sources.items()):
        raise ValueError("DAILY_REPORT_SOURCE_CHANGED_DURING_AUDIT")
    for key, body in content.items():
        ref = objects.put(body)
        objects.get(ref)
        refs[key]["path"] = f"../objects/{ref}"
    audit.update(baseline_id=baseline_id, source_objects=refs)
    write_once(folder / "audit.json", audit)
    flat = io.StringIO(newline="")
    columns = ["symbol", "name", "signal_date", "decision_time", "les", "quality", "erem", "pae",
               "export_parity", "missing_periods", "unresolved_fields", "issues"]
    writer = csv.DictWriter(flat, fieldnames=columns)
    writer.writeheader()
    for row in audit["rows"]:
        writer.writerow({**{k: row[k] for k in columns[:4]},
            "les": row["original_les"].get("signal"), "quality": row["original_quality"].get("status"),
            "erem": row["original_erem"].get("EREM 決策訊號"), "pae": row["original_pae"].get("PAE Decision"),
            "export_parity": row["export_parity"]["status"],
            "missing_periods": canonical_json(row["financial_verification"].get("missing_periods", [])),
            "unresolved_fields": canonical_json(row["financial_verification"].get("unresolved_fields", [])),
            "issues": canonical_json(row["issues"])})
    rendered = {"audit.csv": flat.getvalue().encode("utf-8-sig"), "audit.md": render_audit(audit).encode()}
    for name, body in rendered.items():
        write_bytes_once(folder / name, body)
    write_once(folder / "receipt.json", dict(schema=SCHEMA, baseline_id=baseline_id,
        audit_sha256=digest(audit), source_objects=refs,
        rendered_sha256={name: hashlib.sha256(body).hexdigest() for name, body in rendered.items()},
        status="COMPLETE_REPORT_AUDIT_NOT_REPLAY"))
    return folder
