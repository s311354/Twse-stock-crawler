"""Readable daily v7 research signals, distinct from PAE allocation and orders."""
from __future__ import annotations

import html
import re
from domain.les_v7.fractals import is_active_fractal_feature
from application.v7_report_input import read_v7_report_json
from infrastructure.report.contracts import HtmlRenderer
from infrastructure.storage.valuation_source_verification import (
    valuation_history_verifier_scope, verify_report_valuation_sources,
)


def text(value): return html.escape("N/A" if value is None else str(value))


def number(value): return "N/A" if value is None else f"{float(value):,.2f}"


def report_mode(report):
    """Label the actual report path, never infer deployment from a filename."""
    if report["diagnostics"].get("rule_baseline") is not None: return "RULE_BASELINE"
    if (report["diagnostics"].get("run_scope") or {}).get("diagnostics_only"): return "DIAGNOSTIC"
    if report["envelope"]["model"].get("bundle_sha256"):
        profile = report["diagnostics"].get("model_input_profile")
        if profile is None:
            # Old FR2 reports retain the actual, hash-validated predictor.
            profile = (report["diagnostics"].get("fr2") or {}).get("predictor_bundle", {}).get("payload", {})
        if any(is_active_fractal_feature(name) for name in profile.get("feature_names", ())):
            return "FRACTAL_MODEL"
        return "MODEL"
    return "NO_MODEL"


MODE_LABELS = {"RULE_BASELINE": "V7 模型", "FRACTAL_MODEL": "V7 碎形模型",
               "MODEL": "V7 已載入模型", "DIAGNOSTIC": "V7 診斷模式", "NO_MODEL": "V7 未載入模型"}
MODE_NOTES = {"RULE_BASELINE": "使用 v6 型態規則與固定配分 · 門檻依各報告政策版本",
              "FRACTAL_MODEL": "使用研究 bundle 的碎形特徵、五預測頭、校準與凍結權重 · 僅供研究",
              "MODEL": "已載入模型，保留原生 v7 預測與計分路徑",
              "DIAGNOSTIC": "僅供診斷，不產生正式進場資格",
              "NO_MODEL": "未提供模型，不宣稱已有預測或校準"}


def model_label(modes):
    return MODE_LABELS[next(iter(modes))] if len(modes) == 1 else "V7 混合模式" if modes else None


def rendered_model_label(markup):
    """Read presentation metadata only; never use this as model/order authority.

    The email subject reuses the renderer's validated mode without parsing the
    large financial reports again or inferring it from a command/filename.
    """
    labels = {html.unescape(value) for value in re.findall(r'data-model-label="([^"]+)"', markup)}
    allowed = set(MODE_LABELS.values()) | {"V7 混合模式"}
    return next(iter(labels)) if len(labels) == 1 and labels <= allowed else None


def mode_badge(mode):
    return (f'<span data-report-mode="{mode}" style="display:inline-block;padding:4px 10px;'
            f'border:1px solid #78848d;border-radius:6px;font-size:13px;line-height:1.5;'
            f'font-weight:600;white-space:normal">{text(MODE_LABELS[mode])}</span>')


def score_text(score, *, legacy_caps=False, show_arithmetic=False):
    detail = f'技術原始分 {number(score["raw_score"])} − 扣分 {number(score["total_penalty"])}'
    if (show_arithmetic and score["effective_cap"] is None and score["raw_score"] is not None
            and score["raw_score"] < score["total_penalty"]):
        detail += f'<br>扣分後 {number(score["raw_score"] - score["total_penalty"])} · 分數下限 0'
    if score["effective_cap"] is None:
        return detail + f'<br><strong>Final {number(score["final_score"])}</strong>'
    label = "舊版封頂" if legacy_caps else "封頂"
    return detail + f'<br>{label} {number(score["effective_cap"])} → Final {number(score["final_score"])}'


def valuation_note(valuation):
    """A price-band PASS never relabels a native bear-warning FAIL."""
    if valuation.get("bear_warning"):
        label = "價格條件通過" if valuation["band"] == "PASS" else f'價格 {text(valuation["band"] or valuation["state"])}'
        return (f'{label} · 扣分 {number(valuation["points"])}<br>'
            f'情境警示：僅 BUY_SMALL · 部位倍率 ×{float(valuation["multiplier"]):g}')
    # Preserve the retained R9--R12 presentation byte-for-byte.
    return (f'估值 {text(valuation["band"] or valuation["state"])} · 扣分 {number(valuation["points"])}'
        f'<br>部位倍率 ×{number(valuation["multiplier"])}')


def context_label(report):
    plan = report["diagnostics"].get("reference_plan") or {}
    if plan.get("setup_variant") == "SUPPORT_RESISTANCE_LIMIT_R18":
        return "支撐／阻力條件限價（非三型態）"
    observation = (report["diagnostics"].get("rule_baseline") or {}).get("score_evidence", {}).get("scoring_interpretation", {})
    if observation.get("interpretation") == "LOW_ZONE_OBSERVATION_NOT_ENTRY_SETUP":
        return report["envelope"]["event"]["context"] + " · 低位觀察（非進場型態）"
    return report["envelope"]["event"]["context"]


def report_for(decision, *, source_verifier=None):
    if decision.value("LES Model Version") != "7.0.0": raise ValueError("V7_RENDERER_REQUIRES_V7")
    report = read_v7_report_json(decision.value("LES v7 Report"))
    verify_report_valuation_sources(report, verifier=source_verifier)
    if (report["symbol"] != decision.value("證券代號")
            or report["envelope"]["strategy"]["strategy_decision"] != decision.value("低點決策")
            or report["envelope"]["score"]["final_score"] != decision.value("低點分數")):
        raise ValueError("V7_RENDERER_PROJECTION_MISMATCH")
    return report


def table(headers, rows):
    head = "".join(f"<th>{text(h)}</th>" for h in headers)
    # One content container keeps the existing mobile label/value grid stable
    # even when a value has multiple lines or a highlighted Final score.
    body = "".join("<tr>" + "".join(f'<td data-label="{text(h)}"><div>{v}</div></td>' for h, v in zip(headers, row)) + "</tr>" for row in rows)
    return f'<div style="overflow-x:auto"><table class="responsive-decision-table" border="1" cellpadding="8"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _decision_label(fr2_modes):
    if fr2_modes == {True}: return "FR2 研究決策"
    if fr2_modes == {True, False}: return "LES 決策（配置前）"
    return "v7 原始決策"


def architecture_comparison(*, modes, models, policies, components):
    """Explain paths from validated report metadata, not command/file names.

    This is presentation only: thresholds describe the policy that ran and do
    not confer model readiness, forecast superiority or execution permission.
    """
    if not modes:
        return ""
    trained, baseline = bool(modes & {"MODEL", "FRACTAL_MODEL"}), "RULE_BASELINE" in modes
    fr2 = any(record.get("entry", {}).get("version") == "FR_ENTRY_R2" for record in components)
    eps_price = any(record.get("valuation") for record in components)
    learned_current = "本次採用 · FR2" if trained and fr2 else "本次採用 · 原模型政策" if trained else "對照說明（非本次）"
    daily_current = "本次採用" if baseline else "對照說明（非本次）"
    learned_policy = ("FR_ENTRY_R2 · BUY ≥75／BUY_SMALL ≥65；已核實估值警訊扣5分，僅小倉"
        if fr2 else "原 bundle 進場政策 · BUY ≥80／BUY_SMALL ≥72；不套用每日規則門檻")
    daily_policy = "截止當時部署的規則政策；不套用模型預測／校準"
    if len(policies) == 1:
        version, buy, small = next(iter(policies))
        daily_policy = f'{text(version)} · BUY ≥{float(buy):g}／BUY_SMALL ≥{float(small):g}'
    elif len(policies) > 1:
        daily_policy = "混合規則版本；各檔保留原有政策與門檻"
    rows = [
        ["目前報告", f'<strong>{learned_current}</strong>', f'<strong>{daily_current}</strong>'],
        ["預測與計分", "已凍結特徵、五預測頭與校準；六因子採模型凍結權重",
         "v6 型態規則與固定六因子配分；未使用機率預測模型"],
        ["進場政策", learned_policy, daily_policy],
        ["計畫與執行", "沿用凍結計畫事件；完整來源／風險驗證，價格不符合不成交",
         "規則式因果計畫；條件限價不是成交，仍須完整來源／風險驗證"],
        ["使用範圍", "研究 bundle 僅供研究、不可下單；不自動取代正式每日流程",
         "一般每日流程；PAE 候選授權不等於配置、成交或送單"],
    ]
    if eps_price:
        rows.insert(3, ["研究估值", "核實年度 EPS 價格估值；僅價格報酬／不含股利，不推估缺失股數",
                        "依當日規則政策；不套用研究 bundle 的準入政策"])
    identities = []
    for sha in sorted(models):
        identities.append(f'<span title="{text(sha)}">bundle {text(sha[:12])}</span>')
    seen = set()
    for record in components:
        for name, label in (("prediction", "預測"), ("entry", "進場"), ("valuation", "估值"), ("execution", "執行")):
            component = record.get(name, {})
            version, component_id = component.get("version"), component.get("component_id")
            if not version or not component_id or (name, version, component_id) in seen:
                continue
            seen.add((name, version, component_id))
            identities.append(f'<span title="{text(component_id)}">{label} {text(version)} · {text(component_id[:12])}</span>')
    identity_line = ('<p data-section="research-component-identities" style="font-size:12px;line-height:1.6;overflow-wrap:anywhere">'
        + '<br>'.join(identities) + '</p>') if identities else ""
    return ('<section data-section="model-architecture-comparison" style="margin:10px 0 16px;font-size:13px;line-height:1.6">'
        '<h3 style="font-size:15px;margin:8px 0">兩種決策架構</h3>'
        + table(["項目", "V7 碎形模型（研究 bundle）" if "FRACTAL_MODEL" in modes or not trained else "V7 已載入模型（研究 bundle）",
                 "V7 模型（一般 --profile elec）"], rows) + identity_line
        + '<p style="font-size:12px;margin:6px 0">碎形特徵依 bundle 宣告；已訓練不代表已證明增益。'
          '需以同條件樣本外與成本後結果驗收；UNKNOWN、重大財務風險、交易限制與失效停損仍阻擋。'
          '部位上限不回頭改 LES Score。</p></section>')


class LowEntryV7Renderer(HtmlRenderer):
    def render_entry_summary(self, decisions, *, _source_verifier=None):
        if _source_verifier is None:
            with valuation_history_verifier_scope() as verifier:
                return self.render_entry_summary(decisions, _source_verifier=verifier)
        rows, fr2_modes = [], set()
        for item in decisions:
            report = report_for(item, source_verifier=_source_verifier)
            e, d = report["envelope"], report["diagnostics"]
            fr2_modes.add(d.get("fr2") is not None)
            if e["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}: continue
            plan = d["reference_plan"] or {}
            authorization = report.get("pae_authorization", {})
            eligibility = (f'PAE 候選已授權<br>股數上限 {text(authorization["indicative_shares"])}<br>待 PAE 配置／尚未成交'
                if authorization.get("authorized") is True else
                'PAE 候選未授權<br>' + '<br>'.join(text(reason) for reason in authorization.get("reason_codes", ["ACCOUNT_AUTHORIZATION_MISSING"])))
            if d.get("fr2"):
                assessment = report.get("fr2_execution")
                if assessment is None:
                    eligibility = '研究配置尚未驗證<br>尚未提供帳戶驗證／不可下單'
                else:
                    eligibility = (f'研究配置 {text(assessment["state"])}<br>'
                        f'指示股數上限 {text(assessment["indicative_shares"])}<br>尚未成交／不可下單')
                    eligibility += '<br>' + '<br>'.join(text(reason) for reason in assessment["reason_codes"])
                    total_cap = assessment.get("maximum_total_weight")
                    eligibility += ('<br>持倉上限待帳戶驗證' if total_cap is None else
                        f'<br>研究持倉總上限 {number(total_cap * 100)}%')
            if d.get("entry_admission"):
                total_cap = authorization.get("maximum_total_weight")
                eligibility += ('<br>持倉上限待帳戶驗證' if total_cap is None else
                    f'<br>核准持倉總上限 {number(total_cap * 100)}%')
            conditional = plan.get("entry_basis") == "STRUCTURAL_CONDITIONAL_BAND"
            projected = plan.get("target_method") == "MODEL_PROJECTED_TARGET"
            plan_label = ('條件式／尚未成交<br>僅下一官方交易日開盤入帶有效；收盤到期<br>'
                          if conditional else '')
            target_label = '模型投射目標' if projected else '目標'
            rows.append([text(f'{item.value("證券代號")} {item.value("證券名稱")}'),
                text(e["strategy"]["strategy_decision"]), number(e["score"]["final_score"]),
                text(context_label(report)),
                plan_label + f'進場帶 NT${number(plan.get("band", [None, None])[0])}～{number(plan.get("band", [None, None])[1])}<br>停損 {number(plan.get("stop"))}／{target_label} {number(plan.get("target"))}',
                eligibility, text(item.value("PAE Decision", "N/A"))])
        return '<section data-section="entry-summary" data-les-version="7.0.0"><h3>進場決策彙整（BUY / BUY_SMALL）</h3>' + (
            table(["股票", _decision_label(fr2_modes), "Final", "型態", "因果交易計畫", "執行資格", "PAE"], rows)
            if rows else '<p>本次沒有 v7 BUY / BUY_SMALL；條件與證據檢查見下方分析。</p>') + '</section>'

    def render_analysis(self, decisions, *, include_entry_summary=True, _source_verifier=None):
        if _source_verifier is None:
            with valuation_history_verifier_scope() as verifier:
                return self.render_analysis(decisions, include_entry_summary=include_entry_summary,
                                            _source_verifier=verifier)
        rows, modes, versions = [], [], set()
        row_versions = []
        models, policies, components, fr2_modes = set(), set(), [], set()
        for item in decisions:
            r = report_for(item, source_verifier=_source_verifier); e, d = r["envelope"], r["diagnostics"]
            fr2_modes.add(d.get("fr2") is not None)
            s, q, loc = e["score"], e["quality"], e["localization"]
            admission = d.get("entry_admission")
            quality_text = f'{text(q["status"])} / QS {number(q["quality_score"])}<br>DC {number(q["data_confidence"])}'
            if admission:
                aq, av = admission["quality"], admission["valuation"]
                quality_text += f'<br>品質級別 {text(aq["tier"])}'
                location_note = valuation_note(av)
            else: location_note = None
            baseline = d.get("rule_baseline") is not None
            if baseline:
                policy = d["rule_baseline"]["policy"]
                versions.add(policy["policy_version"])
                thresholds = policy.get("unified_technical_policy", {})
                policies.add((policy["policy_version"], thresholds.get("buy_minimum", 72.),
                              thresholds.get("small_minimum", 62.)))
            row_versions.append(policy["policy_version"] if baseline else None)
            if e["model"].get("bundle_sha256"):
                models.add(e["model"]["bundle_sha256"])
            if d.get("research_components"):
                components.append(d["research_components"])
            modes.append(report_mode(r))
            location_text = ('規則評分' if baseline else
                f'{text(loc["status"])}<br>底部≤3交易日 {number(loc["p_bottom_soon"])}'
                f'<br>成交 {number(loc["p_fill"])}'
                f'<br><span title="已成交條件下，接近未來底部且反彈的聯合機率；不是無條件勝率">'
                f'成交後近底反彈 {number(loc["p_entry_given_fill"])}</span>')
            if location_note: location_text += '<br>' + location_note
            gates = '<br>'.join(f'{text(g["gate_id"])}: {text(g["state"])}' for g in d["gate_receipts"])
            # The daily projection already carries TWD_x1000. This display is
            # explicitly scaled; native reference-plan prices remain in TWD.
            rows.append([text(f'{item.value("證券代號")} {item.value("證券名稱")}') + f'<br>現價 NTD {number(item.value("收盤"))}<br>資料日 {text(r["signal_date"])}',
                text(e["strategy"]["strategy_decision"]),
                f'{text(context_label(r))}<br>' + score_text(s, legacy_caps=baseline,
                    show_arithmetic=(d.get("rule_baseline") or {}).get("policy", {}).get("policy_version") in {
                        "V7_RULE_BASED_BASELINE_R21", "V7_RULE_BASED_BASELINE_R22", "V7_RULE_BASED_BASELINE_R23", "V7_RULE_BASED_BASELINE_R24", "V7_RULE_BASED_BASELINE_R25"}),
                quality_text,
                location_text,
                gates, f'{number(item.value("EREM 分數"))}<br>{text(item.value("EREM 決策訊號"))}',
                '<br>'.join(text(reason) for reason in e["strategy"]["reason_codes"]) or "NONE"])
        unique_modes = set(modes)
        if len(unique_modes) == 1:
            mode = modes[0]
            if mode == "RULE_BASELINE":
                mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                    '<strong>V7 模型</strong></div>'
                    '<div style="font-size:12px;line-height:1.6;margin-top:4px">使用 v6 型態規則與固定配分 · 未使用機率預測模型 · BUY 72／BUY_SMALL 62</div>')
                if len(versions) > 1:
                    for row, version in zip(rows, row_versions):
                        row[0] += ('<br><span data-rule-policy="' + text(version) + '" style="font-size:12px">'
                            + text(version.rsplit("_", 1)[-1]) + '</span>')
                    declared = '<br>'.join(f'{text(v.rsplit("_", 1)[-1])} · BUY {buy:g}／BUY_SMALL {small:g}'
                        for v, buy, small in sorted(policies))
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7">'
                        '<strong>V7 模型</strong></div>'
                        '<div data-entry-policy="MIXED" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        '混合規則版本 · 各檔保留原政策與門檻<br>' + declared + '</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R25"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v7 規則式基線 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R25" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R25 · BUY 70／BUY_SMALL 55 · 正常量能漸進計分 · Quality R7 與必要風控不變 · 條件未達不得成交</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R24"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v7 規則式基線 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R24" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R24 · BUY 70／BUY_SMALL 60 · 沿用 R23 計分與風控 · Quality R7 來源綁定核實</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R23"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v7 規則式基線 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R23" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R23 · BUY 70／BUY_SMALL 60 · 沿用 R22 計分 · 衝突 P/E 隔離，價格與必要風控仍逐項驗證</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R22"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v7 規則式基線 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R22" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R22 · BUY 70／BUY_SMALL 60 · 守住支撐的突破進度漸進計分 · 軟性缺值只影響分數 · 必要來源與交易風控不變</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R21"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v7 規則式基線 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R21" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R21 · BUY 70／BUY_SMALL 60 · 低位觀察採短期回穩解讀，非進場資格 · 產業證據分週期核對 · 安全與交易計畫不變</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R20"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">v6 Hybrid 型態解讀 · 固定六因子配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R20" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R20 · BUY 70／BUY_SMALL 60 · 早期止跌部分計分／停損連續配分 · 沿用 R19 估值與風控 · 次日開盤確認</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R19"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">使用 v6 型態規則與固定配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R19" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R19 · BUY 70／BUY_SMALL 60 · 進場價帶依波動擴展，仍受停損距離與成本後風險報酬限制 · 次日開盤確認</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R18"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">使用 v6 型態規則與固定配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R18" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R18 · 六因子統一計分 · BUY 70／BUY_SMALL 60 · 支撐／阻力條件限價僅小倉 · 未符合開盤價帶不成交</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R17"}:
                    mode_header = (f'<div data-report-mode="{mode}" style="font-size:14px;line-height:1.7;white-space:normal;overflow-wrap:anywhere">'
                        '<strong>V7 模型</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">使用 v6 型態規則與固定配分 · 未使用機率預測模型</div>'
                        '<div data-entry-policy="R17" style="font-size:12px;line-height:1.6;margin-top:4px">'
                        'R17 · 因果型態解讀 · 六因子統一計分 · BUY 72／BUY_SMALL 62 · 來源與交易計畫安全標準不變</div>')
                elif versions == {"V7_RULE_BASED_BASELINE_R10"}:
                    mode_header += '<div data-entry-policy="R10" style="font-size:12px;line-height:1.6;margin-top:4px">R10 · 近期復甦／產業估值 · 型態一致交易計畫</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R11"}:
                    mode_header += '<div data-entry-policy="R11" style="font-size:12px;line-height:1.6;margin-top:4px">R11 · 來源綁定產業強弱 · 沿用 R10 估值／風控</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R12"}:
                    mode_header += '<div data-entry-policy="R12" style="font-size:12px;line-height:1.6;margin-top:4px">R12 · 收盤確認支撐一致 · 沿用 R10 估值／風控</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R15"}:
                    mode_header += '<div data-entry-policy="R15" style="font-size:12px;line-height:1.6;margin-top:4px">R15 · 核心 BUY／戰術 BUY_SMALL · 來源驗證不放寬 · 限倉不影響分數</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R16"}:
                    mode_header += '<div data-entry-policy="R16" style="font-size:12px;line-height:1.6;margin-top:4px">R16 · 早期回檔條件價帶／價格發現投射目標 · 僅小倉 · 未觸價不成交</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R14"}:
                    mode_header += '<div data-entry-policy="R14" style="font-size:12px;line-height:1.6;margin-top:4px">R14 · 復甦／產業估值 · 短盤整突破試單 · 分級限倉</div>'
                elif versions == {"V7_RULE_BASED_BASELINE_R13"}:
                    mode_header += '<div data-entry-policy="R13" style="font-size:12px;line-height:1.6;margin-top:4px">R13 · 年度 EPS 價格估值 · 分級限倉 · 型態一致風控</div>'
            else:
                mode_header = mode_badge(mode) + f'<span style="display:inline-block;margin:6px 0 6px 10px;font-size:13px;line-height:1.6">{text(MODE_NOTES[mode])}</span>'
                if mode in {"MODEL", "FRACTAL_MODEL"} and any(record.get("entry", {}).get("version") == "FR_ENTRY_R2" for record in components):
                    mode_header = (f'<div data-report-mode="{mode}" data-research-policy="FR_ENTRY_R2" '
                        'style="font-size:14px;line-height:1.7;overflow-wrap:anywhere">'
                        f'<strong>{text(MODE_LABELS[mode])}｜FR2 研究政策｜不可下單</strong></div>'
                        '<div style="font-size:12px;line-height:1.6;margin-top:4px">'
                        '預測機率與原 bundle 不變；進場政策、交易驗證各自版本化 · BUY 75／BUY_SMALL 65</div>')
                    if any(record.get("valuation") for record in components):
                        mode_header += ('<div data-valuation-mode="FR2_EPS_PRICE" '
                            'style="font-size:12px;line-height:1.6;margin-top:4px">'
                            'FR2｜核實年度EPS價格估值｜僅價格報酬／不含股利</div>')
        elif unique_modes:
            mode_header = '<span style="font-size:13px">混合模式 · 各檔依實際資料標示</span>'
            for row, mode in zip(rows, modes):
                row[0] += '<br>' + mode_badge(mode)
        else:
            mode_header = '<span style="font-size:13px">尚無可顯示的研究報告</span>'
        label_attribute = f' data-model-label="{text(model_label(unique_modes))}"' if unique_modes else ""
        body = '<section data-section="v7-analysis" data-les-version="7.0.0"' + label_attribute + '><h2>低點進場判斷（Low-Entry Model v7）＋ 預期報酬出場判斷（EREM v3）</h2><div data-section="report-mode" style="margin:8px 0 12px">' + mode_header + '</div><p style="font-size:13px;line-height:1.6">技術分數不等於進場許可；必要證據與安全風控仍須通過。限倉不回頭扣分，原始品質與估值判定保留。PAE 候選授權不等於配置、成交或送單授權。</p>'
        body += architecture_comparison(modes=unique_modes, models=models, policies=policies, components=components)
        body += table(["股票 / 現價", _decision_label(fr2_modes), "LES 技術分數", "Quality / DC", "評分模式 / 預測", "必要閘門", "EREM v3", "原因 / 缺口"], rows) + '</section>'
        return (self.render_entry_summary(decisions, _source_verifier=_source_verifier)
                if include_entry_summary else "") + body

    def render(self, decisions): return self.render_analysis(decisions)


class _ProjectedReportDecision:
    """Minimal read-only-by-interface adapter; no account or strategy evaluation."""
    __slots__ = ("_values",)

    def __init__(self, values):
        self._values = values

    def value(self, name, default=None):
        return self._values.get(name, default)


def render_reports(reports):
    """Render validated native reports as a local, responsive HTML document.

    Missing EREM/PAE inputs stay N/A; this presentation adapter never creates
    a portfolio, infers an empty account, sends mail or evaluates a strategy.
    Native prices are TWD/share and the existing daily display is TWD_x1000.
    """
    from application.low_entry_v7_daily import project_report
    from infrastructure.report.html_renderer import DailyStrategyEmailRenderer
    decisions = []
    for report in reports:
        values = project_report(report)
        close = report["diagnostics"]["features"].get("close")
        values.update({"證券代號": report["symbol"], "證券名稱": "",
                       "收盤": None if close is None else close * 1000.})
        decisions.append(_ProjectedReportDecision(values))
    body = LowEntryV7Renderer().render(decisions)
    title = (rendered_model_label(body) or "Low Entry Model") + "｜研究報告"
    return ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + text(title) + '</title>' + DailyStrategyEmailRenderer._RESPONSIVE_STYLE
        + '</head><body><main class="twse-daily-report">'
          '<p style="font-size:13px;line-height:1.6">本頁僅呈現提供的 LES 原生報告；'
          '未提供的 EREM／PAE 顯示 N/A。未建立帳戶或成交，不寄信、不下單。</p>'
        + body + '</main></body></html>')
