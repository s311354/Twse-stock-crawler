"""Render the actual frozen v6 report; do not relabel a v5 table."""
from __future__ import annotations

import html
import json
from collections.abc import Sequence

from domain.model_contracts import StrategyDecision
from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import SETUPS
from infrastructure.report.contracts import HtmlRenderer


def text(value) -> str:
    return html.escape("N/A" if value is None else str(value))


def numeric(value, places=2) -> str:
    return "N/A" if value is None else f"{float(value):,.{places}f}"


def history_summary(report: dict) -> str:
    data = report.get("data", {})
    observations = []
    for label, key in (("日線", "stock"), ("大盤", "market")):
        count, required = (number(data.get(f"{key}_history_{suffix}")) for suffix in ("observations", "required"))
        if count is not None and required is not None:
            observations.append(f'{label} {numeric(count, 0)}/{numeric(required, 0)}')
    return "；".join(observations)


def setup_score_summary(report: dict) -> str:
    decision = report["decision"]
    if decision.get("policy_version") in {"context-aware-unified-score/v1", "context-aware-unified-score/v2", "context-aware-convex-score/v1"}:
        selection = decision["selection"]
        headline = (f'Hybrid｜型態 {text(selection["selected_setup"])}<br>'
                    f'{text(selection["status"])}（依結構先選型，不比較三個分數）<br>'
                    f'唯一 Final {numeric(decision.get("final_score"))}<br>'
                    f'Raw {numeric(decision.get("raw_score"))} - Penalty {numeric(decision.get("penalty"))}'
                    f' = Adjusted {numeric(decision.get("adjusted_score"))}<br>'
                    f'Safety Cap {numeric(decision.get("safety_cap"))}（多項取最小值）')
        labels = dict(structure="結構", trend="趨勢", relative_strength="相對強弱",
                      demand="買盤", momentum="動能", execution="執行風控")
        headline += '<br>' + '／'.join(f'{label} {numeric(decision["budget_groups"][key]["earned"])}'
            f'/{numeric(decision["budget_groups"][key]["budget"], 0)}' for key, label in labels.items())
        headline += (f'<br>固定分母100；覆蓋率 {numeric(decision.get("score_coverage"), 1)}%'
                     '；缺失槽位不給分、不回補預算、不比例估分')
        weighting = decision.get("weighting")
        if weighting is not None:
            headline += (f'<br>凍結凸估計權重｜{text(weighting["applied_context"])}'
                f'｜{text(weighting["artifact_sha256"][:12])}'
                '<br>執行因子內部比例：停損40%／淨RR30%／流動性20%／價差10%'
                '<br>載入權重不額外阻擋；仍需原生安全閘門與次交易日驗證'
                '<br>績效尚未驗證；收斂不等於預測有效或機率校準')
        else:
            headline += '<br>停損4／淨RR3／流動性2／價差1；單一因果交易計畫'
        if selection["status"] == "NO_CLEAR_CONTEXT":
            headline += '<br>型態未明：共用解讀規則；不強制零分，採反轉型保守風控'
        if decision.get("final_score") is None:
            headline += '<br>無可計算證據；非實際 0 分'
        if decision.get("rr_policy") == "PRICE_DISCOVERY_EXTENSION_SUPPORT":
            headline += '<br>價格發現：目標／淨RR N/A；獨立政策，不補RR分'
        adjustments = [f'{item["reason"]}: -{numeric(item["points"])}' for item in decision.get("penalties", [])]
        adjustments += [f'{item["reason"]}: cap {numeric(item["maximum"])}' for item in decision.get("safety_caps", [])]
        if adjustments:
            headline += '<br>' + '<br>'.join(text(value) for value in adjustments)
        return headline + '<br>研究候選配分／75、65比較門檻未校準；不等同舊版分數或成交授權'
    if decision.get("policy_version") == "unified-penalty-safety-cap/v2":
        headline = (f'選定型態：{text(decision.get("setup"))}<br>'
                    f'Final {numeric(decision.get("final_score"))}<br>'
                    f'Raw {numeric(decision.get("raw_score"))} - Penalty {numeric(decision.get("penalty"))}'
                    f' = Adjusted {numeric(decision.get("adjusted_score"))}<br>'
                    f'Safety Cap {numeric(decision.get("safety_cap"))}（多項取最小值）')
        execution, selection = decision.get("execution_score", {}), decision.get("selection", {})
        if execution:
            headline += (f'<br>共用執行風控 {numeric(execution.get("earned_points"))}/10'
                         '（停損4／淨RR3／流動性2／價差1）')
        overlap = decision.get("correlation_adjustment", {})
        if overlap.get("policy_version") == "source-family-max-pooling/v1":
            headline += f'<br>相關訊號去重：-{numeric(overlap.get("removed_points"))}（同族取最大，不回補分數）'
        if selection:
            highest = " / ".join(selection.get("highest_score_setups", [])) or "N/A"
            headline += f'<br>最高 Final：{text(highest)} {numeric(selection.get("highest_final_score"))}'
            headline += ('<br>選型：可執行優先 → BUY 層級 → 覆蓋率 → 淨RR餘裕 → 停損比例 → 固定順序；不一定取最高分'
                         if selection.get("mode") == "EXECUTABLE_RISK_FIRST" else
                         '<br>無可執行型態：依 Final → 覆蓋率 → 固定順序選擇診斷型態')
        if decision.get("rr_policy") == "PRICE_DISCOVERY_EXTENSION_SUPPORT":
            headline += '<br>價格發現：獨立延伸／支撐政策；RR N/A，不補目標、不補 RR 分'
        reasons = [f'{item["reason"]}: -{numeric(item["points"])}' for item in decision.get("penalties", [])]
        reasons += [f'{item["reason"]}: cap {numeric(item["maximum"])}' for item in decision.get("safety_caps", [])]
        headline += f'<br>覆蓋率 {numeric(decision.get("score_coverage"), 1)}%；{text(decision.get("evidence_status"))}'
        observations = history_summary(report)
        if observations:
            headline += '<br>' + observations
        if decision.get("evidence_status") == "PARTIAL":
            headline += f'<br>可得證據比例估分 {numeric(decision.get("estimated_score"))}（不替代 Final）'
        if decision.get("final_score") is None:
            headline += '<br>無可計算證據；非實際 0 分'
        if reasons:
            headline += '<br>' + '<br>'.join(text(reason) for reason in reasons)
        leadership = report.get("leadership", {})
        if leadership:
            labels = {True: "符合", False: "未符合", None: "UNKNOWN"}
            metrics = leadership.get("metrics", {})
            headline += (f'<br>產業基準 {text(leadership.get("industry_status"))}；'
                         f'RS20 {numeric(metrics.get("industry_rs20"), 3)}'
                         f'<br>抗跌盤整 {labels[leadership.get("resilient_base")]}；'
                         f'領先突破 {labels[leadership.get("leading_breakout")]}'
                         '（研究條件，非進場授權）')
        return headline + '<br>型態／收盤 Trigger 已列入條件明細，不重複設軟性閘門'
    selected = decision.get("setup")
    score, source = number(decision.get("score")), decision.get("score_setup", selected)
    data = report.get("data", {})
    incomplete = (decision.get("score_basis") == "INSUFFICIENT_HISTORY"
                  or data.get("completed_history") is False or data.get("market_aligned") is False)
    partial = decision.get("score_basis") == "PARTIAL_DATA"
    evidence = decision.get("score_evidence", {})
    # Old reports used a zero sentinel when setup=None. Display their stored
    # component-derived score, without upgrading/re-versioning the old signal.
    if not selected and "score_basis" not in decision and not incomplete:
        scored = [(name, number(report["setups"].get(name, {}).get("score"))) for name in SETUPS]
        scored = [(name, value) for name, value in scored if value is not None]
        source, score = max(scored, key=lambda item: item[1]) if scored else (None, None)
    if partial and score is not None:
        headline = f'{numeric(score)}<br>{text(source)}（部分資料估算；不可進場）'
    elif incomplete or score is None:
        headline = "N/A（技術資料不足）"
    elif selected:
        headline = f'{numeric(score)}<br>{text(selected)}'
    else:
        headline = f'{numeric(score)}<br>{text(source)}（參考評分；無合格型態）'
    coverage = number(evidence.get("coverage_percent"))
    if coverage is not None:
        formal = number(decision.get("formal_score"))
        if formal is not None:
            headline += f'<br>正式評分 {numeric(formal)}（固定滿分權重）'
        headline += (f'<br>評分覆蓋率 {numeric(coverage, 1)}%'
                     f'；已得 {numeric(evidence.get("earned_points"))} / 可計算 {numeric(evidence.get("available_weight"))} 分')
        if not partial and coverage < 100 and number(evidence.get("estimated_score")) is not None:
            headline += f'<br>可用資料估算 {numeric(evidence["estimated_score"])}（不替代進場分數）'
        observations = history_summary(report)
        if observations:
            headline += "<br>" + observations
    if not partial and decision.get("entry_tier") == "STARTER_ONLY":
        headline += "<br>修復型：僅限 BUY_SMALL"
    elif not partial and decision.get("entry_tier") == "FORMING":
        headline += "<br>型態形成中，尚未確認進場"
    details = []
    for name, setup in report["setups"].items():
        if not isinstance(setup, dict) or "score" not in setup:
            continue
        status = ("形成中" if setup.get("entry_tier") == "FORMING" else "修復型" if setup.get("entry_tier") == "STARTER_ONLY" else "型態成立" if setup.get("valid")
                  else "形成中" if setup.get("recognized") else "未成立")
        blockers = " / ".join(setup.get("blocking_conditions", []))
        measured = setup.get("score_evidence", {})
        value = number(measured.get("estimated_score")) if partial else None if incomplete else number(setup["score"])
        label = f'估算；覆蓋率 {numeric(measured.get("coverage_percent"), 1)}%' if partial else status
        details.append(f'<span title="{text(blockers)}">{text(name)} {numeric(value)}（{label}）</span>')
    return headline + ("<br>" + "<br>".join(details) if details else "")


def gate_summary(report: dict) -> str:
    gates = entry_gate_checks(report)
    labels = {"quality": "Quality Gate", "no_veto": "No Veto", "score": "Score",
              "trigger": "Trigger（收盤型態）", "risk": "Risk Gate"}
    if report.get("decision", {}).get("policy_version") in {"unified-penalty-safety-cap/v2", "context-aware-unified-score/v1", "context-aware-unified-score/v2", "context-aware-convex-score/v1"}:
        labels["trigger"] = "收盤條件明細一致性（非額外資格）"
    return "<br>".join(f'<span data-entry-gate="{key}" data-status="{text(gates[key])}">'
                       f'{label}: {text(gates[key])}</span>' for key, label in labels.items())


def industry_summary(report: dict) -> str:
    profile = report.get("industry_profile", {})
    if not profile:
        return ""
    quality = profile.get("quality_evaluation", {})
    detail = ""
    if quality:
        missing = [name for name, metric in quality.get("metrics", {}).items() if not metric.get("available")]
        detail = (f'<br>產業 QS {numeric(quality.get("score"))} / {text(quality.get("status"))}'
                  f'<br>DC {numeric(quality.get("data_confidence"))}；覆蓋權重 {numeric(quality.get("available_weight"))}/100'
                  + (f'<br><span title="{text(", ".join(missing))}">缺失指標 {len(missing)} 項</span>' if missing else ""))
    return (f'<br>{text(profile.get("name"))} / {text(profile.get("mode"))}'
            f'<br><small>{text(profile.get("version"))}</small>'
            + detail
            + ("<br>產業門檻待校準，僅觀察、不進場" if profile.get("mode") == "SHADOW"
               else "<br>進場已啟用；產業參數尚未完成績效校準"
               if profile.get("calibration_status") == "UNCALIBRATED" else ""))


def report_for(decision: StrategyDecision) -> dict:
    if not isinstance(decision, StrategyDecision):
        raise TypeError("HtmlRenderer accepts StrategyDecision objects only")
    if decision.value("LES Model Version") != "6.0.0":
        raise ValueError("v6 renderer requires a real v6 decision")
    report = json.loads(decision.value("LES v6 Report"))
    if report["model_version"] != "6.0.0" or report["decision"]["signal"] != decision.value("低點決策"):
        raise ValueError("v6 renderer decision/report mismatch")
    return report


def table(headers, rows) -> str:
    head = "".join(f"<th>{text(label)}</th>" for label in headers)
    body = "".join("<tr>" + "".join(f'<td data-label="{text(label)}">{value}</td>' for label, value in zip(headers, row)) + "</tr>" for row in rows)
    return f'<div class="decision-table-scroll" style="overflow-x:auto"><table class="responsive-decision-table" style="border-collapse:collapse;width:100%" border="1" cellpadding="8"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class LowEntryV6Renderer(HtmlRenderer):
    def render_entry_summary(self, decisions: Sequence[StrategyDecision]) -> str:
        candidates = [(item, report_for(item)) for item in decisions if item.value("低點決策") in {"BUY", "BUY_SMALL"}]
        candidates = [(item, report) for item, report in candidates if report_failure(report, allow_replay=True) is None]
        rows = []
        for item, report in candidates:
            d, entry, sizing = report["decision"], report["entry"], report["sizing"]
            rows.append([
                text(f'{report["symbol"]} {report["name"]}'), text(d["signal"]),
                f'{text(d["setup"])}<br>LES {numeric(d["score"])}',
                gate_summary(report),
                f'突破訊號日高點 NT${numeric(entry.get("trigger_price"))}<br>下一交易日確認，尚未成交',
                f'停損 NT${numeric(entry.get("stop_price"))}<br>目標 NT${numeric(entry.get("target_price"))}<br>毛 RR {numeric(entry.get("reward_risk"))}／淨 RR {numeric(entry.get("net_reward_risk"))}',
                f'試算 {text(sizing["indicative_shares"])} 股<br>{text(sizing["reason"])}',
                text(item.value("PAE Decision", "PAE_NOT_EVALUATED")),
            ])
        buy = sum(report["decision"]["signal"] == "BUY" for _, report in candidates)
        replay_cutoffs = sorted({report_for(item).get("replay_as_of") for item in decisions if report_for(item).get("replay_as_of")})
        notice = (f'<p><strong>HISTORICAL_REPLAY｜歷史重播／不可下單</strong><br>歷史決策截止：{text(", ".join(replay_cutoffs))}。這是歷史研究結果，不是今日可執行訊號。</p>' if replay_cutoffs else "")
        return notice + '<section data-section="entry-summary" data-les-version="6.0.0"><h3>進場決策彙整（BUY / BUY_SMALL）</h3>' + (
            f'<p>共 {len(candidates)} 檔｜BUY {buy}｜BUY_SMALL {len(candidates)-buy}。已發布訊號不等於可成交訂單。</p>'
            + table(["股票", "v6 決策", "Setup / 分數", "五項進場檢核", "執行條件", "風控（TWD）", "試算部位", "PAE 配置"], rows)
            if candidates else '<p>本次沒有 BUY / BUY_SMALL 訊號；阻擋原因見下方 v6 分析。</p>'
        ) + '<p>實際數量仍受 PAE、即時報價、RR、流動性與可用資金限制；缺少必要資料不成交。</p></section>'

    def render_analysis(self, decisions: Sequence[StrategyDecision], *, include_entry_summary=True) -> str:
        rows = []
        for item in decisions:
            report = report_for(item)
            d, q, data = report["decision"], report["quality"], report["data"]
            reasons = list(d.get("reasons", [])) + list(q.get("reasons", [])) + list(report["downtrend_gate"].get("reasons", []))
            failure = report_failure(report, allow_replay=True) if d["signal"] in {"BUY", "BUY_SMALL"} else None
            if failure:
                reasons.append(failure)
            rows.append([
                text(f'{report["symbol"]} {report["name"]}') + f'<br>資料日 {text(report["signal_date"])}',
                text(f'訊號無效（{d["signal"]}）' if failure else d["signal"]),
                setup_score_summary(report),
                f'{text(q["status"])} / {numeric(q.get("score"))}<br>DC {numeric(data.get("confidence"))}<br>{text(data["integrity"])}' + industry_summary(report),
                gate_summary(report),
                f'{text(report["valuation"]["interpretation"])}（P/E 時機參考）<br>PAE Valuation {text(item.value("Valuation Gate"))}',
                f'{numeric(item.value("EREM 分數"))}<br>HC {numeric(item.value("EREM Holding Confidence"))}<br>{text(item.value("EREM 決策訊號"))}',
                "<br>".join(text(reason) for reason in dict.fromkeys(reasons)),
            ])
        section = '<section data-section="v6-analysis" data-les-version="6.0.0"><h2>低點進場判斷（Low-Entry Model v6）＋ 預期報酬出場判斷（EREM v3）</h2><p>Quality-First Adaptive Entry：REVERSAL / TREND_PULLBACK / LEADERSHIP_CONTINUATION。分數與理由來自實際 v6 plugin；PAE 使用的既有估值／CER 另列，不冒充 v6 估值。</p>'
        if any(report_for(item)["decision"].get("policy_version") in {"context-aware-unified-score/v1", "context-aware-unified-score/v2", "context-aware-convex-score/v1"} for item in decisions):
            section += '<p>Hybrid：先選型態 → 六因子固定100分 → 明確扣分 → 最小安全封頂 → 唯一決策。配分與門檻為未校準研究候選；PAE與即時驗證僅限制配置／執行，不改LES原始訊號。</p>'
        cutoffs = sorted({report_for(item).get("replay_as_of") for item in decisions if report_for(item).get("replay_as_of")})
        if cutoffs:
            section += f'<p>HISTORICAL_REPLAY｜歷史研究，不可下單｜決策截止 {text(", ".join(cutoffs))}｜PAE 不使用正式持續性紀錄。</p>'
        section += table(["股票", "進場決策", "LES v6 分數", "Quality / DC", "五項進場檢核", "估值", "EREM v3 持有分數", "決策原因"], rows)
        section += '</section>'
        return ((self.render_entry_summary(decisions) if include_entry_summary else "")
                + section)

    def render(self, decisions: Sequence[StrategyDecision]) -> str:
        return self.render_analysis(decisions)
