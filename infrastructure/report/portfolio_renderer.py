"""Version-neutral Portfolio Allocation Engine HTML renderer."""

from __future__ import annotations

import html
from collections.abc import Iterable
from collections.abc import Sequence

import pandas as pd

from domain.model_contracts import StrategyDecision
from domain.strategies.portfolio_allocation_v1 import PAE_V1_MODEL_VERSION
from infrastructure.report.contracts import HtmlRenderer


class PortfolioAllocationFormatter:
    """Format scalar PAE values without interpreting model policy."""

    @staticmethod
    def _metric(label: str, value: str) -> str:
        return (
            '<div style="min-width:125px;padding:7px 9px;border:1px solid #cad8e1;'
            'background:#fff"><small style="color:#62737c">{}</small><br>'
            "<strong>{}</strong></div>"
        ).format(html.escape(label), value)

    @staticmethod
    def _text(value: object) -> str:
        if value is None:
            return "N/A"
        try:
            if bool(pd.isna(value)):
                return "N/A"
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return html.escape(text) if text else "N/A"

    @staticmethod
    def _number(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if pd.notna(numeric) else 0.0

    @classmethod
    def _percent(cls, value: object) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return "{:.1f}%".format(numeric * 100)

    @classmethod
    def _signed_percent(cls, value: object) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return "{:+.1f}%".format(numeric * 100)

    @classmethod
    def _decimal(cls, value: object, precision: int) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return ("{:.%df}" % precision).format(numeric)

    @classmethod
    def _signed_decimal(cls, value: object, precision: int) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return ("{:+.%df}" % precision).format(numeric)

    @classmethod
    def _flag(cls, value: object) -> str:
        return "YES" if cls._truth(value) else "NO"

    @staticmethod
    def _truth(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "pass"}

    @classmethod
    def _badge(cls, value: object) -> str:
        text = cls._text(value)
        normalized = html.unescape(text).upper()
        if any(
            token in normalized for token in ("EXIT", "FAIL", "INSUFFICIENT")
        ):
            colors = ("#8f1d1d", "#fdecec")
        elif any(
            token in normalized for token in ("HOLD", "WATCH", "CANDIDATE")
        ):
            colors = ("#755000", "#fff4d6")
        else:
            colors = ("#176b4d", "#e7f5ee")
        return (
            '<span style="display:inline-block;padding:2px 6px;border-radius:10px;'
            'font-weight:600;color:{};background:{}">{}</span>'
        ).format(colors[0], colors[1], text)


class PortfolioAllocationTableRenderer(PortfolioAllocationFormatter):
    """Select and render only PAE actions that move portfolio capital."""

    _DISPLAY_ACTIONS = frozenset(
        {
            "STRONG_ADD",
            "ADD",
            "ROTATE_25",
            "ROTATE_50",
            "ROTATE_75",
            "FAST_TRACK_ROTATE",
            "EXIT",
            "EXIT_AND_REALLOCATE",
            "EXIT_TO_CASH",
            "REDUCE_25",
            "TRIM_25",
            "TRIM_TO_TARGET",
            "STARTER",
        }
    )
    _DISPLAY_PRIORITY = {
        "EXIT_TO_CASH": 0,
        "EXIT_AND_REALLOCATE": 1,
        "EXIT": 2,
        "FAST_TRACK_ROTATE": 3,
        "ROTATE_75": 4,
        "ROTATE_50": 5,
        "ROTATE_25": 6,
        "REDUCE_25": 7,
        "TRIM_25": 8,
        "TRIM_TO_TARGET": 9,
        "STRONG_ADD": 10,
        "ADD": 11,
        "STARTER": 12,
    }

    @classmethod
    def latest_by_symbol(
        cls,
        decisions: Iterable[StrategyDecision],
    ) -> tuple[StrategyDecision, ...]:
        """Keep one deterministic PAE row per stock code."""
        latest: dict[str, StrategyDecision] = {}
        for decision in decisions:
            symbol = cls._raw_text(decision.value("證券代號"))
            if not symbol:
                continue
            current = latest.get(symbol)
            if current is None or decision.evaluated_at >= current.evaluated_at:
                latest[symbol] = decision
        return tuple(latest[symbol] for symbol in sorted(latest))

    @staticmethod
    def _raw_text(value: object) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    def _selected(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> tuple[StrategyDecision, ...]:
        selected = [
            decision
            for decision in decisions
            if self._text(decision.value("PAE Decision"))
            in self._DISPLAY_ACTIONS
        ]
        return tuple(
            sorted(
                selected,
                key=lambda decision: (
                    self._DISPLAY_PRIORITY[
                        self._text(decision.value("PAE Decision"))
                    ],
                    -self._number(decision.value("PAE Target Weight")),
                    self._text(decision.value("證券代號")),
                ),
            )[:12]
        )

    def _row(self, decision: StrategyDecision) -> str:
        stock = "<strong>{}</strong><br><small>{}</small><br>{}".format(
            self._text(decision.value("證券代號")),
            self._text(decision.value("證券名稱")),
            self._badge(decision.value("PAE Role")),
        )
        returns = "Quality <strong>{}</strong><br>CER <strong>{}</strong><br><small>Required {}｜Edge {}</small>".format(
            self._decimal(decision.value("PAE Quality Score"), 2),
            self._percent(decision.value("PAE CER")),
            self._percent(decision.value("PAE Required Return")),
            self._percent(decision.value("PAE Opportunity Edge")),
        )
        eligibility = "Strategic {}｜Executable {}<br><small>Binding {}｜Data {}｜Claim {}</small>".format(
            self._flag(decision.value("PAE Strategic Candidate")),
            self._flag(decision.value("PAE Executable Candidate")),
            self._text(decision.value("PAE Binding")),
            self._text(decision.value("PAE Data Quality")),
            self._decimal(decision.value("PAE Capital Claim"), 4),
        )
        weights = "{} → <strong>{}</strong><br><small>Δ {}</small>".format(
            self._percent(decision.value("PAE Current Weight")),
            self._percent(decision.value("PAE Target Weight")),
            self._signed_percent(decision.value("PAE Weight Change")),
        )
        incumbent = decision.value(
            "PAE Incumbent",
            decision.value("PAE Dominance Incumbent"),
        )
        replacement = "Incumbent {}<br><small>Quality {}｜CER {}<br>Excess {} / Hurdle {}<br>Persistence {}｜Fast {}｜Improve {}<br>Atomic {}</small>".format(
            self._text(incumbent),
            self._text(decision.value("PAE Quality Dominance Gate")),
            self._text(decision.value("PAE CER Dominance Gate")),
            self._percent(decision.value("PAE Excess Opportunity")),
            self._percent(decision.value("PAE Switch Hurdle")),
            self._flag(decision.value("PAE Persistence Gate")),
            self._flag(decision.value("PAE Obvious Opportunity")),
            self._flag(decision.value("PAE Portfolio Improvement")),
            self._text(decision.value("PAE Atomic Group")),
        )
        final = "{}<br><small>{}</small><br><small>Blocking：{}｜Execution：{}</small>".format(
            self._badge(decision.value("PAE Decision")),
            self._text(decision.value("PAE Reason")),
            self._text(decision.value("PAE Blocking Reasons")),
            self._text(decision.value("PAE Execution Status")),
        )
        labels = (
            "股票／角色",
            "品質／CER",
            "候選資格",
            "目前 → 目標",
            "品質＋CER 支配證據",
            "PAE 決策",
        )
        cells = (
            stock,
            returns,
            eligibility,
            weights,
            replacement,
            final,
        )
        return "<tr>{}</tr>".format(
            "".join(
                '<td data-label="{}" style="vertical-align:top">{}</td>'.format(
                    html.escape(label),
                    cell,
                )
                for label, cell in zip(labels, cells)
            )
        )

    def _table(self, decisions: Sequence[StrategyDecision]) -> str:
        rows = "".join(self._row(decision) for decision in decisions)
        if not rows:
            return (
                '<div style="padding:9px;background:#fff">'
                "本次沒有涉及進場、加碼、輪動、減碼或退出的 PAE 訊號。"
                "</div>"
            )
        return """
            <div class="decision-table-scroll" style="overflow-x:auto">
              <table class="responsive-decision-table" aria-label="投資組合配置決策" style="border-collapse:collapse;width:100%;font-size:13px;background:#fff" border="1" cellpadding="6">
                <thead style="background:#edf3f7">
                  <tr>
                    <th>股票／角色</th><th>品質／CER</th><th>候選資格</th>
                    <th>目前 → 目標</th><th>品質＋CER 支配證據</th><th>PAE 決策</th>
                  </tr>
                </thead>
                <tbody>{}</tbody>
              </table>
            </div>
            """.format(rows)


class PortfolioAllocationRenderer(
    PortfolioAllocationTableRenderer,
    HtmlRenderer,
):
    """Render versioned aggregate PAE decisions without recalculating rules."""

    _MODE_NOTES = {
        "DIRECT_SELECTION": (
            "目前少於 3 檔有效持股；PAE 為 NON_BINDING，LES 控制進場、EREM v3 控制持股處置。"
        ),
        "PAE_ACTIVE_OPEN_SLOT": (
            "目前 3 檔有效持股；PAE 控制第四席，候選須通過絕對門檻與 Portfolio Improvement。"
        ),
        "PAE_ACTIVE_FULL": (
            "目前 4 檔有效持股；新標的必須通過 Dual Dominance、Switch Hurdle、Persistence 與原子換股。"
        ),
        "POLICY_BREACH_REDUCE_ONLY": (
            "有效持股超過 4 檔；禁止新增持股及淨加碼，只允許降低風險。"
        ),
        "COLD_START": (
            "目前 0 檔有效持股；只有完整通過 Quality、Valuation、CER、LES 與風控的候選才配置資金。"
        ),
        "BUILDING": (
            "目前 1 檔有效持股，處於第二檔建構期；候選須通過絕對品質、估值、CER、LES 與風控門檻，但不套用成對支配與輪動門檻。"
        ),
        "STEADY_STATE": (
            "目前至少 2 檔有效持股；新標的必須通過品質優勢與 Switch Hurdle 的 CER 優勢，才可取得資金。"
        ),
    }

    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        for decision in decisions:
            if not isinstance(decision, StrategyDecision):
                raise TypeError(
                    "HtmlRenderer accepts StrategyDecision objects only"
                )
        pae_decisions = self.latest_by_symbol(
            decision
            for decision in decisions
            if decision.value("PAE Status") is not None
        )
        if not pae_decisions:
            return ""

        first = pae_decisions[0]
        selected = self._selected(pae_decisions)
        mode = self._text(first.value("PAE Mode"))
        model_version = self._text(first.value("PAE Model Version"))
        if model_version == "N/A":
            model_version = PAE_V1_MODEL_VERSION
        mode_note = self._MODE_NOTES.get(
            mode,
            "無法判定投資組合階段；本次結果僅供資料檢查。",
        )
        summary_note = (
            "僅列涉及資金異動的 STARTER、ADD、STRONG_ADD、TRIM、"
            "ROTATE_25/50/75、FAST_TRACK_ROTATE 與 EXIT 類訊號。"
        )
        return """
        <section data-section="portfolio-allocation" style="margin:16px 0;padding:12px;border:1px solid #bfd0dc;border-left:5px solid #295f7a;background:#f5f9fb">
          <h3 style="margin:0 0 6px;color:#204f66">投資組合配置（PAE v{}）</h3>
          <div style="margin-bottom:9px;color:#3f5662">
            <strong>{}</strong>：{} PAE 消費 LES／EREM v3 已發布決策，不重算技術指標。
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px">
            {}
            {}
            {}
            {}
            {}
            {}
          </div>
          <div style="margin-bottom:8px;font-size:12px;color:#526770">
            分析 {} 檔｜進出場訊號 {}｜已篩選 {}｜{}
          </div>
          {}
        </section>
        """.format(
            html.escape(model_version),
            html.escape(mode or "DATA_INSUFFICIENT"),
            html.escape(mode_note),
            self._metric(
                "必要報酬",
                self._percent(first.value("PAE Required Return")),
            ),
            self._metric(
                "目標現金",
                self._percent(first.value("PAE Target Cash Weight")),
            ),
            self._metric(
                "組合 CER",
                self._percent(first.value("PAE Portfolio Expected CER")),
            ),
            self._metric("進出場訊號", str(len(selected))),
            self._metric("狀態", self._text(first.value("PAE Status"))),
            self._metric(
                "約束權限" if model_version.startswith("2.") else "持股基準",
                self._text(first.value("PAE Binding"))
                if model_version.startswith("2.")
                else self._text(first.value("PAE Portfolio Input")),
            ),
            len(pae_decisions),
            len(selected),
            len(pae_decisions) - len(selected),
            summary_note,
            self._table(selected),
        )


# Preserve the established import path while the renderer itself is version-neutral.
PortfolioAllocationV1Renderer = PortfolioAllocationRenderer


__all__ = [
    "PortfolioAllocationFormatter",
    "PortfolioAllocationRenderer",
    "PortfolioAllocationTableRenderer",
    "PortfolioAllocationV1Renderer",
]
