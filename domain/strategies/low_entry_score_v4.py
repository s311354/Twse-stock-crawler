"""Low Entry Model v4 with fundamental, veto, trigger, and risk gates."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.features import SharedFeaturePipeline
from domain.fundamentals import GateStatus
from domain.fundamentals import QualityGateConfig
from domain.fundamentals import QualityGateResult
from domain.fundamentals import evaluate_quality_gate
from domain.industry_profiles import IndustryProfileMode
from domain.strategy import Strategy

LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v4"
ANNUAL_FUNDAMENTALS_ATTR = "les_v4_annual_fundamentals"
FUNDAMENTAL_SNAPSHOT_ATTR = "les_v4_fundamental_snapshot"
FUNDAMENTAL_STATUS_ATTR = "les_v4_fundamental_status"
FUNDAMENTAL_STATUS_DETAIL_ATTR = "les_v4_fundamental_status_detail"
SIGNAL_DATE_ATTR = "les_v4_signal_date"
ACCOUNT_EQUITY_ATTR = "les_v4_account_equity"
PRICE_SCALE_ATTR = "les_v4_price_scale"
LES_V4_MODEL_VERSION = "4.1.0"
LES_V4_PARAMETER_VERSION = "2026-07-21-qs-dc-v2"

_VETO_OUTPUT_COLUMNS = {
    "new_20_day_low": "Veto 創20日新低",
    "accelerating_downtrend": "Veto 下降趨勢加速",
    "high_volume_bearish_bar": "Veto 放量長黑",
    "repeated_breakdown": "Veto 連續破底",
    "destructive_gap": "Veto 跳空破壞結構",
    "taiex_risk_off": "Veto 大盤Risk-Off",
    "insufficient_liquidity": "Veto 流動性不足",
    "negative_volatility_expansion": "Veto 負向波動擴張",
    "stop_too_far": "Veto 停損距離過遠",
}

LOW_ENTRY_OUTPUT_COLUMNS = [
    "低點分數",
    "低點決策",
    "Quality Gate",
    "Quality Score",
    "Industry Quality Profile",
    "Industry Profile Version",
    "Industry Profile Mode",
    "Data Confidence",
    "Available Metric Weight",
    "DC Coverage",
    "DC Freshness",
    "DC Validation",
    "DC Source Reliability",
    "Quality QA Flags",
    "Quality Reasons",
    "Quality Hard Failures",
    "ROIC 5Y Median",
    "Positive CFO Year Ratio",
    "Gross Margin 5Y Percentile",
    "Operating Margin Trend",
    "Fundamental Available At",
    "Fundamental Availability Source",
    "Fundamental Filing ID",
    "Fundamental Revision",
    "Fundamental SHA256",
    "Fundamental Source URL",
    "Fundamental Correction Announced On",
    "Fundamental Derivation Version",
    "Market Data Gate",
    "Market Data Reasons",
    "No Veto",
    "Falling Knife Veto",
    *_VETO_OUTPUT_COLUMNS.values(),
    "LES v4 Trigger",
    "Risk Gate",
    "Risk Gate Reasons",
    "市場相對趨勢分數",
    "合理價格區分數",
    "止跌結構分數",
    "買盤確認分數",
    "波動風報比分數",
    "流動性執行分數",
    "EMA20",
    "EMA20 Slope",
    "ATR14",
    "CLV",
    "CMF20",
    "Volume Ratio",
    "Amount MA20",
    "Spread",
    "VWAP20",
    "RS20",
    "60日位置",
    "Drawdown60",
    "RewardRisk",
    "Entry Candidate",
    "Entry Rule",
    "策略停損",
    "策略停利",
    "建議股數",
    "訊號有效交易日",
    "Signal Date",
    "LES Model Version",
    "LES Parameter Version",
    "Risk Percent",
    "Maximum Stop ATR",
    "Minimum RewardRisk",
    "低點理由",
]

_VETO_LABELS = {
    "new_20_day_low": "創20日新低",
    "accelerating_downtrend": "下降趨勢加速",
    "high_volume_bearish_bar": "放量長黑",
    "repeated_breakdown": "連續破底",
    "destructive_gap": "跳空破壞結構",
    "taiex_risk_off": "大盤Risk-Off",
    "insufficient_liquidity": "流動性不足",
    "negative_volatility_expansion": "異常負向波動擴張",
    "stop_too_far": "停損距離過遠",
}


@dataclass(frozen=True)
class LowEntryScoreV4Config:
    allow_timing_only_buy: bool = False
    strict_market_risk_off: bool = True
    atr_window: int = 14
    rolling_window: int = 20
    range_window: int = 60
    minimum_amount_ma20: float = 30_000_000.0
    maximum_spread: float = 0.01
    maximum_order_amount_ratio: float = 0.05
    maximum_stop_atr: float = 1.5
    minimum_reward_risk: float = 2.0
    account_equity: float = 1_000_000.0
    risk_percent: float = 0.005
    lot_size: int = 1000
    armed_sessions: int = 5
    epsilon: float = 1e-9
    quality: QualityGateConfig = QualityGateConfig()


def attach_low_entry_v4_context(
    data: pd.DataFrame,
    *,
    annual_fundamentals: pd.DataFrame,
    fundamental_snapshot: dict[str, Any],
    signal_date: str | pd.Timestamp,
    account_equity: float | None = None,
    price_scale: float = 1.0,
) -> pd.DataFrame:
    """Attach point-in-time MOPS facts without coupling Strategy to storage."""

    result = data.copy()
    result.attrs[ANNUAL_FUNDAMENTALS_ATTR] = annual_fundamentals.copy()
    result.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] = dict(fundamental_snapshot)
    result.attrs[SIGNAL_DATE_ATTR] = signal_date
    result.attrs[PRICE_SCALE_ATTR] = price_scale
    if account_equity is not None:
        result.attrs[ACCOUNT_EQUITY_ATTR] = account_equity
    return result


class LowEntryScoreV4Strategy(Strategy):
    """Quality Gate + No Veto + Score + Trigger + Risk Gate entry model."""

    name = LOW_ENTRY_STRATEGY_NAME

    def __init__(self, config: LowEntryScoreV4Config | None = None) -> None:
        self.config = config or LowEntryScoreV4Config()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        features = SharedFeaturePipeline(
            self._shared_feature_config()
        ).build(data)
        return self.run_with_features(data, features)

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self._empty_result()])

        context = dict(data.attrs)
        df = data.copy()
        indicators = self._calculate_indicators(df, features)
        market_data_gate, market_data_reasons = self._evaluate_market_data(
            df,
            indicators,
            context,
        )
        plan = self._build_trade_plan(indicators, context)
        vetoes = self._evaluate_vetoes(indicators, plan)
        components = self._score_components(indicators, plan)
        score = components.sum(axis=1).clip(lower=0, upper=100)
        trigger = self._evaluate_trigger(indicators)
        risk_gate, risk_reasons = self._evaluate_risk_gate(
            indicators,
            plan,
            market_data_gate,
        )
        quality = self._evaluate_quality(df, context)
        state = self._resolve_state(
            quality=quality,
            vetoes=vetoes,
            score=score,
            trigger=trigger,
            risk_gate=risk_gate,
            market_data_gate=market_data_gate,
            indicators=indicators,
        )

        veto_text = vetoes.apply(self._format_vetoes, axis=1)
        quality_reasons = "；".join(quality.reasons + quality.warnings)
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        signal_date = context.get(SIGNAL_DATE_ATTR)
        veto_outputs = {
            output_column: vetoes[veto_name].astype(int)
            for veto_name, output_column in _VETO_OUTPUT_COLUMNS.items()
        }
        result = pd.DataFrame(
            {
                "低點分數": score.round(2),
                "低點決策": state,
                "Quality Gate": quality.status.value,
                "Quality Score": quality.score,
                "Industry Quality Profile": quality.industry_profile,
                "Industry Profile Version": quality.industry_profile_version,
                "Industry Profile Mode": quality.industry_profile_mode,
                "Data Confidence": quality.data_confidence,
                "Available Metric Weight": quality.available_metric_weight,
                "DC Coverage": quality.confidence_components.get("coverage"),
                "DC Freshness": quality.confidence_components.get("freshness"),
                "DC Validation": quality.confidence_components.get("validation"),
                "DC Source Reliability": quality.confidence_components.get(
                    "source_reliability"
                ),
                "Quality QA Flags": "；".join(quality.qa_flags) or "NONE",
                "Quality Reasons": quality_reasons,
                "Quality Hard Failures": (
                    "；".join(quality.hard_red_flags) or "NONE"
                ),
                "ROIC 5Y Median": quality.metrics.get("roic_5y_median"),
                "Positive CFO Year Ratio": quality.metrics.get(
                    "positive_cfo_year_ratio"
                ),
                "Gross Margin 5Y Percentile": quality.metrics.get(
                    "gross_margin_5y_percentile"
                ),
                "Operating Margin Trend": quality.metrics.get(
                    "operating_margin_trend"
                ),
                "Audit Opinion": snapshot.get("audit_opinion"),
                "Accounting Concern Classification": snapshot.get(
                    "accounting_concern_classification"
                ),
                "Fundamental Available At": snapshot.get("available_at"),
                "Fundamental Availability Source": snapshot.get("availability_source"),
                "Fundamental Filing ID": snapshot.get("filing_id"),
                "Fundamental Revision": snapshot.get("revision_number"),
                "Fundamental SHA256": snapshot.get("content_sha256"),
                "Fundamental Source URL": snapshot.get("source_url"),
                "Fundamental Correction Announced On": snapshot.get(
                    "correction_announced_on"
                ),
                "Fundamental Derivation Version": snapshot.get(
                    "derivation_version"
                ),
                "Market Data Gate": market_data_gate,
                "Market Data Reasons": market_data_reasons,
                "No Veto": (~vetoes.any(axis=1)).astype(int),
                "Falling Knife Veto": veto_text,
                **veto_outputs,
                "LES v4 Trigger": trigger.astype(int),
                "Risk Gate": risk_gate,
                "Risk Gate Reasons": risk_reasons,
                "市場相對趨勢分數": components["market_relative_trend"],
                "合理價格區分數": components["reasonable_price_zone"],
                "止跌結構分數": components["stabilization"],
                "買盤確認分數": components["demand_confirmation"],
                "波動風報比分數": components["risk_reward"],
                "流動性執行分數": components["liquidity_execution"],
                "EMA20": indicators["ema20"].round(2),
                "EMA20 Slope": indicators["ema20_slope"].round(4),
                "ATR14": indicators["atr14"].round(2),
                "CLV": indicators["clv"].round(4),
                "CMF20": indicators["cmf20"].round(4),
                "Volume Ratio": indicators["volume_ratio"].round(4),
                "Amount MA20": indicators["amount_ma20"].round(2),
                "Spread": indicators["spread"].round(4),
                "VWAP20": indicators["vwap20"].round(2),
                "RS20": indicators["rs20"].round(4),
                "60日位置": indicators["price_position60"].round(4),
                "Drawdown60": indicators["drawdown60"].round(4),
                "RewardRisk": plan["reward_risk"].round(4),
                "Entry Candidate": plan["entry"].round(2),
                "Entry Rule": "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                "策略停損": plan["stop"].round(2),
                "策略停利": plan["target"].round(2),
                "建議股數": plan["shares"].astype("Int64"),
                "訊號有效交易日": np.where(
                    state == "ARMED",
                    self.config.armed_sessions,
                    np.where(state == "BUY", 1, 0),
                ),
                "Signal Date": signal_date,
                "LES Model Version": LES_V4_MODEL_VERSION,
                "LES Parameter Version": LES_V4_PARAMETER_VERSION,
                "Risk Percent": self.config.risk_percent,
                "Maximum Stop ATR": self.config.maximum_stop_atr,
                "Minimum RewardRisk": self.config.minimum_reward_risk,
                "低點理由": self._build_reasons(
                    quality=quality,
                    components=components,
                    veto_text=veto_text,
                    trigger=trigger,
                    risk_gate=risk_gate,
                    risk_reasons=risk_reasons,
                    state=state,
                ),
            },
            index=df.index,
        )
        return pd.concat([df, result], axis=1)

    def _evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        """Validate the completed, aligned daily bars required by the specification."""

        reasons: list[str] = []
        required_columns = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "TAIEX Close",
        }
        missing_columns = sorted(required_columns - set(data.columns))
        if missing_columns:
            reasons.append("MISSING_COLUMNS={}".format(missing_columns))
        if len(data) < self.config.range_window:
            reasons.append("INSUFFICIENT_COMPLETED_BARS")

        dates = (
            pd.to_datetime(data.get("Date"), errors="coerce")
            if "Date" in data
            else pd.Series(dtype="datetime64[ns]")
        )
        if len(dates) != len(data) or dates.isna().any():
            reasons.append("INVALID_TRADE_DATES")
        elif not dates.is_monotonic_increasing or dates.duplicated().any():
            reasons.append("UNORDERED_OR_DUPLICATE_TRADE_DATES")

        signal_date = pd.to_datetime(context.get(SIGNAL_DATE_ATTR), utc=True, errors="coerce")
        if pd.isna(signal_date):
            reasons.append("INVALID_SIGNAL_DATE")
        elif not dates.empty and not dates.isna().any():
            signal_taipei = signal_date.tz_convert("Asia/Taipei")
            latest_trade_date = dates.iloc[-1].date()
            if latest_trade_date > signal_taipei.date():
                reasons.append("BAR_AFTER_SIGNAL_DATE")
            elif (
                latest_trade_date == signal_taipei.date()
                and signal_taipei.time() < dt.time(13, 30)
            ):
                reasons.append("LATEST_BAR_NOT_COMPLETED")

        invalid_ohlc = (
            indicators[["open", "high", "low", "close"]].isna().any(axis=1)
            | (indicators[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (indicators["high"] < indicators[["open", "close"]].max(axis=1))
            | (indicators["low"] > indicators[["open", "close"]].min(axis=1))
            | (indicators["high"] < indicators["low"])
        )
        if invalid_ohlc.any():
            reasons.append("INVALID_OHLC")
        if indicators["volume"].isna().any() or (indicators["volume"] < 0).any():
            reasons.append("INVALID_VOLUME")
        if indicators["taiex"].isna().any() or (indicators["taiex"] <= 0).any():
            reasons.append("UNALIGNED_TAIEX_HISTORY")

        status = "PASS" if not reasons else "FAIL"
        reason_text = "PASS" if not reasons else "；".join(reasons)
        return (
            pd.Series(status, index=data.index),
            pd.Series(reason_text, index=data.index),
        )

    def _calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame | None = None,
    ) -> pd.DataFrame:
        shared = features or SharedFeaturePipeline(
            self._shared_feature_config()
        ).build(data)
        shared.assert_compatible(data)
        if shared.config != self._shared_feature_config():
            raise ValueError(
                "FeatureFrame config does not match Low Entry strategy"
            )
        feature_data = shared.to_pandas()
        open_price = feature_data["open"]
        high = feature_data["high"]
        low = feature_data["low"]
        close = feature_data["close"]
        volume = feature_data["volume"]
        amount = feature_data["amount"]
        trades = feature_data["trades"]
        bid = feature_data["bid"]
        ask = feature_data["ask"]
        pe = feature_data["pe"]
        taiex = feature_data["taiex"]

        ema5 = feature_data["ema5_strict"]
        ema10 = feature_data["ema10_strict"]
        ema20 = feature_data["ema20_strict"]
        ema50 = feature_data["ema50_strict"]
        taiex_ema20 = feature_data["taiex_ema20_strict"]
        r1 = feature_data["r1"]
        r5 = feature_data["r5"]
        taiex_r5 = feature_data["taiex_r5"]
        rs20 = feature_data["r20"] - feature_data["taiex_r20"]

        previous_close = close.shift(1)
        atr14 = feature_data["atr14_wilder"]
        atr_percent = feature_data["atr_percent_wilder"]
        atr_percent_ma20 = feature_data["atr_percent_wilder_ma20"]
        ema20_slope = self._safe_series_div(ema20 - ema20.shift(5), 5 * atr14)

        day_range = high - low
        clv = self._safe_series_div(((close - low) - (high - close)), day_range).fillna(0)
        volume_ma20 = feature_data["volume_ma20_strict"]
        amount_ma20 = amount.rolling(self.config.rolling_window, min_periods=self.config.rolling_window).mean()
        trades_ma20 = trades.rolling(self.config.rolling_window, min_periods=self.config.rolling_window).mean()
        volume_ratio = self._safe_series_div(volume, volume_ma20)
        amount_ratio = self._safe_series_div(amount, amount_ma20)
        trades_ratio = self._safe_series_div(trades, trades_ma20)
        cmf20 = self._safe_series_div(
            (clv * volume).rolling(self.config.rolling_window, min_periods=self.config.rolling_window).sum(),
            feature_data["volume_sum20_strict"],
        )
        obv = feature_data["obv"]
        normalized_obv_slope5 = self._safe_series_div(obv - obv.shift(5), 5 * volume_ma20)

        vwap20 = feature_data["vwap20_typical"]
        mid_quote = (ask + bid) / 2
        spread = self._safe_series_div(ask - bid, mid_quote).where(ask >= bid)
        amihud = self._safe_series_div(r1.abs(), amount)
        amihud20 = amihud.rolling(self.config.rolling_window, min_periods=self.config.rolling_window).mean()
        gap = self._safe_series_div(open_price, previous_close) - 1
        normalized_gap = self._safe_series_div(gap, atr_percent.shift(1))

        highest_high60 = high.rolling(self.config.range_window, min_periods=self.config.range_window).max()
        lowest_low60 = low.rolling(self.config.range_window, min_periods=self.config.range_window).min()
        max_close60 = close.rolling(self.config.range_window, min_periods=self.config.range_window).max()
        price_position60 = self._safe_series_div(close - lowest_low60, highest_high60 - lowest_low60)
        drawdown60 = self._safe_series_div(close, max_close60) - 1
        pe_lower_bound60 = pe.where(pe > 0).rolling(self.config.range_window, min_periods=20).quantile(0.35)

        previous_low20 = low.shift(1).rolling(self.config.rolling_window, min_periods=self.config.rolling_window).min()
        new_20_day_low = close < previous_low20
        swing_candidate = (
            (low < low.shift(1))
            & (low < low.shift(2))
            & (low <= low.shift(-1))
            & (low <= low.shift(-2))
        )
        confirmed_swing_event = low.where(swing_candidate).shift(2)
        previous_confirmed_event = confirmed_swing_event.ffill().shift(1)
        higher_low_event = confirmed_swing_event > previous_confirmed_event + (0.2 * atr14)
        recent_confirmed_swing_low = confirmed_swing_event.ffill()
        higher_low = higher_low_event.where(confirmed_swing_event.notna()).ffill().fillna(False).astype(bool)

        return pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "trades": trades,
                "pe": pe,
                "taiex": taiex,
                "ema5": ema5,
                "ema10": ema10,
                "ema20": ema20,
                "ema50": ema50,
                "ema5_slope_positive": ema5 > ema5.shift(1),
                "taiex_ema20": taiex_ema20,
                "r1": r1,
                "r5": r5,
                "taiex_r5": taiex_r5,
                "rs20": rs20,
                "atr14": atr14,
                "atr_percent": atr_percent,
                "atr_percent_ma20": atr_percent_ma20,
                "ema20_slope": ema20_slope,
                "clv": clv,
                "volume_ratio": volume_ratio,
                "amount_ratio": amount_ratio,
                "trades_ratio": trades_ratio,
                "amount_ma20": amount_ma20,
                "cmf20": cmf20,
                "cmf20_improving": (cmf20 > cmf20.shift(1)) & (cmf20 >= -0.05),
                "normalized_obv_slope5": normalized_obv_slope5,
                "vwap20": vwap20,
                "spread": spread,
                "amihud20": amihud20,
                "gap": gap,
                "normalized_gap": normalized_gap,
                "highest_high60": highest_high60,
                "price_position60": price_position60,
                "drawdown60": drawdown60,
                "pe_lower_bound60": pe_lower_bound60,
                "previous_low20": previous_low20,
                "new_20_day_low": new_20_day_low,
                "recent_confirmed_swing_low": recent_confirmed_swing_low,
                "higher_low": higher_low,
            },
            index=data.index,
        )

    def _build_trade_plan(self, indicators: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        entry = indicators["high"]
        stop = indicators["recent_confirmed_swing_low"] - (0.3 * indicators["atr14"])
        target = indicators["high"].shift(1).rolling(self.config.range_window, min_periods=20).max()
        risk = entry - stop
        reward_risk = self._safe_series_div(target - entry, risk).where(risk > 0)

        account_equity = self._positive_float(context.get(ACCOUNT_EQUITY_ATTR), self.config.account_equity)
        price_scale = self._positive_float(context.get(PRICE_SCALE_ATTR), 1.0)
        risk_per_share = risk / price_scale
        raw_shares = (account_equity * self.config.risk_percent) / risk_per_share.where(risk_per_share > 0)
        lots = np.floor(raw_shares / self.config.lot_size)
        shares = pd.Series(lots * self.config.lot_size, index=indicators.index).where(raw_shares.notna())
        order_amount = shares * (entry / price_scale)

        return pd.DataFrame(
            {
                "entry": entry,
                "stop": stop,
                "target": target,
                "risk": risk,
                "reward_risk": reward_risk,
                "shares": shares,
                "order_amount": order_amount,
            },
            index=indicators.index,
        )

    def _evaluate_vetoes(self, indicators: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
        new_low_count3 = indicators["new_20_day_low"].astype(int).rolling(3, min_periods=3).sum()
        average_clv3 = indicators["clv"].rolling(3, min_periods=3).mean()
        known_liquidity_failure = (
            (indicators["amount_ma20"].notna() & (indicators["amount_ma20"] < self.config.minimum_amount_ma20))
            | (indicators["spread"].notna() & (indicators["spread"] > self.config.maximum_spread))
            | (
                plan["order_amount"].notna()
                & indicators["amount_ma20"].notna()
                & (plan["order_amount"] > self.config.maximum_order_amount_ratio * indicators["amount_ma20"])
            )
        )
        return pd.DataFrame(
            {
                "new_20_day_low": indicators["new_20_day_low"].fillna(False),
                "accelerating_downtrend": (
                    (indicators["ema20_slope"] < -0.15)
                    & (indicators["r5"] < -1.5 * indicators["atr_percent"])
                ).fillna(False),
                "high_volume_bearish_bar": (
                    (indicators["close"] < indicators["open"])
                    & (indicators["clv"] < -0.5)
                    & (indicators["volume_ratio"] > 1.5)
                ).fillna(False),
                "repeated_breakdown": ((new_low_count3 >= 2) & (average_clv3 < 0)).fillna(False),
                "destructive_gap": (
                    (indicators["normalized_gap"] < -1.0)
                    & (indicators["close"] < indicators["close"].shift(1))
                ).fillna(False),
                "taiex_risk_off": (
                    self.config.strict_market_risk_off
                    & (indicators["taiex"] < indicators["taiex_ema20"])
                    & (indicators["taiex_r5"] < -0.03)
                ).fillna(False),
                "insufficient_liquidity": known_liquidity_failure.fillna(False),
                "negative_volatility_expansion": (
                    (indicators["atr_percent"] > 1.5 * indicators["atr_percent_ma20"])
                    & (indicators["r5"] < 0)
                ).fillna(False),
                "stop_too_far": (
                    plan["risk"].notna()
                    & (plan["risk"] > self.config.maximum_stop_atr * indicators["atr14"])
                ).fillna(False),
            },
            index=indicators.index,
        )

    def _score_components(self, indicators: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
        support_distance = self._safe_series_div(
            indicators["close"] - indicators["recent_confirmed_swing_low"],
            indicators["atr14"],
        )
        no_new_low3 = indicators["new_20_day_low"].astype(int).rolling(3, min_periods=3).sum() == 0
        atr_not_expanding = indicators["atr_percent"] <= indicators["atr_percent_ma20"]
        target_distance = plan["target"] - plan["entry"]
        amihud_limit = indicators["amihud20"].rolling(self.config.range_window, min_periods=20).quantile(0.8)

        market_relative_trend = (
            4 * (indicators["taiex"] > indicators["taiex_ema20"]).astype(int)
            + 5 * (indicators["rs20"] > 0).astype(int)
            + 3 * (indicators["ema20_slope"] >= -0.05).astype(int)
            + 3 * (indicators["close"] > indicators["ema20"]).astype(int)
        )
        reasonable_price_zone = (
            5 * indicators["price_position60"].between(0.10, 0.35, inclusive="both").astype(int)
            + 4 * indicators["drawdown60"].between(-0.25, -0.08, inclusive="both").astype(int)
            + 4 * (
                (indicators["pe"] > 0)
                & indicators["pe_lower_bound60"].notna()
                & (indicators["pe"] <= indicators["pe_lower_bound60"])
            ).astype(int)
            + 2 * (
                indicators["recent_confirmed_swing_low"].notna()
                & (support_distance >= 0)
                & (support_distance <= 1.0)
            ).astype(int)
        )
        stabilization = (
            5 * no_new_low3.astype(int)
            + 6 * indicators["higher_low"].astype(int)
            + 6 * (indicators["close"] > indicators["high"].shift(1)).astype(int)
            + 4 * indicators["ema5_slope_positive"].astype(int)
            + 4
            * (
                (indicators["close"] > indicators["vwap20"])
                | (indicators["close"] > indicators["ema10"])
            ).astype(int)
        )
        demand_confirmation = (
            4 * (indicators["clv"] > 0.4).astype(int)
            + 4 * indicators["cmf20_improving"].astype(int)
            + 3 * (
                (indicators["close"] > indicators["close"].shift(1))
                & indicators["volume_ratio"].between(0.9, 1.8, inclusive="both")
                & (indicators["clv"] >= 0)
            ).astype(int)
            + 4 * (indicators["normalized_obv_slope5"] > 0).astype(int)
        )
        risk_reward = (
            4 * atr_not_expanding.astype(int)
            + 5
            * (
                (plan["risk"] > 0)
                & (plan["risk"] <= self.config.maximum_stop_atr * indicators["atr14"])
            ).astype(int)
            + 6 * (plan["reward_risk"] >= self.config.minimum_reward_risk).astype(int)
            + 3 * (target_distance >= indicators["atr14"]).astype(int)
            + 2 * (indicators["normalized_gap"] <= 0.5).astype(int)
        )
        liquidity_execution = (
            4 * (indicators["amount_ma20"] >= self.config.minimum_amount_ma20).astype(int)
            + 3 * (indicators["spread"] <= self.config.maximum_spread).astype(int)
            + 2 * indicators["trades_ratio"].between(0.5, 2.0, inclusive="both").astype(int)
            + 1 * (indicators["amihud20"].notna() & (indicators["amihud20"] <= amihud_limit)).astype(int)
        )
        return pd.DataFrame(
            {
                "market_relative_trend": market_relative_trend,
                "reasonable_price_zone": reasonable_price_zone,
                "stabilization": stabilization,
                "demand_confirmation": demand_confirmation,
                "risk_reward": risk_reward,
                "liquidity_execution": liquidity_execution,
            },
            index=indicators.index,
        )

    def _evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series:
        return (
            (
                (indicators["close"] > indicators["high"].shift(1))
                | ((indicators["close"] > indicators["ema5"]) & (indicators["close"] > indicators["vwap20"]))
            )
            & (indicators["close"] > indicators["open"])
            & (indicators["clv"] > 0.4)
            & indicators["cmf20_improving"]
            & (indicators["volume_ratio"] >= 0.9)
        ).fillna(False)

    def _evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_data_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        liquidity_known = (
            indicators["amount_ma20"].notna()
            & indicators["spread"].notna()
            & plan["order_amount"].notna()
        )
        liquidity_pass = (
            liquidity_known
            & (indicators["amount_ma20"] >= self.config.minimum_amount_ma20)
            & (indicators["spread"] <= self.config.maximum_spread)
            & (plan["order_amount"] <= self.config.maximum_order_amount_ratio * indicators["amount_ma20"])
        )
        market_history_known = indicators[
            ["taiex", "taiex_ema20", "taiex_r5", "rs20", "price_position60", "drawdown60"]
        ].notna().all(axis=1)
        plan_known = (
            plan[["entry", "stop", "target", "risk", "reward_risk", "shares"]].notna().all(axis=1)
            & indicators["atr14"].notna()
            & liquidity_known
            & market_history_known
            & (market_data_gate == "PASS")
        )
        risk_pass = (
            plan_known
            & (plan["risk"] > 0)
            & (plan["risk"] <= self.config.maximum_stop_atr * indicators["atr14"])
            & (plan["reward_risk"] >= self.config.minimum_reward_risk)
            & (plan["shares"] > 0)
            & liquidity_pass
        )
        known_failure = plan_known & ~risk_pass
        status = pd.Series(
            np.select([risk_pass, known_failure], ["PASS", "FAIL"], default="UNKNOWN"),
            index=indicators.index,
        )

        reason_frame = pd.DataFrame(
            {
                "NO_CONFIRMED_SWING_LOW": plan["stop"].isna(),
                "INVALID_STOP": plan["risk"].notna() & (plan["risk"] <= 0),
                "STOP_TOO_FAR": plan["risk"].notna()
                & (plan["risk"] > self.config.maximum_stop_atr * indicators["atr14"]),
                "INVALID_TARGET": plan["target"].isna() | (plan["target"] <= plan["entry"]),
                "INSUFFICIENT_REWARD_RISK": plan["reward_risk"].notna()
                & (plan["reward_risk"] < self.config.minimum_reward_risk),
                "LIQUIDITY_UNKNOWN": ~liquidity_known,
                "LIQUIDITY_FAILED": liquidity_known & ~liquidity_pass,
                "MARKET_HISTORY_INSUFFICIENT": ~market_history_known,
                "MARKET_DATA_INVALID": market_data_gate != "PASS",
                "POSITION_NOT_EXECUTABLE": plan["shares"].notna() & (plan["shares"] <= 0),
            },
            index=indicators.index,
        )
        reasons = reason_frame.apply(
            lambda row: "；".join(row.index[row].tolist()) if row.any() else "PASS",
            axis=1,
        )
        return status, reasons

    def _evaluate_quality(self, data: pd.DataFrame, context: dict[str, Any]) -> QualityGateResult:
        annual = context.get(ANNUAL_FUNDAMENTALS_ATTR)
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        signal_date = context.get(SIGNAL_DATE_ATTR)
        unavailable_status = str(context.get(FUNDAMENTAL_STATUS_ATTR, "")).strip()
        unavailable_detail = str(
            context.get(FUNDAMENTAL_STATUS_DETAIL_ATTR, "")
        ).strip()
        if unavailable_status and not isinstance(annual, pd.DataFrame):
            warnings = ["MOPS_STATUS={}".format(unavailable_status)]
            if unavailable_detail:
                warnings.append(unavailable_detail)
            return QualityGateResult(
                status=GateStatus.UNKNOWN,
                score=None,
                reasons=["MOPS point-in-time 基本面資料不可用"],
                warnings=warnings,
            )
        if signal_date is None and "Date" in data.columns and not data["Date"].empty:
            signal_date = data["Date"].iloc[-1]
        if not isinstance(annual, pd.DataFrame):
            annual = pd.DataFrame()
        if not isinstance(snapshot, dict):
            snapshot = {}
        if signal_date is None:
            return QualityGateResult(
                status=GateStatus.UNKNOWN,
                score=None,
                reasons=["缺少可重現的訊號日期"],
            )
        return evaluate_quality_gate(annual, snapshot, signal_date, self.config.quality)

    def _resolve_state(
        self,
        *,
        quality: QualityGateResult,
        vetoes: pd.DataFrame,
        score: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        market_data_gate: pd.Series,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        if quality.status == GateStatus.REJECT:
            return pd.Series("REJECT", index=score.index)
        if quality.status == GateStatus.SECTOR_UNSUPPORTED:
            return pd.Series("NO_BUY_MODEL_UNSUPPORTED", index=score.index)
        if quality.status == GateStatus.UNKNOWN:
            return pd.Series("DATA_INSUFFICIENT", index=score.index)
        if quality.industry_profile_mode == IndustryProfileMode.SHADOW.value:
            blocked = vetoes.any(axis=1) | (market_data_gate != "PASS")
            return pd.Series(np.where(blocked, "NO_BUY", "WATCH_ONLY"), index=score.index)
        if quality.status in {GateStatus.PROVISIONAL, GateStatus.WATCH_QUALITY}:
            return pd.Series("WATCH_ONLY", index=score.index)

        vetoed = vetoes.any(axis=1)
        market_risk_off = (indicators["taiex"] < indicators["taiex_ema20"]) & (indicators["taiex_r5"] < -0.03)
        if self.config.strict_market_risk_off:
            flexible_buy_allowed = pd.Series(True, index=score.index)
            buy_threshold = pd.Series(80, index=score.index)
        else:
            flexible_buy_allowed = (~market_risk_off) | (indicators["rs20"] > 0)
            buy_threshold = pd.Series(np.where(market_risk_off, 85, 80), index=score.index)
        buy_candidate = (score >= buy_threshold) & trigger & flexible_buy_allowed

        return pd.Series(
            np.select(
                [
                    vetoed,
                    market_data_gate != "PASS",
                    score < 60,
                    score < 70,
                    buy_candidate & (risk_gate == "FAIL"),
                    buy_candidate & (risk_gate == "PASS"),
                ],
                ["NO_BUY", "NO_BUY", "AVOID", "WATCH", "NO_BUY", "BUY"],
                default="ARMED",
            ),
            index=score.index,
        )

    def _build_reasons(
        self,
        *,
        quality: QualityGateResult,
        components: pd.DataFrame,
        veto_text: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        risk_reasons: pd.Series,
        state: pd.Series,
    ) -> pd.Series:
        quality_evidence = "；".join(quality.reasons + quality.warnings)
        quality_text = "Quality={}(QS={}；DC={}；{})".format(
            quality.status.value,
            quality.score if quality.score is not None else "N/A",
            quality.data_confidence,
            quality_evidence,
        )
        detail = pd.DataFrame(
            {
                "market": components["market_relative_trend"],
                "price": components["reasonable_price_zone"],
                "stabilization": components["stabilization"],
                "demand": components["demand_confirmation"],
                "risk": components["risk_reward"],
                "liquidity": components["liquidity_execution"],
                "veto": veto_text,
                "trigger": trigger,
                "risk_gate": risk_gate,
                "risk_reasons": risk_reasons,
                "state": state,
            },
            index=components.index,
        )
        return detail.apply(
            lambda row: (
                "{}；分數={:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}；"
                "Veto={}；Trigger={}；Risk={}({})；決策={}"
            ).format(
                quality_text,
                row["market"],
                row["price"],
                row["stabilization"],
                row["demand"],
                row["risk"],
                row["liquidity"],
                row["veto"],
                "YES" if row["trigger"] else "NO",
                row["risk_gate"],
                row["risk_reasons"],
                row["state"],
            ),
            axis=1,
        )

    def _shared_feature_config(self) -> SharedFeatureConfig:
        return SharedFeatureConfig(
            atr_window=self.config.atr_window,
            atr_ma_window=self.config.rolling_window,
            volume_window=self.config.rolling_window,
            vwap_short_window=self.config.rolling_window,
            epsilon=self.config.epsilon,
        )

    def _numeric_column(self, data: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in data.columns:
                return pd.to_numeric(data[name], errors="coerce")
        return pd.Series(np.nan, index=data.index, dtype="float64")

    def _safe_series_div(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        valid_denominator = denominator.where(denominator.abs() > self.config.epsilon)
        return numerator / valid_denominator

    def _format_vetoes(self, row: pd.Series) -> str:
        labels = [_VETO_LABELS[name] for name in row.index[row.astype(bool)]]
        return "；".join(labels) if labels else "NONE"

    def _positive_float(self, value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if np.isfinite(parsed) and parsed > 0 else fallback

    def _empty_result(self) -> dict[str, object]:
        result = {column: np.nan for column in LOW_ENTRY_OUTPUT_COLUMNS}
        result.update(
            {
                "低點分數": 0,
                "低點決策": "DATA_INSUFFICIENT",
                "Quality Gate": GateStatus.UNKNOWN.value,
                "Industry Quality Profile": "UNSUPPORTED",
                "Industry Profile Version": "N/A",
                "Industry Profile Mode": "UNSUPPORTED",
                "Data Confidence": 0,
                "Available Metric Weight": 0,
                "Quality Reasons": "基本面與日線資料不足",
                "Market Data Gate": "FAIL",
                "Market Data Reasons": "DATA_INSUFFICIENT",
                "No Veto": 0,
                "Falling Knife Veto": "N/A",
                "LES v4 Trigger": 0,
                "Risk Gate": "UNKNOWN",
                "Risk Gate Reasons": "DATA_INSUFFICIENT",
                "訊號有效交易日": 0,
                "LES Model Version": LES_V4_MODEL_VERSION,
                "LES Parameter Version": LES_V4_PARAMETER_VERSION,
                "Risk Percent": self.config.risk_percent,
                "Maximum Stop ATR": self.config.maximum_stop_atr,
                "Minimum RewardRisk": self.config.minimum_reward_risk,
                "低點理由": "資料不足；決策=DATA_INSUFFICIENT",
            }
        )
        return result


__all__ = [
    "ACCOUNT_EQUITY_ATTR",
    "ANNUAL_FUNDAMENTALS_ATTR",
    "FUNDAMENTAL_STATUS_ATTR",
    "FUNDAMENTAL_STATUS_DETAIL_ATTR",
    "FUNDAMENTAL_SNAPSHOT_ATTR",
    "LOW_ENTRY_OUTPUT_COLUMNS",
    "LOW_ENTRY_STRATEGY_NAME",
    "LES_V4_MODEL_VERSION",
    "LES_V4_PARAMETER_VERSION",
    "LowEntryScoreV4Config",
    "LowEntryScoreV4Strategy",
    "PRICE_SCALE_ATTR",
    "SIGNAL_DATE_ATTR",
    "attach_low_entry_v4_context",
]
