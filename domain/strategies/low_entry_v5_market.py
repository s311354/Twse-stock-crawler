"""Market-data, breadth, indicator and execution-risk rules for LES v5."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.industry_profiles import ELECTRONICS_SECTOR_CODES
from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_BREADTH_HISTORY_ATTR
from domain.market_breadth import MARKET_DATA_THROUGH_ATTR
from domain.market_breadth import MARKET_SESSION_FINAL_ATTR
from domain.market_breadth import normalize_market_breadth_snapshot
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_v5_contracts import BroadRallyTier


class LowEntryV5MarketAdapter(Protocol):
    def shared_feature_config(self) -> SharedFeatureConfig: ...

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame: ...

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]: ...

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame: ...

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]: ...

    def safe_series_div(
        self,
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series: ...

    def positive_float(self, value: Any, fallback: float) -> float: ...


class LowEntryV5MarketEvaluator:
    """Own all market-facing calculations used by the LES v5 pipeline."""

    def __init__(self, config: Any, market_adapter: LowEntryV5MarketAdapter) -> None:
        self.config = config
        self.market_adapter = market_adapter

    def shared_feature_config(self) -> SharedFeatureConfig:
        return self.market_adapter.shared_feature_config()

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        gate, reasons = self.market_adapter.evaluate_market_data(
            data,
            indicators,
            context,
        )
        snapshot = normalize_market_breadth_snapshot(
            context.get(MARKET_BREADTH_ATTR)
        )
        breadth_history = context.get(MARKET_BREADTH_HISTORY_ATTR)
        breadth_history = (
            dict(breadth_history)
            if isinstance(breadth_history, dict)
            else {}
        )
        indicators["taiex_r1"] = indicators["taiex"].pct_change(
            fill_method=None
        )
        fundamental_snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        sector = (
            str(fundamental_snapshot.get("sector") or "").strip()
            if isinstance(fundamental_snapshot, dict)
            else ""
        )
        # An unknown classification cannot opt into an industry-specific path.
        indicators["is_electronics"] = pd.Series(
            sector in ELECTRONICS_SECTOR_CODES if sector else pd.NA,
            index=indicators.index,
            dtype="boolean",
        )
        self._attach_breadth_facts(indicators, snapshot)
        indicators["broad_rally_tier"] = self._evaluate_breadth_rally_tier(
            data,
            indicators,
            breadth_snapshot=snapshot,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            market_session_final=bool(
                context.get(MARKET_SESSION_FINAL_ATTR, False)
            ),
        )
        indicators["broad_rally_regime"] = (
            indicators["broad_rally_tier"] == BroadRallyTier.STRONG.value
        )
        selloff_regime, selloff_count = self._evaluate_recent_breadth_regime(
            data,
            indicators,
            breadth_history=breadth_history,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            direction="selloff",
        )
        rally_regime, rally_count = self._evaluate_recent_breadth_regime(
            data,
            indicators,
            breadth_history=breadth_history,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            direction="rally",
        )
        indicators["recent_broad_selloff_regime"] = selloff_regime
        indicators["recent_selloff_sessions"] = selloff_count
        indicators["recent_broad_rally_regime"] = rally_regime
        indicators["recent_rally_sessions"] = rally_count
        return gate, reasons

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.market_adapter.build_trade_plan(indicators, context)

    def build_broad_rally_plan(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        price_scale = self.market_adapter.positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        entry = self._next_twse_stock_tick(
            indicators["high"],
            price_scale=price_scale,
        )
        repriced = plan.copy()
        repriced["entry"] = entry
        signal_bar_stop = (
            indicators["low"]
            - self.config.breadth_rally_signal_stop_buffer_atr
            * indicators["atr14"]
        )
        repriced["stop"] = pd.concat(
            [repriced["stop"], signal_bar_stop],
            axis=1,
        ).max(axis=1)
        repriced["risk"] = entry - repriced["stop"]
        repriced["reward_risk"] = self.market_adapter.safe_series_div(
            repriced["target"] - entry,
            repriced["risk"],
        ).where(repriced["risk"] > 0)

        account_equity = self.market_adapter.positive_float(
            context.get(ACCOUNT_EQUITY_ATTR),
            self.config.account_equity,
        )
        risk_per_share = repriced["risk"] / price_scale
        raw_shares = (
            account_equity * self.config.risk_percent
        ) / risk_per_share.where(risk_per_share > 0)
        lots = np.floor(raw_shares / self.config.lot_size)
        repriced["shares"] = pd.Series(
            lots * self.config.lot_size,
            index=plan.index,
        ).where(raw_shares.notna())
        repriced["order_amount"] = (
            repriced["shares"] * (repriced["entry"] / price_scale)
        )
        return repriced

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self.market_adapter.evaluate_risk_gate(
            indicators,
            plan,
            market_gate,
        )

    def evaluate_broad_rally_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        liquidity_known = (
            indicators["amount_ma20"].notna()
            & indicators["spread"].notna()
            & plan["order_amount"].notna()
        )
        executable_order_amount = (
            plan["order_amount"]
            * self.config.breadth_rally_full_buy_position_fraction
        )
        liquidity_pass = (
            liquidity_known
            & (
                indicators["amount_ma20"]
                >= self.config.breadth_rally_minimum_amount_ma20
            )
            & (indicators["spread"] <= self.config.maximum_spread)
            & (
                executable_order_amount
                <= self.config.maximum_order_amount_ratio
                * indicators["amount_ma20"]
            )
        )
        market_history_known = indicators[
            ["taiex", "taiex_ema20", "taiex_r5", "rs20", "price_position60"]
        ].notna().all(axis=1)
        plan_known = (
            plan[
                ["entry", "stop", "target", "risk", "reward_risk", "shares"]
            ].notna().all(axis=1)
            & indicators["atr14"].notna()
            & liquidity_known
            & market_history_known
            & (market_gate == "PASS")
        )
        passed = (
            plan_known
            & (plan["risk"] > 0)
            & (
                plan["risk"]
                <= self.config.breadth_rally_maximum_stop_atr
                * indicators["atr14"]
            )
            & (
                plan["reward_risk"]
                >= self.config.breadth_rally_minimum_reward_risk
            )
            & (plan["shares"] > 0)
            & liquidity_pass
        )
        known_failure = plan_known & ~passed
        status = pd.Series(
            np.select(
                [passed, known_failure],
                ["PASS", "FAIL"],
                default="UNKNOWN",
            ),
            index=indicators.index,
        )
        reason_frame = pd.DataFrame(
            {
                "NO_CONFIRMED_SWING_LOW": plan["stop"].isna(),
                "INVALID_STOP": plan["risk"].notna() & (plan["risk"] <= 0),
                "STOP_TOO_FAR": plan["risk"].notna()
                & (
                    plan["risk"]
                    > self.config.breadth_rally_maximum_stop_atr
                    * indicators["atr14"]
                ),
                "INVALID_TARGET": plan["target"].isna()
                | (plan["target"] <= plan["entry"]),
                "INSUFFICIENT_REWARD_RISK": plan["reward_risk"].notna()
                & (
                    plan["reward_risk"]
                    < self.config.breadth_rally_minimum_reward_risk
                ),
                "LIQUIDITY_UNKNOWN": ~liquidity_known,
                "LIQUIDITY_FAILED": liquidity_known & ~liquidity_pass,
                "MARKET_HISTORY_INSUFFICIENT": ~market_history_known,
                "MARKET_DATA_INVALID": market_gate != "PASS",
                "POSITION_NOT_EXECUTABLE": plan["shares"].notna()
                & (plan["shares"] <= 0),
            },
            index=indicators.index,
        )
        reasons = reason_frame.apply(
            lambda row: (
                "；".join(row.index[row].tolist()) if row.any() else "PASS"
            ),
            axis=1,
        )
        return status, reasons

    @staticmethod
    def _next_twse_stock_tick(
        prices: pd.Series,
        *,
        price_scale: float,
    ) -> pd.Series:
        actual = pd.to_numeric(prices, errors="coerce") / price_scale
        tick = pd.Series(
            np.select(
                [
                    actual < 10,
                    actual < 50,
                    actual < 100,
                    actual < 500,
                    actual < 1000,
                ],
                [0.01, 0.05, 0.10, 0.50, 1.00],
                default=5.00,
            ),
            index=prices.index,
            dtype=float,
        )
        next_price = (
            (np.floor((actual / tick) + 1e-9) + 1)
            * tick
            * price_scale
        )
        return next_price.where(actual.notna())

    @staticmethod
    def _attach_breadth_facts(
        indicators: pd.DataFrame,
        snapshot: dict[str, object],
    ) -> None:
        indicators["breadth_as_of"] = str(snapshot.get("as_of") or "")
        indicators["breadth_universe"] = str(
            snapshot.get("universe") or "UNKNOWN"
        ).upper()
        for output, source in (
            ("breadth_coverage", "coverage"),
            ("breadth_valid_members", "valid_return_count"),
            ("sector_advancer_ratio", "advancer_ratio"),
            ("sector_median_r1", "median_return"),
        ):
            indicators[output] = pd.to_numeric(
                snapshot.get(source),
                errors="coerce",
            )

    def _evaluate_breadth_rally_regime(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_snapshot: dict[str, object],
        market_data_through: object,
        market_session_final: bool,
    ) -> pd.Series:
        tier = self._evaluate_breadth_rally_tier(
            data,
            indicators,
            breadth_snapshot=breadth_snapshot,
            market_data_through=market_data_through,
            market_session_final=market_session_final,
        )
        return tier.eq(BroadRallyTier.STRONG.value)

    def _evaluate_breadth_rally_tier(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_snapshot: dict[str, object],
        market_data_through: object,
        market_session_final: bool,
    ) -> pd.Series:
        none = pd.Series(BroadRallyTier.NONE.value, index=indicators.index)
        if not breadth_snapshot or "Date" not in data:
            return none

        universe = str(breadth_snapshot.get("universe") or "").upper()
        if universe == "SEMI":
            minimum_members = self.config.semiconductor_breadth_minimum_members
            strong_advancer_ratio = (
                self.config.semiconductor_breadth_minimum_advancer_ratio
            )
            strong_median_return = (
                self.config.semiconductor_breadth_minimum_median_return
            )
            moderate_advancer_ratio = (
                self.config.semiconductor_moderate_breadth_minimum_advancer_ratio
            )
            moderate_median_return = (
                self.config.semiconductor_moderate_breadth_minimum_median_return
            )
        elif universe == "ELEC":
            minimum_members = self.config.electronics_breadth_minimum_members
            strong_advancer_ratio = (
                self.config.electronics_breadth_minimum_advancer_ratio
            )
            strong_median_return = (
                self.config.electronics_breadth_minimum_median_return
            )
            moderate_advancer_ratio = (
                self.config.electronics_moderate_breadth_minimum_advancer_ratio
            )
            moderate_median_return = (
                self.config.electronics_moderate_breadth_minimum_median_return
            )
        else:
            return none

        coverage = pd.to_numeric(breadth_snapshot.get("coverage"), errors="coerce")
        members = pd.to_numeric(
            breadth_snapshot.get("valid_return_count"),
            errors="coerce",
        )
        advancers = pd.to_numeric(
            breadth_snapshot.get("advancer_ratio"),
            errors="coerce",
        )
        median_return = pd.to_numeric(
            breadth_snapshot.get("median_return"),
            errors="coerce",
        )
        breadth_as_of = str(breadth_snapshot.get("as_of") or "")
        market_through = str(market_data_through or "")
        evidence_gate = bool(
            market_session_final
            and breadth_snapshot.get("source_final") is True
            and breadth_as_of
            and breadth_as_of == market_through
            and pd.notna(coverage)
            and coverage >= self.config.breadth_rally_minimum_coverage
            and pd.notna(members)
            and members >= minimum_members
            and pd.notna(advancers)
            and pd.notna(median_return)
        )
        if not evidence_gate:
            return none

        taiex_return = pd.to_numeric(
            breadth_snapshot.get("taiex_return"),
            errors="coerce",
        )
        row_dates = pd.to_datetime(data["Date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        if pd.notna(taiex_return):
            indicators.loc[row_dates == breadth_as_of, "taiex_r1"] = float(
                taiex_return
            )
        aligned = (
            (row_dates == breadth_as_of)
            & (
                indicators["taiex_r1"]
                >= self.config.breadth_rally_minimum_taiex_return
            )
        ).fillna(False)
        strong = bool(
            advancers >= strong_advancer_ratio
            and median_return >= strong_median_return
        )
        moderate = bool(
            advancers >= moderate_advancer_ratio
            and median_return >= moderate_median_return
        )
        tier = none.copy()
        if moderate:
            tier.loc[aligned] = BroadRallyTier.MODERATE.value
        if strong:
            tier.loc[aligned] = BroadRallyTier.STRONG.value
        return tier

    def _evaluate_recent_breadth_regime(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_history: dict[str, object],
        market_data_through: object,
        direction: str,
    ) -> tuple[pd.Series, pd.Series]:
        false = pd.Series(False, index=indicators.index)
        zero = pd.Series(0, index=indicators.index, dtype=int)
        if direction not in {"selloff", "rally"}:
            raise ValueError("direction must be selloff or rally")
        if not breadth_history or "Date" not in data:
            return false, zero

        signal_date = pd.to_datetime(market_data_through, errors="coerce")
        if pd.isna(signal_date):
            return false, zero

        usable: list[tuple[pd.Timestamp, dict[str, object]]] = []
        for key, raw_snapshot in breadth_history.items():
            snapshot = normalize_market_breadth_snapshot(raw_snapshot)
            snapshot_date = pd.to_datetime(
                snapshot.get("as_of") or key,
                errors="coerce",
            )
            if pd.isna(snapshot_date) or snapshot_date >= signal_date:
                continue
            if snapshot.get("source_final") is not True:
                continue
            usable.append((snapshot_date.normalize(), snapshot))

        usable.sort(key=lambda item: item[0])
        recent = usable[-self.config.breadth_memory_lookback_sessions :]
        if len(recent) < self.config.breadth_memory_minimum_sessions:
            return false, zero

        matching_sessions = 0
        for _, snapshot in recent:
            universe = str(snapshot.get("universe") or "").upper()
            if universe == "SEMI":
                minimum_members = self.config.semiconductor_breadth_minimum_members
            elif universe == "ELEC":
                minimum_members = self.config.electronics_breadth_minimum_members
            else:
                continue
            coverage = pd.to_numeric(snapshot.get("coverage"), errors="coerce")
            members = pd.to_numeric(
                snapshot.get("valid_return_count"), errors="coerce"
            )
            advancers = pd.to_numeric(
                snapshot.get("advancer_ratio"), errors="coerce"
            )
            median_return = pd.to_numeric(
                snapshot.get("median_return"), errors="coerce"
            )
            taiex_return = pd.to_numeric(
                snapshot.get("taiex_return"), errors="coerce"
            )
            data_valid = bool(
                pd.notna(coverage)
                and coverage >= self.config.breadth_rally_minimum_coverage
                and pd.notna(members)
                and members >= minimum_members
                and pd.notna(advancers)
                and pd.notna(median_return)
                and pd.notna(taiex_return)
            )
            if not data_valid:
                continue
            if direction == "selloff":
                matches = bool(
                    advancers
                    <= self.config.panic_selloff_maximum_advancer_ratio
                    and median_return
                    <= self.config.panic_selloff_maximum_median_return
                    and taiex_return
                    <= self.config.panic_selloff_maximum_taiex_return
                )
            else:
                breadth_led_recovery = bool(
                    advancers
                    >= self.config.recent_rally_lead_minimum_advancer_ratio
                    and median_return
                    >= self.config.recent_rally_lead_minimum_median_return
                )
                matches = bool(
                    advancers >= self.config.recent_rally_minimum_advancer_ratio
                    and median_return
                    >= self.config.recent_rally_minimum_median_return
                    and (
                        taiex_return
                        >= self.config.recent_rally_minimum_taiex_return
                        or breadth_led_recovery
                    )
                )
            matching_sessions += int(matches)

        row_dates = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        current_row = row_dates == signal_date.normalize()
        regime = current_row & (
            matching_sessions >= self.config.breadth_memory_minimum_sessions
        )
        count = zero.copy()
        count.loc[current_row] = matching_sessions
        return regime.fillna(False), count

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        indicators = self.market_adapter.calculate_indicators(data, features)
        shared = features.to_pandas()
        close = indicators["close"]
        low = indicators["low"]
        high = indicators["high"]
        open_price = indicators["open"]
        true_range = shared["true_range"]
        lower_shadow = open_price.combine(close, min) - low

        indicators["rsi14"] = shared["rsi14_wilder"]
        indicators["macd"] = shared["macd_strict"]
        indicators["macd_signal"] = shared["macd_signal_strict"]
        indicators["macd_histogram"] = shared["macd_histogram_strict"]
        indicators["true_range"] = true_range
        indicators["drawdown20"] = (
            self.market_adapter.safe_series_div(
                close,
                close.rolling(20, min_periods=20).max(),
            )
            - 1
        )
        indicators["pe_median60"] = (
            indicators["pe"]
            .where(indicators["pe"] > 0)
            .rolling(self.config.range_window, min_periods=20)
            .median()
        )
        indicators["ema5_slope"] = self.market_adapter.safe_series_div(
            indicators["ema5"] - indicators["ema5"].shift(1),
            indicators["atr14"],
        )
        indicators["rs20_slope"] = indicators["rs20"] - indicators["rs20"].shift(5)
        indicators["taiex_ema20_slope"] = (
            indicators["taiex_ema20"] - indicators["taiex_ema20"].shift(5)
        )
        taiex_return = indicators["taiex"].pct_change(fill_method=None)
        indicators["taiex_volatility20"] = taiex_return.rolling(
            20,
            min_periods=20,
        ).std(ddof=0)
        indicators["taiex_volatility_ma20"] = indicators[
            "taiex_volatility20"
        ].rolling(20, min_periods=20).mean()
        indicators["previous_high"] = high.shift(1)
        indicators["previous_midpoint"] = (high.shift(1) + low.shift(1)) / 2
        indicators["lower_shadow_ratio"] = self.market_adapter.safe_series_div(
            lower_shadow,
            true_range,
        )
        indicators["new_low_count3"] = (
            indicators["new_20_day_low"]
            .fillna(False)
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
        )
        indicators["weak_clv3"] = (
            (indicators["clv"] < 0)
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
            >= 3
        )
        return indicators


__all__ = ["LowEntryV5MarketAdapter", "LowEntryV5MarketEvaluator"]
