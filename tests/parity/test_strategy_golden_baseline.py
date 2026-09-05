"""Golden strategy outputs protected during behavior-preserving refactors."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from domain.strategies.expected_return_exit_v3 import EREM_V3_MODEL_VERSION
from domain.strategies.expected_return_exit_v3 import (
    EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS,
)
from domain.strategies.expected_return_exit_v3 import (
    ExpectedReturnExitV3Strategy,
)
from domain.strategies.portfolio_allocation_v2 import (
    PortfolioAllocationEngineV2,
)
from domain.strategies.portfolio_allocation_v2 import (
    PortfolioAllocationRequestV2,
)
from domain.strategies.portfolio_allocation_v2 import PortfolioSecurityV2


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "strategy_golden_v1.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _erem_history() -> pd.DataFrame:
    count = 240
    closes = [
        100 + index * 0.12 + math.sin(index / 4) * 0.8
        for index in range(count)
    ]
    closes[-1] = closes[-2] + 1
    return pd.DataFrame(
        {
            "Open": [close - 0.5 for close in closes],
            "High": [close + 3 for close in closes],
            "Low": [close - 8 for close in closes],
            "Close": closes,
            "Volume": [1000.0] * (count - 1) + [2500.0],
            "TAIEX Close": [100 + index * 0.03 for index in range(count)],
            "Entry LES": [math.nan] * (count - 30)
            + [80.0]
            + [math.nan] * 29,
            "Holding Days": [math.nan] * (count - 30) + list(range(30)),
            "Turnover": [1.0] * (count - 1) + [2.5],
        }
    )


def _erem_observation() -> dict[str, object]:
    latest = ExpectedReturnExitV3Strategy().run(_erem_history()).iloc[-1]
    fields = (
        "EREM 分數",
        "EREM Raw Score",
        "EREM Holding Confidence",
        "EREM Entry Weight",
        "EREM 決策訊號",
        "EREM 建議減倉比例",
        "EREM Relative Strength60",
        "EREM RewardRisk",
        "EREM Trend Score",
        "EREM Momentum Score",
        "EREM Volatility Score",
        "EREM Distribution Score",
        "EREM Remaining Return Score",
        "EREM Relative Strength Score",
        "EREM Structure Score",
        "EREM Entry Quality Score",
        "EREM Turnover Score",
        "EREM VWAP Score",
        "EREM Turnover Source",
    )
    observed = {field: latest[field] for field in fields}
    observed.update(
        {
            "model_version": EREM_V3_MODEL_VERSION,
            "output_columns_sha256": _digest(
                EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS
            ),
            "reason_sha256": hashlib.sha256(
                str(latest["EREM 理由"]).encode("utf-8")
            ).hexdigest(),
        }
    )
    return observed


def _pae_v2_observation() -> dict[str, object]:
    candidate = PortfolioSecurityV2(
        stock_id="2330",
        sector_id="24",
        correlation_cluster_id="SEMI",
        quality_status="PASS",
        quality_score=88.0,
        quality_confidence=90.0,
        data_confidence=90.0,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=0.12,
        base_cagr=0.18,
        bull_cagr=0.24,
        valuation_confidence=90.0,
        les_decision="BUY_SMALL",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        liquidity_status="PASS",
        executable_shares=1000,
        les_position_cap=0.10,
        risk_budget_weight=0.10,
        liquidity_weight_cap=0.10,
        erem_decision="HOLD",
        erem_holding_confidence=75.0,
        erem_data_sufficient=True,
        current_price=100.0,
        stop_price=90.0,
    )
    request = PortfolioAllocationRequestV2(
        portfolio_id="baseline-owner",
        decision_time=dt.datetime(
            2026,
            8,
            30,
            13,
            30,
            tzinfo=dt.timezone(dt.timedelta(hours=8)),
        ),
        account_equity=1_000_000,
        available_cash=1_000_000,
        candidates=(candidate,),
    )
    decision = PortfolioAllocationEngineV2().allocate(request)
    return {
        "schema_version": decision.schema_version,
        "model_version": decision.model_version,
        "mode": decision.mode.value,
        "binding": decision.binding.value,
        "required_return": decision.required_return,
        "holding_count_before": decision.holding_count_before,
        "holding_count_after": decision.holding_count_after,
        "cash_weight_before": decision.cash_weight_before,
        "cash_weight_after": decision.cash_weight_after,
        "reason_codes": list(decision.reason_codes),
        "lines": [
            {
                "stock_id": line.stock_id,
                "role": line.role,
                "action": line.action.value,
                "current_weight": line.current_weight,
                "target_weight": line.target_weight,
                "capital_claim": round(line.evaluation.capital_claim, 12),
                "strategic_candidate": (
                    line.evaluation.strategic_candidate
                ),
                "executable_candidate": (
                    line.evaluation.executable_candidate
                ),
                "reason_codes": list(line.reason_codes),
                "blocking_reasons": list(line.blocking_reasons),
            }
            for line in decision.lines
        ],
    }


def test_erem_v3_golden_behavior_is_stable() -> None:
    assert _erem_observation() == _fixture()["erem_v3_trend"]


def test_pae_v2_golden_behavior_is_stable() -> None:
    assert _pae_v2_observation() == _fixture()["pae_v2_direct_buy_small"]
