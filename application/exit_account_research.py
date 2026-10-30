"""Native account-dependent recalculation over a frozen retrospective signal tape.

This is not point-in-time LES reconstruction. Saved LES/Quality/valuation stay
fixed; authorization, EREM and PAE are recomputed for each independent account.
"""
from copy import deepcopy
import datetime as dt
import math

import pandas as pd

from application.low_entry_v7_daily import finalize_daily_report, project_report
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.strategy_contracts import StrategyDecisionAssembler
from domain.model_contracts import DecisionField, FieldSource, StrategyDecision
from domain.models import PortfolioPosition
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategies.erem_soft_reduction import SoftReductionResearchPolicy


VERSION = 'FROZEN_OPPORTUNITY_ACCOUNT_REEVOLUTION_R1'


def recompute_exit(frame, position, *, soft_reduction=False):
    """Full native warm-up, actual entry anchor and causal R1 prefix transform."""
    data = frame.copy(deep=True)
    dates = pd.to_datetime(data['Date'], errors='raise')
    ages = (dates.dt.normalize() - pd.Timestamp(position['entryDate'])).dt.days
    active = ages >= 0
    data['Holding Days'] = ages.where(active)
    data['Entry LES'] = pd.Series(float(position['entryLES']), index=data.index).where(active)
    data['Entry Decision'] = pd.Series(position['entryDecision'], index=data.index).where(active)
    strategy = ExpectedReturnExitV3Strategy(soft_reduction_policy=
        SoftReductionResearchPolicy() if soft_reduction else None)
    result = strategy.run(data).iloc[-1].to_dict()
    return {key: (None if pd.isna(value) else value.item() if hasattr(value, 'item') else value)
            for key, value in result.items()}


def scalar(value):
    if not isinstance(value, str):
        return value
    if value == '':
        return None
    if value in {'True', 'False'}:
        return value == 'True'
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return value


def rebind_security(report, csv_row, *, cash, equity, positions, erem, pipeline):
    """Reauthorize a verified native report and use the existing PAE adapter.

    The report's actual analysis clock is retained. It is never backdated to
    the market day. Source verification happened in the native CAS reader and
    report validator; CSV provenance is retained in the enclosing tape receipt.
    """
    report = deepcopy(report)
    account = dict(value=equity, available_cash=cash, intended_weight=.25,
        capacity_weight=.25, positions=[dict(symbol=c, shares=p['shares']) for c, p in sorted(positions.items())],
        as_of=report['envelope']['event']['cutoff_at'])
    report = finalize_daily_report(report, account=account,
        daily_input_errors=report['diagnostics'].get('daily_input_errors', []))
    row = {key: scalar(value) for key, value in csv_row.items() if not key.startswith('PAE ')}
    row.update(project_report(report))
    row.update(erem or {'EREM Holding Confidence': None, 'EREM 決策訊號': 'NO_POSITION', 'EREM 理由': ''})
    row['證券代號'] = report['symbol']
    row['證券名稱'] = report.get('name', csv_row['證券名稱'])
    row['日期'] = report['signal_date']
    at = dt.datetime.fromisoformat(account['as_of'])
    fields = tuple(DecisionField(name=k, value=v, source=(FieldSource.LES_V5_VALUATION
        if k == 'Valuation Model Version' else FieldSource.UNKNOWN), available_at=at,
        confidence=0., unit=StrategyDecisionAssembler._unit_for(k), model_version=VERSION)
        for k, v in row.items())
    decision = StrategyDecision(strategy_name=VERSION, evaluated_at=at, fields=fields)
    holdings = tuple(PortfolioPosition(symbol=c, shares=p['shares'], average_cost=p['average'],
        entry_les=p['entryLES'], entry_decision=p['entryDecision'],
        entry_date=dt.date.fromisoformat(p['entryDate']), sector='ELECTRONICS') for c,p in sorted(positions.items()))
    context = PortfolioRuntimeContext(account_equity=equity, current_cash=cash,
        positions=holdings, default_sector='ELECTRONICS')
    position = next((p for p in holdings if p.symbol == report['symbol']), None)
    security = pipeline.verified_security(decision, context, position=position)
    return report, row, security
