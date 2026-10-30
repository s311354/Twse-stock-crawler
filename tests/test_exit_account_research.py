import datetime as dt
from dataclasses import replace

import pandas as pd
import pytest

from application.exit_account_research import recompute_exit
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioAllocationInputError
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationRequestV2
from infrastructure.storage.research_portfolio import SQLiteResearchPortfolioPersistence


def history():
    return pd.DataFrame(dict(Date=pd.date_range('2025-01-01',periods=260).strftime('%Y-%m-%d'),
        Open=[100.]*260,High=[101.]*260,Low=[99.]*260,Close=[100.]*260,
        Volume=[10000.]*260,**{'TAIEX Close':[20000.]*260}))


def test_exit_recomputed_from_actual_entry_not_old_account():
    data=history()
    position=dict(entryDate=data.iloc[-1].Date,entryLES=90.,entryDecision='BUY_SMALL')
    original=data.copy(deep=True)
    a=recompute_exit(data,position)
    b=recompute_exit(data,dict(position,entryLES=20.,entryDate='2025-01-01'))
    assert a['EREM Entry LES']==90 and b['EREM Entry LES']==20
    assert a['EREM Holding Days']==0 and b['EREM Holding Days']==259
    assert a['EREM Holding Confidence'] != b['EREM Holding Confidence']
    pd.testing.assert_frame_equal(data,original)


def test_soft_policy_preserves_native_scores_and_hard_exit():
    position=dict(entryDate='2025-01-01',entryLES=20.,entryDecision='BUY_SMALL')
    a=recompute_exit(history(),position)
    b=recompute_exit(history(),position,soft_reduction=True)
    assert b['EREM Native Decision']==a['EREM 決策訊號']
    assert b['EREM Holding Confidence']==a['EREM Holding Confidence']
    if a['EREM 決策訊號']=='EXIT':assert b['EREM 決策訊號']=='EXIT'


def test_native_pae_public_boundary_is_account_scoped_and_idempotent(tmp_path):
    store=SQLiteResearchPortfolioPersistence(tmp_path/'pae.sqlite3',{'account':'synthetic-A'})
    pipeline=PortfolioAllocationV2Pipeline(persistence=store,portfolio_id='synthetic-A')
    request=PortfolioAllocationRequestV2(portfolio_id='synthetic-A',
        decision_time=dt.datetime(2026,10,26,tzinfo=dt.timezone.utc),
        account_equity=1000000.,available_cash=1000000.)
    first=pipeline.evaluate_request(request,dt.date(2026,10,26))
    assert pipeline.evaluate_request(request,dt.date(2026,10,26))==first
    with pytest.raises(PortfolioAllocationInputError,match='PORTFOLIO_ID'):
        pipeline.evaluate_request(replace(request,portfolio_id='synthetic-B'),dt.date(2026,10,26))
