from copy import deepcopy
from math import exp

import pytest

from domain.les_v7.fill_calibration import deploy_fill_pair
from domain.les_v7.fill_calibration_selection import select_shrinkage


FIT='2026-06-01T13:30:00+08:00'


def rows(count=500,one_fold=False,pair=(.8,-.5)):
    result=[]
    for i in range(count):
        first=i<count//2 or one_fold
        fit='2026-03-02T13:30:00+08:00' if first else '2026-04-20T13:30:00+08:00'
        cutoff='2026-03-03T13:30:00+08:00' if first else '2026-04-21T13:30:00+08:00'
        available='2026-03-31T13:30:00+08:00' if first else '2026-05-20T13:30:00+08:00'
        candidates={}
        for alpha,key in ((.25,'0.25'),(.5,'0.5'),(1.,'1.0')):
            deployed=deploy_fill_pair(pair,alpha)
            candidates[key]=1/(1+exp(-deployed[1]))
        result.append(dict(row_id=str(i),fold_id='march' if first else 'april',
            cutoff=cutoff,label_available_at=available,observation_weight=1.,labels={'fill':i%2},
            raw_logits={'fill':0.},head_fit_cutoff=fit,head_last_label_available_at=fit,
            calibration_fit_cutoff=fit,calibration_last_label_available_at=fit,
            prior_fill_pair=list(pair),fill_shrink_candidates=candidates))
    return result


def test_two_mature_prequential_folds_select_real_shrinkage():
    result=select_shrinkage(rows(),fit_cutoff=FIT)
    assert result['alpha']==.25
    assert result['report']['status']=='SELECTED_FROM_PAST_PREQUENTIAL'
    assert result['report']['candidate_brier']['0.25']<result['report']['candidate_brier']['1.0']


@pytest.mark.parametrize('sample',[[],rows(499),rows(500,one_fold=True)])
def test_declared_selection_warmup_does_not_claim_an_optimizer_fit(sample):
    result=select_shrinkage(sample,fit_cutoff=FIT)
    assert result['alpha']==1.
    assert result['report']['status']=='DECLARED_SELECTION_WARMUP_ALPHA'
    assert result['report']['actual_current_calibrator_fit_required_separately'] is True


def test_equal_scores_prefer_current_alpha_one_and_order_is_irrelevant():
    sample=rows(pair=(1.,0.))
    forward=select_shrinkage(sample,fit_cutoff=FIT)
    backward=select_shrinkage(list(reversed(sample)),fit_cutoff=FIT)
    assert forward==backward
    assert forward['alpha']==1.


@pytest.mark.parametrize('field,value',[
    ('label_available_at','2026-06-02T13:30:00+08:00'),
    ('calibration_last_label_available_at','2026-03-03T13:30:00+08:00'),
    ('calibration_fit_cutoff','2026-03-03T13:30:00+08:00'),
    ('head_last_label_available_at','2026-03-03T13:30:00+08:00')])
def test_future_or_mismatched_fit_provenance_is_rejected(field,value):
    sample=rows();sample[0][field]=value
    with pytest.raises(ValueError,match='STRICTLY_PAST_MATURE'):
        select_shrinkage(sample,fit_cutoff=FIT)


def test_candidate_values_cannot_be_forged_or_extra_alphas_searched():
    sample=rows();sample[0]['fill_shrink_candidates']['0.25']=.1
    with pytest.raises(ValueError,match='FORMULA_MISMATCH'):
        select_shrinkage(sample,fit_cutoff=FIT)
    sample=rows();sample[0]['fill_shrink_candidates']['0.01']=.5
    with pytest.raises(ValueError,match='EXACT_CANDIDATE_GRID'):
        select_shrinkage(sample,fit_cutoff=FIT)


def test_duplicate_or_renamed_fold_cannot_inflate_prequential_support():
    sample=rows();sample[1]['fold_id']='fake-second-fold'
    with pytest.raises(ValueError,match='DISTINCT_FOLD_FITS'):
        select_shrinkage(sample,fit_cutoff=FIT)
    sample=rows();sample[1]['row_id']=sample[0]['row_id']
    with pytest.raises(ValueError,match='UNIQUE_ROW'):
        select_shrinkage(sample,fit_cutoff=FIT)
