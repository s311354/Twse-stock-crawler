"""Select fixed shrinkage using only matured, earlier prequential forecasts.

This function does not fit a calibrator. The controller must successfully fit
the current native identity-prior model and create/recompute every candidate
forecast from actual earlier native fits. No holdout or identity fallback.
"""
from __future__ import annotations

from math import exp, fsum

from domain.les_v7.contracts import finite, identity, timestamp
from domain.les_v7.fill_calibration import deploy_fill_pair, profile as prior_profile

VERSION='V7_FILL_PREQUENTIAL_SHRINK_SELECTION_R1'
ALPHAS=(.25,.5,1.)
ALPHA_KEYS=('0.25','0.5','1.0')
MIN_FOLDS=2
MIN_ROWS=500
MAX_ROWS=100000
TIE_TOLERANCE=1e-12


def profile():
    return dict(version=VERSION,alpha_candidates=list(ALPHAS),minimum_mature_inner_folds=MIN_FOLDS,
        minimum_mature_inner_rows=MIN_ROWS,maximum_rows=MAX_ROWS,
        warmup_alpha=1.,tie_tolerance=TIE_TOLERANCE,tie_break='LARGER_ALPHA',
        objective='OBSERVATION_WEIGHTED_MATURE_EARLIER_PREQUENTIAL_FILL_BRIER',
        candidate_generation='ACTUAL_NATIVE_PRIOR_FIT_THEN_DEPLOY_FILL_PAIR',
        prior_fit_profile=prior_profile(),requires_actual_current_prior_fit=True,
        no_solver_failure_fallback=True,no_raw_identity_candidate=True,
        warmup_is_selection_rule_not_calibrator_fallback=True)


def _number(value,name):
    if isinstance(value,bool):
        raise ValueError('SHRINK_SELECTION_FINITE_NUMBER_REQUIRED:'+name)
    return finite(value,name)


def _sigmoid(value):
    return 1./(1.+exp(-value)) if value>=0 else exp(value)/(1.+exp(value))


def select_shrinkage(rows,*,fit_cutoff):
    if not isinstance(rows,(list,tuple)) or len(rows)>MAX_ROWS:
        raise ValueError('SHRINK_SELECTION_BOUNDED_PREQUENTIAL_ROWS_REQUIRED')
    fit_at=timestamp(fit_cutoff);seen=set();records=[];folds={};fit_folds={}
    for row in rows:
        key=row.get('row_id');fold=row.get('fold_id')
        if (not isinstance(key,str) or not key or key in seen
                or not isinstance(fold,str) or not fold):
            raise ValueError('SHRINK_SELECTION_UNIQUE_ROW_AND_FOLD_IDS_REQUIRED')
        seen.add(key)
        cutoff,available,head_fit,head_last,cal_fit,cal_last=(timestamp(row[name]) for name in
            ('cutoff','label_available_at','head_fit_cutoff','head_last_label_available_at',
             'calibration_fit_cutoff','calibration_last_label_available_at'))
        if not head_last<=head_fit==cal_fit<cutoff<available<=fit_at or not cal_last<=cal_fit:
            raise ValueError('SHRINK_SELECTION_STRICTLY_PAST_MATURE_PREQUENTIAL_REQUIRED')
        if fold in folds and folds[fold]!=head_fit:
            raise ValueError('SHRINK_SELECTION_FOLD_FIT_BINDING_MISMATCH')
        if head_fit in fit_folds and fit_folds[head_fit]!=fold:
            raise ValueError('SHRINK_SELECTION_DISTINCT_FOLD_FITS_REQUIRED')
        folds[fold]=head_fit;fit_folds[head_fit]=fold
        fill=row.get('labels',{}).get('fill')
        if type(fill) is not int or fill not in (0,1):
            raise ValueError('SHRINK_SELECTION_OBSERVED_FILL_REQUIRED')
        weight=_number(row.get('observation_weight',1.),'observation_weight')
        if weight<=0:
            raise ValueError('SHRINK_SELECTION_POSITIVE_WEIGHT_REQUIRED')
        raw=_number(row.get('raw_logits',{}).get('fill'),'raw_fill_logit')
        pair=row.get('prior_fill_pair')
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:
            raise ValueError('SHRINK_SELECTION_ACTUAL_PRIOR_PAIR_REQUIRED')
        pair=[_number(value,'prior_pair') for value in pair]
        candidates=row.get('fill_shrink_candidates')
        if not isinstance(candidates,dict) or set(candidates)!=set(ALPHA_KEYS):
            raise ValueError('SHRINK_SELECTION_EXACT_CANDIDATE_GRID_REQUIRED')
        probabilities=[]
        for alpha,label in zip(ALPHAS,ALPHA_KEYS):
            probability=_number(candidates[label],'candidate_probability')
            deployed=deploy_fill_pair(pair,alpha)
            expected=_sigmoid(deployed[0]*raw+deployed[1])
            if not 0<=probability<=1 or abs(probability-expected)>1e-12:
                raise ValueError('SHRINK_SELECTION_CANDIDATE_FORMULA_MISMATCH')
            probabilities.append(probability)
        records.append(dict(row_id=key,fold_id=fold,cutoff=cutoff.isoformat(),
            label_available_at=available.isoformat(),head_fit_cutoff=head_fit.isoformat(),
            head_last_label_available_at=head_last.isoformat(),calibration_fit_cutoff=cal_fit.isoformat(),
            calibration_last_label_available_at=cal_last.isoformat(),fill=fill,
            observation_weight=weight,raw_fill_logit=raw,prior_fill_pair=pair,
            candidate_probabilities=probabilities))
    records.sort(key=lambda row:(row['cutoff'],row['row_id']))
    losses=None
    if records:
        largest=max(row['observation_weight'] for row in records)
        scaled=[row['observation_weight']/largest for row in records];total=fsum(scaled)
        weights=[value/total for value in scaled]
        if any(value<=0 for value in weights):
            raise ValueError('SHRINK_SELECTION_WEIGHT_DYNAMIC_RANGE_UNREPRESENTABLE')
        losses={label:fsum(weight*(row['candidate_probabilities'][index]-row['fill'])**2
            for row,weight in zip(records,weights)) for index,label in enumerate(ALPHA_KEYS)}
    ready=len(records)>=MIN_ROWS and len(folds)>=MIN_FOLDS
    if ready:
        best=min(losses.values())
        alpha=max(value for value,label in zip(ALPHAS,ALPHA_KEYS) if losses[label]<=best+TIE_TOLERANCE)
    else:
        alpha=1.
    report=dict(schema='les-v7-fill-prequential-shrink-selection/v1',profile=profile(),
        fit_cutoff=fit_at.isoformat(),row_count=len(records),mature_inner_folds=len(folds),
        fold_ids=sorted(folds),row_ids_sha256=identity([row['row_id'] for row in records]),
        data_sha256=identity(records),candidate_brier=losses,selected_alpha=alpha,
        status='SELECTED_FROM_PAST_PREQUENTIAL' if ready else 'DECLARED_SELECTION_WARMUP_ALPHA',
        last_label_available_at=max((row['label_available_at'] for row in records),default=None),
        last_calibration_fit_cutoff=max((row['calibration_fit_cutoff'] for row in records),default=None),
        interpretation='DEVELOPMENT_SELECTION_NOT_FRESH_CONFIRMATION_OR_MODEL_ACCEPTANCE',
        actual_current_calibrator_fit_required_separately=True,
        research_only=True,approved_for_live=False,execution_action='NO_ACTION')
    return dict(alpha=alpha,report=report)
