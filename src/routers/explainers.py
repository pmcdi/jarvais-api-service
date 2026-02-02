import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, Request

from ..config import settings
from ..modules.plot import (
    get_distribution_chart,
    get_partial_dependence_chart,
    get_permutation_importance_chart,
    get_shap_bar_chart,
    get_shap_dependence_chart,
    get_shap_summary_chart,
)
from ..modules.explain import (
    compute_pdp,
    compute_permutation_importance,
    compute_shap,
    load_trainer,
)
from ..utils.rate_limit import apply_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explainers", tags=["explainers"])


@router.get("/{trainer_id}/shap/summary")
@apply_rate_limit(settings.rate_limit_general)
async def shap_summary(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    top_n: int = Query(20, ge=1, le=50),
    sample_size: int = Query(200, ge=10, le=1000),
):
    """Return a SHAP summary (beeswarm-style) Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    shap_values, X_sample = compute_shap(trainer, sample_size=sample_size)
    return get_shap_summary_chart(shap_values, X_sample.columns, top_n=top_n)


@router.get("/{trainer_id}/shap/bar")
@apply_rate_limit(settings.rate_limit_general)
async def shap_bar(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    top_n: int = Query(20, ge=1, le=50),
    sample_size: int = Query(200, ge=10, le=1000),
):
    """Return a SHAP mean-absolute bar Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    shap_values, X_sample = compute_shap(trainer, sample_size=sample_size)
    return get_shap_bar_chart(shap_values, X_sample.columns, top_n=top_n)


@router.get("/{trainer_id}/shap/dependence")
@apply_rate_limit(settings.rate_limit_general)
async def shap_dependence(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    feature: str = Query(..., description="Feature for SHAP dependence"),
    interaction: Optional[str] = Query(None, description="Optional interaction feature for coloring"),
    sample_size: int = Query(200, ge=10, le=1000),
):
    """Return a SHAP dependence Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    shap_values, X_sample = compute_shap(trainer, sample_size=sample_size)

    if feature not in X_sample.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found")

    shap_matrix = np.asarray(getattr(shap_values, "values", shap_values))
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, 1 if shap_matrix.shape[1] > 1 else 0, :]
    feature_idx = list(X_sample.columns).index(feature)
    shap_feature_values = shap_matrix[:, feature_idx]

    interaction_values = X_sample[interaction] if interaction and interaction in X_sample.columns else None

    return get_shap_dependence_chart(
        X_sample[feature],
        shap_feature_values,
        feature,
        interaction_values=interaction_values,
    )


@router.get("/{trainer_id}/pdp")
@apply_rate_limit(settings.rate_limit_general)
async def partial_dependence(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    feature: str = Query(..., description="Feature to evaluate"),
    grid_size: int = Query(15, ge=3, le=50),
):
    """Return a partial dependence Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    grid, preds = compute_pdp(trainer, feature, grid_size=grid_size)
    return get_partial_dependence_chart(grid, preds, feature_name=feature, task=trainer.settings.task)


@router.get("/{trainer_id}/permutation_importance")
@apply_rate_limit(settings.rate_limit_general)
async def permutation_importance(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    top_n: int = Query(20, ge=1, le=100),
):
    """Return permutation feature importance Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    importance_df = compute_permutation_importance(trainer)
    return get_permutation_importance_chart(importance_df, top_n=top_n)


@router.get("/{trainer_id}/distribution")
@apply_rate_limit(settings.rate_limit_general)
async def distribution_comparison(
    request: Request,
    trainer_id: str = Path(..., description="Trainer identifier"),
    feature: str = Query(..., description="Feature to compare across train/test"),
):
    """Return train vs test distribution Highcharts configuration."""
    trainer = load_trainer(trainer_id)
    X_train = getattr(trainer, "X_train", None)
    X_test = getattr(trainer, "X_test", None)
    if X_train is None or X_test is None:
        raise HTTPException(status_code=500, detail="Trainer data is incomplete")
    if feature not in X_train.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in training data")
    if feature not in X_test.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in test data")

    return get_distribution_chart(
        X_train[feature],
        X_test[feature],
        feature_name=feature,
    )

