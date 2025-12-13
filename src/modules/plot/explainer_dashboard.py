from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .distribution import get_distribution_chart
from .pdp import get_partial_dependence_chart
from .permutation_importance import get_permutation_importance_chart
from .shap_bar import get_shap_bar_chart
from .shap_dependence import get_shap_dependence_chart
from .shap_summary import get_shap_summary_chart
from ...utils.shap import to_numpy, top_indices_by_mean_abs


def get_explainer_dashboard(
    trainer,
    sample_size: int = 200,
    top_n: int = 10,
    grid_size: int = 15,
) -> List[Dict[str, Any]]:
    """
    Generate a lightweight explainer dashboard as a list of Highcharts configs.
    The dashboard includes:
    - SHAP summary (beeswarm)
    - SHAP bar importance
    - SHAP dependence for the top feature
    - Partial dependence for the top feature
    - Permutation importance
    - Train/Test distribution for the top feature
    """
    charts: List[Dict[str, Any]] = []

    shap_values, X_sample = compute_shap(trainer, sample_size=sample_size)
    shap_matrix = to_numpy(shap_values)
    feature_names = list(X_sample.columns)
    top_idx = top_indices_by_mean_abs(shap_matrix, top_n)
    top_feature = feature_names[top_idx[0]] if top_idx else feature_names[0]

    charts.append(get_shap_summary_chart(shap_values, feature_names, top_n=top_n))
    charts.append(get_shap_bar_chart(shap_values, feature_names, top_n=top_n))

    shap_feat_values = shap_matrix[:, feature_names.index(top_feature)]
    charts.append(
        get_shap_dependence_chart(
            X_sample[top_feature],
            shap_feat_values,
            top_feature,
        )
    )

    grid, preds = compute_pdp(trainer, top_feature, grid_size=grid_size)
    charts.append(get_partial_dependence_chart(grid, preds, feature_name=top_feature, task=trainer.settings.task))

    importance_df = compute_permutation_importance(trainer)
    charts.append(get_permutation_importance_chart(importance_df, top_n=top_n))

    X_train = getattr(trainer, "X_train", None)
    X_test = getattr(trainer, "X_test", None)
    if X_train is not None and X_test is not None and top_feature in X_train.columns and top_feature in X_test.columns:
        charts.append(get_distribution_chart(X_train[top_feature], X_test[top_feature], feature_name=top_feature))

    return charts


