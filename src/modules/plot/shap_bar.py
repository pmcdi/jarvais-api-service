from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from ...utils.shap import to_numpy, top_indices_by_mean_abs


def get_shap_bar_chart(
    shap_values: Any,
    feature_names: Iterable[str],
    top_n: int = 20,
) -> Dict[str, Any]:
    """Return a horizontal bar chart of mean absolute SHAP values."""
    shap_matrix = to_numpy(shap_values)
    names = list(feature_names)
    top_idx = top_indices_by_mean_abs(shap_matrix, top_n)
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    categories = [names[i] for i in top_idx][::-1]
    values = [float(mean_abs[i]) for i in top_idx][::-1]

    config = {
        "chart": {"type": "bar"},
        "title": {"text": "SHAP Feature Importance"},
        "xAxis": {"categories": categories, "title": {"text": ""}},
        "yAxis": {"title": {"text": "Mean |SHAP value|"}},
        "legend": {"enabled": False},
        "series": [{"name": "Importance", "data": values}],
    }

    return config




