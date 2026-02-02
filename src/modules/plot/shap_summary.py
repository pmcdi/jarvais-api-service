from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from ...utils.shap import to_numpy, top_indices_by_mean_abs


def get_shap_summary_chart(
    shap_values: Any,
    feature_names: Iterable[str],
    top_n: int = 20,
    max_points: int = 200,
) -> Dict[str, Any]:
    """Return a beeswarm-style Highcharts scatter configuration."""
    shap_matrix = to_numpy(shap_values)
    names = list(feature_names)
    top_idx = top_indices_by_mean_abs(shap_matrix, top_n)

    rng = np.random.default_rng(0)
    series = []

    for y_idx, feat_idx in enumerate(top_idx):
        values = shap_matrix[:, feat_idx]
        if len(values) > max_points:
            values = rng.choice(values, size=max_points, replace=False)
        series.append(
            {
                "type": "scatter",
                "name": names[feat_idx],
                "data": [{"x": float(val), "y": y_idx} for val in values],
                "tooltip": {"pointFormat": "{point.x:.4f}"},
                "marker": {"radius": 3, "symbol": "circle", "fillOpacity": 0.45},
            }
        )

    config = {
        "chart": {"type": "scatter", "zoomType": "xy"},
        "title": {"text": "SHAP Summary"},
        "xAxis": {"title": {"text": "SHAP value"}, "gridLineWidth": 1},
        "yAxis": {
            "title": {"text": ""},
            "categories": [names[i] for i in top_idx],
            "tickInterval": 1,
            "gridLineWidth": 1,
        },
        "legend": {"enabled": False},
        "series": series,
    }

    return config


