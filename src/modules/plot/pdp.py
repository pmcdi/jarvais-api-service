from __future__ import annotations

from typing import Any, Dict, Iterable


def get_partial_dependence_chart(
    grid: Iterable[float],
    predictions: Iterable[float],
    feature_name: str,
    task: str,
) -> Dict[str, Any]:
    """Return a line chart for 1D partial dependence."""
    grid_list = [float(x) for x in grid]
    pred_list = [float(y) for y in predictions]

    config = {
        "chart": {"type": "line"},
        "title": {"text": f"Partial Dependence: {feature_name}"},
        "xAxis": {"title": {"text": feature_name}, "categories": grid_list},
        "yAxis": {
            "title": {"text": "Predicted probability" if task in {"binary", "multiclass"} else "Predicted value"}
        },
        "legend": {"enabled": False},
        "series": [{"name": feature_name, "data": pred_list}],
    }

    return config

