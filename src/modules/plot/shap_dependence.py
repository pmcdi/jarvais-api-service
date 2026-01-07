from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def get_shap_dependence_chart(
    feature_values: Iterable[Any],
    shap_feature_values: Iterable[float],
    feature_name: str,
    interaction_values: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Return a scatter plot of SHAP values against feature values."""
    data = []
    interaction_list = list(interaction_values) if interaction_values is not None else None

    for idx, (f_val, s_val) in enumerate(zip(feature_values, shap_feature_values)):
        point = {"x": float(f_val), "y": float(s_val)}
        if interaction_list is not None and idx < len(interaction_list):
            point["colorValue"] = interaction_list[idx]
        data.append(point)

    config: Dict[str, Any] = {
        "chart": {"type": "scatter", "zoomType": "xy"},
        "title": {"text": f"SHAP Dependence: {feature_name}"},
        "xAxis": {"title": {"text": feature_name}},
        "yAxis": {"title": {"text": "SHAP value"}},
        "legend": {"enabled": False},
        "series": [{"name": feature_name, "data": data}],
        "tooltip": {"pointFormat": f"{feature_name}: "+"{point.x:.4f}<br>SHAP: {point.y:.4f}"},
    }

    if interaction_list is not None:
        try:
            color_min = float(np.min(interaction_list))
            color_max = float(np.max(interaction_list))
            config["colorAxis"] = {"min": color_min, "max": color_max}
            config["series"][0]["colorKey"] = "colorValue"
        except Exception:
            categories = sorted(set(str(v) for v in interaction_list))
            color_map = {cat: idx for idx, cat in enumerate(categories)}
            for point, raw in zip(data, interaction_list):
                point["colorValue"] = color_map[str(raw)]
            config["colorAxis"] = {"categories": categories}
            config["series"][0]["colorKey"] = "colorValue"

    return config




