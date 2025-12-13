from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def get_permutation_importance_chart(importance_df: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """Return a bar chart using an AutoGluon-style importance dataframe."""
    df = importance_df.sort_values("importance", ascending=False).head(top_n)
    categories = df.index.tolist()[::-1]
    values = [float(v) for v in df["importance"].tolist()][::-1]

    config = {
        "chart": {"type": "bar"},
        "title": {"text": "Permutation Feature Importance"},
        "xAxis": {"categories": categories, "title": {"text": ""}},
        "yAxis": {"title": {"text": "Importance"}},
        "legend": {"enabled": False},
        "series": [{"name": "importance", "data": values}],
    }

    return config

