from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def get_distribution_chart(
    train_series: pd.Series,
    test_series: pd.Series,
    feature_name: str,
    bins: int = 15,
) -> Dict[str, Any]:
    """Return overlapping histograms (numeric) or bar chart (categorical) for train vs test."""
    if pd.api.types.is_numeric_dtype(train_series):
        combined = pd.concat([train_series, test_series])
        _, bin_edges = np.histogram(combined.dropna(), bins=bins)
        train_hist, _ = np.histogram(train_series.dropna(), bins=bin_edges)
        test_hist, _ = np.histogram(test_series.dropna(), bins=bin_edges)
        categories = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]
        series = [
            {"name": "train", "data": train_hist.tolist(), "type": "column", "pointPadding": 0},
            {"name": "test", "data": test_hist.tolist(), "type": "column", "pointPadding": 0.1},
        ]
    else:
        train_counts = train_series.value_counts()
        test_counts = test_series.value_counts()
        categories = sorted(set(train_counts.index).union(set(test_counts.index)))
        series = [
            {"name": "train", "data": [int(train_counts.get(cat, 0)) for cat in categories]},
            {"name": "test", "data": [int(test_counts.get(cat, 0)) for cat in categories]},
        ]

    config = {
        "chart": {"type": "column"},
        "title": {"text": f"Train vs Test Distribution: {feature_name}"},
        "xAxis": {"categories": categories, "title": {"text": feature_name}},
        "yAxis": {"title": {"text": "Count"}},
        "plotOptions": {"column": {"grouping": True, "shadow": False}},
        "series": series,
    }

    return config

