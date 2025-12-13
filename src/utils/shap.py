from __future__ import annotations

from typing import Any, List

import numpy as np


def to_numpy(values: Any) -> np.ndarray:
    """Convert various SHAP outputs into a 2D numpy array of shape (n_samples, n_features)."""
    arr = getattr(values, "values", values)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        class_axis = 1
        if arr.shape[class_axis] > 1:
            arr = arr[:, 1, :]
        else:
            arr = arr[:, 0, :]
    return arr


def top_indices_by_mean_abs(shap_matrix: np.ndarray, top_n: int) -> List[int]:
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return order[: min(top_n, len(order))].tolist()


