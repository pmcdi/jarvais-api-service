import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from ..storage import storage_manager

logger = logging.getLogger(__name__)


def _encode_frame_for_shap(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, pd.Index]]:
    """Convert non-numeric columns to categorical codes for SHAP."""
    encoded = df.copy()
    cat_maps: dict[str, pd.Index] = {}
    for col in encoded.columns:
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            cat = pd.Categorical(encoded[col])
            cat_maps[col] = cat.categories
            encoded[col] = cat.codes.astype(float)
    return encoded, cat_maps


def _decode_frame_from_shap(df: pd.DataFrame, cat_maps: dict[str, pd.Index]) -> pd.DataFrame:
    """Restore categorical codes back to original labels after SHAP perturbations."""
    decoded = df.copy()
    for col, categories in cat_maps.items():
        # SHAP returns floats; round to nearest int code and map back
        decoded[col] = (
            decoded[col]
            .round()
            .astype(int)
            .map(lambda idx: categories[idx] if 0 <= idx < len(categories) else None)
        )
    return decoded


def load_trainer(trainer_id: str):
    """Load a TrainerSupervised instance from stored metadata."""
    if not storage_manager.check_trainer(trainer_id):
        raise HTTPException(status_code=404, detail="Trainer not found")

    trainer_data = storage_manager.get_trainer(trainer_id)
    if not trainer_data:
        raise HTTPException(status_code=404, detail="Trainer not found")

    output_dir = trainer_data.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=500, detail="Trainer output directory not set")

    try:
        from jarvais.trainer import TrainerSupervised

        return TrainerSupervised.load_trainer(output_dir)
    except FileNotFoundError as exc:
        logger.error("Trainer files missing at %s: %s", output_dir, exc)
        raise HTTPException(status_code=500, detail="Trainer artifacts missing on disk") from exc
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load trainer %s: %s", trainer_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load trainer") from exc


def sample_frame(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    """Return a small, reproducible sample for expensive computations."""
    if df.empty:
        raise HTTPException(status_code=400, detail="Trainer data is empty")
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=0)


def select_feature_frame(trainer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return X (features) and y (target) for downstream explainer utilities."""
    X = getattr(trainer, "X_train", None)
    y = getattr(trainer, "y_train", None)
    if X is None or y is None:
        raise HTTPException(status_code=500, detail="Trainer is missing training data")
    return X, y


def predict_mean(trainer, frame: pd.DataFrame) -> float:
    """Average prediction helper across tasks."""
    predictor = trainer.predictor
    if getattr(predictor, "can_predict_proba", False) and trainer.settings.task in {"binary", "multiclass"}:
        proba = predictor.predict_proba(frame)
        target_col = proba.columns[1] if proba.shape[1] > 1 else proba.columns[0]
        return float(proba[target_col].mean())
    preds = predictor.predict(frame, as_pandas=False)
    return float(np.asarray(preds).mean())


def compute_shap(trainer, sample_size: int = 200) -> Tuple[Any, pd.DataFrame]:
    """Compute SHAP values on a small sample."""
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="SHAP is not available in the environment") from exc

    X, _ = select_feature_frame(trainer)
    X_sample = sample_frame(X, max_rows=sample_size)

    # SHAP's tabular masker expects numeric input; encode categoricals and
    # decode them back before passing to the predictor.
    X_encoded, cat_maps = _encode_frame_for_shap(X_sample)
    cols = X_encoded.columns
    predictor = trainer.predictor
    is_clf = getattr(predictor, "can_predict_proba", False) and trainer.settings.task in {"binary", "multiclass"}

    def predict_fn(data):
        encoded_df = pd.DataFrame(data, columns=cols)
        decoded_df = _decode_frame_from_shap(encoded_df, cat_maps)
        if is_clf:
            return predictor.predict_proba(decoded_df).values
        return predictor.predict(decoded_df, as_pandas=False)

    explainer = shap.Explainer(predict_fn, X_encoded)
    shap_values = explainer(X_encoded)

    # Return the original sample (with human-friendly columns) for charting
    return shap_values, X_sample


def compute_pdp(trainer, feature: str, grid_size: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple 1D partial dependence using model predictions."""
    X, _ = select_feature_frame(trainer)
    if feature not in X.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found")

    base = sample_frame(X, max_rows=300).copy()
    series = base[feature]
    if pd.api.types.is_numeric_dtype(series):
        grid = np.linspace(series.min(), series.max(), grid_size)
    else:
        grid = series.dropna().unique()

    preds = []
    for value in grid:
        modified = base.copy()
        modified[feature] = value
        preds.append(predict_mean(trainer, modified))

    return np.asarray(grid), np.asarray(preds)


def compute_permutation_importance(trainer) -> pd.DataFrame:
    """Proxy to AutoGluon feature_importance with a safe default."""
    predictor = trainer.predictor
    try:
        X, y = select_feature_frame(trainer)
        frame = pd.concat([X, y], axis=1)
        importance = predictor.feature_importance(frame)
    except Exception as exc:
        logger.error("Permutation importance failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to compute permutation importance") from exc

    if not isinstance(importance, pd.DataFrame):
        raise HTTPException(status_code=500, detail="Unexpected importance format from predictor")
    if "importance" not in importance.columns:
        raise HTTPException(status_code=500, detail="Predictor importance response missing 'importance' column")
    return importance
