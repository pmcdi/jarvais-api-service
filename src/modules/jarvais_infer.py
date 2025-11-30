"""
Inference business logic for Jarvais.

This module contains inference functions extracted from the routers.
Handles NaN/inf values for JSON serialization.
"""

import logging
import io
import math
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from jarvais import Analyzer
from jarvais.trainer import TrainerSupervised

logger = logging.getLogger(__name__)


def prepare_csv_data(file_content: bytes) -> pd.DataFrame:
    """
    Prepare DataFrame from CSV file content.
    
    Args:
        file_content: CSV file content as bytes
        
    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(io.BytesIO(file_content))
    logger.info(f"Loaded CSV with shape {df.shape}, NaN count: {df.isna().sum().sum()}")
    return df


def prepare_json_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare DataFrame from JSON data.
    
    Args:
        data: List of dictionaries with feature values
        
    Returns:
        DataFrame with parsed data
    """
    df = pd.DataFrame(data)
    
    # Handle JSON NaN values (None becomes NaN in DataFrame)
    # AutoGluon can handle NaN, so we just log it
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"JSON data contains {nan_count} NaN values")
    
    return df


def remove_target_variable(df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    """
    Remove target variable from DataFrame if present.
    
    Args:
        df: Input DataFrame
        target_variable: Name of target variable to remove
        
    Returns:
        DataFrame without target variable
    """
    if target_variable in df.columns:
        df = df.drop(columns=[target_variable])
        logger.info(f"Dropped target variable '{target_variable}' from inference data")
    return df


def transform_inference_data(
    df: pd.DataFrame,
    analyzer: Analyzer,
    target_variable: str
) -> pd.DataFrame:
    """
    Transform inference data to match the format expected by the trained model.
    
    The Analyzer one-hot encodes categorical variables during training. This function
    applies the same transformation to inference data and aligns columns to match
    the training data format exactly.
    
    Args:
        df: Raw inference DataFrame with original column names
        analyzer: Analyzer instance used during training (contains the expected format)
        target_variable: Target variable name (will be excluded from transformation)
        
    Returns:
        DataFrame with columns matching the training data format
    """
    # Get the expected columns from the analyzer's processed data
    # These are the one-hot encoded column names the model was trained on
    training_columns = list(analyzer.data.columns)
    
    # Remove target variable from expected columns if present
    if target_variable in training_columns:
        training_columns.remove(target_variable)
    
    logger.info(f"Training data has {len(training_columns)} features (excluding target)")
    logger.info(f"Inference data has {len(df.columns)} columns")
    
    # Get categorical columns from analyzer settings
    categorical_columns = analyzer.settings.categorical_columns or []
    continuous_columns = analyzer.settings.continuous_columns or []
    
    logger.info(f"Categorical columns from analyzer: {categorical_columns}")
    logger.info(f"Continuous columns from analyzer: {continuous_columns}")
    
    # Start building the transformed DataFrame
    transformed_df = pd.DataFrame(index=df.index)
    
    # Add continuous columns directly (no transformation needed)
    for col in continuous_columns:
        if col in df.columns and col != target_variable:
            transformed_df[col] = df[col]
        elif col in training_columns:
            # Column expected but not in inference data - fill with NaN
            logger.warning(f"Continuous column '{col}' missing from inference data, filling with NaN")
            transformed_df[col] = np.nan
    
    # One-hot encode categorical columns to match training format
    for cat_col in categorical_columns:
        if cat_col == target_variable:
            continue
            
        if cat_col not in df.columns:
            logger.warning(f"Categorical column '{cat_col}' missing from inference data")
            # Add all expected one-hot columns for this category as 0
            for train_col in training_columns:
                if train_col.startswith(f"{cat_col}|"):
                    transformed_df[train_col] = 0
            continue
        
        # Get unique values from inference data
        inference_values = df[cat_col].astype(str)
        
        # Create one-hot encoded columns matching training format
        for train_col in training_columns:
            if train_col.startswith(f"{cat_col}|"):
                # Extract the category value from the column name
                category_value = train_col.split("|", 1)[1]
                # Create the one-hot column
                transformed_df[train_col] = (inference_values == category_value).astype(int)
    
    # Ensure all expected columns are present, in the right order
    final_df = pd.DataFrame(index=df.index)
    missing_columns = []
    for col in training_columns:
        if col in transformed_df.columns:
            final_df[col] = transformed_df[col]
        else:
            # Column expected by model but not generated - fill with 0 for categorical, NaN for continuous
            if "|" in col:
                final_df[col] = 0
            else:
                final_df[col] = np.nan
            missing_columns.append(col)
    
    if missing_columns:
        logger.warning(f"Added {len(missing_columns)} missing columns with default values")
        logger.debug(f"Missing columns: {missing_columns[:10]}...")
    
    # Check for extra columns in inference data that weren't expected
    extra_cols = set(transformed_df.columns) - set(training_columns)
    if extra_cols:
        logger.warning(f"Ignoring {len(extra_cols)} extra columns from inference data: {list(extra_cols)[:5]}...")
    
    logger.info(f"Transformed inference data: {final_df.shape[0]} samples, {final_df.shape[1]} features")
    
    return final_df


def _is_nan_or_inf(obj) -> bool:
    """Check if a value is NaN or Inf using multiple methods for robustness."""
    try:
        # Handle Python float
        if isinstance(obj, float):
            return math.isnan(obj) or math.isinf(obj)
        # Handle numpy types
        if isinstance(obj, (np.floating, np.integer)):
            return bool(np.isnan(obj)) or bool(np.isinf(obj))
        # Handle pandas NA
        if pd.isna(obj):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _replace_nan_with_none(obj):
    """
    Recursively replace NaN values with None for JSON serialization.
    
    Args:
        obj: Object to process (can be list, dict, float, etc.)
        
    Returns:
        Object with NaN replaced by None
    """
    if isinstance(obj, list):
        return [_replace_nan_with_none(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _replace_nan_with_none(value) for key, value in obj.items()}
    elif isinstance(obj, (np.floating, np.integer)):
        # Convert numpy types to Python types first
        if _is_nan_or_inf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, float):
        if _is_nan_or_inf(obj):
            return None
        return obj
    elif obj is None or (hasattr(obj, '__class__') and obj.__class__.__name__ == 'NAType'):
        # Handle None and pandas NA type
        return None
    else:
        return obj


def _count_nans_in_nested(obj, path="root") -> int:
    """Debug helper to count NaN values in nested structures."""
    count = 0
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            count += _count_nans_in_nested(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            count += _count_nans_in_nested(v, f"{path}.{k}")
    elif _is_nan_or_inf(obj):
        logger.warning(f"DEBUG: Found NaN/Inf at {path}: {obj} (type: {type(obj).__name__})")
        count = 1
    return count


def perform_inference(
    df: pd.DataFrame,
    trainer_instance: TrainerSupervised,
    trainer_data: Dict[str, Any]
) -> Tuple[List[Any], Optional[List[List[float]]]]:
    """
    Perform inference on data using trained model.
    
    Args:
        df: DataFrame with features
        trainer_instance: Trained model instance (TrainerSupervised)
        trainer_data: Trainer metadata including task type
        
    Returns:
        Tuple of (predictions list, probabilities list or None)
        
    Raises:
        Exception: If prediction fails
    """
    try:
        # Generate predictions using AutoGluon predictor directly
        # The jarvais infer() method may have issues, so we'll use the predictor directly
        logger.info(f"Calling predictor.predict() on {len(df)} samples with {len(df.columns)} features")
        logger.debug(f"Features: {list(df.columns)}")
        
        # Access the underlying AutoGluon predictor
        if not hasattr(trainer_instance, 'predictor') or trainer_instance.predictor is None:
            raise Exception("Trainer does not have a predictor")
        
        # Use AutoGluon's predictor directly
        predictions = trainer_instance.predictor.predict(df, as_pandas=False)
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        logger.info(f"Predictions generated: {len(predictions_list)} values")
        
        # DEBUG: Check for NaNs before replacement
        nan_count_before = _count_nans_in_nested(predictions_list, "predictions_before")
        logger.info(f"DEBUG: NaN count in predictions BEFORE replacement: {nan_count_before}")
        
        # Replace NaN values in predictions with None
        predictions_list = _replace_nan_with_none(predictions_list)
        
        # DEBUG: Check for NaNs after replacement
        nan_count_after = _count_nans_in_nested(predictions_list, "predictions_after")
        logger.info(f"DEBUG: NaN count in predictions AFTER replacement: {nan_count_after}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise Exception(f"Prediction failed: {str(e)}")
    
    # Generate probabilities for classification tasks
    # Note: jarvais trainer doesn't expose predict_proba directly yet
    # We'll try to access the underlying predictor if available
    probabilities = None
    if trainer_data['task'] in ['binary', 'multiclass']:
        try:
            logger.info("Generating probabilities for classification task")
            # Access the underlying AutoGluon predictor
            if hasattr(trainer_instance, 'predictor') and trainer_instance.predictor is not None:
                proba = trainer_instance.predictor.predict_proba(df)
                
                # DEBUG: Log proba type and check for NaNs
                logger.info(f"DEBUG: proba type: {type(proba).__name__}")
                if hasattr(proba, 'isna'):
                    logger.info(f"DEBUG: proba NaN count (DataFrame): {proba.isna().sum().sum()}")
                
                # Convert to list of lists for JSON serialization
                if hasattr(proba, 'values'):  # DataFrame or Series
                    probabilities = proba.values.tolist()
                elif hasattr(proba, 'tolist'):  # numpy array
                    probabilities = proba.tolist()
                else:
                    probabilities = list(proba)
                
                logger.info(f"Probabilities generated: {len(probabilities)} samples")
                
                # DEBUG: Check for NaNs before replacement
                nan_count_before = _count_nans_in_nested(probabilities, "probabilities_before")
                logger.info(f"DEBUG: NaN count in probabilities BEFORE replacement: {nan_count_before}")
                
                # Replace NaN values in probabilities with None
                probabilities = _replace_nan_with_none(probabilities)
                
                # DEBUG: Check for NaNs after replacement
                nan_count_after = _count_nans_in_nested(probabilities, "probabilities_after")
                logger.info(f"DEBUG: NaN count in probabilities AFTER replacement: {nan_count_after}")
                
                # DEBUG: Sample some values to check types
                if probabilities and len(probabilities) > 0:
                    sample = probabilities[0]
                    logger.info(f"DEBUG: First probability row: {sample}")
                    if isinstance(sample, list) and len(sample) > 0:
                        logger.info(f"DEBUG: First value type: {type(sample[0]).__name__}, value: {sample[0]}")
                
            else:
                logger.warning("Predictor not available for probability generation")
        except Exception as e:
            logger.warning(f"Could not generate probabilities: {e}", exc_info=True)
    
    return predictions_list, probabilities

