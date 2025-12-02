"""
Training logic for Jarvais.
"""

import logging
import shutil
import os
from datetime import datetime
from typing import Optional, Tuple
from fastapi import HTTPException

from jarvais.trainer import TrainerSupervised

logger = logging.getLogger(__name__)


def validate_task_type(task: str) -> None:
    """
    Validate that the task type is valid.
    
    Args:
        task: Task type to validate
        
    Raises:
        HTTPException: If task type is invalid
    """
    valid_tasks = ['binary', 'multiclass', 'regression', 'survival']
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task type. Must be one of: {', '.join(valid_tasks)}"
        )


def validate_feature_reduction_method(method: Optional[str]) -> None:
    """
    Validate that the feature reduction method is valid.
    
    Args:
        method: Feature reduction method to validate
        
    Raises:
        HTTPException: If method is invalid
    """
    if method:
        valid_methods = ['mrmr', 'variance_threshold', 'corr', 'chi2']
        if method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature reduction method. Must be one of: {', '.join(valid_methods)}"
            )


def run_trainer(analyzer, trainer_request, output_dir: str) -> Tuple[str, float]:
    """
    Run training and return the path to the saved model.
    
    Args:
        analyzer: Analyzer instance with data
        trainer_request: TrainerRequest with configuration
        output_dir: Directory for trainer output
        
    Returns:
        Tuple of (output_dir path, training time in seconds)
        
    Raises:
        Exception: If training fails
        
    Note:
        The trainer is saved to disk at output_dir and should be loaded
        using TrainerSupervised.load_trainer(output_dir) when needed.
    """
    # Initialize trainer
    trainer_kwargs = {
        'output_dir': output_dir,
        'target_variable': trainer_request.target_variable,
        'task': trainer_request.task,
        'k_folds': trainer_request.k_folds,
    }
    
    # Note: time_limit is not passed to TrainerSupervised as it's not supported
    # It's kept in the API for future compatibility
    
    if trainer_request.feature_reduction_method:
        trainer_kwargs['feature_reduction_method'] = trainer_request.feature_reduction_method
        if trainer_request.n_features:
            trainer_kwargs['n_features'] = trainer_request.n_features
    
    trainer = TrainerSupervised(**trainer_kwargs)
    
    # Run training
    start_time = datetime.now()
    trainer.run(analyzer.data)
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Training completed successfully in {training_time:.2f}s")
    logger.info(f"Trainer saved to: {output_dir}")
    
    # Return the output_dir path instead of the trainer object
    # The trainer will be loaded from disk when needed
    return output_dir, training_time


def cleanup_trainer_files(output_dir: str) -> None:
    """
    Clean up trainer output directory.
    
    Args:
        output_dir: Directory to clean up
    """
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            logger.info(f"Cleaned up output directory: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up output directory: {e}")

