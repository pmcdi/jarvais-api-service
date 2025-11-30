import logging
import uuid
from datetime import datetime
from typing import Dict, Any
import tempfile
import os

from fastapi import APIRouter, HTTPException, Path, Request, File, UploadFile

from ..storage import storage_manager
from ..models import (
    TrainerRequest, TrainerInfo, TrainerList, TrainerListItem,
    TrainerResults, ModelLeaderboardEntry, ModelScore, SuccessResponse,
    InferenceRequest, InferenceResult
)
from ..config import settings
from ..utils.rate_limit import apply_rate_limit
from ..modules import jarvais_train, jarvais_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trainers", tags=["trainers"])

"""
STRETCH GOAL
- model download option?
- return top 3 instead of top 1 trained model
"""

@router.post("", response_model=TrainerInfo, status_code=200)
@apply_rate_limit(settings.rate_limit_general)
async def create_trainer(
    request: Request,
    trainer_request: TrainerRequest
):
    """
    Create a trainer instance from an analyzer.
    
    Args:
        request: FastAPI request object (required for rate limiting)
        trainer_request: Trainer configuration
        
    Returns:
        TrainerInfo: Information about the completed trainer
    """
    # Check if analyzer exists
    if not storage_manager.check_analyzer(trainer_request.analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    # Validate inputs
    jarvais_train.validate_task_type(trainer_request.task)
    jarvais_train.validate_feature_reduction_method(trainer_request.feature_reduction_method)
    
    # Generate unique ID for this trainer session
    trainer_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # Create temporary output directory for trainer
    output_dir = os.path.join(tempfile.gettempdir(), f"trainer_{trainer_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get analyzer
        analyzer = storage_manager.get_analyzer(trainer_request.analyzer_id)
        if not analyzer:
            raise HTTPException(status_code=404, detail="Analyzer not found")
        
        # Run training (returns output_dir path where trainer is saved)
        saved_output_dir, training_time = jarvais_train.run_trainer(
            analyzer, trainer_request, output_dir
        )
        
        # Store trainer metadata (no trainer instance, just the path)
        trainer_data: Dict[str, Any] = {
            'trainer_id': trainer_id,
            'analyzer_id': trainer_request.analyzer_id,
            'target_variable': trainer_request.target_variable,
            'task': trainer_request.task,
            'k_folds': trainer_request.k_folds,
            'created_at': created_at,
            'output_dir': saved_output_dir,
            'training_time': training_time
        }
        
        storage_manager.store_trainer(trainer_id, trainer_data)
        
        logger.info(f"Created and trained trainer {trainer_id} in {training_time:.2f}s")
        
        return TrainerInfo(
            trainer_id=trainer_id,
            analyzer_id=trainer_request.analyzer_id,
            target_variable=trainer_request.target_variable,
            task=trainer_request.task,
            k_folds=trainer_request.k_folds,
            created_at=created_at,
            training_time=training_time
        )
        
    except HTTPException:
        # Clean up output directory on failure
        jarvais_train.cleanup_trainer_files(output_dir)
        raise
    except Exception as e:
        # Clean up output directory on failure
        jarvais_train.cleanup_trainer_files(output_dir)
        logger.error(f"Failed to create trainer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create trainer: {str(e)}")


@router.get("", response_model=TrainerList)
@apply_rate_limit(settings.rate_limit_general)
async def list_trainers(request: Request):
    """List all trainer sessions."""
    trainer_ids = storage_manager.list_trainer_ids()
    
    trainer_list = []
    for tid in trainer_ids:
        trainer_data = storage_manager.get_trainer(tid)
        if trainer_data:
            trainer_list.append(
                TrainerListItem(
                    trainer_id=tid,
                    analyzer_id=trainer_data['analyzer_id'],
                    target_variable=trainer_data['target_variable'],
                    task=trainer_data['task']
                )
            )
    
    return TrainerList(count=len(trainer_list), trainers=trainer_list)


@router.get("/{trainer_id}", response_model=TrainerInfo)
@apply_rate_limit(settings.rate_limit_general)
async def trainer_info(
    request: Request,
    trainer_id: str = Path(..., description="Unique identifier for the trainer instance")
):
    """Get information about a specific trainer."""
    if not storage_manager.check_trainer(trainer_id):
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    try:
        trainer_data = storage_manager.get_trainer(trainer_id)
        if not trainer_data:
            raise HTTPException(status_code=404, detail="Trainer not found")
        
        return TrainerInfo(
            trainer_id=trainer_id,
            analyzer_id=trainer_data['analyzer_id'],
            target_variable=trainer_data['target_variable'],
            task=trainer_data['task'],
            k_folds=trainer_data['k_folds'],
            created_at=trainer_data['created_at'],
            training_time=trainer_data.get('training_time')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trainer info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trainer info: {str(e)}")


"""
TO-DO
Let's output this in a JSON / HighCharts format.
"""
@router.get("/{trainer_id}/results", response_model=TrainerResults)
@apply_rate_limit(settings.rate_limit_general)
async def trainer_results(
    request: Request,
    trainer_id: str = Path(..., description="Unique identifier for the trainer instance")
):
    """Get training results and leaderboard for a specific trainer."""
    if not storage_manager.check_trainer(trainer_id):
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    try:
        trainer_data = storage_manager.get_trainer(trainer_id)
        if not trainer_data:
            raise HTTPException(status_code=404, detail="Trainer not found")
        
        # Load trainer from disk
        output_dir = trainer_data.get('output_dir')
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=500, detail="Trainer output directory not found")
        
        from jarvais.trainer import TrainerSupervised
        trainer_instance = TrainerSupervised.load_trainer(output_dir)
        
        # Extract leaderboard data from trainer
        leaderboard_entries = []
        
        # Get the leaderboard from AutoGluon trainer
        if hasattr(trainer_instance, 'leaderboard') and trainer_instance.leaderboard is not None:
            leaderboard_df = trainer_instance.leaderboard
            
            # Process leaderboard to extract metrics
            for idx, row in leaderboard_df.iterrows():
                model_name = str(idx) if not isinstance(idx, str) else idx
                
                # Extract scores
                scores_test = []
                scores_val = []
                scores_train = []
                
                # Parse the score columns from the leaderboard
                for col in leaderboard_df.columns:
                    if 'score_test' in col or 'test' in col.lower():
                        value = row[col]
                        if isinstance(value, (int, float)):
                            scores_test.append(ModelScore(
                                metric=col,
                                mean=float(value),
                                min=float(value),
                                max=float(value)
                            ))
                    elif 'score_val' in col or 'val' in col.lower():
                        value = row[col]
                        if isinstance(value, (int, float)):
                            scores_val.append(ModelScore(
                                metric=col,
                                mean=float(value),
                                min=float(value),
                                max=float(value)
                            ))
                
                leaderboard_entries.append(ModelLeaderboardEntry(
                    model_name=model_name,
                    scores_test=scores_test if scores_test else [ModelScore(metric='N/A', mean=0.0, min=0.0, max=0.0)],
                    scores_val=scores_val if scores_val else [ModelScore(metric='N/A', mean=0.0, min=0.0, max=0.0)],
                    scores_train=[ModelScore(metric='N/A', mean=0.0, min=0.0, max=0.0)]
                ))
        
        # Get best model name
        best_model = "N/A"
        if hasattr(trainer_instance, 'best_model') and trainer_instance.best_model:
            best_model = str(trainer_instance.best_model)
        
        return TrainerResults(
            trainer_id=trainer_id,
            leaderboard=leaderboard_entries if leaderboard_entries else [],
            best_model=best_model,
            training_time=trainer_data.get('training_time', 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trainer results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trainer results: {str(e)}")


@router.delete("/{trainer_id}", response_model=SuccessResponse)
@apply_rate_limit(settings.rate_limit_general)
async def delete_trainer(
    request: Request,
    trainer_id: str = Path(..., description="Unique identifier for the trainer instance")
):
    """Delete a trainer session."""
    try:
        # Get trainer data to clean up output directory
        trainer_data = storage_manager.get_trainer(trainer_id)
        if trainer_data and 'output_dir' in trainer_data:
            output_dir = trainer_data['output_dir']
            jarvais_train.cleanup_trainer_files(output_dir)
        
        # Delete trainer from storage
        if storage_manager.delete_trainer(trainer_id):
            logger.info(f"Deleted trainer {trainer_id}")
            return SuccessResponse(message=f"Trainer {trainer_id} deleted successfully")
        else:
            raise HTTPException(status_code=404, detail="Trainer not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete trainer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete trainer: {str(e)}")


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================

@router.post("/{trainer_id}/infer", response_model=InferenceResult, status_code=200)
@apply_rate_limit(settings.rate_limit_general)
async def infer_from_csv(
    request: Request,
    trainer_id: str = Path(..., description="Unique identifier for the trainer instance"),
    file: UploadFile = File(...)
):
    """
    Generate predictions from a CSV file using a trained model.
    
    Args:
        request: FastAPI request object (required for rate limiting)
        trainer_id: Trainer ID to use for inference
        file: CSV file with features matching training data
        
    Returns:
        InferenceResult: Predictions and probabilities
    """
    # Check if trainer exists
    if not storage_manager.check_trainer(trainer_id):
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    # Get trainer data
    trainer_data = storage_manager.get_trainer(trainer_id)
    if not trainer_data:
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    logger.info(f"Retrieved trainer data: output_dir={trainer_data.get('output_dir')}")
    
    try:
        # Read CSV file
        file_content = await file.read()
        logger.info(f"Received CSV file of size {len(file_content)} bytes")
        df = jarvais_infer.prepare_csv_data(file_content)
        logger.info(f"Parsed CSV to DataFrame with shape {df.shape}")
        
        # Load trainer from disk
        output_dir = trainer_data.get('output_dir')
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=500, detail="Trainer output directory not found")
        
        from jarvais.trainer import TrainerSupervised
        logger.info(f"Loading trainer from: {output_dir}")
        trainer_instance = TrainerSupervised.load_trainer(output_dir)
        logger.info(f"Trainer loaded successfully")
        
        # Get the analyzer to transform inference data
        analyzer_id = trainer_data['analyzer_id']
        analyzer = storage_manager.get_analyzer(analyzer_id)
        if not analyzer:
            raise HTTPException(
                status_code=404, 
                detail=f"Analyzer {analyzer_id} not found. The analyzer may have expired. "
                       "Please re-upload the training data and create a new trainer."
            )
        
        # Remove target variable if present
        target_variable = trainer_data['target_variable']
        df = jarvais_infer.remove_target_variable(df, target_variable)
        logger.info(f"DataFrame after removing target: {df.shape}")
        
        # Transform inference data to match training data format
        logger.info("Transforming inference data to match training format...")
        df = jarvais_infer.transform_inference_data(df, analyzer, target_variable)
        logger.info(f"DataFrame after transformation: {df.shape}")
        
        # Generate predictions
        logger.info(f"Starting inference for {len(df)} samples")
        predictions_list, probabilities = jarvais_infer.perform_inference(
            df, trainer_instance, trainer_data
        )
        logger.info(f"Inference completed: {len(predictions_list)} predictions, probabilities={'Yes' if probabilities else 'No'}")
        
        created_at = datetime.now().isoformat()
        
        logger.info(f"Inference completed for trainer {trainer_id} with {len(predictions_list)} samples")
        
        return InferenceResult(
            trainer_id=trainer_id,
            predictions=predictions_list,
            probabilities=probabilities,
            num_samples=len(predictions_list),
            created_at=created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/{trainer_id}/infer/json", response_model=InferenceResult, status_code=200)
@apply_rate_limit(settings.rate_limit_general)
async def infer_from_json(
    request: Request,
    trainer_id: str = Path(..., description="Unique identifier for the trainer instance"),
    inference_request: InferenceRequest = ...
):
    """
    Generate predictions from JSON data using a trained model.
    
    Args:
        request: FastAPI request object (required for rate limiting)
        trainer_id: Trainer ID to use for inference
        inference_request: JSON payload with feature values
        
    Returns:
        InferenceResult: Predictions and probabilities
    """
    # Check if trainer exists
    if not storage_manager.check_trainer(trainer_id):
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    # Get trainer data
    trainer_data = storage_manager.get_trainer(trainer_id)
    if not trainer_data:
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    try:
        # Convert JSON to DataFrame
        logger.info(f"Received JSON data with {len(inference_request.data)} samples")
        df = jarvais_infer.prepare_json_data(inference_request.data)
        logger.info(f"Converted JSON to DataFrame with shape {df.shape}")
        
        # Load trainer from disk
        output_dir = trainer_data.get('output_dir')
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=500, detail="Trainer output directory not found")
        
        from jarvais.trainer import TrainerSupervised
        logger.info(f"Loading trainer from: {output_dir}")
        trainer_instance = TrainerSupervised.load_trainer(output_dir)
        logger.info(f"Trainer loaded successfully")
        
        # Get the analyzer to transform inference data
        analyzer_id = trainer_data['analyzer_id']
        analyzer = storage_manager.get_analyzer(analyzer_id)
        if not analyzer:
            raise HTTPException(
                status_code=404, 
                detail=f"Analyzer {analyzer_id} not found. The analyzer may have expired. "
                       "Please re-upload the training data and create a new trainer."
            )
        
        # Remove target variable if present
        target_variable = trainer_data['target_variable']
        df = jarvais_infer.remove_target_variable(df, target_variable)
        logger.info(f"DataFrame after removing target: {df.shape}")
        
        # Transform inference data to match training data format
        logger.info("Transforming inference data to match training format...")
        df = jarvais_infer.transform_inference_data(df, analyzer, target_variable)
        logger.info(f"DataFrame after transformation: {df.shape}")
        
        # Generate predictions
        logger.info(f"Starting inference for {len(df)} samples")
        predictions_list, probabilities = jarvais_infer.perform_inference(
            df, trainer_instance, trainer_data
        )
        logger.info(f"Inference completed: {len(predictions_list)} predictions, probabilities={'Yes' if probabilities else 'No'}")
        
        created_at = datetime.now().isoformat()
        
        logger.info(f"Inference completed for trainer {trainer_id} with {len(predictions_list)} samples")
        
        return InferenceResult(
            trainer_id=trainer_id,
            predictions=predictions_list,
            probabilities=probabilities,
            num_samples=len(predictions_list),
            created_at=created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

