from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AnalyzerInfo(BaseModel):
    analyzer_id: str = Field(..., description="Unique identifier for the analyzer")
    filename: Optional[str] = Field(None, description="Original filename")
    file_shape: tuple = Field(..., description="Shape of the uploaded data")
    categorical_variables: List[str] = Field(..., description="List of categorical variables")
    continuous_variables: List[str] = Field(..., description="List of continuous variables")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")


class AnalyzerListItem(BaseModel):
    analyzer_id: str = Field(..., description="Unique identifier for the analyzer")
    has_data: bool = Field(..., description="Whether the analyzer has data")


class AnalyzerList(BaseModel):
    count: int = Field(..., description="Number of analyzers")
    analyzers: List[AnalyzerListItem] = Field(..., description="List of analyzers")


class HealthStatus(BaseModel):
    status: str = Field(..., description="Service status")
    storage: str = Field(..., description="Storage type being used")
    timestamp: str = Field(..., description="Current timestamp")
    redis: Optional[str] = Field(None, description="Redis connection status")
    version: str = Field(..., description="API version")
    mode: str = Field(..., description="Application mode")


class SuccessResponse(BaseModel):
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class TrainerRequest(BaseModel):
    analyzer_id: str = Field(..., description="Analyzer ID to train from")
    target_variable: str = Field(..., description="Target variable for training")
    task: str = Field(..., description="Task type: 'binary', 'multiclass', 'regression', or 'survival'")
    k_folds: int = Field(default=5, description="Number of cross-validation folds", ge=2, le=10)
    time_limit: Optional[int] = Field(default=600, description="Time limit for training in seconds (currently not used by TrainerSupervised)", ge=60)
    feature_reduction_method: Optional[str] = Field(default=None, description="Feature reduction method: 'mrmr', 'variance_threshold', 'corr', 'chi2', or None")
    n_features: Optional[int] = Field(default=None, description="Number of features to select (if using feature reduction)", ge=1)


class TrainerInfo(BaseModel):
    trainer_id: str = Field(..., description="Unique identifier for the trainer")
    analyzer_id: str = Field(..., description="Associated analyzer ID")
    target_variable: str = Field(..., description="Target variable")
    task: str = Field(..., description="Task type")
    k_folds: int = Field(..., description="Number of folds")
    created_at: str = Field(..., description="Creation timestamp")
    training_time: Optional[float] = Field(None, description="Training time in seconds")


class TrainerListItem(BaseModel):
    trainer_id: str = Field(..., description="Unique identifier for the trainer")
    analyzer_id: str = Field(..., description="Associated analyzer ID")
    target_variable: str = Field(..., description="Target variable")
    task: str = Field(..., description="Task type")


class TrainerList(BaseModel):
    count: int = Field(..., description="Number of trainers")
    trainers: List[TrainerListItem] = Field(..., description="List of trainers")


class ModelScore(BaseModel):
    metric: str = Field(..., description="Metric name (e.g., 'AUROC', 'F1', 'AUPRC')")
    mean: float = Field(..., description="Mean score across folds")
    min: float = Field(..., description="Minimum score across folds")
    max: float = Field(..., description="Maximum score across folds")


class ModelLeaderboardEntry(BaseModel):
    model_name: str = Field(..., description="Model name")
    scores_test: List[ModelScore] = Field(..., description="Test scores")
    scores_val: List[ModelScore] = Field(..., description="Validation scores")
    scores_train: List[ModelScore] = Field(..., description="Training scores")


class TrainerResults(BaseModel):
    trainer_id: str = Field(..., description="Unique identifier for the trainer")
    leaderboard: List[ModelLeaderboardEntry] = Field(..., description="Model leaderboard")
    best_model: str = Field(..., description="Best performing model name")
    training_time: float = Field(..., description="Total training time in seconds")


class InferenceRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of records to predict")


class InferenceResult(BaseModel):
    trainer_id: str = Field(..., description="Associated trainer ID")
    predictions: List[Any] = Field(..., description="Predicted values")
    probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities for classification tasks")
    num_samples: int = Field(..., description="Number of samples predicted")
    created_at: str = Field(..., description="Creation timestamp")


#
# NOTE: Highcharts response wrapper removed.
# Endpoints now return raw chart payloads directly.


# =============================================================================
# DICOM Database Models
# =============================================================================

class DicomDatabaseCreate(BaseModel):
    """Response model for DICOM database creation."""
    database_id: str = Field(..., description="Unique identifier for the DICOM database")
    created_at: str = Field(..., description="Creation timestamp")
    message: str = Field(..., description="Status message")


class DicomPatientUpload(BaseModel):
    """Request model for uploading a single patient record."""
    # Accepts any key-value pairs as DICOM tags
    class Config:
        extra = "allow"


class DicomPatientsUpload(BaseModel):
    """Request model for uploading multiple patient records."""
    patients: List[Dict[str, Any]] = Field(..., description="List of patient records as DICOM tag key-value pairs")


class DicomUploadResponse(BaseModel):
    """Response model for patient upload operations."""
    database_id: str = Field(..., description="DICOM database identifier")
    patients_added: int = Field(..., description="Number of patients added")
    total_patients: int = Field(..., description="Total patients in database")
    message: str = Field(..., description="Status message")


class DicomDatabaseInfo(BaseModel):
    """Response model for DICOM database information."""
    database_id: str = Field(..., description="Unique identifier for the DICOM database")
    patient_count: int = Field(..., description="Number of patients in the database")
    keys: List[str] = Field(..., description="List of all unique DICOM tag keys in the database")
    created_at: str = Field(..., description="Creation timestamp")


class DicomDatabaseListItem(BaseModel):
    """List item for DICOM databases."""
    database_id: str = Field(..., description="Unique identifier for the DICOM database")
    patient_count: int = Field(..., description="Number of patients in the database")


class DicomDatabaseList(BaseModel):
    """Response model for listing DICOM databases."""
    count: int = Field(..., description="Number of databases")
    databases: List[DicomDatabaseListItem] = Field(..., description="List of databases")


class DicomDashboardResponse(BaseModel):
    """Response model for DICOM dashboard creation."""
    database_id: str = Field(..., description="DICOM database identifier")
    analyzer_id: str = Field(..., description="Created analyzer identifier")
    message: str = Field(..., description="Status message")