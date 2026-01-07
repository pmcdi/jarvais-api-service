"""
DICOM Router for managing DICOM patient databases.

This module provides endpoints for creating and managing DICOM patient databases,
uploading patient records with DICOM tag key-value pairs, exporting data to CSV,
and generating analytical dashboards.

Endpoints:
- POST /dicom - Create empty database, return database_id
- POST /dicom/{database_id}/patient - Upload single patient JSON dict
- POST /dicom/{database_id}/patients - Upload array of patient JSON dicts
- GET /dicom/{database_id}/export - Export database as CSV text (comma-delimited)
- GET /dicom/{database_id}/dashboard - Convert DB to DataFrame, create Analyzer, return dashboard JSON
- GET /dicom/{database_id} - Get database info
- DELETE /dicom/{database_id} - Delete database

Example usage for export endpoint:
    
    ```python
    import requests
    
    # Export DICOM database to CSV
    response = requests.get(f"http://localhost:8888/dicom/{database_id}/export")
    
    # Save to local file
    with open("patients.csv", "w") as f:
        f.write(response.text)
    ```
"""

import io
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import PlainTextResponse
import pandas as pd

from jarvais import Analyzer

from ..storage import storage_manager, DicomDatabase
from ..models import (
    DicomDatabaseCreate,
    DicomPatientsUpload,
    DicomUploadResponse,
    DicomDatabaseInfo,
    DicomDatabaseList,
    DicomDatabaseListItem,
    DicomDashboardResponse,
    SuccessResponse,
)
from ..config import settings
from ..utils.rate_limit import apply_rate_limit
from ..modules.dicom_tags import VALID_DICOM_KEYWORDS, get_tag_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dicom", tags=["dicom"])


def get_umap(data: pd.DataFrame, continuous_columns: list) -> pd.DataFrame:
    """
    Generate UMAP projection for continuous variables.
    
    Args:
        data: Input DataFrame containing the data.
        continuous_columns: List of continuous variable column names.
        
    Returns:
        UMAP transformed data as DataFrame.
    """
    from umap import UMAP
    umap_data = UMAP(n_components=2, random_state=42).fit_transform(data[continuous_columns])
    return pd.DataFrame(umap_data, columns=pd.Index(['UMAP1', 'UMAP2']), index=data.index)


def drop_uid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with number of unique values equal to the number of rows.
    These are typically UID columns that don't provide analytical value.
    """
    uid_columns = [col for col in df.columns if df[col].nunique() == df.shape[0]]
    if uid_columns:
        logger.info(f"Dropping UID columns: {uid_columns}")
    return df.drop(columns=uid_columns)


@router.post("", response_model=DicomDatabaseCreate, status_code=201)
@apply_rate_limit(settings.rate_limit_general)
async def create_database(request: Request):
    """
    Create a new empty DICOM patient database.
    
    Returns a unique database_id that can be used to upload patient records
    and perform analysis operations.
    
    Returns:
        DicomDatabaseCreate: Contains the database_id and creation timestamp
    """
    database_id = str(uuid.uuid4())
    database = DicomDatabase(database_id)
    
    storage_manager.store_dicom(database_id, database)
    
    logger.info(f"Created new DICOM database: {database_id}")
    
    return DicomDatabaseCreate(
        database_id=database_id,
        created_at=database.created_at,
        message="DICOM database created successfully"
    )


@router.post("/{database_id}/patient", response_model=DicomUploadResponse)
@apply_rate_limit(settings.rate_limit_upload)
async def upload_patient(
    request: Request,
    patient_data: Dict[str, Any],
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Upload a single patient record to the DICOM database.
    
    The patient record should be a JSON dictionary with DICOM tags as keys
    and their values. Both standard DICOM keywords (e.g., "PatientID", "Modality")
    and custom keys are accepted.
    
    Args:
        database_id: Unique identifier for the DICOM database
        patient_data: JSON dictionary with DICOM tag key-value pairs
        
    Returns:
        DicomUploadResponse: Upload confirmation with patient counts
        
    Example request body:
        ```json
        {
            "PatientID": "12345",
            "PatientName": "John Doe",
            "PatientAge": "045Y",
            "Modality": "CT",
            "StudyDescription": "Chest CT with contrast"
        }
        ```
    """
    if not storage_manager.check_dicom(database_id):
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    database = storage_manager.get_dicom(database_id)
    if not database:
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    try:
        database.add_patient(patient_data)
        storage_manager.store_dicom(database_id, database)
        
        logger.info(f"Added 1 patient to database {database_id}")
        
        return DicomUploadResponse(
            database_id=database_id,
            patients_added=1,
            total_patients=database.get_patient_count(),
            message="Patient record uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Failed to upload patient: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload patient: {str(e)}")


@router.post("/{database_id}/patients", response_model=DicomUploadResponse)
@apply_rate_limit(settings.rate_limit_upload)
async def upload_patients(
    request: Request,
    patients_data: DicomPatientsUpload,
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Upload multiple patient records to the DICOM database.
    
    Each patient record should be a JSON dictionary with DICOM tags as keys
    and their values. Both standard DICOM keywords and custom keys are accepted.
    
    Args:
        database_id: Unique identifier for the DICOM database
        patients_data: Object containing list of patient records
        
    Returns:
        DicomUploadResponse: Upload confirmation with patient counts
        
    Example request body:
        ```json
        {
            "patients": [
                {
                    "PatientID": "12345",
                    "PatientName": "John Doe",
                    "Modality": "CT"
                },
                {
                    "PatientID": "12346",
                    "PatientName": "Jane Smith",
                    "Modality": "MR"
                }
            ]
        }
        ```
    """
    if not storage_manager.check_dicom(database_id):
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    database = storage_manager.get_dicom(database_id)
    if not database:
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    try:
        count = database.add_patients(patients_data.patients)
        storage_manager.store_dicom(database_id, database)
        
        logger.info(f"Added {count} patients to database {database_id}")
        
        return DicomUploadResponse(
            database_id=database_id,
            patients_added=count,
            total_patients=database.get_patient_count(),
            message=f"{count} patient records uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Failed to upload patients: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload patients: {str(e)}")


@router.get("/{database_id}/export", response_class=PlainTextResponse)
@apply_rate_limit(settings.rate_limit_general)
async def export_database(
    request: Request,
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Export the DICOM database as comma-delimited CSV text.
    
    Returns the database contents as plain text in CSV format that can be
    saved directly to a file using standard file I/O operations.
    
    Args:
        database_id: Unique identifier for the DICOM database
        
    Returns:
        Plain text response with CSV content (text/csv content type)
        - Header row contains all DICOM tag names present in the database
        - Each subsequent row represents one patient record
        
    Example usage:
        ```python
        import requests
        
        response = requests.get(f"http://localhost:8888/dicom/{database_id}/export")
        
        # Save to local CSV file
        with open("patients.csv", "w") as f:
            f.write(response.text)
        ```
    """
    if not storage_manager.check_dicom(database_id):
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    database = storage_manager.get_dicom(database_id)
    if not database:
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    try:
        # Convert to DataFrame for easy CSV export
        if not database.patients:
            # Return empty CSV with no headers if no patients
            return PlainTextResponse(
                content="",
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=dicom_export_{database_id}.csv"}
            )
        
        df = pd.DataFrame(database.patients)
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        logger.info(f"Exported database {database_id} with {len(database.patients)} patients")
        
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=dicom_export_{database_id}.csv"}
        )
    except Exception as e:
        logger.error(f"Failed to export database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export database: {str(e)}")


@router.get("/{database_id}/dashboard")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_dashboard(
    request: Request,
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Generate an analytical dashboard for the DICOM database.
    
    Converts the current state of the DICOM database into a DataFrame,
    creates a jarvais Analyzer instance, and returns the dashboard JSON
    with visualizations including:
    - Distribution plots for continuous vs categorical variables
    - UMAP scatterplot (if continuous variables are available)
    - Statistical significance analysis
    
    The endpoint also creates and stores an analyzer instance that can be
    accessed via the analyzer_id for further analysis.
    
    Args:
        database_id: Unique identifier for the DICOM database
        
    Returns:
        JSON response containing:
        - database_id: The DICOM database identifier
        - analyzer_id: The created analyzer identifier for further analysis
        - dashboard: List of Highcharts configuration objects for visualizations
        
    Raises:
        404: If database not found
        400: If database is empty
        500: If dashboard generation fails
    """
    if not storage_manager.check_dicom(database_id):
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    database = storage_manager.get_dicom(database_id)
    if not database:
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    if not database.patients:
        raise HTTPException(status_code=400, detail="DICOM database is empty. Upload patient records first.")
    
    try:
        # Convert database to DataFrame
        df = pd.DataFrame(database.patients)
        df = drop_uid_columns(df)
        
        # Generate unique analyzer ID
        analyzer_id = str(uuid.uuid4())
        
        # Initialize Analyzer
        analyzer = Analyzer(df, settings.upload_folder)
        analyzer.run()
        
        # Calculate UMAP of continuous variables if available
        if analyzer.settings.continuous_columns:
            try:
                analyzer.umap_data = get_umap(
                    analyzer.data,
                    continuous_columns=analyzer.settings.continuous_columns
                )
            except Exception as e:
                logger.warning(f"Failed to compute UMAP: {e}")
                analyzer.umap_data = None
        
        # Store analyzer instance
        storage_manager.store_analyzer(analyzer_id, analyzer)
        
        # Generate dashboard
        from ..modules.plot import get_dashboard_json
        dashboard_json = get_dashboard_json(analyzer)
        
        # Convert numpy types for JSON serialization
        import numpy as np
        
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable Python types."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_dashboard = convert_numpy_types(dashboard_json)
        
        logger.info(f"Generated dashboard for database {database_id}, created analyzer {analyzer_id}")
        
        return {
            "database_id": database_id,
            "analyzer_id": analyzer_id,
            "patient_count": database.get_patient_count(),
            "dashboard": serializable_dashboard
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")


@router.get("/{database_id}", response_model=DicomDatabaseInfo)
@apply_rate_limit(settings.rate_limit_general)
async def get_database_info(
    request: Request,
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Get information about a specific DICOM database.
    
    Args:
        database_id: Unique identifier for the DICOM database
        
    Returns:
        DicomDatabaseInfo: Database information including patient count and keys
    """
    if not storage_manager.check_dicom(database_id):
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    database = storage_manager.get_dicom(database_id)
    if not database:
        raise HTTPException(status_code=404, detail="DICOM database not found")
    
    return DicomDatabaseInfo(
        database_id=database.database_id,
        patient_count=database.get_patient_count(),
        keys=database.get_all_keys(),
        created_at=database.created_at
    )


@router.delete("/{database_id}", response_model=SuccessResponse)
@apply_rate_limit(settings.rate_limit_general)
async def delete_database(
    request: Request,
    database_id: str = Path(..., description="Unique identifier for the DICOM database")
):
    """
    Delete a DICOM database.
    
    Args:
        database_id: Unique identifier for the DICOM database
        
    Returns:
        SuccessResponse: Confirmation of deletion
    """
    if storage_manager.delete_dicom(database_id):
        logger.info(f"Deleted DICOM database {database_id}")
        return SuccessResponse(message=f"DICOM database {database_id} deleted successfully")
    else:
        raise HTTPException(status_code=404, detail="DICOM database not found")


@router.get("", response_model=DicomDatabaseList)
@apply_rate_limit(settings.rate_limit_general)
async def list_databases(request: Request):
    """
    List all DICOM databases.
    
    Returns:
        DicomDatabaseList: List of all DICOM databases with their info
    """
    database_ids = storage_manager.list_dicom_ids()
    
    databases = []
    for db_id in database_ids:
        database = storage_manager.get_dicom(db_id)
        if database:
            databases.append(DicomDatabaseListItem(
                database_id=db_id,
                patient_count=database.get_patient_count()
            ))
    
    return DicomDatabaseList(count=len(databases), databases=databases)
