"""
Dashboard router for generating combined visualizations.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

from ..modules.plot import get_dashboard_json
from ..storage import storage_manager
from ..config import settings
from ..utils.rate_limit import apply_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/{analyzer_id}")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_dashboard(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance")
):
    """
    Get a comprehensive dashboard with default visualizations.
    
    This endpoint generates a dashboard containing:
    - Data summary statistics
    - Correlation heatmap for continuous variables
    - Frequency heatmaps for categorical variable pairs
    - Pie charts for categorical variables
    - UMAP scatterplot (if available)
    - Distribution plots (box and violin) for continuous vs categorical variables
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        
    Returns:
        JSON response with dashboard data containing multiple visualizations
    """
    if not storage_manager.check_analyzer(analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    try:
        # Generate dashboard with default settings
        # The dashboard module should have been run during analyzer.run() in upload
        dashboard_json = get_dashboard_json(analyzer)
        print("\n\nDASHBOARD RESULTS RECEIVED!!!")

        import json
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
        
        # Convert numpy types before JSON serialization
        serializable_dashboard_json = convert_numpy_types(dashboard_json)
        
        with open(f"dashboard_{analyzer_id}.json", "w") as f:
            json.dump(serializable_dashboard_json, f, indent=2)
        
        return serializable_dashboard_json
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")
