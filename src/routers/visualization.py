import logging
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Path, Request

from ..modules.plot import get_corr_heatmap_json, get_freq_heatmaps_json, get_pie_chart_json, get_umap_json, get_violin_plot_json, get_box_plot_json #, get_grouped_box_plot_json
from ..storage import storage_manager
from ..config import settings
from ..utils.rate_limit import apply_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["visualization"])


@router.get("/{analyzer_id}/correlation_heatmap")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_correlation_heatmap(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    method: Optional[str] = Query(None, description="Method to use for generating the heatmap (e.g., 'pearson', 'spearman')")
):
    """
    Get correlation heatmap for continuous variables.
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        method: Method to use for generating the heatmap (e.g., 'pearson', 'spearman')
        
    Returns:
        JSON response with chart data
    """
    if not storage_manager.check_analyzer(analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    try:
        analyzer = storage_manager.get_analyzer(analyzer_id)
        if not analyzer:
            raise HTTPException(status_code=404, detail="Analyzer not found")
        
        # Generate correlation heatmap
        chart_json = get_corr_heatmap_json(analyzer.input_data[analyzer.settings.continuous_columns], method=method or "pearson") # type: ignore
        return chart_json
        
    except Exception as e:
        logger.error(f"Failed to generate correlation heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")


@router.get("/{analyzer_id}/frequency_heatmap")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_frequency_heatmap(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    column1: str = Query(..., description="First categorical column"),
    column2: str = Query(..., description="Second categorical column")
):
    """
    Get frequency heatmap for categorical variables.
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        column1: First categorical column
        column2: Second categorical column
        
    Returns:
        JSON response with chart data
    """
    if not storage_manager.check_analyzer(analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    if column1 not in analyzer.settings.categorical_columns or column2 not in analyzer.settings.categorical_columns:
        raise HTTPException(status_code=400, detail="Invalid or missing categorical columns")

    try:
        # Generate frequency heatmap
        chart_json = get_freq_heatmaps_json(analyzer.input_data, column1, column2)
        return chart_json
    except Exception as e:
        logger.error(f"Failed to generate frequency heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate frequency heatmap: {str(e)}")


@router.get("/{analyzer_id}/pie_chart")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_pie_chart(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    var: str = Query(..., description="The variable to plot")
):
    """
    Get pie chart for a specific variable.
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        var: The variable to plot

    Returns:
        JSON response with chart data
    """
    if not storage_manager.check_analyzer(analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")

    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")

    try:
        chart_json = get_pie_chart_json(analyzer.input_data, var)
        return chart_json
    except Exception as e:
        logger.error(f"Failed to generate pie chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate pie chart: {str(e)}")


@router.get("/{analyzer_id}/umap_scatterplot")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_umap_plot(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    hue: Optional[str] = Query(None, description="Column to use for color coding")
):
    """
    Get UMAP scatterplot for continuous variables.
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        hue: Column to use for color coding (optional)

    Returns:
        JSON response with chart data
    """
    if not storage_manager.check_analyzer(analyzer_id):
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")
    
    if not hasattr(analyzer, 'umap_data') or analyzer.umap_data is None: # type: ignore
        raise HTTPException(status_code=400, detail="UMAP data not available for this analyzer")
    
    try:
        hue_data = None
        if hue and hue in analyzer.input_data.columns:
            hue_data = analyzer.input_data[hue]

        chart_json = get_umap_json(analyzer.umap_data, hue_data) # type: ignore
        return chart_json
    except Exception as e:
        logger.error(f"Failed to generate UMAP plot: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate UMAP plot: {str(e)}") 

@router.get("/{analyzer_id}/violin_plot")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_violin_plot(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    var_categorical: str = Query(..., description="Categorical variable"),
    var_continuous: str = Query(..., description="Continuous variable")
):
    """
    Get violin plot for a specific variable.
    """
    
    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")
    if var_categorical not in analyzer.input_data.columns:
        raise HTTPException(status_code=400, detail=f"Categorical variable '{var_categorical}' not found in data")
    if var_continuous not in analyzer.input_data.columns:
        raise HTTPException(status_code=400, detail=f"Continuous variable '{var_continuous}' not found in data")
    
    try:
        chart_json = get_violin_plot_json(
            analyzer.input_data,
            var_categorical=var_categorical,
            var_continuous=var_continuous
        )
        return chart_json
    except Exception as e:
        logger.error(f"Failed to generate violin plot: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate violin plot: {str(e)}")


@router.get("/{analyzer_id}/box_plot")
@apply_rate_limit(settings.rate_limit_visualization)
async def get_box_plot(
    request: Request,
    analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
    var_categorical: str = Query(..., description="Categorical variable"),
    var_continuous: str = Query(..., description="Continuous variable")
):
    """
    Get box plot for a specific variable.
    
    Args:
        analyzer_id: Unique identifier for the analyzer instance
        var_categorical: Categorical variable (x-axis)
        var_continuous: Continuous variable (y-axis)
        
    Returns:
        JSON response with chart data
    """
    
    analyzer = storage_manager.get_analyzer(analyzer_id)
    if not analyzer:
        raise HTTPException(status_code=404, detail="Analyzer not found")
    if var_categorical not in analyzer.input_data.columns:
        raise HTTPException(status_code=400, detail=f"Categorical variable '{var_categorical}' not found in data")
    if var_continuous not in analyzer.input_data.columns:
        raise HTTPException(status_code=400, detail=f"Continuous variable '{var_continuous}' not found in data")
    
    try:
        chart_json = get_box_plot_json(
            analyzer.input_data,
            var_categorical=var_categorical,
            var_continuous=var_continuous
        )
        return chart_json
    except Exception as e:
        logger.error(f"Failed to generate box plot: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate box plot: {str(e)}")


# @router.get("/{analyzer_id}/grouped_box_plot")
# @apply_rate_limit(settings.rate_limit_visualization)
# async def get_grouped_box_plot(
#     request: Request,
#     analyzer_id: str = Path(..., description="Unique identifier for the analyzer instance"),
#     var_categorical: str = Query(..., description="Categorical variable"),
#     var_continuous: str = Query(..., description="Continuous variable"),
#     var_grouping: str = Query(..., description="Grouping variable")
# ):
#     """
#     Get grouped box plot for specific variables.
    
#     Args:
#         analyzer_id: Unique identifier for the analyzer instance
#         var_categorical: Categorical variable (x-axis)
#         var_continuous: Continuous variable (y-axis)
#         var_grouping: Grouping variable (creates multiple series)
        
#     Returns:
#         JSON response with chart data
#     """
    
#     analyzer = storage_manager.get_analyzer(analyzer_id)
#     if not analyzer:
#         raise HTTPException(status_code=404, detail="Analyzer not found")
#     if var_categorical not in analyzer.input_data.columns:
#         raise HTTPException(status_code=400, detail=f"Categorical variable '{var_categorical}' not found in data")
#     if var_continuous not in analyzer.input_data.columns:
#         raise HTTPException(status_code=400, detail=f"Continuous variable '{var_continuous}' not found in data")
#     if var_grouping not in analyzer.input_data.columns:
#         raise HTTPException(status_code=400, detail=f"Grouping variable '{var_grouping}' not found in data")
    
#     try:
#         chart_json = get_grouped_box_plot_json(
#             analyzer.input_data,
#             var_categorical=var_categorical,
#             var_continuous=var_continuous,
#             var_grouping=var_grouping
#         )
#         return chart_json
#     except Exception as e:
#         logger.error(f"Failed to generate grouped box plot: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Failed to generate grouped box plot: {str(e)}")