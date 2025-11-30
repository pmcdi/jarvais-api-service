"""
Dashboard module for generating Highcharts objects from jarvAIs Analyzer DashboardModule results.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from jarvais import Analyzer

from .violinplot import get_violin_plot_json
from .umap import get_umap_json

logger = logging.getLogger(__name__)


def get_dashboard_json(analyzer: Analyzer) -> List[Dict[str, Any]]:
    """
    Generate a list of Highcharts objects based on the DashboardModule's significant results.
    
    This function returns a list of Highcharts configurations for:
    1. Box plots and violin plots for each significant categorical vs continuous interaction
    2. UMAP scatterplot if available
    
    Args:
        analyzer: The jarvAIs Analyzer instance with DashboardModule results
        
    Returns:
        List of Highcharts configuration objects
    """
    charts = []
    
    # Check if we have DashboardModule and significant results
    if not hasattr(analyzer, 'dashboard_module'):
        logger.warning("Analyzer does not have a dashboard_module, results may be limited")
        # If no dashboard module, generate basic results
        significant_results = []
    else:
        # Get significant results from the DashboardModule (using private attribute)
        significant_results = analyzer.dashboard_module._significant_results
    
    if not significant_results:
        logger.info("No significant results found in dashboard_module, computing them now...")
        # If no significant results, try to compute them
        from jarvais.utils.statistical_ranking import find_top_multiplots
        
        significant_results = find_top_multiplots(
            data=getattr(analyzer, 'input_data'),
            categorical_columns=analyzer.settings.categorical_columns,
            continuous_columns=analyzer.settings.continuous_columns,
            output_dir=analyzer.settings.output_dir,
            n_top=10,
            significance_threshold=0.05
        )
        
        # Store them in the dashboard module if it exists
        if hasattr(analyzer, 'dashboard_module'):
            analyzer.dashboard_module._significant_results = significant_results
    
    # Generate box and violin plots for each significant result
    for result in significant_results:
        print("processing result: ", result)
        cat_var = result['categorical_var']
        cont_var = result['continuous_var']
        p_value = result.get('p_value', None)
        effect_size = result.get('effect_size', None)
        test_type = result.get('test_type', 'unknown')
        
        # Skip if variables don't exist in the data
        if cat_var not in analyzer.input_data.columns or cont_var not in analyzer.input_data.columns:
            logger.warning(f"Variables {cat_var} or {cont_var} not found in data")
            continue
        
        try:            
            # Generate violin plot with statistical info in title
            violin_title = f"{cont_var} distribution by {cat_var}"
            if p_value is not None:
                violin_title += f" (p={p_value:.3E})"
            
            violin_plot = get_violin_plot_json(
                analyzer.input_data,
                var_categorical=cat_var,
                var_continuous=cont_var
            )
            # Add custom title with statistical significance
            if isinstance(violin_plot, dict) and 'title' in violin_plot:
                violin_plot['title']['text'] = violin_title
            elif isinstance(violin_plot, dict):
                violin_plot['title'] = {'text': violin_title}
            
            # Add metadata about significance
            violin_plot['_metadata'] = {
                'type': 'violin_plot',
                'categorical_var': cat_var,
                'continuous_var': cont_var,
                'p_value': p_value,
                'effect_size': effect_size,
                'test_type': test_type,
                'significant': result.get('significant', False)
            }
            
            charts.append(violin_plot)
            
        except Exception as e:
            logger.error(f"Failed to generate plots for {cat_var} vs {cont_var}: {e}")
            continue
    
    print(analyzer)

    # Add UMAP plot if available
    if hasattr(analyzer, 'umap_data') and analyzer.umap_data is not None:
        try:
            # Find the best categorical variable for coloring (prefer significant ones)
            hue_var = None
            if significant_results:
                # Use the categorical variable from the most significant result
                cat_vars_by_significance = {}
                for result in significant_results:
                    cat_var = result['categorical_var']
                    if cat_var not in cat_vars_by_significance:
                        cat_vars_by_significance[cat_var] = result['p_value']
                    else:
                        cat_vars_by_significance[cat_var] = min(cat_vars_by_significance[cat_var], result['p_value'])
                
                # Sort by p-value and pick the most significant
                sorted_cat_vars = sorted(cat_vars_by_significance.items(), key=lambda x: x[1])
                for cat_var, _ in sorted_cat_vars:
                    if cat_var in analyzer.input_data.columns and analyzer.input_data[cat_var].nunique() <= 10:
                        hue_var = cat_var
                        break
            
            # Fallback to any categorical variable if no significant ones
            if not hue_var and analyzer.settings.categorical_columns:
                for col in analyzer.settings.categorical_columns:
                    if col in analyzer.input_data.columns and analyzer.input_data[col].nunique() <= 10:
                        hue_var = col
                        break
            
            hue_data = analyzer.input_data[hue_var] if hue_var else None
            
            umap_plot = get_umap_json(analyzer.umap_data, hue_data)
            
            # Add title with hue information
            umap_title = "UMAP Projection"
            if hue_var:
                umap_title += f" (colored by {hue_var})"
            
            if isinstance(umap_plot, dict) and 'title' in umap_plot:
                umap_plot['title']['text'] = umap_title
            elif isinstance(umap_plot, dict):
                umap_plot['title'] = {'text': umap_title}
            
            # Add metadata
            umap_plot['_metadata'] = {
                'type': 'umap_scatterplot',
                'hue_variable': hue_var
            }
        
            charts.append(umap_plot)
            
        except Exception as e:
            logger.error(f"Failed to generate UMAP plot: {e}")
    
    print("logger")
    # Log summary
    logger.info(f"Generated {len(charts)} dashboard charts from {len(significant_results)} significant results")
    
    return charts


def get_custom_dashboard_json(analyzer: Analyzer, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a custom dashboard based on configuration.
    For now, this just calls the main dashboard function.
    
    Args:
        analyzer: The jarvAIs Analyzer instance
        config: Configuration dictionary (not used yet)
        
    Returns:
        List of Highcharts configuration objects
    """
    # For now, just return the default dashboard
    return get_dashboard_json(analyzer)
