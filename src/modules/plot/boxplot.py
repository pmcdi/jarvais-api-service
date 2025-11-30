import pandas as pd
import numpy as np
from typing import Dict

def get_box_plot_json(
    data: pd.DataFrame,
    var_categorical: str,
    var_continuous: str
) -> Dict:
    """
    Generates a Highcharts JSON object for a box plot.
    
    Args:
        data (pd.DataFrame): The input dataset.
        var_categorical (str): Name of the categorical variable.
        var_continuous (str): Name of the continuous variable.
    
    Returns:
        Dict: A dictionary representing a Highcharts JSON configuration.
    """
    # Remove rows with missing values
    clean_data = data[[var_categorical, var_continuous]].dropna()
    
    # Get unique categories
    categories = sorted(clean_data[var_categorical].unique()) # type: ignore
    
    # Prepare data for box plots
    box_data = []
    outliers_data = []
    
    # Calculate box plot data for each category
    for i, category in enumerate(categories):
        category_data = clean_data[clean_data[var_categorical] == category][var_continuous]
        
        if len(category_data) > 1:
            # Calculate box plot statistics
            q1 = np.quantile(category_data, 0.25)
            q3 = np.quantile(category_data, 0.75)
            median = np.median(category_data)
            
            # Calculate outliers (values beyond 1.5 * IQR from quartiles)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            # Filter out outliers for whiskers
            filtered_data = category_data[(category_data >= lower_fence) & (category_data <= upper_fence)]
            whisker_min = filtered_data.min() if len(filtered_data) > 0 else category_data.min()
            whisker_max = filtered_data.max() if len(filtered_data) > 0 else category_data.max()
            
            # Box plot data: [x, low, q1, median, q3, high]
            box_data.append([i, whisker_min, q1, median, q3, whisker_max])
            
            # Find outliers
            outliers = category_data[(category_data < lower_fence) | (category_data > upper_fence)]
            for outlier in outliers:
                outliers_data.append([i, outlier])
    
    # Create the series
    series = [
        {
            "name": "Box Plot",
            "type": "boxplot",
            "data": box_data,
            "tooltip": {
                "headerFormat": "<em>{point.key}</em><br/>",
                "pointFormat": (
                    "Maximum: {point.high}<br/>"
                    "Upper quartile: {point.q3}<br/>"
                    "Median: {point.median}<br/>"
                    "Lower quartile: {point.q1}<br/>"
                    "Minimum: {point.low}<br/>"
                )
            }
        }
    ]
    
    # Add outliers series if there are any
    if outliers_data:
        series.append({
            "name": "Outliers",
            "type": "scatter",
            "data": outliers_data,
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": "red"
            },
            "tooltip": {
                "pointFormat": "Outlier: {point.y}"
            }
        })
    
    # Create the Highcharts configuration
    highcharts_config = {
        "chart": {
            "type": "boxplot"
        },
        "title": {
            "text": f"Box Plot: {var_continuous} by {var_categorical}"
        },
        "xAxis": {
            "categories": [str(cat) for cat in categories],
            "title": {
                "text": var_categorical
            }
        },
        "yAxis": {
            "title": {
                "text": var_continuous
            }
        },
        "plotOptions": {
            "boxplot": {
                "fillColor": "rgba(255, 255, 255, 0.8)",
                "lineWidth": 2,
                "medianColor": "#0C5DA5",
                "medianWidth": 3,
                "stemColor": "#A63400",
                "stemDashStyle": "dot",
                "stemWidth": 1,
                "whiskerColor": "#3D9200",
                "whiskerLength": "20%",
                "whiskerWidth": 3
            },
            "scatter": {
                "marker": {
                    "radius": 3
                }
            }
        },
        "series": series
    }
    
    return highcharts_config


def get_grouped_box_plot_json(
    data: pd.DataFrame,
    var_categorical: str,
    var_continuous: str,
    var_grouping: str
) -> Dict:
    """
    Generates a Highcharts JSON object for a grouped box plot.
    
    Args:
        data (pd.DataFrame): The input dataset.
        var_categorical (str): Name of the categorical variable (x-axis).
        var_continuous (str): Name of the continuous variable (y-axis).
        var_grouping (str): Name of the grouping variable (creates multiple series).
    
    Returns:
        Dict: A dictionary representing a Highcharts JSON configuration.
    """
    # Remove rows with missing values
    clean_data = data[[var_categorical, var_continuous, var_grouping]].dropna()
    
    # Get unique categories and groups
    categories = sorted(clean_data[var_categorical].unique()) # type: ignore
    groups = sorted(clean_data[var_grouping].unique()) # type: ignore
    
    # Prepare series for each group
    series = []
    colors = ["#7cb5ec", "#434348", "#90ed7d", "#f7a35c", "#8085e9", "#f15c80", "#e4d354", "#2b908f", "#f45b5b", "#91e8e1"]
    
    for group_idx, group in enumerate(groups):
        group_data = clean_data[clean_data[var_grouping] == group]
        box_data = []
        outliers_data = []
        
        for i, category in enumerate(categories):
            category_data = group_data[group_data[var_categorical] == category][var_continuous]
            
            if len(category_data) > 1:
                # Calculate box plot statistics
                q1 = np.quantile(category_data, 0.25)
                q3 = np.quantile(category_data, 0.75)
                median = np.median(category_data)
                
                # Calculate outliers (values beyond 1.5 * IQR from quartiles)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                
                # Filter out outliers for whiskers
                filtered_data = category_data[(category_data >= lower_fence) & (category_data <= upper_fence)]
                whisker_min = filtered_data.min() if len(filtered_data) > 0 else category_data.min()
                whisker_max = filtered_data.max() if len(filtered_data) > 0 else category_data.max()
                
                # Box plot data: [x, low, q1, median, q3, high]
                box_data.append([i, whisker_min, q1, median, q3, whisker_max])
                
                # Find outliers
                outliers = category_data[(category_data < lower_fence) | (category_data > upper_fence)]
                for outlier in outliers:
                    outliers_data.append([i, outlier])
        
        # Add box plot series for this group
        color = colors[group_idx % len(colors)]
        series.append({
            "name": f"{group}",
            "type": "boxplot",
            "data": box_data,
            "color": color,
            "tooltip": {
                "headerFormat": f"<em>{group} - {{point.key}}</em><br/>",
                "pointFormat": (
                    "Maximum: {point.high}<br/>"
                    "Upper quartile: {point.q3}<br/>"
                    "Median: {point.median}<br/>"
                    "Lower quartile: {point.q1}<br/>"
                    "Minimum: {point.low}<br/>"
                )
            }
        })
        
        # Add outliers series for this group if there are any
        if outliers_data:
            series.append({
                "name": f"{group} Outliers",
                "type": "scatter",
                "data": outliers_data,
                "color": color,
                "marker": {
                    "fillColor": "white",
                    "lineWidth": 1,
                    "lineColor": color
                },
                "tooltip": {
                    "pointFormat": f"{group} Outlier: {{point.y}}"
                },
                "linkedTo": ":previous"
            })
    
    # Create the Highcharts configuration
    highcharts_config = {
        "chart": {
            "type": "boxplot"
        },
        "title": {
            "text": f"Grouped Box Plot: {var_continuous} by {var_categorical} (grouped by {var_grouping})"
        },
        "xAxis": {
            "categories": [str(cat) for cat in categories],
            "title": {
                "text": var_categorical
            }
        },
        "yAxis": {
            "title": {
                "text": var_continuous
            }
        },
        "plotOptions": {
            "boxplot": {
                "fillColor": "rgba(255, 255, 255, 0.8)",
                "lineWidth": 2,
                "medianColor": "#0C5DA5",
                "medianWidth": 3,
                "stemColor": "#A63400",
                "stemDashStyle": "dot",
                "stemWidth": 1,
                "whiskerColor": "#3D9200",
                "whiskerLength": "20%",
                "whiskerWidth": 3
            },
            "scatter": {
                "marker": {
                    "radius": 3
                }
            }
        },
        "series": series
    }
    
    return highcharts_config 