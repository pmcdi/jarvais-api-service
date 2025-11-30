import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import gaussian_kde


def get_violin_plot_json(
    data: pd.DataFrame,
    var_categorical: str,
    var_continuous: str
) -> Dict:
    """
    Generates a Highcharts JSON object for a violin plot.
    
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
    violin_series = []
    
    # Calculate violin plot data for each category
    for i, category in enumerate(categories):
        category_data = clean_data[clean_data[var_categorical] == category][var_continuous]
        
        if len(category_data) > 1:
            # Calculate box plot statistics
            q1 = np.quantile(category_data, 0.25)
            q3 = np.quantile(category_data, 0.75)
            median = np.median(category_data)
            min_val = category_data.min()
            max_val = category_data.max()
            
            # Calculate outliers (values beyond 1.5 * IQR from quartiles)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            # Filter out outliers for whiskers
            filtered_data = category_data[(category_data >= lower_fence) & (category_data <= upper_fence)]
            whisker_min = filtered_data.min() if len(filtered_data) > 0 else min_val
            whisker_max = filtered_data.max() if len(filtered_data) > 0 else max_val
            
            # Box plot data: [x, low, q1, median, q3, high]
            box_data.append([i, whisker_min, q1, median, q3, whisker_max])
            
            # Calculate density curve for violin effect
            if len(category_data) > 2:
                try:
                    # Use kernel density estimation
                    kde = gaussian_kde(category_data)
                    
                    # Create points for density curve
                    data_range = np.linspace(category_data.min(), category_data.max(), 100)
                    density = kde(data_range)
                    
                    # Normalize density to reasonable width (0.3 units on each side)
                    max_density = density.max()
                    if max_density > 0:
                        normalized_density = (density / max_density) * 0.3
                        
                        # Create violin shape (mirror the density curve)
                        violin_points_left = [[i - d, y] for d, y in zip(normalized_density, data_range)]
                        violin_points_right = [[i + d, y] for d, y in zip(normalized_density, data_range)]
                        
                        # Combine left and right sides
                        violin_points = violin_points_left + violin_points_right[::-1]
                        
                        violin_series.append({
                            "name": f"{category} Density",
                            "type": "polygon",
                            "data": violin_points,
                            "fillOpacity": 0.3,
                            "lineWidth": 1,
                            "color": f"rgba({50 + i * 40}, {100 + i * 30}, {200 - i * 20}, 0.6)",
                            "showInLegend": False
                        })
                except Exception as e:
                    print(f"KDE failed: {e}")
                    # If KDE fails, fall back to box plot only (violin series will be empty)
                    # This ensures we always have at least box plot visualization
                    continue
    
    # Create the Highcharts configuration
    # Determine chart title based on whether violin shapes were created
    chart_title = f"Violin Plot: {var_continuous} by {var_categorical}" if violin_series else f"Box Plot: {var_continuous} by {var_categorical}"
    
    highcharts_config = {
        "chart": {
            "type": "boxplot"
        },
        "title": {
            "text": chart_title
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
            }
        },
        "series": [
            {
                "name": "Box Plot",
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
        ] + violin_series
    }
    
    return highcharts_config 