from itertools import combinations
import pandas as pd
from typing import Dict, List

def get_freq_heatmaps_json(
    data: pd.DataFrame,
    column_1: str, 
    column_2: str
) -> List[Dict]:
    """
    Generates Highcharts heatmap JSON for all column pair combinations.

    Args:
        data (pd.DataFrame): Input dataset.
        columns (list): List of column names to create frequency tables for.

    Returns:
        List[Dict]: A list of dictionaries, each a Highcharts JSON config.
    """
    heatmap_data = pd.crosstab(data[column_1], data[column_2])

    # Prepare data for Highcharts
    series_data = []
    y_categories = heatmap_data.index.astype(str).tolist()
    x_categories = heatmap_data.columns.astype(str).tolist()

    for y, row_label in enumerate(y_categories):
        for x, col_label in enumerate(x_categories):
            value = heatmap_data.loc[row_label, col_label]
            series_data.append([x, y, int(value)])

    # Create the Highcharts JSON object for this pair
    highcharts_config = {
        "chart": {"type": 'heatmap', "plotBorderWidth": 1},
        "title": {"text": f'Frequency Table for {column_1} and {column_2}'},
        "xAxis": {"categories": x_categories, "title": {"text": column_2}},
        "yAxis": {"categories": y_categories, "title": {"text": column_1}},
        "colorAxis": {"minColor": '#FFFFFF', "maxColor": '#007bff'}, # Example: white to blue
        "series": [{
            "name": 'Frequency',
            "borderWidth": 1,
            "data": series_data,
            "dataLabels": {
                "enabled": True,
                "color": '#000000',
                "format": '{point.value}'
            }
        }]
    }

    return highcharts_config