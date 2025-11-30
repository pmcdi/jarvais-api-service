import pandas as pd
from typing import Dict

def get_pie_chart_json(data: pd.DataFrame, var: str) -> Dict:
    """
    Generates a Highcharts JSON object for a pie chart

    Args:
        data (pd.DataFrame): The input data.
        var (str): The variable to plot.
    """
    pie_data_grouped = data.groupby(var, observed=False).size().sort_values(ascending=False)
    pie_series_data = [
        {"name": str(label), "y": value}
        for label, value in pie_data_grouped.items()
    ]
    pie_chart_json = {
        "chart": {"type": "pie"},
        "title": {"text": f"{var} Distribution. N: {data[var].count()}"},
        "tooltip": {"pointFormat": '{series.name}: <b>{point.percentage:.1f}%</b>'},
        "plotOptions": {
            "pie": {
                "allowPointSelect": True,
                "cursor": 'pointer',
                "dataLabels": {"enabled": True, "format": '<b>{point.name}</b>: {point.percentage:.1f} %'}
            }
        },
        "series": [{"name": "Count", "data": pie_series_data}]
    }
    return pie_chart_json

