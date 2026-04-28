import os
from pathlib import Path
import sys

import dash
import dash_bootstrap_components as dbc
import yaml
from dash import Input, Output, callback, dcc, html

# Project root and config
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

import requests
import pandas as pd
from app_ui.utils import load_data, create_figure

with open(project_root / "conf" / "base" / "parameters.yml") as f:
    params = yaml.safe_load(f)
    config = params["ui"]
    runner_config = params["pipeline_runner"]

ACTUAL_DATA_PATH = project_root / config["actual_data_path"]

# Global store for predictions history
predictions_history = pd.DataFrame(columns=["datetime", "prediction"])

# App layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Interval(id="interval", interval=config["update_interval_ms"], n_intervals=0),
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Control Panel", style={"color": "#222"}),
            html.Div([
                html.Label("Plots display time (last N hours)", style={"color": "#222"}),
                dcc.Input(id="lookback-hours", type="number", min=1, step=1,
                          value=config["default_lookback_hours"], style={"width": "100%"}),
            ], className="card"),
            html.Div([
                html.H5("ML Application Overview", style={"marginTop": "8px", "color": "#222"}),
                html.Ul([
                    html.Li("ML App forecasts the amount of rented bikes for the next hour", style={"marginBottom": "8px"}),
                    html.Li("ML App consists of feature engineering, training, and inference pipelines", style={"marginBottom": "8px"}),
                    html.Li("Inference runs automatically every 1 second simulating 1h of dataset time", style={"marginBottom": "8px"}),
                    html.Li("Plot shows forecasted vs actual rented bike count", style={"marginBottom": "8px"}),
                    html.Li("The UI app and inference pipeline run in Docker containers", style={"marginBottom": "8px"}),
                    html.Li("The data and model are stored and shared in Docker volumes"),
                ], style={"color": "#444", "fontSize": "14px", "paddingLeft": "20px"}),
            ], className="card card-margin-top"),
        ], width=3, style={"paddingTop": "10px"}),
        # Main chart
        dbc.Col([
            html.H5("Real-time Bike Count Predictions", style={"color": "#222", "fontSize": "28px"}),
            dcc.Graph(id="graph", style={"backgroundColor": "#fff", "borderRadius": "12px", "padding": "8px"}),
        ], width=9, style={"paddingTop": "10px"}),
    ], align="start"),
], fluid=True, style={"backgroundColor": "#e9e9f0", "minHeight": "100vh", "padding": "20px"})


@callback(
    Output("graph", "figure"), 
    [
        Input("lookback-hours", "value"), 
        Input("interval", "n_intervals")
        ]
    )
def update_graph(lookback_hours, n_intervals):
    global predictions_history
    
    df_actual = load_data(ACTUAL_DATA_PATH)
    
    # Simulation: find the batch for the current step (n_intervals)
    first_timestamp = pd.to_datetime(runner_config["first_timestamp"])
    first_idx = df_actual[df_actual["datetime"] >= first_timestamp].index[0]
    
    current_idx = first_idx + n_intervals
    batch_size = runner_config["batch_size"]
    
    batch_start = max(0, current_idx - batch_size + 1)
    batch_end = current_idx + 1
    
    if batch_end <= len(df_actual):
        batch = df_actual.iloc[batch_start:batch_end].copy()
        batch["datetime"] = batch["datetime"].astype(str)
        
        try:
            # Petición HTTP al servicio de inferencia usando requests
            response = requests.post("http://ml-inference:8000/predict", json=batch.to_dict(orient="records"))
            if response.status_code == 200:
                pred_data = response.json()
                new_preds = pd.DataFrame(pred_data)
                new_preds["datetime"] = pd.to_datetime(new_preds["datetime"])
                
                # Update global history
                if not new_preds.empty:
                    predictions_history = pd.concat([predictions_history, new_preds]).drop_duplicates(subset=["datetime"], keep="last")
        except Exception as e:
            print(f"Error calling API: {e}")

    # Set default lookback hours if not provided
    if not lookback_hours or lookback_hours < 1:
        lookback_hours = config["default_lookback_hours"]
        
    figure = create_figure(df_actual, predictions_history, lookback_hours)
    return figure

server = app.server

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8050)