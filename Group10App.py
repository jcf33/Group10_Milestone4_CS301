import dash
from dash import dcc, html, Input, Output
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Data Analysis and Prediction App", style={'textAlign': 'center'}),

    # Upload Component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    # Target Variable Dropdown
    html.Div([
        html.Label("Select Target Variable:"),
        dcc.Dropdown(id='target-dropdown', options=[], placeholder="Select a target variable")
    ], style={'margin': '20px'}),

    # Bar Charts
    html.Div([
        html.Div([
            html.Label("Bar Chart 1: Average Target by Category"),
            dcc.RadioItems(id='categorical-radio', options=[], inline=True),
            dcc.Graph(id='bar-chart-1')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Bar Chart 2: Correlation with Target"),
            dcc.Graph(id='bar-chart-2')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),

    # Train Component
    html.Div([
        html.H3("Train Model"),
        html.Label("Select Features:"),
        dcc.Checklist(id='feature-checklist', options=[], inline=True),
        html.Button("Train", id='train-button'),
        html.Div(id='model-performance', style={'marginTop': '10px'})
    ], style={'margin': '20px'}),

    # Predict Component
    html.Div([
        html.H3("Predict Target"),
        html.Label("Enter Feature Values (comma-separated):"),
        dcc.Input(id='feature-input', type='text', placeholder="Enter values"),
        html.Button("Predict", id='predict-button'),
        html.Div(id='prediction-output', style={'marginTop': '10px'})
    ], style={'margin': '20px'})
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)

