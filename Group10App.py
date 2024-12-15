import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import io
import base64

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Group 10's Data Analysis and Prediction App"

# Global dataset variable
global_data = None

# Define app layout
app.layout = html.Div([
    html.H1("Data Analysis and Prediction App", style={'textAlign': 'center'}),

    # Upload Component
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='upload-feedback'),

    # Dropdown and Charts
    html.Div([
        html.Label("Select Target Variable:"),
        dcc.Dropdown(id='target-dropdown', placeholder="Select a target variable"),
    ]),
    html.Div([
        html.Div([dcc.Graph(id='bar-chart-1')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-2')], style={'width': '48%', 'display': 'inline-block'}),
    ]),

    # Train Model
    html.Div([
        html.H3("Train Model"),
        dcc.Checklist(id='feature-checklist', inline=True),
        html.Button("Train", id='train-button'),
        html.Div(id='model-performance'),
    ]),

    # Predict Target
    html.Div([
        html.H3("Predict Target"),
        dcc.Input(id='feature-input', type='text', placeholder="Enter values"),
        html.Button("Predict", id='predict-button'),
        html.Div(id='prediction-output'),
    ])
])

# Upload Callback
@app.callback(
    [Output('upload-feedback', 'children'),
     Output('target-dropdown', 'options'),
     Output('feature-checklist', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global global_data
    if not contents:
        return "", [], []

    content_type, content_string = contents.split(',')
    global_data = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))

    numeric_columns = [{'label': col, 'value': col} for col in global_data.select_dtypes(include=['number']).columns]
    return f"Uploaded file: {filename}", numeric_columns, numeric_columns

# Chart Update Callback
@app.callback(
    [Output('bar-chart-1', 'figure'),
     Output('bar-chart-2', 'figure')],
    [Input('target-dropdown', 'value')]
)
def update_charts(target):
    if global_data is None or target is None:
        return {}, {}

    avg_target = global_data.groupby(global_data.columns[0])[target].mean().reset_index()
    correlation = global_data.corr()[[target]].abs().sort_values(by=target, ascending=False)

    bar_chart_1 = {'data': [{'x': avg_target[global_data.columns[0]], 'y': avg_target[target], 'type': 'bar'}]}
    bar_chart_2 = {'data': [{'x': correlation.index, 'y': correlation[target], 'type': 'bar'}]}
    return bar_chart_1, bar_chart_2

# Model Training Callback
@app.callback(
    Output('model-performance', 'children'),
    Input('train-button', 'n_clicks'),
    [State('feature-checklist', 'value'), State('target-dropdown', 'value')]
)
def train_model(n_clicks, features, target):
    if not n_clicks or not features or not target:
        return "Select features and target."

    X = global_data[features]
    y = global_data[target]
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    r2 = r2_score(y_test, pipeline.predict(X_test))
    return f"Model trained. R² Score: {r2:.2f}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
