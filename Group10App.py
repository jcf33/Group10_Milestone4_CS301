import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import io
import base64

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Data Analysis and Prediction App"

global_data = None

app.layout = html.Div([
    html.H1("Group 10's Data Analysis and Prediction App", style={'textAlign': 'center'}),

    # To upload layout
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
        },
        multiple=False
    ),
    html.Div(id='upload-feedback'),

    # Select Target Variable
    html.Div([
        html.Label("Select Target Variable:"),
        dcc.Dropdown(id='target-dropdown', placeholder="Select a target variable"),
    ], style={'margin': '20px'}),

    # Bar Charts
    html.Div([
        html.Div([
            html.Label("Bar Chart 1: Average Target by Category"),
            dcc.RadioItems(id='categorical-radio', inline=True),
            dcc.Graph(id='bar-chart-1')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Bar Chart 2: Correlation with Target"),
            dcc.Graph(id='bar-chart-2')
        ], style={'width': '48%', 'display': 'inline-block'}),
    ]),

    # Train Component
    html.Div([
        html.H3("Train Model"),
        html.Label("Select Features:"),
        dcc.Checklist(id='feature-checklist', inline=True),
        html.Button("Train", id='train-button'),
        html.Div(id='model-performance', style={'marginTop': '10px'}),
    ], style={'margin': '20px'}),

    # Predict Component
    html.Div([
        html.H3("Predict Target"),
        html.Label("Enter Feature Values (comma-separated):"),
        dcc.Input(id='feature-input', type='text', placeholder="Enter values"),
        html.Button("Predict", id='predict-button'),
        html.Div(id='prediction-output', style={'marginTop': '10px'}),
    ], style={'margin': '20px'})
])

# Dash callbacks
@app.callback(
    [Output('upload-feedback', 'children'),
     Output('target-dropdown', 'options'),
     Output('categorical-radio', 'options'),
     Output('feature-checklist', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global global_data
    if contents is None:
        return "", [], [], []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)  
    global_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))  

    numeric_columns = [{'label': col, 'value': col} for col in global_data.select_dtypes(include=['float64', 'int64']).columns]
    categorical_columns = [{'label': col, 'value': col} for col in global_data.select_dtypes(include=['object', 'category']).columns]
    all_columns = [{'label': col, 'value': col} for col in global_data.select_dtypes(include=['object', 'category', 'float64', 'int64']).columns]

    return f"Uploaded file: {filename}", numeric_columns, categorical_columns, all_columns


@app.callback(
    [Output('bar-chart-1', 'figure'),
     Output('bar-chart-2', 'figure')],
    [Input('target-dropdown', 'value'),
     Input('categorical-radio', 'value')]
)
def update_charts(target, categorical):
    if global_data is None or target is None or categorical is None:
        return {}, {}

    try:
        # Bar Chart 1
        if global_data[categorical].dtype not in ['object', 'category']:
            return {}, {}  

        avg_target = global_data.groupby(categorical)[target].mean().reset_index()

        # Bar Chart 2
        numeric_data = global_data.select_dtypes(include=['number']) 
        if target not in numeric_data.columns:
            return {}, {}

        correlation = numeric_data.corr()[[target]].abs().sort_values(by=target, ascending=False).reset_index()

        # Create Charts
        bar_chart_1 = {
            'data': [{'x': avg_target[categorical], 'y': avg_target[target], 'type': 'bar'}],
            'layout': {'title': f"Average {target} by {categorical}"}
        }
        bar_chart_2 = {
            'data': [{'x': correlation['index'], 'y': correlation[target], 'type': 'bar'}],
            'layout': {'title': f"Correlation of Features with {target}"}
        }

        return bar_chart_1, bar_chart_2

    except Exception as e:
        print(f"Error updating charts: {e}")
        return {}, {}

@app.callback(
    Output('model-performance', 'children'),
    Input('train-button', 'n_clicks'),
    [State('feature-checklist', 'value'),
     State('target-dropdown', 'value')]
)
def train_model(n_clicks, features, target):
    if n_clicks is None or global_data is None or not features or not target:
        return "Please upload data, select features, and a target variable."

    # Prepping the data
    X = global_data[features].copy()
    y = global_data[target]

    # missing values handling
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].mean())  
        elif X[col].dtype in ['object', 'category']:
            X[col] = X[col].fillna(X[col].mode()[0])  

    # Encode categorical variables
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].factorize()[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Regressor with Grid Search
    param_grid_gbr = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    grid_gbr = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gbr, cv=5, scoring='r2')
    grid_gbr.fit(X_train, y_train)
    best_gbr = grid_gbr.best_estimator_

    # Evaluate model
    y_pred = best_gbr.predict(X_test)
    r2_gbr = r2_score(y_test, y_pred)

    return f"Model trained. Best Gradient Boosting RÂ² Score: {r2_gbr:.2f}"


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('feature-input', 'value'),
     State('feature-checklist', 'value')]
)
def predict_target(n_clicks, input_values, features):
    if n_clicks is None or global_data is None or not features or not input_values:
        return "Please provide input values and select features."

    input_values = list(map(float, input_values.split(',')))
    if len(input_values) != len(features):
        return "Invalid input length. Ensure values match selected features."

    # Encode categorical features
    for col in global_data[features].select_dtypes(include=['object', 'category']).columns:
        global_data[col] = global_data[col].factorize()[0]

    pipeline = GradientBoostingRegressor(random_state=42)
    X = global_data[features]
    y = global_data.iloc[:, 0]  
    pipeline.fit(X, y)
    prediction = pipeline.predict([input_values])[0]

    return f"Predicted Target Value: {prediction:.2f}"

# To host the app
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
