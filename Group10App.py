from dash import dcc, html, Input, Output, State
import dash
import pandas as pd
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Data Analysis and Prediction App"

global_data = None
best_model = None

app.layout = html.Div([
    html.H1("Group 10's Data Analysis and Prediction App", style={'textAlign': 'center'}),

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

    html.Div([
        html.Label("Select Target Variable:"),
        dcc.Dropdown(id='target-dropdown', placeholder="Select a target variable"),
    ], style={'margin': '20px'}),

    html.Div([
        html.H3("Train Model"),
        html.Label("Select Features:"),
        dcc.Checklist(id='feature-checklist', inline=True),
        html.Button("Train", id='train-button'),
        html.Div(id='model-performance', style={'marginTop': '10px'}),
    ], style={'margin': '20px'}),

    html.Div([
        html.H3("Predict Target"),
        html.Label("Enter Feature Values (comma-separated):"),
        dcc.Input(id='feature-input', type='text', placeholder="Enter values"),
        html.Button("Predict", id='predict-button'),
        html.Div(id='prediction-output', style={'marginTop': '10px'}),
    ], style={'margin': '20px'})
])


# File Upload Callback
@app.callback(
    [Output('upload-feedback', 'children'),
     Output('target-dropdown', 'options'),
     Output('feature-checklist', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global global_data
    if contents is None:
        return "", [], []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    numeric_columns = [{'label': col, 'value': col} for col in global_data.select_dtypes(include=['float64', 'int64']).columns]
    return f"Uploaded file: {filename}", numeric_columns, numeric_columns


# Model Training Callback
@app.callback(
    Output('model-performance', 'children'),
    Input('train-button', 'n_clicks'),
    [State('feature-checklist', 'value'),
     State('target-dropdown', 'value')]
)
def train_model(n_clicks, features, target):
    global best_model

    if n_clicks is None or global_data is None or not features or not target:
        return "Please upload data, select features, and a target variable."

    try:
        X = global_data[features]
        y = global_data[target]

        # Define preprocessing for numeric and categorical features
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Create pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)
        best_model = model

        # Evaluate the model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return f"Model trained successfully. R² Score: {r2:.2f}"

    except Exception as e:
        print(f"Error during model training: {e}")
        return "Model training failed. Please check your data and selections."


# Prediction Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('feature-input', 'value'),
     State('feature-checklist', 'value')]
)
def predict_target(n_clicks, input_values, features):
    global best_model

    if n_clicks is None or global_data is None or not features or not input_values:
        return "Please provide input values and select features."

    try:
        # Convert input values to float
        input_values = list(map(float, input_values.split(',')))
        if len(input_values) != len(features):
            return "Invalid input length. Ensure values match selected features."

        # Use the trained model to make predictions
        prediction = best_model.predict([input_values])[0]

        return f"Predicted Target Value: {prediction:.2f}"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction failed. Please check your input format."


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
