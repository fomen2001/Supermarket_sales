from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

app = Flask(__name__)

# Define the custom mse function and register it
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load model and scaler with custom objects
model = load_model('sales_model.h5', custom_objects={'mse': mse})
scaler = joblib.load('scaler.joblib')

# Load data
data = pd.read_csv('sales.csv')

def encode_dates(df, column):
    df[column] = pd.to_datetime(df[column], dayfirst=True)
    df[f'{column}_year'] = df[column].dt.year
    df[f'{column}_month'] = df[column].dt.month
    df[f'{column}_day'] = df[column].dt.day
    df = df.drop(column, axis=1)
    return df

def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df = df.copy()
    df = df.drop(['Row ID', 'Customer Name', 'Country', 'Product Name'], axis=1)
    df = df.drop(['Order ID', 'Customer ID'], axis=1)
    df = encode_dates(df, column='Order Date')
    df = encode_dates(df, column='Ship Date')
    for column in ['Ship Mode', 'Segment', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category']:
        df = onehot_encode(df, column=column)
    return df

def generate_strategy(predicted_sales):
    if predicted_sales < 1000:
        return "The predicted sales are quite low. Consider running promotions or discounts to boost sales."
    elif 1000 <= predicted_sales < 5000:
        return "Sales are moderate. Look into optimizing inventory and improving customer engagement strategies."
    elif 5000 <= predicted_sales < 10000:
        return "Sales are good. Consider expanding product lines and enhancing marketing efforts."
    else:
        return "Sales are excellent! Maintain current strategies and explore opportunities for expansion."
    
@app.route('/')
def index():
    csv_file_path = os.path.abspath('sales.csv')
    sales = pd.read_csv(csv_file_path)
    data = sales.head(200).to_dict(orient='records')
    columns = sales.columns.tolist()

    # Example description dictionary
    description = {column: "Description of " + column for column in columns}

    return render_template('index.html', data=data, column_names=columns, description=description)
    # return render_template('index.html', data=data.to_html(), description=data.describe().to_html())

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_data = {k: v for k, v in input_data.items() if v}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    for column in ['Ship Mode', 'Segment', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category']:
        if column in input_data:
            input_df = onehot_encode(input_df, column=column)
    
    # Ensure all necessary columns are present
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Scale the input data
    input_scaled = scaler.transform(input_df[X_train.columns])

    # Make prediction
    prediction = model.predict(input_scaled)
    
    predicted_sales = float(prediction[0][0])
    strategy = generate_strategy(predicted_sales)
    
    return render_template('result.html', predicted_sales=predicted_sales, strategy=strategy)

    
    #return jsonify({'predicted_sales': float(prediction[0][0])})

if __name__ == '__main__':
    # Preprocess the dataset to fit the model's requirements
    data_processed = preprocess_inputs(data)
    X_train = data_processed.drop('Sales', axis=1)
    
    app.run(debug=True)
