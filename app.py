from flask import Flask, request, render_template, jsonify
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load('model.joblib')


# Preprocessing function
def preprocess(data):
    current_year = 2024
    data['Years_Since_Manufacture'] = current_year - data['Year']
    data.drop(columns=['Year'], inplace=True)
    data['Log_Kms_Driven'] = np.log1p(data['Kms_Driven'])

    data_encoded = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

    # Ensure all necessary columns are present
    required_columns = model.feature_names_in_
    for col in required_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    data_encoded = data_encoded[required_columns]
    return data_encoded


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form.to_dict()

    # Convert data types
    data['Year'] = int(data['Year'])
    data['Present_Price'] = float(data['Present_Price'])
    data['Kms_Driven'] = int(data['Kms_Driven'])
    data['Owner'] = int(data['Owner'])

    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Preprocess the data
    df_processed = preprocess(df)

    # Make a prediction
    prediction = model.predict(df_processed)

    # Return the prediction
    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
