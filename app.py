from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and encoder
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "crime_model.pkl")
encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "label_encoder.pkl")
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city'].strip().title()
    year = int(request.form['year'])

    try:
        # Encode the city
        city_encoded = encoder.transform([city])[0]
        # Prepare input DataFrame with encoded city
        input_data = pd.DataFrame({"city_encoded": [city_encoded], "year": [year]})
        prediction = model.predict(input_data)[0]
        risk_level = {0: 'Low', 1: 'Medium', 2: 'High'}[prediction]
        return render_template('index.html', prediction_text=f"Predicted Crime Risk Level in {city} for {year}: {risk_level}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}. Make sure the city name exists in the training data.")

if __name__ == '__main__':
    app.run(debug=True)
