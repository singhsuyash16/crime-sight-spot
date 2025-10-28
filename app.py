from flask import Flask, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "crime_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return '''
    <h2>Crime Prediction System</h2>
    <form method="POST" action="/predict">
      <label>City:</label><br>
      <input type="text" name="city" placeholder="Enter city (e.g., Delhi)" required><br><br>
      <label>Year:</label><br>
      <input type="number" name="year" placeholder="Enter year (e.g., 2025)" required><br><br>
      <button type="submit">Predict Crime Count</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city'].strip().title()
    year = int(request.form['year'])

    # Prepare input DataFrame for the pipeline
    input_data = pd.DataFrame({"city": [city], "year": [year]})

    try:
        prediction = model.predict(input_data)[0]
        return f"<h3>Predicted Crime Count in {city} for {year}: <b>{prediction:.2f}</b></h3>"
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><p>Make sure the city name exists in the training data.</p>"

if __name__ == '__main__':
    app.run(debug=True)
