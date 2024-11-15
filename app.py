import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("XGBoostModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Input Validation
    if not (location and bhk and bath and sqft):
        return "Error: All fields are required!"
    
    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
    except ValueError:
        return "Error: Please enter valid numeric values for BHK, Bathroom, and Square Feet."

    # Prepare input for prediction
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    try:
        prediction = pipe.predict(input)[0] * 1e5  # Prediction in rupees
        formatted_prediction = f"{prediction:,.2f}"  # Format with commas
        return formatted_prediction
    except Exception as e:
        return f"Error: Unable to make a prediction. {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
