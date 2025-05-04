from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('earthquake_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('magType_encoder.pkl', 'rb') as f:
    magType_encoder = pickle.load(f)


@app.route('/')
def home():
    return render_template('home.html')  # Input form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        magType_str = request.form['magType']
        nst = float(request.form['nst'])
        gap = float(request.form['gap'])
        dmin = float(request.form['dmin'])
        rms = float(request.form['rms'])
        year = int(request.form['year'])
        magNst = float(request.form['magNst'])
        horizontalError = float(request.form['horizontalError'])
        magError = float(request.form['magError'])

        # Encode magType
        magType_encoded = magType_encoder.transform([magType_str])[0]

        # Create feature array
        features = np.array([[
            latitude, longitude, depth, magType_encoded, nst, gap,
            dmin, rms, horizontalError, year, magNst, magError
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict magnitude
        prediction = model.predict(features_scaled)[0]

        return render_template('result.html',
    prediction_text=f"Predicted Earthquake Magnitude: {prediction:.2f}",
    latitude=latitude,
    longitude=longitude,
    depth=depth,
    magType=magType_str,
    year=year,
    nst=nst,
    gap=gap,
    dmin=dmin,
    rms=rms,
    magNst=magNst,
    horizontalError=horizontalError,
    magError=magError
)


    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)

