import joblib
import numpy as np
from flask import Flask, request, jsonify

bundle = joblib.load("model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["features"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)

    # normalisation
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
