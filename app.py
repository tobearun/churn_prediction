from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Get expected feature count
        expected_features = scaler.n_features_in_

        # Check if input matches expected feature count
        if features.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"}), 400

        # Apply scaling
        features = scaler.transform(features)
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
