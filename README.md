# 🔮 Churn Prediction API

A lightweight Flask API that uses a **Logistic Regression** model to predict customer churn. Designed for fast integration into products or dashboards, this API helps businesses proactively identify customers at risk of leaving.

---

## 🚀 Overview

This project exposes a trained churn prediction model through a Flask API. By sending customer data as input, the API returns a prediction indicating whether the customer is likely to churn.

It uses a pre-trained **Logistic Regression** model and a **scaler** to preprocess inputs and return consistent results.

---

## 🧠 How It Works

- `scaler.plk`: Scales incoming feature values to match the training distribution.
- `churn_model.plk`: A trained **Logistic Regression** model used for churn inference.
- `app.py`: Hosts the Flask API. Accepts input via POST requests, processes it, and returns predictions.

---

## 🧪 Example Use Case

Send a POST request with customer data to get the churn prediction:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "feature1": value,
  "feature2": value,
  ...
}'
```
## The API returns:

```bash
{
  "prediction": "Churn" // or "No Churn"
}
