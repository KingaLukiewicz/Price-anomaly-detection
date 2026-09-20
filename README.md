# Price Anomaly Detection

Machine learning system for detecting unusual accommodation prices based on similar listings and market context.

## Overview

The project identifies potential price anomalies in accommodation listings using **unsupervised learning**.

Two approaches are implemented:

* **K-Means** — simple and interpretable baseline
* **HDBSCAN** — density-based clustering that does not require a predefined number of clusters

Listings are grouped based on their characteristics, and prices are compared with the typical price of similar properties.

The project covers the complete ML workflow - from data preparation and model training to REST API deployment and online A/B testing.

## ML Pipeline

```text
Data → Preprocessing → Clustering → Price Anomaly Detection → Evaluation → REST API
```

Key preprocessing steps include missing-value handling, categorical encoding, price transformation, feature scaling, and train/validation/test splitting.

## Tech Stack

**Python · pandas · NumPy · scikit-learn · HDBSCAN · SciPy · Flask**

## Microservice

The trained model is exposed through a Flask API.

### Start the service

```bash
python app.py
```

The API runs on:

```text
http://localhost:8080
```

### Detect a price anomaly

Send a `POST` request to `/detect`:

```bash
curl -X POST http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{
    "neighbourhood_cleansed": "Bronowice",
    "property_type": "Apartment",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bathrooms": 1,
    "bedrooms": 2,
    "beds": 2,
    "price": 250
  }'
```

Response:

```json
{
  "anomaly": 0
}
```

* `0` — normal price
* `1` — potential price anomaly

## A/B Testing

The API can randomly assign requests to the **K-Means** or **HDBSCAN** model in a 50/50 A/B test.

Each request is logged with:

* selected model
* prediction
* input data
* inference latency

The collected data can then be analyzed using `evaluate_ab.py`, including latency statistics, anomaly rates and statistical tests.

## Authors
Kinga Łukiewicz · Aleksandra Raczyńska

Developed as a university project for the Machine Learning Engineering course at Warsaw University of Technology.
