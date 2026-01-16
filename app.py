from flask import Flask, request, jsonify
from ab_test import log_ab_test, random_model, preprocess_input
import time
import pandas as pd

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect_anomaly():
    try:
        data = request.get_json()
        required_fields = [
            "neighbourhood_cleansed", "property_type", "room_type",
            "accommodates", "bathrooms", "bedrooms", "beds", "price"
        ]
        for field in required_fields:
            if field not in data:
                return (
                    jsonify({"error": f"Missing required field: {field}"}),
                    400
                )
        model_type, model = random_model()

        df_processed = preprocess_input(data)
        df_processed = df_processed[model.feature_columns]
        price_series = pd.Series([float(data["price"])],
                                 index=df_processed.index)

        start_time = time.perf_counter()
        result = model.predict(df_processed, price_series)[0]
        prediction = int(result)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        log_ab_test(
            input_data=data,
            model_used=model_type,
            prediction=prediction,
            latency_ms=latency_ms
        )

        return jsonify({"anomaly": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
