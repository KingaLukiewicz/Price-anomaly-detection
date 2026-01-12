from flask import Flask, request, jsonify
from ab_test import log_ab_test, random_model

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect_anomaly():
    try:
        data = request.get_json()
        # Validation
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
        result = model.detect_anomaly(data)
        log_ab_test(
            input_data=data,
            model_used=model_type,
            prediction=result
        )

        return jsonify({"anomaly": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
