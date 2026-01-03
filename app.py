from flask import Flask, request, jsonify
from models import BaseModel, AdvancedModel

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect_anomaly():
    try:
        data = request.get_json()
        model_type = data.get("model", "base")
        listing_id = data["listing_id"]
        date = data["date"]
        price = data["price"]

        if model_type == "advanced":
            model = AdvancedModel()
        else:
            model = BaseModel()

        result = model.detect_anomaly(listing_id, date, price)
        return jsonify({"anomaly": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
