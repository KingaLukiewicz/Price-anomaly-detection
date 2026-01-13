import json
from datetime import datetime
import random
from model_wrapper import BaseModelWrapper, AdvancedModelWrapper


def random_model():
    models = {
        "base": BaseModelWrapper,
        "advanced": AdvancedModelWrapper
    }
    model_type = random.choice(["base", "advanced"])
    return model_type, models[model_type]()


def log_ab_test(
        input_data,
        model_used,
        prediction,
        ground_truth=None,
        latency_ms=None):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_data": input_data,
        "model_used": model_used,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "latency_ms": latency_ms
    }
    with open("ab_test_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
