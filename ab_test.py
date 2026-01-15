import json
from datetime import datetime
import random
import joblib
import pandas as pd
import numpy as np

BASE_MODEL_PATH = "models/base_model.pkl"
ADV_MODEL_PATH = "models/advanced_model.pkl"

base_model_loaded = joblib.load(BASE_MODEL_PATH)
adv_model_loaded = joblib.load(ADV_MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")
encoded_columns = joblib.load("models/encoded_columns.pkl")


def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    print(df.dtypes)

    cat_cols = ["neighbourhood_cleansed", "property_type", "room_type"]
    num_cols = ["accommodates", "bathrooms", "bedrooms", "beds"]

    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    missing_cols = [col for col in encoded_columns if col not in df_cat.columns]
    if missing_cols:
        zeros_df = pd.DataFrame(0, index=df_cat.index, columns=missing_cols)
        df_cat = pd.concat([df_cat, zeros_df], axis=1)

    extra_cols = [col for col in df_cat.columns if col not in encoded_columns]
    if extra_cols:
        df_cat = df_cat.drop(columns=extra_cols)

    df_cat = df_cat[encoded_columns]

    df_num = df[num_cols].copy()
    df_num = pd.DataFrame(scaler.transform(df_num), columns=num_cols, index=df.index)
    df_num["log_price"] = np.log1p(df["price"])
    df_num["price"] = df["price"]

    df_processed = pd.concat([df_cat, df_num], axis=1)

    return df_processed


def random_model():
    models = {
        "base": base_model_loaded,
        "advanced": adv_model_loaded
    }
    model_type = random.choice(["base", "advanced"])
    return model_type, models[model_type]


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
