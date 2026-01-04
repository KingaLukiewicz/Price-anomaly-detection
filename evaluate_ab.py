import pandas as pd
import json


def evaluate_ab_test():
    log_file = "ab_test_logs.json"
    logs = []

    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    df = pd.DataFrame(logs)

    stats = df.groupby("model_used").agg(
        total_calls=pd.NamedAgg(column="prediction", aggfunc="count"),
        anomaly_detected=pd.NamedAgg(column="prediction", aggfunc="sum"),
        avg_latency=pd.NamedAgg(column="latency_ms", aggfunc="mean")
    ).reset_index()

    stats["anomaly_rate_pct"] = (
        stats["anomaly_detected"] / stats["total_calls"]) * 100

    return stats


if __name__ == "__main__":
    results = evaluate_ab_test()
    print(results)
