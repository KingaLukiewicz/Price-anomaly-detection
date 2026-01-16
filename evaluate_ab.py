import pandas as pd
import json
import numpy as np
from scipy import stats as scipy_stats


def evaluate_ab_test():
    log_file = "ab_test_logs.json"
    logs = []

    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic stats
    # Calculate basic metrics for each model
    basic_stats = (
        df.groupby("model_used")
        .agg(
            total_calls=pd.NamedAgg(column="prediction", aggfunc="count"),
            anomaly_detected=pd.NamedAgg(column="prediction", aggfunc="sum"),
            avg_latency=pd.NamedAgg(column="latency_ms", aggfunc="mean"),
            median_latency=pd.NamedAgg(column="latency_ms", aggfunc="median"),
            std_latency=pd.NamedAgg(column="latency_ms", aggfunc="std"),
            min_latency=pd.NamedAgg(column="latency_ms", aggfunc="min"),
            max_latency=pd.NamedAgg(column="latency_ms", aggfunc="max"),
            # P95 (95th percentile) - 95% of requests have latency less than or equal to this value
            # Used for defining SLA and monitoring performance
            p95_latency=pd.NamedAgg(
                column="latency_ms", aggfunc=lambda x: np.percentile(x, 95)
            ),
            # P99 (99th percentile) - 99% of requests have latency less than or equal to this value
            # Shows how the model performs in the worst cases
            p99_latency=pd.NamedAgg(
                column="latency_ms", aggfunc=lambda x: np.percentile(x, 99)
            ),
        )
        .reset_index()
    )

    basic_stats["anomaly_rate_pct"] = (
        basic_stats["anomaly_detected"] / basic_stats["total_calls"]
    ) * 100

    # Price range statistics
    df["price_range"] = pd.cut(
        df["input_data"].apply(lambda x: x["price"]),
        bins=[0, 200, 500, 1000, float("inf")],
        labels=["0-200", "200-500", "500-1000", "1000+"],
    )

    price_stats = (
        df.groupby(["model_used", "price_range"])
        .agg(
            count=pd.NamedAgg(column="prediction", aggfunc="count"),
            anomalies=pd.NamedAgg(column="prediction", aggfunc="sum"),
        )
        .reset_index()
    )
    price_stats["anomaly_rate"] = (
        price_stats["anomalies"] / price_stats["count"]
    ) * 100

    # Temporal statistics
    df["hour"] = df["timestamp"].dt.hour
    temporal_stats = (
        df.groupby(["model_used", "hour"])
        .agg(count=pd.NamedAgg(column="prediction", aggfunc="count"))
        .reset_index()
    )

    # Statistical test - comparing latency between models
    base_latency = df[df["model_used"] == "base"]["latency_ms"]
    advanced_latency = df[df["model_used"] == "advanced"]["latency_ms"]

    if len(base_latency) > 0 and len(advanced_latency) > 0:
        t_stat, p_value = scipy_stats.ttest_ind(base_latency, advanced_latency)
        latency_test = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    else:
        latency_test = None

    # Chi-square test for differences in anomaly detection
    contingency_table = pd.crosstab(df["model_used"], df["prediction"])
    if contingency_table.shape == (2, 2):
        chi2, p_val_chi, dof, expected = scipy_stats.chi2_contingency(contingency_table)
        anomaly_test = {
            "chi2_statistic": chi2,
            "p_value": p_val_chi,
            "significant": p_val_chi < 0.05,
        }
    else:
        anomaly_test = None

    # Prediction distribution
    prediction_dist = (
        df.groupby(["model_used", "prediction"]).size().reset_index(name="count")
    )

    # Stability analysis - does the model change predictions for the same data
    df["input_hash"] = df["input_data"].apply(lambda x: hash(frozenset(x.items())))
    stability_stats = (
        df.groupby(["model_used", "input_hash"])
        .agg(
            unique_predictions=pd.NamedAgg(column="prediction", aggfunc="nunique"),
            count=pd.NamedAgg(column="prediction", aggfunc="count"),
        )
        .reset_index()
    )

    # How many inputs had inconsistent predictions
    inconsistent = stability_stats[stability_stats["unique_predictions"] > 1]
    consistency_stats = (
        inconsistent.groupby("model_used")
        .size()
        .reset_index(name="inconsistent_inputs")
    )

    results = {
        "basic_statistics": basic_stats,
        "price_range_analysis": price_stats,
        "temporal_distribution": temporal_stats,
        "latency_comparison_test": latency_test,
        "anomaly_detection_test": anomaly_test,
        "prediction_distribution": prediction_dist,
        "consistency_analysis": consistency_stats,
    }

    return results


def print_results(results):
    """Wyświetla wyniki analizy A/B test w czytelny sposób."""
    print("=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    print(results["basic_statistics"].to_string(index=False))

    print("\n" + "=" * 80)
    print("PRICE RANGE ANALYSIS")
    print("=" * 80)
    print(results["price_range_analysis"].to_string(index=False))

    print("\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)
    print(results["prediction_distribution"].to_string(index=False))

    if results["latency_comparison_test"]:
        print("\n" + "=" * 80)
        print("STATISTICAL TEST - LATENCY COMPARISON")
        print("=" * 80)
        test = results["latency_comparison_test"]
        print(f"t-statistic: {test['t_statistic']:.4f}")
        print(f"p-value: {test['p_value']:.4f}")
        print(
            f"The difference is statistically significant: {'YES' if test['significant'] else 'NO'}"
        )

    if results["anomaly_detection_test"]:
        print("\n" + "=" * 80)
        print("CHI-SQUARE TEST - DIFFERENCES IN ANOMALY DETECTION")
        print("=" * 80)
        test = results["anomaly_detection_test"]
        print(f"Chi2 statistic: {test['chi2_statistic']:.4f}")
        print(f"p-value: {test['p_value']:.4f}")
        print(
            f"The difference is statistically significant: {'YES' if test['significant'] else 'NO'}"
        )

    if not results["consistency_analysis"].empty:
        print("\n" + "=" * 80)
        print(
            "CONSISTENCY ANALYSIS (how many times the model changed its prediction for the same data)"
        )
        print("=" * 80)
        print(results["consistency_analysis"].to_string(index=False))


if __name__ == "__main__":
    results = evaluate_ab_test()
    print_results(results)
