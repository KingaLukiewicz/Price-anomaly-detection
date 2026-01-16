import pandas as pd
import json
import numpy as np
from scipy import stats as scipy_stats


def load_logs(log_file="ab_test_logs.json"):
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def compute_basic_statistics(df):
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
            p95_latency=pd.NamedAgg(column="latency_ms", aggfunc=lambda x: np.percentile(x, 95)),
            p99_latency=pd.NamedAgg(column="latency_ms", aggfunc=lambda x: np.percentile(x, 99)),
        )
        .reset_index()
    )

    basic_stats["anomaly_rate_pct"] = (
        basic_stats["anomaly_detected"] / basic_stats["total_calls"]
    ) * 100

    return basic_stats


def compute_price_range_statistics(df):
    df = df.copy()

    df["price_range"] = pd.cut(
        df["input_data"].apply(lambda x: x["price"]),
        bins=[0, 200, 500, 1000, float("inf")],
        labels=["0-200", "200-500", "500-1000", "1000+"],
    )

    price_stats = (
        df.groupby(["model_used", "price_range"], observed=True)
        .agg(
            count=pd.NamedAgg(column="prediction", aggfunc="count"),
            anomalies=pd.NamedAgg(column="prediction", aggfunc="sum"),
        )
        .reset_index()
    )

    price_stats["anomaly_rate"] = (
        price_stats["anomalies"] / price_stats["count"]
    ) * 100

    return price_stats


def compute_temporal_statistics(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    temporal_stats = (
        df.groupby(["model_used", "hour"])
        .agg(count=pd.NamedAgg(column="prediction", aggfunc="count"))
        .reset_index()
    )

    return temporal_stats


def run_latency_t_test(df):
    base_latency = df[df["model_used"] == "base"]["latency_ms"]
    advanced_latency = df[df["model_used"] == "advanced"]["latency_ms"]

    if len(base_latency) > 0 and len(advanced_latency) > 0:
        t_stat, p_value = scipy_stats.ttest_ind(base_latency, advanced_latency)
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    return None


def run_anomaly_chi_square_test(df):
    contingency_table = pd.crosstab(df["model_used"], df["prediction"])

    if contingency_table.shape == (2, 2):
        chi2, p_val, dof, expected = scipy_stats.chi2_contingency(contingency_table)
        return {
            "chi2_statistic": chi2,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    return None


def compute_prediction_distribution(df):
    return (
        df.groupby(["model_used", "prediction"])
        .size()
        .reset_index(name="count")
    )


def compute_consistency_statistics(df):
    df = df.copy()
    df["input_hash"] = df["input_data"].apply(lambda x: hash(frozenset(x.items())))

    stability_stats = (
        df.groupby(["model_used", "input_hash"])
        .agg(
            unique_predictions=pd.NamedAgg(column="prediction", aggfunc="nunique"),
            count=pd.NamedAgg(column="prediction", aggfunc="count"),
        )
        .reset_index()
    )

    inconsistent = stability_stats[stability_stats["unique_predictions"] > 1]

    return (
        inconsistent.groupby("model_used")
        .size()
        .reset_index(name="inconsistent_inputs")
    )


def evaluate_ab_test(log_file="ab_test_logs.json"):
    df = load_logs(log_file)

    return {
        "basic_statistics": compute_basic_statistics(df),
        "price_range_analysis": compute_price_range_statistics(df),
        "temporal_distribution": compute_temporal_statistics(df),
        "latency_comparison_test": run_latency_t_test(df),
        "anomaly_detection_test": run_anomaly_chi_square_test(df),
        "prediction_distribution": compute_prediction_distribution(df),
        "consistency_analysis": compute_consistency_statistics(df),
    }


def print_results(results):
    separator = "-" * 80

    print(separator)
    print("A/B TEST EVALUATION REPORT")
    print(separator)

    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print(separator)
    print(results["basic_statistics"].to_string(index=False))

    # Price range analysis
    print("\n2. PRICE RANGE ANALYSIS")
    print(separator)
    print(results["price_range_analysis"].to_string(index=False))

    # Prediction distribution
    print("\n3. PREDICTION DISTRIBUTION")
    print(separator)
    print(results["prediction_distribution"].to_string(index=False))

    # Latency statistical test
    if results["latency_comparison_test"]:
        test = results["latency_comparison_test"]
        print("\n4. STATISTICAL TEST: LATENCY COMPARISON (t-test)")
        print(separator)
        print(f"t-statistic: {test['t_statistic']:.4f}")
        print(f"p-value:     {test['p_value']:.4f}")
        print(
            "Result:      "
            + (
                "Statistically significant difference detected"
                if test["significant"]
                else "No statistically significant difference detected"
            )
        )

    # Anomaly detection statistical test
    if results["anomaly_detection_test"]:
        test = results["anomaly_detection_test"]
        print("\n5. STATISTICAL TEST: ANOMALY DETECTION (Chi-square test)")
        print(separator)
        print(f"Chi-square statistic: {test['chi2_statistic']:.4f}")
        print(f"p-value:              {test['p_value']:.4f}")
        print(
            "Result:               "
            + (
                "Statistically significant difference detected"
                if test["significant"]
                else "No statistically significant difference detected"
            )
        )

    # Consistency analysis
    if not results["consistency_analysis"].empty:
        print("\n6. CONSISTENCY ANALYSIS")
        print(separator)
        print(
            "Number of input cases for which the same model produced different predictions:"
        )
        print(results["consistency_analysis"].to_string(index=False))


if __name__ == "__main__":
    results = evaluate_ab_test()
    print_results(results)
