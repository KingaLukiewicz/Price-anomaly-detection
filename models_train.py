import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import hdbscan
from typing import List, Optional, Tuple, Dict
import warnings
import sys

warnings.filterwarnings("ignore")


class AnomalyDetectionError(Exception):
    """Custom exception for anomaly detection errors"""

    pass


def load_data(
    filepath: str, selected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load data from CSV file with optional column selection

    Args:
        filepath: Path to CSV file
        selected_columns: List of columns to load. If None, loads all columns

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If file doesn't exist
        AnomalyDetectionError: If selected columns don't exist
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except pd.errors.ParserError as e:
        raise AnomalyDetectionError(f"Error parsing CSV: {e}") from e

    if selected_columns:
        missing_cols = [col for col in selected_columns if col not in df.columns]
        if missing_cols:
            raise AnomalyDetectionError(f"Columns not found in data: {missing_cols}")
        df = df[selected_columns]

    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for training - one-hot encode categorical columns

    Args:
        df: Input DataFrame (numeric columns already standardized)

    Returns:
        DataFrame ready for model training
    """
    df_prep = df.copy()

    # One-hot encode categorical columns
    categorical_cols = df_prep.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print(f"One-hot encoding categorical columns: {categorical_cols}")
        df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)

    print(f"Feature count: {len(df_prep.columns)}")
    return df_prep


def perform_clustering(
    X: np.ndarray, n_clusters: int = 5, method: str = "kmeans"
) -> Tuple[np.ndarray, float]:
    """Perform clustering on the data

    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'hdbscan')

    Returns:
        Tuple of (cluster labels, silhouette score)

    Raises:
        AnomalyDetectionError: If clustering fails
    """
    try:
        if method == "kmeans":
            print(f"Performing K-Means clustering with {n_clusters} clusters...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
        elif method == "hdbscan":
            print("Performing HDBSCAN clustering...")
            # More lenient parameters to find clusters in high-dimensional data
            min_cluster_size = max(10, len(X) // 80)
            min_samples = max(3, len(X) // 200)
            print(f"  min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
                metric="euclidean",
                cluster_selection_method="leaf",  # Changed from 'eom' to 'leaf' for more clusters
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"HDBSCAN found {n_clusters} clusters")
        else:
            raise AnomalyDetectionError(f"Unknown clustering method: {method}")

        # Calculate silhouette score (only for non-noise points)
        valid_mask = labels != -1
        if valid_mask.sum() > 1:
            silhouette = silhouette_score(X[valid_mask], labels[valid_mask])
            print(f"Silhouette score: {silhouette:.4f}")
        else:
            silhouette = 0.0
            print("Warning: Too few valid clusters for silhouette score")

        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"  {cluster_name}: {count} samples ({count/len(labels)*100:.1f}%)")

        return labels, silhouette

    except (ValueError, RuntimeError) as e:
        raise AnomalyDetectionError(f"Clustering failed: {e}") from e


def train_base_model(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[LocalOutlierFactor, Dict[str, float]]:
    """Train base model using LocalOutlierFactor with hyperparameter tuning

    Args:
        X_train: Training features
        X_val: Validation features for hyperparameter selection

    Returns:
        Tuple of (best trained LOF model, best params dict)
    """
    print("\n" + "=" * 60)
    print("TRAINING BASE MODEL (LocalOutlierFactor)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("\nTuning hyperparameters...")

    # Grid search parameters
    contamination_values = [0.05, 0.1, 0.15, 0.2]
    n_neighbors_values = [10, 20, 30, 50]

    best_model = None
    best_score = float("inf")
    best_params = {}

    for contamination in contamination_values:
        for n_neighbors in n_neighbors_values:
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination, novelty=True
            )
            model.fit(X_train)

            # Score on validation set (lower is better for anomaly detection)
            val_scores = -model.score_samples(X_val)
            avg_score = np.mean(val_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_model = model
                best_params = {
                    "contamination": contamination,
                    "n_neighbors": n_neighbors,
                    "validation_score": avg_score,
                }

    print(f"\nBest hyperparameters:")
    print(f"  Contamination: {best_params['contamination']}")
    print(f"  N neighbors: {best_params['n_neighbors']}")
    print(f"  Validation score: {best_params['validation_score']:.4f}")

    # Get predictions on training data
    train_pred = best_model.predict(X_train)
    n_outliers = (train_pred == -1).sum()
    print(
        f"Detected outliers in training data: {n_outliers} ({n_outliers/len(X_train)*100:.2f}%)"
    )

    return best_model, best_params


def train_advanced_model(
    X_train: np.ndarray, X_val: np.ndarray, cluster_labels: np.ndarray, clusterer
) -> Dict[str, any]:
    """Train advanced model using separate IsolationForest for each HDBSCAN cluster

    Args:
        X_train: Training features
        X_val: Validation features for hyperparameter selection
        cluster_labels: Cluster assignments from HDBSCAN
        clusterer: Fitted HDBSCAN clusterer object

    Returns:
        Dictionary containing IsolationForest models per cluster and cluster info
    """
    print("\n" + "=" * 60)
    print("TRAINING ADVANCED MODEL (IsolationForest per cluster + HDBSCAN)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Get unique clusters (excluding noise cluster -1)
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    n_clusters = len(unique_clusters)
    print(f"Number of clusters: {n_clusters}")

    # Grid search parameters
    contamination_values = [0.05, 0.1, 0.15, 0.2]
    n_estimators_values = [50, 100, 200]

    # Train separate Isolation Forest for each cluster
    cluster_models = {}
    cluster_params = {}
    total_outliers = 0

    print("\nTraining Isolation Forest for each cluster...")
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        X_cluster = X_train[cluster_mask]
        cluster_size = len(X_cluster)

        print(f"\n--- Cluster {cluster} ({cluster_size} samples) ---")

        if cluster_size < 10:
            print(f"  Warning: Too few samples, skipping this cluster")
            continue

        # Hyperparameter tuning for this cluster
        best_model = None
        best_score = float("inf")
        best_params = {}

        for contamination in contamination_values:
            for n_estimators in n_estimators_values:
                # Adjust contamination if cluster is very small
                adjusted_contamination = min(contamination, 0.5)

                try:
                    if_model = IsolationForest(
                        n_estimators=n_estimators,
                        contamination=adjusted_contamination,
                        random_state=42,
                        max_samples=min(256, cluster_size),
                    )
                    if_model.fit(X_cluster)

                    # Score on cluster samples
                    cluster_scores = -if_model.score_samples(X_cluster)
                    avg_score = np.mean(cluster_scores)

                    if avg_score < best_score:
                        best_score = avg_score
                        best_model = if_model
                        best_params = {
                            "contamination": adjusted_contamination,
                            "n_estimators": n_estimators,
                            "validation_score": avg_score,
                        }
                except Exception as e:
                    # Skip this parameter combination if it fails
                    continue

        if best_model is None:
            print(f"  Warning: Failed to train model for cluster {cluster}")
            continue

        cluster_models[cluster] = best_model
        cluster_params[cluster] = best_params

        # Get predictions
        train_pred = best_model.predict(X_cluster)
        n_outliers = (train_pred == -1).sum()
        total_outliers += n_outliers

        print(
            f"  Best params: contamination={best_params['contamination']}, n_estimators={best_params['n_estimators']}"
        )
        print(
            f"  Outliers detected: {n_outliers}/{cluster_size} ({n_outliers/cluster_size*100:.1f}%)"
        )

    # Handle noise cluster (-1) with a separate global model
    noise_mask = cluster_labels == -1
    if noise_mask.sum() > 10:
        print(f"\n--- Noise cluster ({noise_mask.sum()} samples) ---")
        X_noise = X_train[noise_mask]
        # Use lower contamination for noise cluster to avoid too many false positives
        noise_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Lower contamination to reduce false positives
            random_state=42,
            max_samples="auto",
        )
        noise_model.fit(X_noise)
        cluster_models[-1] = noise_model
        cluster_params[-1] = {"contamination": 0.05, "n_estimators": 100}
        print(f"  Trained noise cluster model with contamination=0.05")

    print(f"\nTotal models trained: {len(cluster_models)}")
    print(
        f"Total outliers across all clusters: {total_outliers}/{len(X_train)} ({total_outliers/len(X_train)*100:.1f}%)"
    )

    model_dict = {
        "cluster_models": cluster_models,  # Dict: cluster_id -> IsolationForest
        "clusterer": clusterer,  # HDBSCAN object for predicting cluster of new data
        "cluster_labels": cluster_labels,
        "n_clusters": n_clusters,
        "cluster_params": cluster_params,
    }

    return model_dict


def evaluate_model(
    model, X_test: np.ndarray, model_name: str = "Model", is_advanced: bool = False
) -> Dict[str, float]:
    """Evaluate anomaly detection model

    Args:
        model: Trained model (LOF or dict with IF)
        X_test: Test features
        model_name: Name for printing
        is_advanced: Whether this is the advanced model

    Returns:
        Dictionary with metrics
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    if is_advanced:
        # Predict cluster for each test sample using HDBSCAN
        test_cluster_labels, _ = hdbscan.approximate_predict(model["clusterer"], X_test)
        cluster_models = model["cluster_models"]

        # Make predictions using cluster-specific models
        predictions = np.zeros(len(X_test))
        scores = np.zeros(len(X_test))

        for cluster_id, if_model in cluster_models.items():
            cluster_mask = test_cluster_labels == cluster_id
            if cluster_mask.sum() == 0:
                continue

            X_cluster = X_test[cluster_mask]
            cluster_predictions = if_model.predict(X_cluster)
            cluster_scores = if_model.score_samples(X_cluster)

            predictions[cluster_mask] = cluster_predictions
            scores[cluster_mask] = cluster_scores

        # Handle samples assigned to clusters without models
        unassigned_mask = predictions == 0
        if unassigned_mask.sum() > 0:
            # Use noise model if available, otherwise mark as normal
            if -1 in cluster_models:
                predictions[unassigned_mask] = cluster_models[-1].predict(
                    X_test[unassigned_mask]
                )
                scores[unassigned_mask] = cluster_models[-1].score_samples(
                    X_test[unassigned_mask]
                )
            else:
                predictions[unassigned_mask] = 1  # Mark as normal
                scores[unassigned_mask] = 0.0

        predictions = predictions.astype(int)
    else:
        predictions = model.predict(X_test)
        scores = model.score_samples(X_test)

    n_outliers = (predictions == -1).sum()
    outlier_rate = n_outliers / len(X_test) * 100

    print(f"Test samples: {len(X_test)}")
    print(f"Detected outliers: {n_outliers} ({outlier_rate:.2f}%)")

    # Calculate anomaly scores
    print("Anomaly score statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")

    # For advanced model, show distribution by cluster
    if is_advanced:
        print("\nOutlier distribution by cluster:")
        unique_test_clusters = sorted(set(test_cluster_labels))
        for cluster in unique_test_clusters:
            cluster_mask = test_cluster_labels == cluster
            cluster_outliers = ((predictions == -1) & cluster_mask).sum()
            cluster_size = cluster_mask.sum()
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            if cluster_size > 0:
                print(
                    f"  {cluster_name}: {cluster_outliers}/{cluster_size} "
                    + f"({cluster_outliers/cluster_size*100:.1f}% outliers)"
                )

    metrics = {
        "outlier_count": n_outliers,
        "outlier_rate": outlier_rate,
        "score_mean": scores.mean(),
        "score_std": scores.std(),
        "score_min": scores.min(),
        "score_max": scores.max(),
    }

    return metrics


def save_models(
    base_model,
    advanced_model,
    scaler: Optional[StandardScaler] = None,
    base_path: str = "models/base_model.pkl",
    advanced_path: str = "models/advanced_model.pkl",
    scaler_path: str = "models/scaler.pkl",
) -> None:
    """Save trained models and scaler

    Args:
        base_model: Trained base model
        advanced_model: Trained advanced model
        scaler: Fitted StandardScaler (optional, already saved by prepare_data.py)
        base_path: Path to save base model
        advanced_path: Path to save advanced model
        scaler_path: Path to save scaler
    """
    try:
        # Create models directory if it doesn't exist
        import os

        os.makedirs("models", exist_ok=True)

        joblib.dump(base_model, base_path)
        joblib.dump(advanced_model, advanced_path)

        if scaler is not None:
            joblib.dump(scaler, scaler_path)

        print("\n" + "=" * 60)
        print("MODELS SAVED")
        print("=" * 60)
        print(f"Base model: {base_path}")
        print(f"Advanced model: {advanced_path}")
        if scaler is not None:
            print(f"Scaler: {scaler_path}")
        else:
            print(f"Scaler: (already saved by prepare_data.py)")

    except (IOError, OSError) as e:
        raise IOError(f"Error saving models: {e}") from e


def train_anomaly_detection_models(
    train_path: str = "data/data.csv",
    test_path: str = "data/test_data.csv",
) -> None:
    """Main function to train anomaly detection models

    Args:
        train_path: Path to training data (already preprocessed and standardized)
        test_path: Path to test data (already preprocessed and standardized)
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")

    # Load data (already split and standardized by prepare_data.py)
    print("\nLoading training data...")
    train_df = load_data(train_path)

    print("\nLoading test data...")
    test_df = load_data(test_path)

    # Prepare features (data already preprocessed)
    train_features = prepare_features(train_df)
    test_features = prepare_features(test_df)

    # Ensure same columns in all sets
    common_cols = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

    # Convert to numpy arrays (already standardized)
    X_train = train_features.values
    X_test = test_features.values

    print(f"\nData loaded:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {len(common_cols)} columns")

    # Clustering for data exploration and advanced model
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)

    # Try different cluster numbers for K-means and select best
    best_silhouette = -1
    best_n_clusters = 5
    for n_clusters in [3, 5, 7, 10]:
        _, silhouette = perform_clustering(
            X_train, n_clusters=n_clusters, method="kmeans"
        )
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_n_clusters = n_clusters

    print(
        f"\nBest K-means configuration: {best_n_clusters} clusters (silhouette: {best_silhouette:.4f})"
    )

    # Use HDBSCAN for advanced model (automatic cluster detection)
    # We need to keep the clusterer object to predict clusters for new data
    print("\nFitting HDBSCAN clusterer for advanced model...")
    min_cluster_size = max(10, len(X_train) // 80)
    min_samples = max(3, len(X_train) // 200)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        prediction_data=True,  # Enable prediction for new data
    )
    cluster_labels_hdbscan = clusterer.fit_predict(X_train)
    n_clusters_found = len(set(cluster_labels_hdbscan)) - (
        1 if -1 in cluster_labels_hdbscan else 0
    )
    print(f"HDBSCAN found {n_clusters_found} clusters")

    # Show cluster distribution
    unique, counts = np.unique(cluster_labels_hdbscan, return_counts=True)
    for label, count in zip(unique, counts):
        cluster_name = "Noise" if label == -1 else f"Cluster {label}"
        print(
            f"  {cluster_name}: {count} samples ({count/len(cluster_labels_hdbscan)*100:.1f}%)"
        )

    # Train base model (LOF) with hyperparameter tuning
    base_model, base_params = train_base_model(X_train, X_train)

    # Train advanced model (separate IsolationForest per cluster + HDBSCAN)
    advanced_model = train_advanced_model(
        X_train, X_train, cluster_labels_hdbscan, clusterer
    )

    # Evaluate models on test set
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    base_metrics = evaluate_model(base_model, X_test, "BASE MODEL (LOF)")
    advanced_metrics = evaluate_model(
        advanced_model, X_test, "ADVANCED MODEL (IF+HDBSCAN)", is_advanced=True
    )

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Base (LOF)':<20} {'Advanced (IF+HDBSCAN)':<20}")
    print("-" * 70)
    print(
        f"{'Outliers detected':<30} {base_metrics['outlier_count']:<20} {advanced_metrics['outlier_count']:<20}"
    )
    print(
        f"{'Outlier rate (%)':<30} {base_metrics['outlier_rate']:<20.2f} {advanced_metrics['outlier_rate']:<20.2f}"
    )
    print(
        f"{'Mean anomaly score':<30} {base_metrics['score_mean']:<20.4f} {advanced_metrics['score_mean']:<20.4f}"
    )
    print(
        f"{'Std anomaly score':<30} {base_metrics['score_std']:<20.4f} {advanced_metrics['score_std']:<20.4f}"
    )
    print("-" * 70)

    # Recommendation
    print("\nRecommendation:")
    if abs(base_metrics["outlier_rate"] - 10.0) < abs(
        advanced_metrics["outlier_rate"] - 10.0
    ):
        print("  → Base model (LOF) has more reasonable outlier detection rate")
    else:
        print(
            "  → Advanced model (IF+HDBSCAN) has more reasonable outlier detection rate"
        )

    if base_metrics["score_std"] < advanced_metrics["score_std"]:
        print("  → Base model shows more stable anomaly scores")
    else:
        print("  → Advanced model shows more stable anomaly scores")

    # Save models (without scaler - it's saved by prepare_data.py)
    save_models(base_model, advanced_model, None)

    # Save column names for feature alignment during inference
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump(list(common_cols), "models/feature_columns.pkl")
    print(f"Feature columns saved to: models/feature_columns.pkl")

    # Save hyperparameters and metrics
    import json
    import os

    os.makedirs("models", exist_ok=True)

    results = {
        "base_model": {"hyperparameters": base_params, "test_metrics": base_metrics},
        "advanced_model": {
            "hyperparameters": {
                str(k): v for k, v in advanced_model["cluster_params"].items()
            },
            "test_metrics": advanced_metrics,
            "n_clusters": advanced_model["n_clusters"],
        },
    }

    with open("models/training_results.json", "w") as f:
        # Convert numpy types to python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): convert_numpy(value) for key, value in obj.items()}
            return obj

        json.dump(convert_numpy(results), f, indent=2)

    print("Training results saved to: models/training_results.json")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--train", default="data/data.csv", help="Training data path")
    parser.add_argument("--test", default="data/test_data.csv", help="Test data path")

    args = parser.parse_args()

    try:
        train_anomaly_detection_models(train_path=args.train, test_path=args.test)
    except (FileNotFoundError, AnomalyDetectionError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user", file=sys.stderr)
        sys.exit(130)
