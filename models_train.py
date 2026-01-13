import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from typing import List, Optional, Tuple, Dict
import warnings
import sys
warnings.filterwarnings('ignore')


class AnomalyDetectionError(Exception):
    """Custom exception for anomaly detection errors"""
    pass


def load_data(filepath: str, selected_columns: Optional[List[str]] = None) -> pd.DataFrame:
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
    """Prepare features for training

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with prepared features
    """
    df_prep = df.copy()

    # One-hot encode categorical columns
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"One-hot encoding categorical columns: {categorical_cols}")
        df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)

    print(f"Final feature count: {len(df_prep.columns)}")
    return df_prep


def perform_clustering(X: np.ndarray, n_clusters: int = 5,
                      method: str = 'kmeans') -> Tuple[np.ndarray, float]:
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
        if method == 'kmeans':
            print(f"Performing K-Means clustering with {n_clusters} clusters...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
        elif method == 'hdbscan':
            print("Performing HDBSCAN clustering...")
            # More lenient parameters to find clusters in high-dimensional data
            min_cluster_size = max(10, len(X) // 80)
            min_samples = max(3, len(X) // 200)
            print(f"  min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
                metric='euclidean',
                cluster_selection_method='leaf',  # Changed from 'eom' to 'leaf' for more clusters
                allow_single_cluster=True
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


def train_base_model(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[LocalOutlierFactor, Dict[str, float]]:
    """Train base model using LocalOutlierFactor with hyperparameter tuning

    Args:
        X_train: Training features
        X_val: Validation features for hyperparameter selection

    Returns:
        Tuple of (best trained LOF model, best params dict)
    """
    print("\n" + "="*60)
    print("TRAINING BASE MODEL (LocalOutlierFactor)")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("\nTuning hyperparameters...")

    # Grid search parameters
    contamination_values = [0.05, 0.1, 0.15, 0.2]
    n_neighbors_values = [10, 20, 30, 50]

    best_model = None
    best_score = float('inf')
    best_params = {}

    for contamination in contamination_values:
        for n_neighbors in n_neighbors_values:
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=True
            )
            model.fit(X_train)

            # Score on validation set (lower is better for anomaly detection)
            val_scores = -model.score_samples(X_val)
            avg_score = np.mean(val_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_model = model
                best_params = {
                    'contamination': contamination,
                    'n_neighbors': n_neighbors,
                    'validation_score': avg_score
                }

    print(f"\nBest hyperparameters:")
    print(f"  Contamination: {best_params['contamination']}")
    print(f"  N neighbors: {best_params['n_neighbors']}")
    print(f"  Validation score: {best_params['validation_score']:.4f}")

    # Get predictions on training data
    train_pred = best_model.predict(X_train)
    n_outliers = (train_pred == -1).sum()
    print(f"Detected outliers in training data: {n_outliers} ({n_outliers/len(X_train)*100:.2f}%)")

    return best_model, best_params


def train_advanced_model(X_train: np.ndarray, X_val: np.ndarray,
                        cluster_labels: np.ndarray) -> Dict[str, any]:
    """Train advanced model using IsolationForest with HDBSCAN clustering

    Args:
        X_train: Training features
        X_val: Validation features for hyperparameter selection
        cluster_labels: Cluster assignments from HDBSCAN

    Returns:
        Dictionary containing IsolationForest model and cluster info
    """
    print("\n" + "="*60)
    print("TRAINING ADVANCED MODEL (IsolationForest + HDBSCAN)")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("\nTuning hyperparameters...")

    # Grid search parameters
    contamination_values = [0.05, 0.1, 0.15, 0.2]
    n_estimators_values = [50, 100, 200]

    best_model = None
    best_score = float('inf')
    best_params = {}

    for contamination in contamination_values:
        for n_estimators in n_estimators_values:
            if_model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                max_samples='auto'
            )
            if_model.fit(X_train)

            # Score on validation set (lower is better)
            val_scores = -if_model.score_samples(X_val)
            avg_score = np.mean(val_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_model = if_model
                best_params = {
                    'contamination': contamination,
                    'n_estimators': n_estimators,
                    'validation_score': avg_score
                }

    print(f"\nBest hyperparameters:")
    print(f"  Contamination: {best_params['contamination']}")
    print(f"  N estimators: {best_params['n_estimators']}")
    print(f"  Validation score: {best_params['validation_score']:.4f}")

    # Get predictions with best model
    train_pred = best_model.predict(X_train)
    n_outliers = (train_pred == -1).sum()
    print(f"Detected outliers in training data: {n_outliers} ({n_outliers/len(X_train)*100:.2f}%)")

    # Analyze outliers by cluster
    print("\nOutlier distribution by cluster:")
    unique_clusters = sorted(set(cluster_labels))
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_outliers = ((train_pred == -1) & cluster_mask).sum()
        cluster_size = cluster_mask.sum()
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"  {cluster_name}: {cluster_outliers}/{cluster_size} " +
              f"({cluster_outliers/cluster_size*100:.1f}% outliers)")

    model_dict = {
        'isolation_forest': best_model,
        'cluster_labels': cluster_labels,
        'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
        'best_params': best_params
    }

    return model_dict


def evaluate_model(
    model, X_test: np.ndarray,
    model_name: str = "Model",
    is_advanced: bool = False
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
    print("\n" + "="*60)
    print(f"EVALUATING {model_name}")
    print("="*60)

    if is_advanced:
        predictions = model['isolation_forest'].predict(X_test)
    else:
        predictions = model.predict(X_test)

    n_outliers = (predictions == -1).sum()
    outlier_rate = n_outliers / len(X_test) * 100

    print(f"Test samples: {len(X_test)}")
    print(f"Detected outliers: {n_outliers} ({outlier_rate:.2f}%)")

    # Calculate anomaly scores
    if is_advanced:
        scores = model['isolation_forest'].score_samples(X_test)
    else:
        scores = model.score_samples(X_test)

    print("Anomaly score statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")

    metrics = {
        'outlier_count': n_outliers,
        'outlier_rate': outlier_rate,
        'score_mean': scores.mean(),
        'score_std': scores.std(),
        'score_min': scores.min(),
        'score_max': scores.max()
    }

    return metrics


def save_models(base_model, advanced_model, scaler: StandardScaler,
                base_path: str = "models/base_model.pkl",
                advanced_path: str = "models/advanced_model.pkl",
                scaler_path: str = "models/scaler.pkl") -> None:
    """Save trained models and scaler

    Args:
        base_model: Trained base model
        advanced_model: Trained advanced model
        scaler: Fitted StandardScaler
        base_path: Path to save base model
        advanced_path: Path to save advanced model
        scaler_path: Path to save scaler
    """
    try:
        # Create models directory if it doesn't exist
        import os
        os.makedirs('models', exist_ok=True)

        joblib.dump(base_model, base_path)
        joblib.dump(advanced_model, advanced_path)
        joblib.dump(scaler, scaler_path)

        print("\n" + "="*60)
        print("MODELS SAVED")
        print("="*60)
        print(f"Base model: {base_path}")
        print(f"Advanced model: {advanced_path}")
        print(f"Scaler: {scaler_path}")

    except (IOError, OSError) as e:
        raise IOError(f"Error saving models: {e}") from e


def train_anomaly_detection_models(
    train_path: str = 'data/data.csv',
    test_path: str = 'data/test_data.csv',
    val_split: float = 0.2
) -> None:
    """Main function to train anomaly detection models

    Args:
        train_path: Path to training data (data.csv - already split)
        test_path: Path to test data (test_data.csv - already split)
        val_split: Proportion of training data to use for validation
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")

    # Load data (already split into train/test by prepare_data.py)
    print("\nLoading training data...")
    train_df = load_data(train_path)

    print("\nLoading test data...")
    test_df = load_data(test_path)

    # Prepare features
    train_features = prepare_features(train_df)
    test_features = prepare_features(test_df)

    # Ensure same columns in train and test
    common_cols = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

    # Convert to numpy arrays
    X_train_full = train_features.values
    X_test = test_features.values

    # Split training data into train + validation for hyperparameter tuning
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(X_train_full, test_size=val_split, random_state=42)
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Standardization
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Clustering for data exploration and advanced model
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)

    # Try different cluster numbers for K-means and select best
    best_silhouette = -1
    best_n_clusters = 5
    for n_clusters in [3, 5, 7, 10]:
        _, silhouette = perform_clustering(
            X_train_scaled, n_clusters=n_clusters, method='kmeans'
        )
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_n_clusters = n_clusters

    print(f"\nBest K-means configuration: {best_n_clusters} clusters (silhouette: {best_silhouette:.4f})")

    # Use HDBSCAN for advanced model (automatic cluster detection)
    cluster_labels_hdbscan, silhouette_hdbscan = perform_clustering(
        X_train_scaled, method='hdbscan'
    )

    # Train base model (LOF) with hyperparameter tuning
    base_model, base_params = train_base_model(X_train_scaled, X_val_scaled)

    # Train advanced model (IsolationForest + HDBSCAN) with hyperparameter tuning
    advanced_model = train_advanced_model(
        X_train_scaled, X_val_scaled, cluster_labels_hdbscan
    )

    # Evaluate models on test set
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)

    base_metrics = evaluate_model(
        base_model,
        X_test_scaled,
        "BASE MODEL (LOF)"
    )
    advanced_metrics = evaluate_model(
        advanced_model,
        X_test_scaled,
        "ADVANCED MODEL (IF+HDBSCAN)",
        is_advanced=True
    )

    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<30} {'Base (LOF)':<20} {'Advanced (IF+HDBSCAN)':<20}")
    print("-"*70)
    print(f"{'Outliers detected':<30} {base_metrics['outlier_count']:<20} {advanced_metrics['outlier_count']:<20}")
    print(f"{'Outlier rate (%)':<30} {base_metrics['outlier_rate']:<20.2f} {advanced_metrics['outlier_rate']:<20.2f}")
    print(f"{'Mean anomaly score':<30} {base_metrics['score_mean']:<20.4f} {advanced_metrics['score_mean']:<20.4f}")
    print(f"{'Std anomaly score':<30} {base_metrics['score_std']:<20.4f} {advanced_metrics['score_std']:<20.4f}")
    print("-"*70)

    # Recommendation
    print("\nRecommendation:")
    if abs(base_metrics['outlier_rate'] - 10.0) < abs(advanced_metrics['outlier_rate'] - 10.0):
        print("  → Base model (LOF) has more reasonable outlier detection rate")
    else:
        print("  → Advanced model (IF+HDBSCAN) has more reasonable outlier detection rate")

    if base_metrics['score_std'] < advanced_metrics['score_std']:
        print("  → Base model shows more stable anomaly scores")
    else:
        print("  → Advanced model shows more stable anomaly scores")

    # Save models
    save_models(base_model, advanced_model, scaler)

    # Save hyperparameters and metrics
    import json
    import os
    os.makedirs('models', exist_ok=True)

    results = {
        'base_model': {
            'hyperparameters': base_params,
            'test_metrics': base_metrics
        },
        'advanced_model': {
            'hyperparameters': advanced_model['best_params'],
            'test_metrics': advanced_metrics,
            'n_clusters': advanced_model['n_clusters']
        }
    }

    with open('models/training_results.json', 'w') as f:
        # Convert numpy types to python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            return obj

        json.dump(convert_numpy(results), f, indent=2)

    print("Training results saved to: models/training_results.json")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    parser.add_argument('--train', default='data/data.csv', help='Training data path')
    parser.add_argument('--test', default='data/test_data.csv', help='Test data path')
    parser.add_argument('--val-split', type=float, default=0.25,
                       help='Proportion of training data for validation (default: 0.25)')

    args = parser.parse_args()

    try:
        train_anomaly_detection_models(
            train_path=args.train,
            test_path=args.test,
            val_split=args.val_split
        )
    except (FileNotFoundError, AnomalyDetectionError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user", file=sys.stderr)
        sys.exit(130)
