import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import hdbscan
from typing import Optional, List


def compute_ground_truth(df: pd.DataFrame, cluster_col: str = None
                         ) -> pd.Series:
    df = df.copy()
    if cluster_col is None:
        high = df['log_price'].quantile(0.95)
        return ((df['log_price'] > high)).astype(int)
    else:
        labels = []
        for _, group in df.groupby(cluster_col):
            if len(group) < 2:
                labels.extend([0] * len(group))
                continue
            high = group['log_price'].quantile(0.95)
            labels.extend(((group['log_price'] > high)).astype(int))
        return pd.Series(labels, index=df.index)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "fpr": fpr}


class Model:
    _detector = None

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


class BaseModel(Model):
    """
    KMeans + threshold from mean price
    """
    def __init__(self, n_clusters: int = 5, threshold: float = 0.3):
        self.feature_columns_: Optional[List[str]] = None
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.best_params = {}

    def fit(self, X: pd.DataFrame):
        self.feature_columns = X.columns.tolist()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(X)
        return self

    def tune(self, X_train, X_val, price_val,
             cluster_values=[3, 5, 7], threshold_values=[0.2, 0.3, 0.4]):
        best_score = 0
        for k in cluster_values:
            for thr in threshold_values:
                self.n_clusters = k
                self.threshold = thr
                self.fit(X_train)
                y_pred = self.predict(X_val, price_val)
                cluster_labels_val = self.kmeans.predict(X_val)
                df_val_gt = pd.DataFrame({
                    "log_price": np.log1p(price_val),
                    "cluster_labels": cluster_labels_val
                })
                y_true = compute_ground_truth(df_val_gt, cluster_col="cluster_labels")
                metrics = evaluate(y_true, y_pred)
                if metrics["f1"] > best_score:
                    best_score = metrics["f1"]
                    self.best_params = {"n_clusters": k, "threshold": thr}
        self.n_clusters = self.best_params["n_clusters"]
        self.threshold = self.best_params["threshold"]
        self.fit(X_train)
        return self

    def predict(self, X: pd.DataFrame, price: pd.Series) -> np.ndarray:
        labels = self.kmeans.predict(X)
        y_pred = np.zeros(len(X), dtype=int)

        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            cluster_prices = price.iloc[idx]
            mu = cluster_prices.median()
            high = mu * (1 + self.threshold)
            y_pred[idx] = ((cluster_prices > high)).astype(int)

        return y_pred


class AdvancedModel(Model):
    """
    HDBSCAN + threshold from mean price
    """
    def __init__(self, min_cluster_size: int = 5, threshold: float = 0.3):
        self.feature_columns_: Optional[List[str]] = None
        self.min_cluster_size = min_cluster_size
        self.threshold = threshold
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.best_params = {}

    def fit(self, X: pd.DataFrame,):
        self.feature_columns = X.columns.tolist()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                         prediction_data=True)
        self.cluster_labels = self.clusterer.fit_predict(X)
        return self

    def tune(self, X_train, X_val, price_val,
             cluster_sizes=[5, 10, 15], threshold_values=[0.2, 0.3, 0.4]):
        best_score = 0
        for size in cluster_sizes:
            for thr in threshold_values:
                self.min_cluster_size = size
                self.threshold = thr
                self.fit(X_train)
                y_pred = self.predict(X_val, price_val)
                cluster_labels_val, _ = hdbscan.approximate_predict(self.clusterer, X_val)
                df_val_gt = pd.DataFrame({
                    "log_price": np.log1p(price_val),
                    "cluster_labels": cluster_labels_val
                })
                y_true = compute_ground_truth(df_val_gt,
                                              cluster_col="cluster_labels")
                metrics = evaluate(y_true, y_pred)
                if metrics["f1"] > best_score:
                    best_score = metrics["f1"]
                    self.best_params = {"min_cluster_size": size,
                                        "threshold": thr}
        self.min_cluster_size = self.best_params["min_cluster_size"]
        self.threshold = self.best_params["threshold"]
        self.fit(X_train)
        return self

    def predict(self, X: pd.DataFrame, price: pd.Series) -> np.ndarray:
        cluster_labels_val, _ = hdbscan.approximate_predict(self.clusterer, X)
        y_pred = np.zeros(len(X), dtype=int)

        for label in np.unique(cluster_labels_val):
            idx = np.where(cluster_labels_val == label)[0]
            cluster_prices = price.iloc[idx]
            mu = cluster_prices.median()
            high = mu * (1 + self.threshold)
            y_pred[idx] = ((cluster_prices > high)).astype(int)

        return y_pred
