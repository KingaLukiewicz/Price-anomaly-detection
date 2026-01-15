import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import hdbscan
from typing import Optional


def compute_ground_truth(df: pd.DataFrame, cluster_col: str = None
                         ) -> pd.Series:
    """
    Adds ground_truth column: based on log_price and 3 sigma rule
    """
    df = df.copy()
    if cluster_col is None:
        mu = df['log_price'].mean()
        sigma = df['log_price'].std()
        return ((df['log_price'] < mu - 2*sigma) |
                (df['log_price'] > mu + 2*sigma)).astype(int)
    else:
        labels = []
        for _, group in df.groupby(cluster_col):
            mu = group['log_price'].mean()
            sigma = group['log_price'].std()
            labels.extend(((group['log_price'] < mu - 2*sigma) |
                           (group['log_price'] > mu + 2*sigma)).astype(int))
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        1 - anomaly, 0 - normal
        """
        if not hasattr(self, "_detector") or self._detector is None:
            raise ValueError("Model is not trained.")
        return (self._detector.predict(X) == -1).astype(int)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


class BaseModel(Model):
    def __init__(self, n_clusters: int = 5, lof_neighbors: int = 20,
                 contamination: float = 0.01):
        self.n_clusters = n_clusters
        self.lof_neighbors = lof_neighbors
        self.contamination = contamination
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.best_params = {}

    def fit(self, X: pd.DataFrame, log_price: pd.Series):
        """
        Train model: clustering (k-means) and LOF
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(X)

        self._detector = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination=self.contamination,
            novelty=True
        )
        self._detector.fit(X)

        # Ground truth for evaluation
        df_gt = pd.DataFrame({
            'log_price': log_price,
            'cluster_labels': self.cluster_labels
        })
        self.gt = compute_ground_truth(df_gt, cluster_col="cluster_labels")
        return self
    
    def tune(self, X_train, log_price_train, X_val, log_price_val,
             cluster_values=[3, 5, 7], lof_neighbors=[10, 20, 30]):
        best_score = 0
        self.best_params = {"n_clusters": self.n_clusters,
                            "lof_neighbors": self.lof_neighbors}
        for k in cluster_values:
            for n in lof_neighbors:
                self.n_clusters = k
                self.lof_neighbors = n
                self.fit(X_train, log_price_train)
                y_pred = self.predict(X_val)
                df_val_gt = pd.DataFrame({
                    'log_price': log_price_val,
                    'cluster_labels': self.kmeans.predict(X_val)
                })
                y_true = compute_ground_truth(df_val_gt,
                                              cluster_col="cluster_labels")
                metrics_dict = evaluate(y_true, y_pred)
                if metrics_dict["f1"] > best_score:
                    best_score = metrics_dict["f1"]
                    self.best_params = {"n_clusters": k, "lof_neighbors": n}

        self.n_clusters = self.best_params["n_clusters"]
        self.lof_neighbors = self.best_params["lof_neighbors"]
        self.fit(X_train, log_price_train)
        joblib.dump(self._detector, "models/lof_model.pkl")
        return self


class AdvancedModel(Model):
    def __init__(self, min_cluster_size: int = 2,
                 iso_contamination: float = 0.01, n_estimators: int = 100):
        self.min_cluster_size = min_cluster_size
        self.iso_contamination = iso_contamination
        self.n_estimators = n_estimators
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.best_params = {}

    def fit(self, X: pd.DataFrame, log_price: pd.Series):
        """
        Train model: clustering (HDBSCAN) and Isolation Forest
        """
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                         prediction_data=True)
        self.cluster_labels = self.clusterer.fit_predict(X)

        self._detector = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.iso_contamination,
            random_state=42
        )
        self._detector.fit(X)

        # Ground truth for evaluation
        df_gt = pd.DataFrame({
            'log_price': log_price,
            'cluster_labels': self.cluster_labels
        })
        self.gt = compute_ground_truth(df_gt, cluster_col="cluster_labels")
        return self
    
    def tune(self, X_train: pd.DataFrame, log_price_train: pd.Series,
             X_val: pd.DataFrame, log_price_val: pd.Series,
             cluster_sizes=[5, 10, 15], contamination_values=[0.01, 0.02, 0.03]):
        best_score = 0
        self.best_params = {
            "min_cluster_size": self.min_cluster_size,
            "iso_contamination": self.iso_contamination
        }
        for c_size in cluster_sizes:
            for cont in contamination_values:
                self.min_cluster_size = c_size
                self.iso_contamination = cont
                self.fit(X_train, log_price_train)
                y_pred = self.predict(X_val)
                cluster_labels_val, strengths = hdbscan.approximate_predict(self.clusterer, X_val)
                df_val_gt = pd.DataFrame({
                    'log_price': log_price_val,
                    'cluster_labels': cluster_labels_val
                })
                y_true = compute_ground_truth(df_val_gt, cluster_col="cluster_labels")
                metrics_dict = evaluate(y_true, y_pred)
                if metrics_dict["f1"] > best_score:
                    best_score = metrics_dict["f1"]
                    self.best_params = {"min_cluster_size": c_size,
                                        "iso_contamination": cont}

        self.min_cluster_size = self.best_params["min_cluster_size"]
        self.iso_contamination = self.best_params["iso_contamination"]
        self.fit(X_train, log_price_train)
        return self
