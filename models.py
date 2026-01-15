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
    df = df.copy()
    if cluster_col is None:
        low = df['log_price'].quantile(0.05)
        high = df['log_price'].quantile(0.95)
        return ((df['log_price'] < low) | (df['log_price'] > high)).astype(int)
    else:
        labels = []
        for _, group in df.groupby(cluster_col):
            if len(group) < 2:
                labels.extend([0] * len(group))
                continue
            low = group['log_price'].quantile(0.05)
            high = group['log_price'].quantile(0.95)
            labels.extend(((group['log_price'] < low) |
                           (group['log_price'] > high)).astype(int))
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
    def __init__(self, n_clusters: int = 5, lof_neighbors: int = 20,
                 contamination: float = 0.01):
        self.n_clusters = n_clusters
        self.lof_neighbors = lof_neighbors
        self.contamination = contamination
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.detectors: dict[int, LocalOutlierFactor] = {}
        self.best_params = {}

    def fit(self, X: pd.DataFrame, log_price: pd.Series):
        """
        Train model: clustering (k-means) and LOF
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(X)

        self.detectors = {}
        for label in np.unique(self.cluster_labels):
            idx = np.where(self.cluster_labels == label)[0]
            cluster_X = X.iloc[idx].copy()
            cluster_X['log_price'] = log_price.iloc[idx]
            cluster_X = cluster_X[X.columns.tolist() + ['log_price']]
            if len(cluster_X) < self.lof_neighbors:
                n_neighbors = max(2, len(cluster_X) - 1)
            else:
                n_neighbors = self.lof_neighbors

            detector = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                novelty=True
            )
            detector.fit(cluster_X)
            self.detectors[label] = detector

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
                y_pred = self.predict(X_val, log_price_val)
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
    
    def predict(self, X: pd.DataFrame, log_price: pd.Series) -> np.ndarray:
        labels = self.kmeans.predict(X)
        y_pred = np.zeros(len(X), dtype=int)
        
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            cluster_X = X.iloc[idx].copy()
            cluster_X['log_price'] = log_price.iloc[idx]
            cluster_X.columns = self.detectors[label].feature_names_in_
            
            y_pred[idx] = (self.detectors[label].predict(cluster_X) == -1).astype(int)
        
        return y_pred


class AdvancedModel(Model):
    def __init__(self, min_cluster_size: int = 2,
                 iso_contamination: float = 0.01, n_estimators: int = 100):
        self.min_cluster_size = min_cluster_size
        self.iso_contamination = iso_contamination
        self.n_estimators = n_estimators
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.detectors: dict[int, IsolationForest] = {}
        self.best_params = {}

    def fit(self, X: pd.DataFrame, log_price: pd.Series):
        """
        Train model: clustering (HDBSCAN) and Isolation Forest
        """
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                         prediction_data=True)
        self.cluster_labels = self.clusterer.fit_predict(X)

        self.detectors = {}
        for label in np.unique(self.cluster_labels):
            idx = np.where(self.cluster_labels == label)[0]
            cluster_X = X.iloc[idx].copy()
            cluster_X['log_price'] = log_price.iloc[idx]
            cluster_X = cluster_X[X.columns.tolist() + ['log_price']]
            if len(cluster_X) < 2:
                continue
            detector = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.iso_contamination,
                random_state=42
            )
            detector.fit(cluster_X)
            self.detectors[label] = detector

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
                y_pred = self.predict(X_val, log_price_val)
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
    
    def predict(self, X: pd.DataFrame, log_price: pd.Series) -> np.ndarray:
        cluster_labels_val, _ = hdbscan.approximate_predict(self.clusterer, X)
        y_pred = np.zeros(len(X), dtype=int)

        for label in np.unique(cluster_labels_val):
            idx = np.where(cluster_labels_val == label)[0]
            cluster_X = X.iloc[idx].copy()
            cluster_X['log_price'] = log_price.iloc[idx]
            cluster_X.columns = self.detectors[label].feature_names_in_

            if label in self.detectors:
                y_pred[idx] = (self.detectors[label].predict(cluster_X) == -1).astype(int)

        return y_pred
