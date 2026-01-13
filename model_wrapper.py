"""
Wrapper for trained ML models to provide compatible interface with existing app.py
Loads trained models and provides detect_anomaly() method.
"""

import joblib
import pandas as pd
import numpy as np
import hdbscan
from typing import Dict, Any


class ModelWrapper:
    """Base wrapper for anomaly detection models"""

    def __init__(self):
        self.scaler = None
        self.model = None
        self.training_columns = None  # Store column names from training
        self._load_model()

    def _load_model(self):
        """Load model and scaler - to be implemented by subclasses"""
        raise NotImplementedError

    def _prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Prepare features from input data

        Args:
            data: Dictionary with keys: neighbourhood_cleansed, property_type,
                  room_type, accommodates, bathrooms, bedrooms, beds, price

        Returns:
            Numpy array with prepared features
        """
        # Create DataFrame from input
        df = pd.DataFrame([data])

        # FIRST: Scale numeric columns (before one-hot encoding!)
        numeric_cols = ["accommodates", "bathrooms", "bedrooms", "beds"]
        if self.scaler is not None:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # SECOND: One-hot encode categorical columns
        categorical_cols = ["neighbourhood_cleansed", "property_type", "room_type"]
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # THIRD: Align columns with training data
        # Add missing columns with 0
        for col in self.training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Keep only columns that were in training (in same order)
        df_encoded = df_encoded[self.training_columns]

        return df_encoded.values

    def detect_anomaly(self, data: Dict[str, Any]) -> int:
        """Detect if input data is anomalous

        Args:
            data: Dictionary with features

        Returns:
            1 if anomaly, 0 if normal
        """
        raise NotImplementedError


class BaseModelWrapper(ModelWrapper):
    """Wrapper for trained LocalOutlierFactor model"""

    def _load_model(self):
        """Load trained base model"""
        try:
            self.model = joblib.load("models/base_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            self.training_columns = joblib.load("models/feature_columns.pkl")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found. Please run models_train.py first. Error: {e}"
            )

    def detect_anomaly(self, data: Dict[str, Any]) -> int:
        """Detect anomaly using LocalOutlierFactor

        Args:
            data: Input features dictionary

        Returns:
            1 if anomaly, 0 if normal
        """
        # Prepare features
        X = self._prepare_features(data)

        # Predict (-1 for outlier, 1 for inlier)
        prediction = self.model.predict(X)[0]

        # Convert to 1/0 (1=anomaly, 0=normal)
        return 1 if prediction == -1 else 0


class AdvancedModelWrapper(ModelWrapper):
    """Wrapper for trained IsolationForest + HDBSCAN model"""

    def _load_model(self):
        """Load trained advanced model"""
        try:
            self.model = joblib.load("models/advanced_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            self.training_columns = joblib.load("models/feature_columns.pkl")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found. Please run models_train.py first. Error: {e}"
            )

    def detect_anomaly(self, data: Dict[str, Any]) -> int:
        """Detect anomaly using IsolationForest per cluster

        Args:
            data: Input features dictionary

        Returns:
            1 if anomaly, 0 if normal
        """
        # Prepare features
        X = self._prepare_features(data)

        # Predict cluster using HDBSCAN
        clusterer = self.model["clusterer"]
        cluster_label, _ = hdbscan.approximate_predict(clusterer, X)
        cluster_label = cluster_label[0]

        # Get appropriate IsolationForest model for this cluster
        cluster_models = self.model["cluster_models"]

        if cluster_label in cluster_models:
            if_model = cluster_models[cluster_label]
        elif -1 in cluster_models:
            # Use noise model if sample doesn't belong to any cluster
            if_model = cluster_models[-1]
        else:
            # No model available, default to normal
            return 0

        # Predict (-1 for outlier, 1 for inlier)
        prediction = if_model.predict(X)[0]

        # Convert to 1/0 (1=anomaly, 0=normal)
        return 1 if prediction == -1 else 0
