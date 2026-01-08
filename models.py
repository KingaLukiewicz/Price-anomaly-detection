import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


def compute_ground_truth(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    """
    Dodaje kolumnę ground truth: 1=anomalia, 0=normal, None=za mało danych
    """
    df = df.copy()
    df["gt"] = None

    for _, group in df.groupby("listing_id"):
        idx = group.index
        if len(group) < min_rows:
            df.loc[idx, "gt"] = None
            continue
        mean = group["price"].mean()
        std = group["price"].std()
        threshold = mean + 3 * std
        df.loc[idx, "gt"] = (group["price"] > threshold).astype(int)
    return df


def metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """
    Recall (TPR) i FPR
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return recall, fpr


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               val_size: float = 0.2, random_state: int = 42):
    """
    Podział danych na zbiory treningowe, walidacyjne i testowe
    """
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)
    return train, val, test


def tune_base_model(train_df, val_df, history_values=[0.2, 0.3, 0.4, 0.5],
                    property_values=[0.2, 0.3, 0.4, 0.5]):
    best_model = None
    best_recall = 0
    for h in history_values:
        for p in property_values:
            model = BaseModel(history_threshold=h, property_threshold=p)
            y_true = val_df["gt"].dropna()
            y_pred = []
            for _, row in val_df.dropna(subset=["gt"]).iterrows():
                features = {"property_type": row["property_type"]}
                pred = model.detect_anomaly(row["date"], row["price"],
                                            features,
                                            listing_id=row["listing_id"])
                y_pred.append(pred)
            recall, fpr = metrics(y_true, pd.Series(y_pred))
            if recall > best_recall and fpr < 0.05:
                best_model = model
                best_model.history_threshold = h
                best_model.property_threshold = p
                best_recall = recall
    return best_model


def tune_advanced_model(train_df, val_df, history_values=[0.2, 0.3, 0.4, 0.5],
                        group_values=[0.2, 0.3, 0.4, 0.5]):
    best_model = None
    best_recall = 0
    for h in history_values:
        for g in group_values:
            model = AdvancedModel(history_threshold=h, group_threshold=g)
            y_true = val_df["gt"].dropna()
            y_pred = []
            for idx, row in val_df.dropna(subset=["gt"]).iterrows():
                features = {
                    "property_type": row["property_type"],
                    "room_type": row["room_type"],
                    "accommodates": row["accommodates"],
                    "bathrooms": row["bathrooms"],
                    "bedrooms": row["bedrooms"],
                    "beds": row["beds"]
                }
                pred = model.detect_anomaly(row["date"], row["price"],
                                            features,
                                            listing_id=row["listing_id"])
                y_pred.append(pred)
            recall, fpr = metrics(y_true, pd.Series(y_pred))
            if recall > best_recall and fpr < 0.05:
                best_model = model
                best_model.history_threshold = h
                best_model.group_threshold = g
                best_recall = recall
    return best_model


def evaluate(model, test_df, model_type="Base"):
    y_true = test_df["gt"].dropna()
    y_pred = []
    for _, row in test_df.dropna(subset=["gt"]).iterrows():
        features = {k: row[k] for k in ["property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds"] if k in row}
        listing_id = row.get("listing_id", None)
        pred = model.detect_anomaly(row["date"], row["price"], features, listing_id)
        y_pred.append(pred)
    recall, fpr = metrics(y_true, pd.Series(y_pred))
    print(f"{model_type} model test set evaluation -> Recall: {recall:.3f}, FPR: {fpr:.3f}")
    return recall, fpr


def train_and_export_models():
    # Przygotowanie danych
    base = BaseModel()
    adv = AdvancedModel()
    base.data = compute_ground_truth(base.data)
    adv.data = compute_ground_truth(adv.data)

    # Podział na train/val/test
    train_base, val_base, test_base = split_data(base.data)
    train_adv, val_adv, test_adv = split_data(adv.data)

    # Strojenie modeli
    best_base = tune_base_model(train_base, val_base)
    best_adv = tune_advanced_model(train_adv, val_adv)

    evaluate(best_base, test_base, "Base")
    evaluate(best_adv, test_adv, "Advanced")

    # Eksport
    joblib.dump(best_base, "base_model.pkl")
    joblib.dump(best_adv, "advanced_model.pkl")
    print("Modele zapisane do base_model.pkl i advanced_model.pkl")


class Model:
    def __init__(self, listings_path: str = "./data/listings.csv",
                 sessions_path: str = "./data/sessions.csv"):
        self.listings, self.sessions = self.load_and_prepare_data(
            listings_path, sessions_path)

    # przygotowanie danych
    def clean_price(self, price: float | str) -> float:
        if isinstance(price, str):
            return float(price.replace("$", "").replace(",", ""))
        return float(price)

    def load_and_prepare_data(self, listings_path: str, sessions_path: str
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # wczytanie danych
        listings = pd.read_csv(listings_path)
        sessions = pd.read_csv(sessions_path)

        # zachowujemy tylko wiersze z niepustą ceną (inne akcje niz book listing)
        sessions = sessions.dropna(subset=["price"])

        # przygotowanie danych
        sessions["date"] = pd.to_datetime(sessions["timestamp"]).dt.date
        sessions["price"] = sessions["price"].apply(self.clean_price)
        listings["price"] = listings["price"].apply(self.clean_price)
        return listings, sessions


class BaseModel(Model):
    def __init__(self, history_threshold: float = 0.4,
                 property_threshold: float = 0.3):
        super().__init__()
        self.history_threshold = history_threshold
        self.property_threshold = property_threshold
        self.data = self.sessions.merge(
            self.listings[["id", "property_type"]],
            left_on="listing_id",
            right_on="id",
            how="left"
        )
        self.data.drop(columns=["id"], inplace=True)

    # funkcje pomocnicze
    def mean_price_history(self, listing_id: int,
                           date: pd.Timestamp) -> Optional[float]:
        start_date = date - timedelta(days=60)

        mask = (
            (self.data["listing_id"] == listing_id) &
            (self.data["date"] < date) &
            (self.data["date"] >= start_date)
        )

        prices = self.data.loc[mask, "price"]

        if len(prices) < 3:
            return None

        return prices.mean()

    def mean_price_property_type(self, property_type: str,
                                 date: pd.Timestamp) -> Optional[float]:
        mask = (
            (self.data["property_type"] == property_type) &
            (self.data["date"] == date)
        )

        prices = self.data.loc[mask, "price"]

        if len(prices) < 3:
            return None

        return prices.mean()

    def detect_anomaly(self, date: pd.Timestamp, price: float,
                       features: dict, listing_id: Optional[int] = None
                       ) -> int:
        property_type = features["property_type"]

        # kryterium 1 - historia
        if listing_id is not None:
            mean_history = self.mean_price_history(listing_id, date)
            history_anomaly = False

            if mean_history is not None:
                history_anomaly = price > (1 + self.history_threshold) * mean_history

        # Kryterium 2 – rodzaj lokalu
        property_mean = self.mean_price_property_type(property_type, date)
        property_anomaly = False

        if property_mean is not None:
            property_anomaly = price > (1 + self.property_threshold) * property_mean

        # Decyzja końcowa
        if mean_history is not None and property_mean is not None:
            return int(history_anomaly and property_anomaly)

        if mean_history is not None:
            return int(history_anomaly)

        if property_mean is not None:
            return int(property_anomaly)

        return 0


class AdvancedModel(Model):
    def __init__(self, history_threshold: float = 0.4,
                 group_threshold: float = 0.3):
        super().__init__()
        self.history_threshold = history_threshold
        self.group_threshold = group_threshold
        # łączenie tabel sessions i listings
        self.data = self.sessions.merge(
            self.listings[['id', 'property_type', 'room_type', 'accommodates',
                           'bathrooms', 'bedrooms', 'beds']],
            left_on='listing_id',
            right_on='id',
            how='left'
        )
        self.data.drop(columns=['id'], inplace=True)

    # przewidywanie ceny za pomocą szeregów czasowych z oknem 365 dni (rok)
    def predict_price(self, listing_id: int, date: pd.Timestamp,
                      window: int = 365) -> Optional[float]:
        df = self.data[self.data['listing_id'] == listing_id].sort_values('date')
        past_window = df[df['date'] < date].tail(window)
        if len(past_window) < 10:
            return None
        series = past_window.set_index('date')['price']
        decomposition = seasonal_decompose(series, model='additive',
                                           period=min(30, len(series)//2),
                                           extrapolate_trend='freq')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        predicted = trend.iloc[-1] + seasonal.iloc[-1]
        return predicted

    # obliczanie średniej ceny lokalu w grupie
    def group_mean(self, date: pd.Timestamp, features: dict
                   ) -> Optional[float]:
        feat = ['property_type', 'room_type', 'accommodates',
                'bathrooms', 'bedrooms', 'beds']
        df = self.listings[feat + ['id']].copy()
        df['accommodates'] = df['accommodates'].fillna(0)
        df['bathrooms'] = df['bathrooms'].fillna(0)
        df['bedrooms'] = df['bedrooms'].fillna(0)
        df['beds'] = df['beds'].fillna(0)

        # one-hot encoding dla danych dyskretnych
        string_features = ['property_type', 'room_type']
        df_string = pd.get_dummies(df[string_features], drop_first=True)

        # dane liczbowe
        num_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
        df_num = df[num_features]

        X = pd.concat([df_num, df_string], axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        # grupowanie za pomocą HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=5)
        clusters = clusterer.fit_predict(X_scaled)
        df['cluster'] = clusters

        input_df = pd.DataFrame([features])
        for f in ['accommodates', 'bathrooms', 'bedrooms', 'beds']:
            if f not in input_df:
                input_df[f] = 0
        
        input_string = pd.get_dummies(input_df[string_features], drop_first=True)

        for col in df_string.columns:
            if col not in input_string:
                input_string[col] = 0
        input_string = input_string[df_string.columns]  # uporządkowanie kolumn
        input_num = input_df[['accommodates', 'bathrooms', 'bedrooms', 'beds']]
        X_input = np.concatenate([input_num.values, input_string.values], axis=1)

        cluster_centroids = []
        valid_clusters = df[df['cluster'] != -1]['cluster'].unique()
        for cl in valid_clusters:
            members = X_scaled[df['cluster'] == cl]
            centroid = members.mean(axis=0)
            cluster_centroids.append((cl, centroid))
        if not cluster_centroids:
            return None

        dists = [np.linalg.norm(X_input[0] - centroid) for cl, centroid in cluster_centroids]
        nearest_cluster = cluster_centroids[np.argmin(dists)][0]

        cluster_listings = df[df['cluster'] == nearest_cluster]['id'].values

        prices = self.sessions[(self.sessions['listing_id']
                                .isin(cluster_listings)) &
                               (self.sessions['date'] == date)]['price']

        if len(prices) == 0:
            return None

        return prices.mean()

    def detect_anomaly(self, date: pd.Timestamp, price: float,
                       features: dict, listing_id: Optional[int] = None
                       ) -> int:
        # kryterium 1 - historia
        if listing_id is not None:
            predicted_price = self.predict_price(self.data, listing_id, date)
            history_anomaly = False

            if predicted_price is not None:
                history_anomaly = price > (1 + self.history_threshold) * predicted_price

        # Kryterium 2 – grupowanie
        mean_price = self.group_mean(date, features)
        group_anomaly = False

        if mean_price is not None:
            group_anomaly = price > (1 + self.group_threshold) * mean_price

        # Decyzja końcowa
        if predicted_price is not None and mean_price is not None:
            return int(history_anomaly and group_anomaly)

        if predicted_price is not None:
            return int(history_anomaly)

        if mean_price is not None:
            return int(group_anomaly)

        return 0