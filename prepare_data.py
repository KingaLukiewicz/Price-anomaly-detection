"""
Przygotowanie i etykietowanie danych do trenowania modeli
Łączy dane z sessions (ceny, daty) z listings (cechy objektów)
Tworzy ground truth (czy anomalia czy nie)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Przygotowanie danych do trenowania"""

    def __init__(
        self,
        listings_path: str = "./data/listings.csv",
        sessions_path: str = "./data/sessions.csv",
    ):
        self.listings = pd.read_csv(listings_path)
        self.sessions = pd.read_csv(sessions_path)

    def clean_price(self, price):
        """Czyszczenie cen - usuwanie $ i ,"""
        if pd.isna(price):
            return np.nan
        if isinstance(price, str):
            return float(price.replace("$", "").replace(",", ""))
        return float(price)

    def prepare_data(self):
        """Przygotowanie dataframu do trenowania"""
        # Wczytanie i czyszczenie sessions
        sessions = self.sessions[self.sessions["action"] == "book_listing"].copy()
        sessions = sessions.dropna(subset=["price"])
        sessions["price"] = sessions["price"].apply(self.clean_price)
        sessions["date"] = pd.to_datetime(sessions["timestamp"]).dt.date

        # Wybranie kolumn z sessions: listing_id, date, price
        sessions = sessions[["listing_id", "date", "price"]]

        # Przygotowanie listings
        listings_features = self.listings[
            [
                "id",
                "neighbourhood_cleansed",
                "property_type",
                "room_type",
                "accommodates",
                "bathrooms",
                "bedrooms",
                "beds",
                "review_scores_rating",
                "review_scores_accuracy",
                "review_scores_cleanliness",
                "review_scores_checkin",
                "review_scores_communication",
                "review_scores_location",
                "review_scores_value",
                "reviews_per_month",
            ]
        ]
        listings_features = listings_features.rename(columns={"id": "listing_id"})

        # Czyszczenie listings - konwersja numerycznych kolumn
        for col in [
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds",
            "review_scores_rating",
            "review_scores_accuracy",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value",
            "reviews_per_month",
        ]:
            listings_features[col] = pd.to_numeric(listings_features[col], errors="coerce")

        # Merge sessions z listings po listing_id
        data = sessions.merge(listings_features, on="listing_id", how="left")

        # Usuwanie wierszy bez kluczowych informacji
        data = data.dropna(subset=["price", "property_type", "room_type"])

        # Fillna dla review scores i numerycznych features
        for col in data.select_dtypes(include=[np.number]).columns:
            if col != "price":  # price już nie ma NaN
                data[col] = data[col].fillna(data[col].median())

        return data

    def compute_ground_truth(self, data, z_score_threshold=2.5):
        """
        Ground truth: cena jest anomalią jeśli jest > mean + z_score_threshold*std
        Obliczane dla każdego property_type osobno
        """
        data = data.copy()
        data["anomaly"] = 0

        for prop_type, group in data.groupby("property_type"):
            if len(group) < 5:  # Za mało danych dla tego typu
                continue

            mean = group["price"].mean()
            std = group["price"].std()

            if std > 0:
                anomalies = group["price"] > (mean + z_score_threshold * std)
                data.loc[group.index[anomalies], "anomaly"] = 1

        return data


def main():
    print("=" * 70)
    print("PRZYGOTOWANIE I ETYKIETOWANIE DANYCH")
    print("=" * 70)

    # Przygotowanie danych
    print("\n1. Wczytywanie i czyszczenie danych...")
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data()
    print(f"   ✓ Wczytano {len(data)} rekordów z cenami i cechami")
    print(f"   ✓ Kolumny: {list(data.columns)}")

    # Etykietowanie anomalii
    print("\n2. Etykietowanie anomalii (ground truth)...")
    data = preprocessor.compute_ground_truth(data)
    n_anomalies = data["anomaly"].sum()
    print(
        f"   ✓ Anomalii: {n_anomalies} z {len(data)} ({n_anomalies/len(data)*100:.2f}%)"
    )

    # Statystyki
    print("\n3. Statystyki danych:")
    print(f"   - Data range: {data['date'].min()} do {data['date'].max()}")
    print(f"   - Ceny - Min: ${data['price'].min():.2f}, Max: ${data['price'].max():.2f}")
    print(f"   - Ceny - Mean: ${data['price'].mean():.2f}, Std: ${data['price'].std():.2f}")
    print(f"   - Property types: {data['property_type'].nunique()}")
    print(f"   - Room types: {data['room_type'].nunique()}")
    print(f"   - Neighbourhoods: {data['neighbourhood_cleansed'].nunique()}")

    print("\n   Rozkład anomalii po property_type:")
    anomaly_by_type = data.groupby("property_type")["anomaly"].agg(["sum", "count"])
    anomaly_by_type["pct"] = anomaly_by_type["sum"] / anomaly_by_type["count"] * 100
    print(anomaly_by_type.to_string())

    # Podział danych
    print("\n4. Podział danych na train/val/test (60/20/20)...")
    train_val, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    print(f"   ✓ Train: {len(train)} próbek (anomalii: {train['anomaly'].sum()})")
    print(f"   ✓ Val:   {len(val)} próbek (anomalii: {val['anomaly'].sum()})")
    print(f"   ✓ Test:  {len(test)} próbek (anomalii: {test['anomaly'].sum()})")

    # Zapis danych
    print("\n5. Zapis przygotowanych danych...")
    train.to_csv("data_train.csv", index=False)
    val.to_csv("data_val.csv", index=False)
    test.to_csv("data_test.csv", index=False)
    data.to_csv("data_full.csv", index=False)

    print("   ✓ data_train.csv")
    print("   ✓ data_val.csv")
    print("   ✓ data_test.csv")
    print("   ✓ data_full.csv")

    print("\n" + "=" * 70)
    print("PRZYGOTOWANIE DANYCH UKOŃCZONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
