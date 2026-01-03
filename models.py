import pandas as pd
from datetime import timedelta
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, Tuple


class Model:
    def __init__(self, listings_path: str = "./data/listings.csv",
                 calendar_path: str = "./data/calendar.csv"):
        self.listings, self.calendar = self.load_and_prepare_data(
            listings_path, calendar_path)

    # przygotowanie danych
    def clean_price(self, price: float | str) -> float:
        if isinstance(price, str):
            return float(price.replace("$", "").replace(",", ""))
        return float(price)

    def load_and_prepare_data(self, listings_path: str, calendar_path: str
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # wczytanie danych
        listings = pd.read_csv(listings_path)
        calendar = pd.read_csv(calendar_path)

        # przygotowanie danych
        calendar["date"] = pd.to_datetime(calendar["date"])
        calendar["price"] = calendar["price"].apply(self.clean_price)
        listings["price"] = listings["price"].apply(self.clean_price)
        return listings, calendar


class BaseModel(Model):
    def __init__(self):
        super().__init__()
        self.data = self.calendar.merge(
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

    def detect_anomaly(self, listing_id: int, date: pd.Timestamp,
                       price: float, history_threshold: float = 0.4,
                       property_threshold: float = 0.3) -> int:

        # dołączenie kolumny property_type do tabeli calendar
        row = self.data[
            (self.data["listing_id"] == listing_id) &
            (self.data["date"] == date)
        ]

        if row.empty:
            return 0

        property_type = row.iloc[0]["property_type"]

        # kryterium 1 - historia
        mean_history = self.mean_price_history(listing_id, date)
        history_anomaly = False

        if mean_history is not None:
            history_anomaly = price > (1 + history_threshold) * mean_history

        # Kryterium 2 – rodzaj lokalu
        property_mean = self.mean_price_property_type(property_type, date)
        property_anomaly = False

        if property_mean is not None:
            property_anomaly = price > (1 + property_threshold) * property_mean

        # Decyzja końcowa
        if mean_history is not None and property_mean is not None:
            return int(history_anomaly and property_anomaly)

        if mean_history is not None:
            return int(history_anomaly)

        if property_mean is not None:
            return int(property_anomaly)

        return 0


class AdvancedModel(Model):
    def __init__(self):
        super().__init__()
        # łączenie tabel calendar i listings
        self.data = self.calendar.merge(
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
    def group_mean(self, listing_id: int, date: pd.Timestamp
                   ) -> Optional[float]:
        features = ['property_type', 'room_type', 'accommodates',
                    'bathrooms', 'bedrooms', 'beds']
        df = self.listings[features + ['id']].copy()
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

        try:
            listing_cluster = df.loc[df['id'] == listing_id, 'cluster'].values[0]
        except IndexError:
            return None

        # jeśli lokal nie posiada innych lokali w swojej grupie
        # (jest nietypowy i został uznany za szum)
        if listing_cluster == -1:
            return None

        cluster_listings = df[df['cluster'] == listing_cluster]['id'].values

        prices = self.calendar[(self.calendar['listing_id']
                                .isin(cluster_listings)) &
                               (self.calendar['date'] == date)]['price']

        if len(prices) == 0:
            return None

        return prices.mean()

    def detect_anomaly(self, listing_id: int,
                       date: pd.Timestamp, price: float,
                       history_threshold: float = 0.4,
                       group_threshold: float = 0.3) -> int:
        row = self.data[
            (self.data["listing_id"] == listing_id) &
            (self.data["date"] == date)
        ]

        if row.empty:
            return 0

        # kryterium 1 - historia
        predicted_price = self.predict_price(self.data, listing_id, date)
        history_anomaly = False

        if predicted_price is not None:
            history_anomaly = price > (1 + history_threshold) * predicted_price

        # Kryterium 2 – grupowanie
        mean_price = self.group_mean(self.data, listing_id, date)
        group_anomaly = False

        if mean_price is not None:
            group_anomaly = price > (1 + group_threshold) * mean_price

        # Decyzja końcowa
        if predicted_price is not None and mean_price is not None:
            return int(history_anomaly and group_anomaly)

        if predicted_price is not None:
            return int(history_anomaly)

        if mean_price is not None:
            return int(group_anomaly)

        return 0
