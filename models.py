import pandas as pd
from datetime import timedelta
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional

# wczytanie danych
listings = pd.read_csv("listings.csv")
calendar = pd.read_csv("calendar.csv")

calendar["date"] = pd.to_datetime(calendar["date"])


# przygotowanie danych
def clean_price(price: float | str) -> float:
    if isinstance(price, str):
        return float(price.replace("$", "").replace(",", ""))
    return float(price)


calendar["price"] = calendar["price"].apply(clean_price)
listings["price"] = listings["price"].apply(clean_price)

# dołączenie kolumny property_type do tabeli calendar
data = calendar.merge(
    listings[["id", "property_type"]],
    left_on="listing_id",
    right_on="id",
    how="left"
)

data.drop(columns=["id"], inplace=True)


# funkcje pomocnicze
def mean_price_history(df: pd.DataFrame, listing_id: int, date: pd.Timestamp
                       ) -> Optional[float]:
    start_date = date - timedelta(days=60)

    mask = (
        (df["listing_id"] == listing_id) &
        (df["date"] < date) &
        (df["date"] >= start_date)
    )

    prices = df.loc[mask, "price"]

    if len(prices) < 3:
        return None

    return prices.mean()


def mean_price_property_type(df: pd.DataFrame, property_type: str,
                             date: pd.Timestamp) -> Optional[float]:
    mask = (
        (df["property_type"] == property_type) &
        (df["date"] == date)
    )

    prices = df.loc[mask, "price"]

    if len(prices) < 3:
        return None

    return prices.mean()


def base_model(df: pd.DataFrame, listing_id: int, date: pd.Timestamp,
               price: float, history_threshold: float = 0.4,
               property_threshold: float = 0.3) -> int:
    row = df[
        (df["listing_id"] == listing_id) &
        (df["date"] == date)
    ]

    if row.empty:
        return 0

    property_type = row.iloc[0]["property_type"]

    # kryterium 1 - historia
    mean_history = mean_price_history(df, listing_id, date)
    history_anomaly = False

    if mean_history is not None:
        history_anomaly = price > (1 + history_threshold) * mean_history

    # Kryterium 2 – rodzaj lokalu
    property_mean = mean_price_property_type(df, property_type, date)
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


# łączenie tabel calendar i listings
data2 = calendar.merge(
    listings[['id', 'property_type', 'room_type', 'accommodates',
              'bathrooms', 'bedrooms', 'beds']],
    left_on='listing_id',
    right_on='id',
    how='left'
)
data2.drop(columns=['id'], inplace=True)


# przewidywanie ceny za pomocą szeregów czasowych z oknem 365 dni (rok)
def predict_price(df: pd.DataFrame, listing_id: int, date: pd.Timestamp,
                  window: int = 365) -> Optional[float]:
    df_local = df[df['listing_id'] == listing_id].sort_values('date')
    past_window = df_local[df_local['date'] < date].tail(window)
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
def group_mean(df: pd.DataFrame, listing_id: int, date: pd.Timestamp
               ) -> Optional[float]:
    features = ['property_type', 'room_type', 'accommodates',
                'bathrooms', 'bedrooms', 'beds']
    df = listings[features + ['id']].copy()
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

    prices = calendar[(calendar['listing_id'].isin(cluster_listings)) &
                      (calendar['date'] == date)]['price']

    if len(prices) == 0:
        return None

    return prices.mean()


def advanced_model(df: pd.DataFrame, listing_id: int, date: pd.Timestamp,
                   price: float, history_threshold: float = 0.4,
                   group_threshold: float = 0.3) -> int:
    row = df[
        (df["listing_id"] == listing_id) &
        (df["date"] == date)
    ]

    if row.empty:
        return 0

    # kryterium 1 - historia
    predicted_price = predict_price(df, listing_id, date)
    history_anomaly = False

    if predicted_price is not None:
        history_anomaly = price > (1 + history_threshold) * predicted_price

    # Kryterium 2 – grupowanie
    mean_price = group_mean(df, listing_id, date)
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
