import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import sys
import numpy as np
import joblib
import os


def load_data(filepath: str) -> pd.DataFrame:
    try:
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"File is empty: {filepath}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV file: {filepath}") from e

    return df


def clean_data(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    # clean numeric columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col + "_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    # clean price
    df['price'] = df['price'].astype(str).str.replace("$", "").str.replace(",", "")
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    return df


def scale_numeric(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  test_df: pd.DataFrame, numeric_columns: List[str]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_columns])

    train_df[numeric_columns] = scaler.transform(train_df[numeric_columns])
    val_df[numeric_columns] = scaler.transform(val_df[numeric_columns])
    test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

    return train_df, val_df, test_df, scaler


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # create price bins for stratification
    df["price_bin"] = pd.qcut(
        df["price"],
        q=5,
        labels=False,
        duplicates="drop"
    )

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["price_bin"]
    )

    # how much of train_val set is val_size from df
    val_relative_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df["price_bin"]
    )

    train_df = train_df.drop("price_bin", axis=1)
    val_df = val_df.drop("price_bin", axis=1)
    test_df = test_df.drop("price_bin", axis=1)

    return train_df, val_df, test_df


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: str = "data/data.csv",
    val_path: str = "data/val_data.csv",
    test_path: str = "data/test_data.csv",
) -> None:
    try:
        print("Saving data...")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    except (IOError, OSError) as e:
        raise IOError(f"Error saving files: {e}") from e


def prepare_data(
    input_path: str = "data/listings.csv",
    train_output: str = "data/data.csv",
    val_output: str = "data/val_data.csv",
    test_output: str = "data/test_data.csv",
) -> None:
    required_columns = [
        "neighbourhood_cleansed",
        "property_type",
        "room_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "price"
    ]
    numeric_columns = ["accommodates", "bathrooms", "bedrooms", "beds"]
    categorical_columns = ['neighbourhood_cleansed', 'property_type',
                           'room_type']

    # Load and select columns
    df = load_data(input_path)
    df = df[required_columns].copy()

    # Clean data
    df = clean_data(df, numeric_columns)

    # # Log-transform price
    df["log_price"] = np.log1p(df["price"])

    # One-hot encoding categorical data
    df_cat = pd.get_dummies(df[categorical_columns], drop_first=True)

    # Save encoded column names (categorical only)
    encoded_columns = df_cat.columns.tolist()
    joblib.dump(encoded_columns, "models/encoded_columns.pkl")
    print("Encoded columns saved to: models/encoded_columns.pkl")

    # Combine numeric columns and categorical dummy columns back
    df_processed = pd.concat([df[numeric_columns + ["log_price", "price"]],
                              df_cat], axis=1)

    # Split into train, validation, test sets
    train_df, val_df, test_df = split_train_val_test(df_processed)

    # Standardize numeric columns AFTER split (prevents data leakage)
    train_df, val_df, test_df, scaler = scale_numeric(
        train_df, val_df, test_df, numeric_columns
    )

    # Save scaler for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to: models/scaler.pkl")

    # Save datasets
    save_datasets(train_df, val_df, test_df, train_output,
                  val_output, test_output)

    print("Data preparation complete!")
    print(f"Files saved: {train_output}, {val_output} and {test_output}")


if __name__ == "__main__":
    try:
        prepare_data()
    except (FileNotFoundError, IOError) as e:
        print(f" Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user", file=sys.stderr)
        sys.exit(130)
