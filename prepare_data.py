import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ast
import re
from typing import Set, List, Tuple
import sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


class DataPreparationError(Exception):
    """Custom exception for data preparation errors"""

    pass


def parse_amenities(amenities_str) -> Set[str]:
    """Parse amenities string and return a set of amenities

    Args:
        amenities_str: String representation of amenities list

    Returns:
        Set of amenity strings
    """
    try:
        if pd.isna(amenities_str):
            return set()

        amenities_str = str(amenities_str).strip()

        # Try parsing as JSON/Python list
        try:
            amenities_list = ast.literal_eval(amenities_str)
            return set(amenities_list)
        except (ValueError, SyntaxError):
            # Fallback: simple string parsing
            amenities_str = amenities_str.strip("[]{}")
            amenities = [a.strip().strip("\"'") for a in amenities_str.split(",")]
            return set(a for a in amenities if a)
    except (AttributeError, TypeError) as e:
        print(f"Warning: Could not parse amenities: {e}")
        return set()


def load_and_select_columns(filepath: str, required_columns: List[str]) -> pd.DataFrame:
    """Load CSV file and select required columns

    Args:
        filepath: Path to the CSV file
        required_columns: List of column names to select

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If the file doesn't exist
        DataPreparationError: If required columns are missing
    """
    try:
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except pd.errors.EmptyDataError as e:
        raise DataPreparationError(f"File is empty: {filepath}") from e
    except pd.errors.ParserError as e:
        raise DataPreparationError(f"Error parsing CSV file: {filepath}") from e

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataPreparationError(f"Missing required columns: {missing_cols}")

    df = df[required_columns].copy()

    print(f"Rows before removing NaN: {len(df)}")
    df = df.dropna()
    print(f"Rows after removing NaN: {len(df)}")

    if len(df) == 0:
        raise DataPreparationError("No data remaining after removing NaN values")

    return df


def extract_unique_amenities(df: pd.DataFrame) -> List[str]:
    """Extract all unique amenities from the amenities column

    Args:
        df: DataFrame with 'amenities' column

    Returns:
        Sorted list of unique amenity names
    """
    print("Processing amenities...")
    all_amenities = set()

    for amenities_str in df["amenities"]:
        all_amenities.update(parse_amenities(amenities_str))

    # Remove empty strings
    all_amenities = sorted([a for a in all_amenities if a and a.strip()])
    print(f"Found {len(all_amenities)} unique amenities")

    return all_amenities


def create_amenity_columns(df: pd.DataFrame, all_amenities: List[str]) -> pd.DataFrame:
    """Create binary columns for each amenity

    Args:
        df: DataFrame with 'amenities' column
        all_amenities: List of all unique amenity names

    Returns:
        DataFrame with amenity columns added and original amenities column
        removed
    """
    # Parse amenities once for all rows
    parsed_amenities = df["amenities"].apply(parse_amenities)

    # Create all amenity columns at once using dictionary
    amenity_columns = {}
    for amenity in all_amenities:
        col_name = f"amenity_{re.sub(r'[^a-zA-Z0-9]', '_', amenity).lower()}"
        amenity_columns[col_name] = parsed_amenities.apply(
            lambda x: 1 if amenity in x else 0
        )

    # Create DataFrame from dictionary and concatenate
    amenity_df = pd.DataFrame(amenity_columns)
    df = pd.concat([df.drop("amenities", axis=1), amenity_df], axis=1)

    return df


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert price column to numeric

    Args:
        df: DataFrame with 'price' column

    Returns:
        DataFrame with cleaned price column

    Raises:
        DataPreparationError: If price column is missing or cannot be converted
    """
    if "price" not in df.columns:
        raise DataPreparationError("Price column is missing")

    try:
        df["price"] = df["price"].astype(str).str.replace("$", "", regex=False)
        df["price"] = df["price"].str.replace(",", "", regex=False)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])

        if len(df) == 0:
            raise DataPreparationError("No valid price data remaining")

        return df
    except (ValueError, KeyError) as e:
        raise DataPreparationError(f"Error processing price column: {e}")


def standardize_numeric_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numeric columns using StandardScaler

    Fits scaler on training data only, then transforms both train and test.
    This prevents data leakage.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        numeric_columns: List of column names to standardize

    Returns:
        Tuple of (standardized train_df, standardized test_df, fitted scaler)

    Raises:
        DataPreparationError: If standardization fails
    """
    numeric_columns = [col for col in numeric_columns if col in train_df.columns]

    if not numeric_columns:
        raise DataPreparationError("No numeric columns found for standardization")

    print(f"\nStandardizing numeric columns: {numeric_columns}")
    print("  Fitting scaler on TRAINING data only (preventing data leakage)")

    try:
        scaler = StandardScaler()
        # Fit only on training data
        scaler.fit(train_df[numeric_columns])

        # Transform both train and test
        train_df[numeric_columns] = scaler.transform(train_df[numeric_columns])
        test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

        print(f"  Scaler params (from training data):")
        print(f"    Mean: {scaler.mean_}")
        print(f"    Std: {scaler.scale_}")

        return train_df, test_df, scaler
    except ValueError as e:
        raise DataPreparationError(f"Error during standardization: {e}") from e


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with stratification

    Args:
        df: DataFrame to split
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        DataPreparationError: If splitting fails
    """
    try:
        # Create price bins for stratification
        df["price_bin"] = pd.qcut(df["price"], q=5, labels=False, duplicates="drop")

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["price_bin"]
        )

        # Remove the price_bin column
        train_df = train_df.drop("price_bin", axis=1)
        test_df = test_df.drop("price_bin", axis=1)

        return train_df, test_df
    except ValueError as e:
        raise DataPreparationError(f"Error splitting data: {e}") from e


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: str = "data/data.csv",
    test_path: str = "data/test_data.csv",
) -> None:
    """Save train and test datasets to CSV files

    Args:
        train_df: Training dataset
        test_df: Test dataset
        train_path: Path to save training data
        test_path: Path to save test data

    Raises:
        IOError: If saving fails
    """
    try:
        print("Saving data...")
        total_rows = len(train_df) + len(test_df)
        print(
            f"Training set size: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)"
        )
        print(
            f"Test set size: {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)"
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("\nPrice distribution comparison:")
        print(
            f"Train set - Mean: {train_df['price'].mean():.4f}, Std: {
                train_df['price'].std():.4f}"
        )
        print(
            f"Test set  - Mean: {test_df['price'].mean():.4f}, Std: {
                test_df['price'].std():.4f}"
        )

    except (IOError, OSError) as e:
        raise IOError(f"Error saving files: {e}") from e


def visualize_price_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str = "data/price_distribution.png",
) -> None:
    """
    Create visualization comparing price distributions between
    train and test sets

    Args:
        train_df: Training dataset
        test_df: Test dataset
        output_path: Path to save the visualization
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Price Distribution Comparison: Train vs Test",
            fontsize=16,
            fontweight="bold",
        )

        train_prices = train_df["price"].values
        test_prices = test_df["price"].values

        # 1. Histograms
        ax1 = axes[0, 0]
        bins = 50
        ax1.hist(
            train_prices,
            bins=bins,
            alpha=0.6,
            label="Train",
            color="blue",
            density=True,
        )
        ax1.hist(
            test_prices,
            bins=bins,
            alpha=0.6,
            label="Test",
            color="orange",
            density=True,
        )
        ax1.set_xlabel("Standardized Price")
        ax1.set_ylabel("Density")
        ax1.set_title("Histogram Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plots
        ax2 = axes[0, 1]
        box_data = [train_prices, test_prices]
        bp = ax2.boxplot(box_data, labels=["Train", "Test"], patch_artist=True)
        bp["boxes"][0].set_facecolor("blue")
        bp["boxes"][1].set_facecolor("orange")
        ax2.set_ylabel("Standardized Price")
        ax2.set_title("Box Plot Comparison")
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax3 = axes[1, 0]
        # Sort both datasets
        sorted_train = np.sort(train_prices)
        sorted_test = np.sort(test_prices)
        # Use same number of quantiles
        n_quantiles = min(len(sorted_train), len(sorted_test))
        train_quantiles = np.percentile(sorted_train, np.linspace(0, 100, n_quantiles))
        test_quantiles = np.percentile(sorted_test, np.linspace(0, 100, n_quantiles))

        ax3.scatter(train_quantiles, test_quantiles, alpha=0.5, s=10)
        min_val = min(train_quantiles.min(), test_quantiles.min())
        max_val = max(train_quantiles.max(), test_quantiles.max())
        ax3.plot(
            [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect match"
        )
        ax3.set_xlabel("Train Quantiles")
        ax3.set_ylabel("Test Quantiles")
        ax3.set_title("Q-Q Plot")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        ax4.hist(
            train_prices,
            bins=100,
            alpha=0.6,
            label="Train",
            color="blue",
            cumulative=True,
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax4.hist(
            test_prices,
            bins=100,
            alpha=0.6,
            label="Test",
            color="orange",
            cumulative=True,
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax4.set_xlabel("Standardized Price")
        ax4.set_ylabel("Cumulative Probability")
        ax4.set_title("Cumulative Distribution Function")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Price distribution visualization saved to: {output_path}")
        plt.close()

    except (ValueError, IOError) as e:
        print(f"Warning: Could not create visualization: {e}")


def statistical_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Perform statistical tests to compare train and test distributions

    Args:
        train_df: Training dataset
        test_df: Test dataset
    """
    train_prices = train_df["price"].values
    test_prices = test_df["price"].values

    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(train_prices, test_prices)

    # Mann-Whitney U test (non-parametric)
    mw_statistic, mw_pvalue = stats.mannwhitneyu(
        train_prices, test_prices, alternative="two-sided"
    )

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON OF PRICE DISTRIBUTIONS")
    print("=" * 60)

    print("\nDescriptive Statistics:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Train':<20} {'Test':<20}")
    print("-" * 60)
    print(f"{'Mean':<20} {train_prices.mean():<20.4f} {test_prices.mean():<20.4f}")
    print(f"{'Std Dev':<20} {train_prices.std():<20.4f} {test_prices.std():<20.4f}")
    print(
        f"{'Median':<20} {np.median(train_prices):<20.4f} {np.median(test_prices):<20.4f}"
    )
    print(f"{'Min':<20} {train_prices.min():<20.4f} {test_prices.min():<20.4f}")
    print(f"{'Max':<20} {train_prices.max():<20.4f} {test_prices.max():<20.4f}")
    print(
        f"{'25th percentile':<20} {np.percentile(train_prices, 25):<20.4f} {np.percentile(test_prices, 25):<20.4f}"
    )
    print(
        f"{'75th percentile':<20} {np.percentile(train_prices, 75):<20.4f} {np.percentile(test_prices, 75):<20.4f}"
    )

    print("\n\nStatistical Tests:")
    print("-" * 60)
    print("Kolmogorov-Smirnov Test:")
    print(f"Statistic: {ks_statistic:.6f}")
    print(f"p-value: {ks_pvalue:.6f}")
    if ks_pvalue > 0.05:
        print("Distributions are similar (p > 0.05)")
    else:
        print("Distributions may differ (p ≤ 0.05)")

    print("Mann-Whitney U Test:")
    print(f"Statistic: {mw_statistic:.2f}")
    print(f"p-value: {mw_pvalue:.6f}")
    if mw_pvalue > 0.05:
        print("Distributions are similar (p > 0.05)")
    else:
        print("Distributions may differ (p ≤ 0.05)")

    print("=" * 60)


def prepare_data(
    input_path: str = "data/listings.csv",
    train_output: str = "data/data.csv",
    test_output: str = "data/test_data.csv",
) -> None:
    """Main function to prepare and split data

    Args:
        input_path: Path to input CSV file
        train_output: Path to save training data
        test_output: Path to save test data
    """
    required_columns = [
        "neighbourhood_cleansed",
        "property_type",
        "room_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "price",
        # "amenities",  # Commented out - not using amenities for now
    ]

    # Only standardize features, NOT the target price!
    numeric_columns = ["accommodates", "bathrooms", "bedrooms", "beds"]

    # Load and select columns
    df = load_and_select_columns(input_path, required_columns)

    # Process amenities - COMMENTED OUT (not using amenities for now)
    # all_amenities = extract_unique_amenities(df)
    # df = create_amenity_columns(df, all_amenities)

    # Clean price column
    df = clean_price_column(df)

    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")

    # Split into train and test FIRST (before scaling!)
    train_df, test_df = split_train_test(df)

    # Standardize numeric columns AFTER split (prevents data leakage)
    train_df, test_df, scaler = standardize_numeric_columns(
        train_df, test_df, numeric_columns
    )

    # Save scaler for later use
    import joblib
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Scaler saved to: models/scaler.pkl")

    # Save datasets
    save_datasets(train_df, test_df, train_output, test_output)

    # Visualize and compare distributions
    visualize_price_distribution(train_df, test_df)
    statistical_comparison(train_df, test_df)

    print("Data preparation complete!")
    print(f"Files saved: {train_output} and {test_output}")


if __name__ == "__main__":
    try:
        prepare_data()
    except (FileNotFoundError, DataPreparationError, IOError) as e:
        print(f" Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user", file=sys.stderr)
        sys.exit(130)
