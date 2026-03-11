"""
Extract random sample(s) from the test split for demo purposes.

Usage:
    python scripts/generate_samples.py                  # 3 files, 200 rows each
    python scripts/generate_samples.py --n_files 5      # 5 files
    python scripts/generate_samples.py --n_rows 100     # 100 rows each
"""

import argparse
import pickle
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
OUTPUT_DIR = PROJECT_ROOT / "app" / "samples"


def load_test_data():
    with open(SPLITS_DIR / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(SPLITS_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return X_test, y_test


def generate_sample(X_test, y_test, n_rows=200, seed=None):
    """Sample n_rows from the test set, ensuring all attack types are included."""
    if seed is not None:
        random.seed(seed)

    df = X_test.copy()
    df["Attack Type"] = y_test.values

    samples = []
    for label in df["Attack Type"].unique():
        subset = df[df["Attack Type"] == label]
        n = max(2, int(len(subset) / len(df) * n_rows))
        n = min(n, len(subset))
        samples.append(subset.sample(n=n, random_state=seed))

    sample_df = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Return features only (without label) for prediction use
    features = sample_df.drop(columns=["Attack Type"])
    return features


def main():
    parser = argparse.ArgumentParser(description="Generate sample test data for NIDS dashboard")
    parser.add_argument("--n_files", type=int, default=3, help="Number of sample files to generate")
    parser.add_argument("--n_rows", type=int, default=200, help="Approximate rows per file")
    args = parser.parse_args()

    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"  Test set: {len(X_test):,} rows\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.n_files + 1):
        seed = 42 + i
        sample = generate_sample(X_test, y_test, n_rows=args.n_rows, seed=seed)
        filename = f"sample_{i}.csv"
        path = OUTPUT_DIR / filename
        sample.to_csv(path, index=False)
        print(f"  Created {filename} ({len(sample)} rows)")

    print(f"\nDone! Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
