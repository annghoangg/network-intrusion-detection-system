import numpy as np
import pandas as pd


# Nhóm nhãn tấn công từ CICIDS2017 → nhãn tổng quát
_LABEL_MAP: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Các lớp bị loại khỏi phân tích (quá hiếm / nhiễu)
_CLASSES_TO_REMOVE = ["Infiltration", "Miscellaneous"]


# 1. Các bước làm sạch riêng lẻ

def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    dropped = before - len(df)
    print(f"  remove_duplicates: removed {dropped:,} duplicate rows "
          f"({dropped / before * 100:.2f}%)")
    return df


def remove_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()
    keep = list(columns)
    to_drop: list[str] = []

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            if col2 in keep and df[col1].equals(df[col2]):
                keep.remove(col2)
                to_drop.append(col2)

    if to_drop:
        print(f"  remove_identity_columns: dropping {len(to_drop)} redundant column(s): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("  remove_identity_columns: no identical columns found.")
    return df


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    inf_counts = np.isinf(df[num_cols]).sum()
    total_inf = inf_counts[inf_counts > 0].sum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"  handle_infinite_values: replaced {total_inf:,} infinite value(s) with NaN.")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    before = len(df)
    if strategy == "drop":
        df = df.dropna().reset_index(drop=True)
        dropped = before - len(df)
        print(f"  handle_missing_values (drop): removed {dropped:,} rows containing NaN "
              f"({dropped / before * 100:.2f}%)")
    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet.")
    return df


def standardise_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Label" in df.columns and "Attack Type" not in df.columns:
        df.rename(columns={"Label": "Attack Type"}, inplace=True)

    df["Attack Type"] = df["Attack Type"].str.strip()
    df["Attack Type"] = df["Attack Type"].map(_LABEL_MAP).fillna(df["Attack Type"])

    before = len(df)
    df = df[~df["Attack Type"].isin(_CLASSES_TO_REMOVE)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  standardise_label_column: removed {removed:,} rows "
              f"belonging to {_CLASSES_TO_REMOVE}.")

    dist = df["Attack Type"].value_counts()
    print(f"\n  Attack Type distribution after standardisation:")
    print(dist.to_string())
    print()
    return df


# 2. Pipeline tổng hợp

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = strip_column_names(df)
    df = remove_duplicates(df)
    df = remove_identity_columns(df)
    df = handle_infinite_values(df)
    df = handle_missing_values(df, strategy="drop")
    df = standardise_label_column(df)

    print("=" * 60)
    print(f"  ✓ Preprocessing complete. Final shape: {df.shape[0]:,} × {df.shape[1]}")
    print("=" * 60 + "\n")
    return df
