import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cột bị loại do đa cộng tuyến cao (|r| ≥ 0.95)
_HIGH_MULTICOLLINEARITY_COLS: list[str] = [
    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]

# Cột bị loại do điểm quan trọng thấp (RF + Kruskal-Wallis)
_LOW_IMPORTANCE_COLS: list[str] = [
    "ECE Flag Count",
    "RST Flag Count",
    "Fwd URG Flags",
    "Idle Std",
    "Fwd PSH Flags",
    "Active Std",
    "Down/Up Ratio",
    "URG Flag Count",
]


# 1. Phân loại đặc trưng

def get_feature_types(
    df: pd.DataFrame,
    target_col: str = "Attack Type",
) -> tuple[list, list]:
    num_feat = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]
    cat_feat = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col != target_col
    ]
    print(f"  get_feature_types: {len(num_feat)} numerical, {len(cat_feat)} categorical features.")
    return num_feat, cat_feat


# 2. Phân tích tương quan

def correlation_analysis(
    df: pd.DataFrame,
    num_feat: list,
    threshold: float = 0.85,
    plot: bool = True,
) -> list[tuple]:
    corr_matrix = df[num_feat].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    if plot:
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            linewidth=0.5,
        )
        plt.title("Feature Correlation Heatmap")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    high_corr = [
        (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
        for i, j in zip(*np.where((np.abs(corr_matrix) > threshold) & mask))
    ]
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"  correlation_analysis: found {len(high_corr)} pair(s) with |r| > {threshold}.")
    return high_corr


# 3. Loại cột đa cộng tuyến

def drop_high_multicollinearity(
    df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    if cols_to_drop is None:
        cols_to_drop = _HIGH_MULTICOLLINEARITY_COLS

    present = [c for c in cols_to_drop if c in df.columns]
    missing = [c for c in cols_to_drop if c not in df.columns]

    if missing:
        print(f"  drop_high_multicollinearity: columns not found (already removed?): {missing}")
    if present:
        print(f"  drop_high_multicollinearity: dropping {len(present)} column(s): {present}")
        df = df.drop(columns=present)
    return df


# 4. Loại đặc trưng kém quan trọng

def drop_low_importance_features(
    df: pd.DataFrame,
    cols_to_remove: list[str] | None = None,
) -> pd.DataFrame:
    if cols_to_remove is None:
        cols_to_remove = _LOW_IMPORTANCE_COLS

    present = [c for c in cols_to_remove if c in df.columns]
    missing = [c for c in cols_to_remove if c not in df.columns]

    if missing:
        print(f"  drop_low_importance_features: columns not found: {missing}")
    if present:
        print(f"  drop_low_importance_features: dropping {len(present)} column(s): {present}")
        df = df.drop(columns=present)
    return df


# 5. Pipeline tổng hợp

def run_feature_engineering_pipeline(
    df: pd.DataFrame,
    target_col: str = "Attack Type",
    corr_threshold: float = 0.85,
    plot_heatmap: bool = True,
) -> pd.DataFrame:
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    num_feat, cat_feat = get_feature_types(df, target_col=target_col)

    _ = correlation_analysis(df, num_feat, threshold=corr_threshold, plot=plot_heatmap)

    df = drop_high_multicollinearity(df)
    df = drop_low_importance_features(df)

    num_feat_final, _ = get_feature_types(df, target_col=target_col)

    print("=" * 60)
    print(f"  ✓ Feature engineering complete.")
    print(f"    Final shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Numerical features remaining: {len(num_feat_final)}")
    print("=" * 60 + "\n")
    return df
