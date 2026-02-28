import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


# 1. Phân tích ngoại lệ

def calculate_outlier_percentage(df: pd.DataFrame) -> dict:
    outlier_pct: dict = {}
    quartiles = df.quantile([0.25, 0.75])
    total_rows = len(df)

    for col in df.columns:
        q1 = quartiles.loc[0.25, col]
        q3 = quartiles.loc[0.75, col]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)
        outlier_pct[col] = is_outlier.sum() / total_rows * 100

    return outlier_pct


# 2. Kiểm định tính đồng nhất phương sai (Levene)

def analyze_variance_homogeneity(
    df: pd.DataFrame,
    numerical_features: list,
    target_col: str = "Attack Type",
) -> dict:
    results: dict = {}
    grouped = df.groupby(target_col)

    for feat in numerical_features:
        groups = [
            group[feat].dropna().values
            for _, group in grouped
            if len(group[feat].dropna()) > 0 and np.var(group[feat].dropna()) > 0
        ]

        if len(groups) < 2:
            print(f"  ⚠ Skipping Levene's test for '{feat}' — insufficient valid groups.")
            continue

        stat, p_value = stats.levene(*groups)
        results[feat] = {"Statistic": stat, "p-value": p_value}

    sig = sum(1 for r in results.values() if r["p-value"] <= 0.05)
    print(f"  Levene's test: {sig}/{len(results)} features show significant variance differences (p ≤ 0.05).")
    return results


# 3. Mức độ quan trọng đặc trưng — Kruskal-Wallis

def analyze_feature_importance_kruskal(
    df: pd.DataFrame,
    num_feat: list,
    target_col: str = "Attack Type",
) -> pd.DataFrame:
    h_scores: dict = {}

    for feature in num_feat:
        groups = [
            group[feature].dropna().values
            for _, group in df.groupby(target_col)
        ]
        h_stat, p_val = stats.kruskal(*groups)
        h_scores[feature] = {"H-statistic": h_stat, "p-value": p_val}

    h_df = (
        pd.DataFrame.from_dict(h_scores, orient="index")
        .reset_index()
        .rename(columns={"index": "Feature"})
        .sort_values("H-statistic", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(18, 10))
    plt.bar(range(len(h_df)), h_df["H-statistic"], color="skyblue")
    plt.xticks(range(len(h_df)), h_df["Feature"], rotation=90)
    plt.title("Feature Importance — Kruskal-Wallis H-statistic")
    plt.xlabel("Features")
    plt.ylabel("H-statistic")
    plt.tight_layout()
    plt.show()

    return h_df


# 4. Mức độ quan trọng đặc trưng — Random Forest

def analyze_feature_importance_rf(
    df: pd.DataFrame,
    num_feat: list,
    target_col: str = "Attack Type",
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_split: int = 5,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 3,
) -> tuple:
    X = df[num_feat]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv_folds, n_jobs=-1)
    print(f"  Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    y_pred = rf.predict(X_test)
    rf_labels = rf.classes_
    cm = confusion_matrix(y_test, y_pred)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=rf_labels))

    importance_df = (
        pd.DataFrame({"Feature": num_feat, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(18, 12))
    plt.bar(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.title("Feature Importance — Random Forest")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return importance_df, cm, rf_labels, cv_scores


# 5. Biểu đồ kết hợp RF + Kruskal-Wallis

def plot_feature_importance_combined(
    comp_tb_sorted: pd.DataFrame,
    h_threshold: float = 100_000,
) -> None:
    colors_cmap = sns.color_palette("coolwarm", as_cmap=True)

    fig, ax1 = plt.subplots(figsize=(25, 10))
    ax2 = ax1.twinx()

    p_min = comp_tb_sorted["p-value"].min()
    p_norm = (comp_tb_sorted["p-value"] - p_min) / (0.1 - p_min + 1e-9)

    ax1.bar(
        comp_tb_sorted["Feature"],
        comp_tb_sorted["Importance"],
        alpha=0.6,
        color=[colors_cmap(v) for v in p_norm],
        edgecolor="black",
    )

    ax2.plot(
        comp_tb_sorted["Feature"],
        comp_tb_sorted["H-statistic"],
        color="black",
        linewidth=2,
        marker="o",
        label="H-statistic",
    )
    ax2.axhline(y=h_threshold, color="red", linestyle="--",
                label=f"H-statistic Threshold ({h_threshold:,})")
    ax2.legend()

    ax1.set_ylabel("Feature Importance", fontsize=12)
    ax1.set_xticks(range(len(comp_tb_sorted)))
    ax1.set_xticklabels(comp_tb_sorted["Feature"], rotation=90, ha="center", fontsize=10)

    plt.title(
        "Feature Importance (Random Forest) with H-statistics and p-values",
        fontsize=14,
        pad=20,
    )

    sm = plt.cm.ScalarMappable(
        cmap=colors_cmap,
        norm=plt.Normalize(vmin=p_min, vmax=0.1),
    )
    cbar = plt.colorbar(sm, ax=ax1, orientation="vertical")
    cbar.set_label("p-value", fontsize=10)

    for ax in [ax1, ax2]:
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.show()
