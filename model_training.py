import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# 1. DATA LOADING & STRATIFIED SPLIT

TARGET_COL = "Attack Type"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = os.path.join(os.path.dirname(__file__), "cicids2017_cleaned.csv")
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "splits")


def load_and_split(
    data_path: str = DATA_PATH,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save: bool = True,
) -> tuple:
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  → Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── Verify stratification ──
    print("Class distribution (%):")
    dist = pd.DataFrame({
        "Full":   y.value_counts(normalize=True).mul(100).round(2),
        "Train":  y_train.value_counts(normalize=True).mul(100).round(2),
        "Test":   y_test.value_counts(normalize=True).mul(100).round(2),
    })
    print(dist.to_string())
    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}\n")

    if save:
        os.makedirs(SPLIT_DIR, exist_ok=True)
        for name, obj in [
            ("X_train", X_train),
            ("X_test", X_test),
            ("y_train", y_train),
            ("y_test", y_test),
        ]:
            path = os.path.join(SPLIT_DIR, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        print(f"Splits saved to {SPLIT_DIR}/")

    return X_train, X_test, y_train, y_test


def load_splits(split_dir: str = SPLIT_DIR) -> tuple:
    """Load previously saved train/test splits from pickle files."""
    result = []
    for name in ("X_train", "X_test", "y_train", "y_test"):
        path = os.path.join(split_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            result.append(pickle.load(f))
    print("Loaded splits from disk.")
    return tuple(result)


# 2. EVALUATION FRAMEWORK

def evaluate_model(
    y_true,
    y_pred,
    model_name: str,
    y_pred_proba=None,
    labels=None,
    print_report: bool = True,
) -> dict:
    if labels is None:
        labels = sorted(y_true.unique()) if hasattr(y_true, "unique") else sorted(set(y_true))

    acc    = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_w  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_m  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa  = cohen_kappa_score(y_true, y_pred)
    mcc    = matthews_corrcoef(y_true, y_pred)

    results = {
        "Model":              model_name,
        "Accuracy":           round(acc, 4),
        "Precision (weighted)": round(prec_w, 4),
        "Precision (macro)":  round(prec_m, 4),
        "Recall (weighted)":  round(rec_w, 4),
        "Recall (macro)":     round(rec_m, 4),
        "F1 (weighted)":      round(f1_w, 4),
        "F1 (macro)":         round(f1_m, 4),
        "Cohen Kappa":        round(kappa, 4),
        "MCC":                round(mcc, 4),
    }

    # ROC-AUC (one-vs-rest) — needs probability estimates
    if y_pred_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=labels)
            roc_w = roc_auc_score(y_bin, y_pred_proba, average="weighted", multi_class="ovr")
            roc_m = roc_auc_score(y_bin, y_pred_proba, average="macro", multi_class="ovr")
            results["ROC-AUC (weighted)"] = round(roc_w, 4)
            results["ROC-AUC (macro)"]    = round(roc_m, 4)
        except ValueError as e:
            print(f"  ⚠ ROC-AUC skipped: {e}")

    if print_report:
        print(f"\n{'='*60}")
        print(f"  {model_name} — Evaluation Results")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=[str(l) for l in labels], zero_division=0))
        for k, v in results.items():
            if k != "Model":
                print(f"  {k:.<30s} {v}")
        print()

    return results


# 3. MODEL COMPARISON HELPER

def compare_models(results_list: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
    return df


# 4. CONFUSION MATRIX VISUALIZATION

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    model_name: str = "Model",
    normalize: bool = True,
    figsize: tuple = (10, 8),
    save_path: str | None = None,
):
    if labels is None:
        labels = sorted(y_true.unique()) if hasattr(y_true, "unique") else sorted(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        fmt = ".1f"
        title_suffix = " (Normalized %)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}{title_suffix}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to {save_path}")

    plt.show()


# 5. MAIN — run when script is executed directly

if __name__ == "__main__":
    # ── Step 1: Load & split ──
    X_train, X_test, y_train, y_test = load_and_split()

    # ── Step 2: Quick demo — Decision Tree baseline ──
    from sklearn.tree import DecisionTreeClassifier

    print("\n─── Training Decision Tree (baseline demo) ───")
    t0 = time.time()
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=20)
    dt.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)

    dt_results = evaluate_model(
        y_test, y_pred,
        model_name="Decision Tree (depth=20)",
        y_pred_proba=y_proba,
        labels=dt.classes_.tolist(),
    )

    plot_confusion_matrix(y_test, y_pred, labels=dt.classes_.tolist(), model_name="Decision Tree")

    # ── Step 3: Show comparison table (single model for now) ──
    comparison = compare_models([dt_results])
    print("\n─── Model Comparison ───")
    print(comparison.to_string(index=False))
