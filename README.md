# CICIDS2017 — Network Intrusion Detection

A machine-learning pipeline for multi-class network intrusion detection using the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.

## Project Structure

```
Project/
├── Input/                          # Raw CICIDS2017 CSV files (8 traffic captures)
├── splits/                         # Stratified train/test splits (pickle)
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── data_ingestion.py           # Load & merge raw CSVs
│   ├── preprocessing.py            # Data cleaning pipeline
│   ├── feature_engineering.py      # Correlation analysis & feature selection
│   ├── eda.py                      # Statistical EDA helpers & visualisations
│   └── model_training.py           # Train/test split, evaluation, comparison
│
├── notebooks/
│   └── 01_data_pipeline.ipynb      # Orchestrating notebook (runs full pipeline)
│
├── cicids2017_cleaned.csv          # Cleaned & feature-engineered output dataset
├── code.ipynb                      # Original monolithic notebook (kept for reference)
└── README.md
```

## ML Pipeline Overview

```
Input CSVs  →  data_ingestion  →  preprocessing  →  feature_engineering  →  cicids2017_cleaned.csv
                                                                                       ↓
                                                                             model_training
                                                                         (split → train → evaluate)
```

## How to Run

### Step 1 — Data Pipeline

Open and run **`notebooks/01_data_pipeline.ipynb`** sequentially.  
This will generate `cicids2017_cleaned.csv`.

> **Skip if** `cicids2017_cleaned.csv` already exists and is up to date.

### Step 2 — Model Training (script)

```bash
cd /path/to/Project
python -m src.model_training
```

Or load the splits in any notebook / script:

```python
from src.model_training import load_splits, evaluate_model, compare_models

X_train, X_test, y_train, y_test = load_splits()
```

### Step 3 — Importing Individual Modules

```python
from src.data_ingestion     import load_raw_data
from src.preprocessing      import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering_pipeline
from src.eda                 import analyze_feature_importance_kruskal
from src.model_training      import evaluate_model, compare_models
```

## Target Classes (Attack Types)

| Label | Description |
|-------|-------------|
| BENIGN | Normal traffic |
| DoS | DoS Hulk / GoldenEye / slowloris / Slowhttptest |
| DDoS | Distributed Denial of Service |
| PortScan | Port scanning activity |
| Brute Force | FTP-Patator / SSH-Patator |
| Web Attack | Brute Force / XSS / SQL Injection |
| Bot | Botnet activity |
| Heartbleed | Heartbleed vulnerability exploit |

## Key Design Decisions

- **No scaling/resampling in the pipeline notebook** — SMOTE, RobustScaler, etc., are applied *inside* the training pipeline (post-split) to prevent data leakage.
- **Stratified split** — class proportions are preserved in both train and test sets.
- **Feature selection rationale** — low-importance features were identified by combining Random Forest importance scores with Kruskal-Wallis H-statistics, removing features that scored low on both.
