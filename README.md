# CICIDS2017 — Network Intrusion Detection

A machine-learning pipeline for multi-class network intrusion detection using the [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/) dataset. A DAP391m project - Group 1.

## Members

- Nguyen Hoang An
- Vu Ngoc Hai Dang
- Le Trung Kien
- Le Trung Hieu
- Do Anh Thu

## Project Structure

```
Project/
├── Input/                        
├── models/
│   ├── label_encoder.joblib
│   └── xgboost_best_model.joblib
│
├── splits/                       
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── src/                                                 
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── eda.py
│   ├── preprocessing.py - Data cleaning + correlation analysis and feature selection
│   ├── feature_engineering.py
│   └── model_training.py - Train/test split, evaluation, comparison
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_decision_tree.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_extra_trees.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   ├── 09_model_comparison.ipynb
│   ├── 10_ensemble_model.ipynb
│   └── results/
│
├── cicids2017_cleaned.csv
├── README.md
└── requirements.txt
```
## Installation

To install all the necessary libraries used in this project, run the following command to install them from the provided text file:

```bash
pip install -r requirements.txt
```

## ML Pipeline Overview

```
Input CSVs  →  data_ingestion  →  eda  →  preprocessing  →  feature_engineering  →  cicids2017_cleaned.csv  →  model training
                                                                                                                     ↓
                                                                                                          (split → train → evaluate)
                                           
                                                                         
```


Run **`notebooks/01_data_pipeline.ipynb`**
This will generate `cicids2017_cleaned.csv`.

**Importing Individual Modules**

```python
from src.data_ingestion     import load_raw_data
from src.preprocessing      import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering_pipeline
from src.eda                 import analyze_feature_importance_kruskal
from src.model_training      import evaluate_model, compare_models
```

## Dashboard Demo

To launch the interactive dashboard for testing the trained model:

```bash
streamlit run app/app.py
```

Upload a CSV file containing network flow features, or click **Use Sample Data** to try the demo with pre-loaded test data.

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
