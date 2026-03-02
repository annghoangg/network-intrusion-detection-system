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
│   └── 01_data_pipeline.ipynb
│
├── cicids2017_cleaned.csv
├── README.md
└── requirements.txt
```

## ML Pipeline Overview

```
Input CSVs  →  data_ingestion  →  eda  →  preprocessing  →  feature_engineering  →  cicids2017_cleaned.csv  →  model training
                                                                                                                     ↓
                                                                                                          (split → train → evaluate)
                                           
                                                                         
```


Run **`notebooks/01_data_pipeline.ipynb`**
This will generate `cicids2017_cleaned.csv`.


**Model Training (script)**

```bash
cd /path/to/Project
python -m src.model_training
```

Or load the splits in any notebook / script:

```python
from src.model_training import load_splits, evaluate_model, compare_models

X_train, X_test, y_train, y_test = load_splits()
```

**Importing Individual Modules**

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
