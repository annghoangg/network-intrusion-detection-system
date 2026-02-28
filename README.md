# CICIDS2017 — Network Intrusion Detection

A machine-learning pipeline for multi-class network intrusion detection using the [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) dataset.

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
│   ├── preprocessing.py          
│   ├── feature_engineering.py     
│   ├── eda.py                    
│   └── model_training.py          
│
├── notebooks/
│   └── 01_data_pipeline.ipynb    
│
├── cicids2017_cleaned.csv         
├── code legacy (dont run).ipynb                     
└── README.md
```

## ML Pipeline Overview

```
Input CSVs  →  data_ingestion  →  preprocessing  →  feature_engineering  →  cicids2017_cleaned.csv
                                                                                       ↓
                                                                             model_training
                                                                         (split → train → evaluate)
```

## How to Run (you dumb)

### 1. — Data Pipeline

notebooks/01_data_pipeline.ipynb
generate cleaned data csv

### 2. — Model Training (script)

```bash
cd /path/to/Project
python -m src.model_training
```

Or load the splits in any notebook / script:

```python
from src.model_training import load_splits, evaluate_model, compare_models

X_train, X_test, y_train, y_test = load_splits()
```

### 3. — Importing Individual Modules

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

mẹ cmay béo
