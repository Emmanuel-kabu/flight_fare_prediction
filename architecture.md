# Architecture — Flight Fare Prediction MLOps Platform

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / BROWSER                               │
│                   http://localhost:8501                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Streamlit UI
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      DOCKER COMPOSE STACK                            │
│                                                                      │
│  ┌──────────────┐   REST    ┌──────────────┐                        │
│  │  Streamlit    │ ───────► │   FastAPI     │                        │
│  │  (port 8501)  │ ◄─────── │  (port 8000)  │                        │
│  └──────────────┘   JSON    └──────┬───────┘                        │
│                                     │ loads model                    │
│                                     ▼                                │
│                            ┌──────────────┐                          │
│                            │  artifacts/   │                         │
│                            │ catboost_model│                         │
│                            │ preprocessor  │                         │
│                            └──────┬───────┘                          │
│                                   │ produced by                      │
│  ┌────────────────────────────────┴──────────────────────────────┐   │
│  │                  AIRFLOW  (port 8082)                          │   │
│  │                                                                │   │
│  │  data_extraction → preprocessing → training → evaluation      │   │
│  │                                                    │          │   │
│  │                                          ┌─────────┴────────┐ │   │
│  │                                          ▼                  ▼ │   │
│  │                                   mlflow_logging    evidently_ │   │
│  │                                          │          monitoring │   │
│  │                                          └────────┬─────────┘ │   │
│  │                                                   ▼           │   │
│  │                                            notify_complete    │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   MLflow      │  │   MinIO       │  │  Evidently   │              │
│  │  (port 5000)  │  │ (port 9000/  │  │  (port 8001) │              │
│  │  Tracking +   │  │  9001)       │  │  Monitoring   │              │
│  │  Registry     │  │  S3 Artifact │  │  Dashboard    │              │
│  └──────┬───────┘  │  Store       │  └──────────────┘               │
│         │          └──────────────┘                                  │
│         ▼                                                            │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │ Postgres      │  │ Postgres     │                                 │
│  │ (MLflow DB)   │  │ (Airflow DB) │                                 │
│  │ port 5433     │  │ port 5432    │                                 │
│  └──────────────┘  └──────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

| Component | Description |
|-----------|-------------|
| **Raw Dataset** | `notebooks/data/kaggle/.../Flight_Price_Dataset_of_Bangladesh.csv` — domestic & international flight fares from Bangladesh |
| **Data Extraction** | `data_extraction/data_extraction.py` — loads CSV, saves as Parquet artifact |
| **Artifacts** | `artifacts/` — serialized model (`catboost_model.pkl`), preprocessor (`preprocessor.pkl`), train/test splits |

### 2. ML Pipeline (`model/`)

| Module | Responsibility |
|--------|----------------|
| **`model/training/preprocessing.py`** | `FlightFarePreprocessor` — feature engineering (date decomposition, cyclic encoding, interaction features, frequency encoding), ordinal encoding, temporal train/test split |
| **`model/training/training.py`** | `FlightFareTrainer` — CatBoost regressor training with native categorical support, early stopping, pickle serialization |
| **`model/evaluation/evaluation.py`** | `FlightFareEvaluator` — regression metrics (R², MAE, RMSE, MAPE, MedAE), segment-level breakdowns, diagnostic plots |
| **`model/prediction/prediction.py`** | Inference helper for batch prediction |

### 3. Experiment Tracking (`experiment_tracking/`)

| Module | Responsibility |
|--------|----------------|
| **`feature_store.py`** | `FeatureStore` — logs feature schema, statistics, frequency maps, and data samples to a dedicated MLflow experiment |
| **`model_store.py`** | `ModelStore` — logs hyperparameters, metrics, feature importances, model artifacts to MLflow; registers models in the MLflow Model Registry |

### 4. Performance Monitoring (`performance_tracking/`)

| Module | Responsibility |
|--------|----------------|
| **`data_drift.py`** | `DataDriftMonitor` — Evidently `DataDriftPreset` to detect feature distribution shifts between training (reference) and new data (current) |
| **`concept_drift.py`** | `ConceptDriftMonitor` — Evidently `RegressionPreset` to detect model performance degradation (R², MAE, RMSE changes over time) |

Reports are saved as HTML + JSON in `reports/data_drift/` and `reports/concept_drift/`.

### 5. Orchestration (`orchestration/ML_pipeline.py`)

An **Apache Airflow DAG** (`flight_fare_ml_pipeline`) runs weekly and chains:

```
data_extraction → preprocessing → training → evaluation
                                                   │
                                       ┌───────────┴───────────┐
                                  mlflow_logging       evidently_monitoring
                                       └───────────┬───────────┘
                                              notify_complete
```

Each task is a `PythonOperator`. Artifacts pass between tasks via XCom and the shared filesystem.

### 6. Serving (`deployment/`)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **`serving.py`** | FastAPI + Uvicorn | REST API — `/predict` (POST), `/health` (GET), `/options` (GET) |
| **`streamlit_app.py`** | Streamlit | Web UI — interactive form for flight details, calls FastAPI, displays predicted fare |

### 7. Infrastructure (`docker/`)

All services run via `docker-compose-ml.yml`:

| Service | Image / Build | Ports |
|---------|---------------|-------|
| `fastapi` | `Dockerfile.fastapi` | 8003 → 8000 |
| `streamlit` | `Dockerfile.streamlit` | 8501 → 8501 |
| `mlflow` | `Dockerfile.mlflow` | 5000 → 5000 |
| `evidently` | `Dockerfile.evidently` | 8001 → 8001 |
| `airflow-webserver` | `Dockerfile.airflow` | 8082 → 8080 |
| `airflow-scheduler` | `Dockerfile.airflow` | — |
| `mlflow-db` | `postgres:16-alpine` | 5433 → 5432 |
| `airflow-db` | `postgres:16-alpine` | 5432 → 5432 |
| `minio` | `minio/minio` | 9000, 9001 |

**Networking:** All services join the `mlops` bridge network. Inter-service communication uses Docker DNS (e.g., `http://mlflow:5000`).

**Volumes:**
- `mlflow-db-data` / `airflow-db-data` — Postgres persistence
- `minio-data` — S3 artifact persistence
- `airflow-logs` — Airflow log persistence
- `shared-artifacts` — model artifacts shared between Airflow & FastAPI
- `shared-reports` — Evidently reports shared between Airflow & Evidently UI

---

## Data Flow

```
CSV Dataset
    │
    ▼
┌─────────────────────┐
│  FlightFarePreprocessor │
│  • Date decomposition   │
│  • Cyclic encoding      │
│  • Route / Airline_Class│
│  • Frequency encoding   │
│  • Booking window bins  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐         ┌──────────────┐
│  CatBoost Regressor │ ──────► │  MLflow      │
│  • depth=8          │ metrics │  Tracking +  │
│  • lr=0.02          │ model   │  Registry    │
│  • 2000 iterations  │         └──────────────┘
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
artifacts/    Evidently Reports
    │         (drift analysis)
    ▼
┌───────────┐     ┌───────────┐
│  FastAPI   │ ◄── │ Streamlit │
│  /predict  │     │    UI     │
└───────────┘     └───────────┘
```

## Key Design Decisions

1. **CatBoost with native categoricals** — avoids one-hot explosion; handles high-cardinality features (Route, Airline_Class) efficiently
2. **Temporal train/test split** — prevents data leakage from future flights into training
3. **Frequency encoding (leak-safe)** — fitted only on training data, applied to test/inference
4. **Pickle serialization** — lightweight model persistence; MLflow also stores the model via `mlflow.catboost`
5. **Evidently for monitoring** — separate data drift and concept drift detection with HTML reports
6. **Airflow for orchestration** — weekly retraining pipeline with MLflow logging and Evidently monitoring
7. **Docker Compose** — single-command deployment of the entire MLOps stack
