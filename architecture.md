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
| **Artifacts** | `artifacts/` — serialized champion model (`champion_model.pkl`), backward-compatible copy (`catboost_model.pkl`), preprocessor (`preprocessor.pkl`), train/test splits |

### 2. ML Pipeline

| Module | Responsibility |
|--------|----------------|
| **`model/training/preprocessing.py`** | `FlightFarePreprocessor` — feature engineering (date decomposition, cyclic encoding, interaction features, frequency encoding), ordinal encoding, temporal train/test split |
| **`model/training/training.py`** | `FlightFareTrainer` — CatBoost regressor training with native categorical support, early stopping, pickle serialization (legacy single-model interface) |
| **`model/evaluation/evaluation.py`** | `FlightFareEvaluator` — regression metrics (R², MAE, RMSE, MAPE, MedAE), segment-level breakdowns, diagnostic plots |
| **`model/prediction/prediction.py`** | Inference helper for batch prediction |

### 2b. Continuous Training (`continuous_training/`)

| Module | Responsibility |
|--------|----------------|
| **`continuous_training.py`** | `ContinuousTrainer` — trains 4 models in parallel (RandomForest, XGBoost, LightGBM, CatBoost), evaluates all on validation set, selects champion by best R², saves artifacts |
| | `check_retraining_needed()` — analyses drift reports to decide whether retraining is necessary |
| | `ModelResult` — container holding trained model, metrics, feature importances, and training time |
| | `MODEL_CONFIGS` — hyperparameter configurations for all four model types |

**Champion/Challenger Flow:**
```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ RandomForest  │   │    XGBoost    │   │   LightGBM    │   │   CatBoost    │
│  500 trees    │   │  2000 rounds  │   │  2000 rounds  │   │  2000 iters   │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │               │               │               │
        └───────┬───────┴───────┬───────┘
                │               │
          Compare R²      Select Best
                │               │
                ▼               ▼
        ┌───────────────────────┐
        │   Champion Model      │ ──→ artifacts/champion_model.pkl
        │   (best R² wins)      │ ──→ MLflow Model Registry
        └───────────────────────┘ ──→ POST /reload → FastAPI
```

### 3. Experiment Tracking (`experiment_tracking/`)

| Module | Responsibility |
|--------|----------------|
| **`feature_store.py`** | `FeatureStore` — logs feature schema, statistics, frequency maps, and data samples to a dedicated MLflow experiment |
| **`model_store.py`** | `ModelStore` — multi-model logging with flavour-aware MLflow calls (`mlflow.sklearn`, `mlflow.xgboost`, `mlflow.lightgbm`, `mlflow.catboost`); registers models in the MLflow Model Registry; tags each run with `model_type`, `model_family`, `mlflow_flavour`, and `champion` |

**MLflow Tags per Model:**

| Tag | RandomForest | XGBoost | LightGBM | CatBoost |
|-----|-------------|---------|----------|----------|
| `model_type` | RandomForest | XGBoost | LightGBM | CatBoost |
| `model_family` | ensemble_bagging | gradient_boosting | gradient_boosting | gradient_boosting |
| `mlflow_flavour` | sklearn | xgboost | lightgbm | catboost |
| `champion` | true/false | true/false | true/false | true/false |

### 4. Performance Monitoring (`performance_tracking/`)

| Module | Responsibility |
|--------|----------------|
| **`data_drift.py`** | `DataDriftMonitor` — Evidently `DataDriftPreset` to detect feature distribution shifts between training (reference) and new data (current) |
| **`concept_drift.py`** | `ConceptDriftMonitor` — Evidently `RegressionPreset` to detect model performance degradation (R², MAE, RMSE changes over time) |

Reports are saved as HTML + JSON in `reports/data_drift/` and `reports/concept_drift/`.

**Drift thresholds that trigger retraining:**
- Data drift: ≥ 30% of features drifted
- Concept drift: R² degradation > 10%

### 5. Orchestration (`orchestration/`)

#### DAG 1: `flight_fare_ml_pipeline` (Weekly)

An **Apache Airflow DAG** for standard single-model training:

```
slack_started → data_extraction → preprocessing → training → evaluation
                                                                  │
                                                      ┌───────────┴───────────┐
                                                 mlflow_logging       evidently_monitoring
                                                      └───────────┬───────────┘
                                                             notify_complete
```

#### DAG 2: `flight_fare_continuous_training` (Every 3 Days)

Drift-triggered multi-model retraining pipeline with `BranchPythonOperator`:

```
check_drift ─┬─→ notify_retraining_started → extract_data → preprocess
             │         → train_all_models → log_to_mlflow → reload_endpoint
             └─→ skip_retraining (no drift)
```

**Task breakdown:**

| Task | Description |
|------|-------------|
| `check_drift` | Reads drift reports, decides retrain vs. skip via `BranchPythonOperator` |
| `notify_retraining_started` | Sends Slack alert with list of models to train |
| `extract_data` | Loads latest CSV via `FlightFarePreprocessor.load_data()` |
| `preprocess` | Temporal split → `fit_transform` (train) → `transform` (test) |
| `train_all_models` | `ContinuousTrainer.train_all_models()` → champion selection |
| `log_to_mlflow` | Logs all 4 runs with tags, registers champion in Model Registry |
| `reload_endpoint` | `POST /reload` on FastAPI to hot-swap the production model |

Each task is a `PythonOperator`. Artifacts pass between tasks via XCom and the shared filesystem.

#### Slack Notifications (`slack_notifications.py`)

| Function | Triggered When |
|----------|----------------|
| `notify_pipeline_started()` | Weekly DAG begins |
| `notify_task_failure()` | Any task fails (`on_failure_callback`) |
| `notify_pipeline_complete()` | Weekly DAG finishes successfully |
| `notify_retraining_needed()` | Drift check detects retraining is needed |
| `notify_retraining_started()` | Multi-model training is about to begin |
| `notify_retraining_failed()` | A retraining step errors out |
| `notify_retraining_complete()` | Champion selected, includes comparison table |
| `notify_model_promoted()` | Champion model deployed to production endpoint |

### 6. Serving (`deployment/`)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **`serving.py`** | FastAPI + Uvicorn | REST API — `/predict` (POST), `/reload` (POST), `/health` (GET), `/options` (GET) |
| **`streamlit_app.py`** | Streamlit | Web UI — interactive form for flight details, calls FastAPI, displays predicted fare |

**Model Loading:**
- At startup: loads `champion_model.pkl` (falls back to `catboost_model.pkl` for backward compatibility)
- At runtime: `POST /reload` hot-swaps the model without restarting the container
- The `/health` endpoint reports the active model class (e.g., `XGBRegressor`, `CatBoostRegressor`)

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

1. **Multi-model champion/challenger** — trains 4 models every retraining cycle and promotes the best R² to production, ensuring the endpoint always runs the optimal model
2. **CatBoost with native categoricals** — avoids one-hot explosion; handles high-cardinality features (Route, Airline_Class) efficiently
3. **Temporal train/test split** — prevents data leakage from future flights into training
4. **Frequency encoding (leak-safe)** — fitted only on training data, applied to test/inference
5. **Pickle serialization** — lightweight model persistence; MLflow also stores models via flavour-specific APIs (`mlflow.catboost`, `mlflow.xgboost`, `mlflow.lightgbm`, `mlflow.sklearn`)
6. **Evidently for monitoring** — separate data drift and concept drift detection with HTML reports; thresholds trigger automatic retraining
7. **Drift-triggered continuous training** — `BranchPythonOperator` checks drift every 3 days and branches to retrain or skip, avoiding unnecessary compute
8. **Hot-reload endpoint** — `POST /reload` on FastAPI swaps the model in-memory without container restart
9. **Slack notifications at every stage** — drift detected, retraining started/failed/completed, model promoted — full visibility into the ML lifecycle
10. **Airflow for orchestration** — two DAGs (weekly training + drift-triggered continuous training) with shared notification infrastructure
11. **Docker Compose** — single-command deployment of the entire MLOps stack
