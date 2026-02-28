# ✈️ Flight Fare Prediction — MLOps Platform

An end-to-end machine learning system that predicts domestic and international flight fares from Bangladesh, deployed with a full MLOps stack including multi-model continuous training, experiment tracking, model registry, automated drift-triggered retraining, performance monitoring, and Slack notifications.

The platform trains and compares **four models** — RandomForest, XGBoost, LightGBM, and CatBoost — and automatically promotes the best performer (champion) to production.

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) v2+
- (Optional) Python 3.11+ for local development

### Launch the Full Stack

```bash
docker compose -f docker/docker-compose-ml.yml up -d --build
```

This starts all services:

| Service | URL | Description |
|---------|-----|-------------|
| **Nginx LB** | [localhost:8003](http://localhost:8003/lb-health) | Load balancer — round-robin across 3 FastAPI replicas |
| **Streamlit UI** | [localhost:8501](http://localhost:8501) | Interactive fare prediction interface |
| **FastAPI ×3** | *(internal only)* | REST API replicas behind Nginx LB |
| **MLflow** | [localhost:5000](http://localhost:5000) | Experiment tracking & model registry |
| **Airflow** | [localhost:8082](http://localhost:8082) | Pipeline orchestration (user: `airflow` / pass: `airflow`) |
| **Evidently** | [localhost:8001](http://localhost:8001) | Data & concept drift dashboards |
| **MinIO** | [localhost:9001](http://localhost:9001) | S3-compatible artifact store (user: `minio` / pass: `minio123`) |

### Stop Everything

```bash
docker compose -f docker/docker-compose-ml.yml down
```

Add `-v` to also remove persistent volumes (databases, artifact store).

---

## 📁 Project Structure

```
flight_fare_prediction/
├── model/                          # ML pipeline modules
│   ├── training/
│   │   ├── preprocessing.py        # Feature engineering & encoding
│   │   └── training.py             # CatBoost model training (legacy)
│   ├── evaluation/
│   │   └── evaluation.py           # Regression metrics & diagnostics
│   └── prediction/
│       └── prediction.py           # Batch inference helper
├── continuous_training/
│   └── continuous_training.py      # Multi-model trainer (RF, XGB, LGBM, CB)
├── deployment/
│   ├── serving.py                  # FastAPI API (/predict, /reload, /health)
│   └── streamlit_app.py            # Streamlit web UI
├── experiment_tracking/
│   ├── feature_store.py            # MLflow-backed feature registry
│   └── model_store.py              # Multi-model MLflow logging & registry
├── performance_tracking/
│   ├── data_drift.py               # Evidently data drift monitoring
│   └── concept_drift.py            # Evidently concept drift monitoring
├── orchestration/
│   ├── ML_pipeline.py              # Airflow DAG — weekly training pipeline
│   ├── continuous_training_dag.py  # Airflow DAG — drift-triggered retraining
│   └── slack_notifications.py      # Slack webhook notifications
├── data_extraction/
│   └── data_extraction.py          # Data loading utilities
├── notebooks/
│   ├── flight_fare_prototype.ipynb  # EDA & model prototyping notebook
│   └── experiment_with_leaky.ipynb  # Leakage analysis notebook
├── docker/
│   ├── docker-compose-ml.yml       # Full stack compose file
│   ├── nginx/
│   │   └── nginx.conf               # Nginx LB config (round-robin, health checks)
│   ├── Dockerfile.nginx             # Nginx load balancer image
│   ├── Dockerfile.fastapi           # FastAPI image
│   ├── Dockerfile.streamlit         # Streamlit image
│   ├── Dockerfile.mlflow            # MLflow image
│   ├── Dockerfile.airflow           # Airflow image
│   ├── Dockerfile.evidently         # Evidently image
│   └── Dockerfile.training          # Training runner image
├── artifacts/                       # Model & preprocessor pickles
├── reports/                         # Generated drift reports (HTML/JSON)
├── architecture.md                  # System architecture documentation
├── requirements.txt                 # Full Python dependencies
└── requirements.api.txt             # API-only dependencies
```

---

## 🧠 Models

The platform uses a **champion/challenger** approach — four models are trained in parallel, and the best performer on the validation set is promoted to production.

| Model | Family | Key Hyperparameters |
|-------|--------|---------------------|
| **RandomForest** | Ensemble (bagging) | 500 trees, max_depth=20, min_samples_split=5 |
| **XGBoost** | Gradient boosting | 2000 rounds, lr=0.02, depth=8, subsample=0.8 |
| **LightGBM** | Gradient boosting | 2000 rounds, lr=0.02, depth=8, num_leaves=63 |
| **CatBoost** | Gradient boosting | 2000 iterations, lr=0.02, depth=8, early_stop=100 |

- **Target:** Total Fare (BDT) — Bangladeshi Taka
- **Features:** 30+ engineered features including:
  - Date decomposition (month, day-of-week, hour) with cyclic encoding
  - Route & airline-class interaction features
  - Frequency encoding (leak-safe, fitted on training data only)
  - Booking window and haul-type categories
- **Champion Selection:** Best R² on validation set
- **Split:** Temporal 80/20 (walk-forward) to prevent future data leakage

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R² | Coefficient of determination |
| MAE | Mean Absolute Error (BDT) |
| RMSE | Root Mean Squared Error (BDT) |
| MAPE | Mean Absolute Percentage Error (%) |
| MedAE | Median Absolute Error (BDT) |

---

## 🔌 API Reference

### `POST /predict`

Predict the total fare for a flight.

**Request body:**
```json
{
  "airline": "Biman Bangladesh Airlines",
  "source": "DAC",
  "destination": "CXB",
  "departure_datetime": "2026-03-15 08:30:00",
  "duration_hrs": 1.5,
  "stopovers": "Direct",
  "aircraft_type": "Boeing 737",
  "travel_class": "Economy",
  "booking_source": "Online Website",
  "seasonality": "Regular",
  "days_before_departure": 15
}
```

**Response:**
```json
{
  "predicted_fare_bdt": 5250.00,
  "input_summary": {
    "airline": "Biman Bangladesh Airlines",
    "route": "DAC → CXB",
    "class": "Economy",
    "departure": "2026-03-15 08:30:00",
    "days_before": 15
  }
}
```

### `GET /health`

Returns service status, whether the model is loaded, active model class, load timestamp, and `instance_id` identifying which replica is responding.

### `POST /reload`

Hot-swap the production model after retraining. The continuous training DAG broadcasts this to **all 3 replicas** directly (bypassing the LB). For manual reload through the LB:

```bash
curl -X POST http://localhost:8003/reload
```

**Response:**
```json
{
  "status": "reloaded",
  "model_class": "XGBRegressor",
  "loaded_at": "2026-02-28T12:00:00+00:00",
  "instance_id": "1"
}
```

### `GET /lb-health`

Nginx-level health check. Returns `200 OK` if the load balancer itself is running (does **not** probe backend replicas).

### `GET /options`

Returns all valid dropdown options (airlines, airports, classes, etc.).

---

## ⚙️ MLOps Pipelines

### Weekly Training DAG (`flight_fare_ml_pipeline`)

Runs **weekly** for standard single-model (CatBoost) training:

```
slack_started → data_extraction → preprocessing → training → evaluation
                                                                  │
                                                      ┌───────────┴───────────┐
                                                 mlflow_logging       evidently_monitoring
                                                      └───────────┬───────────┘
                                                             notify_complete
```

### Continuous Training DAG (`flight_fare_continuous_training`)

Runs **every 3 days** with drift-triggered multi-model retraining:

```
check_drift ─┬─→ notify_retraining_started → extract_data → preprocess
             │         → train_all_models → log_to_mlflow → reload_endpoint
             └─→ skip_retraining
```

1. **Drift Check** — analyses data drift reports; branches to retrain or skip
2. **Data Extraction** — loads latest CSV data
3. **Preprocessing** — temporal split, feature engineering, frequency encoding
4. **Train All Models** — trains RandomForest, XGBoost, LightGBM, CatBoost in sequence
5. **Champion Selection** — picks the model with the best R² on the validation set
6. **MLflow Logging** — logs all 4 runs with tags (`model_type`, `model_family`, `champion`, `mlflow_flavour`)
7. **Model Registry** — registers and promotes the champion to Production
8. **Endpoint Reload** — broadcasts `POST /reload` to all 3 FastAPI replicas behind the load balancer
9. **Slack Notifications** — at every stage (see below)

### 🔔 Slack Notifications

The pipeline sends structured Slack messages to the `#ml_pipeline_report` channel at every key event:

| Event | Emoji | When |
|-------|-------|------|
| Pipeline started | 🚀 | Weekly DAG kicks off |
| Retraining needed | ⚠️ | Drift detected above threshold |
| Retraining started | 🔄 | Multi-model training begins |
| Retraining failed | ❌ | Any step throws an error |
| Retraining complete | 🏆 | Champion selected, includes comparison table |
| Model promoted | 🚀 | Champion deployed to production endpoint |
| Task failure | ❌ | Any task in either DAG fails |

---

## 📊 Monitoring & Drift Detection

### Data Drift

Detects feature distribution shifts between training data and new production data using Evidently's `DataDriftPreset`. Reports are saved to `reports/data_drift/`.

- **Threshold:** If ≥ 30% of features drift, retraining is triggered automatically.

### Concept Drift

Detects model performance degradation over time using Evidently's `RegressionPreset`. Compares R², MAE, RMSE, and error distributions between reference and current windows. Reports are saved to `reports/concept_drift/`.

- **Threshold:** If R² drops by more than 10%, the model is flagged as degraded.

### Continuous Training Trigger

The `flight_fare_continuous_training` DAG checks drift reports every 3 days. When drift exceeds thresholds, it automatically:
1. Sends a Slack alert about the detected drift
2. Retrains all 4 models on fresh data
3. Selects and deploys the new champion

---

## 🛠️ Local Development

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn deployment.serving:app --reload --port 8000

# Run Streamlit locally
streamlit run deployment/streamlit_app.py
```

---

## 📄 License

This project is for educational and demonstration purposes.
