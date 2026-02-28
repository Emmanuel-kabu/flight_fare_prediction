"""
Airflow DAG — Flight Fare Prediction ML Pipeline
=================================================

Orchestrates the full ML lifecycle:

    data_extraction  →  preprocessing  →  training  →  evaluation
                                                           │
                                              ┌────────────┴────────────┐
                                         mlflow_logging         evidently_monitoring
                                              │                         │
                                              └────────────┬────────────┘
                                                     notify_complete

Each task runs as a PythonOperator inside the Airflow worker so that
heavy artefacts (DataFrames, models) can be passed via XCom or the
shared filesystem without needing Docker-in-Docker.
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from slack_notifications import (
    notify_pipeline_started,
    notify_task_failure,
    notify_pipeline_complete,
)

# ── Project root inside the container ────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = (
    PROJECT_ROOT / "notebooks" / "data" / "kaggle"
    / "flight-price-dataset-of-bangladesh"
    / "Flight_Price_Dataset_of_Bangladesh.csv"
)

# ── MLflow tracking URI (points to the MLflow container) ─────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ── Default DAG arguments ───────────────────────────────────────────────────
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": notify_task_failure,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Task callables
# ═══════════════════════════════════════════════════════════════════════════

def task_data_extraction(**context):
    """Load the raw CSV and persist it as a parquet artefact."""
    from model.training.preprocessing import FlightFarePreprocessor

    df = FlightFarePreprocessor.load_data(CSV_PATH)
    out_path = ARTIFACTS_DIR / "raw_data.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[DataExtraction] Saved {len(df):,} rows → {out_path}")
    return str(out_path)


def task_preprocessing(**context):
    """Split data, fit preprocessor, and persist X/y artefacts."""
    import pandas as pd
    from model.training.preprocessing import FlightFarePreprocessor

    raw_path = context["ti"].xcom_pull(task_ids="data_extraction")
    df = pd.read_parquet(raw_path)

    preprocessor = FlightFarePreprocessor()
    train_df, test_df = preprocessor.split_data(df, test_size=0.2, temporal=True)

    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)

    # Persist splits
    X_train.to_parquet(ARTIFACTS_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(ARTIFACTS_DIR / "X_test.parquet", index=False)
    y_train.to_frame("target").to_parquet(ARTIFACTS_DIR / "y_train.parquet", index=False)
    y_test.to_frame("target").to_parquet(ARTIFACTS_DIR / "y_test.parquet", index=False)

    # Persist preprocessor
    prep_path = ARTIFACTS_DIR / "preprocessor.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"[Preprocessing] Train={X_train.shape}  Test={X_test.shape}")
    return {
        "X_train": str(ARTIFACTS_DIR / "X_train.parquet"),
        "X_test": str(ARTIFACTS_DIR / "X_test.parquet"),
        "y_train": str(ARTIFACTS_DIR / "y_train.parquet"),
        "y_test": str(ARTIFACTS_DIR / "y_test.parquet"),
        "preprocessor": str(prep_path),
    }


def task_training(**context):
    """Train CatBoost and persist the model."""
    import pandas as pd
    from model.training.training import FlightFareTrainer

    paths = context["ti"].xcom_pull(task_ids="preprocessing")
    X_train = pd.read_parquet(paths["X_train"])
    y_train = pd.read_parquet(paths["y_train"])["target"]
    X_test = pd.read_parquet(paths["X_test"])
    y_test = pd.read_parquet(paths["y_test"])["target"]

    with open(paths["preprocessor"], "rb") as f:
        preprocessor = pickle.load(f)

    trainer = FlightFareTrainer(
        cat_feature_indices=preprocessor.cat_feature_indices,
    )
    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

    model_path = ARTIFACTS_DIR / "catboost_model.pkl"
    trainer.save_model(model_path)

    print(f"[Training] Model saved → {model_path}")
    return str(model_path)


def task_evaluation(**context):
    """Compute evaluation metrics on the test set."""
    import pandas as pd
    from model.evaluation.evaluation import FlightFareEvaluator

    paths = context["ti"].xcom_pull(task_ids="preprocessing")
    model_path = context["ti"].xcom_pull(task_ids="training")

    X_test = pd.read_parquet(paths["X_test"])
    y_test = pd.read_parquet(paths["y_test"])["target"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    evaluator = FlightFareEvaluator()
    metrics = evaluator.compute_metrics(y_test, y_pred)
    evaluator.print_metrics(metrics, title="CatBoost — Airflow Pipeline")

    # Persist predictions for downstream tasks
    import numpy as np
    pred_path = ARTIFACTS_DIR / "y_pred.parquet"
    pd.DataFrame({"prediction": np.asarray(y_pred)}).to_parquet(pred_path, index=False)

    return {
        "metrics": metrics,
        "y_pred_path": str(pred_path),
    }


def task_mlflow_logging(**context):
    """Log features, model, and metrics to MLflow; register the model."""
    import pandas as pd
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    paths = context["ti"].xcom_pull(task_ids="preprocessing")
    model_path = context["ti"].xcom_pull(task_ids="training")
    eval_result = context["ti"].xcom_pull(task_ids="evaluation")
    metrics = eval_result["metrics"]

    X_train = pd.read_parquet(paths["X_train"])
    X_test = pd.read_parquet(paths["X_test"])

    with open(paths["preprocessor"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Feature importances
    fi = pd.Series(
        model.get_feature_importance(), index=model.feature_names_
    ).sort_values(ascending=False)

    # ── Feature Store ──
    from experiment_tracking.feature_store import FeatureStore

    fs = FeatureStore(tracking_uri=MLFLOW_TRACKING_URI)
    fs_run_id = fs.log_feature_profile(
        X_train, X_test,
        preprocessor=preprocessor,
        run_name="feature-profile-catboost",
    )

    # ── Model Store ──
    from experiment_tracking.model_store import ModelStore

    ms = ModelStore(tracking_uri=MLFLOW_TRACKING_URI)

    # Retrieve trainer params from the model
    params = {
        "iterations": model.tree_count_,
        "learning_rate": model.get_all_params().get("learning_rate", 0.02),
        "depth": model.get_all_params().get("depth", 8),
        "loss_function": model.get_all_params().get("loss_function", "RMSE"),
    }

    ms_run_id = ms.log_training_run(
        model=model,
        params=params,
        metrics=metrics,
        feature_importances=fi,
        X_train=X_train,
        preprocessor=preprocessor,
        run_name="catboost-airflow-run",
        tags={"feature_store_run": fs_run_id, "orchestrator": "airflow"},
    )
    version = ms.register_model(ms_run_id)

    print(f"[MLflow] feature_store={fs_run_id}  model_store={ms_run_id}  version={version}")
    return {"fs_run_id": fs_run_id, "ms_run_id": ms_run_id, "model_version": version}


def task_evidently_monitoring(**context):
    """Generate Evidently data-drift and concept-drift reports."""
    import numpy as np
    import pandas as pd

    paths = context["ti"].xcom_pull(task_ids="preprocessing")
    model_path = context["ti"].xcom_pull(task_ids="training")
    eval_result = context["ti"].xcom_pull(task_ids="evaluation")

    X_train = pd.read_parquet(paths["X_train"])
    X_test = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"])["target"]
    y_test = pd.read_parquet(paths["y_test"])["target"]
    y_pred = pd.read_parquet(eval_result["y_pred_path"])["prediction"].values

    with open(paths["preprocessor"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    reports_dir = PROJECT_ROOT / "reports"

    # ── Data Drift ──
    from performance_tracking.data_drift import DataDriftMonitor

    drift_monitor = DataDriftMonitor(
        reference_data=X_train,
        cat_features=preprocessor.cat_feature_names,
        num_features=preprocessor.num_feature_names,
        reports_dir=str(reports_dir / "data_drift"),
    )
    drift_snap = drift_monitor.run(current_data=X_test)
    drift_monitor.save_html(drift_snap, "data_drift_report.html")
    drift_monitor.save_json(drift_snap, "data_drift_report.json")

    # ── Concept Drift ──
    from performance_tracking.concept_drift import ConceptDriftMonitor

    ref_pred_train = model.predict(X_train)
    ref_df = ConceptDriftMonitor.build_monitor_dataframe(X_train, y_train, ref_pred_train)
    cur_df = ConceptDriftMonitor.build_monitor_dataframe(X_test, y_test, y_pred)

    concept_monitor = ConceptDriftMonitor(
        reference_data=ref_df,
        reports_dir=str(reports_dir / "concept_drift"),
    )
    concept_snap = concept_monitor.run(current_data=cur_df)
    concept_monitor.save_html(concept_snap, "concept_drift_report.html")
    concept_monitor.save_json(concept_snap, "concept_drift_report.json")

    print("[Evidently] Data drift + concept drift reports saved.")
    return {"data_drift": "ok", "concept_drift": "ok"}


def task_notify_complete(**context):
    """Final task — print summary and send Slack notification."""
    eval_result = context["ti"].xcom_pull(task_ids="evaluation")
    mlflow_result = context["ti"].xcom_pull(task_ids="mlflow_logging")
    evidently_result = context["ti"].xcom_pull(task_ids="evidently_monitoring")
    metrics = eval_result["metrics"]

    print("\n" + "=" * 60)
    print("  ✅  FLIGHT FARE ML PIPELINE — COMPLETE")
    print("=" * 60)
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:,.0f} BDT")
    print(f"  RMSE = {metrics['RMSE']:,.0f} BDT")
    if mlflow_result:
        print(f"  MLflow model version = {mlflow_result.get('model_version', 'N/A')}")
    if evidently_result:
        print(f"  Evidently reports    = ✅")
    print("=" * 60)

    # Send Slack completion notification
    notify_pipeline_complete(context)


# ═══════════════════════════════════════════════════════════════════════════
#  DAG definition
# ═══════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="flight_fare_ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: data → train → evaluate → log → monitor",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "flight-fare", "catboost"],
    max_active_runs=1,
) as dag:

    t_slack_start = PythonOperator(
        task_id="slack_pipeline_started",
        python_callable=notify_pipeline_started,
    )

    t_extract = PythonOperator(
        task_id="data_extraction",
        python_callable=task_data_extraction,
    )

    t_preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
    )

    t_train = PythonOperator(
        task_id="training",
        python_callable=task_training,
    )

    t_evaluate = PythonOperator(
        task_id="evaluation",
        python_callable=task_evaluation,
    )

    t_mlflow = PythonOperator(
        task_id="mlflow_logging",
        python_callable=task_mlflow_logging,
    )

    t_evidently = PythonOperator(
        task_id="evidently_monitoring",
        python_callable=task_evidently_monitoring,
    )

    t_notify = PythonOperator(
        task_id="notify_complete",
        python_callable=task_notify_complete,
    )

    # ── Task dependencies (pipeline graph) ──
    t_slack_start >> t_extract >> t_preprocess >> t_train >> t_evaluate
    t_evaluate >> [t_mlflow, t_evidently] >> t_notify
