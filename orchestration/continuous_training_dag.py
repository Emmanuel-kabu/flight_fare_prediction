"""
Airflow DAG — Continuous Training Pipeline for Flight Fare Prediction.

This DAG automates the full retraining lifecycle:
  1. Check drift (data + concept) → decide if retraining is needed
  2. Notify Slack when retraining is triggered
  3. Extract fresh data
  4. Preprocess features
  5. Train all 4 models (RandomForest, XGBoost, LightGBM, CatBoost)
  6. Select champion by best R² on validation set
  7. Log all runs to MLflow with proper tags
  8. Save champion artifacts & reload prediction endpoint
  9. Notify Slack at every stage (needed / started / failed / completed / promoted)

Schedule: Every 3 days — but the retraining step only fires when drift is detected.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

# ── Ensure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = "/opt/airflow"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
#  Default args
# ═══════════════════════════════════════════════════════════════════════════

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Task callables
# ═══════════════════════════════════════════════════════════════════════════

def task_check_drift(**context):
    """
    Run data-drift and concept-drift monitors.
    Pushes drift results to XCom and decides whether to retrain.
    """
    from orchestration.slack_notifications import notify_retraining_needed

    drift_report_path = os.path.join(
        PROJECT_ROOT, "reports", "data_drift", "data_drift_report.json"
    )

    # Use the ContinuousTrainer helper to check drift
    from continuous_training.continuous_training import check_retraining_needed

    needs_retraining, reasons = check_retraining_needed(
        drift_report_path=drift_report_path,
        r2_threshold=0.85,
        drift_share_threshold=0.3,
    )

    context["ti"].xcom_push(key="needs_retraining", value=needs_retraining)
    context["ti"].xcom_push(key="drift_reasons", value=reasons)

    if needs_retraining:
        notify_retraining_needed(reasons, context)
        return "notify_retraining_started"
    else:
        return "skip_retraining"


def task_skip_retraining(**context):
    """No drift detected — skip retraining."""
    print("No drift detected. Skipping retraining.")
    return {"status": "skipped", "reason": "no_drift"}


def task_notify_retraining_started(**context):
    """Send Slack notification that retraining is starting."""
    from orchestration.slack_notifications import notify_retraining_started

    models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    notify_retraining_started(models, context)
    return {"status": "notified", "models": models}


def task_extract_data(**context):
    """Extract fresh training data from the source CSV."""
    from model.training.preprocessing import FlightFarePreprocessor

    csv_path = os.path.join(
        PROJECT_ROOT,
        "notebooks", "data", "kaggle",
        "flight-price-dataset-of-bangladesh",
        "Flight_Price_Dataset_of_Bangladesh.csv",
    )

    df = FlightFarePreprocessor.load_data(csv_path)
    print(f"[ContinuousTraining] Extracted {len(df)} rows")

    # Save to staging location for downstream tasks
    staging_path = os.path.join(PROJECT_ROOT, "artifacts", "staging_data.pkl")
    os.makedirs(os.path.dirname(staging_path), exist_ok=True)
    with open(staging_path, "wb") as f:
        pickle.dump(df, f)

    context["ti"].xcom_push(key="data_path", value=staging_path)
    context["ti"].xcom_push(key="n_rows", value=len(df))
    return {"data_path": staging_path, "n_rows": len(df)}


def task_preprocess(**context):
    """Run preprocessing and feature engineering."""
    from model.training.preprocessing import FlightFarePreprocessor

    ti = context["ti"]
    data_path = ti.xcom_pull(task_ids="extract_data", key="data_path")

    with open(data_path, "rb") as f:
        df = pickle.load(f)

    preprocessor = FlightFarePreprocessor()

    # 1. Temporal train/test split on raw data
    train_df, test_df = preprocessor.split_data(df, test_size=0.2, temporal=True)

    # 2. Fit on training data, then transform both
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)

    print(
        f"[ContinuousTraining] Preprocessed — "
        f"train={len(X_train)}, val={len(X_test)}, "
        f"features={X_train.shape[1]}"
    )

    # Persist for downstream tasks
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    splits_path = os.path.join(artifacts_dir, "staging_splits.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "preprocessor": preprocessor,
            },
            f,
        )

    context["ti"].xcom_push(key="splits_path", value=splits_path)
    context["ti"].xcom_push(key="n_features", value=X_train.shape[1])
    context["ti"].xcom_push(
        key="cat_feature_indices",
        value=preprocessor.cat_feature_indices,
    )
    context["ti"].xcom_push(
        key="cat_feature_names",
        value=preprocessor.cat_feature_names,
    )
    return {"splits_path": splits_path, "n_features": X_train.shape[1]}


def task_train_all_models(**context):
    """Train all 4 models and select the champion."""
    from continuous_training.continuous_training import ContinuousTrainer
    from orchestration.slack_notifications import (
        notify_retraining_complete,
        notify_retraining_failed,
    )

    ti = context["ti"]
    splits_path = ti.xcom_pull(task_ids="preprocess", key="splits_path")
    cat_indices = ti.xcom_pull(task_ids="preprocess", key="cat_feature_indices")
    cat_names = ti.xcom_pull(task_ids="preprocess", key="cat_feature_names")

    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    preprocessor = splits["preprocessor"]

    trainer = ContinuousTrainer(
        artifacts_dir=os.path.join(PROJECT_ROOT, "artifacts")
    )

    try:
        # Train all models
        results = trainer.train_all_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            cat_feature_indices=cat_indices,
            cat_feature_names=cat_names,
        )

        if not results:
            raise RuntimeError("All models failed to train!")

        # Select champion
        champion = trainer.select_champion(results)

        # Build comparison table
        comparison_df = trainer.build_comparison_table(results)
        comparison_str = comparison_df.to_string(index=False)

        # Save champion artifacts
        saved_paths = trainer.save_champion(champion, preprocessor)

        # Notify retraining complete
        notify_retraining_complete(
            champion_name=champion.model_name,
            comparison_table=comparison_str,
            champion_metrics=champion.metrics,
            context=context,
        )

        # Push results for downstream tasks
        results_summary = []
        for r in results:
            results_summary.append({
                "model_name": r.model_name,
                "metrics": r.metrics,
                "params": r.params,
                "training_time_seconds": r.training_time_seconds,
            })

        ti.xcom_push(key="champion_name", value=champion.model_name)
        ti.xcom_push(key="champion_metrics", value=champion.metrics)
        ti.xcom_push(key="comparison_table", value=comparison_str)
        ti.xcom_push(key="results_summary", value=results_summary)
        ti.xcom_push(key="n_models_trained", value=len(results))
        ti.xcom_push(
            key="saved_paths",
            value={k: str(v) for k, v in saved_paths.items()},
        )

        return {
            "champion": champion.model_name,
            "n_models": len(results),
            "champion_R2": champion.metrics["R2"],
        }

    except Exception as e:
        notify_retraining_failed(str(e), "train_all_models", context)
        raise


def task_log_to_mlflow(**context):
    """Log all training results to MLflow with correct tags."""
    from experiment_tracking.model_store import ModelStore

    ti = context["ti"]
    splits_path = ti.xcom_pull(task_ids="preprocess", key="splits_path")
    champion_name = ti.xcom_pull(task_ids="train_all_models", key="champion_name")
    results_summary = ti.xcom_pull(task_ids="train_all_models", key="results_summary")

    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    preprocessor = splits["preprocessor"]
    X_train = splits["X_train"]

    # Recreate model results for MLflow logging
    # We need to load the full model objects from the artifacts
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")

    ms = ModelStore()

    # Log individual results from summary data
    champion_run_id = None
    run_ids = {}

    for result in results_summary:
        model_name = result["model_name"]
        is_champ = model_name == champion_name

        # For the champion, load the saved model to log it
        if is_champ:
            model_path = os.path.join(artifacts_dir, "champion_model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            run_id = ms.log_training_run(
                model=model,
                params=result["params"],
                metrics=result["metrics"],
                model_type=model_name,
                X_train=X_train,
                preprocessor=preprocessor,
                run_name=f"retrain-{model_name.lower()}-champion",
                tags={
                    "training_time_seconds": str(result["training_time_seconds"]),
                    "retrain": "true",
                    "pipeline": "continuous_training",
                },
                is_champion=True,
            )
            champion_run_id = run_id
        else:
            # For non-champion models, log metrics and params only
            # (we don't persist non-champion model objects to save space)
            import mlflow
            mlflow.set_experiment("flight-fare-model-store")
            with mlflow.start_run(run_name=f"retrain-{model_name.lower()}"):
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("model_family",
                               "gradient_boosting" if model_name != "RandomForest" else "ensemble_bagging")
                mlflow.set_tag("champion", "false")
                mlflow.set_tag("retrain", "true")
                mlflow.set_tag("pipeline", "continuous_training")
                for k, v in result["params"].items():
                    mlflow.log_param(k, v)
                for k, v in result["metrics"].items():
                    mlflow.log_metric(k, v)
                mlflow.log_metric(
                    "training_time_seconds", result["training_time_seconds"]
                )
                run_id = mlflow.active_run().info.run_id

        run_ids[model_name] = run_id

    # Register the champion model in the registry
    if champion_run_id:
        registry_name = f"flight-fare-{champion_name.lower()}"
        try:
            version = ms.register_model(champion_run_id, model_name=registry_name)
            ms.transition_model_stage(version, "Production", model_name=registry_name)
        except Exception as e:
            print(f"[ContinuousTraining] Registry error (non-fatal): {e}")
            version = "N/A"

        ti.xcom_push(key="champion_run_id", value=champion_run_id)
        ti.xcom_push(key="model_version", value=version)
        ti.xcom_push(key="run_ids", value=run_ids)
        ti.xcom_push(key="registry_name", value=registry_name)

    return {
        "champion_run_id": champion_run_id,
        "run_ids": run_ids,
    }


def task_reload_endpoint(**context):
    """
    Broadcast reload to ALL FastAPI replicas behind the load balancer.
    Each replica loads model artifacts from the shared volume independently.
    """
    import requests
    from orchestration.slack_notifications import (
        notify_model_promoted,
        notify_retraining_failed,
    )

    ti = context["ti"]
    champion_name = ti.xcom_pull(task_ids="train_all_models", key="champion_name")
    model_version = ti.xcom_pull(task_ids="log_to_mlflow", key="model_version") or "N/A"

    # Get replica URLs — broadcast reload to each one
    replicas_str = os.getenv(
        "FASTAPI_REPLICAS",
        "http://fastapi-1:8000,http://fastapi-2:8000,http://fastapi-3:8000",
    )
    replica_urls = [u.strip() for u in replicas_str.split(",") if u.strip()]
    lb_url = os.getenv("FASTAPI_BASE_URL", "http://nginx-lb:80")

    reloaded = []
    failed = []

    for url in replica_urls:
        try:
            resp = requests.post(f"{url}/reload", timeout=30)
            if resp.status_code == 200:
                reloaded.append(url)
                print(f"[ContinuousTraining] Reloaded: {url}")
            else:
                failed.append(f"{url} → {resp.status_code}")
                print(f"[ContinuousTraining] Reload warning: {url} → {resp.status_code}")
        except Exception as e:
            failed.append(f"{url} → {e}")
            print(f"[ContinuousTraining] Reload failed: {url} → {e}")

    # Notify promotion regardless — model is saved to shared volume
    notify_model_promoted(
        model_name=champion_name,
        model_version=str(model_version),
        endpoint_url=f"{lb_url}/predict",
        context=context,
    )

    summary = {
        "status": "all_reloaded" if not failed else "partial",
        "champion": champion_name,
        "reloaded": len(reloaded),
        "failed": len(failed),
        "details": failed if failed else "all replicas reloaded",
    }
    print(f"[ContinuousTraining] Reload summary: {summary}")
    return summary


def task_notify_failure(context):
    """on_failure_callback for the DAG — wraps Slack notification."""
    from orchestration.slack_notifications import notify_retraining_failed

    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "unknown"
    exception = context.get("exception", "Unknown error")
    notify_retraining_failed(str(exception), task_id, context)


# ═══════════════════════════════════════════════════════════════════════════
#  DAG Definition
# ═══════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="flight_fare_continuous_training",
    default_args={
        **default_args,
        "on_failure_callback": task_notify_failure,
    },
    description="Automated multi-model retraining triggered by drift detection",
    schedule_interval="0 2 */3 * *",   # Every 3 days at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining", "continuous-training", "flight-fare"],
) as dag:

    # 1. Check for drift — branches to retrain or skip
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
        provide_context=True,
    )

    # 2a. No drift — skip
    skip = PythonOperator(
        task_id="skip_retraining",
        python_callable=task_skip_retraining,
        provide_context=True,
    )

    # 2b. Drift detected — notify retraining started
    notify_started = PythonOperator(
        task_id="notify_retraining_started",
        python_callable=task_notify_retraining_started,
        provide_context=True,
    )

    # 3. Extract fresh data
    extract = PythonOperator(
        task_id="extract_data",
        python_callable=task_extract_data,
        provide_context=True,
    )

    # 4. Preprocess
    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
        provide_context=True,
    )

    # 5. Train all 4 models & select champion
    train = PythonOperator(
        task_id="train_all_models",
        python_callable=task_train_all_models,
        provide_context=True,
    )

    # 6. Log to MLflow
    mlflow_log = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=task_log_to_mlflow,
        provide_context=True,
    )

    # 7. Reload prediction endpoint
    reload_ep = PythonOperator(
        task_id="reload_endpoint",
        python_callable=task_reload_endpoint,
        provide_context=True,
    )

    # ── Wiring ──────────────────────────────────────────────────────────────
    check_drift >> [notify_started, skip]
    notify_started >> extract >> preprocess >> train >> mlflow_log >> reload_ep
