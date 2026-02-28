"""
Slack Notification Helpers for the Flight Fare ML Pipeline.

Sends structured messages to a Slack channel via an Incoming Webhook.
Used by the Airflow DAGs to report:
  - Pipeline triggered (start)
  - Task failure (with error details)
  - Pipeline completed (with metrics summary)
  – Retraining needed (drift detected)
  – Retraining started / completed / failed
  – Champion model promoted to production
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL = os.getenv("CHANNEL_NAME", "ml_pipeline_report")

DAG_NAME = "flight_fare_ml_pipeline"
RETRAIN_DAG_NAME = "flight_fare_continuous_training"
AIRFLOW_UI_BASE = os.getenv("AIRFLOW_UI_BASE_URL", "http://localhost:8082")


def _post_to_slack(payload: Dict[str, Any]) -> bool:
    """
    Send a JSON payload to the Slack webhook.

    Returns True on success, False otherwise (never raises).
    """
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL is not set — skipping notification.")
        return False

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.error(
                "Slack webhook returned %s: %s", resp.status_code, resp.text
            )
            return False
        return True
    except Exception as exc:
        logger.error("Failed to send Slack notification: %s", exc)
        return False


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ═══════════════════════════════════════════════════════════════════════════
#  Public notification functions
# ═══════════════════════════════════════════════════════════════════════════

def notify_pipeline_started(context: Dict[str, Any]) -> None:
    """
    Called at the very beginning of the DAG to announce it has been triggered.
    Wire this as the first PythonOperator task.
    """
    dag_run = context.get("dag_run")
    run_id = dag_run.run_id if dag_run else "unknown"
    execution_date = context.get("execution_date", _now_str())

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":rocket:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚀 ML Pipeline Triggered",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Execution Date:*\n{execution_date}",
                    },
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"<{AIRFLOW_UI_BASE}/dags/{DAG_NAME}/grid"
                            f"|View in Airflow UI>"
                        ),
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_task_failure(context: Dict[str, Any]) -> None:
    """
    Airflow ``on_failure_callback`` — called when any task fails.
    Attach this to ``default_args['on_failure_callback']``.
    """
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "unknown"
    dag_run = context.get("dag_run")
    run_id = dag_run.run_id if dag_run else "unknown"
    exception = context.get("exception", "No exception info")
    log_url = task_instance.log_url if task_instance else ""

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":red_circle:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "❌ Pipeline Task Failed",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Task:*\n`{task_id}`"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{str(exception)[:1000]}```",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"<{log_url}|View Task Logs>" if log_url else "Check Airflow UI for logs",
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_pipeline_complete(context: Dict[str, Any]) -> None:
    """
    Called as the final task in the DAG after all work is done.
    Pulls metrics from XCom and sends a success summary to Slack.
    """
    ti = context["ti"]
    dag_run = context.get("dag_run")
    run_id = dag_run.run_id if dag_run else "unknown"

    # Pull results from upstream tasks
    eval_result = ti.xcom_pull(task_ids="evaluation") or {}
    mlflow_result = ti.xcom_pull(task_ids="mlflow_logging") or {}
    evidently_result = ti.xcom_pull(task_ids="evidently_monitoring") or {}
    metrics = eval_result.get("metrics", {})

    # Format metrics
    r2 = metrics.get("R2", 0)
    mae = metrics.get("MAE", 0)
    rmse = metrics.get("RMSE", 0)
    mape = metrics.get("MAPE", 0)
    model_version = mlflow_result.get("model_version", "N/A")
    drift_status = "✅ Generated" if evidently_result.get("data_drift") == "ok" else "⚠️ Skipped"

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":white_check_mark:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ ML Pipeline Completed Successfully",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Model Version:*\n`{model_version}`",
                    },
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*📊 Evaluation Metrics*\n"
                        f"```"
                        f"R²    = {r2:.4f}\n"
                        f"MAE   = {mae:,.0f} BDT\n"
                        f"RMSE  = {rmse:,.0f} BDT\n"
                        f"MAPE  = {mape:.2f}%"
                        f"```"
                    ),
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Drift Reports:*\n{drift_status}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*MLflow Logged:*\n✅ Params, Metrics, Model",
                    },
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"<{AIRFLOW_UI_BASE}/dags/{DAG_NAME}/grid"
                            f"|Airflow UI>  •  "
                            f"<http://localhost:5000|MLflow UI>  •  "
                            f"<http://localhost:8001|Evidently UI>"
                        ),
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


# ═══════════════════════════════════════════════════════════════════════════
#  Continuous-training / retraining notifications
# ═══════════════════════════════════════════════════════════════════════════

def notify_retraining_needed(
    reasons: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Called when drift detection determines that retraining is required.
    """
    triggers = reasons.get("triggers", [])
    drift_share = reasons.get("drift_share", 0.0)
    n_drifted = reasons.get("n_drifted_features", 0)
    triggers_text = "\n".join(f"• {t}" for t in triggers) or "Unknown trigger"

    run_id = ""
    if context:
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else ""

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":warning:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "⚠️ Retraining Needed — Drift Detected",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{RETRAIN_DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Drift Share:*\n{drift_share:.1%}"},
                    {"type": "mrkdwn", "text": f"*Drifted Features:*\n{n_drifted}"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Triggers:*\n{triggers_text}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Automatic retraining will begin shortly.",
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_retraining_started(
    models: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Called when multi-model retraining begins."""
    model_list = ", ".join(models)
    run_id = ""
    if context:
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else ""

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":gear:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🔄 Retraining Started",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{RETRAIN_DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Models:*\n{model_list}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"Training all 4 models: {model_list}. "
                            "Champion will be selected by best R² on validation set."
                        ),
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_retraining_failed(
    error: str,
    failed_task: str = "retraining",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Called when retraining encounters an error."""
    run_id = ""
    if context:
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else ""

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":x:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "❌ Retraining Failed",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{RETRAIN_DAG_NAME}`"},
                    {"type": "mrkdwn", "text": f"*Failed Step:*\n`{failed_task}`"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{str(error)[:1500]}```",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"<{AIRFLOW_UI_BASE}/dags/{RETRAIN_DAG_NAME}/grid"
                            f"|View in Airflow UI>"
                        ),
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_retraining_complete(
    champion_name: str,
    comparison_table: str,
    champion_metrics: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Called after all models are trained and the champion is selected."""
    run_id = ""
    if context:
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else ""

    r2 = champion_metrics.get("R2", 0)
    mae = champion_metrics.get("MAE", 0)
    rmse = champion_metrics.get("RMSE", 0)
    mape = champion_metrics.get("MAPE", 0)

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":trophy:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🏆 Retraining Completed — Champion Selected",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Champion:*\n`{champion_name}`"},
                    {"type": "mrkdwn", "text": f"*R²:*\n{r2:.4f}"},
                    {"type": "mrkdwn", "text": f"*MAE:*\n{mae:,.0f}"},
                    {"type": "mrkdwn", "text": f"*RMSE:*\n{rmse:,.0f}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📊 Model Comparison:*\n```{comparison_table}```",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Timestamp:* {_now_str()}",
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)


def notify_model_promoted(
    model_name: str,
    model_version: str,
    endpoint_url: str = "http://localhost:8003/predict",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Called when the champion model is pushed to the production endpoint."""
    run_id = ""
    if context:
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else ""

    payload = {
        "channel": SLACK_CHANNEL,
        "username": "Airflow Bot",
        "icon_emoji": ":tada:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚀 New Model Deployed to Production",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Model:*\n`{model_name}`"},
                    {"type": "mrkdwn", "text": f"*Version:*\n`{model_version}`"},
                    {"type": "mrkdwn", "text": f"*Endpoint:*\n`{endpoint_url}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{_now_str()}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "The champion model has been saved to the artifacts "
                        "directory and the prediction endpoint has been reloaded. "
                        "The new model is now serving live traffic."
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"<{AIRFLOW_UI_BASE}/dags/{RETRAIN_DAG_NAME}/grid"
                            f"|Airflow UI>  •  "
                            f"<http://localhost:5000|MLflow UI>  •  "
                            f"<{endpoint_url.replace('/predict', '/health')}|Health Check>"
                        ),
                    }
                ],
            },
        ],
    }
    _post_to_slack(payload)
