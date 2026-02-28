"""
Model Store — MLflow experiment tracking & model registry.

Logs hyper-parameters, training metrics, evaluation results, feature
importances, and model artefacts to MLflow for multiple model types:
  - RandomForest (sklearn)
  - XGBoost
  - LightGBM
  - CatBoost

Optionally registers the model in the MLflow Model Registry for
stage promotion (Staging → Production).

Usage
-----
    from experiment_tracking.model_store import ModelStore

    ms = ModelStore(tracking_uri="http://localhost:5000")
    run_id = ms.log_training_run(
        model=trainer.model,
        params=trainer.params,
        metrics=metrics,
        model_type="CatBoost",
        feature_importances=trainer.feature_importances,
        X_train=X_train,
        preprocessor=preprocessor,
    )
    ms.register_model(run_id, model_name="flight-fare-catboost")
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.catboost
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd

# Mapping of model type → model family tag
MODEL_FAMILY_MAP = {
    "RandomForest": "ensemble_bagging",
    "XGBoost": "gradient_boosting",
    "LightGBM": "gradient_boosting",
    "CatBoost": "gradient_boosting",
}

# Mapping of model type → MLflow logging flavour
MODEL_LOG_FLAVOUR = {
    "RandomForest": "sklearn",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
}


class ModelStore:
    """
    MLflow-backed model store for the Flight Fare Prediction pipeline.

    Responsibilities:
    * Log every training run (params, metrics, artefacts)
    * Store models using the correct MLflow flavour per type
    * Store the preprocessor pickle as an artefact
    * Register models in the MLflow Model Registry
    * Load models for inference from the registry
    """

    EXPERIMENT_NAME = "flight-fare-model-store"
    REGISTERED_MODEL_NAME = "flight-fare-catboost"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        self._tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        )
        mlflow.set_tracking_uri(self._tracking_uri)

        self._experiment_name = experiment_name or self.EXPERIMENT_NAME
        mlflow.set_experiment(self._experiment_name)

    # ── Log a complete training run ─────────────────────────────────────────

    def log_training_run(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_type: str = "CatBoost",
        feature_importances: Optional[pd.Series] = None,
        X_train: Optional[pd.DataFrame] = None,
        preprocessor: Any = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        is_champion: bool = False,
    ) -> str:
        """
        Log a full training run to MLflow.

        Parameters
        ----------
        model : trained model instance.
        params : dict
            Hyperparameters used for training.
        metrics : dict
            Evaluation metrics (R2, MAE, RMSE, MAPE, MedAE).
        model_type : str
            One of: RandomForest, XGBoost, LightGBM, CatBoost.
        feature_importances : pd.Series, optional
            Feature importance scores from the model.
        X_train : pd.DataFrame, optional
            Training feature matrix — used to infer the MLflow model signature.
        preprocessor : FlightFarePreprocessor, optional
            Fitted preprocessor to store as an artefact.
        run_name : str, optional
            Human-readable name for the run.
        tags : dict, optional
            Additional MLflow tags.
        is_champion : bool
            If True, tag this run as the champion model.

        Returns
        -------
        str   The MLflow run ID.
        """
        default_run_name = f"{model_type.lower()}-training"
        flavour = MODEL_LOG_FLAVOUR.get(model_type, "sklearn")
        family = MODEL_FAMILY_MAP.get(model_type, "unknown")

        with mlflow.start_run(run_name=run_name or default_run_name) as run:
            # ── Tags ────────────────────────────────────────────────────
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_family", family)
            mlflow.set_tag("mlflow_flavour", flavour)
            if is_champion:
                mlflow.set_tag("champion", "true")
                mlflow.set_tag("stage", "champion")
            if tags:
                mlflow.set_tags(tags)

            # ── Params ──────────────────────────────────────────────────
            for key, val in params.items():
                mlflow.log_param(key, val)
            if X_train is not None:
                mlflow.log_param("n_train_samples", X_train.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])

            # ── Metrics ─────────────────────────────────────────────────
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)

            # Model-specific metrics
            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is not None:
                mlflow.log_metric("best_iteration", best_iter)
            tree_count = getattr(model, "tree_count_", None)
            if tree_count is not None:
                mlflow.log_metric("tree_count", tree_count)

            # ── Model artefact (flavour-aware) ──────────────────────────
            input_example = None
            signature = None
            if X_train is not None:
                try:
                    input_example = X_train.head(5)
                    from mlflow.models import infer_signature
                    preds = model.predict(X_train.head(5))
                    signature = infer_signature(input_example, preds)
                except Exception:
                    input_example = None
                    signature = None

            self._log_model_by_flavour(
                model=model,
                flavour=flavour,
                signature=signature,
                input_example=input_example,
            )

            # ── Artefacts ───────────────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmpdir:
                # Feature importances
                if feature_importances is not None:
                    fi_path = Path(tmpdir) / "feature_importances.json"
                    fi_path.write_text(
                        json.dumps(
                            feature_importances.to_dict(), indent=2, default=str,
                        )
                    )
                    mlflow.log_artifact(str(fi_path), artifact_path="evaluation")

                    fi_csv = Path(tmpdir) / "feature_importances.csv"
                    feature_importances.reset_index().rename(
                        columns={"index": "feature", 0: "importance"}
                    ).to_csv(fi_csv, index=False)
                    mlflow.log_artifact(str(fi_csv), artifact_path="evaluation")

                # Preprocessor pickle
                if preprocessor is not None:
                    prep_path = Path(tmpdir) / "preprocessor.pkl"
                    with open(prep_path, "wb") as f:
                        pickle.dump(preprocessor, f)
                    mlflow.log_artifact(str(prep_path), artifact_path="preprocessor")

                # Params JSON
                params_path = Path(tmpdir) / "training_params.json"
                params_path.write_text(
                    json.dumps(params, indent=2, default=str)
                )
                mlflow.log_artifact(str(params_path), artifact_path="config")

            run_id = run.info.run_id
            print(
                f"[ModelStore] Training run logged — "
                f"model={model_type}  run_id={run_id}  "
                f"R2={metrics.get('R2', 'N/A'):.4f}  "
                f"MAE={metrics.get('MAE', 'N/A'):,.0f}"
            )
            return run_id

    # ── Log all retraining results at once ──────────────────────────────────

    def log_retraining_results(
        self,
        results: list,
        champion_name: str,
        preprocessor: Any = None,
        X_train: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Log all model results from a ContinuousTrainer session.

        Parameters
        ----------
        results : list of ModelResult
        champion_name : str
            Name of the champion model.
        preprocessor : optional preprocessor object.
        X_train : optional training data for signature inference.

        Returns
        -------
        dict  mapping model_name → run_id.
        """
        run_ids: Dict[str, str] = {}

        for result in results:
            is_champ = result.model_name == champion_name
            run_id = self.log_training_run(
                model=result.model,
                params=result.params,
                metrics=result.metrics,
                model_type=result.model_name,
                feature_importances=result.feature_importances,
                X_train=X_train,
                preprocessor=preprocessor if is_champ else None,
                run_name=f"retrain-{result.model_name.lower()}",
                tags={
                    "training_time_seconds": str(result.training_time_seconds),
                    "retrain": "true",
                },
                is_champion=is_champ,
            )
            run_ids[result.model_name] = run_id

        return run_ids

    # ── Flavour-aware model logging ─────────────────────────────────────────

    @staticmethod
    def _log_model_by_flavour(
        model: Any,
        flavour: str,
        signature: Any = None,
        input_example: Any = None,
    ) -> None:
        """Log a model using the appropriate MLflow flavour."""
        kwargs = dict(
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        if flavour == "catboost":
            mlflow.catboost.log_model(cb_model=model, **kwargs)
        elif flavour == "xgboost":
            mlflow.xgboost.log_model(xgb_model=model, **kwargs)
        elif flavour == "lightgbm":
            mlflow.lightgbm.log_model(lgb_model=model, **kwargs)
        elif flavour == "sklearn":
            mlflow.sklearn.log_model(sk_model=model, **kwargs)
        else:
            # Generic fallback
            mlflow.sklearn.log_model(sk_model=model, **kwargs)

    # ── Model Registry ──────────────────────────────────────────────────────

    def register_model(
        self,
        run_id: str,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Register a logged model in the MLflow Model Registry.

        Parameters
        ----------
        run_id : str
            The MLflow run ID that contains the logged model.
        model_name : str, optional
            Registry name. Defaults to ``REGISTERED_MODEL_NAME``.

        Returns
        -------
        str   The model version string.
        """
        model_name = model_name or self.REGISTERED_MODEL_NAME
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        print(
            f"[ModelStore] Model registered — "
            f"name='{model_name}'  version={version}"
        )
        return str(version)

    def transition_model_stage(
        self,
        version: str,
        stage: str = "Production",
        model_name: Optional[str] = None,
    ) -> None:
        """
        Transition a registered model version to a new stage.

        Parameters
        ----------
        version : str
            Model version number.
        stage : str
            Target stage — ``"Staging"``, ``"Production"``, or ``"Archived"``.
        model_name : str, optional
            Registry name. Defaults to ``REGISTERED_MODEL_NAME``.
        """
        model_name = model_name or self.REGISTERED_MODEL_NAME
        client = mlflow.MlflowClient()

        try:
            # MLflow >= 2.9: alias-based workflow
            alias = stage.lower().replace(" ", "-")
            client.set_registered_model_alias(model_name, alias, version)
            print(
                f"[ModelStore] Set alias '{alias}' on "
                f"model='{model_name}' version={version}"
            )
        except Exception:
            # Fallback: legacy stage transition
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True,
            )
            print(
                f"[ModelStore] Transitioned model='{model_name}' "
                f"v{version} → {stage}"
            )

    # ── Load from registry ──────────────────────────────────────────────────

    def load_production_model(
        self,
        model_name: Optional[str] = None,
    ) -> Any:
        """
        Load the latest Production model from the MLflow Model Registry.

        Tries alias-based loading first, then falls back to latest version.
        Uses ``mlflow.pyfunc`` for flavour-agnostic loading.

        Returns
        -------
        Loaded model (any flavour).
        """
        model_name = model_name or self.REGISTERED_MODEL_NAME
        client = mlflow.MlflowClient()

        try:
            # Try alias-based loading first (MLflow >= 2.9)
            mv = client.get_model_version_by_alias(model_name, "production")
            model_uri = f"models:/{model_name}@production"
        except Exception:
            # Fallback: load latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise RuntimeError(
                    f"No registered model found with name '{model_name}'"
                )
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            model_uri = f"models:/{model_name}/{latest.version}"

        # Use pyfunc for flavour-agnostic loading
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"[ModelStore] Loaded model from {model_uri}")
        return model

    def load_preprocessor(self, run_id: str) -> Any:
        """
        Download and load the preprocessor pickle from a specific run.

        Parameters
        ----------
        run_id : str
            The MLflow run ID that contains the preprocessor artefact.

        Returns
        -------
        FlightFarePreprocessor
        """
        client = mlflow.MlflowClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(
                run_id, "preprocessor/preprocessor.pkl", tmpdir,
            )
            with open(local_path, "rb") as f:
                preprocessor = pickle.load(f)
        print(f"[ModelStore] Preprocessor loaded from run {run_id}")
        return preprocessor

    # ── Query helpers ───────────────────────────────────────────────────────

    def get_best_run(
        self,
        metric: str = "R2",
        ascending: bool = False,
    ) -> Dict[str, Any]:
        """
        Find the best run in the experiment by a given metric.

        Returns
        -------
        dict  with keys 'run_id', 'metrics', 'params'.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise RuntimeError(f"Experiment '{self._experiment_name}' not found")

        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            raise RuntimeError("No runs found")

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metrics": {
                k.replace("metrics.", ""): v
                for k, v in best.items()
                if k.startswith("metrics.")
            },
            "params": {
                k.replace("params.", ""): v
                for k, v in best.items()
                if k.startswith("params.")
            },
        }
