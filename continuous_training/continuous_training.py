"""
Continuous Training Module — Multi-Model Retraining Pipeline.

Trains and compares four regression models on the flight fare dataset:
  1. Random Forest
  2. XGBoost
  3. LightGBM
  4. CatBoost

Selects the best performer (champion) based on R² on the validation set,
saves artifacts, and provides hooks for MLflow logging and Slack notifications.

Usage
-----
    from continuous_training.continuous_training import ContinuousTrainer

    ct = ContinuousTrainer(artifacts_dir="artifacts")
    results = ct.train_all_models(X_train, y_train, X_val, y_val, cat_indices)
    champion = ct.select_champion(results)
    ct.save_champion(champion)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Configurations
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "needs_numeric_only": True,
        "description": "Ensemble of decision trees with bagging",
    },
    "XGBoost": {
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "enable_categorical": True,
        },
        "needs_numeric_only": False,
        "description": "Gradient boosting with regularisation (XGBoost)",
    },
    "LightGBM": {
        "class": "lightgbm.LGBMRegressor",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "needs_numeric_only": False,
        "description": "Gradient boosting with leaf-wise growth (LightGBM)",
    },
    "CatBoost": {
        "class": "catboost.CatBoostRegressor",
        "params": {
            "iterations": 2000,
            "learning_rate": 0.02,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "early_stopping_rounds": 100,
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1,
        },
        "needs_numeric_only": False,
        "description": "Gradient boosting with native categorical handling (CatBoost)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  Model Result Container
# ═══════════════════════════════════════════════════════════════════════════

class ModelResult:
    """Holds training output for a single model."""

    def __init__(
        self,
        model_name: str,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        feature_importances: Optional[pd.Series],
        training_time_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.params = params
        self.metrics = metrics
        self.feature_importances = feature_importances
        self.training_time_seconds = training_time_seconds

    def __repr__(self) -> str:
        r2 = self.metrics.get("R2", 0)
        mae = self.metrics.get("MAE", 0)
        return f"<ModelResult {self.model_name} R²={r2:.4f} MAE={mae:,.0f}>"


# ═══════════════════════════════════════════════════════════════════════════
#  Continuous Trainer
# ═══════════════════════════════════════════════════════════════════════════

class ContinuousTrainer:
    """
    Orchestrates multi-model training, evaluation, champion selection,
    and artifact persistence for the continuous training pipeline.
    """

    CHAMPION_MODEL_FILENAME = "champion_model.pkl"
    CHAMPION_PREPROCESSOR_FILENAME = "preprocessor.pkl"
    # For backward compatibility with the FastAPI serving endpoint
    CATBOOST_MODEL_FILENAME = "catboost_model.pkl"

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute metrics ─────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": float(
                np.mean(np.abs(y_true - y_pred) / np.clip(y_true, 1, None)) * 100
            ),
            "MedAE": float(median_absolute_error(y_true, y_pred)),
        }

    # ── Prepare data for sklearn-style models ──────────────────────────────

    @staticmethod
    def prepare_numeric_features(
        X: pd.DataFrame,
        cat_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert categorical columns to numeric codes for models that
        cannot handle pandas category dtype (e.g. Random Forest).
        """
        X_out = X.copy()
        if cat_columns is None:
            cat_columns = [c for c in X_out.columns if X_out[c].dtype.name == "category"]
        for col in cat_columns:
            X_out[col] = X_out[col].astype(str).astype("category").cat.codes
        return X_out

    # ── Train a single model ───────────────────────────────────────────────

    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_feature_indices: Optional[List[int]] = None,
        cat_feature_names: Optional[List[str]] = None,
    ) -> ModelResult:
        """
        Train one model and return a ModelResult.

        Parameters
        ----------
        model_name : str
            Key from MODEL_CONFIGS (RandomForest, XGBoost, LightGBM, CatBoost).
        X_train, y_train : training data.
        X_val, y_val : validation data.
        cat_feature_indices : for CatBoost native categoricals.
        cat_feature_names : for LightGBM categorical features.
        """
        config = MODEL_CONFIGS[model_name]
        params = dict(config["params"])

        # ── Prepare data ────────────────────────────────────────────────
        if config["needs_numeric_only"]:
            X_tr = self.prepare_numeric_features(X_train)
            X_va = self.prepare_numeric_features(X_val)
        else:
            X_tr = X_train.copy()
            X_va = X_val.copy()

        # ── Instantiate model ───────────────────────────────────────────
        model = self._create_model(model_name, params, cat_feature_indices)

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"  Training: {model_name}")
        print(f"{'─' * 60}")

        start = time.time()

        if model_name == "CatBoost":
            model.fit(X_tr, y_train, eval_set=(X_va, y_val))
        elif model_name == "XGBoost":
            model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                verbose=False,
            )
        elif model_name == "LightGBM":
            cat_cols = cat_feature_names or []
            # Mark categorical columns for LightGBM
            for col in cat_cols:
                if col in X_tr.columns:
                    X_tr[col] = X_tr[col].astype("category")
                if col in X_va.columns:
                    X_va[col] = X_va[col].astype("category")
            model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                callbacks=[],
            )
        else:
            # RandomForest — no eval_set
            model.fit(X_tr, y_train)

        elapsed = time.time() - start

        # ── Evaluate ────────────────────────────────────────────────────
        y_pred = model.predict(X_va)
        metrics = self.compute_metrics(y_val, y_pred)

        # ── Feature importances ─────────────────────────────────────────
        fi = self._extract_feature_importances(model, X_tr)

        print(
            f"  {model_name} — R²={metrics['R2']:.4f}  "
            f"MAE={metrics['MAE']:,.0f}  RMSE={metrics['RMSE']:,.0f}  "
            f"({elapsed:.1f}s)"
        )

        return ModelResult(
            model_name=model_name,
            model=model,
            params=params,
            metrics=metrics,
            feature_importances=fi,
            training_time_seconds=elapsed,
        )

    # ── Train all models ───────────────────────────────────────────────────

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_feature_indices: Optional[List[int]] = None,
        cat_feature_names: Optional[List[str]] = None,
    ) -> List[ModelResult]:
        """
        Train all four models and return a list of ModelResult objects.
        """
        results: List[ModelResult] = []

        for name in MODEL_CONFIGS:
            try:
                result = self.train_single_model(
                    model_name=name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    cat_feature_indices=cat_feature_indices,
                    cat_feature_names=cat_feature_names,
                )
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  {name} training failed: {e}")

        return results

    # ── Champion selection ─────────────────────────────────────────────────

    @staticmethod
    def select_champion(
        results: List[ModelResult],
        metric: str = "R2",
        higher_is_better: bool = True,
    ) -> ModelResult:
        """
        Select the best model based on a given metric.

        Parameters
        ----------
        results : list of ModelResult
        metric : str
            Metric key to compare (default: R2).
        higher_is_better : bool
            If True, highest wins; if False, lowest wins.

        Returns
        -------
        ModelResult for the champion model.
        """
        if not results:
            raise ValueError("No model results to compare.")

        sorted_results = sorted(
            results,
            key=lambda r: r.metrics.get(metric, float("-inf")),
            reverse=higher_is_better,
        )
        champion = sorted_results[0]
        print(
            f"\n🏆 Champion: {champion.model_name} — "
            f"R²={champion.metrics['R2']:.4f}  "
            f"MAE={champion.metrics['MAE']:,.0f}"
        )
        return champion

    # ── Build comparison table ─────────────────────────────────────────────

    @staticmethod
    def build_comparison_table(results: List[ModelResult]) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        rows = []
        for r in results:
            rows.append({
                "Model": r.model_name,
                "R²": r.metrics.get("R2", 0),
                "MAE": r.metrics.get("MAE", 0),
                "RMSE": r.metrics.get("RMSE", 0),
                "MAPE (%)": r.metrics.get("MAPE", 0),
                "MedAE": r.metrics.get("MedAE", 0),
                "Time (s)": r.training_time_seconds,
            })
        df = pd.DataFrame(rows).sort_values("R²", ascending=False)
        return df.reset_index(drop=True)

    # ── Save champion artifacts ────────────────────────────────────────────

    def save_champion(
        self,
        champion: ModelResult,
        preprocessor: Any = None,
    ) -> Dict[str, Path]:
        """
        Save the champion model and preprocessor as pickle artifacts.

        Also saves as ``catboost_model.pkl`` for backward compatibility
        with the FastAPI serving endpoint.
        """
        paths: Dict[str, Path] = {}

        # Champion model
        champion_path = self.artifacts_dir / self.CHAMPION_MODEL_FILENAME
        with open(champion_path, "wb") as f:
            pickle.dump(champion.model, f)
        paths["champion_model"] = champion_path
        print(f"Champion saved → {champion_path}")

        # Backward-compatible serving path (FastAPI loads catboost_model.pkl)
        compat_path = self.artifacts_dir / self.CATBOOST_MODEL_FILENAME
        with open(compat_path, "wb") as f:
            pickle.dump(champion.model, f)
        paths["serving_model"] = compat_path
        print(f"Serving model saved → {compat_path}")

        # Preprocessor
        if preprocessor is not None:
            prep_path = self.artifacts_dir / self.CHAMPION_PREPROCESSOR_FILENAME
            with open(prep_path, "wb") as f:
                pickle.dump(preprocessor, f)
            paths["preprocessor"] = prep_path
            print(f"Preprocessor saved → {prep_path}")

        return paths

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _create_model(
        model_name: str,
        params: Dict[str, Any],
        cat_feature_indices: Optional[List[int]] = None,
    ) -> Any:
        """Instantiate a model by name."""
        if model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)

        elif model_name == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor(**params)

        elif model_name == "LightGBM":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**params)

        elif model_name == "CatBoost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                cat_features=cat_feature_indices,
                **params,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def _extract_feature_importances(
        model: Any, X: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Extract feature importances from a fitted model."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                importances = model.get_feature_importance()
            else:
                return None

            names = (
                model.feature_names_
                if hasattr(model, "feature_names_")
                else list(X.columns)
            )
            return pd.Series(importances, index=names).sort_values(ascending=False)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  Drift-based retraining trigger
# ═══════════════════════════════════════════════════════════════════════════

def check_retraining_needed(
    drift_report_path: str | Path,
    r2_threshold: float = 0.85,
    drift_share_threshold: float = 0.3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if retraining is needed based on drift reports.

    Parameters
    ----------
    drift_report_path : path to the data drift JSON report.
    r2_threshold : R² below this triggers retraining.
    drift_share_threshold : fraction of drifted features above this triggers retraining.

    Returns
    -------
    (needs_retraining: bool, reasons: dict)
    """
    import json

    drift_path = Path(drift_report_path)
    reasons: Dict[str, Any] = {
        "drift_detected": False,
        "drift_share": 0.0,
        "n_drifted_features": 0,
        "triggers": [],
    }

    if not drift_path.exists():
        reasons["triggers"].append("No drift report found — retraining as safety measure")
        return True, reasons

    try:
        with open(drift_path) as f:
            report = json.load(f)

        # Try to extract drift info from the Evidently JSON structure
        # The exact structure depends on Evidently version
        metric_results = report.get("metric_results", report.get("metrics", []))

        n_drifted = 0
        n_total = 0

        for mr in metric_results:
            metric_type = str(mr.get("type", mr.get("metric", "")))
            value = mr.get("value", mr.get("result", {}))

            if "DriftedColumnsCount" in metric_type:
                if isinstance(value, dict):
                    n_drifted = value.get("count", value.get("number_of_drifted_columns", 0))
                elif isinstance(value, (int, float)):
                    n_drifted = int(value)

        # Count total features from ValueDrift entries
        for mr in metric_results:
            metric_type = str(mr.get("type", mr.get("metric", "")))
            if "ValueDrift" in metric_type:
                n_total += 1

        drift_share = n_drifted / max(n_total, 1)
        reasons["drift_share"] = drift_share
        reasons["n_drifted_features"] = n_drifted

        if drift_share >= drift_share_threshold:
            reasons["drift_detected"] = True
            reasons["triggers"].append(
                f"Data drift detected: {n_drifted}/{n_total} features "
                f"({drift_share:.1%}) exceeded {drift_share_threshold:.0%} threshold"
            )

    except Exception as e:
        reasons["triggers"].append(f"Could not parse drift report: {e}")
        return True, reasons

    needs_retraining = len(reasons["triggers"]) > 0
    return needs_retraining, reasons
