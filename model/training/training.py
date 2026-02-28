"""
Training module for the Flight Fare Prediction pipeline.

Trains a CatBoost regressor (the best model identified in the prototype notebook)
with native categorical handling and saves/loads models via pickle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class FlightFareTrainer:
    """
    Trains and manages a CatBoost regressor for flight fare prediction.

    The default hyperparameters match the best configuration found during
    the notebook experimentation:
      - CatBoost with raw target (no log transform)
      - RMSE loss function
      - depth=8, learning_rate=0.02, 2 000 iterations, early stopping
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "early_stopping_rounds": 100,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
        "border_count": 254,
    }

    def __init__(
        self,
        cat_feature_indices: Optional[List[int]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        cat_feature_indices : list[int], optional
            Positional indices of categorical columns in the feature matrix.
            Obtained from ``FlightFarePreprocessor.cat_feature_indices``.
        params : dict, optional
            CatBoost hyperparameters. Merged on top of ``DEFAULT_PARAMS``.
        """
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._cat_feature_indices = cat_feature_indices
        self._model: Optional[CatBoostRegressor] = None

    # ── Training ────────────────────────────────────────────────────────────
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> CatBoostRegressor:
        """
        Train the CatBoost model.

        Parameters
        ----------
        X_train, y_train : training features and target.
        X_val, y_val : optional validation set used for early stopping.

        Returns
        -------
        CatBoostRegressor
            The fitted model instance.
        """
        model = CatBoostRegressor(
            cat_features=self._cat_feature_indices,
            **self._params,
        )

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)

        model.fit(X_train, y_train, **fit_kwargs)
        self._model = model

        best_iter = getattr(model, "best_iteration_", model.tree_count_)
        print(
            f"Training complete — "
            f"best iteration: {best_iter}, "
            f"trees: {model.tree_count_}"
        )
        return model

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the fitted model."""
        if self._model is None:
            raise RuntimeError("No trained model available. Call train() or load_model() first.")
        return self._model.predict(X)

    # ── Persistence (pickle) ───────────────────────────────────────────────
    def save_model(self, path: str | Path) -> Path:
        """
        Serialise the trained CatBoost model to disk using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"artifacts/catboost_model.pkl"``).

        Returns
        -------
        Path
            The resolved path the model was saved to.
        """
        if self._model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._model, f)

        print(f"Model saved to {path}  ({path.stat().st_size / 1024:.1f} KB)")
        return path

    def load_model(self, path: str | Path) -> CatBoostRegressor:
        """
        Load a previously saved CatBoost model from a pickle file.

        Returns
        -------
        CatBoostRegressor
            The loaded model (also stored internally for ``predict()``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            self._model = pickle.load(f)

        print(f"Model loaded from {path}")
        return self._model

    # ── Accessors ───────────────────────────────────────────────────────────
    @property
    def model(self) -> Optional[CatBoostRegressor]:
        """Access the underlying CatBoost model."""
        return self._model

    @property
    def params(self) -> Dict[str, Any]:
        """Return the hyperparameters used for training."""
        return dict(self._params)

    @property
    def feature_importances(self) -> Optional[pd.Series]:
        """Return feature importance from the fitted model (if available)."""
        if self._model is None:
            return None
        importances = self._model.get_feature_importance()
        names = self._model.feature_names_
        return pd.Series(importances, index=names).sort_values(ascending=False)


# ── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to sys.path so sibling packages are importable
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from model.training.preprocessing import FlightFarePreprocessor
    from model.evaluation.evaluation import FlightFareEvaluator

    # ── Paths ───────────────────────────────────────────────────────────────
    CSV_PATH = (
        PROJECT_ROOT
        / "notebooks"
        / "data"
        / "kaggle"
        / "flight-price-dataset-of-bangladesh"
        / "Flight_Price_Dataset_of_Bangladesh.csv"
    )
    MODEL_SAVE_PATH = PROJECT_ROOT / "artifacts" / "catboost_model.pkl"

    # ── 1. Load & split ────────────────────────────────────────────────────
    preprocessor = FlightFarePreprocessor()
    df = preprocessor.load_data(CSV_PATH)
    train_df, test_df = preprocessor.split_data(df, test_size=0.2, temporal=True)

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)

    # ── 3. Train ───────────────────────────────────────────────────────────
    trainer = FlightFareTrainer(
        cat_feature_indices=preprocessor.cat_feature_indices,
    )
    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    y_pred = trainer.predict(X_test)

    evaluator = FlightFareEvaluator()
    metrics = evaluator.compute_metrics(y_test, y_pred)
    evaluator.print_metrics(metrics, title="CatBoost — Hold-out Test Set")

    # Segment breakdown
    seg_df = evaluator.evaluate_by_fare_segment(y_test, y_pred)
    print("\nPerformance by fare segment:")
    print(seg_df.to_string(index=False))

    # Top 10 feature importances
    fi = trainer.feature_importances
    if fi is not None:
        print("\nTop 10 Feature Importances:")
        print(fi.head(10).to_string())

    # ── 5. Save model and preprocessor with pickle ──────────────────────────
    trainer.save_model(MODEL_SAVE_PATH)

    PREPROCESSOR_SAVE_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"
    with open(PREPROCESSOR_SAVE_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

    # ── 6. MLflow: log feature profile & training run ───────────────────────
    try:
        from experiment_tracking.feature_store import FeatureStore
        from experiment_tracking.model_store import ModelStore

        print("\n── MLflow Logging ────────────────────────────────────────")

        # Feature store
        fs = FeatureStore()
        fs_run_id = fs.log_feature_profile(
            X_train, X_test,
            preprocessor=preprocessor,
            run_name="feature-profile-catboost",
        )

        # Model store
        ms = ModelStore()
        ms_run_id = ms.log_training_run(
            model=trainer.model,
            params=trainer.params,
            metrics=metrics,
            feature_importances=fi,
            X_train=X_train,
            preprocessor=preprocessor,
            run_name="catboost-raw-rmse",
            tags={"feature_store_run": fs_run_id},
        )

        # Register model
        version = ms.register_model(ms_run_id)
        print(f"MLflow run IDs → features: {fs_run_id}, model: {ms_run_id}")

    except Exception as e:
        print(f"[MLflow] Logging skipped ({e})")

    # ── 7. Evidently: data drift & concept drift reports ────────────────────
    try:
        from performance_tracking.data_drift import DataDriftMonitor
        from performance_tracking.concept_drift import ConceptDriftMonitor

        print("\n── Evidently Monitoring ──────────────────────────────────")

        # Data drift: compare train vs test feature distributions
        drift_monitor = DataDriftMonitor(
            reference_data=X_train,
            cat_features=preprocessor.cat_feature_names,
            num_features=preprocessor.num_feature_names,
        )
        drift_snap = drift_monitor.run(current_data=X_test)
        drift_monitor.save_html(drift_snap, "data_drift_report.html")
        drift_monitor.save_json(drift_snap, "data_drift_report.json")

        # Concept drift: compare model performance ref vs current
        ref_pred_train = trainer.predict(X_train)
        ref_df = ConceptDriftMonitor.build_monitor_dataframe(
            X_train, y_train, ref_pred_train,
        )
        cur_df = ConceptDriftMonitor.build_monitor_dataframe(
            X_test, y_test, y_pred,
        )
        concept_monitor = ConceptDriftMonitor(reference_data=ref_df)
        concept_snap = concept_monitor.run(current_data=cur_df)
        concept_monitor.save_html(concept_snap, "concept_drift_report.html")
        concept_monitor.save_json(concept_snap, "concept_drift_report.json")

        print("Evidently reports saved to reports/")

    except Exception as e:
        print(f"[Evidently] Monitoring skipped ({e})")

    print(f"\nPipeline complete. Artifacts saved to: {MODEL_SAVE_PATH.parent}")
