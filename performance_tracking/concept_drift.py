"""
Concept Drift / Model Performance Monitoring — Evidently v0.7+ API.

Detects model performance degradation (concept drift) by comparing
prediction quality metrics between a reference window and a current window.
Uses Evidently's ``RegressionPreset`` for regression-specific diagnostics.

Concept drift occurs when the relationship between features and target
changes over time, even if the feature distributions remain stable.

Usage
-----
    from performance_tracking.concept_drift import ConceptDriftMonitor

    monitor = ConceptDriftMonitor(
        reference_data=ref_df,     # df with features + 'prediction' + 'target'
    )
    snapshot = monitor.run(current_data=current_df)
    monitor.save_html(snapshot)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import RegressionPreset
from evidently.metrics import MAE, MAPE, RMSE, R2Score, MeanError


class ConceptDriftMonitor:
    """
    Monitors model performance drift for regression tasks.

    Compares prediction quality (R², MAE, RMSE, MAPE, error distribution)
    between a reference period and a current period.  A significant drop
    signals concept drift — the model's learned patterns no longer hold.
    """

    DEFAULT_TARGET_COL = "target"
    DEFAULT_PREDICTION_COL = "prediction"

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_col: str = "target",
        prediction_col: str = "prediction",
        cat_features: Optional[List[str]] = None,
        num_features: Optional[List[str]] = None,
        reports_dir: str | Path = "reports/concept_drift",
    ) -> None:
        """
        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline data containing features, a ``target`` column, and a
            ``prediction`` column.
        target_col : str
            Name of the actual-value column.
        prediction_col : str
            Name of the model-prediction column.
        cat_features, num_features : list[str], optional
            Explicit feature lists for column definition.
        reports_dir : str or Path
            Output directory for HTML/JSON reports.
        """
        self._target_col = target_col
        self._prediction_col = prediction_col
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Build DataDefinition for Evidently v0.7+
        self._data_definition = DataDefinition(
            regression=[
                Regression(
                    target=target_col,
                    prediction=prediction_col,
                ),
            ],
            categorical_columns=cat_features,
            numerical_columns=num_features,
        )

        # Store reference as an Evidently Dataset with the definition
        self._reference = Dataset.from_pandas(
            self._prepare_df(reference_data),
            data_definition=self._data_definition,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, current_data: pd.DataFrame):
        """
        Run a regression performance report comparing current vs reference.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current-period data with the same schema as reference_data.

        Returns
        -------
        evidently Snapshot
        """
        current_ds = Dataset.from_pandas(
            self._prepare_df(current_data),
            data_definition=self._data_definition,
        )

        report = Report(
            metrics=[RegressionPreset()],
        )
        snapshot = report.run(
            current_data=current_ds,
            reference_data=self._reference,
        )
        return snapshot

    def run_detailed(self, current_data: pd.DataFrame):
        """
        Run a more granular report with individual regression metrics.

        Returns
        -------
        evidently Snapshot
        """
        current_ds = Dataset.from_pandas(
            self._prepare_df(current_data),
            data_definition=self._data_definition,
        )

        report = Report(
            metrics=[
                R2Score(),
                MAE(),
                RMSE(),
                MAPE(),
                MeanError(),
            ],
        )
        snapshot = report.run(
            current_data=current_ds,
            reference_data=self._reference,
        )
        return snapshot

    # ── Convenience: build monitor dataframes ───────────────────────────────

    @staticmethod
    def build_monitor_dataframe(
        X: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        target_col: str = "target",
        prediction_col: str = "prediction",
    ) -> pd.DataFrame:
        """
        Assemble a monitoring dataframe from features, actuals, and predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y_true : array-like
            Actual target values.
        y_pred : array-like
            Model predictions.

        Returns
        -------
        pd.DataFrame  ready for ``ConceptDriftMonitor``.
        """
        df = X.copy()
        for col in df.columns:
            if df[col].dtype.name == "category":
                df[col] = df[col].astype(str)
        df[target_col] = np.asarray(y_true)
        df[prediction_col] = np.asarray(y_pred)
        return df

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_html(self, snapshot, filename: Optional[str] = None) -> Path:
        """Save the report as a standalone HTML file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"concept_drift_{ts}.html"
        path = self._reports_dir / filename
        snapshot.save_html(str(path))
        print(f"[ConceptDrift] HTML report saved → {path}")
        return path

    def save_json(self, snapshot, filename: Optional[str] = None) -> Path:
        """Save the report as a JSON snapshot."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"concept_drift_{ts}.json"
        path = self._reports_dir / filename
        snapshot.save_json(str(path))
        print(f"[ConceptDrift] JSON snapshot saved → {path}")
        return path

    # ── Programmatic result extraction ──────────────────────────────────────

    @staticmethod
    def get_performance_summary(snapshot) -> Dict[str, Any]:
        """
        Extract a performance comparison summary from the report snapshot.

        Returns
        -------
        dict with keys:
            reference : dict of regression metrics
            current   : dict of regression metrics
            degraded  : bool  (True if R² dropped > 10%)
        """
        result = snapshot.dict()
        summary: Dict[str, Any] = {
            "reference": {},
            "current": {},
            "degraded": False,
        }

        # Walk through metric results
        for metric_result in result.get("metric_results", result.get("metrics", [])):
            metric_type = str(metric_result.get("type", metric_result.get("metric", "")))
            value = metric_result.get("value", metric_result.get("result", {}))

            # Map metric types to friendly names
            metric_map = {
                "R2Score": "r2",
                "MAE": "mae",
                "RMSE": "rmse",
                "MAPE": "mape",
                "MeanError": "mean_error",
            }

            for key, name in metric_map.items():
                if key in metric_type:
                    if isinstance(value, dict):
                        summary["current"][name] = value.get("value", value.get("current"))
                        summary["reference"][name] = value.get("reference")
                    elif isinstance(value, (int, float)):
                        summary["current"][name] = value

        # Detect degradation: R² dropped > 10% compared to reference
        ref_r2 = summary["reference"].get("r2")
        cur_r2 = summary["current"].get("r2")
        if ref_r2 is not None and cur_r2 is not None:
            try:
                summary["degraded"] = float(cur_r2) < float(ref_r2) * 0.9
            except (TypeError, ValueError):
                pass

        return summary

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert category columns to string for Evidently compatibility."""
        out = df.copy()
        for col in out.columns:
            if out[col].dtype.name == "category":
                out[col] = out[col].astype(str)
        return out
