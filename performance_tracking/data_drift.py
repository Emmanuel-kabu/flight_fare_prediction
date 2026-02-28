"""
Data Drift Monitoring — Evidently v0.7+ API.

Detects feature-level distribution shifts between a reference dataset
(training) and a current dataset (production / new batch) using Evidently's
``DataDriftPreset``.  Generates HTML reports and JSON snapshots that can be
served by the Evidently UI or inspected programmatically.

Usage
-----
    from performance_tracking.data_drift import DataDriftMonitor

    monitor = DataDriftMonitor(reference_data=X_train)
    snapshot = monitor.run(current_data=X_new)
    monitor.save_html(snapshot, "reports/data_drift_report.html")
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evidently import DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import DriftedColumnsCount, ValueDrift


class DataDriftMonitor:
    """
    Monitors feature-level data drift between reference and current data.

    Produces Evidently HTML reports and JSON results that quantify:
    * Per-feature drift (stat test, p-value, drift detected flag)
    * Dataset-level drift score (share of drifted features)
    * Data quality summary (nulls, duplicates, new/missing categories)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        cat_features: Optional[List[str]] = None,
        num_features: Optional[List[str]] = None,
        reports_dir: str | Path = "reports/data_drift",
    ) -> None:
        """
        Parameters
        ----------
        reference_data : pd.DataFrame
            The training / baseline feature matrix to compare against.
        cat_features : list[str], optional
            Categorical column names (auto-detected if omitted).
        num_features : list[str], optional
            Numerical column names (auto-detected if omitted).
        reports_dir : str or Path
            Directory where generated reports are stored.
        """
        self._reference = self._prepare_df(reference_data)
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Build DataDefinition for Evidently v0.7+
        self._data_definition = DataDefinition(
            categorical_columns=cat_features,
            numerical_columns=num_features,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, current_data: pd.DataFrame):
        """
        Run a data-drift report comparing current data to the reference.

        Parameters
        ----------
        current_data : pd.DataFrame
            New / production data to test for drift.

        Returns
        -------
        evidently Snapshot
        """
        current = self._prepare_df(current_data)

        report = Report(
            metrics=[DataDriftPreset()],
        )
        snapshot = report.run(
            current_data=current,
            reference_data=self._reference,
        )
        return snapshot

    def run_with_quality(self, current_data: pd.DataFrame):
        """
        Run a combined data-drift + data-quality report.

        Returns
        -------
        evidently Snapshot
        """
        current = self._prepare_df(current_data)

        report = Report(
            metrics=[
                DataDriftPreset(),
                DataSummaryPreset(),
            ],
        )
        snapshot = report.run(
            current_data=current,
            reference_data=self._reference,
        )
        return snapshot

    # ── Persistence helpers ─────────────────────────────────────────────────

    def save_html(self, snapshot, filename: Optional[str] = None) -> Path:
        """Save the report as a standalone HTML file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_drift_{ts}.html"
        path = self._reports_dir / filename
        snapshot.save_html(str(path))
        print(f"[DataDrift] HTML report saved → {path}")
        return path

    def save_json(self, snapshot, filename: Optional[str] = None) -> Path:
        """Save the report as a JSON snapshot."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_drift_{ts}.json"
        path = self._reports_dir / filename
        snapshot.save_json(str(path))
        print(f"[DataDrift] JSON snapshot saved → {path}")
        return path

    # ── Programmatic result extraction ──────────────────────────────────────

    @staticmethod
    def get_drift_summary(snapshot) -> Dict[str, Any]:
        """
        Extract a concise summary from a drift report snapshot.

        Returns
        -------
        dict with keys:
            dataset_drift   : bool
            drift_share     : float
            n_drifted       : int
            n_features      : int
            drifted_features: list[str]
        """
        result = snapshot.dict()

        drift_info: Dict[str, Any] = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "n_drifted": 0,
            "n_features": 0,
            "drifted_features": [],
        }

        # Walk through metric results
        for metric_result in result.get("metric_results", result.get("metrics", [])):
            metric_id = str(metric_result.get("type", metric_result.get("metric", "")))

            # DriftedColumnsCount gives dataset-level summary
            if "DriftedColumnsCount" in metric_id:
                mr = metric_result.get("value", metric_result.get("result", {}))
                if isinstance(mr, dict):
                    drift_info["n_drifted"] = mr.get("count", mr.get("number_of_drifted_columns", 0))
                elif isinstance(mr, (int, float)):
                    drift_info["n_drifted"] = int(mr)

            # ValueDrift per column
            if "ValueDrift" in metric_id:
                mr = metric_result.get("value", metric_result.get("result", {}))
                col = metric_result.get("column", metric_result.get("column_name", ""))
                drifted = False
                if isinstance(mr, dict):
                    drifted = mr.get("drift_detected", False)
                if drifted and col:
                    drift_info["drifted_features"].append(col)

        # Compute derived fields
        total = drift_info.get("n_features", 0)
        n_drifted = drift_info.get("n_drifted", len(drift_info["drifted_features"]))
        if total > 0:
            drift_info["drift_share"] = n_drifted / total
        drift_info["dataset_drift"] = drift_info["drift_share"] > 0.3

        return drift_info

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert category columns to string for Evidently compatibility."""
        out = df.copy()
        for col in out.columns:
            if out[col].dtype.name == "category":
                out[col] = out[col].astype(str)
        return out
