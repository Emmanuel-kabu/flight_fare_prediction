"""
Feature Store — MLflow-backed feature registry.

Logs feature metadata, statistics, and transformation artefacts to an MLflow
experiment so that every training run has a reproducible record of which
features were used, how they were engineered, and what their distributions
looked like at training time.

Usage
-----
    from experiment_tracking.feature_store import FeatureStore

    fs = FeatureStore(tracking_uri="http://localhost:5000")
    fs.log_feature_profile(
        X_train, X_test,
        preprocessor=preprocessor,
        run_name="v1-baseline"
    )
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd


class FeatureStore:
    """
    Lightweight feature registry backed by MLflow.

    Records:
    * Feature schema (names, dtypes, categorical vs. numeric)
    * Descriptive statistics for every feature (mean, std, min, max, nulls)
    * Frequency-encoding maps (so they can be reproduced exactly)
    * Train / test split sizes
    * Full feature matrix sample artefact (Parquet) for lineage audits
    """

    EXPERIMENT_NAME = "flight-fare-feature-store"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tracking_uri : str, optional
            MLflow tracking server URI.
            Falls back to ``MLFLOW_TRACKING_URI`` env-var, then ``./mlruns``.
        experiment_name : str, optional
            Override the default experiment name.
        """
        self._tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        )
        mlflow.set_tracking_uri(self._tracking_uri)

        self._experiment_name = experiment_name or self.EXPERIMENT_NAME
        mlflow.set_experiment(self._experiment_name)

    # ── Public API ──────────────────────────────────────────────────────────

    def log_feature_profile(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        preprocessor: Any = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create an MLflow run that captures the full feature profile.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Feature matrices **after** preprocessing (fit_transform / transform).
        preprocessor : FlightFarePreprocessor, optional
            If provided, its internal frequency maps and column lists are
            logged as artefacts.
        run_name : str, optional
            Human-readable run name.
        tags : dict, optional
            Extra MLflow tags.

        Returns
        -------
        str   The MLflow run ID.
        """
        with mlflow.start_run(run_name=run_name or "feature-profile") as run:
            # ── Tags ────────────────────────────────────────────────────────
            mlflow.set_tag("stage", "feature_engineering")
            if tags:
                mlflow.set_tags(tags)

            # ── Params: schema overview ─────────────────────────────────────
            cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
            num_cols = [c for c in X_train.columns if c not in cat_cols]

            mlflow.log_param("num_features_total", X_train.shape[1])
            mlflow.log_param("num_categorical", len(cat_cols))
            mlflow.log_param("num_numeric", len(num_cols))
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])

            # ── Metrics: numeric feature stats ──────────────────────────────
            stats = self.compute_feature_stats(X_train, num_cols)
            for col, col_stats in stats.items():
                safe_col = self.sanitize_metric_name(col)
                for stat_name, val in col_stats.items():
                    mlflow.log_metric(f"feat_{safe_col}__{stat_name}", val)

            # ── Metrics: null counts ────────────────────────────────────────
            train_nulls = int(X_train.isnull().sum().sum())
            test_nulls = int(X_test.isnull().sum().sum())
            mlflow.log_metric("train_null_count", train_nulls)
            mlflow.log_metric("test_null_count", test_nulls)

            # ── Artefacts ───────────────────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Feature schema JSON
                schema = self.build_schema(X_train, cat_cols, num_cols)
                schema_path = Path(tmpdir) / "feature_schema.json"
                schema_path.write_text(json.dumps(schema, indent=2))
                mlflow.log_artifact(str(schema_path), artifact_path="features")

                # 2. Full descriptive stats CSV
                desc = X_train.describe(include="all").T
                desc_path = Path(tmpdir) / "feature_statistics.csv"
                desc.to_csv(desc_path)
                mlflow.log_artifact(str(desc_path), artifact_path="features")

                # 3. Frequency encoding maps (from preprocessor)
                if preprocessor and hasattr(preprocessor, "_freq_maps"):
                    freq_maps_serialisable = {
                        col: freq.to_dict()
                        for col, freq in preprocessor.freq_maps.items()
                    }
                    freq_path = Path(tmpdir) / "frequency_maps.json"
                    freq_path.write_text(json.dumps(freq_maps_serialisable, indent=2))
                    mlflow.log_artifact(str(freq_path), artifact_path="features")

                # 4. Sample data (first 200 rows, Parquet)
                sample_path = Path(tmpdir) / "train_sample.parquet"
                X_train.head(200).to_parquet(sample_path, index=False)
                mlflow.log_artifact(str(sample_path), artifact_path="features")

                # 5. Categorical value counts
                cat_dist: Dict[str, Dict[str, int]] = {}
                for col in cat_cols:
                    cat_dist[col] = (
                        X_train[col].value_counts().head(30).to_dict()
                    )
                cat_dist_path = Path(tmpdir) / "categorical_distributions.json"
                cat_dist_path.write_text(
                    json.dumps(cat_dist, indent=2, default=str)
                )
                mlflow.log_artifact(str(cat_dist_path), artifact_path="features")

            run_id = run.info.run_id
            print(
                f"[FeatureStore] Feature profile logged — "
                f"run_id={run_id}  features={X_train.shape[1]}"
            )
            return run_id

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def compute_feature_stats(
        df: pd.DataFrame, num_cols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/std/min/max/median for numeric columns."""
        stats: Dict[str, Dict[str, float]] = {}
        for col in num_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": float(series.mean()) if not series.isna().all() else 0.0,
                "std": float(series.std()) if not series.isna().all() else 0.0,
                "min": float(series.min()) if not series.isna().all() else 0.0,
                "max": float(series.max()) if not series.isna().all() else 0.0,
                "median": float(series.median()) if not series.isna().all() else 0.0,
            }
        return stats

    @staticmethod
    def sanitize_metric_name(name: str) -> str:
        """Replace chars not allowed in MLflow metric names."""
        import re
        # Keep alphanumerics, underscores, dashes, dots, spaces, slashes
        return re.sub(r"[^a-zA-Z0-9_\-\. /]", "_", name)

    @staticmethod
    def build_schema(
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
    ) -> Dict[str, Any]:
        """Build a JSON-serialisable feature schema."""
        return {
            "total_features": df.shape[1],
            "categorical_features": cat_cols,
            "numeric_features": num_cols,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "columns_ordered": list(df.columns),
        }
