"""
Evaluation module for the Flight Fare Prediction pipeline.

Computes regression metrics (R², MAE, RMSE, MAPE, MedAE), segment-level
breakdowns, and diagnostic plots — mirroring the notebook analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


class FlightFareEvaluator:
    """Evaluate regression predictions for the flight fare model."""

    # ── Core metrics ────────────────────────────────────────────────────────
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute standard regression metrics.

        Returns
        -------
        dict with keys: R2, MAE, RMSE, MAPE, MedAE
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        medae = median_absolute_error(y_true, y_pred)
        mape = float(np.mean(np.abs(y_true - y_pred) / np.clip(y_true, 1, None)) * 100)

        return {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "MedAE": medae,
        }

    # ── Segment-level evaluation ────────────────────────────────────────────
    @staticmethod
    def evaluate_by_fare_segment(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        bins: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute metrics per fare-value segment.

        Parameters
        ----------
        bins : list[float]
            Bin edges for ``pd.cut`` (default: <10K, 10–25K, 25–50K, 50–100K, 100–200K, >200K).
        labels : list[str]
            Human-readable segment labels matching ``bins``.

        Returns
        -------
        pd.DataFrame  with one row per segment.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if bins is None:
            bins = [0, 10_000, 25_000, 50_000, 100_000, 200_000, float("inf")]
        if labels is None:
            labels = ["<10K", "10-25K", "25-50K", "50-100K", "100-200K", ">200K"]

        segments = pd.cut(y_true, bins=bins, labels=labels)

        rows = []
        for label in labels:
            mask = segments == label
            n = mask.sum()
            if n < 2:
                continue
            seg_true = y_true[mask]
            seg_pred = y_pred[mask]
            rows.append(
                {
                    "Segment": label,
                    "Count": int(n),
                    "Mean Fare": float(seg_true.mean()),
                    "R2": r2_score(seg_true, seg_pred) if n > 10 else float("nan"),
                    "MAE": mean_absolute_error(seg_true, seg_pred),
                    "MAPE%": float(
                        np.mean(np.abs(seg_true - seg_pred) / np.clip(seg_true, 1, None)) * 100
                    ),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_by_column(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        group_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute metrics grouped by an arbitrary categorical Series
        (e.g. ``X_test['Class']`` or ``X_test['Airline']``).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        rows = []
        for group in sorted(group_series.unique()):
            mask = (group_series == group).values
            n = mask.sum()
            if n < 2:
                continue
            rows.append(
                {
                    "Group": group,
                    "Count": int(n),
                    "Mean Fare": float(y_true[mask].mean()),
                    "R2": r2_score(y_true[mask], y_pred[mask]) if n > 10 else float("nan"),
                    "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
                }
            )
        return pd.DataFrame(rows)

    # ── Pretty-print ────────────────────────────────────────────────────────
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results") -> None:
        """Print metrics in a formatted box."""
        print(f"\n{'═' * 50}")
        print(f"  {title}")
        print(f"{'═' * 50}")
        print(f"  R²   : {metrics['R2']:.4f}")
        print(f"  MAE  : {metrics['MAE']:>10,.0f} BDT")
        print(f"  RMSE : {metrics['RMSE']:>10,.0f} BDT")
        print(f"  MAPE : {metrics['MAPE']:>10.1f} %")
        print(f"  MedAE: {metrics['MedAE']:>10,.0f} BDT")
        print(f"{'═' * 50}")

    # ── Diagnostic plots ───────────────────────────────────────────────────
    @staticmethod
    def plot_diagnostics(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        title: str = "CatBoost (raw target)",
    ) -> None:
        """
        Generate a 4-panel diagnostic figure:
          1. Actual vs Predicted scatter
          2. Residuals vs Predicted scatter
          3. Residual distribution histogram
          4. R² by fare segment bar chart
        """
        import matplotlib.pyplot as plt

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        # 1. Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.15, s=4, c="steelblue")
        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
        axes[0].plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        axes[0].set_xlabel("Actual Fare (BDT)")
        axes[0].set_ylabel("Predicted Fare (BDT)")
        axes[0].set_title("Actual vs Predicted")
        axes[0].legend()

        # 2. Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.15, s=4, c="steelblue")
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Predicted Fare (BDT)")
        axes[1].set_ylabel("Residual (BDT)")
        axes[1].set_title("Residuals vs Predicted")

        # 3. Residual distribution
        axes[2].hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
        axes[2].axvline(x=0, color="red", linestyle="--", linewidth=1)
        axes[2].axvline(
            x=np.median(residuals),
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Median: {np.median(residuals):,.0f}",
        )
        axes[2].set_xlabel("Error (BDT)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Residual Distribution")
        axes[2].legend()

        # 4. R² by fare segment
        bins = [0, 20_000, 50_000, 100_000, 200_000, float("inf")]
        labels = ["<20K", "20-50K", "50-100K", "100-200K", ">200K"]
        segments = pd.cut(y_true, bins=bins, labels=labels)

        seg_r2 = []
        for label in labels:
            mask = segments == label
            n = mask.sum()
            if n > 10:
                seg_r2.append(r2_score(y_true[mask], y_pred[mask]))
            else:
                seg_r2.append(0.0)

        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in seg_r2]
        bars = axes[3].barh(labels, seg_r2, color=colors, alpha=0.8)
        axes[3].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        for bar, r in zip(bars, seg_r2):
            axes[3].text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{r:.2f}",
                va="center",
                fontsize=9,
            )
        axes[3].set_xlabel("R²")
        axes[3].set_title("R² by Fare Segment")

        fig.suptitle(
            f"Best Model: {title} — R²={r2:.4f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
