"""
Preprocessing module for the Flight Fare Prediction pipeline.

Handles all data loading, feature engineering, encoding, and train/test splitting
exactly as performed in the prototype notebook. The best model (CatBoost with raw
target and native categoricals) relies on:
  - Advanced feature engineering (date, cyclic, interaction, booking-window features)
  - Frequency encoding (leak-safe, fitted on training data only)
  - OrdinalEncoder for categorical columns (CatBoost handles them natively)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from sklearn.preprocessing import OrdinalEncoder


class FlightFarePreprocessor:
    """End-to-end preprocessor for the Bangladesh flight fare dataset."""

    # ── Column constants ────────────────────────────────────────────────────
    TARGET = "Total Fare (BDT)"
    LEAKY_FEATURES = ["Base Fare (BDT)", "Tax & Surcharge (BDT)"]
    REDUNDANT_FEATURES = ["Source Name", "Destination Name", "Arrival Date & Time"]
    CAT_FEATURES = [
        "Airline", "Source", "Destination", "Class",
        "Stopovers", "Aircraft Type", "Booking Source",
    ]
    DATE_FEATURES = ["Departure Date & Time"]

    # Categorical columns after feature engineering (includes engineered ones)
    ADV_CAT_COLS = [
        "Airline", "Source", "Destination", "Class", "Stopovers",
        "Aircraft Type", "Booking Source", "Seasonality",
        "Route", "Airline_Class", "Booking_Window", "Haul_Type",
    ]

    # Frequency-encoding source columns
    FREQ_COLS = ["Route", "Airline", "Airline_Class", "Destination"]

    def __init__(self) -> None:
        self._ordinal_encoder: Optional[OrdinalEncoder] = None
        self._freq_maps: Dict[str, pd.Series] = {}
        self._cat_indices: Optional[List[int]] = None
        self._adv_cat_cols: List[str] = []
        self._adv_num_cols: List[str] = []
        self._is_fitted: bool = False

    # ── Data loading ────────────────────────────────────────────────────────
    @staticmethod
    def load_data(csv_path: str | Path) -> pd.DataFrame:
        """Load the raw CSV dataset and return a DataFrame."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path} | shape={df.shape}")
        return df

    # ── Feature engineering ─────────────────────────────────────────────────
    def advanced_feature_engineering(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Build a rich feature matrix from the raw flight data.
        All transformations are deterministic and leak-safe.
        """
        df = df_raw.copy()

        # ── Date features ───────────────────────────────────────────────────
        dt = pd.to_datetime(df["Departure Date & Time"], errors="coerce")
        df["Month"] = dt.dt.month
        df["DayOfWeek"] = dt.dt.dayofweek
        df["Hour"] = dt.dt.hour
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["Is_Peak"] = df["Month"].isin([1, 4, 5, 6, 12]).astype(int)

        # ── Interaction features ────────────────────────────────────────────
        df["Route"] = df["Source"] + "__" + df["Destination"]
        df["Airline_Class"] = df["Airline"] + "__" + df["Class"]

        # ── Numeric transforms ──────────────────────────────────────────────
        dur = df["Duration (hrs)"].clip(lower=0)
        df["Log_Duration"] = np.log1p(dur)
        df["Duration_Sq"] = dur ** 2

        dbd = df["Days Before Departure"].clip(lower=0)
        df["Log_DaysBefore"] = np.log1p(dbd)

        # Booking window categories
        df["Booking_Window"] = pd.cut(
            dbd,
            bins=[-1, 3, 7, 14, 30, 60, 999],
            labels=["last_min", "3_7d", "1_2wk", "2_4wk", "1_2mo", "2mo_plus"],
        ).astype(str)

        # Duration haul category
        df["Haul_Type"] = pd.cut(
            dur,
            bins=[-1, 2, 5, 10, 999],
            labels=["short", "medium", "long", "ultra_long"],
        ).astype(str)

        # Is direct flag
        df["Is_Direct"] = (df["Stopovers"] == "Direct").astype(int)

        # ── Drop columns not used in modelling ──────────────────────────────
        drop = (
            [self.TARGET]
            + self.LEAKY_FEATURES
            + self.REDUNDANT_FEATURES
            + self.DATE_FEATURES
        )
        df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

        return df

    # ── Frequency encoding ──────────────────────────────────────────────────
    def _fit_frequency_encoding(self, df: pd.DataFrame) -> None:
        """Compute and store normalised value-count maps on training data."""
        self._freq_maps = {}
        for col in self.FREQ_COLS:
            if col in df.columns:
                self._freq_maps[col] = df[col].value_counts(normalize=True)

    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map pre-fitted frequency values into new columns."""
        out = df.copy()
        for col, freq in self._freq_maps.items():
            out[f"{col}_freq"] = df[col].map(freq).fillna(0)
        return out

    # ── Public fit / transform API ──────────────────────────────────────────
    def fit_transform(
        self,
        df_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline on **training** data.

        Returns
        -------
        X_train : pd.DataFrame
            Feature matrix with categorical columns cast to ``category`` dtype
            (ready for CatBoost with ``cat_features``).
        y_train : pd.Series
            Target values.
        """
        y = df_raw[self.TARGET].copy()

        # 1. Advanced feature engineering
        X = self.advanced_feature_engineering(df_raw)

        # 2. Identify column types
        self._adv_cat_cols = [c for c in self.ADV_CAT_COLS if c in X.columns]
        self._adv_num_cols = [c for c in X.columns if c not in self._adv_cat_cols]

        # 3. Fit & apply frequency encoding (leak-safe)
        self._fit_frequency_encoding(X)
        X = self._apply_frequency_encoding(X)

        # Re-derive numeric list after adding freq columns
        self._adv_num_cols = [c for c in X.columns if c not in self._adv_cat_cols]

        # 4. Cast categoricals to ``category`` dtype for CatBoost
        for col in self._adv_cat_cols:
            X[col] = X[col].astype(str).astype("category")

        # 5. Record categorical column indices (needed by CatBoost)
        self._cat_indices = [X.columns.get_loc(c) for c in self._adv_cat_cols]

        self._is_fitted = True
        print(
            f"Preprocessor fitted — {X.shape[1]} features "
            f"({len(self._adv_cat_cols)} cat + {len(self._adv_num_cols)} num)"
        )
        return X, y

    def transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply the same preprocessing pipeline to **test / inference** data.

        The frequency maps and column schema from ``fit_transform`` are re-used.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        y = df_raw[self.TARGET].copy() if self.TARGET in df_raw.columns else None

        X = self.advanced_feature_engineering(df_raw)
        X = self._apply_frequency_encoding(X)

        for col in self._adv_cat_cols:
            X[col] = X[col].astype(str).astype("category")

        return (X, y) if y is not None else (X, None)

    # ── Train / test split ──────────────────────────────────────────────────
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        temporal: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the raw DataFrame into train / test partitions.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset **before** feature engineering.
        test_size : float
            Fraction reserved for test set (default 0.2).
        temporal : bool
            If True (default), sort by departure date and take the last
            ``test_size`` fraction as the test set (walk-forward).
            If False, use a random 80/20 split.

        Returns
        -------
        train_df, test_df : (pd.DataFrame, pd.DataFrame)
        """
        if temporal:
            df = df.copy()
            df["_sort_dt"] = pd.to_datetime(df["Departure Date & Time"])
            df = df.sort_values("_sort_dt").drop(columns=["_sort_dt"]).reset_index(drop=True)

        split_idx = int(len(df) * (1 - test_size))
        if temporal:
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            train_df = df.sample(frac=1 - test_size, random_state=42)
            test_df = df.drop(train_df.index)

        print(f"Split → Train: {len(train_df):,}  |  Test: {len(test_df):,}")
        return train_df, test_df

    # ── Accessors ───────────────────────────────────────────────────────────
    @property
    def cat_feature_indices(self) -> List[int]:
        """Categorical column indices for CatBoost ``cat_features`` parameter."""
        if self._cat_indices is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self._cat_indices

    @property
    def cat_feature_names(self) -> List[str]:
        """Categorical column names used after feature engineering."""
        return list(self._adv_cat_cols)

    @property
    def num_feature_names(self) -> List[str]:
        """Numeric column names used after feature engineering."""
        return list(self._adv_num_cols)
