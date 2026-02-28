"""
FastAPI backend for Flight Fare Prediction.

Loads the trained model (champion from continuous training) and fitted
preprocessor from pickle artifacts, exposes:
  - /predict  — returns predicted fare in BDT
  - /reload   — hot-swap model artifacts after retraining (POST)
  - /health   — readiness probe
  - /options  — dropdown values for the frontend
"""

from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Resolve project root & add to sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.training.preprocessing import FlightFarePreprocessor

# ── Paths ───────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# Champion model path (written by ContinuousTrainer.save_champion)
# Falls back to catboost_model.pkl for backward compatibility
CHAMPION_MODEL_PATH = ARTIFACTS_DIR / "champion_model.pkl"
LEGACY_MODEL_PATH = ARTIFACTS_DIR / "catboost_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


# ── Load artifacts ──────────────────────────────────────────────────────────
def _load_artifacts():
    """Load model and preprocessor from disk. Returns (model, preprocessor, model_info)."""
    model_path = CHAMPION_MODEL_PATH if CHAMPION_MODEL_PATH.exists() else LEGACY_MODEL_PATH

    if not model_path.exists() or not PREPROCESSOR_PATH.exists():
        print(
            f"[WARNING] Artifacts not found at {ARTIFACTS_DIR}. "
            "The /predict endpoint will return 503 until the training pipeline runs."
        )
        return None, None, {}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    model_info = {
        "model_path": str(model_path),
        "model_class": type(model).__name__,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[Serving] Loaded {model_info['model_class']} from {model_path}")

    return model, preprocessor, model_info


model, preprocessor, model_info = _load_artifacts()

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flight Fare Prediction API",
    description=(
        "Predict domestic & international flight fares from Bangladesh "
        "using the champion model from continuous training "
        "(RandomForest / XGBoost / LightGBM / CatBoost)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dropdown option constants (from the dataset) ───────────────────────────
AIRLINES = [
    "Air Arabia", "Air Astra", "Air India", "AirAsia",
    "Biman Bangladesh Airlines", "British Airways", "Cathay Pacific",
    "Emirates", "Etihad Airways", "FlyDubai", "Gulf Air", "IndiGo",
    "Kuwait Airways", "Lufthansa", "Malaysian Airlines", "NovoAir",
    "Qatar Airways", "Saudia", "Singapore Airlines", "SriLankan Airlines",
    "Thai Airways", "Turkish Airlines", "US-Bangla Airlines", "Vistara",
]
SOURCES = ["BZL", "CGP", "CXB", "DAC", "JSR", "RJH", "SPD", "ZYL"]
DESTINATIONS = [
    "BKK", "BZL", "CCU", "CGP", "CXB", "DAC", "DEL", "DOH",
    "DXB", "IST", "JED", "JFK", "JSR", "KUL", "LHR", "RJH",
    "SIN", "SPD", "YYZ", "ZYL",
]
CLASSES = ["Economy", "Business", "First Class"]
STOPOVERS = ["Direct", "1 Stop", "2 Stops"]
AIRCRAFT_TYPES = [
    "Airbus A320", "Airbus A350", "Boeing 737", "Boeing 777", "Boeing 787",
]
BOOKING_SOURCES = ["Direct Booking", "Online Website", "Travel Agency"]
SEASONALITIES = ["Regular", "Eid", "Hajj", "Winter Holidays"]


# ── Request / Response schemas ──────────────────────────────────────────────
class FlightInput(BaseModel):
    """Raw flight details — mirrors the original dataset columns."""

    airline: str = Field(..., examples=["Biman Bangladesh Airlines"])
    source: str = Field(..., examples=["DAC"])
    destination: str = Field(..., examples=["CXB"])
    departure_datetime: str = Field(
        ...,
        description="ISO format: YYYY-MM-DD HH:MM:SS",
        examples=["2026-03-15 08:30:00"],
    )
    duration_hrs: float = Field(..., ge=0, examples=[1.5])
    stopovers: str = Field(..., examples=["Direct"])
    aircraft_type: str = Field(..., examples=["Boeing 737"])
    travel_class: str = Field(..., examples=["Economy"])
    booking_source: str = Field(..., examples=["Online Website"])
    seasonality: str = Field(..., examples=["Regular"])
    days_before_departure: int = Field(..., ge=0, examples=[15])


class PredictionResponse(BaseModel):
    predicted_fare_bdt: float
    input_summary: Dict[str, Any]


class OptionsResponse(BaseModel):
    airlines: List[str]
    sources: List[str]
    destinations: List[str]
    classes: List[str]
    stopovers: List[str]
    aircraft_types: List[str]
    booking_sources: List[str]
    seasonalities: List[str]


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_class": model_info.get("model_class", "N/A"),
        "loaded_at": model_info.get("loaded_at", "N/A"),
    }


@app.post("/reload")
def reload_model():
    """
    Hot-swap model artifacts after retraining.
    Called by the continuous training DAG after champion is saved.
    """
    global model, preprocessor, model_info
    try:
        new_model, new_preprocessor, new_info = _load_artifacts()
        if new_model is None or new_preprocessor is None:
            raise HTTPException(
                status_code=503,
                detail="Artifacts not found. Cannot reload.",
            )
        model = new_model
        preprocessor = new_preprocessor
        model_info = new_info
        return {
            "status": "reloaded",
            "model_class": new_info.get("model_class"),
            "loaded_at": new_info.get("loaded_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.get("/options", response_model=OptionsResponse)
def get_options():
    """Return all valid dropdown options for the frontend."""
    return OptionsResponse(
        airlines=AIRLINES,
        sources=SOURCES,
        destinations=DESTINATIONS,
        classes=CLASSES,
        stopovers=STOPOVERS,
        aircraft_types=AIRCRAFT_TYPES,
        booking_sources=BOOKING_SOURCES,
        seasonalities=SEASONALITIES,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlightInput):
    """
    Accept raw flight details, run preprocessing, and return predicted fare.
    """
    global model, preprocessor, model_info
    # Lazy-load: try loading artifacts if they weren't available at startup
    if model is None or preprocessor is None:
        model, preprocessor, model_info = _load_artifacts()
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run the training pipeline first.",
        )
    try:
        # Build a single-row DataFrame matching the raw dataset schema
        row = {
            "Airline": payload.airline,
            "Source": payload.source,
            "Destination": payload.destination,
            "Departure Date & Time": payload.departure_datetime,
            "Duration (hrs)": payload.duration_hrs,
            "Stopovers": payload.stopovers,
            "Aircraft Type": payload.aircraft_type,
            "Class": payload.travel_class,
            "Booking Source": payload.booking_source,
            "Seasonality": payload.seasonality,
            "Days Before Departure": payload.days_before_departure,
            # Columns the preprocessor drops — provide dummies
            "Source Name": "",
            "Destination Name": "",
            "Arrival Date & Time": payload.departure_datetime,
            "Base Fare (BDT)": 0.0,
            "Tax & Surcharge (BDT)": 0.0,
            "Total Fare (BDT)": 0.0,
        }
        df = pd.DataFrame([row])

        # Preprocess (uses fitted frequency maps from training)
        X, _ = preprocessor.transform(df)

        # Predict
        prediction = float(model.predict(X)[0])
        prediction = max(prediction, 0.0)

        return PredictionResponse(
            predicted_fare_bdt=round(prediction, 2),
            input_summary={
                "airline": payload.airline,
                "route": f"{payload.source} → {payload.destination}",
                "class": payload.travel_class,
                "departure": payload.departure_datetime,
                "days_before": payload.days_before_departure,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run with: uvicorn deployment.serving:app --reload ───────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deployment.serving:app", host="0.0.0.0", port=8000, reload=True)
