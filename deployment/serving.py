from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Flight Fare Prediction API")


class PredictRequest(BaseModel):
	# Keep this generic until feature schema is finalized
	features: Dict[str, Any]


class PredictResponse(BaseModel):
	prediction: Optional[float] = None
	details: Dict[str, Any] = {}


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
	# Placeholder: wire to your trained model when available.
	# This keeps the service runnable in Docker immediately.
	return PredictResponse(prediction=None, details={"received_features": payload.features})
