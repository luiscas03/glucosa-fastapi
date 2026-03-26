from __future__ import annotations

import json
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict


MODEL_VARIANT = "cluster_0"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent
PROJECT_DIR = MODELS_DIR.parent

load_dotenv(MODELS_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("postprandial_cluster_0")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEY = os.getenv("API_KEY")

METADATA_PATH = BASE_DIR / "deploy_metadata.json"
MODEL_PATH = BASE_DIR / "model_cluster_0.joblib"
ISO_PATH = BASE_DIR / "iso_cluster_0.joblib"

metadata: Dict[str, Any] = {}
model: Any = None
calibrator: Any = None


def install_sklearn_pickle_compat() -> None:
    """Alias old private sklearn pickle module names to current locations."""
    if "_loss" not in sys.modules:
        try:
            sys.modules["_loss"] = importlib.import_module("sklearn._loss._loss")
        except Exception:
            logger.warning("No se pudo registrar alias de compatibilidad para _loss", exc_info=True)


class PeakRiskInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: Dict[str, float]


app = FastAPI(
    title="Postprandial Cluster 0 API",
    version="1.0.0",
    description="Servicio individual para model_cluster_0.joblib",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    global metadata, model, calibrator

    install_sklearn_pickle_compat()

    if not METADATA_PATH.exists():
        raise RuntimeError(f"No existe el metadata del modelo: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe el modelo: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(ISO_PATH) if ISO_PATH.exists() else None


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")
    return api_key


def get_features() -> List[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(metadata.get("m1_features", []))


def score_model(payload: Dict[str, float]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    features = get_features()
    missing = [name for name in features if name not in payload]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing, "model_variant": MODEL_VARIANT})

    X = pd.DataFrame([{name: float(payload[name]) for name in features}], columns=features)

    try:
        if hasattr(model, "predict_proba"):
            raw_score = float(model.predict_proba(X)[0][1])
        else:
            raw_score = float(model.predict(X)[0])
    except Exception as exc:
        logger.error("Error prediciendo con %s: %s", MODEL_VARIANT, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la prediccion: {exc}") from exc

    calibrated_score = raw_score
    if calibrator is not None:
        try:
            calibrated_score = float(np.asarray(calibrator.predict([raw_score])).ravel()[0])
        except Exception:
            logger.warning("No se pudo aplicar calibracion isotonic", exc_info=True)

    if calibrated_score < 0.20:
        risk_band = "LOW"
    elif calibrated_score < 0.30:
        risk_band = "MEDIUM"
    else:
        risk_band = "HIGH"

    return {
        "model_variant": MODEL_VARIANT,
        "target": metadata.get("target"),
        "risk_score_raw": round(raw_score, 4),
        "risk_score_calibrated": round(calibrated_score, 4),
        "risk_band": risk_band,
        "features_used": features,
        "input_data": {col: float(X.iloc[0][col]) for col in X.columns},
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "not_ready",
        "service": "postprandial_cluster_0",
        "model_variant": MODEL_VARIANT,
        "metadata_loaded": bool(metadata),
        "calibrator_loaded": calibrator is not None,
        "target": metadata.get("target"),
    }


@app.get("/metadata")
def get_metadata():
    return metadata


@app.get("/features")
def features():
    return {
        "model_variant": MODEL_VARIANT,
        "features": get_features(),
        "total": len(get_features()),
    }


@app.post("/predict", dependencies=[Security(get_api_key)])
def predict(item: PeakRiskInput):
    return score_model(item.features)
