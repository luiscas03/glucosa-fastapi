from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent
PROJECT_DIR = MODELS_DIR.parent

load_dotenv(MODELS_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glucose_regression_v2")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEY = os.getenv("API_KEY")

MODEL_PATH = Path(os.getenv("GLUCOSE_V2_MODEL_PATH", BASE_DIR / "glucose_model_v2_clean.pkl"))
SCALER_PATH = Path(os.getenv("GLUCOSE_V2_SCALER_PATH", BASE_DIR / "scaler_v2_clean.pkl"))
FEATURES_PATH = Path(os.getenv("GLUCOSE_V2_FEATURES_PATH", BASE_DIR / "features_v2.txt"))

model: Any = None
scaler: Any = None
feature_names: List[str] = []


class GlucoseRegressionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    food_time_since: Optional[float] = None
    food_has_recent: Optional[float] = None
    food_carbs: Optional[float] = None
    food_protein: Optional[float] = None
    food_gi: Optional[float] = None
    food_size: Optional[float] = None
    food_absorption: Optional[float] = None
    demo_age_norm: Optional[float] = None
    demo_gender: Optional[float] = None
    demo_bmi_norm: Optional[float] = None
    demo_circ_norm: Optional[float] = None
    demo_antihypertensive: Optional[float] = None
    demo_family_diabetes: Optional[float] = None
    demo_glucose_history: Optional[float] = None
    demo_activity: Optional[float] = None
    demo_diet: Optional[float] = None
    demo_hba1c_norm: Optional[float] = None
    glucose_lag1: Optional[float] = None
    glucose_lag2: Optional[float] = None
    glucose_lag3: Optional[float] = None
    glucose_delta_1: Optional[float] = None
    glucose_delta_2: Optional[float] = None
    glucose_ma3_past: Optional[float] = None
    glucose_std3_past: Optional[float] = None


app = FastAPI(
    title="Glucose Regression V2 API",
    version="2.0.0",
    description="Servicio separado para glucose_model_v2_clean.pkl",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")
    return api_key


def load_feature_names() -> List[str]:
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"No existe el archivo de features: {FEATURES_PATH}")
    return [line.strip() for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Valor invalido para feature: {value!r}") from exc


def build_input_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {feature: to_float(payload.get(feature, 0.0)) for feature in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def predict_glucose(payload: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    X = build_input_frame(payload)
    X_input = scaler.transform(X) if scaler is not None else X

    try:
        raw_prediction = model.predict(X_input)[0]
    except Exception as exc:
        logger.error("Error ejecutando prediccion V2: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la prediccion: {exc}") from exc

    glucose_value = float(raw_prediction)
    if glucose_value < 100:
        category = "Normal"
    elif glucose_value < 126:
        category = "Prediabetes"
    else:
        category = "Diabetes"

    return {
        "prediction": round(glucose_value, 2),
        "category": category,
        "model_name": MODEL_PATH.name,
        "features_used": feature_names,
        "input_data": {feature: float(X.iloc[0][feature]) for feature in feature_names},
    }


@app.on_event("startup")
def startup_event() -> None:
    global model, scaler, feature_names

    feature_names = load_feature_names()

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe el modelo V2: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info("Modelo V2 cargado desde %s", MODEL_PATH)

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler V2 cargado desde %s", SCALER_PATH)
    else:
        scaler = None
        logger.warning("No existe scaler V2 en %s; se predecira sin escalado", SCALER_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "not_ready",
        "service": "glucose_regression_v2",
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "features_path": str(FEATURES_PATH),
        "feature_count": len(feature_names),
    }


@app.get("/features")
def get_features() -> Dict[str, Any]:
    return {
        "features": feature_names,
        "total": len(feature_names),
    }


@app.post("/predict", dependencies=[Security(get_api_key)])
def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return predict_glucose(payload)


@app.post("/predict_typed", dependencies=[Security(get_api_key)])
def predict_typed(item: GlucoseRegressionInput) -> Dict[str, Any]:
    return predict_glucose(item.model_dump())
