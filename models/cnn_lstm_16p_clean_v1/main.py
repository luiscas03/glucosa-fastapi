from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

MODEL_NAME = "clean_v1"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent
PROJECT_DIR = MODELS_DIR.parent

load_dotenv(MODELS_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnn_lstm_16p_clean_v1")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEY = os.getenv("API_KEY")

MODEL_PATH = BASE_DIR / "glucose_cnn_lstm_16p_CLEAN_v1.keras"
tf: Any = None
keras_model: Any = None
load_error: str | None = None


class FixedSequencePredictionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    values: list[float]
    shape: list[int] | None = Field(default=None)


app = FastAPI(
    title="CNN LSTM 16P Clean V1 API",
    version="1.0.0",
    description="Servicio individual para glucose_cnn_lstm_16p_CLEAN_v1.keras",
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
    global tf, keras_model, load_error
    try:
        import tensorflow as tensorflow_module
    except Exception as exc:
        tf = None
        load_error = str(exc)
        logger.warning("TensorFlow no disponible: %s", exc)
        return

    tf = tensorflow_module
    if not MODEL_PATH.exists():
        load_error = f"No existe el archivo {MODEL_PATH.name}"
        return

    try:
        keras_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        load_error = str(exc)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")
    return api_key


def predict_sequence(values: list[float], shape: list[int] | None):
    if tf is None:
        raise HTTPException(status_code=503, detail="TensorFlow no esta instalado en este entorno")
    if keras_model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {MODEL_NAME}")

    array = np.asarray(values, dtype=np.float32)
    if shape:
        expected_size = int(np.prod(shape))
        if array.size != expected_size:
            raise HTTPException(status_code=400, detail=f"values tiene {array.size} elementos y shape requiere {expected_size}")
        X = array.reshape((1, *shape))
    else:
        input_shape = getattr(keras_model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        dims = [dim for dim in input_shape[1:] if dim is not None] if input_shape else []
        if dims:
            expected_size = int(np.prod(dims))
            if array.size != expected_size:
                raise HTTPException(status_code=400, detail=f"El modelo espera {expected_size} valores para shape {dims}; recibidos {array.size}")
            X = array.reshape((1, *dims))
        else:
            X = array.reshape((1, array.size))

    prediction = keras_model.predict(X, verbose=0)
    prediction_value = float(np.asarray(prediction).ravel()[0])
    return {"model_name": MODEL_NAME, "input_shape_used": list(X.shape), "prediction": round(prediction_value, 4)}


@app.get("/health")
def health():
    return {
        "status": "ok" if tf is not None and keras_model is not None else "dependency_missing",
        "service": "cnn_lstm_16p_clean_v1",
        "model_name": MODEL_NAME,
        "tensorflow_available": tf is not None,
        "model_loaded": keras_model is not None,
        "load_error": load_error,
    }


@app.post("/predict", dependencies=[Security(get_api_key)])
def predict(item: FixedSequencePredictionInput):
    return predict_sequence(item.values, item.shape)
