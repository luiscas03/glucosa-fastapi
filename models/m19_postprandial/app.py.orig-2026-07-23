"""
M19 — Riesgo posprandial. FastAPI para Azure Container Apps.
Mismo patrón que los GlucoWise: ingress HTTPS, POST /predict.

Incluye un ADAPTADOR: acepta el payload tal como lo manda la app biomarcadores
(features CamelCase anidadas en `features`) y lo traduce a las 10 features de M19.
También acepta el dict plano de M19 (curl/Postman).
"""
import os
import json
import logging
import datetime
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.security import APIKeyHeader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("m19")

# Auth igual que los GlucoWise: header X-API-Key vs env API_KEY.
# Si API_KEY no está seteada, el endpoint queda abierto (mismo contrato que el resto).
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    return api_key

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model_posprandial_global.onnx")
FEATURE_ORDER = ["carbs", "protein", "fat", "fiber", "sugar", "kcal",
                 "gender", "hba1c", "hour", "basal"]

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT = sess.get_inputs()[0].name
app = FastAPI(title="M19 Riesgo Posprandial")


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip().upper()
    if s in ("F", "FEMENINO", "MUJER"):
        return 1.0
    if s in ("M", "MASCULINO", "HOMBRE"):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _first(*vals):
    for v in vals:
        n = _num(v)
        if not (isinstance(n, float) and np.isnan(n)):
            return n
    return np.nan


def adapt(body: dict) -> dict:
    """payload de la app (o plano) -> 10 features de M19."""
    feats = body.get("features") if isinstance(body.get("features"), dict) else {}
    g = lambda *ks: [body.get(k) for k in ks] + [feats.get(k) for k in ks]
    out = {
        "carbs":   _first(*g("carbs", "Meal_Carbs", "food_carbs")),
        "protein": _first(*g("protein", "Meal_Protein", "food_protein")),
        "fat":     _first(*g("fat", "Meal_Fat", "food_fat")),
        "fiber":   _first(*g("fiber", "Meal_Fiber", "food_fiber")),
        "sugar":   _first(*g("sugar", "Meal_Sugar", "food_sugar")),
        "kcal":    _first(*g("kcal", "Total_Energy", "calorie")),
        "gender":  _first(*g("gender", "Gender", "sexo", "demo_gender")),
        "hba1c":   _first(*g("hba1c", "HbA1c", "hba1c_pct")),
        "hour":    _first(*g("hour", "Hour", "Time_Of_Day")),
        "basal":   _first(*g("basal", "Glucosa_Basal", "glucosa_basal_mgdl")),
    }
    if isinstance(out["hour"], float) and np.isnan(out["hour"]):
        out["hour"] = float(datetime.datetime.utcnow().hour)
    return out


def predict(d10: dict) -> dict:
    x = np.array([[d10[k] for k in FEATURE_ORDER]], dtype=np.float32)
    label, probs = sess.run(None, {INPUT: x})
    p = float(probs[0, 1])
    return {
        "prob_pico_alto": round(p, 4),
        "banda": "ALTO" if p >= 0.5 else "BAJO",
        "label": int(label[0]),
        "basal_imputada": bool(np.isnan(d10["basal"])),
    }


@app.get("/")
def health():
    return {"status": "ok", "model": "M19 postprandial global",
            "input": INPUT, "features": FEATURE_ORDER, "umbral": 0.5}


@app.post("/predict", dependencies=[Security(get_api_key)])
async def predict_endpoint(req: Request):
    body = await req.json()
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        body = body["data"]
    d10 = adapt(body)
    res = predict(d10)
    log.info("predict %s -> %s", json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                             for k, v in d10.items()}), res)
    return res
