"""
M19 — Riesgo posprandial. FastAPI con CONTRATO ESTRICTO + RECALIBRACIÓN INTERNA.
Reemplazo listo para `models/m19_postprandial/app.py` (repo NepturaTech/glucosa-fastapi).
Implementa INF-009 (validador) + recalibración isotónica para pruebas internas.

QUÉ HACE (frente al app.py actual):
  VALIDADOR (transparencia de entrada):
  1. `adapt()` deja de descartar campos en silencio: alias conocidos se aceptan
     (compatibilidad con lo que la app manda hoy), pero un campo DESCONOCIDO se reporta
     con `quisiste_decir`. Los 8 obligatorios ausentes -> 422. Opcionales (hba1c, basal)
     pueden faltar -> se imputan DECLARÁNDOLO.
  2. Se ELIMINA la imputación silenciosa de `hour` a UTC (bug: modelo entrenado con hora
     local). `hour` pasa a obligatorio.
  3. `GET /schema` devuelve el contrato. Rollout en 2 pasos vía env STRICT (0=permisivo+log).

  RECALIBRACIÓN (transparencia de salida) — para pruebas internas:
  4. El endpoint devuelve la probabilidad CRUDA (prob_pico_alto, sin cambios: compatibilidad)
     Y una probabilidad RECALIBRADA (isotónica, por régimen), ambas etiquetadas. NUNCA
     sustituye ni oculta la cruda; la recalibrada es un campo aparte, marcado ORIENTATIVO.
  5. La recalibración es una tabla (JSON) ajustada sobre BIG IDEAs por régimen de faltantes:
     m19_recalibrador_isotonic_bigideas.json. Se aplica por interpolación (sin dependencia
     de sklearn en runtime). ⚠️ Ajustada a prevalencia BIG IDEAs (~0.48); NO validada en
     cohorte colombiana. Es una recalibración interna a evaluar, no calibrada para Colombia.

  Distinción clave con la capa heurística vieja de predict-risk (prohibida): aquella SUMABA
  +0.10 a +0.38 de forma arbitraria y OCULTA. Esto es una transformación estadística ajustada
  a datos, DECLARADA, y con la cruda siempre visible al lado.

Compatibilidad: en el payload real de producción da prob_pico_alto IDÉNTICA al app.py actual.
"""
import os
import json
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("m19")

API_KEY = os.getenv("API_KEY")
STRICT = os.getenv("STRICT", "0") == "1"          # rollout en 2 pasos: 0=permisivo+log, 1=422 duro
RECAL_ON = os.getenv("RECAL", "1") == "1"         # devolver la prob recalibrada (default sí)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    return api_key

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model_posprandial_global.onnx")
RECAL_FILE = os.path.join(HERE, "m19_recalibrador_isotonic_bigideas.json")
FEATURE_ORDER = ["carbs", "protein", "fat", "fiber", "sugar", "kcal",
                 "gender", "hba1c", "hour", "basal"]
OBLIGATORIOS = ["carbs", "protein", "fat", "fiber", "sugar", "kcal", "hour", "gender"]
OPCIONALES = ["hba1c", "basal"]

ALIASES = {
    "carbs":   ["carbs", "Meal_Carbs", "food_carbs"],
    "protein": ["protein", "Meal_Protein", "food_protein"],
    "fat":     ["fat", "Meal_Fat", "food_fat"],
    "fiber":   ["fiber", "Meal_Fiber", "food_fiber"],
    "sugar":   ["sugar", "Meal_Sugar", "food_sugar"],
    "kcal":    ["kcal", "Total_Energy", "calorie"],
    "gender":  ["gender", "Gender", "sexo", "demo_gender"],
    "hba1c":   ["hba1c", "HbA1c", "hba1c_pct"],
    "hour":    ["hour", "Hour", "Time_Of_Day"],
    "basal":   ["basal", "Glucosa_Basal", "glucosa_basal_mgdl"],
}
ALIAS_TO_CANON = {a: canon for canon, al in ALIASES.items() for a in al}

VALIDO = {"carbs": (0, 500), "protein": (0, 300), "fat": (0, 300), "fiber": (0, 100),
          "sugar": (0, 400), "kcal": (0, 5000), "hour": (0, 23),
          "hba1c": (3, 20), "basal": (20, 600)}
ENTRENAMIENTO = {"basal": (48, 219), "hba1c": (5.5, 6.4), "fiber": (0, 17), "sugar": (0, 145)}
MEDIANA_TRAIN = {"hba1c": 5.7, "basal": 105.0}

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT = sess.get_inputs()[0].name

# --- cargar recalibrador (tabla isotónica por régimen); si falta, se degrada a solo-cruda ---
RECAL = None
if RECAL_ON and os.path.exists(RECAL_FILE):
    try:
        RECAL = json.load(open(RECAL_FILE))
        log.info("recalibrador cargado: %s", RECAL.get("fit"))
    except Exception as e:
        log.warning("no se pudo cargar el recalibrador: %s", e)

app = FastAPI(title="M19 Riesgo Posprandial (validador + recalibración interna)")


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


def _sugerencia(clave):
    k = clave.lower().replace("_", "").replace(" ", "")
    for canon in FEATURE_ORDER:
        if canon in k or k in canon:
            return canon
    tabla = {"energy": "kcal", "cal": "kcal", "carb": "carbs", "prot": "protein",
             "azucar": "sugar", "grasa": "fat", "fibra": "fiber"}
    for frag, canon in tabla.items():
        if frag in k:
            return canon
    return None


def validar(body: dict):
    feats = body.get("features") if isinstance(body.get("features"), dict) else {}
    fuente = {**(feats or {}), **{k: v for k, v in body.items() if k != "features"}}
    ok, no_reconocidos, errores_valor, valores = [], [], [], {}
    for clave, valor in fuente.items():
        canon = ALIAS_TO_CANON.get(clave)
        if canon is None:
            no_reconocidos.append({"enviado": clave, "quisiste_decir": _sugerencia(clave)})
            continue
        n = _num(valor)
        if not (isinstance(n, float) and np.isnan(n)):
            if canon in VALIDO:
                lo, hi = VALIDO[canon]
                if not (lo <= n <= hi):
                    errores_valor.append({"campo": canon, "recibido": valor, "esperado": f"{lo}..{hi}"})
                    continue
            valores[canon] = n
            ok.append(canon)
    reporte = {"campos_ok": ok, "campos_no_reconocidos": no_reconocidos,
               "campos_faltantes_obligatorios": [c for c in OBLIGATORIOS if c not in valores],
               "campos_opcionales_ausentes": [c for c in OPCIONALES if c not in valores],
               "errores_de_valor": errores_valor}
    return valores, reporte


def _422(reporte):
    return JSONResponse(status_code=422, content={
        "error": "payload_invalido", **reporte,
        "ejemplo_valido": {"carbs": 60, "protein": 20, "fat": 15, "fiber": 4, "sugar": 10,
                           "kcal": 500, "gender": 1, "hour": 12, "hba1c": 5.7, "basal": 95},
        "schema": "GET /m19/schema"})


def _recalibrar(prob_cruda, sin_basal):
    """Interpola la tabla isotónica del régimen que aplica. Devuelve (prob_recal | None, meta)."""
    if RECAL is None:
        return None, None
    reg = "sin_basal" if sin_basal else "con_basal"
    t = RECAL.get("regimenes", {}).get(reg)
    if not t:
        return None, None
    p = float(np.interp(prob_cruda, t["x"], t["y"]))
    return round(p, 4), {"regimen_calibrador": reg, "prevalencia_fit": t.get("prevalencia_fit"),
                         "advertencia": RECAL.get("advertencia")}


def interpretar(valores: dict):
    imputados, advertencias = [], []
    d10 = {}
    for k in FEATURE_ORDER:
        if k in valores:
            d10[k] = valores[k]
            if k in ENTRENAMIENTO:
                lo, hi = ENTRENAMIENTO[k]
                if not (lo <= valores[k] <= hi):
                    advertencias.append(f"{k}={valores[k]} fuera del rango de entrenamiento [{lo},{hi}]; extrapola.")
        else:
            d10[k] = np.nan
            if k in MEDIANA_TRAIN:
                imputados.append({"campo": k, "valor_usado": MEDIANA_TRAIN[k], "origen": "mediana del entrenamiento"})

    x = np.array([[d10[k] for k in FEATURE_ORDER]], dtype=np.float32)
    label, probs = sess.run(None, {INPUT: x})
    p_cruda = float(probs[0, 1])                 # <- NUNCA se altera

    sin_basal = bool(np.isnan(d10["basal"]))
    regimen = "sin_basal_medida" if sin_basal else "con_basal_medida"
    auc_ref = 0.68 if sin_basal else 0.79

    resp = {
        # --- compatibilidad: la CRUDA, igual que el contrato actual ---
        "prob_pico_alto": round(p_cruda, 4),
        "banda": "ALTO" if p_cruda >= 0.5 else "BAJO",
        "label": int(label[0]),
        "basal_imputada": sin_basal,
        # --- contexto declarado ---
        "regimen": regimen,
        "auc_referencia": {"valor": auc_ref, "regimen": regimen,
                           "fuente": "LOPOCV BIG IDEAs (11 sujetos, EE.UU., CGM Dexcom G6)",
                           "advertencia": "No validado en cohorte colombiana ni con macros por foto."},
        "imputados": imputados,
        "advertencias": advertencias,
    }
    if sin_basal:
        resp["advertencias"].insert(0, "Se imputó la glucosa pre-comida; resultado ORIENTATIVO "
                                    "(AUC de referencia ~0.68 sin basal, ~0.79 con basal medida).")

    # --- recalibración (campo aparte, la cruda queda intacta) ---
    p_recal, meta = _recalibrar(p_cruda, sin_basal)
    if p_recal is not None:
        resp["prob_recalibrada"] = p_recal
        resp["banda_recalibrada"] = "ALTO" if p_recal >= 0.5 else "BAJO"
        resp["recalibracion"] = {"metodo": "isotonic", **meta}
    return resp


@app.get("/")
def health():
    return {"status": "ok", "model": "M19 postprandial global", "input": INPUT,
            "features": FEATURE_ORDER, "umbral": 0.5, "strict": STRICT,
            "recalibracion": bool(RECAL)}


@app.get("/schema")
def schema():
    return {"obligatorios": OBLIGATORIOS, "opcionales": OPCIONALES, "orden": FEATURE_ORDER,
            "unidades": {"carbs": "g", "protein": "g", "fat": "g", "fiber": "g", "sugar": "g",
                         "kcal": "kcal", "hour": "0-23 hora LOCAL de la comida",
                         "gender": "1=femenino, 0=masculino", "hba1c": "%",
                         "basal": "mg/dL glucosa pre-comida"},
            "rango_valido": VALIDO, "rango_entrenamiento": ENTRENAMIENTO, "aliases": ALIASES,
            "recalibracion": (RECAL.get("advertencia") if RECAL else "no cargada"),
            "ejemplo": {"carbs": 60, "protein": 20, "fat": 15, "fiber": 4, "sugar": 10,
                        "kcal": 500, "gender": 1, "hour": 12, "hba1c": 5.7, "basal": 95}}


@app.post("/predict", dependencies=[Security(get_api_key)])
async def predict_endpoint(req: Request):
    body = await req.json()
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        body = body["data"]
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="El cuerpo debe ser un objeto JSON")

    valores, reporte = validar(body)
    hay_problema = (reporte["campos_faltantes_obligatorios"] or reporte["campos_no_reconocidos"]
                    or reporte["errores_de_valor"])
    if hay_problema:
        if STRICT:
            return _422(reporte)
        log.warning("m19 payload_dudoso (permisivo) %s", json.dumps(reporte))

    res = interpretar(valores)
    log.info("predict regimen=%s cruda=%s recal=%s", res["regimen"],
             res["prob_pico_alto"], res.get("prob_recalibrada"))
    return res
