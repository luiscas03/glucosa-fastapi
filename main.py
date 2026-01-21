# main.py


from fastapi import FastAPI, HTTPException, Security, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


from fastapi.security import APIKeyHeader


from pydantic import BaseModel, Field, field_validator


from typing import Optional, Dict, List, Any, Tuple


import os


import logging


import joblib


import numpy as np


import pandas as pd


import math


import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv





# ------------------------ Config / Logging ------------------------------------
load_dotenv()


logging.basicConfig(level=logging.INFO)


logger = logging.getLogger("api")





API_KEY_NAME = "X-API-Key"


api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)





MODEL_PATH = os.getenv("MODEL_PATH", "assets/models/root/modelo_gradient_boosting_2.joblib")


API_KEY = os.getenv("API_KEY")

MONITOR_BASE_DIR = Path(__file__).resolve().parent / "glucose-ml-monitor-main"
MONITOR_MODELS_DIR = Path(__file__).resolve().parent / "assets" / "models" / "monitor"
MONITOR_MODEL_FILES = {
    "xgboost": MONITOR_MODELS_DIR / "XGBoost.joblib",
    "random_forest": MONITOR_MODELS_DIR / "Random_Forest.joblib",
    "lightgbm": MONITOR_MODELS_DIR / "LightGBM.joblib",
    "gradient_boosting": MONITOR_MODELS_DIR / "Gradient_Boosting.joblib",
    "ridge": MONITOR_MODELS_DIR / "Ridge.joblib",
    "lasso": MONITOR_MODELS_DIR / "Lasso.joblib",
    "elasticnet": MONITOR_MODELS_DIR / "ElasticNet.joblib",
}
MONITOR_PREPROCESSING_FILE = MONITOR_MODELS_DIR / "preprocessing_objects.pkl"

MISSING_TOKEN = "__MISSING__"





# Defaults numéricos para imputar si llegan NaN (configurable por ENV JSON)


DEFAULT_NUMS = {


    "edad": 40.0,


    "tas": 120.0,


    "tad": 80.0,


    "peso": 70.0,


    "talla": 1.65,


    "perimetro_abdominal": 90.0,


    "imc": 24.0,


}


try:


    _env_defaults = os.getenv("NUMERIC_DEFAULTS")


    if _env_defaults:


        DEFAULT_NUMS.update(json.loads(_env_defaults))


except Exception as e:


    logger.warning(f"No se pudo parsear NUMERIC_DEFAULTS: {e}")





def _num_default(col: str) -> float:


    return float(DEFAULT_NUMS.get(col, 0.0))





# ------------------------ Globales (se rellenan en startup) -------------------


pipe = None


num_cols: List[str] = []


cat_cols: List[str] = []


FEATURE_COLS: List[str] = []

monitor_models: Dict[str, Any] = {}
monitor_label_encoders: Dict[str, Any] = {}
monitor_scaler = None
monitor_feature_names: Optional[List[str]] = None
monitor_ready = False

# ------------------------ Utils ----------------------------------------------


def to_float_or_none(v: Any) -> Optional[float]:


    if v is None or isinstance(v, bool):


        return None


    try:


        return float(v)


    except Exception:


        return None





def compute_imc(peso: Any, talla: Any) -> Optional[float]:


    p = to_float_or_none(peso)


    t = to_float_or_none(talla)


    if p is None or t is None:


        return None


    try:


        t_m = t / 100.0 if t > 3 else t


        if t_m and t_m > 0 and p > 0:


            return round(p / (t_m ** 2), 2)


    except (TypeError, ZeroDivisionError) as e:


        logger.warning(f"No se pudo calcular IMC con peso={peso}, talla={talla}. Error: {e}")


    return None





def sanitize_value(v: Any):


    if isinstance(v, list):


        return sanitize_value(v[0]) if v else None


    if isinstance(v, dict):


        return None


    if isinstance(v, str):


        s = v.strip()


        if s.lower() in {"", "null", "none", "nan"}:


            return None


        return s


    return v





def safe_isna(v: Any) -> bool:


    if v is None:


        return True


    if isinstance(v, bool):


        return False


    if isinstance(v, (int, float, np.floating, np.integer)):


        return isinstance(v, float) and math.isnan(v)


    if isinstance(v, str):


        return False


    return False





def align_row(payload: Dict) -> Tuple[pd.DataFrame, List[str]]:


    """


    Alinea dict a FEATURE_COLS, sanea tipos (ManyChat), calcula IMC y


    evita NaN imputando numéricos con DEFAULT_NUMS y categóricas a __MISSING__.


    Devuelve (df, columnas_que_estaban_faltantes).


    """


    global FEATURE_COLS, num_cols, cat_cols





    clean = {k: sanitize_value(v) for k, v in payload.items()}


    cols = FEATURE_COLS or list(clean.keys())


    row = {c: np.nan for c in cols}





    # Copiar valores saneados


    for k, v in clean.items():


        if k in row:


            row[k] = v





    # IMC si corresponde (antes de marcar faltantes)


    if "imc" in cols and ("peso" in clean or "talla" in clean):


        imc_val = compute_imc(clean.get("peso"), clean.get("talla"))


        if imc_val is not None:


            row["imc"] = imc_val





    df = pd.DataFrame([row])





    # Numéricas → a número


    for c in num_cols:


        if c in df.columns:


            df[c] = pd.to_numeric(df[c], errors="coerce")





    # Detectar faltantes ANTES de imputar (para reportar)


    missing_before: List[str] = []


    for c in num_cols:


        if c in df.columns and safe_isna(df.at[0, c]):


            missing_before.append(c)


    for c in cat_cols:


        if c in df.columns and safe_isna(df.at[0, c]):


            missing_before.append(c)





    # Si 'imc' quedó NaN, intentar calcularla con peso/talla


    if "imc" in df.columns and safe_isna(df.at[0, "imc"]):


        imc_try = compute_imc(df.at[0, "peso"] if "peso" in df.columns else None,


                              df.at[0, "talla"] if "talla" in df.columns else None)


        if imc_try is not None:


            df.at[0, "imc"] = imc_try


            # si estaba en missing_before, ya no la retiramos para reportar que se derivó





    # Imputación numérica definitiva


    for c in num_cols:


        if c in df.columns and safe_isna(df.at[0, c]):


            df.at[0, c] = _num_default(c)





    # Categóricas → string o __MISSING__


    for c in cat_cols:


        if c in df.columns:


            df[c] = df[c].map(lambda x: MISSING_TOKEN if safe_isna(x) else str(x))





    return df, sorted(set(missing_before))







# ------------------------ PPG Helpers --------------------------------------
def _infer_fps(fps, timestamps):
    if fps is not None and fps > 0:
        return float(fps)
    if not timestamps or len(timestamps) < 2:
        return None
    ts = np.array(timestamps, dtype=float)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    med = float(np.median(diffs))
    if med <= 0:
        return None
    if med > 10.0:
        return 1000.0 / med
    return 1.0 / med

def _normalize_timestamps(timestamps, fps, n):
    if timestamps and len(timestamps) >= 2:
        ts = np.array(timestamps, dtype=float)
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if diffs.size > 0 and float(np.median(diffs)) > 10.0:
            ts = ts / 1000.0
        return (ts - ts[0]).tolist()
    return (np.arange(n) / float(fps)).tolist()

def _bandpass_fft(signal, fs, low=0.7, high=4.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal)
    mask = (freqs >= low) & (freqs <= high)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=n)

def _estimate_bpm_fft(signal, fs, low=0.7, high=4.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return 0.0, 0.0, freqs, power
    band_freqs = freqs[band]
    band_power = power[band]
    peak_idx = int(np.argmax(band_power))
    peak_freq = float(band_freqs[peak_idx])
    bpm = peak_freq * 60.0
    return bpm, peak_freq, freqs, power

def _snr_db(power, freqs, peak_freq, low=0.7, high=4.0, bw=0.1):
    band = (freqs >= low) & (freqs <= high)
    peak = (freqs >= (peak_freq - bw)) & (freqs <= (peak_freq + bw))
    signal_power = float(np.sum(power[band & peak]))
    noise_power = float(np.sum(power[band & ~peak]))
    if signal_power <= 0 or noise_power <= 0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)

def _motion_pct(signal):
    diff = np.abs(np.diff(signal))
    if diff.size == 0:
        return 0.0
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    thresh = med + 3.0 * mad
    if thresh <= 0:
        return 0.0
    return float(np.sum(diff > thresh) / diff.size * 100.0)

def _chrom_signal(r, g, b):
    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))
    if r_mean <= 0 or g_mean <= 0 or b_mean <= 0:
        raise HTTPException(status_code=400, detail="Invalid RGB means for CHROM")
    r_n = (r / r_mean) - 1.0
    g_n = (g / g_mean) - 1.0
    b_n = (b / b_mean) - 1.0
    x = 3.0 * r_n - 2.0 * g_n
    y = 1.5 * r_n + g_n - 1.5 * b_n
    y_std = float(np.std(y))
    alpha = float(np.std(x) / y_std) if y_std > 1e-6 else 0.0
    return x - alpha * y
# ------------------------ Seguridad ------------------------------------------


async def get_api_key(incoming_key: str = Security(api_key_header)):


    if API_KEY and incoming_key == API_KEY:


        return incoming_key


    raise HTTPException(status_code=403, detail="Could not validate credentials")





# ------------------------ App -------------------------------------------------


app = FastAPI(title="API Glucosa RF", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)





@app.get("/", include_in_schema=False)
def root():
    index_path = MONITOR_BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return FileResponse(index_path)





def _iter_estimators(est):


    """Itera recursivamente para localizar OneHotEncoder dentro del pipeline."""


    from sklearn.pipeline import Pipeline


    from sklearn.compose import ColumnTransformer


    if isinstance(est, Pipeline):


        for _, step in est.steps:


            yield from _iter_estimators(step)


    elif isinstance(est, ColumnTransformer):


        for _, trans, _cols in est.transformers_:


            if trans == "drop" or trans == "passthrough":


                continue


            yield from _iter_estimators(trans)


    else:


        yield est





def _set_onehot_ignore_unknown():
    try:
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return
    if pipe is None:
        return
    for est in _iter_estimators(pipe):
        if isinstance(est, OneHotEncoder):
            est.handle_unknown = "ignore"


def _patch_onehot_categories():


    """Limpia categories_ de cada OneHotEncoder para evitar np.isnan sobre objetos."""


    try:


        from sklearn.preprocessing import OneHotEncoder


    except Exception:


        return


    count = 0


    for est in _iter_estimators(pipe):


        if isinstance(est, OneHotEncoder) and hasattr(est, "categories_"):


            new_cats = []


            for cats in est.categories_:


                cleaned = []


                for v in list(cats):
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        cleaned.append(MISSING_TOKEN)
                    else:
                        cleaned.append(str(v))

                new_cats.append(np.array(cleaned, dtype=object))


            est.categories_ = new_cats


            try:


                est.handle_unknown = "ignore"


            except Exception:


                pass


            count += 1


    logger.info(f"Parcheados {count} OneHotEncoder(s) categories_ (añadido {MISSING_TOKEN}).")





@app.on_event("startup")


def load_artifacts():

    from sklearn.compose import ColumnTransformer

    import sklearn, numpy as _np

    global pipe, num_cols, cat_cols, FEATURE_COLS

    # (Opcional) limitar hilos para evitar crashes nativos raros

    try:


        from threadpoolctl import threadpool_limits, threadpool_info


        threadpool_limits(1)


        logger.info(f"threadpools → {threadpool_info()}")


    except Exception as _e:


        logger.info(f"No se ajustaron threadpools: {_e}")





    logger.info(f"VERSIONS → sklearn={sklearn.__version__} numpy={_np.__version__} pandas={pd.__version__}")


    logger.info(f"Cargando modelo desde: {MODEL_PATH}")


    try:


        pipe = joblib.load(MODEL_PATH)


    except Exception as e:


        logger.error(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}", exc_info=True)


        raise





    # Intentar ubicar ColumnTransformer


    pre = None


    try:


        ns = list(getattr(pipe, "named_steps", {}).keys())


        logger.info(f"named_steps: {ns}")


    except Exception:


        pass


    if hasattr(pipe, "named_steps"):


        pre = pipe.named_steps.get("preprocess") or pipe.named_steps.get("preprocessor")


    if pre is None and hasattr(pipe, "steps"):


        for name, step in pipe.steps:


            if isinstance(step, ColumnTransformer):


                pre = step


                logger.info(f"Usando ColumnTransformer encontrado en step: {name}")


                break





    # Extraer columnas


    if pre is not None:


        try:


            ncols, ccols = [], []


            for name, transformer, cols in pre.transformers_:


                if name == "num":


                    ncols = list(cols)


                elif name == "cat":


                    ccols = list(cols)


            num_cols[:] = ncols


            cat_cols[:] = ccols


            FEATURE_COLS[:] = ncols + ccols


            logger.info(f"num_cols={len(num_cols)} cat_cols={len(cat_cols)} FEATURES={len(FEATURE_COLS)}")


        except Exception as e:


            logger.warning(f"No se pudieron derivar columnas desde ColumnTransformer: {e}")





    if not FEATURE_COLS:


        fallback = list(getattr(pipe, "feature_names_in_", []))


        if fallback:


            FEATURE_COLS[:] = fallback


            logger.info(f"Usando feature_names_in_: {len(FEATURE_COLS)} columnas")


        else:


            logger.warning("No se pudieron determinar FEATURE_COLS; se usarán keys del payload en cada request.")






    _set_onehot_ignore_unknown()
    # Parchear OHE para evitar isnan sobre objetos


    #_patch_onehot_categories()


@app.on_event("startup")
def load_monitor_models():
    global monitor_models, monitor_label_encoders, monitor_scaler
    global monitor_feature_names, monitor_ready

    monitor_models = {}
    monitor_label_encoders = {}
    monitor_scaler = None
    monitor_feature_names = None
    monitor_ready = False

    if not MONITOR_BASE_DIR.exists():
        logger.warning(f"No se encontró {MONITOR_BASE_DIR}; monitor deshabilitado.")
        return

    try:
        if not MONITOR_PREPROCESSING_FILE.exists():
            logger.warning(f"Preprocesamiento no encontrado: {MONITOR_PREPROCESSING_FILE}")
            return

        preprocessing = joblib.load(MONITOR_PREPROCESSING_FILE)
        monitor_label_encoders = preprocessing.get("label_encoders", {})
        monitor_scaler = preprocessing.get("scaler", None)
        monitor_feature_names = preprocessing.get("feature_names", None)

        models_loaded = 0
        for model_name, model_path in MONITOR_MODEL_FILES.items():
            if not model_path.exists():
                logger.warning(f"Modelo monitor no encontrado: {model_path}")
                continue
            try:
                monitor_models[model_name] = joblib.load(model_path)
                models_loaded += 1
                logger.info(f"Modelo monitor cargado: {model_name}")
            except Exception as e:
                logger.error(f"Error al cargar {model_name}: {e}")

        if models_loaded == 0:
            logger.warning("No se pudo cargar ningun modelo monitor.")
            return

        monitor_ready = True
        logger.info(f"Monitor listo: {models_loaded}/7 modelos cargados.")
    except Exception as e:
        logger.error(f"Error cargando modelos monitor: {e}", exc_info=True)

# ------------------------ Schemas (endpoint estricto opcional) ----------------


class PredictItem(BaseModel):


    edad: Optional[float] = Field(None, ge=0)


    tas: Optional[float] = None


    tad: Optional[float] = None


    perimetro_abdominal: Optional[float] = None


    peso: Optional[float] = Field(None, gt=0)


    talla: Optional[float] = Field(None, gt=0)


    realiza_ejercicio: Optional[str] = None


    frecuencia_frutas: Optional[str] = None


    medicamentos_hta: Optional[float] = None


    ips_codigo: Optional[float] = None





    class Config:


        extra = "allow"





    @field_validator("talla")


    def talla_valida(cls, v):


        if v is not None and v <= 0:


            raise ValueError("talla debe ser > 0")


        return v







class PPGMeasureRequest(BaseModel):
    r: List[float]
    g: List[float]
    b: List[float]
    fps: Optional[float] = Field(None, gt=0)
    timestamps: Optional[List[float]] = None

    class Config:
        extra = "allow"
# ------------------------ Monitor Schemas ------------------------------------


class MonitorPredictionInput(BaseModel):
    edad: int = Field(..., ge=18, le=120)
    sexo: str
    peso: float = Field(..., gt=0)
    talla: float = Field(..., gt=0)
    imc: Optional[float] = None
    perimetro_cintura: float = Field(..., gt=0)
    spo2: int = Field(..., ge=70, le=100)
    frecuencia_cardiaca: int = Field(..., ge=40, le=200)
    actividad_fisica: str
    consumo_frutas: str
    tiene_hipertension: str
    tiene_diabetes: str
    puntaje_findrisc: int = Field(..., ge=0, le=26)


def monitor_preprocess_input(data: MonitorPredictionInput) -> pd.DataFrame:
    imc_value = data.imc if data.imc is not None else (data.peso / (data.talla ** 2))
    input_dict = {
        "Edad": data.edad,
        "Sexo": data.sexo.lower().capitalize(),
        "Peso": data.peso,
        "Talla": data.talla,
        "IMC": imc_value,
        "Perimetro_Cintura": data.perimetro_cintura,
        "SpO2": data.spo2,
        "Frecuencia_Cardiaca": data.frecuencia_cardiaca,
        "Actividad_Fisica": data.actividad_fisica.lower().capitalize(),
        "Consumo_Frutas": data.consumo_frutas.lower().capitalize(),
        "Tiene_Hipertension": data.tiene_hipertension.lower().capitalize(),
        "Tiene_Diabetes": data.tiene_diabetes.lower().capitalize(),
        "Puntaje_FINDRISC": data.puntaje_findrisc,
    }

    for col, encoder in monitor_label_encoders.items():
        if col in input_dict:
            try:
                input_dict[col] = encoder.transform([input_dict[col]])[0]
            except ValueError:
                input_dict[col] = 0

    if monitor_feature_names is not None:
        df = pd.DataFrame([{col: input_dict.get(col, 0) for col in monitor_feature_names}])
    else:
        df = pd.DataFrame([input_dict])

    if monitor_scaler is not None:
        X_scaled = monitor_scaler.transform(df)
        return pd.DataFrame(X_scaled, columns=df.columns)
    return df


def _predict_monitor(data: MonitorPredictionInput) -> Dict[str, Any]:
    X_preprocessed = monitor_preprocess_input(data)

    predicciones_individuales = {}
    predicciones_validas = []

    for model_name, model in monitor_models.items():
        try:
            pred = model.predict(X_preprocessed)[0]
            predicciones_individuales[model_name] = float(pred)
            predicciones_validas.append(pred)
        except Exception:
            predicciones_individuales[model_name] = None

    if len(predicciones_validas) == 0:
        raise HTTPException(status_code=500, detail="Ningun modelo pudo predecir")

    prediccion_final = float(np.mean(predicciones_validas))

    if prediccion_final < 100:
        categoria = "Normal"
    elif prediccion_final < 126:
        categoria = "Prediabetes"
    else:
        categoria = "Diabetes"

    std_predicciones = float(np.std(predicciones_validas))
    confidence = 1.0 - (std_predicciones / prediccion_final) if prediccion_final > 0 else 0.5
    confidence = max(0.0, min(1.0, confidence))

    intervalo_min = prediccion_final - 1.96 * std_predicciones
    intervalo_max = prediccion_final + 1.96 * std_predicciones

    mejor_modelo = min(
        [(k, v) for k, v in predicciones_individuales.items() if v is not None],
        key=lambda x: abs(x[1] - prediccion_final),
    )[0]

    return {
        "prediccion_final": round(prediccion_final, 2),
        "categoria": categoria,
        "predicciones_individuales": {
            k: round(v, 2) if v is not None else None
            for k, v in predicciones_individuales.items()
        },
        "confidence": round(confidence, 3),
        "intervalo_confianza": [round(intervalo_min, 2), round(intervalo_max, 2)],
        "mejor_modelo": mejor_modelo,
        "timestamp": datetime.now().isoformat(),
        "input_data": data.model_dump(),
    }

# ------------------------ Endpoints ------------------------------------------


@app.get("/health")


def health():


    return {


        "status": "ok",


        "features_esperadas": len(FEATURE_COLS),


        "tiene_named_steps": bool(getattr(pipe, "named_steps", {})),


        "num_cols": num_cols,


        "cat_cols": cat_cols,


    }


@app.get("/monitor/health")
def monitor_health():
    return {
        "status": "ready" if monitor_ready else "not_ready",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(monitor_models),
        "models_available": list(monitor_models.keys()),
        "feature_names_loaded": monitor_feature_names is not None,
    }


@app.get("/monitor")
def monitor_index():
    index_path = MONITOR_BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return FileResponse(index_path)


@app.get("/monitor/models")
def monitor_models_list():
    if not monitor_ready:
        raise HTTPException(status_code=503, detail="Monitor no disponible")
    return {
        "models": list(monitor_models.keys()),
        "total": len(monitor_models),
        "feature_names": monitor_feature_names,
    }


# Tolerante (recomendado para ManyChat)


@app.post("/predict", dependencies=[Security(get_api_key)])


def predict(payload: Dict[str, Any] = Body(...)):


    global pipe


    try:
        if not monitor_ready:
            raise HTTPException(status_code=503, detail="Monitor no disponible")

        data = MonitorPredictionInput(**payload)
        return _predict_monitor(data)


    except Exception as e:


        logger.error(f"Error en /predict: {e}", exc_info=True)


        raise HTTPException(status_code=400, detail=str(e))



# Estricto (útil para pruebas manuales)


@app.post("/predict_typed", dependencies=[Security(get_api_key)])
def predict_typed(item: PredictItem):
    return predict(item.model_dump())


@app.post("/api/v1/glucose/predict", dependencies=[Security(get_api_key)])
def predict_v1(payload: Dict[str, Any] = Body(...)):
    return predict(payload)


@app.post("/api/v1/glucose/predict_typed", dependencies=[Security(get_api_key)])
def predict_v1_typed(item: PredictItem):
    return predict(item.model_dump())


@app.post("/monitor/predict", dependencies=[Security(get_api_key)])
def monitor_predict(data: MonitorPredictionInput):
    if not monitor_ready:
        raise HTTPException(status_code=503, detail="Monitor no disponible")

    try:
        return _predict_monitor(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /monitor/predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {e}")


@app.post("/api/v1/ppg/measure", dependencies=[Security(get_api_key)])
def ppg_measure(item: PPGMeasureRequest):
    r = np.array(item.r, dtype=float)
    g = np.array(item.g, dtype=float)
    b = np.array(item.b, dtype=float)

    if r.ndim != 1 or g.ndim != 1 or b.ndim != 1:
        raise HTTPException(status_code=400, detail="r/g/b must be 1D arrays")
    if not (len(r) == len(g) == len(b)):
        raise HTTPException(status_code=400, detail="r/g/b length mismatch")
    if len(r) < 100:
        raise HTTPException(status_code=400, detail="At least 100 samples required")

    fps = _infer_fps(item.fps, item.timestamps)
    if fps is None or fps <= 0:
        raise HTTPException(status_code=400, detail="fps or timestamps required")

    chrom = _chrom_signal(r, g, b)
    filtered = _bandpass_fft(chrom, fps)
    bpm, peak_freq, freqs, power = _estimate_bpm_fft(filtered, fps)
    snr_db = _snr_db(power, freqs, peak_freq)
    motion = _motion_pct(g)
    confidence = max(0.0, min(1.0, snr_db / 10.0))

    quality = {
        "snr_db": round(snr_db, 2),
        "motion_pct": round(motion, 2),
        "valid": bool(bpm > 0 and snr_db >= 3.0 and motion <= 10.0),
    }

    return {
        "bpm": round(bpm, 2),
        "confidence": round(confidence, 3),
        "quality": quality,
        "signal": filtered.astype(float).tolist(),
        "timestamps": _normalize_timestamps(item.timestamps, fps, len(r)),
        "fps": round(fps, 2),
        "method": "CHROM",
    }
