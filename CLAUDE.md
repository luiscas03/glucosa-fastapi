# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Idioma

El proyecto y su documentación están en español. Responde y escribe comentarios/documentación en español.

## Memoria del proyecto

La memoria de Claude para este repo vive **solo** en `.claude/memory/`, con índice en `.claude/MEMORY.md`, y se versiona con el repo para que el contexto sea portable. No escribir memorias de este proyecto en el directorio global `~/.claude/`.

## Fuentes de contexto

**Repositorio canónico: `https://github.com/NepturaTech/glucosa-fastapi`** (es el `origin` real; el `git clone` que aparece en `README.md` y en `deploy_contabo/README.md` apunta a `luiscas03/glucosa-fastapi`, que está obsoleto).

**Base de conocimiento en Devin (DeepWiki)** — fuente **secundaria**, vía las herramientas MCP `mcp__devin__*` con `repoName: "NepturaTech/glucosa-fastapi"`:

- `read_wiki_structure` → índice de temas. `read_wiki_contents` → contenido. `ask_question` → respuesta puntual con citas.
- Tiene wiki completa: overview, `main.py`, cada microservicio, despliegue (Contabo + Azure), artefactos, los dos informes de validación y un glosario.
- Útil sobre todo para **orientarse rápido** ("¿dónde vive X?", "¿cómo encaja Y?") y para el histórico de decisiones, sin quemar contexto leyendo archivos.

Dos límites reales, comprobados el 2026-07-23:

1. **Está indexada sobre el último commit pusheado, no sobre el working tree.** Describe el `m19_postprandial` antiguo y no conoce la recalibración isotónica ni `GET /schema` que ya están en disco sin commitear. Ante cualquier discrepancia, **manda el archivo local**.
2. **Repite lo que dicen los `.md` del repo, incluidos sus errores.** Afirma que los artefactos de `assets/models/monitor/` piden sklearn 1.4.2 porque así lo dice `VALIDACION_MODELOS_2026-06-04.md`; la verificación empírica dice **1.3.2** (bajo 1.4.2 `Gradient_Boosting` ni siquiera carga). Para versiones, shapes, features o cualquier hecho de runtime, **verifica ejecutando**, no preguntando.

Regla práctica: Devin para navegar e historia; el código y `joblib.load` para la verdad.

## Entorno de desarrollo

Todos los venvs usan **Python 3.12** (`/usr/bin/python3.12`). El `python3` por defecto de la máquina es pyenv 3.14, que **no sirve**: las versiones de sklearn que exigen los artefactos no tienen wheels para 3.14. Se gestionan con `uv`.

Hay **tres venvs** porque los artefactos del repo son de tres generaciones de scikit-learn incompatibles entre sí (ver la sección de compatibilidad). Todos están gitignored.

| Venv | Para qué | sklearn / numpy |
|---|---|---|
| `.venv` | `main.py` (gateway). **Reproduce producción**, incluido su bug de 5/7 modelos. | 1.6.1 / ≥2 |
| `.venv-models` | microservicios `models/*` | 1.4.2 / <2 |
| `.venv-monitor` | evaluar el ensemble monitor con fidelidad a los artefactos (7/7). **No arranca `main.py`.** | 1.3.2 / <2 |

Recrearlos:

```bash
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python "scikit-learn==1.6.1" "numpy>=2"

uv venv --python /usr/bin/python3.12 .venv-models
uv pip install --python .venv-models/bin/python fastapi "uvicorn[standard]" \
  "scikit-learn==1.4.2" "numpy<2" pandas joblib python-dotenv

uv venv --python /usr/bin/python3.12 .venv-monitor
uv pip install --python .venv-monitor/bin/python fastapi "uvicorn[standard]" \
  "scikit-learn==1.3.2" "numpy<2" pandas joblib python-dotenv xgboost lightgbm
```

`.env` (gitignored) solo necesita `API_KEY`. Ver `.env.example`. Sin `API_KEY` el gateway devuelve 403 en todo.

## Comandos

```bash
# API principal (gateway) — puerto 8000
.venv/bin/python -m uvicorn main:app --reload

# Microservicios de modelo — como módulo desde la raíz (tienen __init__.py) y con .venv-models
.venv-models/bin/python -m uvicorn models.glucose_regression_v2.main:app  --port 8001 --reload
.venv-models/bin/python -m uvicorn models.postprandial_global_m1.main:app --port 8002 --reload
.venv-models/bin/python -m uvicorn models.postprandial_cluster_0.main:app --port 8003 --reload
.venv-models/bin/python -m uvicorn models.postprandial_cluster_1.main:app --port 8004 --reload
.venv-models/bin/python -m uvicorn models.cnn_lstm_16p_clean_v1.main:app --port 8005 --reload

# m19 es la excepción: app.py (no main.py) y SIN __init__.py → hay que entrar a la carpeta
cd models/m19_postprandial && ../../.venv-models/bin/python -m uvicorn app:app --port 8006 --reload
```

m19 necesita `onnxruntime` (ya instalado en `.venv-models`); no usa sklearn.

`models/model_catalog.json` lista los import paths y los puertos sugeridos de los servicios sklearn.

**No hay suite de tests** — no hay pytest ni conftest en el repo, así que no hay convención previa que seguir. La verificación es manual: `test_client.html` (abrirlo en el navegador; formulario que pega contra la API con `X-API-Key`) y `POSTMAN_ENDPOINTS.md`, que documenta cada endpoint con su body de ejemplo.

Smoke test del gateway (verificado 2026-07-23; requiere `API_KEY` en `.env`):

```bash
curl -s localhost:8000/monitor/health          # espera models_loaded: 6 con .venv (ver compatibilidad)
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -H "X-API-Key: $API_KEY" \
  -d '{"edad":45,"sexo":"Masculino","peso":82.5,"talla":1.72,"perimetro_cintura":95.0,
       "spo2":96,"frecuencia_cardiaca":78,"actividad_fisica":"Moderada","consumo_frutas":"Diario",
       "tiene_hipertension":"No","tiene_diabetes":"No","puntaje_findrisc":12}'
```

Los microservicios exponen `GET /health` y `GET /features` **sin auth**, que es la vía rápida para confirmar que sus artefactos deserializaron.

## Arquitectura

Hay **dos capas de servicio que conviven** y que se despliegan por separado:

### 1. `main.py` — API principal / gateway (puerto 8000)

Monolito FastAPI que carga en `@app.on_event("startup")`:
- **El ensemble "monitor"**: 7 regresores (`assets/models/monitor/*.joblib`: XGBoost, Random Forest, LightGBM, Gradient Boosting, Ridge, Lasso, ElasticNet) + `preprocessing_objects.pkl` (label encoders, scaler, feature_names). Predice la media de los 7, categoriza en Normal / Prediabetes / Diabetes y estima confianza vía desviación estándar entre modelos.
- **Un pipeline sklearn suelto** (`MODEL_PATH`, por defecto `assets/models/root/modelo_gradient_boosting_2.joblib`).

**Detalle crítico que no se ve leyendo un solo endpoint:** `/predict`, `/predict_typed`, `/api/v1/glucose/predict` y `/api/v1/glucose/predict_typed` **todos delegan en `_predict_monitor()`**, es decir en el ensemble. El pipeline `pipe` y toda la maquinaria que lo acompaña (`align_row()`, `DEFAULT_NUMS`, `_patch_onehot_categories()`, `MISSING_TOKEN`, `num_cols`/`cat_cols`) **ya no participa en ninguna predicción**; solo se refleja en `GET /health`. Es código heredado del modelo anterior. Al tocar `/predict` no busques el bug en `align_row`.

Consecuencia práctica: `predict_typed` valida contra `PredictItem` (`edad`, `tas`, `tad`, `perimetro_abdominal`, …) pero luego reenvía a `MonitorPredictionInput`, que exige otros campos (`sexo`, `perimetro_cintura`, `spo2`, `frecuencia_cardiaca`, `puntaje_findrisc`, …). Solo funciona porque `PredictItem.Config.extra = "allow"` deja pasar esos campos como extras; un body "típico" de `PredictItem` devuelve 400.

También expone `POST /api/v1/ppg/measure`: procesamiento de señal PPG puro NumPy (sin modelo) — CHROM sobre canales R/G/B, bandpass FFT 0.7–4 Hz, BPM por pico espectral, SNR y porcentaje de movimiento como quality gate.

El README menciona un `GET /monitor` que sirve una UI desde `glucose-ml-monitor-main/index.html`. **Esa ruta no existe en `main.py`** y esa carpeta no está en el repo; `static/index.html` tampoco está montado (`static/` está gitignored). Documentación desactualizada, no un bug a "arreglar" sin confirmarlo.

### 2. `models/<id>/` — un microservicio FastAPI por modelo

Cada carpeta es autocontenida y desplegable por sí sola: su `main.py`, su `Dockerfile`, su `requirements.txt` y sus artefactos. Comparten un patrón casi idéntico:

- Auth por header `X-API-Key` contra `os.getenv("API_KEY")`.
- `load_dotenv()` de `models/.env` y luego del `.env` de la raíz.
- CORS abierto (`allow_origins=["*"]`).
- Carga de artefactos en `@app.on_event("startup")`.
- Rutas: `GET /health`, `GET /features`, `POST /predict` (protegido). Los postprandiales añaden `GET /metadata`; `glucose_regression_v2` añade `POST /predict_typed`.

**Diferencia de auth importante entre las dos capas:** en los microservicios la comprobación es `if API_KEY and api_key != API_KEY` → **sin `API_KEY` en el entorno el endpoint queda abierto**. En `main.py` es `if API_KEY and incoming_key == API_KEY: return ...` seguido de raise → **sin `API_KEY` todo devuelve 403**. Al arrancar local, si `/predict` da 403 revisa `.env` antes que el código.

Formato del body de `/predict`, que difiere por familia:
- Postprandiales (`postprandial_*`): `{"features": {...}}` anidado, con **todas** las features de `model.feature_names_in_` (o `m1_features` del metadata) presentes; si falta alguna devuelve 400 con `missing_features`.
- `glucose_regression_v2`: dict plano; las features ausentes se rellenan con `0.0` en silencio (`to_float`).
- `cnn_lstm_*`: `{"values": [...], "shape": [...]}`; valida el tamaño contra el input shape del modelo Keras.
- `m19_postprandial`: acepta **dict plano O el payload de la app biomarcadores** (features anidadas en `features`); resuelve alias a las 10 features canónicas del ONNX. Es el único con adaptador de payload.

`m19_postprandial` es el más divergente del resto y conviene leerlo entero antes de tocarlo:

- **No usa sklearn**: es ONNX Runtime. Su archivo es `app.py` (no `main.py`), no tiene `__init__.py`, carga el modelo a nivel de módulo (no en startup) y su health está en `GET /` (no `/health`). Añade `GET /schema` con unidades, rangos y alias — el mejor punto de partida para construir un payload.
- **Valida y reporta en vez de fallar**: `validar()` clasifica cada campo en reconocido / no reconocido (con sugerencia "quisiste_decir") / fuera de rango. Con `STRICT` activo devuelve 422 con el reporte; por defecto es **permisivo**: loguea `payload_dudoso` y predice igual. Un payload con nombres mal escritos **no da error**, da una predicción sobre features imputadas.
- **Imputa con medianas de entrenamiento** los campos ausentes y lo declara en `imputados` / `advertencias`.
- **Recalibración isotónica** desde `m19_recalibrador_isotonic_bigideas.json`, con tablas separadas para régimen `sin_basal` / `con_basal`. Clave del contrato: `prob_pico_alto` **sigue siendo siempre la probabilidad cruda**, intacta; la recalibrada va aparte en `prob_recalibrada` / `banda_recalibrada`. No cambies cuál se devuelve en `prob_pico_alto` — hay consumidores atados a ese campo.
- La respuesta incluye `auc_referencia` (0.68 sin basal medida, 0.79 con ella) y advertencias explícitas de que **no está validado en cohorte colombiana** y es para pruebas internas, no uso clínico.

`models/postprandial_global_m1/` existe y funciona pero **no está en el despliegue de Contabo** (solo en `model_catalog.json`). `models/archive_non_api/` guarda artefactos explícitamente no aptos para producción.

### 3. `deploy_contabo/` — despliegue en producción

`docker compose` con 5 de los microservicios + Caddy (ruteo por prefijo de ruta) + `cloudflared` (Cloudflare Tunnel; no se abren puertos en el VPS, el TLS lo pone Cloudflare en el edge). Cada servicio recibe **su propia API key** por variable de entorno; nunca se hornean en la imagen.

El mapeo ruta ↔ carpeta es load-bearing y no es deducible por el nombre:

| Ruta Caddy | Carpeta | Salida |
|---|---|---|
| `/m19/predict` | `m19_postprandial` | `prob_pico_alto` + banda |
| `/m2/predict` | `cnn_lstm_16p_clean_v1` | `prediction` mg/dL |
| `/m5/predict` | `postprandial_cluster_0` | `risk_calibrated` |
| `/m6/predict` | `postprandial_cluster_1` | `risk` |
| `/m3/predict` | `glucose_regression_v2` | `prediction` mg/dL |

`handle_path` en el Caddyfile **quita el prefijo**: `/m19/predict` llega al contenedor como `/predict`. Health sin auth: M19 en `/m19/`, el resto en `/m2/health`, `/m5/health`, etc.

Actualizar en el VPS: `cd deploy_contabo && ./update.sh` (git pull + rebuild incremental). Cambiar una key: editar `.env` y `docker compose up -d` **sin** `--build`.

`models/Despliegue de modelos.txt` son notas históricas del despliegue anterior en Azure Container Apps. **Contiene una `API_KEY` real en claro (`sk_79ab…`), versionada en git**, y sus URLs de `az acr build` apuntan al repo viejo `luiscas03/`. Esa key debe considerarse comprometida: no reutilizarla ni copiarla a ningún `.env`. Limpiarla exige reescribir historia de git, así que no se ha tocado — es una decisión del dueño del repo.

## Compatibilidad de artefactos — la fuente nº1 de fallos

Los artefactos del repo son de **tres generaciones de scikit-learn incompatibles entre sí**. No existe una sola combinación de versiones que las cargue todas; por eso hay tres venvs. Matriz verificada empíricamente (2026-07-23) probando 1.2.2 / 1.3.2 / 1.4.2 / 1.5.2 / 1.6.1 / 1.7.0 / 1.9.0:

| Artefactos | Requieren | Síntoma si te equivocas de versión |
|---|---|---|
| `assets/models/monitor/*` | sklearn **1.3.2** + numpy<2 | En ≥1.4: `Gradient_Boosting` → `No module named 'sklearn.ensemble._gb_losses'`; `Random_Forest` carga pero al predecir da `'DecisionTreeRegressor' object has no attribute 'monotonic_cst'` |
| `assets/models/root/*` | sklearn **1.5–1.6** + numpy**≥2** | Fuera de rango: `Can't get attribute '_RemainderColsList'`. Con numpy<2: `MT19937 is not a known BitGenerator module` |
| `models/*` | sklearn **1.4.2** + numpy<2 | `__pyx_unpickle_CyHalfBinomialLoss` |

**Consecuencia operativa, importante:** bajo sklearn 1.6.1 el ensemble monitor entrega **5 de 7** modelos — `random_forest` y `gradient_boosting` salen como `null` en `predicciones_individuales` y **la media se calcula solo con los supervivientes**. El fallo es silencioso: `_predict_monitor()` captura la excepción por modelo y continúa. El `Dockerfile` de la raíz (`python:3.9-slim` + `requirements.txt` sin pinear) resuelve a sklearn 1.6.1, así que **producción también está sirviendo con 5/7 sin avisar**. Si ves `null` en `predicciones_individuales`, es esto, no un problema del payload.

No se puede simplemente bajar el gateway a sklearn 1.3.2: `load_artifacts()` hace `raise` si no puede cargar el pipeline de `MODEL_PATH`, que bajo 1.3.2 no carga → la app no arranca. Ese pipeline es justamente el código muerto descrito arriba. Hacerlo opcional recuperaría los 7/7, pero **es un cambio de la semántica fail-fast de producción: consultar antes de aplicarlo.**

Los postprandiales necesitan además un shim de módulos privados que aplican en `install_sklearn_pickle_compat()`, antes de cualquier `joblib.load`:

```python
sys.modules["_loss"] = importlib.import_module("sklearn._loss._loss")
```

Si un modelo nuevo no deserializa, la causa casi siempre es la versión de sklearn/numpy o este alias, no el archivo. Antes de depurar código, comprueba la carga en aislamiento con `joblib.load` en el venv correcto.

`cnn_lstm_*` fija `tensorflow==2.16.1` (imagen Docker ~2 GB, primer build lento).

## Estado de validación de los modelos — leer antes de tocar cualquier predicción

`VALIDACION_MODELOS_2026-06-04.md` e `INFORME_VALIDACION_BIOMARCADORES_2026-06-12.md` documentan que **la mayoría de los modelos están rotos** y por qué. Resumen operativo:

- `postprandial_cluster_0` (M5): el RF discrimina (raw 0.001–0.73) pero el **calibrador isotónico aplasta** el rango operativo (0.02–0.30) a ≈0.208 → siempre "MEDIUM".
- `postprandial_cluster_1` (M6): el más sano; discrimina y **no tiene isotónico**.
- `postprandial_global_m1`: raw comprimido (máx 0.084) → siempre 0.06 / "LOW". Es el `fallback_policy` de todos los demás, lo cual propaga el problema.
- `glucose_regression_v2` (M3): **colapsado**, devuelve ~49 mg/dL constante para cualquier entrada.
- `cnn_lstm_16p_clean_v1` (M2): devuelve ~121–129 mg/dL sobre ondas PPG distintas; no sigue la señal, MAE ~16.
- **Ningún modelo postprandial responde a las features de comida**: variar `carbs` de 20 a 120 con demografía fija no mueve la salida.

Al modificar código de predicción, no interpretes estas salidas constantes como un bug recién introducido — son el estado conocido de los artefactos. Y al contrario: si una salida empieza a variar, verifica que no sea un cambio accidental de features/escalado.
