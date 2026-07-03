# Validación de modelos — glucosa-fastapi (para el equipo ML)

> Fecha: 2026-06-04. Validación local de los modelos serializados + confirmación
> contra los endpoints Azure en vivo. Objetivo: identificar qué modelos están
> sanos, cuáles están rotos y por qué, con evidencia reproducible.
>
> **Entorno de carga:** los `.joblib/.pkl` se serializaron con **scikit-learn
> 1.4.2** y NO cargan con sklearn 1.8 (`__pyx_unpickle_CyHalfBinomialLoss`).
> Requiere además el shim `sys.modules["_loss"] = sklearn._loss._loss` (el mismo
> que aplica `main.py`). → **Riesgo de reproducibilidad: fijar sklearn==1.4.2.**

---

## 1. Veredicto por modelo

| Modelo | Estado | Causa raíz |
|---|---|---|
| **postprandial_cluster_0** (default) | ⚠️ **Calibrador roto** | El modelo RF **sí discrimina** (raw 0.001–0.73), pero el **isotónico aplasta** todo el rango práctico (raw 0.02–0.30) a **0.208** → siempre "MEDIUM" |
| **postprandial_cluster_1** | ✅ El más sano | raw discrimina (0.019–0.50), **sin isotónico** → su salida varía de verdad |
| **postprandial_global_m1** | ❌ **Congelado** | El RF está **sub-discriminando**: raw máx = 0.0836 → isotónico → **0.06 siempre LOW** |
| **glucose_regression_v2** (Regrix) | ❌ **Colapsado** | RandomForest devuelve **~49 mg/dL constante** para cualquier entrada |
| **RF_GlucosaMujeres** (base 7-modelos) | 🟡 Vivo | Responde (~103–109 mg/dL) pero con bug de género (auditoría Delfos) |
| **cnn_lstm_16p_clean** (M2) | 🟡 Piloto | Vivo (124 mg/dL), TFLite OK, pero MAE~16, sin paper P01 |
| **cnn_lstm_p001_p002** (M1) | ❌ 500 | Endpoint Azure da Internal Server Error; piloto 2 pacientes |

---

## 2. Evidencia

### 2.1 Sensibilidad (200 inputs aleatorios, rangos clínicos, seed 42)

```
postprandial_cluster_0   raw: min=0.0011 max=0.7299 std=0.1211  (174 únicos)
postprandial_cluster_1   raw: min=0.0193 max=0.5029 std=0.1138  (141 únicos)
postprandial_global_m1   raw: min=0.0005 max=0.0836 std=0.0103  (119 únicos)  ← comprimido
glucose_regression_v2    mg/dL: min=48.98 max=49.60 std=0.13    (12 únicos)   ← colapsado
```

### 2.2 Variando SOLO la comida (carbs 20→120, demografía fija)

```
cluster_0:  carbs 20/55/120 -> raw=0.0359 (constante)  cal=0.2085  MEDIUM
cluster_1:  carbs 20/55/120 -> raw=0.0492 (constante)  LOW
global_m1:  carbs 20/55/120 -> raw=0.0050 (constante)  cal=0.06    LOW
```
→ **Ningún modelo postprandial responde a la comida** cuando la demografía es fija.
La variación del punto 2.1 viene de las features demográficas, **no** de la comida.
**Para un modelo "postprandial" esto es al revés de lo esperado** (debería pesar
carbohidratos/energía/IG).

### 2.3 Calibradores isotónicos (raw → calibrado)

```
cluster_0:  raw 0.02→0.205 · 0.05–0.30 TODOS →0.208 · 0.50→0.239   (rango 0–0.242)
global_m1:  raw 0.02→0.185 · 0.10→0.275 · 0.50→0.402 · 0.90→0.632  (rango 0–0.632)
```
→ **El isotónico de cluster_0 mapea casi todo el rango operativo (0.02–0.30) a
≈0.208.** Como las predicciones reales caen ~0.04, **siempre** sale 0.208 → MEDIUM.
El de global es razonable, pero el RAW de global está tan comprimido (≤0.08) que
igual termina fijo en 0.06.

### 2.4 Endpoints Azure en vivo (test directo, 2026-06-04)

```
RF base            200  prediccion_final=103.36  Prediabetes (conf 0.936)
Regrix v2          200  prediction=49.25  "Normal"        ← coincide con 2.1 (colapsado)
Prandial Global    200  risk=0.06  LOW                     ← coincide con 2.3 (congelado)
Prandial C1        200  risk=0.0492  LOW
Prandial C0        200  risk_calibrated=0.2085  MEDIUM     ← coincide con 2.3 (calibrador)
Neural 16p         200  prediction=124.21 mg/dL  (shape [1,300,1])
Neural V1          500  Internal Server Error
```

---

## 3. Causas raíz y recomendaciones

1. **cluster_0 — recalibrar (prioridad alta).** El modelo discrimina; el problema
   es el **isotónico** que aplasta el rango operativo a 0.208. Re-ajustar la
   calibración con datos reales (o quitarla, como cluster_1) recuperaría la señal.

2. **global_m1 — reentrenar.** El RAW está comprimido (máx 0.08) → no discrimina.
   No usar como fallback hasta corregir (hoy es el `fallback_policy` de todos).

3. **glucose_regression_v2 (Regrix) — bug grave.** Devuelve ~49 mg/dL constante
   (std 0.13 sobre 200 inputs). Posibles causas: target degenerado en
   entrenamiento, fuga/limpieza que dejó el target plano, o mismatch
   scaler↔modelo. **No usar en producción** hasta depurar. Además 49 mg/dL es
   hipoglucemia severa (rango no fisiológico para "Normal").

4. **Features de comida sin peso (todos los postprandiales).** Variar carbs 20→120
   no mueve la salida. Revisar `feature_importances_` — si los `Meal_*` tienen
   importancia ~0, el modelo no aprendió la señal posprandial.

5. **Considerar cluster_1 como default** mientras se arregla cluster_0: discrimina
   (0.019–0.50) y no tiene el calibrador roto.

6. **Reproducibilidad:** fijar `scikit-learn==1.4.2` en todos los `requirements.txt`
   y documentar el shim `_loss`. Idealmente reserializar con `skops` o exportar a
   ONNX para no depender de la versión exacta de sklearn.

7. **Móvil offline (CNN TFLite):** `glucose_cnn_lstm_16p_CLEAN_v1_unroll.tflite`
   (95 KB, nativo, input [1,17,1]) sirve como respaldo offline, **pero es piloto**
   (MAE~16, sin P01). Para entrega clínica oficial, convertir **M5/M6** (sklearn
   HistGB) vía ONNX→TFLite.

---

## 4. Inventario de modelos (`models/`)

| Carpeta | Archivo | Tamaño |
|---|---|---|
| glucose_regression_v2 | glucose_model_v2_clean.pkl (+ scaler) | 67 MB |
| postprandial_cluster_0 | model_cluster_0.joblib + iso_cluster_0.joblib | 929 KB |
| postprandial_cluster_1 | model_cluster_1.joblib | 615 KB |
| postprandial_global_m1 | model_global_m1.joblib + iso_global_m1.joblib | 884 KB |
| cnn_lstm_16p_clean_v1 | glucose_cnn_lstm_16p_CLEAN_v1.keras | 196 KB |
| cnn_lstm_p001_p002_v1 | glucose_cnn_lstm_p001_p002_v1.keras | 150 KB |
| assets/models/root | RF_GlucosaMujeres.joblib, modelo_gradient_boosting_2.joblib | — |
| assets/models/monitor | ElasticNet/Lasso/Ridge/RF/XGBoost/LightGBM/GB + preprocessing | — |

Notebooks de entrenamiento: `Modelos Biomarcadores/` (v1, v2, v4, exploración
Pac001-003, análisis ML). Según el MANIFEST de TFLite, validación 80/20 random
**sin LOPO**, MAE ~16 mg/dL.
