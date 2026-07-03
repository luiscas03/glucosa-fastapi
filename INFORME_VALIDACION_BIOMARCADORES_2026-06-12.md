# Informe de validación — Biomarcadores (SpO2 on-device + glucosa cloud)

> Fecha: 2026-06-12 · Para: equipo ML · Contexto: requerimiento de validación de
> la Universidad Distrital (Proyecto 111626) antes de reanudar campo.
> Toda la evidencia es reproducible (consola/psql/onnxruntime/endpoints en vivo).

---

## 0. Resumen ejecutivo

Hoy **ni el SpO2 on-device ni los modelos de glucosa en la nube pasarían** una
validación contra dispositivo de referencia. Ambos predicen cerca de la **media
poblacional** sin discriminar la señal real:

- **SpO2-CNN (ONNX on-device):** 7 de 8 capturas reales se **rechazan** por el
  quality-gate; cuando corre da **71-82 % (hipoxémico) vs 94-95 % real** (error
  ~15-20 pts). Causa estructural (no es del Note 8): con dedo+flash el **rojo
  satura** y **verde/azul quedan en el piso de ruido**, y la cámara captura a
  **~11-20 Hz, no a 30 fps**.
- **Glucosa Neural (cnn_lstm_16p, el que la app realmente usa):** sobre 8 ondas
  PPG reales distintas devuelve **121-129 mg/dL** (≈constante ~125). No sigue la
  señal. MAE~16 mg/dL.
- **glucose_regression_v2 (Regrix):** en el escenario real de la app (sin
  historial de glucosa) colapsa a **~50 mg/dL constante**. Ya está **oculto** en
  la app por features faltantes.

---

## 1. SpO2-CNN on-device

### 1.1 Verificación del artefacto (OK)
- ONNX desplegado (`assets/models/spo2_cnn_acdc.onnx`) == el de
  `Downloads/Modelo/` (md5 idéntico). Config == `flutter_config.json` == stats de
  `preproc_acdc.npz`. Preprocesado Dart reproduce el contrato (AC/DC →
  `(x-mu)/sd` → [1,3,90] R,G,B → SpO2[70,100]). I/O del modelo: `input`/`spo2`.
- Calidad del modelo (entrenamiento, Hoffman et al. 2022): **MAE 6.46 % LOOCV,
  AUC 0.796**, varianza enorme entre sujetos (subj_2 MAE 13 %). No clínico.

### 1.2 Validación con datos reales (8 capturas, Redmi Note 8)
| id | sr (Hz) | DC rojo | DC verde | DC azul | gate | CNN si corre | empírico |
|---|---|---|---|---|---|---|---|
| 4 | 13.5 | 130.6 | 5.1 | 2.2 | ❌ azul | 74.2 | 94 |
| 5 | 17.2 | 105.7 | 4.5 | 3.7 | ❌ verde | 71.3 | 95 |
| 6 | 13.5 | 232.7 | 13.4 | 5.6 | ✅ pasa | 79.2 | — |
| 7 | 13.0 | 132.0 | 2.4 | 1.3 | ❌ verde | 82.2 | 95 |
| 8 | 12.4 | 134.1 | 3.8 | 2.1 | ❌ verde | 76.3 | 94 |
| 9 | 12.3 | 136.1 | 4.0 | 1.9 | ❌ verde | 76.3 | 95 |
| 10 | 20.2 | 131.7 | 2.3 | 1.5 | ❌ verde | 74.6 | 94 |
| 11 | 11.4 | 106.4 | 5.0 | 4.2 | ❌ verde | 75.8 | 94 |

### 1.3 ¿Es específico del Note 8? No.
- **Desbalance R/G/B = física** del PPG por reflexión con torch blanco a través
  del dedo → universal. La app misma lo reconoce: `dcGreen<20 ⇒ usa ROJO para
  BPM` (todas nuestras capturas tienen green DC 2-13).
- **fps bajo:** mezcla de rendimiento del equipo y de la implementación (no fija
  30 fps ni bloquea exposición). Un gama alta mejora el fps pero no cumple el
  contrato 30fps-locked sin cambios de captura.
- **Conclusión:** esperar el mismo comportamiento en la mayoría de la flota.

### 1.3b Medición EN VIVO en el Note 8 (2026-06-12, logcat)
- **FPS real = 12.0 Hz** (log `[SpO2] dcRed=111.7 dcGreen=2.7 fs=12.0`) vs 30 fps
  requeridos. La HAL descarta frames (`mm-camera ... skip`, `BufferQueue
  abandoned`): el equipo no sostiene 30fps con torch+ImageReader.
- **Exposición OK y bloqueada:** rojo arrancó saturado (dc=238.9), offset -2.2 →
  dc=117.1 → `Bloqueado dc_final=117.7`. Esta parte funciona; el rojo queda bien
  expuesto sin clipping.
- **Verde = 2.7 (piso de ruido):** la app conmuta a rojo (`Canal ROJO para BPM`,
  `Warm-LED detectado`).
- **El empírico YA usa rojo de canal único:** `[SpO2-SC] dcRed=111.7 acRed=0.826
  piRed=0.0074 → spo2_estimated=95`. Es el rediseño correcto, ya implementado en
  la vía empírica. El CNN se rechaza por exigir verde (`DC[1]=2.8 < 5`).
- **Veredicto:** la exposición NO es el problema; lo son (a) 12 fps vs 30 (HW/HAL)
  y (b) el modelo exige verde/azul inexistentes. El método piRed (rojo) es la guía.

### 1.3c A/B de resolución de captura (Note 8, medido en vivo)
| ResolutionPreset | ImageReader | fps real |
|---|---|---|
| `.medium` | 720x480 | 12.0 Hz |
| `.low` | 320x240 | 14.1 Hz |

→ Bajar resolución sube solo **~2 Hz (+18 %)**, sigue muy por debajo de 30. Confirma
que el cuello NO es el procesamiento (la app ya submuestrea: `step=14`, recorte
central → ~440 px/frame) sino el **AE/HAL con torch**. El plugin `camera ^0.11` no
expone `CONTROL_AE_TARGET_FPS_RANGE` → para forzar 30fps haría falta camera2/CameraX
nativo. Alternativa realista: **reentrenar el modelo a la tasa real (~12-15 fps)**.

### 1.4 Camino recomendado (ML)
1. Captura: **bloquear AE/AWB y fijar 30 fps**, evitar clipping del rojo.
2. Rediseñar el modelo para usar el **canal ROJO** (que sí tiene señal fuerte y
   limpia) en vez de gatear por verde/azul; o ratio rojo/verde.
3. **Reentrenar con datos reales phone+torch** (no el rig de Hoffman). El ground
   truth ya se recolecta: `ppg_training_samples.ref_spo2` + `raw_*_json`.

---

## 2. Modelos de glucosa en la nube (Azure)

### 2.1 Test definitivo con onda PPG REAL (validado contra producción)
Se replicó **exacto** el preprocesado de la app (`_denoiseGreen` NLMS green/red →
90 muestras → resample 30 Hz → baseline → z-score) y se mandó al endpoint Neural.
Validación de fidelidad: id=11 → `121.1928`, idéntico al log de producción
(`121.1927`).

| id | bpm | glucosa Neural cloud (mg/dL) |
|---|---|---|
| 4 | 84 | 127.98 |
| 5 | 84 | 128.09 |
| 6 | 72 | 123.55 |
| 7 | 120 | 129.22 |
| 8 | 86 | 125.40 |
| 9 | 80 | 129.13 |
| 10 | 76 | 122.11 |
| 11 | 106 | 121.19 |

→ rango total **8 mg/dL** sobre 8 ondas distintas: el modelo **no discrimina**,
predice ~125 (prediabetes leve). Agravado porque el input verde está en el piso
de ruido (la app alimenta verde al CNN de glucosa aunque sepa que es ruido).

### 2.2 Estado por modelo (reconfirmado en vivo 2026-06-12)
| Modelo | Estado | Evidencia |
|---|---|---|
| **glucose_regression_v2** (Regrix) | ❌ Fugado | con lags calca glucosa previa (85→85.8, 185→185.7); **sin lags → ~50 mg/dL fijo** |
| postprandial_cluster_0 | ⚠️ Calibrador roto | isotónico aplasta a 0.208 → siempre MEDIUM |
| postprandial_global_m1 | ❌ Congelado | raw ≤0.08 → siempre LOW |
| **postprandial_cluster_1** | ✅ El más sano | sin isotónico, discrimina (raw 0.019-0.50) |
| RF base "7-modelos" (RF_GlucosaMujeres) | 🟡 Vivo, bug género | ensamble Monitor (XGB/RF/LGBM/GB/Ridge/Lasso/ElasticNet) |
| **cnn_lstm_16p** (usado por la app) | 🟡 Piloto MAE~16 | ~125 constante (sección 2.1) |
| cnn_lstm_p001_p002 | ❌ 500 | Internal Server Error |

### 2.3 Insensibilidad al índice glucémico
La validación 2026-06-04 ya mostró que **ningún modelo postprandial responde a la
comida** (carbs/IG) con demografía fija; toda la variación viene de demografía.
Para un modelo "postprandial" esto está al revés de lo esperado.

---

## 3. Arquitectura de selección en la app (qué se usa de verdad)

`measurement_model_catalog.dart` filtra y puntúa modelos:
1. **Bloquea** RF base (bug de género, auditoría Delfos).
2. **Oculta** glucose_regression_v2, postprandial_global, cluster_0 y cluster_1
   por **features faltantes** (necesitan 24-26 campos clínicos/comida que la app
   no provee: Meal_Carbs, HbA1c, Abd_Circum, glucose_lags…).
3. → En la práctica **solo sobrevive el Neural CNN** (~125 constante).

**Hallazgos accionables:**
- **glucose_regression_v2 ya está evitado** (oculto por features). No requiere
  acción para "evitarlo".
- **Bug latente de preferencia:** el scoring premia `cluster_0` (+600) con un
  comentario obsoleto ("varía"); la auditoría 2026-06-04 probó que **cluster_0
  tiene el calibrador roto y cluster_1 es el sano**. Si se habilitan los
  postprandiales, **invertir la preferencia a `cluster_1`**
  (`measurement_model_catalog.dart`, `_modelSuitabilityScore`).
- **Para que el IG pese:** hay que (a) hacer llegar los features de comida
  (Meal_Carbs/GI) — el meal-sheet hoy solo pregunta ayunas/reciente/transcurrido,
  no cantidad/IG —, y (b) reentrenar los postprandiales para que ponderen comida.

---

## 4. Arreglos de infraestructura aplicados (2026-06-12)

- **`ppg_training_samples.id`**: era `bigint NOT NULL` sin default → todos los
  inserts de la app fallaban en silencio (tabla vacía). Arreglado con
  `GENERATED BY DEFAULT AS IDENTITY`. **Verificado en prod:** el backlog local
  sincronizó (0→8 filas; id=11 medido tras el fix).
- **RLS endurecido** en esa tabla: RLS ON; `anon` sin grants; `authenticated`
  solo INSERT/SELECT de lo propio (`user_id = auth.uid()`); `service_role`
  conserva todo. Probado con roles simulados.
- **Flag `enable_spo2_cnn`**: pasado a control remoto vía `app_config` (kill-switch
  sin release). **Recomendado dejarlo en `false`** hasta rediseñar SpO2.
- Pendiente menor: `ppg_training_samples.created_at` sin default (NULL) →
  `ALTER ... SET DEFAULT now()`.

---

## 5. Brecha vs requerimiento Universidad Distrital (`logs/Analisis.txt`)

Piden validación comparativa app vs dispositivo de referencia (oxímetro/glucómetro),
con metodología, n° de mediciones, condiciones, márgenes y análisis de discrepancias.

| Requerido | Estado hoy |
|---|---|
| SpO2 consistente vs oxímetro | ❌ CNN gateado/erróneo; solo el empírico aporta |
| Glucosa consistente vs glucómetro | ❌ Neural ~125 constante; tabulares rotos/fugados |
| Reproducibilidad técnica | ⚠️ artefactos OK y pipeline verificado, pero modelos no válidos |
| Dataset de validación | 🟢 recolección ya funcional (`ppg_training_samples` + `ref_*`) |

**Mínimo para llegar a la mesa técnica:** (1) captura 30fps+exposición bloqueada;
(2) SpO2 sobre canal rojo reentrenado; (3) modelo de glucosa que discrimine (no la
media) y, si es postprandial, que pondere IG/carbohidratos; (4) protocolo de
captura con referencia para construir el set de validación.

---

## Anexos / reproducir
- SpO2 real: `Downloads/Modelo/` + `onnxruntime`; datos en
  `public.ppg_training_samples` (Target `grrrvhuqhevphqnbvvgf`).
- Glucosa cloud: endpoints `*.ambitiousmushroom-987b2f42.eastus.azurecontainerapps.io/predict`,
  `X-API-Key` en `glucosa-fastapi/.env`; validación previa
  `VALIDACION_MODELOS_2026-06-04.md`.
- App: `Documents/flutter_projects/biomarcadores-app-flutter`
  (`heart_rate_page.dart` pipeline, `measurement_model_catalog.dart` selección).
