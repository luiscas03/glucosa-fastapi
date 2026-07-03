# Postman Endpoints

Este documento resume las URLs locales para probar los servicios del proyecto en Postman, explica para que sirve cada endpoint y detalla que modelo usa por detras.

## Header Comun

Usa estos headers en los endpoints protegidos (todos los `POST /predict`):

```http
Content-Type: application/json
X-API-Key: TU_API_KEY
```

La API key se configura en la variable de entorno `API_KEY` (archivo `.env`).

## Que Hace Cada Endpoint

- `GET /health`
  Comprueba si el servicio arranco bien, si el modelo cargo y si las dependencias estan disponibles.

- `GET /features`
  Devuelve la lista de variables esperadas por el modelo. Util para revisar nombres exactos antes de construir el body.

- `GET /metadata`
  Solo existe en los postprandiales. Devuelve metadatos del deploy, target y configuracion base del modelo.

- `POST /predict`
  Ejecuta la prediccion normal del servicio.

- `POST /predict_typed`
  Solo existe en `glucose_regression_v2` y en la API principal. Hace la misma prediccion pero con validacion tipada por schema.

---

## 0. API Principal (Gateway)

Base URL local:

```text
http://127.0.0.1:8000
```

Modelo(s):

| Algoritmo | Archivo |
|---|---|
| XGBoost | `assets/models/monitor/XGBoost.joblib` |
| Random Forest | `assets/models/monitor/Random_Forest.joblib` |
| LightGBM | `assets/models/monitor/LightGBM.joblib` |
| Gradient Boosting | `assets/models/monitor/Gradient_Boosting.joblib` |
| Ridge | `assets/models/monitor/Ridge.joblib` |
| Lasso | `assets/models/monitor/Lasso.joblib` |
| ElasticNet | `assets/models/monitor/ElasticNet.joblib` |
| Preprocesamiento | `assets/models/monitor/preprocessing_objects.pkl` |

Tipo: Ensemble (promedio de los 7 modelos). La prediccion final es la media y la respuesta indica cual modelo individual estuvo mas cerca.

Endpoints:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/monitor/health`
- `GET http://127.0.0.1:8000/monitor/models`
- `POST http://127.0.0.1:8000/predict`
- `POST http://127.0.0.1:8000/predict_typed`
- `POST http://127.0.0.1:8000/api/v1/glucose/predict` (alias de /predict)
- `POST http://127.0.0.1:8000/api/v1/glucose/predict_typed` (alias de /predict_typed)
- `POST http://127.0.0.1:8000/monitor/predict`
- `POST http://127.0.0.1:8000/api/v1/ppg/measure`
- `GET http://127.0.0.1:8000/docs`

Body para `/predict`, `/monitor/predict` y `/api/v1/glucose/predict`:

```json
{
  "edad": 45,
  "sexo": "Masculino",
  "peso": 82.5,
  "talla": 1.72,
  "perimetro_cintura": 95.0,
  "spo2": 96,
  "frecuencia_cardiaca": 78,
  "actividad_fisica": "Moderada",
  "consumo_frutas": "Diario",
  "tiene_hipertension": "No",
  "tiene_diabetes": "No",
  "puntaje_findrisc": 12
}
```

Campos requeridos:

| Campo | Tipo | Validacion |
|---|---|---|
| `edad` | int | 18 - 120 |
| `sexo` | string | "Masculino" / "Femenino" |
| `peso` | float | > 0 |
| `talla` | float | > 0 (en metros) |
| `imc` | float | Opcional, se calcula si no se envia |
| `perimetro_cintura` | float | > 0 |
| `spo2` | int | 70 - 100 |
| `frecuencia_cardiaca` | int | 40 - 200 |
| `actividad_fisica` | string | Nivel de actividad |
| `consumo_frutas` | string | Frecuencia de consumo |
| `tiene_hipertension` | string | "Si" / "No" |
| `tiene_diabetes` | string | "Si" / "No" |
| `puntaje_findrisc` | int | 0 - 26 |

Respuesta:

```json
{
  "prediccion_final": 105.34,
  "categoria": "Prediabetes",
  "predicciones_individuales": {
    "xgboost": 103.21,
    "random_forest": 107.45,
    "lightgbm": 104.89,
    "gradient_boosting": 106.12,
    "ridge": 105.67,
    "lasso": 104.33,
    "elasticnet": 105.89
  },
  "confidence": 0.923,
  "intervalo_confianza": [98.12, 112.56],
  "mejor_modelo": "lightgbm",
  "timestamp": "2026-06-02T10:30:00",
  "input_data": { "..." : "..." }
}
```

Categorias de glucosa:

- `< 100 mg/dL` = Normal
- `100 - 125 mg/dL` = Prediabetes
- `>= 126 mg/dL` = Diabetes

Body para `/api/v1/ppg/measure` (analisis de senal PPG):

```json
{
  "r": [0.45, 0.46, 0.44, 0.47, "...minimo 100 muestras"],
  "g": [0.52, 0.53, 0.51, 0.54, "...misma longitud que r"],
  "b": [0.31, 0.32, 0.30, 0.33, "...misma longitud que r"],
  "fps": 30
}
```

| Campo | Tipo | Requerido | Nota |
|---|---|---|---|
| `r` | List[float] | Si | Canal rojo, minimo 100 muestras |
| `g` | List[float] | Si | Canal verde, misma longitud que r |
| `b` | List[float] | Si | Canal azul, misma longitud que r |
| `fps` | float | Uno u otro | Frames por segundo |
| `timestamps` | List[float] | Uno u otro | Alternativa a fps |

Notas:

- El endpoint PPG no usa modelos ML. Usa el algoritmo CHROM para analisis de senal.
- Devuelve BPM estimado, confianza y metricas de calidad de senal.

---

## 1. Glucose Regression V2

Base URL local:

```text
http://127.0.0.1:8001
```

Modelo: `models/glucose_regression_v2/glucose_model_v2_clean.pkl` (64 MB)
Scaler: `models/glucose_regression_v2/scaler_v2_clean.pkl`
Features: `models/glucose_regression_v2/features_v2.txt` (24 variables)

Tipo: Regresion tabular. Predice valor de glucosa en mg/dL.

Endpoints:

- `GET http://127.0.0.1:8001/health`
- `GET http://127.0.0.1:8001/features`
- `POST http://127.0.0.1:8001/predict`
- `POST http://127.0.0.1:8001/predict_typed`
- `GET http://127.0.0.1:8001/docs`

Body para `/predict` (plano, 24 features):

```json
{
  "food_time_since": 45,
  "food_has_recent": 1,
  "food_carbs": 55,
  "food_protein": 22,
  "food_gi": 65,
  "food_size": 1,
  "food_absorption": 0.7,
  "demo_age_norm": 0.45,
  "demo_gender": 1,
  "demo_bmi_norm": 0.52,
  "demo_circ_norm": 0.48,
  "demo_antihypertensive": 0,
  "demo_family_diabetes": 1,
  "demo_glucose_history": 0,
  "demo_activity": 0.6,
  "demo_diet": 0.7,
  "demo_hba1c_norm": 0.5,
  "glucose_lag1": 98,
  "glucose_lag2": 96,
  "glucose_lag3": 94,
  "glucose_delta_1": 2,
  "glucose_delta_2": 4,
  "glucose_ma3_past": 96,
  "glucose_std3_past": 2
}
```

Grupos de features:

- **Alimentacion** (food_*): tiempo desde comida, carbohidratos, proteina, indice glucemico, tamano, absorcion
- **Demografia normalizada** (demo_*): edad, genero, BMI, circunferencia, antihipertensivos, historial familiar, actividad, dieta, HbA1c
- **Historial de glucosa** (glucose_*): 3 mediciones previas (lag), deltas, media movil, desviacion estandar

Notas:

- `/predict` acepta el body plano. Campos faltantes se imputan con `0.0`.
- `/predict_typed` usa el mismo contenido pero fuerza validacion de tipos.
- Respuesta incluye `model_name: "glucose_model_v2_clean.pkl"`.

---

## 2. Postprandial Global M1

Base URL local:

```text
http://127.0.0.1:8002
```

Modelo: `models/postprandial_global_m1/model_global_m1.joblib`
Calibrador: `models/postprandial_global_m1/iso_global_m1.joblib` (isotonic regression)
Metadata: `models/postprandial_global_m1/deploy_metadata.json`

Tipo: Clasificacion de riesgo postprandial. Predice probabilidad de pico glucemico > 120 mg/dL (`is_high_peak_120`). Modelo global entrenado con todos los pacientes.

Endpoints:

- `GET http://127.0.0.1:8002/health`
- `GET http://127.0.0.1:8002/features`
- `GET http://127.0.0.1:8002/metadata`
- `POST http://127.0.0.1:8002/predict`
- `GET http://127.0.0.1:8002/docs`

Body para `/predict` (26 features dentro de `"features"`):

```json
{
  "features": {
    "Age": 28,
    "Gender": 1,
    "BMI": 24.4,
    "Abd_Circum": 90,
    "Antihypertensive": 0,
    "Family_Diabetes": 1,
    "Glucose_History": 0,
    "Physical_Activity_Score": 0.6,
    "Diet_Quality_Score": 0.7,
    "HbA1c": 5.4,
    "Time_Since_Meal": 45,
    "Has_Recent_Meal": 1,
    "Meal_Carbs": 55,
    "Meal_Protein": 22,
    "Meal_Fat": 14,
    "Meal_GI": 65,
    "Meal_Size": 1,
    "Carb_Absorption": 0.7,
    "Activity_Mean": 0.24,
    "Activity_Std": 0.08,
    "Activity_Max": 0.51,
    "Jerk_Magnitude": 0.12,
    "Walk_Power_Spectral": 0.34,
    "Activity_Intensity": 0.29,
    "Is_Sedentary": 0,
    "Total_Energy": 135
  }
}
```

Bandas de riesgo:

- `< 0.20` = LOW
- `0.20 - 0.29` = MEDIUM
- `>= 0.30` = HIGH

Respuesta incluye `model_variant: "global"`.

Notas:

- El body NO es plano. Debe venir dentro de `"features"`.
- Todas las 26 features son requeridas. Si falta alguna, devuelve error con `missing_features`.

---

## 3. Postprandial Cluster 0

Base URL local:

```text
http://127.0.0.1:8003
```

Modelo: `models/postprandial_cluster_0/model_cluster_0.joblib`
Calibrador: `models/postprandial_cluster_0/iso_cluster_0.joblib` (isotonic regression)
Metadata: `models/postprandial_cluster_0/deploy_metadata.json`

Tipo: Clasificacion de riesgo postprandial para pacientes del Cluster 0 (11 pacientes, patron metabolico tipo A). Target: `is_high_peak_120`.

Endpoints:

- `GET http://127.0.0.1:8003/health`
- `GET http://127.0.0.1:8003/features`
- `GET http://127.0.0.1:8003/metadata`
- `POST http://127.0.0.1:8003/predict`
- `GET http://127.0.0.1:8003/docs`

Body para `/predict` (mismas 26 features que el global):

```json
{
  "features": {
    "Age": 28,
    "Gender": 1,
    "BMI": 24.4,
    "Abd_Circum": 90,
    "Antihypertensive": 0,
    "Family_Diabetes": 1,
    "Glucose_History": 0,
    "Physical_Activity_Score": 0.6,
    "Diet_Quality_Score": 0.7,
    "HbA1c": 5.4,
    "Time_Since_Meal": 45,
    "Has_Recent_Meal": 1,
    "Meal_Carbs": 55,
    "Meal_Protein": 22,
    "Meal_Fat": 14,
    "Meal_GI": 65,
    "Meal_Size": 1,
    "Carb_Absorption": 0.7,
    "Activity_Mean": 0.24,
    "Activity_Std": 0.08,
    "Activity_Max": 0.51,
    "Jerk_Magnitude": 0.12,
    "Walk_Power_Spectral": 0.34,
    "Activity_Intensity": 0.29,
    "Is_Sedentary": 0,
    "Total_Energy": 135
  }
}
```

Respuesta incluye `model_variant: "cluster_0"`.

---

## 4. Postprandial Cluster 1

Base URL local:

```text
http://127.0.0.1:8004
```

Modelo: `models/postprandial_cluster_1/model_cluster_1.joblib`
Calibrador: No disponible (`iso_cluster_1.joblib` no existe, funciona sin calibracion)
Metadata: `models/postprandial_cluster_1/deploy_metadata.json`

Tipo: Clasificacion de riesgo postprandial para pacientes del Cluster 1 (4 pacientes, patron metabolico tipo B). Target: `is_high_peak_120`.

Endpoints:

- `GET http://127.0.0.1:8004/health`
- `GET http://127.0.0.1:8004/features`
- `GET http://127.0.0.1:8004/metadata`
- `POST http://127.0.0.1:8004/predict`
- `GET http://127.0.0.1:8004/docs`

Body para `/predict` (mismas 26 features):

```json
{
  "features": {
    "Age": 28,
    "Gender": 1,
    "BMI": 24.4,
    "Abd_Circum": 90,
    "Antihypertensive": 0,
    "Family_Diabetes": 1,
    "Glucose_History": 0,
    "Physical_Activity_Score": 0.6,
    "Diet_Quality_Score": 0.7,
    "HbA1c": 5.4,
    "Time_Since_Meal": 45,
    "Has_Recent_Meal": 1,
    "Meal_Carbs": 55,
    "Meal_Protein": 22,
    "Meal_Fat": 14,
    "Meal_GI": 65,
    "Meal_Size": 1,
    "Carb_Absorption": 0.7,
    "Activity_Mean": 0.24,
    "Activity_Std": 0.08,
    "Activity_Max": 0.51,
    "Jerk_Magnitude": 0.12,
    "Walk_Power_Spectral": 0.34,
    "Activity_Intensity": 0.29,
    "Is_Sedentary": 0,
    "Total_Energy": 135
  }
}
```

Respuesta incluye `model_variant: "cluster_1"`.

Nota: Este servicio opera sin calibracion isotonica porque el archivo `iso_cluster_1.joblib` no existe. El score raw se devuelve como calibrated tambien.

---

## 5. CNN LSTM 16P Clean V1

Base URL local:

```text
http://127.0.0.1:8005
```

Modelo: `models/cnn_lstm_16p_clean_v1/glucose_cnn_lstm_16p_CLEAN_v1.keras`

Tipo: Red neuronal CNN-LSTM. Entrenada con 16 pacientes (datos limpios). Requiere TensorFlow instalado.

Endpoints:

- `GET http://127.0.0.1:8005/health`
- `POST http://127.0.0.1:8005/predict`
- `GET http://127.0.0.1:8005/docs`

Body para `/predict`:

```json
{
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "shape": [6, 1]
}
```

| Campo | Tipo | Requerido | Nota |
|---|---|---|---|
| `values` | List[float] | Si | Secuencia numerica de entrada |
| `shape` | List[int] | No | Dimensiones del tensor. Si no se envia, se infiere del modelo |

Respuesta incluye `model_name: "clean_v1"`.

Notas:

- Si TensorFlow no esta instalado, `/health` devuelve `status: "dependency_missing"` y `/predict` devuelve 503.
- Si el numero de valores no coincide con el shape esperado, devuelve error 400.

---

## 6. CNN LSTM P001 P002 V1

Base URL local:

```text
http://127.0.0.1:8006
```

Modelo: `models/cnn_lstm_p001_p002_v1/glucose_cnn_lstm_p001_p002_v1.keras`

Tipo: Red neuronal CNN-LSTM. Entrenada con pacientes P001 y P002 especificamente. Requiere TensorFlow instalado.

Endpoints:

- `GET http://127.0.0.1:8006/health`
- `POST http://127.0.0.1:8006/predict`
- `GET http://127.0.0.1:8006/docs`

Body para `/predict`:

```json
{
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "shape": [6, 1]
}
```

Respuesta incluye `model_name: "p001_p002_v1"`.

Mismas notas que CNN LSTM 16P Clean V1.

---

## Mapeo Rapido: Puerto a Modelo

| Puerto | Servicio | Archivo(s) del Modelo | Tipo |
|---|---|---|---|
| 8000 | API Principal (Gateway) | 7 modelos en `assets/models/monitor/*.joblib` | Ensemble (promedio) |
| 8001 | Glucose Regression V2 | `glucose_model_v2_clean.pkl` + `scaler_v2_clean.pkl` | Regresion tabular |
| 8002 | Postprandial Global M1 | `model_global_m1.joblib` + `iso_global_m1.joblib` | Clasificacion con calibracion |
| 8003 | Postprandial Cluster 0 | `model_cluster_0.joblib` + `iso_cluster_0.joblib` | Clasificacion con calibracion |
| 8004 | Postprandial Cluster 1 | `model_cluster_1.joblib` (sin calibrador) | Clasificacion sin calibracion |
| 8005 | CNN LSTM 16P Clean | `glucose_cnn_lstm_16p_CLEAN_v1.keras` | Red neuronal (TensorFlow) |
| 8006 | CNN LSTM P001 P002 | `glucose_cnn_lstm_p001_p002_v1.keras` | Red neuronal (TensorFlow) |

## Flujo Recomendado En Postman

1. Probar `GET /health` para verificar que el servicio y modelo estan cargados
2. Probar `GET /features` o `GET /metadata` si existe, para ver las variables esperadas
3. Enviar `POST /predict` con el body correspondiente
4. Si un servicio tiene `/predict_typed`, usarlo solo cuando quieras validacion mas estricta

## Diagnostico Rapido

Si un endpoint no responde como esperas:

- `GET /health` te dice si el modelo cargo correctamente
- `GET /features` te confirma los nombres exactos de las variables
- `GET /monitor/models` (solo puerto 8000) lista los 7 modelos del ensemble
- `GET /metadata` (solo postprandiales) muestra target, features y configuracion de clusters
