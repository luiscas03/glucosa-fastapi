# Postprandial Cluster 0 API

## Run Local

```bash
uvicorn models.postprandial_cluster_0.main:app --host 0.0.0.0 --port 8003
```

## Routes

- `GET /health`
- `GET /metadata`
- `GET /features`
- `POST /predict`

## Headers

- `Content-Type: application/json`
- `X-API-Key: <valor>` si `API_KEY` existe en `.env`

## Notes

- Usa `model_cluster_0.joblib`
- `iso_cluster_0.joblib` no es otro predictor: es un calibrador opcional que ajusta el score final
- Las features por defecto salen de `deploy_metadata.json`
- `POST /predict` requiere que el body venga envuelto dentro de `features`

## JSON Para `/predict`

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

## Flujo Recomendado

- `GET /health`: confirma si el predictor y el calibrador quedaron cargados
- `GET /features`: devuelve el orden exacto esperado por el modelo
- `POST /predict`: recibe el objeto `features`

## Error Comun

Si envias el JSON plano, FastAPI respondera `422` porque falta `body.features`.
