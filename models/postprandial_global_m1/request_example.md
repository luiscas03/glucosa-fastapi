# Postprandial Global M1

## Servicio

- App: `models.postprandial_global_m1.main:app`
- Endpoint: `POST /predict`
- Tipo: riesgo de pico glucemico postprandial
- El body debe enviarse como `{ "features": { ... } }`

## JSON de ejemplo

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

## Header

- `Content-Type: application/json`
- `X-API-Key: <tu_clave>` si `API_KEY` existe en `.env`

## Nota

- Si quieres ver la lista exacta del modelo cargado, consulta `GET /features`.
- Si envias el JSON plano en vez de `features`, el endpoint devolvera `422`.
