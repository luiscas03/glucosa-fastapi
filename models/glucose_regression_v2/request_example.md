# Glucose Regression V2

## Servicio

- App: `models.glucose_regression_v2.main:app`
- Endpoint: `POST /predict`
- Tipo: regresion tabular de glucosa
- El body se envia plano, sin `features`

## JSON de ejemplo

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

## Header

- `Content-Type: application/json`
- `X-API-Key: <tu_clave>` si `API_KEY` existe en `.env`
