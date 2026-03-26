# Glucose Regression V2 API

## Run Local

```bash
uvicorn models.glucose_regression_v2.main:app --host 0.0.0.0 --port 8001
```

## Routes

- `GET /health`
- `GET /features`
- `POST /predict`
- `POST /predict_typed`

## Headers

- `Content-Type: application/json`
- `X-API-Key: <valor>` si `API_KEY` existe en `.env`

## Notes

- Usa `glucose_model_v2_clean.pkl`
- Usa `scaler_v2_clean.pkl`
- El orden de entrada se lee desde `features_v2.txt`
- `POST /predict` recibe el JSON plano, sin wrapper `features`

## JSON Para `/predict`

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

## JSON Para `/predict_typed`

```json
{
  "food_time_since": 45.0,
  "food_has_recent": 1.0,
  "food_carbs": 55.0,
  "food_protein": 22.0,
  "food_gi": 65.0,
  "food_size": 1.0,
  "food_absorption": 0.7,
  "demo_age_norm": 0.45,
  "demo_gender": 1.0,
  "demo_bmi_norm": 0.52,
  "demo_circ_norm": 0.48,
  "demo_antihypertensive": 0.0,
  "demo_family_diabetes": 1.0,
  "demo_glucose_history": 0.0,
  "demo_activity": 0.6,
  "demo_diet": 0.7,
  "demo_hba1c_norm": 0.5,
  "glucose_lag1": 98.0,
  "glucose_lag2": 96.0,
  "glucose_lag3": 94.0,
  "glucose_delta_1": 2.0,
  "glucose_delta_2": 4.0,
  "glucose_ma3_past": 96.0,
  "glucose_std3_past": 2.0
}
```
