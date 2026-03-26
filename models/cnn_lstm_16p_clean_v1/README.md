# CNN LSTM 16P Clean V1

## Estado

- Modelo Keras archivado como referencia o posible uso local
- No esta marcado hoy como despliegue API productivo
- Requiere `tensorflow`

## Run Local

Desde esta carpeta:

```bash
uvicorn main:app --host 0.0.0.0 --port 8005
```

## Routes

- `GET /health`
- `POST /predict`

## Headers

- `Content-Type: application/json`
- `X-API-Key: <valor>` si `API_KEY` existe en `models/.env`

## Notes

- Usa `glucose_cnn_lstm_16p_CLEAN_v1.keras`
- El body usa `values` y `shape`
- Si omites `shape`, el servicio intenta inferirlo desde `input_shape`
- Si falta TensorFlow o el `shape` no coincide, respondera con error

## JSON Para `/predict`

```json
{
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "shape": [6, 1]
}
```
