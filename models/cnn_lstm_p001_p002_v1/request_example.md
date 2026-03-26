# CNN LSTM P001 P002 V1

## Servicio

- App local: `main:app`
- Endpoint: `POST /predict`
- Tipo: modelo secuencial Keras
- Estado: referencia / local, no API productiva principal

## JSON de ejemplo

```json
{
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "shape": [6, 1]
}
```

## Header

- `Content-Type: application/json`
- `X-API-Key: <tu_clave>` si `API_KEY` existe en `.env`

## Nota

- Si no sabes el `shape`, consulta `GET /health` y prueba con la longitud que espera el modelo.
- El body usa `values` y `shape`, no `features`.
