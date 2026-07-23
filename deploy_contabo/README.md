# BIOMON en Contabo — despliegue de los 5 modelos

Un solo `docker compose` con los 5 modelos + Caddy (HTTPS automático). Una ruta por
modelo bajo `https://$DOMAIN`. Las keys van en `.env` (no se hornean en la imagen).

## Modelos y rutas

| Ruta | Modelo | code interno | Salida |
|---|---|---|---|
| `/m19/predict` | Riesgo posprandial (M19) — primario | `m19_postprandial` | `prob_pico_alto` + banda |
| `/m2/predict`  | Glucosa neural PPG (M2) | `cnn_lstm_16p_clean_v1` | `prediction` mg/dL |
| `/m5/predict`  | Riesgo posprandial C0 (M5) | `postprandial_cluster_0` | `risk_calibrated` |
| `/m6/predict`  | Riesgo posprandial C1 (M6) — el sano | `postprandial_cluster_1` | `risk` |
| `/m3/predict`  | Glucosa tabular Regrix (M3) | `glucose_regression_v2` | `prediction` mg/dL |

Auth: header `X-API-Key: <key del modelo>` en todos. Health sin auth:
M19 en `/m19/`, el resto en `/m2/health`, `/m5/health`, etc.

## Primer despliegue (VPS Ubuntu/Debian limpio) — vía Cloudflare Tunnel

No se abren puertos en Contabo. cloudflared sale hacia Cloudflare y Cloudflare pone el TLS.

```sh
# 1. Docker + plugin compose
curl -fsSL https://get.docker.com | sh

# 2. Crear el túnel en Cloudflare (una vez):
#    Zero Trust > Networks > Tunnels > Create tunnel (Cloudflared)
#    - copia el TUNNEL_TOKEN
#    - Public hostname:  biomon.delfos.lat  ->  Service: HTTP  ->  caddy:80
#    (el dominio debe estar en tu cuenta Cloudflare; el DNS lo crea el túnel solo)

# 3. Repo + secretos
git clone https://github.com/NepturaTech/glucosa-fastapi.git
cd glucosa-fastapi/deploy_contabo
cp .env.example .env
nano .env        # pon DOMAIN, TUNNEL_TOKEN y las 5 keys

# 4. Levantar
docker compose up -d --build
docker compose ps
docker compose logs -f cloudflared   # debe decir "Registered tunnel connection"
```

`DOMAIN` solo lo usa Delfos para construir la URL; el ruteo real lo define el
"Public hostname" del túnel (-> `caddy:80`).

## Actualizar

```sh
cd glucosa-fastapi/deploy_contabo
./update.sh        # git pull + rebuild de lo que cambió + limpia imágenes viejas
```

Cambiar una key: edita `.env` y `docker compose up -d` (sin `--build`).

## Verificar

```sh
curl https://$DOMAIN/m19/                      # health M19
curl https://$DOMAIN/m2/health                 # health M2
curl -X POST https://$DOMAIN/m19/predict \
  -H "X-API-Key: $API_KEY_M19" -H "Content-Type: application/json" \
  -d '{"carbs":60,"protein":15,"fat":10,"fiber":5,"sugar":20,"kcal":400,"gender":"F","hba1c":5.6,"hour":13}'
```

## Notas

- M3 (Regrix) fija `scikit-learn==1.4.2` — el `.pkl` no carga con otra versión.
- M2 trae TensorFlow: su imagen es grande (~2 GB) y el primer build tarda.
- Recursos: con 5 contenedores + Caddy, un VPS Contabo de 4 vCPU / 8 GB va holgado.
