---
name: conflicto-versiones-artefactos
description: Los artefactos del repo son de 3 generaciones de sklearn incompatibles entre sí; ningún entorno único los carga todos (hallazgo verificado 2026-07-23)
metadata:
  type: project
---

Verificado empíricamente el 2026-07-23 cargando cada artefacto bajo sklearn 1.2.2 / 1.3.2 / 1.4.2 / 1.5.2 / 1.6.1 / 1.7.0 / 1.9.0. **Los artefactos serializados pertenecen a tres generaciones mutuamente incompatibles y no existe una sola combinación que los cargue todos:**

- `assets/models/monitor/*` → serializados con **sklearn 1.3.2 + numpy<2**. Bajo 1.6.1, `Gradient_Boosting` no carga (`sklearn.ensemble._gb_losses` se eliminó en 1.4) y `Random_Forest` carga pero **falla al predecir** (`DecisionTreeRegressor` sin `monotonic_cst`, atributo añadido en 1.4).
- `assets/models/root/*` → necesitan **sklearn 1.5–1.6 + numpy≥2** (usan `_RemainderColsList`, ausente en 1.2/1.3/1.4 y también en 1.7+; y el pickle de `MT19937` exige numpy 2).
- `models/*` (postprandiales, regrix) → **sklearn 1.4.2 + numpy<2**, como ya fija su `requirements.txt`.

Consecuencia medida en el gateway: con sklearn 1.6.1 el ensemble monitor entrega **5 de 7** modelos (`random_forest` y `gradient_boosting` salen como `null` en `predicciones_individuales`) y la media se calcula solo con los 5 que sobreviven — 146.93 vs 147.43 con los 7. **El fallo es silencioso**: `_predict_monitor()` captura la excepción por modelo y sigue.

Esto reproduce lo que pasa en producción: el `Dockerfile` de la raíz usa `python:3.9-slim` con `requirements.txt` sin pinear, y pip resuelve a sklearn 1.6.1 (la última que soporta 3.9). **Producción está sirviendo predicciones con 5/7 modelos sin avisar.**

**Decisión pendiente del usuario** (no aplicada): el gateway no arranca bajo sklearn 1.3.2 porque `load_artifacts()` hace `raise` si no puede cargar el pipeline de `MODEL_PATH` — pipeline que además **no participa en ninguna predicción**, solo se refleja en `GET /health`. Hacer opcional esa carga permitiría correr con 1.3.2 y recuperar los 7/7. Es un cambio de semántica fail-fast en producción, así que requiere aprobación explícita antes de tocarlo.

Ver [[entorno-local-python]] y el matriz de venvs documentada en `CLAUDE.md`.
