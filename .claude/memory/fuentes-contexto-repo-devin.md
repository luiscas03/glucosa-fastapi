---
name: fuentes-contexto-repo-devin
description: Repo canónico NepturaTech/glucosa-fastapi y base de conocimiento Devin/DeepWiki como fuente secundaria, con sus dos límites conocidos
metadata:
  type: reference
---

Repositorio canónico: **https://github.com/NepturaTech/glucosa-fastapi** (coincide con el `origin` del working copy). La URL `luiscas03/glucosa-fastapi` que aparece en `README.md` y `deploy_contabo/README.md` está obsoleta.

Fuente secundaria de contexto: **base de conocimiento de Devin (DeepWiki)**, vía MCP `mcp__devin__read_wiki_structure` / `read_wiki_contents` / `ask_question` con `repoName: "NepturaTech/glucosa-fastapi"`. Cubre overview, `main.py`, cada microservicio, despliegue Contabo y Azure, artefactos, los informes de validación y un glosario. Sirve para orientarse rápido y para el histórico de decisiones sin leer medio repo.

**Why:** el usuario la señaló como segunda fuente para trabajar en varios frentes, pero medí sus dos límites el 2026-07-23 y ambos pueden inducir a error si se toma como autoridad:

1. Está indexada sobre el **último commit pusheado**, no sobre el working tree — describía el `m19_postprandial` antiguo, sin la recalibración isotónica ni `GET /schema` que ya estaban en disco sin commitear.
2. **Hereda los errores de los `.md` del repo**: afirma que los artefactos de `assets/models/monitor/` requieren sklearn 1.4.2 porque lo dice `VALIDACION_MODELOS_2026-06-04.md`, cuando la carga real demuestra 1.3.2 (ver [[conflicto-versiones-artefactos]]).

**How to apply:** usar Devin para navegar e historia; para versiones, features, shapes o cualquier hecho de runtime, verificar ejecutando (`joblib.load`, arrancar el servicio) antes de escribirlo como cierto. Ante discrepancia entre Devin y el archivo local, manda el archivo local.
