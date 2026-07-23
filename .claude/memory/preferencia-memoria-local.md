---
name: preferencia-memoria-local
description: La memoria de Claude para este proyecto debe vivir solo en .claude/ dentro del repo, no en la global del usuario
metadata:
  type: feedback
---

La memoria de Claude para este proyecto vive **únicamente** en `.claude/memory/` dentro del repositorio, con su índice en `.claude/MEMORY.md`. No escribir memorias de este proyecto en el directorio global `~/.claude/projects/*/memory/`.

**Why:** el usuario quiere portabilidad — que el contexto viaje con el repo (git clone, otra máquina, otro colaborador) sin depender de la instalación local de Claude Code. `.claude/` no está en `.gitignore` a propósito.

**How to apply:** al guardar una memoria nueva, escribir el archivo en `.claude/memory/<slug>.md` y añadir la línea de índice en `.claude/MEMORY.md`. Si el harness ofrece la ruta global, ignorarla. Ver [[entorno-local-python]] y [[conflicto-versiones-artefactos]].
