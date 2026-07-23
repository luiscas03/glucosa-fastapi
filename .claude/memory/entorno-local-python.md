---
name: entorno-local-python
description: En esta máquina solo hay pyenv 3.14 y system 3.12; los venvs del proyecto deben crearse con /usr/bin/python3.12 usando uv
metadata:
  type: project
---

En la máquina de desarrollo el `python3` por defecto es **pyenv 3.14.6**, que no sirve para este proyecto: las versiones de scikit-learn que necesitan los artefactos (1.3.2 / 1.4.2 / 1.6.1) no tienen wheels para 3.14. El único intérprete utilizable es el del sistema, **`/usr/bin/python3.12`**.

`uv` está instalado en `~/.local/bin/uv` y es la vía rápida para crear/poblar los venvs (`uv venv --python /usr/bin/python3.12 <dir>`, `uv pip install --python <dir>/bin/python ...`). También puede descargar intérpretes que no estén en la máquina (`uv venv --python 3.11`), útil para probar combinaciones de versiones sin ensuciar los venvs del proyecto.

**Why:** crear el venv con el `python3` por defecto falla al instalar o al deserializar, y el error aparece tarde (en `joblib.load`), no al instalar.

**How to apply:** nunca `python -m venv` con el python del PATH; siempre fijar `/usr/bin/python3.12`. Ver [[conflicto-versiones-artefactos]] para qué venv usar en cada caso.
