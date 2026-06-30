#!/bin/sh
# Actualizar en Contabo: trae cambios y reconstruye sólo lo que cambió.
set -e
cd "$(dirname "$0")"
git pull
docker compose up -d --build
docker image prune -f
docker compose ps
