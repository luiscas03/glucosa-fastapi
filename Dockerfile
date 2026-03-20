# 1. Usamos una imagen oficial de Python ligera (puedes cambiar 3.9 por tu versión)
FROM python:3.9-slim

# 2. Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos el archivo de dependencias primero (para aprovechar el caché de Docker)
COPY requirements.txt .

# 4. Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos el resto de tu código (tu modelo .joblib, tus scripts, etc.)
COPY . .

# 6. Exponemos el puerto que usará FastAPI
EXPOSE 8000

# 7. Comando para encender el servidor FastAPI
# NOTA: Cambia "main:app" si tu archivo principal de python tiene otro nombre 
# (ej. si se llama api.py, pon "api:app")
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]