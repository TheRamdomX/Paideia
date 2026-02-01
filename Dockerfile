# Dockerfile para Paideia Backend
FROM python:3.12-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY backend/ /app/backend/

# Puerto de la aplicación
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
