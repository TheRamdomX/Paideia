# Migración de base de datos SurrealDB 

Este proyecto usa SurrealDB como base de datos principal, la persistencia se realiza mediante almacenamiento por archivo, lo que permite mover la base de datos completa entre máquinas sin reindexar ni reingestar datos.

## Estructura del proyecto

```
project/
├── docker-compose.yml
└── surreal-data/
    └── surreal.db
```

El archivo `surreal.db` contiene:

- Esquema completo
- Datos
- Relaciones del grafo
- Embeddings vectoriales
- Identificadores internos

Mover este archivo equivale a mover la base de datos completa.

## Configuración usada (Docker Compose)

El servicio se ejecuta en modo archivo con bind mount al filesystem del host:

```yaml
version: "3.9"

services:
  surrealdb:
    image: surrealdb/surrealdb:latest
    container_name: surrealdb
    command: start --log info --user root --pass root file:/data/surreal.db
    ports:
      - "8000:8000"
    volumes:
      - ./surreal-data:/data
    restart: unless-stopped
```

Este enfoque evita volúmenes Docker opacos y facilita la portabilidad entre dispositivos.

## Procedimiento de migración (máquina → máquina)

En la máquina origen:

1. Detener SurrealDB:

```bash
docker compose down
```

2. Copiar el directorio de datos al destino (ejemplo con rsync):

```bash
rsync -av surreal-data/ usuario@maquina-destino:/ruta/proyecto/surreal-data/
```

También puede copiarse por USB, SCP o cualquier medio equivalente. No copiar el archivo mientras el contenedor está corriendo.

En la máquina destino:

1. Copiar el mismo `docker-compose.yml` al proyecto.
2. Verificar que el archivo existe:

```bash
ls -l surreal-data/surreal.db
```

3. Levantar el servicio:

```bash
docker compose up -d
```

La base de datos quedará exactamente en el mismo estado que en la máquina original.

## Backups simples

Un backup completo puede realizarse copiando el archivo:

```bash
cp surreal-data/surreal.db surreal-data/surreal.db.bak
```

Esto genera una copia íntegra del estado de la base.