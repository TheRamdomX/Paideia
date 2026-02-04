# 🎓 Paideia

**Sistema RAG Educativo con Grafos de Conocimiento**

Paideia es un asistente educativo conversacional que combina Retrieval-Augmented Generation (RAG) con grafos de conocimiento para ofrecer respuestas personalizadas y pedagógicamente adaptadas al nivel de cada estudiante.

---

## ✨ Características Principales

### 🧠 RAG Híbrido Inteligente
- **Búsqueda Semántica**: Embeddings vectoriales para encontrar contenido conceptualmente similar
- **Búsqueda Literal (BM25)**: Para términos técnicos y definiciones exactas
- **Navegación de Grafos**: Explora relaciones entre conceptos para contexto enriquecido
- **Ranking Híbrido**: Combina las tres fuentes con pesos adaptativos

### 📊 Grafos de Conocimiento
- **Conceptos Interconectados**: Relaciones jerárquicas (es-un, parte-de) y asociativas
- **Evidencia Vinculada**: Cada concepto conectado a chunks de contenido fuente
- **Traversal Inteligente**: Expansión de vecindad con control de profundidad

### 🎯 Personalización por Estudiante
- **Perfiles Adaptativos**: Nivel de dominio por concepto
- **Estilos de Aprendizaje**: Visual, textual, ejemplos, analogías
- **Detección de Debilidades**: Identifica vacíos de conocimiento
- **Respuestas Adaptadas**: Ajusta complejidad y estilo según el perfil

### 🔄 Sistema de Agentes
- **Retrieval Agent**: Decide estrategia óptima de búsqueda
- **Reasoning Agent**: Genera respuestas pedagógicas
- **Reflection Agent**: Evalúa calidad y decide si reintentar

### 💾 Memoria y Cache
- **Memoria de Sesión**: Contexto conversacional con ventana deslizante
- **Cache Inteligente**: Respuestas validadas para consultas frecuentes
- **Invalidación Automática**: Por tiempo o actualización de contenido

### 📈 Sistema de Feedback
- **Feedback Explícito**: Útil/no útil, rating, comentarios
- **Señales Implícitas**: Tiempo de lectura, scroll, clicks
- **Mejora Continua**: Actualiza pesos del grafo según feedback

---

## 🚀 Inicio Rápido

### Requisitos
- Python 3.12+
- Docker y Docker Compose
- API Keys (OpenAI, Google AI, o Ollama local)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/TheRamdomX/Paideia.git
cd Paideia

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

### Configuración (.env)

```env
# Base de Datos
SURREAL_URL=ws://localhost:8000/rpc
SURREAL_USER=root
SURREAL_PASS=root
SURREAL_NS=paideia
SURREAL_DB=education

# LLM (elegir uno)
LLM_PROVIDER=openai          # openai, google, ollama
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...

# Embeddings
EMBEDDING_PROVIDER=openai    # openai, google, local
EMBEDDING_MODEL=text-embedding-3-small

# RAG
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=10
MIN_SCORE_THRESHOLD=0.3
```

### Iniciar Servicios

#### Opción 1: Docker Compose

```bash
# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys (OPENAI_API_KEY o GOOGLE_API_KEY)

# Levantar todo el sistema (SurrealDB + Backend)
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

Una vez iniciado:
- **API Backend**: http://localhost:8080
- **SurrealDB**: http://localhost:8000
- **Documentación API**: http://localhost:8080/docs

#### Opción 2: Desarrollo Local

```bash
# Terminal 1: Iniciar solo SurrealDB
docker-compose up -d surrealdb

# Terminal 2: Ejecutar servidor en modo desarrollo
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

---

## 📖 Uso de la API

### Base URL
```
http://localhost:8080/api/v1
```

### 📥 Ingestión de Contenido

#### Subir Archivo
```bash
curl -X POST "http://localhost:8080/api/v1/ingest/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf" \
  -F "metadata={\"subject\": \"matemáticas\", \"level\": \"secundaria\"}"
```

#### Ingestar URL
```bash
curl -X POST "http://localhost:8080/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://ejemplo.com/articulo-educativo",
    "extract_links": true,
    "max_depth": 1
  }'
```

#### Ingestar Texto Directo
```bash
curl -X POST "http://localhost:8080/api/v1/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "La fotosíntesis es el proceso por el cual...",
    "title": "Fotosíntesis",
    "metadata": {"subject": "biología"}
  }'
```

### 💬 Consultas

#### Consulta Simple
```bash
curl -X POST "http://localhost:8080/api/v1/query/" \
  -H "Content-Type: application/json" \
  -H "X-Student-ID: estudiante123" \
  -d '{
    "question": "¿Qué es la fotosíntesis?",
    "use_cache": true
  }'
```

#### Respuesta
```json
{
  "query_id": "q_abc123",
  "answer": "La fotosíntesis es el proceso mediante el cual las plantas...",
  "sources": [
    {
      "content": "...",
      "score": 0.92,
      "concepts": ["fotosíntesis", "cloroplasto"]
    }
  ],
  "confidence": 0.89,
  "learning_mode": "concept",
  "followup_suggestions": [
    "¿Cuáles son las fases de la fotosíntesis?",
    "¿Qué papel juega la clorofila?"
  ]
}
```

#### Modos pedagógicos (LearningMode)

Podés forzar el comportamiento del asistente usando `learning_mode`.

- **Automático** (recomendado): omití `learning_mode` y el backend lo detecta.
- **`concept`**: explicaciones teóricas/definiciones.
- **`practice`**: resolución paso a paso.
- **`exercise_list`**: lista ejercicios (sin resolverlos).

#### Consulta con Streaming
```bash
curl -X POST "http://localhost:8080/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explícame el ciclo de Krebs",
    "stream": true,
    "learning_mode": "practice"
  }'
```

#### Consulta forzando modo (sin streaming)
```bash
curl -X POST "http://localhost:8080/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Lista ejercicios de derivadas",
    "learning_mode": "exercise_list"
  }'
```

#### Consulta Debug (ver contexto intermedio)
```bash
curl -X POST "http://localhost:8080/api/v1/query/debug" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es la mitosis?",
    "include_retrieval_details": true
  }'
```

### 📊 Feedback

#### Feedback Explícito
```bash
curl -X POST "http://localhost:8080/api/v1/feedback/explicit" \
  -H "Content-Type: application/json" \
  -H "X-Student-ID: estudiante123" \
  -d '{
    "query_id": "q_abc123",
    "feedback_type": "helpful",
    "rating": 5,
    "comment": "Muy clara la explicación"
  }'
```

#### Feedback Implícito (métricas de comportamiento)
```bash
curl -X POST "http://localhost:8080/api/v1/feedback/implicit" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "q_abc123",
    "signal_type": "dwell_time",
    "value": 45.5,
    "metadata": {"scroll_depth": 0.8}
  }'
```

#### Resumen de Feedback del Estudiante
```bash
curl "http://localhost:8080/api/v1/feedback/summary/student/estudiante123?days=30"
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI                               │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐                     │
│  │ /query  │  │ /ingest │  │ /feedback│                     │
│  └────┬────┘  └────┬────┘  └────┬─────┘                     │
└───────┼────────────┼────────────┼───────────────────────────┘
        │            │            │
        ▼            ▼            ▼
┌───────────────────────────────────────────────────────────┐
│                    Capa de Agentes                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Retrieval  │  │ Reasoning  │  │ Reflection │           │
│  │   Agent    │──│   Agent    │──│   Agent    │           │
│  └─────┬──────┘  └─────┬──────┘  └────────────┘           │
└────────┼───────────────┼──────────────────────────────────┘
         │               │
         ▼               ▼
┌────────────────┐  ┌────────────────┐
│  GraphRAG      │  │    LLM         │
│  ┌──────────┐  │  │  ┌──────────┐  │
│  │ Vector   │  │  │  │ OpenAI   │  │
│  │ Search   │  │  │  │ Google   │  │
│  ├──────────┤  │  │  │ Ollama   │  │
│  │ BM25     │  │  │  └──────────┘  │
│  ├──────────┤  │  └────────────────┘
│  │ Graph    │  │
│  │ Traversal│  │
│  └──────────┘  │
└───────┬────────┘
        │
        ▼
┌────────────────────────────────────────┐
│              SurrealDB                  │
│  ┌──────────┐  ┌──────────┐            │
│  │ Nodos    │  │ Edges    │            │
│  │ Concept  │──│ RELATES  │            │
│  │ Chunk    │  │ EVIDENCES│            │
│  │ Source   │  │ CONTAINS │            │
│  └──────────┘  └──────────┘            │
└────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
frontend/
├── index.html              # UI (chat + selector de modo)
├── app.js                  # Lógica del cliente (envía learning_mode opcional)
├── styles.css              # Estilos
├── nginx.conf              # Proxy /api -> backend
└── Dockerfile              # Imagen del frontend

backend/
├── main.py                 # Aplicación FastAPI
├── settings.py             # Configuración
├── deps.py                 # Inyección de dependencias
│
├── api/                    # Endpoints REST
│   ├── query.py            # Consultas Q&A
│   ├── ingest.py           # Ingesta de contenido
│   └── feedback.py         # Sistema de feedback
│
├── agents/                 # Agentes inteligentes
│   ├── mode_router.py      # Eleccion de Modo
│   ├── retrieval_agent.py  # Decisión de estrategia
│   ├── reasoning_agent.py  # Generación de respuestas
│   └── reflection_agent.py # Evaluación de calidad
│
├── graphs/                 # Grafos LangGraph
│   ├── source_graph.py     # Pipeline de ingesta
│   ├── transform_graph.py  # Enriquecimiento
│   └── retrieval_graph.py  # Búsqueda híbrida
│
├── ingestion/              # Procesamiento de contenido
│   ├── content_processor.py
│   ├── chunking.py
│   └── vectorizer.py
│
├── graph/                  # Operaciones de grafo
│   ├── schema.py           # Definición de nodos/edges
│   ├── builders.py         # Constructores
│   └── traversal.py        # Navegación
│
├── retrieval/              # Sistemas de búsqueda
│   ├── vector.py           # Búsqueda semántica
│   ├── bm25.py             # Búsqueda literal
│   └── hybrid_ranker.py    # Combinación de scores
│
├── memory/                 # Memoria y estado
│   ├── student_profile.py  # Perfiles de estudiante
│   ├── session_memory.py   # Contexto de sesión
│   └── cache.py            # Cache de respuestas
│
├── feedback/               # Sistema de feedback
│   ├── signals.py          # Procesamiento de señales
│   ├── analytics.py        # Métricas y análisis
│   └── graph_updates.py    # Actualización del grafo
│
├── models/                 # Integraciones de modelos
│   ├── llm.py              # OpenAI, Google, Ollama
│   ├── embeddings.py       # Vectorización
│   └── stt.py              # Speech-to-text
│
├── db/                     # Base de datos
│   └── surreal.py          # Cliente SurrealDB
│
└── utils/                  # Utilidades
    └── text.py             # Procesamiento de texto
```

---

## 🖥️ Frontend

El frontend es una SPA simple (HTML/CSS/JS) servida por Nginx (en Docker). Usa `API_BASE_URL = /api/v1` y el proxy de Nginx redirige `/api` al backend.

### Selector de modo pedagógico

En la parte inferior del chat hay un selector con 4 opciones:

- **Auto**: no envía `learning_mode` (el backend detecta el modo).
- **Concepto**: envía `learning_mode: "concept"`
- **Práctica**: envía `learning_mode: "practice"`
- **Ejercicios**: envía `learning_mode: "exercise_list"`

La selección se guarda en `localStorage` como `paideia_learning_mode`.

---

## 🔧 Configuración Avanzada

### Ajustar Chunking
```python
# settings.py
CHUNK_SIZE = 512          # Tamaño de chunk en tokens
CHUNK_OVERLAP = 64        # Solapamiento entre chunks
MIN_CHUNK_SIZE = 100      # Mínimo para no descartar
```

### Configurar Retrieval
```python
# Pesos del ranking híbrido
VECTOR_WEIGHT = 0.4       # Búsqueda semántica
BM25_WEIGHT = 0.3         # Búsqueda literal
GRAPH_WEIGHT = 0.3        # Expansión de grafo

TOP_K = 10                # Resultados a retornar
MIN_SCORE = 0.3           # Umbral mínimo
```

### Modos pedagógicos (LearningMode) - Configuración

El backend soporta modos pedagógicos. Podés dejarlo en automático u forzar uno en la request con `learning_mode`:

- **Automático**: omití `learning_mode` y el backend lo detecta.
- `concept`: explicaciones teóricas/definiciones.
- `practice`: resolución paso a paso.
- `exercise_list`: lista ejercicios disponibles (sin resolverlos).

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

<p align="center">
  <b>Paideia</b> - Educación personalizada potenciada por IA 🎓
</p>


