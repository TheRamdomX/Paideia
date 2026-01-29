# Paideia

Este repositorio contiene los componentes y la orquestación propuesta para un asistente educativo conversacional multimodal.

## Arquitectura propuesta

```text

┌──────────────────────────────┐
│          Estudiante          │
│        (voz / texto)         │
└──────────────┬───────────────┘
               │ audio stream
               ▼
┌──────────────────────────────┐
│ Native Audio Dialog          │
│ ──────────────────────────── │
│ • ASR + NLU                  │
│ • Gestión de turnos          │
│ • Clasificación de intención │
│ • Router simple              │
└──────────────┬───────────────┘
               │ intención / task
               ▼
┌──────────────────────────────┐
│        Orquestador           │
│      (control explícito)     │
│ ──────────────────────────── │
│ • DAG de agentes             │
│ • Límites de loops           │
│ • Cache check                │
│ • Estado del flujo           │
└───────┬───────────┬──────────┘
        │           │
   cache hit?       │
        │           │
        ▼           ▼
┌─────────────┐   ┌──────────────────────────┐
│ Cache       │   │ Retrieval Agent          │
│ (Redis)     │   │ ──────────────────────── │
│ respuestas  │   │ • Navegación del grafo   │
│ validadas   │   │ • Expansión semántica    │
└─────┬───────┘   │ • Contexto curado        │
      │           └──────────┬───────────────┘
      │                      │
      │                      ▼
      │           ┌──────────────────────────┐
      │           │ GraphRAG Layer           │
      │           │ ──────────────────────── │
      │           │ • Grafo educativo        │
      │           │   (Neo4j)                │
      │           │ • Vector DB (FAISS)      │
      │           │ • Subgrafo relevante     │
      │           └──────────┬───────────────┘
      │                      │ contexto
      │                      ▼
      │           ┌──────────────────────────┐
      │           │ Reasoning Agent          │
      │           │ ──────────────────────── │
      │           │ • Razonamiento           │
      │           │ • Explicación educativa  │
      │           │ • Síntesis               │
      │           └──────────┬───────────────┘
      │                      │ respuesta draft
      │                      ▼
      │           ┌──────────────────────────┐
      │           │ Pedagogical Critic Agent │
      │           │ ──────────────────────── │
      │           │ • Corrección conceptual  │
      │           │ • Nivel adecuado         │
      │           │ • Criterios curriculares │
      │           └──────────┬───────────────┘
      │                      │
      │              ¿aprobado?
      │                      │
      │            ┌─────────┴─────────┐
      │            │                   │
      │           sí                  no
      │            │                   │
      ▼            ▼                   │
┌──────────────────────────┐           │
│ Actualizar Cache         │◄──────────┘
│ + Memoria Estudiante     │   (1 loop máx)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Gemini Audio                 │
│ • Verbalización              │
│ • Continuidad conversacional │
└──────────────┬───────────────┘
               audio
               ▼
┌──────────────────────────────┐
│          Estudiante          │
└──────────────────────────────┘

```

```text
┌──────────────────────────────┐
│ Memoria del Estudiante       │
│ ──────────────────────────── │
│ • Dominio por concepto       │
│ • Errores recurrentes        │
│ • Progreso por unidad        │
│ • Preferencias pedagógicas   │
└──────────────────────────────┘


```

## Estructura del proyecto

```text
project/
├── docker-compose.yml
├── README.md
├── .env.example
│
├── surreal-data/
│   └── surreal.db
│
├── backend/
│   ├── main.py
│   ├── settings.py
│   ├── deps.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── query.py
│   │   └── feedback.py
│   │
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── source_graph.py
│   │   ├── transform_graph.py
│   │   └── retrieval_graph.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── retrieval_agent.py
│   │   ├── reasoning_agent.py
│   │   └── reflection_agent.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── content_processor.py
│   │   ├── chunking.py
│   │   └── vectorizer.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── builders.py
│   │   └── traversal.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── vector.py
│   │   └── hybrid_ranker.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── student_profile.py
│   │   ├── session_memory.py
│   │   └── cache.py
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── signals.py
│   │   ├── analytics.py
│   │   └── graph_updates.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   └── stt.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── surreal.py
│   │   └── migrations/
│   │       ├── 001_sources.surql
│   │       ├── 002_embeddings.surql
│   │       ├── 003_graph.surql
│   │       └── 004_search_functions.surql
│   │
│   └── utils/
│       ├── __init__.py
│       ├── ids.py
│       ├── logging.py
│       └── text.py
```


