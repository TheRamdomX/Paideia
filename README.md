# Paideia

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


