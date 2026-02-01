# Estructura del Proyecto

``` text
project/
├── docker-compose.yml
├── .env
├── README.md
│
├── backend/
│ ├── main.py
│ ├── settings.py
│ ├── deps.py
│ │
│ ├── api/
│ │ ├── ingest.py
│ │ ├── query.py
│ │ ├── feedback.py
│ │
│ ├── agents/
│ │ ├── retrieval_agent.py
│ │ ├── reasoning_agent.py
│ │ └── reflection_agent.py
│ │
│ ├── graphs/
│ │ ├── source_graph.py
│ │ ├── transform_graph.py
│ │ └── retrieval_graph.py
│ │
│ ├── ingestion/
│ │ ├── content_processor.py
│ │ ├── chunking.py
│ │ └── vectorizer.py
│ │
│ ├── graph/
│ │ ├── schema.py
│ │ ├── builders.py
│ │ └── traversal.py
│ │
│ ├── retrieval/
│ │ ├── bm25.py
│ │ ├── vector.py
│ │ └── hybrid_ranker.py
│ │
│ ├── memory/
│ │ ├── student_profile.py
│ │ ├── session_memory.py
│ │ └── cache.py
│ │
│ ├── feedback/
│ │ ├── signals.py
│ │ ├── analytics.py
│ │ └── graph_updates.py
│ │
│ ├── models/
│ │ ├── llm.py
│ │ ├── embeddings.py
│ │ └── stt.py
│ │
│ ├── db/
│ │ └── surreal.py
│ │
│ └── utils/
│ └── text.py
│
└── surreal-data/
  └── surreal.db
```

# Backend

## backend/main.py

### create_app
Qué hace: construye la aplicación FastAPI, registra middlewares y routers.
Usa: settings.load_settings, deps.get_db, routers de api/*.

### startup
Qué hace: inicializa DB, modelos, colas y cache.
Usa: db.surreal.connect, models.llm.select_model, models.embeddings.batch_embed (warm-up opcional).

### shutdown
Qué hace: cierre limpio de conexiones y workers.
Usa: db.surreal.execute (flush), cierre de pools.

## backend/settings.py

### load_settings
Qué hace: carga .env, valida variables críticas.
Usa: utilidades internas de parsing.

### get_model_config
Qué hace: define routing de modelos (Gemma / Gemini / GPT).
Usa: variables cargadas por load_settings.

### get_rag_config
Qué hace: devuelve chunk sizes, thresholds, flags GraphRAG.
Usa: valores normalizados por load_settings.

## backend/deps.py

### get_db
Qué hace: entrega conexión viva a SurrealDB.
Usa: db.surreal.connect.

### get_llm
Qué hace: devuelve wrapper LLM listo para usar.
Usa: models.llm.select_model.

### get_embedding_model
Qué hace: devuelve proveedor de embeddings.
Usa: models.embeddings.embed_text.

### get_agents
Qué hace: construye instancias de agentes.
Usa: retrieval_agent, reasoning_agent, reflection_agent.

# API

## api/ingest.py

### ingest_file
Qué hace: recibe archivo y dispara pipeline de ingestión.
Usa: graphs.source_graph.run_source_graph.

### ingest_url
Qué hace: procesa URLs web.
Usa: ingestion.content_processor.process_content.

### ingest_media
Qué hace: audio/video → texto → ingestión.
Usa: models.stt.transcribe, graphs.source_graph.run_source_graph.

### get_ingest_status
Qué hace: consulta estado del job.
Usa: db.surreal.execute.

## api/query.py

### query_student
Qué hace: flujo completo pregunta → respuesta.
Usa: retrieval_agent.retrieve, reasoning_agent.generate_answer, memory.session_memory.store_turn.

### query_debug
Qué hace: devuelve contexto intermedio.
Usa: retrieval_graph.merge_results.

### query_cached
Qué hace: responde desde cache si existe.
Usa: memory.cache.get_cached_answer.

## api/feedback.py

### submit_feedback
Qué hace: registra feedback explícito.
Usa: feedback.signals.parse_feedback, feedback.analytics.aggregate_metrics.

### implicit_feedback
Qué hace: registra señales implícitas.
Usa: feedback.signals.weight_feedback.

### feedback_summary
Qué hace: métricas por usuario.
Usa: feedback.analytics.aggregate_metrics.

# Graphs

## graphs/source_graph.py

### content_process
Qué hace: extracción y normalización del contenido.
Usa: ingestion.content_processor.process_content.

### save_source
Qué hace: persiste fuente base.
Usa: db.surreal.execute.

### trigger_transformations
Qué hace: decide qué transformaciones ejecutar.
Usa: graphs.transform_graph.run_transform_graph.

### run_source_graph
Qué hace: orquesta ingestión completa.
Usa: todas las anteriores + ingestion.vectorizer.submit_vectorization.

## graphs/transform_graph.py

### extract_concepts
Qué hace: obtiene conceptos candidatos.
Usa: models.llm.generate.

### resolve_entities
Qué hace: deduplica y normaliza conceptos.
Usa: graph.schema.validate_graph.

### generate_summary
Qué hace: crea resumen pedagógico.
Usa: models.llm.generate.

### extract_definitions
Qué hace: definiciones precisas.
Usa: models.llm.generate.

### run_transform_graph
Qué hace: ejecuta transformaciones en paralelo.
Usa: todas las anteriores + graph.builders.create_concept_node.

## graphs/retrieval_graph.py

### graph_retrieval
Qué hace: traversal del grafo educativo.
Usa: graph.traversal.expand_concepts.

### vector_retrieval
Qué hace: búsqueda semántica.
Usa: retrieval.vector.search_vectors.

### bm25_retrieval
Qué hace: búsqueda literal.
Usa: retrieval.bm25.search_text.

### merge_results
Qué hace: unifica resultados.
Usa: retrieval.hybrid_ranker.combine_scores.

# Agents
## agents/retrieval_agent.py

### decide_strategy
Qué hace: elige grafo/vector/BM25.
Usa: heurísticas + memory.student_profile.load_profile.

### retrieve
Qué hace: ejecuta retrieval seleccionado.
Usa: graphs.retrieval_graph.*.

### score_results
Qué hace: scoring híbrido.
Usa: retrieval.hybrid_ranker.normalize_scores.

## agents/reasoning_agent.py

### build_prompt
Qué hace: construye prompt con contexto curado.
Usa: utils.text.truncate_context.

### generate_answer
Qué hace: genera respuesta final.
Usa: models.llm.generate.

### adapt_to_student
Qué hace: ajusta nivel pedagógico.
Usa: memory.student_profile.load_profile.

## agents/reflection_agent.py

### evaluate_answer
Qué hace: evalúa cobertura y coherencia.
Usa: models.llm.generate.

### decide_retry
Qué hace: decide si reintentar.
Usa: evaluate_answer.

### request_more_context
Qué hace: fuerza nuevo retrieval.
Usa: retrieval_agent.retrieve.

# Ingestion

## ingestion/content_processor.py

### process_content
Qué hace: wrapper sobre content-core.
Usa: librería externa.

### detect_content_type
Qué hace: identifica tipo de input.
Usa: heurísticas simples.

### cleanup_text
Qué hace: limpieza básica.
Usa: utils.text.clean_markdown.

## ingestion/chunking.py

### hierarchical_chunk
Qué hace: parent/child chunking.
Usa: utils.text.token_count.

### link_chunks
Qué hace: enlaza jerarquía.
Usa: db.surreal.execute.

### validate_chunks
Qué hace: control de calidad.
Usa: reglas internas.

## ingestion/vectorizer.py

### submit_vectorization
Qué hace: lanza job asíncrono.
Usa: embed_chunk.

### embed_chunk
Qué hace: genera embedding individual.
Usa: models.embeddings.embed_text.

### retry_failed_chunks
Qué hace: reprocesa fallos.
Usa: embed_chunk.

# Graph

## graph/schema.py

### define_nodes
Qué hace: define tipos de nodos.
Usa: constantes internas.

### define_edges
Qué hace: define relaciones válidas.
Usa: constantes internas.

### validate_graph
Qué hace: valida consistencia.
Usa: reglas de negocio.

## graph/builders.py

### create_concept_node
Qué hace: inserta concepto.
Usa: db.surreal.execute.

### link_concepts
Qué hace: crea relaciones.
Usa: graph.schema.define_edges.

### attach_evidence
Qué hace: enlaza chunk preciso.
Usa: db.surreal.execute.

## graph/traversal.py

### expand_concepts
Qué hace: explora vecindad.
Usa: db.surreal.execute.

### rank_paths
Qué hace: ordena caminos.
Usa: scoring interno.

### limit_depth
Qué hace: evita explosión combinatoria.
Usa: parámetros de config.

# Retrieval

## retrieval/bm25.py

### search_text
Qué hace: búsqueda literal.
Usa: función SurrealDB.

### highlight_matches
Qué hace: snippets.
Usa: resultados BM25.

## retrieval/vector.py

### search_vectors
Qué hace: cosine similarity.
Usa: models.embeddings.embed_text.

### filter_by_threshold
Qué hace: corte por score.
Usa: config.

## retrieval/hybrid_ranker.py

### normalize_scores
Qué hace: escala scores.
Usa: reglas internas.

### combine_scores
Qué hace: mezcla scores.
Usa: normalize_scores.

### select_top_k
Qué hace: selección final.
Usa: resultados combinados.

# Memory

## memory/student_profile.py

### load_profile
Qué hace: carga perfil.
Usa: db.surreal.execute.

### update_profile
Qué hace: ajusta nivel.
Usa: feedback.

### infer_weaknesses
Qué hace: detecta vacíos.
Usa: historial + feedback.

## memory/session_memory.py

### store_turn
Qué hace: guarda turno.
Usa: cache local o DB.

### get_recent_context
Qué hace: ventana corta.
Usa: store_turn.

### prune_context
Qué hace: controla tamaño.
Usa: config.

### memory/cache.py

### get_cached_answer
Qué hace: busca cache.
Usa: hashing.

### store_cache
Qué hace: guarda respuesta.
Usa: DB o Redis.

### invalidate_cache
Qué hace: limpieza selectiva.
Usa: reglas de caducidad.

# Feedback

## feedback/signals.py

### parse_feedback
Qué hace: normaliza feedback.
Usa: validaciones.

### weight_feedback
Qué hace: asigna impacto.
Usa: heurísticas.

### feedback/analytics.py

#### detect_poor_chunks
Qué hace: identifica chunks débiles.
Usa: feedback + métricas.

#### track_confusion
Qué hace: detecta conceptos confusos.
Usa: historial.

#### aggregate_metrics
Qué hace: métricas globales.
Usa: DB.

## feedback/graph_updates.py

### reinforce_edges
Qué hace: aumenta peso.
Usa: db.surreal.execute.

### weaken_edges
Qué hace: penaliza relaciones.
Usa: db.surreal.execute.

### schedule_revectorization
Qué hace: marca reprocesado.
Usa: ingestion.vectorizer.submit_vectorization.

# Models

## models/llm.py

### generate
Qué hace: llamada genérica LLM.
Usa: proveedor activo.

### select_model
Qué hace: routing de modelo.
Usa: settings.get_model_config.

### handle_fallback
Qué hace: failover.
Usa: generate.

## models/embeddings.py

### embed_text
Qué hace: texto → vector.
Usa: proveedor embeddings.

### batch_embed
Qué hace: embedding en lote.
Usa: embed_text.

## models/stt.py

### transcribe
Qué hace: audio → texto.
Usa: modelo STT.

### detect_language
Qué hace: detecta idioma.
Usa: STT metadata.

# db

## db/surreal.py

### connect
Qué hace: abre conexión.
Usa: credenciales.

### execute
Qué hace: ejecuta query.
Usa: conexión activa.

### transaction
Qué hace: manejo transaccional.
Usa: execute.

# Utils

## utils/text.py

### token_count
Qué hace: cuenta tokens.
Usa: tokenizer.

### clean_markdown
Qué hace: limpieza estructural.
Usa: regex.

### truncate_context
Qué hace: ajusta longitud.
Usa: token_count.