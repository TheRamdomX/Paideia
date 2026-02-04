# Estructura del Proyecto

``` text
project/
├── docker-compose.yml
├── .env
├── README.md
│
├── frontend/
│ ├── index.html
│ ├── app.js
│ ├── styles.css
│ ├── nginx.conf
│ └── Dockerfile
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
│ │ ├── mode_router.py
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

### lifespan
Qué hace: gestiona el ciclo de vida (startup/shutdown) vía async lifespan.
Usa: deps.lifespan / deps.startup / deps.shutdown (según wiring).

### health_check
Qué hace: healthcheck básico.
Usa: deps.check_db_health, deps.check_models_health.

### detailed_health_check
Qué hace: healthcheck detallado con dependencias.
Usa: deps.check_db_health, deps.check_models_health.

### get_config
Qué hace: devuelve configuración efectiva (sanitizada) para debug.
Usa: settings.get_settings.

## backend/settings.py

### load_settings
Qué hace: carga .env, valida variables críticas.
Usa: utilidades internas de parsing.

### get_settings
Qué hace: singleton/cachea Settings cargado.
Usa: load_settings.

### get_model_config
Qué hace: define routing de modelos (Gemma / Gemini / GPT).
Usa: variables cargadas por load_settings.

### get_model_config_with_user_keys
Qué hace: routing de modelo permitiendo keys por request.
Usa: get_model_config.

### get_embedding_config
Qué hace: devuelve config del proveedor de embeddings.
Usa: load_settings.

### get_rag_config
Qué hace: devuelve chunk sizes, thresholds, flags GraphRAG.
Usa: valores normalizados por load_settings.

### get_db_config
Qué hace: devuelve config de conexión a SurrealDB.
Usa: load_settings.

## backend/deps.py

### get_settings
Qué hace: inyecta Settings en endpoints.
Usa: settings.get_settings.

### get_db
Qué hace: entrega conexión viva a SurrealDB.
Usa: db.surreal.connect.

### get_llm_model
Qué hace: devuelve wrapper LLM listo para usar.
Usa: models.llm.get_llm.

### get_embedding
Qué hace: devuelve proveedor de embeddings.
Usa: models.embeddings.get_embedding_model.

### get_agents
Qué hace: construye instancias de agentes.
Usa: retrieval_agent, reasoning_agent, reflection_agent.

### get_session_id
Qué hace: resuelve/crea session_id.
Usa: request headers/cookies.

### get_student_id
Qué hace: resuelve student_id (si aplica).
Usa: request headers/cookies.

### lifespan
Qué hace: wrapper lifespan para FastAPI.
Usa: startup, shutdown.

### startup
Qué hace: inicializa dependencias (DB/LLM/embeddings/cache).
Usa: init_db, get_llm_model, get_embedding.

### shutdown
Qué hace: cierre limpio de conexiones y caches.
Usa: close_db.

### check_db_health
Qué hace: chequeo de salud de DB.
Usa: db.surreal.get_db / SurrealDBClient.is_connected.

### check_models_health
Qué hace: chequeo de LLM/embeddings.
Usa: models.llm.get_llm, models.embeddings.get_embedding_model.

# API

## api/ingest.py

### ingest_file
Qué hace: recibe archivo y dispara pipeline de ingestión.
Usa: graphs.source_graph.run_source_graph.

### ingest_url
Qué hace: procesa URLs web.
Usa: ingestion.content_processor.process_content.

### ingest_text
Qué hace: ingesta de texto plano.
Usa: graphs.source_graph.run_source_graph (vía content_processor).

### ingest_media
Qué hace: audio/video → texto → ingestión.
Usa: models.stt.transcribe, graphs.source_graph.run_source_graph.

### ingest_batch
Qué hace: ingesta por lote (varios items).
Usa: ingest_* internos por tipo.

### get_ingest_status_endpoint
Qué hace: consulta estado del job.
Usa: _get_ingest_status.

### list_ingests
Qué hace: lista ingestas recientes.
Usa: db.surreal.execute.

### cancel_ingest
Qué hace: cancela una ingesta en curso.
Usa: _set_ingest_status.

### list_sources
Qué hace: lista fuentes disponibles.
Usa: db.surreal.execute.

## api/query.py

### get_or_create_session
Qué hace: crea/reusa sesión conversacional.
Usa: db/memory (según config).

### query_stream
Qué hace: streaming de respuesta (SSE/stream).
Usa: reasoning_agent.generate_answer_stream.

### generate_stream
Qué hace: generador interno de chunks de streaming.
Usa: query_stream.

### query_student
Qué hace: flujo completo pregunta → respuesta.
Usa: retrieval_agent.retrieve, reasoning_agent.generate_answer, memory.session_memory.store_turn.

### query_debug
Qué hace: devuelve contexto intermedio.
Usa: retrieval_graph.merge_results.

### get_suggestions
Qué hace: sugiere próximos pasos/preguntas.
Usa: reasoning_agent.get_suggested_followups.

### get_cached
Qué hace: responde desde cache si existe.
Usa: memory.cache.get_cached_answer.

### clear_cache
Qué hace: invalida cache (por sesión/tag).
Usa: memory.cache.clear / invalidate.

## api/feedback.py

### submit_explicit_feedback
Qué hace: registra feedback explícito.
Usa: feedback.signals.parse_feedback, feedback.analytics.record_feedback.

### submit_implicit_feedback
Qué hace: registra señales implícitas.
Usa: feedback.signals.weight_feedback.

### get_student_feedback_summary
Qué hace: resumen de feedback por estudiante.
Usa: feedback.analytics.aggregate_metrics.

### get_content_feedback_summary
Qué hace: resumen de feedback por contenido.
Usa: feedback.analytics.aggregate_metrics.

### get_engagement_metrics
Qué hace: métricas de engagement.
Usa: feedback.analytics.compute_engagement.

# Graphs

## graphs/source_graph.py

### content_process
Qué hace: extracción y normalización del contenido.
Usa: ingestion.content_processor.process_content.

### save_source
Qué hace: persiste fuente base.
Usa: db.surreal.execute.

### chunk_content
Qué hace: chunking del contenido procesado.
Usa: ingestion.chunking.*.

### save_chunks
Qué hace: persiste chunks.
Usa: db.surreal.execute.

### vectorize_chunks
Qué hace: computa embeddings de chunks.
Usa: ingestion.vectorizer.submit_vectorization.

### trigger_transformations
Qué hace: decide qué transformaciones ejecutar.
Usa: graphs.transform_graph.run_transform_graph.

### finalize
Qué hace: cierre del estado de ingesta.
Usa: db.surreal.execute.

### run_source_graph
Qué hace: orquesta ingestión completa.
Usa: todas las anteriores + ingestion.vectorizer.submit_vectorization.

### get_ingestion_status
Qué hace: consulta estado de ingesta.
Usa: db.surreal.execute.

### reprocess_source
Qué hace: re-procesa una fuente.
Usa: run_source_graph.

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

### extract_relationships
Qué hace: extrae relaciones entre conceptos.
Usa: models.llm.generate.

### run_transform_graph
Qué hace: ejecuta transformaciones (conceptos/definiciones/relaciones).
Usa: graph.builders.*.

### run_transform_graph
Qué hace: ejecuta transformaciones en paralelo.
Usa: todas las anteriores + graph.builders.create_concept_node.

## graphs/retrieval_graph.py

### get_retrieval_config
Qué hace: devuelve configuración de retrieval.
Usa: settings.get_rag_config.

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

### retrieve
Qué hace: punto de entrada que orquesta retrieval híbrido.
Usa: graph_retrieval / vector_retrieval / bm25_retrieval.

# Agents

## agents/mode_router.py

### detect_mode
Qué hace: autodetecta el modo pedagógico.
Usa: heurísticas/patrones de query.

### detect_mode_with_details
Qué hace: autodetecta modo + razón/metadata.
Usa: detect_mode.

### get_mode_description
Qué hace: descripción humana del modo.
Usa: LearningMode.
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

### retrieve_for_concepts
Qué hace: retrieval focalizado por conceptos.
Usa: graphs.retrieval_graph.graph_retrieval.

### get_retrieval_summary
Qué hace: resumen del retrieval para debug.
Usa: resultados de retrieve.

## agents/reasoning_agent.py

### build_prompt
Qué hace: construye prompt con contexto curado.
Usa: utils.text.truncate_context.

### generate_answer
Qué hace: genera respuesta final.
Usa: models.llm.generate.

### generate_answer_stream
Qué hace: genera respuesta en streaming.
Usa: models.llm.generate_stream.

### adapt_to_student
Qué hace: ajusta nivel pedagógico.
Usa: memory.student_profile.load_profile.

### get_suggested_followups
Qué hace: sugiere preguntas/ejercicios siguientes.
Usa: heurísticas sobre respuesta + modo.

## agents/reflection_agent.py

### evaluate_answer
Qué hace: evalúa cobertura y coherencia.
Usa: models.llm.generate.

### decide_retry
Qué hace: decide si reintentar.
Usa: evaluate_answer.

### get_retry_adjustments
Qué hace: calcula ajustes para reintento.
Usa: EvaluationResult + LearningMode.

### request_more_context
Qué hace: fuerza nuevo retrieval.
Usa: retrieval_agent.retrieve.

### generate_fallback_response
Qué hace: respuesta segura cuando todo falla.
Usa: razonamiento mínimo.

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

### extract_title_from_content
Qué hace: intenta extraer título desde el contenido.
Usa: heurísticas.

## ingestion/chunking.py

### hierarchical_chunk
Qué hace: parent/child chunking.
Usa: utils.text.token_count.

### simple_chunk
Qué hace: chunking simple por longitud.
Usa: split_by_separators.

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

### embed_chunks_batch
Qué hace: embeddings en batch.
Usa: models.embeddings.batch_embed.

### retry_failed_chunks
Qué hace: reprocesa fallos.
Usa: embed_chunk.

### get_vectorization_status
Qué hace: estado del job de vectorización.
Usa: VectorizationQueue.

# Graph

## graph/schema.py

### define_nodes
Qué hace: define tipos de nodos.
Usa: constantes internas.

### define_edges
Qué hace: define relaciones válidas.
Usa: constantes internas.

### validate_graph
Qué hace: valida consistencia global.
Usa: validate_node, validate_edge.

### validate_node
Qué hace: valida nodo individual.
Usa: schema.

### validate_edge
Qué hace: valida arista individual.
Usa: schema.

## graph/builders.py

### create_concept_node
Qué hace: inserta concepto.
Usa: db.surreal.execute.

### create_chunk_node
Qué hace: inserta chunk.
Usa: db.surreal.execute.

### create_source_node
Qué hace: inserta fuente.
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

### find_concepts_by_query
Qué hace: busca conceptos relevantes por texto.
Usa: DB + heurísticas.

### extract_subgraph
Qué hace: extrae subgrafo para contexto.
Usa: expand_concepts.

# Retrieval

## retrieval/bm25.py

### search_text
Qué hace: búsqueda literal.
Usa: función SurrealDB.

### highlight_matches
Qué hace: snippets.
Usa: resultados BM25.

### create_fulltext_index
Qué hace: crea índice fulltext.
Usa: SurrealDB.

## retrieval/vector.py

### search_vectors
Qué hace: cosine similarity.
Usa: models.embeddings.embed_text.

### filter_by_threshold
Qué hace: corte por score.
Usa: config.

### create_vector_index
Qué hace: crea índice vectorial.
Usa: SurrealDB.

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

### deduplicate_results
Qué hace: deduplica resultados por fuente/id.
Usa: heurísticas.

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

### get_recommended_concepts
Qué hace: sugiere conceptos recomendados.
Usa: perfil + mastery.

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

## memory/cache.py

### get_cached_answer
Qué hace: busca cache.
Usa: hashing.

### store_cache
Qué hace: guarda respuesta.
Usa: DB o Redis.

### invalidate_cache
Qué hace: limpieza selectiva.
Usa: reglas de caducidad.

### clear_all_caches
Qué hace: limpia todas las caches.
Usa: CacheManager.clear.

# Feedback

## feedback/signals.py

### parse_feedback
Qué hace: normaliza feedback.
Usa: validaciones.

### weight_feedback
Qué hace: asigna impacto.
Usa: heurísticas.

### parse_multiple_feedbacks
Qué hace: parsea múltiples entradas.
Usa: parse_feedback.

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

#### compute_engagement
Qué hace: computa engagement.
Usa: métricas de señales.

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

### apply_feedback_to_graph
Qué hace: aplica feedback y actualiza grafo.
Usa: reinforce_edges/weaken_edges/schedule_revectorization.

# Models

## models/llm.py

### generate
Qué hace: llamada genérica LLM.
Usa: proveedor activo.

### generate_stream
Qué hace: generación en streaming.
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

### select_embedding_model
Qué hace: selecciona proveedor de embeddings.
Usa: settings.get_embedding_config.

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
Usa: SurrealDBClient.connect.

### execute
Qué hace: ejecuta query.
Usa: SurrealDBClient.execute.

### transaction
Qué hace: manejo transaccional.
Usa: SurrealDBClient.transaction.

### close
Qué hace: cierra conexión.
Usa: SurrealDBClient.disconnect.

# Utils

## utils/text.py

### get_tokenizer
Qué hace: obtiene tokenizer configurado.
Usa: provider/tokenizer local.

### token_count
Qué hace: cuenta tokens.
Usa: tokenizer.

### clean_markdown
Qué hace: limpieza estructural.
Usa: regex.

### normalize_text
Qué hace: normaliza casing/espacios.
Usa: clean_whitespace.

### truncate_context
Qué hace: ajusta longitud.
Usa: token_count.