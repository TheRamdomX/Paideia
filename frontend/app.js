/**
 * Paideia Frontend - Chat Application
 * JavaScript para interactuar con la API de Paideia
 */

// ==========================================
// Configuración
// ==========================================

// Usa ruta relativa para funcionar tanto en desarrollo como en Docker con nginx proxy
const API_BASE_URL = '/api/v1';

// Estado de la aplicación
const state = {
    studentId: localStorage.getItem('paideia_student_id') || null,
    sessionId: localStorage.getItem('paideia_session_id') || generateSessionId(),
    openaiKey: localStorage.getItem('paideia_openai_key') || null,
    googleKey: localStorage.getItem('paideia_google_key') || null,
    preferredProvider: localStorage.getItem('paideia_preferred_provider') || null,
    model: localStorage.getItem('paideia_model') || null,
    learningMode: localStorage.getItem('paideia_learning_mode') || null,  // null = automático
    database: localStorage.getItem('paideia_database') || 'education',  // Base de datos activa
    isLoading: false,
    chatHistory: JSON.parse(localStorage.getItem('paideia_chat_history') || '[]'),
    documents: []  // Lista de documentos ingestados
};

// ==========================================
// Utilidades
// ==========================================

function generateSessionId() {
    const id = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('paideia_session_id', id);
    return id;
}

function generateMessageId() {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMessage(text) {
    // Convertir saltos de línea
    let formatted = escapeHtml(text).replace(/\n/g, '<br>');
    
    // Convertir **texto** a negrita
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convertir `código` a monospace
    formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
    
    return formatted;
}

function saveChatHistory() {
    // Guardar solo los últimos 50 mensajes
    const toSave = state.chatHistory.slice(-50);
    localStorage.setItem('paideia_chat_history', JSON.stringify(toSave));
}

// ==========================================
// Elementos del DOM
// ==========================================

const elements = {
    chatMessages: document.getElementById('chat-messages'),
    messageInput: document.getElementById('message-input'),
    btnSend: document.getElementById('btn-send'),
    btnUserConfig: document.getElementById('btn-user-config'),
    userModal: document.getElementById('user-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    studentIdInput: document.getElementById('student-id-input'),
    openaiKeyInput: document.getElementById('openai-key-input'),
    googleKeyInput: document.getElementById('google-key-input'),
    preferredProviderSelect: document.getElementById('preferred-provider'),
    modelSelect: document.getElementById('model-select'),
    btnSaveUser: document.getElementById('btn-save-user'),
    btnClearUser: document.getElementById('btn-clear-user'),
    currentUserDisplay: document.getElementById('current-user-display'),
    // Secciones dinámicas de API keys y modelo
    openaiKeySection: document.getElementById('openai-key-section'),
    googleKeySection: document.getElementById('google-key-section'),
    modelSection: document.getElementById('model-section'),
    modelInfo: document.getElementById('model-info'),
    // Elementos de base de datos
    databaseSelect: document.getElementById('database-select'),
    btnNewDatabase: document.getElementById('btn-new-database'),
    newDatabaseForm: document.getElementById('new-database-form'),
    newDatabaseInput: document.getElementById('new-database-input'),
    btnCreateDatabase: document.getElementById('btn-create-database'),
    btnCancelDatabase: document.getElementById('btn-cancel-database'),
    // Elementos del modal de documentos
    btnDocuments: document.getElementById('btn-documents'),
    docsModal: document.getElementById('docs-modal'),
    btnCloseDocsModal: document.getElementById('btn-close-docs-modal'),
    uploadZone: document.getElementById('upload-zone'),
    docFileInput: document.getElementById('doc-file-input'),
    btnSelectFile: document.getElementById('btn-select-file'),
    uploadProgress: document.getElementById('upload-progress'),
    uploadFileName: document.getElementById('upload-file-name'),
    uploadStatus: document.getElementById('upload-status'),
    progressFill: document.getElementById('progress-fill'),
    docsList: document.getElementById('docs-list'),
    btnRefreshDocs: document.getElementById('btn-refresh-docs'),
    docsCountBadge: document.getElementById('docs-count-badge')
};

// Cache de modelos del backend
let cachedModels = {
    openai: [],
    google: [],
    loaded: false
};

// ==========================================
// Modelos Dinámicos
// ==========================================

async function loadModelsFromBackend() {
    if (cachedModels.loaded) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) throw new Error('Error cargando modelos');
        
        const data = await response.json();
        
        // Separar por proveedor
        cachedModels.openai = data.models.filter(m => m.provider === 'openai');
        cachedModels.google = data.models.filter(m => m.provider === 'google');
        cachedModels.loaded = true;
        
        console.log(`✅ Modelos cargados: ${cachedModels.openai.length} OpenAI, ${cachedModels.google.length} Google`);
        
    } catch (error) {
        console.error('Error cargando modelos:', error);
        // Usar fallback con modelos básicos
        cachedModels.openai = [
            { api_name: 'gpt-4o-mini', name: 'GPT-4o Mini', context_window: 128000 },
            { api_name: 'gpt-4o', name: 'GPT-4o', context_window: 128000 },
        ];
        cachedModels.google = [
            { api_name: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', context_window: 1000000 },
            { api_name: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', context_window: 1000000 },
        ];
        cachedModels.loaded = true;
    }
}

function updateModelsForProvider(provider) {
    const models = provider === 'openai' ? cachedModels.openai : 
                   provider === 'google' ? cachedModels.google : [];
    
    elements.modelSelect.innerHTML = '';
    
    if (models.length === 0) {
        elements.modelSelect.innerHTML = '<option value="">No hay modelos disponibles</option>';
        return;
    }
    
    // Opción por defecto
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = '-- Selecciona un modelo --';
    elements.modelSelect.appendChild(defaultOpt);
    
    // Agregar modelos
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.api_name;
        // Formato: nombre (contexto)
        const contextK = Math.round(model.context_window / 1000);
        option.textContent = `${model.name} (${contextK}K contexto)`;
        option.dataset.contextWindow = model.context_window;
        elements.modelSelect.appendChild(option);
    });
    
    // Si había un modelo guardado de este proveedor, seleccionarlo
    if (state.model && models.some(m => m.api_name === state.model)) {
        elements.modelSelect.value = state.model;
        updateModelInfo(state.model, models);
    }
}

function updateModelInfo(modelName, models) {
    const model = models.find(m => m.api_name === modelName);
    if (model && elements.modelInfo) {
        const contextK = Math.round(model.context_window / 1000);
        elements.modelInfo.textContent = `Ventana de contexto: ${contextK}K tokens`;
    } else if (elements.modelInfo) {
        elements.modelInfo.textContent = '';
    }
}

function handleProviderChange() {
    const provider = elements.preferredProviderSelect.value;
    
    // Ocultar todo primero
    elements.openaiKeySection.classList.add('hidden');
    elements.googleKeySection.classList.add('hidden');
    elements.modelSection.classList.add('hidden');
    
    if (!provider) {
        return;
    }
    
    // Mostrar sección de API key según proveedor
    if (provider === 'openai') {
        elements.openaiKeySection.classList.remove('hidden');
    } else if (provider === 'google') {
        elements.googleKeySection.classList.remove('hidden');
    }
    
    // Mostrar selector de modelos
    elements.modelSection.classList.remove('hidden');
    
    // Actualizar modelos disponibles
    updateModelsForProvider(provider);
}

function handleModelSelectChange() {
    const provider = elements.preferredProviderSelect.value;
    const modelName = elements.modelSelect.value;
    const models = provider === 'openai' ? cachedModels.openai : cachedModels.google;
    updateModelInfo(modelName, models);
}

// ==========================================
// Renderizado de Mensajes
// ==========================================

function createMessageElement(message) {
    const div = document.createElement('div');
    div.className = `message ${message.role}`;
    div.dataset.id = message.id;

    if (message.role === 'user') {
        const initial = state.studentId ? state.studentId.charAt(0).toUpperCase() : '?';
        div.innerHTML = `
            <div class="message-avatar">${initial}</div>
            <div class="message-content">
                <p>${formatMessage(message.content)}</p>
            </div>
        `;
    } else if (message.role === 'assistant') {
        let sourcesHtml = '';
        if (message.sources && message.sources.length > 0) {
            sourcesHtml = `
                <details class="message-sources">
                    <summary>📚 ${message.sources.length} fuente(s)</summary>
                    <ul>
                        ${message.sources.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
                    </ul>
                </details>
            `;
        }
        
        // Botones de feedback
        const feedbackHtml = `
            <div class="message-feedback" data-message-id="${message.id}">
                <button class="feedback-btn" data-feedback="positive" title="Respuesta útil">
                    👍
                </button>
                <button class="feedback-btn" data-feedback="negative" title="Respuesta no útil">
                    👎
                </button>
            </div>
        `;
        
        div.innerHTML = `
            <div class="message-avatar">🎓</div>
            <div class="message-content">
                <p>${formatMessage(message.content)}</p>
                <div class="message-footer">
                    ${sourcesHtml}
                    ${feedbackHtml}
                </div>
            </div>
        `;
        
        // Agregar event listeners para feedback
        setTimeout(() => {
            const feedbackBtns = div.querySelectorAll('.feedback-btn');
            feedbackBtns.forEach(btn => {
                btn.addEventListener('click', () => handleFeedback(message.id, btn.dataset.feedback, btn));
            });
        }, 0);
    } else if (message.role === 'system') {
        div.className = `message system ${message.isError ? 'error' : ''}`;
        div.innerHTML = `
            <div class="message-content">
                ${message.isError ? '❌' : '✅'} ${escapeHtml(message.content)}
            </div>
        `;
    }

    return div;
}

function createLoadingMessage() {
    const div = document.createElement('div');
    div.className = 'message assistant loading';
    div.id = 'loading-message';
    div.innerHTML = `
        <div class="message-avatar">🎓</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return div;
}

function addMessage(message) {
    // Remover mensaje de loading si existe
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) {
        loadingMsg.remove();
    }

    const messageElement = createMessageElement(message);
    elements.chatMessages.appendChild(messageElement);
    scrollToBottom();

    // Guardar en historial (excepto mensajes de sistema temporales)
    if (message.role !== 'system' || message.persist) {
        state.chatHistory.push(message);
        saveChatHistory();
    }
}

function showLoading() {
    const loadingElement = createLoadingMessage();
    elements.chatMessages.appendChild(loadingElement);
    scrollToBottom();
}

function hideLoading() {
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function loadChatHistory() {
    // Limpiar mensajes existentes excepto el de bienvenida
    const welcomeMsg = elements.chatMessages.querySelector('.message.assistant');
    elements.chatMessages.innerHTML = '';
    if (welcomeMsg) {
        elements.chatMessages.appendChild(welcomeMsg);
    }

    // Cargar historial
    state.chatHistory.forEach(message => {
        const messageElement = createMessageElement(message);
        elements.chatMessages.appendChild(messageElement);
    });

    scrollToBottom();
}

// ==========================================
// Feedback
// ==========================================

async function handleFeedback(messageId, feedbackType, buttonElement) {
    const feedbackContainer = buttonElement.closest('.message-feedback');
    const allBtns = feedbackContainer.querySelectorAll('.feedback-btn');
    
    // Si ya se dio feedback, no hacer nada
    if (feedbackContainer.classList.contains('submitted')) {
        return;
    }
    
    // Marcar como enviado visualmente
    allBtns.forEach(btn => btn.classList.remove('selected'));
    buttonElement.classList.add('selected');
    feedbackContainer.classList.add('submitted');
    
    try {
        const response = await fetch(`${API_BASE_URL}/feedback/explicit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query_id: messageId,
                feedback_type: feedbackType === 'positive' ? 'helpful' : 'not_helpful',
                rating: feedbackType === 'positive' ? 5 : 1,
                student_id: state.studentId,
                session_id: state.sessionId,
            })
        });
        
        if (!response.ok) {
            throw new Error('Error enviando feedback');
        }
        
        console.log(`✅ Feedback "${feedbackType}" enviado para mensaje ${messageId}`);
        
    } catch (error) {
        console.error('Error enviando feedback:', error);
        // Revertir estado visual en caso de error
        feedbackContainer.classList.remove('submitted');
        buttonElement.classList.remove('selected');
    }
}

// ==========================================
// Database Management
// ==========================================

async function loadDatabases() {
    try {
        const response = await fetch(`${API_BASE_URL}/databases`);
        if (!response.ok) throw new Error('Error cargando bases de datos');
        
        const data = await response.json();
        // La API devuelve { databases: [{name, is_current}], current, namespace }
        const databases = data.databases || [];
        
        // Limpiar y poblar el select
        elements.databaseSelect.innerHTML = '';
        databases.forEach(db => {
            const option = document.createElement('option');
            option.value = db.name;
            option.textContent = db.name;
            elements.databaseSelect.appendChild(option);
        });
        
        // Seleccionar la base de datos actual
        const current = data.current || state.database;
        if (current) {
            state.database = current;
            elements.databaseSelect.value = current;
            localStorage.setItem('paideia_database', current);
        }
        
    } catch (error) {
        console.error('Error cargando bases de datos:', error);
    }
}

async function createDatabase(name) {
    if (!name || !name.trim()) {
        alert('Ingresa un nombre para la base de datos');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/databases`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name.trim() })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error creando base de datos');
        }
        
        // Limpiar input y ocultar formulario
        elements.newDatabaseInput.value = '';
        elements.newDatabaseForm.classList.add('hidden');
        
        // Recargar lista y seleccionar la nueva
        await loadDatabases();
        state.database = name.trim();
        elements.databaseSelect.value = state.database;
        
        // Cambiar a la nueva base de datos
        await switchDatabase(name.trim());
        
        console.log(`✅ Base de datos "${name.trim()}" creada`);
        
    } catch (error) {
        console.error('Error creando base de datos:', error);
        alert('Error creando base de datos: ' + error.message);
    }
}

async function switchDatabase(dbName) {
    try {
        const response = await fetch(`${API_BASE_URL}/databases/switch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ database: dbName })
        });
        
        if (!response.ok) throw new Error('Error cambiando base de datos');
        
        state.database = dbName;
        localStorage.setItem('paideia_database', dbName);
        
        console.log(`✅ Cambiado a base de datos: ${dbName}`);
        
        // Recargar documentos para la nueva base de datos
        await loadDocumentsList();
        
    } catch (error) {
        console.error('Error cambiando base de datos:', error);
    }
}

// ==========================================
// API Calls
// ==========================================

async function sendQuery(question) {
    const headers = {
        'Content-Type': 'application/json'
    };

    // Agregar headers de identificación
    if (state.studentId) {
        headers['X-Student-ID'] = state.studentId;
    }
    if (state.sessionId) {
        headers['X-Session-ID'] = state.sessionId;
    }
    
    // Agregar headers de API keys (SOLO si el usuario las configuró)
    if (state.openaiKey) {
        headers['X-OpenAI-Key'] = state.openaiKey;
    }
    if (state.googleKey) {
        headers['X-Google-Key'] = state.googleKey;
    }
    if (state.preferredProvider) {
        headers['X-Preferred-Provider'] = state.preferredProvider;
    }
    if (state.model) {
        headers['X-Model'] = state.model;
    }
    if (state.database) {
        headers['X-Database'] = state.database;
    }

    // Construir body con learning_mode opcional
    const body = {
        question,
        session_id: state.sessionId,
        student_id: state.studentId,
        use_cache: true
    };
    
    // Solo agregar learning_mode si está seleccionado (no automático)
    if (state.learningMode) {
        body.learning_mode = state.learningMode;
    }

    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Error desconocido' }));
        throw new Error(error.detail || error.message || `Error ${response.status}`);
    }

    return response.json();
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify({
        uploaded_by: state.studentId || 'anonymous',
        session_id: state.sessionId
    }));

    // Preparar headers con API keys
    const headers = {};
    if (state.openaiKey) {
        headers['X-OpenAI-Key'] = state.openaiKey;
    }
    if (state.googleKey) {
        headers['X-Google-Key'] = state.googleKey;
    }
    if (state.database) {
        headers['X-Database'] = state.database;
    }

    const response = await fetch(`${API_BASE_URL}/ingest/file`, {
        method: 'POST',
        headers: headers,
        body: formData
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Error subiendo archivo' }));
        throw new Error(error.detail || `Error ${response.status}`);
    }

    return response.json();
}

// ==========================================
// Event Handlers
// ==========================================

async function handleSendMessage() {
    const message = elements.messageInput.value.trim();
    
    if (!message) {
        return;
    }

    if (state.isLoading) {
        return;
    }

    state.isLoading = true;
    elements.btnSend.disabled = true;

    try {
        // Agregar mensaje del usuario
        addMessage({
            id: generateMessageId(),
            role: 'user',
            content: message
        });

        // Limpiar input
        elements.messageInput.value = '';
        autoResizeTextarea();

        // Mostrar indicador de carga
        showLoading();

        // Enviar consulta
        const response = await sendQuery(message);

        // Agregar respuesta del asistente
        addMessage({
            id: response.query_id || generateMessageId(),
            role: 'assistant',
            content: response.answer,
            sources: response.sources || [],
            confidence: response.confidence
        });

    } catch (error) {
        hideLoading();
        addMessage({
            id: generateMessageId(),
            role: 'system',
            content: `Error: ${error.message}`,
            isError: true
        });
    } finally {
        state.isLoading = false;
        elements.btnSend.disabled = false;
        elements.messageInput.focus();
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSendMessage();
    }
}

function autoResizeTextarea() {
    const textarea = elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// ==========================================
// User Modal
// ==========================================

async function openUserModal() {
    elements.userModal.classList.remove('hidden');
    
    // Cargar modelos del backend si no están cargados
    await loadModelsFromBackend();
    
    // Restaurar valores guardados
    elements.studentIdInput.value = state.studentId || '';
    elements.openaiKeyInput.value = state.openaiKey || '';
    elements.googleKeyInput.value = state.googleKey || '';
    elements.preferredProviderSelect.value = state.preferredProvider || '';
    
    // Configurar UI según proveedor guardado
    if (state.preferredProvider) {
        handleProviderChange();
        // Restaurar modelo seleccionado
        if (state.model) {
            elements.modelSelect.value = state.model;
            handleModelSelectChange();
        }
    }
    
    elements.studentIdInput.focus();
}

function closeUserModal() {
    elements.userModal.classList.add('hidden');
}

function saveUser() {
    const studentId = elements.studentIdInput.value.trim();
    const openaiKey = elements.openaiKeyInput.value.trim();
    const googleKey = elements.googleKeyInput.value.trim();
    const preferredProvider = elements.preferredProviderSelect.value;
    const model = elements.modelSelect.value;
    
    // Guardar ID de estudiante
    if (studentId) {
        state.studentId = studentId;
        localStorage.setItem('paideia_student_id', studentId);
    } else {
        state.studentId = null;
        localStorage.removeItem('paideia_student_id');
    }
    
    // Guardar API keys
    if (openaiKey) {
        state.openaiKey = openaiKey;
        localStorage.setItem('paideia_openai_key', openaiKey);
    } else {
        state.openaiKey = null;
        localStorage.removeItem('paideia_openai_key');
    }
    
    if (googleKey) {
        state.googleKey = googleKey;
        localStorage.setItem('paideia_google_key', googleKey);
    } else {
        state.googleKey = null;
        localStorage.removeItem('paideia_google_key');
    }
    
    // Guardar proveedor preferido
    if (preferredProvider) {
        state.preferredProvider = preferredProvider;
        localStorage.setItem('paideia_preferred_provider', preferredProvider);
    } else {
        state.preferredProvider = null;
        localStorage.removeItem('paideia_preferred_provider');
    }
    
    // Guardar modelo
    if (model) {
        state.model = model;
        localStorage.setItem('paideia_model', model);
    } else {
        state.model = null;
        localStorage.removeItem('paideia_model');
    }
    
    updateUserDisplay();
    
    // Mensaje de confirmación
    const configuredItems = [];
    if (studentId) configuredItems.push(`ID: ${studentId}`);
    if (preferredProvider === 'openai' && openaiKey) configuredItems.push('🤖 OpenAI');
    if (preferredProvider === 'google' && googleKey) configuredItems.push('💎 Google');
    if (model) configuredItems.push(`Modelo: ${model}`);
    
    if (configuredItems.length > 0) {
        addMessage({
            id: generateMessageId(),
            role: 'system',
            content: `Configuración guardada: ${configuredItems.join(', ')}`,
            persist: true
        });
    } else {
        addMessage({
            id: generateMessageId(),
            role: 'system',
            content: 'Usando configuración del servidor',
            persist: true
        });
    }
    
    closeUserModal();
}

function clearUser() {
    state.studentId = null;
    state.openaiKey = null;
    state.googleKey = null;
    state.preferredProvider = null;
    state.model = null;
    
    localStorage.removeItem('paideia_student_id');
    localStorage.removeItem('paideia_openai_key');
    localStorage.removeItem('paideia_google_key');
    localStorage.removeItem('paideia_preferred_provider');
    localStorage.removeItem('paideia_model');
    
    elements.studentIdInput.value = '';
    elements.openaiKeyInput.value = '';
    elements.googleKeyInput.value = '';
    elements.preferredProviderSelect.value = '';
    elements.modelSelect.value = '';
    
    // Ocultar secciones dinámicas
    elements.openaiKeySection.classList.add('hidden');
    elements.googleKeySection.classList.add('hidden');
    elements.modelSection.classList.add('hidden');
    if (elements.modelInfo) elements.modelInfo.textContent = '';
    
    updateUserDisplay();
    
    addMessage({
        id: generateMessageId(),
        role: 'system',
        content: 'Configuración limpiada - Usando configuración del servidor'
    });
    
    closeUserModal();
}

function updateUserDisplay() {
    const hasConfig = state.studentId || state.preferredProvider;
    
    if (hasConfig) {
        const parts = [];
        if (state.studentId) parts.push(state.studentId);
        // Mostrar solo el proveedor configurado
        if (state.preferredProvider === 'openai' && state.openaiKey) parts.push('🤖');
        if (state.preferredProvider === 'google' && state.googleKey) parts.push('💎');
        if (state.model) parts.push(`[${state.model}]`);
        
        elements.currentUserDisplay.textContent = parts.join(' ');
        elements.currentUserDisplay.classList.add('identified');
    } else {
        elements.currentUserDisplay.textContent = 'Sin configurar';
        elements.currentUserDisplay.classList.remove('identified');
    }
}

// ==========================================
// Modal de Documentos
// ==========================================

function openDocsModal() {
    elements.docsModal.classList.remove('hidden');
    loadDocumentsList();
}

function closeDocsModal() {
    elements.docsModal.classList.add('hidden');
    resetUploadProgress();
}

function resetUploadProgress() {
    elements.uploadProgress.classList.add('hidden');
    elements.progressFill.style.width = '0%';
    elements.progressFill.classList.remove('complete');
    elements.uploadStatus.textContent = '';
    elements.uploadStatus.className = '';
}

async function loadDocumentsList() {
    try {
        const headers = {};
        if (state.database) {
            headers['X-Database'] = state.database;
        }
        
        const response = await fetch(`${API_BASE_URL}/ingest/sources`, { headers });
        if (!response.ok) {
            throw new Error('Error cargando documentos');
        }
        
        const data = await response.json();
        state.documents = data.sources || [];
        renderDocumentsList();
        updateDocsCount();
    } catch (error) {
        console.error('Error loading documents:', error);
        // Mostrar lista vacía si hay error
        state.documents = [];
        renderDocumentsList();
    }
}

function renderDocumentsList() {
    if (state.documents.length === 0) {
        elements.docsList.innerHTML = `
            <div class="docs-empty">
                <span>📭</span>
                <p>No hay documentos cargados</p>
            </div>
        `;
        return;
    }
    
    elements.docsList.innerHTML = state.documents.map(doc => {
        const icon = getFileIcon(doc.source_type || doc.type || 'text');
        const status = doc.status || 'completed';
        const statusClass = status.toLowerCase();
        const statusText = {
            'completed': 'Listo',
            'processing': 'Procesando',
            'pending': 'Pendiente',
            'failed': 'Error'
        }[statusClass] || status;
        
        const date = doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '';
        const chunks = doc.chunk_count ? `${doc.chunk_count} chunks` : '';
        const meta = [date, chunks].filter(Boolean).join(' • ');
        
        return `
            <div class="doc-item" data-id="${doc.source_id || doc.id}">
                <div class="doc-info">
                    <span class="doc-icon">${icon}</span>
                    <div class="doc-details">
                        <div class="doc-name">${escapeHtml(doc.title || doc.name || 'Sin título')}</div>
                        <div class="doc-meta">${meta}</div>
                    </div>
                </div>
                <div class="doc-status">
                    <span class="status-badge ${statusClass}">${statusText}</span>
                </div>
            </div>
        `;
    }).join('');
}

function getFileIcon(type) {
    const icons = {
        'pdf': '📕',
        'text': '📄',
        'txt': '📄',
        'markdown': '📝',
        'md': '📝',
        'docx': '📘',
        'doc': '📘',
        'web': '🌐',
        'youtube': '🎬',
        'audio': '🎵',
        'video': '🎥'
    };
    return icons[type.toLowerCase()] || '📄';
}

function updateDocsCount() {
    const count = state.documents.length;
    if (count > 0) {
        elements.docsCountBadge.textContent = count > 99 ? '99+' : count;
        elements.docsCountBadge.classList.remove('hidden');
    } else {
        elements.docsCountBadge.classList.add('hidden');
    }
}

async function handleDocumentUpload(file) {
    // Mostrar progreso
    elements.uploadProgress.classList.remove('hidden');
    elements.uploadFileName.textContent = file.name;
    elements.uploadStatus.textContent = 'Subiendo...';
    elements.uploadStatus.className = '';
    elements.progressFill.style.width = '30%';
    
    try {
        const result = await uploadFile(file);
        
        // Actualizar progreso
        elements.progressFill.style.width = '100%';
        elements.progressFill.classList.add('complete');
        elements.uploadStatus.textContent = `✓ ${result.chunks_created || 0} chunks creados`;
        elements.uploadStatus.className = 'success';
        
        // Recargar lista
        setTimeout(() => {
            loadDocumentsList();
            // Resetear después de 3 segundos
            setTimeout(resetUploadProgress, 3000);
        }, 500);
        
    } catch (error) {
        elements.progressFill.style.width = '100%';
        elements.uploadStatus.textContent = `✗ ${error.message}`;
        elements.uploadStatus.className = 'error';
    }
}

function setupDragAndDrop() {
    const zone = elements.uploadZone;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, () => {
            zone.classList.add('drag-over');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, () => {
            zone.classList.remove('drag-over');
        });
    });
    
    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleDocumentUpload(files[0]);
        }
    });
}

// ==========================================
// Mode Selector (Select de Modo Pedagógico)
// ==========================================

function initModeSelector() {
    const modeSelect = document.getElementById('mode-select');
    
    // Restaurar modo guardado
    if (state.learningMode) {
        modeSelect.value = state.learningMode;
    }
    
    // Event listener
    modeSelect.addEventListener('change', () => {
        const newMode = modeSelect.value || null;
        state.learningMode = newMode;
        
        if (newMode) {
            localStorage.setItem('paideia_learning_mode', newMode);
        } else {
            localStorage.removeItem('paideia_learning_mode');
        }
        
        // Log
        const modeNames = {
            '': 'Automático',
            'concept': 'Concepto',
            'practice': 'Práctica', 
            'exercise_list': 'Lista de Ejercicios'
        };
        console.log(`🎯 Modo pedagógico: ${modeNames[modeSelect.value] || 'Automático'}`);
    });
}

// ==========================================
// Inicialización
// ==========================================

function init() {
    // Actualizar display de usuario
    updateUserDisplay();

    // Cargar historial de chat
    loadChatHistory();
    
    // Cargar lista de documentos inicial
    loadDocumentsList();
    
    // Cargar lista de bases de datos
    loadDatabases();
    
    // Inicializar selector de modo
    initModeSelector();

    // Event listeners - Envío de mensajes
    elements.btnSend.addEventListener('click', handleSendMessage);
    elements.messageInput.addEventListener('keypress', handleKeyPress);
    elements.messageInput.addEventListener('input', autoResizeTextarea);

    // Event listeners - Modal de usuario
    elements.btnUserConfig.addEventListener('click', openUserModal);
    elements.btnCloseModal.addEventListener('click', closeUserModal);
    elements.btnSaveUser.addEventListener('click', saveUser);
    elements.btnClearUser.addEventListener('click', clearUser);
    
    // Event listeners - Base de datos
    elements.databaseSelect.addEventListener('change', (e) => {
        switchDatabase(e.target.value);
    });
    
    // Event listeners - Proveedor y Modelo dinámico
    elements.preferredProviderSelect.addEventListener('change', handleProviderChange);
    elements.modelSelect.addEventListener('change', handleModelSelectChange);
    
    elements.btnNewDatabase.addEventListener('click', () => {
        elements.newDatabaseForm.classList.toggle('hidden');
        if (!elements.newDatabaseForm.classList.contains('hidden')) {
            elements.newDatabaseInput.focus();
        }
    });
    elements.btnCreateDatabase.addEventListener('click', () => {
        createDatabase(elements.newDatabaseInput.value);
    });
    elements.btnCancelDatabase.addEventListener('click', () => {
        elements.newDatabaseForm.classList.add('hidden');
        elements.newDatabaseInput.value = '';
    });
    elements.newDatabaseInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            createDatabase(elements.newDatabaseInput.value);
        }
    });

    // Event listeners - Modal de documentos
    elements.btnDocuments.addEventListener('click', openDocsModal);
    elements.btnCloseDocsModal.addEventListener('click', closeDocsModal);
    elements.btnSelectFile.addEventListener('click', () => elements.docFileInput.click());
    elements.uploadZone.addEventListener('click', (e) => {
        if (e.target !== elements.btnSelectFile) {
            elements.docFileInput.click();
        }
    });
    elements.docFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleDocumentUpload(e.target.files[0]);
            e.target.value = ''; // Reset para permitir subir el mismo archivo otra vez
        }
    });
    elements.btnRefreshDocs.addEventListener('click', loadDocumentsList);
    
    // Setup drag and drop
    setupDragAndDrop();

    // Cerrar modales al hacer clic fuera
    elements.userModal.addEventListener('click', (e) => {
        if (e.target === elements.userModal) {
            closeUserModal();
        }
    });
    elements.docsModal.addEventListener('click', (e) => {
        if (e.target === elements.docsModal) {
            closeDocsModal();
        }
    });

    // Cerrar modales con Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!elements.userModal.classList.contains('hidden')) {
                closeUserModal();
            }
            if (!elements.docsModal.classList.contains('hidden')) {
                closeDocsModal();
            }
        }
    });

    // Enter en input de estudiante
    elements.studentIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveUser();
        }
    });

    // Focus en input de mensaje
    elements.messageInput.focus();

    console.log('🎓 Paideia Chat initialized');
    console.log(`Session ID: ${state.sessionId}`);
    console.log(`Student ID: ${state.studentId || 'Not set'}`);
}

// Iniciar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', init);
