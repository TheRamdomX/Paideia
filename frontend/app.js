/**
 * Paideia Frontend - Chat Application
 * JavaScript para interactuar con la API de Paideia
 */

// ==========================================
// Configuración
// ==========================================

const API_BASE_URL = 'http://localhost:8080/api/v1';

// Estado de la aplicación
const state = {
    studentId: localStorage.getItem('paideia_student_id') || null,
    sessionId: localStorage.getItem('paideia_session_id') || generateSessionId(),
    openaiKey: localStorage.getItem('paideia_openai_key') || null,
    googleKey: localStorage.getItem('paideia_google_key') || null,
    preferredProvider: localStorage.getItem('paideia_preferred_provider') || null,
    model: localStorage.getItem('paideia_model') || null,
    selectedFile: null,
    isLoading: false,
    chatHistory: JSON.parse(localStorage.getItem('paideia_chat_history') || '[]')
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
    btnUpload: document.getElementById('btn-upload'),
    fileInput: document.getElementById('file-input'),
    uploadPreview: document.getElementById('upload-preview'),
    fileName: document.getElementById('file-name'),
    btnCancelUpload: document.getElementById('btn-cancel-upload'),
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
    currentUserDisplay: document.getElementById('current-user-display')
};

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
        
        div.innerHTML = `
            <div class="message-avatar">🎓</div>
            <div class="message-content">
                <p>${formatMessage(message.content)}</p>
                ${sourcesHtml}
            </div>
        `;
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

    const response = await fetch(`${API_BASE_URL}/query/`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            question,
            session_id: state.sessionId,
            student_id: state.studentId,
            use_cache: true
        })
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

    const response = await fetch(`${API_BASE_URL}/ingest/file`, {
        method: 'POST',
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
    
    if (!message && !state.selectedFile) {
        return;
    }

    if (state.isLoading) {
        return;
    }

    state.isLoading = true;
    elements.btnSend.disabled = true;

    try {
        // Si hay un archivo seleccionado, subirlo primero
        if (state.selectedFile) {
            addMessage({
                id: generateMessageId(),
                role: 'system',
                content: `Subiendo archivo: ${state.selectedFile.name}...`,
                persist: true
            });

            try {
                const uploadResult = await uploadFile(state.selectedFile);
                addMessage({
                    id: generateMessageId(),
                    role: 'system',
                    content: `Archivo "${state.selectedFile.name}" procesado correctamente. ${uploadResult.chunks_created || ''} chunks creados.`,
                    persist: true
                });
            } catch (error) {
                addMessage({
                    id: generateMessageId(),
                    role: 'system',
                    content: `Error subiendo archivo: ${error.message}`,
                    isError: true
                });
            }

            // Limpiar selección de archivo
            clearFileSelection();
        }

        // Si hay un mensaje, enviarlo
        if (message) {
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
        }

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

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        state.selectedFile = file;
        elements.fileName.textContent = `📄 ${file.name}`;
        elements.uploadPreview.classList.remove('hidden');
    }
}

function clearFileSelection() {
    state.selectedFile = null;
    elements.fileInput.value = '';
    elements.uploadPreview.classList.add('hidden');
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

function openUserModal() {
    elements.userModal.classList.remove('hidden');
    elements.studentIdInput.value = state.studentId || '';
    elements.openaiKeyInput.value = state.openaiKey || '';
    elements.googleKeyInput.value = state.googleKey || '';
    elements.preferredProviderSelect.value = state.preferredProvider || '';
    elements.modelSelect.value = state.model || '';
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
    if (openaiKey) configuredItems.push('OpenAI ✓');
    if (googleKey) configuredItems.push('Google ✓');
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
    
    updateUserDisplay();
    
    addMessage({
        id: generateMessageId(),
        role: 'system',
        content: 'Configuración limpiada - Usando configuración del servidor'
    });
    
    closeUserModal();
}

function updateUserDisplay() {
    const hasConfig = state.studentId || state.openaiKey || state.googleKey;
    
    if (hasConfig) {
        const parts = [];
        if (state.studentId) parts.push(state.studentId);
        if (state.openaiKey) parts.push('🤖');
        if (state.googleKey) parts.push('💎');
        if (state.model) parts.push(`[${state.model}]`);
        
        elements.currentUserDisplay.textContent = parts.join(' ');
        elements.currentUserDisplay.classList.add('identified');
    } else {
        elements.currentUserDisplay.textContent = 'Sin configurar';
        elements.currentUserDisplay.classList.remove('identified');
    }
}

// ==========================================
// Inicialización
// ==========================================

function init() {
    // Actualizar display de usuario
    updateUserDisplay();

    // Cargar historial de chat
    loadChatHistory();

    // Event listeners - Envío de mensajes
    elements.btnSend.addEventListener('click', handleSendMessage);
    elements.messageInput.addEventListener('keypress', handleKeyPress);
    elements.messageInput.addEventListener('input', autoResizeTextarea);

    // Event listeners - Subida de archivos
    elements.btnUpload.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.btnCancelUpload.addEventListener('click', clearFileSelection);

    // Event listeners - Modal de usuario
    elements.btnUserConfig.addEventListener('click', openUserModal);
    elements.btnCloseModal.addEventListener('click', closeUserModal);
    elements.btnSaveUser.addEventListener('click', saveUser);
    elements.btnClearUser.addEventListener('click', clearUser);

    // Cerrar modal al hacer clic fuera
    elements.userModal.addEventListener('click', (e) => {
        if (e.target === elements.userModal) {
            closeUserModal();
        }
    });

    // Cerrar modal con Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !elements.userModal.classList.contains('hidden')) {
            closeUserModal();
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
