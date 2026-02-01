"""
text.py
Helpers de limpieza, tokenización y scoring.
"""

from __future__ import annotations

import re
from typing import List, Optional

import tiktoken


# ==========================================
# Tokenización
# ==========================================

# Cache del tokenizer
_tokenizer: Optional[tiktoken.Encoding] = None


def get_tokenizer(model: str = "gpt-4") -> tiktoken.Encoding:
    """
    Obtiene el tokenizer para un modelo.
    
    Args:
        model: Nombre del modelo (default: gpt-4)
        
    Returns:
        Tokenizer de tiktoken
    """
    global _tokenizer
    
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback a cl100k_base (usado por GPT-4, GPT-3.5-turbo)
            _tokenizer = tiktoken.get_encoding("cl100k_base")
    
    return _tokenizer


def token_count(text: str, model: str = "gpt-4") -> int:
    """
    Cuenta el número de tokens en un texto.
    
    Args:
        text: Texto a contar
        model: Modelo para el tokenizer
        
    Returns:
        Número de tokens
    """
    if not text:
        return 0
    
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def tokenize(text: str, model: str = "gpt-4") -> List[int]:
    """
    Tokeniza un texto.
    
    Args:
        text: Texto a tokenizar
        model: Modelo para el tokenizer
        
    Returns:
        Lista de token IDs
    """
    if not text:
        return []
    
    tokenizer = get_tokenizer(model)
    return tokenizer.encode(text)


def detokenize(tokens: List[int], model: str = "gpt-4") -> str:
    """
    Convierte tokens de vuelta a texto.
    
    Args:
        tokens: Lista de token IDs
        model: Modelo para el tokenizer
        
    Returns:
        Texto decodificado
    """
    if not tokens:
        return ""
    
    tokenizer = get_tokenizer(model)
    return tokenizer.decode(tokens)


# ==========================================
# Limpieza de Texto
# ==========================================

def clean_markdown(text: str) -> str:
    """
    Limpia y normaliza texto markdown.
    
    Args:
        text: Texto markdown a limpiar
        
    Returns:
        Texto limpio
    """
    if not text:
        return ""
    
    # Eliminar bloques de código pero mantener contenido
    text = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
    
    # Eliminar código inline
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Eliminar imágenes pero mantener alt text
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    
    # Convertir links a solo texto
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Eliminar énfasis pero mantener texto
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)      # Bold alt
    text = re.sub(r'_([^_]+)_', r'\1', text)        # Italic alt
    
    # Eliminar headers pero mantener texto
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Eliminar blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # Normalizar listas
    text = re.sub(r'^[\*\-\+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '• ', text, flags=re.MULTILINE)
    
    # Eliminar líneas horizontales
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Normalizar espacios en blanco
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def clean_whitespace(text: str) -> str:
    """
    Normaliza espacios en blanco.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto con espacios normalizados
    """
    if not text:
        return ""
    
    # Eliminar espacios al inicio/final de líneas
    lines = [line.strip() for line in text.split('\n')]
    
    # Eliminar líneas vacías consecutivas
    result = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                result.append(line)
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False
    
    return '\n'.join(result).strip()


def normalize_text(text: str) -> str:
    """
    Normalización completa de texto.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    if not text:
        return ""
    
    # Normalizar unicode
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # Eliminar caracteres de control excepto newlines y tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar comillas
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalizar guiones
    text = text.replace('–', '-').replace('—', '-')
    
    return clean_whitespace(text)


# ==========================================
# Truncamiento de Contexto
# ==========================================

def truncate_context(
    text: str,
    max_tokens: int,
    strategy: str = "end",
    ellipsis: str = "..."
) -> str:
    """
    Trunca texto a un número máximo de tokens.
    
    Args:
        text: Texto a truncar
        max_tokens: Número máximo de tokens
        strategy: Estrategia de truncamiento:
            - "end": Corta al final
            - "start": Corta al inicio
            - "middle": Corta en el medio
        ellipsis: Texto a usar como indicador de truncamiento
        
    Returns:
        Texto truncado
    """
    if not text:
        return ""
    
    current_tokens = token_count(text)
    
    if current_tokens <= max_tokens:
        return text
    
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    ellipsis_tokens = tokenizer.encode(ellipsis)
    available_tokens = max_tokens - len(ellipsis_tokens)
    
    if available_tokens <= 0:
        return ellipsis
    
    if strategy == "end":
        truncated_tokens = tokens[:available_tokens]
        return tokenizer.decode(truncated_tokens) + ellipsis
    
    elif strategy == "start":
        truncated_tokens = tokens[-available_tokens:]
        return ellipsis + tokenizer.decode(truncated_tokens)
    
    elif strategy == "middle":
        half = available_tokens // 2
        start_tokens = tokens[:half]
        end_tokens = tokens[-(available_tokens - half):]
        return (
            tokenizer.decode(start_tokens) +
            ellipsis +
            tokenizer.decode(end_tokens)
        )
    
    else:
        raise ValueError(f"Estrategia no soportada: {strategy}")


def truncate_by_sentences(
    text: str,
    max_tokens: int,
    min_sentences: int = 1
) -> str:
    """
    Trunca texto manteniendo oraciones completas.
    
    Args:
        text: Texto a truncar
        max_tokens: Número máximo de tokens
        min_sentences: Mínimo de oraciones a mantener
        
    Returns:
        Texto truncado por oraciones
    """
    if not text:
        return ""
    
    if token_count(text) <= max_tokens:
        return text
    
    # Dividir en oraciones
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = []
    current_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = token_count(sentence)
        
        if current_count + sentence_tokens > max_tokens and i >= min_sentences:
            break
        
        result.append(sentence)
        current_count += sentence_tokens
    
    return ' '.join(result)


# ==========================================
# Utilidades de Scoring
# ==========================================

def calculate_overlap(text1: str, text2: str) -> float:
    """
    Calcula el solapamiento de palabras entre dos textos.
    
    Args:
        text1: Primer texto
        text2: Segundo texto
        
    Returns:
        Score de solapamiento [0, 1]
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)  # Jaccard similarity


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extrae palabras clave de un texto (simple).
    
    Args:
        text: Texto de entrada
        max_keywords: Número máximo de keywords
        
    Returns:
        Lista de palabras clave
    """
    if not text:
        return []
    
    # Stopwords básicos en español e inglés
    stopwords = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'de', 'del', 'al', 'a', 'en', 'con', 'por', 'para',
        'que', 'es', 'son', 'fue', 'ser', 'está', 'están',
        'y', 'o', 'pero', 'si', 'no', 'como', 'más', 'menos',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
        'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for',
        'with', 'this', 'that', 'it', 'as', 'by', 'from',
    }
    
    # Extraer palabras
    words = re.findall(r'\b\w{3,}\b', text.lower())
    
    # Filtrar stopwords y contar frecuencias
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Ordenar por frecuencia
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in sorted_words[:max_keywords]]
