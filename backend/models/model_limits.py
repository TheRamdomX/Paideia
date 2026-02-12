"""
model_limits.py
Configuración de límites de tokens por modelo.
Define la ventana de contexto y otros límites para cada modelo soportado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configuración de un modelo específico."""
    name: str                    # Nombre legible
    api_name: str               # Nombre en la API
    provider: str               # 'openai' o 'google'
    context_window: int         # Ventana de contexto en tokens
    max_output_tokens: int      # Máximo de tokens de salida
    default_max_output: int     # Valor por defecto para max_tokens
    supports_vision: bool = False
    supports_functions: bool = True
    
    def get_safe_context_limit(self, output_tokens: int = 0) -> int:
        """
        Calcula el límite seguro de contexto dejando espacio para la respuesta.
        
        Args:
            output_tokens: Tokens reservados para la respuesta
            
        Returns:
            Tokens disponibles para contexto
        """
        reserved = output_tokens or self.default_max_output
        # Dejar un margen de seguridad del 5%
        margin = int(self.context_window * 0.05)
        return self.context_window - reserved - margin


# ==========================================
# Configuración de Modelos OpenAI
# ==========================================

OPENAI_MODELS: Dict[str, ModelConfig] = {
    # GPT-5.x Series (400K context)
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        api_name="gpt-5.2",
        provider="openai",
        context_window=400_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-5": ModelConfig(
        name="GPT-5",
        api_name="gpt-5",
        provider="openai",
        context_window=400_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-5-mini": ModelConfig(
        name="GPT-5 Mini",
        api_name="gpt-5-mini",
        provider="openai",
        context_window=400_000,
        max_output_tokens=16_384,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-5-nano": ModelConfig(
        name="GPT-5 Nano",
        api_name="gpt-5-nano",
        provider="openai",
        context_window=400_000,
        max_output_tokens=8_192,
        default_max_output=2_048,
        supports_vision=False,
    ),
    
    # GPT-4.1 Series (1M context)
    "gpt-4.1": ModelConfig(
        name="GPT-4.1",
        api_name="gpt-4.1",
        provider="openai",
        context_window=1_047_576,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-4.1-mini": ModelConfig(
        name="GPT-4.1 Mini",
        api_name="gpt-4.1-mini",
        provider="openai",
        context_window=1_047_576,
        max_output_tokens=16_384,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-4.1-nano": ModelConfig(
        name="GPT-4.1 Nano",
        api_name="gpt-4.1-nano",
        provider="openai",
        context_window=1_047_576,
        max_output_tokens=8_192,
        default_max_output=2_048,
        supports_vision=False,
    ),
    
    # GPT-4o Series (128K context)
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        api_name="gpt-4o",
        provider="openai",
        context_window=128_000,
        max_output_tokens=16_384,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        api_name="gpt-4o-mini",
        provider="openai",
        context_window=128_000,
        max_output_tokens=16_384,
        default_max_output=4_096,
        supports_vision=True,
    ),
    
    # GPT-3.5 Series (16K context)
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        api_name="gpt-3.5-turbo",
        provider="openai",
        context_window=16_385,
        max_output_tokens=4_096,
        default_max_output=2_048,
        supports_vision=False,
    ),
}


# ==========================================
# Configuración de Modelos Google (Gemini)
# ==========================================

GOOGLE_MODELS: Dict[str, ModelConfig] = {
    # Gemini 3 Series (1M context)
    "gemini-3-pro": ModelConfig(
        name="Gemini 3 Pro",
        api_name="gemini-3-pro-preview",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gemini-3-flash": ModelConfig(
        name="Gemini 3 Flash",
        api_name="gemini-3-flash-preview",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=16_384,
        default_max_output=4_096,
        supports_vision=True,
    ),

    # Gemini 2.5 Pro
    "gemini-2.5-pro": ModelConfig(
        name="Gemini 2.5 Pro",
        api_name="gemini-2.5-pro",
        provider="google",
        context_window=250_000_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    # Gemini 2.5 Series (1M context)
    "gemini-2.5-flash": ModelConfig(
        name="Gemini 2.5 Flash",
        api_name="gemini-2.5-flash",
        provider="google",
        context_window=250_000_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),
    # Gemini 2.5 Flash Lite
    "gemini-2.5-flash-lite": ModelConfig(
        name="Gemini 2.5 Flash Lite",
        api_name="gemini-2.5-flash-lite",
        provider="google",
        context_window=250_000_000,
        max_output_tokens=32_768,
        default_max_output=4_096,
        supports_vision=True,
    ),

    # Gemini 2.0 pro
    "gemini-2.0-pro": ModelConfig(
        name="Gemini 2.0 Pro",
        api_name="gemini-2.0-pro",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=8_192,
        default_max_output=4_096,
        supports_vision=True,
    ),
    # Gemini 2.0 Series (1M context)
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        api_name="gemini-2.0-flash",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=8_192,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gemini-2.0-flash-lite": ModelConfig(
        name="Gemini 2.0 Flash Lite",
        api_name="gemini-2.0-flash-lite",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=8_192,
        default_max_output=2_048,
        supports_vision=True,
    ),
    
    # Legacy models para compatibilidad
    "gemini-pro": ModelConfig(
        name="Gemini Pro",
        api_name="gemini-pro",
        provider="google",
        context_window=32_768,
        max_output_tokens=8_192,
        default_max_output=2_048,
        supports_vision=False,
    ),
    "gemini-1.5-pro": ModelConfig(
        name="Gemini 1.5 Pro",
        api_name="gemini-1.5-pro",
        provider="google",
        context_window=2_000_000,
        max_output_tokens=8_192,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gemini-1.5-flash": ModelConfig(
        name="Gemini 1.5 Flash",
        api_name="gemini-1.5-flash",
        provider="google",
        context_window=1_000_000,
        max_output_tokens=8_192,
        default_max_output=4_096,
        supports_vision=True,
    ),
    "gemma": ModelConfig(
        name="Gemma 3",
        api_name="gemma-3-27b-it",
        provider="google",
        context_window=12_000,
        max_output_tokens=4_096,
        default_max_output=2_048,
        supports_vision=True,
    )
}


# ==========================================
# Registro unificado de todos los modelos
# ==========================================

ALL_MODELS: Dict[str, ModelConfig] = {
    **OPENAI_MODELS,
    **GOOGLE_MODELS,
}


# ==========================================
# Funciones de utilidad
# ==========================================

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Obtiene la configuración de un modelo por su nombre.
    
    Args:
        model_name: Nombre del modelo (api_name)
        
    Returns:
        ModelConfig o None si no existe
    """
    return ALL_MODELS.get(model_name)


def get_context_window(model_name: str, default: int = 128_000) -> int:
    """
    Obtiene la ventana de contexto de un modelo.
    
    Args:
        model_name: Nombre del modelo
        default: Valor por defecto si el modelo no existe
        
    Returns:
        Ventana de contexto en tokens
    """
    config = get_model_config(model_name)
    return config.context_window if config else default


def get_max_output_tokens(model_name: str, default: int = 4_096) -> int:
    """
    Obtiene el máximo de tokens de salida de un modelo.
    
    Args:
        model_name: Nombre del modelo
        default: Valor por defecto si el modelo no existe
        
    Returns:
        Máximo de tokens de salida
    """
    config = get_model_config(model_name)
    return config.max_output_tokens if config else default


def get_safe_context_limit(
    model_name: str,
    output_tokens: int = 0,
    default_context: int = 128_000,
) -> int:
    """
    Calcula el límite seguro de contexto para un modelo.
    
    Args:
        model_name: Nombre del modelo
        output_tokens: Tokens reservados para respuesta
        default_context: Contexto por defecto si modelo no existe
        
    Returns:
        Tokens disponibles para contexto
    """
    config = get_model_config(model_name)
    if config:
        return config.get_safe_context_limit(output_tokens)
    
    # Fallback para modelos desconocidos
    reserved = output_tokens or 4_096
    margin = int(default_context * 0.05)
    return default_context - reserved - margin


def list_models_by_provider(provider: str) -> Dict[str, ModelConfig]:
    """
    Lista modelos filtrados por proveedor.
    
    Args:
        provider: 'openai' o 'google'
        
    Returns:
        Diccionario de modelos del proveedor
    """
    return {
        name: config
        for name, config in ALL_MODELS.items()
        if config.provider == provider
    }


def get_models_summary() -> Dict[str, Dict[str, any]]:
    """
    Obtiene un resumen de todos los modelos disponibles.
    
    Returns:
        Diccionario con información resumida de cada modelo
    """
    return {
        name: {
            "name": config.name,
            "provider": config.provider,
            "context_window": config.context_window,
            "max_output_tokens": config.max_output_tokens,
            "supports_vision": config.supports_vision,
        }
        for name, config in ALL_MODELS.items()
    }


def validate_token_count(
    model_name: str,
    input_tokens: int,
    output_tokens: int = 0,
) -> tuple[bool, str]:
    """
    Valida que los tokens no excedan los límites del modelo.
    
    Args:
        model_name: Nombre del modelo
        input_tokens: Tokens de entrada
        output_tokens: Tokens de salida esperados
        
    Returns:
        Tupla (es_válido, mensaje)
    """
    config = get_model_config(model_name)
    
    if not config:
        return True, f"Modelo '{model_name}' no encontrado, usando límites por defecto"
    
    total_tokens = input_tokens + output_tokens
    
    if total_tokens > config.context_window:
        return False, (
            f"Total de tokens ({total_tokens:,}) excede la ventana de contexto "
            f"del modelo {config.name} ({config.context_window:,} tokens)"
        )
    
    if output_tokens > config.max_output_tokens:
        return False, (
            f"Tokens de salida ({output_tokens:,}) exceden el máximo permitido "
            f"para {config.name} ({config.max_output_tokens:,} tokens)"
        )
    
    return True, "OK"


# ==========================================
# Configuración por defecto según proveedor
# ==========================================

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}


def get_default_model(provider: str) -> str:
    """
    Obtiene el modelo por defecto para un proveedor.
    
    Args:
        provider: 'openai' o 'google'
        
    Returns:
        Nombre del modelo por defecto
    """
    return DEFAULT_MODELS.get(provider, "gpt-4o-mini")
