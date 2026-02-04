"""
llm.py
Abstracción sobre OpenAI GPT / Google Gemini.
Soporta múltiples proveedores configurables por variable de entorno.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from google.genai import types

from backend.settings import (
    LLMProvider,
    get_model_config,
    get_model_config_with_user_keys,
    get_settings,
)


# ==========================================
# Interfaz Base para LLM
# ==========================================

class BaseLLM(ABC):
    """Interfaz abstracta para proveedores de LLM."""
    
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Genera una respuesta dada un prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Genera una respuesta en streaming."""
        pass
    
    @abstractmethod
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Genera respuesta a partir de lista de mensajes."""
        pass


# ==========================================
# Implementación OpenAI
# ==========================================

class OpenAILLM(BaseLLM):
    """Implementación de LLM usando OpenAI GPT."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization del cliente OpenAI."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Genera una respuesta usando OpenAI.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt de sistema opcional
            **kwargs: Argumentos adicionales para la API
            
        Returns:
            Texto generado
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content or ""
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Genera respuesta en streaming usando OpenAI.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt de sistema opcional
            
        Yields:
            Chunks de texto generado
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Genera respuesta a partir de mensajes.
        
        Args:
            messages: Lista de mensajes [{role, content}]
            
        Returns:
            Texto generado
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content or ""


# ==========================================
# Implementación Google Gemini
# ==========================================

class GoogleLLM(BaseLLM):
    """Implementación de LLM usando Google Gemini (google-genai)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization del cliente Google GenAI."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Genera una respuesta usando Google Gemini.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt de sistema opcional
            **kwargs: Argumentos adicionales
            
        Returns:
            Texto generado
        """
        try:            
            # Construir contenido
            contents = []
            if system_prompt:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"System: {system_prompt}")]
                ))
            contents.append(types.Content(
                role="user", 
                parts=[types.Part(text=prompt)]
            ))
            
            # Configuración de generación
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            # Generar respuesta (la nueva API es async-friendly)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
            )
            
            return response.text if response.text else ""
            
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "404" in error_msg or "could not find model" in error_msg:
                raise ValueError(
                    f"Modelo '{self.model}' no encontrado. "
                    f"Modelos válidos: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro, gemma-3-27b-it"
                ) from e
            elif "api key" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg:
                raise ValueError(
                    "API key de Google inválida o sin permisos. "
                    "Verifica tu key en https://aistudio.google.com/apikey"
                ) from e
            else:
                raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Genera respuesta en streaming usando Gemini.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt de sistema opcional
            
        Yields:
            Chunks de texto generado
        """        
        # Construir contenido
        contents = []
        if system_prompt:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"System: {system_prompt}")]
            ))
        contents.append(types.Content(
            role="user", 
            parts=[types.Part(text=prompt)]
        ))
        
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        loop = asyncio.get_event_loop()
        
        def stream_generate():
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            )
        
        response = await loop.run_in_executor(None, stream_generate)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Genera respuesta a partir de mensajes.
        Convierte formato ChatML a formato Gemini.
        
        Args:
            messages: Lista de mensajes [{role, content}]
            
        Returns:
            Texto generado
        """        
        # Convertir mensajes a formato Gemini
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # System prompt como mensaje de usuario con prefijo
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"System: {content}")]
                ))
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                ))
        
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
        )
        
        return response.text if response.text else ""


# ==========================================
# Factory y Funciones de Conveniencia
# ==========================================

_llm_instance: Optional[BaseLLM] = None


def select_model(
    provider: Optional[LLMProvider] = None,
    **kwargs
) -> BaseLLM:
    """
    Selecciona y configura el modelo LLM según el proveedor.
    
    Args:
        provider: Proveedor de LLM (usa config si no se especifica)
        **kwargs: Argumentos adicionales para el modelo
        
    Returns:
        Instancia de LLM configurada
    """
    config = get_model_config()
    
    if provider is None:
        provider = config["provider"]
    
    # Obtener parámetros
    model = kwargs.get("model", config["model"])
    api_key = kwargs.get("api_key", config["api_key"])
    temperature = kwargs.get("temperature", config["temperature"])
    max_tokens = kwargs.get("max_tokens", config["max_tokens"])
    
    if not api_key:
        raise ValueError(f"API key no configurada para proveedor: {provider}")
    
    if provider == LLMProvider.OPENAI:
        return OpenAILLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == LLMProvider.GOOGLE:
        return GoogleLLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


def get_llm() -> BaseLLM:
    """
    Obtiene la instancia singleton del LLM.
    
    Returns:
        Instancia de LLM configurada
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = select_model()
    
    return _llm_instance


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Genera texto usando el LLM configurado.
    
    Args:
        prompt: Prompt del usuario
        system_prompt: Prompt de sistema opcional
        **kwargs: Argumentos adicionales
        
    Returns:
        Texto generado
    """
    llm = get_llm()
    return await llm.generate(prompt, system_prompt, **kwargs)


async def generate_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Genera texto en streaming usando el LLM configurado.
    
    Args:
        prompt: Prompt del usuario
        system_prompt: Prompt de sistema opcional
        
    Yields:
        Chunks de texto
    """
    llm = get_llm()
    async for chunk in llm.generate_stream(prompt, system_prompt, **kwargs):
        yield chunk


async def handle_fallback(
    prompt: str,
    system_prompt: Optional[str] = None,
    primary_provider: Optional[LLMProvider] = None,
    fallback_provider: Optional[LLMProvider] = None,
    **kwargs
) -> str:
    """
    Genera texto con fallback a otro proveedor si falla.
    
    Args:
        prompt: Prompt del usuario
        system_prompt: Prompt de sistema opcional
        primary_provider: Proveedor primario
        fallback_provider: Proveedor de respaldo
        
    Returns:
        Texto generado
    """
    settings = get_settings()
    
    # Determinar proveedores
    if primary_provider is None:
        primary_provider = settings.llm_provider
    
    if fallback_provider is None:
        # Usar el otro proveedor como fallback
        fallback_provider = (
            LLMProvider.GOOGLE
            if primary_provider == LLMProvider.OPENAI
            else LLMProvider.OPENAI
        )
    
    try:
        # Intentar con proveedor primario
        primary_llm = select_model(primary_provider)
        return await primary_llm.generate(prompt, system_prompt, **kwargs)
        
    except Exception as primary_error:
        print(f"Error con {primary_provider}: {primary_error}")
        
        try:
            # Intentar con fallback
            fallback_llm = select_model(fallback_provider)
            return await fallback_llm.generate(prompt, system_prompt, **kwargs)
            
        except Exception as fallback_error:
            raise RuntimeError(
                f"Error con ambos proveedores. "
                f"Primary ({primary_provider}): {primary_error}. "
                f"Fallback ({fallback_provider}): {fallback_error}"
            )


def reset_llm() -> None:
    """Resetea la instancia del LLM (útil para testing)."""
    global _llm_instance
    _llm_instance = None


def get_llm_with_user_keys(
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
) -> BaseLLM:
    """
    Obtiene una instancia de LLM usando configuración del usuario o servidor.
    
    MODO EXCLUSIVO:
    - Si el usuario envía API key → usa SOLO config del usuario
    - Si NO envía API key → usa SOLO config del servidor
    
    Args:
        user_openai_key: API key de OpenAI del usuario
        user_google_key: API key de Google del usuario
        preferred_provider: Proveedor preferido ('openai' o 'google')
        user_model: Modelo específico (ej: 'gpt-4', 'gpt-4o-mini', 'gemini-pro')
        
    Returns:
        Instancia de LLM configurada
        
    Raises:
        ValueError: Si no hay API key disponible
    """
    config = get_model_config_with_user_keys(
        user_openai_key=user_openai_key,
        user_google_key=user_google_key,
        preferred_provider=preferred_provider,
        user_model=user_model,
    )
    
    if not config.get("has_api_key", False):
        raise ValueError(
            "No hay API key disponible. Configure OPENAI_API_KEY o GOOGLE_API_KEY "
            "en el servidor, o envíe su API key en los headers X-OpenAI-Key o X-Google-Key"
        )
    
    return select_model(
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


async def generate_with_user_keys(
    prompt: str,
    system_prompt: Optional[str] = None,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
    **kwargs
) -> str:
    """
    Genera texto usando configuración del usuario o servidor.
    
    Args:
        prompt: Prompt del usuario
        system_prompt: Prompt de sistema opcional
        user_openai_key: API key de OpenAI del usuario
        user_google_key: API key de Google del usuario
        preferred_provider: Proveedor preferido
        user_model: Modelo específico a usar
        **kwargs: Argumentos adicionales
        
    Returns:
        Texto generado
    """
    llm = get_llm_with_user_keys(
        user_openai_key=user_openai_key,
        user_google_key=user_google_key,
        preferred_provider=preferred_provider,
        user_model=user_model,
    )
    return await llm.generate(prompt, system_prompt, **kwargs)
