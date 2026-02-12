"""
models.py
Endpoints para consultar información de modelos disponibles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.models.model_limits import (
    ALL_MODELS,
    OPENAI_MODELS,
    GOOGLE_MODELS,
    get_model_config,
    get_context_window,
    get_safe_context_limit,
    get_models_summary,
    validate_token_count,
    get_default_model,
)


# ==========================================
# Router
# ==========================================

router = APIRouter(prefix="/models", tags=["Models"])


# ==========================================
# Response Models
# ==========================================

class ModelInfo(BaseModel):
    """Información de un modelo."""
    name: str = Field(..., description="Nombre legible del modelo")
    api_name: str = Field(..., description="Nombre para usar en la API")
    provider: str = Field(..., description="Proveedor (openai/google)")
    context_window: int = Field(..., description="Ventana de contexto en tokens")
    max_output_tokens: int = Field(..., description="Máximo de tokens de salida")
    default_max_output: int = Field(..., description="Valor por defecto para max_tokens")
    supports_vision: bool = Field(default=False, description="Soporta imágenes")
    supports_functions: bool = Field(default=True, description="Soporta function calling")
    safe_context_limit: int = Field(..., description="Límite seguro de contexto")


class ModelsListResponse(BaseModel):
    """Lista de modelos disponibles."""
    models: List[ModelInfo]
    total: int
    providers: List[str]


class TokenValidationRequest(BaseModel):
    """Request para validar tokens."""
    model: str = Field(..., description="Nombre del modelo")
    input_tokens: int = Field(..., ge=0, description="Tokens de entrada")
    output_tokens: int = Field(default=0, ge=0, description="Tokens de salida esperados")


class TokenValidationResponse(BaseModel):
    """Response de validación de tokens."""
    valid: bool
    message: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_window: int
    available_tokens: int


class ContextCalculationResponse(BaseModel):
    """Response de cálculo de contexto disponible."""
    model: str
    context_window: int
    safe_context_limit: int
    output_tokens_reserved: int
    available_for_input: int


# ==========================================
# Endpoints
# ==========================================

@router.get(
    "",
    response_model=ModelsListResponse,
    summary="Listar modelos disponibles",
    description="Lista todos los modelos soportados con sus límites",
)
async def list_models(
    provider: Optional[str] = Query(
        default=None,
        description="Filtrar por proveedor (openai/google)"
    ),
) -> ModelsListResponse:
    """Lista todos los modelos disponibles."""
    
    if provider == "openai":
        models_dict = OPENAI_MODELS
    elif provider == "google":
        models_dict = GOOGLE_MODELS
    else:
        models_dict = ALL_MODELS
    
    models = [
        ModelInfo(
            name=config.name,
            api_name=config.api_name,
            provider=config.provider,
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
            default_max_output=config.default_max_output,
            supports_vision=config.supports_vision,
            supports_functions=config.supports_functions,
            safe_context_limit=config.get_safe_context_limit(),
        )
        for config in models_dict.values()
    ]
    
    providers = list(set(m.provider for m in models))
    
    return ModelsListResponse(
        models=models,
        total=len(models),
        providers=providers,
    )


@router.get(
    "/{model_name}",
    response_model=ModelInfo,
    summary="Obtener información de un modelo",
    description="Obtiene la configuración y límites de un modelo específico",
)
async def get_model(model_name: str) -> ModelInfo:
    """Obtiene información de un modelo específico."""
    config = get_model_config(model_name)
    
    if not config:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no encontrado"
        )
    
    return ModelInfo(
        name=config.name,
        api_name=config.api_name,
        provider=config.provider,
        context_window=config.context_window,
        max_output_tokens=config.max_output_tokens,
        default_max_output=config.default_max_output,
        supports_vision=config.supports_vision,
        supports_functions=config.supports_functions,
        safe_context_limit=config.get_safe_context_limit(),
    )


@router.post(
    "/validate-tokens",
    response_model=TokenValidationResponse,
    summary="Validar tokens",
    description="Valida que los tokens no excedan los límites del modelo",
)
async def validate_tokens(request: TokenValidationRequest) -> TokenValidationResponse:
    """Valida tokens contra los límites del modelo."""
    
    valid, message = validate_token_count(
        model_name=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
    )
    
    context_window = get_context_window(request.model)
    total = request.input_tokens + request.output_tokens
    
    return TokenValidationResponse(
        valid=valid,
        message=message,
        model=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=total,
        context_window=context_window,
        available_tokens=context_window - total,
    )


@router.get(
    "/{model_name}/context",
    response_model=ContextCalculationResponse,
    summary="Calcular contexto disponible",
    description="Calcula cuánto contexto está disponible para un modelo",
)
async def calculate_context(
    model_name: str,
    output_tokens: int = Query(
        default=0,
        ge=0,
        description="Tokens reservados para la respuesta"
    ),
) -> ContextCalculationResponse:
    """Calcula el contexto disponible para un modelo."""
    
    config = get_model_config(model_name)
    
    if not config:
        # Usar valores por defecto
        context_window = 128_000
        safe_limit = get_safe_context_limit(model_name, output_tokens)
    else:
        context_window = config.context_window
        safe_limit = config.get_safe_context_limit(output_tokens)
    
    output_reserved = output_tokens or (config.default_max_output if config else 4096)
    
    return ContextCalculationResponse(
        model=model_name,
        context_window=context_window,
        safe_context_limit=safe_limit,
        output_tokens_reserved=output_reserved,
        available_for_input=safe_limit,
    )


@router.get(
    "/defaults/{provider}",
    summary="Modelo por defecto",
    description="Obtiene el modelo por defecto para un proveedor",
)
async def get_provider_default(provider: str) -> Dict[str, Any]:
    """Obtiene el modelo por defecto para un proveedor."""
    
    if provider not in ["openai", "google"]:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Proveedor '{provider}' no soportado. Use 'openai' o 'google'"
        )
    
    default_model = get_default_model(provider)
    config = get_model_config(default_model)
    
    return {
        "provider": provider,
        "default_model": default_model,
        "context_window": config.context_window if config else None,
        "max_output_tokens": config.max_output_tokens if config else None,
    }
