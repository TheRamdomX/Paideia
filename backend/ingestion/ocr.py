"""
ocr.py
Módulo de OCR para procesamiento de documentos escaneados e imágenes.

Este módulo es independiente del pipeline de RAG, permitiendo:
- Cambiar motores OCR sin tocar RAG
- Usar OCR confidence
- Reprocesar sin LLM
- Versionar raw_text
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ==========================================
# Configuración
# ==========================================

OCR_CONFIG = {
    "default_language": "spa",
    "fallback_languages": ["spa", "eng"],
    "dpi": 200,
    "text_threshold": 100,  # Caracteres mínimos para considerar que un PDF tiene texto
}


@dataclass
class OCRResult:
    """Resultado del procesamiento OCR."""
    
    text: str = ""
    was_ocr_applied: bool = False
    pages_processed: int = 0
    confidence: Optional[float] = None
    duration_seconds: float = 0.0
    language: str = "spa"
    errors: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "was_ocr_applied": self.was_ocr_applied,
            "pages_processed": self.pages_processed,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "text_length": len(self.text),
            "errors": self.errors,
        }


# ==========================================
# Detectores de Tipo
# ==========================================

def is_pdf(path: str) -> bool:
    """
    Detecta si un archivo es PDF por extensión y mimetype.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        True si es PDF
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    
    mime_type, _ = mimetypes.guess_type(path)
    
    return suffix == ".pdf" or (mime_type and "pdf" in mime_type.lower())


def is_image(path: str) -> bool:
    """
    Detecta si un archivo es una imagen soportada.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        True si es imagen
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    
    mime_type, _ = mimetypes.guess_type(path)
    
    return suffix in image_extensions or (mime_type and mime_type.startswith("image/"))


def is_text_file(path: str) -> bool:
    """
    Detecta si un archivo es de texto plano.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        True si es archivo de texto
    """
    text_extensions = {".txt", ".text", ".md", ".markdown", ".rst", ".html", ".htm"}
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    
    return suffix in text_extensions


# ==========================================
# Análisis de PDF
# ==========================================

def pdf_has_text(path: str, threshold: int = None) -> bool:
    """
    Detecta si un PDF tiene texto nativo (no es escaneado).
    
    Args:
        path: Ruta al PDF
        threshold: Mínimo de caracteres para considerar que tiene texto
        
    Returns:
        True si el PDF tiene texto extraíble
    """
    threshold = threshold or OCR_CONFIG["text_threshold"]
    
    try:
        reader = PdfReader(path)
        text = ""
        
        # Revisar solo las primeras 2 páginas para eficiencia
        pages_to_check = min(2, len(reader.pages))
        
        for i in range(pages_to_check):
            page_text = reader.pages[i].extract_text() or ""
            text += page_text
            
            # Si ya superamos el threshold, no necesitamos seguir
            if len(text.strip()) > threshold:
                return True
        
        return len(text.strip()) > threshold
        
    except Exception as e:
        logger.warning(f"Error verificando texto en PDF {path}: {e}")
        return False


def extract_native_pdf_text(path: str) -> str:
    """
    Extrae texto nativo de un PDF digital.
    
    Args:
        path: Ruta al PDF
        
    Returns:
        Texto extraído
    """
    try:
        reader = PdfReader(path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        logger.error(f"Error extrayendo texto de PDF {path}: {e}")
        raise RuntimeError(f"Error extrayendo texto de PDF: {e}") from e


def get_pdf_page_count(path: str) -> int:
    """
    Obtiene el número de páginas de un PDF.
    
    Args:
        path: Ruta al PDF
        
    Returns:
        Número de páginas
    """
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception:
        return 0


# ==========================================
# Rasterización
# ==========================================

def rasterize_pdf(path: str, dpi: int = None) -> List[Image.Image]:
    """
    Convierte un PDF a imágenes (una por página).
    
    Args:
        path: Ruta al PDF
        dpi: Resolución de las imágenes
        
    Returns:
        Lista de imágenes PIL
        
    Note:
        Requiere poppler instalado en el sistema.
    """
    try:
        from pdf2image import convert_from_path
        
        dpi = dpi or OCR_CONFIG["dpi"]
        images = convert_from_path(path, dpi=dpi)
        
        logger.info(f"PDF rasterizado: {len(images)} páginas a {dpi} DPI")
        return images
        
    except ImportError:
        raise ImportError(
            "pdf2image no está instalado. Ejecuta: pip install pdf2image"
        )
    except Exception as e:
        logger.error(f"Error rasterizando PDF {path}: {e}")
        raise RuntimeError(f"Error rasterizando PDF: {e}") from e


# ==========================================
# OCR con Tesseract
# ==========================================

def ocr_image(
    image: Image.Image,
    lang: str = None,
    config: str = ""
) -> Tuple[str, Optional[float]]:
    """
    Aplica OCR a una imagen usando Tesseract.
    
    Args:
        image: Imagen PIL
        lang: Idioma(s) para OCR (ej: "spa", "spa+eng")
        config: Configuración adicional de Tesseract
        
    Returns:
        Tupla (texto_extraído, confianza_promedio)
    """
    try:
        import pytesseract
        
        lang = lang or OCR_CONFIG["default_language"]
        
        # Extraer texto
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        
        # Intentar obtener datos de confianza
        try:
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
        except Exception:
            avg_confidence = None
        
        return text, avg_confidence
        
    except ImportError:
        raise ImportError(
            "pytesseract no está instalado. Ejecuta: pip install pytesseract"
        )
    except Exception as e:
        logger.error(f"Error en OCR: {e}")
        raise RuntimeError(f"Error en OCR: {e}") from e


def ocr_image_from_path(path: str, lang: str = None) -> Tuple[str, Optional[float]]:
    """
    Aplica OCR a una imagen desde su ruta.
    
    Args:
        path: Ruta a la imagen
        lang: Idioma para OCR
        
    Returns:
        Tupla (texto_extraído, confianza_promedio)
    """
    image = Image.open(path)
    return ocr_image(image, lang=lang)


# ==========================================
# OCR Completo
# ==========================================

def run_ocr(path: str, lang: str = None) -> OCRResult:
    """
    Ejecuta OCR completo en un archivo PDF o imagen.
    
    Esta es la función principal para OCR.
    
    Args:
        path: Ruta al archivo
        lang: Idioma para OCR
        
    Returns:
        OCRResult con texto y metadatos
    """
    start_time = time()
    result = OCRResult(language=lang or OCR_CONFIG["default_language"])
    
    try:
        if is_pdf(path):
            # Rasterizar PDF
            images = rasterize_pdf(path)
            result.pages_processed = len(images)
            
            # OCR cada página
            text_parts = []
            confidences = []
            
            for i, img in enumerate(images):
                logger.debug(f"Procesando página {i+1}/{len(images)}")
                page_text, page_conf = ocr_image(img, lang=result.language)
                text_parts.append(page_text)
                
                if page_conf is not None:
                    confidences.append(page_conf)
            
            result.text = "\n\n".join(text_parts)
            result.was_ocr_applied = True
            
            if confidences:
                result.confidence = sum(confidences) / len(confidences)
                
        elif is_image(path):
            # OCR directo en imagen
            text, confidence = ocr_image_from_path(path, lang=result.language)
            result.text = text
            result.confidence = confidence
            result.pages_processed = 1
            result.was_ocr_applied = True
            
        else:
            result.errors.append(f"Tipo de archivo no soportado para OCR: {path}")
            
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"Error en OCR para {path}: {e}")
    
    result.duration_seconds = time() - start_time
    
    logger.info(
        f"OCR completado: applied={result.was_ocr_applied}, "
        f"pages={result.pages_processed}, "
        f"duration={result.duration_seconds:.2f}s, "
        f"confidence={result.confidence}"
    )
    
    return result


# ==========================================
# Funciones de Utilidad
# ==========================================

def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Lee un archivo de texto.
    
    Args:
        path: Ruta al archivo
        encoding: Codificación del archivo
        
    Returns:
        Contenido del archivo
    """
    try:
        return Path(path).read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Intentar con latin-1 como fallback
        return Path(path).read_text(encoding="latin-1")


def check_tesseract_installed() -> Tuple[bool, str]:
    """
    Verifica si Tesseract está instalado y disponible.
    
    Returns:
        Tupla (está_instalado, mensaje)
    """
    try:
        import pytesseract
        
        # Intentar obtener la versión
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} disponible"
        
    except ImportError:
        return False, "pytesseract no está instalado"
    except Exception as e:
        return False, f"Tesseract no disponible: {e}"


def get_available_languages() -> List[str]:
    """
    Obtiene los idiomas disponibles en Tesseract.
    
    Returns:
        Lista de códigos de idioma
    """
    try:
        import pytesseract
        
        return pytesseract.get_languages()
        
    except Exception as e:
        logger.warning(f"Error obteniendo idiomas: {e}")
        return []
