import unicodedata
import re
from typing import Optional, Tuple


def normalizar_texto(texto: str) -> str:
    """
    Remove acentos e converte o texto para minúsculas, sem espaços extras.
    """
    nfkd = unicodedata.normalize("NFKD", texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def parse_resposta_classificacao(resposta: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai a área e subárea a partir da resposta do LLM.
    Espera o formato:
    Área: [área]
    Subárea: [subárea]
    """
    area_match = re.search(r"Área:\s*(.*)", resposta, re.IGNORECASE)
    subarea_match = re.search(r"Subárea:\s*(.*)", resposta, re.IGNORECASE)

    area = area_match.group(1).strip() if area_match else None
    subarea = subarea_match.group(1).strip() if subarea_match else None

    return area, subarea
