from sentence_transformers import SentenceTransformer
from typing import List, Union

from .utils import normalizar_texto


class GeradorEmbeddings:
    def __init__(self, modelo_nome: str = "all-MiniLM-L6-v2"):
        self.modelo = SentenceTransformer(modelo_nome)

    def gerar(self, texto: Union[str, List[str]], normalizar: bool = True) -> Union[List[float], List[List[float]]]:
        if isinstance(texto, list):
            if normalizar:
                texto = [normalizar_texto(t) for t in texto]
        else:
            if normalizar:
                texto = normalizar_texto(texto)

        return self.modelo.encode(texto).tolist()
