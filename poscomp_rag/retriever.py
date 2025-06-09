
import random
from typing import List, Dict

from poscomp_rag.core.embeddings import GeradorEmbeddings
from poscomp_rag.core.utils import normalizar_texto
from chromadb.api.models.Collection import Collection


class BuscadorQuestoes:
    def __init__(self, gerador_embeddings: GeradorEmbeddings):
        self.gerador_embeddings = gerador_embeddings

    def buscar_similares(
        self,
        colecao: Collection,
        consulta: str,
        top_k: int = 5,
        limiar: float = 0.4
    ) -> List[Dict]:
        """
        Busca questões similares em uma coleção do ChromaDB, aplicando
        heurística gulosa por faixa de similaridade para selecionar as melhores.
        """
        try:
            texto_normalizado = normalizar_texto(consulta)
            embedding = self.gerador_embeddings.gerar(texto_normalizado)

            resultados = colecao.query(
                query_embeddings=[embedding],
                n_results=15  # mais do que top_k para diversidade
            )

            documentos = resultados.get('documents', [[]])[0]
            metadados = resultados.get('metadatas', [[]])[0]
            distancias = resultados.get('distances', [[]])[0]

            questoes_similares = []
            for doc, meta, dist in zip(documentos, metadados, distancias):
                similaridade = 1 - dist
                if similaridade >= limiar:
                    questoes_similares.append({
                        "id": meta.get("id", ""),
                        "enunciado": meta.get("enunciado", ""),
                        "alternativas": meta.get("alternativas", "").split("\n"),
                        "area_conhecimento": meta.get("area_conhecimento", ""),
                        "area": meta.get("area", ""),
                        "subarea": meta.get("subarea", ""),
                        "gabarito": meta.get("gabarito", ""),
                        "atributo_rag": meta.get("atributo_rag", ""),
                        "similaridade": round(similaridade, 3)
                    })

            return self._selecionar_guloso(questoes_similares, top_k)

        except Exception as e:
            print(f"⚠️ Erro na busca de questões similares: {e}")
            return []

    def _selecionar_guloso(self, questoes: List[Dict], top_k: int) -> List[Dict]:
        """
        Aplica heurística gulosa com agrupamento por faixas de similaridade
        e embaralhamento para diversidade dentro das faixas.
        """
        if not questoes:
            return []

        # Agrupamento por faixa de similaridade (ex: 0.9, 0.8, ...)
        faixas = {}
        for q in questoes:
            faixa = round(q["similaridade"], 1)
            faixas.setdefault(faixa, []).append(q)

        # Processamento guloso por ordem decrescente de similaridade
        questoes_finais = []
        for faixa in sorted(faixas.keys(), reverse=True):
            grupo = faixas[faixa]
            random.shuffle(grupo)  # diversidade

            vagas_restantes = top_k - len(questoes_finais)
            if vagas_restantes <= 0:
                break

            if len(grupo) <= vagas_restantes:
                questoes_finais.extend(grupo)
            else:
                questoes_finais.extend(grupo[:vagas_restantes])

        return questoes_finais[:top_k]
