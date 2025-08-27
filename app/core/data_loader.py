import json
from typing import Dict, Optional

from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .utils import normalizar_texto


class CarregadorBanco:
    def __init__(
        self,
        db_path: str,
        modelo_embedding: SentenceTransformer,
        area_para_colecao: Dict[str, str],
    ):
        self.client_chroma = PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.modelo_embedding = modelo_embedding
        self.area_para_colecao = area_para_colecao

    def criar_ou_obter_colecao(self, nome_colecao: str):
        return self.client_chroma.get_or_create_collection(name=nome_colecao)

    def colecao_precisa_dados(self, nome_colecao: str) -> bool:
        try:
            colecao = self.criar_ou_obter_colecao(nome_colecao)
            return colecao.count() == 0
        except Exception:
            return True

    def carregar_questoes_arquivo(self, caminho_arquivo: str, nome_colecao: Optional[str] = None) -> int:
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                questoes = json.load(f)

            if not questoes:
                print(f"Nenhuma questão encontrada em {caminho_arquivo}")
                return 0

            questoes_inseridas = 0
            for questao in questoes:
                try:
                    id = questao.get("id", "")
                    enunciado = questao.get("enunciado", "")
                    alternativas = questao.get("alternativas", [])
                    area_conhecimento = questao.get("area_conhecimento", "")
                    area = questao.get("area", area_conhecimento)
                    subarea = questao.get("subarea", "")
                    atributo_rag = questao.get("atributo_rag", "")
                    gabarito = questao.get("gabarito", "")

                    if not enunciado or not id:
                        continue

                    colecao_nome = nome_colecao or self._inferir_colecao(atributo_rag)
                    colecao = self.criar_ou_obter_colecao(colecao_nome)

                    texto_embedding = normalizar_texto(atributo_rag)
                    embedding = self.modelo_embedding.encode(texto_embedding).tolist()

                    colecao.add(
                        documents=[atributo_rag],
                        embeddings=[embedding],
                        metadatas=[{
                            "id": id,
                            "enunciado": enunciado,
                            "alternativas": "\n".join(alternativas),
                            "area_conhecimento": area_conhecimento,
                            "area": area,
                            "subarea": subarea,
                            "gabarito": gabarito,
                            "atributo_rag": atributo_rag,
                        }],
                        ids=[id],
                    )
                    questoes_inseridas += 1

                except Exception as e:
                    print(f"⚠️ Erro ao processar questão {questao.get('id', '?')}: {e}")
            print(f"✅ {questoes_inseridas} questões carregadas em '{nome_colecao or colecao_nome}'")
            return questoes_inseridas

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {caminho_arquivo}")
            return 0
        except json.JSONDecodeError:
            print(f"Erro de formato JSON: {caminho_arquivo}")
            return 0
        except Exception as e:
            print(f"Erro inesperado ao carregar {caminho_arquivo}: {e}")
            return 0

    def _inferir_colecao(self, atributo_rag: str) -> str:
        area_principal = atributo_rag.split(" - ")[0] if " - " in atributo_rag else atributo_rag
        return self.area_para_colecao.get(area_principal, "tecnologia_computacao")

    def deletar_colecao(self, nome_colecao: str):
        try:
            self.client_chroma.delete_collection(name=nome_colecao)
            print(f" Coleção '{nome_colecao}' removida")
        except Exception:
            pass  # pode não existir

    def contar_questoes_por_colecao(self) -> Dict[str, int]:
        contagem = {}
        for area, nome_colecao in self.area_para_colecao.items():
            try:
                colecao = self.criar_ou_obter_colecao(nome_colecao)
                contagem[area] = colecao.count()
            except Exception as e:
                contagem[area] = -1  # erro
        return contagem
