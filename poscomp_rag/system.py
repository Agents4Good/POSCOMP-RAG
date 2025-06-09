import os
from typing import List, Tuple, Optional

from poscomp_rag.core.embeddings import GeradorEmbeddings
from poscomp_rag.core.llm import GerenciadorLLM
from poscomp_rag.core.data_loader import CarregadorBanco
from poscomp_rag.core.utils import normalizar_texto
from poscomp_rag.retriever import BuscadorQuestoes
from poscomp_rag.question_generator import GeradorQuestoes


class SistemaPOSCOMP:
    def __init__(self, auto_carregar_dados: bool = True):
        self.api_key = os.getenv("DEEPINFRA_API_KEY")
        self.base_url = "https://api.deepinfra.com/v1/openai"

        self.modelo_embedding = GeradorEmbeddings()
        self.llm = GerenciadorLLM(api_key=self.api_key, base_url=self.base_url)
        
        self.area_para_colecao = {
            "Matemática": "matematica",
            "Fundamentos da Computação": "fundamentos_computacao",
            "Tecnologia de Computação": "tecnologia_computacao"
        }

        self.arquivos_questoes = {
            "fundamentos_computacao": "poscomp_rag/data/questoes_fundamentos_computacao.json",
            "tecnologia_computacao": "poscomp_rag/data/questoes_tecnologia_computacao.json",
            "matematica": "poscomp_rag/data/questoes_matematica.json"
        }

        self.carregador = CarregadorBanco("poscomp_chroma_db", self.modelo_embedding.modelo, self.area_para_colecao)
        self.buscador = BuscadorQuestoes(self.modelo_embedding)
        self.gerador = GeradorQuestoes(self.llm)

        if auto_carregar_dados:
            self.inicializar_banco()

    def inicializar_banco(self):
        print("🔄 Inicializando banco de dados...")
        for nome_colecao, arquivo in self.arquivos_questoes.items():
            if self.carregador.colecao_precisa_dados(nome_colecao):
                self.carregador.carregar_questoes_arquivo(arquivo, nome_colecao)
            else:
                colecao = self.carregador.criar_ou_obter_colecao(nome_colecao)
                print(f"✅ Coleção '{nome_colecao}' já possui {colecao.count()} questões.")

    def buscar_questoes(self, consulta: str, area: str, top_k: int = 5, limiar: float = 0.4):
        colecao_nome = self.area_para_colecao.get(area)
        if not colecao_nome:
            print(f"⚠️ Área '{area}' não mapeada.")
            return []
        colecao = self.carregador.criar_ou_obter_colecao(colecao_nome)
        return self.buscador.buscar_similares(colecao, consulta, top_k=top_k, limiar=limiar)

    def gerar_nova_questao_com_rag(self, consulta: str, area_conhecimento: str, area: str, subarea: str) -> List[str]:
        questoes_base = self.buscar_questoes(consulta, area_conhecimento)
        return self.gerador.gerar_com_base_em_rag(consulta, questoes_base, area, subarea)

    def gerar_nova_questao_llm(self, area: str, subarea: str) -> List[str]:
        return self.gerador.gerar_perguntas_llm(area, subarea)


    def status_banco(self):
        print("\n📊 STATUS DO BANCO:")
        total = 0
        contagem = self.carregador.contar_questoes_por_colecao()
        for area, qtd in contagem.items():
            if qtd >= 0:
                print(f"{area}: {qtd} questões")
                total += qtd
            else:
                print(f"{area}: Erro na leitura")
        print(f"TOTAL: {total} questões\n")
