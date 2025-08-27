from typing import List, Optional, Union
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI


class GerenciadorLLM:
    """
    Wrapper para inicialização e uso de modelos LLM via LangChain.
    Suporte atual: OpenAI API e compatíveis via DeepInfra (ou outros provedores OpenAI-like).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        modelo_nome: str = "meta-llama/Llama-3.3-70B-Instruct",
        temperatura: float = 0.5,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.modelo_nome = modelo_nome
        self.temperatura = temperatura
        self.llm: Optional[BaseChatModel] = self._criar_modelo()

    def _criar_modelo(self) -> Optional[BaseChatModel]:
        if not self.api_key:
            print("⚠️ API Key não configurada. LLM desabilitado.")
            return None

        try:
            return ChatOpenAI(
                model_name=self.modelo_nome,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=self.temperatura
            )
        except Exception as e:
            print(f"⚠️ Erro ao inicializar o modelo LLM: {e}")
            return None

    def trocar_modelo(self, novo_modelo: str, temperatura: Optional[float] = None):
        """Permite trocar dinamicamente o modelo e/ou temperatura"""
        self.modelo_nome = novo_modelo
        if temperatura is not None:
            self.temperatura = temperatura
        self.llm = self._criar_modelo()

    def gerar_resposta(self, mensagens: List[Union[str, BaseMessage]]) -> Optional[str]:
        """Envia mensagens para o modelo LLM e retorna a resposta"""
        if not self.llm:
            print("Modelo LLM não inicializado.")
            return None

        try:
            mensagens_formatadas = [
                HumanMessage(content=msg) if isinstance(msg, str) else msg
                for msg in mensagens
            ]
            resposta = self.llm.invoke(mensagens_formatadas)
            return resposta.content.strip()
        except Exception as e:
            print(f"Erro ao gerar resposta LLM: {e}")
            return None
