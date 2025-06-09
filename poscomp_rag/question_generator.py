import pandas as pd
from typing import List
from langchain.schema import HumanMessage
from poscomp_rag.core.llm import GerenciadorLLM


class GeradorQuestoes:
    def __init__(self, llm: GerenciadorLLM):
        self.llm = llm
    
    def gerar_zeroshot_prompt(area, subarea):
        formatted_topic = area if pd.Series(subarea).isna().any() else f"{area} - {subarea}"

        return f"Gere uma questão (e apenas uma) no estilo do POSCOMP relacionada ao tema “{formatted_topic}”."

        
    def gerar_com_base_em_rag(self,consulta:str, questoes: List[str], area: str, subarea:str, qtd_perguntas: int = 1) -> List[str]:
        """
        Gera uma nova questão baseada nas questões recuperadas via RAG e nos conhecimentos do LLM.
        
        """
        if not questoes or not self.llm.llm:
            print("⚠️ LLM indisponível ou nenhuma questão base fornecida.")
            return []
    
        texto_base = ""
        for q in questoes:
                for idx, q in enumerate(questoes, 1):
                    enunciado = q.get("enunciado", "")
                    alternativas = q.get("alternativas", [])
                    gabarito = q.get("gabarito", "")
                    if enunciado and alternativas and gabarito:
                        texto_base += f"Questão {idx}:\nEnunciado: {enunciado}\nAlternativas: {', '.join(alternativas)}\nGabarito: {gabarito}\n\n"

        topico = area if pd.Series(subarea).isna().any() else f"{area} - {subarea}"
        
        prompt = f"""Abaixo estão alguns exemplos de questões do exame POSCOMP relacionadas ao tema: “{topico}”.

        {texto_base}

        Agora, com base nos exemplos acima, gere uma (e apenas uma) nova questão de múltipla escolha no estilo POSCOMP sobre o mesmo tema.
        """

        try:
            resposta = self.llm.gerar_resposta([HumanMessage(content=prompt)])
            print(resposta)
                
        except Exception as e:
            print(f"⚠️ Erro ao gerar perguntas: {e}")
            return []    


    def gerar_perguntas_llm(self, area: str, subarea:str, qtd_perguntas: int = 1) -> List[str]:
        """Gera novas perguntas com base apenas nas questões recuperadas via RAG"""
        topico = area if pd.Series(subarea).isna().any() else f"{area} - {subarea}"
        prompt = f"Gere uma questão (e apenas uma) no estilo do POSCOMP relacionada ao tema “{topico}”."
        try:
            resposta = self.llm.gerar_resposta([HumanMessage(content=prompt)])
            print(resposta)
             
        except Exception as e:
            print(f"⚠️ Erro ao gerar perguntas: {e}")
            return []
        
    

