from dotenv import load_dotenv
from poscomp_rag.system import SistemaPOSCOMP

def main():
    load_dotenv()
    sistema = SistemaPOSCOMP(auto_carregar_dados=True)
    sistema.status_banco()

    query = "Fundamentos da Computação - Linguagens de Programação - Paradigmas de Linguagens de Programação"
    area_conhecimento = query.split(' - ')[0]
    area = query.split(' - ')[1]
    subarea = query.split(' - ')[2] if len(query) > 2 and query[2] else ""

    print("\n Gerando nova questão baseada em RAG + LLM:")
    print("=" * 50 + "\n")
    sistema.gerar_nova_questao_com_rag(query, area_conhecimento, area, subarea)

    print("\n Gerando nova questão sem RAG, apenas com LLM:")
    print("=" * 50 + "\n")
    sistema.gerar_nova_questao_llm(area, subarea)

if __name__ == "__main__":
    main()
