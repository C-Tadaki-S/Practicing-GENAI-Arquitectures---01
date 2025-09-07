# --- SEÇÃO 1: IMPORTS E CONFIGURAÇÕES INICIAIS ---
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_experimental.tools import PythonAstREPLTool

# --- SEÇÃO 2: CARREGAMENTO DOS COMPONENTES ---
print("Inicializando o sistema...")
FAISS_INDEX_PATH = "faiss_index"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

llm = ChatOllama(model="llama3:instruct")

cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
print("Componentes de RAG e Re-ranker carregados.")


# --- SEÇÃO 3: DEFINIÇÃO DAS FERRAMENTAS REFINADAS ---

def run_rag_pipeline(pergunta: str) -> str:
    print(f"\n--- Roteado para: Ferramenta RAG ---")
    
    prompt_expansao_template = ChatPromptTemplate.from_messages([
        ("system", "Gere 3 versões alternativas para a pergunta do usuário para melhorar a busca. Responda apenas com as 3 perguntas, uma por linha."),
        ("human", "{pergunta_original}")
    ])
    chain_expansao = prompt_expansao_template | llm
    res = chain_expansao.invoke({"pergunta_original": pergunta})
    perguntas_para_busca = [pergunta] + res.content.strip().split('\n')
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    retrieved_docs_list = retriever.batch(perguntas_para_busca)
    
    documentos_unicos = {doc.page_content: doc for result_list in retrieved_docs_list for doc in result_list}
    candidatos = list(documentos_unicos.values())
    
    pares_para_reranking = [[pergunta, doc.page_content] for doc in candidatos]
    scores = cross_encoder.predict(pares_para_reranking)
    
    docs_com_scores = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)
    
    # MUDANÇA: Aumentando para 6 documentos para mais contexto
    documentos_finais = [doc for doc, score in docs_com_scores[:6]]
    
    # MUDANÇA: Prompt RAG final ainda mais direto
    prompt_rag = ChatPromptTemplate.from_template(
        """Com base exclusivamente no contexto abaixo, extraia e apresente a resposta para a pergunta do usuário.
        Se a resposta não estiver no contexto, diga 'A informação não foi encontrada nos documentos.'

        Contexto:
        {context}
        
        Pergunta:
        {input}
        """
    )
    chain_rag = prompt_rag | llm
    
    resposta_final = chain_rag.invoke({"input": pergunta, "context": documentos_finais})
    return resposta_final.content

# Nossa ferramenta de calculadora
calculator = PythonAstREPLTool()

# MUDANÇA: Função 'run_calculator' agora traduz para Python
def run_calculator(pergunta: str) -> str:
    print(f"\n--- Roteado para: Ferramenta Calculadora ---")
    
    # Prompt para traduzir a linguagem natural para código Python
    prompt_calculo_template = ChatPromptTemplate.from_template(
        """Sua tarefa é traduzir a pergunta do usuário em uma única linha de código Python para ser executada.
        Responda apenas com o código. Não adicione explicações ou a palavra 'python'.

        Exemplos:
        Pergunta: quanto é 5 mais 3?
        Resposta: 5 + 3

        Pergunta: 15% de 5000
        Resposta: 0.15 * 5000

        Pergunta: qual a raiz quadrada de 81?
        Resposta: 81**0.5

        Pergunta: {input}
        Resposta:
        """
    )
    
    # Chain para criar o código
    chain_calculo = prompt_calculo_template | llm
    
    # Traduz a pergunta para código
    codigo_python = chain_calculo.invoke({"input": pergunta}).content.strip()
    
    print(f"Código Python gerado: {codigo_python}")
    
    # Executa o código Python gerado
    resultado = calculator.invoke(codigo_python)
    return f"O resultado é: {resultado}"


# --- SEÇÃO 4: O ROTEADOR INTELIGENTE (sem alterações) ---
router_prompt_template = """
Sua tarefa é classificar a pergunta de um usuário em uma de três categorias: 'pesquisa_documentos', 'calculadora', ou 'geral'.
Responda apenas com a palavra da categoria. Não adicione nenhuma outra palavra ou pontuação.
- Use 'pesquisa_documentos' para perguntas sobre a Petrobras, suas políticas, finanças, relatórios, governança ou código de conduta.
- Use 'calculadora' para perguntas que envolvam cálculos matemáticos explícitos.
- Use 'geral' para cumprimentos, perguntas gerais ou qualquer outra coisa.
Pergunta do usuário:
{input}
Categoria:
"""
router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
router_chain = router_prompt | llm

# --- SEÇÃO 5: LOOP DE INTERAÇÃO PRINCIPAL ---
if __name__ == "__main__":
    while True:
        pergunta_usuario = input("\nSua pergunta: ")
        if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
            break

        rota = router_chain.invoke({"input": pergunta_usuario}).content.strip().lower()
        resposta_final = ""
        if "pesquisa_documentos" in rota:
            resposta_final = run_rag_pipeline(pergunta_usuario)
        elif "calculadora" in rota:
            resposta_final = run_calculator(pergunta_usuario)
        else:
            print(f"\n--- Roteado para: Resposta Geral ---")
            resposta_final = llm.invoke(pergunta_usuario).content

        print("\n\033[92mResposta Final:\033[0m")
        print(resposta_final)