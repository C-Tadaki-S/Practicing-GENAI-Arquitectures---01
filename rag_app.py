#importação de bibliotecas
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings # <-- CORRIGIDO: Importação correta que faltava
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers.cross_encoder import CrossEncoder
# Carregar e preparar os dados 
DATA_PATH = "docs/"
print("Carregando documentos PDF...")
documentos = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)
        loader = PyPDFLoader(pdf_path)
        documentos.extend(loader.load())
print(f"Total de páginas carregados: {len(documentos)}")

# Dividir os documentos em pedaços menores CHUNKS
print("Dividindo documentos em pedaços menores...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
chunks = text_splitter.split_documents(documentos)
print(f"Total de pedaços criados: {len(chunks)}")


# --- 2. EMBEDDINGS E VECTOR STORE ---
FAISS_INDEX_PATH = "faiss_index"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(FAISS_INDEX_PATH):
    print("Carregando Vector Store existente...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector Store carregado com sucesso.")
else:
    print("Criando embeddings e armazenando no FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Vector Store criado e salvo com sucesso.")


# --- 3: CONFIGURAÇÃO DAS CHAINS E COMPONENTES ---
llm = OllamaLLM(model="llama3:8b")

# NOVO: Carrega o modelo de Cross-Encoder para Re-ranking
print("Carregando modelo de Re-ranker (Cross-Encoder)...")
cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
print("Re-ranker carregado.")


# Função de expansão de pergunta (sem alterações)
def expandir_pergunta(pergunta, llm):
    prompt_expansao = PromptTemplate.from_template(
        """
        Você é um assistente de IA. Sua tarefa é gerar 3 versões alternativas para a seguinte pergunta de um usuário, 
        para melhorar a recuperação de documentos. Mantenha o mesmo significado, mas use sinônimos e diferentes formulações.
        Responda apenas com as 3 perguntas, uma por linha.
        Pergunta Original: {pergunta_original}
        Perguntas Alternativas:
        """
    )
    chain_expansao = LLMChain(llm=llm, prompt=prompt_expansao)
    res = chain_expansao.invoke({"pergunta_original": pergunta})
    perguntas_expandidas = [pergunta] + res['text'].strip().split('\n')
    return perguntas_expandidas

# Prompt para a geração da resposta final (sem alterações)
prompt_template_texto = """
Você é um assistente de pesquisa especializado. Sua tarefa é responder à pergunta do usuário de forma clara e direta, baseando-se exclusivamente no contexto fornecido.

Siga estas regras estritamente:
1.  Primeiro, responda diretamente à pergunta do usuário com "Sim.", "Não." ou "A informação não é conclusiva com base no contexto.".
2.  Após a resposta direta, justifique-a citando as informações mais relevantes encontradas no contexto.
3.  Se o contexto não contiver NENHUMA informação relacionada à pergunta, e somente nesse caso, responda: 'Não há informações sobre este assunto nos documentos fornecidos.'.

Contexto:
{context}

Pergunta:
{input}

Resposta:
"""
prompt = PromptTemplate.from_template(prompt_template_texto)

# Chain que efetivamente gera a resposta a partir do contexto e da pergunta
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# --- SEÇÃO 4: EXECUÇÃO E TESTE (Grandes alterações aqui) ---

# MUDANÇA: Aumentamos o 'k' para a recuperação inicial (Recall)
K_RECUPERACAO_INICIAL = 20
retriever = vectorstore.as_retriever(search_kwargs={"k": K_RECUPERACAO_INICIAL})
print(f"Retriever criado para a fase de RECUPERAÇÃO com k={K_RECUPERACAO_INICIAL}.")

print("\n--- Inicie a conversa com o assistente de pesquisa (digite 'sair' para terminar) ---")
while True:
    pergunta_usuario = input("\nSua pergunta: ")
    if pergunta_usuario.lower() == 'sair':
        break

    # 1. FASE DE EXPANSÃO DE PERGUNTA
    print("\n--- Fase 1: Expandindo a pergunta... ---")
    perguntas_para_busca = expandir_pergunta(pergunta_usuario, llm)
    print("Perguntas usadas na busca:", perguntas_para_busca)

    # 2. FASE DE RECUPERAÇÃO (RECALL)
    print(f"\n--- Fase 2: Recuperando até {K_RECUPERACAO_INICIAL} chunks candidatos... ---")
    retrieved_docs_list = retriever.batch(perguntas_para_busca)
    
    documentos_unicos = {}
    for result_list in retrieved_docs_list:
        for doc in result_list:
            if doc.page_content not in documentos_unicos:
                documentos_unicos[doc.page_content] = doc
    
    candidatos = list(documentos_unicos.values())
    print(f"Total de {len(candidatos)} chunks candidatos únicos recuperados.")

    # 3. FASE DE RE-RANKING (PRECISION)
    print(f"\n--- Fase 3: Re-rankeando os {len(candidatos)} candidatos... ---")
    # Prepara os pares (pergunta, chunk) para o cross-encoder
    pares_para_reranking = [[pergunta_usuario, doc.page_content] for doc in candidatos]
    
    # Obtém as pontuações de relevância
    scores = cross_encoder.predict(pares_para_reranking)
    
    # Combina os documentos com suas novas pontuações e ordena
    docs_com_scores = list(zip(candidatos, scores))
    docs_re_rankeados = sorted(docs_com_scores, key=lambda x: x[1], reverse=True)
    
    # Seleciona o Top N final para enviar ao LLM
    K_FINAL = 4
    documentos_finais = [doc for doc, score in docs_re_rankeados[:K_FINAL]]

    print(f"\n--- {len(documentos_finais)} Documentos Finais após Re-ranking (Diagnóstico) ---")
    for i, doc in enumerate(documentos_finais):
        # Mostra o score do re-ranker para diagnóstico
        print(f"--- Documento {i+1} (Score: {docs_re_rankeados[i][1]:.4f}) ---\n{doc.page_content}\n--------------------------\n")

    # 4. FASE DE GERAÇÃO
    response = combine_docs_chain.invoke({
        "input": pergunta_usuario, 
        "context": documentos_finais # Enviamos apenas os documentos re-rankeados
    })

    print("\nResposta do Assistente:")
    print(response)