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
Você é um assistente de pesquisa altamente qualificado. Sua tarefa é responder à pergunta do usuário de forma precisa e concisa, baseando-se estritamente no contexto fornecido.
Analise todos os trechos de contexto abaixo antes de formular sua resposta.
Se a informação necessária não estiver presente em nenhum dos trechos, responda exatamente: 'Com base nos documentos fornecidos, não encontrei a informação solicitada.'.
Não utilize nenhum conhecimento prévio.
Contexto:
{context}
Pergunta:
{input}
Resposta Precisa:
"""
prompt = PromptTemplate.from_template(prompt_template_texto)

# Chain que efetivamente gera a resposta a partir do contexto e da pergunta
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# --- 4. EXECUÇÃO E TESTE ---

# CORREÇÃO 1: Definimos o retriever UMA VEZ, aqui.
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("Retriever criado com k=6.")

print("\n--- Inicie a conversa com o assistente de pesquisa (digite 'sair' para terminar) ---")
while True:
    pergunta_usuario = input("\nSua pergunta: ")
    if pergunta_usuario.lower() == 'sair':
        break

    # PASSO DE EXPANSÃO DE PERGUNTA
    print("\n--- Expandindo a pergunta... ---")
    perguntas_para_busca = expandir_pergunta(pergunta_usuario, llm)
    print("Perguntas usadas na busca:", perguntas_para_busca)

    # PASSO DE BUSCA USANDO TODAS AS PERGUNTAS
    # Usamos o retriever definido anteriormente
    retrieved_docs_list = retriever.batch(perguntas_para_busca)
    
    documentos_unicos = {}
    for result_list in retrieved_docs_list:
        for doc in result_list:
            if doc.page_content not in documentos_unicos:
                documentos_unicos[doc.page_content] = doc
    
    documentos_recuperados = list(documentos_unicos.values())
    
    print(f"\n--- {len(documentos_recuperados)} Documentos Únicos Recuperados (Diagnóstico) ---")
    for i, doc in enumerate(documentos_recuperados[:10]): # Mostra até 6
        print(f"--- Documento {i+1} ---\n{doc.page_content}\n--------------------------\n")

    # CORREÇÃO 2: Invocamos a chain que GERA a resposta, passando o contexto que JÁ ENCONTRAMOS
    response = combine_docs_chain.invoke({
        "input": pergunta_usuario, 
        "context": documentos_recuperados
    })

    print("\nResposta do Assistente:")
    print(response)