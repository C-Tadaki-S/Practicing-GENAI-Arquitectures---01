# Importações necessárias e ATUALIZADAS
from langchain_ollama import OllamaLLM # <-- MUDANÇA AQUI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Carregamento do modelo
print("Carregando o modelo LLM...")
llm = OllamaLLM(model="llama3:8b") # <-- MUDANÇA AQUI (OllamaLLM)
print("Modelo carregado.")

# 1. Definição do Template do Prompt
prompt_template_texto = "Responda a seguinte pergunta de forma concisa: {pergunta}"
prompt_template = PromptTemplate.from_template(prompt_template_texto)

# 2. Criação da Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# 3. Invocação da Chain
print("Enviando pergunta através da chain...")
minha_pergunta = "Qual a capital do Brasil?"
resposta_dict = chain.invoke({"pergunta": minha_pergunta})

# 4. Impressão da Resposta
print("\n--- Resposta da Chain (Dicionário Completo) ---")
print(resposta_dict)

print("\n--- Apenas o Texto da Resposta ---")
print(resposta_dict['text'])
print("---------------------------------")