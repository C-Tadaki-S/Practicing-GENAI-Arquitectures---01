import os
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

from crewai.tools import tool
from crewai_tools import SerperDevTool

from langchain_community.cache import SQLiteCache 
from langchain.globals import set_llm_cache

load_dotenv()

set_llm_cache(SQLiteCache(database_path="cache.db"))

serper_tool = SerperDevTool()

@tool("Ferramenta de Busca Segura na Internet")
async def safe_serper_search(query: str) -> str:
    """
    Uma ferramenta de busca na internet assíncrona e segura.
    Extrai o texto da pesquisa mesmo que o input venha em um formato incorreto.
    """
    if isinstance(query, dict):
        search_query = query.get('description') or query.get('search_query', '')
    else:
        search_query = query
    
    return await serper_tool.arun(search_query)

# --- CONFIGURAÇÃO DOS AGENTES E LLM ---
ollama_llm = OllamaLLM(model="ollama/llama3")

city_researcher = Agent(
    role='Pesquisador de Destinos Turísticos',
    goal='Encontrar as atrações mais interessantes e eventos culturais para uma cidade específica.',
    backstory="""Você é um pesquisador experiente...""",
    verbose=True,
    allow_delegation=False,
    tools=[safe_serper_search],
    llm=ollama_llm
)

food_critic = Agent(
    role='Especialista em Gastronomia Local',
    goal='Descobrir as melhores e mais autênticas experiências gastronômicas...',
    backstory="""Com um paladar refinado...""",
    verbose=True,
    allow_delegation=False,
    tools=[safe_serper_search],
    llm=ollama_llm
)

travel_concierge = Agent(
    role='Concierge de Viagens',
    goal='Criar um roteiro de viagem detalhado...',
    backstory="""Você é um planejador de viagens meticuloso...""",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# --- DEFINIÇÃO DAS TAREFAS ---
task_city_research = Task(
    description='Pesquisar a cidade de Porto, Portugal...',
    expected_output='Um parágrafo resumindo a cidade e uma lista com 5 lugares...',
    agent=city_researcher
    # async_execution=True # ## REMOVIDO PARA CORRIGIR O ERRO DE CONTEXTO ##
)

task_food_research = Task(
    description="""Com base nas informações da cidade de Porto, encontre 5 opções de restaurantes...""",
    expected_output='Uma lista de 5 restaurantes...',
    agent=food_critic,
    context=[task_city_research]
    # async_execution=True # ## REMOVIDO PARA CORRIGIR O ERRO DE CONTEXTO ##
)

task_create_itinerary = Task(
    description='Usando os resultados da pesquisa de atrações e restaurantes, crie um roteiro detalhado...',
    expected_output='Um roteiro completo formatado em Markdown...',
    agent=travel_concierge,
    context=[task_city_research, task_food_research]
)

# --- MONTAGEM E EXECUÇÃO DA CREW ---
travel_crew = Crew(
    agents=[city_researcher, food_critic, travel_concierge],
    tasks=[task_city_research, task_food_research, task_create_itinerary],
    verbose=True,
    process=Process.sequential 
)

async def main():
    print("######################")
    print("## A Crew de Viagem Otimizada está pronta para a decolagem! ##")
    print("######################")
    
    result = await travel_crew.kickoff_async()

    print("\n\n######################")
    print("## Roteiro de Viagem Finalizado: ##")
    print("######################\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())