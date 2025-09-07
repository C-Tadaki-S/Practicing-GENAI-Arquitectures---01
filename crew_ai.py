import os
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (nossa chave de API) do arquivo .env
load_dotenv() 

# Importando a ferramenta Serper
from crewai_tools import SerperDevTool

# Inicializando o LLM local
ollama_llm = OllamaLLM(model="ollama/llama3")

# Inicializando a nova ferramenta de busca
search_tool = SerperDevTool()

# --- DEFINIÇÃO DOS AGENTES ---

# Agente 1: Pesquisador de Cidades
city_researcher = Agent(
  role='Pesquisador de Destinos Turísticos',
  goal='Encontrar as atrações mais interessantes e eventos culturais para uma cidade específica.',
  backstory="""Você é um pesquisador experiente, mestre em usar a internet para
  descobrir joias escondidas e pontos turísticos imperdíveis em qualquer cidade do mundo.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_llm
)

# Agente 2: Curador de Restaurantes
food_critic = Agent(
  role='Especialista em Gastronomia Local',
  goal='Descobrir as melhores e mais autênticas experiências gastronômicas em uma cidade.',
  backstory="""Com um paladar refinado e um faro para comida boa, você é o guia
  definitivo para encontrar desde tascas tradicionais até restaurantes com estrelas Michelin.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_llm
)

# Agente 3: Agente de Logística e Roteiro
travel_concierge = Agent(
  role='Concierge de Viagens',
  goal='Criar um roteiro de viagem detalhado, dia a dia, que seja prático, emocionante e bem organizado.',
  backstory="""Você é um planejador de viagens meticuloso que transforma informações
  dispersas em um roteiro perfeito, equilibrando atividades, descanso e gastronomia.""",
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm
)

# --- DEFINIÇÃO DAS TAREFAS ---

task_city_research = Task(
  description='Pesquisar a cidade de Porto, Portugal. Focar em monumentos históricos, museus, parques e possíveis eventos culturais que acontecerão nos próximos meses. Faça um resumo dos top 5 pontos a não perder.',
  expected_output='Um parágrafo resumindo a cidade e uma lista com 5 lugares imperdíveis com uma breve descrição de cada um.',
  agent=city_researcher
)

task_food_research = Task(
  description='Com base nas informações da cidade de Porto, encontre 5 opções de restaurantes que ofereçam uma experiência local autêntica. Inclua uma opção econômica, uma de gama média e uma mais sofisticada. Descreva o tipo de comida e o que torna cada lugar especial.',
  expected_output='Uma lista de 5 restaurantes, com nome, faixa de preço, tipo de cozinha e um pequeno parágrafo descritivo para cada.',
  agent=food_critic,
  context=[task_city_research]
)

task_create_itinerary = Task(
  description='Usando os resultados da pesquisa de atrações e restaurantes, crie um roteiro detalhado de 3 dias para Porto, Portugal. Organize as atividades por dia (Manhã, Tarde, Noite). Certifique-se de que o roteiro seja lógico em termos de localização das atividades e inclua sugestões de restaurantes para almoço e jantar a cada dia.',
  expected_output='Um roteiro completo formatado em Markdown, com cabeçalhos para Dia 1, Dia 2 e Dia 3, e subdivisões para Manhã, Tarde e Noite.',
  agent=travel_concierge,
  context=[task_city_research, task_food_research]
)

# --- MONTAGEM DA CREW ---

travel_crew = Crew(
  agents=[city_researcher, food_critic, travel_concierge],
  tasks=[task_city_research, task_food_research, task_create_itinerary],
  verbose=True, # ## CORREÇÃO APLICADA AQUI ##
  process=Process.sequential
)

# --- EXECUÇÃO ---

print("######################")
print("## A Crew de Viagem está pronta para a decolagem! ##")
print("######################")
result = travel_crew.kickoff()

print("\n\n######################")
print("## Roteiro de Viagem Finalizado: ##")
print("######################\n")
print(result)