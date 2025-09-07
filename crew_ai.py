import os
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

# # Importe a anotação @tool para criar ferramentas personalizadas
from crewai.tools import tool

load_dotenv()
ollama_llm = OllamaLLM(model="ollama/llama3")

# #############################################################################
# ## ESTRATÉGIA 2: CRIANDO NOSSA FERRAMENTA SEGURA                       ##
# #############################################################################

# Inicializamos a ferramenta original que queremos "envolver"
serper_tool = SerperDevTool()

@tool("Ferramenta de Busca Segura na Internet")
def safe_serper_search(query: str) -> str:
    """
    Uma ferramenta de busca na internet que é mais segura de usar.
    Ela extrai automaticamente o texto da pesquisa mesmo que o input venha em um formato incorreto.
    Use esta ferramenta para qualquer pesquisa na web.
    """
    # Verifica se o input (query) é um dicionário, como no erro que vimos
    if isinstance(query, dict):
        # Se for, extrai o valor da chave 'description' ou 'search_query'
        search_query = query.get('description') or query.get('search_query', '')
    else:
        # Se não, o input já é um texto simples (string)
        search_query = query
    
    # Executa a busca com o texto limpo e retorna o resultado
    return serper_tool.run(search_query)

# #############################################################################
# ## AGORA USAMOS a 'safe_serper_search' em vez da 'SerperDevTool'         ##
# #############################################################################

city_researcher = Agent(
  role='Pesquisador de Destinos Turísticos',
  goal='Encontrar as atrações mais interessantes e eventos culturais para uma cidade específica.',
  backstory="""Você é um pesquisador experiente...""",
  verbose=True,
  allow_delegation=False,
  tools=[safe_serper_search], # <-- USANDO A NOVA FERRAMENTA
  llm=ollama_llm
)

food_critic = Agent(
  role='Especialista em Gastronomia Local',
  goal='Descobrir as melhores e mais autênticas experiências gastronômicas...',
  backstory="""Com um paladar refinado...""",
  verbose=True,
  allow_delegation=False,
  tools=[safe_serper_search], # <-- USANDO A NOVA FERRAMENTA
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
  description="""Com base nas informações da cidade de Porto, encontre 5 opções de restaurantes que ofereçam uma experiência local autêntica. Inclua uma opção econômica, uma de gama média e uma mais sofisticada. Descreva o tipo de comida e o que torna cada lugar especial.
  Para usar a ferramenta de busca, passe um texto simples para o argumento 'search_query'.
  Por exemplo: 'restaurantes de comida tradicional no Porto'.""",
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