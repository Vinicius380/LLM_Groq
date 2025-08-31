# Código llm_mult.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq # Ou outro LLM
from langchain_community.utilities import SQLDatabase # Para representar o DB

# Importações para o Agente SQL + Python
from langchain_community.agent_toolkits import SQLDatabaseToolkit # Toolkit com ferramentas de DB
# Mantenha esta importação se a anterior falhou, caso contrário, use a mais direta se funcionar.
# from langchain_community.agent_toolkits.sql import create_sql_agent # Local alternativo para create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent # Local alternativo para create_sql_agent


from langchain.agents import create_tool_calling_agent, AgentExecutor # Forma moderna de criar agente

# CORREÇÃO AQUI: Adicionando MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Para criar o prompt e o placeholder de mensagens

from langchain_core.messages import AIMessage, HumanMessage # Tipos de mensagens
from langchain_core.tools import Tool # Classe base para ferramentas

# Importação para executar código Python (para plotting)
from langchain_experimental.tools.python.tool import PythonREPL # Ferramenta para executar Python

# ... (resto do seu código continua abaixo) ...


# Importações comuns dentro do PythonREPL (o agente pode precisar delas)
# Não precisamos importar aqui no topo do script principal,
# mas o agente pode precisar gerá-las no Action Input para o PythonREPL.
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np # Pode ser útil para manipulação numérica


# --- 1. Carregar variáveis de ambiente (chaves de API e DB) ---
load_dotenv()

# --- 2. Obter credenciais do banco de dados ---
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_port = os.getenv("MYSQL_PORT")
db_name = os.getenv("MYSQL_DB_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Verificar se as variáveis de ambiente foram carregadas ---
if not all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
    print("Erro: Verifique se todas as variáveis (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB_NAME, GROQ_API_KEY) estão definidas no seu arquivo .env")
    print("Variáveis obtidas:")
    print(f"MYSQL_USER: {db_user}")
    print(f"MYSQL_PASSWORD: {'*' * len(db_password) if db_password else 'None'}") # Não imprimir senha real
    print(f"MYSQL_HOST: {db_host}")
    print(f"MYSQL_PORT: {db_port}")
    print(f"MYSQL_DB_NAME: {db_name}")
    print(f"GROQ_API_KEY: {'*' * len(groq_api_key) if groq_api_key else 'None'}") # Não imprimir chave real
    exit()

# --- 3. Criar a URL de conexão do banco de dados para SQLAlchemy ---
db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# Se usar PyMySQL, use: db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


# --- 4. Criar um objeto SQLDatabase do LangChain ---
try:
    # Limitamos o agente a ver apenas a tabela 'waveforms'
    db = SQLDatabase.from_uri(db_uri, include_tables=["waveforms"])
    print(f"Conexão com o banco de dados '{db_name}' estabelecida com sucesso.")
    print(f"Tabela 'waveforms' acessível para o agente.")

except Exception as e:
    print(f"Erro ao conectar ao banco de dados ou inicializar SQLDatabase: {e}")
    print(f"URI de conexão tentada: {db_uri}") # Mostrar a URI para depuração de conexão
    print("Verifique suas credenciais no .env (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB_NAME) e se o servidor MySQL está rodando e acessível.")
    exit()

# --- 5. Inicializar o LLM (Groq) ---
llm = ChatGroq(temperature=0.2, model="llama3-70b-8192") # Temperatura levemente maior para criatividade do agente
# Se preferir o modelo 8B:
# llm = ChatGroq(temperature=0.2, model="llama3-8b-8192")


# --- 6. Criar as Ferramentas para o Agente ---

# Ferramentas de Banco de Dados (via Toolkit)
# O Toolkit já cria ferramentas como sql_db_query, sql_db_schema, etc.
db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_tools = db_toolkit.get_tools() # Obtém a lista de ferramentas de DB

# Ferramenta de Execução de Código Python
# Permitimos 'allow_dangerous_code=True' pois o agente gerará código que pode plotar.
# CUIDADO ao usar em ambientes de produção com inputs de usuários não confiáveis!
python_repl = PythonREPL()
python_tool = Tool(
    name="python_repl",
    description="A Python REPL. Use this to execute python commands. Input is a valid python command. Use useful packages like pandas, matplotlib.",
    func=python_repl.run,
)

# Lista completa de ferramentas para o agente
tools = db_tools + [python_tool] # Junta as ferramentas de DB com a ferramenta Python


# --- 7. Criar o Prompt do Agente ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é um agente especializado em analisar dados em um banco de dados MySQL.
Você tem acesso à tabela 'waveforms' que contém dados de vibração.
Você pode executar consultas SQL para obter dados e usar um interpretador Python para analisar e plotar.

Sempre siga este formato de raciocínio e ação:
Thought: Seu raciocínio sobre o próximo passo, considerando a pergunta e as ferramentas disponíveis.
Action: nome_da_ferramenta (DEVE ser uma das ferramentas disponíveis: sql_db_list_tables, sql_db_schema, sql_db_query, python_repl)
Action Input: A entrada para a ferramenta, no formato apropriado para a ferramenta.
Observation: O resultado retornado pela ferramenta.
... (Este ciclo Thought/Action/Observation se repete quantas vezes forem necessárias)
Thought: Agora, com base nas observações, determine a resposta final para o usuário.
Final Answer: A resposta final para o usuário.

Ao responder a perguntas:
- Se a pergunta for sobre dados no DB (contar, filtrar, sumarizar), use a ferramenta apropriada do banco de dados (geralmente 'sql_db_query').
- Se a pergunta pedir um GRÁFICO ou VISUALIZAÇÃO, você DEVE usar a ferramenta 'python_repl'.
- Para plotar usando 'python_repl':
    1. Use a ferramenta 'sql_db_query' para obter os dados NECESSÁRIOS para o gráfico (apenas as colunas e linhas relevantes). Para grandes volumes de dados (sua tabela tem ~40M linhas no total), SELECIONE APENAS UMA AMOSTRA (use LIMIT) ou AGREGUE os dados antes de obter para não sobrecarregar a memória.
    2. Quando você usar 'sql_db_query', a 'Observation' conterá os **dados retornados** pela consulta SQL. Estes dados estarão em um formato que pode ser passado DIRETAMENTE para a ferramenta 'python_repl' como uma **variável chamada 'data'**. Os dados serão uma **lista de tuplas**.
    3. No interpretador Python ('python_repl'), você DEVE importar 'pandas', 'matplotlib.pyplot' (como plt) e 'seaborn' (como sns).
    4. Crie um DataFrame pandas a partir da lista de tuplas 'data' que foi passada. Use os nomes das colunas corretos da tabela 'waveforms' para nomear as colunas do DataFrame. Se você usou 'SELECT *', use os nomes das colunas do esquema da tabela (X-Axis, Ch1 Y-Axis, Ch2 Y-Axis, Ch3 Y-Axis, Sample_id, Condition, RPM, Load_kW). Exemplo: `df = pd.DataFrame(data, columns=['X-Axis', 'Ch1 Y-Axis', ...])`.
    5. Use matplotlib ou seaborn para gerar o gráfico a partir do DataFrame `df`. df = pd.DataFrame(data, columns=['X-Axis', 'Ch1 Y-Axis', 'Ch2 Y-Axis', 'Ch3 Y-Axis', 'Sample_id', 'Condition', 'RPM', 'Load_kW'])
    6. Gere código Python que **SALVE o gráfico em um arquivo** Sempre use 'plt.savefig("grafico.png")' e não use 'plt.show()'. Após salvar, informe o caminho do arquivo salvo.

    7. O resultado da sua 'python_repl' Action deve ser a saída do código (sucesso/erro) ou a confirmação do salvamento.
- Seja conciso em suas respostas, mas forneça o resultado, descreva o gráfico gerado/salvo, ou forneça o código Python para o usuário se a execução não for possível.
- Se a execução do Python falhar (a 'Observation' da ferramenta 'python_repl' mostrará o erro), você DEVE reportar o erro para o usuário.
- **Sempre use o formato Action: nome_da_ferramenta e Action Input: entrada_para_a_ferramenta quando for usar uma ferramenta.**

""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# --- 8. Criar o Agente ---
# create_tool_calling_agent é uma forma comum de criar agentes modernos no LangChain
agent = create_tool_calling_agent(llm, tools, prompt)

# --- 9. Criar o Executor do Agente ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True para ver os passos

print("Agente SQL + Plotting pronto!")
print("Agora você pode fazer perguntas sobre os dados na tabela 'waveforms', incluindo pedidos de gráficos.")
print("Lembre-se de pedir gráficos para AMOSTRAS ou DADOS AGREGADOS devido ao volume.")
print("Exemplos: 'Mostre as primeiras 5 linhas da tabela.', 'Qual a média de RPM?', 'Plote X-Axis vs Ch1 Y-Axis para as primeiras 100 linhas'.")
print("Digite 'sair' a qualquer momento para encerrar.")

# --- 10. Loop Interativo ---
chat_history = [] # Para manter o histórico da conversa (útil para follow-ups)

while True:
    question = input("\nSua pergunta sobre os dados/gráficos: ")

    if question.lower() == 'sair':
        print("Saindo...")
        break

    if not question.strip():
        print("Por favor, digite uma pergunta válida.")
        continue

    try:
        # Executa o agente com a pergunta e o histórico
        response = agent_executor.invoke({"input": question, "chat_history": chat_history})

        print("\n--- Resposta do Agente ---")
        agent_response = response.get('output', 'Não foi possível obter uma resposta.')
        print(agent_response)

        # Atualiza o histórico para a próxima iteração
        chat_history.extend([HumanMessage(content=question), AIMessage(content=agent_response)])

    except Exception as e:
        print(f"\nOcorreu um erro ao processar a pergunta: {e}")
        print("Tente reformular a pergunta ou verifique os logs do agente (se verbose=True).")
        # Se o erro for na execução do PythonREPL (como OverflowError), o agente tentará lidar,
        # mas erros graves podem parar a cadeia e aparecer aqui.


# Fim do script
print("Programa finalizado.")