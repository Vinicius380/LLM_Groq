import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq # Ou outro LLM
from langchain_community.utilities import SQLDatabase # Para representar o DB
from langchain_community.agent_toolkits import create_sql_agent


# --- 1. Carregar variáveis de ambiente (chaves de API e DB) ---
load_dotenv()

# --- 2. Obter credenciais do banco de dados ---
db_user = os.getenv("MYSQL_USER") # CORRETO: Obtém o VALOR da variável chamada "MYSQL_USER"
db_password = os.getenv("MYSQL_PASSWORD") # CORRETO: Obtém o VALOR da variável chamada "MYSQL_PASSWORD"
db_host = os.getenv("MYSQL_HOST") # CORRETO: Obtém o VALOR da variável chamada "MYSQL_HOST"
db_port = os.getenv("MYSQL_PORT") # CORRETO: Obtém o VALOR da variável chamada "MYSQL_PORT"
db_name = os.getenv("MYSQL_DB_NAME") # CORRETO: Obtém o VALOR da variável chamada "MYSQL_DB_NAME"
groq_api_key = os.getenv("GROQ_API_KEY") # Já estava correto

# --- Verificar se as variáveis de ambiente foram carregadas ---
if not all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
    print("Erro: Verifique se todas as variáveis (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB_NAME, GROQ_API_KEY) estão definidas no seu arquivo .env")
    exit()

# --- 3. Criar a URL de conexão do banco de dados para SQLAlchemy ---
# Formato: 'dialect+driver://user:password@host:port/database'
# Usaremos mysql+mysqlconnector ou mysql+pymysql
db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# Se usar PyMySQL, use: db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


# --- 4. Criar um objeto SQLDatabase do LangChain ---
# Isso permite que o LangChain "entenda" a estrutura do seu banco de dados
try:
    # Incluímos explicitamente a tabela 'waveforms' para que o agente saiba sobre ela.
    # Isso é útil se o seu DB tiver muitas tabelas.
    db = SQLDatabase.from_uri(db_uri, include_tables=["waveforms"])
    print(f"Conexão com o banco de dados '{db_name}' estabelecida com sucesso.")
    print(f"Tabela 'waveforms' acessível para o agente.")

except Exception as e:
    print(f"Erro ao conectar ao banco de dados ou inicializar SQLDatabase: {e}")
    print("Verifique suas credenciais no .env e se o servidor MySQL está rodando.")
    exit()

# --- 5. Inicializar o LLM (Groq) ---
# Usaremos o mesmo modelo que você usou anteriormente
llm = ChatGroq(temperature=0, model="llama3-70b-8192")
# Se preferir o modelo 8B:
# llm = ChatGroq(temperature=0, model="llama3-8b-8192")

# --- 6. Criar o Agente SQL ---
# O agente usará o LLM para traduzir perguntas em SQL, executar o SQL no 'db' e responder.
# verbose=True mostra os passos (Thought, Action, Action Input)
print("\nInicializando o Agente SQL...")
agent_executor = create_sql_agent(llm, db=db, agent_type="zero-shot-react-description", verbose=True)
print("Agente SQL pronto!")
print("Agora você pode fazer perguntas sobre os dados na tabela 'waveforms'.")
print("Digite 'sair' a qualquer momento para encerrar.")

# --- 7. Loop Interativo para fazer perguntas ---
while True:
    # Pede a entrada do usuário
    question = input("\nSua pergunta sobre os dados: ")

    # Verifica se o usuário quer sair
    if question.lower() == 'sair':
        print("Saindo...")
        break # Sai do loop

    # Verifica se a pergunta não está vazia
    if not question.strip():
        print("Por favor, digite uma pergunta válida.")
        continue # Volta para o início do loop

    # --- 8. Executar o agente com a pergunta ---
    try:
        print(f"\nProcessando pergunta: '{question}'")
        print("Executando agente...")

        # O agente irá pensar, gerar SQL, executar e formular uma resposta.
        response = agent_executor.invoke({"input": question})

        print("\n--- Resposta do Agente ---")
        # A resposta final do agente estará na chave 'output'
        print(response.get('output', 'Não foi possível obter uma resposta.'))

    except Exception as e:
        print(f"\nOcorreu um erro ao processar a pergunta: {e}")
        print("Tente reformular a pergunta ou verifique os logs do agente (se verbose=True).")
        # Erros comuns aqui podem ser SQL inválido gerado pelo LLM ou problemas de conexão.


# Fim do script após sair do loop
print("Programa finalizado.")