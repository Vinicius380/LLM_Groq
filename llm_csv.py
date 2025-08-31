# llm.py

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# === 1. Carrega a chave da API Groq de um arquivo .env ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Erro: GROQ_API_KEY não configurada no .env.")
    exit()
else:
    print(" Chave carregada com sucesso.")

# === 2. Caminho para o arquivo CSV com sinais vibracionais ===
csv_file_path = r"C:\Users\welli\OneDrive\Documentos\GitHub\Projeto_UniSENAI_Ciencia_De_Dados\EDA_Tractian\arquivos\BEARINGS\waveforms_Bearings_inner_raceway_completo.csv"

# === 3. Carrega os dados com pandas para entender a estrutura ===
try:
    df = pd.read_csv(csv_file_path, decimal=",")
    df.drop(columns=["Load_kW"], inplace=True)
    print(f" CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
except Exception as e:
    print(f"Erro ao carregar CSV: {e}")
    exit()

# === 4. Gera um resumo técnico dos dados (head e colunas) ===
colunas = ', '.join(df.columns.tolist())
amostra_head = df.head(3).to_string(index=False)

contexto_tecnico = f"""
Você está interagindo com um agente treinado para analisar sinais de vibração de motores elétricos coletados pela Tractian.

Descrição técnica do CSV:
- Colunas de vibração:
    - 'Ch1 Y-Axis': vibração axial
    - 'Ch2 Y-Axis': vibração vertical
    - 'Ch3 Y-Axis': vibração horizontal
- Cada linha é uma amostra temporal da vibração.
- Os sinais estão em aceleração [g].
- Pode haver colunas de regime operacional: 'Condition', 'RPM', 'Load [kW]'.

Seu objetivo é interpretar esses sinais e apoiar diagnósticos de falhas como:
- Falhas em rolamentos (INNER_RACEWAY, OUTER_RACEWAY)
- Folgas estruturais (STRUCTURAL_LOOSENESS)
- Vibração saudável (HEALTHY)

Estrutura do CSV:
Colunas: {colunas}
Amostra dos dados:
{amostra_head}
"""

# === 5. Inicializa o LLM com Groq ===
llm = ChatGroq(temperature=0, model="llama3-70b-8192")

# === 6. Cria o agente para trabalhar com o CSV ===
agent = create_csv_agent(
    llm=llm,
    path=csv_file_path,
    verbose=True,
    allow_dangerous_code=True
)

# === 7. Passa o contexto técnico para o LLM ===
print(" Inicializando agente com conhecimento técnico...")
agent.invoke(contexto_tecnico)

print(f"\n Agente pronto! Baseado no arquivo: {csv_file_path}")
print("Digite perguntas sobre os dados ou 'sair' para encerrar.")

# === 8. Loop interativo ===
while True:
    pergunta = input("\n Pergunta: ")

    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando...")
        break

    if not pergunta.strip():
        print(" Por favor, digite uma pergunta válida.")
        continue

    try:
        resposta = agent.invoke(pergunta)
        print("\n Resposta:\n")
        print(resposta.get('output', resposta))  # Para casos de retorno puro
    except Exception as e:
        print(f"Erro: {e}")
