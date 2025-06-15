# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Isso carrega as variáveis do arquivo .env para o ambiente

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" # ou o modelo que você está usando

# Linha crucial para o bot do Discord:
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Opcional: Adicione estas verificações para te ajudar a depurar
if GROQ_API_KEY is None:
    print("AVISO (config.py): GROQ_API_KEY não foi encontrado nas variáveis de ambiente.")
if DISCORD_BOT_TOKEN is None:
    print("AVISO (config.py): DISCORD_BOT_TOKEN não foi encontrado nas variáveis de ambiente.")
    print("Certifique-se que seu arquivo .env contém a linha: DISCORD_BOT_TOKEN='seu_token_aqui'")