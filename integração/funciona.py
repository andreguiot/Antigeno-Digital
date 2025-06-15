import os
from groq import Groq, APIConnectionError, APIStatusError

prompt = input()

# Recuperando a chave da API
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("A variável de ambiente GROQ_API_KEY não está definida.")

# Instanciando o cliente Groq (sem base_url!)
client = Groq(api_key=api_key)

try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":f"{prompt}"            
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    print(chat_completion.choices[0].message.content)


except APIConnectionError as e:
    print(f"Erro de conexão com a API: {e}")
except APIStatusError as e:
    print(f"Erro de status da API: {e.status_code} - {e.response}")
except Exception as e:
    print(f"Um erro inesperado ocorreu: {e}")