# groq_client.py
from groq import Groq, APIConnectionError, APIStatusError
from config import GROQ_API_KEY, GROQ_MODEL # GROQ_MODEL aqui é o que você configurou, ex: "llama3-8b-8192"
import os

# Instancia o cliente Groq uma vez quando o módulo é importado.
# É importante que GROQ_API_KEY seja carregado corretamente do seu .env via config.py
if not GROQ_API_KEY:
    print("ERRO CRÍTICO (Groq Client): GROQ_API_KEY não foi carregada. Verifique seu .env e config.py.")
    # Você pode querer levantar um erro aqui ou ter um cliente "dummy" que sempre falha.
    CLIENT_GROQ = None
else:
    try:
        CLIENT_GROQ = Groq(api_key=GROQ_API_KEY)
        print("Cliente Groq inicializado com sucesso.")
    except Exception as e_init_groq:
        print(f"ERRO CRÍTICO ao inicializar o cliente Groq: {e_init_groq}")
        CLIENT_GROQ = None


def query_groq(prompt: str) -> str:
    """
    Envia um prompt para a API da Groq e retorna a resposta do modelo.
    """
    if CLIENT_GROQ is None:
        return "ERRO: Cliente Groq não inicializado. Verifique a API Key."

    try:
        # print(f"DEBUG (Groq): Enviando prompt: '{prompt[:50]}...' para o modelo: {GROQ_MODEL}")
        chat_completion = CLIENT_GROQ.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL, # Usa o modelo definido em config.py
            temperature=0.7,
            # max_tokens=1024, # Opcional: defina um limite de tokens
            # top_p=1, # Opcional
            # stop=None, # Opcional: sequências para parar a geração
        )
        resposta = chat_completion.choices[0].message.content.strip()
        # print(f"DEBUG (Groq): Resposta recebida: '{resposta[:50]}...'")
        return resposta
    
    except APIConnectionError as e_conn:
        print(f"Erro de Conexão com API Groq: {e_conn}")
        return f"Erro de Conexão com API Groq: {e_conn}"
    except APIStatusError as e_stat:
        print(f"Erro de Status da API Groq: Status {e_stat.status_code}, Response: {e_stat.response}")
        return f"Erro de Status da API Groq: Status {e_stat.status_code}"
    except Exception as e:
        print(f"Erro inesperado ao consultar a API Groq: {e}")
        return f"Erro inesperado na API Groq: {e}"

# Teste simples (opcional)
if __name__ == '__main__':
    print("Testando groq_client.py...")
    if CLIENT_GROQ:
        prompt_teste = "Qual a capital do Brasil e por que ela foi planejada?"
        print(f"Enviando prompt de teste para Groq: '{prompt_teste}'")
        resposta_teste = query_groq(prompt_teste)
        print("\nResposta do Groq:")
        print(resposta_teste)
        
        prompt_teste_2 = "Conte uma piada curta sobre programação."
        print(f"\nEnviando segundo prompt de teste para Groq: '{prompt_teste_2}'")
        resposta_teste_2 = query_groq(prompt_teste_2)
        print("\nResposta do Groq:")
        print(resposta_teste_2)
    else:
        print("Não foi possível testar o cliente Groq pois ele não foi inicializado (verifique API Key).")

