# utils_antigeno.py (sem mais palavras-chave)

import csv # Mantido caso você tenha outras funções usando csv aqui

# Se você tiver outras funções utilitárias neste arquivo, elas devem permanecer aqui.

# utils_antigeno.py (com as regras desativadas)

def detectar_por_regras_simples(frase: str) -> bool:
    """
    Função de regras manuais. DESATIVADA para priorizar o modelo de IA.
    Retorna sempre False para que a decisão seja 100% da IA.
    """
    return False



def obter_escolha_fonte_dados():
    """Pergunta ao usuário de onde ler os prompts."""
    print("\nDe onde você gostaria de ler os prompts para teste?")
    print("  1: Apenas do arquivo CSV (chamado 'prompts_para_testar.csv')")
    print("  2: Apenas do arquivo TXT (chamado 'prompts_para_testar.txt')")
    print("  3: De AMBOS os arquivos (CSV e TXT)")
    while True:
        escolha = input("Digite o número da sua escolha (1, 2 ou 3): ").strip()
        if escolha in ['1', '2', '3']:
            return escolha
        else:
            print("Escolha inválida. Por favor, digite 1, 2 ou 3.")

def ler_prompts_de_txt(nome_arquivo_txt: str) -> list:
    """Lê prompts de um arquivo TXT, um por linha, removendo linhas vazias."""
    prompts_lidos = []
    try:
        with open(nome_arquivo_txt, "r", encoding="utf-8") as arquivo_txt:
            for linha_do_arquivo in arquivo_txt:
                frase_limpa = linha_do_arquivo.strip()
                if frase_limpa: # Adiciona apenas se a linha não estiver vazia após strip
                    prompts_lidos.append(frase_limpa)
        if prompts_lidos:
            print(f"Lidas {len(prompts_lidos)} frases do arquivo TXT '{nome_arquivo_txt}'.")
        else:
            print(f"Nenhuma frase válida encontrada no arquivo TXT '{nome_arquivo_txt}'.")
    except FileNotFoundError:
        print(f"AVISO (TXT): Arquivo TXT '{nome_arquivo_txt}' não encontrado.")
    except Exception as e:
        print(f"AVISO (TXT): Ocorreu um erro ao ler o arquivo TXT '{nome_arquivo_txt}': {e}")
    return prompts_lidos

# Se você tiver uma função `ler_prompts_de_csv` aqui, ela também deve ser mantida.
# Exemplo de como ela poderia ser (apenas para contexto, adapte à sua original):
def ler_prompts_de_csv(nome_arquivo_csv: str, nome_coluna_prompt: str, nome_coluna_esperada: str = None) -> list:
    """
    Lê prompts e opcionalmente a classificação esperada de um arquivo CSV.
    Retorna uma lista de dicionários: [{'frase': str, 'esperada': str}]
    """
    dados_lidos = []
    try:
        with open(nome_arquivo_csv, mode="r", encoding="utf-8", newline='') as arquivo_csv_leitura:
            leitor_csv = csv.DictReader(arquivo_csv_leitura)
            if nome_coluna_prompt not in leitor_csv.fieldnames:
                print(f"AVISO (CSV): A coluna de prompt '{nome_coluna_prompt}' não foi encontrada.")
                return dados_lidos
            
            for linha_dict in leitor_csv:
                prompt = linha_dict.get(nome_coluna_prompt, "").strip()
                esperada = "DESCONHECIDO"
                if nome_coluna_esperada and nome_coluna_esperada in leitor_csv.fieldnames:
                    esperada = linha_dict.get(nome_coluna_esperada, "DESCONHECIDO").strip()
                    if not esperada:
                        esperada = "DESCONHECIDO"
                
                if prompt:
                    dados_lidos.append({"frase": prompt, "esperada": esperada.upper()})
            
        if dados_lidos:
            print(f"Lidas {len(dados_lidos)} linhas do arquivo CSV '{nome_arquivo_csv}'.")
        else:
            print(f"Nenhuma linha válida lida do arquivo CSV '{nome_arquivo_csv}'.")
            
    except FileNotFoundError:
        print(f"AVISO (CSV): Arquivo CSV '{nome_arquivo_csv}' não encontrado.")
    except Exception as e:
        print(f"AVISO (CSV): Ocorreu um erro ao ler o arquivo CSV '{nome_arquivo_csv}': {e}")
    return dados_lidos
