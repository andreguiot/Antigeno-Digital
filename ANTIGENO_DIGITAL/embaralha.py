import pandas as pd
import numpy as np

# --- 1. CONFIGURAÇÃO DO ARQUIVO ---
# Coloque aqui o nome do seu arquivo que JÁ CONTÉM os prompts antigos e novos juntos.
caminho_dataset_completo = 'prompts_para_testar.csv' 

# Você pode escolher salvar com o mesmo nome (sobrescrever) ou com um novo nome.
# Usar um novo nome é mais seguro para não perder o original.
caminho_dataset_final_embaralhado = 'dataset_treino_final_embaralhado.csv'

print("--- Iniciando o embaralhamento do dataset de treino ---")

try:
    # --- 2. CARREGAR O DATASET COMPLETO ---
    print(f"\n[PASSO 1/3] Carregando o dataset de '{caminho_dataset_completo}'...")
    df_completo = pd.read_csv(caminho_dataset_completo)
    total_linhas_inicial = len(df_completo)
    print(f"-> Dataset carregado com sucesso. Total de {total_linhas_inicial} linhas.")
    
    # --- 3. EMBARALHAR O DATASET ---
    print("\n[PASSO 2/3] Embaralhando as linhas do dataset...")
    
    # Embaralha todas as linhas do dataframe de forma aleatória
    # frac=1 significa que queremos usar 100% das linhas.
    # reset_index(drop=True) cria um novo índice limpo após o embaralhamento.
    df_embaralhado = df_completo.sample(frac=1).reset_index(drop=True)
    
    print("-> Linhas embaralhadas com sucesso.")

    # --- 4. SALVAR O RESULTADO ---
    print(f"\n[PASSO 3/3] Salvando o dataset final embaralhado em '{caminho_dataset_final_embaralhado}'...")
    
    # Salva o dataframe final em um novo arquivo CSV, sem a coluna de índice
    df_embaralhado.to_csv(caminho_dataset_final_embaralhado, index=False, encoding='utf-8')
    
    print("\n--- Processo Concluído com Sucesso! ---")
    print(f"Seu novo dataset de treino, pronto e embaralhado, está em: '{caminho_dataset_final_embaralhado}'")
    print("Agora, atualize o seu script 'FineTuningModelo.py' para usar este novo arquivo.")

except FileNotFoundError:
    print(f"\nERRO: Arquivo não encontrado!")
    print(f"O arquivo '{caminho_dataset_completo}' não foi localizado. Verifique o nome e o caminho do arquivo e tente novamente.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")