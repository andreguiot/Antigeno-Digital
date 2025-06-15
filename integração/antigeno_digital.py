# antigeno_digital.py
from classifier.model import analisar_prompt_pela_ia # Importa a função que retorna o dicionário
from utils_antigeno import detectar_por_regras_simples
from groq_client import query_groq
# import time # Descomente se quiser medir o tempo

# LIMIAR_PROB_INJECAO_BLOQUEIO = 0.7 # Exemplo: bloquear se prob. de injeção >= 70%
# Ajuste este limiar conforme os resultados e sua tolerância a falsos positivos/negativos.
# Um limiar mais alto = menos bloqueios (mais tolerante), mais baixo = mais bloqueios (mais sensível).
LIMIAR_PROB_INJECAO_BLOQUEIO = 0.5 # Começar com 0.5 pode ser um bom ponto de partida

def obter_analise_antigeno(prompt_usuario: str) -> dict:
    """
    Analisa o prompt do usuário usando regras e o modelo de IA.
    Retorna um dicionário com:
        - "classificacao_final": "SEGURO" ou "INJECAO"
        - "prob_injecao": float (estimativa da probabilidade de ser uma injeção)
        - "motivo_deteccao": str (origem da decisão: REGRAS, IA, ou ERRO)
        - "detalhes_ia": dict (o resultado bruto de analisar_prompt_pela_ia, para log/debug)
    """
    # start_time = time.time() # Descomente para medir o tempo

    # 1. Verificar por Regras Simples (prioridade máxima)
    if detectar_por_regras_simples(prompt_usuario):
        # end_time = time.time() # Descomente para medir o tempo
        return {
            "classificacao_final": "INJECAO",
            "prob_injecao": 1.0,  # Regras são consideradas detecção com 100% de certeza
            "motivo_deteccao": "REGRAS_MANUAIS",
            "detalhes_ia": None, # IA não foi consultada ou não é o motivo primário
            # "tempo_analise_ms": round((end_time - start_time) * 1000, 2) # Descomente
        }

    # 2. Se não pego por regras, consultar a IA 
    analise_ia = analisar_prompt_pela_ia(prompt_usuario)
    #DESATIVEI ISSO MAS N VOU TIRAR 
    # analise_ia é um dict: {"label_ia": "SEGURO"|"INJECAO"|ERRO_..., "score_ia": float}

    label_da_ia = analise_ia.get("label_ia", "ERRO_INESPERADO_IA")
    score_da_ia = analise_ia.get("score_ia", 0.0)
    
    probabilidade_estimada_injecao = 0.0
    classificacao_derivada_ia = "SEGURO" # Padrão

    if label_da_ia == "INJECAO":
        probabilidade_estimada_injecao = score_da_ia
        classificacao_derivada_ia = "INJECAO"
    elif label_da_ia == "SEGURO":
        # Se a IA diz "SEGURO" com score S, a probabilidade de ser "INJECAO" é (1-S)
        # Isso assume que o modelo é binário e os scores somam 1 para as duas classes.
        # Se o score_da_ia já é P(SEGURO), então P(INJECAO) = 1 - P(SEGURO)
        probabilidade_estimada_injecao = 1.0 - score_da_ia
        classificacao_derivada_ia = "SEGURO" # Mantém, pois a IA primariamente disse SEGURO
                                          # A prob_injecao será usada para o limiar.
    elif label_da_ia.startswith("ERRO_"):
        # Em caso de erro da IA, podemos assumir uma probabilidade de injeção conservadora (ex: 0.0 ou 0.5)
        # ou até mesmo tratar como bloqueio dependendo da política de falha segura.
        # Por ora, vamos assumir que não é uma injeção detectada pela IA.
        probabilidade_estimada_injecao = 0.0 # Falha segura (não bloqueia por erro da IA)
        classificacao_derivada_ia = "SEGURO" # Ou um estado de erro específico
        # end_time = time.time() # Descomente para medir o tempo
        return {
            "classificacao_final": "SEGURO", # Ou "ERRO_ANTIGENO"
            "prob_injecao": probabilidade_estimada_injecao,
            "motivo_deteccao": f"IA_RETORNOU_ERRO: {label_da_ia}",
            "detalhes_ia": analise_ia,
            # "tempo_analise_ms": round((end_time - start_time) * 1000, 2) # Descomente
        }
    else: # Label desconhecido ou não mapeado
        probabilidade_estimada_injecao = 0.0
        classificacao_derivada_ia = "SEGURO"
        # end_time = time.time() # Descomente para medir o tempo
        return {
            "classificacao_final": "SEGURO", # Ou "ANTIGENO_IA_DESCONHECIDO"
            "prob_injecao": probabilidade_estimada_injecao,
            "motivo_deteccao": f"IA_LABEL_DESCONHECIDO: {label_da_ia}",
            "detalhes_ia": analise_ia,
            # "tempo_analise_ms": round((end_time - start_time) * 1000, 2) # Descomente
        }

    # A classificação final do Antígeno será baseada na prob_injecao e no limiar
    # Se a prob_injecao (derivada da IA) ultrapassar o limiar, consideramos INJECAO.
    # Mesmo que a IA tenha classificado primariamente como "SEGURO", se (1-score_ia) for alto.
    classificacao_antigeno_final = "SEGURO"
    if probabilidade_estimada_injecao >= LIMIAR_PROB_INJECAO_BLOQUEIO:
        classificacao_antigeno_final = "INJECAO"
    
    # end_time = time.time() # Descomente para medir o tempo
    return {
        "classificacao_final": classificacao_antigeno_final,
        "prob_injecao": float(probabilidade_estimada_injecao),
        "motivo_deteccao": f"IA_MODELO_FINETUNADO (Label IA: {label_da_ia}, Score IA: {score_da_ia:.4f})",
        "detalhes_ia": analise_ia,
        # "tempo_analise_ms": round((end_time - start_time) * 1000, 2) # Descomente
    }


def main():
    print("Bem-vindo ao Antígeno Digital (Modo de Teste Interativo)!")
    print("O modelo de IA será carregado na primeira análise de prompt.")
    print(f"Limiar de probabilidade de injeção para bloqueio: {LIMIAR_PROB_INJECAO_BLOQUEIO*100:.0f}%")
    
    # Chama uma vez para "aquecer" e carregar o modelo, se desejado.
    # obter_analise_antigeno("Teste inicial para carregar modelo.") 
    # print("Modelos pré-carregados (se aplicável).\n")


    while True:
        prompt_usuario = input("\nDigite um prompt (ou 'sair' para encerrar): ")
        if prompt_usuario.lower() == "sair":
            print("Encerrando.")
            break
        if not prompt_usuario.strip():
            print("Prompt vazio, por favor digite algo.")
            continue

        analise_completa = obter_analise_antigeno(prompt_usuario)
        
        classificacao = analise_completa["classificacao_final"]
        prob_injecao = analise_completa["prob_injecao"]
        motivo = analise_completa["motivo_deteccao"]
        # detalhes_ia_log = analise_completa["detalhes_ia"] # Para log mais detalhado se precisar
        # tempo_ms = analise_completa.get("tempo_analise_ms", "N/A")

        print("\n--- Análise do Antígeno Digital ---")
        print(f"  Prompt: '{prompt_usuario}'")
        # print(f"  Tempo de Análise: {tempo_ms} ms") # Descomente se medir tempo
        print(f"  Classificação Final do Antígeno: {classificacao}")
        print(f"  Probabilidade Estimada de Injeção: {prob_injecao:.2%}") # Mostra como porcentagem
        print(f"  Motivo/Detalhe da Detecção: {motivo}")
        # print(f"  Detalhes IA (bruto): {detalhes_ia_log}") # Para debug
        print("---------------------------------")
        
        if classificacao == "INJECAO": # A decisão de bloqueio agora é direta pela classificacao_final
            print("------------------------------------------------------")
            print(">>> ALERTA DO ANTÍGENO DIGITAL <<<")
            print(f"Motivo da Detecção: {motivo}")
            print(f"Probabilidade de Injeção: {prob_injecao:.2%}")
            print("Ação: Prompt BLOQUEADO.")
            print("------------------------------------------------------")
        elif classificacao == "SEGURO":
            print("Antígeno Digital: Prompt parece seguro. Enviando para o modelo principal (Groq)...")
            try:
                resposta_groq = query_groq(prompt_usuario)
                print("\nResposta do Modelo Principal (Groq):")
                print(resposta_groq)
            except Exception as e_groq:
                print(f"Erro ao obter resposta do modelo Groq: {e_groq}")
        else: # Casos de erro do Antígeno, como ERRO_ANTIGENO, ANTIGENO_IA_DESCONHECIDO
            print("------------------------------------------------------")
            print(">>> ATENÇÃO - ANTÍGENO DIGITAL <<<")
            print(f"Houve um problema na análise do Antígeno: {classificacao}")
            print(f"Motivo: {motivo}")
            print("Ação: Por segurança, o prompt NÃO será enviado ao modelo principal.")
            print("------------------------------------------------------")
        print("=================================\n")

if __name__ == "__main__":
    main()
