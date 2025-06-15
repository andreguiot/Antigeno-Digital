# detetive.py (Ajustado para também salvar Falsos Negativos)
import csv
import utils_antigeno as utils # Seu arquivo com as funções auxiliares
from transformers import pipeline

def main():
    """Função principal para executar o detetive de prompts e salvar os resultados."""
    print("Olá! Iniciando nosso Detetive de Prompt Injections (com Regras + IA)...")

    # Configurações
    nome_do_modelo_ia = "modelo_detector_injecao_v2" #neuralmind/bert-base-portuguese-cased
    nome_arquivo_csv_padrao = "prompts_para_testar.csv" # CSV de entrada
    nome_coluna_prompt_csv = "prompt_text"
    nome_coluna_esperada_csv = "classificacao_esperada" # Coluna que você adicionará ao seu CSV de entrada
    
    # Nomes dos arquivos de saída
    nome_arquivo_saida_completo_csv = "resultados_detetive_completo.csv"
    nome_arquivo_saida_fn_csv = "falsos_negativos_detectados.csv"

    detector_ia = None
    todos_os_resultados_detalhados = []

    try:
        print(f"Tentando carregar o modelo de IA '{nome_do_modelo_ia}'...")
        print("(Isso pode demorar um pouco, especialmente na primeira vez ou com modelos maiores)")
        detector_ia = pipeline(
            "text-classification",
            model=nome_do_modelo_ia,
            device=0 # Usar CPU = -1. Usar GPU = 0 (Vai mais rápido)
        )
        print(f"Modelo de IA '{nome_do_modelo_ia}' carregado com sucesso!")
    except Exception as e:
        print(f"Xiii! Deu um problema para carregar o modelo de IA: {e}")
        return # Sai da função se o modelo não puder ser carregado

    # Apenas leitura do CSV (simplificado conforme seu último código)
    print(f"\n--- Lendo frases do arquivo CSV: {nome_arquivo_csv_padrao} ---")
    prompts_com_classificacao = [] # Lista de dicionários {'frase': str, 'esperada': str}

    try:
        with open(nome_arquivo_csv_padrao, mode="r", encoding="utf-8", newline='') as arquivo_csv_leitura:
            leitor_csv = csv.DictReader(arquivo_csv_leitura)
            colunas_csv = leitor_csv.fieldnames if leitor_csv.fieldnames else []
            
            if nome_coluna_prompt_csv not in colunas_csv:
                print(f"AVISO (CSV): A coluna de prompt '{nome_coluna_prompt_csv}' não foi encontrada no arquivo '{nome_arquivo_csv_padrao}'.")
                # Não precisa definir frases_do_csv_raw aqui, pois o loop abaixo não executará se não houver prompt
            else:
                linhas_lidas_csv = 0
                for linha_dict in leitor_csv:
                    prompt = linha_dict.get(nome_coluna_prompt_csv, "").strip()
                    esperada = "DESCONHECIDO" # Padrão
                    if nome_coluna_esperada_csv in colunas_csv:
                        esperada = linha_dict.get(nome_coluna_esperada_csv, "DESCONHECIDO").strip()
                        if not esperada:
                            esperada = "DESCONHECIDO"
                    
                    if prompt:
                        prompts_com_classificacao.append({"frase": prompt, "esperada": esperada.upper()}) # Padroniza 'esperada'
                        linhas_lidas_csv += 1
                if linhas_lidas_csv > 0:
                     print(f"Lidas {linhas_lidas_csv} frases (e classificações esperadas, se disponíveis) do CSV.")
                elif nome_coluna_prompt_csv in colunas_csv: # Se a coluna de prompt existe, mas nenhum prompt válido foi lido
                    print(f"Nenhum prompt válido encontrado na coluna '{nome_coluna_prompt_csv}' do CSV.")

    except FileNotFoundError:
        print(f"ERRO (CSV): Arquivo CSV '{nome_arquivo_csv_padrao}' não encontrado.")
        return
    except Exception as e_csv:
        print(f"ERRO (CSV): Erro ao ler arquivo CSV '{nome_arquivo_csv_padrao}': {e_csv}")
        return

    if not prompts_com_classificacao:
        print("\nNenhuma frase foi carregada do CSV para teste. Verifique o arquivo de entrada.")
        return

    frases_para_analise_ia = [item["frase"] for item in prompts_com_classificacao]
    print(f"\nTotal de {len(frases_para_analise_ia)} frases carregadas para análise.") # Removido "ÚNICAS" pois a deduplicação foi removida para simplificar
    print("Analisando as frases uma por uma...")

    resultados_pipeline_ia = [] # Inicializa a lista
    try:
        if frases_para_analise_ia: # Só roda o pipeline se houver frases
            resultados_pipeline_ia = detector_ia(frases_para_analise_ia)
    except Exception as e_pipeline:
        print(f"ERRO ao tentar analisar frases com o pipeline de IA: {e_pipeline}")
        # Decide se quer continuar sem resultados da IA ou parar.
        # Por ora, continuará e os resultados da IA serão de erro.
        # Para forçar parada: return

    print("\n--- O Que os Detetives Descobriram ---")

    for i, item_processado in enumerate(prompts_com_classificacao):
        frase_original = item_processado["frase"]
        classificacao_esperada_original = item_processado["esperada"] # Já está em maiúsculas

        deteccao_por_regra_bool = utils.detectar_por_regras_simples(frase_original)
        resultado_regra_str = "Detectou" if deteccao_por_regra_bool else "Não Detectou"

        resultado_ia_desta_frase = resultados_pipeline_ia[i] if i < len(resultados_pipeline_ia) else {'label': 'ERRO_PIPELINE_IA', 'score': 0.0}
        rotulo_ia = resultado_ia_desta_frase.get('label', 'ERRO_PIPELINE_IA')
        confianca_ia = resultado_ia_desta_frase.get('score', 0.0)

        print(f"\nFrase Testada ({i+1}): '{frase_original}'")
        if deteccao_por_regra_bool:
            print("  Resultado das Regras: POSSÍVEL INJEÇÃO DETECTADA!")
        else:
            print("  Resultado das Regras: Nada suspeito.")
        print(f"  Resultado da IA: {rotulo_ia} (Confiança: {confianca_ia:.2f})")

        # Lógica de decisão combinada para acao_antigeno_final
        if deteccao_por_regra_bool:
            mensagem_final_antigeno = "ALERTA (Regra)"
        elif rotulo_ia == 'LABEL_1':
            mensagem_final_antigeno = "ALERTA (IA)"
        elif rotulo_ia == 'LABEL_0':
            mensagem_final_antigeno = "SEGURO (IA)"
        elif rotulo_ia == 'ERRO_PIPELINE_IA':
            mensagem_final_antigeno = "ERRO (Análise IA)"
        else: # Caso o modelo retorne algum outro rótulo inesperado
            mensagem_final_antigeno = f"DESCONHECIDO (IA: {rotulo_ia})"
        
        print(f"  >> AÇÃO DO ANTÍGENO (Combinado): {mensagem_final_antigeno}")

        todos_os_resultados_detalhados.append({
            'Nº': i + 1,
            'frase_testada': frase_original,
            'classificacao_esperada': classificacao_esperada_original,
            'resultado_regra': resultado_regra_str,
            'resultado_ia_label': rotulo_ia,
            'resultado_ia_confianca': round(confianca_ia, 4),
            'acao_antigeno_final': mensagem_final_antigeno
        })

    # Salvar os resultados completos
    if todos_os_resultados_detalhados:
        try:
            with open(nome_arquivo_saida_completo_csv, mode='w', newline='', encoding='utf-8') as arquivo_saida:
                nomes_colunas = ['Nº', 'frase_testada', 'classificacao_esperada', 
                                 'resultado_regra', 'resultado_ia_label', 
                                 'resultado_ia_confianca', 'acao_antigeno_final']
                escritor_csv = csv.DictWriter(arquivo_saida, fieldnames=nomes_colunas)
                escritor_csv.writeheader()
                escritor_csv.writerows(todos_os_resultados_detalhados)
            print(f"\nResultados detalhados completos salvos em: {nome_arquivo_saida_completo_csv}")
        except IOError:
            print(f"ERRO: Não foi possível escrever no arquivo CSV completo: {nome_arquivo_saida_completo_csv}")
        except Exception as e_csv_save:
            print(f"ERRO inesperado ao salvar o CSV completo: {e_csv_save}")

        # --- NOVO TRECHO: Filtrar e salvar os Falsos Negativos ---
        falsos_negativos = []
        for resultado in todos_os_resultados_detalhados:
            # Um Falso Negativo ocorre se:
            # 1. A classificação esperada era INJECAO
            # 2. A ação final do antígeno NÃO foi um ALERTA (ou seja, foi SEGURO, ERRO ou DESCONHECIDO)
            if resultado['classificacao_esperada'] == "INJECAO" and \
               not resultado['acao_antigeno_final'].startswith("ALERTA"):
                falsos_negativos.append(resultado)
        
        if falsos_negativos:
            try:
                with open(nome_arquivo_saida_fn_csv, mode='w', newline='', encoding='utf-8') as arquivo_fn:
                    # Usar as mesmas colunas do arquivo completo para consistência
                    nomes_colunas_fn = ['Nº', 'frase_testada', 'classificacao_esperada', 
                                      'resultado_regra', 'resultado_ia_label', 
                                      'resultado_ia_confianca', 'acao_antigeno_final']
                    escritor_fn_csv = csv.DictWriter(arquivo_fn, fieldnames=nomes_colunas_fn)
                    escritor_fn_csv.writeheader()
                    escritor_fn_csv.writerows(falsos_negativos)
                print(f"Falsos Negativos detectados ({len(falsos_negativos)}) salvos em: {nome_arquivo_saida_fn_csv}")
            except IOError:
                print(f"ERRO: Não foi possível escrever no arquivo CSV de Falsos Negativos: {nome_arquivo_saida_fn_csv}")
            except Exception as e_fn_save:
                print(f"ERRO inesperado ao salvar o CSV de Falsos Negativos: {e_fn_save}")
        else:
            if any(r['classificacao_esperada'] == "INJECAO" for r in todos_os_resultados_detalhados):
                print("\nNenhum Falso Negativo detectado para salvar (todas as injeções esperadas foram alertadas).")
            else:
                print("\nNenhum Falso Negativo detectado para salvar (não havia injeções esperadas ou os resultados não puderam ser processados).")
        # --- FIM DO NOVO TRECHO ---

    else:
        print("\nNenhum resultado detalhado para salvar.")

    print("\n--- Fim da demonstração do Detetive de Prompts ---")

if __name__ == "__main__":
    main()