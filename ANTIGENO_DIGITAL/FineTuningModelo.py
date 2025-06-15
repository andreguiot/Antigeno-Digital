import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, ClassLabel
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
os.environ["WANDB_DISABLED"] = "true"


# --- 1. Configurações Iniciais ---
CAMINHO_DO_SEU_CSV = "ANTIGENO_DIGITAL/dataset_treino_final_embaralhado.csv" # Certifique-se que é o CSV de TREINO
COLUNA_TEXTO = "prompt_text"
COLUNA_LABEL_ORIGINAL = "classificacao_esperada"

MODELO_BASE = "ANTIGENO_DIGITAL/modelo_detector_injecao_finetunado"
NOME_DO_MODELO_FINETUNADO = "ANTIGENO_DIGITAL/modelo_detector_injecao_v2"

NUM_EPOCHS_TREINO = 3
TAMANHO_BATCH_TREINO = 16
TAMANHO_BATCH_AVALIACAO = 64
TAXA_APRENDIZAGEM = 2e-5
PROPORCAO_TESTE = 0.1
PROPORCAO_VALIDACAO = 0.1 # Esta proporção é sobre o conjunto de treino+validação após remover o de teste

# --- 2. Carregar e Preparar o Dataset ---
print(f"Carregando dataset de: {CAMINHO_DO_SEU_CSV}")
try:
    df_completo = pd.read_csv(CAMINHO_DO_SEU_CSV, encoding="utf-8")
    print(f"Dataset carregado com {len(df_completo)} linhas.")
except FileNotFoundError:
    print(f"ERRO: Arquivo CSV '{CAMINHO_DO_SEU_CSV}' não encontrado. Verifique o caminho.")
    exit()
except Exception as e:
    print(f"ERRO ao carregar o CSV: {e}")
    exit()

if COLUNA_TEXTO not in df_completo.columns or COLUNA_LABEL_ORIGINAL not in df_completo.columns:
    print(f"ERRO: As colunas '{COLUNA_TEXTO}' ou '{COLUNA_LABEL_ORIGINAL}' não foram encontradas no CSV.")
    print(f"Colunas encontradas: {df_completo.columns.tolist()}")
    exit()

print(f"Limpando e mapeando labels da coluna '{COLUNA_LABEL_ORIGINAL}'...")
COLUNA_LABEL_NUMERICA = 'label_numerica_mapeada'
df_completo[COLUNA_LABEL_ORIGINAL] = df_completo[COLUNA_LABEL_ORIGINAL].astype(str).str.strip().str.upper()
df_completo[COLUNA_LABEL_ORIGINAL] = df_completo[COLUNA_LABEL_ORIGINAL].str.replace('\xa0', '', regex=False)
df_completo[COLUNA_LABEL_ORIGINAL] = df_completo[COLUNA_LABEL_ORIGINAL].str.strip()
df_completo[COLUNA_LABEL_ORIGINAL] = df_completo[COLUNA_LABEL_ORIGINAL].replace({'SEGUR': 'SEGURO'}) # Corrige erro de digitação comum
label_map = {"SEGURO": 0, "INSEGURO": 1, "INJECAO": 1} # Mapeia para 0 e 1
df_completo[COLUNA_LABEL_NUMERICA] = df_completo[COLUNA_LABEL_ORIGINAL].map(label_map)

if df_completo[COLUNA_LABEL_NUMERICA].isnull().any():
    print(f"ERRO: Mesmo após a limpeza, alguns valores na coluna '{COLUNA_LABEL_ORIGINAL}' não puderam ser mapeados para 0 ou 1.")
    print("Valores únicos na sua coluna de label original (APÓS LIMPEZA E UPPERSCASING):")
    print(df_completo[COLUNA_LABEL_ORIGINAL].unique())
    print("Valores problemáticos (que resultaram em NaN após o mapeamento):")
    print(df_completo[df_completo[COLUNA_LABEL_NUMERICA].isnull()][COLUNA_LABEL_ORIGINAL].unique())
    exit()
print(f"Mapeamento concluído. Usando a nova coluna '{COLUNA_LABEL_NUMERICA}' para os labels.")

df_para_dataset = df_completo[[COLUNA_TEXTO, COLUNA_LABEL_NUMERICA]].rename(
    columns={COLUNA_TEXTO: 'text', COLUNA_LABEL_NUMERICA: 'label'}
)
df_para_dataset.dropna(subset=['label'], inplace=True) # Remove linhas onde o label é NaN
df_para_dataset['label'] = df_para_dataset['label'].astype(int) # Converte labels para inteiro

print(f"Labels encontrados no dataset (após processamento e conversão para int): {np.unique(df_para_dataset['label'])}")
if not all(label_val in [0, 1] for label_val in np.unique(df_para_dataset['label'])): # Renomeada variável para evitar conflito
    print("ERRO: Os labels finais devem ser apenas 0 e 1. Verifique o mapeamento e sua coluna de labels original.")
    exit()

print("Dividindo dataset em treino, validação e teste...")
df_treino_val, df_teste = train_test_split(
    df_para_dataset, test_size=PROPORCAO_TESTE, random_state=42, stratify=df_para_dataset['label']
)
# Ajustar proporção de validação para ser sobre o conjunto treino_val
df_treino, df_validacao = train_test_split(
    df_treino_val, test_size=PROPORCAO_VALIDACAO / (1.0 - PROPORCAO_TESTE), 
    random_state=42, stratify=df_treino_val['label']
)
print(f"Tamanho do dataset de Treino: {len(df_treino)}")
print(f"Tamanho do dataset de Validação: {len(df_validacao)}")
print(f"Tamanho do dataset de Teste: {len(df_teste)}")

features = Dataset.from_pandas(df_treino).features.copy()
features['label'] = ClassLabel(num_classes=2, names=['SEGURO', 'INJECAO'])
train_dataset = Dataset.from_pandas(df_treino, features=features)
val_dataset = Dataset.from_pandas(df_validacao, features=features)
test_dataset = Dataset.from_pandas(df_teste, features=features)
dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
print("\nDataset preparado e dividido:")
print(dataset_dict)

print(f"\nCarregando tokenizer e modelo base: {MODELO_BASE}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_BASE, num_labels=2)
except Exception as e:
    print(f"ERRO ao carregar modelo ou tokenizer: {e}")
    exit()

print("\nTokenizando os datasets...")
def tokenizar_funcao(exemplos):
    return tokenizer(exemplos["text"], padding="max_length", truncation=True, max_length=512)
dataset_tokenizado = dataset_dict.map(tokenizar_funcao, batched=True)
print("Tokenização completa.")

# --- 5. Definir Argumentos de Treinamento (VERSÃO MINIMALISTA EXTREMA) ---
print("\nDefinindo argumentos de treinamento (versão minimalista extrema)...")

training_args = TrainingArguments(
    output_dir=f"{NOME_DO_MODELO_FINETUNADO}_checkpoints_minimo", # Diretório para salvar
    num_train_epochs=NUM_EPOCHS_TREINO,
    per_device_train_batch_size=TAMANHO_BATCH_TREINO,
    # per_device_eval_batch_size=TAMANHO_BATCH_AVALIACAO, # Comentado para simplificar
    learning_rate=TAXA_APRENDIZAGEM,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_treino_antigeno_minimo',
    logging_steps=100, # Aumentar logging_steps se a avaliação for removida

    # Removendo todos os parâmetros de estratégia de avaliação e salvamento problemáticos
    # Se load_best_model_at_end der erro sem avaliação, comente também
    # load_best_model_at_end=False, # Desabilitar se causar erro sem avaliação
    # save_strategy="no", # Se a versão suportar, para não salvar checkpoints intermediários
)
# --- FIM DA VERSÃO MINIMALISTA EXTREMA ---

# ... e na inicialização do Trainer, se a avaliação estiver a causar problemas:
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_tokenizado["train"],
#     # eval_dataset=dataset_tokenizado["validation"], # COMENTE ESTA LINHA TEMPORARIAMENTE
#     tokenizer=tokenizer,
#     # compute_metrics=compute_metrics # E ESTA TAMBÉM SE eval_dataset FOR COMENTADO
# )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) # Para evitar erro se só houver uma classe
    return {
        'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
        'tp (injecao_detectada)': int(tp), 'tn (seguro_correto)': int(tn),
        'fp (seguro_como_injecao)': int(fp), 'fn (injecao_como_seguro)': int(fn)
    }

print("\nInicializando o Trainer...")

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=dataset_tokenizado["train"], eval_dataset=dataset_tokenizado["validation"],
    tokenizer=tokenizer, compute_metrics=compute_metrics
)

print("\nIniciando o treinamento do modelo...")
print(f"Usando dispositivo: {trainer.args.device}")
if torch.cuda.is_available():
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
else:
    print("AVISO: Nenhuma GPU detectada. O treinamento será em CPU e pode ser MUITO lento.")

try:
    trainer.train()
    print("Treinamento concluído!")
except Exception as e_train:
    print(f"ERRO durante o treinamento: {e_train}")
    exit()

print(f"\nSalvando o modelo fine-tunado em: {NOME_DO_MODELO_FINETUNADO}")
try:
    trainer.save_model(NOME_DO_MODELO_FINETUNADO)
    tokenizer.save_pretrained(NOME_DO_MODELO_FINETUNADO)
    print("Modelo e tokenizer salvos com sucesso.")
except Exception as e_save:
    print(f"ERRO ao salvar o modelo: {e_save}")

if "test" in dataset_tokenizado and len(dataset_tokenizado["test"]) > 0:
    print("\nIniciando avaliação no conjunto de teste...")
    try:
        test_results = trainer.evaluate(eval_dataset=dataset_tokenizado["test"])
        print("\n--- Resultados da Avaliação no Conjunto de Teste ---")
        for key, value in test_results.items():
            print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}" if isinstance(value, float) else f"{key.replace('eval_', '').capitalize()}: {value}")
    except Exception as e_eval:
        print(f"ERRO durante a avaliação no conjunto de teste: {e_eval}")
else:
    print("\nConjunto de teste não disponível ou vazio. Avaliação final não realizada.")

print("\n--- Processo de Fine-tuning Concluído ---")
print(f"Seu modelo fine-tunado está salvo em: {NOME_DO_MODELO_FINETUNADO}")
print("Para usá-lo, carregue-o com AutoModelForSequenceClassification.from_pretrained('./caminho/para/seu_modelo')")

