# 🛡️ Antígeno Digital: Detector de Prompt Injection para LLMs

**Antígeno Digital** é um sistema de segurança baseado em Inteligência Artificial, projetado para atuar como uma camada de proteção para Modelos de Linguagem de Grande Escala (LLMs). Seu objetivo é detectar e neutralizar, em tempo real, tentativas maliciosas de _prompt injection_ e ataques de engenharia social, garantindo o uso seguro e ético dos LLMs.

O projeto foi desenvolvido com foco especial na língua portuguesa, resultando em um modelo fine-tunado com desempenho superior a **98% de _Recall_** na detecção de ameaças.

---

## ✨ Funcionalidades Principais

- **🎯 Detecção de Alta Performance**  
  Utiliza um modelo BERTimbau fine-tunado, com _Recall_ superior a 98% na identificação de prompts maliciosos.

- **🗣️ Especialista em Português**  
  Treinado com mais de 1.200 exemplos em português, o modelo compreende o contexto e nuances do idioma.

- **🛡️ Defesa Dupla**  
  Detecta tanto injeções técnicas (comandos manipulativos) quanto engenharia social (pedidos disfarçados de conteúdo nocivo).

- **🧠 Arquitetura "AI-First"**  
  Após testes, o sistema abandonou regras manuais e passou a confiar 100% no modelo de IA, reduzindo falsos positivos e aumentando a precisão.

- **🤖 Integração com Discord**  
  Implementa um bot funcional no Discord, que filtra mensagens maliciosas antes de consultar um LLM principal (via Groq).

---

## 🚀 Jornada do Projeto

**Fase 1 — Análise de Viabilidade**  
Testes com modelos genéricos (ex: Llama-Guard-2) apresentaram _Recall_ < 35% em português, indicando a necessidade de uma solução dedicada.

**Fase 2 — Sistema Híbrido**  
Combinação de IA genérica com regras manuais. Embora eficaz contra ataques óbvios, mostrou-se frágil contra tentativas mais sutis.

**Fase 3 — O Ponto de Virada (Fine-Tuning)**  
Fine-tuning do BERTimbau com dataset customizado elevou o _Recall_ para mais de 98%, validando a abordagem especializada.

**Fase 4 — Arquitetura AI-First**  
Com o modelo superando as regras manuais, o sistema foi refatorado para confiar exclusivamente na IA treinada.

---

## 🛠️ Tecnologias Utilizadas

# **Linguagem:** Python 3.9+

Hugging Face Transformers

Hugging Face Datasets

Scikit-learn

Pandas


**Machine Learning:**


- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas

**Integração:**

- `discord.py` (bot Discord)
- # `groq` (cliente do LLM principal)
  discord.py (bot Discord)

groq (cliente do LLM principal)

Controle de Versão:


**Controle de Versão:**


- Git & GitHub
- Git LFS (para os arquivos de modelo)

---

## 📂 Estrutura do Repositório

```
=======
Git LFS (para os arquivos de modelo)

📂 Estrutura do Repositório
bash
Copiar
Editar
.
├── ANTIGENO_DIGITAL/
│   ├── modelo_detector_injecao_v2/     # Modelo fine-tunado (via Git LFS)
│   ├── FineTuningModelo.py             # Script de treinamento
│   └── ...                             # Datasets e análises
└── integração/
    ├── discord_bot.py                  # Bot principal do Discord
    ├── antigeno_digital.py             # Módulo de análise de segurança
    └── classifier/                     # Carregamento e inferência do modelo
```

---

## ⚙️ Instalação e Execução

### 1. Clone o Repositório

> Este projeto utiliza Git LFS para armazenar o modelo. Certifique-se de tê-lo instalado.

```bash
=======
⚙️ Instalação e Execução
1. Clone o Repositório
Este projeto utiliza Git LFS para armazenar o modelo. Certifique-se de tê-lo instalado.

bash
Copiar
Editar
git lfs install
git clone https://github.com/andreguiot/antigeno-digital-detector.git
cd antigeno-digital-detector
git lfs pull
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
```

Ative o ambiente:

- **Windows:** `venv\Scripts\activate`
- **Linux/macOS:** `source venv/bin/activate`

### 3. Instale as Dependências

Crie um arquivo `requirements.txt` com:

```
=======
2. Crie um Ambiente Virtual
bash
Copiar
Editar
python -m venv venv
Ative o ambiente:

Windows: venv\Scripts\activate

Linux/macOS: source venv/bin/activate

3. Instale as Dependências
Crie um arquivo requirements.txt com:

nginx
Copiar
Editar
torch
transformers
datasets
scikit-learn
pandas
discord.py
python-dotenv
groq
```

Instale com:

```bash
pip install -r requirements.txt
```

### 4. Configure as Chaves de API

Crie um arquivo `.env` dentro da pasta `integração/`:

```env
DISCORD_BOT_TOKEN="SEU_TOKEN_DISCORD"
GROQ_API_KEY="SUA_CHAVE_GROQ"
```

O script `config.py` já está pronto para ler essas variáveis.

### 5. Execute o Bot

```bash
python integração/discord_bot.py
```

O bot entrará online no seu servidor e responderá a comandos iniciados com `!antigeno`.

---

## 🔮 Próximos Passos

- **🧩 Melhor detecção de eufemismos**  
  Expandir o dataset com linguagem codificada (ex: “saco de carne de 70kg”).

- **🚫 Detecção de conteúdo ilegal**  
  Adicionar mais exemplos de solicitações prejudiciais explícitas.

- **🎯 Redução de falsos positivos**  
  Incluir exemplos seguros com palavras sensíveis como “prompt”.

---

## 👨‍💻 Autoria

**Desenvolvido por:**  
Equipe IEEE

- André Guiot – [@andreguiot](https://github.com/andreguiot)
- Matheus Dinis – [@Madinni](https://github.com/Madinni)

