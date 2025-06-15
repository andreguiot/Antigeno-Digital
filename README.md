🛡️ Antígeno Digital: Detector de Prompt Injection para LLMs
Antígeno Digital é um sistema de segurança baseado em Inteligência Artificial, projetado para atuar como uma camada de proteção para Modelos de Linguagem de Grande Escala (LLMs). O seu objetivo principal é detetar e neutralizar, em tempo real, tentativas maliciosas de injeção de prompt (prompt injection) e de engenharia social, garantindo a integridade e o uso seguro dos LLMs.

O projeto foi desenvolvido com um foco especial nas nuances da língua portuguesa, culminando num modelo de IA fine-tunado com uma performance de mais de 98% de Recall na deteção de ameaças.

✨ Funcionalidades Principais
Deteção de Alta Performance: Utiliza um modelo BERT (BERTimbau) fine-tunado, alcançando mais de 98% de sensibilidade (Recall) na identificação de prompts maliciosos.

Especialista em Português: Treinado com um dataset customizado de mais de 1.200 exemplos em português, o modelo é capaz de compreender o contexto e as subtilezas do idioma.

Defesa Dupla: Deteta tanto injeções técnicas (comandos para manipular o comportamento do LLM) quanto engenharia social (pedidos de conteúdo prejudicial disfarçados).

Arquitetura "AI-First": Após um processo iterativo, o sistema evoluiu para confiar 100% no modelo de IA, abandonando regras manuais para uma maior precisão e menos falsos positivos.

Integração com Discord: Inclui uma implementação funcional de um bot para o Discord que utiliza o Antígeno Digital como um filtro de segurança antes de consultar um LLM principal (Groq).

🚀 A Jornada do Projeto
O desenvolvimento do Antígeno Digital seguiu uma metodologia iterativa em quatro fases:

Fase 1 - Análise de Viabilidade: Testes iniciais com modelos genéricos (ex: Llama-Guard-2) revelaram uma performance insatisfatória para o português (Recall < 35%), demonstrando a necessidade de uma solução customizada.

Fase 2 - Sistema Híbrido: Uma abordagem inicial combinou um sistema de regras manuais com a IA genérica. Embora tenha melhorado a deteção de ataques óbvios, mostrou-se frágil a ataques mais subtis.

Fase 3 - O Ponto de Virada (Fine-Tuning): O grande salto de qualidade veio com o fine-tuning de um modelo BERTimbau com o nosso dataset customizado, que elevou o Recall para mais de 98%, validando a abordagem.

Fase 4 - Refinamento e "AI-First": Após identificar que o modelo de IA superava a eficácia das regras manuais (principalmente na redução de falsos positivos), o sistema foi refatorado para uma arquitetura "AI-First", confiando plenamente no modelo treinado.

🛠️ Tecnologias Utilizadas
Linguagem: Python 3.9+

Machine Learning:

PyTorch

Hugging Face Transformers (para o modelo, tokenizer e o Trainer)

Hugging Face Datasets (para o processamento de dados)

Scikit-learn (para métricas e divisão do dataset)

Pandas (para manipulação de CSV)

Integração:

discord.py (para o bot do Discord)

groq (como cliente do LLM principal)

Controlo de Versão:

Git & GitHub

Git LFS (para o armazenamento dos ficheiros do modelo)

📂 Estrutura do Repositório
O projeto está organizado em duas pastas principais:

.
├── 📁 ANTIGENO_DIGITAL/      # Núcleo de IA e scripts de treino
│   ├── modelo_detector_injecao_v2/  # O modelo fine-tunado (requer Git LFS)
│   ├── FineTuningModelo.py   # Script para treinar o modelo
│   └── ...                   # Datasets e scripts de análise
│
└── 📁 integração/           # Aplicação prática e integração com o Discord
    ├── discord_bot.py      # O código principal do bot
    ├── antigeno_digital.py # Orquestra a análise de segurança
    ├── classifier/         # Módulo para carregar e usar o modelo
    └── ...                 # Ficheiros de configuração

⚙️ Instalação e Execução
Para executar o bot do Discord localmente, siga estes passos:

1. Clone o Repositório

Como o projeto usa Git LFS para os ficheiros do modelo, é crucial que você tenha o Git LFS instalado.

# Primeiro, instale o Git LFS (se ainda não o tiver)
# https://git-lfs.github.com

# Clone o repositório
git clone https://github.com/andreguiot/antigeno-digital-detector.git

# Navegue para a pasta do projeto
cd antigeno-digital-detector

# Puxe os ficheiros grandes do LFS
git lfs pull

2. Crie um Ambiente Virtual

É uma boa prática isolar as dependências do projeto.

# Crie o ambiente
python -m venv venv

# Ative o ambiente
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

3. Instale as Dependências

Crie um ficheiro requirements.txt na raiz do projeto com o seguinte conteúdo:

torch
transformers
datasets
scikit-learn
pandas
discord.py
python-dotenv
groq

E depois instale-o:

pip install -r requirements.txt

4. Configure as Chaves de API

Na pasta integração, crie um ficheiro chamado .env.

Dentro dele, adicione as suas chaves:

DISCORD_BOT_TOKEN="O_SEU_TOKEN_AQUI"
GROQ_API_KEY="A_SUA_CHAVE_GROQ_AQUI"

O ficheiro config.py já está preparado para ler estas variáveis.

5. Execute o Bot

python integração/discord_bot.py

O bot ficará online no seu servidor do Discord e responderá a comandos iniciados com !antigeno.

🔮 Próximos Passos e Melhorias
O modelo atual, embora robusto, ainda tem espaço para melhorias:

Melhorar a Deteção de Eufemismos: Treinar o modelo com mais exemplos de linguagem codificada (ex: "saco de carne de 70kg").

Fortalecer a Deteção de Conteúdo Ilegal: Aumentar o número de exemplos de pedidos diretos de conteúdo prejudicial para aumentar a sensibilidade do modelo a estes casos.

Reduzir a Hipersensibilidade: Adicionar mais exemplos de uso seguro de palavras-chave como "prompt" para diminuir os Falsos Positivos restantes.

👨‍💻 Autores 
  Equipe IEEE : 

André Guiot  - andreguiot



Sinta-se à vontade para explorar o código!