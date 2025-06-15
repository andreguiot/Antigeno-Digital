# discord_bot.py
import discord
from discord.ext import commands
import asyncio
import os

# Importações do seu projeto Antígeno Digital
import config # Para carregar o token do bot e outras chaves
from antigeno_digital import obter_analise_antigeno, query_groq
from classifier.model import inicializar_classificador # Para "aquecer" o modelo
from groq_client import CLIENT_GROQ # Para verificar se o cliente Groq está pronto

# --- Configurações do Bot ---
BOT_TOKEN = config.DISCORD_BOT_TOKEN
COMMAND_PREFIX = "!antigeno" # Ou qualquer prefixo que você preferir

if not BOT_TOKEN:
    print("ERRO CRÍTICO (Discord Bot): DISCORD_BOT_TOKEN não encontrado. Verifique seu .env e config.py.")
    exit()

# Define as intents necessárias para o bot
# message_content é crucial para ler o conteúdo das mensagens
intents = discord.Intents.default()
intents.message_content = True 
intents.messages = True 

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# --- Funções de Apoio ---
async def run_blocking_io(blocking_func, *args):
    """Executa uma função síncrona bloqueadora em um thread separado."""
    # asyncio.to_thread é preferível no Python 3.9+
    if hasattr(asyncio, 'to_thread'):
        return await asyncio.to_thread(blocking_func, *args)
    else: # Fallback para versões mais antigas do Python
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, blocking_func, *args)

async def inicializar_sistemas_antigeno():
    """
    Inicializa os componentes do Antígeno Digital que podem ter carregamento demorado.
    """
    print("Inicializando sistema de classificação do Antígeno Digital...")
    try:
        # "Aquece" o modelo de IA (carrega se ainda não carregado)
        # A função analisar_prompt_pela_ia (chamada por obter_analise_antigeno) 
        # já chama inicializar_classificador na primeira vez.
        # Podemos fazer uma chamada à função de inicialização diretamente aqui para garantir.
        await run_blocking_io(inicializar_classificador) # Executa a inicialização síncrona em um thread
        print("Modelo de classificação de IA pronto (ou já estava).")
    except Exception as e_cls:
        print(f"Erro ao inicializar o classificador de IA: {e_cls}")

    if CLIENT_GROQ is None: # CLIENT_GROQ é carregado quando groq_client.py é importado
        print("AVISO (Discord Bot): Cliente Groq não parece estar inicializado. Verifique groq_client.py e API Key.")
    else:
        print("Cliente Groq parece estar pronto.")
    print("Sistemas do Antígeno Digital inicializados.")


# --- Eventos do Bot ---
@bot.event
async def on_ready():
    """Chamado quando o bot está conectado e pronto."""
    print(f'Bot conectado como {bot.user.name} (ID: {bot.user.id})')
    print(f'Prefixo de comando: {COMMAND_PREFIX}')
    print('------')
    await inicializar_sistemas_antigeno() # Inicializa os modelos
    print("Bot está pronto para receber comandos!")

@bot.event
async def on_message(message: discord.Message):
    """Chamado quando uma mensagem é enviada em qualquer canal que o bot pode ver."""
    # Ignorar mensagens do próprio bot para evitar loops
    if message.author == bot.user:
        return

    # Verificar se a mensagem começa com o prefixo de comando
    if message.content.startswith(COMMAND_PREFIX):
        # Extrair o prompt do usuário (texto após o prefixo e um espaço)
        if len(message.content) > len(COMMAND_PREFIX) + 1:
            prompt_usuario = message.content[len(COMMAND_PREFIX):].strip()
            if not prompt_usuario:
                await message.channel.send(f"Por favor, forneça um prompt após `{COMMAND_PREFIX}`.")
                return
        else:
            await message.channel.send(f"Uso: `{COMMAND_PREFIX} seu prompt aqui`")
            return
        
        print(f"\nComando recebido de '{message.author.name}': {COMMAND_PREFIX} {prompt_usuario}")
        # Envia uma mensagem de "processando" para dar feedback ao usuário
        processing_message = await message.channel.send(f"Analisando seu prompt com o Antígeno Digital: \"{prompt_usuario[:50]}...\" 🔬")

        # 1. Analisar com o Antígeno Digital (executado em thread para não bloquear)
        try:
            analise_completa = await run_blocking_io(obter_analise_antigeno, prompt_usuario)
            classificacao = analise_completa["classificacao_final"]
            prob_injecao = analise_completa["prob_injecao"]
            motivo_deteccao = analise_completa["motivo_deteccao"]
            
            print(f"  Resultado Antígeno: Classificação='{classificacao}', Prob.Injeção='{prob_injecao:.2%}', Motivo='{motivo_deteccao}'")

        except Exception as e_antigeno:
            print(f"ERRO ao processar com Antígeno Digital: {e_antigeno}")
            await processing_message.edit(content=f"Desculpe, ocorreu um erro ao analisar seu prompt com o Antígeno Digital: `{e_antigeno}`")
            return

        # 2. Decidir ação e responder
        if classificacao == "INJECAO":
            resposta_bot_texto = (
                f"🚨 **ALERTA DO ANTÍGENO DIGITAL** 🚨\n"
                f"> **Prompt:** `{prompt_usuario}`\n"
                f"> **Motivo da Detecção:** `{motivo_deteccao}`\n"
                f"> **Probabilidade de Injeção:** `{prob_injecao:.2%}`\n"
                f"⚠️ **Ação:** Prompt BLOQUEADO."
            )
            await processing_message.edit(content=resposta_bot_texto)
        
        elif classificacao == "SEGURO":
            await processing_message.edit(content=f"Antígeno Digital: Prompt \"{prompt_usuario[:50]}...\" parece seguro (Prob. Injeção: {prob_injecao:.2%}). Enviando para o modelo principal (Groq)... 🧠")
            try:
                # Chamar Groq (executado em thread para não bloquear)
                resposta_groq = await run_blocking_io(query_groq, prompt_usuario)
                
                print(f"  Resposta Groq: '{resposta_groq[:100]}...'")
                
                # Respostas longas do Groq podem precisar ser divididas ou enviadas em embeds
                header_groq = f"**Resposta do Modelo Principal (Groq) para:** `{prompt_usuario}`\n>>> "
                if len(header_groq + resposta_groq) > 2000: # Limite de caracteres do Discord por mensagem é 2000
                    await message.channel.send(f"{header_groq}{resposta_groq[:2000 - len(header_groq) - 3]}...") # Trunca para caber
                else:
                    await message.channel.send(f"{header_groq}{resposta_groq}")

            except Exception as e_groq:
                print(f"ERRO ao obter resposta do modelo Groq: {e_groq}")
                await message.channel.send(f"Desculpe, ocorreu um erro ao buscar a resposta do modelo Groq: `{e_groq}`")
        
        else: # Casos de erro do Antígeno (ERRO_ANTIGENO, ANTIGENO_IA_DESCONHECIDO, etc.)
            resposta_bot_texto = (
                f"⚠️ **ATENÇÃO - ANTÍGENO DIGITAL** ⚠️\n"
                f"> **Prompt:** `{prompt_usuario}`\n"
                f"> **Problema na Análise:** `{classificacao}`\n"
                f"> **Detalhe:** `{motivo_deteccao}`\n"
                f"🚫 **Ação:** Por segurança, o prompt NÃO será enviado ao modelo principal."
            )
            await processing_message.edit(content=resposta_bot_texto)
    
# --- Iniciar o Bot ---
if __name__ == "__main__":
    print("Iniciando o Bot Antígeno Digital para Discord...")
    if BOT_TOKEN:
        try:
            bot.run(BOT_TOKEN)
        except discord.errors.LoginFailure:
            print("ERRO CRÍTICO (Discord Bot): Falha no login. O token fornecido é inválido.")
        except Exception as e_bot_run:
            print(f"ERRO CRÍTICO ao tentar executar o bot: {e_bot_run}")
    else:
        print("Bot não pode iniciar: Token do Discord não configurado.")

