# Telegram AI Character

Bot privado do Telegram com personagem virtual adulta, memória estruturada,
memória semântica, estado emocional simulado, relacionamento progressivo,
Gemini e autonomia conservadora. A arquitetura existente foi mantida.

## Fluxo

Telegram -> FastAPI/Webhook -> AgentBrain

AgentBrain integra:

- MemoryExtractor + Deduplicator + MemoryManager
- SemanticMemoryManager + embeddings locais 384D + pgvector
- EmotionEngine
- RelationshipEngine
- ContextManager
- GeminiLLM com fallback local
- ImageService desacoplado

Autonomia:

GitHub Actions (15 min) -> POST /autonomous/tick -> AutonomyService
-> DecisionEngine -> Telegram

## O que foi implementado nesta versão

### Memória semântica

Novos arquivos:

- `app/brain/embedder.py`
- `app/brain/semantic_memory.py`

Cada mensagem do usuário com pelo menos 8 caracteres pode virar uma
`semantic_memory`. O embedding é determinístico, local, CPU-only e tem 384
dimensões, portanto não exige API externa, GPU ou computador local.

A busca usa distância de cosseno do pgvector. O embedding combina tokens,
n-gramas de caracteres e pequenas expressões, ajudando a recuperar textos
mesmo quando flexões como "viajar" e "viagem" não são idênticas.

### DecisionEngine

`app/brain/decision_engine.py` decide `message` ou `wait`.

A decisão exige:

- autonomia habilitada;
- ausência de `quiet_until`;
- cooldown respeitado;
- limite diário não atingido;
- relação mínima;
- contexto real de conversa;
- memória estruturada ou semântica;
- sinais suficientes de proximidade, curiosidade e confiança.

A decisão é conservadora para evitar mensagens genéricas repetitivas.

### AutonomyService

`app/brain/autonomy.py` usa uma sessão PostgreSQL curta por tick e trava a
linha de `autonomy_states` com `FOR UPDATE`, reduzindo risco de dois ticks
simultâneos enviarem a mesma mensagem.

Defaults:

- intervalo mínimo: 90 minutos
- máximo diário: 3 mensagens

Cada mensagem espontânea é persistida em `conversation_messages`.

### API protegida

Endpoint principal:

`POST /autonomous/tick`

Header:

`X-Autonomy-Token: <AUTONOMY_TOKEN>`

O endpoint antigo `/internal/tick` permanece como compatibilidade e usa a
mesma autenticação.

Webhook:

`POST /telegram/webhook`

Header validado:

`X-Telegram-Bot-Api-Secret-Token`

## Migration

A cadeia é:

`0001_initial -> 0002_brain_memory -> 0003_emotion_relationship -> 0004_semantic_autonomy`

A migration 0004 cria:

- `semantic_memories`
- `autonomy_states`
- índice HNSW de cosine para `semantic_memories.embedding`

## Variáveis

Veja `.env.example`.

As principais são:

```text
TELEGRAM_BOT_TOKEN=
WEBHOOK_BASE_URL=
WEBHOOK_SECRET=
DATABASE_URL=

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_TIMEOUT_SECONDS=60
LLM_MAX_OUTPUT_TOKENS=500

AUTONOMY_TOKEN=
AUTONOMY_MIN_INTERVAL_MINUTES=90
AUTONOMY_MAX_DAILY_MESSAGES=3

CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_IMAGE_MODEL=
POLLINATIONS_API_KEY=
POLLINATIONS_IMAGE_MODEL=
IMAGE_DAILY_LIMIT=
IMAGE_MONTHLY_LIMIT=
IMAGE_TIMEOUT_SECONDS=
```

## Deploy pelo celular

### 1. Telegram

No BotFather, crie o bot e copie o token.

Não coloque o token no GitHub.

### 2. Supabase

Crie um projeto PostgreSQL e copie a connection string.

O projeto usa SQLAlchemy async com `asyncpg`. Se o Supabase fornecer uma URL
com prefixo `postgresql://`, use a forma async:

`postgresql+asyncpg://...`

Se a senha tiver caracteres especiais, use a URL corretamente codificada.

### 3. GitHub

Crie um repositório privado e envie todos os arquivos deste ZIP.

Não envie `.env`.

### 4. Render

Crie um Web Service a partir do repositório, usando Docker. O projeto já está preparado para usar a porta `PORT` fornecida pelo Render.

Configure as variáveis do `.env.example`.

`WEBHOOK_BASE_URL` deve ser a URL HTTPS pública do Render.

### 5. Migration

O container já executa `alembic upgrade head` automaticamente antes de iniciar o bot. Se quiser conferir ou repetir manualmente, abra o Shell do serviço no Render e execute:

```bash
alembic upgrade head
```

O resultado esperado termina em:

```text
0004_semantic_autonomy
```

### 6. Webhook

O aplicativo registra o webhook automaticamente na inicialização quando
`WEBHOOK_BASE_URL` e `WEBHOOK_SECRET` estão configurados.

### 7. GitHub Actions

Em Settings -> Secrets and variables -> Actions, crie:

- `BOT_URL`: URL pública do Render, sem `/autonomous/tick`
- `AUTONOMY_TOKEN`: exatamente o mesmo valor configurado no Render

O workflow `.github/workflows/autonomous-tick.yml` executa a cada 15 minutos.

O workflow só acorda o endpoint; o cooldown e o limite diário são aplicados
no banco.

## Teste

1. Abra o bot no Telegram.
2. Envie `/start` ou uma mensagem normal.
3. Diga algo explícito, por exemplo: `minha comida favorita é pizza`.
4. Continue a conversa.
5. Confira se a resposta mantém o contexto.
6. Teste `/foto uma personagem adulta em uma cafeteria`.
7. Deixe a autonomia habilitada.
8. Após o intervalo configurado, o GitHub Actions poderá executar uma decisão.
9. Se houver contexto suficiente, o bot poderá enviar uma mensagem espontânea.

## Imagens

A camada existente foi preservada:

`ImageService -> ImageProviderRouter -> Cloudflare Workers AI / Pollinations`

O cérebro não depende da implementação do provider de imagem.

## Segurança comportamental

A personagem é explicitamente virtual e adulta. Ela pode ser calorosa e
romântica como estilo de personagem, mas não deve alegar ser uma pessoa humana
real, nem usar chantagem, ameaça, coerção ou pressão emocional.

## Fallback do Gemini

Se `GEMINI_API_KEY` não estiver configurada, houver timeout, erro HTTP ou 429,
o bot continua funcionando e usa a resposta local existente. Isso evita que
uma falha da API derrube o webhook.

## Nota sobre plano gratuito

Os serviços e limites gratuitos de Render, Supabase, Gemini, Cloudflare,
Pollinations e GitHub podem mudar. A arquitetura não exige GPU ou software
rodando no celular, mas os limites atuais de cada provedor devem ser conferidos
na conta antes do deploy.

## Face identity post-processing

A geração de imagens agora tem uma segunda etapa obrigatória, separada dos
providers de geração:

`ImageProvider -> FaceSwapService -> Telegram`

Quando `FACE_SWAP_REQUIRED=true`, o bot **não entrega a imagem gerada sem o
face swap**. Se o processamento facial falhar, a geração é marcada como falha
e a imagem sem o rosto aplicado não é enviada.

### Referência facial

O projeto inclui a referência facial em:

`assets/pamela_face_reference.jpg`

Ela é usada somente pela camada de geração/edição de imagens. Não é enviada
para o Gemini e não participa das memórias conversacionais.

### Provider gratuito preferencial

O provider padrão é o Hugging Face Gradio Space configurado em
`HF_FACE_SWAP_SPACE`. O Space de referência atual usa FaceFusion/ReActor com
Hyperswap e CPU. A integração chama o endpoint `/generate_image`, passando:

- referência facial como `Source (Face)`;
- imagem recém-gerada como `Target (Body)`;
- `target_index=0` para o maior rosto;
- `hyperswap_1b_256.onnx` como modelo padrão;
- sem face restoration adicional por padrão.

O provider é externo e pode mudar de disponibilidade ou de API. Por isso os
nomes do Space, endpoint e modelo são configuráveis por ambiente.

Para privacidade, prefira duplicar o Space para sua própria conta do Hugging
Face e deixá-lo privado, depois configure `HF_FACE_SWAP_SPACE` para essa cópia.

### Fallback opcional

Também existe fallback opcional via Replicate. Ele só é usado se o provider
Hugging Face falhar e `REPLICATE_API_TOKEN` estiver configurado. Esse fallback
pode gerar cobrança e, portanto, permanece desativado na prática quando nenhum
token é informado.

### Importante sobre identidade

O prompt da geração continua pedindo consistência visual, mas isso não é mais
o mecanismo principal de identidade facial. A referência facial é aplicada
**depois** da geração. Isso reduz a variação do rosto entre cenas.

Ainda assim, nenhum algoritmo de face swap pode garantir identidade perfeita em
100% das imagens: rostos parcialmente ocultos, ângulos extremos, múltiplas
pessoas, mãos cobrindo o rosto ou imagens sem um rosto detectável podem fazer a
operação falhar. Nesses casos, com `FACE_SWAP_REQUIRED=true`, a imagem não é
enviada ao Telegram.

### Variáveis adicionais

```text
FACE_SWAP_ENABLED=true
FACE_SWAP_REQUIRED=true
FACE_SWAP_PROVIDER=huggingface
FACE_REFERENCE_IMAGE_PATH=assets/pamela_face_reference.jpg
FACE_SWAP_TIMEOUT_SECONDS=180
HF_FACE_SWAP_SPACE=V0pr0S/FaceFusion-Face-Swap-Hyperswap
HF_FACE_SWAP_API_NAME=/generate_image
HF_TOKEN=
HF_FACE_SWAP_MODEL=hyperswap_1b_256.onnx
HF_FACE_SWAP_TARGET_INDEX=0
HF_FACE_RESTORE_MODEL=none
HF_FACE_RESTORE_STRENGTH=0.7

REPLICATE_API_TOKEN=
REPLICATE_FACE_SWAP_VERSION=codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34
```

### Privacidade da referência

Como a referência é um rosto de uma pessoa real fornecido pelo usuário, não
publique `assets/pamela_face_reference.jpg` em um repositório público. Use um
repositório GitHub privado e, se possível, uma cópia privada do Space de
face-swap. O serviço de face-swap recebe a referência e a imagem gerada durante
o processamento.
