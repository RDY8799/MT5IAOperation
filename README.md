# MT5 AI Bot (MT5IAOperation)

Bot de trading com IA para MetaTrader 5, com pipeline de dados, treino LightGBM, backtests por fases, execucao live, camada de risco e menu CLI interativo.

## Resumo rapido

- Coleta candles com fonte hibrida (`auto`: MT5 -> Yahoo fallback).
- Gera features tecnicas e labels (triple barrier).
- Treina LightGBM com validacao temporal (PurgedKFold).
- Roda fases de robustez (`phase4` ate `phase11`).
- Executa bot live/paper/diagnostico.
- Registra modelos em `Model Registry` com versao, schema e metadados.
- Orquestra tudo por menu (`cli_menu`) com selecao de simbolo por lista do MT5.

## Estrutura

```
mt5_ai_bot/
  data/
    raw/
    processed/
  logs/
  models/                      # legado (compatibilidade)
  reports/
    models/
      index.json
      {symbol}/{tf}/{version}/
        model.bin
        features_schema.json
        train_meta.json
        metrics_oof.json
    runs/
      {run_id}/
        run_meta.json
        logs/
          stdout.log
          stderr.log
        outputs/
  src/
  tests/
```

## Requisitos

- Windows + MetaTrader 5 instalado
- Python 3.11+
- Conta demo (recomendado para validacao)

## Instalacao

No diretorio `mt5_ai_bot`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Credenciais MT5

Voce pode configurar credenciais de 2 formas:

1. Variaveis de ambiente do Windows:

```powershell
setx MT5_LOGIN "SEU_LOGIN"
setx MT5_PASSWORD "SUA_SENHA"
setx MT5_SERVER "SEU_SERVIDOR"
setx MT5_PATH "C:\Program Files\MetaTrader 5\terminal64.exe"
```

2. Pelo menu (`opcao 9`), salvando em:

- `reports/runs/mt5_credentials.json`

## OpenAI API (auto-ajuste de parametros)

Opcional, para usar auto-ajuste iterativo no fluxo pos-treino:

- Salve pelo menu na opcao `12) Configurar OpenAI API (key/model)`.
- Arquivo salvo em: `reports/runs/openai_credentials.json`.
- Campos:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (padrao sugerido: `gpt-4.1-mini`)

## Pipeline de treino (manual)

### 1) Coletar dados

```powershell
python -m src.data_feed --symbol EURUSD --tfs M5 M15 M30 H1 H4 --months 24 --source auto
```

Fontes suportadas:

- `--source auto` (padrao): tenta MT5 primeiro; se vier 0 candles, tenta Yahoo.
- `--source mt5`: usa somente MT5.
- `--source yahoo`: usa somente Yahoo.

Observacao:

- A coleta MT5 agora usa janela em blocos (backfill por periodos), sem depender de scroll manual de grafico.

### 2) Gerar dataset

```powershell
python -m src.build_dataset --symbol EURUSD --tf M5
```

### 3) Treinar modelo

```powershell
python -m src.train_lgbm --symbol EURUSD --tf M5 --splits 5 --seed 42
```

## Model Registry

Arquivo principal:

- `reports/models/index.json`

Cada modelo registrado contem:

- `symbol`, `timeframe`, `version`, `trained_at`
- `model_path`
- `features_schema_path`
- `train_meta_path`
- `metrics_oof_path`
- resumo de metricas

API principal (modulo `src/model_registry.py`):

- `list_models()`
- `get_latest_model(symbol, tf)`
- `get_model(symbol, tf, version)`
- `register_model(...)`
- `load_model_object(...)`

## Menu interativo (CLI)

Executar:

```powershell
python -m src.cli_menu
```

O topo do menu mostra status de:

- Credenciais salvas
- MT5 instalado
- Login ativo no MT5

### Desenho do menu e submenus

```text
MT5 AI BOT MENU
├─ 1) Listar modelos treinados
├─ 2) Treinar novo modelo
├─ 3) Rodar backtest/robustez (fase)
├─ 4) Rodar bot (paper)
├─ 5) Rodar bot (demo-trade)
├─ 6) Rodar diagnostico (diagnostic-only)
├─ 7) Configurar simbolos/TFs (perfis)
├─ 8) Iniciar trade (background + submenu flags)
│  ├─ simbolo
│  ├─ tf entrada
│  ├─ modelo (symbol/tf/version ou latest)
│  ├─ modo (paper/demo/diagnostic/no-trade)
│  ├─ gate/policy (quando aplicavel)
│  └─ flags extras (telegram, logs, etc.)
├─ 9) Configurar credenciais MT5
│  ├─ login
│  ├─ senha
│  ├─ servidor
│  └─ caminho do terminal (opcional)
├─ 10) Trades ativos (tempo real)
├─ 12) Configurar OpenAI API (key/model)
│  ├─ salvar/atualizar OPENAI_API_KEY
│  ├─ definir OPENAI_MODEL
│  └─ limpar chave (digitar APAGAR)
└─ 0) Sair
```

### Funcoes de cada opcao

1. `Listar modelos treinados`
- Lê o `Model Registry` e mostra modelos disponíveis por `symbol/tf/version`, data e caminhos.

2. `Treinar novo modelo`
- Fluxo guiado: símbolo, timeframe, splits e seed.
- Se não houver dataset, oferece geração automática.
- Detecta inconsistência de artefatos (`raw/features/dataset`) e oferece limpeza/reconstrução.
- Salva run em `reports/runs/{run_id}` com logs e metadados.

3. `Rodar backtest/robustez (fase)`
- Executa runners de fase (`phase4` até `phase11`) com parâmetros escolhidos.
- Centraliza outputs no `run_id` da execução.

4. `Rodar bot (paper)`
- Inicia bot sem enviar ordens reais.
- Útil para validar sinais, bloqueios e coerência de decisão.

5. `Rodar bot (demo-trade)`
- Inicia bot com envio de ordens na conta demo MT5.

6. `Rodar diagnostico (diagnostic-only)`
- Roda ciclo completo de decisão sem enviar ordens.
- Gera contadores por motivo de bloqueio para auditoria.

7. `Configurar simbolos/TFs (perfis)`
- Salva perfis base por símbolo/timeframe (entry/gate) para reutilização.

8. `Iniciar trade (background + submenu flags)`
- Inicia `bot_live` em segundo plano (não trava o menu).
- Permite montar flags de execução antes de iniciar.
- Registra `PID`, comando e logs da execução.

9. `Configurar credenciais MT5`
- Persiste credenciais em `reports/runs/mt5_credentials.json`.
- O menu aplica no ambiente para subprocessos automaticamente.

10. `Trades ativos (tempo real)`
- Tabela live com posições abertas (ticket, tipo, lote, preço, SL/TP, PnL, duração).
- Pode filtrar por símbolo.

12. `Configurar OpenAI API (key/model)`
- Salva chave/modelo para subprocessos do menu.
- Habilita auto-ajuste iterativo no fluxo pós-treino.

0. `Sair`
- Encerra apenas o menu (processos em background continuam rodando).

### Fluxos recomendados

1. Primeiro treino:
- `2 -> Treinar novo modelo` (com `source=auto`)  
- depois `4 -> Rodar bot (paper)`  
- depois `5 -> Rodar bot (demo-trade)` se paper estiver consistente.

2. Diagnóstico de bloqueios:
- `6 -> Rodar diagnostico`  
- revisar logs/sumários por motivo (`tf_conflict`, volatilidade, risco global etc.).

### O que o menu ja automatiza

- Se dataset nao existir no treino:
  - pergunta se deve gerar automaticamente
  - pergunta a fonte (`auto/mt5/yahoo`)
  - roda coleta (`data_feed`) + build (`build_dataset`) + treino
- Se dataset/features estiverem vazios:
  - aborta com mensagem clara (sem traceback confuso)
- Se detectar inconsistencia (raw/features/dataset vazio ou quebrado):
  - oferece apagar artefatos do simbolo/TF e reconstruir
- Cria `run_id` e salva logs/outputs em `reports/runs/{run_id}`
- Salva ultima selecao em `reports/runs/last_selection.json`
- Aceita entrada em minusculo para simbolo/TF e normaliza automaticamente
- Exibe "Ultima selecao" de forma amigavel (painel com acao/par/tf/run_id)

### Progresso real no terminal (sem barra fake)

Durante execucao de modulos pelo menu (coleta, dataset e treino), a barra agora usa progresso real via eventos `PROGRESS x/y`:

- `data_feed`: progresso por timeframe coletado
- `build_dataset`: 2 etapas (features e dataset)
- `train_lgbm`: progresso por fold do PurgedKFold

Se o modulo nao emitir `PROGRESS`, o menu finaliza com status concluido, sem simular porcentagem ciclica.

### Painel de diagnostico ao treinar

No treino (`opcao 2`), abaixo da barra aparece painel "Treino em tempo real" com linhas `LIVE`, incluindo:

- resumo do dataset (linhas, periodo, quantidade de features)
- distribuicao de classes (`y`)
- por fold: tamanho de treino/validacao, `best_iter`, `val_logloss`
- caminho final do modelo salvo

Isso melhora visibilidade do que o bot esta calculando enquanto treina.

## Simbolo por lista (MT5)

No menu, quando pede simbolo:

- carrega lista de simbolos do MT5
- permite filtro (ex: `EUR`, `XAU`)
- selecao por indice numerico

## Rodar bot live/paper/diagnostico

### Live

```powershell
python -m src.bot_live --symbol EURUSD --tf M5
```

### Paper

```powershell
python -m src.bot_live --symbol EURUSD --tf M5 --paper --paper-bars 800
```

### Diagnostico (nao envia ordens)

```powershell
python -m src.bot_live --symbol EURUSD --tf M5 --diagnostic-only --out reports --out-name diag_m5
```

## Selecao de modelo no bot

`bot_live` suporta:

- `--model-symbol`
- `--model-tf`
- `--model-version`
- `--use-latest-model` (padrao)
- `--no-use-latest-model`

Exemplo (latest):

```powershell
python -m src.bot_live --symbol EURUSD --tf M5 --model-symbol EURUSD --model-tf M5 --use-latest-model
```

Exemplo (versao fixa):

```powershell
python -m src.bot_live --symbol EURUSD --tf M5 --model-symbol EURUSD --model-tf M5 --model-version 20260303_120000 --no-use-latest-model
```

## Validacao de schema antes do trade

Ao iniciar, o bot valida `features_schema.json` do modelo contra as features live.

Se faltar feature obrigatoria do schema:

- gera `kill_switch` com motivo `MISSING_FEATURES_SCHEMA`
- nao opera com shape incompativel

## Fases de robustez

Runners disponiveis (estado atual):

- `phase4_runner.py`
- `phase5_runner.py`
- `phase6_runner.py`
- `phase7_runner.py`
- `phase8_runner.py`
- `phase9_runner.py`
- `phase10_runner.py`
- `phase11_runner.py`

Exemplos:

```powershell
python -m src.phase10_runner --symbol EURUSD --tf_entry M5 --tf_gate M30 --windows 8 --seed 42
python -m src.phase11_runner --symbol EURUSD --windows 8 --seed 42
```

## Logs

### Logs live

- Tecnico JSON: `logs/bot_live.log`
- Humano: `logs/bot_live_human.log`

Visualizar em tempo real:

```powershell
Get-Content logs\bot_live_human.log -Wait
```

### Diagnostico por hora

No modo `--diagnostic-only`, gera sumarios periodicos com contadores de bloqueio.

## Trades ativos em tempo real (menu opcao 10)

Mostra tabela com:

- Ticket
- Simbolo
- Tipo
- Lote
- Preco de entrada
- SL/TP
- PnL atual
- Duracao da posicao

Sem travar o bot em background.

## Configuracao central

Arquivo: `src/config.py`

Principais grupos:

- `RiskConfig`
- `GlobalRiskConfig`
- `LiveConfig`
- `TripleBarrierConfig`
- `FeatureConfig`
- `TimeframeConfig` (M5/M30/M1 etc.)

## Troubleshooting

### Erro: dataset nao encontrado

Gere automaticamente pelo menu (opcao 2), ou rode manualmente:

```powershell
python -m src.data_feed --symbol XAUUSD --tfs M5 --months 24 --source auto
python -m src.build_dataset --symbol XAUUSD --tf M5
```

### Dataset vazio (0 linhas)

Normalmente significa:

- simbolo inexistente no broker (ex: usar `XAUUSDm` em vez de `XAUUSD`)
- historico insuficiente no MT5
- fallback externo sem historico suficiente para esse ativo/intervalo

Use a selecao por lista no menu para achar o nome real do ativo.

### Barra de progresso "parada" no treino

Se parecer travada em um fold:

- com `splits=10`, cada fold pode demorar varios minutos
- o progresso avanca quando termina o fold atual
- verifique `run_meta/logs/stdout.log` para detalhes completos

Para iteracao mais rapida durante desenvolvimento:

- use `splits=5`
- mantenha `seed=42` para reproducibilidade

### "Os dados Yahoo sao iguais ao MT5?"

Nao. Yahoo e fonte de mercado para pesquisa/backtest rapido, mas nao replica exatamente:

- spread/custos do seu broker
- microvariacao de candles do servidor MT5
- condicoes exatas de execucao

Use `MT5` para validacao final de operacao. Use `auto` para acelerar preparacao de dataset.

## Aviso de risco

Trading envolve risco elevado. Use conta demo para validacao.
Resultados passados nao garantem resultado futuro.
