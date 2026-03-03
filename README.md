# MT5IAOperation - Bot de Trading com IA para MetaTrader 5

Projeto de bot para operacao automatizada (foco em conta DEMO) com:
- pipeline de dados e features
- treino de modelo LightGBM
- backtest com custos realistas
- validacao walk-forward/robustez por fases
- execucao live com regras de risco
- logs tecnicos (JSON) e logs humanos (PT-BR)
- alertas no Telegram

## 1) Estrutura principal

- `src/data_feed.py`: coleta candles do MT5 e salva em `data/raw`.
- `src/features.py`: gera features tecnicas e de contexto (inclui suporte/resistencia).
- `src/labeling_triple_barrier.py`: gera labels pelo metodo triple barrier.
- `src/build_dataset.py`: monta dataset final para treino/backtest.
- `src/train_lgbm.py`: treino LightGBM com PurgedKFold.
- `src/backtest_engine.py`: motor de backtest e metricas.
- `src/multitf.py`: alinhamento H4->H1 e politicas multi-timeframe.
- `src/phase4_runner.py ... src/phase9_runner.py`: fases de diagnostico/robustez.
- `src/bot_live.py`: execucao live/paper no MT5.
- `src/config.py`: configuracoes centrais.
- `src/notifier_telegram.py`: notificacoes Telegram.

Pastas:
- `data/raw`, `data/processed`
- `models`
- `logs`
- `reports`

## 2) Requisitos

- Windows com MetaTrader 5 instalado
- Python 3.11+
- Conta MT5 (preferencialmente DEMO para validacao)

## 3) Instalacao

No diretorio raiz `mt5_ai_bot`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4) Variaveis de ambiente MT5

Opcionalmente configure para login automatico:

```powershell
setx MT5_LOGIN "SEU_LOGIN"
setx MT5_PASSWORD "SUA_SENHA"
setx MT5_SERVER "SEU_SERVIDOR"
setx MT5_PATH "C:\Program Files\MetaTrader 5\terminal64.exe"
```

## 5) Pipeline de treino (exemplo)

### 5.1 Coletar dados

```powershell
python -m src.data_feed --symbol EURUSD --tfs M1 M5 M15 M30 H1 H4 --months 24
```

### 5.2 Gerar dataset

```powershell
python -m src.build_dataset --symbol EURUSD --tf M15
python -m src.build_dataset --symbol EURUSD --tf H1
```

### 5.3 Treinar modelo

```powershell
python -m src.train_lgbm --symbol EURUSD --tf M15 --splits 5
python -m src.train_lgbm --symbol EURUSD --tf H1 --splits 5
```

## 6) Backtest e fases

Exemplos:

```powershell
python -m src.backtest_engine --symbol EURUSD --tf H1 --threshold 0.60
python -m src.phase8_runner --symbol EURUSD --seed 42 --windows 8
python -m src.phase9_runner --symbol EURUSD --seed 42
```

Saidas ficam em `reports/` (json/csv por fase).

## 7) Rodar bot live

### 7.1 Teste unico sem enviar ordem

```powershell
python -m src.bot_live --symbol EURUSD --tf M15 --once --no-trade
```

### 7.2 Rodar continuo com ordens

```powershell
python -m src.bot_live --symbol EURUSD --tf M15
```

Observacoes:
- O bot decide no fechamento de candle do timeframe.
- Em `M15`, ha ate 96 verificacoes por dia.
- O envio de ordem depende de probabilidade + filtros + risco.

## 8) Telegram

Configure:

```powershell
setx MT5_TELEGRAM_BOT_TOKEN "SEU_TOKEN"
setx MT5_TELEGRAM_CHAT_ID "SEU_CHAT_ID"
```

Eventos notificados:
- inicio do bot
- nova entrada
- falha de ordem
- sinal bloqueado
- kill switch
- posicao fechada
- alerta de saude do modelo

## 9) Logs

### 9.1 Log tecnico JSON

Arquivo:
- `logs/bot_live.log`

### 9.2 Log humano PT-BR

Arquivo:
- `logs/bot_live_human.log`

Visualizar em tempo real:

```powershell
Get-Content logs\bot_live_human.log -Wait
```

## 10) Configuracao central

Arquivo:
- `src/config.py`

Principais grupos:
- `RiskConfig`: limites de risco e sizing
- `TripleBarrierConfig`: parametros SL/TP/horizonte
- `LiveConfig`: threshold, filtros, time stop, monitor de saude
- `FeatureConfig`: janelas dos indicadores

## 11) Melhorias implementadas na versao atual

- Multi-timeframe e alinhamento sem lookahead
- Auditoria de cobertura por janela
- Camada de risco (sizing/circuit breaker/cooldown) para robustez
- Fechamento por time stop (configuravel)
- Filtros de sessao, noticia e spread
- Features de suporte/resistencia
- Logs humanos coloridos no terminal
- Notificacoes Telegram em PT-BR com emojis

## 12) Aviso de risco

Trading envolve risco elevado.
Use conta DEMO para validacao.
Nao ha garantia de lucro futuro, mesmo com bons resultados passados.

