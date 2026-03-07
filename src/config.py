from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    # Numero maximo de posicoes abertas ao mesmo tempo.
    max_open_positions: int = 10
    # Limite maximo de entradas por dia.
    max_trades_per_day: int = 100
    # Perda diaria maxima em percentual do saldo (kill switch).
    max_daily_loss_pct: float = 10.0
    # Margem minima permitida para operar.
    min_margin_level_pct: float = 100.0
    # Tempo de espera entre entradas (em minutos).
    cooldown_minutes: int = 0
    # Maximo de posicoes simultaneas no mesmo simbolo durante o live.
    max_symbol_positions_live: int = 1
    # Quantidade de perdas consecutivas para acionar pausa temporaria.
    max_consecutive_losses: int = 2
    # Pausa em candles apos atingir a sequencia maxima de perdas.
    loss_streak_cooldown_candles: int = 12
    # Pausa apenas na mesma direcao apos um stop/prejuizo.
    same_side_loss_pause_candles: int = 6
    # Se devolver X% do pico de lucro realizado do periodo, pausa.
    profit_giveback_lock_fraction: float = 0.50
    # Pausa em candles apos devolver lucro demais.
    profit_giveback_cooldown_candles: int = 20
    # Se True, usa lote dinamico por risco/SL; se False, usa lote fixo.
    use_dynamic_position_sizing: bool = True
    # Risco por operacao em % do saldo (usado no lote dinamico).
    default_risk_pct: float = 0.5
    # Lote fixo de fallback (ou lote padrao se dinamico estiver desligado).
    fixed_demo_lot: float = 1.00
    # Lote minimo permitido pelo bot.
    min_lot: float = 0.01
    # Lote maximo permitido pelo bot.
    max_lot: float = 10.0
    # Multiplicador de lote por regime (reduz risco em regimes ruins).
    regime_mult_trend: float = 1.20
    regime_mult_high_vol: float = 0.80
    regime_mult_sideways: float = 0.50
    regime_mult_neutral: float = 1.00


class TripleBarrierConfig(BaseModel):
    # Horizonte de barras por timeframe para o labeling/backtest.
    horizon_by_tf: Dict[str, int] = Field(
        default_factory=lambda: {"M1": 20, "M5": 12, "M15": 8, "M30": 8, "H1": 6}
    )
    # Multiplicador de ATR para Take Profit.
    pt_atr_mult: float = 1.5
    # Multiplicador de ATR para Stop Loss.
    sl_atr_mult: float = 1.0


class LiveConfig(BaseModel):
    # Probabilidade minima para entrar (BUY/SELL).
    threshold: float = 0.50
    # Quantidade de candles buscados no MT5 para gerar features.
    fetch_bars: int = 500
    # Numero maximo de falhas consecutivas de leitura MT5 antes de parar.
    max_mt5_failures: int = 5
    # Desvio maximo em pontos aceito no envio da ordem.
    order_deviation_points: int = 8
    # Magic number para identificar ordens deste bot no MT5.
    magic_number: int = 987654
    # Slippage simulado/usado no motor de avaliacao.
    slippage_points: int = 2
    # Comissao fixa por trade (se sua corretora cobrar por ordem).
    commission_per_trade: float = 0.0
    # Ativo padrao quando nenhum simbolo e informado.
    default_symbol: str = "EURUSD"
    # Timeframe padrao quando nenhum timeframe e informado.
    default_tf: str = "M15"
    # Limite maximo de spread (em pontos) para permitir nova entrada.
    max_spread_points: float = 35.0
    # Se True, aplica filtro de horarios permitidos.
    use_session_filter: bool = True
    # Janelas UTC permitidas para operar (formato HH:MM-HH:MM).
    allowed_sessions_utc: tuple[str, ...] = ("00:00-23:59",)
    # Se True, bloqueia entradas perto de horarios de noticia.
    use_news_blackout: bool = True
    # Horarios UTC sensiveis (formato HH:MM), ex: CPI/FOMC.
    news_blackout_utc: tuple[str, ...] = ("13:30", "15:00", "19:00")
    # Minutos antes/depois do horario de noticia para bloquear entradas.
    news_blackout_minutes: int = 20
    # Minutos antes do evento high-impact (calendario) para bloquear novas entradas.
    news_blackout_minutes_pre: int = 60
    # Minutos depois do evento high-impact (calendario) para bloquear novas entradas.
    news_blackout_minutes_post: int = 30
    # Ativa uso do calendario economico em CSV para blackout e features fundamentais.
    use_fundamental_calendar: bool = True
    # Se True, tenta coletar proxy de forca do USD (DXY/USDX) via MT5.
    use_dxy_proxy: bool = True
    # Ordem de tentativa de simbolos para DXY no broker.
    dxy_symbols: tuple[str, ...] = ("USDX", "DXY", "USDOLLAR")
    # Fecha posicoes abertas apos X candles (0 desliga).
    max_holding_candles: int = 0
    # Frequencia do monitor de saude (eventos) para logar estatisticas.
    health_check_every_n_events: int = 50
    # Alerta se taxa de sinal ficar abaixo deste valor no monitor.
    min_signal_rate_alert: float = 0.01
    # Alerta se confianca media (max prob) ficar abaixo deste valor.
    min_confidence_alert: float = 0.55
    # Delta acima do threshold para considerar "sinal forte bloqueado".
    strong_signal_alert_delta: float = 0.10


class TimeframeConfig(BaseModel):
    # Timeframe de entrada (onde as ordens sao decididas/executadas).
    tf_entry: str
    # Timeframe de gate/direcao.
    tf_gate: str
    # Politica do gate:
    # strict = gate precisa confirmar a direcao
    # allow_wait = gate WAIT nao bloqueia; direcao oposta bloqueia
    # bias_only = gate atua como vies; so bloqueia se vier forte contra
    # off = ignora gate
    gate_mode: str = "strict"
    # Margem minima do gate para bloquear no modo bias_only.
    gate_min_margin_block: float = 0.10
    # Horizonte em candles do tf_entry (uso em validacao/controles).
    horizon_candles: int
    # Maximo de entradas por hora/janela configurada.
    max_trades_per_hour: int
    # Modo da janela de contagem de trades: rolling 60m ou hora fixa.
    trades_window_mode: str = "rolling_60m"
    # Minimo de candles entre trades da mesma direcao.
    min_candles_between_same_direction_trades: int = 0
    # Bloqueio de reentrada apos fechamento (mesma direcao).
    reentry_block_candles: int = 2
    # Modo do threshold minimo de volatilidade.
    volatility_threshold_min_mode: str = "atr_percentile"
    # Modo do threshold maximo de volatilidade.
    volatility_threshold_max_mode: str = "atr_percentile"
    # Percentil minimo de ATR para liberar operacao.
    volatility_p_min: float = 40.0
    # Percentil maximo de ATR para liberar operacao.
    volatility_p_max: float = 95.0
    # Threshold absoluto minimo de ATR (quando modo=absolute).
    volatility_abs_min: float = 0.0
    # Threshold absoluto maximo de ATR (quando modo=absolute).
    volatility_abs_max: float = 1e9
    # Risco por trade deste perfil (em % do saldo).
    risk_pct: float = 0.5
    # Threshold minimo de probabilidade para entrada.
    signal_threshold: float = 0.50
    # Diferenca minima entre p_buy e p_sell para evitar sinais ruidosos.
    min_signal_margin: float = 0.0
    # Threshold especifico para BUY. Se vazio, usa signal_threshold.
    buy_signal_threshold: float | None = None
    # Threshold especifico para SELL. Se vazio, usa signal_threshold.
    sell_signal_threshold: float | None = None
    # Margem minima especifica para BUY. Se vazio, usa min_signal_margin.
    buy_min_signal_margin: float | None = None
    # Margem minima especifica para SELL. Se vazio, usa min_signal_margin.
    sell_min_signal_margin: float | None = None
    # Habilita/desabilita o perfil.
    enabled: bool = True
    # Permite bypass do gate quando gate=WAIT e sinal do entry for muito forte (modo scalp).
    allow_gate_wait_bypass: bool = False
    # Threshold minimo da probabilidade max(p_buy,p_sell) para bypass do gate=WAIT.
    gate_wait_bypass_threshold: float = 0.85
    # Exige impulso recente alinhado com a direcao do trade.
    impulse_alignment_required: bool = False
    # Quantidade de barras para medir o impulso curto (usa log_return_{n} quando existir).
    impulse_lookback_bars: int = 3
    # Magnitude minima do impulso para considerar entrada alinhada.
    impulse_min_abs_return: float = 0.0
    # Thresholds dinamicos por regime detectado em tempo real.
    # Exemplo:
    # {"LOW_VOL_SIDEWAYS":{"signal_threshold":0.60,"min_signal_margin":0.12}}
    regime_thresholds: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class GlobalRiskConfig(BaseModel):
    # Limite de risco total agregado por simbolo (em % do saldo).
    max_total_risk_pct_symbol: float = 2.0
    # Maximo de posicoes abertas por simbolo (global entre TFs).
    max_total_open_positions_symbol: int = 3
    # Perda diaria maxima por simbolo (em % do saldo).
    max_total_daily_loss_symbol: float = 5.0
    # Exposicao maxima em lotes por simbolo.
    max_total_exposure_lots_symbol: float = 10.0
    # Maximo de novas ordens por janela de rate limit.
    max_new_orders_per_minute_symbol: int = 1
    # Tamanho da janela de rate limit.
    order_rate_window_seconds: int = 60


class FeatureConfig(BaseModel):
    # Janela do ATR.
    atr_window: int = 14
    # Janela do RSI.
    rsi_window: int = 14
    # Janelas de medias moveis exponenciais.
    ema_windows: tuple[int, int, int] = (9, 21, 50)
    # Parametro do fractional differencing.
    fracdiff_d: float = 0.35


class FundamentalConfig(BaseModel):
    # Caminho do calendario economico local (CSV).
    calendar_csv: Path = Path(__file__).resolve().parents[1] / "data" / "calendar_2025_2026.csv"
    # Janela (em minutos) para feature "high_impact_in_next_60min".
    next_event_window_minutes: int = 60


class AppConfig(BaseModel):
    # Diretorio raiz do projeto.
    root_dir: Path = Path(__file__).resolve().parents[1]
    # Dados brutos coletados do MT5.
    data_raw_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw"
    # Dados processados (features/datasets).
    data_processed_dir: Path = Path(__file__).resolve().parents[1] / "data" / "processed"
    # Modelos treinados salvos.
    models_dir: Path = Path(__file__).resolve().parents[1] / "models"
    # Logs do bot.
    logs_dir: Path = Path(__file__).resolve().parents[1] / "logs"
    # Relatorios (backtest, fases, paper).
    reports_dir: Path = Path(__file__).resolve().parents[1] / "reports"
    # Fuso horario padrao do app.
    timezone: str = "UTC"
    risk: RiskConfig = RiskConfig()
    global_risk: GlobalRiskConfig = GlobalRiskConfig()
    triple_barrier: TripleBarrierConfig = TripleBarrierConfig()
    live: LiveConfig = LiveConfig()
    timeframe_profiles: Dict[str, TimeframeConfig] = Field(
        default_factory=lambda: {
            "M5": TimeframeConfig(
                tf_entry="M5",
                tf_gate="M30",
                gate_mode="strict",
                gate_min_margin_block=0.10,
                horizon_candles=12,
                max_trades_per_hour=1,
                trades_window_mode="rolling_60m",
                min_candles_between_same_direction_trades=5,
                reentry_block_candles=5,
                volatility_threshold_min_mode="atr_percentile",
                volatility_threshold_max_mode="atr_percentile",
                volatility_p_min=60.0,
                volatility_p_max=84.0,
                risk_pct=0.5,
                signal_threshold=0.59,
                min_signal_margin=0.19,
                buy_signal_threshold=0.62,
                sell_signal_threshold=0.58,
                buy_min_signal_margin=0.22,
                sell_min_signal_margin=0.16,
                enabled=True,
                impulse_alignment_required=False,
                impulse_lookback_bars=3,
                impulse_min_abs_return=0.0,
                regime_thresholds={
                    "LOW_VOL_SIDEWAYS": {
                        "signal_threshold": 0.62,
                        "min_signal_margin": 0.16,
                        "buy_signal_threshold": 0.65,
                        "sell_signal_threshold": 0.61,
                        "buy_min_signal_margin": 0.24,
                        "sell_min_signal_margin": 0.18,
                    },
                    "TRENDING_STRONG": {
                        "signal_threshold": 0.54,
                        "min_signal_margin": 0.10,
                        "buy_signal_threshold": 0.57,
                        "sell_signal_threshold": 0.52,
                        "buy_min_signal_margin": 0.12,
                        "sell_min_signal_margin": 0.08,
                    },
                    "HIGH_VOL_BREAKOUT": {
                        "signal_threshold": 0.58,
                        "min_signal_margin": 0.14,
                        "buy_signal_threshold": 0.61,
                        "sell_signal_threshold": 0.57,
                        "buy_min_signal_margin": 0.18,
                        "sell_min_signal_margin": 0.12,
                    },
                    "POST_NEWS_SHOCK": {
                        "signal_threshold": 0.64,
                        "min_signal_margin": 0.20,
                        "buy_signal_threshold": 0.67,
                        "sell_signal_threshold": 0.63,
                        "buy_min_signal_margin": 0.24,
                        "sell_min_signal_margin": 0.18,
                    },
                    "RANGING": {
                        "signal_threshold": 0.60,
                        "min_signal_margin": 0.14,
                        "buy_signal_threshold": 0.63,
                        "sell_signal_threshold": 0.59,
                        "buy_min_signal_margin": 0.20,
                        "sell_min_signal_margin": 0.12,
                    },
                },
            ),
            "M30": TimeframeConfig(
                tf_entry="M30",
                tf_gate="H4",
                gate_mode="strict",
                gate_min_margin_block=0.10,
                horizon_candles=6,
                max_trades_per_hour=2,
                trades_window_mode="fixed_hour",
                min_candles_between_same_direction_trades=2,
                reentry_block_candles=2,
                volatility_threshold_min_mode="atr_percentile",
                volatility_threshold_max_mode="atr_percentile",
                volatility_p_min=35.0,
                volatility_p_max=97.0,
                risk_pct=0.75,
                enabled=False,
                impulse_alignment_required=False,
                impulse_lookback_bars=3,
                impulse_min_abs_return=0.0,
                regime_thresholds={
                    "LOW_VOL_SIDEWAYS": {"signal_threshold": 0.58, "min_signal_margin": 0.10},
                    "TRENDING_STRONG": {"signal_threshold": 0.47, "min_signal_margin": 0.05},
                    "HIGH_VOL_BREAKOUT": {"signal_threshold": 0.53, "min_signal_margin": 0.09},
                    "POST_NEWS_SHOCK": {"signal_threshold": 0.60, "min_signal_margin": 0.14},
                    "RANGING": {"signal_threshold": 0.56, "min_signal_margin": 0.09},
                },
            ),
            "M1": TimeframeConfig(
                tf_entry="M1",
                tf_gate="M15",
                gate_mode="allow_wait",
                gate_min_margin_block=0.12,
                horizon_candles=30,
                max_trades_per_hour=3,
                trades_window_mode="rolling_60m",
                min_candles_between_same_direction_trades=5,
                reentry_block_candles=3,
                volatility_threshold_min_mode="atr_percentile",
                volatility_threshold_max_mode="atr_percentile",
                volatility_p_min=40.0,
                volatility_p_max=98.0,
                risk_pct=0.25,
                enabled=False,
                allow_gate_wait_bypass=True,
                gate_wait_bypass_threshold=0.80,
                impulse_alignment_required=True,
                impulse_lookback_bars=3,
                impulse_min_abs_return=0.00005,
                regime_thresholds={
                    "LOW_VOL_SIDEWAYS": {"signal_threshold": 0.62, "min_signal_margin": 0.14},
                    "TRENDING_STRONG": {"signal_threshold": 0.50, "min_signal_margin": 0.08},
                    "HIGH_VOL_BREAKOUT": {"signal_threshold": 0.56, "min_signal_margin": 0.11},
                    "POST_NEWS_SHOCK": {"signal_threshold": 0.64, "min_signal_margin": 0.16},
                    "RANGING": {"signal_threshold": 0.60, "min_signal_margin": 0.12},
                },
            ),
        }
    )
    feature: FeatureConfig = FeatureConfig()
    fundamentals: FundamentalConfig = FundamentalConfig()
    fracdiff_threshold: float = 1e-5

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.logs_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()


def _apply_timeframe_overrides() -> None:
    path = CONFIG.reports_dir / "runs" / "timeframe_overrides.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    if not isinstance(profiles, dict):
        return
    for tf, vals in profiles.items():
        if tf not in CONFIG.timeframe_profiles or not isinstance(vals, dict):
            continue
        current = CONFIG.timeframe_profiles[tf]
        allowed = set(current.model_fields.keys())
        updates = {k: v for k, v in vals.items() if k in allowed}
        if not updates:
            continue
        CONFIG.timeframe_profiles[tf] = current.model_copy(update=updates)


_apply_timeframe_overrides()


def _load_symbol_profiles() -> dict[tuple[str, str], dict]:
    path = CONFIG.reports_dir / "runs" / "symbol_profiles.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("profiles", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        tf = str(item.get("tf", "")).upper().strip()
        if not symbol or not tf:
            continue
        out[(symbol, tf)] = item
    return out


def get_symbol_profile_entry(symbol: str, timeframe: str) -> dict:
    return _load_symbol_profiles().get((str(symbol).upper().strip(), str(timeframe).upper().strip()), {})


def resolve_timeframe_profile(symbol: str, timeframe: str) -> TimeframeConfig | None:
    base = CONFIG.timeframe_profiles.get(str(timeframe).upper().strip())
    if base is None:
        return None
    item = get_symbol_profile_entry(symbol=symbol, timeframe=timeframe)
    if not base.enabled:
        enabled_override = False
        if isinstance(item.get("profile_updates"), dict):
            enabled_override = bool(item["profile_updates"].get("enabled", False))
        if not enabled_override:
            return None
    if not item:
        return base
    updates = {}
    if isinstance(item.get("profile_updates"), dict):
        updates.update({k: v for k, v in item["profile_updates"].items() if k in base.model_fields})
    # Compatibilidade com formato antigo do menu.
    if item.get("gate"):
        updates["tf_gate"] = str(item.get("gate")).upper().strip()
    return base.model_copy(update=updates) if updates else base


def resolve_session_settings(symbol: str, timeframe: str) -> tuple[bool, tuple[str, ...]]:
    use_filter = bool(CONFIG.live.use_session_filter)
    windows = tuple(str(x) for x in CONFIG.live.allowed_sessions_utc)
    item = get_symbol_profile_entry(symbol=symbol, timeframe=timeframe)
    if item:
        if "use_session_filter" in item:
            use_filter = bool(item.get("use_session_filter"))
        raw_windows = item.get("allowed_sessions_utc")
        if isinstance(raw_windows, (list, tuple)) and raw_windows:
            windows = tuple(str(x) for x in raw_windows if str(x).strip())
    return use_filter, windows


def resolve_exit_settings(symbol: str, timeframe: str) -> dict[str, float | bool | int]:
    item = get_symbol_profile_entry(symbol=symbol, timeframe=timeframe)
    out: dict[str, float | bool | int] = {
        "break_even_enabled": False,
        "break_even_r": 0.5,
        "break_even_offset_r": 0.0,
        "trailing_enabled": False,
        "trailing_activation_r": 1.0,
        "trailing_atr_mult": 1.0,
        "max_holding_candles_override": 0,
    }
    for k in list(out.keys()):
        if k in item:
            out[k] = item[k]
    return out
