from __future__ import annotations

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
    # Fecha posicoes abertas apos X candles (0 desliga).
    max_holding_candles: int = 0
    # Frequencia do monitor de saude (eventos) para logar estatisticas.
    health_check_every_n_events: int = 50
    # Alerta se taxa de sinal ficar abaixo deste valor no monitor.
    min_signal_rate_alert: float = 0.01
    # Alerta se confianca media (max prob) ficar abaixo deste valor.
    min_confidence_alert: float = 0.55


class FeatureConfig(BaseModel):
    # Janela do ATR.
    atr_window: int = 14
    # Janela do RSI.
    rsi_window: int = 14
    # Janelas de medias moveis exponenciais.
    ema_windows: tuple[int, int, int] = (9, 21, 50)
    # Parametro do fractional differencing.
    fracdiff_d: float = 0.35


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
    triple_barrier: TripleBarrierConfig = TripleBarrierConfig()
    live: LiveConfig = LiveConfig()
    feature: FeatureConfig = FeatureConfig()
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
