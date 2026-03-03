from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import CONFIG


@dataclass
class AccountState:
    balance: float
    equity: float
    margin_level_pct: float
    daily_pnl: float


@dataclass
class TodayStats:
    trades_count: int
    last_trade_time: datetime | None


def can_open_trade(
    account_state: AccountState, open_positions: int, today_stats: TodayStats
) -> tuple[bool, str]:
    cfg = CONFIG.risk
    if open_positions >= cfg.max_open_positions:
        return False, "MAX_OPEN_POSITIONS"
    if today_stats.trades_count >= cfg.max_trades_per_day:
        return False, "MAX_TRADES_PER_DAY"
    if account_state.balance > 0:
        daily_loss_pct = max(0.0, (-account_state.daily_pnl / account_state.balance) * 100.0)
        if daily_loss_pct >= cfg.max_daily_loss_pct:
            return False, "MAX_DAILY_LOSS_PCT"
    if account_state.margin_level_pct < cfg.min_margin_level_pct:
        return False, "MIN_MARGIN_LEVEL_PCT"
    if today_stats.last_trade_time is not None:
        elapsed = datetime.now(timezone.utc) - today_stats.last_trade_time
        if elapsed.total_seconds() < cfg.cooldown_minutes * 60:
            return False, "COOLDOWN"
    return True, "OK"


def compute_position_size(balance: float, risk_pct: float, sl_pips: float) -> float:
    _ = (balance, risk_pct, sl_pips)
    return CONFIG.risk.fixed_demo_lot

