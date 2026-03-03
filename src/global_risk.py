from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import CONFIG

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


@dataclass
class GlobalRiskSnapshot:
    risk_current_pct: float
    positions_count: int
    daily_loss_pct: float
    exposure_lots: float
    new_orders_last_window: int


class GlobalRiskCoordinator:
    """Coordena limites globais de risco por simbolo para evitar conflito entre TFs."""

    def __init__(self) -> None:
        self._open_ticket_risk: dict[str, dict[int, float]] = defaultdict(dict)
        self._new_orders_ts: dict[str, deque[datetime]] = defaultdict(deque)

    def register_open_order(self, symbol: str, ticket: int, risk_pct: float) -> None:
        self._open_ticket_risk[symbol][int(ticket)] = float(max(0.0, risk_pct))
        self._new_orders_ts[symbol].append(datetime.now(timezone.utc))
        self._gc_order_window(symbol)

    def unregister_closed_ticket(self, symbol: str, ticket: int) -> None:
        self._open_ticket_risk[symbol].pop(int(ticket), None)

    def _gc_order_window(self, symbol: str) -> None:
        now = datetime.now(timezone.utc)
        win = max(1, int(CONFIG.global_risk.order_rate_window_seconds))
        dq = self._new_orders_ts[symbol]
        while dq and (now - dq[0]).total_seconds() > win:
            dq.popleft()

    def _positions_metrics(self, symbol: str) -> tuple[int, float]:
        if mt5 is None:
            return 0, 0.0
        pos = mt5.positions_get(symbol=symbol)
        if pos is None:
            return 0, 0.0
        count = len(pos)
        lots = 0.0
        for p in pos:
            lots += float(getattr(p, "volume", 0.0) or 0.0)
        return count, lots

    def snapshot(self, symbol: str, proposed_risk_pct: float = 0.0) -> GlobalRiskSnapshot:
        self._gc_order_window(symbol)
        info = mt5.account_info() if mt5 else None
        balance = float(getattr(info, "balance", 0.0) or 0.0)
        equity = float(getattr(info, "equity", 0.0) or 0.0)
        daily_pnl = equity - balance
        daily_loss_pct = max(0.0, (-daily_pnl / balance) * 100.0) if balance > 0 else 0.0
        positions_count, exposure_lots = self._positions_metrics(symbol)
        tracked = sum(self._open_ticket_risk[symbol].values())
        risk_current_pct = float(max(0.0, tracked + max(0.0, proposed_risk_pct)))
        return GlobalRiskSnapshot(
            risk_current_pct=risk_current_pct,
            positions_count=int(positions_count),
            daily_loss_pct=float(daily_loss_pct),
            exposure_lots=float(exposure_lots),
            new_orders_last_window=int(len(self._new_orders_ts[symbol])),
        )

    def can_open(self, symbol: str, proposed_risk_pct: float, proposed_lot: float) -> tuple[bool, str, GlobalRiskSnapshot]:
        cfg = CONFIG.global_risk
        snap = self.snapshot(symbol, proposed_risk_pct=proposed_risk_pct)
        if snap.risk_current_pct > float(cfg.max_total_risk_pct_symbol):
            return False, "blocked_by_global_risk", snap
        if snap.positions_count >= int(cfg.max_total_open_positions_symbol):
            return False, "blocked_by_global_positions", snap
        if snap.daily_loss_pct >= float(cfg.max_total_daily_loss_symbol):
            return False, "blocked_by_global_daily_loss", snap
        if (snap.exposure_lots + float(proposed_lot)) > float(cfg.max_total_exposure_lots_symbol):
            return False, "blocked_by_global_exposure", snap
        if snap.new_orders_last_window >= int(cfg.max_new_orders_per_minute_symbol):
            return False, "blocked_by_order_rate_limit", snap
        return True, "OK", snap

