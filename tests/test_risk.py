from datetime import datetime, timedelta, timezone

from src.risk_manager import AccountState, TodayStats, can_open_trade


def test_can_open_trade_ok():
    state = AccountState(balance=1000, equity=1000, margin_level_pct=500, daily_pnl=0)
    stats = TodayStats(trades_count=0, last_trade_time=None)
    ok, reason = can_open_trade(state, open_positions=0, today_stats=stats)
    assert ok is True
    assert reason == "OK"


def test_can_open_trade_blocked_by_cooldown():
    state = AccountState(balance=1000, equity=990, margin_level_pct=500, daily_pnl=-10)
    stats = TodayStats(
        trades_count=1, last_trade_time=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    ok, reason = can_open_trade(state, open_positions=0, today_stats=stats)
    assert ok is False
    assert reason == "COOLDOWN"

