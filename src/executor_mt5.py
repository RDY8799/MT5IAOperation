from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG
from .mt5_connect import ensure_logged_in

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


def _json_log(payload: dict[str, Any], file_name: str = "bot_live.log") -> None:
    CONFIG.ensure_dirs()
    path: Path = CONFIG.logs_dir / file_name
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["timestamp_local"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    payload["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def send_order(
    symbol: str,
    side: str,
    lot: float,
    sl: float,
    tp: float,
    deviation_points: int,
) -> tuple[bool, Any]:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not ensure_logged_in():
        _json_log({"event": "order_fail", "reason": "mt5_login_failed"})
        return False, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        _json_log({"event": "order_fail", "symbol": symbol, "reason": "no_tick"})
        return False, None
    order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation_points,
        "magic": CONFIG.live.magic_number,
        "comment": "mt5_ai_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    response = mt5.order_send(request)
    ok = response is not None and getattr(response, "retcode", None) == mt5.TRADE_RETCODE_DONE

    _json_log(
        {
            "event": "order_send",
            "symbol": symbol,
            "side": side,
            "request": request,
            "response": response._asdict() if response else None,
            "ok": ok,
        }
    )
    return ok, response


def close_position(
    symbol: str,
    ticket: int,
    volume: float,
    side: str,
    deviation_points: int,
) -> tuple[bool, Any]:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not ensure_logged_in():
        _json_log({"event": "close_fail", "reason": "mt5_login_failed", "ticket": ticket})
        return False, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        _json_log({"event": "close_fail", "symbol": symbol, "ticket": ticket, "reason": "no_tick"})
        return False, None

    # To close BUY position send SELL, to close SELL position send BUY.
    close_type = mt5.ORDER_TYPE_SELL if side.upper() == "BUY" else mt5.ORDER_TYPE_BUY
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": int(ticket),
        "price": price,
        "deviation": deviation_points,
        "magic": CONFIG.live.magic_number,
        "comment": "mt5_ai_bot_time_stop",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    response = mt5.order_send(request)
    ok = response is not None and getattr(response, "retcode", None) == mt5.TRADE_RETCODE_DONE
    _json_log(
        {
            "event": "position_close_send",
            "symbol": symbol,
            "ticket": int(ticket),
            "request": request,
            "response": response._asdict() if response else None,
            "ok": ok,
        }
    )
    return ok, response
