from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict


try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


@dataclass
class MT5Credentials:
    login: int | None = None
    password: str | None = None
    server: str | None = None


def initialize() -> bool:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    terminal_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if os.path.exists(terminal_path):
        return bool(mt5.initialize(path=terminal_path))
    return bool(mt5.initialize())


def shutdown() -> None:
    if mt5 is None:
        return
    mt5.shutdown()


def ensure_logged_in(credentials: MT5Credentials | None = None) -> bool:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    terminal_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    init_ok = (
        bool(mt5.initialize(path=terminal_path))
        if os.path.exists(terminal_path)
        else bool(mt5.initialize())
    )
    if not init_ok:
        return False
    if credentials is None:
        env_login = os.getenv("MT5_LOGIN")
        env_password = os.getenv("MT5_PASSWORD")
        env_server = os.getenv("MT5_SERVER")
        if env_login and env_password and env_server:
            credentials = MT5Credentials(
                login=int(env_login),
                password=env_password,
                server=env_server,
            )
    if credentials and credentials.login and credentials.password and credentials.server:
        return bool(
            mt5.login(
                login=credentials.login,
                password=credentials.password,
                server=credentials.server,
            )
        )
    account = mt5.account_info()
    return account is not None


def health_check() -> Dict[str, Any]:
    if mt5 is None:
        return {"ok": False, "error": "MetaTrader5 package not installed"}
    terminal = mt5.terminal_info()
    account = mt5.account_info()
    last_error = mt5.last_error()
    return {
        "ok": terminal is not None and account is not None,
        "terminal_connected": bool(terminal.connected) if terminal else False,
        "trade_allowed": bool(terminal.trade_allowed) if terminal else False,
        "account_login": getattr(account, "login", None),
        "balance": getattr(account, "balance", None),
        "margin_level": getattr(account, "margin_level", None),
        "last_error": last_error,
    }
