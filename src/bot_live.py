from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CONFIG, TimeframeConfig, resolve_exit_settings, resolve_session_settings, resolve_timeframe_profile
from .executor_mt5 import close_position, modify_position_sltp, send_order
from .features import build_features
from .fundamentals import fetch_dxy_strength, get_fundamental_window_status
from .global_risk import GlobalRiskCoordinator
from .multitf import BUY, SELL, WAIT, align_h4_to_h1, apply_policy
from .mt5_connect import ensure_logged_in, shutdown
from .model_registry import get_latest_model, get_model
from .notifier_telegram import send_telegram
from .risk_manager import AccountState, TodayStats, can_open_trade
from .utils_time import timeframe_minutes, wait_seconds_to_next_candle

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    from rich.console import Console
except ImportError:  # pragma: no cover
    Console = None


TF_TO_MT5 = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None),
    "M5": getattr(mt5, "TIMEFRAME_M5", None),
    "M15": getattr(mt5, "TIMEFRAME_M15", None),
    "M30": getattr(mt5, "TIMEFRAME_M30", None),
    "H1": getattr(mt5, "TIMEFRAME_H1", None),
    "H4": getattr(mt5, "TIMEFRAME_H4", None),
    "D1": getattr(mt5, "TIMEFRAME_D1", None),
}

DROP_COLS = {"time", "y", "t1", "pt", "sl", "regime_class"}
_CONSOLE = Console() if Console else None
_GLOBAL_COORD = GlobalRiskCoordinator()


@dataclass
class LiveProtectionState:
    realized_pnl: float = 0.0
    realized_peak_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    same_side_block_until: dict[str, datetime | None] = field(
        default_factory=lambda: {"BUY": None, "SELL": None}
    )


@dataclass
class ManagedPositionState:
    ticket: int
    side: str
    entry_price: float
    initial_sl: float
    initial_tp: float
    break_even_done: bool = False
    trailing_active: bool = False


def _fmt_local_ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _fmt_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _pretty_line(payload: dict) -> str:
    event = payload.get("event", "event")
    symbol = payload.get("symbol", "-")
    tf = payload.get("tf", "-")
    decision = payload.get("decision")
    reason = payload.get("reason") or payload.get("can_open_reason")
    probs = payload.get("probs", {}) or {}
    p_buy = probs.get("buy")
    p_sell = probs.get("sell")
    event_pt = {
        "decision": "DECISAO",
        "decision_blocked": "DECISAO_BLOQUEADA",
        "order_send": "ORDEM_ENVIADA",
        "order_fail": "FALHA_ORDEM",
        "kill_switch": "KILL_SWITCH",
        "position_closed_detected": "POSICAO_FECHADA",
        "time_stop_close": "TIME_STOP",
        "health_monitor": "SAUDE_MODELO",
        "strong_signal_blocked": "SINAL_FORTE_BLOQUEADO",
        "telegram_notify_failed": "FALHA_TELEGRAM",
    }.get(str(event), str(event).upper())
    decision_pt = {
        "BUY": "COMPRA",
        "SELL": "VENDA",
        "WAIT": "AGUARDAR",
    }.get(str(decision), str(decision) if decision is not None else None)
    reason_pt = {
        "OK": "OK",
        "COOLDOWN": "EM_COOLDOWN",
        "MAX_OPEN_POSITIONS": "MAX_POSICOES_ABERTAS",
        "MAX_SYMBOL_POSITIONS_LIVE": "MAX_POSICOES_POR_ATIVO",
        "MAX_TRADES_PER_DAY": "MAX_TRADES_DIA",
        "MAX_DAILY_LOSS_PCT": "MAX_PERDA_DIA",
        "MIN_MARGIN_LEVEL_PCT": "MARGEM_BAIXA",
        "SESSION_FILTER": "FORA_DA_SESSAO",
        "NEWS_BLACKOUT": "JANELA_NOTICIA",
        "FUNDAMENTAL_EVENT": "EVENTO_FUNDAMENTAL",
        "LOSS_STREAK_COOLDOWN": "PAUSA_POR_PERDAS",
        "SAME_SIDE_LOSS_PAUSE": "PAUSA_MESMA_DIRECAO",
        "PROFIT_GIVEBACK_LOCK": "PAUSA_POR_DEVOLVER_LUCRO",
        "SPREAD_FILTER": "SPREAD_ALTO",
        "blocked_by_impulse_alignment": "IMPULSO_FRACO",
    }.get(str(reason), str(reason) if reason else "")

    parts = [f"[{event_pt}]", f"{symbol}/{tf}"]
    if decision_pt is not None:
        parts.append(f"sinal={decision_pt}")
    if p_buy is not None and p_sell is not None:
        parts.append(f"compra={float(p_buy)*100:.1f}% venda={float(p_sell)*100:.1f}%")
    if reason_pt:
        parts.append(f"motivo={reason_pt}")
    if "order_id" in payload and payload.get("order_id") is not None:
        parts.append(f"ordem={payload.get('order_id')}")
    if "ticket" in payload:
        parts.append(f"ticket={payload.get('ticket')}")
    if payload.get("entry_signal") is not None:
        parts.append(f"entry={payload.get('entry_signal')}")
    if payload.get("gate_signal") is not None:
        parts.append(f"gate={payload.get('gate_signal')}")
    if payload.get("conflict_detail"):
        parts.append(f"conflito={payload.get('conflict_detail')}")
    if payload.get("has_gate_pred") is not None:
        parts.append(f"gate_pred={payload.get('has_gate_pred')}")
    if payload.get("has_entry_pred") is not None:
        parts.append(f"entry_pred={payload.get('has_entry_pred')}")
    if payload.get("reason_missing_gate_pred"):
        parts.append(f"missing_gate={payload.get('reason_missing_gate_pred')}")
    if payload.get("reason_missing_entry_pred"):
        parts.append(f"missing_entry={payload.get('reason_missing_entry_pred')}")
    if payload.get("entry_prob_diff") is not None:
        parts.append(f"margem_entry={float(payload.get('entry_prob_diff')):.3f}")
    if payload.get("gate_prob_diff") is not None:
        parts.append(f"margem_gate={float(payload.get('gate_prob_diff')):.3f}")
    if payload.get("threshold_used") is not None:
        parts.append(f"thr={float(payload.get('threshold_used')):.2f}")
    if payload.get("min_signal_margin_used") is not None:
        parts.append(f"min_margem={float(payload.get('min_signal_margin_used')):.2f}")
    if payload.get("global_risk_current_pct") is not None:
        parts.append(f"risco_global={float(payload.get('global_risk_current_pct')):.2f}%")
    if payload.get("max_trades_window_count") is not None:
        parts.append(f"trades_60m={int(payload.get('max_trades_window_count'))}")
    return " | ".join(parts)


def _pretty_color(event: str) -> str:
    if event in {"kill_switch", "order_fail", "telegram_notify_failed"}:
        return "red"
    if event in {"order_send", "position_close_send", "time_stop_close", "position_closed_detected"}:
        return "green"
    if event in {"decision_blocked", "health_monitor"}:
        return "yellow"
    return "cyan"


def _sleep_with_countdown(seconds: float, symbol: str, timeframe: str) -> None:
    total = max(0.0, float(seconds))
    whole = int(total)
    frac = total - whole
    next_dt = datetime.now().astimezone() + timedelta(seconds=total)
    for remaining in range(whole, 0, -1):
        mm = remaining // 60
        ss = remaining % 60
        now_str = datetime.now().astimezone().strftime("%H:%M:%S")
        next_str = next_dt.strftime("%H:%M:%S")
        msg = (
            f"Agora {now_str} | Proxima analise {symbol}/{timeframe} "
            f"as {next_str} (em {mm:02d}:{ss:02d})"
        )
        if _CONSOLE is not None:
            _CONSOLE.print(f"[bold blue]{msg}[/]", end="\r", markup=True, soft_wrap=False)
        else:
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()
        time.sleep(1)
    if frac > 0:
        time.sleep(frac)
    # Clear countdown line.
    clear = " " * 80
    if _CONSOLE is not None:
        _CONSOLE.print(clear, end="\r", markup=False, soft_wrap=False)
    else:
        sys.stdout.write("\r" + clear + "\r")
        sys.stdout.flush()


def _json_log(payload: dict, file_name: str = "bot_live.log") -> None:
    CONFIG.ensure_dirs()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["timestamp_local"] = _fmt_local_ts()
    payload["timestamp_utc"] = _fmt_utc_ts()
    path: Path = CONFIG.logs_dir / file_name
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    human_line = f"{payload['timestamp_local']} | {_pretty_line(payload)}"
    human_path: Path = CONFIG.logs_dir / "bot_live_human.log"
    with human_path.open("a", encoding="utf-8") as f:
        f.write(human_line + "\n")
    if _CONSOLE is not None:
        _CONSOLE.print(
            human_line,
            style=_pretty_color(str(payload.get("event", ""))),
            markup=False,
            soft_wrap=True,
        )


def _update_daily_report(payload: dict) -> None:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = CONFIG.reports_dir / f"daily_{day}.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"date": day, "events": 0, "decisions": {"BUY": 0, "SELL": 0, "WAIT": 0}, "last_event": {}}
    report["events"] += 1
    decision = payload.get("decision")
    if decision in report["decisions"]:
        report["decisions"][decision] += 1
    report["last_event"] = payload
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _notify_telegram(message: str, key: str, cooldown_seconds: int = 0) -> None:
    ok = send_telegram(message=message, key=key, cooldown_seconds=cooldown_seconds)
    if not ok:
        _json_log(
            {
                "event": "telegram_notify_failed",
                "key": key,
                "has_token": bool(os.getenv("MT5_TELEGRAM_BOT_TOKEN")),
                "has_chat_id": bool(os.getenv("MT5_TELEGRAM_CHAT_ID")),
            }
        )


def _emoji_decision(decision: str) -> str:
    return {"BUY": "ðŸŸ¢ COMPRA", "SELL": "ðŸ”´ VENDA", "WAIT": "ðŸŸ¡ AGUARDAR"}.get(decision, decision)


def _latest_model(symbol: str, timeframe: str) -> Path:
    return get_latest_model(symbol=symbol, timeframe=timeframe).model_path


def _fetch_recent(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    tf = TF_TO_MT5[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    for col in ("spread", "real_volume"):
        if col not in df.columns:
            df[col] = pd.NA
    return df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]


def _account_state(symbol: str) -> AccountState:
    info = mt5.account_info() if mt5 else None
    if info is None:
        return AccountState(balance=0.0, equity=0.0, margin_level_pct=0.0, daily_pnl=0.0)
    daily_pnl = float(info.equity - info.balance)
    margin_level = float(info.margin_level or 0.0)
    if margin_level <= 0 and _positions_count(symbol) == 0:
        margin_level = 9999.0
    return AccountState(
        balance=float(info.balance),
        equity=float(info.equity),
        margin_level_pct=margin_level,
        daily_pnl=daily_pnl,
    )


def _positions_count(symbol: str) -> int:
    if mt5 is None:
        return 0
    pos = mt5.positions_get(symbol=symbol)
    return 0 if pos is None else len(pos)


def _position_state_path(symbol: str, timeframe: str) -> Path:
    safe_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in {"_", "-"})
    safe_tf = "".join(ch for ch in timeframe.upper() if ch.isalnum() or ch in {"_", "-"})
    return CONFIG.reports_dir / "runs" / f"live_position_state_{safe_symbol}_{safe_tf}.json"


def _load_position_state(symbol: str, timeframe: str) -> dict[int, ManagedPositionState]:
    path = _position_state_path(symbol, timeframe)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, ManagedPositionState] = {}
    rows = data.get("positions", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            state = ManagedPositionState(
                ticket=int(item["ticket"]),
                side=str(item["side"]),
                entry_price=float(item["entry_price"]),
                initial_sl=float(item["initial_sl"]),
                initial_tp=float(item.get("initial_tp", 0.0) or 0.0),
                break_even_done=bool(item.get("break_even_done", False)),
                trailing_active=bool(item.get("trailing_active", False)),
            )
        except Exception:
            continue
        out[state.ticket] = state
    return out


def _save_position_state(symbol: str, timeframe: str, state_map: dict[int, ManagedPositionState]) -> None:
    path = _position_state_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "positions": [
            {
                "ticket": s.ticket,
                "side": s.side,
                "entry_price": s.entry_price,
                "initial_sl": s.initial_sl,
                "initial_tp": s.initial_tp,
                "break_even_done": s.break_even_done,
                "trailing_active": s.trailing_active,
            }
            for s in sorted(state_map.values(), key=lambda x: x.ticket)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_lot(symbol: str, lot: float) -> float:
    cfg = CONFIG.risk
    if mt5 is None:
        return max(cfg.min_lot, min(cfg.max_lot, lot))
    info = mt5.symbol_info(symbol)
    vol_min = float(getattr(info, "volume_min", cfg.min_lot) or cfg.min_lot)
    vol_max = float(getattr(info, "volume_max", cfg.max_lot) or cfg.max_lot)
    vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
    lo = max(vol_min, cfg.min_lot)
    hi = min(vol_max, cfg.max_lot)
    clipped = max(lo, min(hi, lot))
    steps = math.floor(clipped / vol_step)
    normalized = steps * vol_step
    if normalized < lo:
        normalized = lo
    decimals = max(0, len(str(vol_step).split(".")[-1].rstrip("0")))
    return float(round(normalized, decimals))


def _compute_live_lot(
    symbol: str,
    side: str,
    balance: float,
    entry_price: float,
    sl_price: float,
    risk_pct_override: float | None = None,
) -> float:
    cfg = CONFIG.risk
    fallback = _normalize_lot(symbol, cfg.fixed_demo_lot)
    if not cfg.use_dynamic_position_sizing:
        return fallback
    if mt5 is None or balance <= 0:
        return fallback
    risk_pct = float(cfg.default_risk_pct if risk_pct_override is None else risk_pct_override)
    risk_amount = balance * (risk_pct / 100.0)
    if risk_amount <= 0:
        return fallback
    order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
    loss_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, sl_price)
    if loss_one_lot is None:
        return fallback
    risk_per_lot = abs(float(loss_one_lot))
    if risk_per_lot <= 0:
        return fallback
    raw_lot = risk_amount / risk_per_lot
    return _normalize_lot(symbol, raw_lot)


def _compute_shap_top(model, row: pd.DataFrame) -> list[dict]:
    if shap is None:
        return []
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(row)
        arr = np.asarray(values)
        n_features = row.shape[1]

        if arr.ndim == 1:
            v = np.abs(arr)
        elif arr.ndim == 2:
            if arr.shape[0] == 1 and arr.shape[1] == n_features:
                v = np.abs(arr[0])
            elif arr.shape[1] == n_features:
                v = np.mean(np.abs(arr), axis=0)
            elif arr.shape[0] == n_features:
                v = np.mean(np.abs(arr), axis=1)
            else:
                return []
        else:
            feat_axis = next((ax for ax, size in enumerate(arr.shape) if size == n_features), None)
            if feat_axis is None:
                return []
            axes = tuple(ax for ax in range(arr.ndim) if ax != feat_axis)
            v = np.mean(np.abs(arr), axis=axes)

        v = np.asarray(v).reshape(-1)
        if len(v) != n_features:
            return []
        order = np.argsort(v)[::-1][:5]
        return [{"feature": row.columns[i], "value": float(v[i])} for i in order]
    except Exception:
        return []


def _parse_hhmm(value: str) -> tuple[int, int]:
    hh, mm = value.strip().split(":")
    return int(hh), int(mm)


def _is_in_session(now_utc: datetime, windows: tuple[str, ...]) -> bool:
    if not windows:
        return True
    current = now_utc.hour * 60 + now_utc.minute
    for win in windows:
        if "-" not in win:
            continue
        start_s, end_s = win.split("-", 1)
        sh, sm = _parse_hhmm(start_s)
        eh, em = _parse_hhmm(end_s)
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        if start_m <= end_m:
            if start_m <= current <= end_m:
                return True
        else:
            # Overnight window (e.g. 22:00-02:00)
            if current >= start_m or current <= end_m:
                return True
    return False


def _is_news_blackout(now_utc: datetime) -> bool:
    cfg = CONFIG.live
    if not cfg.use_news_blackout:
        return False
    current = now_utc.hour * 60 + now_utc.minute
    for h in cfg.news_blackout_utc:
        hh, mm = _parse_hhmm(h)
        target = hh * 60 + mm
        if abs(current - target) <= cfg.news_blackout_minutes:
            return True
    return False


def _resolve_dynamic_thresholds(
    profile: TimeframeConfig | None,
    regime_class: str,
) -> dict[str, float]:
    if profile is None:
        base_thr = float(CONFIG.live.threshold)
        return {
            "signal_threshold": base_thr,
            "min_signal_margin": 0.0,
            "buy_signal_threshold": base_thr,
            "sell_signal_threshold": base_thr,
            "buy_min_signal_margin": 0.0,
            "sell_min_signal_margin": 0.0,
        }
    thr = float(profile.signal_threshold)
    margin = float(profile.min_signal_margin)
    buy_thr = float(profile.buy_signal_threshold if profile.buy_signal_threshold is not None else thr)
    sell_thr = float(profile.sell_signal_threshold if profile.sell_signal_threshold is not None else thr)
    buy_margin = float(profile.buy_min_signal_margin if profile.buy_min_signal_margin is not None else margin)
    sell_margin = float(profile.sell_min_signal_margin if profile.sell_min_signal_margin is not None else margin)
    reg = str(regime_class or "RANGING").upper()
    dyn = profile.regime_thresholds.get(reg, {}) if isinstance(profile.regime_thresholds, dict) else {}
    if isinstance(dyn, dict):
        if "signal_threshold" in dyn:
            thr = float(dyn["signal_threshold"])
        if "min_signal_margin" in dyn:
            margin = float(dyn["min_signal_margin"])
        if "buy_signal_threshold" in dyn:
            buy_thr = float(dyn["buy_signal_threshold"])
        if "sell_signal_threshold" in dyn:
            sell_thr = float(dyn["sell_signal_threshold"])
        if "buy_min_signal_margin" in dyn:
            buy_margin = float(dyn["buy_min_signal_margin"])
        if "sell_min_signal_margin" in dyn:
            sell_margin = float(dyn["sell_min_signal_margin"])
    buy_thr = max(buy_thr, thr)
    sell_thr = max(sell_thr, thr)
    buy_margin = max(buy_margin, margin)
    sell_margin = max(sell_margin, margin)
    return {
        "signal_threshold": thr,
        "min_signal_margin": margin,
        "buy_signal_threshold": buy_thr,
        "sell_signal_threshold": sell_thr,
        "buy_min_signal_margin": buy_margin,
        "sell_min_signal_margin": sell_margin,
    }


def _impulse_value(row: pd.Series, lookback_bars: int) -> float:
    if lookback_bars in {1, 3, 12}:
        return float(row.get(f"log_return_{lookback_bars}", 0.0) or 0.0)
    return float(row.get("log_return_3", 0.0) or 0.0)


def _gate_policy_live(
    *,
    entry_signal: str,
    gate_signal: str,
    gate_buy_prob: float,
    gate_sell_prob: float,
    confidence: float,
    profile: TimeframeConfig | None,
) -> tuple[str, str, str]:
    if entry_signal == "WAIT":
        return "WAIT", "entry_wait", ""
    if profile is None:
        if gate_signal != "WAIT" and gate_signal != entry_signal:
            return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? {gate_signal}"
        return entry_signal, "OK", ""

    mode = str(profile.gate_mode or "strict").lower().strip()
    gate_margin = abs(float(gate_buy_prob) - float(gate_sell_prob))
    if mode == "off":
        return entry_signal, "OK", ""
    if mode == "allow_wait":
        if gate_signal == "WAIT":
            can_bypass = bool(
                profile.allow_gate_wait_bypass and confidence >= float(profile.gate_wait_bypass_threshold)
            )
            if can_bypass or not profile.allow_gate_wait_bypass:
                return entry_signal, "OK", ""
            return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? WAIT"
        if gate_signal != entry_signal:
            return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? {gate_signal}"
        return entry_signal, "OK", ""
    if mode == "bias_only":
        if gate_signal == "WAIT":
            return entry_signal, "OK", ""
        if gate_signal != entry_signal and gate_margin >= float(profile.gate_min_margin_block):
            return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? {gate_signal}"
        return entry_signal, "OK", ""

    if gate_signal == "WAIT":
        can_bypass_wait_gate = bool(
            profile.allow_gate_wait_bypass and confidence >= float(profile.gate_wait_bypass_threshold)
        )
        if can_bypass_wait_gate:
            return entry_signal, "OK", ""
        return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? WAIT"
    if gate_signal != entry_signal:
        return "WAIT", "blocked_by_tf_conflict", f"entry=? {entry_signal} vs gate=? {gate_signal}"
    return entry_signal, "OK", ""


def _apply_time_stop(symbol: str, timeframe: str, max_holding_override: int | None = None) -> int:
    if mt5 is None:
        return 0
    max_candles = int(CONFIG.live.max_holding_candles if max_holding_override is None else max_holding_override)
    if max_candles <= 0:
        return 0
    tf_min = timeframe_minutes(timeframe)
    max_age_minutes = max_candles * tf_min
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    now_ts = datetime.now(timezone.utc).timestamp()
    closed = 0
    for p in positions:
        open_ts = float(getattr(p, "time", 0.0) or 0.0)
        if open_ts <= 0:
            continue
        age_minutes = (now_ts - open_ts) / 60.0
        if age_minutes < max_age_minutes:
            continue
        p_type = int(getattr(p, "type", -1))
        side = "BUY" if p_type == mt5.ORDER_TYPE_BUY else "SELL"
        ok, _ = close_position(
            symbol=symbol,
            ticket=int(getattr(p, "ticket")),
            volume=float(getattr(p, "volume")),
            side=side,
            deviation_points=CONFIG.live.order_deviation_points,
        )
        if ok:
            closed += 1
            _json_log(
                {
                    "event": "time_stop_close",
                    "symbol": symbol,
                    "tf": timeframe,
                    "ticket": int(getattr(p, "ticket")),
                    "age_minutes": age_minutes,
                    "max_holding_candles": max_candles,
                }
            )
            _notify_telegram(
                message=(
                    f"â±ï¸ Time Stop acionado\n"
                    f"Ativo: {symbol} | TF: {timeframe}\n"
                    f"Ticket: {int(getattr(p, 'ticket'))}\n"
                    f"Tempo aberto: {age_minutes:.1f} min"
                ),
                key=f"time_stop:{symbol}:{timeframe}",
                cooldown_seconds=30,
            )
    return closed


def _current_positions_map(symbol: str) -> dict[int, dict]:
    if mt5 is None:
        return {}
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return {}
    out: dict[int, dict] = {}
    for p in pos:
        ticket = int(getattr(p, "ticket"))
        p_type = int(getattr(p, "type", -1))
        side = "BUY" if p_type == mt5.ORDER_TYPE_BUY else "SELL"
        out[ticket] = {
            "ticket": ticket,
            "side": side,
            "volume": float(getattr(p, "volume", 0.0) or 0.0),
            "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
            "sl": float(getattr(p, "sl", 0.0) or 0.0),
            "tp": float(getattr(p, "tp", 0.0) or 0.0),
            "price_current": float(getattr(p, "price_current", 0.0) or 0.0),
            "time": float(getattr(p, "time", 0.0) or 0.0),
            "profit": float(getattr(p, "profit", 0.0) or 0.0),
        }
    return out


def _closed_position_pnl(symbol: str, ticket: int) -> float | None:
    if mt5 is None:
        return None
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        deals = mt5.history_deals_get(start, end, group=symbol)
        if deals is None or len(deals) == 0:
            return None
        pnl = 0.0
        for d in deals:
            position_id = getattr(d, "position_id", getattr(d, "position", None))
            if int(position_id or -1) != int(ticket):
                continue
            entry_flag = int(getattr(d, "entry", -1) or -1)
            if entry_flag not in {
                getattr(mt5, "DEAL_ENTRY_OUT", 1),
                getattr(mt5, "DEAL_ENTRY_OUT_BY", 3),
                getattr(mt5, "DEAL_ENTRY_INOUT", 2),
            }:
                continue
            pnl += float(getattr(d, "profit", 0.0) or 0.0)
            pnl += float(getattr(d, "commission", 0.0) or 0.0)
            pnl += float(getattr(d, "swap", 0.0) or 0.0)
            pnl += float(getattr(d, "fee", 0.0) or 0.0)
        if pnl == 0.0:
            return None
        return pnl
    except Exception:
        return None


def _candles_to_timedelta(candles: int, timeframe: str) -> timedelta:
    return timedelta(minutes=max(1, candles) * max(1, timeframe_minutes(timeframe)))


def _update_live_protection_after_close(
    guard: LiveProtectionState,
    *,
    pnl: float | None,
    side: str,
    now_utc: datetime,
    timeframe: str,
) -> None:
    if pnl is None:
        return
    guard.realized_pnl += float(pnl)
    guard.realized_peak_pnl = max(guard.realized_peak_pnl, guard.realized_pnl)

    if pnl < 0:
        guard.consecutive_losses += 1
        side_pause = _candles_to_timedelta(CONFIG.risk.same_side_loss_pause_candles, timeframe)
        guard.same_side_block_until[side] = now_utc + side_pause
        if guard.consecutive_losses >= int(CONFIG.risk.max_consecutive_losses):
            streak_pause = _candles_to_timedelta(CONFIG.risk.loss_streak_cooldown_candles, timeframe)
            guard.cooldown_until = now_utc + streak_pause
            guard.consecutive_losses = 0
    else:
        guard.consecutive_losses = 0

    peak = float(guard.realized_peak_pnl)
    giveback_frac = float(CONFIG.risk.profit_giveback_lock_fraction)
    if peak > 0.0 and guard.realized_pnl <= peak * (1.0 - giveback_frac):
        giveback_pause = _candles_to_timedelta(CONFIG.risk.profit_giveback_cooldown_candles, timeframe)
        if guard.cooldown_until is None or (now_utc + giveback_pause) > guard.cooldown_until:
            guard.cooldown_until = now_utc + giveback_pause


def _live_protection_block_reason(
    guard: LiveProtectionState,
    *,
    now_utc: datetime,
    decision: str,
) -> str:
    if guard.cooldown_until is not None and now_utc < guard.cooldown_until:
        peak = float(guard.realized_peak_pnl)
        if peak > 0.0 and guard.realized_pnl <= peak * (1.0 - float(CONFIG.risk.profit_giveback_lock_fraction)):
            return "PROFIT_GIVEBACK_LOCK"
        return "LOSS_STREAK_COOLDOWN"
    if decision in {"BUY", "SELL"}:
        side_until = guard.same_side_block_until.get(decision)
        if side_until is not None and now_utc < side_until:
            return "SAME_SIDE_LOSS_PAUSE"
    return ""


def _notify_closed_positions(
    symbol: str,
    timeframe: str,
    known_positions: dict[int, dict],
    guard: LiveProtectionState,
) -> tuple[dict[int, dict], list[dict]]:
    current = _current_positions_map(symbol)
    closed_tickets = [t for t in known_positions.keys() if t not in current]
    closed_events: list[dict] = []
    for ticket in closed_tickets:
        prev = known_positions[ticket]
        pnl = _closed_position_pnl(symbol, ticket)
        _GLOBAL_COORD.unregister_closed_ticket(symbol, ticket)
        _update_live_protection_after_close(
            guard,
            pnl=pnl,
            side=str(prev.get("side", "WAIT")),
            now_utc=datetime.now(timezone.utc),
            timeframe=timeframe,
        )
        closed_events.append({"ticket": int(ticket), "side": str(prev.get("side", "WAIT"))})
        _json_log(
            {
                "event": "position_closed_detected",
                "symbol": symbol,
                "ticket": ticket,
                "side": prev.get("side"),
                "volume": prev.get("volume"),
                "pnl": pnl,
                "realized_pnl_session": guard.realized_pnl,
                "realized_peak_pnl_session": guard.realized_peak_pnl,
                "consecutive_losses": guard.consecutive_losses,
                "cooldown_until": guard.cooldown_until.isoformat() if guard.cooldown_until else "",
            }
        )
        pnl_txt = f"{pnl:.2f}" if pnl is not None else "n/a"
        _notify_telegram(
            message=(
                f"âœ… PosiÃ§Ã£o fechada\n"
                f"Ativo: {symbol}\n"
                f"Ticket: {ticket}\n"
                f"Lado: {prev.get('side')} | Volume: {prev.get('volume')}\n"
                f"PnL: {pnl_txt}"
            ),
            key=f"close:{symbol}:{ticket}",
            cooldown_seconds=0,
        )
    return current, closed_events


def _sync_managed_positions(
    symbol: str,
    timeframe: str,
    current_positions: dict[int, dict],
    managed: dict[int, ManagedPositionState],
) -> dict[int, ManagedPositionState]:
    synced: dict[int, ManagedPositionState] = {}
    for ticket, pos in current_positions.items():
        state = managed.get(ticket)
        if state is None:
            state = ManagedPositionState(
                ticket=int(ticket),
                side=str(pos.get("side", "WAIT")),
                entry_price=float(pos.get("price_open", 0.0) or 0.0),
                initial_sl=float(pos.get("sl", 0.0) or 0.0),
                initial_tp=float(pos.get("tp", 0.0) or 0.0),
            )
        synced[int(ticket)] = state
    _save_position_state(symbol, timeframe, synced)
    return synced


def _manage_open_positions(
    *,
    symbol: str,
    timeframe: str,
    feats_entry: pd.DataFrame,
    regime_class: str,
    managed: dict[int, ManagedPositionState],
) -> dict[int, ManagedPositionState]:
    if mt5 is None:
        return managed
    current = _current_positions_map(symbol)
    managed = _sync_managed_positions(symbol, timeframe, current, managed)
    if not current:
        return managed

    exit_cfg = resolve_exit_settings(symbol=symbol, timeframe=timeframe)
    atr = float(feats_entry.iloc[-1].get("ATR_14", 0.0) or 0.0)
    high_vol_regimes = {"HIGH_VOL", "HIGH_VOL_BREAKOUT", "POST_NEWS_SHOCK"}
    break_even_enabled = bool(exit_cfg.get("break_even_enabled", False))
    break_even_r = float(exit_cfg.get("break_even_r", 0.5) or 0.5)
    break_even_offset_r = float(exit_cfg.get("break_even_offset_r", 0.0) or 0.0)
    trailing_enabled = bool(exit_cfg.get("trailing_enabled", False))
    trailing_activation_r = float(exit_cfg.get("trailing_activation_r", 1.0) or 1.0)
    trailing_atr_mult = float(exit_cfg.get("trailing_atr_mult", 1.0) or 1.0)
    changed = False

    for ticket, pos in current.items():
        st = managed.get(ticket)
        if st is None:
            continue
        initial_r = abs(float(st.entry_price) - float(st.initial_sl))
        if initial_r <= 0:
            continue
        current_price = float(pos.get("price_current", 0.0) or 0.0)
        current_sl = float(pos.get("sl", 0.0) or 0.0)
        current_tp = float(pos.get("tp", 0.0) or 0.0)
        if current_price <= 0:
            continue

        pnl_r = (
            (current_price - st.entry_price) / initial_r
            if st.side == "BUY"
            else (st.entry_price - current_price) / initial_r
        )
        desired_sl = current_sl
        action_reason = ""

        if break_even_enabled and not st.break_even_done and pnl_r >= break_even_r:
            offset = initial_r * break_even_offset_r
            desired_sl = st.entry_price + offset if st.side == "BUY" else st.entry_price - offset
            st.break_even_done = True
            action_reason = "break_even"

        if trailing_enabled and regime_class in high_vol_regimes and pnl_r >= trailing_activation_r and atr > 0:
            trail_sl = current_price - (atr * trailing_atr_mult) if st.side == "BUY" else current_price + (atr * trailing_atr_mult)
            if st.side == "BUY":
                desired_sl = max(desired_sl, trail_sl)
            else:
                desired_sl = min(desired_sl if desired_sl > 0 else trail_sl, trail_sl)
            st.trailing_active = True
            action_reason = action_reason or "atr_trailing"

        should_modify = False
        if st.side == "BUY" and desired_sl > current_sl + 1e-8:
            should_modify = True
        if st.side == "SELL" and (current_sl <= 0 or desired_sl < current_sl - 1e-8):
            should_modify = True

        if should_modify:
            ok, _ = modify_position_sltp(symbol=symbol, ticket=ticket, sl=desired_sl, tp=current_tp)
            if ok:
                changed = True
                _json_log(
                    {
                        "event": "position_exit_adjust",
                        "symbol": symbol,
                        "tf": timeframe,
                        "ticket": ticket,
                        "reason": action_reason,
                        "new_sl": desired_sl,
                        "current_tp": current_tp,
                        "pnl_r": pnl_r,
                        "regime_class": regime_class,
                    }
                )
    if changed:
        _save_position_state(symbol, timeframe, managed)
    return managed


def _decision_from_proba(
    proba: np.ndarray,
    classes: list[int],
    *,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
) -> tuple[str, float, float]:
    p_sell = float(proba[classes.index(0)]) if 0 in classes else 0.0
    p_buy = float(proba[classes.index(2)]) if 2 in classes else 0.0
    buy_thr = float(CONFIG.live.threshold if buy_threshold is None else buy_threshold)
    sell_thr = float(CONFIG.live.threshold if sell_threshold is None else sell_threshold)
    decision = "WAIT"
    if p_buy >= buy_thr and p_buy > p_sell:
        decision = "BUY"
    elif p_sell >= sell_thr and p_sell > p_buy:
        decision = "SELL"
    return decision, p_buy, p_sell


def _decision_to_int(decision: str) -> int:
    if decision == "BUY":
        return BUY
    if decision == "SELL":
        return SELL
    return WAIT


def _int_to_decision(signal: int) -> str:
    if int(signal) == BUY:
        return "BUY"
    if int(signal) == SELL:
        return "SELL"
    return "WAIT"


def _resolve_tf_profile(symbol: str, timeframe: str) -> TimeframeConfig | None:
    return resolve_timeframe_profile(symbol=symbol, timeframe=timeframe)


def _window_trades_count(
    now_utc: datetime,
    profile: TimeframeConfig,
    open_times: deque[datetime],
) -> int:
    if profile.trades_window_mode == "fixed_hour":
        hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
        return sum(1 for t in open_times if t >= hour_start)
    cutoff = now_utc - timedelta(minutes=60)
    return sum(1 for t in open_times if t >= cutoff)


def _atr_threshold_values(feats: pd.DataFrame, profile: TimeframeConfig) -> tuple[float, float, float, float]:
    atr_series = feats["ATR_14"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    atr_value = float(feats.iloc[-1]["ATR_14"] or 0.0)
    if len(atr_series) == 0:
        return atr_value, 0.0, 1e9, 0.0
    if profile.volatility_threshold_min_mode == "atr_percentile":
        pmin = float(np.percentile(atr_series.values, profile.volatility_p_min))
    else:
        pmin = float(profile.volatility_abs_min)
    if profile.volatility_threshold_max_mode == "atr_percentile":
        pmax = float(np.percentile(atr_series.values, profile.volatility_p_max))
    else:
        pmax = float(profile.volatility_abs_max)
    rank = float((atr_series <= atr_value).mean() * 100.0)
    return atr_value, pmin, pmax, rank


def _frequency_blocked(
    now_utc: datetime,
    profile: TimeframeConfig,
    open_times: deque[datetime],
) -> bool:
    if profile.max_trades_per_hour <= 0:
        return False
    if profile.trades_window_mode == "fixed_hour":
        hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
        n = sum(1 for t in open_times if t >= hour_start)
        return n >= profile.max_trades_per_hour
    cutoff = now_utc - timedelta(minutes=60)
    n = sum(1 for t in open_times if t >= cutoff)
    return n >= profile.max_trades_per_hour


def _candles_since(last_time: datetime | None, now_utc: datetime, timeframe: str) -> int:
    if last_time is None:
        return 10**9
    tf_min = max(1, timeframe_minutes(timeframe))
    diff_min = (now_utc - last_time).total_seconds() / 60.0
    return int(diff_min // tf_min)


def _resolve_model_feature_cols(model, feats: pd.DataFrame) -> list[str]:
    model_cols = list(getattr(model, "feature_name_", []))
    if model_cols:
        missing = [c for c in model_cols if c not in feats.columns]
        for col in missing:
            # Keep live inference shape-compatible with training schema.
            feats[col] = 0.0
        return model_cols
    return [c for c in feats.columns if c not in DROP_COLS]


def _load_model_from_registry(
    model_symbol: str,
    model_tf: str,
    model_version: str | None = None,
    use_latest_model: bool = True,
) -> tuple[Any, dict[str, Any]]:
    entry = (
        get_latest_model(symbol=model_symbol, timeframe=model_tf)
        if use_latest_model or not model_version
        else get_model(symbol=model_symbol, timeframe=model_tf, version=model_version)
    )
    model = joblib.load(entry.model_path)
    schema_features: list[str] = []
    if entry.features_schema_path and entry.features_schema_path.exists():
        try:
            schema = json.loads(entry.features_schema_path.read_text(encoding="utf-8"))
            schema_features = list(schema.get("features", []))
        except Exception:
            schema_features = []
    meta = {
        "model_path": str(entry.model_path),
        "trained_at": entry.trained_at,
        "model_symbol": model_symbol,
        "model_tf": model_tf,
        "model_version": entry.version,
        "schema_features": schema_features,
    }
    return model, meta


def run_live(
    symbol: str,
    timeframe: str,
    once: bool = False,
    no_trade: bool = False,
    diagnostic_only: bool = False,
    diagnostic_out_dir: Path | None = None,
    diagnostic_out_name: str = "diagnostic_summary",
    model_symbol: str | None = None,
    model_tf: str | None = None,
    model_version: str | None = None,
    use_latest_model: bool = True,
) -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    CONFIG.ensure_dirs()
    if not ensure_logged_in():
        raise RuntimeError("MT5 login failed")

    profile = _resolve_tf_profile(symbol, timeframe)
    tf_entry = timeframe
    tf_gate = profile.tf_gate if profile else timeframe
    use_session_filter, allowed_sessions = resolve_session_settings(symbol=symbol, timeframe=tf_entry)
    exit_cfg = resolve_exit_settings(symbol=symbol, timeframe=tf_entry)
    model_symbol = model_symbol or symbol
    model_tf = model_tf or tf_entry
    model_entry, model_entry_meta = _load_model_from_registry(
        model_symbol=model_symbol,
        model_tf=model_tf,
        model_version=model_version,
        use_latest_model=use_latest_model,
    )
    if tf_gate != tf_entry:
        model_gate, model_gate_meta = _load_model_from_registry(
            model_symbol=model_symbol,
            model_tf=tf_gate,
            model_version=None,
            use_latest_model=True,
        )
    else:
        model_gate = None
        model_gate_meta = {}

    _notify_telegram(
        message=(
            f"Bot iniciado\n"
            f"Ativo: {symbol} | TF entrada: {tf_entry} | TF gate: {tf_gate}\n"
            f"Modelo: {model_entry_meta.get('model_symbol')}/{model_entry_meta.get('model_tf')} v={model_entry_meta.get('model_version')}\n"
            f"Treinado em: {model_entry_meta.get('trained_at')}\n"
            f"Modo sem ordem: {'SIM' if (no_trade or diagnostic_only) else 'NAO'}"
        ),
        key=f"boot:{symbol}:{tf_entry}:{tf_gate}",
        cooldown_seconds=120,
    )
    _json_log(
        {
            "event": "bot_boot",
            "symbol": symbol,
            "tf": tf_entry,
            "tf_entry": tf_entry,
            "tf_gate": tf_gate,
            "model_path": model_entry_meta.get("model_path"),
            "trained_at": model_entry_meta.get("trained_at"),
            "model_symbol": model_entry_meta.get("model_symbol"),
            "model_tf": model_entry_meta.get("model_tf"),
            "model_version": model_entry_meta.get("model_version"),
            "gate_model_path": model_gate_meta.get("model_path") if model_gate_meta else None,
            "diagnostic_only": diagnostic_only,
            "no_trade": no_trade,
        }
    )
    failures = 0
    today_stats = TodayStats(trades_count=0, last_trade_time=None)
    live_guard = LiveProtectionState()
    known_positions = _current_positions_map(symbol)
    managed_positions = _load_position_state(symbol, tf_entry)
    health_buf = deque(maxlen=max(10, CONFIG.live.health_check_every_n_events))
    event_count = 0
    open_times: deque[datetime] = deque(maxlen=3000)
    last_open_same_dir: dict[str, datetime | None] = {"BUY": None, "SELL": None}
    last_close_same_dir: dict[str, datetime | None] = {"BUY": None, "SELL": None}
    diagnostic_out_dir = diagnostic_out_dir or CONFIG.reports_dir
    diagnostic_out_dir.mkdir(parents=True, exist_ok=True)
    last_diag_dump = datetime.now(timezone.utc)
    diag_counter = {
        "total_cycles": 0,
        "ok_count": 0,
        "blocked_by_tf_conflict_count": 0,
        "blocked_by_missing_gate_pred_count": 0,
        "blocked_by_missing_entry_pred_count": 0,
        "blocked_by_volatility_low_count": 0,
        "blocked_by_volatility_high_count": 0,
        "blocked_by_impulse_alignment_count": 0,
        "blocked_by_frequency_count": 0,
        "blocked_by_global_risk_count": 0,
        "blocked_by_spread_count": 0,
        "blocked_by_session_count": 0,
        "blocked_by_news_blackout_count": 0,
        "blocked_by_fundamental_count": 0,
        "blocked_by_loss_streak_count": 0,
        "blocked_by_profit_giveback_count": 0,
        "blocked_by_same_side_loss_count": 0,
        "blocked_by_symbol_positions_count": 0,
    }

    try:
        while True:
            known_positions, closed_events = _notify_closed_positions(symbol, tf_entry, known_positions, live_guard)
            now_utc = datetime.now(timezone.utc)
            for ev in closed_events:
                side = str(ev.get("side", "WAIT"))
                if side in last_close_same_dir:
                    last_close_same_dir[side] = now_utc

            _apply_time_stop(
                symbol=symbol,
                timeframe=tf_entry,
                max_holding_override=int(exit_cfg.get("max_holding_candles_override", 0) or 0),
            )
            raw_entry = _fetch_recent(symbol=symbol, timeframe=tf_entry, bars=CONFIG.live.fetch_bars)
            if raw_entry.empty:
                failures += 1
                _json_log({"event": "fetch_fail", "failures": failures, "symbol": symbol, "tf": tf_entry})
                if failures >= CONFIG.live.max_mt5_failures:
                    _json_log({"event": "kill_switch", "reason": "MAX_MT5_FAILURES"})
                    break
                time.sleep(3)
                continue
            failures = 0

            dxy_strength = 0.0
            if CONFIG.live.use_dxy_proxy:
                dxy_strength = fetch_dxy_strength(
                    timeframe_mt5=TF_TO_MT5.get(tf_entry),
                    bars=max(80, int(CONFIG.live.fetch_bars // 2)),
                    symbols=CONFIG.live.dxy_symbols,
                )
            feats_entry = build_features(raw_entry, dxy_strength=dxy_strength)
            if feats_entry.empty:
                time.sleep(3)
                continue

            regime_class = str(feats_entry.iloc[-1].get("regime_class", "RANGING"))
            managed_positions = _manage_open_positions(
                symbol=symbol,
                timeframe=tf_entry,
                feats_entry=feats_entry,
                regime_class=regime_class,
                managed=managed_positions,
            )
            entry_thresholds = _resolve_dynamic_thresholds(profile, regime_class=regime_class)
            buy_threshold_used = float(entry_thresholds["buy_signal_threshold"])
            sell_threshold_used = float(entry_thresholds["sell_signal_threshold"])
            buy_min_signal_margin_used = float(entry_thresholds["buy_min_signal_margin"])
            sell_min_signal_margin_used = float(entry_thresholds["sell_min_signal_margin"])

            entry_cols = _resolve_model_feature_cols(model_entry, feats_entry)
            schema_feats = model_entry_meta.get("schema_features", [])
            if schema_feats:
                missing_schema = [c for c in schema_feats if c not in feats_entry.columns]
                if missing_schema:
                    _json_log(
                        {
                            "event": "kill_switch",
                            "reason": "MISSING_FEATURES_SCHEMA",
                            "symbol": symbol,
                            "tf": tf_entry,
                            "missing_features_count": len(missing_schema),
                            "missing_features_sample": missing_schema[:10],
                            "model_path": model_entry_meta.get("model_path"),
                        }
                    )
                    break
            row_entry = feats_entry.iloc[[-1]][entry_cols]
            entry_proba = model_entry.predict_proba(row_entry)[0]
            entry_classes = model_entry.classes_.tolist()
            has_entry_pred = True
            reason_missing_entry_pred: str | None = None
            entry_signal, p_buy, p_sell = _decision_from_proba(
                entry_proba,
                entry_classes,
                buy_threshold=buy_threshold_used,
                sell_threshold=sell_threshold_used,
            )
            prob_diff = abs(p_buy - p_sell)
            confidence = max(p_buy, p_sell)
            threshold_used = (
                buy_threshold_used if entry_signal == "BUY" else sell_threshold_used if entry_signal == "SELL" else float(entry_thresholds["signal_threshold"])
            )
            min_signal_margin_used = (
                buy_min_signal_margin_used if entry_signal == "BUY" else sell_min_signal_margin_used if entry_signal == "SELL" else float(entry_thresholds["min_signal_margin"])
            )

            gate_signal = entry_signal
            gate_probs = {"buy": p_buy, "sell": p_sell}
            has_gate_pred = True
            reason_missing_gate_pred: str | None = None
            if model_gate is not None:
                raw_gate = _fetch_recent(symbol=symbol, timeframe=tf_gate, bars=CONFIG.live.fetch_bars)
                gate_dxy_strength = dxy_strength if tf_gate == tf_entry else 0.0
                if CONFIG.live.use_dxy_proxy and tf_gate != tf_entry:
                    gate_dxy_strength = fetch_dxy_strength(
                        timeframe_mt5=TF_TO_MT5.get(tf_gate),
                        bars=max(80, int(CONFIG.live.fetch_bars // 2)),
                        symbols=CONFIG.live.dxy_symbols,
                    )
                feats_gate = build_features(raw_gate, dxy_strength=gate_dxy_strength) if not raw_gate.empty else pd.DataFrame()
                if not feats_gate.empty:
                    gate_regime_class = str(feats_gate.iloc[-1].get("regime_class", "RANGING"))
                    gate_profile = CONFIG.timeframe_profiles.get(tf_gate) or profile
                    gate_thresholds = _resolve_dynamic_thresholds(gate_profile, regime_class=gate_regime_class)
                    gate_cols = _resolve_model_feature_cols(model_gate, feats_gate)
                    gate_idx = -2 if len(feats_gate) >= 2 else -1
                    row_gate = feats_gate.iloc[[gate_idx]][gate_cols]
                    gate_proba = model_gate.predict_proba(row_gate)[0]
                    gate_classes = model_gate.classes_.tolist()
                    gate_signal, g_buy, g_sell = _decision_from_proba(
                        gate_proba,
                        gate_classes,
                        buy_threshold=float(gate_thresholds["buy_signal_threshold"]),
                        sell_threshold=float(gate_thresholds["sell_signal_threshold"]),
                    )
                    gate_probs = {"buy": g_buy, "sell": g_sell}
                else:
                    gate_signal = "WAIT"
                    gate_probs = {"buy": 0.0, "sell": 0.0}
                    has_gate_pred = False
                    reason_missing_gate_pred = "missing_gate_features"
            gate_prob_diff = abs(float(gate_probs["buy"]) - float(gate_probs["sell"]))

            spread_points = float(feats_entry.iloc[-1].get("spread", 0.0) or 0.0)
            high_impact_next_60 = bool(int(feats_entry.iloc[-1].get("high_impact_in_next_60min", 0) or 0))
            hours_since_high_impact = float(feats_entry.iloc[-1].get("hours_since_last_high_impact", 1e9) or 1e9)
            atr_val, atr_p_min_val, atr_p_max_val, atr_rank = (
                _atr_threshold_values(feats_entry, profile)
                if profile
                else (float(feats_entry.iloc[-1].get("ATR_14", 0.0) or 0.0), 0.0, 1e9, 0.0)
            )
            fund_status = get_fundamental_window_status(
                now_utc,
                pre_minutes=CONFIG.live.news_blackout_minutes_pre,
                post_minutes=CONFIG.live.news_blackout_minutes_post,
                next_window_minutes=CONFIG.fundamentals.next_event_window_minutes,
            )

            final_decision = entry_signal
            block_reason = ""
            conflict_detail = ""
            trades_in_window = _window_trades_count(now_utc, profile, open_times) if profile else 0

            state = _account_state(symbol)
            if state.balance > 0:
                loss_pct = max(0.0, (-state.daily_pnl / state.balance) * 100.0)
                if loss_pct >= CONFIG.risk.max_daily_loss_pct:
                    _json_log({"event": "kill_switch", "reason": "MAX_DAILY_LOSS"})
                    break
            if state.margin_level_pct < CONFIG.risk.min_margin_level_pct:
                _json_log({"event": "kill_switch", "reason": "LOW_MARGIN_LEVEL"})
                break

            if final_decision != "WAIT":
                if model_gate is not None and not has_gate_pred:
                    final_decision = "WAIT"
                    block_reason = "blocked_by_missing_gate_pred"
                    conflict_detail = f"entry={tf_entry} {entry_signal} vs gate={tf_gate} WAIT"
                if not block_reason and model_gate is not None:
                    gate_decision, gate_reason, gate_conflict = _gate_policy_live(
                        entry_signal=entry_signal,
                        gate_signal=gate_signal,
                        gate_buy_prob=float(gate_probs["buy"]),
                        gate_sell_prob=float(gate_probs["sell"]),
                        confidence=confidence,
                        profile=profile,
                    )
                    final_decision = gate_decision
                    if gate_reason not in {"", "OK"}:
                        block_reason = gate_reason
                    if gate_conflict:
                        conflict_detail = gate_conflict.replace("entry=?", f"entry={tf_entry}").replace(
                            "gate=?", f"gate={tf_gate}"
                        )
                if not block_reason and use_session_filter and not _is_in_session(now_utc, allowed_sessions):
                    final_decision = "WAIT"
                    block_reason = "SESSION_FILTER"
                if not block_reason and _is_news_blackout(now_utc):
                    final_decision = "WAIT"
                    block_reason = "NEWS_BLACKOUT"
                if not block_reason and CONFIG.live.use_fundamental_calendar and fund_status.blocked:
                    final_decision = "WAIT"
                    block_reason = "FUNDAMENTAL_EVENT"
                if not block_reason and spread_points > CONFIG.live.max_spread_points:
                    final_decision = "WAIT"
                    block_reason = "SPREAD_FILTER"
                if not block_reason and prob_diff < float(min_signal_margin_used):
                    final_decision = "WAIT"
                    block_reason = "blocked_by_low_signal_margin"

                if not block_reason and profile and profile.impulse_alignment_required:
                    impulse_val = _impulse_value(feats_entry.iloc[-1], int(profile.impulse_lookback_bars))
                    impulse_min = float(profile.impulse_min_abs_return)
                    if final_decision == "BUY" and impulse_val < impulse_min:
                        final_decision = "WAIT"
                        block_reason = "blocked_by_impulse_alignment"
                    elif final_decision == "SELL" and impulse_val > -impulse_min:
                        final_decision = "WAIT"
                        block_reason = "blocked_by_impulse_alignment"

                if not block_reason and profile:
                    if atr_val < atr_p_min_val:
                        final_decision = "WAIT"
                        block_reason = "blocked_by_volatility_low"
                    elif atr_val > atr_p_max_val:
                        final_decision = "WAIT"
                        block_reason = "blocked_by_volatility_high"

                if not block_reason and profile and _frequency_blocked(now_utc, profile, open_times):
                    final_decision = "WAIT"
                    block_reason = "blocked_by_frequency"

                if not block_reason and profile:
                    candles_since_open = _candles_since(last_open_same_dir.get(entry_signal), now_utc, tf_entry)
                    if candles_since_open < int(profile.min_candles_between_same_direction_trades):
                        final_decision = "WAIT"
                        block_reason = "blocked_by_reentry_rule"
                    candles_since_close = _candles_since(last_close_same_dir.get(entry_signal), now_utc, tf_entry)
                    if not block_reason and candles_since_close < int(profile.reentry_block_candles):
                        final_decision = "WAIT"
                        block_reason = "blocked_by_reentry_rule"

                if not block_reason and _positions_count(symbol) >= int(CONFIG.risk.max_symbol_positions_live):
                    final_decision = "WAIT"
                    block_reason = "MAX_SYMBOL_POSITIONS_LIVE"

                if not block_reason:
                    guard_reason = _live_protection_block_reason(
                        live_guard,
                        now_utc=now_utc,
                        decision=entry_signal,
                    )
                    if guard_reason:
                        final_decision = "WAIT"
                        block_reason = guard_reason

                if not block_reason:
                    can_open, reason = can_open_trade(state, _positions_count(symbol), today_stats)
                    if not can_open:
                        final_decision = "WAIT"
                        block_reason = reason

            price = float(feats_entry.iloc[-1]["close"])
            atr = float(feats_entry.iloc[-1].get("ATR_14", 0.0) or 0.0)
            sl_dist = CONFIG.triple_barrier.sl_atr_mult * atr
            tp_dist = CONFIG.triple_barrier.pt_atr_mult * atr
            sl = price - sl_dist if final_decision == "BUY" else price + sl_dist
            tp = price + tp_dist if final_decision == "BUY" else price - tp_dist
            risk_pct_used = float(profile.risk_pct if profile else CONFIG.risk.default_risk_pct)
            stop_distance_points = float(abs(price - sl) * 1e5) if final_decision != "WAIT" else 0.0

            lot = 0.0
            regime_mult = 1.0
            resp = None
            if final_decision != "WAIT":
                lot = _compute_live_lot(
                    symbol=symbol,
                    side=final_decision,
                    balance=state.balance,
                    entry_price=price,
                    sl_price=sl,
                    risk_pct_override=risk_pct_used,
                )
                regime_mult_map = {
                    "TREND": CONFIG.risk.regime_mult_trend,
                    "TRENDING_STRONG": CONFIG.risk.regime_mult_trend,
                    "HIGH_VOL": CONFIG.risk.regime_mult_high_vol,
                    "HIGH_VOL_BREAKOUT": CONFIG.risk.regime_mult_high_vol,
                    "POST_NEWS_SHOCK": CONFIG.risk.regime_mult_high_vol,
                    "SIDEWAYS": CONFIG.risk.regime_mult_sideways,
                    "LOW_VOL_SIDEWAYS": CONFIG.risk.regime_mult_sideways,
                    "RANGING": CONFIG.risk.regime_mult_neutral,
                    "NEUTRAL": CONFIG.risk.regime_mult_neutral,
                }
                regime_mult = float(regime_mult_map.get(regime_class, CONFIG.risk.regime_mult_neutral))
                lot = _normalize_lot(symbol, lot * regime_mult)
                g_ok, g_reason, _ = _GLOBAL_COORD.can_open(symbol, proposed_risk_pct=risk_pct_used, proposed_lot=lot)
                if not g_ok:
                    final_decision = "WAIT"
                    block_reason = g_reason
                else:
                    block_reason = block_reason or "OK"
            global_snap = _GLOBAL_COORD.snapshot(symbol)

            order_ticket = 0
            if final_decision != "WAIT":
                ok = False
                if not no_trade and not diagnostic_only:
                    ok, resp = send_order(
                        symbol=symbol,
                        side=final_decision,
                        lot=lot,
                        sl=sl,
                        tp=tp,
                        deviation_points=CONFIG.live.order_deviation_points,
                    )
                if ok:
                    order_ticket = int(getattr(resp, "order", 0) or 0)
                    today_stats.trades_count += 1
                    today_stats.last_trade_time = now_utc
                    open_times.append(now_utc)
                    last_open_same_dir[final_decision] = now_utc
                    if order_ticket > 0:
                        _GLOBAL_COORD.register_open_order(symbol, order_ticket, risk_pct_used)
                        managed_positions[order_ticket] = ManagedPositionState(
                            ticket=order_ticket,
                            side=final_decision,
                            entry_price=price,
                            initial_sl=sl,
                            initial_tp=tp,
                        )
                        _save_position_state(symbol, tf_entry, managed_positions)
                elif no_trade or diagnostic_only:
                    block_reason = "OK"
                else:
                    block_reason = "ORDER_FAIL"

            event_name = "decision" if final_decision != "WAIT" or entry_signal == "WAIT" else "decision_blocked"
            strong_signal = max(p_buy, p_sell) >= (threshold_used + float(CONFIG.live.strong_signal_alert_delta))
            if final_decision == "WAIT" and block_reason and strong_signal:
                _json_log(
                    {
                        "event": "strong_signal_blocked",
                        "symbol": symbol,
                        "tf": tf_entry,
                        "entry_signal": entry_signal,
                        "gate_signal": gate_signal,
                        "reason": block_reason,
                        "p_buy": p_buy,
                        "p_sell": p_sell,
                        "threshold_used": threshold_used,
                    }
                )

            payload = {
                "event": event_name,
                "symbol": symbol,
                "tf": tf_entry,
                "tf_entry": tf_entry,
                "tf_gate": tf_gate,
                "gate_mode": str(profile.gate_mode if profile else "strict"),
                "gate_signal": gate_signal,
                "entry_signal": entry_signal,
                "has_entry_pred": has_entry_pred,
                "has_gate_pred": has_gate_pred,
                "reason_missing_entry_pred": reason_missing_entry_pred,
                "reason_missing_gate_pred": reason_missing_gate_pred,
                "decision": final_decision,
                "final_decision": final_decision,
                "block_reason": block_reason,
                "reason": block_reason if final_decision == "WAIT" and block_reason else "",
                "conflict_entry_tf": tf_entry if block_reason == "blocked_by_tf_conflict" else "",
                "conflict_gate_tf": tf_gate if block_reason == "blocked_by_tf_conflict" else "",
                "conflict_detail": conflict_detail,
                "probs": {"buy": p_buy, "sell": p_sell},
                "gate_probs": gate_probs,
                "entry_buy_prob": p_buy,
                "entry_sell_prob": p_sell,
                "gate_buy_prob": float(gate_probs["buy"]),
                "gate_sell_prob": float(gate_probs["sell"]),
                "entry_prob_diff": prob_diff,
                "gate_prob_diff": gate_prob_diff,
                "prob_diff": prob_diff,
                "threshold_used": threshold_used,
                "min_signal_margin_used": min_signal_margin_used,
                "buy_threshold_used": buy_threshold_used,
                "sell_threshold_used": sell_threshold_used,
                "buy_min_signal_margin_used": buy_min_signal_margin_used,
                "sell_min_signal_margin_used": sell_min_signal_margin_used,
                "impulse_alignment_required": bool(profile.impulse_alignment_required) if profile else False,
                "impulse_lookback_bars": int(profile.impulse_lookback_bars) if profile else 0,
                "impulse_min_abs_return": float(profile.impulse_min_abs_return) if profile else 0.0,
                "impulse_value": float(_impulse_value(feats_entry.iloc[-1], int(profile.impulse_lookback_bars))) if profile else 0.0,
                "price": price,
                "spread_points": spread_points,
                "sl": sl if final_decision != "WAIT" else None,
                "tp": tp if final_decision != "WAIT" else None,
                "order_id": order_ticket,
                "order_ticket": order_ticket,
                "no_trade": no_trade,
                "can_open_reason": block_reason,
                "atr_14_value": atr_val,
                "atr_p_min_value": atr_p_min_val,
                "atr_p_max_value": atr_p_max_val,
                "atr_percentile_current": atr_rank,
                "dxy_strength": float(dxy_strength),
                "high_impact_in_next_60min": high_impact_next_60,
                "hours_since_last_high_impact": hours_since_high_impact,
                "fundamental_blackout_active": bool(fund_status.blocked),
                "next_high_impact_event_at": fund_status.next_event_at,
                "next_high_impact_event_name": fund_status.next_event_name,
                "risk_pct_used": risk_pct_used if final_decision != "WAIT" else 0.0,
                "stop_distance_points": stop_distance_points,
                "position_size_lots": lot,
                "min_lot": CONFIG.risk.min_lot,
                "lot_step": float(getattr(mt5.symbol_info(symbol), "volume_step", 0.01) or 0.01) if mt5 else 0.01,
                "global_risk_current_pct": global_snap.risk_current_pct,
                "global_positions_count": global_snap.positions_count,
                "global_daily_loss_pct": global_snap.daily_loss_pct,
                "new_orders_last_60s": global_snap.new_orders_last_window,
                "cooldown_active": bool(
                    today_stats.last_trade_time is not None
                    and (now_utc - today_stats.last_trade_time).total_seconds() < (CONFIG.risk.cooldown_minutes * 60)
                ),
                "loss_streak_cooldown_active": bool(live_guard.cooldown_until is not None and now_utc < live_guard.cooldown_until),
                "consecutive_losses": int(live_guard.consecutive_losses),
                "realized_pnl_session": float(live_guard.realized_pnl),
                "realized_peak_pnl_session": float(live_guard.realized_peak_pnl),
                "same_side_block_until_buy": live_guard.same_side_block_until["BUY"].isoformat() if live_guard.same_side_block_until["BUY"] else "",
                "same_side_block_until_sell": live_guard.same_side_block_until["SELL"].isoformat() if live_guard.same_side_block_until["SELL"] else "",
                "max_trades_window_count": int(trades_in_window),
                "volatility_state": (
                    "LOW" if atr_val < atr_p_min_val else ("HIGH" if atr_val > atr_p_max_val else "OK")
                ),
                "use_session_filter": bool(use_session_filter),
                "allowed_sessions_utc": list(allowed_sessions),
                "gate_policy": str(profile.gate_mode if profile else "strict"),
                "break_even_enabled": bool(exit_cfg.get("break_even_enabled", False)),
                "break_even_r": float(exit_cfg.get("break_even_r", 0.5) or 0.5),
                "trailing_enabled": bool(exit_cfg.get("trailing_enabled", False)),
                "trailing_activation_r": float(exit_cfg.get("trailing_activation_r", 1.0) or 1.0),
                "impulse_alignment_required": bool(profile.impulse_alignment_required) if profile else False,
                "impulse_lookback_bars": int(profile.impulse_lookback_bars) if profile else 0,
                "impulse_min_abs_return": float(profile.impulse_min_abs_return) if profile else 0.0,
                "impulse_value": float(_impulse_value(feats_entry.iloc[-1], int(profile.impulse_lookback_bars))) if profile else 0.0,
                "decision_panel": {
                    "threshold": threshold_used,
                    "p_buy": p_buy,
                    "p_sell": p_sell,
                    "confidence": confidence,
                    "regime_class": regime_class,
                    "spread_points": spread_points,
                    "regime_mult": regime_mult,
                },
                "model_info": {
                    "model_path": model_entry_meta.get("model_path"),
                    "trained_at": model_entry_meta.get("trained_at"),
                    "model_symbol": model_entry_meta.get("model_symbol"),
                    "model_tf": model_entry_meta.get("model_tf"),
                    "model_version": model_entry_meta.get("model_version"),
                    "gate_model_path": model_gate_meta.get("model_path"),
                },
            }
            _json_log(payload)
            _update_daily_report(payload)

            diag_counter["total_cycles"] += 1
            if final_decision != "WAIT" and block_reason == "OK":
                diag_counter["ok_count"] += 1
            if block_reason == "blocked_by_tf_conflict":
                diag_counter["blocked_by_tf_conflict_count"] += 1
            if block_reason == "blocked_by_missing_gate_pred":
                diag_counter["blocked_by_missing_gate_pred_count"] += 1
            if block_reason == "blocked_by_missing_entry_pred":
                diag_counter["blocked_by_missing_entry_pred_count"] += 1
            if block_reason == "blocked_by_volatility_low":
                diag_counter["blocked_by_volatility_low_count"] += 1
            if block_reason == "blocked_by_volatility_high":
                diag_counter["blocked_by_volatility_high_count"] += 1
            if block_reason == "blocked_by_impulse_alignment":
                diag_counter["blocked_by_impulse_alignment_count"] += 1
            if block_reason == "blocked_by_frequency":
                diag_counter["blocked_by_frequency_count"] += 1
            if block_reason.startswith("blocked_by_global_"):
                diag_counter["blocked_by_global_risk_count"] += 1
            if block_reason == "SPREAD_FILTER":
                diag_counter["blocked_by_spread_count"] += 1
            if block_reason == "SESSION_FILTER":
                diag_counter["blocked_by_session_count"] += 1
            if block_reason == "NEWS_BLACKOUT":
                diag_counter["blocked_by_news_blackout_count"] += 1
            if block_reason == "FUNDAMENTAL_EVENT":
                diag_counter["blocked_by_fundamental_count"] += 1
            if block_reason == "LOSS_STREAK_COOLDOWN":
                diag_counter["blocked_by_loss_streak_count"] += 1
            if block_reason == "PROFIT_GIVEBACK_LOCK":
                diag_counter["blocked_by_profit_giveback_count"] += 1
            if block_reason == "SAME_SIDE_LOSS_PAUSE":
                diag_counter["blocked_by_same_side_loss_count"] += 1
            if block_reason == "MAX_SYMBOL_POSITIONS_LIVE":
                diag_counter["blocked_by_symbol_positions_count"] += 1

            if (now_utc - last_diag_dump).total_seconds() >= 3600:
                hour_key = now_utc.strftime("%Y-%m-%d_%H")
                diag_payload = {
                    "symbol": symbol,
                    "tf_entry": tf_entry,
                    "tf_gate": tf_gate,
                    "diagnostic_only": diagnostic_only,
                    "timestamp": now_utc.isoformat(),
                    **diag_counter,
                }
                diag_name = (
                    f"{diagnostic_out_name}_{hour_key}.json"
                    if diagnostic_out_name and diagnostic_out_name != "diagnostic_summary"
                    else f"diagnostic_summary_{hour_key}.json"
                )
                diag_path = diagnostic_out_dir / diag_name
                diag_path.write_text(json.dumps(diag_payload, indent=2, default=str), encoding="utf-8")
                _json_log({"event": "diagnostic_summary", "symbol": symbol, "tf": tf_entry, "path": str(diag_path), **diag_counter})
                last_diag_dump = now_utc

            event_count += 1
            health_buf.append({"signal": 1 if final_decision != "WAIT" else 0, "confidence": confidence})
            if event_count % max(1, CONFIG.live.health_check_every_n_events) == 0 and len(health_buf) >= 10:
                signal_rate = float(np.mean([x["signal"] for x in health_buf]))
                avg_conf = float(np.mean([x["confidence"] for x in health_buf]))
                _json_log(
                    {
                        "event": "health_monitor",
                        "symbol": symbol,
                        "tf": tf_entry,
                        "window_events": len(health_buf),
                        "signal_rate": signal_rate,
                        "avg_confidence": avg_conf,
                    }
                )

            if once and len(feats_entry) >= 2:
                prev_row = feats_entry.iloc[[-2]][entry_cols]
                prev_close = float(feats_entry.iloc[-2]["close"])
                curr_close = float(feats_entry.iloc[-1]["close"])
                prev_proba = model_entry.predict_proba(prev_row)[0]
                prev_decision, _, _ = _decision_from_proba(prev_proba, entry_classes)
                ret = 0.0
                if prev_decision == "BUY":
                    ret = curr_close - prev_close
                elif prev_decision == "SELL":
                    ret = prev_close - curr_close
                _json_log(
                    {
                        "event": "hypothetical_result",
                        "symbol": symbol,
                        "tf": tf_entry,
                        "prev_decision": prev_decision,
                        "prev_close": prev_close,
                        "curr_close": curr_close,
                        "hypothetical_pnl_1bar": ret,
                    }
                )

            if once:
                known_positions, _ = _notify_closed_positions(symbol, tf_entry, known_positions, live_guard)
                break
            sleep_s = wait_seconds_to_next_candle(feats_entry.iloc[-1]["time"], tf_entry)
            _sleep_with_countdown(max(1.0, sleep_s), symbol=symbol, timeframe=tf_entry)
    finally:
        shutdown()

def run_paper(
    symbol: str,
    timeframe: str,
    bars: int = 800,
    out_name: str = "phase5_paper_log.json",
    model_symbol: str | None = None,
    model_tf: str | None = None,
    model_version: str | None = None,
    use_latest_model: bool = True,
) -> Path:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    CONFIG.ensure_dirs()
    if not ensure_logged_in():
        raise RuntimeError("MT5 login failed")
    try:
        model, _ = _load_model_from_registry(
            model_symbol=model_symbol or symbol,
            model_tf=model_tf or timeframe,
            model_version=model_version,
            use_latest_model=use_latest_model,
        )
        raw = _fetch_recent(symbol=symbol, timeframe=timeframe, bars=bars)
        if raw.empty:
            raise RuntimeError("No bars returned from MT5 for paper mode")
        feats = build_features(raw)
        if len(feats) < 5:
            feat_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
            if feat_path.exists():
                feats = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True).tail(bars)
        if len(feats) < 5:
            raise RuntimeError("Not enough feature rows for paper mode")
        feature_cols = _resolve_model_feature_cols(model, feats)
        classes = model.classes_.tolist()
        logs = []
        decisions = []
        ret_series = []
        latencies = []
        rng = np.random.default_rng(42)

        for i in range(len(feats) - 1):
            row = feats.iloc[[i]][feature_cols]
            t0 = time.perf_counter()
            proba = model.predict_proba(row)[0]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            decision, p_buy, p_sell = _decision_from_proba(proba, classes)
            close_t = float(feats.iloc[i]["close"])
            close_t1 = float(feats.iloc[i + 1]["close"])
            spread = float(feats.iloc[i + 1].get("spread", 0.0) or 0.0) * 1e-5
            slip = max(0.0, spread * rng.uniform(0.3, 1.2))
            pnl = 0.0
            if decision == "BUY":
                pnl = (close_t1 - close_t) - spread - slip
            elif decision == "SELL":
                pnl = (close_t - close_t1) - spread - slip
            ret_series.append(pnl)
            decisions.append(decision)
            latencies.append(latency_ms)
            logs.append(
                {
                    "time": str(feats.iloc[i]["time"]),
                    "decision": decision,
                    "probs": {"buy": p_buy, "sell": p_sell},
                    "latency_ms": latency_ms,
                    "sim_spread_cost": spread,
                    "sim_slippage": slip,
                    "paper_pnl_1bar": pnl,
                }
            )

        active_rets = [r for d, r in zip(decisions, ret_series) if d != "WAIT"]
        report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_used": bars,
            "events": len(logs),
            "paper_trades": int(sum(1 for d in decisions if d != "WAIT")),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "avg_sim_slippage": float(np.mean([x["sim_slippage"] for x in logs])) if logs else 0.0,
            "paper_expectancy": float(np.mean(active_rets)) if active_rets else 0.0,
            "paper_cum_return": float(np.sum(active_rets)) if active_rets else 0.0,
            "sample": logs[-50:],
        }
        out_path = CONFIG.reports_dir / out_name
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out_path
    finally:
        shutdown()


def run_paper_multitf(
    symbol: str,
    tf_entry: str = "H1",
    tf_gate: str = "H4",
    policy: str = "H4_GATE_DIRECTION",
    bars: int = 1200,
    out_name: str = "phase8_paper_multitf_log.json",
    model_symbol: str | None = None,
) -> Path:
    CONFIG.ensure_dirs()
    # Use processed features for stable paper replay; fallback to live fetch if needed.
    entry_path = CONFIG.data_processed_dir / f"{symbol}_{tf_entry}_features.parquet"
    gate_path = CONFIG.data_processed_dir / f"{symbol}_{tf_gate}_features.parquet"
    if entry_path.exists():
        entry_df = pd.read_parquet(entry_path).sort_values("time").reset_index(drop=True).tail(bars)
    else:
        if not ensure_logged_in():
            raise RuntimeError("MT5 login failed")
        raw_entry = _fetch_recent(symbol=symbol, timeframe=tf_entry, bars=bars + 200)
        entry_df = build_features(raw_entry)
        shutdown()
    if gate_path.exists():
        gate_df = pd.read_parquet(gate_path).sort_values("time").reset_index(drop=True).tail(bars)
    else:
        if not ensure_logged_in():
            raise RuntimeError("MT5 login failed")
        raw_gate = _fetch_recent(symbol=symbol, timeframe=tf_gate, bars=bars + 200)
        gate_df = build_features(raw_gate)
        shutdown()

    model_symbol = model_symbol or symbol
    model_entry, _ = _load_model_from_registry(
        model_symbol=model_symbol,
        model_tf=tf_entry,
        model_version=None,
        use_latest_model=True,
    )
    model_gate, _ = _load_model_from_registry(
        model_symbol=model_symbol,
        model_tf=tf_gate,
        model_version=None,
        use_latest_model=True,
    )
    model_entry_cols = list(getattr(model_entry, "feature_name_", []))
    model_gate_cols = list(getattr(model_gate, "feature_name_", []))
    entry_cols = model_entry_cols if model_entry_cols else [c for c in entry_df.columns if c not in DROP_COLS]
    gate_cols = model_gate_cols if model_gate_cols else [c for c in gate_df.columns if c not in DROP_COLS]
    missing_e = [c for c in entry_cols if c not in entry_df.columns]
    missing_g = [c for c in gate_cols if c not in gate_df.columns]
    if missing_e:
        raise RuntimeError(f"Missing entry features for paper multitf: {missing_e[:8]}")
    if missing_g:
        raise RuntimeError(f"Missing gate features for paper multitf: {missing_g[:8]}")
    classes_e = model_entry.classes_.tolist()
    classes_g = model_gate.classes_.tolist()

    entry_probs = model_entry.predict_proba(entry_df[entry_cols])
    gate_probs = model_gate.predict_proba(gate_df[gate_cols])
    idx_e_sell = classes_e.index(0) if 0 in classes_e else 0
    idx_e_buy = classes_e.index(2) if 2 in classes_e else len(classes_e) - 1
    idx_g_sell = classes_g.index(0) if 0 in classes_g else 0
    idx_g_buy = classes_g.index(2) if 2 in classes_g else len(classes_g) - 1

    entry_signal = []
    for p in entry_probs:
        d, _, _ = _decision_from_proba(p, classes_e)
        entry_signal.append(_decision_to_int(d))
    gate_signal = []
    for p in gate_probs:
        d, _, _ = _decision_from_proba(p, classes_g)
        gate_signal.append(_decision_to_int(d))

    h1 = pd.DataFrame(
        {
            "time": pd.to_datetime(entry_df["time"], utc=True),
            "close": entry_df["close"].astype(float).values,
            "spread": entry_df.get("spread", pd.Series(0.0, index=entry_df.index)).fillna(0.0).astype(float).values,
            "signal_h1": np.array(entry_signal, dtype=int),
            "prob_h1_buy": entry_probs[:, idx_e_buy],
            "prob_h1_sell": entry_probs[:, idx_e_sell],
        }
    )
    h4 = pd.DataFrame(
        {
            "time": pd.to_datetime(gate_df["time"], utc=True),
            "signal_h4_aligned": np.array(gate_signal, dtype=int),
            "prob_h4_buy_aligned": gate_probs[:, idx_g_buy],
            "prob_h4_sell_aligned": gate_probs[:, idx_g_sell],
        }
    )
    aligned = align_h4_to_h1(h1, h4, "time", "time")
    decisions, reasons, enriched = apply_policy(aligned, policy=policy, threshold_final=0.60)

    logs = []
    ret_series = []
    for i in range(len(enriched) - 1):
        d = int(decisions[i])
        close_t = float(enriched.iloc[i]["close"])
        close_t1 = float(enriched.iloc[i + 1]["close"])
        spread = float(enriched.iloc[i + 1]["spread"]) * 1e-5
        pnl = 0.0
        if d == BUY:
            pnl = (close_t1 - close_t) - spread
        elif d == SELL:
            pnl = (close_t - close_t1) - spread
        ret_series.append(pnl)
        logs.append(
            {
                "time": str(enriched.iloc[i]["time"]),
                "has_h4_pred": bool(enriched.iloc[i].get("has_h4_pred", False)),
                "has_h1_pred": bool(enriched.iloc[i].get("has_h1_pred", True)),
                "signal_h4_aligned": int(enriched.iloc[i]["signal_h4_aligned"]),
                "signal_h1": int(enriched.iloc[i]["signal_h1"]),
                "decision_final": int(d),
                "blocked_reason": reasons[i],
                "reason_missing": (
                    reasons[i]
                    if reasons[i] in {"blocked_by_missing_h4_pred", "blocked_by_missing_h1_pred"}
                    else ""
                ),
                "prob_h4": {
                    "buy": float(enriched.iloc[i]["prob_h4_buy_aligned"]),
                    "sell": float(enriched.iloc[i]["prob_h4_sell_aligned"]),
                },
                "prob_h1": {
                    "buy": float(enriched.iloc[i]["prob_h1_buy"]),
                    "sell": float(enriched.iloc[i]["prob_h1_sell"]),
                },
                "paper_pnl_1bar": pnl,
            }
        )

    active = [x["paper_pnl_1bar"] for x in logs if x["decision_final"] != 0]
    report = {
        "symbol": symbol,
        "tf_entry": tf_entry,
        "tf_gate": tf_gate,
        "policy": policy,
        "events": len(logs),
        "paper_trades": int(sum(1 for x in logs if x["decision_final"] != 0)),
        "paper_expectancy": float(np.mean(active)) if active else 0.0,
        "paper_cum_return": float(np.sum(active)) if active else 0.0,
        "blocked_reason_counts": pd.Series(reasons).value_counts().to_dict(),
        "sample": logs[-80:],
    }
    out_path = CONFIG.reports_dir / out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MT5 live bot loop.")
    parser.add_argument("--symbol", default=CONFIG.live.default_symbol)
    parser.add_argument("--tf", default=CONFIG.live.default_tf)
    parser.add_argument("--timeframe", default=None, help="Alias for --tf")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-trade", action="store_true", help="Do not send orders; only log decisions.")
    parser.add_argument("--paper", action="store_true", help="Run paper simulation mode without real orders.")
    parser.add_argument("--paper-bars", type=int, default=800)
    parser.add_argument("--tf_entry", default="H1")
    parser.add_argument("--tf_gate", default="H4")
    parser.add_argument("--policy", default=None, help="Multi-TF paper policy: H4_GATE_DIRECTION|DOUBLE_CONFIRMATION|ENSEMBLE_SCORE")
    parser.add_argument("--out-name", default=None, help="Output report file name in reports/.")
    parser.add_argument("--diagnostic-only", action="store_true", help="Run full decision pipeline but never send orders.")
    parser.add_argument("--out", default=None, help="Output directory for diagnostic summaries.")
    parser.add_argument("--model-symbol", default=None, help="Model symbol override (default = --symbol)")
    parser.add_argument("--model-tf", default=None, help="Model timeframe override (default = tf entry)")
    parser.add_argument("--model-version", default=None, help="Specific model version from registry")
    parser.add_argument("--use-latest-model", dest="use_latest_model", action="store_true", help="Use latest model from registry (default)")
    parser.add_argument("--no-use-latest-model", dest="use_latest_model", action="store_false", help="Disable latest auto-pick and allow --model-version")
    parser.set_defaults(use_latest_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf = args.timeframe or args.tf
    if args.paper:
        if args.policy:
            out = run_paper_multitf(
                symbol=args.symbol,
                tf_entry=args.tf_entry,
                tf_gate=args.tf_gate,
                policy=args.policy,
                bars=args.paper_bars,
                out_name=args.out_name or "phase8_paper_multitf_log.json",
                model_symbol=args.model_symbol,
            )
        else:
            out = run_paper(
                symbol=args.symbol,
                timeframe=tf,
                bars=args.paper_bars,
                model_symbol=args.model_symbol,
                model_tf=args.model_tf,
                model_version=args.model_version,
                use_latest_model=args.use_latest_model,
            )
        print(f"saved={out}")
    else:
        diag_dir = Path(args.out) if args.out else CONFIG.reports_dir
        run_live(
            symbol=args.symbol,
            timeframe=tf,
            once=args.once,
            no_trade=args.no_trade,
            diagnostic_only=args.diagnostic_only,
            diagnostic_out_dir=diag_dir,
            diagnostic_out_name=args.out_name or "diagnostic_summary",
            model_symbol=args.model_symbol,
            model_tf=args.model_tf,
            model_version=args.model_version,
            use_latest_model=args.use_latest_model,
        )


if __name__ == "__main__":
    main()

