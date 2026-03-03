from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CONFIG
from .executor_mt5 import close_position, send_order
from .features import build_features
from .multitf import BUY, SELL, WAIT, align_h4_to_h1, apply_policy
from .mt5_connect import ensure_logged_in, shutdown
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
        "MAX_TRADES_PER_DAY": "MAX_TRADES_DIA",
        "MAX_DAILY_LOSS_PCT": "MAX_PERDA_DIA",
        "MIN_MARGIN_LEVEL_PCT": "MARGEM_BAIXA",
        "SESSION_FILTER": "FORA_DA_SESSAO",
        "NEWS_BLACKOUT": "JANELA_NOTICIA",
        "SPREAD_FILTER": "SPREAD_ALTO",
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
    return {"BUY": "🟢 COMPRA", "SELL": "🔴 VENDA", "WAIT": "🟡 AGUARDAR"}.get(decision, decision)


def _latest_model(symbol: str, timeframe: str) -> Path:
    files = sorted(CONFIG.models_dir.glob(f"{symbol}_{timeframe}_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No model found for {symbol}/{timeframe}")
    return files[-1]


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


def _compute_live_lot(symbol: str, side: str, balance: float, entry_price: float, sl_price: float) -> float:
    cfg = CONFIG.risk
    fallback = _normalize_lot(symbol, cfg.fixed_demo_lot)
    if not cfg.use_dynamic_position_sizing:
        return fallback
    if mt5 is None or balance <= 0:
        return fallback
    risk_amount = balance * (cfg.default_risk_pct / 100.0)
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


def _apply_time_stop(symbol: str, timeframe: str) -> int:
    if mt5 is None:
        return 0
    max_candles = int(CONFIG.live.max_holding_candles)
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
                    f"⏱️ Time Stop acionado\n"
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
            "time": float(getattr(p, "time", 0.0) or 0.0),
            "profit": float(getattr(p, "profit", 0.0) or 0.0),
        }
    return out


def _closed_position_pnl(ticket: int) -> float | None:
    if mt5 is None:
        return None
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        deals = mt5.history_deals_get(start, end, position=ticket)
        if deals is None or len(deals) == 0:
            return None
        pnl = 0.0
        for d in deals:
            pnl += float(getattr(d, "profit", 0.0) or 0.0)
            pnl += float(getattr(d, "commission", 0.0) or 0.0)
            pnl += float(getattr(d, "swap", 0.0) or 0.0)
            pnl += float(getattr(d, "fee", 0.0) or 0.0)
        return pnl
    except Exception:
        return None


def _notify_closed_positions(symbol: str, known_positions: dict[int, dict]) -> dict[int, dict]:
    current = _current_positions_map(symbol)
    closed_tickets = [t for t in known_positions.keys() if t not in current]
    for ticket in closed_tickets:
        prev = known_positions[ticket]
        pnl = _closed_position_pnl(ticket)
        _json_log(
            {
                "event": "position_closed_detected",
                "symbol": symbol,
                "ticket": ticket,
                "side": prev.get("side"),
                "volume": prev.get("volume"),
                "pnl": pnl,
            }
        )
        pnl_txt = f"{pnl:.2f}" if pnl is not None else "n/a"
        _notify_telegram(
            message=(
                f"✅ Posição fechada\n"
                f"Ativo: {symbol}\n"
                f"Ticket: {ticket}\n"
                f"Lado: {prev.get('side')} | Volume: {prev.get('volume')}\n"
                f"PnL: {pnl_txt}"
            ),
            key=f"close:{symbol}:{ticket}",
            cooldown_seconds=0,
        )
    return current


def _decision_from_proba(proba: np.ndarray, classes: list[int]) -> tuple[str, float, float]:
    p_sell = float(proba[classes.index(0)]) if 0 in classes else 0.0
    p_buy = float(proba[classes.index(2)]) if 2 in classes else 0.0
    decision = "WAIT"
    if p_buy >= CONFIG.live.threshold and p_buy > p_sell:
        decision = "BUY"
    elif p_sell >= CONFIG.live.threshold and p_sell > p_buy:
        decision = "SELL"
    return decision, p_buy, p_sell


def _resolve_model_feature_cols(model, feats: pd.DataFrame) -> list[str]:
    model_cols = list(getattr(model, "feature_name_", []))
    if model_cols:
        missing = [c for c in model_cols if c not in feats.columns]
        for col in missing:
            # Keep live inference shape-compatible with training schema.
            feats[col] = 0.0
        return model_cols
    return [c for c in feats.columns if c not in DROP_COLS]


def run_live(symbol: str, timeframe: str, once: bool = False, no_trade: bool = False) -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    CONFIG.ensure_dirs()
    if not ensure_logged_in():
        raise RuntimeError("MT5 login failed")

    model = joblib.load(_latest_model(symbol, timeframe))
    _notify_telegram(
        message=(
            f"🤖 Bot iniciado\n"
            f"Ativo: {symbol} | TF: {timeframe}\n"
            f"Modo sem ordem: {'SIM' if no_trade else 'NÃO'}"
        ),
        key=f"boot:{symbol}:{timeframe}",
        cooldown_seconds=120,
    )
    failures = 0
    today_stats = TodayStats(trades_count=0, last_trade_time=None)
    known_positions = _current_positions_map(symbol)
    health_buf = deque(maxlen=max(10, CONFIG.live.health_check_every_n_events))
    event_count = 0

    try:
        while True:
            known_positions = _notify_closed_positions(symbol, known_positions)
            _apply_time_stop(symbol=symbol, timeframe=timeframe)
            raw = _fetch_recent(symbol=symbol, timeframe=timeframe, bars=CONFIG.live.fetch_bars)
            if raw.empty:
                failures += 1
                _json_log({"event": "fetch_fail", "failures": failures, "symbol": symbol, "tf": timeframe})
                if failures >= CONFIG.live.max_mt5_failures:
                    _json_log({"event": "kill_switch", "reason": "MAX_MT5_FAILURES"})
                    _notify_telegram(
                        message=(
                            f"🛑 Kill Switch: falhas MT5\n"
                            f"Ativo: {symbol} | TF: {timeframe}\n"
                            f"Motivo: MAX_MT5_FAILURES"
                        ),
                        key=f"kill:max_fail:{symbol}:{timeframe}",
                        cooldown_seconds=120,
                    )
                    break
                time.sleep(3)
                continue
            failures = 0

            feats = build_features(raw)
            if feats.empty:
                time.sleep(3)
                continue

            feature_cols = _resolve_model_feature_cols(model, feats)
            row = feats.iloc[[-1]][feature_cols]
            proba = model.predict_proba(row)[0]
            classes = model.classes_.tolist()
            decision, p_buy, p_sell = _decision_from_proba(proba, classes)
            confidence = max(p_buy, p_sell)
            regime_class = str(feats.iloc[-1].get("regime_class", "NEUTRAL"))
            spread_points = float(feats.iloc[-1].get("spread", 0.0) or 0.0)
            now_utc = datetime.now(timezone.utc)

            state = _account_state(symbol)
            if state.balance > 0:
                loss_pct = max(0.0, (-state.daily_pnl / state.balance) * 100.0)
                if loss_pct >= CONFIG.risk.max_daily_loss_pct:
                    _json_log({"event": "kill_switch", "reason": "MAX_DAILY_LOSS"})
                    _notify_telegram(
                        message=(
                            f"🛑 Kill Switch: perda diária\n"
                            f"Ativo: {symbol} | TF: {timeframe}\n"
                            f"Loss diário: {loss_pct:.2f}%"
                        ),
                        key=f"kill:max_loss:{symbol}:{timeframe}",
                        cooldown_seconds=120,
                    )
                    break
            if state.margin_level_pct < CONFIG.risk.min_margin_level_pct:
                _json_log({"event": "kill_switch", "reason": "LOW_MARGIN_LEVEL"})
                _notify_telegram(
                    message=(
                        f"🛑 Kill Switch: margem baixa\n"
                        f"Ativo: {symbol} | TF: {timeframe}\n"
                        f"Nível de margem: {state.margin_level_pct:.2f}%"
                    ),
                    key=f"kill:margin:{symbol}:{timeframe}",
                    cooldown_seconds=120,
                )
                break

            if decision != "WAIT":
                if CONFIG.live.use_session_filter and not _is_in_session(now_utc, CONFIG.live.allowed_sessions_utc):
                    reason = "SESSION_FILTER"
                    _json_log(
                        payload := {
                            "event": "decision_blocked",
                            "symbol": symbol,
                            "tf": timeframe,
                            "decision": decision,
                            "reason": reason,
                            "decision_panel": {
                                "threshold": CONFIG.live.threshold,
                                "p_buy": p_buy,
                                "p_sell": p_sell,
                                "confidence": confidence,
                                "regime_class": regime_class,
                                "spread_points": spread_points,
                            },
                        }
                    )
                    _update_daily_report(payload)
                    event_count += 1
                    health_buf.append({"signal": 0, "confidence": confidence})
                    if once:
                        break
                    sleep_s = wait_seconds_to_next_candle(feats.iloc[-1]["time"], timeframe)
                    _sleep_with_countdown(max(1.0, sleep_s), symbol=symbol, timeframe=timeframe)
                    continue
                if _is_news_blackout(now_utc):
                    reason = "NEWS_BLACKOUT"
                    _json_log(
                        payload := {
                            "event": "decision_blocked",
                            "symbol": symbol,
                            "tf": timeframe,
                            "decision": decision,
                            "reason": reason,
                            "decision_panel": {
                                "threshold": CONFIG.live.threshold,
                                "p_buy": p_buy,
                                "p_sell": p_sell,
                                "confidence": confidence,
                                "regime_class": regime_class,
                                "spread_points": spread_points,
                            },
                        }
                    )
                    _update_daily_report(payload)
                    event_count += 1
                    health_buf.append({"signal": 0, "confidence": confidence})
                    if once:
                        break
                    sleep_s = wait_seconds_to_next_candle(feats.iloc[-1]["time"], timeframe)
                    _sleep_with_countdown(max(1.0, sleep_s), symbol=symbol, timeframe=timeframe)
                    continue
                if spread_points > CONFIG.live.max_spread_points:
                    reason = "SPREAD_FILTER"
                    _json_log(
                        payload := {
                            "event": "decision_blocked",
                            "symbol": symbol,
                            "tf": timeframe,
                            "decision": decision,
                            "reason": reason,
                            "decision_panel": {
                                "threshold": CONFIG.live.threshold,
                                "p_buy": p_buy,
                                "p_sell": p_sell,
                                "confidence": confidence,
                                "regime_class": regime_class,
                                "spread_points": spread_points,
                            },
                        }
                    )
                    _update_daily_report(payload)
                    event_count += 1
                    health_buf.append({"signal": 0, "confidence": confidence})
                    if once:
                        break
                    sleep_s = wait_seconds_to_next_candle(feats.iloc[-1]["time"], timeframe)
                    _sleep_with_countdown(max(1.0, sleep_s), symbol=symbol, timeframe=timeframe)
                    continue

                can_open, reason = can_open_trade(
                    state,
                    _positions_count(symbol),
                    today_stats,
                )
                if can_open:
                    atr = float(feats.iloc[-1]["ATR_14"])
                    price = float(feats.iloc[-1]["close"])
                    sl_dist = CONFIG.triple_barrier.sl_atr_mult * atr
                    tp_dist = CONFIG.triple_barrier.pt_atr_mult * atr
                    if decision == "BUY":
                        sl = price - sl_dist
                        tp = price + tp_dist
                    else:
                        sl = price + sl_dist
                        tp = price - tp_dist

                    lot = _compute_live_lot(
                        symbol=symbol,
                        side=decision,
                        balance=state.balance,
                        entry_price=price,
                        sl_price=sl,
                    )
                    regime_mult_map = {
                        "TREND": CONFIG.risk.regime_mult_trend,
                        "HIGH_VOL": CONFIG.risk.regime_mult_high_vol,
                        "SIDEWAYS": CONFIG.risk.regime_mult_sideways,
                        "NEUTRAL": CONFIG.risk.regime_mult_neutral,
                    }
                    regime_mult = float(regime_mult_map.get(regime_class, CONFIG.risk.regime_mult_neutral))
                    lot = _normalize_lot(symbol, lot * regime_mult)
                    ok, resp = (False, None)
                    if not no_trade:
                        ok, resp = send_order(
                            symbol=symbol,
                            side=decision,
                            lot=lot,
                            sl=sl,
                            tp=tp,
                            deviation_points=CONFIG.live.order_deviation_points,
                        )
                    if ok:
                        today_stats.trades_count += 1
                        today_stats.last_trade_time = datetime.now(timezone.utc)
                        _notify_telegram(
                            message=(
                                f"🚀 Nova entrada {_emoji_decision(decision)}\n"
                                f"Ativo: {symbol} | TF: {timeframe}\n"
                                f"Preço: {price:.5f}\n"
                                f"Prob. compra: {p_buy*100:.1f}% | Prob. venda: {p_sell*100:.1f}%\n"
                                f"SL: {sl:.5f} | TP: {tp:.5f} | Lote: {lot:.2f}\n"
                                f"Regime: {regime_class} | Spread: {spread_points:.1f}"
                            ),
                            key=f"entry:{symbol}:{timeframe}",
                            cooldown_seconds=30,
                        )
                    elif not no_trade:
                        _notify_telegram(
                            message=(
                                f"⚠️ Falha ao enviar ordem {_emoji_decision(decision)}\n"
                                f"Ativo: {symbol} | TF: {timeframe}\n"
                                f"Prob. compra: {p_buy*100:.1f}% | Prob. venda: {p_sell*100:.1f}%"
                            ),
                            key=f"order_fail:{symbol}:{timeframe}",
                            cooldown_seconds=60,
                        )
                    elif no_trade:
                        _notify_telegram(
                            message=(
                                f"🧪 Sinal (sem envio de ordem) {_emoji_decision(decision)}\n"
                                f"Ativo: {symbol} | TF: {timeframe}\n"
                                f"Prob. compra: {p_buy*100:.1f}% | Prob. venda: {p_sell*100:.1f}%"
                            ),
                            key=f"dry_signal:{symbol}:{timeframe}",
                            cooldown_seconds=60,
                        )
                    _json_log(
                        payload := {
                            "event": "decision",
                            "symbol": symbol,
                            "tf": timeframe,
                            "price": price,
                            "decision": decision,
                            "probs": {"buy": p_buy, "sell": p_sell},
                            "sl": sl,
                            "tp": tp,
                            "order_id": getattr(resp, "order", None) if resp else None,
                            "no_trade": no_trade,
                            "shap_top_features": _compute_shap_top(model, row),
                            "can_open_reason": reason,
                            "decision_panel": {
                                "threshold": CONFIG.live.threshold,
                                "p_buy": p_buy,
                                "p_sell": p_sell,
                                "confidence": confidence,
                                "regime_class": regime_class,
                                "spread_points": spread_points,
                                "regime_mult": regime_mult,
                            },
                        }
                    )
                    _update_daily_report(payload)
                else:
                    _json_log(payload := {
                            "event": "decision_blocked",
                            "symbol": symbol,
                            "tf": timeframe,
                            "decision": decision,
                            "reason": reason,
                            "decision_panel": {
                                "threshold": CONFIG.live.threshold,
                                "p_buy": p_buy,
                                "p_sell": p_sell,
                                "confidence": confidence,
                                "regime_class": regime_class,
                                "spread_points": spread_points,
                            },
                        })
                    _notify_telegram(
                        message=(
                            f"⛔ Sinal bloqueado\n"
                            f"Ativo: {symbol} | TF: {timeframe}\n"
                            f"Sinal: {_emoji_decision(decision)}\n"
                            f"Motivo: {reason}"
                        ),
                        key=f"blocked:{symbol}:{timeframe}:{reason}",
                        cooldown_seconds=120,
                    )
                    _update_daily_report(payload)
            else:
                _json_log(payload := {
                        "event": "decision",
                        "symbol": symbol,
                        "tf": timeframe,
                        "decision": decision,
                        "no_trade": no_trade,
                        "probs": {"buy": p_buy, "sell": p_sell},
                        "decision_panel": {
                            "threshold": CONFIG.live.threshold,
                            "p_buy": p_buy,
                            "p_sell": p_sell,
                            "confidence": confidence,
                            "regime_class": regime_class,
                            "spread_points": spread_points,
                        },
                    })
                _update_daily_report(payload)
            event_count += 1
            health_buf.append({"signal": 1 if decision != "WAIT" else 0, "confidence": confidence})
            if event_count % max(1, CONFIG.live.health_check_every_n_events) == 0 and len(health_buf) >= 10:
                signal_rate = float(np.mean([x["signal"] for x in health_buf]))
                avg_conf = float(np.mean([x["confidence"] for x in health_buf]))
                _json_log(
                    {
                        "event": "health_monitor",
                        "symbol": symbol,
                        "tf": timeframe,
                        "window_events": len(health_buf),
                        "signal_rate": signal_rate,
                        "avg_confidence": avg_conf,
                    }
                )
                if signal_rate < CONFIG.live.min_signal_rate_alert or avg_conf < CONFIG.live.min_confidence_alert:
                    _notify_telegram(
                        message=(
                            f"📉 Alerta de saúde do modelo\n"
                            f"Ativo: {symbol} | TF: {timeframe}\n"
                            f"Taxa de sinais: {signal_rate*100:.1f}%\n"
                            f"Confiança média: {avg_conf*100:.1f}%"
                        ),
                        key=f"health_alert:{symbol}:{timeframe}",
                        cooldown_seconds=300,
                    )

            # Hypothetical 1-bar result for sanity check in --once mode.
            if once and len(feats) >= 2:
                prev_row = feats.iloc[[-2]][feature_cols]
                prev_close = float(feats.iloc[-2]["close"])
                curr_close = float(feats.iloc[-1]["close"])
                prev_proba = model.predict_proba(prev_row)[0]
                prev_decision, _, _ = _decision_from_proba(prev_proba, classes)
                ret = 0.0
                if prev_decision == "BUY":
                    ret = curr_close - prev_close
                elif prev_decision == "SELL":
                    ret = prev_close - curr_close
                _json_log(
                    {
                        "event": "hypothetical_result",
                        "symbol": symbol,
                        "tf": timeframe,
                        "prev_decision": prev_decision,
                        "prev_close": prev_close,
                        "curr_close": curr_close,
                        "hypothetical_pnl_1bar": ret,
                    }
                )

            if once:
                known_positions = _notify_closed_positions(symbol, known_positions)
                break
            sleep_s = wait_seconds_to_next_candle(feats.iloc[-1]["time"], timeframe)
            _sleep_with_countdown(max(1.0, sleep_s), symbol=symbol, timeframe=timeframe)
    finally:
        shutdown()


def run_paper(symbol: str, timeframe: str, bars: int = 800, out_name: str = "phase5_paper_log.json") -> Path:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    CONFIG.ensure_dirs()
    if not ensure_logged_in():
        raise RuntimeError("MT5 login failed")
    try:
        model = joblib.load(_latest_model(symbol, timeframe))
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


def _decision_to_int(decision: str) -> int:
    if decision == "BUY":
        return BUY
    if decision == "SELL":
        return SELL
    return WAIT


def run_paper_multitf(
    symbol: str,
    tf_entry: str = "H1",
    tf_gate: str = "H4",
    policy: str = "H4_GATE_DIRECTION",
    bars: int = 1200,
    out_name: str = "phase8_paper_multitf_log.json",
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

    model_entry = joblib.load(_latest_model(symbol, tf_entry))
    model_gate = joblib.load(_latest_model(symbol, tf_gate))
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
            )
        else:
            out = run_paper(symbol=args.symbol, timeframe=tf, bars=args.paper_bars)
        print(f"saved={out}")
    else:
        run_live(symbol=args.symbol, timeframe=tf, once=args.once, no_trade=args.no_trade)


if __name__ == "__main__":
    main()
