from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, get_symbol_profile_entry, resolve_timeframe_profile
from .model_registry import get_latest_model, list_models
from .mt5_connect import ensure_logged_in
from .openai_tuner import suggest_profile_updates

try:
    import msvcrt  # Windows-only; usado para sair do monitor com tecla Q
except ImportError:  # pragma: no cover
    msvcrt = None

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Group = None
    Live = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TimeElapsedColumn = None
    Table = None


RUNS_DIR = CONFIG.reports_dir / "runs"
LAST_SELECTION = RUNS_DIR / "last_selection.json"
SYMBOL_PROFILES = RUNS_DIR / "symbol_profiles.json"
MT5_CREDS_FILE = RUNS_DIR / "mt5_credentials.json"
OPENAI_CREDS_FILE = RUNS_DIR / "openai_credentials.json"
TF_OVERRIDES_FILE = RUNS_DIR / "timeframe_overrides.json"
_CONSOLE = Console() if Console else None
_ACTION_LABELS = {
    "train": "Treino concluido",
    "train_empty_dataset": "Treino abortado (dataset vazio)",
    "train_prepare_empty_raw": "Preparacao falhou (sem candles)",
    "train_prepare_empty_features": "Preparacao falhou (features vazias)",
    "train_prepare_fail": "Preparacao falhou",
    "train_inconsistent_cancel": "Treino cancelado (inconsistencia)",
    "bot": "Execucao do bot",
    "trade_live": "Trade iniciado (background)",
    "post_train_flow": "Fluxo pos-treino",
    "scalp_preset": "Scalping preset aplicado",
    "scalp_start": "Scalping iniciado",
    "intraday_active_preset": "Preset intraday ativo aplicado",
    "intraday_active_start": "Intraday ativo iniciado",
}


def _print(msg: str, style: str | None = None) -> None:
    if _CONSOLE is not None and style:
        _CONSOLE.print(msg, style=style)
    elif _CONSOLE is not None:
        _CONSOLE.print(msg)
    else:
        print(msg)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(symbol: str, tf: str, action: str) -> str:
    return f"{_now_utc().strftime('%Y%m%d_%H%M%S')}_{symbol}_{tf}_{action}"


def _ensure_dirs() -> None:
    CONFIG.ensure_dirs()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    creds = _load_mt5_credentials()
    for k, v in creds.items():
        if v:
            os.environ[k] = v
    openai_creds = _load_openai_credentials()
    if openai_creds.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(openai_creds["OPENAI_API_KEY"])
    if openai_creds.get("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = str(openai_creds["OPENAI_MODEL"])


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _upsert_symbol_profile(item: dict[str, Any]) -> None:
    data = _load_json(SYMBOL_PROFILES, {"profiles": []})
    profiles = [p for p in data.get("profiles", []) if not (p.get("symbol") == item.get("symbol") and p.get("tf") == item.get("tf"))]
    profiles.append(item)
    data["profiles"] = profiles
    _save_json(SYMBOL_PROFILES, data)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_tf_overrides() -> dict[str, Any]:
    data = _load_json(TF_OVERRIDES_FILE, {"profiles": {}})
    if not isinstance(data, dict):
        return {"profiles": {}}
    if not isinstance(data.get("profiles"), dict):
        data["profiles"] = {}
    return data


def _save_tf_override(tf: str, updates: dict[str, Any], reason: str) -> None:
    data = _load_tf_overrides()
    profiles = data.setdefault("profiles", {})
    base = profiles.get(tf, {})
    if not isinstance(base, dict):
        base = {}
    merged = {**base, **updates, "updated_at": _now_utc().isoformat(), "reason": reason}
    profiles[tf] = merged
    _save_json(TF_OVERRIDES_FILE, data)


def _save_last_selection(payload: dict[str, Any]) -> None:
    _save_json(LAST_SELECTION, payload)


def _load_mt5_credentials() -> dict[str, str]:
    data = _load_json(MT5_CREDS_FILE, {})
    out: dict[str, str] = {}
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        v = str(data.get(k, "")).strip() if data.get(k) is not None else ""
        if v:
            out[k] = v
    return out


def _load_openai_credentials() -> dict[str, str]:
    data = _load_json(OPENAI_CREDS_FILE, {})
    out: dict[str, str] = {}
    for k in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        raw = data.get(k)
        v = str(raw).strip() if raw is not None else ""
        if v:
            out[k] = v
    return out


def _mt5_status() -> dict[str, Any]:
    creds = _load_mt5_credentials()
    has_creds = bool(creds.get("MT5_LOGIN") and creds.get("MT5_PASSWORD") and creds.get("MT5_SERVER"))
    installed = mt5 is not None
    logged_in = False
    account_login = None
    if installed:
        try:
            logged_in = bool(ensure_logged_in())
            if logged_in and mt5 is not None:
                acc = mt5.account_info()
                account_login = getattr(acc, "login", None) if acc else None
        except Exception:
            logged_in = False
    return {
        "has_creds": has_creds,
        "installed": installed,
        "logged_in": logged_in,
        "account_login": account_login,
    }


def _openai_status() -> dict[str, Any]:
    creds = _load_openai_credentials()
    has_key = bool(creds.get("OPENAI_API_KEY"))
    model = str(creds.get("OPENAI_MODEL", "")).strip() or "gpt-4.1-mini"
    return {"has_key": has_key, "model": model}


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    creds = _load_mt5_credentials()
    env.update(creds)
    oai = _load_openai_credentials()
    if oai.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = oai["OPENAI_API_KEY"]
    if oai.get("OPENAI_MODEL"):
        env["OPENAI_MODEL"] = oai["OPENAI_MODEL"]
    return env


def _validate_symbol(symbol: str) -> bool:
    if mt5 is None:
        return True
    if not ensure_logged_in():
        return False
    info = mt5.symbol_info(symbol)
    return info is not None


def _available_symbols() -> list[str]:
    if mt5 is None:
        return []
    try:
        if not ensure_logged_in():
            return []
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        names = sorted({str(getattr(s, "name", "")).upper() for s in symbols if getattr(s, "name", "")})
        return names
    except Exception:
        return []


def _choose_symbol(default: str = "EURUSD") -> str:
    names = _available_symbols()
    if not names:
        return _ask("Simbolo", default, to_upper=True)
    filt = _ask("Filtro de simbolo (ENTER=todos)", "", to_upper=True)
    filtered = [n for n in names if filt in n] if filt else names
    if not filtered:
        _print(f"Nenhum simbolo com filtro '{filt}'. Usando digitacao manual.", style="yellow")
        return _ask("Simbolo", default, to_upper=True)
    max_show = 80
    show = filtered[:max_show]
    if Table is not None and _CONSOLE is not None:
        tb = Table(title="Escolha o Simbolo", show_lines=False)
        tb.add_column("#", style="cyan")
        tb.add_column("Simbolo", style="white")
        for i, s in enumerate(show, start=1):
            tb.add_row(str(i), s)
        _CONSOLE.print(tb)
    else:
        _print("Lista de simbolos:")
        for i, s in enumerate(show, start=1):
            _print(f"{i}) {s}")
    if len(filtered) > max_show:
        _print(f"Mostrando {max_show} de {len(filtered)}. Use filtro para refinar.", style="yellow")
    idx_raw = _ask("Escolha o numero (0 = digitar manualmente)", "1")
    try:
        idx = int(idx_raw)
    except ValueError:
        idx = 0
    if idx <= 0:
        return _ask("Simbolo", default, to_upper=True)
    if 1 <= idx <= len(show):
        return show[idx - 1].upper()
    _print("Indice invalido, usando valor padrao.", style="yellow")
    return default.upper()


def _ask(prompt: str, default: str | None = None, to_upper: bool = False) -> str:
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    raw = input(msg).strip()
    val = raw if raw else (default or "")
    return val.upper() if to_upper else val


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "s" if default else "n"
    raw = _ask(f"{prompt} (s/n)", d).strip().lower()
    return raw in {"s", "sim", "y", "yes", "1", "true"}


def _pause_continue() -> None:
    prompt = "Pressione ENTER para voltar ao menu..."
    if _CONSOLE is not None:
        _CONSOLE.print(f"[dim]{prompt}[/dim]")
    else:
        print(prompt)
    try:
        input()
    except EOFError:
        return


def _tail_lines(path: Path, n: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
    except Exception:
        return []


def _print_run_result(run_dir: Path, meta: dict[str, Any]) -> None:
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")
    rc = int(meta.get("return_code", 1))
    if rc == 0:
        return
    _print(f"Execucao falhou (return_code={rc}).", style="red")
    tail = _tail_lines(run_dir / "logs" / "stderr.log", n=30)
    if not tail:
        tail = _tail_lines(run_dir / "logs" / "stdout.log", n=30)
    if not tail:
        _print("Sem detalhes nos logs.", style="yellow")
        return
    _print("Ultimas linhas do erro:", style="yellow")
    for line in tail[-12:]:
        if line.strip():
            _print(f"  {line}", style="yellow")


def _find_summary_path(meta: dict[str, Any], default_name: str) -> Path | None:
    for p in meta.get("saved_paths", []):
        s = str(p)
        if "summary" in s.lower() and s.lower().endswith(".json"):
            path = Path(s)
            if path.exists():
                return path
    default = CONFIG.reports_dir / default_name
    return default if default.exists() else None


def _metric_bar(value: float, target: float | None = None, *, width: int = 24, invert: bool = False) -> str:
    try:
        val = float(value)
    except Exception:
        val = 0.0
    if target is None or target == 0:
        ratio = max(0.0, min(1.0, abs(val)))
    else:
        ratio = max(0.0, min(1.0, abs(val) / abs(float(target))))
    if invert:
        ratio = max(0.0, min(1.0, 1.0 - ratio))
    filled = int(round(width * ratio))
    return "#" * filled + "-" * max(0, width - filled)


def _print_phase_summary_dashboard(data: dict[str, Any]) -> None:
    result = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
    criteria = data.get("criteria", {}) if isinstance(data.get("criteria"), dict) else {}
    approved = bool(data.get("approved", False))
    symbol = str(data.get("symbol", "-"))
    entry_tf = str(data.get("entry_tf", "-"))
    gate_tf = str(data.get("gate_tf", "-"))

    pf_ratio = float(result.get("pf_gt_1_ratio_valid_only", 0.0))
    pf_target = float(criteria.get("pf_gt_1_ratio_valid_only_min", 0.60))
    stress25_pf = float(result.get("stress25_pf", 0.0))
    coverage = float(result.get("coverage_ratio", 0.0))
    trades_total = int(result.get("trades_total", 0))
    trade_windows = int(result.get("trade_windows_count", 0))
    no_signal = int(result.get("no_signal_windows", 0))
    dd_ok = bool(result.get("dd_windows_ok", False))

    status_txt = "APROVADO" if approved else "REPROVADO"
    status_style = "green" if approved else "red"
    _print(f"Resumo da robustez [{symbol} {entry_tf}/{gate_tf}] => {status_txt}", style=status_style)

    if Table is not None and _CONSOLE is not None:
        tb = Table(title="Painel de Robustez", show_lines=False)
        tb.add_column("Metrica", style="cyan")
        tb.add_column("Valor", style="white")
        tb.add_column("Meta", style="magenta")
        tb.add_column("Leitura", style="green")
        tb.add_row("PF ratio", f"{pf_ratio:.3f}", f">= {pf_target:.2f}", _metric_bar(pf_ratio, pf_target))
        tb.add_row("Stress +25%", f"{stress25_pf:.3f}", "> 1.00", _metric_bar(stress25_pf, 1.0))
        tb.add_row("Coverage", f"{coverage:.3f}", ">= 0.90", _metric_bar(coverage, 0.90))
        tb.add_row("Trades totais", str(trades_total), ">= 120", _metric_bar(trades_total, 120.0))
        tb.add_row("Janelas com trade", str(trade_windows), ">= 6", _metric_bar(trade_windows, 6.0))
        tb.add_row("Janelas sem sinal", str(no_signal), "ideal 0", _metric_bar(no_signal, 8.0, invert=True))
        tb.add_row("Drawdown por janela", "OK" if dd_ok else "FALHOU", "OK", "########################" if dd_ok else "######------------------")
        _CONSOLE.print(tb)
    else:
        _print(f"PF ratio        {pf_ratio:.3f}  meta>={pf_target:.2f}  {_metric_bar(pf_ratio, pf_target)}", style="cyan")
        _print(f"Stress +25%     {stress25_pf:.3f}  meta>1.00  {_metric_bar(stress25_pf, 1.0)}", style="cyan")
        _print(f"Coverage        {coverage:.3f}  meta>=0.90 {_metric_bar(coverage, 0.90)}", style="cyan")
        _print(f"Trades totais   {trades_total}  meta>=120 {_metric_bar(trades_total, 120.0)}", style="cyan")
        _print(f"Janelas trade   {trade_windows}  meta>=6   {_metric_bar(trade_windows, 6.0)}", style="cyan")
        _print(f"Janelas sem sinal {no_signal}  ideal 0   {_metric_bar(no_signal, 8.0, invert=True)}", style="cyan")
        _print(f"DD por janela   {'OK' if dd_ok else 'FALHOU'}", style="cyan")

    _print(
        f"Leitura direta: {'estrutura forte para paper/demo' if approved else 'ainda nao robusto; falta edge/frequencia/saida'}",
        style=("green" if approved else "yellow"),
    )


def _print_phase_summary(summary_path: Path) -> None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _print(f"Falha ao ler summary: {summary_path} ({exc})", style="red")
        return
    if not isinstance(data, dict):
        _print(f"Summary invalido: {summary_path}", style="red")
        return

    # Summary phase10/11 format
    if "result" in data:
        _print_phase_summary_dashboard(data)
        _print(f"Resumo auditavel salvo em: {summary_path}", style="magenta")
        return

    # Summary phase4 format (diagnostico)
    by_tf = data.get("by_timeframe", {})
    if isinstance(by_tf, dict) and by_tf:
        tf, tf_data = next(iter(by_tf.items()))
        best = tf_data.get("best_experiment", {}) if isinstance(tf_data, dict) else {}
        _print(f"Resumo do diagnostico [{data.get('symbol', '-')}/{tf}]", style="cyan")
        if best:
            _print(
                "Melhor experimento: "
                f"{best.get('experiment_type', '-')}"
                f" | PF={float(best.get('profit_factor', 0.0)):.3f}"
                f" | Sharpe={float(best.get('sharpe', 0.0)):.3f}"
                f" | DD={float(best.get('max_drawdown', 0.0)):.3f}"
                f" | Trades={int(best.get('trades', 0))}",
                style="white",
            )
            if float(best.get("profit_factor", 0.0)) > 1.0:
                _print("Leitura simples: existe sinal promissor para esse TF.", style="green")
            else:
                _print("Leitura simples: sem robustez clara nesse TF; usar paper/diagnostico antes de demo.", style="yellow")
        _print(f"Arquivo resumo: {summary_path}", style="magenta")
        return

    _print("Summary em formato nao reconhecido.", style="yellow")
    _print(f"Arquivo resumo: {summary_path}", style="magenta")


def _extract_summary_approval(summary_path: Path) -> tuple[bool | None, dict[str, Any], dict[str, Any]]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}, {}
    if not isinstance(data, dict):
        return None, {}, {}
    if "approved" in data and isinstance(data.get("result"), dict):
        return bool(data.get("approved", False)), data.get("result", {}), data.get("criteria", {})
    return None, {}, {}


def _latest_robustness_summary(tf: str) -> tuple[Path | None, bool | None]:
    tfu = tf.upper()
    candidates: list[Path] = []
    if tfu == "M5":
        candidates.append(CONFIG.reports_dir / "phase11_M5_summary.json")
    candidates.append(CONFIG.reports_dir / f"phase10_{tfu}_summary.json")
    for p in candidates:
        if p.exists():
            approved, _, _ = _extract_summary_approval(p)
            return p, approved
    return None, None


def _check_live_approval_gate(tf: str, action_label: str) -> bool:
    p, approved = _latest_robustness_summary(tf)
    if p is None:
        _print(
            f"Bloqueado: sem resumo de robustez para {tf}. Rode a opcao 11 antes de {action_label}.",
            style="red",
        )
        return False
    if approved is True:
        return True
    _print(f"Bloqueado: robustez reprovada para {tf}.", style="red")
    _print(f"Resumo usado: {p}", style="yellow")
    _print("Para excelencia, demo/live so libera com approved=true.", style="yellow")
    return False


def _auto_tune_profile_from_summary(tf: str, summary_path: Path) -> dict[str, Any]:
    approved, result, criteria = _extract_summary_approval(summary_path)
    if approved is None:
        return {}
    profile = resolve_timeframe_profile(symbol, tf) or CONFIG.timeframe_profiles.get(tf)
    if profile is None:
        return {}
    updates: dict[str, Any] = {}
    pf_ratio = float(result.get("pf_gt_1_ratio_valid_only", 0.0))
    pf_min = float(criteria.get("pf_gt_1_ratio_valid_only_min", 0.60))
    stress_pf = float(result.get("stress25_pf", 0.0))
    trades_total = int(result.get("trades_total", 0))
    trades_min = int(criteria.get("trades_total_min", 120))
    trade_windows = int(result.get("trade_windows_count", 0))
    trade_windows_min = int(criteria.get("trade_windows_count_min", 6))

    # Regra simples: se há excesso de trades e PF/stress ruim, apertar filtros.
    tighten = (pf_ratio < pf_min and trades_total >= trades_min) or (stress_pf <= 1.0 and trades_total > 500)
    # Regra oposta: se faltam trades/janelas, relaxar filtros.
    loosen = (trades_total < trades_min) or (trade_windows < trade_windows_min)

    if tighten and not loosen:
        updates["signal_threshold"] = round(_clamp(float(profile.signal_threshold) + 0.02, 0.50, 0.75), 3)
        updates["min_signal_margin"] = round(_clamp(float(profile.min_signal_margin) + 0.02, 0.05, 0.35), 3)
        updates["volatility_p_min"] = round(_clamp(float(profile.volatility_p_min) + 5.0, 20.0, 80.0), 1)
        updates["volatility_p_max"] = round(_clamp(float(profile.volatility_p_max) - 3.0, 70.0, 99.0), 1)
        updates["reentry_block_candles"] = int(_clamp(float(profile.reentry_block_candles + 1), 1, 12))
    elif loosen and not tighten:
        updates["signal_threshold"] = round(_clamp(float(profile.signal_threshold) - 0.02, 0.45, 0.75), 3)
        updates["min_signal_margin"] = round(_clamp(float(profile.min_signal_margin) - 0.02, 0.00, 0.35), 3)
        updates["volatility_p_min"] = round(_clamp(float(profile.volatility_p_min) - 5.0, 10.0, 80.0), 1)
        updates["volatility_p_max"] = round(_clamp(float(profile.volatility_p_max) + 3.0, 70.0, 99.0), 1)
        updates["reentry_block_candles"] = int(_clamp(float(profile.reentry_block_candles - 1), 1, 12))
    else:
        # Caso misto: ajuste pequeno apenas em threshold para tentar melhorar robustez sem agressividade.
        updates["signal_threshold"] = round(_clamp(float(profile.signal_threshold) + 0.01, 0.45, 0.75), 3)

    return updates


def _normalize_profile_updates(tf: str, updates: dict[str, Any]) -> dict[str, Any]:
    profile = CONFIG.timeframe_profiles.get(tf)
    if profile is None:
        return {}
    out: dict[str, Any] = {}
    for k, v in updates.items():
        try:
            if k == "signal_threshold":
                out[k] = round(_clamp(float(v), 0.45, 0.80), 3)
            elif k == "min_signal_margin":
                out[k] = round(_clamp(float(v), 0.0, 0.40), 3)
            elif k == "enabled":
                out[k] = bool(v)
            elif k in {"tf_entry", "tf_gate"}:
                out[k] = str(v).upper().strip()
            elif k == "horizon_candles":
                out[k] = int(_clamp(float(v), 1.0, 60.0))
            elif k == "trades_window_mode":
                mode = str(v).strip().lower()
                if mode in {"rolling_60m", "fixed_hour"}:
                    out[k] = mode
            elif k in {"buy_signal_threshold", "sell_signal_threshold"}:
                out[k] = round(_clamp(float(v), 0.45, 0.95), 3)
            elif k in {"buy_min_signal_margin", "sell_min_signal_margin"}:
                out[k] = round(_clamp(float(v), 0.0, 0.60), 3)
            elif k == "volatility_p_min":
                out[k] = round(_clamp(float(v), 10.0, 90.0), 1)
            elif k == "volatility_p_max":
                out[k] = round(_clamp(float(v), 60.0, 99.0), 1)
            elif k in {"volatility_abs_min", "volatility_abs_max", "risk_pct"}:
                out[k] = float(v)
            elif k == "reentry_block_candles":
                out[k] = int(_clamp(float(v), 1.0, 12.0))
            elif k == "max_trades_per_hour":
                out[k] = int(_clamp(float(v), 1.0, 20.0))
            elif k == "min_candles_between_same_direction_trades":
                out[k] = int(_clamp(float(v), 0.0, 20.0))
            elif k == "allow_gate_wait_bypass":
                out[k] = bool(v)
            elif k == "gate_wait_bypass_threshold":
                out[k] = round(_clamp(float(v), 0.45, 0.99), 3)
            elif k == "gate_mode":
                mode = str(v).strip().lower()
                if mode in {"strict", "allow_wait", "bias_only", "off"}:
                    out[k] = mode
            elif k == "gate_min_margin_block":
                out[k] = round(_clamp(float(v), 0.0, 0.50), 3)
            elif k == "impulse_alignment_required":
                out[k] = bool(v)
            elif k == "impulse_lookback_bars":
                out[k] = int(_clamp(float(v), 1.0, 12.0))
            elif k == "impulse_min_abs_return":
                out[k] = round(_clamp(float(v), 0.0, 0.01), 6)
        except Exception:
            continue
    if "volatility_p_min" in out and "volatility_p_max" in out:
        if float(out["volatility_p_min"]) >= float(out["volatility_p_max"]):
            out["volatility_p_max"] = round(min(99.0, float(out["volatility_p_min"]) + 5.0), 1)
    return out


def _auto_tune_profile_with_openai(
    *,
    tf: str,
    summary_path: Path,
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    creds = _load_openai_credentials()
    api_key = str(creds.get("OPENAI_API_KEY", "")).strip()
    model = str(creds.get("OPENAI_MODEL", "gpt-4.1-mini")).strip() or "gpt-4.1-mini"
    if not api_key:
        return {}, "OPENAI_API_KEY ausente"
    approved, result, criteria = _extract_summary_approval(summary_path)
    profile = CONFIG.timeframe_profiles.get(tf)
    if approved is None or profile is None:
        return {}, "summary/perfil invalido"
    profile_payload = profile.model_dump()
    updates, rationale = suggest_profile_updates(
        api_key=api_key,
        model=model,
        tf=tf,
        profile=profile_payload,
        result=result,
        criteria=criteria,
        history=history,
    )
    normalized = _normalize_profile_updates(tf, updates)
    return normalized, rationale


def _print_last_selection(last: dict[str, Any]) -> None:
    if not last:
        return
    action = str(last.get("action", "")).strip()
    action_label = _ACTION_LABELS.get(action, action or "-")
    symbol = str(last.get("symbol", "-"))
    tf = str(last.get("tf", "-"))
    phase = str(last.get("phase", "")).strip()
    run_id = str(last.get("run_id", "-"))
    extra = f" | Fase: {phase}" if phase else ""
    if _CONSOLE is not None and Panel is not None:
        body = (
            f"[white]Acao:[/white] [bold]{action_label}[/bold]\n"
            f"[white]Par:[/white] [cyan]{symbol}[/cyan]  "
            f"[white]TF:[/white] [cyan]{tf}[/cyan]{extra}\n"
            f"[white]Run ID:[/white] [magenta]{run_id}[/magenta]"
        )
        _CONSOLE.print(Panel(body, title="Ultima Selecao", border_style="magenta"))
    else:
        _print(f"Ultima selecao | Acao: {action_label} | Par: {symbol} | TF: {tf}{extra} | Run ID: {run_id}", style="magenta")


_PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)\s*/\s*(\d+)(?:\s+(.*))?$")
_SYMBOL_RE = re.compile(r"--symbol\s+([A-Za-z0-9._-]+)")
_TF_RE = re.compile(r"--tf\s+([A-Za-z0-9._-]+)")


def _parse_progress_line(line: str) -> tuple[int, int, str] | None:
    m = _PROGRESS_RE.match(line.strip())
    if not m:
        return None
    cur = int(m.group(1))
    total = int(m.group(2))
    msg = (m.group(3) or "").strip()
    if total <= 0:
        return None
    return cur, total, msg


def _clean_live_line(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    if text.startswith("PROGRESS"):
        return ""
    if text.startswith("LIVE "):
        return text[5:].strip()
    return text


def _fmt_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def _resolve_module_name(module: str) -> str:
    """
    Resolve module path to work when menu is executed as:
    - python -m mt5_ai_bot.src.cli_menu (project root)
    - python -m src.cli_menu (inside mt5_ai_bot/)
    """
    def _has_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    if _has_spec(module):
        return module
    if module.startswith("mt5_ai_bot."):
        alt = module.replace("mt5_ai_bot.", "", 1)
        if _has_spec(alt):
            return alt
    if module.startswith("src."):
        alt = f"mt5_ai_bot.{module}"
        if _has_spec(alt):
            return alt
    return module


def _run_module(
    module: str,
    args: list[str],
    run_dir: Path,
    *,
    allow_quit_with_q: bool = False,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_module = _resolve_module_name(module)
    cmd = [sys.executable, "-m", resolved_module, *args]
    started_at = _now_utc().isoformat()
    env = _build_subprocess_env()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    saw_progress = False
    progress_total = 100
    progress_completed = 0
    cancelled_by_user = False
    if (
        _CONSOLE is not None
        and Progress is not None
        and Live is not None
        and Panel is not None
        and Group is not None
        and proc.stdout is not None
    ):
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=40),
            TimeElapsedColumn(),
            console=_CONSOLE,
            transient=False,
        )
        task = progress.add_task(f"Executando {module}", total=progress_total, completed=0)
        live_lines: deque[str] = deque(maxlen=10)
        start_ts = time.time()
        last_output_ts = start_ts
        line_queue: Queue[Any] = Queue()
        eof = object()

        def _reader() -> None:
            try:
                for raw in proc.stdout:  # type: ignore[union-attr]
                    line_queue.put(raw.rstrip("\r\n"))
            finally:
                line_queue.put(eof)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        def _render() -> Any:
            body = "\n".join(live_lines) if live_lines else "Aguardando atualizacoes..."
            elapsed = _fmt_elapsed(time.time() - start_ts)
            silent = _fmt_elapsed(time.time() - last_output_ts)
            status = f"[dim]Status: rodando | Tempo: {elapsed} | Sem novo log: {silent}[/dim]"
            panel = Panel(f"{body}\n\n{status}", title="Execucao em tempo real", border_style="bright_blue")
            return Group(progress, panel)

        with Live(_render(), console=_CONSOLE, refresh_per_second=5, transient=True) as live:
            reader_done = False
            while True:
                if allow_quit_with_q and msvcrt is not None and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if str(ch).lower() == "q":
                        cancelled_by_user = True
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        live_lines.append("Cancelado pelo usuario (Q).")
                        live.update(_render())
                        break
                try:
                    item = line_queue.get(timeout=0.5)
                except Empty:
                    live.update(_render())
                    if reader_done and proc.poll() is not None:
                        break
                    continue
                if item is eof:
                    reader_done = True
                    live.update(_render())
                    if proc.poll() is not None:
                        break
                    continue

                line = str(item)
                stdout_lines.append(line)
                last_output_ts = time.time()
                parsed = _parse_progress_line(line)
                if parsed is not None:
                    saw_progress = True
                    cur, total, msg = parsed
                    progress_total = max(1, total)
                    progress_completed = max(0, min(cur, progress_total))
                    progress.update(
                        task,
                        total=progress_total,
                        completed=progress_completed,
                        description=(msg or f"Executando {module}"),
                    )
                    live.update(_render())
                    continue
                clean = _clean_live_line(line)
                if clean:
                    live_lines.append(clean)
                    live.update(_render())
            proc.wait()
            if saw_progress:
                progress.update(task, total=progress_total, completed=progress_total, description=f"{module} concluido")
            else:
                progress.update(task, total=100, completed=100, description=f"{module} concluido")
            live_lines.append("Concluido.")
            live.update(_render())
    else:
        stdout, _ = proc.communicate()
        stdout_lines = (stdout or "").splitlines()

    stdout = "\n".join(stdout_lines)
    stderr = ""
    (logs_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (logs_dir / "stderr.log").write_text(stderr, encoding="utf-8")

    saved_paths: list[str] = []
    for line in stdout.splitlines():
        if "saved_" in line and "=" in line:
            saved_paths.append(line.split("=", 1)[1].strip())
    copied_outputs: list[str] = []
    for p in saved_paths:
        src = Path(p)
        if src.exists() and src.is_file():
            dst = out_dir / src.name
            try:
                shutil.copy2(src, dst)
                copied_outputs.append(str(dst))
            except Exception:
                copied_outputs.append(str(src))

    meta = {
        "module": module,
        "resolved_module": resolved_module,
        "args": args,
        "cmd": cmd,
        "started_at": started_at,
        "finished_at": _now_utc().isoformat(),
        "return_code": int(proc.returncode or 0),
        "cancelled_by_user": bool(cancelled_by_user),
        "saved_paths": saved_paths,
        "copied_outputs": copied_outputs,
    }
    _save_json(run_dir / "run_meta.json", meta)
    return meta


def _extract_flag_value(cmdline: str, pattern: re.Pattern[str], default: str = "-") -> str:
    m = pattern.search(cmdline or "")
    return m.group(1).upper() if m else default


def _infer_mode_from_cmd(cmdline: str) -> str:
    c = (cmdline or "").lower()
    if "--diagnostic-only" in c:
        return "diagnostico"
    if "--paper" in c:
        return "paper"
    if "--no-trade" in c:
        return "sem-ordem"
    return "demo/live"


def _running_bot_processes() -> list[dict[str, Any]]:
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { "
        "$_.CommandLine -like '*mt5_ai_bot.src.bot_live*' -and "
        "$_.Name -match 'python' -and "
        "$_.CommandLine -like '* -m mt5_ai_bot.src.bot_live*' "
        "} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return []
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    rows = data if isinstance(data, list) else [data]
    result: list[dict[str, Any]] = []
    for r in rows:
        pid = int(r.get("ProcessId", 0) or 0)
        cmdline = str(r.get("CommandLine", "") or "")
        name = str(r.get("Name", "") or "")
        if pid <= 0:
            continue
        # safety: ignore monitor/query shells even if command line matches
        if "powershell" in name.lower() or "cmd.exe" in name.lower():
            continue
        result.append(
            {
                "pid": pid,
                "name": name,
                "cmdline": cmdline,
                "symbol": _extract_flag_value(cmdline, _SYMBOL_RE, default="-"),
                "tf": _extract_flag_value(cmdline, _TF_RE, default="-"),
                "mode": _infer_mode_from_cmd(cmdline),
            }
        )
    return result


def _running_meta_by_pid() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in (RUNS_DIR).glob("*/run_meta.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = int(data.get("pid", 0) or 0)
        if pid <= 0:
            continue
        out[pid] = {
            "run_id": p.parent.parent.name,
            "stdout_log": str(data.get("stdout_log", "")),
            "stderr_log": str(data.get("stderr_log", "")),
            "started_at": str(data.get("started_at", "")),
        }
    return out


def _tail_text_file(path: Path, n: int = 6) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return [ln for ln in lines[-n:] if ln.strip()]


def _spawn_module(module: str, args: list[str], run_dir: Path) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_module = _resolve_module_name(module)
    cmd = [sys.executable, "-m", resolved_module, *args]
    started_at = _now_utc().isoformat()
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    stdout_f = stdout_path.open("a", encoding="utf-8")
    stderr_f = stderr_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        text=True,
        cwd=str(Path.cwd()),
        env=_build_subprocess_env(),
    )
    meta = {
        "module": module,
        "resolved_module": resolved_module,
        "args": args,
        "cmd": cmd,
        "started_at": started_at,
        "pid": proc.pid,
        "status": "running",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _save_json(run_dir / "run_meta.json", meta)
    return meta


def _print_models() -> None:
    models = list_models()
    if not models:
        _print("Nenhum modelo registrado.", style="yellow")
        return
    if Table is not None and _CONSOLE is not None:
        tb = Table(title="Modelos Treinados", show_lines=False)
        tb.add_column("Symbol/TF", style="cyan")
        tb.add_column("VersÃ£o", style="magenta")
        tb.add_column("Treinado em", style="green")
        tb.add_column("Acc", style="yellow")
        tb.add_column("Caminho", style="white")
        for m in models:
            tb.add_row(
                f"{m.get('symbol')}/{m.get('timeframe')}",
                str(m.get("version")),
                str(m.get("trained_at")),
                str(m.get("metrics_summary", {}).get("class_accuracy")),
                str(m.get("model_path")),
            )
        _CONSOLE.print(tb)
    else:
        _print("Modelos treinados:", style="cyan")
        for m in models:
            _print(
                f"- {m.get('symbol')}/{m.get('timeframe')} v={m.get('version')} "
                f"trained_at={m.get('trained_at')} "
                f"acc={m.get('metrics_summary', {}).get('class_accuracy')} "
                f"path={m.get('model_path')}"
            )


def _action_train() -> None:
    symbol = _choose_symbol("EURUSD")
    tf = _ask("Timeframe", "M5", to_upper=True)
    splits = _ask("Splits PurgedKFold", "5")
    seed = _ask("Seed", "42")
    if not _validate_symbol(symbol):
        _print(f"Simbolo invalido/inacessivel no MT5: {symbol}", style="red")
        return
    run_id = _run_id(symbol, tf, "train")
    run_dir = RUNS_DIR / run_id
    raw_path = CONFIG.data_raw_dir / f"{symbol}_{tf}.parquet"
    feat_path = CONFIG.data_processed_dir / f"{symbol}_{tf}_features.parquet"
    dataset_path = CONFIG.data_processed_dir / f"{symbol}_{tf}_dataset.parquet"

    def _rows_or_none(path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:
            return None

    raw_rows = _rows_or_none(raw_path)
    feat_rows = _rows_or_none(feat_path)
    ds_rows = _rows_or_none(dataset_path)
    inconsistent = (
        (dataset_path.exists() and (ds_rows is None or ds_rows == 0))
        or (feat_path.exists() and (feat_rows is None or feat_rows == 0))
        or (raw_path.exists() and (raw_rows is None or raw_rows == 0))
        or (dataset_path.exists() and not feat_path.exists())
    )

    if inconsistent:
        _print("Detectada inconsistência nos artefatos (raw/features/dataset vazio ou inválido).", style="yellow")
        if _ask_yes_no("Deseja apagar artefatos do símbolo/TF e reconstruir?", default=True):
            for p in (dataset_path, feat_path, raw_path):
                if p.exists():
                    try:
                        p.unlink()
                        _print(f"Removido: {p}", style="cyan")
                    except Exception as exc:
                        _print(f"Falha ao remover {p}: {exc}", style="red")
        else:
            _print("Treino cancelado por inconsistência.", style="yellow")
            _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_inconsistent_cancel", "run_id": run_id})
            return

    if not dataset_path.exists():
        _print(f"Dataset nao encontrado: {dataset_path}", style="yellow")
        if _ask_yes_no("Deseja gerar dataset automaticamente agora?", default=True):
            months = _ask("Meses de historico para coleta", "24")
            source = _ask("Fonte de dados (auto/mt5/yahoo)", "auto").strip().lower()
            if source not in {"auto", "mt5", "yahoo"}:
                source = "auto"
            _print(f"Passo 1/2: coletando dados ({source})...", style="cyan")
            meta_feed = _run_module(
                module="mt5_ai_bot.src.data_feed",
                args=["--symbol", symbol, "--tfs", tf, "--months", months, "--source", source],
                run_dir=run_dir,
            )
            if meta_feed["return_code"] != 0:
                _print("Falha na coleta. Veja stderr em run_meta/logs.", style="red")
                _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_prepare_fail", "run_id": run_id})
                return
            _print("Passo 2/2: construindo features + dataset...", style="cyan")
            meta_ds = _run_module(
                module="mt5_ai_bot.src.build_dataset",
                args=["--symbol", symbol, "--tf", tf],
                run_dir=run_dir,
            )
            if meta_ds["return_code"] != 0:
                _print("Falha ao gerar dataset. Veja stderr em run_meta/logs.", style="red")
                _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_prepare_fail", "run_id": run_id})
                return
            if raw_path.exists():
                raw_rows = len(pd.read_parquet(raw_path))
                if raw_rows == 0:
                    _print("Coleta retornou 0 candles para esse simbolo/TF.", style="red")
                    _print(
                        "Verifique no MT5 se o simbolo existe nesse broker (ex: XAUUSD, XAUUSDm, GOLD) e se ha historico carregado.",
                        style="yellow",
                    )
                    _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_prepare_empty_raw", "run_id": run_id})
                    return
            if feat_path.exists():
                feat_rows = len(pd.read_parquet(feat_path))
                if feat_rows == 0:
                    _print("Features ficaram vazias (0 linhas).", style="red")
                    _print("Provavel falta de historico suficiente para os indicadores.", style="yellow")
                    _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_prepare_empty_features", "run_id": run_id})
                    return
        else:
            _print("Treino cancelado: dataset ausente.", style="yellow")
            return

    if dataset_path.exists():
        ds_rows = len(pd.read_parquet(dataset_path))
        if ds_rows == 0:
            _print("Dataset vazio (0 linhas). Treino abortado.", style="red")
            _print("Isso acontece quando nao ha historico util para gerar labels/triple barrier.", style="yellow")
            _save_last_selection({"symbol": symbol, "tf": tf, "action": "train_empty_dataset", "run_id": run_id})
            return

    _print("Executando treino...", style="cyan")
    meta = _run_module(
        module="mt5_ai_bot.src.train_lgbm",
        args=["--symbol", symbol, "--tf", tf, "--splits", splits, "--seed", seed],
        run_dir=run_dir,
    )
    _save_last_selection({"symbol": symbol, "tf": tf, "action": "train", "run_id": run_id})
    _print(f"Run: {run_id}", style="green")
    _print(f"Return code: {meta['return_code']}", style="yellow")
    _print_run_result(run_dir, meta)


def _action_phase() -> None:
    symbol = _choose_symbol("EURUSD")
    phase = _ask("Fase (4-11)", "11")
    tf = _ask("TF entrada/perfil", "M5", to_upper=True)
    resolved = resolve_timeframe_profile(symbol, tf) or CONFIG.timeframe_profiles.get(tf)
    gate = _ask("TF gate", str(resolved.tf_gate if resolved else "M30"), to_upper=True)
    windows = _ask("Janelas walk-forward", "8")
    seed = _ask("Seed", "42")
    if phase not in {"4", "5", "6", "7", "8", "9", "10", "11"}:
        _print("Fase invalida.", style="red")
        return
    run_id = _run_id(symbol, tf, f"phase{phase}")
    run_dir = RUNS_DIR / run_id
    module = f"mt5_ai_bot.src.phase{phase}_runner"
    args: list[str] = ["--symbol", symbol]
    if phase in {"8", "11"}:
        args += ["--windows", windows, "--seed", seed]
    elif phase == "10":
        args += ["--tf_entry", tf, "--tf_gate", gate, "--windows", windows, "--seed", seed]
    elif phase == "9":
        args += ["--seed", seed]
    meta = _run_module(module=module, args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": tf, "phase": phase, "run_id": run_id})
    _print(f"Run: {run_id}", style="green")
    _print(f"Return code: {meta['return_code']}", style="yellow")
    _print_run_result(run_dir, meta)
    if int(meta.get("return_code", 1)) == 0:
        default_summary = f"phase10_{tf}_summary.json" if phase == "10" else (f"phase11_{tf}_summary.json" if phase == "11" else f"phase{phase}_{tf}_summary.json")
        summary_path = _find_summary_path(meta, default_summary)
        if summary_path is not None:
            _print_phase_summary(summary_path)
        else:
            _print("Resumo nao encontrado, mas os arquivos foram salvos em reports/.", style="yellow")


def _build_bot_args_submenu(
    *,
    paper: bool = False,
    diagnostic: bool = False,
    force_trade: bool = False,
) -> tuple[str, str, list[str]]:
    symbol = _choose_symbol("EURUSD")
    tf = _ask("TF entrada", "M5", to_upper=True)
    model_symbol = _choose_symbol(symbol)
    model_tf = _ask("Model tf", tf, to_upper=True)
    model_version = _ask("Model version (vazio=latest)", "")
    once = _ask_yes_no("Rodar apenas um ciclo (--once)", default=False)
    no_trade = _ask_yes_no("Modo sem ordem (--no-trade)", default=False)
    if force_trade:
        no_trade = False
    diag_only = diagnostic or _ask_yes_no("Modo diagnostico (--diagnostic-only)", default=False)
    args = [
        "--symbol",
        symbol,
        "--tf",
        tf,
        "--model-symbol",
        model_symbol,
        "--model-tf",
        model_tf,
    ]
    if once:
        args.append("--once")
    if no_trade:
        args.append("--no-trade")
    if model_version:
        args += ["--model-version", model_version, "--no-use-latest-model"]
    else:
        args += ["--use-latest-model"]
    if paper:
        args += ["--paper", "--paper-bars", "800"]
    if diag_only:
        out_name = _ask("Nome base do diagnostico (--out-name)", "diag_menu")
        args += ["--diagnostic-only", "--out", str(CONFIG.reports_dir), "--out-name", out_name]
    return symbol, tf, args


def _action_bot(paper: bool, diagnostic: bool = False) -> None:
    symbol, tf, args = _build_bot_args_submenu(paper=paper, diagnostic=diagnostic, force_trade=False)
    if (not paper) and (not diagnostic):
        if not _check_live_approval_gate(tf=tf, action_label="demo-trade"):
            return
    run_id = _run_id(symbol, tf, "bot_paper" if paper else ("bot_diag" if diagnostic else "bot_demo"))
    run_dir = RUNS_DIR / run_id
    meta = _run_module(module="mt5_ai_bot.src.bot_live", args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": tf, "action": "bot", "run_id": run_id})
    _print(f"Run: {run_id}", style="green")
    _print(f"Return code: {meta['return_code']}", style="yellow")
    _print_run_result(run_dir, meta)


def _action_start_trade_background() -> None:
    symbol, tf, args = _build_bot_args_submenu(paper=False, diagnostic=False, force_trade=True)
    if not _check_live_approval_gate(tf=tf, action_label="iniciar trade"):
        return
    # Iniciar trade: garante sem --once e sem --no-trade.
    args = [a for a in args if a != "--once" and a != "--no-trade"]
    run_id = _run_id(symbol, tf, "trade_live")
    run_dir = RUNS_DIR / run_id
    meta = _spawn_module(module="mt5_ai_bot.src.bot_live", args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": tf, "action": "trade_live", "run_id": run_id})
    _print(f"Trade iniciado em background. PID={meta['pid']}", style="green")
    _print(f"Run: {run_id}", style="yellow")
    _print(f"Logs: {meta['stdout_log']}", style="cyan")
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")


def _has_latest_model(symbol: str, tf: str) -> bool:
    try:
        get_latest_model(symbol=symbol, timeframe=tf)
        return True
    except Exception:
        return False


def _apply_scalping_preset_m1() -> dict[str, Any]:
    updates = {
        "enabled": True,
        "tf_entry": "M1",
        "tf_gate": "M5",
        "horizon_candles": 20,
        "max_trades_per_hour": 14,
        "trades_window_mode": "rolling_60m",
        "min_candles_between_same_direction_trades": 0,
        "reentry_block_candles": 0,
        "volatility_threshold_min_mode": "atr_percentile",
        "volatility_threshold_max_mode": "atr_percentile",
        "volatility_p_min": 25.0,
        "volatility_p_max": 99.0,
        "risk_pct": 0.25,
        "signal_threshold": 0.50,
        "min_signal_margin": 0.04,
        "allow_gate_wait_bypass": True,
        "gate_wait_bypass_threshold": 0.74,
        "regime_thresholds": {
            "LOW_VOL_SIDEWAYS": {"signal_threshold": 0.54, "min_signal_margin": 0.08},
            "TRENDING_STRONG": {"signal_threshold": 0.46, "min_signal_margin": 0.03},
            "HIGH_VOL_BREAKOUT": {"signal_threshold": 0.50, "min_signal_margin": 0.06},
            "POST_NEWS_SHOCK": {"signal_threshold": 0.58, "min_signal_margin": 0.10},
            "RANGING": {"signal_threshold": 0.50, "min_signal_margin": 0.05},
        },
    }
    profile = CONFIG.timeframe_profiles.get("M1")
    if profile is not None:
        CONFIG.timeframe_profiles["M1"] = profile.model_copy(update=updates)
    _save_tf_override(tf="M1", updates=updates, reason="menu_scalping_preset_m1")
    return updates


def _active_intraday_preset(tf: str) -> dict[str, Any]:
    tf = str(tf).upper().strip()
    if tf == "M1":
        return {
            "enabled": True,
            "tf_entry": "M1",
            "tf_gate": "M5",
            "gate_mode": "allow_wait",
            "gate_min_margin_block": 0.14,
            "horizon_candles": 10,
            "max_trades_per_hour": 12,
            "trades_window_mode": "rolling_60m",
            "min_candles_between_same_direction_trades": 1,
            "reentry_block_candles": 1,
            "volatility_p_min": 20.0,
            "volatility_p_max": 98.0,
            "risk_pct": 0.20,
            "signal_threshold": 0.50,
            "min_signal_margin": 0.03,
            "allow_gate_wait_bypass": True,
            "gate_wait_bypass_threshold": 0.68,
            "impulse_alignment_required": True,
            "impulse_lookback_bars": 3,
            "impulse_min_abs_return": 0.00003,
        }
    return {
        "enabled": True,
        "tf_entry": "M5",
        "tf_gate": "M15",
        "gate_mode": "bias_only",
        "gate_min_margin_block": 0.12,
        "horizon_candles": 6,
        "max_trades_per_hour": 4,
        "trades_window_mode": "rolling_60m",
        "min_candles_between_same_direction_trades": 1,
        "reentry_block_candles": 1,
        "volatility_p_min": 25.0,
        "volatility_p_max": 95.0,
        "risk_pct": 0.35,
        "signal_threshold": 0.53,
        "min_signal_margin": 0.06,
        "allow_gate_wait_bypass": False,
        "gate_wait_bypass_threshold": 0.80,
        "impulse_alignment_required": True,
        "impulse_lookback_bars": 3,
        "impulse_min_abs_return": 0.00005,
    }


def _apply_active_intraday_preset(symbol: str, tf: str) -> dict[str, Any]:
    updates = _active_intraday_preset(tf)
    item = {
        "symbol": str(symbol).upper().strip(),
        "tf": str(tf).upper().strip(),
        "gate": str(updates["tf_gate"]).upper().strip(),
        "use_session_filter": False,
        "allowed_sessions_utc": list(CONFIG.live.allowed_sessions_utc),
        "break_even_enabled": True,
        "break_even_r": 0.4 if str(tf).upper().strip() == "M1" else 0.5,
        "trailing_enabled": True,
        "trailing_activation_r": 0.8 if str(tf).upper().strip() == "M1" else 1.0,
        "trailing_atr_mult": 0.8 if str(tf).upper().strip() == "M1" else 1.0,
        "max_holding_candles_override": int(updates["horizon_candles"]),
        "profile_updates": _normalize_profile_updates(str(tf).upper().strip(), updates),
        "updated_at": _now_utc().isoformat(),
    }
    _upsert_symbol_profile(item)
    return updates


def _action_active_intraday_menu() -> None:
    _print("Modo intraday ativo: mais frequencia, gate menos duro e impulso curto alinhado.", style="cyan")
    symbol = _choose_symbol("EURUSD")
    tf = _ask("Preset [M1/M5]", "M5", to_upper=True)
    if tf not in {"M1", "M5"}:
        _print("Preset invalido. Use M1 ou M5.", style="red")
        return
    if _ask_yes_no(f"Aplicar preset intraday ativo {tf} agora?", default=True):
        updates = _apply_active_intraday_preset(symbol, tf)
        _print(f"Preset ativo {tf} aplicado:", style="green")
        for k in (
            "tf_gate",
            "gate_mode",
            "max_trades_per_hour",
            "signal_threshold",
            "min_signal_margin",
            "impulse_alignment_required",
            "impulse_lookback_bars",
            "impulse_min_abs_return",
            "risk_pct",
        ):
            _print(f"- {k} = {updates[k]}", style="cyan")
        _print(f"Perfil salvo em: {SYMBOL_PROFILES}", style="magenta")
        _save_last_selection({"symbol": symbol, "tf": tf, "action": "intraday_active_preset", "run_id": "-"})

    gate_tf = str(_active_intraday_preset(tf)["tf_gate"])
    has_entry = _has_latest_model(symbol, tf)
    has_gate = _has_latest_model(symbol, gate_tf)
    _print(f"Modelo {symbol}/{tf}: {'OK' if has_entry else 'AUSENTE'}", style=("green" if has_entry else "yellow"))
    _print(f"Modelo {symbol}/{gate_tf} (gate): {'OK' if has_gate else 'AUSENTE'}", style=("green" if has_gate else "yellow"))
    if not (has_entry and has_gate):
        _print(f"Antes de operar no modo ativo, treine os modelos {tf} e {gate_tf} para esse simbolo.", style="yellow")
        return

    mode = _ask("Iniciar agora? [0=nao,1=paper,2=demo]", "1").strip()
    if mode not in {"1", "2"}:
        _print("Preset ativo configurado. Nao iniciado.", style="yellow")
        return

    run_tag = f"intraday_{tf.lower()}_paper_bg" if mode == "1" else f"intraday_{tf.lower()}_demo_bg"
    args = [
        "--symbol", symbol,
        "--tf", tf,
        "--model-symbol", symbol,
        "--model-tf", tf,
        "--use-latest-model",
    ]
    if mode == "1":
        args += ["--paper", "--paper-bars", "1600"]
    else:
        if not _check_live_approval_gate(tf=tf, action_label=f"demo intraday {tf}"):
            if not _ask_yes_no(f"Robustez {tf} nao aprovada. Iniciar demo mesmo assim (por sua conta)?", default=False):
                return
    run_id = _run_id(symbol, tf, run_tag)
    run_dir = RUNS_DIR / run_id
    meta = _spawn_module(module="mt5_ai_bot.src.bot_live", args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": tf, "action": "intraday_active_start", "run_id": run_id})
    _print(f"Intraday ativo iniciado em background. PID={meta['pid']}", style="green")
    _print(f"Run: {run_id}", style="yellow")
    _print(f"Logs: {meta['stdout_log']}", style="cyan")
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")


def _action_scalping_menu() -> None:
    _print("Modo scalping (M1): mais frequencia com controle de risco.", style="cyan")
    symbol = _choose_symbol("EURUSD")
    if _ask_yes_no("Aplicar preset scalping M1 agora?", default=True):
        updates = _apply_scalping_preset_m1()
        _print("Preset M1 aplicado:", style="green")
        for k in (
            "tf_gate",
            "max_trades_per_hour",
            "min_candles_between_same_direction_trades",
            "reentry_block_candles",
            "signal_threshold",
            "min_signal_margin",
            "risk_pct",
        ):
            _print(f"- {k} = {updates[k]}", style="cyan")
        _print(f"Override salvo em: {TF_OVERRIDES_FILE}", style="magenta")
        _save_last_selection({"symbol": symbol, "tf": "M1", "action": "scalp_preset", "run_id": "-"})

    has_m1 = _has_latest_model(symbol, "M1")
    has_m5 = _has_latest_model(symbol, "M5")
    _print(f"Modelo {symbol}/M1: {'OK' if has_m1 else 'AUSENTE'}", style=("green" if has_m1 else "yellow"))
    _print(f"Modelo {symbol}/M5 (gate): {'OK' if has_m5 else 'AUSENTE'}", style=("green" if has_m5 else "yellow"))
    if not (has_m1 and has_m5):
        _print("Antes de operar scalping, treine os modelos M1 e M5 para esse simbolo.", style="yellow")
        if _ask_yes_no("Deseja abrir treino agora? (opcao 2 apos voltar)", default=True):
            return

    mode = _ask("Iniciar agora? [0=nao,1=paper,2=demo]", "1").strip()
    if mode not in {"1", "2"}:
        _print("Scalping configurado. Nao iniciado.", style="yellow")
        return

    run_tag = "scalp_m1_paper_bg" if mode == "1" else "scalp_m1_demo_bg"
    args = [
        "--symbol",
        symbol,
        "--tf",
        "M1",
        "--model-symbol",
        symbol,
        "--model-tf",
        "M1",
        "--use-latest-model",
    ]
    if mode == "1":
        args += ["--paper", "--paper-bars", "1200"]
    else:
        if not _check_live_approval_gate(tf="M1", action_label="demo scalping"):
            if not _ask_yes_no("Robustez M1 nao aprovada. Iniciar demo mesmo assim (por sua conta)?", default=False):
                return

    run_id = _run_id(symbol, "M1", run_tag)
    run_dir = RUNS_DIR / run_id
    meta = _spawn_module(module="mt5_ai_bot.src.bot_live", args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": "M1", "action": "scalp_start", "run_id": run_id})
    _print(f"Scalping iniciado em background. PID={meta['pid']}", style="green")
    _print(f"Run: {run_id}", style="yellow")
    _print(f"Logs: {meta['stdout_log']}", style="cyan")
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")


def _action_profiles() -> None:
    data = _load_json(SYMBOL_PROFILES, {"profiles": []})
    _print("Configurar perfil por simbolo/TF (gate, sessao, saida e overrides locais).", style="cyan")
    symbol = _choose_symbol("EURUSD")
    tf = _ask("TF", "M5", to_upper=True)
    current = get_symbol_profile_entry(symbol, tf)
    base_profile = resolve_timeframe_profile(symbol, tf) or CONFIG.timeframe_profiles.get(tf)
    if base_profile is None:
        _print(f"Perfil base ausente para {symbol}/{tf}.", style="red")
        return
    gate = _ask("Gate", str(current.get("gate") or base_profile.tf_gate), to_upper=True)
    gate_mode = _ask(
        "Gate mode [strict/allow_wait/bias_only/off]",
        str(current.get("profile_updates", {}).get("gate_mode", base_profile.gate_mode)),
    ).strip().lower()
    gate_min_margin_block = float(
        _ask(
            "Gate min margin p/ bloquear (bias_only)",
            str(current.get("profile_updates", {}).get("gate_min_margin_block", base_profile.gate_min_margin_block)),
        )
    )
    use_session_filter = _ask_yes_no(
        "Usar filtro de sessao para este simbolo/TF?",
        default=bool(current.get("use_session_filter", CONFIG.live.use_session_filter)),
    )
    sess_default = ",".join(current.get("allowed_sessions_utc", CONFIG.live.allowed_sessions_utc))
    sessions_raw = _ask("Janelas UTC (ex: 06:00-17:00,12:00-16:00)", sess_default)
    sessions = [s.strip() for s in sessions_raw.split(",") if s.strip()]
    break_even_enabled = _ask_yes_no(
        "Ativar break-even neste perfil?",
        default=bool(current.get("break_even_enabled", False)),
    )
    break_even_r = float(_ask("Break-even em quantos R", str(current.get("break_even_r", 0.5))))
    trailing_enabled = _ask_yes_no(
        "Ativar trailing ATR em high-vol?",
        default=bool(current.get("trailing_enabled", False)),
    )
    trailing_activation_r = float(_ask("Trailing ativa a partir de quantos R", str(current.get("trailing_activation_r", 1.0))))
    trailing_atr_mult = float(_ask("Trailing ATR multiplicador", str(current.get("trailing_atr_mult", 1.0))))
    max_holding_override = int(_ask("Time stop override (candles, 0=usar padrao)", str(current.get("max_holding_candles_override", 0))))
    impulse_alignment_required = _ask_yes_no(
        "Exigir impulso curto alinhado?",
        default=bool(current.get("profile_updates", {}).get("impulse_alignment_required", base_profile.impulse_alignment_required)),
    )
    impulse_lookback_bars = int(
        _ask(
            "Impulso: lookback em barras",
            str(current.get("profile_updates", {}).get("impulse_lookback_bars", base_profile.impulse_lookback_bars)),
        )
    )
    impulse_min_abs_return = float(
        _ask(
            "Impulso: retorno minimo absoluto",
            str(current.get("profile_updates", {}).get("impulse_min_abs_return", base_profile.impulse_min_abs_return)),
        )
    )
    _print("Overrides de precisao (ENTER para manter base):", style="cyan")
    buy_thr_raw = _ask("BUY threshold", str(current.get("profile_updates", {}).get("buy_signal_threshold", "")))
    sell_thr_raw = _ask("SELL threshold", str(current.get("profile_updates", {}).get("sell_signal_threshold", "")))
    buy_margin_raw = _ask("BUY min margin", str(current.get("profile_updates", {}).get("buy_min_signal_margin", "")))
    sell_margin_raw = _ask("SELL min margin", str(current.get("profile_updates", {}).get("sell_min_signal_margin", "")))
    profile_updates: dict[str, Any] = {}
    if gate_mode:
        profile_updates["gate_mode"] = gate_mode
    profile_updates["gate_min_margin_block"] = gate_min_margin_block
    profile_updates["impulse_alignment_required"] = impulse_alignment_required
    profile_updates["impulse_lookback_bars"] = impulse_lookback_bars
    profile_updates["impulse_min_abs_return"] = impulse_min_abs_return
    for key, raw in (
        ("buy_signal_threshold", buy_thr_raw),
        ("sell_signal_threshold", sell_thr_raw),
        ("buy_min_signal_margin", buy_margin_raw),
        ("sell_min_signal_margin", sell_margin_raw),
    ):
        if str(raw).strip():
            try:
                profile_updates[key] = float(raw)
            except Exception:
                pass
    profile_updates = _normalize_profile_updates(tf, profile_updates)
    item = {
        "symbol": symbol,
        "tf": tf,
        "gate": gate,
        "use_session_filter": use_session_filter,
        "allowed_sessions_utc": sessions,
        "break_even_enabled": break_even_enabled,
        "break_even_r": break_even_r,
        "trailing_enabled": trailing_enabled,
        "trailing_activation_r": trailing_activation_r,
        "trailing_atr_mult": trailing_atr_mult,
        "max_holding_candles_override": max_holding_override,
        "profile_updates": profile_updates,
        "updated_at": _now_utc().isoformat(),
    }
    profiles = [p for p in data.get("profiles", []) if not (p.get("symbol") == symbol and p.get("tf") == tf)]
    profiles.append(item)
    data["profiles"] = profiles
    _save_json(SYMBOL_PROFILES, data)
    _print(f"Perfil salvo em {SYMBOL_PROFILES}", style="green")


def _action_mt5_credentials() -> None:
    current = _load_mt5_credentials()
    _print("Configurar credenciais MT5 (salvas em reports/runs/mt5_credentials.json)", style="cyan")
    login = _ask("MT5_LOGIN", current.get("MT5_LOGIN", ""))
    password = _ask("MT5_PASSWORD", current.get("MT5_PASSWORD", ""))
    server = _ask("MT5_SERVER", current.get("MT5_SERVER", ""))
    path = _ask("MT5_PATH (opcional)", current.get("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe"))
    payload = {
        "MT5_LOGIN": login,
        "MT5_PASSWORD": password,
        "MT5_SERVER": server,
        "MT5_PATH": path,
        "updated_at": _now_utc().isoformat(),
    }
    _save_json(MT5_CREDS_FILE, payload)
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        if payload.get(k):
            os.environ[k] = str(payload[k])
    ok = ensure_logged_in()
    _print(f"Credenciais salvas em: {MT5_CREDS_FILE}", style="green")
    _print(f"Teste de login MT5: {'OK' if ok else 'FALHOU'}", style=("green" if ok else "red"))
    if not ok:
        _print("Revise login/senha/servidor e MT5 aberto.", style="yellow")


def _mask_secret(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}...{s[-4:]}"


def _action_openai_credentials() -> None:
    current = _load_openai_credentials()
    _print("Configurar OpenAI (salvo em reports/runs/openai_credentials.json)", style="cyan")
    cur_key = current.get("OPENAI_API_KEY", "")
    if cur_key:
        _print(f"OPENAI_API_KEY atual: {_mask_secret(cur_key)}", style="dim")
    _print("Dica: ENTER vazio mantém valor atual. Digite APAGAR para limpar.", style="dim")
    key_raw = _ask("OPENAI_API_KEY", "")
    if key_raw.strip().upper() == "APAGAR":
        key = ""
    elif key_raw.strip():
        key = key_raw.strip()
    else:
        key = cur_key

    model_default = current.get("OPENAI_MODEL", "gpt-4.1-mini")
    model = _ask("OPENAI_MODEL", model_default).strip() or "gpt-4.1-mini"

    payload = {
        "OPENAI_API_KEY": key,
        "OPENAI_MODEL": model,
        "updated_at": _now_utc().isoformat(),
    }
    _save_json(OPENAI_CREDS_FILE, payload)
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    os.environ["OPENAI_MODEL"] = model
    _print(f"Credenciais OpenAI salvas em: {OPENAI_CREDS_FILE}", style="green")
    _print(f"Chave ativa: {'SIM' if bool(key) else 'NAO'} | Modelo: {model}", style="cyan")


def _positions_table(symbol_filter: str | None = None) -> tuple[Any, int, float]:
    if mt5 is None:
        return None, 0, 0.0
    if not ensure_logged_in():
        return None, 0, 0.0
    pos = mt5.positions_get(symbol=symbol_filter) if symbol_filter else mt5.positions_get()
    positions = [] if pos is None else list(pos)
    total_profit = 0.0
    if Table is None:
        return positions, len(positions), float(sum(float(getattr(p, "profit", 0.0) or 0.0) for p in positions))
    tb = Table(title="Trades Ativos (tempo real)", show_lines=False)
    tb.add_column("Ticket", style="cyan")
    tb.add_column("Simbolo", style="white")
    tb.add_column("Tipo", style="magenta")
    tb.add_column("Lote", style="white")
    tb.add_column("Entrada", style="white")
    tb.add_column("SL", style="red")
    tb.add_column("TP", style="green")
    tb.add_column("PnL", style="yellow")
    tb.add_column("Duracao", style="blue")
    now_ts = datetime.now(timezone.utc).timestamp()
    for p in positions:
        p_type = int(getattr(p, "type", -1))
        side = "BUY" if p_type == getattr(mt5, "ORDER_TYPE_BUY", 0) else "SELL"
        profit = float(getattr(p, "profit", 0.0) or 0.0)
        total_profit += profit
        open_ts = float(getattr(p, "time", 0.0) or 0.0)
        age_min = (now_ts - open_ts) / 60.0 if open_ts > 0 else 0.0
        tb.add_row(
            str(int(getattr(p, "ticket", 0) or 0)),
            str(getattr(p, "symbol", "")),
            side,
            f"{float(getattr(p, 'volume', 0.0) or 0.0):.2f}",
            f"{float(getattr(p, 'price_open', 0.0) or 0.0):.5f}",
            f"{float(getattr(p, 'sl', 0.0) or 0.0):.5f}",
            f"{float(getattr(p, 'tp', 0.0) or 0.0):.5f}",
            f"{profit:.2f}",
            f"{age_min:.1f}m",
        )
    return tb, len(positions), total_profit


def _action_active_trades() -> None:
    symbol = _ask("Filtrar simbolo (ENTER=todos)", "", to_upper=True)
    symbol_filter = symbol if symbol else None
    duration_s = int(_ask("Duracao de monitoramento (segundos)", "60"))
    refresh_s = float(_ask("Intervalo de refresh (segundos)", "2"))
    if mt5 is None:
        _print("MetaTrader5 package nao instalado no Python.", style="red")
        return
    if not ensure_logged_in():
        _print("Nao foi possivel logar no MT5 para monitorar trades.", style="red")
        return
    start = time.time()
    if _CONSOLE is not None and Live is not None and Table is not None:
        table, count, pnl = _positions_table(symbol_filter=symbol_filter)
        with Live(table if table is not None else "", console=_CONSOLE, refresh_per_second=max(1, int(1 / max(0.2, refresh_s)))) as live:
            while (time.time() - start) < duration_s:
                table, count, pnl = _positions_table(symbol_filter=symbol_filter)
                if table is not None:
                    live.update(table)
                time.sleep(refresh_s)
        _print(f"Monitor finalizado. Posicoes: {count} | PnL total: {pnl:.2f}", style="cyan")
    else:
        while (time.time() - start) < duration_s:
            table, count, pnl = _positions_table(symbol_filter=symbol_filter)
            _print(f"Posicoes: {count} | PnL total: {pnl:.2f}", style="cyan")
            if isinstance(table, list):
                for p in table:
                    _print(str(p))
            time.sleep(refresh_s)
        _print("Monitor finalizado.", style="cyan")


def _action_running_bots_monitor() -> None:
    refresh_s = float(_ask("Intervalo de refresh (segundos)", "1"))
    _print("Monitor continuo. Pressione Q para sair desta tela (ou Ctrl+C).", style="cyan")
    human_log = CONFIG.logs_dir / "bot_live_human.log"

    if _CONSOLE is not None and Live is not None and Table is not None and Panel is not None and Group is not None:
        def _render() -> Any:
            procs = _running_bot_processes()
            meta_map = _running_meta_by_pid()
            tb = Table(title="Bots em execucao", show_lines=False)
            tb.add_column("PID", style="cyan")
            tb.add_column("Par", style="white")
            tb.add_column("TF", style="white")
            tb.add_column("Modo", style="magenta")
            tb.add_column("Run ID", style="yellow")
            tb.add_column("Inicio", style="green")
            if not procs:
                tb.add_row("-", "-", "-", "-", "-", "-")
            else:
                for p in procs:
                    md = meta_map.get(int(p["pid"]), {})
                    tb.add_row(
                        str(p["pid"]),
                        str(p["symbol"]),
                        str(p["tf"]),
                        str(p["mode"]),
                        str(md.get("run_id", "-")),
                        str(md.get("started_at", "-")),
                    )
            lines = _tail_text_file(human_log, n=8)
            body = "\n".join(lines) if lines else "Sem logs recentes em logs/bot_live_human.log"
            panel = Panel(body, title="Logs continuos (human)", border_style="bright_blue")
            return Group(tb, panel)

        try:
            with Live(_render(), console=_CONSOLE, refresh_per_second=max(1, int(1 / max(0.2, refresh_s)))) as live:
                while True:
                    if msvcrt is not None and msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if str(ch).lower() == "q":
                            _print("Saindo do monitor (tecla Q).", style="yellow")
                            break
                    time.sleep(refresh_s)
                    live.update(_render())
        except KeyboardInterrupt:
            _print("Monitor encerrado pelo usuario.", style="yellow")
        return

    try:
        while True:
            if msvcrt is not None and msvcrt.kbhit():
                ch = msvcrt.getwch()
                if str(ch).lower() == "q":
                    _print("Saindo do monitor (tecla Q).", style="yellow")
                    break
            procs = _running_bot_processes()
            _print(f"Bots em execucao: {len(procs)}", style="cyan")
            for p in procs:
                _print(f"- PID={p['pid']} {p['symbol']}/{p['tf']} modo={p['mode']}")
            for ln in _tail_text_file(human_log, n=6):
                _print(ln)
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        _print("Monitor encerrado pelo usuario.", style="yellow")


def _action_post_train_flow() -> None:
    _print("Fluxo recomendado: robustez -> paper -> demo (adaptativo por TF)", style="cyan")
    symbol = _choose_symbol("EURUSD")
    tf = _ask("TF alvo", "M5", to_upper=True)
    windows = _ask("Janelas walk-forward", "8")
    seed = _ask("Seed", "42")

    profile = resolve_timeframe_profile(symbol, tf) or CONFIG.timeframe_profiles.get(tf)
    if tf == "M5":
        module = "mt5_ai_bot.src.phase11_runner"
        args = ["--symbol", symbol, "--windows", windows, "--seed", seed]
        run_tag = "phase11_post_train"
        default_summary = "phase11_M5_summary.json"
        flow_name = "Phase11"
    elif profile is not None and profile.enabled:
        module = "mt5_ai_bot.src.phase10_runner"
        args = [
            "--symbol",
            symbol,
            "--tf_entry",
            tf,
            "--tf_gate",
            profile.tf_gate,
            "--windows",
            windows,
            "--seed",
            seed,
        ]
        run_tag = "phase10_post_train"
        default_summary = f"phase10_{tf}_summary.json"
        flow_name = f"Phase10 ({tf}/{profile.tf_gate})"
    else:
        module = "mt5_ai_bot.src.phase4_runner"
        args = ["--symbol", symbol, "--tfs", tf, "--seed", seed]
        run_tag = "phase4_post_train"
        default_summary = f"phase4_summary_{symbol}_"
        flow_name = f"Phase4 diagnostico ({tf})"

    run_id = _run_id(symbol, tf, run_tag)
    run_dir = RUNS_DIR / run_id
    meta = _run_module(module=module, args=args, run_dir=run_dir)
    _print(f"{flow_name} concluida | return_code={meta['return_code']}", style=("green" if meta["return_code"] == 0 else "red"))
    _print_run_result(run_dir, meta)
    summary_path = _find_summary_path(meta, default_summary)
    if summary_path is None and default_summary.endswith("_"):
        # phase4 fallback: busca por prefixo mais recente
        cands = sorted(CONFIG.reports_dir.glob(f"{default_summary}*.json"))
        if cands:
            summary_path = cands[-1]
    if summary_path is not None:
        _print_phase_summary(summary_path)
        approved, _, _ = _extract_summary_approval(summary_path)
        if approved is False:
            if _ask_yes_no("Aplicar auto-ajuste IA no perfil agora?", default=True):
                use_openai = _ask_yes_no("Usar OpenAI API para sugerir parametros?", default=True)
                continuous_mode = _ask_yes_no(
                    "Modo continuo ate APROVAR? (para apenas com Q ou quando aprovar)",
                    default=True,
                )
                max_iters = int(_ask("Maximo de iteracoes de reteste (0 = infinito)", "0" if continuous_mode else "5"))
                if max_iters < 0:
                    max_iters = 0
                if max_iters > 0:
                    max_iters = min(max_iters, 200)
                history: list[dict[str, Any]] = []
                current_summary = summary_path
                reached_target = False

                if _ask_yes_no("Reexecutar robustez automaticamente agora?", default=True):
                    if msvcrt is not None:
                        _print("Auto-ajuste em andamento. Pressione Q para parar.", style="yellow")
                    i = 0
                    while True:
                        i += 1
                        if max_iters > 0 and i > max_iters:
                            break
                        if msvcrt is not None and msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            if str(ch).lower() == "q":
                                _print("Auto-ajuste interrompido pelo usuario (Q).", style="yellow")
                                break
                        if use_openai:
                            try:
                                updates, rationale = _auto_tune_profile_with_openai(
                                    tf=tf,
                                    summary_path=current_summary,
                                    history=history,
                                )
                            except Exception as exc:
                                _print(f"OpenAI falhou na iteracao {i}: {exc}", style="yellow")
                                updates, rationale = {}, "fallback_heuristico"
                        else:
                            updates, rationale = {}, "heuristico"

                        if not updates:
                            updates = _normalize_profile_updates(tf, _auto_tune_profile_from_summary(tf=tf, summary_path=current_summary))
                            if not updates:
                                _print("Nao foi possivel gerar ajustes nesta iteracao.", style="yellow")
                                break
                            if rationale == "fallback_heuristico":
                                rationale = "fallback_heuristico_sem_sugestao_openai"

                        _save_tf_override(
                            tf=tf,
                            updates=updates,
                            reason=f"auto_tune_iter_{i}:{rationale}",
                        )
                        iter_label = f"{i}/{max_iters}" if max_iters > 0 else f"{i}/INF"
                        _print(f"[Iteracao {iter_label}] Auto-ajuste aplicado:", style="green")
                        for k, v in updates.items():
                            _print(f"- {k} = {v}", style="cyan")

                        rerun_id = _run_id(symbol, tf, f"{run_tag}_retune{i}")
                        rerun_dir = RUNS_DIR / rerun_id
                        meta2 = _run_module(
                            module=module,
                            args=args,
                            run_dir=rerun_dir,
                            allow_quit_with_q=True,
                        )
                        if bool(meta2.get("cancelled_by_user", False)):
                            _print("Reexecucao interrompida pelo usuario (Q).", style="yellow")
                            break
                        _print(
                            f"Reexecucao concluida | return_code={meta2['return_code']}",
                            style=("green" if meta2["return_code"] == 0 else "red"),
                        )
                        _print_run_result(rerun_dir, meta2)
                        summary2 = _find_summary_path(meta2, default_summary)
                        if summary2 is None and default_summary.endswith("_"):
                            cands2 = sorted(CONFIG.reports_dir.glob(f"{default_summary}*.json"))
                            if cands2:
                                summary2 = cands2[-1]
                        if summary2 is None:
                            _print("Summary nao encontrado apos reexecucao.", style="yellow")
                            break
                        current_summary = summary2
                        _print_phase_summary(current_summary)
                        approved2, result2, criteria2 = _extract_summary_approval(current_summary)
                        history.append(
                            {
                                "iteration": i,
                                "updates": updates,
                                "approved": approved2,
                                "pf_ratio": float(result2.get("pf_gt_1_ratio_valid_only", 0.0)) if isinstance(result2, dict) else 0.0,
                            }
                        )
                        pf_ratio = float(result2.get("pf_gt_1_ratio_valid_only", 0.0)) if isinstance(result2, dict) else 0.0
                        pf_min = float(criteria2.get("pf_gt_1_ratio_valid_only_min", 0.6)) if isinstance(criteria2, dict) else 0.6
                        approved_flag = bool(approved2) if approved2 is not None else False
                        if approved_flag:
                            reached_target = True
                            _print(
                                f"Meta atingida: APROVADO | pf_ratio={pf_ratio:.3f} (min={pf_min:.3f}). Encerrando loop.",
                                style="green",
                            )
                            break
                        _print(
                            f"Ainda reprovado: approved={approved_flag} | pf_ratio={pf_ratio:.3f} (min={pf_min:.3f}). Seguindo para proxima iteracao.",
                            style="yellow",
                        )
                        if not continuous_mode and max_iters == 0:
                            # Protecao para evitar loop infinito quando usuario desliga modo continuo.
                            break
                    _print(f"Override salvo em: {TF_OVERRIDES_FILE}", style="magenta")
                    if not reached_target:
                        _print("Meta de pf_ratio nao foi atingida dentro do limite de iteracoes.", style="yellow")
    else:
        _print("Resumo da robustez nao encontrado.", style="yellow")
    if meta["return_code"] != 0:
        _save_last_selection({"symbol": symbol, "tf": tf, "action": "post_train_flow", "run_id": run_id})
        return

    if _ask_yes_no("Iniciar paper em background agora?", default=True):
        paper_args = [
            "--symbol",
            symbol,
            "--tf",
            tf,
            "--model-symbol",
            symbol,
            "--model-tf",
            tf,
            "--use-latest-model",
            "--paper",
            "--paper-bars",
            "800",
        ]
        paper_run_id = _run_id(symbol, tf, "bot_paper_bg")
        paper_run_dir = RUNS_DIR / paper_run_id
        paper_meta = _spawn_module(module="mt5_ai_bot.src.bot_live", args=paper_args, run_dir=paper_run_dir)
        _print(f"Paper iniciado em background. PID={paper_meta['pid']}", style="green")
        _print(f"Logs: {paper_meta['stdout_log']}", style="cyan")

    if _ask_yes_no("Iniciar demo-trade em background agora?", default=False):
        if not _check_live_approval_gate(tf=tf, action_label="demo-trade"):
            _save_last_selection({"symbol": symbol, "tf": tf, "action": "post_train_flow", "run_id": run_id})
            return
        _print("Confirmacao: isso envia ordens na conta demo configurada.", style="yellow")
        if _ask_yes_no("Confirma iniciar demo-trade?", default=False):
            demo_args = [
                "--symbol",
                symbol,
                "--tf",
                tf,
                "--model-symbol",
                symbol,
                "--model-tf",
                tf,
                "--use-latest-model",
            ]
            demo_run_id = _run_id(symbol, tf, "bot_demo_bg")
            demo_run_dir = RUNS_DIR / demo_run_id
            demo_meta = _spawn_module(module="mt5_ai_bot.src.bot_live", args=demo_args, run_dir=demo_run_dir)
            _print(f"Demo-trade iniciado em background. PID={demo_meta['pid']}", style="green")
            _print(f"Logs: {demo_meta['stdout_log']}", style="cyan")

    _save_last_selection({"symbol": symbol, "tf": tf, "action": "post_train_flow", "run_id": run_id})


def _action_delete_trained_models() -> None:
    models = list_models()
    if not models:
        _print("Nenhum modelo para apagar.", style="yellow")
        return
    symbol = _ask("Filtrar simbolo (ENTER=todos)", "", to_upper=True)
    tf = _ask("Filtrar TF (ENTER=todos)", "", to_upper=True)
    filtered = [
        m
        for m in models
        if (not symbol or str(m.get("symbol", "")).upper() == symbol)
        and (not tf or str(m.get("timeframe", "")).upper() == tf)
    ]
    if not filtered:
        _print("Nenhum modelo encontrado com esse filtro.", style="yellow")
        return
    if Table is not None and _CONSOLE is not None:
        tb = Table(title="Modelos para excluir", show_lines=False)
        tb.add_column("#", style="cyan")
        tb.add_column("Symbol/TF", style="white")
        tb.add_column("Versao", style="magenta")
        tb.add_column("Treinado em", style="green")
        tb.add_column("Path", style="yellow")
        for i, m in enumerate(filtered, start=1):
            tb.add_row(
                str(i),
                f"{m.get('symbol')}/{m.get('timeframe')}",
                str(m.get("version")),
                str(m.get("trained_at")),
                str(m.get("model_path", "")),
            )
        _CONSOLE.print(tb)
    else:
        for i, m in enumerate(filtered, start=1):
            _print(f"{i}) {m.get('symbol')}/{m.get('timeframe')} v={m.get('version')} path={m.get('model_path')}")

    def _parse_multi_selection(raw: str, max_n: int) -> list[int]:
        s = raw.strip().lower()
        if not s or s == "0":
            return []
        if s in {"all", "todos"}:
            return list(range(1, max_n + 1))
        out: set[int] = set()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                if not a.isdigit() or not b.isdigit():
                    continue
                start = int(a)
                end = int(b)
                if start > end:
                    start, end = end, start
                for i in range(start, end + 1):
                    if 1 <= i <= max_n:
                        out.add(i)
            else:
                if p.isdigit():
                    i = int(p)
                    if 1 <= i <= max_n:
                        out.add(i)
        return sorted(out)

    idx_raw = _ask("Numero(s) para excluir (ex: 1,3,5-8 | all | 0=cancelar)", "0")
    selected_idx = _parse_multi_selection(idx_raw, len(filtered))
    if not selected_idx:
        _print("Operacao cancelada.", style="yellow")
        return

    targets = [filtered[i - 1] for i in selected_idx]
    _print("Selecionados para exclusao:", style="yellow")
    for t in targets:
        _print(f"- {t.get('symbol')}/{t.get('timeframe')} v={t.get('version')}")

    if len(targets) > 1:
        _print(f"Total: {len(targets)} modelos", style="yellow")

    if not _ask_yes_no("Confirmar exclusao?", default=False):
        _print("Operacao cancelada.", style="yellow")
        return
    confirm_label = _ask("Digite EXCLUIR para confirmar", "")
    if confirm_label.strip().upper() != "EXCLUIR":
        _print("Confirmacao nao corresponde. Cancelado.", style="yellow")
        return

    removed: list[Path] = []
    failed: list[str] = []
    for target in targets:
        paths: list[Path] = []
        for key in ("model_path", "features_schema_path", "train_meta_path", "metrics_oof_path", "calibration_path"):
            raw = str(target.get(key, "") or "").strip()
            if raw:
                paths.append(Path(raw))

        # garante limpeza de artefatos legados em models/
        sym = str(target.get("symbol", "")).strip()
        tf = str(target.get("timeframe", "")).strip()
        ver = str(target.get("version", "")).strip()
        if sym and tf and ver:
            paths.append(CONFIG.models_dir / f"{sym}_{tf}_{ver}.pkl")
            paths.append(CONFIG.models_dir / f"{sym}_{tf}_{ver}.json")

        # dedup mantendo ordem
        uniq_paths: list[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                uniq_paths.append(p)

        for p in uniq_paths:
            try:
                if p.exists() and p.is_file():
                    p.unlink()
                    removed.append(p)
            except Exception as exc:
                failed.append(f"{p}: {exc}")

        # remove pasta da versao (reports/models/<symbol>/<tf>/<version>) quando existir
        model_path = Path(str(target.get("model_path", "")))
        version_dir = model_path.parent if model_path.name.lower() == "model.bin" else None
        if version_dir and version_dir.exists() and version_dir.is_dir():
            try:
                shutil.rmtree(version_dir)
                removed.append(version_dir)
            except Exception as exc:
                failed.append(f"{version_dir}: {exc}")

    index_path = CONFIG.reports_dir / "models" / "index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            old = data.get("models", [])
            delete_keys = {
                (str(t.get("symbol")), str(t.get("timeframe")), str(t.get("version")))
                for t in targets
            }
            kept = [r for r in old if (str(r.get("symbol")), str(r.get("timeframe")), str(r.get("version"))) not in delete_keys]
            data["models"] = kept
            index_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            _print(f"Falha ao atualizar registry: {exc}", style="red")

    _print(f"Modelos removidos: {len(targets)}", style="green")
    if removed:
        _print("Itens removidos:", style="cyan")
        for p in removed:
            _print(f"- {p}")
    if failed:
        _print("Falhas na remocao:", style="red")
        for msg in failed:
            _print(f"- {msg}", style="red")


def run_menu() -> None:
    _ensure_dirs()
    while True:
        last = _load_json(LAST_SELECTION, {})
        st = _mt5_status()
        oai = _openai_status()
        creds_txt = "[green]OK[/green]" if st["has_creds"] else "[red]NAO[/red]"
        installed_txt = "[green]SIM[/green]" if st["installed"] else "[red]NAO[/red]"
        login_txt = (
            f"[green]LOGADO ({st['account_login']})[/green]"
            if st["logged_in"]
            else "[yellow]NAO LOGADO[/yellow]"
        )
        openai_txt = f"[green]OK ({oai['model']})[/green]" if oai["has_key"] else "[red]NAO[/red]"
        if _CONSOLE is not None and Panel is not None:
            _CONSOLE.print(
                Panel(
                    "[bold cyan]MT5 AI BOT MENU[/bold cyan]\n"
                    "[white]Gerenciamento de treino, fases e execucao live[/white]\n"
                    f"[white]Credenciais:[/white] {creds_txt} | "
                    f"[white]MT5 instalado:[/white] {installed_txt} | "
                    f"[white]Login:[/white] {login_txt} | "
                    f"[white]OpenAI:[/white] {openai_txt}",
                    border_style="bright_blue",
                )
            )
        else:
            _print("\n==== MT5 AI BOT MENU ====", style="cyan")
            _print(
                f"Credenciais={'OK' if st['has_creds'] else 'NAO'} | "
                f"MT5_instalado={'SIM' if st['installed'] else 'NAO'} | "
                f"Login={'LOGADO' if st['logged_in'] else 'NAO LOGADO'} | "
                f"OpenAI={'OK' if oai['has_key'] else 'NAO'}"
            )
        _print_last_selection(last)
        _print("1) Listar modelos treinados", style="white")
        _print("2) Treinar novo modelo", style="white")
        _print("3) Rodar backtest/robustez (fase)", style="white")
        _print("4) Rodar bot (paper)", style="white")
        _print("5) Rodar bot (demo-trade)", style="white")
        _print("6) Rodar diagnostico (diagnostic-only)", style="white")
        _print("7) Configurar simbolos/TFs (perfis)", style="white")
        _print("8) Iniciar trade (background + submenu flags)", style="white")
        _print("9) Configurar credenciais MT5", style="white")
        _print("10) Trades ativos (tempo real)", style="white")
        _print("11) Fluxo pos-treino (phase11 -> paper -> demo)", style="white")
        _print("12) Configurar OpenAI API (key/model)", style="white")
        _print("13) Gerenciar modelos (apagar)", style="white")
        _print("14) Bots rodando + logs continuos", style="white")
        _print("15) Modo scalping M1 (preset + iniciar)", style="white")
        _print("16) Modo intraday ativo (M1/M5 preset + iniciar)", style="white")
        _print("0) Sair", style="white")
        opt = _ask("Escolha", "1")
        if opt == "1":
            _print_models()
            _pause_continue()
        elif opt == "2":
            _action_train()
            _pause_continue()
        elif opt == "3":
            _action_phase()
            _pause_continue()
        elif opt == "4":
            _action_bot(paper=True, diagnostic=False)
            _pause_continue()
        elif opt == "5":
            _action_bot(paper=False, diagnostic=False)
            _pause_continue()
        elif opt == "6":
            _action_bot(paper=False, diagnostic=True)
            _pause_continue()
        elif opt == "7":
            _action_profiles()
            _pause_continue()
        elif opt == "8":
            _action_start_trade_background()
            _pause_continue()
        elif opt == "9":
            _action_mt5_credentials()
            _pause_continue()
        elif opt == "10":
            _action_active_trades()
            _pause_continue()
        elif opt == "11":
            _action_post_train_flow()
            _pause_continue()
        elif opt == "12":
            _action_openai_credentials()
            _pause_continue()
        elif opt == "13":
            _action_delete_trained_models()
            _pause_continue()
        elif opt == "14":
            _action_running_bots_monitor()
            _pause_continue()
        elif opt == "15":
            _action_scalping_menu()
            _pause_continue()
        elif opt == "16":
            _action_active_intraday_menu()
            _pause_continue()
        elif opt == "0":
            _print("Saindo.", style="cyan")
            return
        else:
            _print("Opcao invalida.", style="red")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI menu for MT5 AI Bot")
    parser.add_argument("--once", action="store_true", help="Run one listing pass and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.once:
        _print_models()
        return
    run_menu()


if __name__ == "__main__":
    main()

