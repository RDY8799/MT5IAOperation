from __future__ import annotations

import argparse
from collections import deque
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

from .config import CONFIG
from .model_registry import list_models
from .mt5_connect import ensure_logged_in

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


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    creds = _load_mt5_credentials()
    env.update(creds)
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
        result = data.get("result", {})
        criteria = data.get("criteria", {})
        approved = bool(data.get("approved", False))
        symbol = str(data.get("symbol", "-"))
        entry_tf = str(data.get("entry_tf", "-"))
        gate_tf = str(data.get("gate_tf", "-"))

        status_txt = "APROVADO" if approved else "REPROVADO"
        status_style = "green" if approved else "red"
        _print(f"Resumo da robustez [{symbol} {entry_tf}/{gate_tf}]: {status_txt}", style=status_style)
        _print(
            "Metricas: "
            f"pf_ratio={float(result.get('pf_gt_1_ratio_valid_only', 0.0)):.3f} | "
            f"dd_windows_ok={bool(result.get('dd_windows_ok', False))} | "
            f"stress25_pf={float(result.get('stress25_pf', 0.0)):.3f} | "
            f"trade_windows={int(result.get('trade_windows_count', 0))} | "
            f"coverage={float(result.get('coverage_ratio', 0.0)):.3f} | "
            f"trades_total={int(result.get('trades_total', 0))}",
            style="cyan",
        )
        _print(
            "Criterios minimos: "
            f"pf_ratio>={float(criteria.get('pf_gt_1_ratio_valid_only_min', 0.6)):.2f}, "
            f"dd_windows_ok={bool(criteria.get('dd_windows_ok', True))}, "
            f"stress25_pf>1={bool(criteria.get('stress25_pf_gt_1', True))}, "
            f"trade_windows>={int(criteria.get('trade_windows_count_min', 6))}, "
            f"coverage>={float(criteria.get('coverage_ratio_min', 0.9)):.2f}, "
            f"trades_total>={int(criteria.get('trades_total_min', 120))}",
            style="white",
        )
        if approved:
            _print("Leitura simples: setup robusto para paper/demo com risco controlado.", style="green")
        else:
            _print("Leitura simples: ainda NAO esta robusto; precisa revisar filtros/risco antes de operar.", style="yellow")
        _print(f"Arquivo resumo: {summary_path}", style="magenta")
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
    profile = CONFIG.timeframe_profiles.get(tf)
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

    for k, v in updates.items():
        setattr(profile, k, v)
    return updates


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


def _run_module(module: str, args: list[str], run_dir: Path) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module, *args]
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
        "args": args,
        "cmd": cmd,
        "started_at": started_at,
        "finished_at": _now_utc().isoformat(),
        "return_code": int(proc.returncode or 0),
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
    cmd = [sys.executable, "-m", module, *args]
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
    gate = _ask("TF gate", "M30", to_upper=True)
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
    if meta["saved_paths"]:
        _print("Saidas:", style="cyan")
        for p in meta["saved_paths"]:
            _print(f"- {p}")


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


def _action_profiles() -> None:
    data = _load_json(SYMBOL_PROFILES, {"profiles": []})
    _print("Configurar simbolo/tf preferido (nao altera estrategia, apenas orquestracao).", style="cyan")
    symbol = _choose_symbol("EURUSD")
    tf = _ask("TF", "M5", to_upper=True)
    gate = _ask("Gate", "M30", to_upper=True)
    item = {"symbol": symbol, "tf": tf, "gate": gate, "updated_at": _now_utc().isoformat()}
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

    profile = CONFIG.timeframe_profiles.get(tf)
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
                updates = _auto_tune_profile_from_summary(tf=tf, summary_path=summary_path)
                if updates:
                    _save_tf_override(
                        tf=tf,
                        updates=updates,
                        reason=f"auto_tune_after_fail:{summary_path.name}",
                    )
                    _print("Auto-ajuste aplicado:", style="green")
                    for k, v in updates.items():
                        _print(f"- {k} = {v}", style="cyan")
                    _print(f"Override salvo em: {TF_OVERRIDES_FILE}", style="magenta")
                    if _ask_yes_no("Reexecutar robustez agora com ajustes?", default=True):
                        rerun_id = _run_id(symbol, tf, f"{run_tag}_retune")
                        rerun_dir = RUNS_DIR / rerun_id
                        meta2 = _run_module(module=module, args=args, run_dir=rerun_dir)
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
                        if summary2 is not None:
                            _print_phase_summary(summary2)
                else:
                    _print("Nao foi possivel gerar auto-ajuste para esse summary.", style="yellow")
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
        creds_txt = "[green]OK[/green]" if st["has_creds"] else "[red]NAO[/red]"
        installed_txt = "[green]SIM[/green]" if st["installed"] else "[red]NAO[/red]"
        login_txt = (
            f"[green]LOGADO ({st['account_login']})[/green]"
            if st["logged_in"]
            else "[yellow]NAO LOGADO[/yellow]"
        )
        if _CONSOLE is not None and Panel is not None:
            _CONSOLE.print(
                Panel(
                    "[bold cyan]MT5 AI BOT MENU[/bold cyan]\n"
                    "[white]Gerenciamento de treino, fases e execucao live[/white]\n"
                    f"[white]Credenciais:[/white] {creds_txt} | "
                    f"[white]MT5 instalado:[/white] {installed_txt} | "
                    f"[white]Login:[/white] {login_txt}",
                    border_style="bright_blue",
                )
            )
        else:
            _print("\n==== MT5 AI BOT MENU ====", style="cyan")
            _print(
                f"Credenciais={'OK' if st['has_creds'] else 'NAO'} | "
                f"MT5_instalado={'SIM' if st['installed'] else 'NAO'} | "
                f"Login={'LOGADO' if st['logged_in'] else 'NAO LOGADO'}"
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
        _print("13) Gerenciar modelos (apagar)", style="white")
        _print("14) Bots rodando + logs continuos", style="white")
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
        elif opt == "13":
            _action_delete_trained_models()
            _pause_continue()
        elif opt == "14":
            _action_running_bots_monitor()
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

