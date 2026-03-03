from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG
from .model_registry import list_models
from .mt5_connect import ensure_logged_in

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
    filt = _ask("Filtro de simbolo (ENTER=todos)", default[:3], to_upper=True)
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

        def _render() -> Any:
            body = "\n".join(live_lines) if live_lines else "Aguardando atualizacoes..."
            panel = Panel(body, title="Treino em tempo real", border_style="bright_blue")
            return Group(progress, panel)

        with Live(_render(), console=_CONSOLE, refresh_per_second=5, transient=True) as live:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\r\n")
                stdout_lines.append(line)
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
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")


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
    run_id = _run_id(symbol, tf, "bot_paper" if paper else ("bot_diag" if diagnostic else "bot_demo"))
    run_dir = RUNS_DIR / run_id
    meta = _run_module(module="mt5_ai_bot.src.bot_live", args=args, run_dir=run_dir)
    _save_last_selection({"symbol": symbol, "tf": tf, "action": "bot", "run_id": run_id})
    _print(f"Run: {run_id}", style="green")
    _print(f"Return code: {meta['return_code']}", style="yellow")
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")


def _action_start_trade_background() -> None:
    symbol, tf, args = _build_bot_args_submenu(paper=False, diagnostic=False, force_trade=True)
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


def _action_post_train_flow() -> None:
    _print("Fluxo recomendado: robustez (phase11) -> paper -> demo", style="cyan")
    symbol = _choose_symbol("EURUSD")
    tf = _ask("TF alvo", "M5", to_upper=True)
    if tf != "M5":
        _print("Phase11 atual foi desenhada para M5. Use M5 neste fluxo guiado.", style="yellow")
        return
    windows = _ask("Janelas walk-forward", "8")
    seed = _ask("Seed", "42")

    run_id = _run_id(symbol, tf, "phase11_post_train")
    run_dir = RUNS_DIR / run_id
    meta = _run_module(
        module="mt5_ai_bot.src.phase11_runner",
        args=["--symbol", symbol, "--windows", windows, "--seed", seed],
        run_dir=run_dir,
    )
    _print(f"Phase11 concluida | return_code={meta['return_code']}", style=("green" if meta["return_code"] == 0 else "red"))
    _print(f"Run meta: {run_dir / 'run_meta.json'}", style="cyan")
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
        _print("0) Sair", style="white")
        opt = _ask("Escolha", "1")
        if opt == "1":
            _print_models()
        elif opt == "2":
            _action_train()
        elif opt == "3":
            _action_phase()
        elif opt == "4":
            _action_bot(paper=True, diagnostic=False)
        elif opt == "5":
            _action_bot(paper=False, diagnostic=False)
        elif opt == "6":
            _action_bot(paper=False, diagnostic=True)
        elif opt == "7":
            _action_profiles()
        elif opt == "8":
            _action_start_trade_background()
        elif opt == "9":
            _action_mt5_credentials()
        elif opt == "10":
            _action_active_trades()
        elif opt == "11":
            _action_post_train_flow()
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

