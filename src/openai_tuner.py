from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _allowed_update_keys() -> list[str]:
    return [
        "signal_threshold",
        "min_signal_margin",
        "buy_signal_threshold",
        "sell_signal_threshold",
        "buy_min_signal_margin",
        "sell_min_signal_margin",
        "volatility_p_min",
        "volatility_p_max",
        "reentry_block_candles",
        "max_trades_per_hour",
        "min_candles_between_same_direction_trades",
        "gate_mode",
        "gate_min_margin_block",
        "allow_gate_wait_bypass",
        "gate_wait_bypass_threshold",
        "impulse_alignment_required",
        "impulse_lookback_bars",
        "impulse_min_abs_return",
        "horizon_candles",
    ]


def suggest_profile_update_candidates(
    *,
    api_key: str,
    model: str,
    tf: str,
    profile: dict[str, Any],
    result: dict[str, Any],
    criteria: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    num_candidates: int = 5,
    timeout_s: int = 60,
) -> list[dict[str, Any]]:
    """
    Consulta a OpenAI para sugerir varios candidatos de ajuste do perfil.
    Retorna lista no formato: [{"updates": {...}, "rationale": "..."}].
    """
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY vazio")
    if not model.strip():
        raise ValueError("Modelo OpenAI vazio")

    num_candidates = max(1, min(int(num_candidates), 50))
    allowed_keys = _allowed_update_keys()
    prompt = {
        "task": "Ajustar perfil de trading para melhorar robustez sem overfitting agressivo.",
        "constraints": [
            "NAO alterar modelo base nem labeling.",
            "Ajustar apenas filtros operacionais do timeframe.",
            "Gerar candidatos pequenos e defensaveis, nao palpites aleatorios.",
            "Priorizar candidatos aprovados; se nao houver aprovados, retornar os mais proximos da aprovacao.",
            "Evitar candidatos duplicados.",
        ],
        "timeframe": tf,
        "num_candidates": num_candidates,
        "allowed_update_keys": allowed_keys,
        "current_profile": profile,
        "result_metrics": result,
        "criteria": criteria,
        "history": history or [],
        "response_format": {
            "candidates": [
                {
                    "updates": {
                        "signal_threshold": "float",
                        "min_signal_margin": "float",
                        "buy_signal_threshold": "float",
                        "sell_signal_threshold": "float",
                        "buy_min_signal_margin": "float",
                        "sell_min_signal_margin": "float",
                        "volatility_p_min": "float",
                        "volatility_p_max": "float",
                        "reentry_block_candles": "int",
                        "max_trades_per_hour": "int",
                        "min_candles_between_same_direction_trades": "int",
                        "gate_mode": "strict|allow_wait|bias_only|off",
                        "gate_min_margin_block": "float",
                        "allow_gate_wait_bypass": "bool",
                        "gate_wait_bypass_threshold": "float",
                        "impulse_alignment_required": "bool",
                        "impulse_lookback_bars": "int",
                        "impulse_min_abs_return": "float",
                        "horizon_candles": "int",
                    },
                    "rationale": "string curta"
                }
            ]
        },
    }

    body = {
        "model": model,
        "temperature": 0.35,
        "input": [
            {
                "role": "system",
                "content": (
                    "Voce e um quant pragmatico. Responda APENAS JSON valido no formato "
                    "{\"candidates\":[{\"updates\":{...},\"rationale\":\"...\"}]}. "
                    "Nao inclua markdown."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }

    req = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail[:280]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Falha ao chamar OpenAI: {exc}") from exc

    text = _extract_text(payload)
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Resposta OpenAI sem JSON valido")

    raw_candidates = parsed.get("candidates")
    if not isinstance(raw_candidates, list):
        if isinstance(parsed.get("updates"), dict):
            raw_candidates = [parsed]
        else:
            raw_candidates = []

    clean_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        updates = item.get("updates", {})
        rationale = str(item.get("rationale", "")).strip()
        if not isinstance(updates, dict):
            continue
        clean = {k: v for k, v in updates.items() if k in allowed_keys}
        if not clean:
            continue
        key = json.dumps(clean, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        clean_candidates.append({"updates": clean, "rationale": rationale})
        if len(clean_candidates) >= num_candidates:
            break
    return clean_candidates


def suggest_profile_updates(
    *,
    api_key: str,
    model: str,
    tf: str,
    profile: dict[str, Any],
    result: dict[str, Any],
    criteria: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    timeout_s: int = 45,
) -> tuple[dict[str, Any], str]:
    """
    Consulta a OpenAI para sugerir um unico ajuste de filtros/risco sem mudar modelo/labeling.
    Retorna: (updates, rationale)
    """
    candidates = suggest_profile_update_candidates(
        api_key=api_key,
        model=model,
        tf=tf,
        profile=profile,
        result=result,
        criteria=criteria,
        history=history,
        num_candidates=1,
        timeout_s=timeout_s,
    )
    if not candidates:
        return {}, ""
    first = candidates[0]
    return dict(first.get("updates", {})), str(first.get("rationale", "")).strip()


def _extract_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if isinstance(c, dict):
                    txt = c.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
        if parts:
            return "\n".join(parts).strip()

    out_txt = payload.get("output_text")
    if isinstance(out_txt, str) and out_txt.strip():
        return out_txt.strip()
    return json.dumps(payload, ensure_ascii=False)


def _parse_json(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        frag = s[start : end + 1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None
