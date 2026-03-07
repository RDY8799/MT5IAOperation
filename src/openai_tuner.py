from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


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
    Consulta a OpenAI para sugerir ajustes de filtros/risco sem mudar modelo/labeling.
    Retorna: (updates, rationale)
    """
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY vazio")
    if not model.strip():
        raise ValueError("Modelo OpenAI vazio")

    allowed_keys = [
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
    ]
    prompt = {
        "task": "Ajustar perfil de trading para melhorar robustez sem overfitting agressivo.",
        "constraints": [
            "NAO alterar modelo base nem labeling.",
            "Ajustar apenas filtros operacionais do timeframe.",
            "Objetivo principal: pf_gt_1_ratio_valid_only >= criterio minimo.",
            "Manter ajustes pequenos e conservadores por iteracao.",
        ],
        "timeframe": tf,
        "allowed_update_keys": allowed_keys,
        "current_profile": profile,
        "result_metrics": result,
        "criteria": criteria,
        "history": history or [],
        "response_format": {
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
            },
            "rationale": "string",
        },
    }

    body = {
        "model": model,
        "temperature": 0.2,
        "input": [
            {
                "role": "system",
                "content": (
                    "Voce e um quant pragmático. Responda APENAS JSON válido com chaves "
                    "{updates, rationale}. Nao inclua markdown."
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
        raise RuntimeError("Resposta OpenAI sem JSON válido")

    updates = parsed.get("updates", {})
    rationale = str(parsed.get("rationale", "")).strip()
    if not isinstance(updates, dict):
        updates = {}
    clean = {k: v for k, v in updates.items() if k in allowed_keys}
    return clean, rationale


def _extract_text(payload: dict[str, Any]) -> str:
    # Novo formato Responses API
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

    # Fallbacks
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
