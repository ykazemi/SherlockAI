# llm_client_hf.py
from __future__ import annotations
import json
import os
import requests
from typing import Any, Dict, Optional
import re


class LLMError(RuntimeError):
    pass


def _extract_first_json_object(s: str) -> str:
    # Try to find the first {...} block (naive but works well for demos)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise LLMError("Model did not return a JSON object.")
    return m.group(0)


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v.strip() == "":
        raise LLMError(f"Missing env var: {name}")
    return v


def call_hf_chat_json(
    system: str, user: str, temperature: float = 0.2, max_tokens: int = 700
) -> Dict[str, Any]:
    """
    Calls Hugging Face router OpenAI-compatible /chat/completions endpoint.
    Expects the model to return JSON text which we parse here.
    """
    base_url = _env("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    token = _env("HF_TOKEN")
    model = _env("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 400:
        raise LLMError(f"HF HTTP {r.status_code}: {r.text[:600]}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content2 = content.strip().strip("```").strip()
        try:
            return json.loads(content2)
        except json.JSONDecodeError:
            return json.loads(_extract_first_json_object(content2))

    # Parse strict JSON; tolerate code fences
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content2 = content.strip().strip("```").strip()
        return json.loads(content2)
