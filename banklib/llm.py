"""
Tiny Ollama client (no extra SDK needed - just `requests`).

    from banklib.llm import ask, ask_json, status
    ask("Explain this variance ...")
    ask_json("Extract fields ... Return JSON only.")

If Ollama is not running, functions return None and every demo switches to
a rule-based fallback, clearly labelled, so a live session never stalls.
"""
import json
import re
import requests

from . import config

_last_error = None


def status():
    """Return (ok: bool, message: str, models: list[str])."""
    global _last_error
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(m.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for m in models):
            return False, (f"Ollama is running but model '{config.OLLAMA_MODEL}' is not pulled. "
                           f"Run:  ollama pull {config.OLLAMA_MODEL}"), models
        return True, f"Ollama ready - model {config.OLLAMA_MODEL}", models
    except Exception as e:  # noqa: BLE001
        _last_error = str(e)
        return False, f"Ollama not reachable at {config.OLLAMA_HOST} (start the Ollama app or run `ollama serve`)", []


PREFERRED = ["qwen2.5:7b", "qwen2.5", "llama3.1:8b", "llama3.2", "llama3.1", "mistral", "gemma2", "phi3"]


def auto_select():
    """If the configured model is not pulled but others are, switch to an installed one. Returns the model used."""
    ok, _, models = status()
    if ok or not models:
        return config.OLLAMA_MODEL
    names = [m for m in models if "embed" not in m]
    for want in PREFERRED:
        hit = next((m for m in names if m == want or m.split(":")[0] == want.split(":")[0]), None)
        if hit:
            config.OLLAMA_MODEL = hit
            return hit
    if names:
        config.OLLAMA_MODEL = names[0]
    return config.OLLAMA_MODEL


def available():
    return status()[0]


def ask(prompt, system=None, json_mode=False, temperature=0.0, model=None, max_tokens=None):
    """Send one prompt to Ollama and return the reply text, or None if unavailable."""
    global _last_error
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})},
        "keep_alive": "30m",   # keep the model loaded between calls - avoids a reload on every demo step
    }
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=config.OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        _last_error = str(e)
        if not config.ALLOW_FALLBACK:
            raise
        return None


def stream(prompt, system=None, temperature=0.0, model=None, max_tokens=None, deadline_s=120):
    """Yield the reply piece by piece as Ollama writes it (text appears immediately).
    Stops after deadline_s seconds. Yields nothing if Ollama is unavailable."""
    import json as _json
    import time as _time
    global _last_error
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {"model": model or config.OLLAMA_MODEL, "messages": messages, "stream": True, "keep_alive": "30m",
               "options": {"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})}}
    t0 = _time.time()
    try:
        with requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, stream=True,
                           timeout=(3, config.OLLAMA_TIMEOUT)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                part = _json.loads(line)
                piece = part.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if part.get("done") or _time.time() - t0 > deadline_s:
                    break
    except Exception as e:  # noqa: BLE001
        _last_error = str(e)
        return


def parse_json(text):
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    if text is None:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def ask_json(prompt, system=None, max_tokens=None):
    return parse_json(ask(prompt, system=system, json_mode=True, max_tokens=max_tokens))


def last_error():
    return _last_error
