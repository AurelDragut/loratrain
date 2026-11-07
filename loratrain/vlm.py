"""A client for the vision endpoints people actually run locally.

Local inference splits into two shapes:

* OpenAI-compatible chat completions — vLLM, llama.cpp's server, LM Studio and
  Ollama's own ``/v1`` shim. The image travels as a data URI in an ``image_url``
  part.
* Ollama's native ``/api/chat`` — the image travels as raw base64 in an ``images``
  array, and ``format: "json"`` constrains the reply to JSON, which is the most
  reliable way to get parseable captions out of a small vision model.

A bare URL is ambiguous, so it is probed once: if ``/api/tags`` answers, the server
is Ollama, otherwise it is treated as OpenAI-compatible.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OPENAI = "openai"
OLLAMA = "ollama"

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


class VlmError(RuntimeError):
    """Raised when the endpoint cannot be reached or refuses the request."""


@dataclass
class VlmConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0
    retries: int = 2
    max_tokens: int = 400
    flavor: str = "auto"  # auto | openai | ollama
    keep_alive: str = "30m"
    json_mode: bool = True

    def endpoint(self) -> str:
        base = self.base_url.strip().rstrip("/")

        if self.flavor == OLLAMA:
            if base.endswith("/api/chat"):
                return base
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            if base.endswith("/api"):
                return base + "/chat"
            return base + "/api/chat"

        if base.endswith("/chat/completions"):
            return base
        if not urllib.parse.urlsplit(base).path.strip("/"):
            return base + "/v1/chat/completions"
        return base + "/chat/completions"


def _http_json(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except Exception:
        return None


def resolve_flavor(config: VlmConfig) -> str:
    """Settle ``auto`` into a concrete flavour, probing a bare host once."""
    if config.flavor in (OPENAI, OLLAMA):
        return config.flavor

    base = config.base_url.strip().rstrip("/")
    path = urllib.parse.urlsplit(base).path.strip("/")

    if path.endswith("api") or path.endswith("api/chat"):
        config.flavor = OLLAMA
        return OLLAMA
    if path.endswith("v1"):
        config.flavor = OPENAI
        return OPENAI

    if _http_json(base + "/api/tags") is not None:
        config.flavor = OLLAMA
    else:
        config.flavor = OPENAI
    return config.flavor


def list_models(config: VlmConfig) -> list[str] | None:
    """Names the server reports, or None when it cannot be asked."""
    flavor = resolve_flavor(config)
    base = config.base_url.strip().rstrip("/")

    if flavor == OLLAMA:
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        data = _http_json(base + "/api/tags", timeout=10.0)
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            names = [
                str(item["name"])
                for item in data["models"]
                if isinstance(item, dict) and item.get("name")
            ]
            return sorted(names)
        return None

    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not urllib.parse.urlsplit(base).path.strip("/"):
        base = base + "/v1"
    data = _http_json(base + "/models", timeout=10.0)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        names = [
            str(item["id"]) for item in data["data"] if isinstance(item, dict) and item.get("id")
        ]
        return sorted(names)
    return None


def _base64_image(path: Path | str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def data_uri(path: Path | str) -> str:
    path = Path(path)
    mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64," + _base64_image(path)


def build_openai_payload(model: str, prompt: str, image_uri: str, max_tokens: int = 400) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def build_ollama_payload(
    model: str,
    prompt: str,
    image_base64: str,
    max_tokens: int = 400,
    keep_alive: str = "30m",
    json_mode: bool = True,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_base64]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
        "keep_alive": keep_alive,
    }
    if json_mode:
        payload["format"] = "json"
    return payload


def parse_response(text: str) -> dict:
    """Pull the JSON object out of a model reply, falling back to plain text.

    Models wrap JSON in fenced blocks or add a sentence before it, so the outermost
    brace pair is what gets parsed. A reply with no JSON at all is still useful: it
    becomes the caption.
    """
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            caption = parsed.get("caption")
            parsed["caption"] = caption.strip() if isinstance(caption, str) else ""
            return parsed

    return {"caption": raw}


def extract_text(payload: dict, flavor: str) -> str:
    """The assistant text out of either response shape."""
    if flavor == OLLAMA:
        message = payload.get("message")
        if isinstance(message, dict) and "content" in message:
            return str(message.get("content") or "")
        error = payload.get("error")
        raise VlmError(str(error) if error else f"unexpected Ollama reply: {str(payload)[:200]}")

    choices = payload.get("choices") or []
    if not choices:
        error = payload.get("error") or payload
        raise VlmError(f"no choices in response: {str(error)[:200]}")

    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def error_message(detail: str) -> str:
    """Turn an error body into something readable.

    Ollama and vLLM both answer with ``{"error": ...}``; a reverse proxy answers
    with HTML, which is worse than useless, so it gets truncated.
    """
    detail = (detail or "").strip()
    if not detail:
        return "the server sent no error body"
    try:
        data = json.loads(detail)
    except json.JSONDecodeError:
        return detail[:300]

    if isinstance(data, dict):
        error = data.get("error") or data.get("message") or data
        if isinstance(error, dict):
            error = error.get("message") or error
        return str(error)[:300]
    return str(data)[:300]


def _decode(raw: str, flavor: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VlmError(f"endpoint returned something that is not JSON: {raw[:200]}") from exc
    return parse_response(extract_text(payload, flavor))


def caption_image(path: Path | str, prompt: str, config: VlmConfig) -> dict:
    """Send one image to the model and return the parsed reply."""
    flavor = resolve_flavor(config)

    if flavor == OLLAMA:
        payload = build_ollama_payload(
            config.model,
            prompt,
            _base64_image(path),
            config.max_tokens,
            config.keep_alive,
            config.json_mode,
        )
    else:
        payload = build_openai_payload(config.model, prompt, data_uri(path), config.max_tokens)

    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = config.endpoint()
    last_error: VlmError | None = None

    for attempt in range(config.retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
            return _decode(raw, flavor)
        except urllib.error.HTTPError as exc:
            detail = error_message(exc.read().decode("utf-8", "replace"))
            lowered = detail.lower()

            if exc.code in (401, 403):
                raise VlmError(
                    f"the endpoint rejected the credentials ({exc.code}): {detail}. "
                    "loratrain sends LORATRAIN_VLM_API_KEY as a bearer token; unset it "
                    "for a local server that wants none"
                ) from exc
            if exc.code == 404 and "model" in lowered:
                raise VlmError(
                    f"{config.model!r} is not available on {url}: {detail}. list what is "
                    "served with `loratrain doctor --vlm-url <url>`"
                ) from exc
            if exc.code == 404:
                raise VlmError(
                    f"{url} returned 404: {detail}. A bare host is treated as Ollama "
                    "(/api/chat) and a URL with /v1 as OpenAI-compatible; use "
                    "--vlm-flavor to say which"
                ) from exc

            last_error = VlmError(f"HTTP {exc.code} from {url}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = VlmError(
                f"could not reach {url}: {exc}. if the server is loading a large model "
                "the first request can be slow; raise --vlm-timeout"
            )

        if attempt < config.retries:
            time.sleep(1.5 * (attempt + 1))

    raise last_error or VlmError(f"request to {url} failed")
