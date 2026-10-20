"""Single-attempt compatible chat inference with explicit usage accounting."""

import base64
import hashlib
import json
import math
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import ModelResponse


class HTTPBackend:
    def __init__(
        self,
        base_url,
        model,
        *,
        api_key_env="MODEL_API_KEY",
        timeout=60,
        temperature=0.1,
        max_new_tokens=1024,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("A valid HTTP model endpoint is required")
        if timeout <= 0 or max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Timeout and generation limit must be positive")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key_env, self.timeout = api_key_env, timeout
        self.temperature, self.max_new_tokens = temperature, max_new_tokens
        self.version = hashlib.sha256(
            json.dumps(
                {
                    "base_url": self.base_url,
                    "model": model,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "timeout": timeout,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def generate(self, system, user, images=()):
        content = [{"type": "text", "text": user}]
        for file in images:
            file = Path(file)
            mime = mimetypes.guess_type(file.name)[0]
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                raise ValueError("Unsupported image media type")
            encoded = base64.b64encode(file.read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
            )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content if images else user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        start = time.perf_counter()
        response = requests.post(
            self.base_url + "/chat/completions", json=body, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        try:
            text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("Expected textual model output")
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Malformed model response envelope") from error
        usage = payload.get("usage", {})
        return ModelResponse(
            text,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            time.perf_counter() - start,
        )

    def score_binary(self, prompt, images=()):
        raise ValueError(
            "Calibrated verification requires a local backend with conditional label likelihoods"
        )
