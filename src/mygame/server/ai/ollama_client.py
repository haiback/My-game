"""Async HTTP client for a local Ollama server.

Thin wrapper around the /api/generate endpoint. All failures raise
OllamaError so callers can decide how to degrade.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), transport=transport
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        options: dict[str, float | int] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                if not text:
                    raise OllamaError(f"Empty response from Ollama ({model})")
                return text
            except (httpx.HTTPError, OllamaError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                last_error = exc
                break

        raise OllamaError(f"Ollama request failed for {model}: {last_error}")

    async def close(self) -> None:
        await self._client.aclose()
