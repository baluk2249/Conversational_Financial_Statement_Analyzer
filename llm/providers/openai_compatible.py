"""OpenAI-compatible chat-completions adapter using environment credentials."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm.base import LLMProviderError


class OpenAICompatibleClient:
    """Call an OpenAI-compatible endpoint without coupling agent logic to it."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 30,
    ) -> None:
        if not api_key.strip():
            raise LLMProviderError("An API key is required for this provider")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must use HTTP or HTTPS")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> OpenAICompatibleClient:
        """Create a client from OPENAI_API_KEY and optional endpoint settings."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured")
        return cls(
            api_key,
            base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError("The provider returned an empty completion")
            return content.strip()
        except LLMProviderError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMProviderError(
                f"The language-model provider request failed: {error}"
            ) from error
