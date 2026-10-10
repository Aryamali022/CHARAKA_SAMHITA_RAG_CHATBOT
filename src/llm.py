"""The answer model: openai/gpt-oss-20b through NVIDIA's OpenAI-compatible API.

The key is read from NVIDIA_API_KEY in .env (never written in code). Model
and endpoint come from src/config.py and can be changed with LLM_MODEL and
LLM_BASE_URL. This is the only module that talks to the API, so tests use a
fake with the same complete() method.

gpt-oss is a reasoning model: it returns its reasoning separately
(reasoning_content) from the answer (content). Only the answer is used.
"""
from __future__ import annotations

import os

from src import config


class LLMError(RuntimeError):
    """The model could not produce an answer (no key, network, limits, ...)."""


class LLM:
    def __init__(self, model: str = config.LLM_MODEL, base_url: str = config.LLM_BASE_URL,
                 api_key: str | None = None):
        from openai import OpenAI

        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise LLMError("NVIDIA_API_KEY is not set. Add it to .env (see .env.example).")
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=2)

    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 1.0,
                 attempts: int = 2) -> str:
        """The model's answer text.

        Temperature 1.0 is what gpt-oss's makers recommend: at low temperatures
        this small reasoning model can fall into a loop, repeating one word
        until it runs out of tokens. Faithfulness to the passages comes from
        the instructions and the citation checks, not from temperature. An
        empty reply (such a loop) is retried once.
        """
        import openai

        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=temperature,
                )
            except openai.AuthenticationError as error:
                raise LLMError("The API key was rejected. Check NVIDIA_API_KEY in .env.") from error
            except openai.RateLimitError as error:
                raise LLMError("Rate limit reached. Wait a minute and try again.") from error
            except openai.APIConnectionError as error:
                raise LLMError("Could not reach the model API. Check the internet connection.") from error
            except openai.APIStatusError as error:
                raise LLMError(f"The model API returned an error ({error.status_code}).") from error

            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if content:
                return content
        reason = "it ran out of tokens while reasoning" if choice.finish_reason == "length" else \
            f"finish reason: {choice.finish_reason}"
        raise LLMError(f"The model returned no answer ({reason}).")
