import time
import random
from loguru import logger
from groq import Groq, RateLimitError, APIError
from neural_search.config import settings as global_settings
from neural_search.synthesis.prompt import build_prompt

_MAX_BACKOFF = 32  # seconds

# Current recommended model — update here when Groq deprecates again
_DEFAULT_MODEL = "llama-3.1-8b-instant"

# Track deprecated models explicitly
_DEPRECATED_MODELS = {"llama3-8b-8192"}

_FALLBACK_RESPONSE = {
    "answer": "Unable to generate answer — please try again.",
    "sources_used": [],
}


class GroqSynthesizer:
    def __init__(self, settings_obj=None):
        # Allow dependency injection for testability
        self._settings = settings_obj or global_settings

        self._settings.assert_groq_configured()
        self._client = Groq(api_key=self._settings.groq_api_key)

        # Handle deprecated model fallback
        if self._settings.groq_model in _DEPRECATED_MODELS:
            self._model = _DEFAULT_MODEL
            logger.warning(
                f"Model '{self._settings.groq_model}' is decommissioned. "
                f"Switching to '{_DEFAULT_MODEL}'. Update GROQ_MODEL in .env."
            )
        else:
            self._model = self._settings.groq_model

    def synthesize(self, query: str, chunks: list, retries: int = 3) -> dict:
        context_chunks = chunks[:5]
        prompt = build_prompt(query, context_chunks)

        sources = [
            {"source_file": _get(c, "source_file"), "page": _get(c, "page")}
            for c in context_chunks
        ]

        for attempt in range(1, retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                return {
                    "answer": response.choices[0].message.content.strip(),
                    "sources_used": sources,
                    "model": self._model,
                }

            except RateLimitError:
                base = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait = min(base + jitter, _MAX_BACKOFF)

                logger.warning(
                    f"Groq rate limit — retrying in {wait:.1f}s "
                    f"(attempt {attempt}/{retries})"
                )
                time.sleep(wait)

            except APIError as e:
                logger.error(f"Groq API error: {e}")
                break

        return {**_FALLBACK_RESPONSE, "model": self._model}


def _get(chunk, key: str):
    if isinstance(chunk, dict):
        return chunk[key]
    return getattr(chunk, key)
