import time
import random
from loguru import logger
from groq import Groq, RateLimitError, APIError
from neural_search.config import settings
from neural_search.synthesis.prompt import build_prompt

_MAX_BACKOFF = 32  # seconds


class GroqSynthesizer:
    def __init__(self):
        settings.assert_groq_configured()
        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        self._model = settings.groq_model

    def synthesize(self, query: str, chunks: list, retries: int = 3) -> dict:
        if not query.strip():
            return {
                "answer": "",
                "sources_used": [],
                "model": self._model,
            }

        context_chunks = chunks[:5]
        prompt = build_prompt(query, context_chunks)

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

                if not response.choices:
                    raise RuntimeError("Empty response from Groq")

                answer = (response.choices[0].message.content or "").strip()

                return {
                    "answer": answer,
                    "sources_used": [
                        {
                            "source_file": _get(c, "source_file"),
                            "page": _get(c, "page"),
                        }
                        for c in context_chunks
                    ],
                    "model": self._model,
                }

            except RateLimitError:
                base = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait = min(base + jitter, _MAX_BACKOFF)
                logger.warning(
                    f"Groq rate limit — retrying in {wait:.1f}s (attempt {attempt}/{retries})"
                )
                time.sleep(wait)

            except APIError as e:
                logger.error(f"Groq API error: {e}")
                break

            except Exception as e:
                logger.error(f"Unexpected synthesis error: {e}")
                break

        return {
            "answer": "Unable to generate answer — please try again.",
            "sources_used": [],
            "model": self._model,
        }


def _get(chunk, key: str):
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)
