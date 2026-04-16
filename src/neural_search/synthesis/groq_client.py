import time
from loguru import logger
from groq import Groq, RateLimitError, APIError
from neural_search.config import settings
from neural_search.synthesis.prompt import build_prompt


class GroqSynthesizer:
    def __init__(self):
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    def synthesize(self, query: str, chunks: list[dict], retries: int = 3) -> dict:
        """
        Generate an answer from top retrieved chunks.

        Returns:
            {answer, sources_used, model}
        """
        # Cap to top 5 chunks to stay within context window
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
                answer = response.choices[0].message.content.strip()
                return {
                    "answer": answer,
                    "sources_used": [
                        {"source_file": c["source_file"], "page": c["page"]}
                        for c in context_chunks
                    ],
                    "model": self._model,
                }
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Groq rate limit — retrying in {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            except APIError as e:
                logger.error(f"Groq API error: {e}")
                break

        return {"answer": "Unable to generate answer — please try again.", "sources_used": [], "model": self._model}
