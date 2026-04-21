"""
Unit tests for synthesis/groq_client.py
Tests: successful synthesis, retry on rate limit, API error, source capping.
"""

import pytest
from unittest.mock import MagicMock, patch


def _mock_response(answer: str) -> MagicMock:
    """Build a realistic Groq response object."""
    message = MagicMock(content=answer)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


class TestGroqSynthesizer:
    def test_successful_synthesis_returns_answer(self, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            client = MockGroq.return_value
            client.chat.completions.create.return_value = _mock_response("The answer is 42.")

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("What is the answer?", sample_chunks)

        assert client.chat.completions.create.called
        assert result["answer"] == "The answer is 42."
        assert result["model"] == "llama3-8b-8192"
        assert "sources_used" in result

    def test_sources_capped_at_five_even_with_more_chunks(self, sample_chunks):
        assert len(sample_chunks) > 5

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            client = MockGroq.return_value
            client.chat.completions.create.return_value = _mock_response("answer")

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks)

        assert len(result["sources_used"]) == 5

    def test_sources_contain_required_fields(self, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            client = MockGroq.return_value
            client.chat.completions.create.return_value = _mock_response("answer")

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks)

        for source in result["sources_used"]:
            assert "source_file" in source
            assert "page" in source

    def test_retries_once_on_rate_limit_then_succeeds(self, sample_chunks):
        from groq import RateLimitError

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep") as mock_sleep:

            client = MockGroq.return_value
            client.chat.completions.create.side_effect = [
                RateLimitError("rate limit", response=MagicMock(), body={}),
                _mock_response("retry worked"),
            ]

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2])

        assert result["answer"] == "retry worked"
        assert client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    def test_exhausted_retries_returns_fallback_message(self, sample_chunks):
        from groq import RateLimitError

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep"):

            client = MockGroq.return_value
            client.chat.completions.create.side_effect = RateLimitError(
                "rate limit", response=MagicMock(), body={}
            )

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2], retries=3)

        assert result["sources_used"] == []
        assert result["model"] == "llama3-8b-8192"
        assert any(
            word in result["answer"].lower()
            for word in ("unable", "try again", "failed", "error")
        )

    def test_api_error_returns_fallback(self, sample_chunks):
        from groq import APIError

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep"):

            client = MockGroq.return_value
            client.chat.completions.create.side_effect = APIError(
                "server error", MagicMock(), {}
            )

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2])

        assert result["sources_used"] == []
        assert any(
            word in result["answer"].lower()
            for word in ("error", "failed", "unable")
        )

    def test_model_name_comes_from_settings(self, sample_chunks, patch_settings):
        patch_settings.groq_model = "custom-model"

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            client = MockGroq.return_value
            client.chat.completions.create.return_value = _mock_response("ok")

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:1])

        assert result["model"] == "custom-model"
