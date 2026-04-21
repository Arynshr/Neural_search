"""
Unit tests for synthesis/groq_client.py
Tests: successful synthesis, retry on rate limit, API error handling, source capping.
"""
import pytest
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def mock_settings(monkeypatch):
    from neural_search import config
    mock = MagicMock()
    mock.groq_api_key = "test-key"
    mock.groq_model = "llama3-8b-8192"
    monkeypatch.setattr(config, "settings", mock)
    return mock


@pytest.fixture
def sample_chunks():
    return [
        {"source_file": f"doc{i}.pdf", "page": i, "text": f"relevant content {i}"}
        for i in range(7)
    ]


class TestGroqSynthesizer:
    def _make_mock_response(self, answer: str):
        response = MagicMock()
        response.choices[0].message.content = answer
        return response

    def test_successful_synthesis_returns_answer(self, mock_settings, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            client = MockGroq.return_value
            client.chat.completions.create.return_value = self._make_mock_response("The answer is 42.")
            from neural_search.synthesis.groq_client import GroqSynthesizer
            synth = GroqSynthesizer()
            result = synth.synthesize("What is the answer?", sample_chunks)
        assert result["answer"] == "The answer is 42."
        assert result["model"] == "llama3-8b-8192"

    def test_sources_used_capped_at_five(self, mock_settings, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = \
                self._make_mock_response("answer")
            from neural_search.synthesis.groq_client import GroqSynthesizer
            synth = GroqSynthesizer()
            result = synth.synthesize("query", sample_chunks)
        assert len(result["sources_used"]) == 5

    def test_sources_contain_file_and_page(self, mock_settings, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = \
                self._make_mock_response("answer")
            from neural_search.synthesis.groq_client import GroqSynthesizer
            synth = GroqSynthesizer()
            result = synth.synthesize("query", sample_chunks)
        for source in result["sources_used"]:
            assert "source_file" in source
            assert "page" in source

    def test_retries_on_rate_limit(self, mock_settings, sample_chunks):
        from groq import RateLimitError
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            with patch("neural_search.synthesis.groq_client.time.sleep"):
                client = MockGroq.return_value
                client.chat.completions.create.side_effect = [
                    RateLimitError("rate limit", response=MagicMock(), body={}),
                    self._make_mock_response("retry worked"),
                ]
                from neural_search.synthesis.groq_client import GroqSynthesizer
                synth = GroqSynthesizer()
                result = synth.synthesize("query", sample_chunks[:2])
        assert result["answer"] == "retry worked"

    def test_exhausted_retries_returns_fallback(self, mock_settings, sample_chunks):
        from groq import RateLimitError
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            with patch("neural_search.synthesis.groq_client.time.sleep"):
                client = MockGroq.return_value
                client.chat.completions.create.side_effect = RateLimitError(
                    "rate limit", response=MagicMock(), body={}
                )
                from neural_search.synthesis.groq_client import GroqSynthesizer
                synth = GroqSynthesizer()
                result = synth.synthesize("query", sample_chunks[:2], retries=3)
        assert "unable" in result["answer"].lower() or "try again" in result["answer"].lower()
        assert result["sources_used"] == []
