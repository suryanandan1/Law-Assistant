import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

load_dotenv()

# A Vertex call blocks the whole Streamlit run, so bound it and let the SDK
# retry transient failures with backoff instead of failing the answer outright.
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))
RETRY_ATTEMPTS = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "3"))
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# Deterministic by default: a legal assistant should give the same answer to the
# same question. GEMINI_MAX_TOKENS=0 leaves the output uncapped (safer with the
# 2.5 "thinking" models, whose reasoning tokens count against the cap).
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "0"))

# Grounded RAG over supplied statutory text does not need chain-of-thought.
# Disabling it (budget 0) is faster, cheaper and much closer to reproducible.
# Set -1 for the model's dynamic thinking, or a positive token budget.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

# Keep echoed conversation turns from ballooning (or poisoning) the prompt.
MAX_HISTORY_TURNS = 3
MAX_HISTORY_QUESTION_CHARS = 500
MAX_HISTORY_ANSWER_CHARS = 800


class GeminiService:
    def __init__(self):

        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

        if not self.project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and fill it in."
            )

        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=RETRY_ATTEMPTS,
                    initial_delay=1.0,
                    max_delay=16.0,
                    http_status_codes=RETRYABLE_STATUS_CODES,
                ),
            ),
        )

    def _config(self) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            top_p=1.0,
            candidate_count=1,
        )
        if GEMINI_MAX_TOKENS:
            config.max_output_tokens = GEMINI_MAX_TOKENS
        if GEMINI_THINKING_BUDGET >= 0:
            config.thinking_config = types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
        return config

    def _generate(self, prompt: str) -> str:
        """One Vertex call. Timeout + retry/backoff are configured on the client."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(),
            )
        except Exception as error:
            logger.exception(f"Vertex generate_content failed: {error}")
            raise
        return (response.text or "").strip()

    def generate_answer(self, prompt: str) -> str:
        answer = self._generate(prompt)
        if not answer:
            raise RuntimeError("The model returned an empty response.")
        return answer

    def stream_answer(self, prompt: str):
        """Yield answer text as the model produces it.

        If the stream fails before emitting anything, fall back to a single
        non-streamed call; if it fails mid-response, re-raise (the caller has
        already shown partial text and cannot cleanly restart).
        """
        produced = False
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=self._config(),
            ):
                text = getattr(chunk, "text", "") or ""
                if text:
                    produced = True
                    yield text
        except Exception as error:
            if produced:
                logger.exception(f"Vertex streaming failed mid-response: {error}")
                raise
            logger.warning(
                f"Streaming failed before any output ({error}); "
                f"falling back to a non-streamed call."
            )
            yield self.generate_answer(prompt)
            return

        if not produced:
            raise RuntimeError("The model returned an empty response.")

    def build_search_query(self, question: str, history: list) -> str:
        """
        Collapse a (possibly non-English, possibly follow-up) question into a
        standalone English search query the retriever can embed well.

        This lets the retrieval step stay on a single, English-only embedding
        model while still supporting multilingual questions and conversational
        follow-ups ("what about clause 2?").
        """
        if history:
            history_text = "\n".join(
                f"Q: {turn['question'][:MAX_HISTORY_QUESTION_CHARS]}\n"
                f"A: {turn['answer'][:MAX_HISTORY_ANSWER_CHARS]}"
                for turn in history[-MAX_HISTORY_TURNS:]
            )
        else:
            history_text = "None"

        prompt = f"""Rewrite the LATEST QUESTION below into a single standalone English search query, suitable for a document search engine.

RULES:
- If the question is a follow-up (uses "it", "that", "this article", etc.), resolve it using the CONVERSATION HISTORY so it stands alone.
- If the question is written in Hindi or any other language, translate it to English.
- Do not answer the question. Only return the rewritten query text, nothing else, no quotes.

CONVERSATION HISTORY:
{history_text}

LATEST QUESTION:
{question}

STANDALONE ENGLISH SEARCH QUERY:"""

        try:
            rewritten = self._generate(prompt).strip('"').strip()
            return rewritten or question
        except Exception as error:
            logger.warning(f"Query rewrite failed, using the original question: {error}")
            return question
