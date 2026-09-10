import os
import re
import time
import traceback

from loguru import logger

from src.context_builder import ContextBuilder
from src.llm_service import GeminiService
from src.logging_config import configure_logging
from src.retriever import Retriever
from src.tracing import trace_turn


class RAGPipeline:
    """Retrieve document context and generate a grounded answer."""

    def __init__(self, top_k=15, retriever=None, context_builder=None, llm=None):
        # Collaborators can be injected (tests); otherwise built normally.
        logger.info("Initializing RAG Pipeline...")
        self.top_k = top_k
        try:
            self.retriever = retriever or Retriever(top_k=top_k)
            self.context_builder = context_builder or ContextBuilder()
            self.llm = llm or GeminiService()
            logger.info("RAG Pipeline Ready")
        except Exception as error:
            logger.exception(f"Startup Failure: {error}")
            raise

    @staticmethod
    def _result(answer, pages, retrieval_time, generation_time, total_start, **extra):
        return {
            "answer": answer,
            "pages": pages,
            "retrieval_time": round(retrieval_time, 4),
            "generation_time": round(generation_time, 4),
            "total_time": round(time.perf_counter() - total_start, 4),
            **extra,
        }

    # Reference-led openers that cannot stand alone as a search query.
    _REFERENCE_WORDS = {
        "it",
        "its",
        "that",
        "this",
        "they",
        "them",
        "these",
        "those",
        "he",
        "she",
        "him",
        "her",
        "above",
        "below",
    }

    @classmethod
    def _needs_query_rewrite(cls, question: str, history: list) -> bool:
        """Decide whether to spend an extra LLM call turning the question into a
        standalone English search query.

        QUERY_REWRITE=always | never overrides the heuristic; default "auto"
        rewrites follow-ups, non-English questions, and reference-led openers
        ("what about that clause?") that have no history to lean on.
        """
        mode = os.getenv("QUERY_REWRITE", "auto").strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False

        if history:
            return True
        if any(ord(char) > 127 for char in question):
            return True

        first_words = re.findall(r"[a-z]+", question.lower())[:1]
        return bool(first_words) and first_words[0] in cls._REFERENCE_WORDS

    NO_INFO_ANSWER = "The document does not contain this information."

    def retrieve_and_build(self, question: str, history: list = None, language: str = "English"):
        """Everything up to answer generation: query rewrite, retrieval, context.

        Returns a dict with either a ready ``answer`` (short-circuit: nothing
        relevant retrieved) or a ``prompt`` for the caller to generate from.
        """
        history = history or []

        search_query = question
        if self._needs_query_rewrite(question, history):
            search_query = self.llm.build_search_query(question, history)
            logger.info(f"Rewritten search query: {search_query!r}")

        retrieval_start = time.perf_counter()
        sources = self.retriever.retrieve(search_query, top_k=self.top_k)["results"]
        retrieval_time = time.perf_counter() - retrieval_start

        if not sources:
            return {
                "answer": self.NO_INFO_ANSWER,
                "prompt": None,
                "pages": [],
                "sources": [],
                "search_query": search_query,
                "retrieval_time": retrieval_time,
            }

        context = self.context_builder.build_context(
            question, sources, history=history, language=language
        )
        return {
            "answer": None,
            "prompt": context["prompt"],
            "pages": context["pages"],
            "sources": sources,
            "search_query": search_query,
            "retrieval_time": retrieval_time,
        }

    def ask(self, question: str, history: list = None, language: str = "English"):
        """Non-streaming convenience wrapper (CLI, tests, programmatic callers)."""
        total_start = time.perf_counter()
        retrieval_time = generation_time = 0
        try:
            prep = self.retrieve_and_build(question, history, language)
            retrieval_time = prep["retrieval_time"]
            search_query = prep.get("search_query", question)

            if prep["answer"] is not None:
                trace_turn(
                    question=question,
                    search_query=search_query,
                    language=language,
                    sources=[],
                    pages=[],
                    retrieval_time=retrieval_time,
                    generation_time=0,
                    answer=prep["answer"],
                )
                return self._result(
                    prep["answer"],
                    prep["pages"],
                    retrieval_time,
                    0,
                    total_start,
                    sources=prep["sources"],
                )

            generation_start = time.perf_counter()
            answer = self.llm.generate_answer(prep["prompt"])
            generation_time = time.perf_counter() - generation_start
            total_time = time.perf_counter() - total_start
            logger.info(
                f"Question processed | Retrieval={retrieval_time:.3f}s | "
                f"Generation={generation_time:.3f}s | Total={total_time:.3f}s"
            )
            trace_turn(
                question=question,
                search_query=search_query,
                language=language,
                sources=prep["sources"],
                pages=prep["pages"],
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                answer=answer,
            )
            return self._result(
                answer,
                prep["pages"],
                retrieval_time,
                generation_time,
                total_start,
                sources=prep["sources"],
            )
        except Exception as error:
            logger.error(f"Pipeline Failure: {error}\n{traceback.format_exc()}")
            trace_turn(
                question=question,
                search_query=question,
                language=language,
                sources=[],
                pages=[],
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                answer="",
                error=str(error),
            )
            return self._result(
                "An internal error occurred while processing the request.",
                [],
                retrieval_time,
                generation_time,
                total_start,
                sources=[],
                error=str(error),
            )


def test():
    configure_logging("logs/rag_pipeline.log")
    pipeline = RAGPipeline(top_k=15)
    while (question := input("\nQuestion: ")).lower() not in {"exit", "quit"}:
        result = pipeline.ask(question)
        print(
            f"\n{'=' * 80}\n\n"
            f"ANSWER:\n{result['answer']}\n\n"
            f"PAGES:\n{result['pages']}\n\n"
            f"RETRIEVAL TIME:\n{result['retrieval_time']} sec\n\n"
            f"GENERATION TIME:\n{result['generation_time']} sec\n\n"
            f"TOTAL TIME:\n{result['total_time']} sec"
        )


if __name__ == "__main__":
    test()
