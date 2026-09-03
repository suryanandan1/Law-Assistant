import time
import traceback
import re

from loguru import logger

from src.context_builder import ContextBuilder
from src.llm_service import MistralService
from src.retriever import Retriever


class RAGPipeline:
    """Retrieve document context and generate a grounded answer."""

    def __init__(self, top_k=15):
        logger.info("Initializing RAG Pipeline...")
        self.top_k = top_k
        try:
            self.retriever = Retriever(top_k=top_k)
            self.context_builder = ContextBuilder()
            self.llm = MistralService()
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

    @staticmethod
    def _format_answer(answer: str) -> str:
        """Keep the model's answer sections readable in the UI."""
        return re.sub(r"\s*(Supporting Pages:|Confidence:)\s*", r"\n\n\1 ", answer).strip()

    def ask(self, question: str):
        total_start = time.perf_counter()
        retrieval_time = generation_time = 0
        try:
            retrieval_start = time.perf_counter()
            sources = self.retriever.retrieve(question, top_k=self.top_k)["results"]
            retrieval_time = time.perf_counter() - retrieval_start
            if not sources:
                return self._result("The document does not contain this information.", [], retrieval_time, 0, total_start, sources=[])

            context = self.context_builder.build_context(question, sources)
            generation_start = time.perf_counter()
            answer = self._format_answer(self.llm.generate_answer(context["prompt"]))
            generation_time = time.perf_counter() - generation_start
            logger.info(f"Question processed | Retrieval={retrieval_time:.3f}s | Generation={generation_time:.3f}s | Total={time.perf_counter() - total_start:.3f}s")
            return self._result(answer, context["pages"], retrieval_time, generation_time, total_start, sources=sources)
        except Exception as error:
            logger.error(f"Pipeline Failure: {error}\n{traceback.format_exc()}")
            return self._result("An internal error occurred while processing the request.", [], retrieval_time, generation_time, total_start, sources=[], error=str(error))


def setup_logging():
    logger.remove()
    logger.add("logs/rag_pipeline.log", rotation="10 MB", retention=10, level="INFO")
    logger.add(lambda message: print(message, end=""))


def test():
    setup_logging()
    pipeline = RAGPipeline(top_k=15)
    while (question := input("\nQuestion: ")).lower() not in {"exit", "quit"}:
        result = pipeline.ask(question)
        print(f"\n{'=' * 80}\n\nANSWER:\n{result['answer']}\n\nPAGES:\n{result['pages']}\n\nRETRIEVAL TIME:\n{result['retrieval_time']} sec\n\nGENERATION TIME:\n{result['generation_time']} sec\n\nTOTAL TIME:\n{result['total_time']} sec")


if __name__ == "__main__":
    test()
