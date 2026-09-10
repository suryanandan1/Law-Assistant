import json
import time
from pathlib import Path

import streamlit as st
from loguru import logger

from src.logging_config import configure_logging
from src.rag_pipeline import RAGPipeline
from src.tracing import trace_turn

# ----------------------------------
# Page setup
# ----------------------------------

st.set_page_config(
    page_title="Indian Constitution Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
configure_logging("logs/streamlit.log")

LANGUAGES = {"English": "English", "हिन्दी (Hindi)": "Hindi"}
SAMPLE_QUESTIONS = [
    "What is Article 21?",
    "What are the Fundamental Rights?",
    "Explain the Directive Principles of State Policy",
    "Who can be removed by impeachment?",
]


@st.cache_resource
def load_pipeline():
    return RAGPipeline(top_k=15)


try:
    pipeline = load_pipeline()
except Exception as startup_error:
    logger.exception(startup_error)
    st.title(":material/balance: Indian Constitution Assistant")
    st.error("The assistant failed to start.", icon=":material/error:")
    st.exception(startup_error)
    st.caption(
        "Check that GOOGLE_CLOUD_PROJECT and Google credentials are set (copy "
        "`.env.example` to `.env`), and that `processed/faiss.index` exists — "
        "build it with `python ingest.py`."
    )
    st.stop()


@st.cache_data
def load_manifest():
    path = Path("processed/ingest_manifest.json")
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    logger.info(f"Ingest manifest: {manifest}")
    return manifest


manifest = load_manifest()
messages = st.session_state.setdefault("messages", [])
pending_question = None

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:
    st.header("Constitution assistant")

    if st.button("New chat", icon=":material/add:", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()

    language_label = st.segmented_control(
        "Response language",
        list(LANGUAGES.keys()),
        default=next(iter(LANGUAGES)),
        width="stretch",
    )
    st.session_state["language"] = LANGUAGES.get(language_label, "English")

    st.markdown("#### System status")

    faiss_index = getattr(pipeline.retriever.vector_store, "index", None)
    vector_count = faiss_index.ntotal if faiss_index is not None else 0
    chunk_count = len(pipeline.retriever.chunk_lookup)
    gemini_ok = st.session_state.get("gemini_ok")  # None until the first answer

    def status_badge(label, ok):
        if ok is None:
            st.badge(label, icon=":material/help:", color="orange")
        elif ok:
            st.badge(label, icon=":material/check_circle:", color="green")
        else:
            st.badge(label, icon=":material/error:", color="red")

    status_badge(f"FAISS index · {vector_count:,} vectors", vector_count > 0)
    status_badge(f"Retriever · {chunk_count:,} chunks", chunk_count > 0)
    status_badge(
        "Gemini connected" if gemini_ok else "Gemini not verified yet",
        gemini_ok,
    )

    with st.container(border=True):
        reranker = (
            pipeline.retriever.rerank_model_name if pipeline.retriever.rerank_enabled else "off"
        )
        st.markdown(
            f"**Embedding model**  \n`{pipeline.retriever.model_name}`\n\n"
            f"**Reranker**  \n`{reranker}`\n\n"
            f"**LLM**  \n`{pipeline.llm.model}` · Vertex AI\n\n"
            f"**Vector database**  \nFAISS"
        )

    if manifest.get("built_at"):
        st.caption(f"Index built {manifest['built_at'][:10]}")

# ----------------------------------
# Header
# ----------------------------------

st.title(":material/balance: Indian Constitution Assistant")
st.caption(
    "Ask questions about the Constitution — grounded in the source document, with page citations."
)

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not messages:
    st.caption("Try one of these to get started")
    picked = st.pills(
        "Sample questions",
        SAMPLE_QUESTIONS,
        label_visibility="collapsed",
    )
    if picked:
        pending_question = picked


def build_history_pairs(all_messages):
    """Turn the flat message log into (question, answer) pairs for follow-up context."""
    pairs = []
    for i in range(0, len(all_messages) - 1, 2):
        user_msg, assistant_msg = all_messages[i], all_messages[i + 1]
        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
            pairs.append({"question": user_msg["content"], "answer": assistant_msg["content"]})
    return pairs


def show_result(result):
    if result["pages"]:
        with st.expander("Supporting pages", icon=":material/description:"):
            st.write(", ".join(map(str, result["pages"])))

    with st.expander("Retrieval statistics", icon=":material/timer:"):
        stats = (
            ("Retrieval", "retrieval_time"),
            ("Generation", "generation_time"),
            ("Total", "total_time"),
        )
        for column, (label, key) in zip(st.columns(3), stats):
            column.metric(label, f"{result[key]} s", border=True)

    with st.expander("Retrieved sources", icon=":material/menu_book:"):
        for index, chunk in enumerate(result.get("sources", [])[:10], 1):
            with st.container(border=True):
                meta = f"Source {index} · Page {chunk['page']}"
                if chunk.get("heading"):
                    meta += f" · {chunk['heading']}"
                score = chunk.get("rerank_score", chunk.get("score"))
                meta += f" · Score {score}"
                st.caption(meta)
                st.markdown(chunk["text"][:600])


question = st.chat_input("Ask a question…", submit_mode="disable") or pending_question

if question:
    history = build_history_pairs(messages)
    language = st.session_state.get("language", "English")
    turn_start = time.perf_counter()

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            prep = pipeline.retrieve_and_build(question, history=history, language=language)

            search_query = prep.get("search_query", question)

            if prep["answer"] is not None:
                # Nothing relevant retrieved — no generation needed.
                st.markdown(prep["answer"])
                messages.append({"role": "assistant", "content": prep["answer"]})
                trace_turn(
                    question=question,
                    search_query=search_query,
                    language=language,
                    sources=[],
                    pages=[],
                    retrieval_time=prep["retrieval_time"],
                    generation_time=0,
                    answer=prep["answer"],
                )
            else:
                gen_start = time.perf_counter()
                answer = st.write_stream(pipeline.llm.stream_answer(prep["prompt"]))
                st.session_state["gemini_ok"] = True
                messages.append({"role": "assistant", "content": answer})
                generation_time = time.perf_counter() - gen_start
                show_result(
                    {
                        "answer": answer,
                        "pages": prep["pages"],
                        "sources": prep["sources"],
                        "retrieval_time": round(prep["retrieval_time"], 4),
                        "generation_time": round(generation_time, 4),
                        "total_time": round(time.perf_counter() - turn_start, 4),
                    }
                )
                trace_turn(
                    question=question,
                    search_query=search_query,
                    language=language,
                    sources=prep["sources"],
                    pages=prep["pages"],
                    retrieval_time=prep["retrieval_time"],
                    generation_time=generation_time,
                    answer=answer,
                )
        except Exception as error:
            logger.exception(error)
            st.session_state["gemini_ok"] = False
            st.error(f"Could not answer: {error}", icon=":material/error:")
            messages.pop()  # drop the unanswered turn so history stays aligned
