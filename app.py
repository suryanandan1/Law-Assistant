import time

import streamlit as st
from loguru import logger

from src.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Indian Constitution Assistant", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.main{padding-top:1rem}.stChatMessage{border-radius:12px}</style>", unsafe_allow_html=True)
logger.add("logs/streamlit.log", rotation="10 MB")


@st.cache_resource
def load_pipeline():
    return RAGPipeline(top_k=15)


pipeline = load_pipeline()
messages = st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title("⚙️ System Status")
    st.success("Backend Loaded")
    st.divider()
    st.subheader("Models")
    st.write("**Embedding:** BAAI/bge-large-en-v1.5")
    st.write("**LLM:** mistral-large-latest")
    st.write("**Vector DB:** FAISS")
    st.divider()
    st.metric("Indexed Chunks", len(pipeline.retriever.chunk_lookup))
    st.divider()
    st.write("✅ FAISS Loaded\n\n✅ Retriever Ready\n\n✅ Mistral Connected")

st.title("Indian Constitution Assistant")
st.caption("Ask questions about your legal document.")
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def show_result(result):
    if result["pages"]:
        with st.expander("📄 Supporting Pages"):
            st.write(", ".join(map(str, result["pages"])))

    with st.expander("📊 Retrieval Statistics"):
        for column, (label, key) in zip(st.columns(3), (("Retrieval", "retrieval_time"), ("Generation", "generation_time"), ("Total", "total_time"))):
            column.metric(label, f"{result[key]} s")

    with st.expander("📚 Retrieved Sources"):
        for index, chunk in enumerate(result.get("sources", [])[:10], 1):
            st.markdown(f"### Source {index}\n\n**Page:** {chunk['page']}\n\n**Score:** {chunk['score']}\n\n{chunk['text'][:1200]}")


if question := st.chat_input("Ask a question..."):
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = pipeline.ask(question)
            answer, placeholder = result["answer"], st.empty()
            streamed = ""
            for word in answer.split():
                streamed += f"{word} "
                placeholder.markdown(streamed)
                time.sleep(0.01)
            messages.append({"role": "assistant", "content": answer})
            show_result(result)
        except Exception as error:
            logger.exception(error)
            st.error(f"Error: {error}")
