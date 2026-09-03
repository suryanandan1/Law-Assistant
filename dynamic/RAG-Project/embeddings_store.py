from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(split_docs):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en"
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embedding_model
    )

    return vectorstore