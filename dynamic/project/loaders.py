from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_files):
    docs = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source_file"] = pdf

        docs.extend(loaded_docs)

    return docs