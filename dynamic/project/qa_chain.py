from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


def build_qa_chain(vectorstore):
    load_dotenv()

    model = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    prompt_template = """
You are a strict document-based PDF Assistant.

Answer questions ONLY from the provided PDF context and the user information included inside the question.

Rules:
1. Answer ONLY from the uploaded PDF context.
2. Use user/employee information if it is present in the question.
3. If previous user data is provided and the PDF contains matching rules, calculate using that data.
4. Never assume or invent rules, policies, amounts, dates, eligibility, or formulas.
5. If the answer is missing, say exactly:
   "The answer is not available in the uploaded PDF documents."
6. Show calculations clearly when calculation is required.
7. Use tables when comparing multiple categories, leave types, bands, grades, or values.
8. Mention the source PDF name when it helps clarity.
9. Do not answer from general knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain