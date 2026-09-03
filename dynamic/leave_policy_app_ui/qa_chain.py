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
You are a strict Leave Policy Assistant.

Answer employee leave-related questions ONLY from the provided context and employee information given inside the question.

Rules:
1. Answer ONLY from the provided context.
2. Use employee information if it is present in the question.
3. Never assume or invent policy information.
4. If policy information is missing, say exactly:
   "The answer is not available in the provided documents."
5. Clearly separate PL, CL, and SL.
6. For balance questions, calculate:
   Remaining Leave = Total Entitlement - Leave Already Taken
7. Show calculations explicitly.
8. Use tables when multiple leave categories apply.
9. Do not mix PL, CL, and SL.
10. Do not generate HR rules from general knowledge.
11. Give answer according to the band.

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