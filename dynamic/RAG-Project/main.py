from loaders import load_pdfs
from splitter import split_documents
from embeddings_store import create_vectorstore
from qa_chain import build_qa_chain
from employee_data import get_employee_by_id


docs = load_pdfs([
    "data/leave.pdf"
])

split_docs = split_documents(docs)
vectorstore = create_vectorstore(split_docs)
qa_chain = build_qa_chain(vectorstore)


print("\nLeave Policy Assistant Started")

employee_id = input("Enter Employee ID: ")

employee = get_employee_by_id(employee_id)

if employee is None:
    print("Employee not found.")
    exit()


print(f"\nWelcome {employee['name']}")
print(f"Grade/Band: {employee['grade']}")
print("Ask your question or type 'exit' to stop.")


while True:
    query = input("\nEnter your query: ")

    if query.lower() == "exit":
        break

    full_query = f"""
Employee Information:
Employee ID: {employee['employee_id']}
Name: {employee['name']}
Grade/Band: {employee['grade']}
PL Taken: {employee['PL_taken']}
CL Taken: {employee['CL_taken']}
SL Taken: {employee['SL_taken']}

User Question:
{query}
"""

    response = qa_chain.invoke({"query": full_query})

    answer = response["result"].replace("**", "")

    print("\nAnswer:")
    print(answer)