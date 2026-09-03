from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    employee_id: str
    question: str
    employee: dict
    documents: list
    answer: str


def get_employee_data(state: LeaveState):
    employee_id = state["employee_id"]

    employee = {
        "employee_id": employee_id,
        "name": "Rahul",
        "grade": "A",
        "pl_taken": 7,
        "cl_taken": 2,
        "sl_taken": 1
    }

    return {
        "employee": employee
    }


def retrieve_policy(state: LeaveState):
    question = state["question"]

    documents = [
        "Privilege Leave is 18 days per year.",
        "Casual Leave is 7 days per year.",
        "Sick Leave is 7 days per year."
    ]

    return {
        "documents": documents
    }


def generate_answer(state: LeaveState):
    employee = state["employee"]
    documents = state["documents"]

    pl_total = 18
    pl_taken = employee["pl_taken"]
    pl_remaining = pl_total - pl_taken

    answer = f"""
Employee: {employee['name']}
Grade: {employee['grade']}

Privilege Leave total: {pl_total}
Privilege Leave taken: {pl_taken}
Privilege Leave remaining: {pl_remaining}
"""

    return {
        "answer": answer
    }


graph_builder = StateGraph(LeaveState)

graph_builder.add_node("get_employee_data", get_employee_data)
graph_builder.add_node("retrieve_policy", retrieve_policy)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "get_employee_data")
graph_builder.add_edge("get_employee_data", "retrieve_policy")
graph_builder.add_edge("retrieve_policy", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph_builder.add_edge("")

graph = graph_builder.compile()

result = graph.invoke({
    "employee_id": "E101",
    "question": "How many PL leaves do I have?",
    "employee": {},
    "documents": [],
    "answer": ""
})

print(result["answer"])