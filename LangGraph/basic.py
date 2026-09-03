from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    answer: str


def receive_question(state: GraphState):
    question = state["question"]

    return {
        "question": question
    }


def generate_answer(state: GraphState):
    question = state["question"]

    answer = f"You asked: {question}"

    return {
        "answer": answer
    }


graph_builder = StateGraph(GraphState)

graph_builder.add_node("receive_question", receive_question)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "receive_question")
graph_builder.add_edge("receive_question", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()

result = graph.invoke({
    "question": "What is Privilege Leave?",
    "answer": ""
})

print(result)