from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    route: str
    answer: str


def route_question(state: GraphState):
    question = state["question"].lower()

    if "leave" in question or "pl" in question or "cl" in question or "sl" in question:
        return {"route": "leave"}
    else:
        return {"route": "general"}


def decide_next_node(state: GraphState):
    if state["route"] == "leave":
        return "leave_node"
    else:
        return "general_node"


def leave_node(state: GraphState):
    return {
        "answer": "This is a leave-related question. We should search the leave policy PDF."
    }


def general_node(state: GraphState):
    return {
        "answer": "This is a general question. No leave policy search is needed."
    }


graph_builder = StateGraph(GraphState)

graph_builder.add_node("route_question", route_question)
graph_builder.add_node("leave_node", leave_node)
graph_builder.add_node("general_node", general_node)

graph_builder.add_edge(START, "route_question")

graph_builder.add_conditional_edges(
    "route_question",
    decide_next_node,
    {
        "leave_node": "leave_node",
        "general_node": "general_node",
    }
)

graph_builder.add_edge("leave_node", END)
graph_builder.add_edge("general_node", END)

graph = graph_builder.compile()


result = graph.invoke({
    "question": "Tell me about today wheather?",
    "route": "",
    "answer": ""
})

print(result["answer"])