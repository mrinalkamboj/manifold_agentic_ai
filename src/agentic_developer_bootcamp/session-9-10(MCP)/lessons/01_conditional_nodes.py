"""Concept: you write the router. It is ordinary Python that returns the next node name."""
from typing import Literal, TypedDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
load_dotenv()


class Severity(BaseModel):
    level: Literal["high", "low"]
    reason: str


class State(TypedDict):
    messages: list[BaseMessage]
    severity: str
    result: str


classifier = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(Severity)


def classify(state: State):
    decision = classifier.invoke(state["messages"])
    print("Model classified severity as:", decision.level, "-", decision.reason)
    return {"severity": decision.level}


def page_oncall(state: State):
    return {"result": "HIGH: page the on-call engineer now."}


def log_ticket(state: State):
    return {"result": "LOW: log a ticket. No page."}


# You write this. Return the next node name as a string.
def route(state: State) -> str:
    if state["severity"] == "high":
        return "page_oncall"
    return "log_ticket"


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("page_oncall", page_oncall)
builder.add_node("log_ticket", log_ticket)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {"page_oncall": "page_oncall", "log_ticket": "log_ticket"}, # string output from function : node name
)
builder.add_edge("page_oncall", END)
builder.add_edge("log_ticket", END)
graph = builder.compile()

print(graph.get_graph().draw_ascii())
print("--------------------------------")

high = graph.invoke({"messages": [HumanMessage(content="Checkout is failing for 40 percent of users.")]})
print(high["result"])
print("--------------------------------")
low = graph.invoke({"messages": [HumanMessage(content="user is using thi")]})
print(low["result"])
