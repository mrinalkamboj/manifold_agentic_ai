"""Concept: a tool in the same file. The model proposes the call. ToolNode runs it."""
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
load_dotenv()


@tool
def calculate_incident_impact(failed_requests: int, total_requests: int, baseline_error_rate_pct: float) -> dict:
    """Calculate request failure percentage and percentage-point change from baseline.
    Use counts from the same observation window.
    """
    rate = failed_requests / total_requests * 100
    return {"error_rate_pct": rate, "change_percentage_points": rate - baseline_error_rate_pct}


TOOLS = [calculate_incident_impact]
model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def assistant(state: State):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "assistant")
# tools_condition is a ready-made router:
# last AI message has tool_calls -> "tools", else -> END
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

print(graph.get_graph().draw_ascii())
print("--------------------------------")

needs_tool = graph.invoke({"messages": [HumanMessage(content="What is the impact of 60 failed requests out of 1200 total requests, baseline 2 percent?")]})
print(needs_tool["messages"][-1].content)
print("--------------------------------")
no_tool = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
print(no_tool["messages"][-1].content)
