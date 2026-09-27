"""Concept: same graph as 02. Tools are imported from incident_tools.py."""
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
from incident_tools import TOOLS
load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def assistant(state: State):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

print("Imported tools:", [tool.name for tool in TOOLS])
print(graph.get_graph().draw_ascii())
print("--------------------------------")

output = graph.invoke({"messages": [HumanMessage(content="What is the checkout runbook, then the impact of 60 failed out of 1200 with baseline 2 percent?")]})
print(output["messages"][-1].content)
