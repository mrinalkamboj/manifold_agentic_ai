"""Concept: same graph as 02. Tools are imported from incident_tools.py."""

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph, END

#Import llm
from agentic_developer_bootcamp.session_9_10_MCP.lessons.zero.init_llm import llm

#Import tools
from agentic_developer_bootcamp.session_9_10_MCP.lessons.third.incident_tools import TOOLS

#Import state
from agentic_developer_bootcamp.session_9_10_MCP.lessons.third.entities import State

#Fetching the llm object separately and not binding the tools here
#model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(TOOLS)
model = llm.bind_tools(TOOLS)


def assistant(state: State):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant",END)
graph = builder.compile()

print("Imported tools:", [tool.name for tool in TOOLS])
print(graph.get_graph().draw_ascii())
print("--------------------------------")

output = graph.invoke({"messages": [HumanMessage(content="What is the checkout runbook, then the impact of 60 failed out of 1200 with baseline 2 percent?")]})
print(output["messages"][-1].content)
