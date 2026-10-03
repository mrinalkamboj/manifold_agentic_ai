"""Concept: a tool can call a live HTTP API. The graph does not change."""

from langchain_core.messages import HumanMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph,END

from agentic_developer_bootcamp.session_9_10_MCP.lessons.fourth.entities import State
from agentic_developer_bootcamp.session_9_10_MCP.lessons.fourth.current_tools import TOOLS

#fetch llm
from agentic_developer_bootcamp.session_9_10_MCP.lessons.zero.init_llm import llm
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

print(graph.get_graph().draw_ascii())
print("--------------------------------")

output = graph.invoke({"messages": [HumanMessage(content="What is the current weather in London, UK?")]})
print(output["messages"][-1].content)
