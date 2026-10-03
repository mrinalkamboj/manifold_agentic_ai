"""Concept: tools come from an MCP server. Load is async. Graph run must be async too."""

# Python Packages
import asyncio
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient

from agentic_developer_bootcamp.session_9_10_MCP.lessons.fifth.entities import State
from agentic_developer_bootcamp.session_9_10_MCP.lessons.zero.init_llm import llm

# Establish a connection with the MCP server
MCP_SERVER = str(Path(__file__).resolve().parent / "mcp_server.py") # Resolve MCP Server

# Create a MCP Client
mcp_client = MultiServerMCPClient({
    "incident": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [MCP_SERVER],
    },
})


async def main():
    tools = await mcp_client.get_tools()
    print("MCP tools:", [tool.name for tool in tools])
    #model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(tools)
    model = llm.bind_tools(tools)

    async def assistant(state: State):
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)
    graph = builder.compile()

    print(graph.get_graph().draw_ascii())
    print("--------------------------------")

    # MCP tools are async-only. graph.invoke() raises NotImplementedError.
    output = await graph.ainvoke({"messages": [HumanMessage(content="What is the impact of 100 failed requests out of 1200 total requests, baseline 2 percent?")]})
    print(output["messages"][-1].content)


# Initiate Async code
if __name__ == "__main__":
    asyncio.run(main())
