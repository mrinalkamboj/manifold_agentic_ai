"""Concept: tools come from an MCP server. Load is async. Graph run must be async too."""
import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv()

MCP_SERVER = str(Path(__file__).resolve().parent / "mcp_server.py")
mcp_client = MultiServerMCPClient({
    "incident": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [MCP_SERVER],
    },
})


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    tools = await mcp_client.get_tools()
    print("MCP tools:", [tool.name for tool in tools])
    model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(tools)

    async def assistant(state: State):
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    graph = builder.compile()

    print(graph.get_graph().draw_ascii())
    print("--------------------------------")

    # MCP tools are async-only. graph.invoke() raises NotImplementedError.
    output = await graph.ainvoke({"messages": [HumanMessage(content="What is the impact of 60 failed requests out of 1200 total requests, baseline 2 percent?")]})
    print(output["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
