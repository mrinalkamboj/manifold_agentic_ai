import json
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from demo_schema import ImpactInput
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
import sqlite3
import uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import sys
from pathlib import Path
load_dotenv()

# Creating MultiServerMCPClient is sync. Fetching tools is async.
FAST_MCP_PATH = str(Path(__file__).resolve().parent / "fast_mcp.py")
mcp_client = MultiServerMCPClient({
    # stdio requires a command + args that spawn the server process
    "fast_mcp": {
        "transport": "stdio",
        "command": "python",
        "args": ["fast_mcp.py"],
    },
    # HTTP servers use transport="http" (not type="http")
    "docs-langchain": {
        "transport": "http",
        "url": "https://docs.langchain.com/mcp",
    },
})

async def load_mcp_tools():
    return await mcp_client.get_tools()

# create custom tool for perform_search
# @tool
# def perform_search(query: str) -> str:
#     """Perform a search on the web."""
#     return mcp_client.perform_search(query)

async def main():
    # MCP tools are async-only; sync graph.invoke() raises
    # StructuredTool does not support sync invocation.
    TOOLS = await load_mcp_tools()
    print(f"Tools: {[tool.name for tool in TOOLS]}")
    print("--------------------------------")
    model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(TOOLS)

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    async def assistant(state: State):
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition) # tools ? : END
    builder.add_edge("tools","assistant")
    builder.add_edge("assistant", END)

    graph = builder.compile(checkpointer=MemorySaver())

    #  generate a unique id
    thread_id = str(uuid.uuid4()) # random uuid for the thread
    thread_1 = {"configurable": {"thread_id": thread_id}}

    output = await graph.ainvoke({"messages": [HumanMessage(content="What is the impact of 60 failed requests out of 1200 total requests, with a baseline error rate of 2%?")]}, thread_1)
    print(output)
    # for msg in output["messages"]:
    #     role = msg.type.upper()
    #     print(f"\n--- [{role}] ---")

    #     if getattr(msg, "tool_calls", None):
    #         for call in msg.tool_calls:
    #             print(f"Tool Call: {call['name']}({call['args']})")

    #     if msg.content:
    #         print(msg.content)
    print("--------------------------------")
    output = await graph.ainvoke({"messages": [HumanMessage(content="What is the latest news on the stock market?")]}, thread_1)
    # I dont have search tool, so this will result in LLM generation of response not tool call
    print(output)
    print("--------------------------------")
    output = await graph.ainvoke({"messages": [HumanMessage(content="Tell me more about langchain deep agents?")]}, thread_1)
    # I dont have search tool, so this will result in LLM generation of response not tool call
    print(output)
    print("--------------------------------")
    print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    asyncio.run(main())