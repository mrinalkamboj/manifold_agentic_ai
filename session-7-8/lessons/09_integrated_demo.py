"""Integrated demo: you route first, then one assistant uses every tool style."""
import asyncio
import json
import sys
from pathlib import Path
from urllib.request import urlopen
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from incident_tools import lookup_runbook
load_dotenv()


@tool
def format_alert(service: str, summary: str) -> str:
    """Write a short on-call alert line. Local tool, same file."""
    return f"ALERT [{service}]: {summary}"


@tool
def get_live_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather from the Open-Meteo public API. Bangalore is 12.97, 77.59."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,weather_code"
    )
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read()).get("current", {})


class PathChoice(BaseModel):
    path: Literal["work", "chat"]


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    path: str


classifier = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(PathChoice)
MCP_SERVER = str(Path(__file__).resolve().parent / "mcp_server.py")
mcp_client = MultiServerMCPClient({
    "incident": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [MCP_SERVER],
    },
})


def route(state: State) -> str:
    return state["path"]


async def main():
    local_tools = [format_alert]
    module_tools = [lookup_runbook]
    live_tools = [get_live_weather]
    mcp_tools = await mcp_client.get_tools()
    tools = local_tools + module_tools + live_tools + mcp_tools

    print("local:", [t.name for t in local_tools])
    print("module:", [t.name for t in module_tools])
    print("live:", [t.name for t in live_tools])
    print("mcp:", [t.name for t in mcp_tools])

    desk = ChatOpenAI(model="gpt-5.4-mini").bind_tools(tools)
    talk = ChatOpenAI(model="gpt-5.4-mini")

    async def classify(state: State):
        decision = await classifier.ainvoke(state["messages"])
        print("You-route path:", decision.path)
        return {"path": decision.path}

    async def assistant(state: State):
        return {"messages": [await desk.ainvoke(state["messages"])]}

    async def chat(state: State):
        return {"messages": [await talk.ainvoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("chat", chat)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {"work": "assistant", "chat": "chat"},
    )
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("chat", END)
    graph = builder.compile()

    print(graph.get_graph().draw_ascii())
    print("--------------------------------")

    work = await graph.ainvoke({"messages": [HumanMessage(content=(
        "Checkout is down. 60 failed requests out of 1200, baseline 2 percent. "
        "Get the checkout runbook, the impact, Bangalore weather, and format an alert."
    ))]})
    print(work["messages"][-1].content)
    print("--------------------------------")

    chat_out = await graph.ainvoke({"messages": [HumanMessage(content="What is LangGraph in one sentence?")]})
    print(chat_out["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
