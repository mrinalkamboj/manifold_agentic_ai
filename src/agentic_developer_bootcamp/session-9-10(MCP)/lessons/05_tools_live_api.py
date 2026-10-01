"""Concept: a tool can call a live HTTP API. The graph does not change."""
import json
from urllib.request import urlopen
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
load_dotenv()


@tool
def get_live_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather from the Open-Meteo public API. No API key.
    Bangalore is latitude 12.97 longitude 77.59. Use decimal degrees.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,weather_code"
    )
    with urlopen(url, timeout=10) as response:
        data = json.loads(response.read())
    return data.get("current", data)


TOOLS = [get_live_weather]
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

print(graph.get_graph().draw_ascii())
print("--------------------------------")

output = graph.invoke({"messages": [HumanMessage(content="What is the current weather in Bangalore?")]})
print(output["messages"][-1].content)
