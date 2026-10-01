"""Full demo: you classify the ask, then each path uses a different tool style."""
import json
from urllib.request import urlopen
from typing import Literal, TypedDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
from dotenv import load_dotenv
from incident_tools import calculate_incident_impact
load_dotenv()


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


class Intent(BaseModel):
    name: Literal["impact", "weather", "chat"]


class State(TypedDict):
    messages: list[BaseMessage]
    intent: str
    result: str


classifier = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(Intent)
chat = ChatOpenAI(model="gpt-5.4-mini")


def classify(state: State):
    decision = classifier.invoke(state["messages"])
    print("Intent:", decision.name)
    return {"intent": decision.name}


def route(state: State) -> str:
    return state["intent"]


def run_impact(state: State):
    # Local / module tool. Model fills the arguments. We execute.
    model = ChatOpenAI(model="gpt-5.4-mini").bind_tools([calculate_incident_impact])
    ai = model.invoke(state["messages"])
    if not ai.tool_calls:
        return {"result": ai.content}
    result = calculate_incident_impact.invoke(ai.tool_calls[0]["args"])
    return {"result": str(result)}


def run_weather(state: State):
    # Live API tool. Model fills the city coordinates. We execute.
    model = ChatOpenAI(model="gpt-5.4-mini").bind_tools([get_live_weather])
    ai = model.invoke(state["messages"])
    if not ai.tool_calls:
        return {"result": ai.content}
    result = get_live_weather.invoke(ai.tool_calls[0]["args"])
    return {"result": str(result)}


def run_chat(state: State):
    ai = chat.invoke(state["messages"])
    return {"result": ai.content}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("impact", run_impact)
builder.add_node("weather", run_weather)
builder.add_node("chat", run_chat)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {"impact": "impact", "weather": "weather", "chat": "chat"},
)
builder.add_edge("impact", END)
builder.add_edge("weather", END)
builder.add_edge("chat", END)
graph = builder.compile()

print(graph.get_graph().draw_ascii())
print("--------------------------------")

for question in (
    "What is the impact of 60 failed requests out of 1200, baseline 2 percent?",
    "What is the weather in Bangalore right now?",
    "What is LangGraph in one sentence?",
):
    output = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(question)
    print(output["result"])
    print("--------------------------------")
