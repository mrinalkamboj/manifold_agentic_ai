"""
---------------------------------------------------------------------------------
Full demo: you classify the ask, then each path uses a different tool style.
---------------------------------------------------------------------------------
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition

# llm import
from agentic_developer_bootcamp.session_910_MCP.lessons.zero.init_llm import llm

# Entities import
from agentic_developer_bootcamp.session_910_MCP.lessons.sixth.entities import Intent, State


# tools list (filled at runtime with the MCP tools)
tools = []

# Classify function to find the correct intent
def classify(state: State):
    classifier = llm.with_structured_output(Intent)
    decision = classifier.invoke(state["messages"])
    print("Intent:", decision.name)
    return {"intent": decision.name}

# Routing function
def route(state: State) -> str:
    return state["intent"]

# Main Execute function to call the model with tools
def execute(state: State):
    model = llm.bind_tools(tools)
    result = model.invoke(state["messages"])
    return {"messages": [result], "result": result.content}

# function to return tool results to the node that asked for them
def back_to_node(state: State) -> str:
    return state["intent"]  

# Establish a connection with the MCP server
MCP_SERVER = str(Path(__file__).resolve().parent / "mcp_server.py") # Resolve MCP Server

# Create MCP client
mcp_client = MultiServerMCPClient({
    "incident": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [MCP_SERVER],
    },
    })

async def main():
    
    tools = await mcp_client.get_tools() # list of mcp tools
    print("MCP tools:", [tool.name for tool in tools]) # print mcp tools
    globals()["tools"].extend(tools) # make tools visible to execute()
    
    builder = StateGraph(State) #lang graph state graph
    builder.add_node("classify", classify) # adding classify node
    builder.add_node("impact", execute) # adding impact node
    builder.add_node("weather", execute) # adding weather node
    builder.add_node("chat", execute) # adding chat node
    builder.add_node("tools", ToolNode(tools)) # adding tools node

    builder.add_edge(START, "classify") # adding edge from start to classify
    
    builder.add_conditional_edges(
        "classify",
        route,
        {"impact": "impact", "weather": "weather", "chat": "chat"},
    )
    builder.add_conditional_edges("impact", tools_condition) # adding conditional edge from assistant to tools
    builder.add_conditional_edges("weather", tools_condition) # adding conditional edge from assistant to tools
    builder.add_conditional_edges("chat", tools_condition) # adding conditional edge from assistant to tools   

    builder.add_conditional_edges(
        "tools",
        back_to_node,
        {"impact": "impact", "weather": "weather", "chat": "chat"},
    )
    
    builder.add_edge("impact", END)
    builder.add_edge("weather", END)
    builder.add_edge("chat", END)
    graph = builder.compile()

    print(graph.get_graph().draw_ascii())
    print("--------------------------------")

    for question in (
        "What is the weather in Bangalore right now?",
        "What is the impact of 60 failed requests out of 1200, baseline 2 percent?",
        "What is Langchain deep agents, explain in detail?",
    ):
        output = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
        print(question)
        print(output["result"])
        print("--------------------------------")

# Initiate Async code
if __name__ == "__main__":
    asyncio.run(main())
