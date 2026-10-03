"""Concept: a tool in the same file. The model proposes the call. ToolNode runs it."""

# langgraph imports
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# langchain core imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# import state
from agentic_developer_bootcamp.session_9_10_MCP.lessons.second.entities import State

#import llm
from agentic_developer_bootcamp.session_9_10_MCP.lessons.zero.init_llm import llm


'''
Define a tool calculate_incident_impact
'''
@tool
def calculate_incident_impact(failed_requests: int, total_requests: int, baseline_error_rate_pct: float) -> dict:
    """Calculate request failure percentage and percentage-point change from baseline.
    Use counts from the same observation window.
    """
    rate = failed_requests / total_requests * 100
    return {"error_rate_pct": rate, "change_percentage_points": rate - baseline_error_rate_pct}


'''
List of Tools
'''
TOOLS = [calculate_incident_impact]

# Earlier code using the ChatOpenAI
#model = ChatOpenAI(model="gpt-5.4-mini").bind_tools(TOOLS)

# Using the llm object binding with the Tools
# we shall not bind the tools here to the model / llm object since we are anywayt creating the tool node
# model = llm.bind_tools(TOOLS) # Not binding the tools explicitly since we have created an explicit tools node
model = llm

'''
Assistant Invoking Model based on the message provided
'''
def assistant(state: State):
    return {"messages": [model.invoke(state["messages"])]}

'''
Langgraph Builder
'''
builder = StateGraph(State) # Langgraph state graph init
builder.add_node("assistant", assistant) # adding assistant node
builder.add_node("tools", ToolNode(TOOLS)) # adding tools node
builder.add_edge(START, "assistant") # adding edge from start to assistant
# tools_condition is a ready-made router:
# last AI message has tool_calls -> "tools", else -> END
builder.add_conditional_edges("assistant", tools_condition) # adding conditional edge from assistant to tools
builder.add_edge("tools", "assistant") # adding edge from tools to assistant
builder.add_edge("assistant", END) # adding edge from assistant to end
graph = builder.compile() # compiles the graph

print(graph.get_graph().draw_ascii())
print("--------------------------------")

needs_tool = graph.invoke({"messages": [HumanMessage(content="What is the impact of 60 failed requests out of 1200 total requests, baseline 2 percent?")]})
print(needs_tool["messages"][-1].content)
print("--------------------------------")
no_tool = graph.invoke({"messages": [HumanMessage(content="What is the capital of India?")]})
print(no_tool["messages"][-1].content)
