"""Concept: you write the router. It is ordinary Python that returns the next node name."""
#Langchain Core Messages Object
from langchain_core.messages import HumanMessage 

#Langgraph state graph
from langgraph.graph import START, END, StateGraph 

#load environment variables
from dotenv import load_dotenv
load_dotenv()

#import llm
from agentic_developer_bootcamp.session_9_10_MCP.lessons.zero.init_llm import llm

#import entities
from agentic_developer_bootcamp.session_9_10_MCP.lessons.first.entities import Severity, State

'''
Classifier object using Open AI in the background to classify the request
'''
# classifier = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(Severity)
# Creating a classifier object using llm with Severity as structured O/p
classifier = llm.with_structured_output(Severity)

'''
Execute the classification function using llm object with structured output method 
This will ensure that the output of the model is always in the form of Severity object, else it will throw an error
'''
def classify(state: State):
    decision = classifier.invoke(state["messages"])
    print("Model classified severity as:", decision.level, "-", decision.reason)
    return {"severity": decision.level}


'''
page_oncall function is used to page the on-call engineer when the ticket is of high severity
'''
def page_oncall(state: State):
    return {"result": "HIGH: page the on-call engineer now."}


'''
log_ticket function is used to log the ticket when the ticket is of low severity
'''
def log_ticket(state: State):
    return {"result": "LOW: log a ticket. No page."}


'''
Implement routing logic in the system
'''
def route(state: State) -> str:
    if state["severity"] == "high":
        return "page_oncall"
    return "log_ticket"

#lang graph state graph builder
builder = StateGraph(State)
#adding nodes
builder.add_node("classify", classify)
builder.add_node("page_oncall", page_oncall)
builder.add_node("log_ticket", log_ticket)

#adding edges
#edge from start to classify
builder.add_edge(START, "classify")

#conditional edges
builder.add_conditional_edges(
    "classify",
    route,
    {"page_oncall": "page_oncall", "log_ticket": "log_ticket"}, # string output from function : node name
)
builder.add_edge("page_oncall", END)
builder.add_edge("log_ticket", END)
graph = builder.compile()

print(graph.get_graph().draw_ascii())
print("--------------------------------")

# High Severity
query = graph.invoke({"messages": [HumanMessage(content="Checkout is failing for 50 percent of users.")]})
print(query["result"])
print("--------------------------------")

# Low Severity
query = graph.invoke({"messages": [HumanMessage(content="User is experiencing slowness in the website once in a while, may be 1 out of eevry 10 trial")]})
print(query["result"])
print("--------------------------------")

# High Severity
query = graph.invoke({"messages": [HumanMessage(content="Credit Card Information of various users have been leaked on the Internet causing a huge challenge")]})
print(query["result"])
print("--------------------------------")