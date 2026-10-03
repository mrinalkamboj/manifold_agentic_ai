from typing import TypedDict, Annotated

#import BaseMessage and add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


