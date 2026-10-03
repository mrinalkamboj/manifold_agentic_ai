from typing import Literal, TypedDict # Typed Dictionary (Class object nehaves like typed dictionary)
from pydantic import BaseModel # Pydantic Basemodel
from langchain_core.messages import BaseMessage #Langchain object

'''
Severity Class defining the ticket severity, with two fields:
1. level : can be high or low
2. reason : a string explaining why it is high or low
'''
class Severity(BaseModel):
    level: Literal["high", "low"]
    reason: str


'''
State is a dictionary that holds the state of the graph.
It has three fields:
1. messages : list of messages
2. severity : string
3. result : string
'''
class State(TypedDict):
    messages: list[BaseMessage]
    severity: str
    result: str