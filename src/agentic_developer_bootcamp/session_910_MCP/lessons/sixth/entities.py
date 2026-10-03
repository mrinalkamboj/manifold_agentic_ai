from pydantic import BaseModel
from typing import Literal, TypedDict
from langchain_core.messages import BaseMessage

class Intent(BaseModel):
    name: Literal["impact", "weather", "chat"]


class State(TypedDict):
    messages: list[BaseMessage]
    intent: str
    result: str