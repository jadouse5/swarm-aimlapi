from typing import List, Callable, Union, Optional
from pydantic import BaseModel, Field

# Agent function type
AgentFunction = Callable[[], Union[str, "Agent", dict]]

# Default model (adjust as per your API)
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


class Agent(BaseModel):
    name: str = "Agent"
    model: str = Field(default=DEFAULT_MODEL)
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = "auto"

class Function(BaseModel):
    """
    Represents a function that an agent can call.
    """
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call that an agent can make during a conversation.
    """
    id: str
    function: Function
    type: str = "function"


class Response(BaseModel):
    """
    Encapsulates the response from an agent after performing a task.
    """
    messages: List[dict] = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Represents the result of a function call by an agent.
    """
    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
