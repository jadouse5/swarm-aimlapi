import copy
import json
from collections import defaultdict
from typing import List, Callable, Union
from openai import OpenAI

from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
    DEFAULT_MODEL,
)
from .util import function_to_json, debug_print, merge_chunk

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    """
    A class that orchestrates the execution of agents.
    """
    def __init__(self, client: OpenAI):
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ):
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}]
        for msg in history:
            if msg["role"] in ["system", "user", "assistant"]:
                messages.append({"role": msg["role"], "content": str(msg.get("content", ""))})

        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [self._function_to_json(f) for f in agent.functions]

        create_params = {
            "model": model_override or agent.model or DEFAULT_MODEL,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice if tools else None,
            "stream": stream,
        }

        return self.client.chat.completions.create(**create_params)

    def call_function(self, agent: Agent, messages: List[dict], function_name: str, function_args: dict):
        tools = [self._function_to_json(f) for f in agent.functions]
        
        response = self.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        function_call = response.choices[0].message.tool_calls[0].function
        if function_call.name == function_name:
            function = next((f for f in agent.functions if f.__name__ == function_name), None)
            if function:
                result = function(**function_args)
                return result
            else:
                raise ValueError(f"Function {function_name} not found in agent functions.")
        else:
            raise ValueError(f"Expected function {function_name} but model called {function_call.name}")

    def _function_to_json(self, function: Callable):
        return {
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": function.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string"} for param in function.__code__.co_varnames
                        if param != "context_variables"
                    },
                    "required": [
                        param for param in function.__code__.co_varnames
                        if param != "context_variables"
                    ]
                }
            }
        }
