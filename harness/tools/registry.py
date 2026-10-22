"""
This module provides a registry for tools.
It stores tools, generate schemas and dispatch requests to the tools.
"""

from dataclasses import dataclass
from typing import Any, Callable, get_type_hints
from pydantic import TypeAdapter
import inspect

@dataclass
class Tool:
    """
    A tool is a function that can be called by the agent.
    """
    name: str
    description: str
    func: Callable
    schema: dict


class ToolRegistry:
    """
    A registry for tools.
    """
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """
        Get the schemas for all tools in Anthropic tools format.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, name:str, arguments: dict[str, Any]) -> str:

        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")

        try:
            tool = self._tools[name]
            result = tool.func(**arguments)
            return str(result)
        except Exception as e:
            return f"Error calling tool {name}: {e}"

registry = ToolRegistry()


def _input_schema(func: Callable) -> dict:
    hints = get_type_hints(func)
    properties = {}
    required = []

    for name, param in inspect.signature(func).parameters.items():
        param_schema = TypeAdapter(hints.get(name, str)).json_schema()
        param_schema.pop("title", None)
        properties[name] = param_schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(func: Callable) -> Callable:
    """
    A decorator to register a function as a tool.
    """

    name = func.__name__
    description = inspect.getdoc(func) or ""
    schema = _input_schema(func)

    tool = Tool(name=name, description=description, func=func, schema=schema)
    registry.register(tool)
    return func