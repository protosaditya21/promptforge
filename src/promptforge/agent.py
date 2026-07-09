"""A minimal, provider-agnostic tool-calling agent loop.

This does not depend on any specific LLM SDK. You supply a ``chat_fn``
that takes the conversation (list of message dicts) and available tool
specs, and returns an :class:`AgentStep` describing either a final
answer or a tool call. That keeps the loop testable and usable with any
provider -- Anthropic, OpenAI, a local model, or a mock in unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .memory import ConversationMemory


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=dict)

    def spec_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Register callables as tools an agent can invoke by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, parameters: Optional[dict] = None):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                handler=func,
                parameters=parameters or {},
            )
            return func

        return decorator

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def specs(self) -> list[dict]:
        return [t.spec_dict() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).handler(**kwargs)


@dataclass
class AgentStep:
    """One decision from the model: either a final answer or a tool call."""

    kind: str  # "final" | "tool_call"
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


ChatFn = Callable[[list[dict], list[dict]], AgentStep]


class Agent:
    """Drives a chat_fn through a tool-use loop until it produces a final answer."""

    def __init__(
        self,
        chat_fn: ChatFn,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        max_steps: int = 8,
    ):
        self.chat_fn = chat_fn
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)
        for _ in range(self.max_steps):
            step = self.chat_fn(self.memory.as_dicts(), self.tools.specs())
            if step.kind == "final":
                self.memory.add_assistant(step.content)
                return step.content
            if step.kind == "tool_call":
                result = self.tools.call(step.tool_name, **step.tool_args)
                self.memory.add_assistant(
                    f"[called tool {step.tool_name} with {step.tool_args}]"
                )
                self.memory.add("tool", str(result))
                continue
            raise ValueError(f"Unknown step kind: {step.kind}")
        raise RuntimeError("Agent exceeded max_steps without producing a final answer")
