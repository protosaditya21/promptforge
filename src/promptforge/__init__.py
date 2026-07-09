"""promptforge: a lightweight, provider-agnostic toolkit for prompt
engineering and LLM agents.
"""
from .agent import Agent, AgentStep, ToolRegistry, ToolSpec
from .memory import ConversationMemory, Message
from .retry import RetryExhaustedError, with_retry
from .templates import ExtraVariableError, MissingVariableError, PromptTemplate
from .tokens import count_tokens, fits_budget

__version__ = "0.1.0"

__all__ = [
    "PromptTemplate",
    "MissingVariableError",
    "ExtraVariableError",
    "count_tokens",
    "fits_budget",
    "with_retry",
    "RetryExhaustedError",
    "ConversationMemory",
    "Message",
    "Agent",
    "AgentStep",
    "ToolRegistry",
    "ToolSpec",
]
