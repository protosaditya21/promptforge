"""Conversation memory with token-budget aware truncation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .tokens import count_tokens

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Rolling conversation buffer that stays under a token budget.

    System messages are always preserved. Once the budget is exceeded,
    the oldest non-system messages are dropped first (a simple FIFO
    strategy -- good enough for most agent loops; swap in your own
    summarization if you need something smarter).
    """

    def __init__(self, token_budget: int = 4000, encoding_name: str = "cl100k_base"):
        self.token_budget = token_budget
        self.encoding_name = encoding_name
        self._messages: list[Message] = []

    def add(self, role: Role, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        self._enforce_budget()

    def add_system(self, content: str) -> None:
        self.add("system", content)

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def as_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def total_tokens(self) -> int:
        return sum(count_tokens(m.content, self.encoding_name) for m in self._messages)

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [m for m in self._messages if m.role == "system"]
        else:
            self._messages = []

    def _enforce_budget(self) -> None:
        while self.total_tokens() > self.token_budget:
            idx = self._first_droppable_index()
            if idx is None:
                break
            del self._messages[idx]

    def _first_droppable_index(self) -> Optional[int]:
        for i, m in enumerate(self._messages):
            if m.role != "system":
                return i
        return None
