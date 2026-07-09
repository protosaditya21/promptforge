"""Prompt templating with variable validation and versioning.

Deliberately uses a tiny ``{{var}}`` substitution syntax instead of Jinja2
so the library has zero heavyweight dependencies. If you need loops or
conditionals in your prompts, reach for Jinja2 directly -- this stays
simple on purpose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class MissingVariableError(KeyError):
    """Raised when a required template variable is not supplied."""


class ExtraVariableError(ValueError):
    """Raised when strict=True and unexpected variables are supplied."""


_VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


@dataclass
class PromptTemplate:
    """A reusable prompt string with named ``{{variable}}`` placeholders."""

    template: str
    name: str = "unnamed"
    version: str = "1.0.0"
    description: str = ""

    def variables(self) -> set[str]:
        """Return the set of variable names referenced in the template."""
        return set(_VAR_PATTERN.findall(self.template))

    def render(self, strict: bool = True, **kwargs: Any) -> str:
        """Render the template, substituting variables.

        Args:
            strict: if True, raise when kwargs contains names not used by
                the template (helps catch typos / stale call sites).
            **kwargs: variable values to substitute.

        Raises:
            MissingVariableError: a required variable was not provided.
            ExtraVariableError: strict=True and an unused kwarg was passed.
        """
        required = self.variables()
        missing = required - kwargs.keys()
        if missing:
            raise MissingVariableError(
                f"Missing variables for template '{self.name}': {sorted(missing)}"
            )
        if strict:
            extra = kwargs.keys() - required
            if extra:
                raise ExtraVariableError(
                    f"Unexpected variables for template '{self.name}': {sorted(extra)}"
                )

        def _sub(match: "re.Match[str]") -> str:
            key = match.group(1)
            return str(kwargs[key])

        return _VAR_PATTERN.sub(_sub, self.template)

    @classmethod
    def from_file(cls, path: str, **meta: Any) -> "PromptTemplate":
        """Load a template from a text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(template=text, **meta)
