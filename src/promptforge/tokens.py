"""Token counting utilities.

Uses ``tiktoken`` when it's installed (accurate, matches OpenAI-style
BPE tokenizers) and falls back to a word-based heuristic otherwise, so
the core library has no hard dependency on any tokenizer package.
"""
from __future__ import annotations

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:  # pragma: no cover - exercised via fallback tests
    _HAS_TIKTOKEN = False

_DEFAULT_ENCODING = "cl100k_base"


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """Count tokens in ``text``.

    Uses tiktoken's encoding when available for accurate counts. Falls
    back to a heuristic (~0.75 words per token) when tiktoken is not
    installed, which is good enough for budgeting purposes.
    """
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    return _heuristic_count(text)


def _heuristic_count(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    # Rough average for English prose: 1 token ~= 0.75 words.
    return max(1, round(len(words) / 0.75))


def fits_budget(text: str, budget: int, encoding_name: str = _DEFAULT_ENCODING) -> bool:
    """Return True if ``text`` fits within ``budget`` tokens."""
    return count_tokens(text, encoding_name) <= budget
