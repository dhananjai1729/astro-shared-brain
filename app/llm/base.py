from abc import ABC, abstractmethod
from typing import Literal

Role = Literal["chat", "extract"]


class LLMError(Exception):
    """Raised when a provider cannot produce a completion."""


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict],
        *,
        role: Role = "chat",
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> str:
        """Return the model's text. `messages` are {role, content} dicts (user/assistant)."""
