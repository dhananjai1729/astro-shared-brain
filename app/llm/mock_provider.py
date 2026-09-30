"""Deterministic provider for tests/eval: extraction keys off the real user text (rules),
chat replies echo the context they were given so tests can assert on it."""
import json
import re

from app.llm.base import LLMProvider, Role
from app.memory.rules import extract_rules

EXTRACT_MARKER = "MEMORY_EXTRACTION"


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, system, messages, *, role: Role = "chat", max_tokens=800, json_mode=False) -> str:
        self.calls.append({"system": system, "messages": messages, "role": role})
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if EXTRACT_MARKER in system:
            return json.dumps({"memories": [i.model_dump() for i in extract_rules(last_user)]})
        mem = re.search(r"## Relevant memories\n(.*?)(?:\n## |\Z)", system, re.S)
        prof = re.search(r"## User profile\n(.*?)(?:\n## |\Z)", system, re.S)
        return (f"[mock] profile: {(prof.group(1).strip() if prof else 'none')} | "
                f"memories: {(mem.group(1).strip() if mem else 'none')} | you said: {last_user}")
