"""Zero-setup fallback so the app is fully usable with no API key and no local model.

Not an LLM: extraction uses the rule-based extractor (real memory updates), and chat replies are composed
from the context the pipeline selected (profile + retrieved memories), so the Shared Brain behaviour is
visible end to end. Replies are labelled so nobody mistakes them for model output.
"""
import json
import re

from app.llm.base import LLMProvider, Role
from app.llm.mock_provider import EXTRACT_MARKER
from app.memory.classifier import RulesClassifier
from app.memory.rules import extract_rules

LABEL = "[demo mode: no LLM configured] "
_ADVICE = {
    "career": "Pick one concrete next step this month (update your resume, talk to someone in the field, or shortlist roles) and set a date for it.",
    "relationships": "Lead with clear, honest communication and give things time rather than forcing a timeline.",
    "health": "Start with small, steady routines (sleep, movement, food) and check in with a professional for anything persistent.",
    "finance": "Write down your goals, build a cushion first, and avoid rushing big decisions.",
    "education": "Break the syllabus into weekly targets and protect a fixed daily study block.",
    "family": "Make time for open conversations; small, regular check-ins beat big gestures.",
    "spirituality": "A short daily practice you can actually keep matters more than an elaborate one.",
    "personal_growth": "Choose one habit, make it tiny, and track it for two weeks.",
}


def _section(system: str, title: str) -> str:
    m = re.search(rf"## {title}\n(.*?)(?:\n## |\Z)", system, re.S)
    return m.group(1).strip() if m else ""


class DemoProvider(LLMProvider):
    name = "demo"

    def generate(self, system, messages, *, role: Role = "chat", max_tokens=800, json_mode=False) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if EXTRACT_MARKER in system:
            return json.dumps({"memories": [i.model_dump() for i in extract_rules(last)]})

        profile = _section(system, "User profile")
        get = lambda label: (re.search(rf"{label}: ([^;\n(]+)", profile) or [None, "unknown"])[1].strip()
        name, sun, moon = get("name"), get("sun sign"), get("moon sign")
        memories = [l[2:] for l in _section(system, "Relevant memories").splitlines() if l.startswith("- ")]

        parts = [f"Hi {name}." if name != "unknown" else "Hello."]
        an = lambda w: "an" if w[0] in "AEIOU" else "a"
        signs = [f"{an(sun)} {sun} sun" if sun != "unknown" else "", f"{an(moon)} {moon} moon" if moon != "unknown" else ""]
        signs = " and ".join(s for s in signs if s)
        if signs:
            parts.append(f"With {signs}, here is how I'd read this.")
        if memories:
            parts.append("From what you've told me before: " + "; ".join(memories) + ".")
        elif re.search(r"\bremember\b|\bknow about me\b", last, re.I):
            parts.append("I don't have anything stored about that yet; tell me and I'll remember it.")
        areas = RulesClassifier().life_areas(last)
        if areas and areas[0] in _ADVICE:
            parts.append(_ADVICE[areas[0]])
        elif memories:
            parts.append("Keep that in mind as we talk through this.")
        else:
            parts.append("Tell me a bit more about what matters to you and I'll tailor this.")
        return LABEL + " ".join(parts)
