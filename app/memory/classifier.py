"""Query understanding behind one interface: rules (default) or zero-shot Laya (optional)."""
import re
from abc import ABC, abstractmethod

from app.models import LIFE_AREAS

_AREA_KEYWORDS = {
    "career": r"career|job|work|profession|promotion|interview|office|boss|business|startup|salary|resume",
    "relationships": r"love|relationship|marri|partner|dating|spouse|romance|breakup|wife|husband",
    "health": r"health|illness|fitness|diet|sleep|stress|disease|surgery|wellbeing",
    "finance": r"money|finance|wealth|invest|savings|debt|income|loan|stock|rich",
    "education": r"study|exam|college|university|degree|course|student|school",
    "family": r"family|parent|mother|father|sibling|brother|sister|child|kids|son|daughter",
    "spirituality": r"spiritual|meditat|karma|dharma|temple|puja|mantra|remedy|gemstone",
    "personal_growth": r"growth|habit|confidence|purpose|motivation|goal|interest|learn",
}
_TRIVIA = {"ok", "okay", "thanks", "thank you", "thx", "hi", "hello", "hey", "yes", "no", "cool",
           "great", "nice", "hmm", "k", "bye", "got it", "sure"}
_FOLLOWUP = re.compile(
    r"^\s*(why|how come|what do you mean|can you explain|explain that|elaborate|go on|and\b|what about that|"
    r"really\b|tell me more|more on that|how so)\b.{0,60}$", re.I)
_MEMORY_QUERY = re.compile(r"\b(remember|recall|know about me|told you|mentioned|last time|earlier)\b", re.I)
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i am|i've|i'd|i'll|my|me|we|our|mine|born|call me)\b", re.I)
_DIRECTIVE = re.compile(r"\b(reply|respond|answer|speak|talk)\s+(to me\s+)?in\b|\bfrom now on\b|\balways\b|\bprefer\b", re.I)


class Classifier(ABC):
    @abstractmethod
    def life_areas(self, message: str) -> list[str]: ...

    @abstractmethod
    def worth_remembering(self, message: str) -> bool: ...

    def is_followup(self, message: str) -> bool:
        return bool(_FOLLOWUP.match(message)) and not _MEMORY_QUERY.search(message)


class RulesClassifier(Classifier):
    def life_areas(self, message: str) -> list[str]:
        low = message.lower()
        return [a for a in LIFE_AREAS if a in _AREA_KEYWORDS and re.search(_AREA_KEYWORDS[a], low)]

    def worth_remembering(self, message: str) -> bool:
        low = message.strip().lower().rstrip("!.")
        if low in _TRIVIA or len(low.split()) < 3:
            return False
        is_question = message.strip().endswith("?")
        if is_question and not re.search(r"\b(i'm|i am|my name|i was|i have|i want|planning)\b", low):
            return False
        return bool(_FIRST_PERSON.search(message) or _DIRECTIVE.search(message))


def build_classifier(kind: str) -> Classifier:
    if kind == "laya":
        try:
            from app.memory.laya_classifier import LayaClassifier
            return LayaClassifier(fallback=RulesClassifier())
        except Exception:  # optional dependency / weights missing -> rules
            import logging
            logging.getLogger(__name__).warning("Laya unavailable, using rules classifier")
    return RulesClassifier()
