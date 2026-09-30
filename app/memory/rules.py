"""Rule-based extraction: the always-available fallback and the mock LLM's brain."""
import re
from datetime import date, timedelta

from dateutil import parser as dateparser

from app.models import ExtractedItem

_PLACE = r"([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*)"


def _year_hint(msg: str, today: date) -> tuple[int | None, str | None, int | None]:
    """Resolve relative time phrases -> (target_year, raw_phrase, expires_in_days)."""
    low = msg.lower()
    if "next year" in low:
        return today.year + 1, "next year", None
    if "next month" in low:
        return None, "next month", 45
    if "this year" in low:
        return today.year, "this year", None
    m = re.search(r"\b(?:in|by)\s+(20\d{2})\b", low)
    if m:
        return int(m.group(1)), m.group(0), None
    return None, None, None


def extract_rules(message: str, today: date | None = None) -> list[ExtractedItem]:
    today = today or date.today()
    low = message.lower()
    items: list[ExtractedItem] = []

    def add(**kw):
        items.append(ExtractedItem(**kw))

    m = re.search(r"\b(?i:my name is)\s+([A-Z][a-z]+)|\b(?i:call me)\s+([A-Z][a-z]+)", message)
    if m:
        name = m.group(1) or m.group(2)
        add(kind="profile", key="profile:name", value=name, text=f"User's name is {name}",
            confidence=0.95, importance=0.6)

    m = re.search(
        r"\bborn\s+(?:on\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        message)
    if m:
        try:
            dob = dateparser.parse(m.group(1)).date().isoformat()
            add(kind="profile", key="profile:dob", value=dob, text=f"User was born on {dob}",
                confidence=0.95, importance=0.8)
        except (ValueError, OverflowError):
            pass

    m = re.search(r"\b(?:born|birth).*?\bin\s+" + _PLACE, message)
    if m:
        add(kind="profile", key="profile:birth_place", value=m.group(1),
            text=f"User was born in {m.group(1)}", confidence=0.9, importance=0.6)

    m = re.search(r"\bborn\b.*?\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
    if m:
        hh, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        if hh < 24:
            add(kind="profile", key="profile:tob", value=f"{hh:02d}:{mm:02d}",
                text="User's time of birth", confidence=0.9, importance=0.6)

    m = re.search(r"\b(hindi|english|tamil|telugu|bengali|marathi|gujarati|kannada)\b", low)
    if m and re.search(r"\b(prefer|reply|respond|speak|talk|answer)\b", low):
        add(kind="profile", key="profile:language", value=m.group(1).capitalize(),
            text=f"User prefers {m.group(1).capitalize()}", confidence=0.85, importance=0.7)

    year, raw, exp = _year_hint(message, today)
    retract = re.search(
        r"(?:not|no longer|don't|won't|never)\s+(?:\w+\s+){0,2}(?:switch|change|quit|leave)\w*\s+(?:my\s+)?(?:jobs?|career|company)"
        r"|(?:decided|changed my mind).{0,30}\b(?:stay|not)\b", low)
    if retract:
        add(kind="goal", key="goal:career_change", value="retracted",
            text="User no longer plans to change jobs", life_area="career",
            action="retract", confidence=0.85, importance=0.7)
    elif re.search(r"\b(switch|change|quit|leave)\w*\s+(?:my\s+)?(?:jobs?|career|company)", low):
        add(kind="goal", key="goal:career_change", value="career change",
            text="User is planning a career change" + (f" around {year}" if year else ""),
            life_area="career", target_year=year, raw_phrase=raw, expires_in_days=None,
            confidence=0.85, importance=0.8)

    m = re.search(r"\b(?:preparing for|prepping for|have)\s+(?:an?\s+)?([a-z ]{3,40}?)\s+interview", low)
    if m:
        topic = m.group(1).strip()
        add(kind="goal", key=f"goal:interview_{re.sub(r'[^a-z]+', '_', topic).strip('_')}",
            value=f"{topic} interview", text=f"User is preparing for a {topic} interview",
            life_area="career", target_year=year, raw_phrase=raw, expires_in_days=exp or 60,
            confidence=0.85, importance=0.8)

    if re.search(r"\b(start(?:ing)?|launch(?:ing)?|begin(?:ning)?)\s+(?:a\s+|my\s+own\s+)?(business|startup|company)", low):
        add(kind="goal", key="goal:start_business", value="start a business",
            text="User wants to start a business", life_area="career",
            target_year=year, raw_phrase=raw, confidence=0.8, importance=0.8)

    m = re.search(r"\binterested in\s+([a-z][a-z ]{2,30}?)(?:[.,!?]|$| and | but )", low)
    if m:
        topic = m.group(1).strip()
        add(kind="interest", key=f"interest:{re.sub(r'[^a-z]+', '_', topic).strip('_')}",
            value=topic.title(), text=f"User is interested in {topic}",
            life_area="personal_growth", confidence=0.8, importance=0.5)

    return items
