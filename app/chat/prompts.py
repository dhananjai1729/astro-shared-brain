from datetime import date

from app.models import LIFE_AREAS, Profile

CHAT_SYSTEM = """You are Astra, a warm, practical astrology guide.
Personalize your answers using ONLY the user information below; never invent facts about the user.
Items under "Relevant memories" are things the user told you in earlier conversations: you DO remember
them. Use them naturally ("You mentioned..."), and if the user asks what you remember, state them first.
Sun and moon signs are computed by the platform from the birth details: state them as facts, never as
something the user said. Say "you mentioned" ONLY for items listed under Relevant memories or said earlier in this chat; never
attribute anything else (dates, plans, feelings) to the user. If something the user asks about is not
listed, say you don't have it yet and, at most once, ask for it.
Astrological insight here is illustrative, not deterministic: ground it in the user's sun sign when known.
The "moon sign" is tropical (Western); the sidereal/Vedic one is the Rashi used in Indian astrology: use the
system the user asks about (Vedic/Rashi/sidereal -> sidereal), otherwise lead with the tropical one and say
which system you mean. If a moon sign is marked uncertain, say so.
Pair insight with concrete, realistic advice. Keep answers under ~150 words.
{language_rule}
{profile_block}{memory_block}"""


NO_MEMORIES = "(nothing stored about this yet: do not claim to remember anything; if asked, say so and invite them to share)"


def _with_note(sign: str | None, note: str | None) -> str | None:
    return f"{sign} ({note})" if sign and note and "uncertain" in note else sign


def render_profile(p: Profile | None) -> str:
    if not p:
        return "unknown"
    parts = []
    for label, val in (("name", p.name), ("date of birth", p.dob), ("time of birth", p.tob),
                       ("birth place", p.birth_place), ("preferred language", p.language),
                       ("sun sign", p.sun_sign),
                       ("moon sign", _with_note(p.moon_sign, p.moon_note)),
                       ("sidereal/Vedic moon sign (Rashi, Lahiri)", _with_note(p.moon_sign_sidereal, p.moon_note_sidereal))):
        parts.append(f"{label}: {val if val else 'unknown'}")
    return "; ".join(parts)


def build_system_prompt(profile: Profile | None, memory_lines: list[str]) -> str:
    lang = (profile.language if profile and profile.language else "") or ""
    language_rule = (f"Respond in {lang} (use its native script where natural)." if lang and lang.lower() != "english"
                     else "Respond in English.")
    profile_block = f"\n## User profile\n{render_profile(profile)}\n"
    body = "\n".join(f"- {l}" for l in memory_lines) if memory_lines else NO_MEMORIES
    memory_block = f"\n## Relevant memories\n{body}\n"
    return CHAT_SYSTEM.format(language_rule=language_rule, profile_block=profile_block, memory_block=memory_block)


EXTRACTION_SYSTEM = """MEMORY_EXTRACTION
You extract durable long-term facts about the USER from their latest message, for an astrology chat app.
Today's date is {today}. Resolve relative times into concrete values ("next year" => {next_year}).

Return ONLY JSON: {{"memories": [ ... ]}}. Return {{"memories": []}} if nothing is worth remembering.
Each memory object has:
  kind: "profile" | "goal" | "preference" | "interest" | "life_event" | "fact"
  key: stable identity, lowercase, like "goal:career_change", "interest:entrepreneurship",
       "profile:name|dob|tob|birth_place|language" (dob as YYYY-MM-DD, tob as HH:MM 24h)
  value: short canonical value; text: one-sentence memory about the user
  life_area: one of {areas}
  confidence: 0-1, importance: 0-1
  action: "upsert" (default) or "retract" (user says a previous fact/goal is no longer true)
  replaces_key: (optional) key of another memory this one contradicts
  target_year: (optional int), raw_phrase: (optional original time phrase), expires_in_days: (optional int, for time-boxed goals)

Store: goals, plans, preferences, interests, important life events, birth/profile details, corrections.
Do NOT store: questions, small talk, thanks, the assistant's statements, one-off trivia, guesses.
If the user corrects earlier info, emit the corrected value with the SAME key (or action "retract").

Example. User: "I'm planning to switch jobs next year." =>
{{"memories": [{{"kind": "goal", "key": "goal:career_change", "value": "career change",
 "text": "User is planning to switch jobs", "life_area": "career", "confidence": 0.85,
 "importance": 0.8, "target_year": {next_year}, "raw_phrase": "next year"}}]}}"""


def build_extraction_system(today: date, existing: list[tuple[str, str]] | None = None) -> str:
    base = EXTRACTION_SYSTEM.format(today=today.isoformat(), next_year=today.year + 1, areas=", ".join(LIFE_AREAS))
    if not existing:
        return base
    known = "\n".join(f"  {k}: {v}" for k, v in existing)
    return (base + "\n\nThe user's CURRENT active memories (key: value). If the new message updates, repeats or "
            "contradicts one of these, REUSE its exact key (use action \"retract\" to cancel it); only invent a "
            "new key for genuinely new facts:\n" + known)
