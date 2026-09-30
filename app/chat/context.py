from dataclasses import dataclass, field

from app.brain.store import Brain, BrainUnavailable
from app.config import Settings
from app.memory.classifier import Classifier
from app.memory.ranking import rank_memories
from app.models import Profile


@dataclass
class ContextBundle:
    profile: Profile | None
    memory_lines: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False


def _label(m: dict) -> str:
    kind = "goal" if m.get("kind") == "goal" else m.get("kind", "memory")
    return f"{m.get('life_area', 'general')}_{kind}"


def select_context(user_id: str, message: str, profile: Profile | None, recent: list[dict],
                   brain: Brain | None, classifier: Classifier, s: Settings, brain_ok: bool = True) -> ContextBundle:
    """Understand query -> pick relevant memories -> fit budget. Never the whole graph."""
    ctx = ContextBundle(profile=profile, recent=recent)
    if profile and any([profile.name, profile.dob, profile.birth_place, profile.language, profile.sun_sign]):
        ctx.labels.append("user_profile")

    if not brain_ok or brain is None:
        ctx.degraded = True
        ctx.warnings.append("shared brain unavailable; answering without long-term memory")
        return ctx
    if classifier.is_followup(message) and recent:
        return ctx  # short-term context is enough

    areas = classifier.life_areas(message)
    try:
        cands = brain.candidate_memories(user_id, areas or None)
    except BrainUnavailable:
        ctx.degraded = True
        ctx.warnings.append("shared brain unavailable; answering without long-term memory")
        return ctx

    if not areas:
        # no topical match: only a generic "what do you remember" style query gets the top memories
        from app.memory.classifier import _MEMORY_QUERY
        if not _MEMORY_QUERY.search(message):
            return ctx
    chosen = rank_memories(cands, areas, s.memory_half_life_days, s.max_memories)
    budget = s.context_char_budget
    for m in chosen:
        line = m["text"] + (f" (target year {m['target_year']})" if m.get("target_year") else "")
        if len(line) > budget:
            break
        budget -= len(line)
        ctx.memory_lines.append(line)
        ctx.labels.append(_label(m))
    return ctx
