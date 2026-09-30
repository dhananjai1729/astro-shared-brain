from datetime import datetime, timezone


def _age_days(iso: str | None, now: datetime) -> float:
    if not iso:
        return 0.0
    return max((now - datetime.fromisoformat(iso)).total_seconds() / 86400, 0.0)


def score_memory(m: dict, matched_areas: list[str], half_life_days: float, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    relevance = 1.0 if m.get("life_area") in matched_areas else (0.3 if not matched_areas else 0.1)
    decay = 0.5 ** (_age_days(m.get("updated_at"), now) / half_life_days)
    return relevance * (0.5 + 0.5 * m.get("importance", 0.5)) * m.get("confidence", 0.8) * decay


def rank_memories(memories: list[dict], matched_areas: list[str], half_life_days: float, limit: int,
                  now: datetime | None = None) -> list[dict]:
    if matched_areas:  # a topical query never gets off-topic memories
        memories = [m for m in memories if m.get("life_area") in matched_areas]
    scored = sorted(memories, key=lambda m: score_memory(m, matched_areas, half_life_days, now), reverse=True)
    return scored[:limit]
