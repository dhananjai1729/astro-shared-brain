"""Post-response memory update: gate -> extract (LLM, rules fallback) -> validate -> write (outbox on failure)."""
import json
import logging
import re
from datetime import date

from pydantic import ValidationError

from app.brain.store import Brain, BrainUnavailable
from app.chat.prompts import build_extraction_system
from app.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.memory.classifier import Classifier
from app.memory.rules import extract_rules
from app.models import ExtractedItem
from app.session.store import SessionStore

log = logging.getLogger(__name__)


def parse_items(raw: str) -> list[ExtractedItem]:
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
    if not m:
        raise ValueError("no JSON in extractor output")
    data = json.loads(m.group(0))
    rows = data.get("memories", []) if isinstance(data, dict) else data
    items = []
    for r in rows:
        try:
            items.append(ExtractedItem(**r))
        except (ValidationError, TypeError) as e:
            log.info("dropping invalid extracted memory %r: %s", r, e)
    return items


class MemoryUpdater:
    def __init__(self, brain: Brain | None, llm: LLMProvider, classifier: Classifier,
                 sessions: SessionStore, settings: Settings):
        self.brain, self.llm, self.classifier, self.sessions, self.s = brain, llm, classifier, sessions, settings

    def _existing(self, user_id: str) -> list[tuple[str, str]]:
        if self.brain is None:
            return []
        try:
            mems = self.brain.list_memories(user_id)
        except BrainUnavailable:
            return []
        mems.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return [(m["key"], m["value"]) for m in mems[:20]]

    def extract(self, message: str, recent: list[dict], user_id: str | None = None) -> list[ExtractedItem]:
        today = date.today()
        try:
            convo = [*recent[-4:], {"role": "user", "content": message}]
            # extraction looks only at the user's latest message; recent turns disambiguate references
            raw = self.llm.generate(build_extraction_system(today, self._existing(user_id) if user_id else None), convo,
                                    role="extract", max_tokens=700, json_mode=True)
            items = parse_items(raw)
            # regexes are high-precision for profile facts; small LLMs often skip them. LLM wins on conflicts.
            have = {i.key for i in items}
            items += [i for i in extract_rules(message, today) if i.kind == "profile" and i.key not in have]
            return items
        except (LLMError, ValueError, json.JSONDecodeError) as e:
            log.warning("LLM extraction failed (%s); using rule-based extractor", e)
            return extract_rules(message, today)

    def validate(self, items: list[ExtractedItem]) -> list[ExtractedItem]:
        seen, out = set(), []
        for it in items:
            if it.confidence < self.s.min_memory_confidence:
                continue
            sig = (it.key, it.value, it.action)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(it)
        return out

    def write(self, user_id: str, items: list[ExtractedItem]) -> dict:
        """Apply items to the graph. Anything the graph can't take goes to the outbox."""
        stats = {"applied": 0, "queued": 0}
        for idx, it in enumerate(items):
            try:
                if self.brain is None:
                    raise BrainUnavailable("no brain")
                self._apply(user_id, it)
                stats["applied"] += 1
            except BrainUnavailable:
                for rest in items[idx:]:
                    self.sessions.enqueue(user_id, rest.model_dump())
                    stats["queued"] += 1
                break
        return stats

    def _apply(self, user_id: str, it: ExtractedItem):
        if it.kind == "profile":
            field = it.key.split(":", 1)[1]
            if field in ("name", "dob", "tob", "birth_place", "language") and it.action == "upsert":
                prof = self.brain.update_profile(user_id, {field: it.value})
                self.sessions.cache_profile(user_id, prof.model_dump())
            return
        self.brain.apply_item(user_id, it)

    def drain_outbox(self) -> int:
        """Retry queued writes (called at startup and before each update)."""
        if self.brain is None:
            return 0
        done = 0
        for oid, uid, payload in self.sessions.pending():
            try:
                self._apply(uid, ExtractedItem(**payload))
                self.sessions.ack(oid)
                done += 1
            except BrainUnavailable:
                break
            except ValidationError:
                self.sessions.ack(oid)  # poison message
        return done

    def process(self, user_id: str, message: str, recent: list[dict]) -> dict:
        try:
            self.drain_outbox()
            if not self.classifier.worth_remembering(message):
                return {"skipped": "not worth remembering"}
            items = self.validate(self.extract(message, recent, user_id))
            if not items:
                return {"skipped": "nothing extracted"}
            return self.write(user_id, items)
        except Exception:  # never let memory update break or crash the request path
            log.exception("memory update failed")
            return {"error": True}
