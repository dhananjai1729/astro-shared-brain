import logging

from app.brain.store import Brain, BrainUnavailable
from app.chat.context import select_context
from app.chat.prompts import build_system_prompt
from app.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.memory.classifier import Classifier
from app.memory.updater import MemoryUpdater
from app.models import ChatRequest, ChatResponse, Profile
from app.session.store import SessionStore

log = logging.getLogger(__name__)

FALLBACK_REPLY = ("I'm having trouble reaching my insight engine right now, so I can't give you a proper "
                  "answer. Please try again in a moment.")


class ChatService:
    def __init__(self, brain: Brain | None, llm: LLMProvider, classifier: Classifier,
                 sessions: SessionStore, updater: MemoryUpdater, settings: Settings):
        self.brain, self.llm, self.classifier = brain, llm, classifier
        self.sessions, self.updater, self.s = sessions, updater, settings

    def _load_profile(self, user_id: str) -> tuple[Profile | None, bool]:
        """Returns (profile, brain_ok). Falls back to the SQLite cache if the graph is down."""
        try:
            self.brain.ensure_user(user_id)
            prof = self.brain.get_profile(user_id)
            if prof:
                self.sessions.cache_profile(user_id, prof.model_dump())
            return prof, True
        except (BrainUnavailable, AttributeError):
            cached = self.sessions.cached_profile(user_id)
            return (Profile(**cached) if cached else None), False

    def chat(self, req: ChatRequest) -> tuple[ChatResponse, callable]:
        """Returns the response and a zero-arg callable that performs the memory update."""
        profile, brain_ok = self._load_profile(req.user_id)
        recent = self.sessions.recent(req.user_id, req.session_id, self.s.recent_turns * 2)
        ctx = select_context(req.user_id, req.message, profile, recent, self.brain,
                             self.classifier, self.s, brain_ok)
        system = build_system_prompt(ctx.profile, ctx.memory_lines)
        warnings = list(ctx.warnings)
        degraded = ctx.degraded
        try:
            reply = self.llm.generate(system, [*recent, {"role": "user", "content": req.message}],
                                      role="chat", max_tokens=700)
        except LLMError as e:
            log.error("all LLM providers failed: %s", e)
            reply, degraded = FALLBACK_REPLY, True
            warnings.append("llm unavailable")

        if getattr(self.llm, "last_used", None) == "demo" and not degraded:
            warnings.append("demo mode: no real LLM available (set ANTHROPIC_API_KEY or run Ollama)")

        self.sessions.add_turn(req.user_id, req.session_id, "user", req.message)
        self.sessions.add_turn(req.user_id, req.session_id, "assistant", reply)
        resp = ChatResponse(response=reply, user_id=req.user_id, session_id=req.session_id,
                            context_used=ctx.labels, degraded=degraded, warnings=warnings)
        update = lambda: self.updater.process(req.user_id, req.message, recent)
        return resp, update
