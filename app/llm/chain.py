import logging
import threading

from app.llm.base import LLMError, LLMProvider

log = logging.getLogger(__name__)


class FallbackChain(LLMProvider):
    """Try providers in order; raise LLMError only if all fail."""

    name = "chain"

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self._local = threading.local()  # per-thread: a request reads what *its own* call used

    @property
    def last_used(self) -> str | None:
        return getattr(self._local, "used", None)

    def generate(self, system, messages, **kw) -> str:
        errors = [] if self.providers else ["no LLM providers configured"]
        for p in self.providers:
            try:
                out = p.generate(system, messages, **kw)
                self._local.used = p.name
                return out
            except LLMError as e:
                log.warning("LLM provider %s failed: %s", p.name, e)
                errors.append(str(e))
        raise LLMError("; ".join(errors))
