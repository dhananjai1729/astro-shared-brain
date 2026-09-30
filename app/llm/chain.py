import logging

from app.llm.base import LLMError, LLMProvider

log = logging.getLogger(__name__)


class FallbackChain(LLMProvider):
    """Try providers in order; raise LLMError only if all fail."""

    name = "chain"

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self.last_used: str | None = None

    def generate(self, system, messages, **kw) -> str:
        errors = [] if self.providers else ["no LLM providers configured"]
        for p in self.providers:
            try:
                out = p.generate(system, messages, **kw)
                self.last_used = p.name
                return out
            except LLMError as e:
                log.warning("LLM provider %s failed: %s", p.name, e)
                errors.append(str(e))
        raise LLMError("; ".join(errors))
