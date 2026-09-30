import logging

from app.config import Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMError, LLMProvider
from app.llm.chain import FallbackChain
from app.llm.demo_provider import DemoProvider
from app.llm.mock_provider import MockProvider
from app.llm.ollama_provider import OllamaProvider

log = logging.getLogger(__name__)


def build_llm(s: Settings) -> LLMProvider:
    providers: list[LLMProvider] = []
    for name in [n.strip() for n in s.llm_provider.split(",") if n.strip()]:
        try:
            if name == "anthropic":
                providers.append(AnthropicProvider(
                    s.anthropic_api_key, s.anthropic_chat_model, s.anthropic_extract_model, s.llm_timeout,
                    s.anthropic_chat_effort, s.anthropic_api_url))
            elif name == "ollama":
                providers.append(OllamaProvider(s.ollama_base_url, s.ollama_model, s.llm_timeout * 2))
            elif name == "demo":
                providers.append(DemoProvider())
            elif name == "mock":
                providers.append(MockProvider())
            else:
                log.warning("unknown LLM provider %r ignored", name)
        except LLMError as e:
            log.warning("provider %s unavailable: %s", name, e)
    if not providers:
        log.error("NO LLM PROVIDER AVAILABLE (LLM_PROVIDER=%r): every chat will get the canned fallback. "
                  "Set ANTHROPIC_API_KEY, or LLM_PROVIDER=ollama|mock.", s.llm_provider)
    return FallbackChain(providers)
