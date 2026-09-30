import anthropic

from app.llm.base import LLMError, LLMProvider, Role

CHAT_MIN_TOKENS = 4096  # Opus 5.5 always thinks; thinking shares max_tokens with the visible answer


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, chat_model: str, extract_model: str, timeout: float = 60.0,
                 chat_effort: str = "low", base_url: str = "https://api.anthropic.com"):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1)
        self.models = {"chat": chat_model, "extract": extract_model}
        self.chat_effort = chat_effort

    def generate(self, system, messages, *, role: Role = "chat", max_tokens=800, json_mode=False) -> str:
        kwargs = {}
        if role == "chat":
            max_tokens = max(max_tokens, CHAT_MIN_TOKENS)
            if self.chat_effort:  # not sent for the extraction model (Haiku rejects `effort`)
                kwargs["output_config"] = {"effort": self.chat_effort}
        try:
            resp = self.client.messages.create(
                model=self.models[role], max_tokens=max_tokens, system=system, messages=messages, **kwargs)
        except anthropic.APIError as e:  # connection, rate limit, status errors
            raise LLMError(f"anthropic: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("anthropic: refusal")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise LLMError(f"anthropic: no text in response (stop_reason={resp.stop_reason})")
        return text
