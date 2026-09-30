import httpx

from app.llm.base import LLMError, LLMProvider, Role


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system, messages, *, role: Role = "chat", max_tokens=800, json_mode=False) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2 if json_mode else 0.7},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"ollama: {e}") from e
        if not text:
            raise LLMError("ollama: empty response")
        return text
