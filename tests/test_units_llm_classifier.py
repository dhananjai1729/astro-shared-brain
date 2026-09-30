"""Provider/classifier adapters tested with stubs: no network, no torch, no API keys."""
import anthropic
import pytest

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMError
from app.memory.laya_classifier import LayaClassifier


class FakeRouter:
    def __init__(self, choice="career", noul=0.9, boom=False):
        self.choice, self.noul, self.boom, self.calls = choice, noul, boom, 0

    def predict(self, state, questions):
        self.calls += 1
        if self.boom:
            raise RuntimeError("weights missing")
        return {"answers": {"area": {"choice": self.choice}, "keep": {"noul": self.noul}}}


def test_laya_skips_model_when_rules_are_clear():
    r = FakeRouter()
    assert LayaClassifier(router=r).life_areas("Will my career improve?") == ["career"] and r.calls == 0


def test_laya_used_only_for_ambiguous_messages():
    r = FakeRouter(choice="finance")
    assert LayaClassifier(router=r).life_areas("Tell me what the stars hold") == ["finance"] and r.calls == 1


def test_laya_area_errors_fall_back_to_rules():
    assert LayaClassifier(router=FakeRouter(boom=True)).life_areas("Tell me what the stars hold") == []


def test_keep_gate_is_rule_based_and_never_calls_laya():
    r = FakeRouter(noul=0.99)
    c = LayaClassifier(router=r)
    assert c.worth_remembering("thanks") is False and c.worth_remembering("Why do you say that?") is False
    assert c.worth_remembering("I want to start a business this year") is True
    assert r.calls == 0  # measured to be worse than rules, so it is not consulted


def test_anthropic_provider_requires_key():
    with pytest.raises(LLMError):
        AnthropicProvider("", "claude-opus-5-5", "claude-haiku-4-5")


def test_anthropic_provider_maps_sdk_errors_and_refusals():
    p = AnthropicProvider("sk-test", "claude-opus-5-5", "claude-haiku-4-5")

    class Block: type, text = "text", "hi"
    class Resp: content, stop_reason = [Block()], "end_turn"

    class Msgs:
        def __init__(self, result): self.result = result; self.kw = None
        def create(self, **kw):
            self.kw = kw
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    ok = Msgs(Resp()); p.client.messages = ok
    assert p.generate("sys", [{"role": "user", "content": "x"}], role="extract") == "hi"
    assert ok.kw["model"] == "claude-haiku-4-5"  # cheap model for extraction, big model for chat

    class Refusal(Resp): stop_reason = "refusal"
    p.client.messages = Msgs(Refusal())
    with pytest.raises(LLMError):
        p.generate("sys", [])
    p.client.messages = Msgs(anthropic.APIConnectionError(request=None))
    with pytest.raises(LLMError):
        p.generate("sys", [])


def _stub_provider(**kw):
    p = AnthropicProvider("sk-test", "claude-opus-5-5", "claude-haiku-4-5", **kw)

    class Block: type, text = "text", "ok"
    class Resp: content, stop_reason = [Block()], "end_turn"

    class Msgs:
        kw = None
        def create(self, **k):
            Msgs.kw = k
            return Resp()
    p.client.messages = Msgs()
    return p, Msgs


def test_anthropic_chat_has_token_floor_and_effort_but_extraction_does_not():
    p, msgs = _stub_provider(chat_effort="low")
    p.generate("s", [{"role": "user", "content": "x"}], role="chat", max_tokens=700)
    assert msgs.kw["max_tokens"] >= 4096 and msgs.kw["output_config"] == {"effort": "low"}  # thinking needs headroom
    p.generate("s", [{"role": "user", "content": "x"}], role="extract", max_tokens=700)
    assert msgs.kw["max_tokens"] == 700 and "output_config" not in msgs.kw  # Haiku rejects `effort`


def test_anthropic_base_url_is_explicit_not_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://some-proxy.invalid")
    p = AnthropicProvider("sk-test", "m", "m")
    assert str(p.client.base_url).startswith("https://api.anthropic.com")


def test_anthropic_thinking_only_response_is_a_clear_error():
    p = AnthropicProvider("sk-test", "m", "m")

    class Think: type = "thinking"
    class Resp: content, stop_reason = [Think()], "max_tokens"
    class Msgs:
        def create(self, **k): return Resp()
    p.client.messages = Msgs()
    with pytest.raises(LLMError, match="max_tokens"):
        p.generate("s", [])
