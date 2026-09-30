"""Failure paths. Most run without Docker (dead Neo4j URI); outbox replay needs the live graph."""
import pytest

from app.brain.store import Brain
from app.llm.base import LLMError, LLMProvider
from app.llm.chain import FallbackChain
from app.llm.mock_provider import MockProvider

FIRST = "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."


class DownLLM(LLMProvider):
    name = "down"

    def generate(self, *a, **k):
        raise LLMError("boom")


class GarbageExtractor(MockProvider):
    def generate(self, system, messages, **kw):
        if kw.get("role") == "extract":
            return "sorry, I cannot produce JSON"
        return super().generate(system, messages, **kw)


@pytest.fixture
def dead_brain():
    return Brain("bolt://localhost:59999", "neo4j", "x", connect_timeout=1)


def post(c, msg="hello there", uid="t-fail", sid="s"):
    return c.post("/chat", json={"user_id": uid, "session_id": sid, "message": msg})


@pytest.mark.parametrize("body", [
    {"user_id": "u", "session_id": "s", "message": ""},
    {"user_id": "u", "session_id": "s", "message": "   "},
    {"user_id": "bad id", "session_id": "s", "message": "hi"},
    {"user_id": "u", "session_id": "s", "message": "x" * 2001},
    {"user_id": "u", "message": "hi"},
])
def test_invalid_input_is_422(make_client, dead_brain, body):
    assert make_client(brain=dead_brain).post("/chat", json=body).status_code == 422


def test_invalid_profile_is_422(make_client, dead_brain):
    c = make_client(brain=dead_brain)
    assert c.put("/users/u1/profile", json={"dob": "not-a-date"}).status_code == 422
    assert c.put("/users/u1/profile", json={"tob": "99:99"}).status_code == 422


def test_all_llms_down_returns_degraded_canned_reply(make_client, dead_brain):
    r = post(make_client(brain=dead_brain, llm=DownLLM()))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] and "llm unavailable" in body["warnings"] and body["response"]


def test_fallback_chain_uses_next_provider(make_client, dead_brain):
    chain = FallbackChain([DownLLM(), MockProvider()])
    body = post(make_client(brain=dead_brain, llm=chain)).json()
    assert body["response"].startswith("[mock]") and "llm unavailable" not in body["warnings"]


def test_graph_down_chat_still_answers_and_flags_degraded(make_client, dead_brain):
    body = post(make_client(brain=dead_brain)).json()
    assert body["response"].startswith("[mock]") and body["degraded"]
    assert any("shared brain unavailable" in w for w in body["warnings"])
    assert body["context_used"] == []


def test_graph_down_profile_endpoints(make_client, dead_brain):
    c = make_client(brain=dead_brain)
    assert c.put("/users/u1/profile", json={"name": "A"}).status_code == 503
    assert c.get("/users/u1/profile").status_code == 503
    assert c.get("/health").json()["graph"] == "down"


def test_failed_graph_writes_go_to_outbox_then_replay(make_client, dead_brain, live_brain, uid):
    c = make_client(brain=dead_brain)
    post(c, FIRST, uid=uid)
    pending = c.app.state.sessions.pending_count()
    assert pending >= 2  # profile facts + goal were queued, not lost
    # graph comes back: the next update (or startup) drains the outbox
    c.app.state.updater.brain = live_brain
    assert c.app.state.updater.drain_outbox() == pending
    assert c.app.state.sessions.pending_count() == 0
    assert [m["key"] for m in live_brain.list_memories(uid)] == ["goal:career_change"]
    assert live_brain.get_profile(uid).birth_place == "Delhi"


def test_garbage_extractor_output_falls_back_to_rules(live_brain, make_client, uid):
    c = make_client(brain=live_brain, llm=GarbageExtractor())
    post(c, FIRST, uid=uid)
    assert [m["key"] for m in live_brain.list_memories(uid)] == ["goal:career_change"]


def test_missing_profile_and_empty_memory_paths(live_brain, make_client, uid):
    body = post(make_client(brain=live_brain), "What does my horoscope say about money?", uid=uid).json()
    assert not body["degraded"] and body["context_used"] == []


def test_unknown_user_profile_404(live_brain, make_client):
    assert make_client(brain=live_brain).get("/users/t-never-seen-xyz/profile").status_code == 404


def test_profile_put_computes_sun_sign(live_brain, make_client, uid):
    r = make_client(brain=live_brain).put(f"/users/{uid}/profile",
                                          json={"name": "Asha", "dob": "1990-03-21", "tob": "06:30",
                                                "birth_place": "Pune", "language": "Hindi"})
    assert r.status_code == 200 and r.json()["sun_sign"] == "Aries"


def test_rule_based_profile_facts_backfill_a_lazy_llm(live_brain, make_client, uid):
    class EmptyExtractor(MockProvider):
        def generate(self, system, messages, **kw):
            return '{"memories": []}' if kw.get("role") == "extract" else super().generate(system, messages, **kw)
    post(make_client(brain=live_brain, llm=EmptyExtractor()), "My name is Asha and I was born in Pune", uid=uid)
    prof = live_brain.get_profile(uid)
    assert (prof.name, prof.birth_place) == ("Asha", "Pune")


def test_profile_put_computes_moon_sign_and_reacts_to_birth_time(live_brain, make_client, uid):
    c = make_client(brain=live_brain)
    from app.profile.astro import moon_sign
    from datetime import date
    r = c.put(f"/users/{uid}/profile", json={"dob": "1995-08-15", "tob": "06:30"}).json()
    assert r["moon_sign"] == moon_sign(date(1995, 8, 15), "06:30")[0] and r["moon_note"] == "from birth time"
    assert r["sun_sign"] == "Leo"
    edges = live_brain._run("MATCH (:User {id:$u})-[rel:HAS_MOON_SIGN {system:'tropical'}]->(s:Sign) RETURN s.name AS n", u=uid)
    assert [e["n"] for e in edges] == [r["moon_sign"]]  # exactly one tropical moon-sign edge
    # changing the UTC offset recomputes it, still a single edge
    r2 = c.put(f"/users/{uid}/profile", json={"utc_offset": -8}).json()
    assert r2["utc_offset"] == -8 and r2["moon_sign"] == moon_sign(date(1995, 8, 15), "06:30", -8)[0]
    assert len(live_brain._run("MATCH (:User {id:$u})-[:HAS_MOON_SIGN {system:'tropical'}]->() RETURN 1", u=uid)) == 1
    assert c.put(f"/users/{uid}/profile", json={"utc_offset": 99}).status_code == 422


def test_profile_put_returns_tropical_and_sidereal_moon_signs(live_brain, make_client, uid):
    from app.profile.astro import moon_sign
    from datetime import date
    c = make_client(brain=live_brain)
    r = c.put(f"/users/{uid}/profile", json={"dob": "1995-08-15", "tob": "06:30"}).json()
    d = date(1995, 8, 15)
    assert r["moon_sign"] == moon_sign(d, "06:30")[0]
    assert r["moon_sign_sidereal"] == moon_sign(d, "06:30", sidereal=True)[0]
    assert r["moon_sign"] != r["moon_sign_sidereal"]  # ~24 deg apart for this chart
    edges = live_brain._run("MATCH (:User {id:$u})-[r:HAS_MOON_SIGN]->(s:Sign) RETURN r.system AS sys, s.name AS n", u=uid)
    assert {e["sys"]: e["n"] for e in edges} == {"tropical": r["moon_sign"], "sidereal": r["moon_sign_sidereal"]}
    # update recomputes both, still exactly one edge per system
    c.put(f"/users/{uid}/profile", json={"utc_offset": -8})
    assert len(live_brain._run("MATCH (:User {id:$u})-[:HAS_MOON_SIGN]->() RETURN 1", u=uid)) == 2
