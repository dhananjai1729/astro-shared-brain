"""End-to-end scenarios through POST /chat with the real Neo4j graph and the deterministic mock LLM."""
FIRST = "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."


def chat(c, uid, sid, msg):
    r = c.post("/chat", json={"user_id": uid, "session_id": sid, "message": msg})
    assert r.status_code == 200, r.text
    return r.json()


def memories(c, uid, inactive=False):
    return c.get(f"/users/{uid}/memories", params={"include_inactive": inactive}).json()["memories"]


def test_1_new_user_gets_a_reply_without_memories(client, uid):
    r = chat(client, uid, "s1", "What should I focus on for my career?")
    assert r["response"] and r["context_used"] == [] and not r["degraded"]
    assert "nothing stored" in r["response"]  # prompt tells the LLM it remembers nothing


def test_2_creates_long_term_memory_and_profile(client, uid):
    chat(client, uid, "s1", FIRST)
    mems = memories(client, uid)
    assert [m["key"] for m in mems] == ["goal:career_change"]
    assert mems[0]["target_year"] == 2027 and mems[0]["life_area"] == "career"
    prof = client.get(f"/users/{uid}/profile").json()
    assert (prof["name"], prof["birth_place"], prof["dob"], prof["sun_sign"]) == ("Rahul", "Delhi", "1995-08-15", "Leo")


def test_3_retrieves_memory_for_relevant_question(client, uid):
    chat(client, uid, "s1", FIRST)
    r = chat(client, uid, "s1", "What should I focus on for my career?")
    assert "career_goal" in r["context_used"] and "user_profile" in r["context_used"]
    assert "career change" in r["response"]


def test_4_followup_uses_recent_context_not_retrieval(client, uid, mock):
    chat(client, uid, "s1", FIRST)
    chat(client, uid, "s1", "What should I focus on for my career?")
    r = chat(client, uid, "s1", "Why do you say that?")
    assert r["context_used"] == ["user_profile"]  # no long-term retrieval for a follow-up
    last = mock.calls[-1]["messages"]
    assert last[-3]["content"] == "What should I focus on for my career?" and last[-2]["role"] == "assistant"


def test_5_new_session_uses_previous_information(client, uid):
    chat(client, uid, "s1", FIRST)
    r = chat(client, uid, "s2-new", "What do you remember about my career goals?")
    assert "career_goal" in r["context_used"] and "career change" in r["response"]


def test_6_irrelevant_memory_is_not_injected(client, uid):
    chat(client, uid, "s1", FIRST)
    r = chat(client, uid, "s1", "Will I find love and marriage soon?")
    assert "career_goal" not in r["context_used"] and "career change" not in r["response"]


def test_7_user_correction_supersedes_old_memory(client, uid):
    chat(client, uid, "s1", FIRST)
    chat(client, uid, "s1", "Actually I was born in Mumbai, not Delhi.")
    assert client.get(f"/users/{uid}/profile").json()["birth_place"] == "Mumbai"
    chat(client, uid, "s1", "I decided not to switch jobs, I'm starting a business instead.")
    active = {m["key"] for m in memories(client, uid)}
    all_ = {m["key"]: m["status"] for m in memories(client, uid, inactive=True)}
    assert active == {"goal:start_business"}
    assert all_["goal:career_change"] == "superseded"
    r = chat(client, uid, "s3", "What should I focus on for my career?")
    assert "start a business" in r["response"] and "career change" not in r["response"]


def test_7b_same_key_new_value_supersedes_and_links(client, live_brain, uid):
    from app.models import ExtractedItem
    mk = lambda v: ExtractedItem(kind="goal", key="goal:career_change", value=v, text=f"goal {v}",
                                 life_area="career")
    live_brain.apply_item(uid, mk("v1"))
    assert live_brain.apply_item(uid, mk("v1")) == "updated"      # same value: no duplicate
    assert live_brain.apply_item(uid, mk("v2")) == "superseded"   # new value: old archived
    rows = live_brain._run("MATCH (:Memory {status:'active'})-[:SUPERSEDES]->(o:Memory {status:'superseded'}) "
                           "RETURN count(o) AS n")
    assert rows[0]["n"] >= 1
    assert len([m for m in live_brain.list_memories(uid) if m["key"] == "goal:career_change"]) == 1


def test_8_missing_information_does_not_break(client, uid):
    r = chat(client, uid, "s1", "What does my horoscope say about money?")
    assert r["response"] and "unknown" in r["response"]  # prompt tells the LLM what is unknown


def test_small_talk_is_not_stored(client, uid):
    chat(client, uid, "s1", "thanks")
    chat(client, uid, "s1", "Why do you say that?")
    assert memories(client, uid, inactive=True) == []


def test_expired_memory_is_not_retrieved(client, live_brain, uid):
    from app.models import ExtractedItem
    live_brain.apply_item(uid, ExtractedItem(kind="goal", key="goal:interview_pm", value="pm interview",
                                             text="Preparing for a PM interview", life_area="career",
                                             expires_in_days=1))
    live_brain._run("MATCH (m:Memory {key:'goal:interview_pm'}) SET m.expires_at='2020-01-01T00:00:00+00:00'")
    r = chat(client, uid, "s1", "How is my career looking?")
    assert "career_goal" not in r["context_used"]


def test_hindi_preference_changes_prompt_language(client, uid, mock):
    chat(client, uid, "s1", "Please reply in Hindi from now on")
    assert client.get(f"/users/{uid}/profile").json()["language"] == "Hindi"
    chat(client, uid, "s1", "What about my career?")
    assert "Respond in Hindi" in mock.calls[-1]["system"]


def test_memory_persists_across_app_restart(live_brain, make_client, uid):
    chat(make_client(brain=live_brain), uid, "s1", FIRST)
    second_app = make_client(brain=live_brain)  # fresh app + fresh SQLite, same graph
    r = chat(second_app, uid, "new", "What do you remember about my career goals?")
    assert "career change" in r["response"]


def test_extractor_is_shown_existing_memory_keys_so_it_reuses_them(client, uid, mock):
    chat(client, uid, "s1", FIRST)
    chat(client, uid, "s1", "I've decided I'm not going to switch jobs after all")
    extraction = [c for c in mock.calls if c["role"] == "extract"][-1]["system"]
    assert "goal:career_change: career change" in extraction  # existing key is offered for reuse
    assert memories(client, uid) == []  # and the retract targeted that key -> nothing active remains
