"""A/B harness: same scripted conversations with the Shared Brain ON vs OFF (graph unreachable).

Deterministic (mock LLM echoes the context it was given), so it measures the *system*: what was
extracted, what was retrieved, what reached the prompt. For real-model quality, swap the provider
and add an LLM judge (see README "Evaluation").

    python -m eval.run_eval            # needs Neo4j: docker compose up -d neo4j
"""
import json
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.brain.store import Brain
from app.config import Settings
from app.llm.mock_provider import MockProvider
from app.main import create_app

logging.disable(logging.CRITICAL)
FIRST = "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."

CASES = [
    {"name": "recall_goal_new_session", "setup": [FIRST], "probe": "What do you remember about my career goals?",
     "new_session": True, "want_labels": {"career_goal"}, "want_text": ["career change"], "stored": {"goal:career_change"}},
    {"name": "topical_career_question", "setup": [FIRST], "probe": "What should I focus on for my career?",
     "want_labels": {"career_goal"}, "want_text": ["career change", "Rahul"], "stored": {"goal:career_change"}},
    {"name": "irrelevant_memory_excluded", "setup": [FIRST], "probe": "Will I find love and marriage soon?",
     "want_labels": set(), "forbid_labels": {"career_goal"}, "forbid_text": ["career change"], "stored": {"goal:career_change"}},
    {"name": "correction_supersedes", "setup": [FIRST, "I decided not to switch jobs, I'm starting a business instead."],
     "probe": "What should I focus on for my career?", "want_labels": {"career_goal"},
     "want_text": ["start a business"], "forbid_text": ["career change"], "stored": {"goal:start_business"}},
    {"name": "profile_personalization", "setup": [FIRST], "probe": "What does my horoscope say about money?",
     "want_labels": set(), "want_text": ["Leo", "Rahul"], "stored": {"goal:career_change"}},
    {"name": "small_talk_not_stored", "setup": ["thanks", "ok cool"], "probe": "Tell me about my week",
     "want_labels": set(), "stored": set()},
]


def run_case(case, brain, persist_check: bool):
    with tempfile.TemporaryDirectory() as d:
        s = Settings(llm_provider="mock", sqlite_path=str(Path(d) / "e.db"), memory_update_mode="inline")
        uid = f"t-eval-{uuid.uuid4().hex[:8]}"
        post = lambda c, sid, m: c.post("/chat", json={"user_id": uid, "session_id": sid, "message": m}).json()
        with TestClient(create_app(s, llm=MockProvider(), brain=brain)) as c:
            for m in case["setup"]:
                post(c, "setup", m)
            stored = set()
            try:
                stored = {m["key"] for m in brain.list_memories(uid)}
            except Exception:
                pass
            out = post(c, "probe-new" if case.get("new_session") else "setup", case["probe"])
        persisted = None
        if persist_check:  # brand-new app + SQLite, same graph
            with tempfile.TemporaryDirectory() as d2:
                s2 = Settings(llm_provider="mock", sqlite_path=str(Path(d2) / "e2.db"), memory_update_mode="inline")
                with TestClient(create_app(s2, llm=MockProvider(), brain=brain)) as c2:
                    keys = {m["key"] for m in brain.list_memories(uid)}
                    persisted = (case["stored"] <= keys) if case["stored"] else None  # nothing to persist: n/a
        return out, stored, persisted


def score(case, out, stored):
    labels, text = set(out["context_used"]) - {"user_profile"}, out["response"]
    want, forbid = case.get("want_labels", set()), case.get("forbid_labels", set())
    return {
        "memory_accuracy": stored == case["stored"],
        "context_recall": want <= labels,
        "context_precision": (len(labels & want) / len(labels)) if labels else 1.0,
        "irrelevant_context": bool(labels - want),
        "personalized": all(t in text for t in case.get("want_text", [])) and not any(t in text for t in case.get("forbid_text", [])),
    }


def main():
    on = Brain("bolt://localhost:7687", "neo4j", "brain-pass")
    off = Brain("bolt://localhost:59999", "neo4j", "x", connect_timeout=1)  # brain unreachable == brain off
    try:
        on.driver.verify_connectivity()
    except Exception:
        raise SystemExit("Neo4j not reachable. Start it: docker compose up -d neo4j")
    results = {"on": [], "off": []}
    for mode, brain in (("on", on), ("off", off)):
        for case in CASES:
            out, stored, persisted = run_case(case, brain, persist_check=(mode == "on"))
            r = score(case, out, stored)
            r["case"] = case["name"]
            if persisted is not None:
                r["persistence"] = persisted
            results[mode].append(r)
    on.delete_users_with_prefix("t-eval-")
    metrics = ["memory_accuracy", "context_recall", "context_precision", "irrelevant_context", "personalized", "persistence"]
    print(f"{'metric':<22}{'brain ON':>10}{'brain OFF':>11}")
    summary = {}
    for m in metrics:
        row = {}
        for mode in ("on", "off"):
            vals = [r[m] for r in results[mode] if m in r]
            row[mode] = round(sum(vals) / len(vals), 2) if vals else None
        summary[m] = row
        print(f"{m:<22}{row['on'] if row['on'] is not None else '-':>10}{row['off'] if row['off'] is not None else '-':>11}")
    print("\nper-case failures (brain ON):", [(r['case'], k) for r in results["on"] for k, v in r.items()
                                              if k not in ('case', 'irrelevant_context') and v is False or (k == 'context_precision' and v < 1)] or "none")
    Path("eval/results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
