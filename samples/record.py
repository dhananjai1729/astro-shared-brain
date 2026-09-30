"""Record a real conversation (Ollama by default) to samples/transcript.json.
    LLM_PROVIDER=ollama python -m samples.record      (or anthropic, with ANTHROPIC_API_KEY)
"""
import json, logging, os, sys, tempfile, uuid
from pathlib import Path
from fastapi.testclient import TestClient
from app.brain.store import Brain
from app.config import Settings
from app.main import create_app

logging.disable(logging.CRITICAL)
uid = f"t-sample-{uuid.uuid4().hex[:6]}"
steps = [
    ("s1", "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."),
    ("s1", "What should I focus on for my career?"),
    ("s1", "Why do you say that?"),
    ("s2", "What do you remember about my career goals?"),
    ("s2", "Actually I decided not to switch jobs, I'm starting a business instead."),
    ("s3", "What should I focus on for my career now?"),
    ("s3", "Will I find love and marriage soon?"),
]
brain = Brain("bolt://localhost:7687", "neo4j", "brain-pass")
s = Settings(sqlite_path=str(Path(tempfile.mkdtemp()) / "s.db"), memory_update_mode="inline",
             llm_provider=os.environ.get("LLM_PROVIDER", "ollama"))
out = []
with TestClient(create_app(s, brain=brain)) as c:
    for sid, msg in steps:
        req = {"user_id": uid, "session_id": sid, "message": msg}
        r = c.post("/chat", json=req)
        out.append({"request": req, "status": r.status_code, "response": r.json()})
        print(sid, msg[:40], "->", r.json()["context_used"], flush=True)
    mem = c.get(f"/users/{uid}/memories", params={"include_inactive": True}).json()
    prof = c.get(f"/users/{uid}/profile").json()
text = json.dumps({"provider": s.llm_provider, "chat": out, "memories_after": mem, "profile_after": prof}, indent=2)
Path("samples/transcript.json").write_text(text.replace(uid, "user-123"))
brain.delete_users_with_prefix("t-sample-")
print("saved")
