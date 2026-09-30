import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.brain.store import Brain, BrainUnavailable
from app.chat.service import ChatService
from app.config import Settings, get_settings
from app.llm.factory import build_llm
from app.memory.classifier import build_classifier
from app.memory.updater import MemoryUpdater
from app.models import ChatRequest, ChatResponse, ID_PATTERN, Profile, ProfileIn
from app.session.store import SessionStore

import re

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j").setLevel(logging.ERROR)
log = logging.getLogger("sharedbrain")


def build_services(s: Settings, llm=None, brain: Brain | None = None):
    """Wire everything. The brain may be unreachable at boot; the API degrades instead of failing."""
    sessions = SessionStore(s.sqlite_path)
    classifier = build_classifier(s.classifier)
    llm = llm or build_llm(s)
    if brain is None:
        brain = Brain(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
        try:
            brain.wait_until_ready(attempts=3, delay=1.0)
            brain.ensure_schema()
        except BrainUnavailable as e:
            log.warning("Neo4j not ready at startup (%s); running degraded, will retry on use", e)
    updater = MemoryUpdater(brain, llm, classifier, sessions, s)
    return ChatService(brain, llm, classifier, sessions, updater, s), brain, sessions, updater


def create_app(settings: Settings | None = None, llm=None, brain: Brain | None = None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_brain = brain is None
        svc, br, sessions, updater = build_services(s, llm, brain)
        app.state.svc, app.state.brain, app.state.sessions, app.state.updater = svc, br, sessions, updater
        try:
            updater.drain_outbox()
        except Exception:
            log.exception("outbox drain failed at startup")
        yield
        if owns_brain:
            br.close()

    app = FastAPI(title="Shared Brain API", lifespan=lifespan)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, background: BackgroundTasks):
        resp, update = app.state.svc.chat(req)
        if s.memory_update_mode == "inline":
            update()
        else:
            background.add_task(update)
        return resp

    def _check_uid(user_id: str):
        if not re.match(ID_PATTERN, user_id):
            raise HTTPException(422, "invalid user_id")

    @app.put("/users/{user_id}/profile", response_model=Profile)
    def put_profile(user_id: str, body: ProfileIn):
        _check_uid(user_id)
        fields = {k: (v.isoformat() if k == "dob" and v else v) for k, v in body.model_dump().items()}
        try:
            prof = app.state.brain.update_profile(user_id, fields)
        except BrainUnavailable:
            raise HTTPException(503, "shared brain unavailable")
        app.state.sessions.cache_profile(user_id, prof.model_dump())
        return prof

    @app.get("/users/{user_id}/profile", response_model=Profile)
    def get_profile(user_id: str):
        _check_uid(user_id)
        try:
            prof = app.state.brain.get_profile(user_id)
        except BrainUnavailable:
            cached = app.state.sessions.cached_profile(user_id)
            if cached:
                return Profile(**cached)
            raise HTTPException(503, "shared brain unavailable")
        if not prof:
            raise HTTPException(404, "user not found")
        return prof

    @app.get("/users/{user_id}/memories")
    def memories(user_id: str, include_inactive: bool = False):
        _check_uid(user_id)
        try:
            return {"user_id": user_id, "memories": app.state.brain.list_memories(user_id, include_inactive)}
        except BrainUnavailable:
            raise HTTPException(503, "shared brain unavailable")

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        try:
            app.state.brain.driver.verify_connectivity()
            graph = "up"
        except Exception:
            graph = "down"
        return {"status": "ok", "graph": graph, "llm": [p.name for p in app.state.svc.llm.providers]
                if hasattr(app.state.svc.llm, "providers") else [app.state.svc.llm.name], "outbox_pending": app.state.sessions.pending_count()}

    return app


app = create_app()
