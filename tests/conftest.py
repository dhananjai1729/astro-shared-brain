import logging
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.brain.store import Brain
from app.config import Settings
from app.llm.mock_provider import MockProvider
from app.main import create_app

logging.getLogger("neo4j").setLevel(logging.ERROR)
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
PW = os.environ.get("NEO4J_PASSWORD", "brain-pass")


def _brain_or_none():
    try:
        b = Brain(URI, "neo4j", PW, connect_timeout=2)
        b.driver.verify_connectivity()
        b.ensure_schema()
        return b
    except Exception:
        return None


@pytest.fixture(scope="session")
def live_brain():
    b = _brain_or_none()
    if b is None:
        pytest.skip("Neo4j not available (run: docker compose up -d neo4j)")
    yield b
    b.delete_users_with_prefix("t-")
    b.close()


@pytest.fixture
def uid():
    return f"t-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def mock():
    return MockProvider()


@pytest.fixture
def make_client(tmp_path, mock):
    clients = []

    def _make(brain=None, llm=None, **overrides):
        s = Settings(llm_provider="mock", sqlite_path=str(tmp_path / "t.db"), memory_update_mode="inline",
                     neo4j_uri=URI, neo4j_password=PW, **overrides)
        c = TestClient(create_app(s, llm=llm or mock, brain=brain))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(live_brain, make_client):
    return make_client(brain=live_brain)
