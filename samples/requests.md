# Sample API requests

Start the stack: `docker compose up -d neo4j && .venv/bin/uvicorn app.main:app` (LLM via `.env`).
A full recorded multi-session conversation (real Ollama `qwen2.5:7b-instruct` output) is in
[`transcript.json`](transcript.json); regenerate with `LLM_PROVIDER=ollama python -m samples.record`.

## First conversation — profile facts + goal are extracted after the reply
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "user_id": "user-123", "session_id": "session-1",
  "message": "My name is Rahul. I was born on 15 August 1995 in Delhi. I'"'"'m planning to switch jobs next year."}'
```
```json
{"response": "Hello Rahul! ...", "user_id": "user-123", "session_id": "session-1",
 "context_used": [], "degraded": false, "warnings": []}
```

## Later, same session — memory + profile used
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "user_id": "user-123", "session_id": "session-1", "message": "What should I focus on for my career?"}'
```
```json
{"response": "Given your Leo sun sign ...", "user_id": "user-123", "session_id": "session-1",
 "context_used": ["user_profile", "career_goal"], "degraded": false, "warnings": []}
```

## Follow-up — answered from recent turns only (no long-term retrieval)
```json
{"message": "Why do you say that?"}  ->  "context_used": ["user_profile"]
```

## New session — long-term memory
```json
{"session_id": "session-2", "message": "What do you remember about my career goals?"}
 ->  "context_used": ["user_profile", "career_goal"]
```

## Irrelevant memory is not injected
```json
{"message": "Will I find love and marriage soon?"}  ->  "context_used": ["user_profile"]
```

## Profile API
```bash
curl -s -X PUT localhost:8000/users/user-123/profile -H 'content-type: application/json' \
  -d '{"name":"Rahul","dob":"1995-08-15","tob":"06:30","birth_place":"Delhi","language":"Hindi"}'
# -> {"user_id":"user-123","name":"Rahul","dob":"1995-08-15","tob":"06:30","birth_place":"Delhi",
#     "language":"Hindi","sun_sign":"Leo"}
curl -s 'localhost:8000/users/user-123/memories?include_inactive=true'   # includes superseded history
curl -s localhost:8000/health    # {"status":"ok","graph":"up","outbox_pending":0}
```

## Errors / degradation
```text
POST /chat {"message": ""}                      -> 422
PUT  profile {"dob": "not-a-date"}              -> 422
LLM providers all down                          -> 200, "degraded": true, "warnings": ["llm unavailable"]
Neo4j down                                      -> 200, "degraded": true, "warnings": ["shared brain unavailable; ..."]
```
