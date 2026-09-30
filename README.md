# Personalized Astrology Chat with a Shared Brain

A FastAPI service where an LLM answers astrology questions using **short-term session context** and a
**persistent graph of long-term user memory (the Shared Brain, in Neo4j)**.

```
POST /chat → validate → load profile → select context → LLM → respond → [background] update memory
```

## For reviewers

**Run the working app (one command, no API key, no setup)** — needs only Docker:
```bash
docker compose up --build        # Neo4j + API
# then open http://localhost:8000/   (chat UI + live view of the graph memories)
```
With no key and no Ollama the app runs in a clearly labelled **demo mode**: the graph, memory extraction
(rule-based), retrieval, corrections and profile/moon-sign logic are all real; only the reply text is composed
from the retrieved context instead of written by an LLM (each reply is prefixed `[demo mode…]` and carries a
warning). For real answers add `ANTHROPIC_API_KEY` to `.env` (copy `.env.example`) and/or run Ollama
(`ollama pull qwen2.5:7b-instruct`); the default chain is `anthropic → ollama → demo`, so a real model is
always preferred when available. Suggested script in the UI: use the suggestion chips in order.

**Verify the engineering (≈2 min)**
1. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (Neo4j from step above is running)
2. `.venv/bin/python -m pytest tests -q` — 80 tests incl. the 8 scenarios from the brief against a real graph
   (`test_1_…` to `test_8_…` in `tests/test_scenarios.py`; deterministic mock LLM).
3. `.venv/bin/python -m eval.run_eval` — Shared Brain ON vs OFF (memory accuracy 1.00 vs 0.17, personalization 1.00 vs 0.33).
4. [`samples/transcript.json`](samples/transcript.json) — a recorded multi-session conversation with a real local model.

**Where each requirement of the brief lives**

| Brief item | Implementation | Verified by |
|---|---|---|
| 1. Chat API | `POST /chat` in `app/main.py`, orchestration in `app/chat/service.py` | `tests/test_scenarios.py` |
| 2. User profile (+ stubbed astrology) | `PUT /users/{id}/profile`, `app/profile/astro.py` (sun sign, tropical + sidereal moon sign) | `test_units.py`, `test_failures.py` |
| 3. Shared Brain (Neo4j, schema) | `app/brain/store.py`; schema under "Shared Brain schema" below | scenarios 2, 3, 5, 7 |
| 4. Short-term vs long-term memory | `app/session/store.py` (SQLite) vs `app/memory/updater.py` (graph) | scenarios 4, 5; small talk not stored |
| 5. Context selection | `app/chat/context.py`, `app/memory/classifier.py`, `ranking.py` | scenarios 3, 6; `context_used` in every response |
| 6. LLM layer (modular) | `app/llm/` — Anthropic, Ollama, Mock + fallback chain | provider/fallback tests |
| 7. Memory update (what / not / represent / update) | `app/memory/updater.py`, `rules.py`, `brain/store.py` (`SUPERSEDES`) | scenario 7, expiry, retract tests |
| 8. Testing & evaluation | `tests/` (8 scenarios + units + failures), `eval/run_eval.py` | `pytest`, eval script |
| 9. Error handling | degraded mode, outbox, fallbacks — see "Error handling" | `tests/test_failures.py` |
| Bonus done | importance/confidence scores, conflict resolution, decay/expiry, model fallback, token budget, Hindi/multilingual prompt, plus moon signs and an optional Laya classifier | see sections below |
| Bonus not done | conversation summarization, advanced graph traversal | — |

**Honest status.** Verified for real: the graph, all API flows, the tests, the eval, the Docker image, and real
extraction + chat with a local Ollama model. **Not run against the real Anthropic API** (no key was available
while building); that provider is covered by stub tests only — see "Verified vs not" at the end.

## Quick start

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY (or set LLM_PROVIDER=ollama / mock)
docker compose up -d neo4j      # graph DB (Docker Desktop must be running)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload          # API on :8000  (docs: /docs)
.venv/bin/python -m pytest tests -q              # tests (scenario tests need Neo4j, others don't)
.venv/bin/python -m eval.run_eval                # Shared Brain on-vs-off evaluation
```

Or everything in containers: `docker compose up --build` (the API reaches host Ollama at `host.docker.internal`).

**LLM selection** — `LLM_PROVIDER` is a comma list forming a fallback chain (default `anthropic,ollama,demo`):
`anthropic` (default models: `claude-opus-5-5` chat, `claude-haiku-4-5` extraction), `ollama`
(`qwen2.5:7b-instruct`), `demo` (no-LLM fallback, see above), `mock` (deterministic, used by tests). Providers are tried in order;
remove `demo` to require a real model.

### API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | `{user_id, session_id, message}` → `{response, user_id, session_id, context_used[], degraded, warnings[]}` |
| `PUT /users/{id}/profile` | upsert name / dob / tob / birth_place / language / utc_offset; **sun sign** (from dob) and **moon sign** — tropical and sidereal/Vedic — (from dob + tob + utc_offset) are computed |
| `GET /users/{id}/profile`, `GET /users/{id}/memories?include_inactive=` | inspect the brain |
| `GET /health` | graph status + pending outbox writes |
| `GET /` | test UI: chat, graph memories, profile, `context_used` |

Sample requests/responses: [`samples/requests.md`](samples/requests.md) (curl) and
[`samples/transcript.json`](samples/transcript.json) (a recorded multi-session conversation).
`context_used` lists what was injected into the prompt (`user_profile`, `career_goal`, …);
`degraded`/`warnings` are the extension to the response format.

## Architecture

```
app/
  main.py            FastAPI wiring, lifespan (schema init, outbox drain), endpoints
  chat/service.py    orchestration of one turn (degrades instead of failing)
  chat/context.py    context selection  (query understanding → graph retrieval → budget)
  chat/prompts.py    chat + extraction prompts
  brain/store.py     Neo4j Shared Brain (schema, supersede logic, retrieval queries)
  memory/classifier  life-area routing, "worth remembering" gate, follow-up detection (rules | Laya)
  memory/updater.py  post-response memory update (gate → extract → validate → write/outbox)
  memory/rules.py    regex extractor: fallback when the LLM fails, and the mock LLM's brain
  memory/ranking.py  relevance × importance × confidence × time-decay
  session/store.py   SQLite: short-term turns, profile cache, write outbox
  llm/               LLMProvider interface; Anthropic, Ollama, Mock, FallbackChain
  profile/astro.py   light astrology: sun sign (dob), tropical + sidereal moon sign (low-precision lunar longitude)
```

The LLM layer is one 1-method interface (`generate(system, messages, role, ...)`); swapping providers
is configuration, not code. `role="extract"` lets extraction use a cheaper model than chat.

## Shared Brain schema

```
(User {id,name,dob,tob,birth_place,language,sun_sign})
  -[:HAS_MEMORY]->(Memory {id,kind,key,value,text,confidence,importance,status,
                           created_at,updated_at,expires_at,target_year,raw_phrase})
        -[:ABOUT]->(LifeArea {name})                 career|relationships|health|finance|education|family|spirituality|personal_growth|general
        -[:SUPERSEDES]->(Memory)                      older version, status='superseded' (kept for audit)
        -[:DESCRIBES]->(Goal | Topic | Preference)    typed projection of the memory
  -[:HAS_GOAL]->(Goal {title,target_year,status})     e.g. "career change", 2027
  -[:INTERESTED_IN]->(Topic {name})
  -[:PREFERS]->(Preference {name})                    e.g. Hindi
  -[:HAS_ZODIAC]->(Sign {name})                       sun sign, computed from dob
  -[:HAS_MOON_SIGN {system}]->(Sign {name})           tropical + sidereal (Lahiri) moon sign, from dob + tob + utc_offset
```

- **Entities:** users, memories (first-class nodes so they carry confidence/status/history), life areas
  (shared, so retrieval is a graph hop), and typed nodes that mirror the example tree in the brief
  (`HAS_GOAL → Career Change`, `PREFERS → Hindi`, `HAS_ZODIAC → Leo`).
- **Why a `Memory` node rather than only typed edges:** corrections, history, expiry and scoring all need
  properties *on the fact*. An edge-only model makes "supersede but keep history" awkward.
- **Retrieval:** `User → HAS_MEMORY → Memory(active, unexpired) → ABOUT → LifeArea ∈ matched areas`
  (Cypher), then ranked in Python (testable) and trimmed to budget.
- **Why Neo4j:** the preferred option, real Cypher/constraints, inspectable in the browser at `:7474`.
  The store sits behind a small class, so another graph backend is a contained change.

## Astrology computations
- **Sun sign:** tropical date ranges from the date of birth.
- **Moon sign (tropical and sidereal):** computed from a truncated Meeus lunar series (no dependency; ~0.2–0.3°,
  verified against reference positions in the tests). The **sidereal / Vedic moon sign (Rashi)** is the tropical
  longitude minus the **Lahiri ayanamsa** (23.853° at J2000, +0.01397°/yr; checked against published values for
  1950/2000/2025). Both are stored on the profile (`moon_sign`, `moon_sign_sidereal`) and as
  `HAS_MOON_SIGN {system}` edges; they usually differ by one sign. The prompt carries both, labelled, and tells the
  model to use the system the user asks about (Vedic/Rashi/sidereal → sidereal), otherwise lead with tropical and
  name the system.
- **Needs a time:** the Moon changes sign every ~2.3 days, so it uses the birth time and `utc_offset` (default IST
  +5.5; birth place → timezone is not resolved). With no birth time it assumes noon and stores a note; if the Moon
  changed signs that day the note says the result is **uncertain**. A birth time within ~0.3° of a sign boundary
  (~33 min of Moon travel) is flagged the same way. Recomputed whenever dob / tob / utc_offset change.
- **Limits:** the sun sign is still tropical only (a Vedic sun sign would need a solar longitude; the Rashi is
  moon-based so it does not block this). Only the Lahiri ayanamsa is supported (no Raman/KP), mean ayanamsa
  (nutation ignored). No ascendant/houses/nakshatra, and historical/DST offsets are not modelled.

## Memory strategy

**Short-term:** last 6 turns of the session (user *and* assistant) from SQLite, sent on every call. A
follow-up like "Why do you say that?" is answered from this alone — no graph retrieval.

**Long-term, after the response** (background task; inline in tests):
1. **Gate** (rules): trivia ("thanks", "ok"), bare questions, and text with no first-person/directive
   content are never sent to the extractor — most turns cost zero extra LLM calls.
2. **Extract** (LLM, JSON): goals, plans, preferences, interests, life events, profile facts, corrections.
   Today's date is in the prompt so "next year" becomes `target_year: 2027` (the raw phrase is stored too).
   Not stored: questions, small talk, the assistant's own statements, one-off trivia.
3. **Validate:** schema repair (small models omit fields), confidence ≥ 0.6, de-duplication.
4. **Fallback:** if the LLM fails or returns junk, a regex extractor covers the obvious patterns.
5. **Write:** profile facts go to `User` properties (and update `HAS_ZODIAC` / `PREFERS`); everything else
   becomes a `Memory`.

**Updates/conflicts.** Every memory has a stable `key` (`goal:career_change`, `interest:astrology`).
- same key, same value → bump confidence/recency (no duplicate)
- same key, new value → new memory `SUPERSEDES` the old; old becomes `superseded` and stops being retrieved
- `action: retract` ("I'm not switching jobs after all") → old memory superseded, nothing replaces it
- `replaces_key` → explicit cross-key contradiction
- profile corrections ("actually born in Mumbai") overwrite the field in place

Nothing is hard-deleted. **Decay/expiry:** time-boxed goals ("interview next month") get `expires_at`;
ranking applies a 180-day half-life.

## Context-selection approach

1. **Always:** compact profile line + last 6 turns.
2. **Understand the query:** keyword → life-area classifier; follow-ups skip retrieval.
3. **Retrieve:** memories in the matched areas (or, for "what do you remember…", the top memories overall).
   A topical question never receives off-topic memories (asked about love → the career goal is not injected).
4. **Rank & budget:** relevance × importance × confidence × decay, max 5 memories, 1,500 characters.
5. **Report:** `context_used` returns what was included.

The full graph and full history are never sent. `CLASSIFIER=laya` upgrades step 2's area routing with a
zero-shot [Laya](https://huggingface.co/convaiinnovations/laya) fallback: rules decide keyword-clear cases,
Laya only the keyword-free ones, errors fall back to rules (`pip install -r requirements-laya.txt`; first
use downloads ~1.7 GB). The keep/skip gate stays rule-based (measured below).

### Rules vs Laya (measured)
`python -m eval.compare_classifiers` — 38 labelled area messages (half keyword-free paraphrases) and 32 keep/skip
messages, zero-shot Laya 0.3.22 English checkpoint, CPU/MPS:

| Life-area routing | overall | keyword msgs | paraphrases | median latency |
|---|---|---|---|---|
| rules | 0.63 | 1.00 | 0.22 | <0.1 ms |
| Laya alone | 0.74 | 0.80 | 0.67 | ~67 ms |
| **hybrid (rules → Laya)** | **0.82** | 1.00 | 0.61 | ~0 ms on keyword msgs |

| Worth-remembering gate | accuracy | precision | recall |
|---|---|---|---|
| **rules** | **0.97** | 0.94 | 1.00 |
| Laya alone (p≥0.5) | 0.59 | 1.00 | 0.19 (AUC 0.59) |

Takeaways: zero-shot Laya is useful for *routing* (big gain on paraphrases, e.g. "Should I accept the offer
from the new firm?" now retrieves the career goal) but poor as a *keep/skip gate* (it scores clear goals
like "I'm planning to switch jobs next year" at only p≈0.2), so the gate is not delegated to it.
Caveats: small, author-written sets; the keyword half favours rules by construction; the rules were written
before this set but by the same author, so their gate score is optimistic. Fine-tuning Laya on logged
conversations is the obvious next step for the gate.

## Error handling

| Failure | Behaviour |
|---|---|
| invalid input | 422 (empty/oversized message, bad ids, bad dob/tob) |
| LLM failure | fallback chain (e.g. Claude → Ollama); if all fail, a canned reply with `degraded: true` |
| Neo4j down | `/chat` still answers from the SQLite profile cache + recent turns, `degraded: true`; profile endpoints 503 |
| graph write fails | extracted memories go to a SQLite **outbox**, replayed on the next update and at startup |
| extractor returns garbage | regex fallback; a failed memory update never affects the response |
| empty memory / no relevant context | normal path; the prompt simply has no memory section |
| missing profile fields | prompt states them as `unknown`; the model is told not to invent facts |

## Tests & evaluation

`tests/` — unit (sun sign, classifier, extraction rules, ranking, validation, provider and Laya
adapters with stubs), the 8 brief scenarios through `POST /chat` against real Neo4j (new user, create,
retrieve, follow-up, new session, irrelevant memory, correction, missing info) plus expiry, Hindi,
restart persistence, and failure paths (LLM down, graph down, outbox replay, garbage extraction).
Scenario tests use a deterministic mock LLM that echoes its context and **skip cleanly if Neo4j is absent**;
graph-down tests need no Docker. Test users are prefixed `t-` and cleaned up by prefix only.

**Evaluating whether the Shared Brain helps** — `python -m eval.run_eval` runs 6 scripted conversations
with the brain ON vs OFF (graph unreachable) and scores:

| Metric | What it checks |
|---|---|
| memory accuracy | exactly the expected memories were stored (no junk, no misses) |
| context recall / precision | the right memories were retrieved, nothing extra |
| irrelevant-context rate | off-topic memory injected into the prompt |
| personalization | the answer reflects stored facts and not superseded ones |
| persistence | memories survive a brand-new app instance and session |

Latest run: memory accuracy 1.00 vs 0.17, personalization 1.00 vs 0.33, recall 1.00 vs 0.50,
persistence 1.00, irrelevant-context 0.00 (ON vs OFF). Caveat: the mock LLM makes this a test of
*the system* (what gets extracted, retrieved and put in the prompt), not of answer quality; OFF-mode
precision is vacuous because nothing is retrieved. For production I would add an LLM-as-judge on real
model output (personalization, consistency with earlier turns) over logged conversations, human-labelled
retrieval relevance, and online signals (thumbs, follow-up "I already told you" rate).

## Key decisions and trade-offs

- **LLM extraction + rule gate/fallback** over rules-only (brittle) or LLM-only (wasteful, no fallback).
- **Keyword life-area routing over embeddings:** explainable, zero dependencies, good enough for a
  handful of life areas; embeddings/vector index would be the next step for vague queries.
- **Ranking in Python, filtering in Cypher:** graph does traversal, scoring stays unit-testable.
- **Sessions in SQLite, not the graph:** chat logs don't belong in the knowledge graph; the interface
  swaps to Redis/Postgres for multi-worker deployments.
- **Soft supersede instead of delete:** auditability and undo at the cost of some extra nodes.
- **Extraction after the response:** lower latency; a memory said this turn is usable next turn, not this one.
- **Laya is optional and zero-shot, used only where it measured well:** area routing for keyword-free
  messages. Rules stay the default and own the keep/skip gate. Future work: fine-tune on logged conversations.

## Production considerations

- **Scale/ops:** run multiple API workers (sessions → Redis/Postgres, outbox → a real queue such as
  SQS/Redis streams with a worker), managed Neo4j (Aura) with backups, connection pooling (already in the driver).
- **Privacy:** birth details and personal goals are sensitive — encryption at rest, per-user
  export/delete endpoints (GDPR/DPDP), PII-aware logging, consent for memory. Memories are scoped by `user_id`
  in every query.
- **Security:** auth on all endpoints (user_id must come from a token, not the body); prompt-injection
  hardening since users can write memories ("remember that I'm an admin") — memory text is injected as
  data, never as instructions, and is length-capped.
- **Quality:** versioned prompts, extraction eval set, memory review/confidence thresholds, rate limits and
  token/cost budgets per user, caching of stable prompt prefixes.
- **Concurrency:** memory writes for one user should be serialised (per-user lock/queue) to avoid races on
  the same key.
- **Not done (bonus):** conversation summarization, multi-hop graph traversal, real ephemeris calculations.
- **Model quality caveat:** with the local 7B model, prompt wording mattered a lot. Early prompts made it
  ignore injected memories ("I don't have information…"); the first fix made it invent memories when none
  existed. The final prompt lists memories as things the assistant remembers, renders an explicit
  "nothing stored" marker when empty, and forbids attributing anything else to the user. Residual
  over-attribution from chat history is still possible on small models; hence the judge-based eval above.
- **Verified vs not (honest status):**
  - Verified for real: Neo4j graph, all API flows, 67 automated tests, the eval script, the Docker image
    (API container → Neo4j, and → host Ollama via `host.docker.internal`), and real extraction + chat with
    Ollama `qwen2.5:7b-instruct` (see `samples/transcript.json`).
  - **Not run against the real Anthropic API** (no API key was available during development). The provider
    is covered by stub tests for request shape, error mapping, refusals, and the Opus 5.5 specifics (thinking
    always on → chat gets a 4096-token floor and `effort: low`; `effort` is not sent to the Haiku extractor;
    base URL is explicit). Expect to tune the first real run.
  - Laya was installed and measured for real (section "Rules vs Laya"); `CLASSIFIER=laya` was also run end to end
    through the app. It needs torch and a ~1.7 GB checkpoint, so it is an opt-in extra.
