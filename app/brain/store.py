"""Shared Brain: Neo4j graph of User -> Memory/Goal/Topic/Preference/Sign/LifeArea.

Schema
  (User {id,name,dob,tob,birth_place,language,sun_sign})
  (User)-[:HAS_MEMORY]->(Memory {id,kind,key,value,text,confidence,importance,status,
                                 created_at,updated_at,expires_at,target_year,raw_phrase})
  (Memory)-[:ABOUT]->(LifeArea {name})
  (Memory)-[:SUPERSEDES]->(Memory)            old version stays, status='superseded'
  (User)-[:HAS_GOAL]->(Goal {id,title,target_year})       (Memory)-[:DESCRIBES]->(Goal)
  (User)-[:INTERESTED_IN]->(Topic {name})
  (User)-[:PREFERS]->(Preference {name})
  (User)-[:HAS_ZODIAC]->(Sign {name})                     sun sign, from dob
  (User)-[:HAS_MOON_SIGN {system}]->(Sign {name})         tropical + sidereal(Lahiri) moon sign, from dob + tob + utc_offset
"""
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.models import LIFE_AREAS, ExtractedItem, Profile
from app.profile.astro import moon_sign, sun_sign

log = logging.getLogger(__name__)

PROFILE_FIELDS = ("name", "dob", "tob", "birth_place", "language", "utc_offset")


class BrainUnavailable(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Brain:
    def __init__(self, uri: str, user: str, password: str, connect_timeout: float = 5.0):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), connection_timeout=connect_timeout,
            max_transaction_retry_time=3, connection_acquisition_timeout=connect_timeout)

    def close(self):
        self.driver.close()

    # ---- infra -----------------------------------------------------------------
    def _run(self, query: str, **params):
        try:
            with self.driver.session() as s:
                return s.run(query, **params).data()
        except (ServiceUnavailable, Neo4jError, OSError) as e:
            raise BrainUnavailable(str(e)) from e

    def wait_until_ready(self, attempts: int = 30, delay: float = 2.0):
        last = None
        for _ in range(attempts):
            try:
                self.driver.verify_connectivity()
                return
            except Exception as e:  # container still booting
                last = e
                time.sleep(delay)
        raise BrainUnavailable(str(last))

    def ensure_schema(self):
        for q in (
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT area_name IF NOT EXISTS FOR (a:LifeArea) REQUIRE a.name IS UNIQUE",
            "CREATE INDEX memory_key IF NOT EXISTS FOR (m:Memory) ON (m.key)",
        ):
            self._run(q)
        self._run("UNWIND $names AS n MERGE (:LifeArea {name: n})", names=LIFE_AREAS)

    # ---- users / profile -------------------------------------------------------
    def ensure_user(self, user_id: str) -> None:
        self._run("MERGE (u:User {id:$id}) ON CREATE SET u.created_at=$now", id=user_id, now=_now().isoformat())

    def get_profile(self, user_id: str) -> Profile | None:
        rows = self._run("MATCH (u:User {id:$id}) RETURN u", id=user_id)
        if not rows:
            return None
        u = rows[0]["u"]
        return Profile(user_id=user_id, **{f: u.get(f) for f in (*PROFILE_FIELDS, "sun_sign", "moon_sign", "moon_note",
                                                                        "moon_sign_sidereal", "moon_note_sidereal")})

    def update_profile(self, user_id: str, fields: dict) -> Profile:
        fields = {k: v for k, v in fields.items() if k in PROFILE_FIELDS and v not in (None, "")}
        self.ensure_user(user_id)
        if fields:
            if "dob" in fields:
                fields["sun_sign"] = sun_sign(datetime.fromisoformat(str(fields["dob"])).date())
            self._run("MATCH (u:User {id:$id}) SET u += $f", id=user_id, f=fields)
            if "sun_sign" in fields:
                self._run(
                    "MATCH (u:User {id:$id}) OPTIONAL MATCH (u)-[r:HAS_ZODIAC]->() DELETE r "
                    "WITH DISTINCT u MERGE (s:Sign {name:$s}) MERGE (u)-[:HAS_ZODIAC]->(s)",
                    id=user_id, s=fields["sun_sign"])
            cur = self.get_profile(user_id)  # moon signs depend on dob + tob + utc_offset: recompute on any change
            if cur.dob and {"dob", "tob", "utc_offset"} & fields.keys():
                d = datetime.fromisoformat(str(cur.dob)).date()
                trop, tnote = moon_sign(d, cur.tob, cur.utc_offset)
                sid, snote = moon_sign(d, cur.tob, cur.utc_offset, sidereal=True)
                self._run("MATCH (u:User {id:$id}) SET u.moon_sign=$t, u.moon_note=$tn, "
                          "u.moon_sign_sidereal=$s, u.moon_note_sidereal=$sn", id=user_id, t=trop, tn=tnote, s=sid, sn=snote)
                self._run(
                    "MATCH (u:User {id:$id}) OPTIONAL MATCH (u)-[r:HAS_MOON_SIGN]->() DELETE r "
                    "WITH DISTINCT u MERGE (a:Sign {name:$t}) MERGE (b:Sign {name:$s}) "
                    "MERGE (u)-[:HAS_MOON_SIGN {system:'tropical'}]->(a) "
                    "MERGE (u)-[:HAS_MOON_SIGN {system:'sidereal'}]->(b)", id=user_id, t=trop, s=sid)
            if "language" in fields:
                self._run(
                    "MATCH (u:User {id:$id}) OPTIONAL MATCH (u)-[r:PREFERS]->(:Preference) DELETE r "
                    "WITH DISTINCT u MERGE (p:Preference {name:$l}) MERGE (u)-[:PREFERS]->(p)",
                    id=user_id, l=fields["language"])
        return self.get_profile(user_id)

    # ---- memories --------------------------------------------------------------
    def active_by_key(self, user_id: str, key: str) -> dict | None:
        rows = self._run(
            "MATCH (:User {id:$u})-[:HAS_MEMORY]->(m:Memory {key:$k, status:'active'}) RETURN m LIMIT 1",
            u=user_id, k=key)
        return dict(rows[0]["m"]) if rows else None

    def _supersede(self, user_id: str, old_id: str, new_id: str | None):
        self._run(
            "MATCH (m:Memory {id:$old}) SET m.status='superseded', m.updated_at=$now "
            "WITH m OPTIONAL MATCH (m)-[:DESCRIBES]->(g:Goal) SET g.status='superseded' "
            "WITH m, g OPTIONAL MATCH (:User {id:$u})-[r:HAS_GOAL]->(g) DELETE r "
            "WITH DISTINCT m OPTIONAL MATCH (n:Memory {id:$new}) "
            "FOREACH (_ IN CASE WHEN n IS NULL THEN [] ELSE [1] END | MERGE (n)-[:SUPERSEDES]->(m))",
            old=old_id, new=new_id or "", u=user_id, now=_now().isoformat())

    def apply_item(self, user_id: str, item: ExtractedItem) -> str:
        """Write one extracted item. Returns 'created' | 'updated' | 'superseded' | 'retracted' | 'noop'."""
        self.ensure_user(user_id)
        now = _now()
        existing = self.active_by_key(user_id, item.key)

        # explicit cross-key contradiction ("instead of X")
        if item.replaces_key and item.replaces_key != item.key:
            old = self.active_by_key(user_id, item.replaces_key)
            if old:
                self._supersede(user_id, old["id"], None)

        if item.action == "retract":
            if existing:
                self._supersede(user_id, existing["id"], None)
                return "retracted"
            return "noop"

        if existing and existing.get("value") == item.value:
            self._run(
                "MATCH (m:Memory {id:$id}) SET m.confidence=$c, m.importance=$i, m.updated_at=$now",
                id=existing["id"], c=max(existing.get("confidence", 0), item.confidence),
                i=max(existing.get("importance", 0), item.importance), now=now.isoformat())
            return "updated"

        mid = f"m-{uuid.uuid4().hex[:12]}"
        expires = (now + timedelta(days=item.expires_in_days)).isoformat() if item.expires_in_days else None
        area = item.life_area if item.life_area in LIFE_AREAS else "general"
        self._run(
            "MATCH (u:User {id:$u}) MERGE (a:LifeArea {name:$area}) "
            "CREATE (m:Memory {id:$mid, kind:$kind, key:$key, value:$value, text:$text, confidence:$conf, "
            "importance:$imp, status:'active', created_at:$now, updated_at:$now, expires_at:$exp, "
            "target_year:$ty, raw_phrase:$raw}) "
            "CREATE (u)-[:HAS_MEMORY]->(m) CREATE (m)-[:ABOUT]->(a)",
            u=user_id, area=area, mid=mid, kind=item.kind, key=item.key, value=item.value, text=item.text,
            conf=item.confidence, imp=item.importance, now=now.isoformat(), exp=expires,
            ty=item.target_year, raw=item.raw_phrase)
        self._typed_projection(user_id, mid, item)
        if existing:
            self._supersede(user_id, existing["id"], mid)
            return "superseded"
        return "created"

    def _typed_projection(self, user_id: str, mid: str, item: ExtractedItem):
        if item.kind == "goal":
            self._run(
                "MATCH (u:User {id:$u}), (m:Memory {id:$mid}) "
                "CREATE (g:Goal {id:$gid, title:$title, target_year:$ty, status:'active'}) "
                "CREATE (u)-[:HAS_GOAL]->(g) CREATE (m)-[:DESCRIBES]->(g)",
                u=user_id, mid=mid, gid=f"g-{uuid.uuid4().hex[:10]}", title=item.value, ty=item.target_year)
        elif item.kind == "interest":
            self._run(
                "MATCH (u:User {id:$u}), (m:Memory {id:$mid}) MERGE (t:Topic {name:$n}) "
                "MERGE (u)-[:INTERESTED_IN]->(t) MERGE (m)-[:DESCRIBES]->(t)",
                u=user_id, mid=mid, n=item.value)
        elif item.kind == "preference":
            self._run(
                "MATCH (u:User {id:$u}), (m:Memory {id:$mid}) MERGE (p:Preference {name:$n}) "
                "MERGE (u)-[:PREFERS]->(p) MERGE (m)-[:DESCRIBES]->(p)",
                u=user_id, mid=mid, n=item.value)

    def candidate_memories(self, user_id: str, areas: list[str] | None, fetch: int = 50) -> list[dict]:
        """Active, unexpired memories reachable via the given life areas (or all if areas is None)."""
        rows = self._run(
            "MATCH (:User {id:$u})-[:HAS_MEMORY]->(m:Memory {status:'active'})-[:ABOUT]->(a:LifeArea) "
            "WHERE (m.expires_at IS NULL OR m.expires_at > $now) AND ($areas IS NULL OR a.name IN $areas) "
            "RETURN m, a.name AS area ORDER BY m.updated_at DESC LIMIT $fetch",
            u=user_id, areas=areas, now=_now().isoformat(), fetch=fetch)
        return [{**dict(r["m"]), "life_area": r["area"]} for r in rows]

    def list_memories(self, user_id: str, include_inactive: bool = False) -> list[dict]:
        rows = self._run(
            "MATCH (:User {id:$u})-[:HAS_MEMORY]->(m:Memory)-[:ABOUT]->(a:LifeArea) "
            "WHERE $all OR m.status='active' RETURN m, a.name AS area ORDER BY m.created_at",
            u=user_id, all=include_inactive)
        return [{**dict(r["m"]), "life_area": r["area"]} for r in rows]

    def delete_users_with_prefix(self, prefix: str):
        """Test helper: remove only test users and their memory nodes."""
        self._run(
            "MATCH (u:User) WHERE u.id STARTS WITH $p "
            "OPTIONAL MATCH (u)-[:HAS_MEMORY]->(m:Memory) OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:Goal) "
            "OPTIONAL MATCH (m)-[:DESCRIBES]->(g2:Goal) DETACH DELETE m, g, g2, u", p=prefix)
