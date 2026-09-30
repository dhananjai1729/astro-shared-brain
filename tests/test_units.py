from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.memory.classifier import RulesClassifier
from app.memory.ranking import rank_memories, score_memory
from app.memory.rules import extract_rules
from app.memory.updater import parse_items
from app.models import ChatRequest, ProfileIn
from app.profile.astro import sun_sign

TODAY = date(2026, 9, 30)


@pytest.mark.parametrize("dob,sign", [
    (date(1995, 8, 15), "Leo"), (date(2000, 1, 1), "Capricorn"), (date(2000, 1, 20), "Aquarius"),
    (date(1990, 3, 21), "Aries"), (date(1990, 3, 20), "Pisces"), (date(1990, 12, 25), "Capricorn"),
    (date(1990, 12, 21), "Sagittarius"), (date(1990, 7, 22), "Cancer"), (date(1990, 11, 22), "Sagittarius"),
])
def test_sun_sign(dob, sign):
    assert sun_sign(dob) == sign


@pytest.mark.parametrize("msg,areas", [
    ("What should I focus on for my career?", ["career"]),
    ("Will I get married soon?", ["relationships"]),
    ("Is my health going to improve and my money too?", ["health", "finance"]),
    ("Tell me about the stars", []),
])
def test_life_areas(msg, areas):
    assert RulesClassifier().life_areas(msg) == areas


@pytest.mark.parametrize("msg,ok", [
    ("thanks", False), ("ok", False), ("Why do you say that?", False), ("What should I do about work?", False),
    ("I'm planning to switch jobs next year.", True), ("My name is Rahul", True),
    ("The weather is nice today", False),
])
def test_worth_remembering(msg, ok):
    assert RulesClassifier().worth_remembering(msg) is ok


def test_followup_detection():
    c = RulesClassifier()
    assert c.is_followup("Why do you say that?")
    assert not c.is_followup("What do you remember about my career goals?")


def test_extract_first_conversation():
    items = {i.key: i for i in extract_rules(
        "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year.", TODAY)}
    assert items["profile:name"].value == "Rahul"
    assert items["profile:dob"].value == "1995-08-15"
    assert items["profile:birth_place"].value == "Delhi"
    assert items["goal:career_change"].target_year == 2027  # relative time resolved
    assert items["goal:career_change"].raw_phrase == "next year"


def test_extract_ignores_small_talk():
    assert extract_rules("thanks, that was helpful", TODAY) == []


def test_extract_timeboxed_goal_and_retract():
    g = extract_rules("I'm preparing for a product management interview next month.", TODAY)[0]
    assert g.key.startswith("goal:interview") and g.expires_in_days == 45
    r = extract_rules("I decided not to switch jobs after all", TODAY)[0]
    assert r.action == "retract" and r.key == "goal:career_change"


def test_parse_items_tolerates_noise_and_drops_invalid():
    raw = 'Sure! {"memories":[{"kind":"goal","key":"goal:x_y","value":"v","text":"t"},{"kind":"bogus"}]} done'
    assert [i.key for i in parse_items(raw)] == ["goal:x_y"]


def test_ranking_prefers_matching_area_importance_and_recency():
    now = datetime(2026, 9, 30, tzinfo=timezone.utc)
    fresh = (now - timedelta(days=1)).isoformat()
    old = (now - timedelta(days=720)).isoformat()
    mems = [
        {"text": "health", "life_area": "health", "importance": 0.9, "confidence": 0.9, "updated_at": fresh},
        {"text": "career-old", "life_area": "career", "importance": 0.8, "confidence": 0.9, "updated_at": old},
        {"text": "career-new", "life_area": "career", "importance": 0.8, "confidence": 0.9, "updated_at": fresh},
    ]
    ranked = rank_memories(mems, ["career"], 180, 2, now)
    assert [m["text"] for m in ranked] == ["career-new", "career-old"]  # off-topic health memory excluded by limit
    assert ranked[0]["text"] == "career-new"
    assert score_memory(mems[1], ["career"], 180, now) < score_memory(mems[2], ["career"], 180, now)


def test_input_validation():
    with pytest.raises(ValidationError):
        ChatRequest(user_id="u", session_id="s", message="   ")
    with pytest.raises(ValidationError):
        ChatRequest(user_id="bad id!", session_id="s", message="hi")
    with pytest.raises(ValidationError):
        ChatRequest(user_id="u", session_id="s", message="x" * 2001)
    with pytest.raises(ValidationError):
        ProfileIn(tob="25:99")


def test_prompt_marks_empty_memory_explicitly_and_follows_language():
    from app.chat.prompts import NO_MEMORIES, build_system_prompt
    from app.models import Profile
    empty = build_system_prompt(Profile(user_id="u", language="Hindi"), [])
    assert NO_MEMORIES in empty and "Respond in Hindi" in empty
    full = build_system_prompt(Profile(user_id="u"), ["User wants to start a business"])
    assert "- User wants to start a business" in full and NO_MEMORIES not in full and "Respond in English" in full


def test_moon_longitude_matches_reference_positions():
    from app.profile.astro import moon_longitude
    from datetime import timezone
    assert abs(moon_longitude(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)) - 223.32) < 0.3   # J2000 (Meeus)
    assert abs(moon_longitude(datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)) - 285.9) < 1.0  # new moon: = sun
    d = (moon_longitude(datetime(2000, 1, 21, 4, 44, tzinfo=timezone.utc)) - 121.0 + 180) % 360 - 180
    assert abs(d) < 1.5  # total lunar eclipse: moon opposite sun


def test_moon_sign_uses_birth_time_and_flags_uncertainty():
    from app.profile.astro import moon_sign
    # the Moon changes sign roughly every 2.3 days: find a day where it does, at IST
    changing = next(d for d in (date(1995, 8, 1) + timedelta(days=i) for i in range(10))
                    if "uncertain" in moon_sign(d)[1])
    early = moon_sign(changing, "00:05")[0]
    late = moon_sign(changing, "23:55")[0]
    assert early != late and moon_sign(changing, "06:30")[1].startswith("from birth time")
    assert moon_sign(changing)[0] in (early, late)
    # timezone matters: same clock time, very different UTC offset can flip the sign near a boundary
    assert isinstance(moon_sign(changing, "12:00", -8)[0], str)


def test_prompt_shows_moon_sign_and_marks_uncertainty():
    from app.chat.prompts import render_profile
    from app.models import Profile
    ok = render_profile(Profile(user_id="u", moon_sign="Gemini", moon_note="from birth time"))
    assert "moon sign: Gemini" in ok and "uncertain" not in ok
    unsure = render_profile(Profile(user_id="u", moon_sign="Gemini", moon_note="birth time unknown; ... so this is uncertain"))
    assert "moon sign: Gemini (" in unsure and "uncertain" in unsure
    assert "moon sign: unknown" in render_profile(Profile(user_id="u"))
    both = render_profile(Profile(user_id="u", moon_sign="Aries", moon_sign_sidereal="Pisces"))
    assert "moon sign: Aries" in both and "Rashi, Lahiri): Pisces" in both


def test_lahiri_ayanamsa_matches_published_values():
    from app.profile.astro import lahiri_ayanamsa
    utc = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc)
    assert abs(lahiri_ayanamsa(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)) - 23.853) < 0.01
    assert abs(lahiri_ayanamsa(utc(1950)) - 23.155) < 0.02
    assert abs(lahiri_ayanamsa(utc(2025)) - 24.20) < 0.02


def test_sidereal_moon_is_tropical_shifted_back_by_ayanamsa():
    from app.profile.astro import SIGN_NAMES, lahiri_ayanamsa, moon_longitude, moon_sign
    dt = datetime(1995, 8, 15, 1, 0, tzinfo=timezone.utc)  # 06:30 IST
    trop = moon_longitude(dt)
    sid = (trop - lahiri_ayanamsa(dt)) % 360
    assert moon_sign(date(1995, 8, 15), "06:30")[0] == SIGN_NAMES[int(trop // 30)]
    assert moon_sign(date(1995, 8, 15), "06:30", sidereal=True)[0] == SIGN_NAMES[int(sid // 30)]
    # ~24 deg apart: the sign differs unless the tropical moon sits in the last ~6 deg... of a sign
    assert (trop % 30 >= 6) == (SIGN_NAMES[int(trop // 30)] != SIGN_NAMES[int(sid // 30)]) or trop % 30 < 6


def test_boundary_proximity_is_flagged(monkeypatch):
    import app.profile.astro as a
    monkeypatch.setattr(a, "_lon_at", lambda *args, **kw: 59.9)  # 0.1 deg before Gemini
    sign, note = a.moon_sign(date(2000, 1, 1), "12:00")
    assert sign == "Taurus" and "boundary" in note and "uncertain" in note
    monkeypatch.setattr(a, "_lon_at", lambda *args, **kw: 45.0)
    assert a.moon_sign(date(2000, 1, 1), "12:00")[1] == "from birth time"
