from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"

LIFE_AREAS = [
    "career", "relationships", "health", "finance",
    "education", "family", "spirituality", "personal_growth", "general",
]


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=ID_PATTERN)
    session_id: str = Field(pattern=ID_PATTERN)
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    context_used: list[str] = []
    degraded: bool = False
    warnings: list[str] = []


class ProfileIn(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    dob: Optional[date] = None
    tob: Optional[str] = Field(default=None, pattern=r"^([01]?\d|2[0-3]):[0-5]\d$")
    birth_place: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None, max_length=30)
    utc_offset: Optional[float] = Field(default=None, ge=-12, le=14, description="birth place UTC offset in hours; default IST +5.5")


class Profile(BaseModel):
    user_id: str
    name: Optional[str] = None
    dob: Optional[str] = None
    tob: Optional[str] = None
    birth_place: Optional[str] = None
    language: Optional[str] = None
    sun_sign: Optional[str] = None
    utc_offset: Optional[float] = None
    moon_sign: Optional[str] = None
    moon_note: Optional[str] = None
    moon_sign_sidereal: Optional[str] = None  # Vedic Rashi (Lahiri)
    moon_note_sidereal: Optional[str] = None


MemoryKind = Literal["profile", "goal", "preference", "interest", "life_event", "fact"]


class ExtractedItem(BaseModel):
    """One candidate long-term memory produced by the extractor (LLM or rules)."""

    kind: MemoryKind
    key: str = Field(min_length=3, max_length=80)  # stable identity, e.g. goal:career_change
    value: str = Field(min_length=1, max_length=200)
    text: str = Field(default="", max_length=300)  # natural-language memory (defaults to value)
    life_area: str = "general"
    confidence: float = Field(default=0.8, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    action: Literal["upsert", "retract"] = "upsert"
    replaces_key: Optional[str] = None  # explicit contradiction of another key
    target_year: Optional[int] = None
    raw_phrase: Optional[str] = None  # original relative-time phrase ("next year")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data):
        """Small models omit/overlong fields; repair instead of dropping a good memory."""
        if isinstance(data, dict):
            data = dict(data)
            if data.get("value") is not None:
                data["value"] = str(data["value"])[:200]
            data["text"] = str(data.get("text") or data.get("value") or "")[:300]
            for f in ("confidence", "importance"):
                if isinstance(data.get(f), (int, float)):
                    data[f] = min(max(float(data[f]), 0.0), 1.0)
        return data
