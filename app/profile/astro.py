"""Stubbed astrology: tropical sun sign from date of birth (no ephemeris)."""
from datetime import date

# (month, day) where the sign *starts*
_SIGNS = [
    ((1, 20), "Aquarius"), ((2, 19), "Pisces"), ((3, 21), "Aries"),
    ((4, 20), "Taurus"), ((5, 21), "Gemini"), ((6, 21), "Cancer"),
    ((7, 23), "Leo"), ((8, 23), "Virgo"), ((9, 23), "Libra"),
    ((10, 23), "Scorpio"), ((11, 22), "Sagittarius"), ((12, 22), "Capricorn"),
]


def sun_sign(dob: date) -> str:
    md = (dob.month, dob.day)
    sign = "Capricorn"  # Jan 1-19 falls before Aquarius start
    for start, name in _SIGNS:
        if md >= start:
            sign = name
    return sign


# ---- moon sign -------------------------------------------------------------------------------
# Low-precision lunar longitude (truncated Meeus series, ~0.2-0.3 deg). Tropical zodiac, like the sun sign.
import math
from datetime import datetime, time, timedelta, timezone

SIGN_NAMES = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
              "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
DEFAULT_UTC_OFFSET = 5.5  # IST: most users; override per user with utc_offset


def _julian_day(dt_utc: datetime) -> float:
    return 2440587.5 + dt_utc.timestamp() / 86400.0


def moon_longitude(dt_utc: datetime) -> float:
    """Ecliptic longitude of the Moon in degrees [0, 360) (tropical)."""
    T = (_julian_day(dt_utc) - 2451545.0) / 36525.0
    Lp = 218.3164477 + 481267.88123421 * T
    D = math.radians(297.8501921 + 445267.1114034 * T)
    M = math.radians(357.5291092 + 35999.0502909 * T)
    Mp = math.radians(134.9633964 + 477198.8675055 * T)
    F = math.radians(93.2720950 + 483202.0175233 * T)
    lon = (Lp
           + 6.288774 * math.sin(Mp) + 1.274027 * math.sin(2 * D - Mp) + 0.658314 * math.sin(2 * D)
           + 0.213618 * math.sin(2 * Mp) - 0.185116 * math.sin(M) - 0.114332 * math.sin(2 * F)
           + 0.058793 * math.sin(2 * D - 2 * Mp) + 0.057066 * math.sin(2 * D - M - Mp)
           + 0.053322 * math.sin(2 * D + Mp) + 0.045758 * math.sin(2 * D - M) - 0.040923 * math.sin(M - Mp)
           - 0.034720 * math.sin(D) - 0.030383 * math.sin(M + Mp) + 0.015327 * math.sin(2 * D - 2 * F)
           - 0.012528 * math.sin(Mp + 2 * F) + 0.010980 * math.sin(Mp - 2 * F))
    return lon % 360.0


def lahiri_ayanamsa(dt_utc: datetime) -> float:
    """Lahiri (Chitrapaksha) ayanamsa in degrees: 23.853 at J2000, precessing ~50.29"/yr (mean value,
    nutation ignored, ~0.005 deg)."""
    years = (_julian_day(dt_utc) - 2451545.0) / 365.25
    return 23.85306 + 0.013970 * years


def _lon_at(dob: date, local_time: time, utc_offset: float, sidereal: bool) -> float:
    local = datetime.combine(dob, local_time)
    utc = (local - timedelta(hours=utc_offset)).replace(tzinfo=timezone.utc)
    lon = moon_longitude(utc)
    return (lon - lahiri_ayanamsa(utc)) % 360.0 if sidereal else lon


def _sign_at(dob: date, local_time: time, utc_offset: float, sidereal: bool = False) -> str:
    return SIGN_NAMES[int(_lon_at(dob, local_time, utc_offset, sidereal) // 30)]


BOUNDARY_DEG = 0.3  # the Moon moves ~0.55 deg/hour, so 0.3 deg ~ 33 minutes of birth-time error


def moon_sign(dob: date, tob: str | None = None, utc_offset: float | None = None,
              sidereal: bool = False) -> tuple[str, str]:
    """Returns (sign, note). sidereal=True gives the Vedic Rashi (tropical longitude - Lahiri ayanamsa).
    Without a birth time, noon is assumed and the note says whether the Moon changed signs during that day
    (in which case the sign is genuinely uncertain)."""
    off = DEFAULT_UTC_OFFSET if utc_offset is None else utc_offset
    if tob:
        hh, mm = (int(x) for x in tob.split(":"))
        lon = _lon_at(dob, time(hh, mm), off, sidereal)
        note = "from birth time"
        if min(lon % 30, 30 - lon % 30) < BOUNDARY_DEG:
            note += "; very close to a sign boundary, so a small birth-time error could change it (uncertain)"
        return SIGN_NAMES[int(lon // 30)], note
    start, end = _sign_at(dob, time(0, 0), off, sidereal), _sign_at(dob, time(23, 59), off, sidereal)
    noon = _sign_at(dob, time(12, 0), off, sidereal)
    if start != end:
        return noon, f"birth time unknown; the Moon moved from {start} to {end} that day, so this is uncertain"
    return noon, "birth time unknown, but the Moon stayed in this sign all day"
