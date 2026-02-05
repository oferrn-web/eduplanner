# app.py
# Streamlit wizard-based monthly planner + deterministic scheduler + ICS export + optional Gemini policy extraction
# Model: "gemini-3-pro-preview" via st.secrets["GEMINI_API_KEY"]

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo


# =========================
# Constants
# =========================
DEFAULT_TZ = "Asia/Jerusalem"
DEFAULT_WORKDAY_START = "08:00"
DEFAULT_WORKDAY_END = "20:00"
DEFAULT_DAILY_MAX_HOURS = 4.0
DEFAULT_MAX_TASK_HOURS_PER_DAY = 3.0

# UI defaults
DEFAULT_MAX_CONTINUOUS_MINUTES = 120
DEFAULT_BREAK_MINUTES = 15
DEFAULT_BUFFER_HOURS = 48

# Internal stepping for generating candidate start times (NOT user-facing)
START_STEP_MINUTES = 15

WEEKDAYS_HE = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
# Python weekday(): Monday=0 ... Sunday=6
WEEKDAY_NAME_TO_INT = {
    "שני": 0,
    "שלישי": 1,
    "רביעי": 2,
    "חמישי": 3,
    "שישי": 4,
    "שבת": 5,
    "ראשון": 6,
}


# =========================
# Data Models
# =========================
@dataclass
class Task:
    task_id: str
    course: str
    title: str
    deadline: date
    estimated_hours: float
    priority: int
    notes: str = ""


@dataclass
class Event:
    title: str
    start_dt: datetime
    end_dt: datetime
    description: str = ""
    kind: str = "task"  # "task" | "constraint"


# =========================
# Helpers: formatting, parsing
# =========================
def clamp_int(x: int, lo: int, hi: int) -> int:
    try:
        x = int(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return default


def parse_hhmm(s: str) -> time:
    s = str(s).strip()
    if not s:
        raise ValueError("empty time")
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        raise ValueError(f"bad time: {s}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"bad time: {s}")
    return time(hh, mm)


def format_date_ddmmyyyy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def parse_date_any(s: str) -> date:
    """
    Accept:
    - DD/MM/YYYY
    - YYYY-MM-DD
    """
    s = str(s).strip()
    if not s:
        raise ValueError("empty date")
    m1 = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", s)
    if m1:
        dd, mm, yy = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        return date(yy, mm, dd)

    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m2:
        yy, mm, dd = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return date(yy, mm, dd)

    raise ValueError(f"unsupported date format: {s}")


def coerce_date_value_to_date(val) -> date:
    """
    Accepts: date, datetime, pd.Timestamp, or string (DD/MM/YYYY or YYYY-MM-DD).
    Returns: date
    """
    if val is None:
        raise ValueError("empty date")
    try:
        if pd.isna(val):
            raise ValueError("missing date")
    except Exception:
        pass

    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()

    s = str(val).strip()
    if not s:
        raise ValueError("empty date")
    return parse_date_any(s)


def month_date_range(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start, end


def daterange(d1: date, d2_excl: date):
    cur = d1
    while cur < d2_excl:
        yield cur
        cur = cur + timedelta(days=1)


# =========================
# Window arithmetic for availability
# =========================
def subtract_interval(
    windows: List[Tuple[datetime, datetime]],
    block: Tuple[datetime, datetime],
) -> List[Tuple[datetime, datetime]]:
    bs, be = block
    if be <= bs:
        return windows
    out: List[Tuple[datetime, datetime]] = []
    for ws, we in windows:
        if be <= ws or bs >= we:
            out.append((ws, we))
            continue
        if bs > ws:
            out.append((ws, min(bs, we)))
        if be < we:
            out.append((max(be, ws), we))
    out = [(a, b) for a, b in out if b > a]
    return out


def merge_time_ranges(ranges: List[Tuple[time, time]]) -> List[Tuple[time, time]]:
    good = [(s, e) for s, e in ranges if e > s]
    good.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[time, time]] = []
    for s, e in good:
        if not out:
            out.append((s, e))
            continue
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def build_daily_available_windows(
    d: date,
    tz: ZoneInfo,
    work_start: time,
    work_end: time,
    weekday_blocks: Dict[int, List[Tuple[time, time]]],
    date_blocks: Dict[date, List[Tuple[time, time]]],
    fixed_daily_blocks: List[Tuple[time, time]],
) -> List[Tuple[datetime, datetime]]:
    if work_end <= work_start:
        return []

    ws = datetime.combine(d, work_start, tzinfo=tz)
    we = datetime.combine(d, work_end, tzinfo=tz)
    windows: List[Tuple[datetime, datetime]] = [(ws, we)]

    wd = d.weekday()

    # apply fixed daily blocks (all days)
    for bs, be in fixed_daily_blocks:
        bsd = datetime.combine(d, bs, tzinfo=tz)
        bed = datetime.combine(d, be, tzinfo=tz)
        windows = subtract_interval(windows, (bsd, bed))

    # weekday blocks
    for bs, be in weekday_blocks.get(wd, []):
        bsd = datetime.combine(d, bs, tzinfo=tz)
        bed = datetime.combine(d, be, tzinfo=tz)
        windows = subtract_interval(windows, (bsd, bed))

    # date-specific blocks
    for bs, be in date_blocks.get(d, []):
        bsd = datetime.combine(d, bs, tzinfo=tz)
        bed = datetime.combine(d, be, tzinfo=tz)
        windows = subtract_interval(windows, (bsd, bed))

    return windows


def generate_start_times(windows: List[Tuple[datetime, datetime]], step_minutes: int) -> List[datetime]:
    step_minutes = int(step_minutes) if step_minutes else 15
    step_minutes = max(5, min(60, step_minutes))

    out: List[datetime] = []
    step = timedelta(minutes=step_minutes)
    for ws, we in windows:
        cur = ws
        while cur < we:
            out.append(cur)
            cur = cur + step
    return out


def reorder_slots_spread(starts: List[datetime], tz: ZoneInfo) -> List[datetime]:
    """
    Reorder start-times to encourage within-day spreading:
    alternate afternoon and morning starts, so afternoons get used.
    """
    if not starts:
        return starts

    starts_sorted = sorted(starts)
    day = starts_sorted[0].date()
    midday = datetime.combine(day, time(13, 0), tzinfo=tz)

    morning = [s for s in starts_sorted if s < midday]
    afternoon = [s for s in starts_sorted if s >= midday]

    out: List[datetime] = []
    i = j = 0
    take_afternoon = (day.day % 2 == 1)

    while i < len(morning) or j < len(afternoon):
        if take_afternoon and j < len(afternoon):
            out.append(afternoon[j])
            j += 1
        elif (not take_afternoon) and i < len(morning):
            out.append(morning[i])
            i += 1
        else:
            if i < len(morning):
                out.append(morning[i])
                i += 1
            elif j < len(afternoon):
                out.append(afternoon[j])
                j += 1
        take_afternoon = not take_afternoon

    return out


# =========================
# ICS Export (no external deps)
# =========================
def ics_escape(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace(";", r"\;").replace(",", r"\,")
    s = s.replace("\n", r"\n")
    return s


def dt_to_ics(dt: datetime) -> str:
    # For Google Calendar, floating local time is okay, but we include TZID on DTSTART/DTEND.
    return dt.strftime("%Y%m%dT%H%M%S")


def build_ics(events: List[Event], tz_name: str) -> str:
    tzid = tz_name or DEFAULT_TZ
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//EduPlanner//Wizard//HE",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    now_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for ev in sorted(events, key=lambda e: e.start_dt):
        uid = f"{uuid.uuid4()}@eduplanner"
        summary = ics_escape(ev.title)
        desc = ics_escape(ev.description or "")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{now_utc}",
                f"DTSTART;TZID={tzid}:{dt_to_ics(ev.start_dt)}",
                f"DTEND;TZID={tzid}:{dt_to_ics(ev.end_dt)}",
                f"SUMMARY:{summary}",
                f"DESCRIPTION:{desc}",
                "END:VEVENT",
            ]
        )

    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# =========================
# AI: Gemini policy extraction (optional)
# =========================
def gemini_available() -> bool:
    try:
        import google.generativeai as genai  # noqa: F401
        return True
    except Exception:
        return False


def ai_extract_policy(free_text: str, tz_name: str) -> Dict:
    """
    Calls Gemini to parse a free-text 'policy' into JSON. If fails, returns defaults.
    Expected schema:
      {
        "hard": {
          "max_continuous_minutes": 120,
          "break_minutes": 15
        },
        "preferences": {
          "dayparts": ["morning","afternoon","evening"],
          "focus_weekdays": ["שני","שלישי"],
          "avoid_weekdays": [],
          "course_bias": {"קורס X": 1.2}
        },
        "notes": "..."
      }
    """
    defaults = {
        "hard": {
            "max_continuous_minutes": DEFAULT_MAX_CONTINUOUS_MINUTES,
            "break_minutes": DEFAULT_BREAK_MINUTES,
        },
        "preferences": {
            "dayparts": ["morning", "afternoon"],
            "focus_weekdays": [],
            "avoid_weekdays": [],
            "course_bias": {},
        },
        "notes": "ברירת מחדל.",
    }

    free_text = (free_text or "").strip()
    if not free_text:
        return defaults

    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    if not api_key or not gemini_available():
        return defaults | {"notes": "לא בוצעה קריאת AI (מפתח חסר או ספרייה לא זמינה)."}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-3-pro-preview")

        prompt = f"""
אתה מנוע מדיניות לשיבוץ לו״ז לימודי חודשי.
הפלט חייב להיות JSON תקין בלבד, ללא טקסט נוסף.

אזור זמן: {tz_name}

קלט מהמשתמש (policy חופשי בעברית):
\"\"\"{free_text}\"\"\"

הפק JSON במבנה הבא בדיוק:
{{
  "hard": {{
    "max_continuous_minutes": 120,
    "break_minutes": 15
  }},
  "preferences": {{
    "dayparts": ["morning","afternoon","evening"],
    "focus_weekdays": ["שני","שלישי"],
    "avoid_weekdays": ["שישי"],
    "course_bias": {{"שם קורס": 1.2}}
  }},
  "notes": "מחרוזת קצרה"
}}

כללים:
- אם מידע חסר, השתמש בברירות מחדל סבירות.
- dayparts מותרים: morning, afternoon, evening.
- max_continuous_minutes בין 30 ל-240.
- break_minutes בין 0 ל-60.
- course_bias: מספרים בין 0.5 ל-2.0.
"""

        res = model.generate_content(prompt).text.strip()

        # Try to locate JSON object
        m = re.search(r"\{.*\}", res, flags=re.DOTALL)
        if not m:
            return defaults | {"notes": "AI לא החזיר JSON תקין, הופעלה ברירת מחדל."}
        obj = json.loads(m.group(0))

        hard = obj.get("hard", {}) if isinstance(obj.get("hard", {}), dict) else {}
        pref = obj.get("preferences", {}) if isinstance(obj.get("preferences", {}), dict) else {}

        max_cont = clamp_int(hard.get("max_continuous_minutes", DEFAULT_MAX_CONTINUOUS_MINUTES), 30, 240)
        brk = clamp_int(hard.get("break_minutes", DEFAULT_BREAK_MINUTES), 0, 60)

        dayparts = pref.get("dayparts", ["morning", "afternoon"])
        if not isinstance(dayparts, list):
            dayparts = ["morning", "afternoon"]
        dayparts = [p for p in dayparts if p in ["morning", "afternoon", "evening"]]
        if not dayparts:
            dayparts = ["morning", "afternoon"]

        focus_wd = pref.get("focus_weekdays", [])
        if not isinstance(focus_wd, list):
            focus_wd = []
        focus_wd = [w for w in focus_wd if w in WEEKDAYS_HE]

        avoid_wd = pref.get("avoid_weekdays", [])
        if not isinstance(avoid_wd, list):
            avoid_wd = []
        avoid_wd = [w for w in avoid_wd if w in WEEKDAYS_HE]

        course_bias = pref.get("course_bias", {})
        if not isinstance(course_bias, dict):
            course_bias = {}
        cleaned_bias = {}
        for k, v in course_bias.items():
            try:
                kk = str(k).strip()
                vv = float(v)
                vv = max(0.5, min(2.0, vv))
                if kk:
                    cleaned_bias[kk] = vv
            except Exception:
                continue

        notes = obj.get("notes", "")
        if not isinstance(notes, str):
            notes = ""

        return {
            "hard": {"max_continuous_minutes": max_cont, "break_minutes": brk},
            "preferences": {
                "dayparts": dayparts,
                "focus_weekdays": focus_wd,
                "avoid_weekdays": avoid_wd,
                "course_bias": cleaned_bias,
            },
            "notes": notes.strip() or "נוצר ע״י AI.",
        }

    except Exception:
        return defaults | {"notes": "שגיאה בקריאת AI, הופעלה ברירת מחדל."}


# =========================
# Deterministic Scheduling Engine (Round-Robin, policy-aware)
# =========================
def validate_schedule(
    events: List[Event],
    tz: ZoneInfo,
    daily_max_minutes: int,
    weekday_blocks: Dict[int, List[Tuple[time, time]]],
    date_blocks: Dict[date, List[Tuple[time, time]]],
    fixed_daily_blocks: List[Tuple[time, time]],
) -> Dict:
    overlaps = []
    daily_minutes: Dict[date, int] = {}
    blocked_hits = []

    evs = sorted(events, key=lambda e: (e.start_dt, e.end_dt))
    for i in range(1, len(evs)):
        prev = evs[i - 1]
        cur = evs[i]
        if cur.start_dt < prev.end_dt:
            overlaps.append(
                {
                    "prev": {"title": prev.title, "start": prev.start_dt.isoformat(), "end": prev.end_dt.isoformat()},
                    "cur": {"title": cur.title, "start": cur.start_dt.isoformat(), "end": cur.end_dt.isoformat()},
                }
            )

    for ev in evs:
        d = ev.start_dt.date()
        mins = int((ev.end_dt - ev.start_dt).total_seconds() // 60)
        daily_minutes[d] = daily_minutes.get(d, 0) + max(0, mins)

    daily_exceed = []
    for d, mins in daily_minutes.items():
        if mins > daily_max_minutes:
            daily_exceed.append({"date": d.isoformat(), "minutes": mins, "limit": daily_max_minutes})

    # blocks check
    for ev in evs:
        d = ev.start_dt.date()
        wd = d.weekday()

        # fixed daily
        for bs, be in fixed_daily_blocks:
            bsd = datetime.combine(d, bs, tzinfo=tz)
            bed = datetime.combine(d, be, tzinfo=tz)
            if ev.start_dt < bed and ev.end_dt > bsd:
                blocked_hits.append({"event": ev.title, "block": {"type": "fixed_daily", "start": bs.isoformat(), "end": be.isoformat()}})

        for bs, be in weekday_blocks.get(wd, []):
            bsd = datetime.combine(d, bs, tzinfo=tz)
            bed = datetime.combine(d, be, tzinfo=tz)
            if ev.start_dt < bed and ev.end_dt > bsd:
                blocked_hits.append({"event": ev.title, "block": {"type": "weekday", "weekday": wd, "start": bs.isoformat(), "end": be.isoformat()}})

        for bs, be in date_blocks.get(d, []):
            bsd = datetime.combine(d, bs, tzinfo=tz)
            bed = datetime.combine(d, be, tzinfo=tz)
            if ev.start_dt < bed and ev.end_dt > bsd:
                blocked_hits.append({"event": ev.title, "block": {"type": "date", "date": d.isoformat(), "start": bs.isoformat(), "end": be.isoformat()}})

    return {
        "ok": (len(overlaps) == 0 and len(daily_exceed) == 0 and len(blocked_hits) == 0),
        "overlaps_count": len(overlaps),
        "daily_exceed_count": len(daily_exceed),
        "blocked_hits_count": len(blocked_hits),
        "overlaps": overlaps[:10],
        "daily_exceed": daily_exceed[:10],
        "blocked_hits": blocked_hits[:10],
    }


def schedule_tasks(
    tasks: List[Task],
    tz_name: str,
    year: int,
    month: int,
    work_start_hhmm: str,
    work_end_hhmm: str,
    daily_max_hours: float,
    max_task_hours_per_day: float,
    buffer_hours: int,
    weekday_blocks: Dict[int, List[Tuple[str, str]]],
    date_blocks: Dict[str, List[Tuple[str, str]]],
    fixed_daily_blocks: List[Tuple[str, str]],
    policy: Optional[Dict] = None,
) -> Tuple[List[Event], Dict]:
    tz = ZoneInfo(tz_name or DEFAULT_TZ)
    work_start = parse_hhmm(work_start_hhmm)
    work_end = parse_hhmm(work_end_hhmm)

    # Convert blocks
    weekday_blocks_t: Dict[int, List[Tuple[time, time]]] = {}
    for wd, blocks in weekday_blocks.items():
        out = []
        for s, e in blocks:
            try:
                out.append((parse_hhmm(s), parse_hhmm(e)))
            except Exception:
                continue
        weekday_blocks_t[wd] = merge_time_ranges(out)

    date_blocks_t: Dict[date, List[Tuple[time, time]]] = {}
    for ds, blocks in date_blocks.items():
        try:
            d = parse_date_any(ds)
        except Exception:
            continue
        out = []
        for s, e in blocks:
            try:
                out.append((parse_hhmm(s), parse_hhmm(e)))
            except Exception:
                continue
        date_blocks_t[d] = merge_time_ranges(out)

    fixed_daily_blocks_t: List[Tuple[time, time]] = []
    for s, e in fixed_daily_blocks:
        try:
            fixed_daily_blocks_t.append((parse_hhmm(s), parse_hhmm(e)))
        except Exception:
            continue
    fixed_daily_blocks_t = merge_time_ranges(fixed_daily_blocks_t)

    month_start, month_end_excl = month_date_range(year, month)
    now_dt = datetime.now(tz=tz)

    # Clean tasks and drop invalid deadlines
    cleaned_tasks: List[Task] = []
    dropped_tasks: List[Dict] = []
    for t in tasks:
        dl = getattr(t, "deadline", None)
        if dl is None:
            dropped_tasks.append({"task_id": t.task_id, "course": t.course, "title": t.title, "reason": "דדליין חסר."})
            continue
        try:
            if isinstance(dl, (pd.Timestamp, datetime)) and pd.isna(dl):
                dropped_tasks.append({"task_id": t.task_id, "course": t.course, "title": t.title, "reason": "דדליין חסר (NaT)."})
                continue
        except Exception:
            pass
        if isinstance(dl, (pd.Timestamp, datetime)):
            dl = dl.date()
        if not isinstance(dl, date) or isinstance(dl, datetime):
            dropped_tasks.append({"task_id": t.task_id, "course": t.course, "title": t.title, "reason": f"סוג דדליין לא תקין: {type(getattr(t,'deadline',None))}"})
            continue
        t.deadline = dl
        cleaned_tasks.append(t)

    tasks = cleaned_tasks

    # Sort by (deadline, priority desc, hours desc)
    tasks_sorted = sorted(tasks, key=lambda x: (x.deadline, -clamp_int(x.priority, 1, 5), -float(x.estimated_hours)))

    # Policy
    policy = policy or {}
    pol_hard = policy.get("hard", {}) if isinstance(policy.get("hard", {}), dict) else {}
    pol_pref = policy.get("preferences", {}) if isinstance(policy.get("preferences", {}), dict) else {}

    work_block_minutes = clamp_int(pol_hard.get("max_continuous_minutes", DEFAULT_MAX_CONTINUOUS_MINUTES), 30, 240)
    break_minutes = clamp_int(pol_hard.get("break_minutes", DEFAULT_BREAK_MINUTES), 0, 60)

    dayparts = pol_pref.get("dayparts", ["morning", "afternoon"])
    if not isinstance(dayparts, list) or not dayparts:
        dayparts = ["morning", "afternoon"]

    focus_weekdays = pol_pref.get("focus_weekdays", [])
    avoid_weekdays = pol_pref.get("avoid_weekdays", [])
    if not isinstance(focus_weekdays, list):
        focus_weekdays = []
    if not isinstance(avoid_weekdays, list):
        avoid_weekdays = []

    course_bias = pol_pref.get("course_bias", {})
    if not isinstance(course_bias, dict):
        course_bias = {}

    daily_max_minutes = int(float(daily_max_hours) * 60)
    max_task_minutes_per_day = int(float(max_task_hours_per_day) * 60)
    buffer_delta = timedelta(hours=int(buffer_hours))

    # Build day windows and candidate starts
    day_windows: Dict[date, List[Tuple[datetime, datetime]]] = {}
    day_starts: Dict[date, List[datetime]] = {}

    for d in daterange(month_start, month_end_excl):
        windows = build_daily_available_windows(
            d=d,
            tz=tz,
            work_start=work_start,
            work_end=work_end,
            weekday_blocks=weekday_blocks_t,
            date_blocks=date_blocks_t,
            fixed_daily_blocks=fixed_daily_blocks_t,
        )
        day_windows[d] = windows
        starts = generate_start_times(windows, START_STEP_MINUTES)
        starts = reorder_slots_spread(starts, tz)
        day_starts[d] = starts

    # Helper: is (s,e) within any window
    def within_windows(d: date, s_dt: datetime, e_dt: datetime) -> bool:
        for ws, we in day_windows.get(d, []):
            if s_dt >= ws and e_dt <= we:
                return True
        return False

    # Usage tracking
    used_minutes_by_day: Dict[date, int] = {d: 0 for d in day_starts.keys()}
    used_minutes_by_task_day: Dict[Tuple[str, date], int] = {}

    def day_has_capacity(d: date, add: int) -> bool:
        return used_minutes_by_day.get(d, 0) + add <= daily_max_minutes

    def task_day_has_capacity(tid: str, d: date, add: int) -> bool:
        return used_minutes_by_task_day.get((tid, d), 0) + add <= max_task_minutes_per_day

    # Determine planning horizon per task (deadline - buffer)
    start_day = max(month_start, now_dt.date())

    # Build remaining minutes map
    remaining_by_task: Dict[str, int] = {}
    task_by_id: Dict[str, Task] = {}
    for t in tasks_sorted:
        mins = int(max(0.0, float(t.estimated_hours)) * 60)
        if mins <= 0:
            continue
        remaining_by_task[t.task_id] = mins
        task_by_id[t.task_id] = t

    # Optional: create constraint events for export and visibility
    constraint_events: List[Event] = []
    # Weekday blocks as events for each day of month
    for d in daterange(month_start, month_end_excl):
        wd = d.weekday()
        for bs, be in fixed_daily_blocks_t:
            sdt = datetime.combine(d, bs, tzinfo=tz)
            edt = datetime.combine(d, be, tzinfo=tz)
            constraint_events.append(Event(title="חסם קבוע", start_dt=sdt, end_dt=edt, description="אילוץ קבוע (יומי).", kind="constraint"))
        for bs, be in weekday_blocks_t.get(wd, []):
            sdt = datetime.combine(d, bs, tzinfo=tz)
            edt = datetime.combine(d, be, tzinfo=tz)
            constraint_events.append(Event(title="חסם שבועי", start_dt=sdt, end_dt=edt, description="אילוץ שבועי קבוע.", kind="constraint"))
        for bs, be in date_blocks_t.get(d, []):
            sdt = datetime.combine(d, bs, tzinfo=tz)
            edt = datetime.combine(d, be, tzinfo=tz)
            constraint_events.append(Event(title="חסם בתאריך", start_dt=sdt, end_dt=edt, description="אילוץ בתאריך ספציפי.", kind="constraint"))

    # Scoring: balance load, respect preferences, and slightly favor sooner deadlines
    def daypart_of(dt: datetime) -> str:
        h = dt.hour
        if 5 <= h < 12:
            return "morning"
        if 12 <= h < 17:
            return "afternoon"
        return "evening"

    def score_candidate(t: Task, d: date, s_dt: datetime, e_dt: datetime) -> float:
        # balance: prefer days with lower load
        load_ratio = used_minutes_by_day.get(d, 0) / max(1, daily_max_minutes)
        score = 1.0 - load_ratio

        # deadline urgency: earlier deadlines get more score
        days_to_deadline = (t.deadline - d).days
        score += max(0.0, 3.0 - (days_to_deadline / 7.0))

        # preference: dayparts
        dp = daypart_of(s_dt)
        if dp in dayparts:
            score += 0.5
        else:
            score -= 0.3

        # preference: focus weekdays / avoid weekdays
        wd_name = WEEKDAYS_HE[d.weekday()] if d.weekday() <= 6 else ""
        if wd_name in focus_weekdays:
            score += 0.4
        if wd_name in avoid_weekdays:
            score -= 0.8

        # course bias
        bias = float(course_bias.get(t.course, 1.0)) if t.course else 1.0
        score *= bias

        # small tie-breakers
        score += (0.0001 * (24 - s_dt.hour))
        return score

    # Hard constraints: daily cap, per-task daily cap, no overlap with already scheduled events, break minutes
    events: List[Event] = []
    unscheduled: List[Dict] = []

    # For overlap checks, keep per-day scheduled intervals
    scheduled_intervals: Dict[date, List[Tuple[datetime, datetime]]] = {}

    def overlaps_existing(d: date, s_dt: datetime, e_dt: datetime) -> bool:
        for a, b in scheduled_intervals.get(d, []):
            if s_dt < b and e_dt > a:
                return True
        return False

    def respects_break(d: date, s_dt: datetime, e_dt: datetime) -> bool:
        if break_minutes <= 0:
            return True
        pad = timedelta(minutes=break_minutes)
        for a, b in scheduled_intervals.get(d, []):
            # enforce gap: new [s,e] must not touch too close to [a,b]
            if (s_dt < (b + pad)) and (e_dt > (a - pad)):
                # if it overlaps only because of padding, still disallow
                if not (e_dt <= a or s_dt >= b):
                    return False
                # even non-overlap but too close
                if abs((s_dt - b).total_seconds()) < pad.total_seconds():
                    return False
                if abs((a - e_dt).total_seconds()) < pad.total_seconds():
                    return False
        return True

    # Round-robin allocate blocks
    tids = [t.task_id for t in tasks_sorted if t.task_id in remaining_by_task]
    idx = 0
    safety_counter = 0
    max_iterations = 200000  # hard safety

    while any(remaining_by_task.get(tid, 0) > 0 for tid in tids) and safety_counter < max_iterations:
        safety_counter += 1
        tid = tids[idx % max(1, len(tids))]
        idx += 1

        rem = remaining_by_task.get(tid, 0)
        if rem <= 0:
            continue

        task = task_by_id[tid]
        deadline_dt = datetime.combine(task.deadline, time(23, 59), tzinfo=tz)
        latest_allowed = deadline_dt - buffer_delta
        end_day = min(month_end_excl - timedelta(days=1), latest_allowed.date())

        if end_day < start_day:
            unscheduled.append(
                {
                    "task_id": task.task_id,
                    "course": task.course,
                    "title": task.title,
                    "reason": "חלון הזמן האפשרי לפני הדדליין (כולל buffer) אינו נמצא בחודש הנבחר.",
                    "remaining_hours": round(rem / 60.0, 2),
                }
            )
            remaining_by_task[tid] = 0
            continue

        best = None  # (score, d, s_dt, e_dt, alloc)

        # Search days, prefer earlier
        for d in daterange(start_day, end_day + timedelta(days=1)):
            if remaining_by_task[tid] <= 0:
                break

            if used_minutes_by_day.get(d, 0) >= daily_max_minutes:
                continue

            starts = day_starts.get(d, [])
            if not starts:
                continue

            # Try candidate starts for this day
            for s_dt in starts:
                needed = remaining_by_task[tid]
                if needed <= 0:
                    break

                # allocate a real work block
                alloc = min(work_block_minutes, needed)

                # If task still has plenty remaining, avoid tiny fragments
                if alloc < 45 and needed >= 60:
                    continue

                if not day_has_capacity(d, alloc):
                    continue
                if not task_day_has_capacity(tid, d, alloc):
                    continue

                e_dt = s_dt + timedelta(minutes=alloc)

                if not within_windows(d, s_dt, e_dt):
                    continue
                if overlaps_existing(d, s_dt, e_dt):
                    continue
                if not respects_break(d, s_dt, e_dt):
                    continue

                sc = score_candidate(task, d, s_dt, e_dt)
                if best is None or sc > best[0]:
                    best = (sc, d, s_dt, e_dt, alloc)

        if best is None:
            # cannot place any more work for this task within constraints
            unscheduled.append(
                {
                    "task_id": task.task_id,
                    "course": task.course,
                    "title": task.title,
                    "reason": "לא נמצאה משבצת חוקית (חלונות, עומסים, הפסקות, מגבלות) לפני הדדליין-Buffer.",
                    "remaining_hours": round(remaining_by_task[tid] / 60.0, 2),
                }
            )
            remaining_by_task[tid] = 0
            continue

        _, d, s_dt, e_dt, alloc = best

        title = f"{task.course} | {task.title}".strip(" |")
        desc = (
            f"עבודה מתוכננת.\n"
            f"דדליין: {format_date_ddmmyyyy(task.deadline)}\n"
            f"עדיפות: {clamp_int(task.priority, 1, 5)}\n"
            f"זמן בלוק: {alloc} דקות\n"
            f"{task.notes}".strip()
        )
        events.append(Event(title=title, start_dt=s_dt, end_dt=e_dt, description=desc, kind="task"))

        used_minutes_by_day[d] = used_minutes_by_day.get(d, 0) + alloc
        used_minutes_by_task_day[(tid, d)] = used_minutes_by_task_day.get((tid, d), 0) + alloc
        remaining_by_task[tid] = remaining_by_task[tid] - alloc

        scheduled_intervals.setdefault(d, []).append((s_dt, e_dt))
        scheduled_intervals[d].sort(key=lambda x: x[0])

    # Final event list includes constraints as optional export
    all_events = sorted(events + constraint_events, key=lambda e: e.start_dt)

    validation = validate_schedule(all_events, tz, daily_max_minutes, weekday_blocks_t, date_blocks_t, fixed_daily_blocks_t)
    report = {
        "tz": tz_name,
        "month": f"{year:04d}-{month:02d}",
        "daily_max_hours": float(daily_max_hours),
        "work_start": work_start_hhmm,
        "work_end": work_end_hhmm,
        "work_block_minutes": int(work_block_minutes),
        "break_minutes": int(break_minutes),
        "max_task_hours_per_day": float(max_task_hours_per_day),
        "buffer_hours": int(buffer_hours),
        "dropped_tasks": dropped_tasks,
        "unscheduled": unscheduled,
        "validation": validation,
        "events_count": len(all_events),
    }

    return all_events, report


# =========================
# Data conversion (UI tables -> model)
# =========================
TASK_COLS = ["delete", "task_id", "course", "title", "deadline", "estimated_hours", "priority", "notes"]


def ensure_tasks_df(year: int, month: int):
    if "tasks_df" not in st.session_state:
        st.session_state["tasks_df"] = pd.DataFrame(
            [
                {
                    "delete": False,
                    "task_id": "T1",
                    "course": "קורס לדוגמה",
                    "title": "עבודה מסכמת",
                    "deadline": f"20/{month:02d}/{year:04d}",
                    "estimated_hours": 6.0,
                    "priority": 4,
                    "notes": "",
                },
                {
                    "delete": False,
                    "task_id": "T2",
                    "course": "קורס לדוגמה",
                    "title": "קריאת מאמר",
                    "deadline": f"12/{month:02d}/{year:04d}",
                    "estimated_hours": 3.0,
                    "priority": 3,
                    "notes": "",
                },
            ],
            columns=TASK_COLS,
        )
    else:
        df = st.session_state["tasks_df"].copy()
        for c in TASK_COLS:
            if c not in df.columns:
                df[c] = False if c == "delete" else ""
        st.session_state["tasks_df"] = df[TASK_COLS]
    
    if "delete" not in st.session_state["tasks_df"].columns:
        st.session_state["tasks_df"]["delete"] = False


def df_to_tasks(df: pd.DataFrame) -> List[Task]:
    tasks: List[Task] = []
    if df is None or df.empty:
        return tasks

    df = df.copy()
    # remove rows marked delete
    if "delete" in df.columns:
        df = df[~df["delete"].fillna(False)].copy()

    for i, row in df.iterrows():
        task_id = str(row.get("task_id") or f"T{i+1}").strip() or f"T{i+1}"
        course = str(row.get("course") or "").strip()
        title = str(row.get("title") or "").strip()

        if not title and not course:
            continue

        try:
            dl = coerce_date_value_to_date(row.get("deadline"))
        except Exception:
            continue

        est = safe_float(row.get("estimated_hours"), 0.0)
        pr = clamp_int(safe_int(row.get("priority"), 3), 1, 5)
        notes = str(row.get("notes") or "").strip()

        tasks.append(
            Task(
                task_id=task_id,
                course=course,
                title=title,
                deadline=dl,
                estimated_hours=est,
                priority=pr,
                notes=notes,
            )
        )

    return tasks


def ensure_weekday_blocks_df():
    if "weekday_blocks_df" not in st.session_state:
        st.session_state["weekday_blocks_df"] = pd.DataFrame(
            [
                {"delete": False, "weekday": "שני", "start": "17:00", "end": "19:00", "label": "עבודה/לימודים"},
                {"delete": False, "weekday": "רביעי", "start": "08:00", "end": "12:00", "label": "קורס קבוע"},
            ]
        )


def ensure_date_blocks_df(year: int, month: int):
    if "date_blocks_df" not in st.session_state:
        st.session_state["date_blocks_df"] = pd.DataFrame(
            [
                {"delete": False, "date": f"10/{month:02d}/{year:04d}", "start": "18:00", "end": "22:00", "label": "מחויבות מיוחדת"},
            ]
        )


def ensure_fixed_daily_df():
    if "fixed_daily_df" not in st.session_state:
        st.session_state["fixed_daily_df"] = pd.DataFrame(
            [
                {"delete": False, "start": "13:00", "end": "13:30", "label": "הפסקת צהריים"},
            ]
        )


def df_to_weekday_blocks(df: pd.DataFrame) -> Dict[int, List[Tuple[str, str]]]:
    out: Dict[int, List[Tuple[str, str]]] = {}
    if df is None or df.empty:
        return out
    df = df.copy()
    if "delete" in df.columns:
        df = df[~df["delete"].fillna(False)].copy()

    for _, row in df.fillna("").iterrows():
        wd_name = str(row.get("weekday") or "").strip()
        if wd_name not in WEEKDAY_NAME_TO_INT:
            continue
        s = str(row.get("start") or "").strip()
        e = str(row.get("end") or "").strip()
        if not s or not e:
            continue
        out.setdefault(WEEKDAY_NAME_TO_INT[wd_name], []).append((s, e))
    return out


def df_to_date_blocks(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    if df is None or df.empty:
        return out
    df = df.copy()
    if "delete" in df.columns:
        df = df[~df["delete"].fillna(False)].copy()

    for _, row in df.fillna("").iterrows():
        ds = str(row.get("date") or "").strip()
        s = str(row.get("start") or "").strip()
        e = str(row.get("end") or "").strip()
        if not ds or not s or not e:
            continue
        out.setdefault(ds, []).append((s, e))
    return out


def df_to_fixed_daily_blocks(df: pd.DataFrame) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if df is None or df.empty:
        return out
    df = df.copy()
    if "delete" in df.columns:
        df = df[~df["delete"].fillna(False)].copy()

    for _, row in df.fillna("").iterrows():
        s = str(row.get("start") or "").strip()
        e = str(row.get("end") or "").strip()
        if not s or not e:
            continue
        out.append((s, e))
    return out


# =========================
# Wizard state
# =========================
def init_wizard_state():
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 0
    if "wizard_saved_payload" not in st.session_state:
        st.session_state["wizard_saved_payload"] = None
    if "policy" not in st.session_state:
        st.session_state["policy"] = None
    if "events" not in st.session_state:
        st.session_state["events"] = None
    if "report" not in st.session_state:
        st.session_state["report"] = None


def go_step(n: int):
    st.session_state["wizard_step"] = int(n)
    st.rerun()


def step_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


# =========================
# RTL Styling
# =========================
def inject_rtl_css():
    st.markdown("""
<style>
/* RTL לכל האפליקציה */
html, body, [class*="stApp"] { direction: rtl; text-align: right; }
h1,h2,h3,h4,h5,h6,p,li,div,span,label { direction: rtl; text-align: right; }
section[data-testid="stSidebar"] * { direction: rtl; text-align: right; }
input, textarea { direction: rtl !important; text-align: right !important; }

/* =========
   חריגה: טבלאות Streamlit נשארות LTR (כדי לא לשבור Layout),
   אבל הטקסט בתאים מיושר לימין + עטיפה
   ========= */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
  direction: ltr !important;
}

/* יישור לימין של תוכן טבלה */
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataEditor"] [role="columnheader"],
div[data-testid="stDataEditor"] [role="gridcell"] {
  text-align: right !important;
  direction: rtl !important;
  unicode-bidi: plaintext;
}

/* עטיפת טקסט במקום חיתוך */
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataEditor"] [role="gridcell"] {
  white-space: normal !important;
  word-break: break-word !important;
  overflow: visible !important;
  line-height: 1.25 !important;
}

/* גלילה אופקית אם צריך */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  overflow-x: auto !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="EduPlanner Wizard", layout="wide")
inject_rtl_css()
init_wizard_state()

# =========================
# Global RTL (stable) + Tables exception
# =========================
st.markdown("""
<style>
/* RTL רק לטקסטים רגילים, לא לגריד של טבלאות */
.stMarkdown, .stMarkdown *, .stText, .stCaption, .stHeader, .stSubheader, .stTitle {
  direction: rtl !important;
  text-align: right !important;
}

/* Sidebar RTL */
section[data-testid="stSidebar"] * {
  direction: rtl !important;
  text-align: right !important;
}

/* Inputs RTL */
input, textarea {
  direction: rtl !important;
  text-align: right !important;
}

/* טבלאות Streamlit נשארות LTR (כדי לא לשבור ציור/גלילה) */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
  direction: ltr !important;
  text-align: left !important;
  overflow-x: auto !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar: always show "reset" and minimal nav indicator
st.sidebar.header("EduPlanner ⚙️")
st.sidebar.caption("אפליקציית תכנון חודשית מדורגת.")
if st.sidebar.button("🧹 איפוס מלא", type="secondary"):
    st.session_state.clear()
    st.rerun()

step = int(st.session_state["wizard_step"])

def render_html_table(df: pd.DataFrame, height_px: int = 520):
    if df is None or df.empty:
        st.info("אין נתונים להצגה.", icon="ℹ️")
        return

    # HTML table with RTL and wrapping
    html = df.to_html(index=False, escape=True)
    st.markdown(
        f"""
        <div dir="rtl" style="text-align:right; max-height:{height_px}px; overflow:auto; border:1px solid #e6e6e6; border-radius:10px; padding:8px;">
          <style>
            table {{ width:100%; border-collapse:collapse; }}
            th, td {{ border-bottom:1px solid #eee; padding:8px; vertical-align:top; text-align:right; }}
            th {{ position:sticky; top:0; background:#fff; }}
            td {{ white-space:normal; word-break:break-word; }}
          </style>
          {html}
        </div>
        """,
        unsafe_allow_html=True
    )

# Shared settings kept in session_state
if "tz_name" not in st.session_state:
    st.session_state["tz_name"] = DEFAULT_TZ
if "year" not in st.session_state:
    st.session_state["year"] = 2026
if "month" not in st.session_state:
    st.session_state["month"] = datetime.now().month
if "workday_start" not in st.session_state:
    st.session_state["workday_start"] = DEFAULT_WORKDAY_START
if "workday_end" not in st.session_state:
    st.session_state["workday_end"] = DEFAULT_WORKDAY_END
if "daily_max_hours" not in st.session_state:
    st.session_state["daily_max_hours"] = float(DEFAULT_DAILY_MAX_HOURS)
if "max_task_hours_per_day" not in st.session_state:
    st.session_state["max_task_hours_per_day"] = float(DEFAULT_MAX_TASK_HOURS_PER_DAY)
if "buffer_hours" not in st.session_state:
    st.session_state["buffer_hours"] = int(DEFAULT_BUFFER_HOURS)
if "max_continuous_minutes" not in st.session_state:
    st.session_state["max_continuous_minutes"] = int(DEFAULT_MAX_CONTINUOUS_MINUTES)
if "break_minutes" not in st.session_state:
    st.session_state["break_minutes"] = int(DEFAULT_BREAK_MINUTES)

# Wizard progress indicator
steps_titles = [
    "פתיחה",
    "קורסים ומטלות",
    "חסמים",
    "אילוצים קבועים",
    "פרמטרי עבודה",
    "Policy AI ואימות",
    "חישוב ושיבוץ",
]
st.sidebar.markdown("### שלבים")
for i, t in enumerate(steps_titles):
    marker = "✅" if i < step else ("➡️" if i == step else "•")
    st.sidebar.write(f"{marker} {i}. {t}")

# =========================
# Step 0: Welcome
# =========================
if step == 0:
    step_header("ברוכים הבאים ל־EduPlanner", "אשף מדורג לתכנון חודשי, כולל ייצוא ICS.")
    st.markdown(
        """
<div class="edu-card">
מטרת האפליקציה: להפוך מטלות אקדמיות ולוח זמנים תפוס לתוכנית עבודה חודשית, תוך שמירה על:
<ul>
  <li>מרווח ביטחון לפני דדליינים</li>
  <li>איזון עומסים</li>
  <li>זמן עבודה רציף והפסקות</li>
  <li>אילוצים שבועיים ותאריכיים</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        tz_name = st.text_input("אזור זמן (TZID)", value=st.session_state["tz_name"])
    with c2:
        year = st.selectbox("שנה", options=list(range(2024, 2031)), index=list(range(2024, 2031)).index(st.session_state["year"]))
    with c3:
        month = st.selectbox("חודש", options=list(range(1, 13)), index=int(st.session_state["month"]) - 1)

    st.session_state["tz_name"] = tz_name.strip() or DEFAULT_TZ
    st.session_state["year"] = int(year)
    st.session_state["month"] = int(month)

    st.divider()
    if st.button("בוא/י נתחיל", type="primary"):
        go_step(1)

# =========================
# Step 1: Tasks
# =========================
elif step == 1:
    step_header("הזנת קורסים ומטלות", "הכנס מטלות, כולל דדליין בפורמט dd/mm/yyyy.")
    year = int(st.session_state["year"])
    month = int(st.session_state["month"])
    ensure_tasks_df(year, month)

    with st.form("add_task_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 2])
        with c1:
            course = st.text_input("שם הקורס")
            title = st.text_input("שם המטלה")
            notes = st.text_area("הערות", height=90)

        with c2:
            deadline = st.text_input("דדליין (dd/mm/yyyy)", placeholder="לדוגמה: 12/03/2026")
            estimated_hours = st.number_input("שעות משוערות", min_value=0.0, value=3.0, step=0.5)
            priority = st.selectbox("עדיפות (1-5)", options=[1, 2, 3, 4, 5], index=2)

        submit = st.form_submit_button("➕ הוסף מטלה")

    if submit:
        try:
            dl = parse_date_any(deadline)  # הפונקציה שלך שתומכת dd/mm/yyyy
        except Exception:
            st.error("דדליין לא תקין. יש להזין בפורמט dd/mm/yyyy, לדוגמה 12/03/2026.")
            st.stop()

        df = st.session_state["tasks_df"].copy()

        # מזהה אוטומטי
        next_id = f"T{len(df) + 1}"
        row = {
            "task_id": next_id,
            "course": course.strip(),
            "title": title.strip(),
            "deadline": dl.strftime("%d/%m/%Y"),
            "estimated_hours": float(estimated_hours),
            "priority": int(priority),
            "notes": notes.strip(),
            "delete": False,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        st.session_state["tasks_df"] = df
        st.success("המטלה נוספה.")

    st.divider()
    st.subheader("רשימת מטלות")

    # תצוגה קריאה בלבד (מומלץ HTML כדי למנוע תקלות RTL)
    df_view = st.session_state["tasks_df"].copy()
    if "delete" not in df_view.columns:
        df_view["delete"] = False

    render_html_table(df_view.drop(columns=[]), height_px=420)

    st.caption("למחיקה: סמן/י בעמודת delete ואז לחץ/י על כפתור המחיקה למטה.")

    c_del1, c_del2 = st.columns([1, 3])
    with c_del1:
        if st.button("🗑️ מחק שורות מסומנות", type="secondary"):
            df = st.session_state["tasks_df"].copy()
            if "delete" in df.columns:
                df = df[~df["delete"].fillna(False)].copy()
                df["delete"] = False
                st.session_state["tasks_df"] = df
                st.success("השורות המסומנות נמחקו.")
                st.rerun()

        save = st.form_submit_button("💾 שמירה והמשך", type="primary")
        back = st.form_submit_button("⬅️ חזרה", type="secondary")

    if back:
        go_step(0)

    if save:
        df = st.session_state["tasks_df"].copy()
        for c in TASK_COLS:
            if c not in df.columns:
                df[c] = False if c == "delete" else ""
        st.session_state["tasks_df"] = df[TASK_COLS].copy()

        # Validate at least one task
        tasks = df_to_tasks(st.session_state["tasks_df"])
        if not tasks:
            st.error("לא נמצאו מטלות תקינות. ודא/י שיש דדליין בפורמט dd/mm/yyyy ושעות משוערות.")
        else:
            go_step(2)

# =========================
# Step 2: Constraints (weekday/date)
# =========================
elif step == 2:
    step_header("הזנת חסמים", "חסמים הם זמנים תפוסים שאסור לשבץ בהם עבודה.")
    year = int(st.session_state["year"])
    month = int(st.session_state["month"])
    ensure_weekday_blocks_df()
    ensure_date_blocks_df(year, month)

    with st.form("form_blocks", clear_on_submit=False):
        st.subheader("חסמים שבועיים קבועים")
        wd_df = st.data_editor(
            st.session_state["weekday_blocks_df"],
            use_container_width=True,
            height=420,
            num_rows="dynamic",
            key="editor_weekday_blocks_v1",
            column_config={
                "weekday": st.column_config.SelectboxColumn("יום", options=WEEKDAYS_HE, width="small"),
                "start": st.column_config.TextColumn("התחלה", width="small"),
                "end": st.column_config.TextColumn("סיום", width="small"),
                "label": st.column_config.TextColumn("תיאור", width="large"),
                "delete": st.column_config.CheckboxColumn("מחיקה", width="small"),
            },
        )

        st.subheader("חסמים בתאריכים ספציפיים")
        date_df = st.data_editor(
            st.session_state["date_blocks_df"],
            use_container_width=True,
            height=420,
            num_rows="dynamic",
            key="editor_date_blocks_v1",
            column_config={
                "delete": st.column_config.CheckboxColumn("מחיקה", width="small"),
                "date": st.column_config.TextColumn("תאריך (dd/mm/yyyy)", help="לדוגמה: 10/03/2026", width="medium"),
                "start": st.column_config.TextColumn("התחלה (HH:MM)", width="small"),
                "end": st.column_config.TextColumn("סיום (HH:MM)", width="small"),
                "label": st.column_config.TextColumn("תיאור", width="medium"),
            },
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            back = st.form_submit_button("⬅️ חזרה", type="secondary")
        with c2:
            save = st.form_submit_button("💾 שמירה והמשך", type="primary")

    if back:
        go_step(1)

    if save:
        st.session_state["weekday_blocks_df"] = wd_df.copy()
        st.session_state["date_blocks_df"] = date_df.copy()
        go_step(3)

# =========================
# Step 3: Fixed constraints (daily routines)
# =========================
elif step == 3:
    step_header("הזנת אילוצים קבועים", "אילוצים יומיים שחוזרים בכל יום (למשל ארוחות, הסעות, מנוחה).")
    ensure_fixed_daily_df()

    with st.form("form_fixed_daily", clear_on_submit=False):
        fixed_df = st.data_editor(
            st.session_state["fixed_daily_df"],
            use_container_width=True,
            height=360,
            num_rows="dynamic",
            key="editor_fixed_daily_v1",
            column_config={
                "delete": st.column_config.CheckboxColumn("מחיקה", width="small"),
                "start": st.column_config.TextColumn("התחלה (HH:MM)", width="large"),
                "end": st.column_config.TextColumn("סיום (HH:MM)", width="small"),
                "label": st.column_config.TextColumn("תיאור", width="medium"),
            },
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            back = st.form_submit_button("⬅️ חזרה", type="secondary")
        with c2:
            save = st.form_submit_button("💾 שמירה והמשך", type="primary")

    if back:
        go_step(2)

    if save:
        st.session_state["fixed_daily_df"] = fixed_df.copy()
        go_step(4)

# =========================
# Step 4: Work parameters (continuous work, breaks, day start/end, caps)
# =========================
elif step == 4:
    step_header("הגדרת זמן עבודה", "כאן נקבעים תחילת יום, סוף יום, עומסים, זמן עבודה רציף והפסקות.")
    with st.form("form_work_params", clear_on_submit=False):
        tz_name = st.text_input("אזור זמן (TZID)", value=st.session_state["tz_name"])
        st.session_state["tz_name"] = tz_name.strip() or DEFAULT_TZ

        c1, c2 = st.columns(2)
        with c1:
            daily_max_hours = st.number_input(
                "כמה שעות מקסימליות ביום?",
                min_value=1.0,
                max_value=12.0,
                value=float(st.session_state["daily_max_hours"]),
                step=0.5,
            )
            workday_start_t = st.time_input("מתי להתחיל את היום?", value=parse_hhmm(st.session_state["workday_start"]))
            max_continuous_minutes = st.selectbox(
                "זמן עבודה רציף, אורך בלוק עבודה (דקות)",
                options=[45, 60, 75, 90, 120, 150, 180, 240],
                index=[45, 60, 75, 90, 120, 150, 180, 240].index(int(st.session_state["max_continuous_minutes"]))
                if int(st.session_state["max_continuous_minutes"]) in [45, 60, 75, 90, 120, 150, 180, 240]
                else [45, 60, 75, 90, 120, 150, 180, 240].index(120),
            )

        with c2:
            max_task_hours_per_day = st.number_input(
                "מקסימום שעות לאותה מטלה ביום",
                min_value=1.0,
                max_value=8.0,
                value=float(st.session_state["max_task_hours_per_day"]),
                step=0.5,
            )
            workday_end_t = st.time_input("מתי לסיים את היום?", value=parse_hhmm(st.session_state["workday_end"]))
            break_minutes = st.selectbox(
                "הפסקה מינימלית בין בלוקים (דקות)",
                options=[0, 5, 10, 15, 20, 30, 45, 60],
                index=[0, 5, 10, 15, 20, 30, 45, 60].index(int(st.session_state["break_minutes"]))
                if int(st.session_state["break_minutes"]) in [0, 5, 10, 15, 20, 30, 45, 60]
                else [0, 5, 10, 15, 20, 30, 45, 60].index(15),
            )

        buffer_hours = st.selectbox(
            "מרווח ביטחון לפני דדליין (שעות)",
            options=[0, 12, 24, 36, 48, 72, 96],
            index=[0, 12, 24, 36, 48, 72, 96].index(int(st.session_state["buffer_hours"]))
            if int(st.session_state["buffer_hours"]) in [0, 12, 24, 36, 48, 72, 96]
            else [0, 12, 24, 36, 48, 72, 96].index(48),
        )

        c3, c4 = st.columns([1, 1])
        with c3:
            back = st.form_submit_button("⬅️ חזרה", type="secondary")
        with c4:
            save = st.form_submit_button("💾 שמירה והמשך", type="primary")

    if back:
        go_step(3)

    if save:
        st.session_state["daily_max_hours"] = float(daily_max_hours)
        st.session_state["max_task_hours_per_day"] = float(max_task_hours_per_day)
        st.session_state["workday_start"] = workday_start_t.strftime("%H:%M")
        st.session_state["workday_end"] = workday_end_t.strftime("%H:%M")
        st.session_state["max_continuous_minutes"] = int(max_continuous_minutes)
        st.session_state["break_minutes"] = int(break_minutes)
        st.session_state["buffer_hours"] = int(buffer_hours)

        go_step(5)

# =========================
# Step 5: AI policy + review + save payload
# =========================
elif step == 5:
    step_header("Policy בינה מלאכותית ואימות נתונים", "הדבק מדיניות חופשית (אופציונלי), בדוק/י את הנתונים, ושמור/י.")
    st.info("ה־Policy לא מחליף את המנוע. הוא מכוון העדפות, זמן רציף והפסקות, באופן מבוקר.", icon="🧠")

    # Policy input
    policy_text = st.text_area(
        "Policy חופשי (אופציונלי)",
        height=140,
        placeholder="לדוגמה: אני עובד הכי טוב בבוקר. הימנע משישי. תן עדיפות לקורס 'ביולוגיה'. אני רוצה בלוק של 120 דקות והפסקה 15 דקות.",
        key="policy_text_area",
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        apply_ai = st.button("🧠 הפק Policy באמצעות AI", type="secondary")
    with c2:
        back = st.button("⬅️ חזרה", type="secondary")
    with c3:
        save_all = st.button("💾 שמירת נתונים ואישור", type="primary")

    if back:
        go_step(4)

    if apply_ai:
        with st.spinner("המערכת מפיקה policy מהטקסט באמצעות Gemini..."):
            pol = ai_extract_policy(policy_text, st.session_state["tz_name"])
            # override hard values from UI if user already chose, but allow AI to update if user wants
            # Here: AI wins, but you can invert if you prefer
            st.session_state["policy"] = pol
        st.success("Policy עודכן.")

    # Build a "resolved policy" used for scheduling (merge UI + AI)
    pol = st.session_state.get("policy") or {
        "hard": {"max_continuous_minutes": st.session_state["max_continuous_minutes"], "break_minutes": st.session_state["break_minutes"]},
        "preferences": {"dayparts": ["morning", "afternoon"], "focus_weekdays": [], "avoid_weekdays": [], "course_bias": {}},
        "notes": "ברירת מחדל (ללא AI).",
    }

    # Ensure UI-defined hard values are always present
    pol = dict(pol)
    pol_hard = pol.get("hard", {}) if isinstance(pol.get("hard", {}), dict) else {}
    pol_hard["max_continuous_minutes"] = int(pol_hard.get("max_continuous_minutes", st.session_state["max_continuous_minutes"]))
    pol_hard["break_minutes"] = int(pol_hard.get("break_minutes", st.session_state["break_minutes"]))
    pol["hard"] = pol_hard

    # Preview payload
    tasks = df_to_tasks(st.session_state["tasks_df"])
    weekday_blocks = df_to_weekday_blocks(st.session_state["weekday_blocks_df"])
    date_blocks = df_to_date_blocks(st.session_state["date_blocks_df"])
    fixed_daily_blocks = df_to_fixed_daily_blocks(st.session_state["fixed_daily_df"])

    preview = {
        "tz_name": st.session_state["tz_name"],
        "year": int(st.session_state["year"]),
        "month": int(st.session_state["month"]),
        "workday_start": st.session_state["workday_start"],
        "workday_end": st.session_state["workday_end"],
        "daily_max_hours": float(st.session_state["daily_max_hours"]),
        "max_task_hours_per_day": float(st.session_state["max_task_hours_per_day"]),
        "buffer_hours": int(st.session_state["buffer_hours"]),
        "policy": pol,
        "tasks_count": len(tasks),
        "weekday_blocks_count": int(st.session_state["weekday_blocks_df"][~st.session_state["weekday_blocks_df"].get("delete", False)].shape[0]),
        "date_blocks_count": int(st.session_state["date_blocks_df"][~st.session_state["date_blocks_df"].get("delete", False)].shape[0]),
        "fixed_daily_blocks_count": int(st.session_state["fixed_daily_df"][~st.session_state["fixed_daily_df"].get("delete", False)].shape[0]),
    }

    st.subheader("תצוגה לאימות")
    st.json(preview)

    if save_all:
        # Save a stable payload snapshot for compute step
        st.session_state["wizard_saved_payload"] = {
            "tasks": [t.__dict__ | {"deadline": t.deadline.isoformat()} for t in tasks],
            "weekday_blocks_df": st.session_state["weekday_blocks_df"].to_dict(orient="records"),
            "date_blocks_df": st.session_state["date_blocks_df"].to_dict(orient="records"),
            "fixed_daily_df": st.session_state["fixed_daily_df"].to_dict(orient="records"),
            "settings": preview,
        }
        st.session_state["policy"] = pol
        st.success("הנתונים נשמרו. ניתן להמשיך לחישוב.")
        go_step(6)

# =========================
# Step 6: Compute + results + ICS
# =========================
elif step == 6:
    step_header("חישוב שיבוץ וייצוא", "מנוע דטרמיניסטי, עם policy אופציונלי, וייצוא ICS תקני.")

    if not st.session_state.get("wizard_saved_payload"):
        st.warning("לא נמצא snapshot שמור. חזור/י לשלב Policy ואימות ושמור/י.", icon="⚠️")
        if st.button("⬅️ חזרה לשלב 5", type="secondary"):
            go_step(5)

    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            back = st.button("⬅️ חזרה", type="secondary")
        with c2:
            compute = st.button("🚀 חשב לו״ז חודשי", type="primary")
        with c3:
            clear_results = st.button("🧽 נקה תוצאות", type="secondary")

        if back:
            go_step(5)

        if clear_results:
            st.session_state["events"] = None
            st.session_state["report"] = None
            st.rerun()

        if compute:
            # ---- validate tasks (no missing deadlines) ----
            tasks = df_to_tasks(st.session_state["tasks_df"])

            missing = 0
            df_check = st.session_state["tasks_df"]
            if df_check is not None and not df_check.empty:
                for row in df_check.to_dict(orient="records"):
                    if row.get("delete"):
                        continue
                    try:
                        _ = coerce_date_value_to_date(row.get("deadline"))
                    except Exception:
                        missing += 1

            if missing > 0:
                st.error(f"יש {missing} שורות מטלה עם דדליין חסר/לא תקין. תקן/י או סמן/י למחיקה ואז נסה שוב.")
                st.stop()

            weekday_blocks = df_to_weekday_blocks(st.session_state["weekday_blocks_df"])
            date_blocks = df_to_date_blocks(st.session_state["date_blocks_df"])
            fixed_daily_blocks = df_to_fixed_daily_blocks(st.session_state["fixed_daily_df"])

            schedule_params = {
                "tasks": tasks,
                "tz_name": st.session_state["tz_name"],
                "year": int(st.session_state["year"]),
                "month": int(st.session_state["month"]),
                "work_start_hhmm": st.session_state["workday_start"],
                "work_end_hhmm": st.session_state["workday_end"],
                "daily_max_hours": float(st.session_state["daily_max_hours"]),
                "max_task_hours_per_day": float(st.session_state["max_task_hours_per_day"]),
                "buffer_hours": int(st.session_state["buffer_hours"]),
                "weekday_blocks": weekday_blocks,
                "date_blocks": date_blocks,
                "fixed_daily_blocks": fixed_daily_blocks,
                "policy": st.session_state.get("policy"),
            }

            prog = st.progress(0)
            with st.spinner("המנוע מבצע שיבוץ תוך כיבוד אילוצים, הפסקות ועומסים..."):
                prog.progress(20)
                try:
                    events, report = schedule_tasks(**schedule_params)
                    prog.progress(90)
                    st.session_state["events"] = events
                    st.session_state["report"] = report
                    prog.progress(100)
                    st.success(f"השיבוץ הושלם. נוצרו {len(events)} אירועים (כולל חסמים אם הופעלו).")
                except Exception as e:
                    st.error("שגיאה בקריאה ל-schedule_tasks:")
                    st.exception(e)
                    st.stop()

        events = st.session_state.get("events")
        report = st.session_state.get("report")

        if events and report:
            # ----------- quick visibility: policy + counts -----------
            pol = st.session_state.get("policy")
            if pol:
                with st.expander("Policy שנקלט (לבדיקה)", expanded=False):
                    st.json(pol)

            task_events = [e for e in events if getattr(e, "kind", "") != "constraint"]
            constraint_events = [e for e in events if getattr(e, "kind", "") == "constraint"]

            st.caption(f"סה״כ אירועים: {len(events)} | מטלות: {len(task_events)} | חסמים: {len(constraint_events)}")

            # ----------- Report -----------
            st.subheader("דו״ח תקינות")
            st.json(report)

            # ----------- Tables -----------
            def _events_to_rows(ev_list):
                rows = []
                for ev in sorted(ev_list, key=lambda e: e.start_dt):
                    rows.append(
                        {
                            "כותרת": ev.title,
                            "סוג": "חסם" if getattr(ev, "kind", "") == "constraint" else "מטלה",
                            "תאריך": ev.start_dt.strftime("%d/%m/%Y"),
                            "יום": WEEKDAYS_HE[ev.start_dt.weekday()],
                            "התחלה": ev.start_dt.strftime("%H:%M"),
                            "סיום": ev.end_dt.strftime("%H:%M"),
                        }
                    )
                df = pd.DataFrame(rows)
                if not df.empty:
                    # סדר עמודות כדי לתת לכותרת יותר מקום בפועל
                    df = df[["כותרת", "סוג", "תאריך", "יום", "התחלה", "סיום"]]
                return df

            st.subheader("מטלות משובצות")
            df_tasks = _events_to_rows(task_events)
            if df_tasks.empty:
                st.warning("לא נמצאו מטלות משובצות. בדוק/י חלונות עבודה, מגבלות יומיות, ו־Buffer.", icon="⚠️")
            else:
                render_html_table(df_tasks, height_px=520)

            st.subheader("חסמים שנכללו ביומן")
            df_constraints = _events_to_rows(constraint_events)
            if df_constraints.empty:
                st.info("לא הוגדרו/נכללו חסמים בשיבוץ.", icon="ℹ️")
            else:
                render_html_table(df_constraints, height_px=420)

            # ----------- ICS export -----------
            st.subheader("ייצוא ליומן")
            ics_text = build_ics(events, st.session_state["tz_name"])
            st.download_button(
                "⬇️ הורדת קובץ ICS",
                data=ics_text.encode("utf-8"),
                file_name=f"eduplanner_{st.session_state['year']}_{st.session_state['month']:02d}.ics",
                mime="text/calendar",
            )

            # Snapshot export
            payload = st.session_state.get("wizard_saved_payload")
            if payload:
                st.download_button(
                    "⬇️ הורדת Snapshot (JSON)",
                    data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="eduplanner_snapshot.json",
                    mime="application/json",
                )

        else:
            st.info("כדי לראות תוצאות, לחץ/י על 'חשב לו״ז חודשי'.", icon="ℹ️")