# app.py
# Streamlit Academic Planner, monthly scheduling + ICS export (Google Calendar compatible)
# Design goal: keep your general flow and style, add "daily max hours", "when to start the day",
# and convert scheduling from AI-guessing to deterministic constraint-based scheduling.
#
# Run:
#   streamlit run app.py
#
# Notes:
# 1) The scheduling engine is deterministic and validated.
# 2) AI (optional) is used only for explanation and optional parsing, not for core scheduling.
# 3) ICS is generated without external dependencies.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st
import google.generativeai as genai


# =========================
# Configuration
# =========================
APP_TITLE = "מתכנן המטלות האקדמי שלי 🎓"
DEFAULT_TZ = "Asia/Jerusalem"
DEFAULT_WORKDAY_START = "08:30"
DEFAULT_WORKDAY_END = "21:30"
DEFAULT_DAILY_MAX_HOURS = 4.0
DEFAULT_MAX_TASK_HOURS_PER_DAY = 3.0
DEFAULT_SLOT_MINUTES = 60
DEFAULT_BUFFER_HOURS = 48

st.markdown(
    """
    <style>
    html, body, [class*="st-"] {
        direction: rtl;
        text-align: right;
        font-family: "Arial", "Rubik", sans-serif;
    }

    textarea, input {
        direction: rtl;
        text-align: right;
    }

    /* Data editor */
    div[data-testid="stDataEditor"] {
        direction: rtl;
    }

    /* Keep numbers LTR */
    .ltr {
        direction: ltr;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ===== constants =====
TASK_COLS = ["task_id", "course", "title", "deadline", "estimated_hours", "priority", "notes"]

# ===== UI selections =====
year = st.selectbox(
    "שנה",
    options=list(range(2025, 2031)),
    index=1,
)

month = st.selectbox(
    "חודש",
    options=list(range(1, 13)),
    index=1,
)

def ensure_session_defaults(year: int, month: int) -> None:
    if "tasks_df" not in st.session_state:
        st.session_state["tasks_df"] = pd.DataFrame(
            [
                {"task_id": "T1", "course": "קורס לדוגמה", "title": "עבודה מסכמת", "deadline": f"{year:04d}-{month:02d}-20", "estimated_hours": 6.0, "priority": 4, "notes": ""},
                {"task_id": "T2", "course": "קורס לדוגמה", "title": "קריאת מאמר", "deadline": f"{year:04d}-{month:02d}-12", "estimated_hours": 3.0, "priority": 3, "notes": ""},
            ],
            columns=TASK_COLS,
        )

    if "weekday_blocks_df" not in st.session_state:
        st.session_state["weekday_blocks_df"] = pd.DataFrame(
            [
                {"weekday": "שני", "start": "17:00", "end": "19:00", "label": "עבודה/לימודים"},
                {"weekday": "רביעי", "start": "08:00", "end": "12:00", "label": "קורס קבוע"},
            ]
        )

    if "date_blocks_df" not in st.session_state:
        st.session_state["date_blocks_df"] = pd.DataFrame(
            [
                {"date": f"{year:04d}-{month:02d}-10", "start": "18:00", "end": "22:00", "label": "מחויבות מיוחדת"},
            ]
        )

# Call once, BEFORE rendering editors
ensure_session_defaults(year, month)

WEEKDAYS_HE = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
# Python weekday(): Monday=0 ... Sunday=6
WEEKDAY_NAME_TO_INT = {"שני": 0, "שלישי": 1, "רביעי": 2, "חמישי": 3, "שישי": 4, "שבת": 5, "ראשון": 6}

# =========================
# AI Model
# =========================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-3-pro-preview')
except:
    st.error("שגיאה בחיבור ל-AI. בדוק את ה-Secrets.")
    st.stop()

# =========================
# RTL + Hebrew localization (Streamlit)
# Put this near the top of app.py (after st.set_page_config), before rendering UI.
# Goal: enforce RTL layout, Hebrew-friendly typography, correct alignment for labels,
# and keep numbers/dates readable (LTR) inside an RTL interface.
# ========================= 

def apply_hebrew_rtl_ui(app_title: str = "מתכנן המטלות האקדמי שלי 🎓") -> None:
    st.markdown(
        f"""
        <style>
        /* ---------- Global direction ---------- */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            direction: rtl;
            text-align: right;
        }}

        /* Main container */
        section.main > div {{
            direction: rtl;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            direction: rtl;
            text-align: right;
        }}
        [data-testid="stSidebar"] * {{
            direction: rtl;
            text-align: right;
        }}

        /* Headings and text */
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
            text-align: right;
        }}

        /* Inputs: align text to the right (Hebrew) */
        input, textarea {{
            direction: rtl;
            text-align: right;
        }}

        /* BUT: force LTR for numeric/date/time fields to avoid confusion */
        /* You can add class hooks via st.text_input(..., key="...") and target them if needed. */
        input[type="number"], input[type="date"], input[type="time"] {{
            direction: ltr;
            text-align: left;
        }}

        /* Data editor / tables: keep overall RTL, but allow LTR in numeric cells */
        [data-testid="stDataFrame"] {{
            direction: rtl;
        }}
        [data-testid="stDataFrame"] table {{
            direction: rtl;
        }}
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
            text-align: right;
            unicode-bidi: plaintext;
        }}

        /* Code blocks and JSON should remain LTR */
        pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {{
            direction: ltr !important;
            text-align: left !important;
        }}

        /* Buttons alignment */
        button {{
            direction: rtl;
        }}

        /* Small polish: consistent spacing */
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }}

        /* Optional: slightly improve readability on dark theme */
        .he-hint {{
            opacity: 0.88;
        }}
        </style>

        <!-- Set document language for accessibility and better text handling -->
        <script>
        document.documentElement.lang = "he";
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Optional: a clean RTL title row that stays consistent
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:flex-start; gap:0.6rem;">
          <h1 style="margin:0; direction:rtl; text-align:right;">{app_title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Call once early in your app, after st.set_page_config(...)
apply_hebrew_rtl_ui(APP_TITLE)

# =========================
# Hebrew-first constants (optional centralization)
# Keep UI text in one place to avoid mixed-language drift.
# =========================
UI = {
    "caption": "הזנת מטלות, חישוב לו״ז חודשי דטרמיניסטי, וייצוא לקובץ ICS ליומן Google. עיצוב נקי, תכנון אמין.",
    "settings_header": "הגדרות מערכת ⚙️",
    "timezone": "אזור זמן (TZID)",
    "choose_year": "בחר שנת לימודים",
    "choose_month": "בחר חודש לתכנון",
    "daily_planning": "תכנון זמן יומי",
    "daily_max": "כמה שעות מקסימליות ביום?",
    "day_start": "מתי להתחיל את היום? (HH:MM)",
    "day_end": "מתי לסיים את היום? (HH:MM)",
    "rules": "כללי שיבוץ",
    "max_task_per_day": "מקסימום שעות לאותה מטלה ביום",
    "slot_minutes": "גודל משבצת (דקות)",
    "buffer": "מרווח ביטחון לפני דדליין (שעות)",
    "clear_all": "🧹 ניקוי כל הנתונים",
    "tasks_title": "הזנת מטלות 📝",
    "tasks_hint": "מומלץ להזין מטלות בצורה מובנית. ניתן גם להדביק טקסט חופשי, ואז לבצע חילוץ בסיסי.",
    "constraints_title": "הגדרת חסמים ⛔",
    "compute_title": "חישוב לו״ז אסטרטגי וחכם 🚀",
    "compute_btn": "🚀 חשב לו״ז אסטרטגי וחכם",
    "explain_btn": "🧠 צור הסבר והמלצות (אופציונלי)",
    "download_ics": "⬇️ הורד קובץ ICS (Google Calendar)",
}

# Usage example:
# st.sidebar.header(UI["settings_header"])
# st.caption(UI["caption"])

# =========================
# Utilities
# =========================
def coerce_dates_for_editor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Makes df[col] compatible with st.column_config.DateColumn by coercing to datetime64[ns].
    Supports YYYY-MM-DD and DD/MM/YYYY strings, plus existing datetime/date objects.
    """
    if df is None or df.empty or col not in df.columns:
        return df

    s = df[col]

    # Already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(s):
        return df

    def _parse_one(x):
        if x is None:
            return pd.NaT
        try:
            if pd.isna(x):
                return pd.NaT
        except Exception:
            pass

        if isinstance(x, (pd.Timestamp, datetime)):
            return pd.Timestamp(x)
        if isinstance(x, date):
            return pd.Timestamp(datetime.combine(x, time(0, 0)))

        txt = str(x).strip()
        if not txt:
            return pd.NaT

        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return pd.Timestamp(datetime.strptime(txt, fmt))
            except Exception:
                continue

        return pd.NaT

    df = df.copy()
    df[col] = s.apply(_parse_one)
    return df

def parse_hhmm(s: str) -> time:
    s = (s or "").strip()
    if not re.match(r"^\d{2}:\d{2}$", s):
        raise ValueError("פורמט שעה לא תקין, נדרש HH:MM")
    hh, mm = s.split(":")
    hh_i, mm_i = int(hh), int(mm)
    if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
        raise ValueError("שעה לא תקינה")
    return time(hh_i, mm_i)


def minutes_between(t1: time, t2: time) -> int:
    d = date(2000, 1, 1)
    dt1 = datetime.combine(d, t1)
    dt2 = datetime.combine(d, t2)
    return int((dt2 - dt1).total_seconds() // 60)


def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def month_date_range(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start, end


def daterange(d1: date, d2_exclusive: date):
    cur = d1
    while cur < d2_exclusive:
        yield cur
        cur = cur + timedelta(days=1)


def safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


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
    priority: int = 3
    notes: str = ""

    def to_display_name(self) -> str:
        base = f"{self.course} | {self.title}".strip()
        return base if base else self.title or self.task_id


@dataclass
class Event:
    title: str
    start_dt: datetime
    end_dt: datetime
    description: str = ""


# =========================
# Constraints and Slots
# =========================
def build_daily_available_windows(
    day: date,
    tz: ZoneInfo,
    work_start: time,
    work_end: time,
    weekday_blocks: Dict[int, List[Tuple[time, time]]],
    date_blocks: Dict[date, List[Tuple[time, time]]],
) -> List[Tuple[datetime, datetime]]:
    """
    Returns list of available datetime windows for a given day after removing blocked intervals.
    Strategy:
      1) Start with one window [work_start, work_end]
      2) subtract blocked windows (weekday + specific date)
      3) return sorted, non-overlapping windows
    """
    base_start = datetime.combine(day, work_start, tzinfo=tz)
    base_end = datetime.combine(day, work_end, tzinfo=tz)

    if base_end <= base_start:
        return []

    blocks = []
    wd = day.weekday()
    for b in weekday_blocks.get(wd, []):
        blocks.append((datetime.combine(day, b[0], tzinfo=tz), datetime.combine(day, b[1], tzinfo=tz)))
    for b in date_blocks.get(day, []):
        blocks.append((datetime.combine(day, b[0], tzinfo=tz), datetime.combine(day, b[1], tzinfo=tz)))

    blocks = [(s, e) for (s, e) in blocks if e > s]
    blocks.sort(key=lambda x: x[0])

    # Merge overlaps
    merged = []
    for s, e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    blocks = [(x[0], x[1]) for x in merged]

    # Subtract blocks from base window
    windows = [(base_start, base_end)]
    for bs, be in blocks:
        new_windows = []
        for ws, we in windows:
            if be <= ws or bs >= we:
                new_windows.append((ws, we))
                continue
            if bs > ws:
                new_windows.append((ws, bs))
            if be < we:
                new_windows.append((be, we))
        windows = new_windows

    # Remove tiny or invalid windows
    clean = []
    for ws, we in windows:
        if we > ws and int((we - ws).total_seconds() // 60) >= 15:
            clean.append((ws, we))
    clean.sort(key=lambda x: x[0])
    return clean


def generate_daily_slots(
    windows: List[Tuple[datetime, datetime]],
    slot_minutes: int,
) -> List[Tuple[datetime, datetime]]:
    slots = []
    for ws, we in windows:
        cur = ws
        while cur + timedelta(minutes=slot_minutes) <= we:
            slots.append((cur, cur + timedelta(minutes=slot_minutes)))
            cur = cur + timedelta(minutes=slot_minutes)
    return slots


# =========================
# Deterministic Scheduling Engine (with occupancy + progression)
# =========================
def schedule_tasks(
    tasks: List[Task],
    tz_name: str,
    year: int,
    month: int,
    work_start_hhmm: str,
    work_end_hhmm: str,
    daily_max_hours: float,
    max_task_hours_per_day: float,
    slot_minutes: int,
    buffer_hours: int,
    weekday_blocks: Dict[int, List[Tuple[str, str]]],
    date_blocks: Dict[str, List[Tuple[str, str]]],
) -> Tuple[List[Event], Dict]:
    """
    Returns (events, report).
    report includes validation flags and unscheduled work.

    Key properties:
    - No overlaps: slot-level occupancy is enforced (occupied_start_times by date).
    - Time progression: allocation consumes unique slots; it never reuses the same HH:MM on the same date.
    - Respects weekday/date blocks, work window, daily max minutes, and max per task per day.
    """
    tz = ZoneInfo(tz_name)

    work_start = parse_hhmm(work_start_hhmm)
    work_end = parse_hhmm(work_end_hhmm)

    # -------------------------
    # Safety filter: remove tasks with invalid deadlines (NaT/None/Timestamp)
    # -------------------------
    cleaned_tasks: List[Task] = []
    dropped_tasks: List[Dict] = []

    for t in tasks:
        dl = getattr(t, "deadline", None)

        # None deadline
        if dl is None:
            dropped_tasks.append(
                {"task_id": getattr(t, "task_id", ""), "course": getattr(t, "course", ""), "title": getattr(t, "title", ""), "reason": "דדליין חסר (None)."}
            )
            continue

        # pandas NaT / missing timestamps
        try:
            if isinstance(dl, (pd.Timestamp, datetime)) and pd.isna(dl):
                dropped_tasks.append(
                    {"task_id": getattr(t, "task_id", ""), "course": getattr(t, "course", ""), "title": getattr(t, "title", ""), "reason": "דדליין חסר (NaT)."}
                )
                continue
        except Exception:
            pass

        # normalize Timestamp/datetime -> date
        if isinstance(dl, (pd.Timestamp, datetime)):
            dl = dl.date()

        # ensure it's a pure date object
        if not isinstance(dl, date) or isinstance(dl, datetime):
            dropped_tasks.append(
                {"task_id": getattr(t, "task_id", ""), "course": getattr(t, "course", ""), "title": getattr(t, "title", ""), "reason": f"סוג דדליין לא תקין: {type(getattr(t, 'deadline', None))}"}
            )
            continue

        # write back normalized value
        try:
            t.deadline = dl
        except Exception:
            pass

        cleaned_tasks.append(t)

    tasks = cleaned_tasks

    # עכשיו המיון בטוח
    tasks_sorted = sorted(
        tasks,
        key=lambda t: (t.deadline, -int(clamp_float(t.priority, 1, 5)), -float(t.estimated_hours)),
    )

    weekday_blocks_t: Dict[int, List[Tuple[time, time]]] = {}
    for wd, blocks in (weekday_blocks or {}).items():
        out: List[Tuple[time, time]] = []
        for s, e in blocks:
            try:
                out.append((parse_hhmm(s), parse_hhmm(e)))
            except Exception:
                continue
        weekday_blocks_t[wd] = out

    date_blocks_t: Dict[date, List[Tuple[time, time]]] = {}
    for ds, blocks in (date_blocks or {}).items():
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        out: List[Tuple[time, time]] = []
        for s, e in blocks:
            try:
                out.append((parse_hhmm(s), parse_hhmm(e)))
            except Exception:
                continue
        date_blocks_t[d] = out

    month_start, month_end_excl = month_date_range(year, month)
    now_dt = datetime.now(tz=tz)

    tasks_sorted = sorted(
        tasks,
        key=lambda t: (t.deadline, -int(clamp_float(t.priority, 1, 5)), -float(t.estimated_hours)),
    )

    day_slots: Dict[date, List[Tuple[datetime, datetime]]] = {}
    for d in daterange(month_start, month_end_excl):
        windows = build_daily_available_windows(d, tz, work_start, work_end, weekday_blocks_t, date_blocks_t)
        slots = generate_daily_slots(windows, slot_minutes)
        day_slots[d] = slots

    daily_max_minutes = int(daily_max_hours * 60)
    max_task_minutes_per_day = int(max_task_hours_per_day * 60)
    buffer_delta = timedelta(hours=buffer_hours)

    used_minutes_by_day: Dict[date, int] = {d: 0 for d in day_slots.keys()}
    used_minutes_by_task_day: Dict[Tuple[str, date], int] = {}

    # Occupancy: prevent reusing the same start time for multiple events on the same date
    # and optionally track (start_dt, end_dt) to guard against non-uniform slot sizes.
    occupied_starts_by_day: Dict[date, set] = {d: set() for d in day_slots.keys()}
    occupied_ranges_by_day: Dict[date, List[Tuple[datetime, datetime]]] = {d: [] for d in day_slots.keys()}

    events: List[Event] = []
    unscheduled: List[Dict] = []

    def day_has_capacity(d: date, add_minutes: int) -> bool:
        return used_minutes_by_day.get(d, 0) + add_minutes <= daily_max_minutes

    def task_day_has_capacity(task_id: str, d: date, add_minutes: int) -> bool:
        key = (task_id, d)
        return used_minutes_by_task_day.get(key, 0) + add_minutes <= max_task_minutes_per_day

    def slot_is_free(d: date, s_dt: datetime, e_dt: datetime) -> bool:
        # Fast path: prevent exact same start-time collisions
        s_key = s_dt.strftime("%H:%M")
        if s_key in occupied_starts_by_day.get(d, set()):
            return False
        # Robust path: prevent any overlap with already-booked ranges on that day
        for a, b in occupied_ranges_by_day.get(d, []):
            if s_dt < b and e_dt > a:
                return False
        return True

    def mark_slot_used(d: date, s_dt: datetime, e_dt: datetime) -> None:
        occupied_starts_by_day.setdefault(d, set()).add(s_dt.strftime("%H:%M"))
        occupied_ranges_by_day.setdefault(d, []).append((s_dt, e_dt))

    # For each task, allocate minutes into FREE slots before (deadline - buffer)
    for t in tasks_sorted:
        remaining_minutes = int(max(0.0, float(t.estimated_hours)) * 60)
        if remaining_minutes <= 0:
            continue

        deadline_dt = datetime.combine(t.deadline, time(23, 59), tzinfo=tz)
        latest_allowed = deadline_dt - buffer_delta

        start_day = max(month_start, now_dt.date())
        end_day = min(month_end_excl - timedelta(days=1), latest_allowed.date())

        if end_day < start_day:
            unscheduled.append(
                {
                    "task_id": t.task_id,
                    "course": t.course,
                    "title": t.title,
                    "reason": "חלון הזמן האפשרי לפני הדדליין (כולל buffer) אינו נמצא בחודש הנבחר.",
                    "remaining_hours": round(remaining_minutes / 60.0, 2),
                }
            )
            continue

        # Greedy fill with light load-balancing:
        # iterate days in chronological order, but skip days too close to daily max.
        for d in daterange(start_day, end_day + timedelta(days=1)):
            if remaining_minutes <= 0:
                break

            slots = day_slots.get(d, [])
            if not slots:
                continue

            day_load_ratio = used_minutes_by_day.get(d, 0) / max(1, daily_max_minutes)
            if day_load_ratio >= 0.95:
                continue

            for s_dt, e_dt in slots:
                if remaining_minutes <= 0:
                    break

                slot_len = int((e_dt - s_dt).total_seconds() // 60)
                if slot_len <= 0:
                    continue

                # If remaining work is less than a full slot, optionally allow a shorter final slot
                # by trimming the end time (keeps "progression" and reduces over-allocation).
                alloc_minutes = min(slot_len, remaining_minutes)
                if alloc_minutes <= 0:
                    continue

                # Enforce per-day and per-task-per-day caps
                if not day_has_capacity(d, alloc_minutes):
                    continue
                if not task_day_has_capacity(t.task_id, d, alloc_minutes):
                    continue

                s_alloc = s_dt
                e_alloc = s_dt + timedelta(minutes=alloc_minutes)

                # Occupancy check (prevents overlaps and repeated 08:00)
                if not slot_is_free(d, s_alloc, e_alloc):
                    continue

                title = f"{t.course} | {t.title}".strip(" |")
                desc = (
                    f"משימת לימודים מתוכננת.\n"
                    f"דדליין: {t.deadline.isoformat()}\n"
                    f"עדיפות: {t.priority}\n"
                    f"{t.notes}"
                ).strip()

                events.append(Event(title=title, start_dt=s_alloc, end_dt=e_alloc, description=desc))

                used_minutes_by_day[d] = used_minutes_by_day.get(d, 0) + alloc_minutes
                key = (t.task_id, d)
                used_minutes_by_task_day[key] = used_minutes_by_task_day.get(key, 0) + alloc_minutes
                remaining_minutes -= alloc_minutes

                mark_slot_used(d, s_alloc, e_alloc)

        if remaining_minutes > 0:
            unscheduled.append(
                {
                    "task_id": t.task_id,
                    "course": t.course,
                    "title": t.title,
                    "reason": "אין מספיק משבצות פנויות בחלונות הזמן והאילוצים שנבחרו.",
                    "remaining_hours": round(remaining_minutes / 60.0, 2),
                }
            )

    validation = validate_schedule(events, tz, daily_max_minutes, weekday_blocks_t, date_blocks_t)
    report = {
        "tz": tz_name,
        "month": f"{year:04d}-{month:02d}",
        "daily_max_hours": float(daily_max_hours),
        "work_start": work_start_hhmm,
        "work_end": work_end_hhmm,
        "slot_minutes": int(slot_minutes),
        "max_task_hours_per_day": float(max_task_hours_per_day),
        "buffer_hours": int(buffer_hours),
        "unscheduled": unscheduled,
        "validation": validation,
        "events_count": len(events),
    }

    events.sort(key=lambda ev: ev.start_dt)
    return events, report


def validate_schedule(
    events: List[Event],
    tz: ZoneInfo,
    daily_max_minutes: int,
    weekday_blocks: Dict[int, List[Tuple[time, time]]],
    date_blocks: Dict[date, List[Tuple[time, time]]],
) -> Dict:
    overlaps = []
    blocked_hits = []
    daily_minutes: Dict[date, int] = {}

    events_sorted = sorted(events, key=lambda e: (e.start_dt, e.end_dt))

    # Overlaps
    for i in range(1, len(events_sorted)):
        prev = events_sorted[i - 1]
        cur = events_sorted[i]
        if cur.start_dt < prev.end_dt:
            overlaps.append(
                {
                    "prev": {"title": prev.title, "start": prev.start_dt.isoformat(), "end": prev.end_dt.isoformat()},
                    "cur": {"title": cur.title, "start": cur.start_dt.isoformat(), "end": cur.end_dt.isoformat()},
                }
            )

    # Daily max
    for ev in events_sorted:
        d = ev.start_dt.date()
        mins = int((ev.end_dt - ev.start_dt).total_seconds() // 60)
        daily_minutes[d] = daily_minutes.get(d, 0) + max(0, mins)

    daily_exceed = []
    for d, mins in daily_minutes.items():
        if mins > daily_max_minutes:
            daily_exceed.append({"date": d.isoformat(), "minutes": mins, "limit": daily_max_minutes})

    # Block violations
    for ev in events_sorted:
        d = ev.start_dt.date()
        wd = d.weekday()

        for bs, be in weekday_blocks.get(wd, []):
            bsd = datetime.combine(d, bs, tzinfo=tz)
            bed = datetime.combine(d, be, tzinfo=tz)
            if ev.start_dt < bed and ev.end_dt > bsd:
                blocked_hits.append(
                    {
                        "event": {"title": ev.title, "start": ev.start_dt.isoformat(), "end": ev.end_dt.isoformat()},
                        "block": {"type": "weekday", "weekday": wd, "start": bs.isoformat(), "end": be.isoformat()},
                    }
                )

        for bs, be in date_blocks.get(d, []):
            bsd = datetime.combine(d, bs, tzinfo=tz)
            bed = datetime.combine(d, be, tzinfo=tz)
            if ev.start_dt < bed and ev.end_dt > bsd:
                blocked_hits.append(
                    {
                        "event": {"title": ev.title, "start": ev.start_dt.isoformat(), "end": ev.end_dt.isoformat()},
                        "block": {"type": "date", "date": d.isoformat(), "start": bs.isoformat(), "end": be.isoformat()},
                    }
                )

    return {
        "overlaps_count": len(overlaps),
        "overlaps": overlaps[:10],
        "daily_exceed_count": len(daily_exceed),
        "daily_exceed": daily_exceed[:10],
        "blocked_hits_count": len(blocked_hits),
        "blocked_hits": blocked_hits[:10],
        "ok": (len(overlaps) == 0 and len(daily_exceed) == 0 and len(blocked_hits) == 0),
    }

# =========================
# ICS Export
# =========================
def ics_escape(text: str) -> str:
    if text is None:
        return ""
    # RFC5545 basic escaping
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def to_ics(events: List[Event], tz_name: str, cal_name: str = "EduPlanner") -> str:
    tz = ZoneInfo(tz_name)
    dtstamp = datetime.now(tz=tz).strftime("%Y%m%dT%H%M%S")
    lines = []
    lines.append("BEGIN:VCALENDAR")
    lines.append("VERSION:2.0")
    lines.append("PRODID:-//EduPlanner//Streamlit//HE")
    lines.append(f"X-WR-CALNAME:{ics_escape(cal_name)}")
    lines.append("CALSCALE:GREGORIAN")
    lines.append("METHOD:PUBLISH")

    for idx, ev in enumerate(events):
        uid = f"eduplanner-{idx}-{int(ev.start_dt.timestamp())}@local"
        dtstart = ev.start_dt.strftime("%Y%m%dT%H%M%S")
        dtend = ev.end_dt.strftime("%Y%m%dT%H%M%S")

        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{uid}")
        lines.append(f"DTSTAMP:{dtstamp}")
        lines.append(f"SUMMARY:{ics_escape(ev.title)}")
        if ev.description:
            lines.append(f"DESCRIPTION:{ics_escape(ev.description)}")
        lines.append(f"DTSTART;TZID={tz_name}:{dtstart}")
        lines.append(f"DTEND;TZID={tz_name}:{dtend}")
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    return "\n".join(lines)


# =========================
# Optional: AI Helpers (Safe Stubs)
# =========================
def coerce_date_series_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col]

    # אם כבר datetime, לא נוגעים
    if pd.api.types.is_datetime64_any_dtype(s):
        return df

    # אם אובייקטים של date, ננסה להמיר
    try:
        df[col] = pd.to_datetime(s, errors="coerce")
        return df
    except Exception:
        pass

    # אם מחרוזות, ננסה שני פורמטים בצורה דטרמיניסטית
    def _parse(x):
        if x is None or str(x).strip() == "":
            return pd.NaT
        x = str(x).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(x, fmt)
            except Exception:
                continue
        return pd.NaT

    df[col] = s.apply(_parse)
    return df

def try_ai_parse_tasks(free_text: str) -> List[Task]:
    """
    Optional: parse tasks from free text.
    Conservative + deterministic, no external model calls.

    Supports date formats:
      - YYYY-MM-DD
      - DD/MM/YYYY

    Example lines:
      - "ביולוגיה | עבודה סמינריונית | 2026-03-10 | 12 | עדיפות 5"
      - "ביולוגיה | עבודה סמינריונית | 10/03/2026 | 12 | עדיפות 5"
    """
    tasks: List[Task] = []
    if not free_text or not free_text.strip():
        return tasks

    for i, line in enumerate(free_text.splitlines(), start=1):
        line = (line or "").strip()
        if not line:
            continue

        parts = [p.strip() for p in re.split(r"[|,]", line) if p.strip()]
        if len(parts) < 3:
            continue

        course = parts[0]
        title = parts[1]

        # Find a date in either ISO or EU format
        date_str = None
        m_iso = re.search(r"(\d{4}-\d{2}-\d{2})", line)
        m_eu = re.search(r"(\d{2}/\d{2}/\d{4})", line)

        if m_iso:
            date_str = m_iso.group(1)
        elif m_eu:
            date_str = m_eu.group(1)

        if not date_str:
            continue

        try:
            dl = parse_date_any(date_str)
        except Exception:
            continue

        # Find hours (accepts: "12", "12.5", "שעות: 12", "hours=12")
        h_match = re.search(
            r"(?:שעות|hrs|hours)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        est = float(h_match.group(1)) if h_match else 3.0

        # Find priority 1..5 (accepts: "עדיפות 4", "priority:5")
        p_match = re.search(
            r"(?:עדיפות|priority)\s*[:=]?\s*([1-5])",
            line,
            flags=re.IGNORECASE,
        )
        pr = int(p_match.group(1)) if p_match else 3

        # Basic sanity clamps
        est = max(0.0, float(est))
        pr = int(clamp_float(pr, 1, 5))

        tasks.append(
            Task(
                task_id=f"T{i}",
                course=course,
                title=title,
                deadline=dl,
                estimated_hours=est,
                priority=pr,
                notes="נוצר מהזנה חופשית (פיענוח בסיסי).",
            )
        )

    return tasks

# =========================
# UI
# =========================
st.set_page_config(page_title="EduPlanner", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    /* Data Editor container */
    [data-testid="stDataEditor"], 
    [data-testid="stDataEditor"] * {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Header cells */
    [data-testid="stDataEditor"] [role="columnheader"] {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext;
    }

    /* Cell content */
    [data-testid="stDataEditor"] [role="gridcell"] {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext;
    }

    /* Keep code blocks and JSON LTR */
    pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {
        direction: ltr !important;
        text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Minimal styling to keep your dark, clean feel
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
    .title-row { display:flex; align-items:center; gap: 0.6rem; }
    .hint { opacity: 0.85; }
    .small { font-size: 0.9rem; opacity: 0.9; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='title-row'><h1>{APP_TITLE}</h1></div>", unsafe_allow_html=True)
st.caption("הזנת מטלות, חישוב לו״ז חודשי דטרמיניסטי, וייצוא לקובץ ICS ליומן Google. עיצוב נקי, תכנון אמין.")


# -------------------------
# Sidebar: system settings
# -------------------------
st.sidebar.header("הגדרות מערכת ⚙️")

# טאבים בסיידבר, יותר יציב מאשר רשימה ארוכה
tab_basic, tab_rules, tab_reset = st.sidebar.tabs(["בסיסי", "שיבוץ", "איפוס"])

with tab_basic:
    tz_name = st.text_input("אזור זמן (TZID)", value=DEFAULT_TZ)

    year = st.selectbox("בחר שנת לימודים", options=list(range(2024, 2031)), index=list(range(2024, 2031)).index(2026) if 2026 in range(2024, 2031) else 0)
    month = st.selectbox("בחר חודש לתכנון", options=list(range(1, 13)), index=datetime.now().month - 1)

    st.subheader("תכנון זמן יומי")
    c1, c2 = st.columns(2)

    with c1:
        daily_max_hours = st.number_input(
            "כמה שעות מקסימליות ביום?",
            min_value=1.0,
            max_value=12.0,
            value=float(DEFAULT_DAILY_MAX_HOURS),
            step=0.5
        )

        # time_input מחזיר datetime.time, זה חוסך שגיאות HH:MM
        workday_start_t = st.time_input("מתי להתחיל את היום?", value=parse_hhmm(DEFAULT_WORKDAY_START))

    with c2:
        max_task_hours_per_day = st.number_input(
            "מקסימום שעות לאותה מטלה ביום",
            min_value=1.0,
            max_value=6.0,
            value=float(DEFAULT_MAX_TASK_HOURS_PER_DAY),
            step=0.5
        )
        workday_end_t = st.time_input("מתי לסיים את היום?", value=parse_hhmm(DEFAULT_WORKDAY_END))

    # המרה חזרה ל־HH:MM עבור מנוע השיבוץ
    workday_start = workday_start_t.strftime("%H:%M")
    workday_end = workday_end_t.strftime("%H:%M")

with tab_rules:
    st.subheader("רזולוציית שיבוץ ומרווח ביטחון")

    r1, r2 = st.columns(2)
    with r1:
        slot_minutes = st.select_slider("גודל משבצת (דקות)", options=[30, 45, 60, 90, 120], value=DEFAULT_SLOT_MINUTES)
    with r2:
        buffer_hours = st.select_slider("מרווח ביטחון לפני דדליין (שעות)", options=[24, 36, 48, 72], value=DEFAULT_BUFFER_HOURS)

    st.caption("המלצה פרקטית: משבצת 60 דקות + Buffer של 48 שעות היא ברירת מחדל טובה לרוב הסטודנטים.")

with tab_reset:
    st.subheader("ניקוי נתונים")
    if st.button("🧹 ניקוי כל הנתונים", type="secondary"):
        for k in ["tasks_df", "weekday_blocks_df", "date_blocks_df", "events", "report"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # אופציה אגרסיבית, אם אתה רוצה למחוק הכל כולל כל מפתח פנימי
    if st.button("🧨 ניקוי מוחלט (כולל הכל)", type="secondary"):
        st.session_state.clear()
        st.rerun()


# -------------------------
# Main: Task input
# -------------------------
# ודא שברירת מחדל קיימת לפני הכל
ensure_session_defaults(year, month)

# התאמת טיפוסי תאריך לפני data_editor (כדי ש-DateColumn לא יקרוס)
st.session_state["tasks_df"] = coerce_dates_for_editor(st.session_state["tasks_df"], "deadline")

with st.form("planner_form", clear_on_submit=False):

    st.markdown("## הזנת מטלות 📝")
    st.info("אפשר להזין בטבלה או להדביק טקסט חופשי. השמירה מתבצעת רק בלחיצה על הכפתורים למטה.", icon="💡")

    edited_tasks_df = st.data_editor(
        st.session_state["tasks_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "task_id": st.column_config.TextColumn("מזהה"),
            "course": st.column_config.TextColumn("שם הקורס"),
            "title": st.column_config.TextColumn("שם המטלה"),
            "deadline": st.column_config.DateColumn("דדליין", format="DD/MM/YYYY"),
            "estimated_hours": st.column_config.NumberColumn("שעות משוערות", min_value=0.0, step=0.5),
            "priority": st.column_config.NumberColumn("עדיפות 1–5", min_value=1, max_value=5, step=1),
            "notes": st.column_config.TextColumn("הערות"),
        },
        key="w_tasks_editor_main",
    )

    st.markdown("### הדבקת מטלות בטקסט חופשי (אופציונלי)")
    st.caption("דוגמה: קורס | מטלה | 01/02/2026 | 12 | עדיפות 5")
    free_text = st.text_area(
        "הדבק כאן",
        height=120,
        placeholder="לדוגמה:\nביולוגיה | עבודה סמינריונית | 01/02/2026 | 12 | עדיפות 5",
        key="free_text_tasks",
    )

    col1, col2 = st.columns(2)
    with col1:
        save_clicked = st.form_submit_button("💾 שמור נתונים")
    with col2:
        compute_clicked = st.form_submit_button("🚀 שמור וחשב לו״ז", type="primary")

# commit אחרי submit
if save_clicked or compute_clicked:
    st.session_state["tasks_df"] = edited_tasks_df

    # הוספת מטלות מטקסט חופשי לאחר submit בלבד
    txt = (st.session_state.get("free_text_tasks") or "").strip()
    if txt:
        parsed = try_ai_parse_tasks(txt)
        if parsed:
            add_df = pd.DataFrame(
                [{
                    "task_id": t.task_id,
                    "course": t.course,
                    "title": t.title,
                    "deadline": pd.Timestamp(datetime.combine(t.deadline, time(0, 0))),
                    "estimated_hours": float(t.estimated_hours),
                    "priority": int(t.priority),
                    "notes": t.notes,
                } for t in parsed]
            )
            st.session_state["tasks_df"] = pd.concat([st.session_state["tasks_df"], add_df], ignore_index=True)
            st.success(f"נוספו {len(parsed)} מטלות מהטקסט.")
        else:
            st.warning("לא הצלחתי לחלץ מטלות מהטקסט. נסה פורמט כמו בדוגמה.")
                
def parse_date_any(s: str) -> date:
    """
    Parses date in either ISO or EU format.

    Supported:
      - YYYY-MM-DD
      - DD/MM/YYYY

    Returns: datetime.date
    Raises: ValueError if cannot parse.
    """
    if s is None:
        raise ValueError("empty date")

    s = str(s).strip()
    if not s:
        raise ValueError("empty date string")

    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue

    raise ValueError(f"Unsupported date format: {s}")

# -------------------------
# Convert UI tables to model inputs
# -------------------------
def _coerce_date_value_to_date(val) -> date:
    """
    Accepts: date, datetime, pd.Timestamp, or string in YYYY-MM-DD / DD/MM/YYYY.
    Returns: datetime.date
    Raises: ValueError if cannot parse.
    """
    if val is None:
        raise ValueError("empty date")

    # pandas may store missing as NaT (which behaves like NaN)
    try:
        if pd.isna(val):
            raise ValueError("missing/NaT date")
    except Exception:
        pass

    # If it's already a date (but not datetime)
    if isinstance(val, date) and not isinstance(val, datetime):
        return val

    # If it's a datetime-like
    if isinstance(val, (datetime, pd.Timestamp)):
        # guard again for NaT just in case
        try:
            if pd.isna(val):
                raise ValueError("missing/NaT datetime")
        except Exception:
            pass
        return val.date()

    # Otherwise treat as string
    s = str(val).strip()
    if not s:
        raise ValueError("empty date string")

    return parse_date_any(s)


def df_to_tasks(df: pd.DataFrame) -> List[Task]:
    tasks: List[Task] = []
    if df is None or df.empty:
        return tasks

    for i, row in df.iterrows():
        task_id = str(row.get("task_id") or f"T{i+1}").strip() or f"T{i+1}"
        course = str(row.get("course") or "").strip()
        title = str(row.get("title") or "").strip()

        # skip empty rows
        if not title and not course:
            continue

        # deadline: must be valid
        dl_val = row.get("deadline")
        try:
            dl = _coerce_date_value_to_date(dl_val)
        except Exception:
            # deadline missing/invalid -> skip task
            continue

        est = safe_float(row.get("estimated_hours"), 0.0)
        pr = safe_int(row.get("priority"), 3)
        pr = int(clamp_float(pr, 1, 5))
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

def df_to_weekday_blocks(df: pd.DataFrame) -> Dict[int, List[Tuple[str, str]]]:
    out: Dict[int, List[Tuple[str, str]]] = {}
    if df is None or df.empty:
        return out

    for _, row in df.iterrows():
        wd_name = str(row.get("weekday") or "").strip()
        if wd_name not in WEEKDAY_NAME_TO_INT:
            continue

        wd = WEEKDAY_NAME_TO_INT[wd_name]
        s = str(row.get("start") or "").strip()
        e = str(row.get("end") or "").strip()

        if not s or not e:
            continue

        out.setdefault(wd, []).append((s, e))

    return out


def df_to_date_blocks(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dict keyed by ISO date string 'YYYY-MM-DD', to remain compatible
    with schedule_tasks which currently parses ds via datetime.strptime(ds, "%Y-%m-%d").
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    if df is None or df.empty:
        return out

    for _, row in df.iterrows():
        ds_val = row.get("date")
        s = str(row.get("start") or "").strip()
        e = str(row.get("end") or "").strip()

        if not s or not e:
            continue

        try:
            d = _coerce_date_value_to_date(ds_val)
        except Exception:
            continue

        ds = d.strftime("%Y-%m-%d")
        out.setdefault(ds, []).append((s, e))

    return out


# =========================
# 7. Compute Schedule
# =========================
st.divider()
st.markdown("## חישוב לו״ז אסטרטגי וחכם 🚀")
st.caption("הליבה כאן היא שיבוץ דטרמיניסטי עם בדיקות תקינות. ה-AI, אם תרצה, יכול לשמש להסברים, לא לחישוב השעות.")

col_a, col_b = st.columns([1, 1])
with col_a:
    compute_clicked = st.button("🚀 חשב לו״ז אסטרטגי וחכם", type="primary")
with col_b:
    explain_clicked = st.button("🧠 צור הסבר והמלצות (אופציונלי)", type="secondary")

if compute_clicked:
    tasks = df_to_tasks(st.session_state.tasks_df)

    if not tasks:
        st.warning("לא נמצאו מטלות תקינות. ודא שיש דדליין בפורמט YYYY-MM-DD ושעות משוערות.")
    else:
        weekday_blocks = df_to_weekday_blocks(st.session_state.weekday_blocks_df)
        date_blocks = df_to_date_blocks(st.session_state.date_blocks_df)

        workday_start_str = workday_start_t.strftime("%H:%M")
        workday_end_str = workday_end_t.strftime("%H:%M")

        if "deadline" in st.session_state["tasks_df"].columns:
            n_missing = int(st.session_state["tasks_df"]["deadline"].isna().sum())
        if n_missing > 0:
            st.error(f"יש {n_missing} מטלות ללא דדליין. מחק שורות ריקות או מלא תאריך ואז נסה שוב.")
            st.stop()


        schedule_params = {
            "tasks": tasks,
            "tz_name": tz_name.strip() or DEFAULT_TZ,
            "year": int(year),
            "month": int(month),
            "work_start_hhmm": workday_start_str,
            "work_end_hhmm": workday_end_str,
            "daily_max_hours": float(daily_max_hours),
            "max_task_hours_per_day": float(max_task_hours_per_day),
            "slot_minutes": int(slot_minutes),
            "buffer_hours": int(buffer_hours),
            "weekday_blocks": weekday_blocks,
            "date_blocks": date_blocks,
        }

        # DEBUG חייב להיות כאן, כי כאן schedule_params קיים
        st.write("DEBUG schedule_params keys:", list(schedule_params.keys()))
        st.write("DEBUG tasks count:", len(schedule_params["tasks"]))

        with st.spinner("המערכת בונה לו״ז חודשי תוך כיבוד אילוצים ועומסים..."):
            try:
                events, report = schedule_tasks(**schedule_params)
                st.session_state.events = events
                st.session_state.report = report
                st.success(f"הלו״ז הושלם בהצלחה. נוצרו {len(events)} משבצות עבודה.")
            except Exception as e:
                st.error("שגיאה בקריאה ל-schedule_tasks. פירוט מלא:")
                st.exception(e)

# =========================
# Display schedule + export
# =========================
if "events" in st.session_state and st.session_state.events:
    st.markdown("## תוצרים 📅")

    events: List[Event] = st.session_state.events
    tz = ZoneInfo(st.session_state.report.get("tz", DEFAULT_TZ))

    # Build display table
    rows = []
    for ev in events:
        d = ev.start_dt.date().isoformat()
        start_s = ev.start_dt.strftime("%H:%M")
        end_s = ev.end_dt.strftime("%H:%M")
        hours = round((ev.end_dt - ev.start_dt).total_seconds() / 3600.0, 2)

        # Split title into course and task if possible
        course, task_title = "", ev.title
        if "|" in ev.title:
            parts = [p.strip() for p in ev.title.split("|", 1)]
            course = parts[0]
            task_title = parts[1] if len(parts) > 1 else ev.title

        rows.append(
            {
                "תאריך": d,
                "קורס והמטלה": (course + " | " + task_title).strip(" |"),
                "שלב/תת-משימה": "עבודה ממוקדת (משבצת)",
                "שעות שיבוץ": hours,
                "שעות התחלה וסיום משוערות": f"{start_s}–{end_s}",
            }
        )

    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # Report and validation
    report = st.session_state.report
    val = report.get("validation", {})
    if val and not val.get("ok", True):
        st.warning("נמצאו בעיות תקינות. מומלץ לעדכן אילוצים או הגדרות זמן ולחשב מחדש.")
        st.json(val)
    else:
        st.success("בדיקות תקינות עברו בהצלחה. אין חפיפות, אין חריגה ממגבלת שעות, ואין שיבוץ בתוך חסמים.")

    # Unscheduled tasks
    uns = report.get("unscheduled", [])
    if uns:
        st.error("יש מטלות שלא שובצו במלואן. הסיבות מפורטות כאן:")
        st.dataframe(pd.DataFrame(uns), use_container_width=True, hide_index=True)

    # Build JSON for downstream use (clean, deterministic)
    json_events = []
    for ev in events:
        json_events.append(
            {
                "title": ev.title,
                "date": ev.start_dt.date().isoformat(),
                "start_time": ev.start_dt.strftime("%H:%M"),
                "end_time": ev.end_dt.strftime("%H:%M"),
                "timezone": report.get("tz", DEFAULT_TZ),
            }
        )

    with st.expander("בלוק JSON לייצוא / בדיקה", expanded=False):
        st.code(json.dumps(json_events, ensure_ascii=False, indent=2), language="json")

    # ICS download
    ics_text = to_ics(events, tz_name=report.get("tz", DEFAULT_TZ), cal_name=f"EduPlanner {report.get('month','')}")
    st.download_button(
        "⬇️ הורד קובץ ICS (Google Calendar)",
        data=ics_text.encode("utf-8"),
        file_name=f"eduplanner_{report.get('month','month')}.ics",
        mime="text/calendar",
    )

else:
    st.caption("לאחר החישוב, תופיע כאן טבלת שיבוץ ויכולת ייצוא לקובץ ICS.")


# =========================
# Optional: Explanation block (placeholder for AI)
# =========================
if explain_clicked:
    if "events" not in st.session_state or not st.session_state.events:
        st.warning("כדי להפיק הסבר, חשב קודם לו״ז.")
    else:
        report = st.session_state.report
        events = st.session_state.events

        # Deterministic explanation (no AI). You can replace with AI later, safely, since schedule is fixed.
        total_hours = sum((ev.end_dt - ev.start_dt).total_seconds() / 3600.0 for ev in events)
        days = sorted({ev.start_dt.date() for ev in events})
        busiest = None
        day_load = {}
        for ev in events:
            d = ev.start_dt.date()
            day_load[d] = day_load.get(d, 0.0) + (ev.end_dt - ev.start_dt).total_seconds() / 3600.0
        if day_load:
            busiest = max(day_load.items(), key=lambda x: x[1])

        st.markdown("## הסבר והמלצות 🧠")
        st.write(
            f"""
            **תמונת מצב כללית**
            1. מספר משבצות: {len(events)}
            2. סך שעות מתוכננות: {total_hours:.2f}
            3. ימים עם עבודה מתוכננת: {len(days)}
            4. מרווח ביטחון לפני דדליין (buffer): {report.get('buffer_hours')} שעות
            5. גודל משבצת: {report.get('slot_minutes')} דקות
            """
        )

        if busiest:
            st.write(f"**היום העמוס ביותר**: {busiest[0].isoformat()} עם {busiest[1]:.2f} שעות.")

        if report.get("unscheduled"):
            st.write(
                """
                **מה המשמעות של מטלות שלא שובצו**
                לרוב זה מצביע על אחד מאלה:
                1. מגבלת שעות יומית נמוכה מדי ביחס לעומס.
                2. חלון העבודה קצר מדי (שעת התחלה מאוחרת או סיום מוקדם).
                3. חסמים רבים מדי.
                4. דדליין קרוב מדי ביחס ל-buffer.
                """
            )

        st.write(
            """
            **המלצות לשיפור איכות התכנון**
            1. הגדל חלון עבודה יומי, לפני שמגדילים את מגבלת השעות. זה משפר גמישות בלי להעמיס מייד.
            2. אם יש שחיקה, הקטן את מגבלת השעות, אך הגדל מספר ימים מתוכננים, באמצעות הקדמת התחלה או צמצום חסמים.
            3. אם ההערכות לשעות אינן יציבות, עדיף להגדיר אומדן שמרני, ואז לקצר בפועל.
            4. בדוק האם buffer של 48 שעות מתאים לכל מטלה, או שצריך להפוך אותו לתלוי גודל מטלה.
            """
        )

st.divider()
st.caption("EduPlanner, גרסה דטרמיניסטית: תכנון זמן אמין, ואז שפה והמלצות. כך מתקבלים שיבוצים רלוונטיים ושקופים.")