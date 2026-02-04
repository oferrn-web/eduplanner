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

def generate_start_times(
    windows: List[Tuple[datetime, datetime]],
    step_minutes: int
) -> List[datetime]:
    """
    Returns list of possible start datetimes, spaced by step_minutes, within each window.
    """
    step_minutes = int(step_minutes) if step_minutes else 15
    step_minutes = max(5, min(60, step_minutes))

    starts: List[datetime] = []
    step = timedelta(minutes=step_minutes)

    for ws, we in windows:
        cur = ws
        # generate start points up to window end (actual end check happens when booking)
        while cur < we:
            starts.append(cur)
            cur = cur + step
    return starts

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
    start_step_minutes: int,
    buffer_hours: int,
    weekday_blocks: Dict[int, List[Tuple[str, str]]],
    date_blocks: Dict[str, List[Tuple[str, str]]],
    break_minutes: int = 15,
    max_continuous_minutes: int = 120,
    policy: dict | None = None,
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
    # ALWAYS initialize these early (avoid UnboundLocalError due to indentation/branches)
    events: List[Event] = []
    unscheduled: List[Dict] = []

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

    day_windows: Dict[date, List[Tuple[datetime, datetime]]] = {}
    day_slots: Dict[date, List[datetime]] = {}

    for d in daterange(month_start, month_end_excl):
        windows = build_daily_available_windows(d, tz, work_start, work_end, weekday_blocks_t, date_blocks_t)
        START_STEP_MINUTES = 15
        starts = generate_start_times(windows, START_STEP_MINUTES)
        day_slots[d] = starts
        day_windows[d] = windows
    
    daily_max_minutes = int(daily_max_hours * 60)
    max_task_minutes_per_day = int(max_task_hours_per_day * 60)
    buffer_delta = timedelta(hours=buffer_hours)
    
    break_delta = timedelta(minutes=int(max(0, break_minutes)))
    max_continuous = int(max(0, max_continuous_minutes))

    # -------------------------
    # Optional: add constraints as calendar events
    # -------------------------
    constraint_events: List[Event] = []

    def _add_constraint_event(d: date, bs: time, be: time, label: str, kind: str):
        # guard
        if bs >= be:
            return
        sdt = datetime.combine(d, bs, tzinfo=tz)
        edt = datetime.combine(d, be, tzinfo=tz)
        title = f"⛔ חסם: {label}".strip()
        desc = f"חסם ({kind})."
        constraint_events.append(Event(title=title, start_dt=sdt, end_dt=edt, description=desc))

    # 1) Weekly blocks -> expand to concrete dates in the selected month
    for d in daterange(month_start, month_end_excl):
        wd = d.weekday()
        for (bs, be) in weekday_blocks_t.get(wd, []):
            _add_constraint_event(d, bs, be, "אילוץ שבועי", "weekday")

    # 2) Date-specific blocks
    for d, blocks in date_blocks_t.items():
        if month_start <= d < month_end_excl:
            for (bs, be) in blocks:
                _add_constraint_event(d, bs, be, "אילוץ בתאריך", "date")

    # Merge: include constraint events in output (so they export to ICS)
    events.extend(constraint_events)

    def reorder_slots_spread(starts: List[datetime]) -> List[datetime]:
        """
        Reorders start-times to encourage within-day spreading:
        alternate afternoon and morning starts, so afternoons get used.
        """
        if not starts:
            return starts

        # Ensure deterministic order before splitting
        starts_sorted = sorted(starts)

        day = starts_sorted[0].date()
        midday = datetime.combine(day, time(13, 0), tzinfo=tz)  # אפשר לשנות 13:00 לפי צורך

        morning = [s for s in starts_sorted if s < midday]
        afternoon = [s for s in starts_sorted if s >= midday]

        out: List[datetime] = []
        i = j = 0

        # Start alternating side per date to balance: even days -> morning first, odd -> afternoon first
        take_afternoon = (day.day % 2 == 1)

        while i < len(morning) or j < len(afternoon):
            if take_afternoon and j < len(afternoon):
                out.append(afternoon[j]); j += 1
            elif (not take_afternoon) and i < len(morning):
                out.append(morning[i]); i += 1
            else:
                if i < len(morning):
                    out.append(morning[i]); i += 1
                elif j < len(afternoon):
                    out.append(afternoon[j]); j += 1
            take_afternoon = not take_afternoon

        return out


    # reorder each day's slots once
    for d in list(day_slots.keys()):
        day_slots[d] = reorder_slots_spread(day_slots[d])

    # Track last event end per day (for enforcing breaks)
    last_end_by_day: Dict[date, datetime] = {}

    # Track continuous work minutes since last real break
    continuous_minutes_by_day: Dict[date, int] = {d: 0 for d in day_slots.keys()}

    used_minutes_by_day: Dict[date, int] = {d: 0 for d in day_slots.keys()}
    used_minutes_by_task_day: Dict[Tuple[str, date], int] = {}

    # Occupancy: prevent reusing the same start time for multiple events on the same date
    # and optionally track (start_dt, end_dt) to guard against non-uniform slot sizes.
    occupied_starts_by_day: Dict[date, set] = {d: set() for d in day_slots.keys()}
    occupied_ranges_by_day: Dict[date, List[Tuple[datetime, datetime]]] = {d: [] for d in day_slots.keys()}

    events: List[Event] = []
    unscheduled: List[Dict] = []

    # -------------------------
    # Add constraints as calendar events (so they export to ICS)
    # IMPORTANT: must be after "events = []"
    # -------------------------
    constraint_events: List[Event] = []

    def _add_constraint_event(d: date, bs: time, be: time, label: str, kind: str):
        if bs >= be:
            return
        sdt = datetime.combine(d, bs, tzinfo=tz)
        edt = datetime.combine(d, be, tzinfo=tz)
        title = f"⛔ חסם: {label}".strip()
        desc = f"חסם ({kind})."
        constraint_events.append(Event(title=title, start_dt=sdt, end_dt=edt, description=desc))

    # 1) Weekly blocks -> expand to concrete dates in month
    # אם אין לך label במבנה, תן label כללי
    for d in daterange(month_start, month_end_excl):
        wd = d.weekday()
        for block in weekday_blocks_t.get(wd, []):
            # תומך גם ב-(time,time) וגם ב-(time,time,label)
            if len(block) == 2:
                bs, be = block
                label = "אילוץ שבועי"
            else:
                bs, be, label = block
            _add_constraint_event(d, bs, be, label, "weekday")

    # 2) Date-specific blocks
    for d, blocks in date_blocks_t.items():
        if month_start <= d < month_end_excl:
            for block in blocks:
                if len(block) == 2:
                    bs, be = block
                    label = "אילוץ בתאריך"
                else:
                    bs, be, label = block
                _add_constraint_event(d, bs, be, label, "date")

    # Merge
    events.extend(constraint_events)

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

    # =========================
    # Round-Robin (weighted) Scheduling Core + Policy-based slot scoring
    # Replace the old: "for t in tasks_sorted: ..." block with this entire block
    # =========================

    pol = policy or {}
    pol_hard = (pol.get("hard") or {})
    pol_soft = (pol.get("soft") or {})

    # Pull hard overrides (fallback to current parameters if not present)
    break_minutes = int(pol_hard.get("break_minutes") or 0)
    max_continuous_minutes = int(pol_hard.get("max_continuous_minutes") or 0)

    avoid_weekdays = set(pol_hard.get("avoid_weekdays") or [])  # Hebrew names

    # Soft preferences
    prefer_windows = pol_soft.get("prefer_time_windows") or []
    target_daily_load_ratio = float(pol_soft.get("target_daily_load_ratio") or 0.75)
    target_daily_load_ratio = max(0.5, min(0.9, target_daily_load_ratio))
    midday_hhmm = str(pol_soft.get("midday_hhmm") or "13:00").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", midday_hhmm):
        midday_hhmm = "13:00"
    midday_t = parse_hhmm(midday_hhmm)

    break_delta = timedelta(minutes=max(0, break_minutes))

    # Track last end and continuous work per day (for breaks and max continuous)
    last_end_by_day: Dict[date, datetime] = {}
    continuous_minutes_by_day: Dict[date, int] = {d: 0 for d in day_slots.keys()}

    # Remaining minutes per task_id
    remaining_by_task: Dict[str, int] = {}
    task_map: Dict[str, Task] = {t.task_id: t for t in tasks_sorted}

    for t in tasks_sorted:
        remaining_by_task[t.task_id] = int(max(0.0, float(t.estimated_hours)) * 60)

    def _he_weekday_name(d: date) -> str:
        # Python weekday(): Mon=0..Sun=6. Your WEEKDAYS_HE likely starts Sunday.
        # We'll map explicitly:
        # 0 Mon,1 Tue,2 Wed,3 Thu,4 Fri,5 Sat,6 Sun
        m = {
            0: "שני",
            1: "שלישי",
            2: "רביעי",
            3: "חמישי",
            4: "שישי",
            5: "שבת",
            6: "ראשון",
        }
        return m.get(d.weekday(), "")

    def _time_weight(s_dt: datetime) -> float:
        # Soft preference windows weight, sum of matching windows
        w = 0.0
        st = s_dt.timetz().replace(tzinfo=None)
        for win in prefer_windows:
            ws = str(win.get("start") or "").strip()
            we = str(win.get("end") or "").strip()
            if not (re.fullmatch(r"\d{2}:\d{2}", ws) and re.fullmatch(r"\d{2}:\d{2}", we)):
                continue
            try:
                ws_t = parse_hhmm(ws)
                we_t = parse_hhmm(we)
            except Exception:
                continue
            if ws_t <= st < we_t:
                try:
                    ww = float(win.get("weight"))
                except Exception:
                    ww = 0.0
                w += max(0.0, min(1.0, ww))
        # If no windows defined, neutral weight
        return w if prefer_windows else 0.3

    def _day_balance_score(d: date) -> float:
        # Prefer days under target load
        target = int(daily_max_minutes * target_daily_load_ratio)
        used = used_minutes_by_day.get(d, 0)
        if target <= 0:
            return 0.0
        return max(0.0, (target - used) / target)

    def _slot_score(task_id: str, d: date, s_dt: datetime, e_dt: datetime) -> float:
        # Hard filters are applied outside. This is soft score only.
        score = 0.0

        # 1) daily load balancing
        score += 1.5 * _day_balance_score(d)

        # 2) prefer time windows (user energy)
        score += 1.0 * _time_weight(s_dt)

        # 3) encourage spread within day (bigger gap since last event is better)
        prev_end = last_end_by_day.get(d)
        if prev_end is not None:
            gap_m = (s_dt - prev_end).total_seconds() / 60.0
            score += min(1.0, max(0.0, gap_m / 60.0))  # up to +1 for 60m gap
        else:
            # If no previous event, mildly prefer alternating morning/afternoon by date parity
            st = s_dt.timetz().replace(tzinfo=None)
            is_afternoon = (st >= midday_t)
            if (d.day % 2 == 0 and not is_afternoon) or (d.day % 2 == 1 and is_afternoon):
                score += 0.4

        return score

    def _hard_ok(task: Task, d: date, s_dt: datetime, e_dt: datetime, slot_len: int) -> bool:
        # avoid weekdays hard rule
        if _he_weekday_name(d) in avoid_weekdays:
            return False

        # capacity checks
        if not day_has_capacity(d, slot_len):
            return False
        if not task_day_has_capacity(task.task_id, d, slot_len):
            return False

        # break between events in same day
        prev_end = last_end_by_day.get(d)
        if prev_end is not None and s_dt < (prev_end + break_delta):
            return False

        # max continuous work
        if max_continuous_minutes > 0 and continuous_minutes_by_day.get(d, 0) >= max_continuous_minutes:
            return False

        return True

    def _book(task: Task, d: date, s_dt: datetime, e_dt: datetime, slot_len: int):
        title = f"{task.course} | {task.title}".strip(" |")
        desc = f"משימת לימודים מתוכננת.\nדדליין: {task.deadline.strftime('%d/%m/%Y')}\nעדיפות: {task.priority}\n{task.notes}".strip()

        events.append(Event(title=title, start_dt=s_dt, end_dt=e_dt, description=desc))

        used_minutes_by_day[d] = used_minutes_by_day.get(d, 0) + slot_len
        key = (task.task_id, d)
        used_minutes_by_task_day[key] = used_minutes_by_task_day.get(key, 0) + slot_len

        # update continuous + last_end
        prev_end = last_end_by_day.get(d)
        if prev_end is None:
            continuous_minutes_by_day[d] = slot_len
        else:
            gap = s_dt - prev_end
            if gap >= break_delta:
                continuous_minutes_by_day[d] = slot_len
            else:
                continuous_minutes_by_day[d] = continuous_minutes_by_day.get(d, 0) + slot_len
        last_end_by_day[d] = e_dt

    # Build feasible horizon per task (deadline - buffer)
    latest_allowed_by_task: Dict[str, date] = {}
    for t in tasks_sorted:
        deadline_dt = datetime.combine(t.deadline, time(23, 59), tzinfo=tz)
        latest_allowed = deadline_dt - buffer_delta
        latest_allowed_by_task[t.task_id] = latest_allowed.date()

    # Candidate days for the month (from today/month_start to month_end)
    start_day_global = max(month_start, now_dt.date())
    end_day_global = month_end_excl - timedelta(days=1)

    # A bounded loop to prevent infinite loops
    guard_iters = 0
    max_iters = 200000

    # Weighted Round-Robin pointer
    rr_order = [t.task_id for t in tasks_sorted if remaining_by_task.get(t.task_id, 0) > 0]
    rr_idx = 0

    while True:
        guard_iters += 1
        if guard_iters > max_iters:
            break

        # stop condition: all done
        active = [tid for tid, rem in remaining_by_task.items() if rem > 0]
        if not active:
            break

        if not rr_order:
            rr_order = active
            rr_idx = 0

        # pick next task in RR order, but skip completed
        tid = rr_order[rr_idx % len(rr_order)]
        rr_idx += 1
        if remaining_by_task.get(tid, 0) <= 0:
            continue

        task = task_map[tid]
        latest_allowed_day = min(end_day_global, latest_allowed_by_task.get(tid, end_day_global))

        # If no feasible days for this task in month, mark unscheduled and zero it out
        if latest_allowed_day < start_day_global:
            unscheduled.append({
                "task_id": task.task_id,
                "course": task.course,
                "title": task.title,
                "reason": "חלון הזמן האפשרי לפני הדדליין (כולל buffer) אינו נמצא בחודש הנבחר.",
                "remaining_hours": round(remaining_by_task[tid] / 60.0, 2),
            })
            remaining_by_task[tid] = 0
            continue

        # Find best slot (max score) among feasible days/slots
        best = None  # (score, d, s_dt, e_dt, slot_len)
        for d in daterange(start_day_global, latest_allowed_day + timedelta(days=1)):
            starts = day_slots.get(d, [])
            if not starts:
                continue

            # quick skip: if day already at cap
            if used_minutes_by_day.get(d, 0) >= daily_max_minutes:
                continue

            for s_dt in starts:
                needed = remaining_by_task[tid]
                if needed <= 0:
                    break

                # זמן עבודה רציף = אורך בלוק עבודה
                work_block_minutes = int(pol_hard.get("max_continuous_minutes") or 120)
                min_block_minutes = 60  # אפשר לשנות ל-45 אם אתה רוצה
                needed = remaining_by_task[tid]

                alloc = min(work_block_minutes, needed)

                # אם נשאר מעט בסוף מטלה, אפשר לאפשר בלוק קצר
                if alloc < min_block_minutes and needed >= min_block_minutes:
                    # אל תבזבז בלוק קטן באמצע מטלה, דלג על ההתחלה הזו
                    continue

                cand_end = s_dt + timedelta(minutes=alloc)

                # ודא שהבלוק נכנס בתוך אחד מחלונות הזמינות (אחרי החסמים)
                ok_in_window = False
                for ws, we in day_windows.get(d, []):
                    if s_dt >= ws and cand_end <= we:
                        ok_in_window = True
                        break
                if not ok_in_window:
                    continue

                if not _hard_ok(task, d, s_dt, cand_end, alloc):
                    continue

                sc = _slot_score(tid, d, s_dt, cand_end)
                if best is None or sc > best[0]:
                    best = (sc, d, s_dt, cand_end, alloc)

        if best is None:
            # cannot place this task any further, mark remaining as unscheduled and zero out
            unscheduled.append({
                "task_id": task.task_id,
                "course": task.course,
                "title": task.title,
                "reason": "אין מספיק משבצות פנויות בחלונות הזמן והאילוצים שנבחרו (כולל policy).",
                "remaining_hours": round(remaining_by_task[tid] / 60.0, 2),
            })
            remaining_by_task[tid] = 0
            continue

        _, d, s_dt, e_dt, alloc = best
        _book(task, d, s_dt, e_dt, alloc)
        remaining_by_task[tid] -= alloc

    # Any leftover positive remaining should become unscheduled
    for tid, rem in remaining_by_task.items():
        if rem > 0:
            t = task_map[tid]
            unscheduled.append({
                "task_id": t.task_id,
                "course": t.course,
                "title": t.title,
                "reason": "לא שובץ במלואו (סיום לולאה/guard).",
                "remaining_hours": round(rem / 60.0, 2),
            })
            remaining_by_task[tid] = 0

    validation = validate_schedule(events, tz, daily_max_minutes, weekday_blocks_t, date_blocks_t)
    report = {
        "tz": tz_name,
        "month": f"{year:04d}-{month:02d}",
        "daily_max_hours": float(daily_max_hours),
        "work_start": work_start_hhmm,
        "work_end": work_end_hhmm,
        "start_step_minutes": int(start_step_minutes),
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
def delete_selected_and_empty_rows(tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Deletes rows that are either:
    1) Marked for deletion via _delete == True
    2) "Empty" rows: no course+title AND deadline is missing/NaT

    Returns: (new_df, deleted_marked, deleted_empty)
    """
    if tasks_df is None or tasks_df.empty:
        return tasks_df, 0, 0

    df = tasks_df.copy()

    # Ensure columns exist
    if "_delete" not in df.columns:
        df["_delete"] = False
    for col in ["course", "title", "deadline"]:
        if col not in df.columns:
            df[col] = None

    # Normalize deadline to detect NaT reliably
    dl = pd.to_datetime(df["deadline"], errors="coerce")

    marked_mask = df["_delete"].fillna(False).astype(bool)
    empty_mask = (
        df["course"].fillna("").astype(str).str.strip().eq("")
        & df["title"].fillna("").astype(str).str.strip().eq("")
        & dl.isna()
    )

    deleted_marked = int(marked_mask.sum())
    deleted_empty = int(empty_mask.sum())

    df = df[~(marked_mask | empty_mask)].copy()

    # Reset delete column
    df["_delete"] = False

    return df, deleted_marked, deleted_empty

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
    st.subheader("זמן עבודה רציף, רזולוציית התחלה ומרווח ביטחון")

    r1, r2 = st.columns(2)

    with r1:
        max_continuous_minutes = st.selectbox(
            "זמן עבודה רציף (דקות) , אורך בלוק עבודה",
            options=[45, 60, 75, 90, 120, 150, 180, 240],
            index=[45, 60, 75, 90, 120, 150, 180, 240].index(
                int(st.session_state.get("max_continuous_minutes", 120))
                if int(st.session_state.get("max_continuous_minutes", 120)) in [45, 60, 75, 90, 120, 150, 180, 240]
                else 120
            ),
        )

    with r2:
        buffer_hours = st.selectbox(
            "מרווח ביטחון לפני דדליין (שעות)",
            options=[0, 12, 24, 36, 48, 72, 96],
            index=[0, 12, 24, 36, 48, 72, 96].index(
                int(st.session_state.get("buffer_hours", DEFAULT_BUFFER_HOURS))
                if int(st.session_state.get("buffer_hours", DEFAULT_BUFFER_HOURS)) in [0, 12, 24, 36, 48, 72, 96]
                else 48
            ),
        )

        break_minutes = st.selectbox(
            "הפסקה מינימלית בין בלוקים (דקות)",
            options=[0, 5, 10, 15, 20, 30, 45, 60],
            index=[0, 5, 10, 15, 20, 30, 45, 60].index(
                int(st.session_state.get("break_minutes", 15))
                if int(st.session_state.get("break_minutes", 15)) in [0, 5, 10, 15, 20, 30, 45, 60]
                else 15
            ),
        )

    st.session_state["max_continuous_minutes"] = int(max_continuous_minutes)
    st.session_state["buffer_hours"] = int(buffer_hours)
    st.session_state["break_minutes"] = int(break_minutes)

    st.caption("כעת זמן עבודה רציף הוא אורך אירוע העבודה. הרזולוציה משפיעה רק על נקודת ההתחלה.")

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


# =========================
# Inputs Form: Tasks + Constraints + Actions
# =========================

# ודא nonce קיים כדי שנוכל לאפס editor אחרי מחיקה (מונע תקיעות)
if "tasks_editor_nonce" not in st.session_state:
    st.session_state["tasks_editor_nonce"] = 0

# ודא עמודת מחיקה קיימת
if "_delete" not in st.session_state["tasks_df"].columns:
    st.session_state["tasks_df"]["_delete"] = False

# התאמת טיפוסי תאריך לפני DateColumn (כדי שלא יקרוס)
st.session_state["tasks_df"] = coerce_dates_for_editor(st.session_state["tasks_df"], "deadline")
st.session_state["date_blocks_df"] = coerce_dates_for_editor(st.session_state["date_blocks_df"], "date")

def delete_selected_and_empty_rows(tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Deletes rows that are either:
    1) Marked for deletion via _delete == True
    2) Empty rows: course+title empty AND deadline missing/NaT
    """
    if tasks_df is None or tasks_df.empty:
        return tasks_df, 0, 0

    df = tasks_df.copy()

    if "_delete" not in df.columns:
        df["_delete"] = False
    for col in ["course", "title", "deadline"]:
        if col not in df.columns:
            df[col] = None

    dl = pd.to_datetime(df["deadline"], errors="coerce")

    marked_mask = df["_delete"].fillna(False).astype(bool)
    empty_mask = (
        df["course"].fillna("").astype(str).str.strip().eq("")
        & df["title"].fillna("").astype(str).str.strip().eq("")
        & dl.isna()
    )

    deleted_marked = int(marked_mask.sum())
    deleted_empty = int(empty_mask.sum())

    df = df[~(marked_mask | empty_mask)].copy()
    df["_delete"] = False

    return df, deleted_marked, deleted_empty


with st.form("planner_form", clear_on_submit=False):

    st.markdown("## הזנת מטלות 📝")

    edited_tasks_df = st.data_editor(
        st.session_state["tasks_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "_delete": st.column_config.CheckboxColumn("מחיקה", help="סמן שורות למחיקה"),
            "task_id": st.column_config.TextColumn("מזהה"),
            "course": st.column_config.TextColumn("שם הקורס"),
            "title": st.column_config.TextColumn("שם המטלה"),
            "deadline": st.column_config.DateColumn("דדליין", format="DD/MM/YYYY"),
            "estimated_hours": st.column_config.NumberColumn("שעות משוערות", min_value=0.0, step=0.5),
            "priority": st.column_config.NumberColumn("עדיפות 1–5", min_value=1, max_value=5, step=1),
            "notes": st.column_config.TextColumn("הערות"),
        },
        key=f"w_tasks_editor_main_{st.session_state['tasks_editor_nonce']}",
    )

    st.markdown("### הדבקת מטלות בטקסט חופשי (אופציונלי)")
    free_text = st.text_area(
        "הדבק כאן",
        height=120,
        placeholder="לדוגמה:\nביולוגיה | עבודה סמינריונית | 01/02/2026 | 12 | עדיפות 5",
        key="free_text_tasks",
    )

    st.divider()
    st.markdown("## הגדרת חסמים ⛔")

    st.markdown("### חסמים שבועיים קבועים")
    edited_wd_df = st.data_editor(
        st.session_state["weekday_blocks_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "weekday": st.column_config.SelectboxColumn("יום", options=WEEKDAYS_HE),
            "start": st.column_config.TextColumn("התחלה (HH:MM)"),
            "end": st.column_config.TextColumn("סיום (HH:MM)"),
            "label": st.column_config.TextColumn("תיאור"),
        },
        key="weekday_blocks_editor",
    )

    st.markdown("### חסמים בתאריכים ספציפיים")
    edited_date_df = st.data_editor(
        st.session_state["date_blocks_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("תאריך", format="DD/MM/YYYY"),
            "start": st.column_config.TextColumn("התחלה (HH:MM)"),
            "end": st.column_config.TextColumn("סיום (HH:MM)"),
            "label": st.column_config.TextColumn("תיאור"),
        },
        key="date_blocks_editor",
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        save_clicked = st.form_submit_button("💾 שמור נתונים")
    with c2:
        delete_clicked = st.form_submit_button("🗑️ מחק מסומנים/ריקים")
    with c3:
        compute_clicked = st.form_submit_button("🚀 שמור וחשב לו״ז", type="primary")


# =========================
# After form: Commit actions
# =========================
if save_clicked or delete_clicked or compute_clicked:
    st.session_state["tasks_df"] = edited_tasks_df
    st.session_state["weekday_blocks_df"] = edited_wd_df
    st.session_state["date_blocks_df"] = edited_date_df

# מחיקה אמיתית + מניעת תקיעות
if delete_clicked:
    before = len(st.session_state["tasks_df"])
    new_df, deleted_marked, deleted_empty = delete_selected_and_empty_rows(st.session_state["tasks_df"])
    after = len(new_df)

    st.session_state["tasks_df"] = new_df
    st.success(
        f"נמחקו {deleted_marked} שורות שסומנו למחיקה, ועוד {deleted_empty} שורות ריקות/ללא דדליין. "
        f"(סה״כ {before - after} שורות)."
    )

    # איפוס העורך כדי שלא ייתקע
    st.session_state["tasks_editor_nonce"] += 1
    st.rerun()

# הוספת מטלות מהטקסט החופשי (רק בשמירה/חישוב)
if (save_clicked or compute_clicked) and (st.session_state.get("free_text_tasks") or "").strip():
    parsed = try_ai_parse_tasks(st.session_state["free_text_tasks"])
    if parsed:
        add_df = pd.DataFrame(
            [{
                "_delete": False,
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

# =========================
# Smart Policy (Free text -> AI -> constraints/preferences)
# Place this block under "שיבוץ" settings, BEFORE the compute/explain buttons
# =========================
st.markdown("### 🧭 Policy חכם (אופציונלי)")
st.caption("כתוב העדפות ומגבלות בשפה חופשית. ה-AI ינסה לתרגם אותן למדיניות שיבוץ (Policy) שתשפיע על המנוע הדטרמיניסטי.")

if "policy" not in st.session_state:
    st.session_state["policy"] = None
if "policy_notes" not in st.session_state:
    st.session_state["policy_notes"] = ""

def _extract_json_from_model_text(text: str) -> str:
    t = (text or "").strip()
    # Try fenced ```json
    m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try any {...} block (best-effort)
    m2 = re.search(r"(\{.*\})", t, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return t

WEEKDAYS_HE_LIST = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]

def _is_hhmm(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", (s or "").strip()))

def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def sanitize_policy(raw: dict, defaults: dict) -> dict:
    raw = raw or {}
    hard = raw.get("hard") or {}
    soft = raw.get("soft") or {}

    out = {"hard": {}, "soft": {}, "notes": str(raw.get("notes") or "").strip()}

    # Hard
    out["hard"]["work_start"] = hard.get("work_start") if _is_hhmm(hard.get("work_start")) else defaults["work_start"]
    out["hard"]["work_end"]   = hard.get("work_end")   if _is_hhmm(hard.get("work_end"))   else defaults["work_end"]

    def _num(v, d):
        try:
            return float(v)
        except Exception:
            return float(d)

    out["hard"]["daily_max_hours"] = _clamp(_num(hard.get("daily_max_hours"), defaults["daily_max_hours"]), 0.5, 16.0)
    out["hard"]["max_task_hours_per_day"] = _clamp(_num(hard.get("max_task_hours_per_day"), defaults["max_task_hours_per_day"]), 0.5, 8.0)

    out["hard"]["break_minutes"] = int(_clamp(_num(hard.get("break_minutes"), defaults["break_minutes"]), 0, 180))
    out["hard"]["max_continuous_minutes"] = int(_clamp(_num(hard.get("max_continuous_minutes"), defaults["max_continuous_minutes"]), 0, 600))
    out["hard"]["buffer_hours"] = int(_clamp(_num(hard.get("buffer_hours"), defaults["buffer_hours"]), 0, 240))

    avoid = hard.get("avoid_weekdays") or []
    out["hard"]["avoid_weekdays"] = [d for d in avoid if d in WEEKDAYS_HE_LIST]

    # Fixed blocks (weekday-based). Optional.
    fixed_blocks = []
    for b in (hard.get("fixed_blocks") or []):
        wd = str(b.get("weekday") or "").strip()
        s = str(b.get("start") or "").strip()
        e = str(b.get("end") or "").strip()
        label = str(b.get("label") or "אילוץ").strip()
        if wd in WEEKDAYS_HE_LIST and _is_hhmm(s) and _is_hhmm(e):
            fixed_blocks.append({"weekday": wd, "start": s, "end": e, "label": label})
    out["hard"]["fixed_blocks"] = fixed_blocks

    # Soft
    # prefer_time_windows: list of {"start","end","weight"} weight 0..1
    ptw = []
    for w in (soft.get("prefer_time_windows") or []):
        s = str(w.get("start") or "").strip()
        e = str(w.get("end") or "").strip()
        try:
            weight = float(w.get("weight"))
        except Exception:
            continue
        if _is_hhmm(s) and _is_hhmm(e) and 0.0 <= weight <= 1.0:
            ptw.append({"start": s, "end": e, "weight": weight})
    out["soft"]["prefer_time_windows"] = ptw

    # target_daily_load_ratio: 0.5..0.9
    try:
        tdlr = float(soft.get("target_daily_load_ratio"))
    except Exception:
        tdlr = float(defaults["target_daily_load_ratio"])
    out["soft"]["target_daily_load_ratio"] = _clamp(tdlr, 0.5, 0.9)

    # task_switch_penalty 0..1 (higher = prefer staying on same task)
    try:
        tsp = float(soft.get("task_switch_penalty"))
    except Exception:
        tsp = float(defaults["task_switch_penalty"])
    out["soft"]["task_switch_penalty"] = _clamp(tsp, 0.0, 1.0)

    # midday split for morning/afternoon (optional)
    md = str(soft.get("midday_hhmm") or defaults["midday_hhmm"]).strip()
    out["soft"]["midday_hhmm"] = md if _is_hhmm(md) else defaults["midday_hhmm"]

    return out

policy_text = st.text_area(
    "תאר במילים חופשיות את ההעדפות והמגבלות שלך (דוגמה: \"עובד הכי טוב בבוקר 09:00-12:00, אחרי 16:00 לא לומד, הפסקה כל שעה, שישי פנוי רק בבוקר\")",
    height=120,
    placeholder="כתוב כאן...",
    key="w_policy_free_text"
)

pol_col1, pol_col2 = st.columns([1, 1])
with pol_col1:
    analyze_policy_clicked = st.button("🧠 נתח policy מהטקסט", type="secondary")
with pol_col2:
    clear_policy_clicked = st.button("🧹 נקה policy", type="secondary")

if clear_policy_clicked:
    st.session_state["policy"] = None
    st.session_state["policy_notes"] = ""
    st.success("policy נוקה.")

if analyze_policy_clicked:
    # Defaults derived from current UI variables (must exist in your app before this block)
    defaults = {
        "work_start": workday_start_t.strftime("%H:%M"),
        "work_end": workday_end_t.strftime("%H:%M"),
        "daily_max_hours": float(daily_max_hours),
        "max_task_hours_per_day": float(max_task_hours_per_day),
        "break_minutes": int(st.session_state.get("break_minutes", 15)) if "break_minutes" in st.session_state else 15,
        "max_continuous_minutes": int(st.session_state.get("max_continuous_minutes", 120)) if "max_continuous_minutes" in st.session_state else 120,
        "buffer_hours": int(buffer_hours),
        "target_daily_load_ratio": 0.75,
        "task_switch_penalty": 0.2,
        "midday_hhmm": "13:00",
    }

    policy_prompt = f"""
אתה מומחה לניהול זמן. קיבלת טקסט חופשי בעברית עם העדפות/מגבלות למידה.
החזר אך ורק JSON תקין (ללא טקסט נוסף) לפי הסכימה:

{{
  "hard": {{
    "work_start": "HH:MM",
    "work_end": "HH:MM",
    "daily_max_hours": number,
    "max_task_hours_per_day": number,
    "break_minutes": number,
    "max_continuous_minutes": number,
    "buffer_hours": number,
    "avoid_weekdays": ["ראשון"|"שני"|"שלישי"|"רביעי"|"חמישי"|"שישי"|"שבת"],
    "fixed_blocks": [
      {{"weekday":"...","start":"HH:MM","end":"HH:MM","label":"..."}}
    ]
  }},
  "soft": {{
    "prefer_time_windows": [
      {{"start":"HH:MM","end":"HH:MM","weight": number}}
    ],
    "target_daily_load_ratio": number,
    "task_switch_penalty": number,
    "midday_hhmm": "HH:MM"
  }},
  "notes": "סיכום קצר בעברית"
}}

כללים מחייבים:
1) אל תמציא פרטים. אם לא נכתב במפורש, השאר שדה ריק/אל תכלול אותו.
2) שעות תמיד HH:MM.
3) weight בין 0 ל-1. target_daily_load_ratio בין 0.5 ל-0.9. task_switch_penalty בין 0 ל-1.

ברירות מחדל אם אין מידע:
work_start={defaults["work_start"]}, work_end={defaults["work_end"]},
daily_max_hours={defaults["daily_max_hours"]}, max_task_hours_per_day={defaults["max_task_hours_per_day"]},
break_minutes={defaults["break_minutes"]}, max_continuous_minutes={defaults["max_continuous_minutes"]},
buffer_hours={defaults["buffer_hours"]}, midday_hhmm={defaults["midday_hhmm"]}.

הטקסט:
\"\"\"{policy_text}\"\"\"
"""

    try:
        with st.spinner("מנתח policy מהטקסט..."):
            raw_text = model.generate_content(policy_prompt).text
        json_str = _extract_json_from_model_text(raw_text)
        raw_policy = json.loads(json_str)
        pol = sanitize_policy(raw_policy, defaults)

        st.session_state["policy"] = pol
        st.session_state["policy_notes"] = pol.get("notes", "")

        st.success("policy הופק ונשמר. הוא יופעל בחישוב הבא (Compute).")
    except Exception as e:
        st.error("לא הצלחתי לנתח policy. פירוט:")
        st.exception(e)

if st.session_state.get("policy"):
    with st.expander("הצג policy נוכחי", expanded=False):
        st.write(st.session_state.get("policy_notes", ""))
        st.json(st.session_state["policy"])

col_a, col_b = st.columns([1, 1])
with col_a:
    compute_clicked = st.button("🚀 חשב לו״ז אסטרטגי וחכם", type="primary")
with col_b:
    explain_clicked = st.button("🧠 צור הסבר והמלצות (אופציונלי)", type="secondary")


def _count_missing_deadlines(df: pd.DataFrame) -> int:
    if df is None or df.empty or "deadline" not in df.columns:
        return 0
    return int(pd.to_datetime(df["deadline"], errors="coerce").isna().sum())


if compute_clicked:
    st.info("DEBUG: לחצת על כפתור החישוב", icon="🧪")

    # 1) בדיקת דדליינים חסרים בטבלה (רק בחישוב)
    missing = _count_missing_deadlines(st.session_state["tasks_df"])
    st.write({"DEBUG_missing_deadlines": missing})

    if missing > 0:
        st.error(f"יש {missing} מטלות ללא דדליין. מחק שורות ריקות או מלא תאריך ואז נסה שוב.")
        st.stop()

    # 2) המרה למודל
    tasks = df_to_tasks(st.session_state["tasks_df"])
    st.write({"DEBUG_tasks_after_parse": len(tasks)})

    if not tasks:
        st.warning("לא נמצאו מטלות תקינות. ודא שיש דדליין ושעות משוערות.")
        st.stop()

    # 3) חסמים
    weekday_blocks = df_to_weekday_blocks(st.session_state["weekday_blocks_df"])
    date_blocks = df_to_date_blocks(st.session_state["date_blocks_df"])

    # 4) חלון יום עבודה
    workday_start_str = workday_start_t.strftime("%H:%M")
    workday_end_str = workday_end_t.strftime("%H:%M")

    # 5) פרמטרים לשיבוץ
    schedule_params = {
        "tasks": tasks,
        "tz_name": tz_name.strip() or DEFAULT_TZ,
        "year": int(year),
        "month": int(month),
        "work_start_hhmm": workday_start_str,
        "work_end_hhmm": workday_end_str,
        "daily_max_hours": float(daily_max_hours),
        "max_task_hours_per_day": float(max_task_hours_per_day),
        "weekday_blocks": weekday_blocks,
        "date_blocks": date_blocks,
        "policy": st.session_state.get("policy"),
        "start_step_minutes": 15,
        "buffer_hours": int(st.session_state.get("buffer_hours", buffer_hours)),
        "max_continuous_minutes": int(st.session_state.get("max_continuous_minutes", 120)),
    }

    st.write("DEBUG schedule_params keys:", list(schedule_params.keys()))
    st.write("DEBUG tasks sample:", [{"id": t.task_id, "deadline": str(t.deadline)} for t in tasks[:3]])

    # 6) הרצת שיבוץ
    with st.spinner("המערכת בונה לו״ז חודשי תוך כיבוד אילוצים ועומסים..."):
        try:
            events, report = schedule_tasks(**schedule_params)
                    # DEBUG – בדיקת פיזור וזמני שיבוץ
            st.write(
                "דוגמת אירועים ראשונים:",
                [
                    (
                        e.start_dt.strftime("%d/%m %H:%M"),
                        e.end_dt.strftime("%H:%M"),
                        e.title
                    )
                    for e in events[:10]
                ]
            )



            st.session_state["events"] = events
            st.session_state["report"] = report

            st.success(f"הלו״ז הושלם בהצלחה. נוצרו {len(events)} משבצות עבודה.")

            # הצגה מינימלית כדי שלא יהיה "שקט"
            if len(events) == 0:
                st.warning("לא נוצרו אירועים. בדוק חלונות עבודה, מגבלות יומיות, וחסמים.")
                st.write(report)
            else:
                st.dataframe(
                    pd.DataFrame([{
                        "תאריך": ev.start_dt.strftime("%d/%m/%Y"),
                        "התחלה": ev.start_dt.strftime("%H:%M"),
                        "סיום": ev.end_dt.strftime("%H:%M"),
                        "כותרת": ev.title,
                    } for ev in events]),
                    use_container_width=True
                )

        except Exception as e:
            st.error("שגיאה בקריאה ל-schedule_tasks. פירוט מלא:")
            st.exception(e)
            st.stop()

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