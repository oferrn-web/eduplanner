import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json
import re

# --- 1. הגדרות RTL ועיצוב ---
st.set_page_config(page_title="מתכנן המטלות האקדמי", layout="wide")
st.markdown("<style> .stApp { direction: RTL; text-align: right; } </style>", unsafe_allow_html=True)

# --- 2. ניהול זיכרון (Session State) ---
if 'form_version' not in st.session_state: st.session_state.form_version = 0
if 'extracted_tasks' not in st.session_state: st.session_state.extracted_tasks = []
if 'last_sched' not in st.session_state: st.session_state.last_sched = []

def clear_everything():
    for key in list(st.session_state.keys()):
        if key != 'form_version': del st.session_state[key]
    st.session_state.form_version += 1
    st.rerun()

# --- 3. חיבור ל-AI ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-3-pro-preview')
except:
    st.error("שגיאה בחיבור ל-AI. בדוק את ה-Secrets.")
    st.stop()

st.title("🎓 מתכנן המטלות האקדמי שלי")

# --- 4. סרגל צד: אילוצים ושנה ---
with st.sidebar:
    st.header("⚙️ הגדרות מערכת")
    
    # בחירת שנה דינמית
    current_year = datetime.date.today().year
    selected_year = st.selectbox("בחר שנת לימודים:", [current_year, current_year + 1, current_year + 2], index=0)
    
    st.divider()
    st.subheader("🗓️ זמנים תפוסים (אילוצים)")
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("בחר ימים עם אילוץ קבוע:", days_week, key=f"days_{st.session_state.form_version}")
    
    day_constraints = []
    for day in selected_days:
        with st.expander(f"אילוץ ליום {day}", expanded=True):
            headline = st.text_input(f"כותרת (למשל: עבודה)", key=f"h_{day}_{st.session_state.form_version}")
            c1, c2 = st.columns(2)
            with c1: start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"s_{day}_{st.session_state.form_version}")
            with c2: end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"e_{day}_{st.session_state.form_version}")
            day_constraints.append({"day": day, "title": headline, "hours": f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"})

    daily_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 12, 4, key=f"max_{st.session_state.form_version}")
    
    if st.button("🗑️ ניקוי כל הנתונים"): clear_everything()

# --- 5. הזנת מטלות ---
st.header("📝 הזנת מטלות")
st.info(f"💡 **המלצה:** העתק טבלה מהשיטס והדבק כאן. המערכת תשתמש בשנת **{selected_year}** כברירת מחדל.")

raw_input = st.text_area("הדבק כאן תוכן מהשיטס או מהסילבוס:", key=f"raw_in_{st.session_state.form_version}")

if st.button("🔍 חלץ ונתח מטלות"):
    if raw_input:
        with st.spinner("ה-AI מנתח מטלות..."):
            parse_prompt = f"""
            נתח את הטקסט וחלץ מטלות. עבור כל אחת זהה: שם קורס, שם מטלה, תאריך יעד ותתי-משימות.
            אם לא מצוינת שנה בתאריך היעד, השתמש בשנת {selected_year}.
            החזר רשימת JSON: [{{"name": "קורס: שם", "time": 2.0, "deadline": "YYYY-MM-DD", "subs": "פירוט"}}]
            טקסט: {raw_input}
            """
            res = model.generate_content(parse_prompt)
            match = re.search(r'\[.*\]', res.text, re.DOTALL)
            if match:
                st.session_state.extracted_tasks = json.loads(match.group(0))
                st.rerun()

# --- 6. עריכה ---
if st.session_state.extracted_tasks:
    st.subheader("✍️ הגדרת שעות ופירוט לכל מטלה")
    for idx, task in enumerate(st.session_state.extracted_tasks):
        with st.expander(f"עריכה: {task['name']}", expanded=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: st.session_state.extracted_tasks[idx]['name'] = st.text_input("שם", value=task['name'], key=f"n_{idx}")
            with c2: st.session_state.extracted_tasks[idx]['time'] = st.number_input("שעות", value=float(task.get('time', 2.0)), key=f"t_{idx}")
            with c3:
                try: d_val = datetime.datetime.strptime(task['deadline'], "%Y-%m-%d").date()
                except: d_val = datetime.date.today()
                st.session_state.extracted_tasks[idx]['deadline'] = str(st.date_input("דדליין", value=d_val, key=f"d_{idx}"))
            st.session_state.extracted_tasks[idx]['subs'] = st.text_area("תתי משימות", value=task.get('subs', ""), key=f"s_{idx}")

# --- 7. חישוב לו"ז מפורט ---
st.divider()
if st.button("🚀 חשב לו''ז מפורט וסנכרן ליומן"):
    if st.session_state.extracted_tasks:
        final_prompt = f"""
        אתה מתכנן לוחות זמנים מקצועי לסטודנטים. פזר את המטלות האלו בלו"ז: {st.session_state.extracted_tasks}
        אילוצים (זמן תפוס): {day_constraints}.
        מגבלת שעות עבודה ביום: {daily_max_hours}.
        
        הנחיות קריטיות:
        1. צור אירוע נפרד לכל תת-משימה! 
        2. השתמש בשנה שמופיעה בדדליין של כל מטלה.
        3. אל תשבץ בשעות האילוצים.
        4. החזר טבלה בעברית ובסוף בלוק JSON עם: title, date (YYYY-MM-DD), start_time (HH:MM).
        """
        with st.spinner("בונה לו''ז מפורט..."):
            res = model.generate_content(final_prompt).text
            st.markdown(res)
            if "```json" in res:
                st.session_state.last_sched = json.loads(res.split("```json")[1].split("```")[0].strip())
                st.success(f"הלו''ז כולל {len(st.session_state.last_sched)} אירועים נפרדים.")

# --- 8. תצוגה מקדימה וייצוא ---
if st.session_state.last_sched:
    with st.expander("👁️ תצוגה מקדימה של האירועים שייוצאו", expanded=True):
        st.table(st.session_state.last_sched)
    
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item.get('title', item.get('name', 'מטלה'))
        # וידוא פורמט תקין לייצוא
        try:
            e.begin = f"{item.get('date')} {item.get('start_time', '09:00')}:00"
            e.duration = {"hours": 1}
            c.events.add(e)
        except: continue
    
    st.download_button("💾 הורד קובץ ליומן גוגל", str(c), file_name="academic_planner.ics")