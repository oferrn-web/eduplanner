import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json
import re

# --- 1. הגדרות דף ועיצוב RTL ---
st.set_page_config(page_title="מתכנן המטלות החכם", layout="wide")
st.markdown("<style> .stApp { direction: RTL; text-align: right; } </style>", unsafe_allow_html=True)

# --- 2. ניהול זיכרון (Session State) ---
if 'form_version' not in st.session_state:
    st.session_state.form_version = 0
if 'extracted_tasks' not in st.session_state:
    st.session_state.extracted_tasks = []

def clear_everything():
    # מחיקת כל המפתחות בזיכרון
    for key in list(st.session_state.keys()):
        if key != 'form_version':
            del st.session_state[key]
    st.session_state.form_version += 1
    st.rerun()

# --- 3. חיבור ל-AI (Secrets) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-3-pro-preview')
except Exception as e:
    st.error("לא נמצא מפתח ב-Secrets. וודא שהגדרת GEMINI_API_KEY.")
    st.stop()

st.title("📅 מתכנן המטלות החכם")

# --- 4. סרגל צד: אילוצים עם כותרות ---
with st.sidebar:
    st.header("⚙️ אילוצים וזמנים תפוסים")
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("ימים עם אילוץ קבוע:", days_week, key=f"days_{st.session_state.form_version}")
    
    day_constraints = []
    for day in selected_days:
        with st.expander(f"אילוץ ליום {day}", expanded=True):
            headline = st.text_input(f"כותרת (למשל: עבודה)", key=f"h_{day}_{st.session_state.form_version}")
            # כאן התיקון - השמות c1 ו-c2 עקביים
            c1, c2 = st.columns(2)
            with c1:
                start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"s_{day}_{st.session_state.form_version}")
            with c2:
                end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"e_{day}_{st.session_state.form_version}")
            day_constraints.append({"day": day, "title": headline, "hours": f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"})

    daily_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 10, 4, key=f"max_{st.session_state.form_version}")

    st.divider()
    if st.button("🗑️ ניקוי כל הנתונים"):
        clear_everything()

# --- 5. הזנת מטלות והדרכה ---
st.header("📝 הזנת מטלות")
st.info("💡 **טיפ:** סמנו את הטבלה בגוגל שיטס, העתיקו (Ctrl+C) והדביקו כאן. ה-AI כבר ידע להפריד בין קורס למטלה.")

raw_input = st.text_area("הדבק כאן את תוכן הטבלה:", key=f"raw_in_{st.session_state.form_version}")

if st.button("🔍 חלץ ונתח מטלות"):
    if raw_input:
        with st.spinner("ה-AI מנתח את מבנה הטקסט..."):
            parse_prompt = f"""
            נתח את הטקסט הבא וחלץ מטלות. 
            זהה: שם קורס, שם מטלה, תאריך יעד (YYYY-MM-DD), ותתי-משימות.
            החזר רשימת JSON בלבד: [{{"name": "קורס: מטלה", "time": 2.0, "deadline": "YYYY-MM-DD", "subs": "פירוט"}}]
            טקסט: {raw_input}
            """
            try:
                res = model.generate_content(parse_prompt)
                json_match = re.search(r'\[.*\]', res.text, re.DOTALL)
                if json_match:
                    st.session_state.extracted_tasks = json.loads(json_match.group(0))
                    st.success("המטלות חולצו! עברו עליהן למטה כדי לעדכן שעות.")
                    st.rerun()
            except Exception as e:
                st.error(f"שגיאה בניתוח: {e}")

# --- 6. עריכת פרטי המטלות ---
if st.session_state.extracted_tasks:
    st.subheader("✍️ עדכון פרטים אחרון")
    for idx, task in enumerate(st.session_state.extracted_tasks):
        with st.expander(f"עריכה: {task['name']}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.session_state.extracted_tasks[idx]['name'] = st.text_input("שם", value=task['name'], key=f"n_{idx}_{st.session_state.form_version}")
            with col2:
                st.session_state.extracted_tasks[idx]['time'] = st.number_input("שעות", value=float(task.get('time', 2.0)), key=f"t_{idx}_{st.session_state.form_version}")
            with col3:
                try:
                    d_val = datetime.datetime.strptime(task['deadline'], "%Y-%m-%d").date()
                except:
                    d_val = datetime.date.today()
                st.session_state.extracted_tasks[idx]['deadline'] = str(st.date_input("דדליין", value=d_val, key=f"d_{idx}_{st.session_state.form_version}"))
            st.session_state.extracted_tasks[idx]['subs'] = st.text_area("תתי משימות", value=task.get('subs', ""), key=f"s_{idx}_{st.session_state.form_version}")

# --- 7. חישוב לו"ז סופי ---
st.divider()
if st.button("🚀 חשב לו''ז חכם (טבלה)"):
    if st.session_state.extracted_tasks:
        final_prompt = f"""
        פזר את המטלות האלו בלו"ז: {st.session_state.extracted_tasks}
        חסמים (זמנים שבהם המשתמש תפוס ואסור לשבץ): {day_constraints}
        מגבלת שעות עבודה ביום: {daily_max_hours}.
        
        החזר טבלה בעברית (תאריך, מטלה, תת-משימה, שעות) ובלוק JSON בסוף לייצוא ליומן.
        """
        with st.spinner("מחשב פיזור אופטימלי..."):
            res = model.generate_content(final_prompt).text
            st.markdown(res)
            if "```json" in res:
                st.session_state.last_sched = json.loads(res.split("```json")[1].split("```")[0].strip())

# --- 8. ייצוא ---
if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item.get('start_time', '09:00')}:00"
        c.events.add(e)
    st.download_button("💾 הורד ליומן גוגל", str(c), file_name="planner.ics")