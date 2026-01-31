import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json

# --- 1. הגדרות דף ועיצוב RTL ---
st.set_page_config(page_title="מתכנן המטלות החכם", layout="wide")
st.markdown("<style> .stApp { direction: RTL; text-align: right; } </style>", unsafe_allow_html=True)

# --- 2. חיבור ל-AI (עם תיקון ה-Permission) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-3-pro-preview')
except Exception as e:
    st.error("לא נמצא מפתח ב-Secrets או שיש שגיאת חיבור.")
    st.stop()

st.title("📅 מתכנן המטלות החכם")

# --- 3. סרגל צד: אילוצים וניקוי נתונים ---
with st.sidebar:
    st.header("⚙️ הגדרות מערכת")
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("ימים עם לו\"ז קבוע:", days_week)
    
    day_constraints = {}
    for day in selected_days:
        st.write(f"**ביום {day}:**")
        start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"s_{day}")
        end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"e_{day}")
        day_constraints[day] = f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"

    daily_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 10, 4)

    st.divider()
    # כפתור ניקוי הנתונים
    if st.button("🗑️ ניקוי כל הנתונים"):
        st.session_state.extracted_tasks = []
        if 'last_sched' in st.session_state:
            del st.session_state.last_sched
        st.success("הנתונים נמחקו!")
        st.rerun()

# --- 4. הזנת מטלות (ייבוא או ידני) ---
if 'extracted_tasks' not in st.session_state:
    st.session_state.extracted_tasks = []

input_method = st.radio("בחר שיטת הזנה:", ["ייבוא מרשימה/קישור (גוגל שיטס)", "הזנה ידנית"])

if input_method == "ייבוא מרשימה/קישור (גוגל שיטס)":
    raw_input = st.text_area("הדבק קישור או רשימת מטלות:")
    if st.button("🔍 חלץ מטלות"):
        with st.spinner("מזהה מטלות..."):
            prompt = f"חלץ מרשימה זו רק את שמות המטלות העיקריות. החזר JSON של רשימת שמות בלבד: {raw_input}"
            try:
                res = model.generate_content(prompt)
                names = json.loads(res.text.replace("```json", "").replace("```", "").strip())
                st.session_state.extracted_tasks = [{"name": n, "time": 2.0, "deadline": str(datetime.date.today()), "subs": ""} for n in names]
            except Exception as e:
                st.error(f"שגיאה בזיהוי: {e}")
else:
    if st.button("➕ הוסף מטלה ריקה"):
        st.session_state.extracted_tasks.append({"name": f"מטלה {len(st.session_state.extracted_tasks)+1}", "time": 1.0, "deadline": str(datetime.date.today()), "subs": ""})

# --- 5. הזנת שעות ותתי-משימות לכל מטלה ---
if st.session_state.extracted_tasks:
    st.subheader("✍️ הגדר שעות ותתי-משימות לכל מטלה:")
    for idx, task in enumerate(st.session_state.extracted_tasks):
        with st.expander(f"עריכה: {task['name']}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.session_state.extracted_tasks[idx]['name'] = st.text_input("שם", value=task['name'], key=f"n_{idx}")
            with col2:
                st.session_state.extracted_tasks[idx]['time'] = st.number_input("שעות", value=float(task['time']), key=f"t_{idx}")
            with col3:
                curr_date = datetime.datetime.strptime(task['deadline'], "%Y-%m-%d").date()
                st.session_state.extracted_tasks[idx]['deadline'] = str(st.date_input("דדליין", value=curr_date, key=f"d_{idx}"))
            
            st.session_state.extracted_tasks[idx]['subs'] = st.text_area("תתי משימות (הפרד בשורות)", value=task['subs'], key=f"s_{idx}")

# --- 6. יצירת הלו"ז הטבלאי ---
st.divider()
if st.button("🚀 חשב לו''ז חכם"):
    if not st.session_state.extracted_tasks:
        st.warning("נא להזין מטלות.")
    else:
        final_prompt = f"""
        פזר את המטלות האלו בלו"ז: {st.session_state.extracted_tasks}
        אילוצי שעות: {day_constraints}, מקסימום {daily_max_hours} שעות ביום.
        
        החזר תשובה בשני חלקים:
        1. טבלה בעברית (תאריך, מטלה, תת-משימה, שעות עבודה).
        2. בסוף, בלוק ```json ``` עם title, date (YYYY-MM-DD), start_time.
        """
        with st.spinner("בונה תוכנית עבודה..."):
            try:
                res = model.generate_content(final_prompt).text
                st.markdown(res)
                
                if "```json" in res:
                    st.session_state.last_sched = json.loads(res.split("```json")[1].split("```")[0].strip())
            except Exception as e:
                st.error(f"שגיאה בחישוב הלו\"ז: {e}")

# --- 7. ייצוא ---
if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item.get('start_time', '09:00')}:00"
        c.events.add(e)
    st.download_button("💾 הורד ליומן גוגל", str(c), file_name="planner.ics")