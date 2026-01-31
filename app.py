import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json

# --- 1. הגדרות דף ועיצוב RTL ---
st.set_page_config(page_title="מתכנן המטלות החכם שלי", layout="wide")
st.markdown("""
    <style>
    .stApp { direction: RTL; text-align: right; }
    div[data-testid="stSidebar"] { direction: RTL; }
    .stTextArea textarea, .stTextInput input { direction: RTL; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. חיבור לבינה מלאכותית (Secrets) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3-pro-preview')
except Exception as e:
    st.error("שגיאה בחיבור ל-API. וודא שהגדרת את GEMINI_API_KEY ב-Secrets.")
    st.stop()

st.title("📅 מתכנן המטלות החכם לסטודנטים")

# --- 3. סרגל צד: אילוצים ---
with st.sidebar:
    st.header("⚙️ הגדרות ואילוצים")
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("ימים עם לו\"ז קבוע:", days_week)
    
    day_constraints = {}
    for day in selected_days:
        st.write(f"**ביום {day}:**")
        start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"s_{day}")
        end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"e_{day}")
        day_constraints[day] = f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"

    st.divider()
    daily_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 10, 4)

# --- 4. מנגנון הזנת מטלות חכם ---
st.header("📝 הזנת מטלות")

if 'extracted_tasks' not in st.session_state:
    st.session_state.extracted_tasks = []

# בחירת שיטת הזנה
input_method = st.radio("בחר שיטה:", ["הזנה ידנית", "ייבוא מרשימה/קישור גוגל שיטס"])

if input_method == "הזנה ידנית":
    with st.expander("➕ הוספת מטלה אחת"):
        t_name = st.text_input("שם המטלה")
        t_time = st.number_input("שעות", min_value=1.0, step=0.5)
        t_deadline = st.date_input("דדליין", value=datetime.date.today() + datetime.timedelta(days=7))
        t_subs = st.text_area("תתי משימות")
        if st.button("הוסף"):
            st.session_state.extracted_tasks.append({"name": t_name, "time": t_time, "deadline": str(t_deadline), "subtasks": t_subs})
            st.rerun()

else:
    raw_input = st.text_area("הדבק כאן קישור לגוגל שיטס או רשימת מטלות מהסילבוס:")
    if st.button("🔍 חלץ מטלות מהטקסט"):
        with st.spinner("ה-AI מזהה את המטלות שלך..."):
            parse_prompt = f"חלץ מרשימה זו רק את שמות המטלות העיקריות. החזר רשימת JSON של שמות המטלות בלבד: {raw_input}"
            response = model.generate_content(parse_prompt)
            try:
                # חילוץ שמות המטלות
                names = json.loads(response.text.replace("```json", "").replace("```", "").strip())
                st.session_state.extracted_tasks = [{"name": name, "time": 2.0, "deadline": str(datetime.date.today()), "subtasks": ""} for name in names]
                st.success(f"נמצאו {len(names)} מטלות!")
            except:
                st.error("לא הצלחתי לזהות מטלות. נסה להדביק טקסט ברור יותר.")

# --- 5. עריכת פרטים למטלות שחולצו ---
if st.session_state.extracted_tasks:
    st.subheader("⚙️ הגדר פרטים לכל מטלה:")
    for idx, task in enumerate(st.session_state.extracted_tasks):
        with st.expander(f"מטלה: {task['name']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.extracted_tasks[idx]['time'] = st.number_input(f"שעות עבודה נדרשות", value=float(task['time']), key=f"time_{idx}")
            with col2:
                st.session_state.extracted_tasks[idx]['deadline'] = str(st.date_input(f"דדליין", key=f"date_{idx}"))
            st.session_state.extracted_tasks[idx]['subtasks'] = st.text_area(f"תתי משימות עבור {task['name']}", value=task['subtasks'], key=f"sub_{idx}")

# --- 6. חישוב לו"ז סופי ---
st.divider()
if st.button("🚀 חשב לו\"ז חכם בטבלה"):
    if not st.session_state.extracted_tasks:
        st.warning("אין מטלות לחישוב.")
    else:
        final_prompt = f"""
        אתה מומחה לניהול זמן לסטודנטים לחינוך מיוחד.
        מטלות: {st.session_state.extracted_tasks}
        אילוצי שעות: {day_constraints}
        מקסימום {daily_max_hours} שעות ביום.
        
        צור טבלה מסודרת בעברית (תאריך, מטלה, תת-משימה, שעות).
        בסוף, הוסף בלוק JSON עם title, date (YYYY-MM-DD), start_time.
        """
        with st.spinner("בונה תוכנית עבודה..."):
            res = model.generate_content(final_prompt).text
            st.markdown(res)
            
            # שמירה לייצוא ICS
            if "```json" in res:
                st.session_state.last_sched = json.loads(res.split("```json")[1].split("```")[0].strip())

if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item.get('start_time', '09:00')}:00"
        c.events.add(e)
    st.download_button("💾 הורד ליומן גוגל", str(c), file_name="planner.ics")