import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json
import pandas as pd

# --- 1. הגדרות דף ועיצוב RTL (מימין לשמאל) ---
st.set_page_config(page_title="מתכנן המטלות החכם שלי", layout="wide")

st.markdown("""
    <style>
    .stApp { direction: RTL; text-align: right; }
    div[data-testid="stSidebar"] { direction: RTL; }
    .stTextArea textarea, .stTextInput input { direction: RTL; }
    .stMultiSelect div { direction: RTL; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. חיבור לבינה מלאכותית (Secrets) ---
try:
    # משיכת המפתח מה-Secrets שהגדרת ב-Streamlit
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3-pro-preview')
except Exception as e:
    st.error("שגיאה בחיבור ל-API. וודא שהגדרת את GEMINI_API_KEY ב-Secrets של Streamlit.")
    st.stop()

st.title("📅 מתכנן המטלות החכם לסטודנטים")
st.write("ברוכים הבאים! הכלי שיעזור לכם לפרק את המטלות של סמסטר א' ללו\"ז ריאלי ביומן.")

# --- 3. סרגל צד: אילוצים מפורטים ---
with st.sidebar:
    st.header("⚙️ הגדרות ואילוצים")
    
    st.subheader("🗓️ אילוצי ימים ושעות")
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("בחר ימים שבהם יש לך לו\"ז קבוע:", days_week)
    
    day_constraints = {}
    for day in selected_days:
        st.write(f"**שעות תפוסות ביום {day}:**")
        start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"start_{day}")
        end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"end_{day}")
        day_constraints[day] = f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"

    st.divider()
    daily_max_hours = st.slider("מקסימום שעות עבודה ביום על מטלות:", 1, 10, 4)
    additional_info = st.text_area("הערות נוספות ל-AI (למשל: 'בימי חמישי אני מעדיף לעבוד בערב')")

# --- 4. הזנת מטלות (Sheets או רשימה) ---
st.header("📝 הזנת מטלות ותתי-משימות")

input_method = st.radio("בחר שיטת הזנה:", ["רשימה ידנית", "קישור לגוגל שיטס / רשימה מרוכזת"])

if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if input_method == "רשימה ידנית":
    with st.expander("➕ הוספת מטלה חדשה", expanded=True):
        t_name = st.text_input("שם המטלה")
        c1, c2 = st.columns(2)
        with c1:
            t_time = st.number_input("סך שעות עבודה", min_value=1.0, step=0.5)
        with c2:
            t_deadline = st.date_input("תאריך דדליין", value=datetime.date.today() + datetime.timedelta(days=14))
        t_subs = st.text_area("תתי משימות (הפרד בשורות)")
        
        if st.button("הוסף לרשימה"):
            if t_name:
                st.session_state.tasks.append({
                    "name": t_name, "time": t_time, 
                    "deadline": t_deadline.strftime("%Y-%m-%d"), "subtasks": t_subs
                })
                st.rerun()
else:
    sheets_input = st.text_area("הדבק כאן קישור לגוגל שיטס או העתק-הדבק של טבלת המטלות שלך:")
    if st.button("עדכן נתוני טבלה"):
        st.info("ה-AI ינתח את הטקסט/קישור שהזנת בזמן חישוב הלו\"ז.")

# --- 5. הצגת הרשימה הקיימת ---
if st.session_state.tasks:
    st.subheader("📋 המטלות שלך:")
    for idx, task in enumerate(st.session_state.tasks):
        cols = st.columns([8, 1])
        cols[0].write(f"🔹 **{task['name']}** - {task['time']} שעות (עד {task['deadline']})")
        if cols[1].button("🗑️", key=f"del_{idx}"):
            st.session_state.tasks.pop(idx)
            st.rerun()

# --- 6. מנוע ה-AI: חישוב לו"ז חכם ---
st.divider()
if st.button("🚀 חשב לו\"ז חכם (פלט טבלאי)"):
    tasks_to_send = st.session_state.tasks if input_method == "רשימה ידנית" else sheets_input
    
    if not tasks_to_send:
        st.warning("נא להזין מטלות לפני החישוב.")
    else:
        prompt = f"""
        אתה מומחה לניהול זמן לסטודנטים. עליך לבנות תוכנית עבודה חודשית.
        המטלות והזמנים: {tasks_to_send}
        
        אילוצים קבועים (שעות תפוסות): {day_constraints}
        מגבלת שעות עבודה על מטלות ביום: {daily_max_hours} שעות.
        מידע נוסף: {additional_info}
        
        הנחיות לפלט:
        1. הצג את התוכנית בטבלה מסודרת בעברית עם העמודות: תאריך, מטלה, פירוט תת-משימה, שעות עבודה.
        2. לאחר הטבלה, הוסף בלוק קוד JSON בתוך ```json ``` המכיל רשימת אובייקטים עם השדות: title, date (YYYY-MM-DD), start_time (HH:MM).
        """
        
        with st.spinner("ה-AI בונה לך את הלו\"ז האופטימלי..."):
            response = model.generate_content(prompt)
            res_text = response.text
            
            # הצגת הפלט (הטבלה תופיע כאן כחלק מהטקסט)
            st.markdown(res_text)
            
            # ניסיון חילוץ ה-JSON לייצוא ICS
            try:
                if "```json" in res_text:
                    json_part = res_text.split("```json")[1].split("```")[0].strip()
                    st.session_state.last_sched = json.loads(json_part)
                    st.success("הלו\"ז מוכן לייצוא ליומן גוגל!")
            except Exception as e:
                st.info("הלו\"ז הוצג, אך לא ניתן היה ליצור קובץ יומן אוטומטי מהפלט הזה.")

# --- 7. ייצוא ליומן ---
if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item['start_time']}:00"
        c.events.add(e)
    
    st.download_button("💾 הורד את הלו\"ז ליומן גוגל (ICS)", str(c), file_name="my_schedule.ics")