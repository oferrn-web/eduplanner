import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json

# --- הגדרות דף ו-RTL ---
st.set_page_config(page_title="מתכנן מטלות חכם", layout="wide")
st.markdown("<style> .stApp { direction: RTL; text-align: right; } </style>", unsafe_allow_html=True)

# --- הגדרת המודל (תיקון ה-404) ---
API_KEY = "AIzaSyC7kS7dFoqY2XmQtraiApFOGma81j7_2Gw" 

def get_model(key):
    try:
        genai.configure(api_key=key)
        # שינוי ל-flash כדי להבטיח תאימות ושרידות
        return genai.GenerativeModel('gemini-3-pro-preview')
    except Exception as e:
        st.error(f"שגיאת התחברות: {e}")
        return None

model = get_model(API_KEY)

st.title("📅 מתכנן המטלות החכם שלי")

# --- סרגל צד: אילוצים ---
with st.sidebar:
    st.header("⚙️ הגדרות ואילוצים")
    side_constraints = st.text_area("אילוצים קבועים (לימודים/עבודה):", placeholder="למשל: ימי ג' התנסות מעשית")
    side_blocked = st.multiselect("ימים חסומים:", ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"])
    side_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 8, 4)
    
    if side_constraints or side_blocked:
        st.success("✅ האילוצים נשמרו וישלחו ל-AI")

# --- ניהול מטלות (Session State) ---
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

with st.expander("➕ הוספת מטלה חדשה", expanded=True):
    t_name = st.text_input("שם המטלה")
    col1, col2 = st.columns(2)
    with col1:
        t_time = st.number_input("שעות עבודה נדרשות", min_value=1.0, step=0.5)
    with col2:
        t_deadline = st.date_input("דדליין", value=datetime.date.today() + datetime.timedelta(days=10))
    t_subs = st.text_area("תתי משימות (הפרד בפסיק)")

    if st.button("הוסף מטלה"):
        if t_name:
            st.session_state.tasks.append({
                "name": t_name, "time": t_time, 
                "deadline": t_deadline.strftime("%Y-%m-%d"), "subtasks": t_subs
            })
            st.rerun()

# --- הצגת המטלות ---
if st.session_state.tasks:
    st.subheader("📋 רשימת המטלות הנוכחית")
    for idx, task in enumerate(st.session_state.tasks):
        c1, c2, c3 = st.columns([1, 8, 1])
        with c1:
            if st.button("🔼", key=f"u{idx}") and idx > 0:
                st.session_state.tasks[idx], st.session_state.tasks[idx-1] = st.session_state.tasks[idx-1], st.session_state.tasks[idx]
                st.rerun()
        with c2:
            st.info(f"**{task['name']}** | {task['time']} שעות | דדליין: {task['deadline']}")
        with c3:
            if st.button("🗑️", key=f"d{idx}"):
                st.session_state.tasks.pop(idx)
                st.rerun()

# --- הפעלת ה-AI (שילוב האילוצים) ---
st.divider()
if st.button("🚀 חשב לו''ז חכם (AI)"):
    if not model:
        st.error("לא הוגדר מפתח API תקין.")
    elif not st.session_state.tasks:
        st.warning("הוסף לפחות מטלה אחת.")
    else:
        # כאן אנחנו מזריקים את האילוצים מהסרגל צד לתוך הפרומפט
        prompt = f"""
        פזר את המטלות האלו ביומן: {st.session_state.tasks}.
        
        אילוצים חשובים:
        1. אילוצים קבועים: {side_constraints}
        2. ימים חסומים שבהם אסור לשבץ עבודה: {', '.join(side_blocked)}
        3. אל תשבץ יותר מ-{side_max_hours} שעות עבודה ביום אחד.
        
        החזר תשובה בעברית ובסופה בלוק JSON עם השדות: title, date (YYYY-MM-DD), start_time (HH:MM).
        """
        
        with st.spinner("ה-AI מנתח עומסים ואילוצים..."):
            try:
                response = model.generate_content(prompt)
                res_text = response.text
                st.markdown(res_text)
                
                # ניסיון חילוץ JSON לייצוא
                if "```json" in res_text:
                    js = res_text.split("```json")[1].split("```")[0].strip()
                    st.session_state.last_sched = json.loads(js)
                    st.success("הלו''ז מוכן לייצוא!")
            except Exception as e:
                st.error(f"שגיאת AI: {e}")

# --- ייצוא ליומן ---
if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item['start_time']}:00"
        c.events.add(e)
    st.download_button("💾 הורד את הלו''ז ליומן גוגל (ICS)", str(c), file_name="smart_calendar.ics")