import streamlit as st
import google.generativeai as genai
from ics import Calendar, Event
import datetime
import json

# --- 1. הגדרות דף ועיצוב RTL ---
st.set_page_config(page_title="מתכנן המטלות החכם", layout="wide")
st.markdown("<style> .stApp { direction: RTL; text-align: right; } </style>", unsafe_allow_html=True)

# --- 2. חיבור ל-AI (Secrets) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-3-pro-preview')
except Exception as e:
    st.error("לא נמצא מפתח ב-Secrets. וודא שהגדרת GEMINI_API_KEY.")
    st.stop()

st.title("📅 מתכנן המטלות החכם")

# --- 3. סרגל צד: אילוצים עם כותרות ---
with st.sidebar:
    st.header("⚙️ הגדרות מערכת ואילוצים")
    st.write("הגדר זמנים שבהם **אסור** ל-AI לשבץ מטלות.")
    
    days_week = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
    selected_days = st.multiselect("בחר ימים עם אילוץ קבוע:", days_week)
    
    day_constraints = []
    for day in selected_days:
        with st.expander(f"אילוץ ליום {day}", expanded=True):
            headline = st.text_input(f"כותרת האילוץ (למשל: עבודה)", key=f"h_{day}")
            col1, col2 = st.columns(2)
            with col1:
                start_t = st.time_input(f"התחלה", datetime.time(8, 0), key=f"s_{day}")
            with col2:
                end_t = st.time_input(f"סיום", datetime.time(16, 0), key=f"e_{day}")
            day_constraints.append({
                "day": day,
                "title": headline,
                "hours": f"{start_t.strftime('%H:%M')} עד {end_t.strftime('%H:%M')}"
            })

    daily_max_hours = st.slider("מקסימום שעות עבודה ביום:", 1, 10, 4)

    if st.button("🗑️ ניקוי כל הנתונים"):
        st.session_state.extracted_tasks = []
        st.rerun()

# --- 4. הזנת מטלות והדרכה ---
st.header("📝 הזנת מטלות")

# תיבת הדרכה להעתקה משיטס
st.info("""
💡 **טיפ להצלחה:** הדרך הטובה ביותר היא לסמן את הטבלה שלך בגוגל שיטס (התאים עצמם), 
להעתיק (Ctrl+C) ולהדביק בתיבה מטה. ה-AI יזהה את המבנה אוטומטית.
""")

if 'extracted_tasks' not in st.session_state:
    st.session_state.extracted_tasks = []

raw_input = st.text_area("הדבק כאן את תוכן הטבלה או קישור:")

if st.button("🔍 חלץ מטלות"):
    with st.spinner("מזהה מטלות..."):
        parse_prompt = f"חלץ מרשימה זו רק את שמות המטלות העיקריות. החזר אך ורק רשימת JSON של שמות המטלות: {raw_input}"
        try:
            res = model.generate_content(parse_prompt)
            clean_json = res.text.replace("```json", "").replace("```", "").strip()
            names = json.loads(clean_json)
            st.session_state.extracted_tasks = [{"name": n, "time": 2.0, "deadline": str(datetime.date.today()), "subs": ""} for n in names]
            st.rerun()
        except:
            st.error("שגיאה בזיהוי. נסה להדביק טקסט ישיר מהטבלה.")

# --- 5. עריכת פרטי המטלות ---
if st.session_state.extracted_tasks:
    st.subheader("✍️ הגדר שעות ותתי-משימות")
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
            st.session_state.extracted_tasks[idx]['subs'] = st.text_area("תתי משימות", value=task['subs'], key=f"s_{idx}")

# --- 6. יצירת הלו"ז הטבלאי (עם תיקון לוגיקת האילוצים) ---
st.divider()
if st.button("🚀 חשב לו''ז חכם"):
    if not st.session_state.extracted_tasks:
        st.warning("נא להזין מטלות.")
    else:
        # הנחיה מחמירה ל-AI לגבי האילוצים
        final_prompt = f"""
        אתה מנהל לו"ז מקצועי. המטרה: לשבץ את המטלות הבאות בזמנים הפנויים בלבד.
        מטלות לשיבוץ: {st.session_state.tasks}
        
        חסמים (זמנים שבהם אסור לשבץ כלום - אלו זמנים תפוסים):
        {day_constraints}
        
        חוקים נוקשים:
        1. חל איסור מוחלט לשבץ עבודה על מטלות בזמני החסמים שצוינו לעיל. החסמים הם "שטח מת".
        2. בצע את הפיזור החל מהיום ועד לדדליין של כל מטלה.
        3. אל תעבור את המכסה של {daily_max_hours} שעות עבודה ביום.
        
        החזר:
        1. טבלה בעברית (תאריך, מטלה, תת-משימה, שעות עבודה).
        2. בלוק JSON בסוף עם title, date (YYYY-MM-DD), start_time.
        """
        with st.spinner("מחשב לו''ז ומדלג על אילוצים..."):
            res = model.generate_content(final_prompt).text
            st.markdown(res)
            if "```json" in res:
                st.session_state.last_sched = json.loads(res.split("```json")[1].split("```")[0].strip())

# --- 7. ייצוא ---
if 'last_sched' in st.session_state:
    c = Calendar()
    for item in st.session_state.last_sched:
        e = Event()
        e.name = item['title']
        e.begin = f"{item['date']} {item.get('start_time', '09:00')}:00"
        c.events.add(e)
    st.download_button("💾 הורד ליומן גוגל", str(c), file_name="planner.ics")