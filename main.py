import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import re
from groq import Groq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Setup ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

client = Groq(api_key=GROQ_API_KEY)

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="StudyPlanner AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- UI & Styling ----------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def apply_theme(dark_mode):
    if dark_mode:
        st.markdown('<body class="dark-mode">', unsafe_allow_html=True)
    else:
        st.markdown('<body class="light-mode">', unsafe_allow_html=True)

# ---------------- PDF Text Extraction ----------------
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# ---------------- Groq AI Integration ----------------
def generate_study_plan(syllabus_text, exam_date, course_name, all_courses_info, available_days):
    """Generate study plan using Groq AI, considering all courses and available days"""
    
    # Create context about other courses
    other_courses_context = ""
    if len(all_courses_info) > 1:
        other_courses_context = "\n\nOther courses and their exam dates:\n"
        for course_info in all_courses_info:
            if course_info['name'] != course_name:
                other_courses_context += f"- {course_info['name']}: Exam on {course_info['exam_date']}\n"
        other_courses_context += "\nPlease consider these other courses when creating the study plan to avoid conflicts and ensure balanced preparation.\n"
    
    prompt = f"""You are an expert academic tutor creating a comprehensive study plan.

Course: {course_name}
Exam Date: {exam_date}
Available Study Days: {available_days} days
Syllabus Content: {syllabus_text[:2500]}

{other_courses_context}

Create a detailed {available_days}-day study plan. For each day, provide a list of tasks and their estimated duration.

**FORMAT:**
Day 1:
- Task 1 (e.g., "Review Chapter 1") - 1.5 hours
- Task 2 (e.g., "Practice problems for Chapter 1") - 2 hours

Day 2:
- Task 1 (e.g., "Read Chapter 2") - 1 hour
...

**IMPORTANT GUIDELINES:**
- Create exactly {available_days} days of study content.
- Provide a time estimate in hours for each task (e.g., "1.5 hours", "45 minutes").
- Ensure the total study time for any single day does not exceed 4 hours.
- Be specific and actionable in your task descriptions.
- Prioritize the most important topics based on the syllabus.
- Do NOT schedule any study on the actual exam date.
- Format each day exactly as "Day X:" followed by a list of tasks.
"""

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=2000,
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating study plan: {e}")
        return f"❌ Error generating study plan for {course_name}: {e}"

# ---------------- Calendar Logic with Conflict Resolution ----------------
def _parse_time_string(time_str):
    """Parse time string like '1.5 hours' or '45 minutes' into hours (float)"""
    time_str = time_str.lower()
    total_hours = 0

    # Find hours
    hour_match = re.search(r'(\d+\.?\d*)\s*hours?', time_str)
    if hour_match:
        total_hours += float(hour_match.group(1))

    # Find minutes
    min_match = re.search(r'(\d+)\s*minutes?', time_str)
    if min_match:
        total_hours += float(min_match.group(1)) / 60

    return total_hours

def parse_study_plan_to_calendar_items(plan_text, course_name, exam_date, available_days):
    """Parse study plan text into structured calendar items"""
    items = []
    
    # Regex to match each day's content
    day_pattern = r'Day\s*(\d+):\s*(.*?)(?=Day\s*\d+:|$)'
    day_matches = re.findall(day_pattern, plan_text, re.DOTALL | re.IGNORECASE)
    
    for day_num, day_content in day_matches:
        day_number = int(day_num)
        if day_number > available_days:
            continue
            
        study_date = exam_date - timedelta(days=(available_days - day_number + 1))

        # Regex to match tasks within a day's content
        task_pattern = r'-\s*(.*?)\s*-\s*([\d\.]+\s*(?:hours?|minutes?))'
        task_matches = re.findall(task_pattern, day_content, re.IGNORECASE)

        for task_desc, time_str in task_matches:
            duration = _parse_time_string(time_str)
            if duration == 0:
                # Default duration if parsing fails
                duration = 1.0
            
            items.append({
                "date": study_date,
                "course": course_name,
                "task": task_desc.strip(),
                "duration": duration,  # in hours
                "day": f"Day {day_number}",
                "days_until_exam": (exam_date - study_date).days,
                "exam_date": exam_date
            })

    return items

def resolve_calendar_conflicts(all_calendar_items, max_daily_hours=4.0):
    """Resolve conflicts by ensuring daily study load does not exceed a max limit."""
    if not all_calendar_items:
        return all_calendar_items, []

    # Group items by date
    date_groups = {}
    for item in all_calendar_items:
        date_str = item['date'].strftime('%Y-%m-%d')
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(item)

    resolved_items = []
    conflicts_resolved = []
    
    # Sort dates to process chronologically
    sorted_dates = sorted(date_groups.keys())

    for date_str in sorted_dates:
        items_on_day = sorted(date_groups[date_str], key=lambda x: x['days_until_exam'])
        daily_hours = 0

        for item in items_on_day:
            if daily_hours + item['duration'] <= max_daily_hours:
                # No conflict, add to resolved items
                resolved_items.append(item)
                daily_hours += item['duration']
            else:
                # Conflict: try to reschedule
                conflicts_resolved.append({
                    'date': date_str,
                    'courses': [item['course']],
                    'original_count': 1 # Simplified for new logic
                })

                rescheduled = False
                for day_offset in range(1, 8): # Try to move up to a week
                    # Try earlier days first
                    for sign in [-1, 1]:
                        new_date = item['date'] + timedelta(days=day_offset * sign)

                        # Don't reschedule to a date after the exam
                        if new_date >= item['exam_date']:
                            continue

                        new_date_str = new_date.strftime('%Y-%m-%d')

                        # Calculate hours on the potential new day
                        new_day_hours = sum(i['duration'] for i in resolved_items if i['date'] == new_date)

                        if new_day_hours + item['duration'] <= max_daily_hours:
                            modified_item = item.copy()
                            modified_item['date'] = new_date
                            modified_item['task'] = f"[Rescheduled] {item['task']}"
                            resolved_items.append(modified_item)
                            rescheduled = True
                            break
                    if rescheduled:
                        break

                if not rescheduled:
                    # Could not reschedule, mark as conflict
                    conflicted_item = item.copy()
                    conflicted_item['task'] = f"[CONFLICT - Overloaded] {item['task']}"
                    resolved_items.append(conflicted_item)

    return sorted(resolved_items, key=lambda x: x['date']), conflicts_resolved

# ---------------- Enhanced Calendar Visualization ----------------
def create_calendar_visualization(calendar_items, dark_mode=False):
    """Create an enhanced calendar visualization using Plotly"""
    if not calendar_items:
        return None
    
    df = pd.DataFrame(calendar_items)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create a timeline chart
    fig = go.Figure()
    
    # Get unique courses and assign colors
    courses = df['course'].unique()
    colors = px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Vivid
    color_map = dict(zip(courses, colors[:len(courses)]))
    
    # Add timeline traces for each course
    for course in courses:
        course_data = df[df['course'] == course].sort_values('date')
        
        fig.add_trace(go.Scatter(
            x=course_data['date'],
            y=[course] * len(course_data),
            mode='markers+lines',
            name=course,
            marker=dict(
                color=color_map[course],
                size=15,
                line=dict(width=2, color='white' if not dark_mode else '#1E1E1E')
            ),
            line=dict(color=color_map[course], width=3),
            text=[f"<b>{row['day']}</b>: {row['task']}<br><em>{row['duration']:.2f} hours</em>" for _, row in course_data.iterrows()],
            hovertemplate="<b>%{y}</b><br>" +
                         "Date: %{x|%A, %b %d}<br>" +
                         "Info: %{text}<br>" +
                         "<extra></extra>"
        ))
        
        # Add exam date markers
        exam_date = course_data['exam_date'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[exam_date],
            y=[course],
            mode='markers',
            name=f'{course} Exam',
            marker=dict(
                color='#FF4136' if not dark_mode else '#FF6B6B',
                size=20,
                symbol='star',
                line=dict(width=2, color='white' if not dark_mode else '#1E1E1E')
            ),
            text=f"<b>EXAM: {course}</b>",
            hovertemplate="<b>%{text}</b><br>" +
                         "Date: %{x|%A, %b %d}<br>" +
                         "<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title="📅 Study Schedule Timeline",
        xaxis_title="Date",
        yaxis_title="",
        height=max(400, len(courses) * 80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            size=12,
            color= '#1E1E1E' if not dark_mode else '#EAEAEA'
        ),
        title_font_size=20,
        xaxis=dict(
            type='date',
            tickformat='%b %d',
            showgrid=True,
            gridcolor= '#E5E7EB' if not dark_mode else '#374151',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=False,
        ),
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def show_calendar_view(calendar_items, conflicts_resolved, dark_mode=False):
    """Display enhanced calendar view with better graphics"""
    if not calendar_items:
        st.warning("📅 No calendar items to display.")
        return

    # Show conflict resolution summary
    if conflicts_resolved:
        st.markdown('<div class="conflict-warning">', unsafe_allow_html=True)
        st.markdown("⚠️ **Schedule Overload Detected:**")
        st.markdown("Some study sessions were rescheduled or marked as conflicted to avoid overloading your schedule.")
        for conflict in conflicts_resolved:
            courses_str = ", ".join(conflict['courses'])
            st.markdown(f"• **{conflict['date']}**: Overload with course(s) **{courses_str}**.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown("✅ **No Schedule Conflicts Detected** - Your study plan is optimally organized!")
        st.markdown('</div>', unsafe_allow_html=True)

    df = pd.DataFrame(calendar_items)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create timeline visualization
    fig = create_calendar_visualization(calendar_items, dark_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Create metrics dashboard
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📚 Total Courses", len(df['course'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Study Sessions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        # Get the latest exam date
        latest_exam = df['exam_date'].max()
        days_left = (latest_exam - datetime.now().date()).days
        st.metric("⏰ Days Until Last Exam", max(0, days_left))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        next_session = df[df['date'].dt.date >= datetime.now().date()]
        if len(next_session) > 0:
            next_date = next_session.iloc[0]['date'].strftime('%b %d')
        else:
            next_date = "Completed"
        st.metric("🎯 Next Study Session", next_date)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed day-by-day breakdown
    st.markdown('<div class="detailed-schedule">', unsafe_allow_html=True)
    st.markdown("### 📋 Detailed Study Schedule")
    
    for course in df['course'].unique():
        course_data = df[df['course'] == course].sort_values('date')
        
        st.markdown(f'<div class="course-card">', unsafe_allow_html=True)
        st.markdown(f"#### 📖 {course}")
        
        exam_date = course_data['exam_date'].iloc[0]
        st.markdown(f'<div class="exam-date">🎯 Exam: {exam_date.strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
        
        for _, row in course_data.iterrows():
            task_class = "study-task"
            if "[CONFLICT]" in row['task']:
                task_class += " conflict-task"
            elif "[Rescheduled]" in row['task']:
                task_class += " rescheduled-task"
            
            st.markdown(f'''
            <div class="{task_class}">
                <strong>{row['day']} ({row['date'].strftime('%A, %B %d')})</strong>
                <span style="float: right; background-color: #eee; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                    🕒 {row['duration']:.2f} hours
                </span>
                <br>
                {row['task']}
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Main Streamlit App ----------------
def main():
    # --- Initial Setup ---
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    load_css("styles.css")
    apply_theme(st.session_state.dark_mode)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🔧 Settings")

        # Dark Mode Toggle
        st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

        st.markdown("**API Configuration**")
        groq_key = st.text_input("Groq API Key", type="password", 
                                placeholder="Enter your Groq API key here")
        
        if groq_key:
            global client
            client = Groq(api_key=groq_key)
        elif GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            groq_key = GROQ_API_KEY
            client = Groq(api_key=groq_key)
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.info("""
        1. Upload your syllabus PDFs
        2. Set exam dates for each course
        3. Generate AI-powered study plans
        4. View your organized calendar
        5. Conflicts are automatically resolved!
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.success("""
        This app uses **Groq AI** to create personalized study schedules.
        - ✅ Automatic conflict resolution
        - ✅ Adaptive plan length
        - ✅ Enhanced timeline visualization
        """)

    # --- Main Header ---
    st.markdown('''
    <div class="main-header">
        <h1>📚 StudyPlanner AI</h1>
        <p>Plan your test season across multiple subjects with AI-powered schedules</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "📁 Upload Syllabus PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload PDF files containing your course syllabi"
    )
    
    # --- Session State Initialization ---
    if "exam_dates" not in st.session_state:
        st.session_state.exam_dates = {}
    if "generated_plans" not in st.session_state:
        st.session_state.generated_plans = {}
    if "calendar_items" not in st.session_state:
        st.session_state.calendar_items = []
    if "conflicts_resolved" not in st.session_state:
        st.session_state.conflicts_resolved = []
    
    # --- Main Logic ---
    if uploaded_files:
        st.markdown("### 📅 Set Exam Dates")
        
        # Layout for exam dates
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**📚 {file.name}**")
            with col2:
                st.session_state.exam_dates[file.name] = st.date_input(
                    f"Exam date", key=f"exam_date_{i}", min_value=datetime.now().date(),
                    help=f"Select exam date for {file.name}"
                )
        
        st.markdown("---")
        
        # Generate plans button
        if st.button("✨ Generate Study Plans", type="primary", use_container_width=True):
            if not groq_key:
                st.error("⚠️ Please provide a valid Groq API key in the sidebar!")
                return
            
            st.session_state.generated_plans = {}
            st.session_state.calendar_items = []
            st.session_state.conflicts_resolved = []
            
            all_courses_info = [{'name': f.name.replace('.pdf', ''),
                                 'exam_date': st.session_state.exam_dates.get(f.name).strftime("%B %d, %Y"),
                                 'file': f} for f in uploaded_files if st.session_state.exam_dates.get(f.name)]
            
            progress_bar = st.progress(0, text="🚀 Generating plans...")
            all_calendar_items = []
            
            for i, file in enumerate(uploaded_files):
                exam_date = st.session_state.exam_dates.get(file.name)
                if not exam_date:
                    st.warning(f"⚠️ Please select a date for {file.name}")
                    continue
                
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analyzing {file.name}...")
                
                with st.spinner(f"🤖 Generating plan for {file.name}..."):
                    syllabus_text = extract_text_from_pdf(file)
                    if not syllabus_text:
                        st.error(f"❌ Could not extract text from {file.name}")
                        continue
                    
                    course_name = file.name.replace('.pdf', '')
                    days_until_exam = (exam_date - datetime.now().date()).days
                    available_days = min(7, max(1, days_until_exam))
                    
                    raw_plan = generate_study_plan(
                        syllabus_text, exam_date.strftime("%B %d, %Y"), course_name,
                        all_courses_info, available_days
                    )
                    structured_plan = parse_study_plan_to_calendar_items(
                        raw_plan, course_name, exam_date, available_days
                    )
                    
                    all_calendar_items.extend(structured_plan)
                    st.session_state.generated_plans[course_name] = raw_plan
            
            if all_calendar_items:
                with st.spinner("🔄 Resolving schedule conflicts..."):
                    resolved_items, conflicts = resolve_calendar_conflicts(all_calendar_items)
                    st.session_state.calendar_items = resolved_items
                    st.session_state.conflicts_resolved = conflicts
            
            progress_bar.empty()
            st.success("✅ Study plans generated successfully!")
    
    # --- Display Results ---
    if st.session_state.generated_plans:
        st.markdown("## 📘 Study Plans")
        
        tab1, tab2 = st.tabs(["📅 Calendar View", "📝 Detailed Plans"])
        
        with tab1:
            show_calendar_view(st.session_state.calendar_items,
                               st.session_state.conflicts_resolved,
                               st.session_state.dark_mode)
        
        with tab2:
            for course, plan in st.session_state.generated_plans.items():
                with st.expander(f"📄 {course} Study Plan", expanded=True):
                    st.markdown(plan)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Made with ❤️ by YBG | Powered by Groq AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()