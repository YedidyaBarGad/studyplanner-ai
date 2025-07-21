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

# ---------------- Custom CSS for Better Graphics ----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .course-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .study-task {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    
    .study-task strong {
        color: #2c3e50;
    }
    
    .detailed-schedule {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .detailed-schedule h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .exam-date {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .conflict-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

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

Create a detailed {available_days}-day study plan with the following format:
Day 1: [Specific topics and tasks with time estimates]
Day 2: [Specific topics and tasks with time estimates]
...
Day {available_days}: [Final review and exam preparation]

IMPORTANT GUIDELINES:
- Create exactly {available_days} days of study content
- Make each day's tasks specific and actionable
- Include estimated time for each task (e.g., "2 hours", "45 minutes")
- Consider workload balance across all courses
- Include variety in study methods (reading, practice, review)
- Build in breaks and rest periods
- Make the final day focused on review and confidence building
- Do NOT schedule any study on the actual exam date
- Prioritize the most important topics first

Format each day exactly as "Day X: [content]" for proper parsing.
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
def parse_study_plan_to_calendar_items(plan_text, course_name, exam_date, available_days):
    """Parse study plan text into structured calendar items"""
    items = []
    
    # Better regex to match day patterns
    day_pattern = r'Day\s*(\d+):\s*(.*?)(?=Day\s*\d+:|$)'
    matches = re.findall(day_pattern, plan_text, re.DOTALL | re.IGNORECASE)
    
    for day_num, task_text in matches:
        day_number = int(day_num)
        if day_number <= available_days:
            # Calculate study date: exam_date - (available_days - day_number + 1) days
            study_date = exam_date - timedelta(days=(available_days - day_number + 1))
            
            # Clean up task text
            task_text = task_text.strip().replace('\n', ' ').replace('  ', ' ')
            
            items.append({
                "date": study_date,
                "course": course_name,
                "task": task_text,
                "day": f"Day {day_number}",
                "days_until_exam": (exam_date - study_date).days,
                "exam_date": exam_date
            })
    
    return items

def resolve_calendar_conflicts(all_calendar_items):
    """Resolve conflicts in calendar items to prevent overlapping studies"""
    if not all_calendar_items:
        return all_calendar_items, []
    
    # Sort by date
    sorted_items = sorted(all_calendar_items, key=lambda x: x['date'])
    
    # Group by date
    date_groups = {}
    for item in sorted_items:
        date_str = item['date'].strftime('%Y-%m-%d')
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(item)
    
    resolved_items = []
    conflicts_resolved = []
    
    for date_str, items in date_groups.items():
        if len(items) == 1:
            # No conflict
            resolved_items.append(items[0])
        else:
            # Conflict detected - distribute across available days
            conflicts_resolved.append({
                'date': date_str,
                'courses': [item['course'] for item in items],
                'original_count': len(items)
            })
            
            # Strategy: Keep the item with the closest exam date on the original day
            # Move others to nearby days
            items_by_urgency = sorted(items, key=lambda x: x['days_until_exam'])
            
            # Keep the most urgent item on the original day
            resolved_items.append(items_by_urgency[0])
            
            # Redistribute others
            for i, item in enumerate(items_by_urgency[1:], 1):
                # Try to move to a nearby day (prefer earlier days)
                for day_offset in [1, -1, 2, -2, 3, -3]:
                    new_date = item['date'] + timedelta(days=day_offset)
                    
                    # Check if new date is available and not an exam day
                    new_date_str = new_date.strftime('%Y-%m-%d')
                    date_available = True
                    
                    # Check if it's an exam day for any course
                    for existing_item in all_calendar_items:
                        if existing_item['exam_date'] == new_date:
                            date_available = False
                            break
                    
                    # Check if the new date already has fewer than 2 items
                    if date_available and len([x for x in resolved_items if x['date'] == new_date]) < 2:
                        modified_item = item.copy()
                        modified_item['date'] = new_date
                        modified_item['task'] = f"[Rescheduled] {item['task']}"
                        resolved_items.append(modified_item)
                        break
                else:
                    # If no suitable day found, keep original but mark as conflicted
                    conflicted_item = item.copy()
                    conflicted_item['task'] = f"[CONFLICT] {item['task']}"
                    resolved_items.append(conflicted_item)
    
    return resolved_items, conflicts_resolved

# ---------------- Enhanced Calendar Visualization ----------------
def create_calendar_visualization(calendar_items):
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
    colors = px.colors.qualitative.Set3[:len(courses)]
    color_map = dict(zip(courses, colors))
    
    # Add timeline traces for each course
    for course in courses:
        course_data = df[df['course'] == course].sort_values('date')
        
        # Create timeline for this course
        fig.add_trace(go.Scatter(
            x=course_data['date'],
            y=[course] * len(course_data),
            mode='markers+lines',
            name=course,
            marker=dict(
                color=color_map[course],
                size=15,
                line=dict(width=2, color='white')
            ),
            line=dict(color=color_map[course], width=3),
            text=[f"{row['day']}: {row['task'][:50]}..." for _, row in course_data.iterrows()],
            hovertemplate="<b>%{y}</b><br>" +
                         "Date: %{x}<br>" +
                         "Task: %{text}<br>" +
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
                color='red',
                size=20,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            text=f"EXAM: {course}",
            hovertemplate="<b>%{y} EXAM</b><br>" +
                         "Date: %{x}<br>" +
                         "<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title="📅 Study Schedule Timeline",
        xaxis_title="Date",
        yaxis_title="Course",
        height=max(400, len(courses) * 100),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=20,
        xaxis=dict(
            type='date',
            tickformat='%b %d',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='closest'
    )
    
    return fig

def show_calendar_view(calendar_items, conflicts_resolved):
    """Display enhanced calendar view with better graphics"""
    if not calendar_items:
        st.warning("📅 No calendar items to display.")
        return
    
    # Show conflict resolution summary
    if conflicts_resolved:
        st.markdown('<div class="conflict-warning">', unsafe_allow_html=True)
        st.markdown("⚠️ **Schedule Conflicts Detected and Resolved:**")
        for conflict in conflicts_resolved:
            courses_str = ", ".join(conflict['courses'])
            st.markdown(f"• {conflict['date']}: {courses_str} ({conflict['original_count']} courses)")
        st.markdown("Some study sessions have been automatically rescheduled to avoid conflicts.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown("✅ **No Schedule Conflicts Detected** - Your study plan is optimally organized!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(calendar_items)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create timeline visualization
    fig = create_calendar_visualization(calendar_items)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Create metrics dashboard
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
                <strong>{row['day']} ({row['date'].strftime('%A, %B %d')})</strong><br>
                {row['task']}
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Main Streamlit App ----------------
def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>📚 StudyPlanner AI</h1>
        <p>Plan your test season across multiple subjects with AI-powered study schedules</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Settings")
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
        st.markdown("""
        1. Upload your syllabus PDFs
        2. Set exam dates for each course
        3. Generate AI-powered study plans
        4. View your organized calendar
        5. Conflicts are automatically resolved!
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses **Groq AI** to create personalized study schedules based on your course syllabi.
        
        **New Features:**
        - ✅ Automatic conflict resolution
        - ✅ No study on exam days
        - ✅ Adaptive plan length
        - ✅ Enhanced timeline visualization
        """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "📁 Upload Syllabus PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload PDF files containing your course syllabi"
    )
    
    if "exam_dates" not in st.session_state:
        st.session_state.exam_dates = {}
    
    if "generated_plans" not in st.session_state:
        st.session_state.generated_plans = {}
    
    if "calendar_items" not in st.session_state:
        st.session_state.calendar_items = []
    
    if "conflicts_resolved" not in st.session_state:
        st.session_state.conflicts_resolved = []
    
    if uploaded_files:
        st.markdown("### 📅 Set Exam Dates")
        
        # Create a nice layout for exam dates
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📚 {file.name}**")
            
            with col2:
                selected_date = st.date_input(
                    f"Exam date",
                    key=f"exam_date_{i}",
                    min_value=datetime.now().date(),
                    help=f"Select exam date for {file.name}"
                )
                st.session_state.exam_dates[file.name] = selected_date
        
        st.markdown("---")
        
        # Generate plans button
        if st.button("✨ Generate Study Plans", type="primary", use_container_width=True):
            if not groq_key:
                st.error("⚠️ Please provide a valid Groq API key in the sidebar!")
                return
            
            st.session_state.generated_plans = {}
            st.session_state.calendar_items = []
            st.session_state.conflicts_resolved = []
            
            # Prepare all courses info for better planning
            all_courses_info = []
            for file in uploaded_files:
                exam_date = st.session_state.exam_dates.get(file.name)
                if exam_date:
                    all_courses_info.append({
                        'name': file.name.replace('.pdf', ''),
                        'exam_date': exam_date.strftime("%B %d, %Y"),
                        'file': file
                    })
            
            progress_bar = st.progress(0)
            all_calendar_items = []
            
            for i, file in enumerate(uploaded_files):
                exam_date = st.session_state.exam_dates.get(file.name)
                if not exam_date:
                    st.warning(f"⚠️ Please select a date for {file.name}")
                    continue
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                with st.spinner(f"🤖 Generating study plan for {file.name}..."):
                    syllabus_text = extract_text_from_pdf(file)
                    if not syllabus_text:
                        st.error(f"❌ Could not extract text from {file.name}")
                        continue
                    
                    course_name = file.name.replace('.pdf', '')
                    formatted_date = exam_date.strftime("%B %d, %Y")
                    
                    # Calculate available days (excluding exam day)
                    days_until_exam = (exam_date - datetime.now().date()).days
                    available_days = min(7, max(1, days_until_exam))  # 1-7 days
                    
                    raw_plan = generate_study_plan(
                        syllabus_text, formatted_date, course_name, 
                        all_courses_info, available_days
                    )
                    structured_plan = parse_study_plan_to_calendar_items(
                        raw_plan, course_name, exam_date, available_days
                    )
                    
                    all_calendar_items.extend(structured_plan)
                    st.session_state.generated_plans[course_name] = raw_plan
            
            # Resolve conflicts after all plans are generated
            if all_calendar_items:
                with st.spinner("🔄 Resolving schedule conflicts..."):
                    resolved_items, conflicts = resolve_calendar_conflicts(all_calendar_items)
                    st.session_state.calendar_items = resolved_items
                    st.session_state.conflicts_resolved = conflicts
            
            progress_bar.empty()
            st.success("✅ Study plans generated and conflicts resolved successfully!")
    
    # Display results
    if st.session_state.generated_plans:
        st.markdown("## 📘 Study Plans")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📅 Calendar View", "📝 Detailed Plans"])
        
        with tab1:
            show_calendar_view(st.session_state.calendar_items, st.session_state.conflicts_resolved)
        
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