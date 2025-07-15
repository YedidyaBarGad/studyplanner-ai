import streamlit as st
import fitz  # PyMuPDF
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------- PDF Parsing ----------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# ---------------- AI Prompt ----------------
def generate_study_plan(syllabus_text, exam_date):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are an academic tutor helping students plan for exams."},
                {"role": "user", "content": f"""
Create a 7-day study plan based on the following syllabus:
---
{syllabus_text[:3000]}  # truncate to avoid token limit
---
The test is on {exam_date}. List the study plan day-by-day. Keep it clear and motivational.
                """}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error generating study plan: {e}"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="StudyPlanner AI", layout="centered")
st.title("📚 StudyPlanner AI")
st.caption("Plan your entire test season with AI. Upload multiple syllabi and dates.")

# Upload multiple files
uploaded_files = st.file_uploader("Upload your syllabi (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### 📅 Enter exam dates for each course:")
    exam_dates = {}
    for file in uploaded_files:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**{file.name}**")
        with col2:
            exam_dates[file.name] = st.date_input(f"Exam date for {file.name}", key=file.name)

    if st.button("✨ Generate Study Plans"):
        for file in uploaded_files:
            exam_date = exam_dates.get(file.name)
            if not exam_date:
                st.warning(f"Please select a date for {file.name}")
                continue

            with st.spinner(f"Processing {file.name}..."):
                syllabus_text = extract_text_from_pdf(file)
                formatted_date = exam_date.strftime("%B %d, %Y")
                study_plan = generate_study_plan(syllabus_text, formatted_date)

            st.subheader(f"📘 Study Plan for: {file.name}")
            st.markdown(study_plan)
else:
    st.info("Please upload one or more syllabus PDFs.")
