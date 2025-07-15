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
        return f"Error generating study plan: {e}"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="StudyPlanner AI", layout="centered")
st.title("📚 StudyPlanner AI")
st.caption("Plan your exam prep in seconds using AI.")

uploaded_file = st.file_uploader("Upload your syllabus (PDF)", type=["pdf"])
exam_date = st.date_input("Select your exam date")

if uploaded_file and exam_date:
    with st.spinner("Reading syllabus and generating your plan..."):
        syllabus_text = extract_text_from_pdf(uploaded_file)
        formatted_date = exam_date.strftime("%B %d, %Y")
        study_plan = generate_study_plan(syllabus_text, formatted_date)

    st.subheader("Your Study Plan")
    st.markdown(study_plan)
else:
    st.info("Please upload a syllabus PDF and select your exam date.")
