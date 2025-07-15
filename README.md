# StudyPlanner AI

Plan your entire test season with AI! Upload your syllabi as PDFs and get a personalized 7-day study plan for each exam.

## Features

- 📄 Upload multiple syllabus PDFs
- 📅 Enter exam dates for each course
- 🤖 AI-generated, motivational 7-day study plans
- 🖥️ Simple, interactive web interface powered by Streamlit

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/account/api-keys)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/YedidyaBarGad/studyplanner-ai.git
   cd studyplanner-ai
   ```
2. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   - Create a file named `.env` in the root of the project directory.
   - Add your API key to the file:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

### Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open your web browser and go to `http://localhost:8501`.
3. Upload your syllabus PDFs and enter your exam dates.
4. Receive your personalized 7-day study plans and get started on your study journey!

### Contributing

We welcome contributions to StudyPlanner AI! If you have suggestions or improvements, please submit a pull request or open an issue on GitHub.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- [Streamlit](https://streamlit.io/) - The web framework used to build the interactive interface.
- [OpenAI](https://openai.com/) - For their powerful language model used in generating study plans.
