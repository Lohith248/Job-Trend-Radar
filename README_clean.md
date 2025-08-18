# 🎯 Career Trend Radar

**AI-powered job market analysis and skill matching tool**

## What it does
- Extracts skills from your resume using AI
- Fetches real job postings for your target role
- Analyzes skill gaps and provides match percentages
- Generates career recommendations and learning paths
- Exports detailed reports (PDF/CSV)

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
JSEARCH_API_KEY=your_jsearch_api_key
```

### 3. Run the App
```bash
streamlit run app_clean.py
```

## How to Use
1. **Upload Resume** - PDF, DOCX, or TXT files
2. **Enter Target Role** - e.g., "Data Scientist", "Software Engineer"
3. **Click Analyze** - AI will process and match your skills
4. **View Results** - See gaps, recommendations, and download reports

## Features
- 🤖 **AI Skill Extraction** - Intelligent resume parsing
- 🎯 **Smart Matching** - Semantic skill comparison
- 📊 **Market Analysis** - Real job market trends
- 📝 **Career Guidance** - Personalized recommendations
- 📄 **Export Reports** - PDF and CSV downloads

## Requirements
- Python 3.10+
- Google Gemini API key (required)
- JSearch API key (optional, for live job data)

## Project Structure
```
├── app_clean.py                    # Main Streamlit application
├── skill_pipeline_simple.py        # Resume processing & skill extraction
├── jobs_api.py                     # Job posting API integration
├── utils/
│   ├── llm_utils_simple.py        # LLM operations
│   ├── smart_skill_matcher.py     # Intelligent skill matching
│   └── llm_skill_normalizer.py    # Dynamic skill normalization
└── requirements.txt                # Python dependencies
```
