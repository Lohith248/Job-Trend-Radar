## Career Trend Radar (Job + Skill Assistant)

### Overview
Streamlit app that extracts skills from resumes and job descriptions, clusters trends, computes a user–role skill match, and exports a short report (PDF/CSV).

### Quick start
1) Install dependencies (Python 3.10+):
```
pip install -r requirements.txt
```

2) Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key
# Optional (for live job search via JSearch)
JSEARCH_API_KEY=your_jsearch_api_key
```

3) Run the app:
```
streamlit run streamlit_app.py
```

### How to use
- Upload a resume (pdf/docx/txt). Optionally add manual skills or paste a single JD for a quick match.
- To analyze live job posts, toggle “Use live JSearch API” and set role/location, then click “Analyze”.
- View top skills, clusters, gap analysis, role recommendations, and download the PDF/CSV.

### Environment variables
- `GOOGLE_API_KEY` (required): Gemini API key used for skill extraction and summaries. Put it in `.env` or export it in your shell.
- `JSEARCH_API_KEY` (optional): Enables live job search via JSearch. Without it, paste job descriptions manually.

### Notes
- First run may download a small embedding model (for better matching); this can take a minute.
- If you see very few skills, ensure `GOOGLE_API_KEY` is set and valid. You can also add a few manual skills to boost recall.

### Troubleshooting
- “GOOGLE_API_KEY not found”: Create `.env` with the key and restart the app.
- Few or zero skills: Ensure resume is text-based (not scanned). Try DOCX/TXT. Verify internet access and install extras: `pip install sentence-transformers keybert`.
- Slow first run: Models are cached after first use. Keep the app running to avoid reloads.


