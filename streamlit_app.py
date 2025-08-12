import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from skill_pipeline import (
    extract_resume_skills_hybrid,
    _extract_text_enhanced,
    extract_resume_skills_llm,
)
from utils.llm_utils import (
    extract_skills_lists_for_descriptions,
    cluster_skills_llm,
    gap_analysis_llm,
    recommend_roles_llm,
    pathway_for_skill,
    align_skills_to_vocab,
)
from jobs_api import fetch_jobs_jsearch
import os
import hashlib
import io
import csv
import pandas as pd
import plotly.express as px
from fpdf import FPDF

# ---------- Helpers (keep app code small) ----------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_jobs_cached(q: str, p: int):
    return fetch_jobs_jsearch(q, pages=int(p))

@st.cache_data(ttl=900, show_spinner=False)
def extract_skills_cached(descs: list[str]):
    return extract_skills_lists_for_descriptions(descs)

def aggregate_top(per_job_skills: list[list[str]], top_k: int = 30):
    freq: dict[str, int] = {}
    for skills in per_job_skills:
        for s in set(skills):
            freq[s] = freq.get(s, 0) + 1
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return top, freq

def make_pdf(role_query: str, job_descs: list[str], summary: dict, top_items: list[tuple[str,int]], clusters_list, roles_list) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt="Career Trend Radar Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=11)
    pdf.cell(0, 8, txt=f"Role: {role_query}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f"Dataset size: {len(job_descs)} | User match %: {summary.get('match_percent',0)}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 8, txt="Top skills:", ln=True)
    pdf.set_font("Arial", size=11)
    for s,cnt in top_items[:20]:
        pdf.cell(0, 6, txt=f"- {s}: {cnt}", ln=True)
    pdf.ln(2)

    if clusters_list:
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 8, txt="Clusters:", ln=True)
        pdf.set_font("Arial", size=11)
        for c in clusters_list:
            pdf.multi_cell(0, 6, txt=f"- {c.get('cluster','')}: {', '.join(c.get('skills',[]))}")
    pdf.ln(2)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 8, txt="Skill gap:", ln=True)
    pdf.set_font("Arial", size=11)
    have_txt = ", ".join(summary.get('have', [])) or "(none)"
    missing_txt = ", ".join(summary.get('missing', [])) or "(none)"
    for ch in ['—', '–', '\t', '\r']:
        have_txt = have_txt.replace(ch, '-')
        missing_txt = missing_txt.replace(ch, '-')
    
    def _wrap_tokens(text: str, width: int = 30) -> str:
        pieces = []
        for w in text.split(' '):
            if len(w) <= width:
                pieces.append(w)
            else:
                pieces.extend([w[i:i+width] for i in range(0, len(w), width)])
        return ' '.join(pieces)
    
    have_txt = _wrap_tokens(have_txt)
    missing_txt = _wrap_tokens(missing_txt)
    pdf.multi_cell(0, 6, txt=f"Have: {have_txt}")
    pdf.multi_cell(0, 6, txt=f"Missing: {missing_txt}")
    if summary.get('summary'):
        pdf.ln(2)
        pdf.set_font("Arial", 'I', size=11)
        pdf.multi_cell(0, 6, txt=f"Summary: {summary['summary']}")

    if roles_list:
        pdf.ln(2)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 8, txt="Role recommendations:", ln=True)
        pdf.set_font("Arial", size=11)
        for r in roles_list:
            pdf.multi_cell(0, 6, txt=f"- {r['role']} — {r['match_percent']}% | missing: {', '.join(r.get('missing_skills', []))}")
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

st.set_page_config(page_title="Career Trend Radar", layout="wide")
st.title("Career Trend Radar — LLM-only")

col1, col2 = st.columns(2)
with col2:
    uploaded = st.file_uploader("Upload Resume (pdf/docx/txt)", type=["pdf", "docx", "txt"])
    st.caption("LLM extractor runs automatically with heuristic fallback.")

# Manual skill input section
manual_skills_input = st.text_area(
    "Add Missing Skills (comma-separated)",
    placeholder="e.g., Docker, Kubernetes, Team Leadership",
    help="Add any skills missed by automatic extraction",
)

col_inp1, col_inp2 = st.columns([3, 1])
with col_inp1:
    role_query = st.text_input("Target role / search term", value="Data Scientist")
    location = st.text_input("Location (optional)", value="Remote")
with col_inp2:
    analyze_btn = st.button("Analyze", use_container_width=True)
    pages = st.number_input("Pages (JSearch)", min_value=1, max_value=5, value=1, help="More pages = slower, more coverage")

use_live_api = st.toggle("Use live JSearch API (recommended)", value=True, help="Requires JSEARCH_API_KEY in .env")

# Optional: paste a job description to get a quick match score
job_desc = st.text_area(
    "Paste a Job Description (optional)",
    placeholder="Paste full JD here to get a match score",
    help="We'll extract job skills via LLM and compare overlap with your resume skills.",
)

resume_text = ""
extraction_debug = {}
if uploaded:
    with st.spinner("Extracting text from resume..."):
        # Enhanced text extraction with multiple fallbacks
        resume_text, text_debug = _extract_text_enhanced(uploaded, uploaded.name)
        extraction_debug.update(text_debug)
    
    if not resume_text.strip():
        st.error("Could not extract text from resume. Please ensure it's a text-based document (not a scanned image).")
        if text_debug.get('errors'):
            st.write("Extraction errors:", "; ".join(text_debug['errors']))

user_skills: list[str] = []
skills_debug = {}
if resume_text:
    with st.spinner("Extracting resume skills (LLM)..."):
        user_skills, skills_debug = extract_resume_skills_hybrid(
            resume_text, uploaded.name if uploaded else "resume"
        )
    
    # Add manual skills if provided
    if manual_skills_input.strip():
        manual_skills = [s.strip().lower() for s in manual_skills_input.split(',') if s.strip()]
        user_skills.extend(manual_skills)
        user_skills = sorted(set(user_skills))  # Remove duplicates
        skills_debug['manual_skills_added'] = len(manual_skills)
    
    st.write("Resume skills (normalized):", ", ".join(sorted(set(user_skills))[:80]) or "(none)")

# Merge manual skills in any case (even without resume)
if manual_skills_input.strip():
    _manual_skills = [s.strip().lower() for s in manual_skills_input.split(',') if s.strip()]
    user_skills = sorted(set((user_skills or []) + _manual_skills))

# Quick job match against a single pasted JD
if user_skills and job_desc.strip():
    with st.spinner("Analyzing job description and computing match score..."):
        jd_skills = extract_resume_skills_llm(job_desc)
        gap = gap_analysis_llm(role_query, jd_skills, user_skills)
    
    st.subheader("Job Match")
    st.write(f"Match Score: {gap['match_percent']}%")
    st.write("Matched Skills:", ", ".join(gap.get('have', [])) or "(none)")
    st.write("Missing Skills:", ", ".join(gap.get('missing', [])) or "(none)")
    if gap.get("summary"):
        st.info(gap["summary"]) 
        with st.expander("Job Skills (LLM extracted)"):
            st.write(", ".join(sorted(set(jd_skills))) or "(none)")

# Main analysis flow triggered by Analyze button
if analyze_btn:
    if not user_skills:
        st.warning("Please upload a resume or enter skills before analyzing.")
    else:
        st.subheader("Results")
        # Build sources: live API preferred, else pasted multi-JD block
        job_descs: list[str] = []

        query_str = role_query.strip()
        if location.strip():
            query_str = f"{role_query.strip()} in {location.strip()}"

        if use_live_api:
            api_key = os.getenv("JSEARCH_API_KEY", "").strip()
            if not api_key:
                st.error("Missing JSEARCH_API_KEY in .env. Add it and restart.")
            else:
                with st.spinner(f"Fetching jobs from JSearch for: {query_str} (pages={int(pages)})..."):
                    jobs, meta = fetch_jobs_cached(query_str, int(pages))
                if meta.get("errors"):
                    st.warning("; ".join(meta["errors"]))
                job_descs = [j.get("description", "") for j in jobs if j.get("description")]
        else:
            with st.expander("Paste multiple JDs (alternative to API)"):
                jd_block = st.text_area(
                    "Paste multiple JDs (separate with a blank line or '---')",
                    height=160,
                    placeholder="JD 1...\n---\nJD 2...\n---\nJD 3...",
                    key="jd_block_main",
                )
                raw = jd_block.strip() if jd_block else ""
                if raw:
                    if '---' in raw:
                        job_descs = [p.strip() for p in raw.split('---') if p.strip()]
                    else:
                        job_descs = [p.strip() for p in raw.split('\n\n') if p.strip()]

        if not job_descs:
            st.warning("No job descriptions available. Provide JSearch API key or paste JDs.")
        else:
            with st.spinner(f"LLM extracting skills for {len(job_descs)} jobs ..."):
                per_job_skills = extract_skills_cached(job_descs[: min(250, len(job_descs))])

            # Aggregate overall top skills
            top, freq = aggregate_top(per_job_skills)

            # Semantic alignment of resume skills to top skills vocabulary before matching
            top_vocab = [s for s, _ in top]
            mapping = align_skills_to_vocab(user_skills, top_vocab)
            aligned_user = sorted({ mapping.get(s, s) or s for s in user_skills })
            
            # Summary card
            match_gap = gap_analysis_llm(role_query, top_vocab, aligned_user)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dataset size", len(job_descs))
            c2.metric("Unique skills", len(freq))
            c3.metric("User match %", match_gap.get("match_percent", 0))

            # Top skills bar chart
            if top:
                st.subheader("Top skills")
                df_top = pd.DataFrame(top, columns=["skill", "count"])
                fig = px.bar(df_top, x="skill", y="count")
                fig.update_layout(height=360, xaxis_tickangle=-35, margin=dict(t=10,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

            # Clusters
            with st.spinner("Clustering skills..."):
                clusters = cluster_skills_llm([s for s, _ in top])
            if clusters:
                st.subheader("Skill clusters")
                for c in clusters:
                    st.markdown(f"- **{c['cluster']}**: {', '.join(c.get('skills', []))}")

            # Gap list
            st.subheader("Skill gap")
            st.write("**Have:**", ", ".join(match_gap.get('have', [])) or "(none)")
            st.write("**Missing:**", ", ".join(match_gap.get('missing', [])) or "(none)")
            if match_gap.get("summary"):
                st.info(match_gap["summary"]) 

            # Pathway suggestions for top 2 missing
            if match_gap.get('missing'):
                st.subheader("Pathway suggestions")
                for skill in match_gap['missing'][:2]:
                    with st.spinner(f"Generating pathway for {skill}..."):
                        path = pathway_for_skill(skill, user_skills)
                    st.write(f"**{skill} → steps:** ")
                    for step in path.get("steps", [])[:3]:
                        st.write(f"   • {step}")
                    if path.get("project"):
                        st.write(f"**Project Idea:** {path['project']}")

            # Role recommendations with sample job details
            title_to_skills: dict[str, set] = {}
            if use_live_api and os.getenv("JSEARCH_API_KEY", "").strip():
                jobs, _ = fetch_jobs_cached(query_str, int(pages))
                for j, skills in zip(jobs[: len(per_job_skills)], per_job_skills):
                    title = (j.get("title") or role_query).strip() or role_query
                    title_to_skills.setdefault(title, set()).update(skills)
            
            role_skill_map = {role_query: [s for s, _ in top]}
            if title_to_skills:
                sorted_titles = sorted(title_to_skills.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
                for t, sset in sorted_titles:
                    role_skill_map[t] = sorted(sset)
            
            with st.spinner("Recommending roles..."):
                roles = recommend_roles_llm(user_skills, role_skill_map)
            
            if roles:
                st.subheader("Job role recommendations")
                # Attach example companies/descriptions from fetched jobs where possible
                sample_jobs = locals().get('jobs', [])
                if sample_jobs:
                    for r in roles:
                        line = f"• **{r['role']}** — {r['match_percent']}% match (missing: {', '.join(r.get('missing_skills', []))})"
                        st.write(line)
                        # Show one matching job detail if role title appears in fetched titles
                        for j in sample_jobs:
                            if r['role'].lower() in (j.get('title','').lower()):
                                comp = j.get('employer_name') or ''
                                desc = (j.get('job_description') or '')[:180].replace('\n',' ')
                                st.caption(f"Example: {comp} — {desc}...")
                                break

            # Download PDF report
            pdf_bytes = make_pdf(role_query, job_descs, match_gap, top, clusters, roles)
            st.download_button("Download Report (PDF)", data=pdf_bytes, file_name="career_trend_radar_report.pdf", mime="application/pdf")

# Export skills list (if any)
if user_skills:
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["skill"])
    for s in user_skills:
        writer.writerow([s])
    st.download_button("Download Skills CSV", csv_buf.getvalue(), file_name="resume_skills.csv", mime="text/csv")

st.caption("LLM-only resume extraction. Optional JD match scoring and role suggestions included.")

# CSV-based jobs analysis (optional)
with st.expander("Analyze jobs from CSV (optional)"):
    csv_file = st.file_uploader("Upload jobs CSV with a 'description' column", type=["csv"], key="csv_jobs")
    if csv_file and st.button("Analyze CSV"):
        import csv as _csv
        rows = []
        try:
            text = csv_file.read().decode('utf-8', errors='ignore')
            reader = _csv.DictReader(text.splitlines())
            for r in reader:
                if r.get('description'):
                    rows.append(r.get('description'))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            rows = []
        
        st.write(f"Loaded {len(rows)} rows")
        if rows:
            with st.spinner("Extracting skills via LLM and aggregating..."):
                per_job = extract_skills_cached(rows[:200])
            
            top, _ = aggregate_top(per_job)
            
            st.subheader("Top skills across CSV")
            for s, c in top:
                st.write(f"{s}: {c}")