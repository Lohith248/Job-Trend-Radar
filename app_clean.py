import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from skill_pipeline_simple import extract_resume_skills_llm_enhanced, extract_text_enhanced
from utils.llm_utils_simple import extract_skills_lists_for_descriptions, recommend_roles_llm
from utils.smart_skill_matcher import enhanced_gap_analysis_llm
from jobs_api import fetch_jobs_jsearch
import os
import pandas as pd
import plotly.express as px
import io
import csv

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

def aggregate_skills(per_job_skills: list, top_k: int = 20):
    freq = {}
    for skills in per_job_skills:
        for skill in set(skills):
            freq[skill] = freq.get(skill, 0) + 1
    
    sorted_skills = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return sorted_skills[:top_k], freq

def create_pdf_report(role: str, summary: dict, top_skills: list) -> bytes:
    if not PDF_AVAILABLE:
        return b"PDF not available"
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []
    
    content.append(Paragraph("Career Analysis Report", styles['Title']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Role:</b> {role}", styles['Normal']))
    content.append(Paragraph(f"<b>Match Score:</b> {summary.get('match_percent', 0)}%", styles['Normal']))
    content.append(Spacer(1, 12))
    
    # Top skills
    content.append(Paragraph("<b>Top Market Skills:</b>", styles['Heading2']))
    for skill, count in top_skills[:10]:
        content.append(Paragraph(f"• {skill}: {count} jobs", styles['Normal']))
    
    if summary.get('have'):
        content.append(Spacer(1, 12))
        content.append(Paragraph("<b>Skills You Have:</b>", styles['Heading2']))
        skills_text = ", ".join(summary['have'][:15])
        content.append(Paragraph(skills_text, styles['Normal']))
    
    if summary.get('missing'):
        content.append(Spacer(1, 12))
        content.append(Paragraph("<b>Skills to Develop:</b>", styles['Heading2']))
        missing_text = ", ".join(summary['missing'][:10])
        content.append(Paragraph(missing_text, styles['Normal']))
    
    doc.build(content)
    return buffer.getvalue()

st.set_page_config(page_title="Career Trend Radar", layout="wide")
st.title("🎯 Career Trend Radar")
st.caption("Analyze job market trends and match your skills")

col1, col2 = st.columns([1, 1])

with col1:
    role_query = st.text_input("🎯 Target Role", value="Data Scientist", 
                              help="Enter the job role you're targeting")
    pages = st.selectbox("📊 Analysis Depth", [1, 2, 3], 
                        help="More pages = more jobs analyzed (slower)")

with col2:
    uploaded_file = st.file_uploader("📄 Upload Resume", 
                                   type=["pdf", "docx", "txt"],
                                   help="Upload your resume for skill extraction")
    
    manual_skills = st.text_input("➕ Additional Skills", 
                                placeholder="Docker, Kubernetes, Leadership",
                                help="Add skills not found in your resume")

# Analyze button
analyze_btn = st.button("🚀 Analyze Market Trends", type="primary", use_container_width=True)

# Extract skills from resume
user_skills = []
if uploaded_file:
    with st.spinner("Extracting skills from resume..."):
        text, debug = extract_text_enhanced(uploaded_file, uploaded_file.name)
        if text.strip():
            # Use enhanced LLM extraction with normalization
            skill_result = extract_resume_skills_llm_enhanced(text, role_query.lower())
            user_skills = skill_result.get('skills', [])
            
            method = skill_result.get('method', 'unknown')
            if method == 'llm_enhanced':
                st.success("✨ Enhanced LLM extraction successful")
            elif method == 'llm_with_normalization':
                st.info("🔧 LLM extraction with normalization")
                if skill_result.get('removed_skills'):
                    with st.expander(f"🧹 Removed {len(skill_result['removed_skills'])} invalid skills"):
                        st.caption(", ".join(skill_result['removed_skills']))
            elif skill_result.get('fallback'):
                st.warning("⚠️ Using fallback extraction method")
                
            if debug.get('errors'):
                st.warning(f"Text extraction issues: {'; '.join(debug['errors'])}")
        else:
            user_skills = []

if manual_skills.strip():
    manual_list = [s.strip().lower() for s in manual_skills.split(',') if s.strip()]
    user_skills.extend(manual_list)
    user_skills = sorted(set(user_skills))

if user_skills:
    if isinstance(user_skills, dict) and 'skills' in user_skills:
        clean_skills = user_skills.get('skills', [])
        removed_skills = user_skills.get('removed_skills', [])
        
        st.success(f"**Found {len(clean_skills)} validated skills**")
        
        if clean_skills:
            skill_display = []
            for skill in clean_skills[:15]:
                if isinstance(skill, dict):
                    name = skill.get('name', skill)
                    category = skill.get('category', '')
                    skill_display.append(f"{name} ({category})" if category else name)
                else:
                    skill_display.append(skill)
            
            st.write(", ".join(skill_display) + ("..." if len(clean_skills) > 15 else ""))
        if removed_skills:
            with st.expander(f"🧹 Removed {len(removed_skills)} nonsensical entries"):
                st.caption(", ".join(removed_skills))
        
        # Use clean skills for analysis
        user_skills = [s.get('name', s) if isinstance(s, dict) else s for s in clean_skills]
    else:
        # Simple list format
        st.success(f"**Found {len(user_skills)} skills:** {', '.join(user_skills[:15])}" + 
                  ("..." if len(user_skills) > 15 else ""))
        user_skills = list(user_skills)  # Ensure it's a list

# Main Analysis
if analyze_btn:
    if not user_skills:
        st.error("❌ Please upload a resume or enter skills manually first")
    else:
        # Check API key
        api_key = os.getenv("JSEARCH_API_KEY")
        if not api_key:
            st.error("❌ Missing JSEARCH_API_KEY in .env file")
        else:
            # Fetch jobs
            with st.spinner(f"🔍 Fetching {role_query} jobs..."):
                jobs, meta = fetch_jobs_jsearch(role_query, pages=pages)
                
            if meta.get("errors"):
                st.warning(f"⚠️ API issues: {', '.join(meta['errors'])}")
            
            job_descriptions = [job.get("description", "") for job in jobs if job.get("description")]
            
            if not job_descriptions:
                st.error("❌ No job descriptions found. Try a different search term.")
            else:
                # Extract skills from jobs
                with st.spinner(f"🧠 Analyzing {len(job_descriptions)} job descriptions..."):
                    job_skills_lists = extract_skills_lists_for_descriptions(job_descriptions)
                
                # Aggregate results
                top_skills, all_frequencies = aggregate_skills(job_skills_lists)
                top_skill_names = [skill for skill, _ in top_skills]
                
                # Perform intelligent gap analysis with LLM
                gap_result = enhanced_gap_analysis_llm(role_query, top_skill_names, user_skills)
                
                # Display Results
                st.header("📊 Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Jobs Analyzed", len(job_descriptions))
                col2.metric("Unique Skills Found", len(all_frequencies))
                col3.metric("Your Match Score", f"{gap_result.get('match_percent', 0)}%")
                col4.metric("Skills You Have", len(gap_result.get('have', [])))
                
                # Skills chart
                if top_skills:
                    st.subheader("🔥 Most In-Demand Skills")
                    df = pd.DataFrame(top_skills[:15], columns=["Skill", "Job Count"])
                    
                    fig = px.bar(df, x="Skill", y="Job Count", 
                               title=f"Top Skills for {role_query} Roles",
                               color="Job Count", color_continuous_scale="viridis")
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Gap Analysis with detailed explanations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("✅ Skills You Have")
                    matched_skills = gap_result.get('matched_skills', [])
                    if matched_skills:
                        for match in matched_skills[:8]:
                            job_req = match.get('job_requirement', '')
                            user_skill = match.get('candidate_skill', '')
                            strength = match.get('match_strength', 'good')
                            explanation = match.get('explanation', '')
                            
                            # Color code by strength
                            color = "🟢" if strength == "strong" else "🟡" if strength == "good" else "🟠"
                            st.write(f"{color} **{job_req}** (via {user_skill})")
                            if explanation:
                                st.caption(explanation)
                    else:
                        # Fallback to simple list
                        have_skills = gap_result.get('have', [])
                        for skill in have_skills[:10]:
                            st.write(f"• {skill}")
                
                with col2:
                    st.subheader("🎯 Skills to Learn")
                    missing_skills = gap_result.get('missing_skills', [])
                    if missing_skills:
                        for missing in missing_skills[:8]:
                            skill = missing.get('skill', '')
                            priority = missing.get('priority', 'medium')
                            learning_path = missing.get('learning_path', '')
                            related = missing.get('related_to_existing', '')
                            
                            # Priority indicators
                            priority_icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                            st.write(f"{priority_icon} **{skill}**")
                            
                            if learning_path:
                                st.caption(f"💡 {learning_path}")
                            if related:
                                st.caption(f"🔗 {related}")
                    else:
                        # Fallback to simple list
                        missing_simple = gap_result.get('missing', [])
                        for skill in missing_simple[:10]:
                            st.write(f"• {skill}")
                
                # Show cleaned/removed skills info
                cleaning_info = gap_result.get('cleaning_info', {})
                if cleaning_info.get('removed_job_skills'):
                    with st.expander("🧹 Cleaned Job Requirements"):
                        st.write(f"**Removed nonsensical requirements:** {', '.join(cleaning_info['removed_job_skills'])}")
                        st.caption(f"Analyzed {cleaning_info.get('clean_job_count', 0)} valid skills out of {cleaning_info.get('original_job_count', 0)} total")
                
                # Enhanced AI Summary with recommendations
                if gap_result.get('summary'):
                    st.subheader("🤖 AI Career Analysis")
                    st.info(gap_result['summary'])
                    
                    # Show specific recommendations
                    recommendations = gap_result.get('recommendations', [])
                    if recommendations:
                        st.subheader("📋 Action Plan")
                        for i, rec in enumerate(recommendations[:5], 1):
                            st.write(f"{i}. {rec}")
                
                # Show excluded nonsensical requirements
                excluded = gap_result.get('excluded_requirements', [])
                if excluded:
                    st.warning(f"⚠️ Ignored nonsensical job requirements: {', '.join(excluded)}")
                
                # Role Recommendations
                if top_skills:
                    with st.spinner("🎭 Getting role recommendations..."):
                        role_map = {role_query: top_skill_names}
                        recommendations = recommend_roles_llm(user_skills, role_map)
                    
                    if recommendations:
                        st.subheader("💼 Role Match Analysis")
                        for i, role in enumerate(recommendations[:3], 1):
                            match_pct = role['match_percent']
                            color = "success" if match_pct >= 70 else "warning" if match_pct >= 50 else "error"
                            st.write(f"**{i}. {role['role']}** - {match_pct}% match")
                            if role.get('missing_skills'):
                                st.caption(f"To improve: {', '.join(role['missing_skills'][:3])}")
                
                # Download Options
                st.subheader("📥 Download Reports")
                col1, col2 = st.columns(2)
                
                with col1:
                    if PDF_AVAILABLE:
                        pdf_data = create_pdf_report(role_query, gap_result, top_skills)
                        st.download_button(
                            "📄 Download PDF Report",
                            data=pdf_data,
                            file_name=f"{role_query.replace(' ', '_')}_analysis.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.info("Install reportlab for PDF reports")
                
                with col2:
                    # Skills CSV
                    csv_buffer = io.StringIO()
                    writer = csv.writer(csv_buffer)
                    writer.writerow(["Skill", "Frequency", "You Have"])
                    for skill, count in top_skills:
                        has_skill = "Yes" if skill in gap_result.get('have', []) else "No"
                        writer.writerow([skill, count, has_skill])
                    
                    st.download_button(
                        "📊 Download Skills CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{role_query.replace(' ', '_')}_skills.csv",
                        mime="text/csv"
                    )

# Footer
st.markdown("---")
st.caption("🚀 Built with Streamlit • Powered by AI • Open Source")
