import json
import re
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from .llm_skill_normalizer import llm_normalize_skills, normalize_job_skills
import os

# Configure Gemini
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-1.5-flash"

def llm_skill_matcher(user_skills: List[str], job_skills: List[str], role: str) -> Dict[str, Any]:
    """
    Use LLM to intelligently match skills and provide detailed analysis.
    Returns comprehensive skill gap analysis with explanations.
    """
    if not job_skills or not user_skills:
        return {"match_percent": 0, "have": [], "missing": job_skills, "summary": "No skills to compare"}
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Create the analysis prompt
        prompt = f"""
You are an expert career counselor and technical recruiter specializing in {role} roles. 

TASK: Analyze skill alignment between a candidate's skills and job market requirements.

CANDIDATE SKILLS: {', '.join(user_skills)}

JOB MARKET REQUIREMENTS: {', '.join(job_skills)}

ANALYSIS INSTRUCTIONS:
1. CLEAN the job requirements - remove any nonsensical, incomplete, or unclear skills
2. MATCH candidate skills to requirements using semantic understanding 
3. SCORE the overall match percentage (0-100%)
4. PROVIDE detailed explanations for matches

SEMANTIC MATCHING RULES:
- "pandas" qualifies for "data analysis", "data manipulation" 
- "scikit-learn" qualifies for "machine learning", "predictive modeling"
- "SQL" qualifies for "database management", "data querying"
- "Python" qualifies for "programming", "automation", "scripting"
- "Tableau" qualifies for "data visualization", "business intelligence"
- Related skills should be considered matches (e.g., TensorFlow → Deep Learning)

OUTPUT FORMAT (valid JSON only):
{{
    "match_percent": 75,
    "matched_skills": [
        {{
            "job_requirement": "data analysis",
            "candidate_skill": "pandas",
            "match_strength": "strong",
            "explanation": "Pandas is the primary tool for data analysis in Python"
        }}
    ],
    "missing_skills": [
        {{
            "skill": "deep learning",
            "priority": "high", 
            "learning_path": "Start with TensorFlow basics, then neural networks",
            "related_to_existing": "You have machine learning foundation"
        }}
    ],
    "excluded_requirements": [
        "electro", "mission it"
    ],
    "summary": "Strong foundation in data science with 75% skill alignment. Focus on deep learning to become fully qualified.",
    "recommendations": [
        "Take a TensorFlow course",
        "Build a neural network project", 
        "Learn about MLOps deployment"
    ]
}}

RESPOND WITH VALID JSON ONLY:
"""

        response = model.generate_content(prompt, generation_config={
            "temperature": 0.1,
            "max_output_tokens": 2048
        })
        
        if not response.text:
            return _fallback_matching(user_skills, job_skills)
        
        # Clean and parse JSON
        json_text = response.text.strip()
        json_text = re.sub(r'```json\s*', '', json_text)
        json_text = re.sub(r'\s*```', '', json_text)
        
        result = json.loads(json_text)
        
        # Validate and format result
        return {
            "match_percent": result.get("match_percent", 0),
            "matched_skills": result.get("matched_skills", []),
            "missing_skills": result.get("missing_skills", []),
            "excluded_requirements": result.get("excluded_requirements", []),
            "summary": result.get("summary", "Analysis completed"),
            "recommendations": result.get("recommendations", []),
            "have": [m["job_requirement"] for m in result.get("matched_skills", [])],
            "missing": [m["skill"] for m in result.get("missing_skills", [])]
        }
        
    except Exception as e:
        print(f"LLM skill matching failed: {e}")
        return _fallback_matching(user_skills, job_skills)

def _fallback_matching(user_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
    """Simple fallback when LLM fails."""
    user_set = set(s.lower() for s in user_skills)
    job_set = set(s.lower() for s in job_skills)
    
    have = list(job_set.intersection(user_set))
    missing = list(job_set.difference(user_set))
    match_percent = round(100 * len(have) / len(job_set)) if job_set else 0
    
    return {
        "match_percent": match_percent,
        "have": have,
        "missing": missing,
        "summary": f"Basic analysis: {match_percent}% match",
        "matched_skills": [],
        "missing_skills": [],
        "recommendations": []
    }

def llm_skill_cleaner(raw_skills: List[str]) -> Tuple[List[str], List[str]]:
    """
    Use LLM to clean and validate extracted skills, removing nonsense.
    Returns (clean_skills, removed_skills)
    """
    if not raw_skills:
        return [], []
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        prompt = f"""
You are a technical skill validator. Clean this list of extracted skills.

RAW SKILLS: {', '.join(raw_skills)}

RULES:
1. KEEP: Real technical skills, programming languages, tools, frameworks, soft skills
2. REMOVE: Incomplete words, nonsense terms, company names, dates, locations
3. FIX: Common typos (panda→pandas, kera→keras)
4. STANDARDIZE: Use canonical names (tensorflow not tf, postgresql not postgres)

EXAMPLES TO REMOVE: "electro", "mission it", "june 2018", "princeton", "ran", "built"
EXAMPLES TO KEEP: "python", "machine learning", "communication", "aws", "sql"

Return JSON with two arrays:
{{
    "clean_skills": ["python", "pandas", "machine learning"],
    "removed_skills": ["electro", "mission it", "june 2018"]
}}
"""

        response = model.generate_content(prompt, generation_config={"temperature": 0})
        
        if response.text:
            json_text = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            result = json.loads(json_text)
            return result.get("clean_skills", []), result.get("removed_skills", [])
            
    except Exception as e:
        print(f"LLM skill cleaning failed: {e}")
    
    # Fallback: basic cleaning
    clean = [s for s in raw_skills if len(s) > 2 and not any(char.isdigit() for char in s)]
    removed = [s for s in raw_skills if s not in clean]
    return clean, removed

def enhanced_gap_analysis_llm(role: str, job_skills: List[str], user_skills: List[str]) -> Dict[str, Any]:
    """
    Enhanced gap analysis using LLM for intelligent skill matching.
    """
    # First clean the job skills to remove nonsense
    clean_job_skills, removed_job_skills = llm_skill_cleaner(job_skills)
    clean_user_skills, removed_user_skills = llm_skill_cleaner(user_skills)
    
    if removed_job_skills:
        print(f"Removed nonsensical job requirements: {removed_job_skills}")
    if removed_user_skills:
        print(f"Cleaned user skills, removed: {removed_user_skills}")
    
    # Perform intelligent matching
    analysis = llm_skill_matcher(clean_user_skills, clean_job_skills, role)
    
    # Add cleaning info to the result
    analysis["cleaning_info"] = {
        "removed_job_skills": removed_job_skills,
        "removed_user_skills": removed_user_skills,
        "original_job_count": len(job_skills),
        "clean_job_count": len(clean_job_skills)
    }
    
    return analysis
