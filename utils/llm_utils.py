import re
import time
import os
import json
import difflib
from typing import List, Tuple, Dict, Any, Sequence, Mapping, Union, Optional
import google.generativeai as genai

# Remove transformers to speed up startup
SentenceTransformer = None  # type: ignore
st_util = None  # type: ignore

# --- CONFIGURATION & INITIALIZATION ---

# Prefer environment variable. Using a fallback is not recommended for production.
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("[llm_utils] WARNING: GOOGLE_API_KEY environment variable not found.")
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"[llm_utils] Failed to configure Google API key: {e}")

# Use a fast and capable model. 'gemini-1.5-flash' is a great choice.
MODEL_NAME = "gemini-2.5-pro"

# No embedding model to avoid heavy downloads/startup
EMBEDDING_MODEL = None

# --- SKILL NORMALIZATION ---

# Centralized alias map for skill normalization
_RAW_CANON_MAP: Dict[str, str] = {
    "tf": "tensorflow", "tensorflow 2": "tensorflow", "tensorflow2": "tensorflow",
    "pytorch": "pytorch", "torch": "pytorch", "js": "javascript",
    "java script": "javascript", "py": "python", "python3": "python",
    "postgres": "postgresql", "gcloud": "gcp", "ci/cd": "ci cd",
    "powerbi": "power bi", "cplusplus": "c++", "node.js": "nodejs",
    "react.js": "react", "eda": "exploratory data analysis", "viz": "data visualization",
    "ml": "machine learning", "nlp": "natural language processing",
}

def _normalize_skill(token: str) -> str:
    """A robust function to clean and standardize a skill token."""
    if not token:
        return ""
    t = token.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("power-bi", "power bi").replace("+ +", "++")
    # More robustly handle plurals without affecting special cases
    if t.endswith('s') and len(t) > 3 and t not in {"aws", "gcp", "kubernetes", "express", "analysis", "business", "statistics"}:
        t = t[:-1]
    # Remove common bullet point characters
    t = re.sub(r"[•▪▫◦‣⁃→←↑↓–—]", "", t).strip()
    return _RAW_CANON_MAP.get(t, t)

def normalize_skills(skills: Sequence[str]) -> List[str]:
    """Applies normalization to a list of skills and returns a sorted, unique list."""
    if not skills:
        return []
    return sorted({_normalize_skill(s) for s in skills if _normalize_skill(s)})

# --- CORE LLM & SEMANTIC FUNCTIONS ---

def extract_skills_lists_for_descriptions(descriptions: Sequence[str], batch_size: int = 10, delay_sec: int = 5) -> List[List[str]]:
    """Extracts skills from a list of job descriptions using batch processing."""
    if not descriptions:
        return []
    
    model = genai.GenerativeModel(MODEL_NAME)
    all_extracted_skills: List[List[str]] = []
    
    for i in range(0, len(descriptions), batch_size):
        batch_texts = descriptions[i:i + batch_size]
        numbered_jobs = "\n".join(f"{idx + 1}: {text}" for idx, text in enumerate(batch_texts))
        
        prompt = (
            "You are an expert technical recruiter. For EACH job description, extract ALL distinct skills, tools, technologies, and relevant methodologies mentioned. "
            "Be exhaustive with NO arbitrary cap. Include: programming languages, frameworks, libraries, tooling, platforms, cloud, databases, data/ML/AI, DevOps, testing/QA, analytics/BI, security, methodologies (agile/scrum), certifications, and soft skills. "
            "Use canonical, complete names only (e.g., 'react', 'node.js', 'rest api', 'graphql', 'docker', 'kubernetes', 'pandas', 'power bi'). "
            "Do not output truncated tokens (e.g., 'reactj' -> 'react', 'nodej' -> 'node.js', 'html/cs' -> 'html css').\n\n"
            "JOB DESCRIPTIONS:\n"
            f"{numbered_jobs}\n\n"
            "Respond ONLY with a JSON object. The keys should be the job numbers (e.g., '1', '2'). "
            "The values should be an array of skill strings. Example: {\"1\": [\"python\", \"sql\"], \"2\": [\"communication\"]}"
        )
        
        try:
            print(f"Processing batch starting at index {i}...")
            response = model.generate_content(prompt, generation_config={"temperature": 0.1})
            # Clean response text from markdown code blocks
            cleaned_text = re.sub(r"```json\n?|```", "", response.text.strip())
            extracted_data = json.loads(cleaned_text)
            
            # Process results for the current batch
            for j in range(len(batch_texts)):
                # JSON keys are strings, so access with str(j+1)
                skills = extracted_data.get(str(j + 1), [])
                all_extracted_skills.append(normalize_skills(skills))

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Batch at index {i} failed: {e}. Appending empty results for this batch.")
            all_extracted_skills.extend([[] for _ in batch_texts]) # Add empty lists for failed batch items
            
        if i + batch_size < len(descriptions):
            print(f"Waiting {delay_sec} seconds...")
            time.sleep(delay_sec)
            
    return all_extracted_skills


def cluster_skills_llm(top_skills: Sequence[str], max_clusters: int = 6) -> List[Dict[str, Any]]:
    """Groups a list of skills into named clusters using an LLM."""
    if not top_skills:
        return []
    
    model = genai.GenerativeModel(MODEL_NAME)
    skills_str = ", ".join(normalize_skills(top_skills))
    
    prompt = (
        "You are a skills analyst. Group the following list of skills into logical clusters.\n"
        f"SKILLS: {skills_str}\n\n"
        f"Group them into at most {max_clusters} clusters. For each cluster, provide a short, descriptive name.\n"
        "Respond ONLY with a valid JSON array of objects, where each object has two keys: "
        "\"cluster\" (the cluster name) and \"skills\" (an array of skill strings from the original list).\n"
        "Example: [{\"cluster\": \"Data Science & ML\", \"skills\": [\"python\", \"machine learning\"]}]"
    )
    
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        cleaned_text = re.sub(r"```json\n?|```", "", response.text.strip())
        clusters = json.loads(cleaned_text)
        # Ensure data is a list of dictionaries with correct keys
        if isinstance(clusters, list):
            return [{"cluster": c.get("cluster", "Uncategorized"), "skills": normalize_skills(c.get("skills", []))} for c in clusters if isinstance(c, dict)]
    except (json.JSONDecodeError, Exception) as e:
        print(f"Skill clustering failed: {e}")
    
    return []


def _semantic_match_skills(user_skills: Sequence[str], job_skills: Sequence[str], threshold=0.7) -> Tuple[List[str], List[str]]:
    """Lightweight matching using set overlap after alignment.

    Assumes upstream alignment (via align_skills_to_vocab) has already mapped
    resume skills to the job vocabulary, so simple set logic is sufficient.
    """
    if not job_skills:
        return [], []
    user_set = set(normalize_skills(user_skills))
    job_set = set(normalize_skills(job_skills))
    have = sorted(list(job_set.intersection(user_set)))
    missing = sorted(list(job_set.difference(user_set)))
    return have, missing


def gap_analysis_llm(role: str, top_skills: Sequence[str], user_skills: Sequence[str]) -> Dict[str, Any]:
    """Computes match percentage using semantic matching and provides an LLM summary."""
    norm_top_skills = normalize_skills(top_skills)
    norm_user_skills = normalize_skills(user_skills)

    # Use semantic matching for have/missing lists
    have, missing = _semantic_match_skills(norm_user_skills, norm_top_skills)
    
    match_percent = round(100.0 * len(have) / max(1, len(norm_top_skills)), 2)

    # Generate a summary with the LLM
    summary = ""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"A user applying for a '{role}' role has the following skill gap.\n"
            f"Matched Skills: {', '.join(have)}\n"
            f"Missing Skills: {', '.join(missing)}\n"
            "Write a brief, encouraging 2-3 sentence summary analyzing this gap. "
            "Highlight their strengths and point out the key areas for development to better fit the role."
        )
        response = model.generate_content(prompt, generation_config={"temperature": 0.4})
        summary = response.text.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        summary = "Could not generate summary."

    return {"match_percent": match_percent, "have": have, "missing": missing, "summary": summary}


def recommend_roles_llm(user_skills: Sequence[str], role_skill_map: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Scores and recommends roles based on semantic skill match."""
    results = []
    norm_user_skills = normalize_skills(user_skills)
    
    for role, required_skills in role_skill_map.items():
        norm_req_skills = normalize_skills(required_skills)
        have, missing = _semantic_match_skills(norm_user_skills, norm_req_skills)
        
        pct = round(100.0 * len(have) / max(1, len(norm_req_skills)), 2)
        
        results.append({
            "role": role, 
            "match_percent": pct, 
            "missing_skills": missing
        })
        
    return sorted(results, key=lambda x: -x["match_percent"])[:5]


def align_skills_to_vocab(user_skills: Sequence[str], vocab: Sequence[str]) -> Dict[str, str]:
    """Align skills via LLM guidance with explicit rules, no transformers.

    Uses a structured prompt to map each resume skill to the closest job
    vocabulary skill, handling typos, specific->general, library->concept,
    and synonyms. Returns mapping for skills with reasonable matches; others
    are mapped to null and filtered out.
    """
    if not user_skills or not vocab:
        return {}

    model = genai.GenerativeModel(MODEL_NAME)
    usr = ", ".join(normalize_skills(user_skills))
    job = ", ".join(normalize_skills(vocab))

    prompt = (
        "You are an expert tech recruiter specializing in Data Science. Your goal is to accurately map a candidate's resume skills to the skills required in job descriptions.\n\n"
        f"Candidate's Resume Skills:\n{usr}\n\n"
        f"Job Vocabulary Skills:\n{job}\n\n"
        "Your Task: Create a JSON mapping from EACH resume skill to its closest equivalent in the Job Vocabulary. Rules:\n"
        "1. Correct Typos (e.g., 'panda' -> 'pandas').\n"
        "2. Match Specific to General (e.g., 'customer attrition modeling' -> 'predictive modeling').\n"
        "3. Match Libraries to Concepts (e.g., 'scikit-learn' -> 'machine learning').\n"
        "4. Understand Synonyms ('cross-functional collaboration' -> 'collaboration').\n"
        "If a resume skill has no reasonable match, map it to null. Respond ONLY with valid JSON object: {\"resume_skill\": \"job_vocab_or_null\"}."
    )

    mapping: Dict[str, str] = {}
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.1})
        cleaned = re.sub(r"```json\n?|```", "", (resp.text or "").strip())
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.strip():
                    mapping[k] = v.strip()
    except Exception as e:
        print(f"align_skills_to_vocab LLM mapping failed: {e}")

    # Fallback: light fuzzy matching for anything missing
    if mapping:
        remaining = [s for s in user_skills if s not in mapping]
    else:
        remaining = list(user_skills)

    norm_vocab = normalize_skills(vocab)
    for s in remaining:
        cand = difflib.get_close_matches(_normalize_skill(s), norm_vocab, n=1, cutoff=0.83)
        if cand:
            mapping[s] = cand[0]

    return mapping


def pathway_for_skill(skill: str, user_skills: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Generates a learning pathway for a single skill."""
    model = genai.GenerativeModel(MODEL_NAME)
    s = _normalize_skill(skill)
    usr_sk = normalize_skills(user_skills or [])
    
    prompt = (
        f"A user wants to learn the skill: '{s}'.\n"
        f"They already have some related skills: {', '.join(usr_sk) if usr_sk else 'none'}.\n\n"
        "Provide a concise learning plan. The plan should include:\n"
        "1. 'steps': An array of 2-3 short, actionable learning steps (6-12 words each).\n"
        "2. 'project': A one-sentence project idea to practice the skill.\n\n"
        "Respond ONLY with a valid JSON object with keys 'steps' and 'project'."
    )
    
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.3})
        cleaned_text = re.sub(r"```json\n?|```", "", response.text.strip())
        data = json.loads(cleaned_text)
        return {"skill": s, "steps": data.get("steps", []), "project": data.get("project", "")}
    except (json.JSONDecodeError, Exception) as e:
        print(f"Pathway generation failed for skill '{s}': {e}")
    
    return {"skill": s, "steps": [], "project": ""}