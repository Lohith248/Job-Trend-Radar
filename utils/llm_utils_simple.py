import re
import time
import os
import json
from typing import List, Dict, Any
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"

SKILL_ALIASES = {
    "ml": "machine learning", "ai": "artificial intelligence", "dl": "deep learning",
    "nlp": "natural language processing", "py": "python", "js": "javascript",
    "tf": "tensorflow", "torch": "pytorch", "sklearn": "scikit-learn",
    "panda": "pandas", "pandas": "pandas", "kera": "keras", "keras": "keras",
    "postgre": "postgresql", "postgres": "postgresql", "tablea": "tableau", "tableau": "tableau",
    "mysql": "mysql", "mongo": "mongodb", "mongodb": "mongodb", "sqlite": "sqlite", "redis": "redis",
    
    # Cloud platforms
    "aws": "aws", "azure": "azure", "gcp": "gcp", "google cloud": "gcp",
    
    # Web frameworks
    "reactjs": "react", "react": "react",
    "nodejs": "node.js", "node.js": "node.js",
    "flask": "flask", "django": "django",
    
    # DevOps tools
    "k8s": "kubernetes", "kubernetes": "kubernetes",
    "powerbi": "power bi", "power-bi": "power bi",
    
    # Programming languages
    "cplusplus": "c++", "c++": "c++",
    "javascript": "javascript", "typescript": "typescript",
    
    # Data Science tools
    "jupyter": "jupyter", "numpy": "numpy", "scipy": "scipy",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "plotly": "plotly",
    
    # Business Intelligence
    "excel": "excel", "powerpoint": "powerpoint",
    
    # Version control
    "git": "git", "github": "github", "gitlab": "gitlab",
    
    # Soft skills normalization
    "communication": "communication",
    "problem-solving": "problem solving", 
    "problem solving": "problem solving",
    "teamwork": "teamwork", "leadership": "leadership",
    
    # Specific Data Science concepts
    "recommendation engine": "recommendation systems",
    "productionizing model": "model deployment",
    "customer segmentation": "customer segmentation",
}

def normalize_skill(skill: str) -> str:
    """Normalize a skill name."""
    if not skill:
        return ""
    
    skill = skill.lower().strip()
    skill = re.sub(r'\s+', ' ', skill)
    
    # Remove plurals (except for some exceptions)
    if skill.endswith('s') and len(skill) > 3 and skill not in ['aws', 'analysis', 'business']:
        skill = skill[:-1]
    
    return SKILL_ALIASES.get(skill, skill)

def normalize_skills(skills: List[str]) -> List[str]:
    """Normalize a list of skills."""
    return sorted(set(normalize_skill(s) for s in skills if normalize_skill(s)))

def extract_skills_fallback(text: str) -> List[str]:
    """Extract skills using keyword matching when LLM fails."""
    common_skills = [
        'python', 'java', 'javascript', 'sql', 'react', 'node.js', 'aws', 'docker',
        'machine learning', 'data science', 'tensorflow', 'pytorch', 'pandas',
        'tableau', 'power bi', 'git', 'kubernetes', 'mongodb', 'postgresql',
        'communication', 'leadership', 'teamwork', 'problem solving'
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in common_skills:
        pattern = r'\b' + re.escape(skill.replace(' ', r'\s+')) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills

def extract_skills_lists_for_descriptions(descriptions: List[str], batch_size: int = 5) -> List[List[str]]:
    """Extract skills from job descriptions."""
    if not descriptions:
        return []
    
    model = genai.GenerativeModel(MODEL_NAME)
    all_skills = []
    
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        numbered_jobs = "\n".join(f"{idx + 1}: {text[:500]}" for idx, text in enumerate(batch))
        
        prompt = (
            "Extract technical skills, tools, and technologies from these job descriptions. "
            "Return JSON with job numbers as keys and skill arrays as values.\n"
            f"Jobs:\n{numbered_jobs}\n\n"
            "Example: {\"1\": [\"python\", \"sql\"], \"2\": [\"react\", \"javascript\"]}"
        )
        
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0.1})
            if response.text:
                cleaned_text = re.sub(r"```json\n?|```", "", response.text.strip())
                data = json.loads(cleaned_text)
                
                for j in range(len(batch)):
                    skills = data.get(str(j + 1), [])
                    all_skills.append(normalize_skills(skills))
            else:
                # Fallback for blocked responses
                for text in batch:
                    fallback_skills = extract_skills_fallback(text)
                    all_skills.append(normalize_skills(fallback_skills))
                    
        except Exception as e:
            print(f"Batch {i} failed: {e}. Using fallback.")
            for text in batch:
                fallback_skills = extract_skills_fallback(text)
                all_skills.append(normalize_skills(fallback_skills))
        
        if i + batch_size < len(descriptions):
            time.sleep(2)  # Rate limiting
    
    return all_skills

def semantic_match_skills(user_skills: List[str], job_skills: List[str]) -> tuple[List[str], List[str]]:
    """Enhanced matching for Data Science skills with better semantic understanding."""
    if not job_skills:
        return [], []
    
    user_set = set(normalize_skills(user_skills))
    job_set = set(normalize_skills(job_skills))
    
    # Direct matches first
    have = list(job_set.intersection(user_set))
    
    # Enhanced semantic mappings for Data Science
    semantic_relationships = {
        # Core Data Science concepts
        'data analyst': ['python', 'pandas', 'sql', 'mysql', 'postgresql', 'data science', 'statistics'],
        'data analytic': ['python', 'pandas', 'numpy', 'data analysis', 'statistics', 'tableau'],
        'artificial intelligence': ['machine learning', 'tensorflow', 'keras', 'pytorch', 'python', 'scikit-learn'],
        'machine learning': ['scikit-learn', 'tensorflow', 'keras', 'python', 'pandas', 'numpy'],
        'data science': ['python', 'pandas', 'numpy', 'scikit-learn', 'machine learning', 'statistics'],
        
        # Programming & Tools
        'python': ['pandas', 'numpy', 'scikit-learn', 'flask', 'data analysis'],
        'data analysis': ['python', 'pandas', 'sql', 'statistics', 'data science'],
        'database': ['sql', 'mysql', 'postgresql', 'mongodb'],
        'sql': ['mysql', 'postgresql', 'database management'],
        
        # Advanced ML
        'deep learning': ['tensorflow', 'keras', 'pytorch', 'neural networks'],
        'recommendation engine': ['machine learning', 'collaborative filtering', 'python'],
        'productionizing model': ['mlops', 'deployment', 'docker', 'aws', 'flask'],
        
        # Cloud & DevOps
        'cloud': ['aws', 'azure', 'gcp'],
        'automation': ['python', 'scripting', 'aws'],
        
        # Business Skills
        'customer segmentation': ['clustering', 'machine learning', 'data analysis'],
        'data visualization': ['tableau', 'matplotlib', 'seaborn', 'plotly'],
        
        # Soft Skills
        'consulting': ['communication', 'problem solving', 'client management'],
        'advertising': ['marketing analytics', 'customer analysis'],
    }
    
    # Reverse mapping - if user has advanced skill, they likely have basics
    reverse_mappings = {
        'pandas': ['data analysis', 'data analytic', 'python programming'],
        'scikit-learn': ['machine learning', 'artificial intelligence', 'data science'],
        'tensorflow': ['deep learning', 'artificial intelligence', 'machine learning'],
        'keras': ['deep learning', 'neural networks', 'artificial intelligence'],
        'flask': ['web development', 'python', 'api development'],
        'aws': ['cloud', 'devops', 'deployment'],
        'mysql': ['database', 'sql', 'data management'],
        'postgresql': ['database', 'sql', 'data management'],
        'git': ['version control', 'software development', 'collaboration'],
        'recommendation engine': ['machine learning', 'personalization', 'data science'],
    }
    
    # Check semantic relationships
    for job_skill in job_set:
        if job_skill in have:
            continue
            
        # Check if user has skills that qualify them for this job skill
        if job_skill in semantic_relationships:
            user_related = set(semantic_relationships[job_skill])
            if user_set.intersection(user_related):
                have.append(job_skill)
                continue
        
        # Check reverse mapping - user has advanced skill for basic requirement
        for user_skill in user_set:
            if user_skill in reverse_mappings:
                if job_skill in reverse_mappings[user_skill]:
                    have.append(job_skill)
                    break
            
            # Substring matching for compound skills
            if len(job_skill) > 4 and len(user_skill) > 4:
                if any(word in user_skill for word in job_skill.split() if len(word) > 3):
                    have.append(job_skill)
                    break
                if any(word in job_skill for word in user_skill.split() if len(word) > 3):
                    have.append(job_skill)
                    break
    
    # Remove duplicates and sort
    have = sorted(list(set(have)))
    missing = sorted(job_set.difference(set(have)))
    
    return have, missing

def gap_analysis_llm(role: str, top_skills: List[str], user_skills: List[str]) -> Dict[str, Any]:
    """Analyze skill gap and provide recommendations."""
    norm_top = normalize_skills(top_skills)
    norm_user = normalize_skills(user_skills)
    
    have, missing = semantic_match_skills(norm_user, norm_top)
    match_percent = round(100.0 * len(have) / max(1, len(norm_top)), 1)
    
    # Generate summary
    summary = ""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"User applying for '{role}' role:\n"
            f"Has skills: {', '.join(have[:10])}\n"
            f"Missing: {', '.join(missing[:5])}\n"
            "Write 2 sentences: strengths and key areas to improve."
        )
        response = model.generate_content(prompt, generation_config={"temperature": 0.3})
        summary = response.text.strip() if response.text else "Analysis not available."
    except:
        summary = f"You have {len(have)} of {len(norm_top)} required skills. Focus on developing the missing skills to improve your match."
    
    return {
        "match_percent": match_percent,
        "have": have[:10],
        "missing": missing[:10],
        "summary": summary
    }

def recommend_roles_llm(user_skills: List[str], role_skill_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Recommend roles based on skill match."""
    results = []
    norm_user = normalize_skills(user_skills)
    
    for role, required_skills in role_skill_map.items():
        norm_required = normalize_skills(required_skills)
        have, missing = semantic_match_skills(norm_user, norm_required)
        
        match_pct = round(100.0 * len(have) / max(1, len(norm_required)), 1)
        
        results.append({
            "role": role,
            "match_percent": match_pct,
            "missing_skills": missing[:5]
        })
    
    return sorted(results, key=lambda x: x['match_percent'], reverse=True)
