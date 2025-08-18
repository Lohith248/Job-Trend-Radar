"""
LLM-Powered Skill Normalization System
Dynamically normalizes and standardizes skills using AI instead of hardcoded mappings
"""

import google.generativeai as genai
import os
import json
import re
from typing import List, Dict, Any

def setup_gemini():
    """Setup Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def llm_normalize_skills(skills: List[str], context: str = "general") -> Dict[str, Any]:
    """
    Use LLM to normalize and standardize a list of skills
    
    Args:
        skills: List of raw skills to normalize
        context: Context like 'data science', 'web development', etc.
    
    Returns:
        Dict with normalized skills and mapping information
    """
    model = setup_gemini()
    
    prompt = f"""
You are a skill normalization expert. Your job is to:

1. **Standardize** skill names to their most common industry format
2. **Expand** abbreviations and acronyms 
3. **Correct** typos and variations
4. **Group** related skills under standard categories
5. **Remove** duplicate or overlapping skills
6. **Flag** vague or nonsensical entries

CONTEXT: {context} domain

INPUT SKILLS:
{json.dumps(skills, indent=2)}

RULES:
- Use standard industry terminology (e.g., "panda" → "Pandas", "js" → "JavaScript")
- Expand abbreviations (e.g., "ML" → "Machine Learning", "DL" → "Deep Learning")
- Fix common typos (e.g., "pythn" → "Python", "kera" → "Keras")
- Use title case for technologies (e.g., "react" → "React")
- Remove vague terms like "experience", "knowledge", "good at"
- Flag nonsensical entries like "electro", "mission it", single letters

OUTPUT FORMAT (JSON):
{{
    "normalized_skills": [
        {{
            "original": "original skill name",
            "normalized": "Standard Skill Name", 
            "category": "Programming Language|Framework|Tool|Database|Cloud|etc",
            "confidence": "high|medium|low",
            "explanation": "why this normalization was made"
        }}
    ],
    "removed_skills": [
        {{
            "skill": "removed skill",
            "reason": "why it was removed (vague/nonsensical/duplicate)"
        }}
    ],
    "skill_groups": {{
        "Programming Languages": ["Python", "JavaScript", "Java"],
        "Frameworks": ["React", "Django", "Flask"],
        "Databases": ["PostgreSQL", "MongoDB"],
        "Cloud Platforms": ["AWS", "Azure", "GCP"],
        "Data Science": ["Pandas", "NumPy", "Scikit-learn"],
        "Machine Learning": ["TensorFlow", "PyTorch", "Keras"]
    }},
    "summary": "Brief summary of normalization performed"
}}

Focus on accuracy and industry standards. Be conservative - if unsure about a skill, mark confidence as 'low'.
"""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Validate and clean the result
            normalized_skills = result.get('normalized_skills', [])
            removed_skills = result.get('removed_skills', [])
            skill_groups = result.get('skill_groups', {})
            
            # Create clean skill list
            clean_skills = []
            mapping = {}
            
            for skill_info in normalized_skills:
                original = skill_info.get('original', '')
                normalized = skill_info.get('normalized', '')
                category = skill_info.get('category', 'Other')
                confidence = skill_info.get('confidence', 'medium')
                
                if normalized and confidence != 'low':
                    clean_skills.append(normalized)
                    mapping[original] = {
                        'normalized': normalized,
                        'category': category,
                        'confidence': confidence
                    }
            
            return {
                'success': True,
                'clean_skills': sorted(set(clean_skills)),
                'mapping': mapping,
                'removed_skills': [r.get('skill', '') for r in removed_skills],
                'skill_groups': skill_groups,
                'summary': result.get('summary', ''),
                'original_count': len(skills),
                'normalized_count': len(clean_skills),
                'removed_count': len(removed_skills)
            }
        else:
            # Fallback: basic cleaning
            return basic_skill_cleanup(skills)
            
    except Exception as e:
        print(f"LLM normalization failed: {e}")
        return basic_skill_cleanup(skills)

def basic_skill_cleanup(skills: List[str]) -> Dict[str, Any]:
    """
    Enhanced fallback skill cleanup without LLM
    """
    clean_skills = []
    removed = []
    
    # Enhanced cleaning rules
    common_tech_normalizations = {
        'pythn': 'Python', 'py': 'Python',
        'panda': 'Pandas', 'pandas': 'Pandas',
        'js': 'JavaScript', 'javascript': 'JavaScript',
        'kera': 'Keras', 'keras': 'Keras',
        'tensorflow': 'TensorFlow', 'tf': 'TensorFlow',
        'sklearn': 'Scikit-learn', 'scikit-learn': 'Scikit-learn',
        'postgre': 'PostgreSQL', 'postgres': 'PostgreSQL',
        'mysql': 'MySQL', 'mongo': 'MongoDB',
        'reactj': 'React', 'react': 'React',
        'nodej': 'Node.js', 'nodejs': 'Node.js',
        'aws': 'AWS', 'azure': 'Azure', 'gcp': 'Google Cloud',
        'html': 'HTML', 'css': 'CSS', 'sql': 'SQL'
    }
    
    # Nonsensical skills to remove
    nonsensical_terms = [
        'electro', 'mission it', 'experience', 'knowledge', 'good at',
        'excellent', 'proficient', 'familiar', 'worked with', 'used',
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'years', 'year', 'months', 'month', 'time', 'work', 'job'
    ]
    
    for skill in skills:
        original_skill = skill
        skill = skill.strip().lower()
        
        # Remove if too short or obviously invalid
        if len(skill) < 2:
            removed.append(original_skill)
            continue
            
        # Remove nonsensical terms
        if skill in nonsensical_terms:
            removed.append(original_skill)
            continue
            
        # Remove if it's just numbers or special chars
        if skill.isdigit() or not any(c.isalpha() for c in skill):
            removed.append(original_skill)
            continue
            
        # Apply normalizations
        if skill in common_tech_normalizations:
            clean_skills.append(common_tech_normalizations[skill])
        else:
            # Basic title case for unknown skills
            clean_skill = original_skill.strip().title()
            # Don't add very short or obviously problematic skills
            if len(clean_skill) >= 3 and not any(bad in skill for bad in ['mission', 'electro', 'exp']):
                clean_skills.append(clean_skill)
            else:
                removed.append(original_skill)
    
    return {
        'success': False,
        'clean_skills': sorted(set(clean_skills)),
        'mapping': {},
        'removed_skills': removed,
        'skill_groups': {},
        'summary': f'Basic cleanup: normalized {len(clean_skills)} skills, removed {len(removed)} invalid entries',
        'original_count': len(skills),
        'normalized_count': len(clean_skills),
        'removed_count': len(removed)
    }

def llm_enhance_skill_extraction(text: str, domain: str = "general") -> Dict[str, Any]:
    """
    Use LLM to extract and normalize skills directly from text
    """
    model = setup_gemini()
    
    prompt = f"""
Extract and normalize all technical skills from this resume/profile text.

DOMAIN CONTEXT: {domain}

TEXT:
{text[:3000]}  # Limit text length

EXTRACT:
1. Programming languages (Python, Java, JavaScript, etc.)
2. Frameworks & libraries (React, Django, TensorFlow, etc.)
3. Databases (PostgreSQL, MongoDB, etc.)
4. Cloud platforms (AWS, Azure, GCP, etc.)
5. Tools & technologies (Docker, Git, Jenkins, etc.)
6. Methodologies (Agile, DevOps, Machine Learning, etc.)

NORMALIZE each skill to industry standard format.

OUTPUT JSON:
{{
    "skills": [
        {{
            "name": "Standard Skill Name",
            "category": "Programming Language|Framework|Database|Cloud|Tool|Methodology",
            "confidence": "high|medium|low",
            "evidence": "text snippet that mentions this skill"
        }}
    ],
    "summary": "Brief summary of skills found"
}}

Focus on actual technical skills, not soft skills or job titles.
"""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Filter by confidence
            high_conf_skills = []
            medium_conf_skills = []
            
            for skill in result.get('skills', []):
                if skill.get('confidence') == 'high':
                    high_conf_skills.append(skill['name'])
                elif skill.get('confidence') == 'medium':
                    medium_conf_skills.append(skill['name'])
            
            all_skills = high_conf_skills + medium_conf_skills
            
            return {
                'success': True,
                'skills': sorted(set(all_skills)),
                'detailed_skills': result.get('skills', []),
                'summary': result.get('summary', ''),
                'high_confidence': len(high_conf_skills),
                'medium_confidence': len(medium_conf_skills)
            }
            
    except Exception as e:
        print(f"LLM skill extraction failed: {e}")
        
    # Fallback to basic extraction
    return {
        'success': False,
        'skills': [],
        'detailed_skills': [],
        'summary': 'LLM extraction failed',
        'high_confidence': 0,
        'medium_confidence': 0
    }

def normalize_job_skills(job_skills: List[str], domain: str = "general") -> Dict[str, Any]:
    """
    Normalize skills extracted from job postings using LLM
    """
    if not job_skills:
        return {'clean_skills': [], 'removed_skills': [], 'summary': 'No skills to normalize'}
    
    return llm_normalize_skills(job_skills, context=f"{domain} job requirements")

# Example usage and testing
if __name__ == "__main__":
    # Test normalization
    test_skills = [
        "pythn", "panda", "js", "react", "ML", "DL", "AWS", 
        "electro", "mission it", "kera", "tensorflow", "sql"
    ]
    
    print("Testing LLM Skill Normalization...")
    result = llm_normalize_skills(test_skills, "data science")
    
    print(f"\nOriginal skills: {test_skills}")
    print(f"Normalized skills: {result['clean_skills']}")
    print(f"Removed skills: {result['removed_skills']}")
    print(f"Summary: {result['summary']}")
