import re
import hashlib
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
from utils.llm_skill_normalizer import llm_normalize_skills, llm_enhance_skill_extraction

SKILL_FIXES = {
    "panda": "pandas", "kera": "keras", "tablea": "tableau",
    "postgre": "postgresql", "sklearn": "scikit-learn",
    "reactj": "react", "nodej": "node.js"
}

def normalize_skill(skill: str) -> str:
    if not skill:
        return ""
    
    skill = skill.lower().strip()
    skill = re.sub(r'\s+', ' ', skill)
    skill = SKILL_FIXES.get(skill, skill)
    
    if skill.endswith('s') and len(skill) > 3 and skill not in ['aws', 'analysis', 'business']:
        skill = skill[:-1]
    
    return skill

def extract_text_enhanced(file_obj, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from uploaded files."""
    debug = {'method': None, 'length': 0, 'errors': []}
    text = ""
    
    try:
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            try:
                import pdfplumber
                with pdfplumber.open(file_obj) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                debug['method'] = 'pdfplumber'
            except Exception as e:
                debug['errors'].append(f'PDF: {str(e)[:50]}')
                
        elif ext == 'docx':
            try:
                import docx
                doc = docx.Document(file_obj)
                text = "\n".join(p.text for p in doc.paragraphs)
                debug['method'] = 'docx'
            except Exception as e:
                debug['errors'].append(f'DOCX: {str(e)[:50]}')
                
        else:  # txt and others
            try:
                file_obj.seek(0)
                text = file_obj.read().decode('utf-8', errors='ignore')
                debug['method'] = 'text'
            except Exception as e:
                debug['errors'].append(f'Text: {str(e)[:50]}')
                
    except Exception as e:
        debug['errors'].append(f'General: {str(e)[:50]}')
    
    debug['length'] = len(text)
    return text, debug

# Cache for LLM results
_SKILLS_CACHE = {}

def extract_resume_skills_llm(text: str) -> List[str]:
    """Extract skills from resume text using LLM."""
    if not text.strip():
        return []
    
    # Cache based on text hash
    cache_key = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()[:16]
    if cache_key in _SKILLS_CACHE:
        return _SKILLS_CACHE[cache_key]
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "Extract ONLY technical skills, tools, programming languages, and soft skills from this resume. "
            "Return as JSON array of clean skill names. "
            "Fix typos (panda->pandas, kera->keras). "
            "EXCLUDE: dates, locations, companies, job titles.\n\n"
            f"Resume:\n{text[:3000]}\n\n"
            'Format: {"skills": ["python", "sql", "communication"]}'
        )
        
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        
        if response.text:
            cleaned = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            data = eval(cleaned)  # Quick parsing
            skills = [normalize_skill(s) for s in data.get('skills', [])]
            skills = [s for s in skills if s and len(s) > 1]
            
            _SKILLS_CACHE[cache_key] = sorted(set(skills))
            return _SKILLS_CACHE[cache_key]
            
    except Exception as e:
        print(f"LLM extraction failed: {e}")
    
    # Fallback: keyword extraction
    return extract_skills_keyword(text)

def extract_skills_keyword(text: str) -> List[str]:
    """Fallback keyword-based skill extraction."""
    common_skills = [
        'python', 'java', 'javascript', 'sql', 'react', 'angular', 'vue',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'aws', 'azure', 'docker', 'kubernetes', 'git',
        'tableau', 'power bi', 'excel', 'mysql', 'postgresql',
        'machine learning', 'data science', 'deep learning',
        'communication', 'leadership', 'teamwork', 'project management'
    ]
    
    text_lower = text.lower()
    found = []
    
    for skill in common_skills:
        pattern = r'\b' + re.escape(skill.replace(' ', r'\s+')) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    
    return sorted(set(found))

def extract_resume_skills_llm_enhanced(resume_text: str, domain: str = "general") -> Dict[str, Any]:
    """
    Enhanced skill extraction using LLM with dynamic normalization
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return {
            'skills': [],
            'detailed_skills': [],
            'removed_skills': [],
            'summary': 'Insufficient text for analysis',
            'method': 'none'
        }
    
    try:
        # Use LLM to extract and normalize skills directly
        result = llm_enhance_skill_extraction(resume_text, domain)
        
        if result['success'] and result['skills']:
            return {
                'skills': result['skills'],
                'detailed_skills': result.get('detailed_skills', []),
                'removed_skills': [],
                'summary': result.get('summary', ''),
                'method': 'llm_enhanced',
                'confidence_breakdown': {
                    'high': result.get('high_confidence', 0),
                    'medium': result.get('medium_confidence', 0)
                }
            }
    except Exception as e:
        print(f"Enhanced LLM extraction failed: {e}")
    
    # Fallback to hybrid approach with LLM normalization
    try:
        # Extract skills using original LLM method
        llm_skills = extract_resume_skills_llm(resume_text)
        
        if llm_skills:
            # Normalize extracted skills using LLM
            normalization_result = llm_normalize_skills(llm_skills, domain)
            
            if normalization_result['success']:
                return {
                    'skills': normalization_result['clean_skills'],
                    'detailed_skills': [
                        {
                            'name': skill,
                            'category': normalization_result['mapping'].get(skill, {}).get('category', 'Other'),
                            'confidence': normalization_result['mapping'].get(skill, {}).get('confidence', 'medium')
                        }
                        for skill in normalization_result['clean_skills']
                    ],
                    'removed_skills': normalization_result['removed_skills'],
                    'summary': normalization_result['summary'],
                    'method': 'llm_with_normalization',
                    'original_count': normalization_result['original_count'],
                    'normalized_count': normalization_result['normalized_count']
                }
    except Exception as e:
        print(f"LLM with normalization failed: {e}")
    
    # Final fallback: keyword extraction with basic normalization
    try:
        keyword_skills = extract_skills_keyword(resume_text)
        if keyword_skills:
            # Apply basic normalization
            normalization_result = llm_normalize_skills(keyword_skills, domain)
            return {
                'skills': normalization_result['clean_skills'],
                'detailed_skills': [],
                'removed_skills': normalization_result['removed_skills'],
                'summary': f"Keyword extraction with normalization: {normalization_result['summary']}",
                'method': 'keyword_with_normalization',
                'fallback': True
            }
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
    
    return {
        'skills': [],
        'detailed_skills': [],
        'removed_skills': [],
        'summary': 'All extraction methods failed',
        'method': 'failed'
    }

def extract_resume_skills_hybrid(resume_text: str) -> List[str]:
    """Extract skills using both LLM and keyword methods."""
    debug = {'input_length': len(resume_text or ''), 'methods': []}
    
    # Try LLM first
    llm_skills = extract_resume_skills_llm(resume_text)
    debug['methods'].append(f'LLM: {len(llm_skills)} skills')
    
    if llm_skills:
        debug['success'] = 'LLM'
        return llm_skills
    
    # Fallback to keywords
    keyword_skills = extract_skills_keyword(resume_text)
    debug['methods'].append(f'Keyword: {len(keyword_skills)} skills')
    debug['success'] = 'Keyword'
    
    return keyword_skills

# Aliases for compatibility
_extract_text_enhanced = extract_text_enhanced
