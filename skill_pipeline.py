"""Resume-focused pipeline: enhanced text extraction + robust LLM skill detection with fallback."""
from typing import List, Tuple, Dict, Any
import hashlib
import re
import json
import time
import google.generativeai as genai

# No transformers fallback to keep startup fast

# Canonical mapping for normalization
_RAW_CANON_MAP: Dict[str, str] = {
    "js": "javascript",
    "java script": "javascript",
    "python3": "python",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "postgres": "postgresql",
    "ci/cd": "ci cd",
    "powerbi": "power bi",
    "c plus plus": "c++",
    "cplusplus": "c++",
    # Common truncations/variants from LLM outputs
    "reactjs": "react",
    "react j": "react",
    "reactj": "react",
    "react native": "react native",
    "nodejs": "node.js",
    "node j": "node.js",
    "nodej": "node.js",
    "rest": "rest api",
    "rest architecture": "rest api",
    "graphql": "graphql",
    "websockets": "websocket",
    "ui": "ui design",
    "ux": "ux design",
    "ui/ux": "ui design",
    "git version control": "git",
    "html/cs": "html css",
    "html-css": "html css",
    "html & css": "html css",
    "bash": "bash scripting",
    "algorithmic fairne": "algorithmic fairness",
    "statistical analysi": "statistical analysis",
    "product analytic": "product analytics",
    "sports analytic": "sports analytics",
    "data analytic": "data analytics",
    "genai": "generative ai",
    "mlop": "mlops",
    "problem-solving": "problem solving",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "sci-kit learn": "scikit-learn",
    "kera": "keras",
    "tablea": "tableau",
    "ms excel": "excel",
    "power-bi": "power bi",
}

# Extended skill dictionary for heuristic fallback
_EXTENDED_SKILLS = [
    # Programming languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala', 'r',
    # Frameworks & libraries
    'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'cassandra', 'elasticsearch', 'snowflake', 'bigquery', 'dynamodb', 'redshift',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible', 'mlflow', 'airflow', 'dbt', 'kafka',
    'aws s3', 's3', 'ec2', 'lambda', 'cloudwatch', 'glue', 'athena', 'emr', 'sagemaker', 'eks', 'ecs', 'azure devops', 'databricks', 'azure data factory', 'gke', 'pub/sub', 'cloud functions',
    # Data & Analytics
    'machine learning', 'deep learning', 'data science', 'statistics', 'tableau', 'power bi', 'excel', 'spark', 'hadoop', 'nlp', 'computer vision',
    # Web technologies
    'html', 'css', 'rest api', 'graphql', 'json', 'xml', 'microservices', 'oauth', 'jwt', 'grpc',
    # Soft skills
    'leadership', 'communication', 'teamwork', 'problem solving', 'project management', 'agile', 'scrum', 'collaboration'
]

def _normalize_skill(s: str) -> str:
    t = re.sub(r"\s+", " ", s.lower().strip())
    t = t.replace("power-bi", "power bi").replace("+ +", "++")
    if t.endswith('s') and len(t) > 3 and t not in ['analysis', 'business', 'process', 'express', 'kubernetes', 'statistics']:
        t = t[:-1]
    t = re.sub(r'[•▪▫◦‣⁃→←↑↓–—]', '', t).strip()
    return _RAW_CANON_MAP.get(t, t)


def _canonicalize_list(raw_skills: List[str]) -> List[str]:
    seen: set[str] = set()
    cleaned: List[str] = []
    for s in raw_skills:
        if not s:
            continue
        n = _normalize_skill(s)
        if not n or len(n) < 2:
            continue
        if n not in seen:
            seen.add(n)
            cleaned.append(n)
    return cleaned


def _extract_list_sections(text: str) -> List[str]:
    """Heuristically parse Skills/Technologies/Tools sections to boost recall."""
    if not text:
        return []
    lines = text.splitlines()
    collected: List[str] = []
    heading_rx = re.compile(r"^(skills?|technologies|tech\s*stack|tools?)\b.*$", re.IGNORECASE)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if heading_rx.match(line):
            j = i + 1
            look = 0
            buf: List[str] = []
            while j < len(lines) and look < 8:
                cur = lines[j].strip()
                if not cur:
                    break
                buf.append(cur)
                j += 1
                look += 1
            chunk = " ".join(buf)
            parts = re.split(r"[|,/;]\s*|\s-\s|\s\u2013\s|\s\u2014\s", chunk)
            tokens = [t.strip() for t in parts if len(t.strip()) >= 2]
            collected.extend(tokens)
            i = j
            continue
        if ("," in line or "|" in line) and len(line) > 24 and sum(1 for c in line if c in ",|") >= 3:
            for t in re.split(r"[|,/]", line):
                tt = t.strip()
                if tt:
                    collected.append(tt)
        i += 1

    return _canonicalize_list(collected)

def _extract_text_enhanced(file_obj, filename: str) -> Tuple[str, Dict[str, Any]]:
    debug = {'extraction_method': None, 'text_length': 0, 'ocr_used': False, 'errors': []}
    text = ""
    try:
        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            try:
                import pdfplumber
                with pdfplumber.open(file_obj) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                debug['extraction_method'] = 'pdfplumber'
            except Exception as e:
                debug['errors'].append(f'pdfplumber: {str(e)[:100]}')
            if len(text.strip()) < 50:
                try:
                    import fitz
                    file_obj.seek(0)
                    pdf_doc = fitz.open(stream=file_obj.read(), filetype="pdf")
                    text = "\n".join(page.get_text() for page in pdf_doc)
                    pdf_doc.close()
                    debug['extraction_method'] = 'pymupdf'
                except Exception as e:
                    debug['errors'].append(f'pymupdf: {str(e)[:100]}')
            if len(text.strip()) < 50:
                try:
                    import pytesseract
                    from pdf2image import convert_from_bytes
                    file_obj.seek(0)
                    images = convert_from_bytes(file_obj.read(), dpi=300)
                    text = "\n".join(pytesseract.image_to_string(img, config='--psm 6') for img in images)
                    debug['extraction_method'] = 'ocr'
                    debug['ocr_used'] = True
                except Exception as e:
                    debug['errors'].append(f'ocr: {str(e)[:100]}')
        elif ext == 'docx':
            try:
                import docx
                doc = docx.Document(file_obj)
                text = "\n".join(p.text for p in doc.paragraphs)
                debug['extraction_method'] = 'python-docx'
            except Exception as e:
                debug['errors'].append(f'docx: {str(e)[:100]}')
        else:
            try:
                file_obj.seek(0)
                text = file_obj.read().decode('utf-8', errors='ignore')
                debug['extraction_method'] = 'direct_text'
            except Exception as e:
                debug['errors'].append(f'text: {str(e)[:100]}')
    except Exception as e:
        debug['errors'].append(f'general: {str(e)[:100]}')
    debug['text_length'] = len(text)
    return text, debug

_RESUME_SKILLS_CACHE: Dict[str, List[str]] = {}


def extract_resume_skills_llm(text: str, retries: int = 3, delay: int = 3,
                              chunk_size: int = 10000, overlap: int = 500) -> List[str]:
    """High-recall extractor using LLM over chunks with union of results.

    - Splits long resumes into chunks with small overlap to avoid truncation.
    - Uses a comprehensive prompt encouraging exhaustive extraction across categories.
    - Merges and normalizes all skills across chunks.
    """
    if not text.strip():
        return []

    cache_key = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
    if cache_key in _RESUME_SKILLS_CACHE:
        return list(_RESUME_SKILLS_CACHE[cache_key])

    # Prepare chunks
    chunks: List[str] = []
    t = text
    idx = 0
    n = len(t)
    if n <= chunk_size:
        chunks = [t]
    else:
        step = max(1, chunk_size - overlap)
        while idx < n and len(chunks) < 12:
            chunks.append(t[idx: idx + chunk_size])
            idx += step

    # Use Gemini 2.5 Pro
    model = genai.GenerativeModel('gemini-2.5-pro')

    all_skills: List[str] = []
    base_prompt = (
        "You are an expert tech recruiter. Extract ALL distinct skills, tools, technologies, platforms, and methodologies mentioned. "
        "Be exhaustive with NO cap. Include languages, frameworks, libraries, data/ML/AI, MLOps, BI/analytics, cloud services, databases/warehouses, DevOps, testing/QA, APIs, and soft skills. "
        "Use canonical names only (e.g., 'react', 'node.js', 'rest api', 'graphql', 'docker', 'kubernetes', 'pandas', 'power bi', 'scikit-learn', 'xgboost', 'snowflake', 'airflow'). "
        "Fix obvious truncations/typos (e.g., 'tablea' → 'tableau', 'kera' → 'keras'). Return ONLY valid JSON: {\"skills\":[...]}"
    )

    for ci, chunk in enumerate(chunks):
        prompt = f"{base_prompt}\n\nRESUME CHUNK {ci+1} OF {len(chunks)}:\n{chunk[:chunk_size]}"
        for attempt in range(retries):
            try:
                resp = model.generate_content(prompt, generation_config={"temperature": 0})
                raw = (resp.text or '').strip()
                # Remove common code-fence wrappers like ```json ... ```
                raw = raw.strip()
                if raw.startswith('```'):
                    raw = raw.strip('`')
                    if raw.lower().startswith('json'):
                        raw = raw[4:]
                if raw.endswith('```'):
                    raw = raw.rstrip('`')
                data = json.loads(raw)
                if isinstance(data, dict) and isinstance(data.get('skills'), list):
                    all_skills.extend([str(s) for s in data['skills']])
                    break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    # last attempt failed; continue with next chunk
                    pass

    # Merge with heuristic list-section parser to boost recall further
    heuristic = _extract_list_sections(text)
    skills = _canonicalize_list(all_skills + heuristic)
    skills_sorted = sorted(set(skills))
    _RESUME_SKILLS_CACHE[cache_key] = skills_sorted
    return skills_sorted

def extract_resume_skills_hybrid(text: str, filename: str = "resume") -> Tuple[List[str], Dict[str, Any]]:
    debug = {
        'filename': filename,
        'input_chars': len(text or ''),
        'layers_attempted': ['llm', 'heuristic'],
        'skills_per_layer': {},
        'final_count': 0,
        'success_layer': None
    }
    skills = extract_resume_skills_llm(text)
    debug['skills_per_layer']['llm'] = len(skills)
    if skills:
        debug['success_layer'] = 'llm'
    else:
        txt_low = text.lower()
        # Heuristic patterns to catch common variants
        pattern_map = {
            r"\bnode\s*\.?(?:js)?\b": "node.js",
            r"\breact\s*\.?(?:js)?\b": "react",
            r"\brest\b": "rest api",
            r"\bpower\s*bi\b": "power bi",
            r"\bci/?cd\b": "ci cd",
        }
        hits: set[str] = set()
        for pat, canon in pattern_map.items():
            if re.search(pat, txt_low):
                hits.add(canon)
        heuristic_skills = sorted(hits | {
            term for term in _EXTENDED_SKILLS
            if re.search(r'\b' + re.escape(term) + r'\b', txt_low)
        })
        skills = heuristic_skills
        debug['skills_per_layer']['heuristic'] = len(skills)
        if skills:
            debug['success_layer'] = 'heuristic'

        # Note: transformers-based keyword fallback removed for speed
    debug['final_count'] = len(skills)
    return skills, debug
