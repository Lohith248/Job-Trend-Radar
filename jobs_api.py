"""JSearch API client via RapidAPI (robust with retries).

Requires `.env` with `JSEARCH_API_KEY`.
"""
from typing import List, Dict, Any, Tuple
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_URL = f"https://{JSEARCH_HOST}/search"

''
def _headers() -> Dict[str, str]:
    key = os.getenv("JSEARCH_API_KEY", "").strip()
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": JSEARCH_HOST,
        "User-Agent": "CareerTrendRadar/1.0 (+https://rapidapi.com/)",
    }


def fetch_jobs_jsearch(query: str, pages: int = 1, delay_sec: float = 0.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch job postings from JSearch with retry/backoff.

    Returns (jobs, meta). 'jobs' is a list of dicts. 'meta' contains errors, pages, and elapsed time.
    """
    jobs: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"errors": [], "pages": pages}
    if not query.strip():
        meta["errors"].append("empty_query")
        return jobs, meta

    headers = _headers()
    if not headers["X-RapidAPI-Key"]:
        meta["errors"].append("missing_api_key")
        return jobs, meta

    start = time.time()

    # Robust session with retries (handles transient 5xx/429 and connection resets)
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for page in range(1, max(1, pages) + 1):
        params = {"query": query, "page": page}
        # If user typed "role in location" and did not set country, keep default JSearch behavior.
        # Optionally, you can tune here: add "num_pages" to ask server to aggregate pages in one call.
        if pages > 1:
            params["num_pages"] = pages
        try:
            r = session.get(JSEARCH_URL, headers=headers, params=params, timeout=(10, 30))
            if r.status_code != 200:
                snippet = str(r.text or "")[:140].replace("\n", " ")
                meta["errors"].append(f"page {page} status {r.status_code}: {snippet}")
                continue
            data = r.json()
            results = data.get("data") or data.get("results") or []
            for item in results:
                jobs.append(_normalize_job(item))
        except Exception as e:
            meta["errors"].append(f"page {page}: {str(e)[:120]}")
        if page < pages and delay_sec:
            time.sleep(delay_sec)

    meta["elapsed_sec"] = round(time.time() - start, 2)
    meta["count"] = len(jobs)
    return jobs, meta

def _normalize_job(j: Dict[str, Any]) -> Dict[str, Any]:
    """Pick common fields; keep original for reference."""
    title = j.get("job_title") or j.get("title") or ""
    company = j.get("employer_name") or j.get("company_name") or j.get("company") or ""
    location = j.get("job_city") or j.get("city") or ""
    country = j.get("job_country") or j.get("country") or ""
    description = (
        j.get("job_description")
        or j.get("description")
        or _coalesce_highlights(j)
        or ""
    )
    return {
        "title": title,
        "company": company,
        "location": location,
        "country": country,
        "description": description,
        "raw": j,
    }

def _coalesce_highlights(j: Dict[str, Any]) -> str:
    hl = j.get("job_highlights") or {}
    if isinstance(hl, dict):
        bullets = []
        for k in ("Qualifications", "Responsibilities", "Benefits"):
            v = hl.get(k)
            if isinstance(v, list):
                bullets.extend(v)
        return "\n".join(bullets)
    if isinstance(hl, list):
        return "\n".join(str(x) for x in hl)
    return ""
