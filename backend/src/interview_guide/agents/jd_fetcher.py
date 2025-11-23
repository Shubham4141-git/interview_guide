

"""
JD Fetcher
- Given a URL, fetches the HTML and extracts visible text.
- Minimal: strips scripts/styles, returns cleaned text.
"""
import requests
from bs4 import BeautifulSoup


def fetch_jd_from_url(url: str, max_chars: int = 4000) -> str:
    """Fetch JD text from a URL (truncate to max_chars)."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL {url}: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    if not text:
        raise RuntimeError("No extractable text found in page.")

    return text[:max_chars]
