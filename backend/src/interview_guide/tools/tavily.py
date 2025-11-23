from typing import List, Dict, Optional

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # type: ignore

from interview_guide.configuration import settings

def search(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
) -> List[Dict]:
    """Run a Tavily web search and normalize results.

    Returns a list of dicts:
    {title, url, snippet, score, source}
    """
    key = api_key or settings.tavily_api_key
    if key is None:
        raise RuntimeError("Missing TAVILY_API_KEY. Put it in .env or export it.")
    if TavilyClient is None:
        raise RuntimeError("tavily-python not installed. Run: uv add tavily-python")

    client = TavilyClient(api_key=key)
    resp = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_domains=include_domains,
    )

    items: List[Dict] = []
    seen = set()
    for r in resp.get("results", []):
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        items.append({
            "title": r.get("title") or "Untitled",
            "url": url,
            "snippet": r.get("content") or "",
            "score": r.get("score", 0.0),
            "source": "tavily",
        })
        if len(items) >= max_results:
            break
    return items