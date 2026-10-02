"""
tools.py — web search used by the Fact Checker and Competitor Watch.

Uses Tavily when TAVILY_API_KEY is set, otherwise DuckDuckGo.
Results are cached per query so repeated searches in one run are free.
"""
from functools import lru_cache

from core.config import TAVILY_API_KEY, SEARCH_MAX_RESULTS


def _tavily(query: str, k: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    resp = client.search(query=query, max_results=k, search_depth="basic")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _duckduckgo(query: str, k: int) -> list[dict]:
    try:
        from ddgs import DDGS                 # current package name
    except ImportError:
        from duckduckgo_search import DDGS    # older package name
    with DDGS() as ddg:
        rows = list(ddg.text(query, max_results=k))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in rows
    ]


@lru_cache(maxsize=256)
def _cached_search(query: str, k: int) -> tuple:
    try:
        rows = _tavily(query, k) if TAVILY_API_KEY else _duckduckgo(query, k)
    except Exception:
        rows = []
    return tuple(tuple(sorted(r.items())) for r in rows)


def web_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[dict]:
    """Return a list of {title, url, content}. Never raises — empty list on failure."""
    return [dict(r) for r in _cached_search(query.strip(), max_results)]


def format_results(results: list[dict], max_chars: int = 600) -> str:
    """Turn search results into a numbered evidence block for prompts."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "")[:max_chars].replace("\n", " ")
        lines.append(f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{snippet}")
    return "\n\n".join(lines)