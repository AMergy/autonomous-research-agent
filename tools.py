"""
No API key required. The duckduckgo-search library handles everything.
We return a clean list of result dicts so the agent can work with them easily.
"""
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for a given query and return a list of results.

    Each result is a dict with:
      - title   : page title
      - url     : page URL (used as a citation)
      - snippet : short text extract from the page

    Args:
        query:       The search query string.
        max_results: How many results to fetch (default 5 is usually enough).

    Returns:
        A list of result dicts, or an empty list if the search fails.

    Example:
        >>> results = web_search("what is photosynthesis", max_results=3)
        >>> print(results[0]["title"])
    """
    try:
        with DDGS() as ddgs:
            # region='us-en' forces English-language results regardless of
            # the machine's locale — without it DuckDuckGo may return results
            # in whatever language it detects from your IP or system settings
            raw = list(ddgs.text(query, max_results=max_results, region='us-en'))

        # Normalise field names — the library sometimes uses 'body' or 'snippet'
        results = []
        for r in raw:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", r.get("url", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            })

        return results

    except Exception as e:
        print(f"[search] Warning: search failed for '{query}': {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    """
    Turn the list of result dicts into a readable string we can paste into
    a prompt without it looking like raw JSON.

    Args:
        results: Output from web_search().

    Returns:
        A multi-line string, one result per block.
    """
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")  # blank line between results

    return "\n".join(lines)
