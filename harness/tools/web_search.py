import os
from tavily import TavilyClient

from harness.tools.registry import tool
from harness.config import WEB_SEARCH_MAX_RESULTS




def _format_results(results: dict, query: str) -> str:
    """
    Format the results of the web search.
    """

    answer = results.get("answer") or "[No answer found]"

    results = results.get("results") or []
    lines = [f"### Web Search Results for {query}","", "**Answer**", answer, ""]

    if len(results) > 0:
        lines.append("**Sources**")
        for i, result in enumerate(results, start=1):
            title = result.get("title") or "[No title found]"
            url = result.get("url") or "[No URL found]"
            content = result.get("content") or "[No content found]"
            lines.append(f"{i}. {title} ({url})")
            if content:
                lines.append(f"    {content}")
            lines.append("")
        lines.append("")
    else:
        lines.append("No results found")
        lines.append("")

    return "\n".join(lines)
    

@tool
def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """
    Search the web for information.
    """
    
    if max_results is None:
        max_results = WEB_SEARCH_MAX_RESULTS
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set")
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query, 
            max_results=max_results, 
            include_answer=True,
            include_usage=True
        )
        print(f"Web Search Response: {response}")
    except Exception as e:
        print(f"Error searching the web: {e}")
        return f"Error searching the web: {e}"
    
    final_results = _format_results(response, query)
    return final_results