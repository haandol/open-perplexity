import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tavily import TavilyClient

from ...logger import get_logger

logger = get_logger("web_search_tool")


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", None)
assert TAVILY_API_KEY, "TAVILY_API_KEY environment variable not set"
TAVILY_K = int(os.environ.get("TAVILY_K", 3))

client = TavilyClient(api_key=TAVILY_API_KEY)


class WebSearchInput(BaseModel):
    """
    Input schema for web search operations.

    queries parameter only accepts English keywords.

    Attributes:
        queries (list[str]): The search query strings that will be used to perform the web search.
    """

    queries: list[str] = Field(
        title="Queries",
        description=(
            "A list of query strings for web search."
            "Each query must be in English and should be specific enough to yield relevant results."
            "For best results, use clear and well-formed questions or keyword combinations."
        ),
    )


def _tavily_search(query: str) -> dict:
    """Perform a search using the Tavily API."""
    logger.info(f"Searching for: {query}...")
    return client.search(query, max_results=TAVILY_K)


def web_search(queries: list[str]) -> list:
    """
    Searches given queries on the web and returns the search results.

    ## Tool Parameters
    - queries (list[str]): The search query strings that will be used to perform the web search.

    ### Query generation instructions
    For each element of the queries parameter, please follow the guidelines below:
     - Generate 2-3 relevant search queries based on a given task to web searches.
     - Each query should capture the main intent of the task in different perspectives or contexts.
    """
    TIMEOUT = 10
    SCORE_THRESHOLD = 0.45
    results = []
    try:
        with ThreadPoolExecutor(max_workers=TAVILY_K) as executor:
            future_results = executor.map(
                _tavily_search, queries, timeout=TIMEOUT)
            for result in future_results:
                results.extend([r for r in result["results"]
                               if r["score"] > SCORE_THRESHOLD])
        logger.info(f"Web search results: {len(results)}")
    except TimeoutError:
        traceback.print_exc()
        logger.error("Web search failed with timeout.")
    except Exception:
        traceback.print_exc()
        logger.error("Web search failed with exception")
    return results


tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description=web_search.__doc__,
    args_schema=WebSearchInput,
    return_direct=True,
)
