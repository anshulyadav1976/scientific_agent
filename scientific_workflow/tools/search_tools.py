"""Tools for Scientific Literature and Web Search"""
import logging
import os
import httpx # Use httpx for async requests
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from tavily import TavilyClient # Ensure tavily-python is in requirements.txt

from portia.tool import Tool, ToolRunContext, ToolSoftError, ToolHardError

logger = logging.getLogger(__name__)

# --- ScientificSearchTool --- #

class SearchInputSchema(BaseModel):
    queries: List[str] = Field(..., description="A list of search queries (strings). Prefixes like 'pubmed:', 'arxiv:', 'web:' can be used, otherwise defaults to Semantic Scholar and Web search.")
    max_results_per_query: int = Field(default=3, description="Maximum number of results to return for each query.")

class SearchResult(BaseModel):
    source_type: str = Field(description="Source of the result (e.g., 'semantic_scholar', 'tavily_web', 'pubmed', 'arxiv')")
    title: str = Field(description="Title of the paper or web page.")
    snippet: str = Field(description="A relevant snippet from the abstract or page content.")
    url_or_doi: Optional[str] = Field(default=None, description="URL or DOI link to the source.")
    # Add other relevant fields like authors, publication_date if needed
    publication_date: Optional[str] = None
    authors: Optional[List[str]] = None

class ScientificSearchTool(Tool[List[SearchResult]]):
    """Performs searches on scientific databases (Semantic Scholar) and general web (Tavily)."""
    id: str = "scientific_search_tool"
    name: str = "Scientific and Web Search"
    description: str = (
        "Queries scientific databases (Semantic Scholar) and performs general web searches (using Tavily) based on a list of input queries. "
        "Returns a list of structured findings including source type, title, snippet, and URL/DOI."
    )
    args_schema: type[BaseModel] = SearchInputSchema
    output_schema: tuple[str, str] = (
        "List[SearchResult]",
        "A list of dictionaries, each containing details about a search result."
    )

    def __init__(self):
        # Initialize Tavily client if API key is available
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables. Web search will be disabled.")
            self.tavily_client = None
        else:
            try:
                 self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except Exception as e:
                 logger.error(f"Failed to initialize TavilyClient: {e}", exc_info=True)
                 self.tavily_client = None

    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[SearchResult]:
        """Performs a search using the Semantic Scholar API."""
        logger.info(f"Searching Semantic Scholar for: '{query}'")
        results = []
        # S2 Public API endpoint
        # Use fields parameter to get abstract (snippet), url, title, authors, publicationDate
        # See: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper_search
        search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,url,abstract,authors,publicationDate"
        headers = {"Accept": "application/json"} # Basic headers, add API key if you have one

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(search_url, headers=headers)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()

            if data and data.get("data"):
                for item in data["data"]:
                    snippet = item.get("abstract", "No abstract available.")
                    if not snippet: snippet = "No abstract available."

                    results.append(SearchResult(
                        source_type="semantic_scholar",
                        title=item.get("title", "No title"),
                        snippet=snippet[:500] + ("..." if len(snippet) > 500 else ""), # Limit snippet length
                        url_or_doi=item.get("url"),
                        publication_date=item.get("publicationDate"),
                        authors=[author.get("name") for author in item.get("authors", []) if author.get("name")]
                    ))
            else:
                logger.info(f"No Semantic Scholar results found for query: '{query}'")

        except httpx.HTTPStatusError as e:
            logger.error(f"Semantic Scholar API request failed: {e.response.status_code} - {e.response.text}")
            # Soft error as maybe other searches will succeed
            # raise ToolSoftError(f"Semantic Scholar API request failed: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Semantic Scholar connection error: {e}")
            # raise ToolSoftError(f"Could not connect to Semantic Scholar: {e}")
        except Exception as e:
            logger.error(f"Error processing Semantic Scholar results: {e}", exc_info=True)
            # raise ToolSoftError(f"Error processing Semantic Scholar results: {e}")
        return results

    async def _search_tavily(self, query: str, max_results: int) -> List[SearchResult]:
        """Performs a web search using the Tavily API."""
        logger.info(f"Searching Tavily Web for: '{query}'")
        results = []
        if not self.tavily_client:
            logger.warning("Tavily client not initialized, skipping web search.")
            return results

        try:
            # Use include_raw_content=True? Might be too verbose
            search_response = self.tavily_client.search(query=query, search_depth="basic", max_results=max_results)
            if search_response and search_response.get("results"):
                for item in search_response["results"]:
                    results.append(SearchResult(
                        source_type="tavily_web",
                        title=item.get("title", "No title"),
                        snippet=item.get("content", "No snippet available.")[:500] + "...",
                        url_or_doi=item.get("url"),
                        # Tavily basic search doesn't usually provide authors/date
                    ))
            else:
                logger.info(f"No Tavily web results found for query: '{query}'")

        except Exception as e:
            logger.error(f"Error during Tavily search: {e}", exc_info=True)
            # raise ToolSoftError(f"Error during Tavily search: {e}")
        return results

    # Mark the main run method as async
    async def run(self, ctx: ToolRunContext, queries: List[str], max_results_per_query: int = 3) -> List[SearchResult]:
        """Executes the search queries against Semantic Scholar and Tavily."""
        logger.info(f"[{self.id}] Running search for {len(queries)} queries...")
        all_results: List[SearchResult] = [] # Use type hint

        # TODO: Implement routing based on prefixes (pubmed:, arxiv:, web:)
        # For now, search both Semantic Scholar and Tavily for all queries

        import asyncio # Import asyncio for concurrent execution

        tasks = []
        for query in queries:
            query = query.strip()
            if not query:
                continue

            logger.info(f"Processing query: '{query}'")
            # Create tasks for concurrent execution
            tasks.append(self._search_semantic_scholar(query, max_results_per_query))
            if self.tavily_client: # Only add tavily task if client is available
                tasks.append(self._search_tavily(query, max_results_per_query))

            # --- Placeholder for other sources ---
            # if query.startswith("pubmed:"): tasks.append(self._search_pubmed(...))

        # Run tasks concurrently
        try:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            # Process results, handling potential exceptions
            for result_item in results_list:
                if isinstance(result_item, Exception):
                    logger.error(f"A search task failed: {result_item}")
                elif isinstance(result_item, list): # Successful result is a list of SearchResult
                    all_results.extend(result_item)
                else:
                    logger.warning(f"Unexpected item type in search results: {type(result_item)}")

        except Exception as e:
             logger.error(f"Error gathering search results: {e}", exc_info=True)


        logger.info(f"[{self.id}] Search completed. Found {len(all_results)} total results.")
        # Deduplicate results based on URL/DOI? Optional.
        # unique_results = {res.url_or_doi: res for res in all_results if res.url_or_doi}
        # return list(unique_results.values())
        return all_results
