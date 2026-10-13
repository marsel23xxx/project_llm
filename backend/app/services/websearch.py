"""Search the web for sources a written document can cite.

The search runs through Tavily's API, which returns each page's text along with its link, so the
backend never fetches arbitrary URLs itself. Without a key there is no search and the writer says so.
"""

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.core.config import Settings

ENDPOINT = "https://api.tavily.com/search"
RESULTS_PER_QUERY = 5
WORDS_PER_SOURCE = 450


class SearchError(RuntimeError):
    pass


@dataclass
class WebSource:
    title: str
    url: str
    text: str

    @property
    def site(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")


def _clip(text: str, words: int) -> str:
    parts = re.sub(r"\s+", " ", text or "").strip().split(" ")
    return " ".join(parts[:words])


class WebSearch:
    name = "tavily"

    def __init__(self, api_key: str, max_sources: int = 8, timeout: float = 20.0, transport: httpx.AsyncBaseTransport | None = None):
        self._max_sources = max_sources
        self._client = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    async def _one(self, query: str) -> list[dict]:
        payload = {"query": query[:380], "search_depth": "basic", "max_results": RESULTS_PER_QUERY, "include_raw_content": True}
        try:
            response = await self._client.post(ENDPOINT, json=payload)
        except httpx.HTTPError as exc:
            raise SearchError(f"pencarian web tidak bisa dihubungi: {exc}") from exc
        if response.status_code >= 400:
            raise SearchError(f"pencarian web error {response.status_code}: {response.text[:200]}")
        return response.json().get("results") or []

    async def search(self, queries: list[str]) -> list[WebSource]:
        """Run the queries together; results are interleaved so every query contributes, best hits first."""
        batches = await asyncio.gather(*(self._one(q) for q in queries), return_exceptions=True)
        good = [b for b in batches if not isinstance(b, BaseException)]
        if not good:
            raise SearchError(str(batches[0]) if batches else "tidak ada kueri pencarian")

        sources: dict[str, WebSource] = {}
        for rank in range(RESULTS_PER_QUERY):
            for batch in good:
                if rank >= len(batch):
                    continue
                item = batch[rank]
                url = str(item.get("url") or "")
                if not url.startswith(("http://", "https://")) or url in sources:
                    continue
                # The snippet is the part that matched the query; the page text that follows adds the figures around it.
                snippet, page = _clip(item.get("content", ""), 120), _clip(item.get("raw_content") or "", WORDS_PER_SOURCE)
                text = f"{snippet}\n{page}".strip()
                if len(text.split()) >= 15:
                    sources[url] = WebSource(title=_clip(item.get("title") or url, 30), url=url, text=text)
        return list(sources.values())[: self._max_sources]


def search_from(settings: Settings) -> WebSearch | None:
    """A web search client when a key is configured and the hosted LLM is in use; else None."""
    if settings.llm_mode == "local" or not settings.web_search or not settings.tavily_api_key:
        return None
    return WebSearch(settings.tavily_api_key, settings.web_search_max_sources)
