"""Client for retrieving sports news and injury reports.

This module defines a simple interface to a generic news API.  It is
primarily intended to gather recent headlines and articles about teams
or players, which can be used by the prediction engine to adjust
probabilities based on injuries, suspensions or other contextual
factors.  The default implementation targets the public
newsapi.org service, but you can point it at any API that exposes a
similar ``/v2/everything`` endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models import NewsArticle
from ..utils import fetch_json


class NewsApiClient:
    """Retrieve news articles about teams and players from a news API."""

    DEFAULT_BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_days: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("News API key must be provided")
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.default_days = default_days

    def get_recent_articles(
        self,
        search_term: str,
        from_days: Optional[int] = None,
        max_results: int = 5,
    ) -> List[NewsArticle]:
        """Search for news articles about a team or player.

        Args:
            search_term: Text to search for (e.g. team or player name).
            from_days: Number of days in the past to include in the search.
            max_results: Maximum number of articles to return.

        Returns:
            A list of `NewsArticle` objects, sorted by publication date
            (newest first).  If no articles are found, returns an empty list.
        """
        if not search_term:
            return []
        days = from_days or self.default_days
        from_date = (
            datetime.utcnow() - timedelta(days=days)
        ).strftime("%Y-%m-%d")
        params: Dict[str, str] = {
            "q": search_term,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": str(max_results),
            "apiKey": self.api_key,
        }
        try:
            data = fetch_json(self.base_url, params=params)
        except Exception as e:
            logging.error("Failed to fetch news articles: %s", e)
            return []
        articles: List[NewsArticle] = []
        for item in data.get("articles", []) if isinstance(data, dict) else []:
            try:
                title = item.get("title") or ""
                url = item.get("url") or ""
                published_at_str = item.get("publishedAt") or ""
                try:
                    published_at = datetime.fromisoformat(
                        published_at_str.replace("Z", "+00:00")
                    )
                except Exception:
                    published_at = datetime.utcnow()
                summary = item.get("description") or None
                source = (item.get("source", {}) or {}).get("name")
                articles.append(
                    NewsArticle(
                        title=title,
                        url=url,
                        published_at=published_at,
                        summary=summary,
                        source=source,
                    )
                )
            except Exception:
                continue
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles
