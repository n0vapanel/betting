"""Basic tests for API client classes.

These tests verify that the client classes enforce API key requirements
and that certain methods return sensible defaults when invoked with
minimal input.  They deliberately avoid making real network calls.
"""

from __future__ import annotations

import pytest

from telegram_betting_bot_simple.data.apis import ApiSportsClient, NewsApiClient, TheOddsApiClient

def test_api_sports_client_requires_key() -> None:
    with pytest.raises(ValueError):
        ApiSportsClient(api_key="")
    client = ApiSportsClient(api_key="dummy")
    assert client is not None

def test_news_api_client_returns_empty_on_blank_search() -> None:
    client = NewsApiClient(api_key="dummy")
    articles = client.get_recent_articles("")
    assert articles == []

def test_the_odds_api_client_requires_key() -> None:
    with pytest.raises(ValueError):
        TheOddsApiClient(api_key="")
    client = TheOddsApiClient(api_key="dummy")
    assert client is not None
