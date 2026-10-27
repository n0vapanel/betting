"""Utility functions for data fetching and processing.

This module centralises helper functions used across the data layer:

- `fetch_json` performs an HTTP GET request through the process's shared cache (data/cache.py's TTLCache)
  and returns the parsed JSON response while handling common errors. The cache key is the URL and the query
  without its secret parameters (API keys and tokens): a key never reaches the cache, its table or a log.
- `policy` gives the time-to-live and stale window for a kind of data (data/cache.py's ttl_for table).
- `use_cache` hands this module the process's one TTLCache (services.build_service does it at startup).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

import requests

from .cache import TTLCache, ttl_for

# Query parameters that carry credentials, left out of cache keys (compared without case, "-" and "_"): these
# names, and any name ending in one of SECRET_ENDINGS (x-api-key, refresh_token, client_secret, ...)
SECRET_PARAMS = frozenset({"apikey", "key", "token", "accesstoken", "secret", "clientsecret", "password",
                           "signature", "sig", "auth", "authorization"})
SECRET_ENDINGS = ("apikey", "accesskey", "privatekey", "token", "secret", "password", "signature")

_lock = threading.Lock()
_cache: Optional[TTLCache] = None


def use_cache(cache: Optional[TTLCache]) -> None:
    """Make ``cache`` the one fetch_json (and a client built without one) uses: the process's shared TTLCache."""
    global _cache
    with _lock:
        _cache = cache


def shared_cache() -> TTLCache:
    """The cache set by use_cache, else this module's own in-memory one."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = TTLCache(name="utils")
        return _cache


def policy(kind: str, state: Optional[str] = None) -> Tuple[float, float]:
    """(time to live, stale window) in seconds for ``kind`` of data in ``state`` ("live", "pre", "final", ...):
    data/cache.py's ttl_for, the one policy table (it lists the legacy clients' "http" and "odds_api" kinds too;
    an unknown kind is a KeyError)."""
    found = ttl_for(kind, state)
    return float(found.ttl), float(found.stale)


def _plain(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def safe_params(params: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    """A query as a sorted tuple of (name, value) pairs without its secret parameters (SECRET_PARAMS)."""
    return tuple(sorted((str(name), str(value)) for name, value in (params or {}).items()
                        if _plain(str(name)) not in SECRET_PARAMS and not _plain(str(name)).endswith(SECRET_ENDINGS)))


def cache_key(url: str, params: Optional[Mapping[str, Any]] = None) -> Tuple[Hashable, ...]:
    """The cache key of a GET: its scheme, host, port and path (never a user:password@), and its query (the
    URL's own and ``params``) without secret parameters (safe_params)."""
    parts = urlsplit(url)
    host = (parts.hostname or "") + (f":{parts.port}" if parts.port else "")
    query = {**dict(parse_qsl(parts.query, keep_blank_values=True)), **(params or {})}
    return ("http", f"{parts.scheme}://{host}{parts.path}", safe_params(query))


def kind_of(url: str) -> str:
    """The kind of data a URL answers with: The Odds API's free listing ("odds_sports") or its boards
    ("odds_api"), NewsAPI's articles ("news"), else "http"."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host == "api.the-odds-api.com":
        return "odds_sports" if parts.path.rstrip("/").endswith("/sports") else "odds_api"
    if host == "newsapi.org":
        return "news"
    return "http"


def fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    *,
    cache: Optional[TTLCache] = None,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform an HTTP GET request through the shared cache and return the JSON response.

    Args:
        url: The absolute URL to request.
        params: Query parameters to include in the request.
        headers: Additional headers to include (e.g. API keys); never part of the cache key.
        timeout: Timeout in seconds for the request.
        cache: The TTLCache to go through (default: the process's, see use_cache).
        kind: The kind of data for the policy table (default: from the host, else "http").

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If the request fails or the response is not valid JSON.
    """
    kind = kind or kind_of(url)
    ttl, stale = policy(kind)
    return (cache if cache is not None else shared_cache()).get(
        cache_key(url, params), ttl, lambda: _get_json(url, params, headers, timeout), stale=stale, kind=kind)


def _get_json(url: str, params: Optional[Dict[str, Any]], headers: Optional[Dict[str, str]],
              timeout: float) -> Dict[str, Any]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        # Don't log or chain exc: its message holds the full query string, API keys included
        reason = getattr(exc.response, "status_code", None) or type(exc).__name__
        logging.error("HTTP request to %s failed (%s)", url, reason)
        raise RuntimeError(f"HTTP request failed for {url} ({reason})") from None
    try:
        return response.json()
    except ValueError:
        logging.error("Failed to parse JSON from %s", url)
        raise RuntimeError(f"Invalid JSON response from {url}") from None
