"""Utility functions for data fetching and processing.

This module centralises helper functions used across the data layer.  The
main helper is `fetch_json`, which performs an HTTP GET request and
returns the parsed JSON response while handling common errors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests


def fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Perform an HTTP GET request and return the JSON response.

    Args:
        url: The absolute URL to request.
        params: Query parameters to include in the request.
        headers: Additional headers to include (e.g. API keys).
        timeout: Timeout in seconds for the request.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If the request fails or the response is not valid JSON.
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.exception("HTTP request to %s failed: %s", url, exc)
        raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc
    try:
        return response.json()
    except ValueError as exc:
        logging.exception("Failed to parse JSON from %s: %s", url, exc)
        raise RuntimeError(f"Invalid JSON response from {url}") from exc
