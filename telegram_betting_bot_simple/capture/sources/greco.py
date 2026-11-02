"""Greco1899's UFCStats CSVs (``Greco1899/scrape_ufc_stats`` on GitHub), raw (plan section 3.3: weekly).

``https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/{file}`` (the v4 research notes,
combat_tennis_cricket_other.md). The repo publishes six CSVs, refreshed daily by its own job (5,441 bouts
2016-2026). FILES are their names: the notes' file list says ``ufc_events.csv``, but the event file the research
actually downloaded and joined is ``ufc_event_details.csv`` (``EVENT, URL, DATE, LOCATION``), so that name is used.
``csvs(http, files=...)`` takes another list. Licence unknown: cite, never redistribute.

Each file is params ``{"file": "<name>"}``, game key ``static:greco1899:<file>`` (``game_key``), league
``mma_mixed_martial_arts`` (``league_of``).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Optional

from . import RawFetch, fetch, static_key

SOURCE = "greco1899"
LEAGUE = "mma_mixed_martial_arts"
BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
FILES = ("ufc_event_details.csv", "ufc_fight_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv",
         "ufc_fighter_details.csv", "ufc_fighter_tott.csv")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv")


def file_name(name: Any) -> str:
    text = str(name or "").strip()
    if not _NAME.fullmatch(text) or ".." in text:
        raise ValueError(f"not a CSV file name: {name!r}")
    return text


def file_url(name: Any) -> str:
    return f"{BASE}/{file_name(name)}"


def csv_file(http: Any, name: Any) -> RawFetch:
    """One of the CSVs. Params ``{"file": name}``."""
    text = file_name(name)
    return fetch(http, file_url(text), source=SOURCE, endpoint="csv", params={"file": text}, query={})


def game_key(name: Any) -> str:
    """``static:greco1899:<file>`` for a file name (or a fetch of one)."""
    return static_key(SOURCE, file_name(name.params["file"] if isinstance(name, RawFetch) else name))


def league_of(name: Any = None) -> str:
    """Every Greco1899 file is the UFC's."""
    return LEAGUE


def csvs(http: Any, files: Optional[Iterable[Any]] = None) -> Iterator[RawFetch]:
    """Each CSV in turn (FILES unless ``files``), one RawFetch each: the caller stores each as it comes, so a
    budget's Refused or an Unreachable part-way loses nothing already fetched."""
    for name in (FILES if files is None else tuple(files)):
        yield csv_file(http, name)
