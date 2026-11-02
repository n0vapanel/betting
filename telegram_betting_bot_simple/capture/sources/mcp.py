"""Jeff Sackmann's Match Charting Project (``JeffSackmann/tennis_MatchChartingProject``), raw (plan section 3.3:
monthly).

``https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/{file}`` (the v4 research
notes, combat_tennis_cricket_other.md: the MCP repo is live; Sackmann's tennis_atp is not). FILES are the files the
research fetched and confirmed: ``charting-m-matches.csv`` (match metadata), ``charting-m-points-2020s.csv`` (point
by point) and ``charting-m-stats-Overview.csv`` (per player per set: aces, double faults...). The women's files
(``charting-w-*``) and the other point files are split the same way per the README, but their exact names were not
confirmed, so they are not in the default: ``csvs(http, files=...)`` takes them. Licence: CC BY-NC-SA 4.0.

Each file is params ``{"file": "<name>"}``, game key ``static:mcp:<file>`` (``game_key``); ``league_of(name)`` is
``tennis_wta`` for a women's file (``charting-w-``), else ``tennis_atp``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Optional

from . import RawFetch, fetch, static_key

SOURCE = "mcp"
BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
FILES = ("charting-m-matches.csv", "charting-m-points-2020s.csv", "charting-m-stats-Overview.csv")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv")


def file_name(name: Any) -> str:
    text = str(name or "").strip()
    if not _NAME.fullmatch(text) or ".." in text:
        raise ValueError(f"not a CSV file name: {name!r}")
    return text


def _named(name: Any) -> str:
    return file_name(name.params["file"] if isinstance(name, RawFetch) else name)


def league_of(name: Any) -> str:
    """The tour a file (or a fetch of one) belongs to: ``tennis_wta`` for ``charting-w-*``, else ``tennis_atp``."""
    return "tennis_wta" if _named(name).startswith("charting-w-") else "tennis_atp"


def game_key(name: Any) -> str:
    """``static:mcp:<file>`` for a file name (or a fetch of one)."""
    return static_key(SOURCE, _named(name))


def file_url(name: Any) -> str:
    return f"{BASE}/{file_name(name)}"


def csv_file(http: Any, name: Any) -> RawFetch:
    """One MCP file. Params ``{"file": name}``."""
    text = file_name(name)
    return fetch(http, file_url(text), source=SOURCE, endpoint="csv", params={"file": text}, query={})


def csvs(http: Any, files: Optional[Iterable[Any]] = None) -> Iterator[RawFetch]:
    """Each file in turn (FILES unless ``files``), one RawFetch each, stored by the caller as it comes."""
    for name in (FILES if files is None else tuple(files)):
        yield csv_file(http, name)
