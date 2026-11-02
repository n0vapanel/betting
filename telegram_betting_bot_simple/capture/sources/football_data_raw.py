"""football-data.co.uk's season CSVs, raw (plan section 3.3: twice a week). The existing
``learn/sources/football_data.py`` keeps reading them into ``hist_*``; this keeps the file itself.

Files (as learn/sources/football_data.py reads them; the www. host answers 302 to this one)::

    https://football-data.co.uk/mmz4281/{YYZZ}/{DIV}.csv    one league-season (2425 = 2024/25)
    https://football-data.co.uk/new/{CC}.csv                one country, every season since 2012 (USA: MLS)

CODES: the six divisions the learn source reads (E0, E1, SP1, I1, D1, F1) plus the three other wave-1 leagues the
research downloaded the same way (SC0 Scottish Premiership, N1 Eredivisie, P1 Primeira Liga: soccer_events.md,
``mmz4281/{2324,2425,2526}/{league}.csv``). NEW: the country files (USA).

``season_csv(http, code, season)``: a division's file has params ``{"season": "2425"}`` (``season`` may be the
start year, 2024, or the folder code, "2425"); a country file holds every season, so its params are ``{}`` and
``season`` is not used. The game key is ``static:football_data:<code>:<YYZZ>`` (a country file:
``static:football_data:<code>``; ``game_key``), the league ``league_of(code)`` (E0 -> soccer_epl).
"""

from __future__ import annotations

import re
from typing import Any, Dict

from . import RawFetch, fetch, static_key

SOURCE = "football_data"
BASE = "https://football-data.co.uk"
CODES: Dict[str, str] = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
    "SC0": "soccer_spl",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
}
NEW: Dict[str, str] = {"USA": "soccer_usa_mls"}
_FOLDER = re.compile(r"\d{4}")


def season_code(season: Any) -> str:
    """The season's folder: an int is the start year (2024 -> "2425", the 2024/25 season); text is a folder code
    and stays as it is ("2425"). Text is never read as a year ("2021" is the 2020/21 folder)."""
    if isinstance(season, bool):
        raise ValueError(f"not a season: {season!r}")
    if isinstance(season, int):
        if not 1993 <= season <= 2098:
            raise ValueError(f"not a season: {season!r}")
        return f"{season % 100:02d}{(season + 1) % 100:02d}"
    text = str(season or "").strip()
    if _FOLDER.fullmatch(text) and (int(text[:2]) + 1) % 100 == int(text[2:]):
        return text
    raise ValueError(f"not a season (a start year as an int, or a folder like '2425'): {season!r}")


def _code(code: Any) -> str:
    text = str(code or "").strip()
    if text not in CODES and text not in NEW:
        raise ValueError(f"not a football-data code here ({sorted(CODES) + sorted(NEW)}): {code!r}")
    return text


def _from_fetch(code: Any, season: Any) -> Any:
    """(code, season) of a fetch of this module (read back from its URL and params), else as given."""
    if isinstance(code, RawFetch):
        found = re.search(r"/(?:mmz4281/\d{4}|new)/([A-Za-z0-9]+)\.csv", code.url)
        if not found:
            raise ValueError(f"not a football-data fetch: {code.url}")
        return found.group(1), code.params.get("season")
    return code, season


def league_of(code: Any) -> str:
    """The app's league for a code (or a fetch of one): E0 -> soccer_epl."""
    text = _code(_from_fetch(code, None)[0])
    return CODES.get(text) or NEW[text]


def game_key(code: Any, season: Any = None) -> str:
    """``static:football_data:<code>:<YYZZ>`` for a league-season (``static:football_data:<code>`` for a country
    file, every season in one); ``code`` may be a fetch of this module."""
    code, season = _from_fetch(code, season)
    text = _code(code)
    return static_key(SOURCE, text if text in NEW else f"{text}:{season_code(season)}")


def file_url(code: Any, season: Any = None) -> str:
    text = _code(code)
    if text in NEW:
        return f"{BASE}/new/{text}.csv"
    return f"{BASE}/mmz4281/{season_code(season)}/{text}.csv"


def season_csv(http: Any, code: Any, season: Any = None) -> RawFetch:
    """One league-season's CSV (a country file: every season). See the module's docstring for the params."""
    text = _code(code)
    params = {} if text in NEW else {"season": season_code(season)}
    return fetch(http, file_url(text, season), source=SOURCE, endpoint="csv", params=params, query={})
