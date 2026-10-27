"""Model and rating opinions: ESPN's matchup predictor, ESPN BPI/FPI, Bart Torvik, Tennis Abstract Elo and
eloratings.net, turned into chances with documented formulas. Explanations and a small weight in the blend,
never the price: the research found results-only ratings add nothing to a fresh market.

Endpoints (live-checked 2026-10-27; all free, no key; each fetched at most every 12 hours, the predictor every
hour, by the background worker):

- ESPN matchup predictor: the game summary, EspnClient.summary(league, event) =
  https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={id}
  "predictor": {"homeTeam": {"id", "gameProjection": "61.9", "teamChanceLoss"?}, "awayTeam": {...}} (percent
  strings). NFL, college football and NBA have one; NHL, MLB, soccer and pre-season college basketball don't.
  NFL projections leave the tie chance out (61.9 + 37.8 = 99.7), so the two are scaled to sum to 1.
- ESPN BPI (NBA) and FPI (NFL, college football):
  https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{nba|nfl|college-football}/seasons/{season}/powerindex?limit=400
  items[].team.$ref (ends /teams/{id}), lastUpdated, and stats[] (BPI: name "bpi") or predictives[] (FPI:
  name "fpi"): "expected point margin vs average opponent on neutral court/field". NBA 2026-27 is season 2027
  (30 teams), NFL 2026 (32), FBS 2026 (138). Men's college basketball BPI was still empty (count 0).
  Team ids are named with https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams?limit=1000
  (sports[0].leagues[0].teams[].team: id, abbreviation, displayName, shortDisplayName, location, nickname).
- Bart Torvik T-Rank (men's college basketball): https://barttorvik.com/{season}_team_results.json, a list of
  45-column rows: [1] team, [2] conference, [4] adjusted offensive efficiency (points per 100 possessions),
  [6] adjusted defensive efficiency, [8] barthag, [44] adjusted tempo (possessions per 40 minutes). The
  2027 file (2026-27 pre-season) was live, last-modified 2026-10-27.
- Tennis Abstract Elo: https://www.tennisabstract.com/reports/{atp|wta}_elo_ratings.html, the table
  id="reportable": Elo rank, Player, Age, Elo, (blank), hElo rank, hElo, cElo rank, cElo, gElo rank, gElo...
  ("Updated weekly(ish). Last update: 2026-10-26").
- eloratings.net (national teams): https://www.eloratings.net/World.tsv (tab-separated: rank, rank, team
  code, rating, ...) and https://www.eloratings.net/en.teams.tsv (code, name[, other names]).
- Not used: ClubElo (its API is deactivated or answers 502, per the research notes), so club soccer has no
  rating source; nor do the NHL, MLB, WNBA, MMA or the smaller leagues. Their power() is {} and game_prob None.

Formulas (game_prob):
- Expected-margin ratings (BPI, FPI): margin = r_home - r_away + home edge (0 on a neutral site), and
  p_home = Phi(margin / sd), a normal CDF. Constants per league:
    NBA: home edge 2.0 points ("markets implied 2.05 points over 2022-25", research_notes/Prediction lab
         logic and maths/game_models_math.md), sd 12.0 (same notes: "use a normal sigma of about 11.5-12" to
         turn a spread into a win chance; stage spec).
    NFL: home edge 1.7 points (research_notes/Sports data sources and analysis UX/venues_travel_weather.md,
         "Home-advantage defaults for v3"), sd 12.7 (stats_and_models.md: margin around the spread "SD 12.72";
         stage spec).
    College football: home edge 2.5 points (the figure most outlets assume, per Action Network's college
         home-field write-ups; an East Carolina study of FBS rivalries found about 2.3), sd 15.5 points
         (an assumption: no measured value was found; wider than the NFL's because college mismatches are
         larger. To be refitted from the record).
- Torvik (college basketball): each side's points per 100 possessions = its AdjO x the opponent's AdjD /
  the league-average efficiency, at AdjT_home x AdjT_away / the league-average tempo possessions (the
  adjusted-efficiency matchup KenPom-style ratings use; averages taken from the same file); margin = the
  difference + 3.0 points at home (an assumed, commonly used D-I home edge), p = Phi(margin / 11.0) (stage
  spec: "college basketball about 11").
- Elo: p = 1 / (1 + 10^(-d/400)).
    Tennis: d = r_home - r_away with r = 0.5 x overall Elo + 0.5 x surface Elo when a surface is given
         (Tennis Abstract, "An introduction to tennis Elo", 2019: the 50/50 blend "worked on every surface"),
         else overall Elo; no home edge.
    National teams: d = r_home - r_away + 100 at home (eloratings.net's method, as on Wikipedia's "World
         Football Elo Ratings": "dr equals the difference in ratings (add 100 points for the home team)").
         That gives the expected score We (a draw counts half). The draw chance is 4 x D0 x We x (1 - We),
         D0 = 0.27 at an even match (the stage spec's soccer base-rate draw share) and smaller as the match
         gets more lopsided; this shape is an assumption, to be refitted from the record. home = We - draw/2
         and away = 1 - We - draw/2, so We stays exactly eloratings.net's.
"""

from __future__ import annotations

import datetime
import html
import logging
import math
import re
import statistics
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..cache import ttl_for
from .aliases import _person_score, _plain, _team_score, kind as name_kind, team_key
from .common import HttpError, Keeper, http_get, iso, num, season_now, text

log = logging.getLogger(__name__)

# From the shared cache's policy table (data/cache.py's ttl_for), the one place these times live:
POWER_TTL = ttl_for("ratings").ttl  # ratings: 12 hours
TEAMS_TTL = ttl_for("teams").ttl  # ESPN's team lists change a few times a year: a week
PREDICTOR_TTL = ttl_for("predictor").ttl  # ESPN's predictor: 1 hour
PREDICTOR_KEEP = 3 * 24 * 3600  # predictors saved for a restart are kept this long

CORE = "https://sports.core.api.espn.com/v2/sports"
SITE = "https://site.api.espn.com/apis/site/v2/sports"
TORVIK = "https://barttorvik.com/{season}_team_results.json"
TENNIS_ABSTRACT = "https://www.tennisabstract.com/reports/{tour}_elo_ratings.html"
ELO_WORLD = "https://www.eloratings.net/World.tsv"
ELO_NAMES = "https://www.eloratings.net/en.teams.tsv"

PREDICTOR_LEAGUES = {"americanfootball_nfl": ("football", "nfl"),
                     "americanfootball_ncaaf": ("football", "college-football"),
                     "basketball_nba": ("basketball", "nba")}

# league -> (ESPN sport, ESPN league path, index kind, stat name)
ESPN_INDEX = {
    "basketball_nba": ("basketball", "nba", "BPI", "bpi"),
    "americanfootball_nfl": ("football", "nfl", "FPI", "fpi"),
    "americanfootball_ncaaf": ("football", "college-football", "FPI", "fpi"),
}
# Expected-margin constants (see the module notes for sources): home edge in points, margin SD in points
MARGIN = {
    "basketball_nba": (2.0, 12.0),
    "americanfootball_nfl": (1.7, 12.7),
    "americanfootball_ncaaf": (2.5, 15.5),
    "basketball_ncaab": (3.0, 11.0),
}
ELO_HOME = {"national": 100.0, "tennis": 0.0}  # Elo points added to the home side, per kind of league
DRAW_SHARE = 0.27  # draw chance of an even soccer match (stage spec base rate 45/27/28)
SURFACES = {"hard": "hard", "indoor": "hard", "indoor hard": "hard", "carpet": "hard", "clay": "clay",
            "red clay": "clay", "green clay": "clay", "grass": "grass"}
# A tennis tournament's surface from its name (ESPN's event name, "Mutua Madrid Open") or the odds feed's league key
# (tennis_atp_french_open): the tour-level clay and grass events, by city or title. Anything else is hard (indoor
# counts as hard: Tennis Abstract keeps hard, clay and grass Elo only). Two cities depend on the tour: Stuttgart is
# the WTA's clay event in April and the ATP's grass event in June; Lyon the ATP's clay event and the WTA's indoor
# hard one.
CLAY_EVENTS = (
    "roland garros", "french open", "monte carlo", "madrid", "rome", "italian open", "internazionali", "barcelona",
    "munich", "geneva", "hamburg", "bastad", "gstaad", "umag", "croatia open", "kitzbuhel", "houston", "clay court",
    "marrakech", "hassan", "bucharest", "tiriac", "estoril", "cordoba", "buenos aires", "argentina open", "rio open",
    "rio de janeiro", "santiago", "chile open", "charleston", "bogota", "colsanitas", "rabat", "lalla meryem",
    "strasbourg", "palermo", "budapest", "hungarian grand prix", "lausanne", "iasi", "rouen", "parma", "porsche",
)
GRASS_EVENTS = (
    "wimbledon", "queen s", "queens club", "halle", "terra wortmann", "hertogenbosch", "libema", "mallorca",
    "eastbourne", "nottingham", "birmingham", "rothesay", "bad homburg", "berlin", "hsbc championships", "cinch",
    "boss open",
)
NATIONAL_TAGS = ("world_cup", "nations_league", "copa_america", "european_championship", "euro_qualification",
                 "africa_cup", "gold_cup", "asian_cup", "friendlies", "wc_qualif")


def phi(x: float) -> float:
    """The standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def elo_p(diff: float) -> float:
    """Elo's expected score for a rating difference: 1 / (1 + 10^(-d/400))."""
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def surface_of(league: str, event_name: Any = None) -> Optional[str]:
    """"clay", "grass" or "hard" for a tennis league's match (see CLAY_EVENTS), None outside tennis."""
    tour = tennis_tour(league or "")
    if tour is None:
        return None
    words = f" {_plain(event_name)} {_plain(str(league).replace('_', ' '))} "
    if " stuttgart " in words:
        return "clay" if tour == "wta" else "grass"
    if " lyon " in words:
        return "clay" if tour == "atp" else "hard"
    for surface, names in (("clay", CLAY_EVENTS), ("grass", GRASS_EVENTS)):
        if any(f" {name} " in words for name in names):
            return surface
    return "hard"


def is_national(league: str) -> bool:
    return league.startswith("soccer_") and "club" not in league and any(tag in league for tag in NATIONAL_TAGS)


def tennis_tour(league: str) -> Optional[str]:
    if league == "tennis_atp" or league.startswith("tennis_atp_"):
        return "atp"
    if league == "tennis_wta" or league.startswith("tennis_wta_"):
        return "wta"
    return None


def source_kind(league: str) -> Optional[str]:
    """Which rating a league gets: "BPI", "FPI", "Torvik", "Elo" (tennis, national teams), or None."""
    if league in ESPN_INDEX:
        return ESPN_INDEX[league][2]
    if league == "basketball_ncaab":
        return "Torvik"
    if tennis_tour(league) or is_national(league):
        return "Elo"
    return None


# --- parsers (pure: tests feed them trimmed live captures) ------------------------------------------------

def parse_predictor(summary: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """{"home", "away"} chances from a summary's "predictor" (scaled to sum to 1), or None."""
    predictor = summary.get("predictor") if isinstance(summary, Mapping) else None
    if not isinstance(predictor, Mapping):
        return None
    home, away = predictor.get("homeTeam") or {}, predictor.get("awayTeam") or {}
    p_home, p_away = num(home.get("gameProjection")), num(away.get("gameProjection"))
    if p_home is None or p_away is None or p_home < 0 or p_away < 0 or p_home + p_away <= 0:
        return None
    # The predictor names its teams by id; if they disagree with the header's home/away, follow the header
    competition = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    sides = {str((c.get("team") or {}).get("id")): c.get("homeAway") for c in competition.get("competitors") or []
             if isinstance(c, dict)}
    if sides.get(str(home.get("id"))) == "away" and sides.get(str(away.get("id"))) == "home":
        p_home, p_away = p_away, p_home
    total = p_home + p_away
    return {"home": p_home / total, "away": p_away / total}


def team_names(teams_answer: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """ESPN team id -> {abbreviation, displayName, shortDisplayName, location, nickname, name}."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        teams = teams_answer["sports"][0]["leagues"][0]["teams"]
    except (KeyError, IndexError, TypeError):
        return out
    for item in teams:
        team = item.get("team") if isinstance(item, dict) else None
        if isinstance(team, dict) and team.get("id") is not None:
            out[str(team["id"])] = {field: text(team.get(field), 60) for field in
                                    ("abbreviation", "displayName", "shortDisplayName", "location", "nickname", "name")}
    return out


def _espn_key(league: str, team: Mapping[str, Any]) -> Optional[str]:
    """The team's key: its abbreviation in the table leagues (NBA, NFL), its school name in college."""
    fields = ("abbreviation", "location", "shortDisplayName", "displayName")
    if name_kind(league) != "table":
        fields = ("location", "shortDisplayName", "displayName", "abbreviation")
    for field in fields:
        key = team_key(league, team.get(field))
        if key:
            return key
    return None


def parse_power_index(league: str, answer: Mapping[str, Any], teams: Mapping[str, Mapping[str, Any]]
                      ) -> Dict[str, Dict[str, Any]]:
    """{team key: {"rating", "kind", "as_of", "name", "id", "names"}} from an ESPN powerindex answer."""
    _sport, _path, kind, stat = ESPN_INDEX[league]
    out: Dict[str, Dict[str, Any]] = {}
    for item in (answer or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        ref = str((item.get("team") or {}).get("$ref") or "")
        match = re.search(r"/teams/(\d+)", ref)
        values = [s for group in ("stats", "predictives") for s in item.get(group) or [] if isinstance(s, dict)]
        rating = next((num(s.get("value")) for s in values if s.get("name") == stat), None)
        if not match or rating is None:
            continue
        team = teams.get(match.group(1)) or {}
        key = _espn_key(league, team)
        if not key:
            continue
        names = [team[field] for field in ("abbreviation", "displayName", "shortDisplayName", "location")
                 if team.get(field)]
        out[key] = {"rating": rating, "kind": kind, "as_of": _iso_minute(item.get("lastUpdated")),
                    "name": team.get("displayName"), "id": match.group(1), "names": names}
    return out


def _iso_minute(value: Any) -> Optional[str]:
    """ESPN's "2026-10-26T23:54Z" as "2026-10-26T23:54:00Z"."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.datetime.strptime(value[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_torvik(rows: Sequence[Any], as_of: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """{team key: {"rating" (AdjEM, points per 100 possessions), "kind": "Torvik", "adjoe", "adjde", "tempo",
    "as_of", "name"}} from Torvik's team_results rows."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, list) or len(row) < 45:
            continue
        name = text(row[1], 60)
        adjoe, adjde, tempo = num(row[4]), num(row[6]), num(row[44])
        key = team_key("basketball_ncaab", name)
        if not key or adjoe is None or adjde is None or not tempo:
            continue
        out[key] = {"rating": round(adjoe - adjde, 3), "kind": "Torvik", "adjoe": adjoe, "adjde": adjde,
                    "tempo": tempo, "as_of": as_of, "name": name, "names": [name]}
    return out


_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def parse_tennis_abstract(page: str, league: str) -> Dict[str, Dict[str, Any]]:
    """{player key: {"rating" (overall Elo), "kind": "Elo", "hard", "clay", "grass", "as_of", "name"}} from a
    Tennis Abstract Elo report page."""
    if not isinstance(page, str):
        return {}
    updated = re.search(r"Last update:\s*(\d{4}-\d{2}-\d{2})", page)
    as_of = f"{updated.group(1)}T00:00:00Z" if updated else None
    start = page.find('id="reportable"')
    out: Dict[str, Dict[str, Any]] = {}
    for row in _ROW.findall(page[start:] if start >= 0 else ""):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", cell)).replace("\xa0", " ").strip() for cell in _CELL.findall(row)]
        if len(cells) < 11:
            continue
        name, elo = text(cells[1], 60), num(cells[3])
        key = team_key(league, name)
        if not key or elo is None:
            continue
        out[key] = {"rating": elo, "kind": "Elo", "hard": num(cells[6]), "clay": num(cells[8]),
                    "grass": num(cells[10]), "as_of": as_of, "name": name, "names": [name]}
    return out


def parse_eloratings(world_tsv: str, names_tsv: str, league: str, as_of: Optional[str]
                     ) -> Dict[str, Dict[str, Any]]:
    """{team key: {"rating", "kind": "Elo", "as_of", "name", "code"}} from eloratings.net's World.tsv and
    en.teams.tsv."""
    names: Dict[str, List[str]] = {}
    for line in (names_tsv or "").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 2:
            names[parts[0]] = parts[1:]
    out: Dict[str, Dict[str, Any]] = {}
    for line in (world_tsv or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code, rating = parts[2].strip(), num(parts[3])
        spellings = names.get(code)
        if not spellings or rating is None:
            continue
        key = team_key(league, spellings[0])
        if key:
            out[key] = {"rating": rating, "kind": "Elo", "as_of": as_of, "name": text(spellings[0], 60),
                        "code": code, "names": [text(s, 60) for s in spellings]}
    return out


# --- the client ---------------------------------------------------------------------------------------

class Ratings:
    """Outside model opinions per league. ``power(league)`` and ``espn_predictor(league, event)`` fetch when
    their copy is due (for the background worker); ``cached_power(league)`` never fetches;
    ``game_prob(...)`` turns ratings into chances (with ``fetch=False`` it reads only what is cached).
    Failures keep the last good copy; nothing raises."""

    source = "ratings"

    def __init__(self, espn: Any = None, cache: Any = None, store: Any = None, *,
                 fetch: Optional[Callable[..., Tuple[int, Any]]] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.espn = espn
        self.cache = cache  # accepted for the data clients' shared-cache convention; copies live in the keeper
        self.store = store
        self.clock = clock or time.time
        self._fetch = fetch or http_get
        self.keeper = Keeper(self.source, store, self.clock)
        self._predictors = Keeper("ratings-predictor", None, self.clock)  # per game, in memory ...
        self._predictor_seeded: set = set()  # ... seeded from (and saved to) one store blob per league
        # Name lookups per rating table (the slate asks for every game on every rebuild): league -> (the table they
        # were made on, {spellings: table key or None}, the table's averages). A new table starts a new memo.
        self._memo: Dict[str, Tuple[Any, Dict[Tuple[str, ...], Optional[str]], Dict[str, float]]] = {}

    # --- HTTP

    def _json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        status, body = self._fetch(url, params)
        if status != 200 or body is None:
            raise HttpError(status, url.split("/")[2])
        return body

    def _text(self, url: str) -> str:
        status, body = self._fetch(url, None, text=True)
        if status != 200 or not isinstance(body, str):
            raise HttpError(status, url.split("/")[2])
        return body

    # --- ESPN predictor

    def espn_predictor(self, league: str, espn_event_id: Any) -> Optional[Dict[str, Any]]:
        """{"home", "away", "as_of"} from ESPN's matchup predictor (NFL, college football, NBA), or None."""
        if league not in PREDICTOR_LEAGUES or self.espn is None or espn_event_id in (None, ""):
            return None
        event_id = str(espn_event_id)
        self._seed_predictors(league)
        key = (league, event_id)
        value, fetched_at = self._predictors.get_with_time(key, PREDICTOR_TTL, lambda: self._load_predictor(league, event_id))
        if not value:
            return None
        return {**value, "as_of": iso(fetched_at)}

    def _load_predictor(self, league: str, event_id: str) -> Optional[Dict[str, float]]:
        value = parse_predictor(self.espn.summary(league, event_id))
        self._save_predictors(league, event_id, value)
        return value

    def _blob_key(self, league: str) -> Tuple[str, str]:
        return ("predictors", league)

    def _seed_predictors(self, league: str) -> None:
        if league in self._predictor_seeded:
            return
        self._predictor_seeded.add(league)
        saved = self.keeper.peek(self._blob_key(league))
        for event_id, entry in ((saved[1] if saved else None) or {}).items():
            if isinstance(entry, list) and len(entry) == 2 and num(entry[0]) is not None:
                self._predictors.seed((league, event_id), float(entry[0]), entry[1])

    def _save_predictors(self, league: str, event_id: str, value: Optional[Dict[str, float]]) -> None:
        now = self.clock()
        saved = self.keeper.peek(self._blob_key(league))
        blob = {k: v for k, v in ((saved[1] if saved else None) or {}).items()
                if isinstance(v, list) and len(v) == 2 and now - (num(v[0]) or 0.0) < PREDICTOR_KEEP}
        blob[event_id] = [now, value]
        self.keeper.put(self._blob_key(league), blob)

    # --- power ratings

    def power(self, league: str) -> Dict[str, Dict[str, Any]]:
        """{team key: {"rating", "kind": "BPI"|"FPI"|"Elo"|"Torvik", "as_of", ...}} for the league (refreshed
        every 12 hours); {} where no rating source covers it."""
        loader = self._power_loader(league)
        if loader is None:
            return {}
        return self.keeper.get(("power", self._power_league(league)), POWER_TTL, loader, empty={}) or {}

    def cached_power(self, league: str) -> Dict[str, Dict[str, Any]]:
        copy = self.keeper.peek(("power", self._power_league(league)))
        return (copy[1] if copy else None) or {}

    @staticmethod
    def _power_league(league: str) -> str:
        tour = tennis_tour(league)
        if tour:
            return f"tennis_{tour}"
        if is_national(league):
            return "soccer_national"
        return league

    def _power_loader(self, league: str) -> Optional[Callable[[], Dict[str, Dict[str, Any]]]]:
        if league in ESPN_INDEX:
            return lambda: self._load_espn_index(league)
        if league == "basketball_ncaab":
            return self._load_torvik
        tour = tennis_tour(league)
        if tour:
            return lambda: self._load_tennis(tour)
        if is_national(league):
            return self._load_national
        return None

    def _teams(self, league: str) -> Dict[str, Dict[str, Any]]:
        sport, path = ESPN_INDEX[league][:2]

        def load() -> Dict[str, Dict[str, Any]]:
            names = team_names(self._json(f"{SITE}/{sport}/{path}/teams", {"limit": 1000}))
            if not names:
                raise ValueError("no teams")
            return names

        return self.keeper.get(("teams", league), TEAMS_TTL, load, empty={}) or {}

    def _load_espn_index(self, league: str) -> Dict[str, Dict[str, Any]]:
        sport, path = ESPN_INDEX[league][:2]
        season = season_now(sport, self.clock())
        answer = self._json(f"{CORE}/{sport}/leagues/{path}/seasons/{season}/powerindex", {"limit": 400})
        teams = self._teams(league)
        if not teams:
            raise ValueError("team names unavailable")
        ratings = parse_power_index(league, answer, teams)
        if not ratings:
            raise ValueError("empty power index")
        return ratings

    def _load_torvik(self) -> Dict[str, Dict[str, Any]]:
        season = season_now("basketball", self.clock())
        rows = self._json(TORVIK.format(season=season))
        ratings = parse_torvik(rows, iso(self.clock()))
        if not ratings:
            raise ValueError("empty Torvik file")
        return ratings

    def _load_tennis(self, tour: str) -> Dict[str, Dict[str, Any]]:
        ratings = parse_tennis_abstract(self._text(TENNIS_ABSTRACT.format(tour=tour)), f"tennis_{tour}")
        if not ratings:
            raise ValueError("empty Elo table")
        return ratings

    def _load_national(self) -> Dict[str, Dict[str, Any]]:
        world, names = self._text(ELO_WORLD), self._text(ELO_NAMES)
        ratings = parse_eloratings(world, names, "soccer_national", iso(self.clock()))
        if not ratings:
            raise ValueError("empty Elo table")
        return ratings

    # --- chances

    def _memo_of(self, league: str, table: Mapping[str, Mapping[str, Any]]) -> Any:
        return _memo_entry(self._memo, league, table)

    def _resolve(self, league: str, table: Mapping[str, Mapping[str, Any]], side: Any) -> Optional[str]:
        """_resolve, remembered per table (misses too): a table is replaced, never changed, when it is refreshed."""
        names = self._memo_of(league, table)[1]
        spellings = _spellings(side)
        if spellings not in names:
            if len(names) > 20000:  # a bound, never reached by a real league's names
                names.clear()
            names[spellings] = _resolve(league, table, spellings)
        return names[spellings]

    def game_prob(self, league: str, home: Any, away: Any, neutral: bool = False, surface: Optional[str] = None,
                  *, fetch: bool = True) -> Optional[Dict[str, Any]]:
        """{"home", "away", "draw"?, "source", "as_of", "margin"?, "detail"} for one game from the league's
        ratings (formulas in the module notes), or None when there is no rating for either side. ``home`` and
        ``away`` may be names, abbreviations or the slate's team dicts (name/short/abbr)."""
        kind = source_kind(league)
        if kind is None:
            return None
        table = self.power(league) if fetch else self.cached_power(league)
        if not table:
            return None
        rating_league = "soccer_national" if is_national(league) else league
        h, a = self._resolve(rating_league, table, home), self._resolve(rating_league, table, away)
        if h is None or a is None or h == a:
            return None
        rh, ra = table[h], table[a]
        as_of = min((x for x in (rh.get("as_of"), ra.get("as_of")) if x), default=None)
        if kind in ("BPI", "FPI"):
            edge, sd = MARGIN[league]
            margin = rh["rating"] - ra["rating"] + (0.0 if neutral else edge)
            p = phi(margin / sd)
            detail = (f"{kind} {rh['rating']:+.1f} vs {ra['rating']:+.1f}"
                      f"{'' if neutral else f', home {edge:+.1f}'} = margin {margin:+.1f}, sd {sd}")
            return {"home": p, "away": 1.0 - p, "source": kind, "as_of": as_of, "margin": round(margin, 2),
                    "detail": detail}
        if kind == "Torvik":
            margin = _torvik_margin(table, rh, ra, 0.0 if neutral else MARGIN[league][0],
                                    averages=self._memo_of(rating_league, table)[2])
            sd = MARGIN[league][1]
            p = phi(margin / sd)
            return {"home": p, "away": 1.0 - p, "source": "Torvik", "as_of": as_of, "margin": round(margin, 2),
                    "detail": f"Torvik AdjEM {rh['rating']:+.1f} vs {ra['rating']:+.1f}, margin {margin:+.1f}, sd {sd}"}
        # Elo
        if tennis_tour(league):
            side = SURFACES.get(str(surface or "").lower())

            def blended(row: Mapping[str, Any]) -> float:
                if side and row.get(side) is not None:
                    return 0.5 * row["rating"] + 0.5 * row[side]
                return row["rating"]

            if side and (rh.get(side) is None or ra.get(side) is None):
                side = None
            diff = blended(rh) - blended(ra)
            p = elo_p(diff)
            what = f"50/50 overall and {side} Elo" if side else "overall Elo"
            return {"home": p, "away": 1.0 - p, "source": "Elo", "as_of": as_of,
                    "detail": f"{what}: {blended(rh):.0f} vs {blended(ra):.0f}"}
        diff = rh["rating"] - ra["rating"] + (0.0 if neutral else ELO_HOME["national"])
        expected = elo_p(diff)
        draw = 4.0 * DRAW_SHARE * expected * (1.0 - expected)
        return {"home": expected - draw / 2, "away": 1.0 - expected - draw / 2, "draw": draw, "source": "Elo",
                "as_of": as_of, "detail": f"eloratings.net {rh['rating']:.0f} vs {ra['rating']:.0f}"
                f"{'' if neutral else ', +100 at home'}: expected score {expected:.3f}"}


def _memo_entry(memo: Dict[str, Any], league: str, table: Mapping[str, Mapping[str, Any]]) -> Any:
    entry = memo.get(league)
    if entry is None or entry[0] is not table:
        entry = (table, {}, _torvik_averages(table) if source_kind(league) == "Torvik" else {})
        memo[league] = entry
    return entry


def _torvik_averages(table: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """The league-average efficiency and tempo of a Torvik table."""
    rows = [row for row in table.values() if row.get("adjoe") and row.get("tempo")]
    return {"eff": statistics.fmean(row["adjoe"] for row in rows),
            "tempo": statistics.fmean(row["tempo"] for row in rows)}


def _torvik_margin(table: Mapping[str, Mapping[str, Any]], home: Mapping[str, Any], away: Mapping[str, Any],
                   edge: float, averages: Optional[Dict[str, float]] = None) -> float:
    if not averages:
        averages = _torvik_averages(table)
    avg_eff, avg_tempo = averages["eff"], averages["tempo"]
    possessions = home["tempo"] * away["tempo"] / avg_tempo
    home_ppp = home["adjoe"] * away["adjde"] / avg_eff
    away_ppp = away["adjoe"] * home["adjde"] / avg_eff
    return (home_ppp - away_ppp) * possessions / 100.0 + edge


def _spellings(side: Any) -> Tuple[str, ...]:
    """Every spelling of a side given as a name, an abbreviation or a dict of names."""
    if isinstance(side, Mapping):
        return tuple(str(side[f]) for f in ("abbr", "short", "name", "abbreviation", "displayName",
                                            "shortDisplayName", "location") if side.get(f))
    if isinstance(side, (list, tuple)):
        return tuple(str(s) for s in side if s)
    return (str(side),) if side not in (None, "") else ()


def _resolve(league: str, table: Mapping[str, Mapping[str, Any]], side: Any) -> Optional[str]:
    """The table key for a side given as a name, an abbreviation or a dict of names: an exact key first, then
    a key of one of the table's spellings, then the single best close name (one name inside the other, or the
    same surname and initial)."""
    spellings = list(_spellings(side))
    keys = [k for k in (team_key(league, s) for s in spellings) if k]
    for key in keys:
        if key in table:
            return key
    for key in keys:
        for name, row in table.items():
            if any(team_key(league, other) == key for other in row.get("names") or []):
                return name
    person = name_kind(league) == "person"
    scorer = _person_score if person else _team_score
    wanted = set(keys) | {_plain(s) for s in spellings if _plain(s)}
    # Ranked by score, then by the words shared: "Northern Illinois Huskies" scores 0.85 against both "Illinois" and
    # "Northern Illinois" (each inside it), and the one sharing more words is the team
    best, best_rank, tie = None, (0.0, 0), False
    for name, row in table.items():
        candidates = {name} | {_plain(n) for n in row.get("names") or [] if n}
        rank = max(((scorer(w, c), len(set(w.split()) & set(c.split()))) for w in wanted for c in candidates),
                   default=(0.0, 0))
        if rank > best_rank:
            best, best_rank, tie = name, rank, False
        elif rank == best_rank and rank[0] > 0:
            tie = True
    return best if best is not None and best_rank[0] >= 0.85 and not tie else None
