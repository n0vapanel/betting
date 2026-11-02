"""The player scan: who reaches a number tonight ("20+ points", "10+ rebounds", "anytime TD", "7+ strikeouts").

For one league and one day's games (the board, data/player_board.py), every player gets four numbers:

- **History:** how often he reached the number in his last 5 and last 10 games and this season
  (``espn_players.hit_rates`` against the number minus a half), with his averages over the same windows.
- **Projection:** tonight's expected figure. NBA players use the NBA model's own projection for the game
  (``service.detail``: minutes, usage, pace and injuries); everyone else uses the form model below.
- **Chance:** the chance of reaching the number. For the NBA it is ``calls.stat_probabilities`` on the model's
  mean, exactly as the player card works it out, so the two never disagree on the same line.
- **Posted line:** DraftKings' line and price for the stat (through ESPN, never The Odds API), with the edge
  when the price is for the number asked (a "20+" milestone, or an over at 19.5) and the line isn't stale. An
  under matches only an over/under line (a milestone is an over's bet).

Players are ranked by that chance (then the projection), strongest first; ``sort`` asks for another order
(projection, last-10 hit rate, edge, or "hot" and "cold": this season's last 5 games against a baseline, ``trend``).
A scan can be for named players only (``names``: each name it can't rank is missing, ambiguous or in a game already
under way, sort_names), and scan_edges ranks every stat's posted lines by edge.

The form model (every league without the NBA model, and NBA players the model has no projection for): a
recency-weighted mean of the last 10 games (exponential weights, half-life 5 games), shrunk toward this
season's per-game average by games played (k = min(5, this season's games)). Counts (points, rebounds, shots,
strikeouts) use a negative binomial whose dispersion is fitted from the log's own variance, never below a Poisson;
yards use a normal with the log's spread, at least 15% of the mean and 10 yards; "anytime" asks are 1 - P(0) of the
count, "2+ TDs" is P(2 or more) of the same count. A double- or triple-double is a yes/no event: the
recency-weighted share of games that reached it, shrunk toward the season's share the same way (a beta-binomial).
A yes/no row (``kind`` "yes_no") shows only its chance. Every row says which model it came from (``model``: "nba"
or "form"); the form model is labelled "from recent form".

Thresholds: an ask is a whole number to reach ("20+" is 20). The line behind it is that number minus a half
(19.5), so a game counts as a hit at 20 or more; an "under" ask counts the games below the number. "Over 19.5"
and "at least 20" are both 20; "under 9.5" is under 10.

Everything here is pure: numbers in, numbers out. Nothing fetches.
"""

from __future__ import annotations

import datetime
import math
import re
import time
import unicodedata
from decimal import Decimal
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from ..data.espn_players import hit_rates
# STAT_DISPERSION is engine/calls.STAT_VARIANCE_RATIO under its old name: per-stat variance ratios
# (variance / mean) of an NB1 negative binomial (calls.NB_FAMILY), not NB2 sizes like learn/stats.nb_logpmf's
from .calls import STAT_DISPERSION, _negative_binomial_cdf, stat_probabilities
from .odds_math import decimal_to_american, format_american

_NORMAL = NormalDist()

# --- the stats and their lingo ------------------------------------------------------------------------------

NBA = "basketball_nba"
WNBA = "basketball_wnba"
NFL = "americanfootball_nfl"
NHL = "icehockey_nhl"
MLB = "baseball_mlb"


def _stat(key: str, label: str, unit: str, aliases: Iterable[str], log: Optional[str], kind: str = "count", *,
          nba: Optional[str] = None, parts: Iterable[str] = (), markets: Iterable[str] = (),
          needs: Iterable[str] = (), threshold: Optional[int] = None, short: Optional[str] = None,
          hit_word: Optional[str] = None, key_alias: bool = True, per: Optional[str] = None) -> Dict[str, Any]:
    """One stat of a league's table. ``log`` is the game-log column or sum ("pts+reb+ast", "RUSH TD+REC TD"; a sum
    counts the columns a row has); ``kind`` is "count", "yards", "anytime" (a 1+ yes/no ask) or "double" (double- and
    triple-doubles); ``nba`` the NBA model's stat for ``calls.stat_probabilities`` when it has a measured spread;
    ``parts`` the NBA projection's stats summed for the projection; ``markets`` DraftKings' market keys
    (espn_players.PROP_MARKETS); ``needs`` columns a game-log row must carry to count (an MLB pitcher's "H" is hits
    allowed, so batters' stats need "AB"); ``threshold`` a fixed number to reach (anytime 1, double-double 2);
    ``hit_word`` how a hit reads at 1+ ("a TD in 6 of last 10"); ``key_alias`` False keeps a key that means
    something else in bettor talk out of the phrase matching ("td" is a touchdown, not a triple-double); ``per``
    what a yes/no stat's average counts ("TDs": "0.6 TDs a game")."""
    return {
        "key": key, "label": label, "unit": unit, "short": short or label,
        "aliases": tuple(dict.fromkeys(_norm(a) for a in ((key,) if key_alias else ()) + (label, *aliases))),
        "log": log, "kind": kind, "nba": nba, "parts": tuple(parts), "markets": tuple(markets),
        "needs": tuple(needs), "threshold": threshold, "yes": kind in ("anytime", "double"),
        "hit_word": hit_word, "per": per or label,
    }


def _norm(text: Any) -> str:
    """Lower case, accents off, "+" kept tight ("pts + reb" -> "pts+reb"), single spaces."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    text = text.replace("’", "'").replace("-", " ")
    text = re.sub(r"(?<![0-9])\s*\+\s*(?=[a-z])", "+", text)  # "pts + reb", not "20+ points"
    return " ".join(text.split())


# The usage bump in the NBA model covers these stats (nba_model.USAGE_BUMP), so a teammate out is a reason for them
USAGE_STATS = ("pts", "ast", "fg3m", "reb", "pra", "pr", "pa", "ra")
# The double-double categories, each projected by the NBA model
DOUBLE_PARTS = ("pts", "reb", "ast", "stl", "blk")

BASKETBALL = {s["key"]: s for s in (
    _stat("pts", "points", "pts", ("pts", "point", "pt", "scorers", "scorer", "scoring", "buckets", "bucket",
                                   "score"), "pts", nba="pts", parts=("pts",), markets=("player_points",)),
    _stat("reb", "rebounds", "reb", ("rebound", "rebs", "boards", "board", "rebounding", "rbs"), "reb",
          nba="reb", parts=("reb",), markets=("player_rebounds",)),
    _stat("ast", "assists", "ast", ("assist", "asts", "dimes", "dime", "dishes"), "ast", nba="ast", parts=("ast",),
          markets=("player_assists",)),
    _stat("fg3m", "threes", "3PM", ("three", "3s", "3pm", "3pt", "3pts", "3 pointers", "3 pointer", "three pointers",
                                    "three pointer", "triples", "treys", "trey", "threes made", "made threes",
                                    "3 point field goals", "3ptm"), "fg3m", nba="fg3m", parts=("fg3m",),
          markets=("player_threes",)),
    _stat("stl", "steals", "stl", ("steal", "stls"), "stl", nba="stl", parts=("stl",), markets=("player_steals",)),
    _stat("blk", "blocks", "blk", ("block", "blks", "swats", "rejections"), "blk", nba="blk", parts=("blk",),
          markets=("player_blocks",)),
    _stat("stocks", "steals + blocks", "stl+blk", ("stl+blk", "blk+stl", "steals and blocks", "steals+blocks",
                                                    "blocks+steals", "stock"),
          "stl+blk", parts=("stl", "blk"), markets=("player_blocks_steals",)),
    _stat("pra", "points + rebounds + assists", "PRA", ("pts+reb+ast", "p+r+a", "points rebounds assists",
                                                         "points rebounds and assists", "points+rebounds+assists"),
          "pts+reb+ast", nba="pra", parts=("pts", "reb", "ast"), markets=("player_points_rebounds_assists",),
          short="PRA"),
    _stat("pr", "points + rebounds", "P+R", ("pts+reb", "p+r", "points and rebounds", "points+rebounds"),
          "pts+reb", parts=("pts", "reb"), markets=("player_points_rebounds",), short="P+R"),
    _stat("pa", "points + assists", "P+A", ("pts+ast", "p+a", "points and assists", "points+assists"),
          "pts+ast", parts=("pts", "ast"), markets=("player_points_assists",), short="P+A"),
    _stat("ra", "rebounds + assists", "R+A", ("reb+ast", "r+a", "rebounds and assists", "rebounds+assists",
                                               "assists+rebounds"),
          "reb+ast", parts=("reb", "ast"), markets=("player_rebounds_assists",), short="R+A"),
    _stat("dd", "double-double", "", ("double double", "double doubles", "dubdub", "dub dub"), None, "double",
          markets=("player_double_double",), threshold=2, hit_word="a double-double"),
    _stat("td", "triple-double", "", ("triple double", "triple doubles", "trip dub", "trip dubs"), None, "double",
          markets=("player_triple_double",), threshold=3, hit_word="a triple-double", key_alias=False),
)}

FOOTBALL = {s["key"]: s for s in (
    _stat("pass_yds", "passing yards", "yds", ("pass yards", "pass yds", "passing yds", "passing", "py", "yards",
                                               "yds", "yardage"),
          "PASS YDS", "yards", markets=("player_pass_yds",)),
    _stat("pass_td", "passing TDs", "TD", ("passing touchdowns", "pass tds", "pass td", "passing td", "td passes",
                                           "touchdown passes", "passing touchdown"),
          "PASS TD", markets=("player_pass_tds",), hit_word="a TD pass"),
    _stat("rush_yds", "rushing yards", "yds", ("rush yards", "rush yds", "rushing yds", "rushing", "ground yards"),
          "RUSH YDS", "yards", markets=("player_rush_yds",)),
    _stat("rec_yds", "receiving yards", "yds", ("rec yards", "rec yds", "receiving yds", "receiving"),
          "REC YDS", "yards", markets=("player_reception_yds",)),
    _stat("rec", "receptions", "rec", ("reception", "catches", "catch", "recs"), "REC",
          markets=("player_receptions",)),
    _stat("rush_rec_yds", "rushing + receiving yards", "yds", ("rush+rec yards", "rush+rec yds", "rush+rec",
                                                               "scrimmage yards", "rushing and receiving yards",
                                                               "rush and rec yards"),
          "RUSH YDS+REC YDS", "yards", markets=("player_rush_reception_yds",), short="rush + rec yards"),
    _stat("anytime_td", "anytime TD", "TD", ("anytime touchdown", "attd", "anytime td scorer", "anytime td scorers",
                                             "td scorer", "td scorers", "touchdown scorer", "touchdown scorers",
                                             "scores a td", "score a td", "scores a touchdown", "score a touchdown",
                                             "touchdowns", "touchdown", "tds", "td"),
          "RUSH TD+REC TD", "anytime", markets=("player_anytime_td",), threshold=1, hit_word="a TD", per="TDs"),
)}

HOCKEY = {s["key"]: s for s in (
    _stat("goals", "goals", "G", ("goal", "gls"), "G", markets=("player_goals",), hit_word="a goal"),
    _stat("assists", "assists", "A", ("assist", "apples", "helpers"), "A", markets=("player_assists",)),
    _stat("points", "points", "PTS", ("point", "pts"), "PTS", markets=("player_points",)),
    _stat("sog", "shots on goal", "SOG", ("shots", "shot", "sog", "shots on net", "sogs"), "SOG",
          markets=("player_shots_on_goal",)),
    _stat("saves", "saves", "SV", ("save", "goalie saves", "goaltender saves", "svs"), "SV", needs=("SV",),
          markets=("player_total_saves",)),
    _stat("anytime_goal", "anytime goal", "G", ("anytime goalscorer", "anytime goal scorer", "anytime scorer",
                                                "goalscorer", "goal scorer", "goal scorers", "goalscorers",
                                                "scores a goal", "score a goal", "anytime goals"),
          "G", "anytime", markets=("player_goal_scorer_anytime",), threshold=1, hit_word="a goal", per="goals"),
)}

BASEBALL = {s["key"]: s for s in (
    _stat("hits", "hits", "H", ("base hits",), "H", needs=("AB",), markets=("batter_hits",), hit_word="a hit"),
    _stat("tb", "total bases", "TB", ("total base", "bases"), "H+2B+3B+3B+HR+HR+HR", needs=("AB",),
          markets=("batter_total_bases",)),
    _stat("hr", "home runs", "HR", ("home run", "homers", "homer", "dingers", "dinger", "hrs", "jacks", "taters"),
          "HR", needs=("AB",), markets=("batter_home_runs",), hit_word="a home run"),
    _stat("rbi", "RBIs", "RBI", ("rbis", "runs batted in", "ribbies"), "RBI", needs=("AB",),
          markets=("batter_rbis",)),
    _stat("runs", "runs", "R", ("runs scored",), "R", needs=("AB",), markets=("batter_runs_scored",)),
    _stat("hrr", "hits + runs + RBIs", "H+R+RBI", ("h+r+rbi", "hits runs rbis", "hits+runs+rbis",
                                                   "hits runs and rbis"),
          "H+R+RBI", needs=("AB",), markets=("batter_hits_runs_rbis",), short="H+R+RBI"),
    _stat("k", "strikeouts", "K", ("strikeout", "ks", "k's", "pitcher strikeouts", "punchouts", "whiffs"),
          "K", needs=("IP",), markets=("pitcher_strikeouts",)),
)}

SOCCER = {s["key"]: s for s in (
    _stat("goals", "goals", "G", ("goal", "anytime scorer", "anytime goalscorer", "goalscorer", "goal scorer",
                                  "goalscorers", "anytime goal"),
          "G", markets=(), hit_word="a goal"),
    _stat("shots", "shots", "SHOT", ("shot", "shots taken", "total shots"), "SHOT"),
    _stat("sot", "shots on target", "SOT", ("shots on goal", "sot", "on target", "sog"), "SOG"),
    _stat("assists", "assists", "A", ("assist",), "A"),
)}

_ESPN_LEAGUES = ("soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a", "soccer_germany_bundesliga",
                 "soccer_france_ligue_one", "soccer_usa_mls", "soccer_uefa_champs_league", "soccer_uefa_europa_league",
                 "soccer_efl_champ", "soccer_netherlands_eredivisie", "soccer_portugal_primeira_liga",
                 "soccer_mexico_ligamx", "soccer_spl", "soccer_brazil_campeonato", "soccer_argentina_primera_division")
# league -> {stat key -> stat}; NBA and WNBA share one table, as do the soccer leagues
STATS: Dict[str, Dict[str, Dict[str, Any]]] = {NBA: BASKETBALL, WNBA: BASKETBALL, NFL: FOOTBALL, NHL: HOCKEY,
                                               MLB: BASEBALL}
STATS.update({league: SOCCER for league in _ESPN_LEAGUES})
# The order leagues are tried in when a stat names no league: the NBA first for points, rebounds and assists; the
# NFL for yards and TDs; the NHL for SOG and goals (a soccer league only when one is named)
LEAGUE_ORDER = (NBA, WNBA, NFL, NHL, MLB) + _ESPN_LEAGUES
# The leagues with the NBA model (dashboard/service.py's AI_LEAGUES)
MODEL_LEAGUES = frozenset({NBA})
LEAGUE_NAMES = {
    NBA: ("nba",), WNBA: ("wnba",), NFL: ("nfl",), NHL: ("nhl", "hockey"), MLB: ("mlb", "baseball"),
    "soccer_epl": ("premier league", "epl", "prem"), "soccer_spain_la_liga": ("la liga", "laliga"),
    "soccer_italy_serie_a": ("serie a",), "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_france_ligue_one": ("ligue 1", "ligue one"), "soccer_usa_mls": ("mls",),
    "soccer_uefa_champs_league": ("champions league", "ucl"), "soccer_uefa_europa_league": ("europa league",),
}
TITLES = {NBA: "NBA", WNBA: "WNBA", NFL: "NFL", NHL: "NHL", MLB: "MLB", "soccer_epl": "Premier League"}
OUT_STATUSES = frozenset({"out", "doubtful", "suspension", "suspended", "injured reserve", "inactive"})
STATUS_TAGS = {"questionable": "Q", "day-to-day": "GTD", "day to day": "GTD", "game-time decision": "GTD",
               "game time decision": "GTD", "probable": "P"}

# --- the model's settings ---------------------------------------------------------------------------------

RECENT = 10  # games in the form window
HALF_LIFE = 5.0  # games: the weight halves every 5 games back
SHRINK_K = 5.0  # games of this season's average the form mean is shrunk toward
SPREAD_GAMES = 40  # the newest games the spread is fitted on
# Every "dispersion" in this module is a variance ratio (variance / mean): the NB1 convention of
# engine/calls._negative_binomial_cdf, alpha = ratio - 1 (variance = mean x ratio), never an NB2 size
MAX_DISPERSION = 4.0  # a count's variance is at most this many times its mean (a few odd games can't run away)
YARDS_SD_SHARE = 0.15  # a yards spread is at least 15% of the mean ...
YARDS_SD_MIN = 10.0  # ... and at least 10 yards
CANDIDATE_SHARE = 0.8  # a player projected for 80% of the number is in the conversation
LIKELY = 0.5  # "players" are projected to get there, or at least this likely to
YES_BAR = 0.30  # for yes/no asks (anytime TD, anytime goal, double-double) almost nobody clears 50%: 30% is the bar
CLOSE = 5  # rows in "close"
MAX_ROWS = 25  # the most players a scan returns
MODEL_LABELS = {"nba": "NBA model", "form": "from recent form"}
# How a scan's rows can be ordered (scan's ``sort``): None keeps the ask's own order (chance, or edge for a
# posted-line ask); "hot" and "cold" rank by this season's last 5 games against a baseline (trend.diff)
SORTS = ("chance", "projection", "l10", "edge", "hot", "cold")
# A trend (trend(); hot and cold rank only rows with one): this season's last TREND_GAMES logged games against a
# baseline, this season's other games once it has TREND_BASE, else last season's average when it had TREND_BASE
TREND_GAMES = 5
TREND_BASE = 10
# An edge above +50% is the model's tail (negative binomial or normal) disagreeing with a long price, not an edge: a
# row with one carries ``edge_suspect`` and no edge number, and never ranks by it
EDGE_CAP = 0.5
RUNG_WINDOW = 0.25  # edges mode prices milestone rungs within 25% of the projection (the nearest one either side)
# The stat a scan with no stat named reads for hot and cold (the Lab's and the tool's default); a league not named here
# (the soccer leagues) reads the first of its table (goals)
DEFAULT_STAT = {NBA: "pts", WNBA: "pts", NHL: "points", NFL: "rush_rec_yds", MLB: "tb"}
# Left off a last name ("Jaren Jackson Jr.") and off a typed name ("Payton II", "Marvin Bagley 3rd")
NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "2nd", "3rd"})
NAME_CANDIDATES = 5  # an ambiguous name's candidates listed (ambiguous_names), in alphabetical order
# A typed name ending in a team, as two players with one full name are listed: "Marcus Sutton (MIA)", "... MIA"
_TEAM_QUALIFIER = re.compile(r"^(.*?)[\s,]*\(?\b([A-Za-z]{2,4})\)?\s*$")
WEEKDAYS = {"monday": 0, "tuesday": 1, "tues": 1, "wednesday": 2, "weds": 2, "thursday": 3, "thurs": 3,
            "friday": 4, "saturday": 5, "sunday": 6}
DAY_TZ = ZoneInfo("America/New_York")  # ESPN's day, when a reading has no zone of its own


# --- asks -------------------------------------------------------------------------------------------------

def leagues_for(stat: Any, leagues: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """{league: its stat key} for the leagues whose tables match ``stat`` (a key or a phrase) best, in LEAGUE_ORDER
    (or the order of ``leagues``). Only the strongest matches count: "anytime TD scorers" is the NFL's, not the
    NBA's "scorers"; "20+ points" is the NBA's, the WNBA's and the NHL's; a soccer league comes in only when
    ``leagues`` names it or no other league has the stat."""
    order = list(leagues) if leagues is not None else list(LEAGUE_ORDER)
    scored = [(league, *_match(league, stat)) for league in order]
    best = max((length for _, length, key in scored if key), default=0)
    found = {league: key for league, length, key in scored if key and length == best}
    if leagues is None and any(not league.startswith("soccer_") for league in found):
        # A soccer league only when one is named (or nothing else has the stat): "SOG 3+" is the NHL's
        found = {league: key for league, key in found.items() if not league.startswith("soccer_")}
    return found


def resolve_stat(league: Any, phrase: Any) -> Optional[str]:
    """The stat key ``phrase`` means in ``league``: a key ("pts"), an alias ("boards", "3PM", "ATTD", "Ks") or a
    phrase holding one ("10+ boards tonight"); the longest alias wins. None when the league's table has none."""
    return _match(league, phrase)[1]


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+_']+", " ", text).split())


def _match(league: Any, phrase: Any) -> Tuple[int, Optional[str]]:
    """(how long the matched alias is, the key): an exact alias (a key is one of its own aliases) counts as the
    longest, so "points" is the NBA's as much as the NHL's and "shots" the NHL's as much as soccer's; a key that means
    something else in bettor talk ("td", the NBA's triple-double, isn't one of its aliases) still names its own stat,
    one step below, so "TD" is the NFL's anytime TD."""
    table = STATS.get(str(league or ""))
    text = _norm(phrase)
    if not table or not text:
        return 0, None
    for key, stat in table.items():
        if text in stat["aliases"]:
            return 1000, key
    if text in table:
        return 999, text
    padded = f" {_words(text)} "
    best: Tuple[int, Optional[str]] = (0, None)
    for key, stat in table.items():
        for alias in stat["aliases"]:
            plain = _words(alias)
            if plain and len(plain) > best[0] and f" {plain} " in padded:
                best = (len(plain), key)
    return best


def threshold_for(value: Any) -> Optional[int]:
    """The whole number to reach for a typed number: 20 -> 20, 19.5 -> 20 (over 19.5 is 20 or more; under 19.5 is
    under 20). None for no number; ValueError for one that isn't a positive number under 1000."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("threshold is a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("threshold is a number, e.g. 20") from None
    if not math.isfinite(number) or number <= 0 or number >= 1000:
        raise ValueError("threshold is a number from 1 to 999")
    return max(1, int(math.ceil(number - 1e-9)))


def make_ask(league: str, stat: Any, threshold: Any = None, side: str = "over", day: str = "today") -> Dict[str, Any]:
    """A checked ask: {league, stat, threshold, side, day, line, label, yes}. ``stat`` is a key or an alias;
    ``threshold`` the number to reach (None: each player's own posted line). A double- or triple-double has its own
    number; so has an anytime ask (1+) with no number or 1, while 2 or more is that count ("2+ TDs", "5+ goals": the
    negative binomial, never answered as anytime) and an under the games below it ("under 0.5 TDs"). Raises
    ValueError for anything that doesn't make an ask."""
    if league not in STATS:
        raise ValueError(f"The player scan doesn't cover {league!r}")
    key = resolve_stat(league, stat)
    if key is None:
        raise ValueError(f"No {TITLES.get(league, league)} stat called {stat!r}")
    spec = STATS[league][key]
    side = str(side or "over").lower()
    if side not in ("over", "under"):
        raise ValueError("side is over or under")
    if day not in ("today", "tomorrow"):
        raise ValueError("day is today or tomorrow")
    if spec["threshold"] is None:
        number = threshold_for(threshold)
    elif spec["kind"] != "anytime":
        number, side = spec["threshold"], "over"  # a double- or triple-double: its own number
    else:
        number = threshold_for(threshold)
        if number is None or (number == spec["threshold"] and side == "over"):
            number, side = spec["threshold"], "over"
    return _ask(league, key, number, side, day)


def _ask(league: str, key: str, number: Optional[int], side: str, day: str = "today") -> Dict[str, Any]:
    """make_ask's shape for a stat key and a whole number already checked."""
    spec = STATS[league][key]
    ask = {"league": league, "stat": key, "threshold": number, "side": side, "day": day,
           "line": None if number is None else number - 0.5, "yes": spec["yes"], "unit": spec["unit"]}
    ask["label"] = ask_label(league, key, number, side)
    return ask


def ask_label(league: str, stat: str, threshold: Optional[int], side: str = "over") -> str:
    """How an ask reads: "20+ points", "under 9.5 rebounds", "anytime TD", "double-double", "PRA overs", "2+ TDs"."""
    spec = STATS[league][stat]
    if spec["kind"] == "double" or (spec["kind"] == "anytime" and threshold in (None, 1) and side == "over"):
        return spec["label"]
    name = spec["short"] if spec["short"] != spec["label"] else spec["label"]
    if spec["kind"] == "anytime":
        name = spec["per"]  # "2+ TDs", "under 0.5 goals"
    if threshold is None:
        return f"{name} {'unders' if side == 'under' else 'overs'} (posted lines)"
    if side == "under":
        return f"under {threshold - 0.5:g} {name}"
    return f"{threshold}+ {name}"


def row_kind(spec: Mapping[str, Any]) -> str:
    """A row's ``kind``: "yes_no" (anytime TD or goal, double- and triple-doubles: only the chance means something),
    "yards" or "count"."""
    if spec["yes"]:
        return "yes_no"
    return "yards" if spec["kind"] == "yards" else "count"


def default_stat(league: str) -> str:
    """The stat a scan with none named reads for hot and cold (DEFAULT_STAT: points for the NBA and WNBA, points
    for the NHL, rushing + receiving yards for the NFL, total bases for MLB; else the first of the league's table,
    goals in soccer)."""
    if league not in STATS:
        raise ValueError(f"The player scan doesn't cover {league!r}")
    return DEFAULT_STAT.get(league) or next(iter(STATS[league]))


_NUMBER = r"(\d{1,3}(?:\.\d+)?)"
_ASK_PATTERNS = (
    # (pattern, side, how the number becomes the threshold)
    (re.compile(rf"\b(?:under|below|u)\s*{_NUMBER}"), "under", "line"),  # "u9.5" too
    (re.compile(rf"\b(?:at most|no more than|max(?:imum)?(?: of)?)\s+{_NUMBER}"), "under", "at_most"),
    (re.compile(rf"\b{_NUMBER}\s*(?:or less|or fewer|or under)\b"), "under", "at_most"),
    (re.compile(rf"\b(?:less|fewer)\s+than\s+{_NUMBER}"), "under", "below"),
    (re.compile(rf"\b{_NUMBER}\s*\+"), "over", "reach"),
    (re.compile(rf"\bat\s*least\s+{_NUMBER}"), "over", "reach"),
    (re.compile(rf"\b{_NUMBER}\s*(?:or more|or better|and up|and over|plus)\b"), "over", "reach"),
    (re.compile(rf"\bmore\s+than\s+{_NUMBER}"), "over", "above"),
    (re.compile(rf"\b(?:over|o)\s*{_NUMBER}"), "over", "line"),
    (re.compile(rf"\b(?:score|scoring|scores|get|gets|getting|have|has|record|records|grab|grabs|hit|hits|"
                rf"dish|dishes|drop|drops|put up|puts up|throw|throws|go for|goes for|going for|reach|reaches)"
                rf"\s+{_NUMBER}\b"), "over", "reach"),
)


def _from_number(text: str, how: str) -> int:
    number = float(text)
    if how in ("at_most", "above"):  # "at most 9" is under 10; "more than 20" is 21 or more
        return int(math.floor(number)) + 1
    return max(1, int(math.ceil(number - 1e-9)))  # "20+", "over 19.5", "fewer than 10", "under 9.5"


def weekday_named(text: Any) -> Optional[int]:
    """The weekday a message names (Monday 0 ... Sunday 6: "sunday", "sundays", "thurs"), or None."""
    for word in _words(_norm(text)).split():
        found = WEEKDAYS.get(word[:-1] if word.endswith("days") else word)
        if found is not None:
            return found
    return None


def day_for_weekday(text: Any, now: Any, tz: Any = None) -> Optional[str]:
    """"today" or "tomorrow" for a weekday named in ``text`` ("touchdown sunday") that falls on today or tomorrow in
    ``tz`` (a zone or its name; ESPN's US Eastern day by default) at ``now`` (epoch seconds or a datetime); None when
    no weekday is named or it is further off."""
    wanted = weekday_named(text)
    if wanted is None:
        return None
    zone = ZoneInfo(tz) if isinstance(tz, str) else (tz or DAY_TZ)
    if isinstance(now, datetime.datetime):
        moment = now.astimezone(zone) if now.tzinfo else now.replace(tzinfo=zone)
    else:
        moment = datetime.datetime.fromtimestamp(float(time.time() if now is None else now), zone)
    ahead = (wanted - moment.weekday()) % 7
    return {0: "today", 1: "tomorrow"}.get(ahead)


def parse_ask(text: Any, *, now: Any = None, tz: Any = None) -> Dict[str, Any]:
    """A deterministic reading of a scan ask, for tests and templates (the router does the real reading):
    {stat, league, leagues, stats, threshold, side, day}. "20+", "at least 20", "over 19.5" and "20 or more" give
    20 over; "under 9.5 rebounds" gives 10 under; "1+" and "anytime" give 1; "2+ TDs" gives 2. ``league`` is the
    league named (None when none is); ``stats`` maps each league that knows the stat to its key, in LEAGUE_ORDER (or
    the named league only), and ``stat`` is the first one's key. ``day`` is "today", "tomorrow" (also a weekday
    that is tomorrow at ``now`` in ``tz``: day_for_weekday) or "later" (a weekday further off: no scan covers it)."""
    raw = _norm(text)
    plain = re.sub(r"[?!,;]", " ", raw)
    if re.search(r"\btomorrow\b|\btmrw\b|\btmr\b", plain):
        day = "tomorrow"
    elif weekday_named(plain) is not None:
        day = day_for_weekday(plain, now, tz) or "later"
    else:
        day = "today"
    league = None
    for key, names in LEAGUE_NAMES.items():
        if any(re.search(rf"\b{re.escape(name)}\b", plain) for name in names):
            league = key
            break
    if league is None and re.search(r"\bsoccer\b", plain):
        league = "soccer_epl"
    threshold: Optional[int] = None
    side = "over"
    # Numbers in league and stat names ("ligue 1", "3pm", "3s", "h+r+rbi") aren't thresholds
    scrubbed = re.sub(r"\bligue 1\b|\b3\s*(?:pm|pt|pts|ptm|s|point(?:er)?s?)\b|\bp\+r\+a\b|\b[0-9]+(?:st|nd|rd|th)\b",
                      " ", plain)
    scrubbed = re.sub(r"\b(?:last|l)\s*(?:5|10|15|20)\b|\bl(?:5|10|15|20)\b", " ", scrubbed)
    for pattern, which, how in _ASK_PATTERNS:
        match = pattern.search(scrubbed)
        if match:
            threshold, side = _from_number(match.group(1), how), which
            break
    if re.search(r"\bunders?\b", plain) and threshold is None:
        side = "under"
    stats = leagues_for(plain, [league] if league else None)
    if threshold is None:
        # A bare number in front of the stat: "players with 20 points", "7 strikeouts"
        for match in re.finditer(rf"\b{_NUMBER}\s+((?:[a-z+']+\s*){{1,4}})", scrubbed):
            if leagues_for(match.group(2), list(stats) or None):
                threshold = _from_number(match.group(1), "reach")
                break
    first = next(iter(stats.items()), (None, None))
    stat = first[1]
    spec = STATS[first[0]][stat] if stat else None
    if spec is not None and spec["threshold"] is not None:
        # A double-double, or an anytime ask, has its own number; "2+ TDs" (a count) keeps its own
        counted = spec["kind"] == "anytime" and threshold is not None \
            and (threshold > spec["threshold"] or side == "under")
        if not counted:
            threshold = spec["threshold"]
            side = "over"
    elif spec is not None and re.search(r"\banytime\b", plain) and threshold is None:
        threshold = 1
    return {"stat": stat, "league": league, "leagues": list(stats), "stats": stats, "threshold": threshold,
            "side": side, "day": day}


# --- game-log numbers ---------------------------------------------------------------------------------------

def log_value(row: Mapping[str, Any], spec: Mapping[str, Any]) -> Optional[float]:
    """A game-log row's value for a stat: its column, or the sum of the columns the row has (a receiver's log has
    no "RUSH TD"); None when the row has none of them or lacks a column the stat needs (a pitcher's row for a
    batter's hits). Double- and triple-doubles count the double-digit categories."""
    stats = row.get("stats") or {}
    if any(stats.get(column) is None for column in spec["needs"]):
        return None
    if spec["kind"] == "double":
        values = [stats.get(part) for part in DOUBLE_PARTS]
        if all(value is None for value in values):
            return None
        return float(sum(1 for value in values if value is not None and value >= 10))
    expr = spec["log"]
    if expr in stats:
        value = stats[expr]
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    total, seen = 0.0, False
    for part in str(expr).split("+"):
        value = stats.get(part.strip())
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
            seen = True
    return total if seen else None


def _derived_rows(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The rows with one column, "_v", the stat's value (rows without one keep an empty stats dict)."""
    out = []
    for row in rows:
        value = log_value(row, spec)
        out.append({"season": row.get("season"), "stats": {} if value is None else {"_v": value}})
    return out


def history(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any], threshold: int, side: str = "over"
            ) -> Tuple[Dict[str, List[int]], Dict[str, Optional[float]], List[float]]:
    """(hits, averages, values): hits {l5, l10, season: [hits, games]} from espn_players.hit_rates against
    threshold - 0.5 (an under counts the games below it); averages over the same windows (one decimal, None
    without games); values the stat's figures, newest first. ``rows`` are game-log rows, newest first: this
    season's and, early on, the tail of last season's (the season window counts the newest row's season)."""
    derived = _derived_rows(rows, spec)
    rates = hit_rates(derived, "_v", threshold - 0.5)
    hits: Dict[str, List[int]] = {}
    for name, window in (("l5", "last5"), ("l10", "last10"), ("season", "season")):
        games = int(rates["games"].get(window, 0))
        over = int(rates["hits"].get(window, 0))
        pushes = int(rates["pushes"].get(window, 0))
        hits[name] = [over if side == "over" else games - over - pushes, games]
    played = [(row["season"], row["stats"]["_v"]) for row in derived if "_v" in row["stats"]]
    values = [value for _, value in played]
    season = played[0][0] if played else None
    windows = {"l5": values[:5], "l10": values[:RECENT], "season": [v for s, v in played if s == season]}
    averages = {name: round(sum(sample) / len(sample), 1) if sample else None for name, sample in windows.items()}
    return hits, averages, values


# --- the form model ---------------------------------------------------------------------------------------

def form_mean(values: Sequence[float], season: Sequence[float]) -> Optional[float]:
    """Recency-weighted mean of the last 10 values (newest first; half-life 5 games), shrunk toward this
    season's average by games played: (n x recent + k x season) / (n + k), the season weighing as many games as it
    stands on, at most 5 (k = min(5, season games): a one-game season doesn't outweigh the last ten). None without
    values."""
    recent = list(values[:RECENT])
    if not recent:
        return None
    weights = [0.5 ** (i / HALF_LIFE) for i in range(len(recent))]
    weighted = sum(w * v for w, v in zip(weights, recent)) / sum(weights)
    base = sum(season) / len(season) if season else weighted
    k = min(SHRINK_K, float(len(season))) if season else SHRINK_K
    n = len(recent)
    return (n * weighted + k * base) / (n + k)


def yes_share(values: Sequence[float], season: Sequence[float], threshold: float) -> Optional[float]:
    """The chance of a yes/no event from the log (a double- or triple-double without the NBA model): the
    recency-weighted share of the last 10 games that reached ``threshold``, shrunk toward this season's share with
    k = min(5, season games), the beta-binomial's posterior mean (form_mean on the games' 1s and 0s). None without
    games."""
    return form_mean([1.0 if value >= threshold else 0.0 for value in values],
                     [1.0 if value >= threshold else 0.0 for value in season])


# NB1: the variance ratio count_chance hands to calls._negative_binomial_cdf (not an NB2 size)
def dispersion(values: Sequence[float]) -> float:
    """A count's variance over its mean from the log (the newest SPREAD_GAMES values), never below 1 (a Poisson)
    and at most MAX_DISPERSION; 1 with fewer than 3 games or a zero mean."""
    sample = list(values[:SPREAD_GAMES])
    if len(sample) < 3:
        return 1.0
    mean = sum(sample) / len(sample)
    if mean <= 0:
        return 1.0
    variance = sum((v - mean) ** 2 for v in sample) / (len(sample) - 1)
    return min(max(variance / mean, 1.0), MAX_DISPERSION)


def yards_sd(values: Sequence[float], mean: float) -> float:
    """A yards figure's spread: the log's standard deviation, at least 15% of the mean and 10 yards."""
    sample = list(values[:SPREAD_GAMES])
    sd = 0.0
    if len(sample) >= 2:
        average = sum(sample) / len(sample)
        sd = math.sqrt(sum((v - average) ** 2 for v in sample) / (len(sample) - 1))
    return max(sd, YARDS_SD_SHARE * mean, YARDS_SD_MIN)


def count_chance(mean: float, threshold: int, ratio: float, side: str = "over") -> float:
    """P(X >= threshold) (over) or P(X < threshold) (under) for a count with this mean and variance/mean ratio
    (a negative binomial; a Poisson at 1). An anytime ask is threshold 1: 1 - P(0)."""
    if mean <= 0:
        return 0.0 if side == "over" else 1.0
    below = _negative_binomial_cdf(threshold - 1, mean, max(ratio, 1.0))
    return 1.0 - below if side == "over" else below


def yards_chance(mean: float, threshold: int, sd: float, side: str = "over") -> float:
    """P(X >= threshold) (over) or below it (under) for yards: a normal around ``mean`` read at threshold - 0.5."""
    below = _NORMAL.cdf((threshold - 0.5 - mean) / sd) if sd > 0 else float(mean < threshold - 0.5)
    return 1.0 - below if side == "over" else below


def _played(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    """(the stat's values, newest first; this season's of them: the newest row's season)."""
    played = [(row.get("season"), log_value(row, spec)) for row in rows]
    played = [(season, value) for season, value in played if value is not None]
    if not played:
        return [], []
    newest = played[0][0]
    return [value for _, value in played], [value for s, value in played if s == newest]


def form_projection(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The form model for one player and stat: {mean, dispersion or sd, games} or None without games. A double- or
    triple-double is a yes/no event: ``share`` is its chance (yes_share), and ``mean`` the double-digit categories
    a game (what the candidate filter reads)."""
    values, season = _played(rows, spec)
    if not values:
        return None
    mean = form_mean(values, season)
    if mean is None:
        return None
    out: Dict[str, Any] = {"mean": mean, "games": len(values)}
    if spec["kind"] == "yards":
        out["sd"] = yards_sd(values, mean)
    elif spec["kind"] == "double":
        out["share"] = yes_share(values, season, spec["threshold"])
    else:
        out["dispersion"] = dispersion(values)
    return out


def form_chance(form: Mapping[str, Any], spec: Mapping[str, Any], threshold: int, side: str = "over") -> float:
    if spec["kind"] == "yards":
        return yards_chance(form["mean"], threshold, form["sd"], side)
    if spec["kind"] == "double":
        return form["share"] if side == "over" else 1.0 - form["share"]
    return count_chance(form["mean"], threshold, form["dispersion"], side)


# --- the NBA model ----------------------------------------------------------------------------------------

def nba_mean(means: Mapping[str, Any], spec: Mapping[str, Any]) -> Optional[float]:
    """The NBA projection for a stat: one stat's mean or a combo's sum (PRA and the like)."""
    if not spec["parts"]:
        return None
    total = 0.0
    for part in spec["parts"]:
        value = means.get(part)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        total += float(value)
    return total


def nba_chance(means: Mapping[str, Any], minutes: Optional[float], spec: Mapping[str, Any], threshold: int,
               side: str, values: Sequence[float]) -> Tuple[Optional[float], Optional[float], str]:
    """(projection, chance, chance_source) from the NBA model's means. A stat the model has a measured spread for
    (points, rebounds, assists, threes, steals, blocks, PRA) uses calls.stat_probabilities, exactly as the player
    card does; other combos use the model's mean with the spread fitted from his log ("nba+form"); double- and
    triple-doubles combine the per-stat chances of 10+ (independent), the projection being the expected number of
    double-digit categories."""
    line = threshold - 0.5
    if spec["kind"] == "double":
        chances = []
        for part in DOUBLE_PARTS:
            mean = means.get(part)
            if not isinstance(mean, (int, float)):
                return None, None, "nba"
            chances.append(stat_probabilities(part, float(mean), 9.5, minutes)[0])
        at_least = _at_least(chances, threshold)
        return sum(chances), at_least, "nba"
    mean = nba_mean(means, spec)
    if mean is None:
        return None, None, "nba"
    if spec["nba"] and spec["nba"] in STAT_DISPERSION:
        over, under, _ = stat_probabilities(spec["nba"], mean, line, minutes)
        return mean, over if side == "over" else under, "nba"
    return mean, count_chance(mean, threshold, dispersion(values), side), "nba+form"


def _at_least(chances: Sequence[float], k: int) -> float:
    """P(at least k of independent events with these chances happen)."""
    dist = [1.0] + [0.0] * len(chances)
    for p in chances:
        for j in range(len(chances), 0, -1):
            dist[j] = dist[j] * (1 - p) + dist[j - 1] * p
        dist[0] *= 1 - p
    return sum(dist[k:])


# --- posted lines and edges ---------------------------------------------------------------------------------

def _price(value: Any) -> Optional[Dict[str, Any]]:
    """A decimal price as the app shows one: {decimal (2 places), american}; None for none."""
    if value is None:
        return None
    try:
        number = float(Decimal(str(value)))
    except Exception:
        return None
    if not number > 1:
        return None
    return {"decimal": round(number, 2), "american": format_american(decimal_to_american(number))}


def _raw_price(value: Any) -> Optional[float]:
    try:
        number = float(Decimal(str(value)))
    except Exception:
        return None
    return number if number > 1 else None


def pick_line(props: Sequence[Mapping[str, Any]], spec: Mapping[str, Any], threshold: Optional[int], side: str
              ) -> Optional[Dict[str, Any]]:
    """The posted prop that best answers the ask: the exact rung (a "20+" milestone, an over/under at 19.5, the yes
    bet of an anytime or double-double ask), else the main over/under line, else the nearest milestone.
    {prop, matches} or None when nothing for this stat is posted. An under matches only an over/under line at its
    number (a milestone or a yes bet is an over-only bet: "under 11.5" is never "12+"), else falls back to the
    main line, not matching; a yes stat asked at another number ("2+ TDs") never takes the anytime price."""
    mine = [p for p in props if p.get("market") in spec["markets"]]
    if not mine:
        return None
    if spec["yes"] and side == "over" and threshold in (None, spec["threshold"]):
        yes = [p for p in mine if p.get("kind") == "yes"]
        if yes:
            return {"prop": yes[0], "matches": True}
        line = None if threshold is None else threshold - 0.5
        exact = [p for p in mine if p.get("line") == line]
        return {"prop": (exact or mine)[0], "matches": bool(exact)}
    if threshold is not None:
        line = threshold - 0.5
        exact = [p for p in mine if p.get("line") is not None and float(p["line"]) == line]
        if side == "over":
            priced = [p for p in exact if p.get("over_price") is not None]
        else:
            exact = [p for p in exact if p.get("kind") == "over_under"]
            priced = [p for p in exact if p.get("under_price") is not None]
        if priced or exact:
            ranked = sorted(priced or exact, key=lambda p: p.get("kind") != "milestone")
            return {"prop": ranked[0], "matches": True}
    main = [p for p in mine if p.get("kind") == "over_under" and p.get("line") is not None]
    if main:
        return {"prop": main[0], "matches": False}
    if side == "under":
        return None
    ladder = [p for p in mine if p.get("line") is not None]
    if not ladder:
        return None if spec["yes"] else {"prop": mine[0], "matches": False}
    target = (threshold - 0.5) if threshold is not None else 0.0
    return {"prop": min(ladder, key=lambda p: abs(float(p["line"]) - target)), "matches": False}


def line_view(choice: Optional[Mapping[str, Any]], meta: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The row's ``line``: {kind, line, label, over, under, yes, book, as_of, stale, matches}."""
    if not choice:
        return None
    prop = choice["prop"]
    return {"kind": prop.get("kind"), "line": prop.get("line"), "label": prop.get("label"),
            "market": prop.get("market"), "over": _price(prop.get("over_price")),
            "under": _price(prop.get("under_price")), "yes": _price(prop.get("yes_price")),
            "book": meta.get("book") or "DraftKings", "as_of": meta.get("as_of"), "stale": bool(meta.get("stale")),
            "matches": bool(choice["matches"])}


def edge_for(choice: Optional[Mapping[str, Any]], chance: Optional[float], side: str, yes: bool) -> Optional[float]:
    """chance x price - 1, only when the posted price is for exactly the ask (the matching rung's over or under, or
    the yes bet); None otherwise."""
    if not choice or not choice["matches"] or chance is None:
        return None
    prop = choice["prop"]
    if yes and prop.get("kind") == "yes":
        price = _raw_price(prop.get("yes_price"))
    elif side == "over":
        price = _raw_price(prop.get("over_price"))
    else:
        price = _raw_price(prop.get("under_price"))
    if price is None:
        return None
    return round(chance * price - 1, 4)


# --- the scan -----------------------------------------------------------------------------------------------

def status_tag(status: Any) -> Optional[str]:
    """The chip for an injury status: "Q" (questionable), "GTD" (day-to-day), "P" (probable); None when healthy."""
    return STATUS_TAGS.get(str(status or "").strip().lower())


def ruled_out(status: Any) -> bool:
    return str(status or "").strip().lower() in OUT_STATUSES


def posted_line(props: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The main over/under prop for the stat (what "PRA overs" with no number scans against), or None."""
    for prop in props:
        if prop.get("market") in spec["markets"] and prop.get("kind") == "over_under" and prop.get("line") is not None:
            return {"prop": prop, "matches": True}
    return None


def trend(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any],
          last_season: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """{l5, season, diff, vs}, each number to one decimal (diff from the two as shown): ``l5`` the stat's average over
    this season's last TREND_GAMES logged games (never last season's); ``season`` the baseline, this season's other
    games once it has TREND_BASE (``vs`` "season"), else last season's average when it had TREND_BASE (``vs``
    "last_season"); ``diff`` l5 - season. A double- or triple-double counts the games that reached it (a share).
    None without TREND_GAMES games this season, or without a baseline.

    ``rows`` are the board's game-log rows, newest first (this season's is the newest row's season, as everywhere in
    this module); ``last_season`` last season's whole log (the board reads it while this season is short of
    TREND_BASE games; without it, the rows' own tail of last season is all there is). A newest row from last season
    means no game this season."""
    played = [(row.get("season"), log_value(row, spec)) for row in rows]
    played = [(season, value) for season, value in played if value is not None]
    if not played:
        return None
    current = played[0][0]
    if last_season is not None:
        older = [(row.get("season"), log_value(row, spec)) for row in last_season]
        older = [(season, value) for season, value in older if value is not None]
        if current in {row.get("season") for row in last_season}:
            return None  # his newest game is last season's: none this season yet
    else:
        older = [(season, value) for season, value in played if season != current]
    this = [value for season, value in played if season == current]
    before = [value for _, value in older]
    if spec["kind"] == "double":
        this = [1.0 if value >= spec["threshold"] else 0.0 for value in this]
        before = [1.0 if value >= spec["threshold"] else 0.0 for value in before]
    if len(this) < TREND_GAMES:
        return None
    if len(this) >= TREND_BASE:
        base, vs = this[TREND_GAMES:], "season"
    elif len(before) >= TREND_BASE:
        base, vs = before, "last_season"
    else:
        return None
    l5 = round(sum(this[:TREND_GAMES]) / TREND_GAMES, 1)
    average = round(sum(base) / len(base), 1)
    return {"l5": l5, "season": average, "diff": round(l5 - average, 1), "vs": vs}


def _row(player: Mapping[str, Any], ask: Mapping[str, Any], spec: Mapping[str, Any], game: Mapping[str, Any],
         props_meta: Mapping[str, Any], season_number: bool = False) -> Optional[Dict[str, Any]]:
    """One player's numbers for the ask, or None when there is nothing to project him from (or, with no number
    asked, no line posted for him; with ``season_number``, hot and cold's reading, his season average rounded up is
    the number instead)."""
    rows = player.get("rows") or []
    side = ask["side"]
    props = player.get("props") or []
    threshold = ask["threshold"]
    if threshold is None:
        # "PRA overs": his own posted line is the number (24.5 -> 25+)
        choice = posted_line(props, spec)
        if choice is not None:
            threshold = int(math.floor(float(choice["prop"]["line"]))) + 1
        elif season_number:
            _, season = _played(rows, spec)
            if not season:
                return None
            threshold = max(1, int(math.floor(sum(season) / len(season))) + 1)
        else:
            return None
    else:
        choice = pick_line(props, spec, threshold, side)
    hits, averages, values = history(rows, spec, threshold, side)
    projection = chance = None
    source = model = None
    projected = player.get("projection") or None
    minutes = None
    if projected and projected.get("means"):
        minutes = projected.get("minutes")
        projection, chance, source = nba_chance(projected["means"], minutes, spec, threshold, side, values)
        model = "nba" if chance is not None else None
    if chance is None:
        form = form_projection(rows, spec)
        if form is None:
            return None
        projection, chance, source, model = form["mean"], form_chance(form, spec, threshold, side), "form", "form"
        minutes = None
    status = player.get("status") or None
    kind = row_kind(spec)
    row: Dict[str, Any] = {
        "athlete_id": str(player.get("athlete_id")), "name": player.get("name"), "team": player.get("team"),
        "opponent": player.get("opponent"), "home": player.get("home"), "position": player.get("position"),
        "game": game.get("label"), "event_id": str(player.get("event_id") or game.get("event_id") or ""),
        "start": game.get("start"), "status": status, "status_tag": status_tag(status),
        "threshold": threshold,
        # A yes/no stat's only number is its chance (a double-double's "1.6 projected" means nothing to anyone)
        "projection": None if projection is None or kind == "yes_no" else round(projection, 1),
        "chance": round(min(max(chance, 0.0), 1.0), 4), "chance_source": source, "model": model,
        "model_label": MODEL_LABELS[model], "hits": hits, "avg": averages, "kind": kind,
    }
    if minutes is not None:
        row["minutes"] = round(float(minutes), 1)
    moved = trend(rows, spec, player.get("last_season_rows"))
    if moved is not None:
        row["trend"] = moved
    view = line_view(choice, props_meta)
    if view is not None:
        row["line"] = view
        # A stale line (DraftKings' copy older than the props' time to live) is shown, never priced for an edge
        edge = None if view["stale"] else edge_for(choice, row["chance"], side, spec["yes"])
        if edge is not None and edge > EDGE_CAP:
            row["edge_suspect"] = True  # the model's tail against a long price: no number, no ranking
        elif edge is not None:
            row["edge"] = edge
    row["_projection"] = projection
    row["_values"] = values
    return row


def _qualifies(row: Mapping[str, Any], spec: Mapping[str, Any], side: str) -> bool:
    """In "players": projected at or past the number (an under: at or below the line), or likely enough (50%; 30%
    for a yes/no ask, which almost nobody is 50% to)."""
    threshold = row["threshold"]
    projection = row["_projection"]
    if spec["yes"]:
        return row["chance"] >= YES_BAR
    if row["chance"] >= LIKELY:
        return True
    if projection is None:
        return False
    return projection >= threshold if side == "over" else projection <= threshold - 0.5


def _candidate(row: Mapping[str, Any]) -> bool:
    """In the conversation: reached the number at least once in his last 10, or projected for 80% of it (for an
    under too: a player nowhere near the number isn't an under pick)."""
    threshold = row["threshold"]
    reached = any(value >= threshold for value in row["_values"][:RECENT])
    projection = row["_projection"]
    return reached or (projection is not None and projection >= CANDIDATE_SHARE * threshold)


def why(row: Mapping[str, Any], spec: Mapping[str, Any], side: str, outs: Sequence[str] = ()) -> List[str]:
    """Up to three short reasons from the row's own numbers ("9 of last 10 at 20+", "averaging 26.4 points over
    the last 5", "Irving out", "line 24.5 posted"). Plain facts only: no certainty words."""
    reasons: List[str] = []
    threshold = row["threshold"]
    hits, games = row["hits"]["l10"]
    if games:
        last = f"last {games}" if games != 1 else "last game"
        if spec["hit_word"] and threshold == 1 and side == "under":
            # "under 0.5 goals" counts the games without one: "no goal in 8 of last 10"
            word = re.sub(r"^an? ", "", spec["hit_word"])
            reasons.append(f"no {word} in {hits} of {last}")
        elif spec["hit_word"] and (threshold == 1 or (spec["yes"] and threshold == spec["threshold"])):
            reasons.append(f"{spec['hit_word']} in {hits} of {last}")
        elif side == "under":
            reasons.append(f"under {threshold - 0.5:g} in {hits} of {last}")
        else:
            reasons.append(f"{hits} of {last} at {threshold}+")
    l5, season = row["avg"]["l5"], row["avg"]["season"]
    sample = row["hits"]["l5"][1]
    if l5 is not None and spec["kind"] != "double" and sample:
        if spec["yes"]:
            per = spec["per"][:-1] if l5 == 1 and spec["per"].endswith("s") else spec["per"]  # "1 TD a game"
            reasons.append(f"{l5:g} {per} a game over the last {sample}")
        elif sample > 1:
            reasons.append(f"averaging {l5:g} {spec['label']} over the last {sample}")
        else:
            reasons.append(f"{l5:g} {spec['label']} last game")
    extras: List[str] = []
    if outs and spec["key"] in USAGE_STATS:
        extras.append(f"{', '.join(outs[:2])} out")
    line = row.get("line")
    # The price itself is the row's line (shown in the person's own odds format), never written into a reason; a
    # milestone ("12+") is an over's bet, never an under's reason
    if line and line.get("matches") and line.get("kind") == "milestone" and side == "over":
        extras.append(f"{line.get('label') or f'{threshold}+'} posted")
    elif line and line.get("kind") == "yes" and line.get("yes"):
        extras.append(f"{spec['label']} posted")
    elif line and line.get("line") is not None:
        extras.append(f"line {float(line['line']):g} posted")
    if l5 is not None and season is not None and row["hits"]["season"][1] >= 5 and season > 0 and not spec["yes"]:
        if side == "over" and l5 >= 1.15 * season:
            extras.append(f"up from {season:g} on the season")
        elif side == "under" and l5 <= 0.85 * season:
            extras.append(f"down from {season:g} on the season")
    for extra in extras:
        if len(reasons) >= 3:
            break
        reasons.append(extra)
    if not reasons:  # a projection with no game log behind it (a debut, or ESPN without his log)
        reasons.append("no games logged this season or last")
    return reasons[:3]


def _rank_key(row: Mapping[str, Any], side: str, by_edge: bool) -> Tuple[Any, ...]:
    projection = row["_projection"] if row["_projection"] is not None else 0.0
    edge = row.get("edge")
    head: Tuple[Any, ...] = ()
    if by_edge:
        head = (edge is None, -(edge or 0.0))
    return head + (-row["chance"], -projection if side == "over" else projection, str(row.get("name") or ""))


def _sort_key(row: Mapping[str, Any], side: str, sort: str) -> Tuple[Any, ...]:
    """The order ``sort`` asks for, strongest first (scan's docstring); ties go to the chance, then the name."""
    if sort in ("chance", "edge"):
        return _rank_key(row, side, sort == "edge")
    chance, name = -row["chance"], str(row.get("name") or "")
    if sort == "projection":
        projection = row["_projection"] if row["_projection"] is not None else 0.0
        return (-projection if side == "over" else projection, chance, name)
    if sort == "l10":
        hits, games = row["hits"]["l10"]
        return (-(hits / games) if games else 1.0, -games, chance, name)
    # Hot and cold: by trend.diff; a row without a trend (named players only: hot and cold rank no other) goes last
    moved = row.get("trend")
    diff = float((moved or {}).get("diff") or 0.0)
    return (moved is None, -diff if sort == "hot" else diff, chance, name)


# --- names ------------------------------------------------------------------------------------------------

def _name_words(text: Any) -> List[str]:
    """A name's words, lower case, accents, dots and apostrophes off ("De'Aaron Fox" -> ["deaaron", "fox"])."""
    plain = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    plain = re.sub(r"[.'’`]", "", plain)
    return [word for word in re.split(r"[\s,]+", plain) if word]


def _name_key(words: Sequence[str]) -> str:
    """Words compared without spaces or hyphens: "Gilgeous-Alexander" is "gilgeous alexander" is "gilgeousalexander"."""
    return "".join(re.sub(r"[^a-z0-9]", "", word) for word in words)


def _name_keys(name: Any) -> Tuple[Set[str], str, str]:
    """({full name, with and without a suffix}, first name, last name) of a player, as compared."""
    words = _name_words(name)
    bare = list(words)
    while len(bare) > 2 and bare[-1] in NAME_SUFFIXES:
        bare.pop()
    full = {key for key in (_name_key(words), _name_key(bare)) if key}
    return full, _name_key(bare[:1]), _name_key(bare[1:]) if len(bare) > 1 else ""


def _bare(words: Sequence[str]) -> List[str]:
    """The words with any trailing suffix left off ("payton ii" -> "payton"); one word always stays."""
    out = list(words)
    while len(out) > 1 and out[-1] in NAME_SUFFIXES:
        out.pop()
    return out


def _query_forms(asked: Any) -> List[List[str]]:
    """The word lists a typed name is tried as, in order, each without a trailing suffix ("Payton II", "Porter Jr.",
    "Marvin Bagley 3rd"): "Last, First" read as "First Last" ("Doncic, Luka", "Porter Jr., Michael"; a suffix after
    the comma is just a suffix: "Michael Porter, Jr."), then the words as typed."""
    text = str(asked or "")
    forms: List[List[str]] = []
    if text.count(",") == 1:
        before, after = (_name_words(part) for part in text.split(","))
        first = [word for word in after if word not in NAME_SUFFIXES]
        last = [word for word in before if word not in NAME_SUFFIXES]
        if first and last:
            forms.append(first + last)
    typed = _bare(_name_words(text))
    if typed and typed not in forms:
        forms.append(typed)
    return forms


def _matches(keyed: Sequence[Tuple[str, Set[str], str, str]], words: Sequence[str]) -> List[str]:
    """The ids one form of a typed name matches, by the first rule that matches anyone: the full name, the last name,
    an initial or a first name's start with the last name ("L Doncic", "Steph Curry"), the first name."""
    query = _name_key(words)
    if not query:
        return []
    ids = [pid for pid, full, _, _ in keyed if query in full] \
        or [pid for pid, _, _, last in keyed if last and query == last]
    if not ids and len(words) > 1:
        head, tail = _name_key(words[:1]), _name_key([word for word in words[1:] if word not in NAME_SUFFIXES])
        ids = [pid for pid, _, first, last in keyed if tail and last == tail and head and first.startswith(head)]
    if not ids:
        ids = [pid for pid, _, first, _ in keyed if first == query]
    return list(dict.fromkeys(ids))


def match_names(players: Iterable[Mapping[str, Any]], names: Iterable[Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    """({name asked: [athlete ids]}, [names matching nobody]) against a board's players. Case, accents, dots and a
    suffix don't count; a name matches a player's full name ("Luka Doncic", "Doncic, Luka", "Marvin Bagley 3rd"), his
    last name ("Doncic", "Payton II", "Gilgeous-Alexander"), an initial or a first name's start with the last name
    ("L. Doncic", "Steph Curry"), or his first name ("Shai"). A name several players answer to ("Curry", "Jalen")
    gets every one's id: sort_names calls it ambiguous. A name that matches nobody as typed may end in a team
    ("Marcus Sutton (MIA)", "Marcus Sutton MIA": how an ambiguous name's candidates read): that team's players only.
    Two names in a row that read as one in "Last, First" order ("James" then "LeBron") are asked as one, "LeBron
    James" (the key)."""
    players = list(players)
    keyed = [(str(player.get("athlete_id")), *_name_keys(player.get("name"))) for player in players]
    teams = {str(player.get("athlete_id")): str(player.get("team") or "").upper() for player in players}

    def lookup(asked: Any) -> List[str]:
        ids: List[str] = []
        for words in _query_forms(asked):
            ids = _matches(keyed, words)
            if ids:
                break
        qualified = None if ids else _TEAM_QUALIFIER.match(str(asked or ""))
        if qualified and qualified.group(1).strip():
            for words in _query_forms(qualified.group(1)):
                ids = [pid for pid in _matches(keyed, words) if teams.get(pid) == qualified.group(2).upper()]
                if ids:
                    break
        return ids

    # Two names in a row that read as one in "Last, First" order (the finder splits at commas: "James, LeBron" comes
    # as "James" then "LeBron") are asked as "First Last": the second is one word, "LeBron James" matches a player,
    # and the two alone don't each match a single one
    asked_names: List[Any] = []
    pending = list(names)
    while pending:
        last = pending.pop(0)
        if pending and len(_name_words(pending[0])) == 1 and lookup(f"{pending[0]} {last}") \
                and not (len(lookup(last)) == 1 and len(lookup(pending[0])) == 1):
            last = f"{pending.pop(0)} {last}"
        asked_names.append(last)
    found: Dict[str, List[str]] = {}
    missing: List[str] = []
    for asked in asked_names:
        ids = lookup(asked)
        if ids:
            found[str(asked)] = ids
        elif str(asked) not in missing:
            missing.append(str(asked))
    return found, missing


def sort_names(players: Iterable[Mapping[str, Any]], started: Iterable[Mapping[str, Any]], names: Iterable[Any]
               ) -> Dict[str, Any]:
    """Each asked name in exactly one place (scan's and scan_edges's ``names``), matched (match_names) against the
    board's players and the players of the day's games already under way (``started``: the snapshot's
    started_players, {athlete_id, name, game}): {"found": {name: [athlete id]} (one player, in a game still to come),
    "missing": [names matching nobody, in a game to come or one under way], "ambiguous": [{"name", "candidates"}] (more
    than one player: up to NAME_CANDIDATES full names, in alphabetical order, each with its team when two players
    share one: "Marcus Sutton (MIA)"; one player in two games, a doubleheader, is one player), "started": [{"name",
    "player", "game"}] (one player, whose game has started: ``game`` its label)}. Each list keeps the order the names
    were asked in."""
    later = {str(player.get("athlete_id")): player for player in players}
    begun: Dict[str, Mapping[str, Any]] = {}
    for player in started:
        begun.setdefault(str(player.get("athlete_id")), player)
    for athlete_id in later:  # a doubleheader's player whose second game is still to come is in that game
        begun.pop(athlete_id, None)
    found, missing = match_names(list(later.values()) + list(begun.values()), names)
    out: Dict[str, Any] = {"found": {}, "missing": missing, "ambiguous": [], "started": []}
    for asked, ids in found.items():
        if len(ids) > 1:
            entries = [later.get(pid) or begun.get(pid) or {} for pid in ids]
            keys = [_name_key(_name_words(entry.get("name"))) for entry in entries]
            # Two players with one full name are told apart by team: "Marcus Sutton (MIA)", not the name twice
            full = [f"{entry.get('name')} ({entry.get('team')})" if keys.count(key) > 1 and entry.get("team")
                    else str(entry.get("name")) for entry, key in zip(entries, keys) if entry.get("name")]
            full = sorted(full, key=lambda name: (_name_key(_name_words(name)), name))
            out["ambiguous"].append({"name": asked, "candidates": full[:NAME_CANDIDATES]})
        elif ids[0] in later:
            out["found"][asked] = ids
        else:
            player = begun[ids[0]]
            out["started"].append({"name": asked, "player": str(player.get("name") or asked),
                                   "game": str(player.get("game") or "")})
    return out


# --- ranking a board ------------------------------------------------------------------------------------

def _loading(board: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """{games_ready, games_total} while the board is still building, with ``failed`` (games with a part that
    failed and is still missing) when there are any, built or not; None when neither."""
    ready, total = int(board.get("games_ready") or 0), int(board.get("games_total") or 0)
    failed = int(board.get("games_failed") or 0)
    if ready >= total and not failed:
        return None
    loading = {"games_ready": ready, "games_total": total}
    if failed:
        loading["failed"] = failed
    return loading


def _finish(rows: Sequence[Dict[str, Any]], games: Mapping[str, Any], specs: Any, sides: Any) -> None:
    """Rank, reasons ("why") and no working fields, for the rows a scan returns."""
    for rank, row in enumerate(rows, start=1):
        spec = specs(row)
        game = games.get(row["event_id"]) or {}
        outs = [name for name in (game.get("outs") or {}).get(str(row.get("team")), []) if name != row.get("name")]
        row["rank"] = rank
        row["why"] = why(row, spec, sides(row), outs)
        row.pop("_projection", None)
        row.pop("_values", None)


def scan(board: Mapping[str, Any], ask: Mapping[str, Any], n: int = 10, *, names: Optional[Iterable[Any]] = None,
         sort: Optional[str] = None) -> Dict[str, Any]:
    """Rank one board's players for an ask (make_ask's shape).

    ``board`` is data/player_board.py's snapshot: {league, day, as_of, games: {event_id: {event_id, label, start,
    outs: {team: [names]}, props: {book, as_of, stale}}}, players: [{athlete_id, name, team, opponent, home,
    position, event_id, status, projection: {means, minutes} | None, rows, last_season_rows?, props}],
    started_players: [{athlete_id, name, team, event_id, game}], games_ready, games_total, games_started,
    games_failed}.

    Returns {ask, players, close, more, as_of, games, games_started, loading?, missing_names?, ambiguous_names?,
    started_names?, mode: "scan"}: ``players`` are the candidates projected to reach the number or at least 50% to
    (30% for a yes/no ask), strongest first, at most ``n``; ``close`` up to five more just below; ``more`` how many
    more qualified beyond ``n``; ``loading`` {games_ready, games_total, failed?} while the board is still building or
    a game is missing a part that failed. Ruled-out players (out, doubtful) are dropped; questionable ones carry their
    status and ``status_tag``. With no number (``threshold`` None) each player's posted line is the number and rows
    rank by edge, then chance. Every row has ``kind`` ("yes_no": its projection is None, the chance says it all;
    "count"; "yards") and, when his log allows one (TREND_GAMES games this season and a baseline), ``trend`` {l5,
    season, diff, vs}.

    Every row says whether it qualifies (``qualifies``: projected for the number, or likely enough; in a plain scan
    every row of ``players`` does and no row of ``close``).

    ``names`` (a list): only those players, each whether he qualifies or not, ruled-out ones too (``ruled_out``: True,
    after the others), ranked by chance; ``close`` is empty, and each name is in one place (sort_names): its player's
    rows; ``missing_names`` (matching nobody, or nothing to project him from); ``ambiguous_names`` [{name,
    candidates}] (more than one player); ``started_names`` [{name, player, game}] (his game has started). ``sort``
    (SORTS) reorders: "chance", "projection", "l10" (last-10 hit rate), "edge"; "hot" and "cold" rank every player
    with a ``trend`` by trend.diff, highest or lowest first, without the candidate filter (``close`` empty). With no
    number, hot, cold and named players without a posted line are read against their season average rounded up."""
    league, key, side = ask["league"], ask["stat"], ask["side"]
    spec = STATS[league][key]
    if sort is not None and sort not in SORTS:
        raise ValueError(f"sort is one of {', '.join(SORTS)}")
    n = max(1, min(int(n or 10), MAX_ROWS))
    games = board.get("games") or {}
    players = list(board.get("players") or [])
    named: Optional[Dict[str, List[str]]] = None
    missing: List[str] = []
    sorted_names: Dict[str, Any] = {}
    if names is not None:
        sorted_names = sort_names(players, board.get("started_players") or [],
                                  [name for name in names if str(name or "").strip()])
        named, missing = sorted_names["found"], sorted_names["missing"]
    wanted = {pid for ids in (named or {}).values() for pid in ids}
    trending = named is None and sort in ("hot", "cold")
    by_edge = ask.get("threshold") is None
    rows: List[Dict[str, Any]] = []
    for player in players:
        out = ruled_out(player.get("status"))
        if named is not None:
            if str(player.get("athlete_id")) not in wanted:
                continue
        elif out:
            continue
        game = games.get(str(player.get("event_id"))) or {}
        row = _row(player, ask, spec, game, game.get("props") or {}, season_number=trending or named is not None)
        if row is None:
            continue
        chosen = ((row.get("edge") or 0) > 0 or row["chance"] >= LIKELY) if by_edge else _qualifies(row, spec, side)
        row["qualifies"] = bool(chosen) and not out
        if named is not None:
            if out:
                row["ruled_out"] = True
            rows.append(row)
        elif trending:
            if row.get("trend") is not None:  # only rows with a trend: TREND_GAMES games this season and a baseline
                rows.append(row)
        elif _candidate(row):
            rows.append(row)
    if sort is not None:
        rows.sort(key=lambda row: _sort_key(row, side, sort))
    else:
        rows.sort(key=lambda row: _rank_key(row, side, by_edge and named is None))
    if named is not None:
        rows.sort(key=lambda row: bool(row.get("ruled_out")))  # stable: a ruled-out player after those who play
        shown, close, chosen_count = rows[:MAX_ROWS], [], len(rows)
        returned = {row["athlete_id"] for row in rows}
        for asked, ids in named.items():
            if not returned & set(ids) and asked not in missing:
                missing.append(asked)
    elif trending:
        shown, close, chosen_count = rows[:n], [], len(rows)
    else:
        chosen = [row for row in rows if row["qualifies"]]
        picked = {id(row) for row in chosen}
        shown, chosen_count = chosen[:n], len(chosen)
        close = [row for row in rows if id(row) not in picked][:CLOSE]
    _finish(shown + close, games, lambda row: spec, lambda row: side)
    result: Dict[str, Any] = {
        "ask": dict(ask), "players": shown, "close": close, "more": max(0, chosen_count - len(shown)),
        "as_of": board.get("as_of"), "games": int(board.get("games_total") or 0),
        "games_started": int(board.get("games_started") or 0), "mode": "scan",
    }
    loading = _loading(board)
    if loading is not None:
        result["loading"] = loading
    if named is not None:
        result.update(missing_names=missing, ambiguous_names=sorted_names["ambiguous"],
                      started_names=sorted_names["started"])
    return result


def _ladder(props: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """A market's priced milestone rungs ("15+", "20+", ...) whose prices keep the ladder's order: a rung priced
    longer than one above it (a 2+ at 2.16 under a 5+ at 2.01) can't be right (a capture or feed error), so it is
    left out rather than shown as a huge edge."""
    rungs = sorted(((float(prop["line"]), _raw_price(prop.get("over_price")), prop) for prop in props
                    if prop.get("kind") == "milestone" and prop.get("line") is not None),
                   key=lambda rung: rung[0])
    rungs = [rung for rung in rungs if rung[1] is not None]
    kept = []
    for index, (_, price, prop) in enumerate(rungs):
        if all(price <= higher for _, higher, _ in rungs[index + 1:]):
            kept.append(prop)
    return kept


def _projection_of(player: Mapping[str, Any], spec: Mapping[str, Any]) -> Optional[float]:
    """A player's projection for a stat, whatever the number: the NBA model's mean when it has one, else the form
    model's (as _row works it out)."""
    projected = player.get("projection") or None
    if projected and projected.get("means"):
        mean = nba_mean(projected["means"], spec)
        if mean is not None:
            return mean
    form = form_projection(player.get("rows") or [], spec)
    return None if form is None else form["mean"]


def _near_rungs(rungs: Sequence[Mapping[str, Any]], projection: Optional[float]) -> List[int]:
    """The numbers of the milestone rungs edges mode prices for a player: within RUNG_WINDOW of his projection, at
    most the nearest one at or below it and the nearest one above (a far rung's "edge" is the model's tail)."""
    if projection is None or projection <= 0:
        return []
    numbers = sorted({int(math.floor(float(prop["line"]))) + 1 for prop in rungs})
    near = [number for number in numbers if abs(number - projection) <= RUNG_WINDOW * projection]
    below = [number for number in near if number <= projection]
    above = [number for number in near if number > projection]
    return ([below[-1]] if below else []) + ([above[0]] if above else [])


def scan_edges(board: Mapping[str, Any], league: str, n: int = 10, *, side: Optional[str] = None,
               names: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """The posted lines the numbers like best, every stat of the league at once ("best player props tonight", "any
    good overs"): for each player and stat with lines posted that aren't stale (the main over/under line, either
    side; the yes bet of an anytime or double-double stat; the priced milestone rungs near his projection, "20+", an
    over: within RUNG_WINDOW of it, the nearest one either side), the line and side with the biggest positive edge
    (``side`` "over" or "under": only that side). ESPN's DraftKings over/under items mostly carry no prices, so the
    near milestones and yes bets are most of what can be priced. A far rung is never priced (its "edge" is the model's
    tail against a long price), nor is a rung priced out of its ladder's order (_ladder), and an edge above EDGE_CAP
    is suspect (the row's ``edge_suspect``, no number): it never ranks.

    Returns {mode: "edges", players, more, as_of, games, games_started, loading?, missing_names?, ambiguous_names?,
    started_names?}: one row per player and stat at most, with a positive edge, highest first, at most ``n``; each
    the scan's row for that line's number plus ``stat``, ``stat_label`` and ``side``. There is no candidate filter: a
    long rung the numbers like is there. ``names``: only those players' lines, each name in one place as in scan
    (sort_names: his lines, ``missing_names``, ``ambiguous_names`` or ``started_names``)."""
    if league not in STATS:
        raise ValueError(f"The player scan doesn't cover {league!r}")
    if side is not None:
        side = str(side).lower()
        if side not in ("over", "under"):
            raise ValueError("side is over or under")
    n = max(1, min(int(n or 10), MAX_ROWS))
    sides = ("over", "under") if side is None else (side,)
    games = board.get("games") or {}
    players = list(board.get("players") or [])
    sorted_names: Optional[Dict[str, Any]] = None
    if names is not None:
        sorted_names = sort_names(players, board.get("started_players") or [],
                                  [name for name in names if str(name or "").strip()])
        wanted = {pid for ids in sorted_names["found"].values() for pid in ids}
        players = [player for player in players if str(player.get("athlete_id")) in wanted]
    asks: Dict[Tuple[str, Optional[int], str], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for player in players:
        props = player.get("props") or []
        if ruled_out(player.get("status")) or not props:
            continue
        game = games.get(str(player.get("event_id"))) or {}
        meta = game.get("props") or {}
        if meta.get("stale"):
            continue
        for key, spec in STATS[league].items():
            mine = [prop for prop in props if prop.get("market") in spec["markets"]]
            if not mine:
                continue
            tries: List[Tuple[Optional[int], str]] = []
            main = posted_line(mine, spec)
            if main is not None:
                tries += [(int(math.floor(float(main["prop"]["line"]))) + 1, which) for which in sides]
            if "over" in sides:
                if spec["yes"] and any(prop.get("kind") == "yes" for prop in mine):
                    tries.append((spec["threshold"], "over"))
                if not spec["yes"]:
                    tries += [(number, "over") for number in _near_rungs(_ladder(mine), _projection_of(player, spec))]
            best: Optional[Dict[str, Any]] = None
            for number, which in dict.fromkeys(tries):
                ask = asks.get((key, number, which))
                if ask is None:
                    ask = asks[(key, number, which)] = _ask(league, key, number, which)
                row = _row(player, ask, spec, game, meta)
                if row is not None and (row.get("edge") or 0) > 0 and (best is None or row["edge"] > best["edge"]):
                    row.update(stat=key, stat_label=spec["label"], side=which)
                    best = row
            if best is not None:
                rows.append(best)
    rows.sort(key=lambda row: (-row["edge"], -row["chance"], str(row.get("name") or ""), row["stat"]))
    shown = rows[:n]
    table = STATS[league]
    _finish(shown, games, lambda row: table[row["stat"]], lambda row: row["side"])
    result: Dict[str, Any] = {
        "mode": "edges", "players": shown, "more": max(0, len(rows) - len(shown)), "as_of": board.get("as_of"),
        "games": int(board.get("games_total") or 0), "games_started": int(board.get("games_started") or 0),
    }
    loading = _loading(board)
    if loading is not None:
        result["loading"] = loading
    if sorted_names is not None:
        result.update(missing_names=sorted_names["missing"], ambiguous_names=sorted_names["ambiguous"],
                      started_names=sorted_names["started"])
    return result


def stats_menu(league: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """The scan's stats for a picker: {league: [{key, label, short, unit, yes, threshold}]} (one league, or every
    league the table covers)."""
    leagues = [league] if league else list(STATS)
    return {key: [{"key": stat["key"], "label": stat["label"], "short": stat["short"], "unit": stat["unit"],
                   "yes": stat["yes"], "threshold": stat["threshold"]} for stat in STATS[key].values()]
            for key in leagues if key in STATS}
