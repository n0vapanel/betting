"""The player scan: who reaches a number tonight ("20+ points", "10+ rebounds", "anytime TD", "7+ strikeouts").

For one league and one day's games (the board, data/player_board.py), every player gets four numbers:

- **History:** how often he reached the number in his last 5 and last 10 games and this season
  (``espn_players.hit_rates`` against the number minus a half), with his averages over the same windows.
- **Projection:** tonight's expected figure. NBA players use the NBA model's own projection for the game
  (``service.detail``: minutes, usage, pace and injuries); everyone else uses the form model below.
- **Chance:** the chance of reaching the number. For the NBA it is ``calls.stat_probabilities`` on the model's
  mean, exactly as the player card works it out, so the two never disagree on the same line.
- **Posted line:** DraftKings' line and price for the stat (through ESPN, never The Odds API), with the edge
  when the price is for the number asked (a "20+" milestone, or an over at 19.5).

Players are ranked by that chance (then the projection), strongest first.

The form model (every league without the NBA model, and NBA players the model has no projection for): a
recency-weighted mean of the last 10 games (exponential weights, half-life 5 games), shrunk toward this
season's per-game average by games played (k = 5). Counts (points, rebounds, shots, strikeouts) use a negative
binomial whose dispersion is fitted from the log's own variance, never below a Poisson; yards use a normal with
the log's spread, at least 15% of the mean and 10 yards; "anytime" asks are 1 - P(0) of the count. Every row
says which model it came from (``model``: "nba" or "form"); the form model is labelled "from recent form".

Thresholds: an ask is a whole number to reach ("20+" is 20). The line behind it is that number minus a half
(19.5), so a game counts as a hit at 20 or more; an "under" ask counts the games below the number. "Over 19.5"
and "at least 20" are both 20; "under 9.5" is under 10.

Everything here is pure: numbers in, numbers out. Nothing fetches.
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..data.espn_players import hit_rates
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
MAX_DISPERSION = 4.0  # a count's variance is at most this many times its mean (a few odd games can't run away)
YARDS_SD_SHARE = 0.15  # a yards spread is at least 15% of the mean ...
YARDS_SD_MIN = 10.0  # ... and at least 10 yards
CANDIDATE_SHARE = 0.8  # a player projected for 80% of the number is in the conversation
LIKELY = 0.5  # "players" are projected to get there, or at least this likely to
YES_BAR = 0.30  # for yes/no asks (anytime TD, anytime goal, double-double) almost nobody clears 50%: 30% is the bar
CLOSE = 5  # rows in "close"
MAX_ROWS = 25  # the most players a scan returns
MODEL_LABELS = {"nba": "NBA model", "form": "from recent form"}


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
    ``threshold`` the number to reach (None: each player's own posted line); a yes/no stat (anytime TD, double-double)
    has its own. Raises ValueError for anything that doesn't make an ask."""
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
    number = spec["threshold"] if spec["threshold"] is not None else threshold_for(threshold)
    if spec["yes"]:
        side = "over"
    ask = {"league": league, "stat": key, "threshold": number, "side": side, "day": day,
           "line": None if number is None else number - 0.5, "yes": spec["yes"], "unit": spec["unit"]}
    ask["label"] = ask_label(league, key, number, side)
    return ask


def ask_label(league: str, stat: str, threshold: Optional[int], side: str = "over") -> str:
    """How an ask reads: "20+ points", "under 9.5 rebounds", "anytime TD", "double-double", "PRA overs"."""
    spec = STATS[league][stat]
    if spec["kind"] == "double" or (spec["kind"] == "anytime" and threshold in (None, 1)):
        return spec["label"]
    name = spec["short"] if spec["short"] != spec["label"] else spec["label"]
    if threshold is None:
        return f"{name} {'unders' if side == 'under' else 'overs'} (posted lines)"
    if side == "under":
        return f"under {threshold - 0.5:g} {name}"
    return f"{threshold}+ {name}"


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


def parse_ask(text: Any) -> Dict[str, Any]:
    """A deterministic reading of a scan ask, for tests and templates (the router does the real reading):
    {stat, league, leagues, stats, threshold, side, day}. "20+", "at least 20", "over 19.5" and "20 or more" give
    20 over; "under 9.5 rebounds" gives 10 under; "1+" and "anytime" give 1. ``league`` is the league named (None
    when none is); ``stats`` maps each league that knows the stat to its key, in LEAGUE_ORDER (or the named league
    only), and ``stat`` is the first one's key."""
    raw = _norm(text)
    plain = re.sub(r"[?!,;]", " ", raw)
    day = "tomorrow" if re.search(r"\btomorrow\b|\btmrw\b|\btmr\b", plain) else "today"
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
    season's average by games played: (n x recent + 5 x season) / (n + 5). None without values."""
    recent = list(values[:RECENT])
    if not recent:
        return None
    weights = [0.5 ** (i / HALF_LIFE) for i in range(len(recent))]
    weighted = sum(w * v for w, v in zip(weights, recent)) / sum(weights)
    base = sum(season) / len(season) if season else weighted
    n = len(recent)
    return (n * weighted + SHRINK_K * base) / (n + SHRINK_K)


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


def form_projection(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The form model for one player and stat: {mean, dispersion or sd, games} or None without games."""
    played = [(row.get("season"), log_value(row, spec)) for row in rows]
    played = [(season, value) for season, value in played if value is not None]
    if not played:
        return None
    values = [value for _, value in played]
    newest = played[0][0]
    season = [value for s, value in played if s == newest]
    mean = form_mean(values, season)
    if mean is None:
        return None
    out: Dict[str, Any] = {"mean": mean, "games": len(values)}
    if spec["kind"] == "yards":
        out["sd"] = yards_sd(values, mean)
    else:
        out["dispersion"] = dispersion(values)
    return out


def form_chance(form: Mapping[str, Any], spec: Mapping[str, Any], threshold: int, side: str = "over") -> float:
    if spec["kind"] == "yards":
        return yards_chance(form["mean"], threshold, form["sd"], side)
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
    {prop, matches} or None when nothing for this stat is posted."""
    mine = [p for p in props if p.get("market") in spec["markets"]]
    if not mine:
        return None
    if spec["yes"]:
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
            priced = [p for p in exact if p.get("under_price") is not None]
        if priced or exact:
            ranked = sorted(priced or exact, key=lambda p: p.get("kind") != "milestone")
            return {"prop": ranked[0], "matches": True}
    main = [p for p in mine if p.get("kind") == "over_under" and p.get("line") is not None]
    if main:
        return {"prop": main[0], "matches": False}
    ladder = [p for p in mine if p.get("line") is not None]
    if not ladder:
        return {"prop": mine[0], "matches": False}
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


def _row(player: Mapping[str, Any], ask: Mapping[str, Any], spec: Mapping[str, Any], game: Mapping[str, Any],
         props_meta: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """One player's numbers for the ask, or None when there is nothing to project him from (or, with no number
    asked, no line posted for him)."""
    rows = player.get("rows") or []
    side = ask["side"]
    props = player.get("props") or []
    threshold = ask["threshold"]
    if threshold is None:
        # "PRA overs": his own posted line is the number (24.5 -> 25+)
        choice = posted_line(props, spec)
        if choice is None:
            return None
        threshold = int(math.floor(float(choice["prop"]["line"]))) + 1
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
    row: Dict[str, Any] = {
        "athlete_id": str(player.get("athlete_id")), "name": player.get("name"), "team": player.get("team"),
        "opponent": player.get("opponent"), "home": player.get("home"), "position": player.get("position"),
        "game": game.get("label"), "event_id": str(player.get("event_id") or game.get("event_id") or ""),
        "start": game.get("start"), "status": status, "status_tag": status_tag(status),
        "threshold": threshold, "projection": None if projection is None else round(projection, 1),
        "chance": round(min(max(chance, 0.0), 1.0), 4), "chance_source": source, "model": model,
        "model_label": MODEL_LABELS[model], "hits": hits, "avg": averages,
    }
    if minutes is not None:
        row["minutes"] = round(float(minutes), 1)
    view = line_view(choice, props_meta)
    if view is not None:
        row["line"] = view
        edge = edge_for(choice, row["chance"], side, spec["yes"])
        if edge is not None:
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
        if spec["hit_word"] and (spec["yes"] or threshold == 1):
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
    # The price itself is the row's line (shown in the person's own odds format), never written into a reason
    if line and line.get("matches") and line.get("kind") == "milestone":
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


def scan(board: Mapping[str, Any], ask: Mapping[str, Any], n: int = 10) -> Dict[str, Any]:
    """Rank one board's players for an ask (make_ask's shape).

    ``board`` is data/player_board.py's snapshot: {league, day, as_of, games: {event_id: {event_id, label, start,
    outs: {team: [names]}, props: {book, as_of, stale}}}, players: [{athlete_id, name, team, opponent, home,
    position, event_id, status, projection: {means, minutes} | None, rows, props}], games_ready, games_total,
    games_started}.

    Returns {ask, players, close, more, as_of, games, games_started, loading?}: ``players`` are the candidates
    projected to reach the number or at least 50% to (30% for a yes/no ask), strongest first, at most ``n``;
    ``close`` up to five more just below; ``more`` how many more qualified beyond ``n``; ``loading`` {games_ready,
    games_total} while the board is still building. Ruled-out players (out, doubtful) are dropped; questionable
    ones carry their status and ``status_tag``. With no number (``threshold`` None) each player's posted line is
    the number and rows rank by edge, then chance."""
    league, key, side = ask["league"], ask["stat"], ask["side"]
    spec = STATS[league][key]
    n = max(1, min(int(n or 10), MAX_ROWS))
    games = board.get("games") or {}
    rows: List[Dict[str, Any]] = []
    for player in board.get("players") or []:
        if ruled_out(player.get("status")):
            continue
        game = games.get(str(player.get("event_id"))) or {}
        row = _row(player, ask, spec, game, game.get("props") or {})
        if row is not None and _candidate(row):
            rows.append(row)
    by_edge = ask.get("threshold") is None
    rows.sort(key=lambda row: _rank_key(row, side, by_edge))
    if by_edge:
        chosen = [row for row in rows if (row.get("edge") or 0) > 0 or row["chance"] >= LIKELY]
    else:
        chosen = [row for row in rows if _qualifies(row, spec, side)]
    picked = {id(row) for row in chosen}
    shown = chosen[:n]
    close = [row for row in rows if id(row) not in picked][:CLOSE]
    for rank, row in enumerate(shown + close, start=1):
        game = games.get(row["event_id"]) or {}
        outs = [name for name in (game.get("outs") or {}).get(str(row.get("team")), []) if name != row.get("name")]
        row["rank"] = rank
        row["why"] = why(row, spec, side, outs)
        row.pop("_projection", None)
        row.pop("_values", None)
    result: Dict[str, Any] = {
        "ask": dict(ask), "players": shown, "close": close, "more": max(0, len(chosen) - len(shown)),
        "as_of": board.get("as_of"), "games": int(board.get("games_total") or 0),
        "games_started": int(board.get("games_started") or 0),
    }
    ready, total = int(board.get("games_ready") or 0), int(board.get("games_total") or 0)
    if ready < total:
        result["loading"] = {"games_ready": ready, "games_total": total}
    return result


def stats_menu(league: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """The scan's stats for a picker: {league: [{key, label, short, unit, yes, threshold}]} (one league, or every
    league the table covers)."""
    leagues = [league] if league else list(STATS)
    return {key: [{"key": stat["key"], "label": stat["label"], "short": stat["short"], "unit": stat["unit"],
                   "yes": stat["yes"], "threshold": stat["threshold"]} for stat in STATS[key].values()]
            for key in leagues if key in STATS}
