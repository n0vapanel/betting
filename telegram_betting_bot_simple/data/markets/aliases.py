"""One key per team or player, whatever the spelling, and matching venue events to the app's games.

team_key(league, name_or_abbr) -> str | None
    The NBA, NFL, NHL and MLB have full tables: every abbreviation the venues use (ESPN "WSH", Polymarket
    "was", Kalshi "WAS"; Polymarket's NHL "cal"/"las"/"mon"/"lak" against ESPN's CGY/VGK/MTL/LA; Kalshi's NFL
    "JAC" against ESPN's JAX), the city as Kalshi writes it ("Los Angeles L", "New York R"), nicknames and full
    names map to ESPN's abbreviation. A name a table doesn't know gives None.
    Other team leagues (college, soccer) have too many teams for a table: the key is the normalised name
    (accents, punctuation and club suffixes like FC/CF/SC/SK dropped, "St." at the end read as "State") with a
    short list of common variants folded together ("Man Utd", "Spurs", "Inter", "PSG", "New York RB"...).
    Tennis players and fighters are keyed by their normalised full name.

match_events(league, games, market_events, tol_hours=None) -> {game_id: MarketEvent}
    Pairs each venue event with at most one game and each game with at most one event. Both teams must match
    (same key, or for names outside the tables one name contained in the other, e.g. "Kennesaw St." in
    "Kennesaw State Owls", or a player's surname with the same first initial), and the start times must be
    within ``tol_hours`` when both are known (else the dates within a day). ``tol_hours`` defaults to 3 for teams
    and 12 for tennis and fights: a player has one singles match a day and a fighter one fight a card, while the
    venues' times for them are rough (Polymarket gives every fight the card's start, ESPN the main card's; Kalshi
    and Polymarket a tennis day's first time). The best pairs win first: stronger
    name matches, then the nearest start, so the two games of a double-header each get their own event.
    An event whose home and away are the other way round from the game's is returned turned round (home and
    away quotes swapped, the spread's point negated) with "swapped": True, so its "home" is always the game's.
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from zoneinfo import ZoneInfo

from .common import parse_time

EASTERN = ZoneInfo("America/New_York")  # ESPN's and Kalshi's game days

# --- tables -------------------------------------------------------------------------------------------
# canonical (ESPN's abbreviation) -> every other spelling seen on ESPN, Polymarket, Kalshi and The Odds API
# (abbreviations, cities as each venue writes them, nicknames, full names). Live-checked 2026-10-27.

_NBA = {
    "ATL": ["Atlanta", "Hawks", "Atlanta Hawks"],
    "BOS": ["Boston", "Celtics", "Boston Celtics"],
    "BKN": ["BRK", "NJN", "Brooklyn", "Nets", "Brooklyn Nets"],
    "CHA": ["CHO", "Charlotte", "Hornets", "Charlotte Hornets"],
    "CHI": ["Chicago", "Bulls", "Chicago Bulls"],
    "CLE": ["Cleveland", "Cavaliers", "Cavs", "Cleveland Cavaliers"],
    "DAL": ["Dallas", "Mavericks", "Mavs", "Dallas Mavericks"],
    "DEN": ["Denver", "Nuggets", "Denver Nuggets"],
    "DET": ["Detroit", "Pistons", "Detroit Pistons"],
    "GS": ["GSW", "Golden State", "Warriors", "Golden State Warriors"],
    "HOU": ["Houston", "Rockets", "Houston Rockets"],
    "IND": ["Indiana", "Pacers", "Indiana Pacers"],
    "LAC": ["Los Angeles C", "LA Clippers", "Los Angeles Clippers", "Clippers"],
    "LAL": ["Los Angeles L", "LA Lakers", "Los Angeles Lakers", "Lakers"],
    "MEM": ["Memphis", "Grizzlies", "Memphis Grizzlies"],
    "MIA": ["Miami", "Heat", "Miami Heat"],
    "MIL": ["Milwaukee", "Bucks", "Milwaukee Bucks"],
    "MIN": ["Minnesota", "Timberwolves", "Minnesota Timberwolves"],
    "NO": ["NOP", "NOR", "New Orleans", "Pelicans", "New Orleans Pelicans"],
    "NY": ["NYK", "New York", "Knicks", "New York Knicks"],
    "OKC": ["Oklahoma City", "Thunder", "Oklahoma City Thunder"],
    "ORL": ["Orlando", "Magic", "Orlando Magic"],
    "PHI": ["PHL", "Philadelphia", "76ers", "Sixers", "Philadelphia 76ers"],
    "PHX": ["PHO", "Phoenix", "Suns", "Phoenix Suns"],
    "POR": ["Portland", "Trail Blazers", "Blazers", "Portland Trail Blazers"],
    "SAC": ["Sacramento", "Kings", "Sacramento Kings"],
    "SA": ["SAS", "San Antonio", "Spurs", "San Antonio Spurs"],
    "TOR": ["Toronto", "Raptors", "Toronto Raptors"],
    "UTAH": ["UTA", "Utah", "Jazz", "Utah Jazz"],
    "WSH": ["WAS", "Washington", "Wizards", "Washington Wizards"],
}

_NFL = {
    "ARI": ["Arizona", "Cardinals", "Arizona Cardinals"],
    "ATL": ["Atlanta", "Falcons", "Atlanta Falcons"],
    "BAL": ["Baltimore", "Ravens", "Baltimore Ravens"],
    "BUF": ["Buffalo", "Bills", "Buffalo Bills"],
    "CAR": ["Carolina", "Panthers", "Carolina Panthers"],
    "CHI": ["Chicago", "Bears", "Chicago Bears"],
    "CIN": ["Cincinnati", "Bengals", "Cincinnati Bengals"],
    "CLE": ["Cleveland", "Browns", "Cleveland Browns"],
    "DAL": ["Dallas", "Cowboys", "Dallas Cowboys"],
    "DEN": ["Denver", "Broncos", "Denver Broncos"],
    "DET": ["Detroit", "Lions", "Detroit Lions"],
    "GB": ["GNB", "Green Bay", "Packers", "Green Bay Packers"],
    "HOU": ["Houston", "Texans", "Houston Texans"],
    "IND": ["Indianapolis", "Colts", "Indianapolis Colts"],
    "JAX": ["JAC", "Jacksonville", "Jaguars", "Jacksonville Jaguars"],
    "KC": ["KAN", "Kansas City", "Chiefs", "Kansas City Chiefs"],
    "LV": ["LVR", "Las Vegas", "Raiders", "Las Vegas Raiders"],
    "LAC": ["Los Angeles C", "LA Chargers", "Los Angeles Chargers", "Chargers"],
    "LAR": ["LA", "Los Angeles R", "LA Rams", "Los Angeles Rams", "Rams"],
    "MIA": ["Miami", "Dolphins", "Miami Dolphins"],
    "MIN": ["Minnesota", "Vikings", "Minnesota Vikings"],
    "NE": ["NWE", "New England", "Patriots", "New England Patriots"],
    "NO": ["NOR", "New Orleans", "Saints", "New Orleans Saints"],
    "NYG": ["New York G", "Giants", "New York Giants"],
    "NYJ": ["New York J", "Jets", "New York Jets"],
    "PHI": ["Philadelphia", "Eagles", "Philadelphia Eagles"],
    "PIT": ["Pittsburgh", "Steelers", "Pittsburgh Steelers"],
    "SF": ["SFO", "San Francisco", "49ers", "Niners", "San Francisco 49ers"],
    "SEA": ["Seattle", "Seahawks", "Seattle Seahawks"],
    "TB": ["TAM", "Tampa Bay", "Buccaneers", "Bucs", "Tampa Bay Buccaneers"],
    "TEN": ["Tennessee", "Titans", "Tennessee Titans"],
    "WSH": ["WAS", "Washington", "Commanders", "Washington Commanders"],
}

_NHL = {
    "ANA": ["Anaheim", "Ducks", "Anaheim Ducks"],
    "BOS": ["Boston", "Bruins", "Boston Bruins"],
    "BUF": ["Buffalo", "Sabres", "Buffalo Sabres"],
    "CGY": ["CAL", "Calgary", "Flames", "Calgary Flames"],
    "CAR": ["Carolina", "Hurricanes", "Carolina Hurricanes"],
    "CHI": ["Chicago", "Blackhawks", "Chicago Blackhawks"],
    "COL": ["Colorado", "Avalanche", "Colorado Avalanche"],
    "CBJ": ["CLB", "Columbus", "Blue Jackets", "Columbus Blue Jackets"],
    "DAL": ["Dallas", "Stars", "Dallas Stars"],
    "DET": ["Detroit", "Red Wings", "Detroit Red Wings"],
    "EDM": ["Edmonton", "Oilers", "Edmonton Oilers"],
    "FLA": ["Florida", "Panthers", "Florida Panthers"],
    "LA": ["LAK", "Los Angeles", "Kings", "LA Kings", "Los Angeles Kings"],
    "MIN": ["Minnesota", "Wild", "Minnesota Wild"],
    "MTL": ["MON", "Montreal", "Canadiens", "Montreal Canadiens"],
    "NSH": ["Nashville", "Predators", "Nashville Predators"],
    "NJ": ["NJD", "New Jersey", "Devils", "New Jersey Devils"],
    "NYI": ["New York I", "Islanders", "New York Islanders"],
    "NYR": ["New York R", "Rangers", "New York Rangers"],
    "OTT": ["Ottawa", "Senators", "Ottawa Senators"],
    "PHI": ["Philadelphia", "Flyers", "Philadelphia Flyers"],
    "PIT": ["Pittsburgh", "Penguins", "Pittsburgh Penguins"],
    "SJ": ["SJS", "San Jose", "Sharks", "San Jose Sharks"],
    "SEA": ["Seattle", "Kraken", "Seattle Kraken"],
    "STL": ["St. Louis", "Blues", "St. Louis Blues"],
    "TB": ["TBL", "Tampa Bay", "Lightning", "Tampa Bay Lightning"],
    "TOR": ["Toronto", "Maple Leafs", "Toronto Maple Leafs"],
    "UTAH": ["UTA", "Utah", "Mammoth", "Utah Mammoth", "Utah Hockey Club"],
    "VAN": ["Vancouver", "Canucks", "Vancouver Canucks"],
    "VGK": ["LAS", "VEG", "Vegas", "Golden Knights", "Vegas Golden Knights"],
    "WSH": ["WAS", "Washington", "Capitals", "Washington Capitals"],
    "WPG": ["Winnipeg", "Jets", "Winnipeg Jets"],
}

_MLB = {
    "ARI": ["AZ", "Arizona", "Diamondbacks", "D-backs", "Arizona Diamondbacks"],
    "ATH": ["OAK", "Athletics", "Oakland", "Oakland Athletics", "A's"],
    "ATL": ["Atlanta", "Braves", "Atlanta Braves"],
    "BAL": ["Baltimore", "Orioles", "Baltimore Orioles"],
    "BOS": ["Boston", "Red Sox", "Boston Red Sox"],
    "CHC": ["Chicago C", "Cubs", "Chicago Cubs"],
    "CHW": ["CWS", "Chicago W", "White Sox", "Chicago White Sox"],
    "CIN": ["Cincinnati", "Reds", "Cincinnati Reds"],
    "CLE": ["Cleveland", "Guardians", "Cleveland Guardians"],
    "COL": ["Colorado", "Rockies", "Colorado Rockies"],
    "DET": ["Detroit", "Tigers", "Detroit Tigers"],
    "HOU": ["Houston", "Astros", "Houston Astros"],
    "KC": ["KCR", "Kansas City", "Royals", "Kansas City Royals"],
    "LAA": ["ANA", "Los Angeles A", "Angels", "Los Angeles Angels", "LA Angels"],
    "LAD": ["Los Angeles D", "Dodgers", "Los Angeles Dodgers", "LA Dodgers"],
    "MIA": ["FLA", "Miami", "Marlins", "Miami Marlins"],
    "MIL": ["Milwaukee", "Brewers", "Milwaukee Brewers"],
    "MIN": ["Minnesota", "Twins", "Minnesota Twins"],
    "NYM": ["New York M", "Mets", "New York Mets"],
    "NYY": ["New York Y", "Yankees", "New York Yankees"],
    "PHI": ["Philadelphia", "Phillies", "Philadelphia Phillies"],
    "PIT": ["Pittsburgh", "Pirates", "Pittsburgh Pirates"],
    "SD": ["SDP", "San Diego", "Padres", "San Diego Padres"],
    "SF": ["SFG", "San Francisco", "Giants", "San Francisco Giants"],
    "SEA": ["Seattle", "Mariners", "Seattle Mariners"],
    "STL": ["St. Louis", "Cardinals", "St. Louis Cardinals"],
    "TB": ["TBR", "Tampa Bay", "Rays", "Tampa Bay Rays"],
    "TEX": ["Texas", "Rangers", "Texas Rangers"],
    "TOR": ["Toronto", "Blue Jays", "Toronto Blue Jays"],
    "WSH": ["WSN", "WAS", "Washington", "Nationals", "Washington Nationals"],
}

TABLES: Dict[str, Dict[str, List[str]]] = {
    "basketball_nba": _NBA,
    "americanfootball_nfl": _NFL,
    "icehockey_nhl": _NHL,
    "baseball_mlb": _MLB,
}

# Soccer name variants folded into one key (normalised forms, after club suffixes are dropped)
SOCCER_VARIANTS = {
    "stade brestois 29": "brest", "stade brestois": "brest", "kobenhavn": "copenhagen",
    "man city": "manchester city", "man utd": "manchester united", "man united": "manchester united",
    "manchester utd": "manchester united", "spurs": "tottenham", "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton", "wolverhampton wanderers": "wolverhampton", "brighton and hove albion": "brighton",
    "nottm forest": "nottingham forest", "newcastle united": "newcastle", "west ham united": "west ham",
    "leeds united": "leeds", "leicester city": "leicester", "ipswich town": "ipswich", "hull city": "hull",
    "coventry city": "coventry", "norwich city": "norwich", "cardiff city": "cardiff", "stoke city": "stoke",
    "sheffield utd": "sheffield united", "sheffield weds": "sheffield wednesday", "qpr": "queens park rangers",
    "west brom": "west bromwich albion", "inter": "internazionale", "inter milan": "internazionale",
    "internazionale milano": "internazionale", "milan": "ac milan", "roma": "as roma", "napoli": "ssc napoli",
    "lazio": "ss lazio", "atletico": "atletico madrid", "atletico de madrid": "atletico madrid",
    "club atletico de madrid": "atletico madrid", "atleti": "atletico madrid", "barca": "barcelona",
    "athletic bilbao": "athletic club", "real betis balompie": "real betis", "betis": "real betis",
    "bayern": "bayern munich", "bayern munchen": "bayern munich", "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund", "leverkusen": "bayer leverkusen", "bayer 04 leverkusen": "bayer leverkusen",
    "gladbach": "borussia monchengladbach", "monchengladbach": "borussia monchengladbach",
    "rasenballsport leipzig": "rb leipzig", "leipzig": "rb leipzig", "cologne": "koln", "1 koln": "koln",
    "psg": "paris saint germain", "paris sg": "paris saint germain",
    "olympique marseille": "marseille", "olympique de marseille": "marseille", "olympique lyonnais": "lyon",
    "sporting lisbon": "sporting cp", "sporting clube de portugal": "sporting cp",
    "new york rb": "new york red bulls", "ny red bulls": "new york red bulls", "red bull new york": "new york red bulls",
    "nycfc": "new york city", "lafc": "los angeles", "los angeles football club": "los angeles",
    "la galaxy": "los angeles galaxy", "dc united": "d c united", "montreal impact": "montreal",
    "sporting kc": "sporting kansas city", "st louis city": "st louis", "miami": "inter miami",
    "red bull salzburg": "salzburg", "rb salzburg": "salzburg", "psv eindhoven": "psv",
    # national teams, as eloratings.net names them
    "usa": "united states", "korea republic": "south korea", "republic of korea": "south korea",
    "korea dpr": "north korea", "ir iran": "iran", "cote d ivoire": "ivory coast", "turkiye": "turkey",
    "czech republic": "czechia", "china pr": "china", "republic of ireland": "ireland", "cabo verde": "cape verde",
    "north macedonia": "macedonia", "congo dr": "dr congo", "bosnia herzegovina": "bosnia and herzegovina",
}

# College variants (normalised): the venues' and ESPN's short forms of the same school
COLLEGE_VARIANTS = {
    "uconn": "connecticut", "ole miss": "mississippi", "pitt": "pittsburgh", "nc state": "north carolina state",
    "n c state": "north carolina state", "lsu": "louisiana state", "usc": "southern california",
    "ucf": "central florida", "smu": "southern methodist", "tcu": "texas christian", "byu": "brigham young",
    "unlv": "nevada las vegas", "utep": "texas el paso", "utsa": "texas san antonio", "uab": "alabama birmingham",
    "umass": "massachusetts", "miami fl": "miami", "miami florida": "miami", "miami oh": "miami ohio",
    "miami oh redhawks": "miami ohio", "fiu": "florida international", "fau": "florida atlantic",
    "app state": "appalachian state", "southern miss": "southern mississippi", "vcu": "virginia commonwealth",
    "louisiana monroe": "ul monroe", "la monroe": "ul monroe", "ulm": "ul monroe",
}

# Words too common to identify a team on their own (a name that is only these never matches by containment)
GENERIC = {
    "fc", "cf", "sc", "afc", "united", "city", "real", "sporting", "athletic", "club", "state", "university",
    "new", "york", "los", "angeles", "san", "saint", "st", "de", "the", "of", "and", "town", "county", "rovers",
    "wanderers", "albion", "college", "tech", "a", "north", "south", "east", "west", "central",
    "northern", "southern", "eastern", "western", "inter", "internacional", "atletico", "sport", "sports",
}

CLUB_WORDS = {"fc", "cf", "afc", "sc", "sk", "fk", "cd", "ud", "sd", "rcd", "rc", "ss", "ssc", "ac", "as", "us",
              "sv", "vfb", "vfl", "tsg", "bsc", "fsv", "sl", "ca", "cr", "ec", "se", "club", "calcio", "1"}

INDIVIDUAL_PREFIXES = ("tennis_", "mma_", "boxing_")
TEAM_TOL_HOURS = 3.0  # match_events' start-time tolerance for teams ...
PERSON_TOL_HOURS = 12.0  # ... and for tennis players and fighters

# Letters with no accent to strip (NFKD leaves them whole, and an ASCII fold would drop them): "Bodø" is "Bodo"
_SPELLED = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ł": "l",
                          "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D"})


def _plain(value: Any) -> str:
    """Lowercase, accent-free (ø as o, æ as ae, ł as l...), "&" as "and", punctuation as spaces, single-spaced."""
    text = unicodedata.normalize("NFKD", str(value or "").translate(_SPELLED)).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ").replace("'", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def kind(league: str) -> str:
    """'table' (NBA/NFL/NHL/MLB), 'soccer', 'college', 'person' (tennis, MMA) or 'team'."""
    league = league or ""
    if league in TABLES:
        return "table"
    if league.startswith(INDIVIDUAL_PREFIXES):
        return "person"
    if league.startswith("soccer_"):
        return "soccer"
    if league in ("americanfootball_ncaaf", "basketball_ncaab", "basketball_wncaab") or "ncaa" in league:
        return "college"
    return "team"


def _build_index() -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for league, table in TABLES.items():
        aliases: Dict[str, str] = {}
        for canonical, spellings in table.items():
            for spelling in [canonical, *spellings]:
                key = _plain(spelling)
                if key.startswith("la "):  # "LA Clippers" and "Los Angeles Clippers" are one name
                    aliases.setdefault("los angeles " + key[3:], canonical)
                if aliases.get(key, canonical) != canonical:
                    raise ValueError(f"{league}: {spelling!r} names two teams")
                aliases[key] = canonical
        index[league] = aliases
    return index


_INDEX = _build_index()


def _soccer_name(plain: str) -> str:
    words = [word for word in plain.split() if word not in CLUB_WORDS]
    name = " ".join(words) or plain
    return SOCCER_VARIANTS.get(name, name)


def _college_name(plain: str) -> str:
    words = plain.split()
    if words and words[-1] == "st":  # "Kennesaw St." is Kennesaw State ...
        words[-1] = "state"
    if words and words[0] == "st":  # ... and "St. John's" is Saint John's
        words[0] = "saint"
    name = " ".join(word for word in words if word not in ("university", "the"))
    return COLLEGE_VARIANTS.get(name, name)


def team_key(league: str, name_or_abbr: Any) -> Optional[str]:
    """One key for any spelling of a team (or player) in ``league``; None when there is nothing to key or a
    table league doesn't know the name."""
    plain = _plain(name_or_abbr)
    if not plain:
        return None
    sort = kind(league)
    if sort == "table":
        aliases = _INDEX[league]
        found = aliases.get(plain)
        if found is None and plain.startswith("la "):
            found = aliases.get("los angeles " + plain[3:])
        return found
    if sort == "soccer":
        return _soccer_name(plain)
    if sort == "college":
        return _college_name(plain)
    return plain


# --- matching -----------------------------------------------------------------------------------------

def _names(side: Any) -> List[str]:
    """Every spelling a side carries: a string, or a dict's name/short/abbr/displayName/... fields."""
    if side is None:
        return []
    if isinstance(side, str):
        return [side]
    if isinstance(side, Mapping):
        fields = ("name", "short", "abbr", "abbreviation", "displayName", "shortDisplayName", "location",
                  "nickname", "full")
        return [str(side[field]) for field in fields if isinstance(side.get(field), (str, int)) and str(side[field])]
    if isinstance(side, (list, tuple)):
        return [str(item) for item in side if item]
    return []


def _keys(league: str, names: Iterable[str]) -> Set[str]:
    return {key for key in (team_key(league, name) for name in names) if key}


def _tokens(key: str) -> Set[str]:
    return set(key.split())


def _person_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    if set(ta) == set(tb):
        return 0.95  # the same names in another order ("Zhang Zhizhen" / "Zhizhen Zhang")
    if ta[-1] == tb[-1]:
        if len(ta) == 1 or len(tb) == 1 or ta[0][0] == tb[0][0]:
            return 0.9  # same surname, same first initial ("J Sinner" / "Jannik Sinner")
        return 0.4
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 2 and " ".join(small) in " ".join(big):
        return 0.85  # "Ugo Carabelli" in "Camilo Ugo Carabelli"
    return 0.0


def _team_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if small <= big and small - GENERIC:
        return 0.85  # "kennesaw state" in "kennesaw state owls"; "atlanta" in "atlanta united"
    shared = ta & tb
    if not shared - GENERIC:
        return 0.0  # "los angeles" alone doesn't say which Los Angeles team
    return len(shared) / len(ta | tb)


def side_score(league: str, a: Sequence[str], b: Sequence[str]) -> float:
    """How well two spellings of one side match: 1.0 for the same key, 0 for none."""
    keys_a, keys_b = _keys(league, a), _keys(league, b)
    if keys_a & keys_b:
        return 1.0
    sort = kind(league)
    if sort == "table":
        if keys_a and keys_b:
            return 0.0  # both known to the table and different: different teams
        # one side's spelling isn't in the table: fall back to its names
    scorer = _person_score if sort == "person" else _team_score
    plain_a = keys_a | {_plain(name) for name in a if _plain(name)}
    plain_b = keys_b | {_plain(name) for name in b if _plain(name)}
    return max((scorer(x, y) for x in plain_a for y in plain_b), default=0.0)


MATCH = 0.6  # each side must score at least this


def _game_id(game: Mapping[str, Any]) -> Optional[str]:
    for field in ("id", "game_id", "event_id", "odds_id", "espn_id"):
        if game.get(field) not in (None, ""):
            return str(game[field])
    return None


def _start(game: Mapping[str, Any]) -> Optional[float]:
    ts = game.get("_ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    return parse_time(game.get("start") or game.get("commence_time"))


def _day(ts: Optional[float], fallback: Any = None) -> Optional[datetime.date]:
    if ts is not None:
        return datetime.datetime.fromtimestamp(ts, EASTERN).date()
    if isinstance(fallback, str):
        try:
            return datetime.date.fromisoformat(fallback[:10])
        except ValueError:
            return None
    return None


def _event_names(event: Mapping[str, Any], side: str) -> List[str]:
    names = _names(event.get(side))
    for field in (f"{side}_abbr", f"{side}_name"):
        if event.get(field):
            names.append(str(event[field]))
    return names


def orient(event: Mapping[str, Any], swap: bool) -> Dict[str, Any]:
    """A copy of a MarketEvent; with ``swap`` its home and away are exchanged everywhere."""
    out = dict(event)
    markets = {name: dict(market) for name, market in (event.get("markets") or {}).items()}
    if swap:
        out["home"], out["away"] = event.get("away"), event.get("home")
        for field in ("abbr", "key"):
            out[f"home_{field}"], out[f"away_{field}"] = event.get(f"away_{field}"), event.get(f"home_{field}")
        for name in ("h2h", "spreads"):
            market = markets.get(name)
            if market and "home" in market and "away" in market:
                market["home"], market["away"] = market["away"], market["home"]
                if name == "spreads" and isinstance(market.get("point"), (int, float)):
                    market["point"] = -market["point"]
    out["swapped"] = bool(event.get("swapped")) != swap
    out["markets"] = markets
    return out


def match_events(league: str, games: Iterable[Mapping[str, Any]], market_events: Iterable[Mapping[str, Any]],
                 tol_hours: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """{game id: the venue event for that game, oriented to the game's home and away}. See the module notes."""
    if tol_hours is None:
        tol_hours = PERSON_TOL_HOURS if kind(league) == "person" else TEAM_TOL_HOURS
    tolerance = tol_hours * 3600
    game_list = [(gid, game) for game in games if isinstance(game, Mapping) for gid in [_game_id(game)] if gid]
    event_list = [event for event in market_events if isinstance(event, Mapping)]
    candidates = []
    for gi, (gid, game) in enumerate(game_list):
        g_home, g_away = _names(game.get("home")), _names(game.get("away"))
        if not g_home or not g_away:
            continue
        g_start = _start(game)
        for ei, event in enumerate(event_list):
            e_start = parse_time(event.get("start"))
            if g_start is not None and e_start is not None:
                gap = abs(g_start - e_start)
                if gap > tolerance:
                    continue
            else:
                g_day = _day(g_start, game.get("date"))
                e_day = _day(e_start, event.get("date"))
                if g_day and e_day and abs((g_day - e_day).days) > 1:
                    continue
                gap = 12 * 3600.0  # unknown: ranked after every pair whose starts are both known
            e_home, e_away = _event_names(event, "home"), _event_names(event, "away")
            straight = (side_score(league, g_home, e_home), side_score(league, g_away, e_away))
            crossed = (side_score(league, g_home, e_away), side_score(league, g_away, e_home))
            for scores, swap in ((straight, False), (crossed, True)):
                if min(scores) >= MATCH:
                    candidates.append((-sum(scores), gap, gi, ei, swap))
    candidates.sort()
    used_games: Set[int] = set()
    used_events: Set[int] = set()
    matched: Dict[str, Dict[str, Any]] = {}
    for _score, _gap, gi, ei, swap in candidates:
        if gi in used_games or ei in used_events:
            continue
        used_games.add(gi)
        used_events.add(ei)
        matched[game_list[gi][0]] = orient(event_list[ei], swap)
    return matched
