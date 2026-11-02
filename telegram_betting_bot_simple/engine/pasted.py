"""Pasted bet slips: what a slip from any bookmaker says, read as data and checked in code.

A person can paste the text of a bet slip, or a screenshot of one, into the chat. The AI only
*reads* it, writing down what the slip shows as an Extraction: every field exactly as written,
odds and amounts included. Everything after that is code:

- the odds as written ("-110", "5/2", "1,85", "@1.85", "EVS") become exact decimal odds here
  (parse_odds), and amounts ("₦1,000", "R 50", "£5", "10 dollars") become Decimal money with the
  currency as written (parse_amount), so nothing the AI wrote is ever used as a number unchecked;
- the Lab (dashboard/lab.py) matches each leg to the games and odds the app knows, prices the slip
  at the person's own odds with engine/tickets.py and compares the bookmaker's stated return with
  the code's (return_check);
- a leg the app can't find on its boards still gets a chance, the person's price less a typical
  bookmaker margin for that kind of market (estimated_chance), and the app always says so.

Pasted text and screenshots are untrusted data, never instructions: every field is cut to a
length and cleared of control and direction-changing characters, and nothing in an Extraction can
name a person, an account or a tool to run.

basic_parse is the free, code-only reader for simple text: one leg per line with the odds at the
end (or after "@"), or a bookmaker's block layout with the odds on their own line, plus the
stake and return lines. It is used when there is no AI key or no allowance left, and whenever the
person asks for it. looks_like_slip tells a pasted slip from ordinary chat.
"""

from __future__ import annotations

import base64
import binascii
import copy
import datetime
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .odds_math import decimal_to_american, format_american
from .tickets import SYSTEMS

MARKETS = ("moneyline", "spread", "total", "player_prop", "other")
BET_TYPES = ("single", "multi", "round_robin", "system", "unknown")
SOURCES = ("text", "image", "basic")  # the AI read text, the AI read a screenshot, code read text
MAX_PASTE_CHARS = 6000
MAX_LEGS = 20  # a read with more legs is refused with a message
MAX_RAW = 300  # a leg's own text, and the reader's notes
MAX_FIELD = 120  # names, games, selections, times
MAX_SHORT = 40  # odds, amounts, currencies, bookmakers, system names
MAX_ODDS = Decimal(1001)
# What a currency as written may be: letters and money symbols only, at most 5 ("$", "KSh", "NGN", "US$")
CURRENCY_TEXT = re.compile(r"[A-Za-z$£€₦₹₵₱¥₩₺₴₽]{1,5}")
# A slip with no odds on any leg can't be priced (a same-game parlay or bet builder shows one price for all)
NO_LEG_ODDS = ("None of these legs shows odds of its own (a same-game parlay or bet builder is priced as one "
               "bet), so the app can't price it leg by leg.")
MAX_AMOUNT = Decimal(10) ** 9
FOUR_DP = Decimal("0.0001")
CENTS = Decimal("0.01")
PROP_STATS = ("points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers", "pra")
# Screenshots: what the API takes (a JPEG, PNG or WebP), at most 1.8 MB once decoded (2.4 MB as base64)
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_IMAGE_BYTES = 1_800_000
# The free reader's 400 when it can't find a bet
BASIC_FORMAT_HELP = ("Couldn't find any bets in that. The free reader takes one bet per line with its odds at the "
                     "end, e.g. \"Boston Celtics -3.5 -110\" or \"Arsenal to win @ 2.10\", and lines such as "
                     "\"Stake $10\" and \"To win $19.10\".")

# What a bookmaker typically keeps on each kind of market (the overround as a share of the fair price), used
# only for a leg the app can't find on its boards: the honest chance is then the person's price less this
TYPICAL_MARGIN: Dict[str, float] = {
    "moneyline": 0.045,  # two-way (NBA, NFL, NHL, MLB, tennis, MMA)
    "moneyline_3way": 0.06,  # soccer's home / draw / away
    "spread": 0.045,
    "total": 0.045,
    "player_prop": 0.08,
    "other": 0.08,
}

# Lower-case words a slip uses for a league -> The Odds API's key
LEAGUE_ALIASES: Dict[str, str] = {
    "nba": "basketball_nba", "national basketball association": "basketball_nba",
    "wnba": "basketball_wnba",
    "ncaab": "basketball_ncaab", "college basketball": "basketball_ncaab", "ncaa basketball": "basketball_ncaab",
    "ncaam": "basketball_ncaab", "cbb": "basketball_ncaab", "march madness": "basketball_ncaab",
    "euroleague": "basketball_euroleague",
    "nfl": "americanfootball_nfl", "national football league": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf", "college football": "americanfootball_ncaaf",
    "ncaa football": "americanfootball_ncaaf", "cfb": "americanfootball_ncaaf",
    "mlb": "baseball_mlb", "major league baseball": "baseball_mlb",
    "nhl": "icehockey_nhl", "national hockey league": "icehockey_nhl",
    "epl": "soccer_epl", "premier league": "soccer_epl", "english premier league": "soccer_epl",
    "england premier league": "soccer_epl", "barclays premier league": "soccer_epl",
    "champions league": "soccer_uefa_champs_league", "uefa champions league": "soccer_uefa_champs_league",
    "ucl": "soccer_uefa_champs_league",
    "europa league": "soccer_uefa_europa_league", "uefa europa league": "soccer_uefa_europa_league",
    "uel": "soccer_uefa_europa_league",
    "la liga": "soccer_spain_la_liga", "laliga": "soccer_spain_la_liga", "spain la liga": "soccer_spain_la_liga",
    "primera division": "soccer_spain_la_liga",
    "serie a": "soccer_italy_serie_a", "italy serie a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga", "germany bundesliga": "soccer_germany_bundesliga",
    "ligue 1": "soccer_france_ligue_one", "ligue one": "soccer_france_ligue_one",
    "france ligue 1": "soccer_france_ligue_one",
    "mls": "soccer_usa_mls", "major league soccer": "soccer_usa_mls",
    "efl championship": "soccer_efl_champ", "english championship": "soccer_efl_champ",
    "fa cup": "soccer_fa_cup",
    "eredivisie": "soccer_netherlands_eredivisie",
    "primeira liga": "soccer_portugal_primeira_liga",
    "scottish premiership": "soccer_spl",
    "ufc": "mma_mixed_martial_arts", "mma": "mma_mixed_martial_arts", "mixed martial arts": "mma_mixed_martial_arts",
}
# Other countries' "Premier League" (Kenya, Ghana, Russia, ...) is not the English one
_OTHER_PREMIER = re.compile(
    r"\b(?:kenya|kenyan|ghana|ghanaian|nigeria|nigerian|npfl|south african|russia|russian|egypt|egyptian|ukraine|"
    r"ukrainian|uganda|tanzania|zambia|malawi|bangladesh|hong kong|welsh|wales|northern ireland|irish|india|indian|"
    r"jamaica|israel|israeli|kuwait|lebanon|syria|iraq|armenia|belarus|kazakhstan|azerbaijan|ethiopia|rwanda|"
    r"zimbabwe|botswana|singapore|malaysia|tunisia|algeria|libya)\b")
# A soccer league's name used by other countries ("Brazil Serie A", "Austria Bundesliga", "Argentina Primera
# Division", "Algeria Ligue 1"): any country named beside it that isn't the league's own
_COUNTRY = re.compile(
    r"\b(?:england|english|scotland|scottish|wales|welsh|ireland|irish|spain|spanish|italy|italian|germany|german|"
    r"france|french|netherlands|dutch|holland|portugal|portuguese|belgium|belgian|austria|austrian|switzerland|"
    r"swiss|denmark|danish|sweden|swedish|norway|norwegian|finland|finnish|iceland|icelandic|poland|polish|czech|"
    r"slovakia|slovak|hungary|hungarian|romania|romanian|bulgaria|bulgarian|serbia|serbian|croatia|croatian|"
    r"slovenia|bosnia|greece|greek|turkey|turkish|cyprus|cypriot|russia|russian|ukraine|ukrainian|israel|israeli|"
    r"brazil|brazilian|brasil|brasileiro|brasileirao|ecuador|ecuadorian|argentina|argentine|argentinian|chile|"
    r"chilean|colombia|colombian|peru|peruvian|uruguay|uruguayan|paraguay|paraguayan|bolivia|bolivian|venezuela|"
    r"mexico|mexican|canada|canadian|japan|japanese|korea|korean|china|chinese|australia|australian|india|indian|"
    r"saudi|qatar|qatari|uae|emirates|egypt|egyptian|algeria|algerian|morocco|moroccan|tunisia|tunisian|nigeria|"
    r"nigerian|npfl|ghana|ghanaian|kenya|kenyan|south africa|south african|uganda|ugandan|tanzania|zambia|malawi|"
    r"zimbabwe|botswana|ethiopia|rwanda|libya|bangladesh|pakistan|sri lanka|lanka|caribbean|jamaica|singapore|"
    r"malaysia|indonesia|thailand|vietnam|hong kong|kuwait|lebanon|syria|iraq|iran|armenia|belarus|kazakhstan|"
    r"azerbaijan)\b")
_OWN_COUNTRY: Dict[str, Tuple[str, ...]] = {
    "soccer_epl": ("england", "english"), "soccer_efl_champ": ("england", "english"),
    "soccer_fa_cup": ("england", "english"), "soccer_spain_la_liga": ("spain", "spanish"),
    "soccer_italy_serie_a": ("italy", "italian"), "soccer_germany_bundesliga": ("germany", "german"),
    "soccer_france_ligue_one": ("france", "french"), "soccer_netherlands_eredivisie": ("netherlands", "dutch", "holland"),
    "soccer_portugal_primeira_liga": ("portugal", "portuguese"), "soccer_spl": ("scotland", "scottish"),
    "soccer_usa_mls": ("canada", "canadian"),
}
# Europe's club cups share their names with other confederations' ("CAF Champions League", "AFC Champions League")
_CONFEDERATION = re.compile(r"\b(?:caf|afc|concacaf|conmebol|ofc|africa|african|asia|asian|arab|oceania|"
                            r"libertadores|sudamericana)\b")
# A lower tier or another version of a league ("NBA G League", "Women's Premier League", "Serie A U19")
_VARIANT = re.compile(r"\b(?:women|women's|womens|woman|ladies|female|femenina|femminile|frauen|feminine|wsl|"
                      r"g league|summer league|2k|u-?\s?(?:1[5-9]|2[0-3])(?![.,]\d)|youth|junior|juniors|reserve|"
                      r"reserves|academy|amateur|hypermotion|smartbank|segunda)\b")
_TIER_WORDS = r"(?:ii|iii|2nd|3rd|second|third)"
# A sport named without its league ("soccer", "basketball"): the boards to look among
SPORT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "soccer": ("soccer_",), "football": ("soccer_", "americanfootball_"), "american football": ("americanfootball_",),
    "basketball": ("basketball_",), "ice hockey": ("icehockey_",), "hockey": ("icehockey_",),
    "baseball": ("baseball_",), "tennis": ("tennis_",), "mma": ("mma_",), "boxing": ("boxing_",),
    "cricket": ("cricket_",), "rugby": ("rugbyleague_", "rugbyunion_"),
}

# Words bookmakers use for each market (the reader's own words are mapped to MARKETS with these)
_MARKET_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "moneyline": ("moneyline", "money line", "ml", "h2h", "head to head", "match result", "full time result",
                  "fulltime result", "ft result", "1x2", "match winner", "winner", "to win", "win", "3 way",
                  "three way", "result", "match betting", "match odds", "full time", "game winner", "outright winner"),
    "spread": ("spread", "spreads", "point spread", "points spread", "handicap", "asian handicap", "run line",
               "puck line", "line", "alternate spread", "alt spread", "ah", "spread betting"),
    "total": ("total", "totals", "total points", "total goals", "total runs", "over under", "over/under", "o/u",
              "goals over/under", "goals over under", "total match goals", "game total", "alternate total",
              "alt total", "match goals", "totals over/under", "total match points"),
    "player_prop": ("player prop", "player props", "player_prop", "prop", "props", "player points",
                    "player rebounds", "player assists", "player threes", "player points + rebounds + assists",
                    "player shots", "points", "rebounds", "assists", "threes", "player", "anytime goalscorer",
                    "first goalscorer", "anytime touchdown scorer", "to score"),
    "other": ("both teams to score", "btts", "gg/ng", "double chance", "draw no bet", "correct score",
              "half time/full time", "ht/ft", "half time result", "first half", "1st half", "corners", "cards",
              "bookings", "odd/even", "other"),
}
_MARKET_LOOKUP = {words: market for market, phrases in _MARKET_SYNONYMS.items() for words in phrases}

_SYSTEM_KEYS = {re.sub(r"[^a-z0-9]", "", key): key for key in SYSTEMS}
_SYSTEM_KEYS.update({re.sub(r"[^a-z0-9]", "", system.title.lower()): key for key, system in SYSTEMS.items()})

_STAT_WORDS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("pra", re.compile(r"pts\s*\+\s*reb\s*\+\s*ast|points\s*\+\s*rebounds\s*\+\s*assists|p\s*\+\s*r\s*\+\s*a|\bpra\b",
                       re.I)),
    ("threes", re.compile(r"\bthrees\b|\b3-?pointers?\b|\bthree[- ]pointers?\b|\b3pm\b|\b3pt\b|\b3-pt\b|\bfg3m\b",
                          re.I)),
    ("points", re.compile(r"\bpoints?\b|\bpts\b", re.I)),
    ("rebounds", re.compile(r"\brebounds?\b|\brebs?\b", re.I)),
    ("assists", re.compile(r"\bassists?\b|\basts?\b", re.I)),
    ("steals", re.compile(r"\bsteals?\b|\bstl\b", re.I)),
    ("blocks", re.compile(r"\bblocks?\b|\bblk\b|\bblocked shots\b", re.I)),
    ("turnovers", re.compile(r"\bturnovers?\b|\btov\b", re.I)),
)
# Other sports' player stats, kept in the slip's own words (the app models NBA players only)
_OTHER_STATS = re.compile(r"\b(?:passing|rushing|receiving)\s+(?:yards|tds?|touchdowns)\b|\breceptions\b|"
                          r"\bstrikeouts\b|\bhits\b|\btotal bases\b|\bshots on (?:goal|target)\b|\bshots\b|"
                          r"\bsaves\b|\bgoals\b|\bhome runs?\b|\brbis?\b|\btackles\b|\baces\b", re.I)

# Control characters and the invisible ones that reorder or hide text
_HIDDEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ‪-‮⁠-⁤"
                     r"⁦-⁩﻿]")
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "‒": "-", "‐": "-",
                         "‑": "-", "＋": "+"})


# --- what the reader returns ---------------------------------------------------------------------

def clean_text(text: Any, limit: int = MAX_PASTE_CHARS) -> str:
    """Pasted text as plain lines: compatibility forms folded (full-width digits), minus signs and dashes as
    "-", hidden characters gone, at most ``limit`` characters."""
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_DASHES)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return _HIDDEN.sub("", value)[:limit]


def _short(value: Any, limit: int) -> Optional[str]:
    """One field of a read: text on one line, at most ``limit`` characters, None when empty."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        value = str(value)
    if not isinstance(value, str):
        return None
    text = " ".join(clean_text(value, limit * 4).split())[:limit].strip()
    return text or None


def market_of(value: Any) -> str:
    """A market in the slip's or the reader's words ("Match Result", "h2h", "Over/Under") as one of MARKETS."""
    text = (_short(value, MAX_FIELD) or "").lower()
    if text in MARKETS:
        return text
    words = re.sub(r"\s*[+-]?\d+(?:\.\d+)?$", "", re.sub(r"\(.*?\)", " ", text).strip())  # "over/under 2.5"
    words = " ".join(re.sub(r"[^a-z0-9/+ ]", " ", words.replace("_", " ")).split())
    return _MARKET_LOOKUP.get(words) or _MARKET_LOOKUP.get(words.replace("/", " ")) or "other"


def bet_type_of(value: Any) -> str:
    """A bet type in a slip's words ("Parlay", "4 Fold Accumulator", "Singles", "Lucky 15") as one of BET_TYPES."""
    text = (_short(value, MAX_FIELD) or "").lower()
    if text in BET_TYPES:
        return text
    if re.search(r"round\s*robin|\brr\b", text):
        return "round_robin"
    if system_key(text) or re.search(r"\bsystem\b", text):
        return "system"
    if re.search(r"\bsingles?\b|\bstraight\b", text):
        return "single"
    if re.search(r"parlay|multi|accumulator|\bacca\b|treble|\bdouble\b|\bfold\b|\bsgp\b|combo|bet\s*builder|"
                 r"\bteaser\b", text):
        return "multi"
    return "unknown"


def system_key(value: Any) -> Optional[str]:
    """A named system bet's key in engine/tickets.SYSTEMS ("Lucky 15" -> "lucky15"), or None."""
    text = re.sub(r"[^a-z0-9]", "", (_short(value, MAX_SHORT) or "").lower())
    if text in _SYSTEM_KEYS:
        return _SYSTEM_KEYS[text]
    return next((key for name, key in sorted(_SYSTEM_KEYS.items(), key=lambda item: -len(item[0]))
                 if len(name) >= 5 and name in text), None)


def prop_stat_of(value: Any) -> Optional[str]:
    """A player stat in a slip's words ("Points", "3PM", "Pts + Reb + Ast") as one of PROP_STATS; another
    sport's stat in the slip's own words ("passing yards"); None when there is none."""
    text = _short(value, MAX_FIELD)
    if not text:
        return None
    lowered = text.lower()
    if lowered in PROP_STATS:
        return lowered
    for stat, pattern in _STAT_WORDS:
        if pattern.search(lowered):
            # "Points + Rebounds" (no assists) is a combination the model doesn't project
            if stat == "points" and re.search(r"\+\s*(?:reb|ast|rebounds|assists)", lowered):
                return " ".join(re.sub(r"[^a-z+ ]", " ", lowered).split())[:MAX_SHORT] or None
            return stat
    other = _OTHER_STATS.search(lowered)
    return other.group(0).lower()[:MAX_SHORT] if other else None


def _line_value(value: Any) -> Optional[float]:
    """A spread or total line (-4.5, 224.5) from a number or text ("+3.5", "o224.5", "PK"), or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
    else:
        text = (_short(value, MAX_SHORT) or "").lower().replace(" ", "")
        if text in ("pk", "pick", "pickem", "pick'em", "even"):
            return 0.0
        match = re.fullmatch(r"[ou]?([+-]?\d{1,4}(?:[.,]\d{1,2})?)", text)
        if not match:
            return None
        number = float(match.group(1).replace(",", "."))
    if number != number or abs(number) > 1000:  # NaN, or nothing a line can be
        return None
    return round(number, 2)


def _over_under(value: Any) -> Optional[str]:
    text = (_short(value, MAX_SHORT) or "").lower()
    if text in ("over", "o", "more", "over (+)", "+"):
        return "over"
    if text in ("under", "u", "less", "under (-)", "-"):
        return "under"
    return None


class ExtractedLeg(BaseModel):
    """One leg of a slip as the reader (the AI or basic_parse) wrote it down: all text as written."""

    model_config = ConfigDict(extra="ignore")

    raw: str  # the leg's own text from the slip (<= 300 chars)
    sport_hint: Optional[str] = None  # "NBA", "Premier League", "NFL", "tennis", ...
    event_text: Optional[str] = None  # "Lakers @ Celtics", "Arsenal v Chelsea"
    selection: str  # "Boston Celtics", "Over", "Draw", "LeBron James"
    market: str  # one of MARKETS
    line: Optional[float] = None  # -4.5, 224.5, 27.5 (sign as written for spreads)
    prop_stat: Optional[str] = None  # points, rebounds, assists, threes, steals, blocks, turnovers, pra
    over_under: Optional[str] = None  # "over" / "under" for totals and props
    odds_text: Optional[str] = None  # exactly as written: "1.85", "+150", "-110", "5/2", "EVS"
    start_text: Optional[str] = None  # "Tue 7:30 PM", "26 Oct 19:45"

    @field_validator("raw", mode="before")
    @classmethod
    def _raw(cls, value: Any) -> str:
        return _short(value, MAX_RAW) or ""

    @field_validator("selection", mode="before")
    @classmethod
    def _selection(cls, value: Any) -> str:
        return _short(value, MAX_FIELD) or ""

    @field_validator("sport_hint", "event_text", "start_text", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Optional[str]:
        return _short(value, MAX_FIELD)

    @field_validator("odds_text", mode="before")
    @classmethod
    def _odds_text(cls, value: Any) -> Optional[str]:
        return _short(value, MAX_SHORT)

    @field_validator("market", mode="before")
    @classmethod
    def _market(cls, value: Any) -> str:
        return market_of(value)

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, value: Any) -> Optional[float]:
        return _line_value(value)

    @field_validator("prop_stat", mode="before")
    @classmethod
    def _stat(cls, value: Any) -> Optional[str]:
        return prop_stat_of(value)

    @field_validator("over_under", mode="before")
    @classmethod
    def _side(cls, value: Any) -> Optional[str]:
        return _over_under(value)

    @model_validator(mode="after")
    def _filled(self) -> "ExtractedLeg":
        # A leg is at least some text: the selection falls back to the raw text and the other way round
        if not self.selection:
            self.selection = self.raw[:MAX_FIELD]
        if not self.raw:
            self.raw = self.selection
        if self.market in ("total", "player_prop") and self.over_under is None:
            side = re.search(r"\b(over|under)\b", f"{self.selection} {self.raw}", re.I)
            self.over_under = side.group(1).lower() if side else None
        return self


class Extraction(BaseModel):
    """A whole slip as read: the bookmaker, the kind of bet, the money lines and the legs, all as written."""

    model_config = ConfigDict(extra="ignore")

    bookmaker: Optional[str] = None
    bet_type: str = "unknown"  # one of BET_TYPES ("singles" -> "single")
    k: Optional[int] = None  # round robin size, if stated
    system: Optional[str] = None  # trixie, yankee, ... if stated
    stake_text: Optional[str] = None  # as written, per bet if the slip says so: "$10", "10.00", "£5", "R50"
    total_stake_text: Optional[str] = None
    currency: Optional[str] = None  # "$", "£", "€", "R", "₦", "KSh", ... as written
    potential_return_text: Optional[str] = None
    legs: List[ExtractedLeg]
    notes: Optional[str] = None  # anything the reader couldn't place (<= 300 chars)

    @field_validator("bookmaker", "stake_text", "total_stake_text", "potential_return_text", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Optional[str]:
        return _short(value, MAX_SHORT)

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> Optional[str]:
        # A symbol or a code as written ("$", "£", "R", "KSh", "NGN", "US$"): anything else isn't a currency,
        # and the money lines' own symbols are used instead (Lab._paste_money)
        text = _short(value, MAX_SHORT)
        return text if text and CURRENCY_TEXT.fullmatch(text) else None

    @field_validator("system", mode="before")
    @classmethod
    def _system(cls, value: Any) -> Optional[str]:
        return _short(value, MAX_SHORT)

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, value: Any) -> Optional[str]:
        return _short(value, MAX_RAW)

    @field_validator("bet_type", mode="before")
    @classmethod
    def _bet_type(cls, value: Any) -> str:
        return bet_type_of(value) if value is not None else "unknown"

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            k = int(str(value).strip()) if value is not None else None
        except ValueError:
            return None
        return k if k is not None and 2 <= k <= 7 else None

    @field_validator("legs", mode="before")
    @classmethod
    def _legs(cls, value: Any) -> Any:
        # One more than the most a slip may have, so a refusal can say there were too many
        return list(value)[:MAX_LEGS + 1] if isinstance(value, (list, tuple)) else value


_NULL_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_NULL_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_NULL_INTEGER = {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def _described(schema: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {**schema, "description": text}


_LEG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "raw": {"type": "string", "description": "This leg's own text on the slip, copied as shown."},
        "sport_hint": _described(_NULL_STRING, "The sport or league if the slip shows it (NBA, Premier League)."),
        "event_text": _described(_NULL_STRING, "The game as written (Lakers @ Celtics, Arsenal v Chelsea)."),
        "selection": {"type": "string", "description": "What was picked: a team, Over, Under, Draw or a player."},
        "market": {"type": "string", "enum": list(MARKETS)},
        "line": _described(_NULL_NUMBER, "The spread or total line as written, sign included (-4.5, 224.5)."),
        "prop_stat": _described(_NULL_STRING, "For a player prop: the stat (points, rebounds, assists, threes, "
                                              "steals, blocks, turnovers, pra, or the slip's words)."),
        "over_under": {"anyOf": [{"type": "string", "enum": ["over", "under"]}, {"type": "null"}]},
        "odds_text": _described(_NULL_STRING, "The odds exactly as shown (1.85, -110, +150, 5/2, EVS); null if "
                                              "unreadable. Never work out or guess odds."),
        "start_text": _described(_NULL_STRING, "The start time as written, if shown."),
    },
    "required": ["raw", "sport_hint", "event_text", "selection", "market", "line", "prop_stat", "over_under",
                 "odds_text", "start_text"],
    "additionalProperties": False,
}
_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bookmaker": _described(_NULL_STRING, "The bookmaker or app, if shown."),
        "bet_type": {"type": "string", "enum": list(BET_TYPES)},
        "k": _described(_NULL_INTEGER, "A round robin's legs per bet, if stated."),
        "system": _described(_NULL_STRING, "A named system bet (Trixie, Yankee, Lucky 15), if stated."),
        "stake_text": _described(_NULL_STRING, "The stake as written, per bet if the slip says so."),
        "total_stake_text": _described(_NULL_STRING, "The total stake as written, if shown."),
        "currency": _described(_NULL_STRING, "The currency symbol or code as written ($, £, €, R, KSh, NGN)."),
        "potential_return_text": _described(_NULL_STRING, "The potential return, payout or to-win amount as written."),
        "legs": {"type": "array", "items": _LEG_SCHEMA},
        "notes": _described(_NULL_STRING, "Anything on the slip you couldn't place, briefly."),
    },
    "required": ["bookmaker", "bet_type", "k", "system", "stake_text", "total_stake_text", "currency",
                 "potential_return_text", "legs", "notes"],
    "additionalProperties": False,
}


def extraction_schema() -> Dict[str, Any]:
    """The JSON schema for Claude's structured output (every field listed and required, nullable where a slip
    may not show it, no extra fields). A fresh copy each time, so a caller can't change the shared one."""
    return copy.deepcopy(_SCHEMA)


# --- odds and amounts ----------------------------------------------------------------------------

_EVENS = {"evs", "evens", "even", "evn", "evns", "ev's", "1/1"}
_FRACTION_DENOMINATORS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40, 50, 100}


def _bounded(value: Decimal) -> Optional[Decimal]:
    if not value.is_finite():
        return None
    odds = value.quantize(FOUR_DP, rounding=ROUND_HALF_UP)
    return odds if 1 < odds <= MAX_ODDS else None


def parse_odds(text: Optional[str]) -> Optional[Decimal]:
    """Decimal odds (4 places) from odds as a slip writes them, or None.

    Decimal (1.85, 1,85, @1.85), American (+150, -110), fractional (5/2, 11/10) and "EVS"/"evens"/"even"
    (2.0). Odds must come out above 1 and at most 1001; anything else (a line like -4.5, a date, words) is
    None. An unsigned whole number is decimal odds (a long shot's 150), never American.
    """
    if text is None or isinstance(text, bool):
        return None
    value = clean_text(str(text), MAX_SHORT * 2).strip().lower()
    value = re.sub(r"^(?:odds|price|at)\s*[:=]?\s*", "", value)
    value = value.strip(" ()[]").lstrip("@").strip().replace(" ", "")
    if not value:
        return None
    if value in _EVENS:
        return Decimal("2.0000")
    try:
        fraction = re.fullmatch(r"(\d{1,4})/(\d{1,4})", value)
        if fraction:
            top, bottom = int(fraction.group(1)), int(fraction.group(2))
            return _bounded(1 + Decimal(top) / Decimal(bottom)) if top > 0 and bottom > 0 else None
        american = re.fullmatch(r"([+-])(\d{3,6})(?:\.0+)?", value)
        if american:
            number = Decimal(american.group(2))
            if number < 100:
                return None
            return _bounded(1 + number / 100 if american.group(1) == "+" else 1 + Decimal(100) / number)
        plain = re.fullmatch(r"(\d{1,4})(?:([.,])(\d{1,4}))?", value)
        if plain:
            whole, mark, part = plain.groups()
            if mark == "," and len(part) == 3:  # "1,000": a thousands separator
                return _bounded(Decimal(whole + part))
            return _bounded(Decimal(f"{whole}.{part}" if part else whole))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def _fractional(odds: Decimal) -> str:
    """Decimal odds as the bookmakers' fraction: the smallest denominator (up to 100) that gives the odds to 4
    places (1.85 -> 17/20, 1.9091 -> 10/11, 2.5 -> 3/2)."""
    profit = odds - 1
    for bottom in range(1, 101):
        top = (profit * bottom).to_integral_value(rounding=ROUND_HALF_UP)
        if top > 0 and abs(top / bottom - profit) <= Decimal("0.00005"):
            return f"{top}/{bottom}"
    nearest = Fraction(str(profit)).limit_denominator(100)
    return f"{max(nearest.numerator, 1)}/{nearest.denominator}"


def odds_view(decimal: Decimal) -> Dict[str, str]:
    """Odds in the three formats the app shows: {"decimal": "1.85", "american": "-118", "fractional": "17/20"}."""
    odds = decimal if isinstance(decimal, Decimal) else Decimal(str(decimal))
    return {"decimal": str(odds.quantize(CENTS, rounding=ROUND_HALF_UP)),
            "american": format_american(decimal_to_american(float(odds))),
            "fractional": _fractional(odds)}


# Currencies as slips write them: symbols and codes before or after the amount, and words after it
_CURRENCY_BEFORE = (r"US\$|C\$|A\$|NZ\$|R\$|KSh|Ksh|KSH|K[Ss]hs|USD|GBP|EUR|NGN|GHS|ZAR|KES|UGX|TZS|ZMW|CAD|AUD|"
                    r"INR|[$£€₦₹₵₱¥₩₺₴₽]|(?<![A-Za-z])[RNK](?=\s?\d)")
_CURRENCY_WORDS = {"dollars": "$", "dollar": "$", "bucks": "$", "usd": "USD", "pounds": "£", "pound": "£",
                   "quid": "£", "euros": "€", "euro": "€", "naira": "₦", "rand": "R", "shillings": "KSh",
                   "shilling": "KSh", "cedis": "₵", "cedi": "₵", "rupees": "₹", "kwacha": "K"}
_CURRENCY_AFTER = (r"USD|GBP|EUR|NGN|GHS|ZAR|KES|UGX|TZS|ZMW|CAD|AUD|INR|KSh|[$£€₦₹₵]|"
                   + "|".join(sorted(_CURRENCY_WORDS, key=len, reverse=True)))
_AMOUNT = re.compile(rf"(?P<before>{_CURRENCY_BEFORE})?\s?(?P<number>\d[\d,.]*\d|\d)"
                     rf"(?:\s?(?P<after>{_CURRENCY_AFTER})(?![A-Za-z]))?", re.I)


def _number(raw: str) -> Optional[Decimal]:
    """A written amount as a number: "1,000.50", "1.000,50", "5,50" (a decimal comma) or "1,000"."""
    text = raw.strip(".,")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):  # 1.000,50
            text = text.replace(".", "").replace(",", ".")
        else:  # 1,000.50
            text = text.replace(",", "")
    elif "," in text:
        groups = text.split(",")
        thousands = len(groups) > 2 or len(groups[-1]) == 3
        text = text.replace(",", "") if thousands else text.replace(",", ".")
    elif text.count(".") > 1:  # 1.000.000
        text = text.replace(".", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_amount(text: Optional[str]) -> Tuple[Optional[Decimal], Optional[str]]:
    """(amount in Decimal cents, currency as written) from money as a slip writes it.

    "$10", "10.00", "£5", "R 50", "₦1,000", "KSh 200", "NGN 1,000.00", "€5,50" and "10 dollars" all read;
    a currency written as a word comes back as its symbol. (None, None) when there is no amount.
    """
    if text is None or isinstance(text, bool):
        return None, None
    value = clean_text(str(text), MAX_SHORT * 4)
    match = _AMOUNT.search(value)
    if not match or value[:match.start("number")].rstrip().endswith("-"):  # no amount, or a negative one
        return None, None
    amount = _number(match.group("number"))
    currency = match.group("before") or match.group("after")
    if currency:
        currency = currency.strip()
        currency = _CURRENCY_WORDS.get(currency.lower(), currency)
    if amount is None or not amount.is_finite() or amount <= 0 or amount >= MAX_AMOUNT:
        return None, currency
    return amount.quantize(CENTS, rounding=ROUND_HALF_UP), currency


def return_check(computed: Optional[Decimal], stated: Optional[Decimal]) -> str:
    """"matches" when the bookmaker's stated return is within 1% or 0.05 of the code's (bookmakers round each
    leg and the total), "differs" otherwise, "unknown" when either is missing."""
    if computed is None or stated is None:
        return "unknown"
    gap = abs(Decimal(computed) - Decimal(stated))
    return "matches" if gap <= Decimal("0.05") or gap <= abs(Decimal(computed)) * Decimal("0.01") else "differs"


# --- leagues and chances -------------------------------------------------------------------------

def _words(text: Optional[str]) -> str:
    """Lower-case words only ("England - Premier League" -> "england premier league")."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9+]", " ", ascii_text).split())


def _another_league(alias: str, league: str, words: str, text: str) -> bool:
    """Whether words naming a league by ``alias`` mean another league of that name: another country's ("Brazil
    Serie A"), a lower tier ("2. Bundesliga", "Bundesliga 2") or a variant ("NBA G League", "Women's Premier
    League"), or another confederation's club cup ("CAF Champions League"). ``text`` keeps dots and slashes."""
    rest = re.sub(rf"\b{re.escape(alias)}\b", " ", words)
    spaced = re.escape(alias).replace(r"\ ", r"\s+")
    if _VARIANT.search(re.sub(rf"\b{spaced}\b", " ", text)):
        return True
    if re.search(rf"(?:\b[23]\.\s*|\b{_TIER_WORDS}\s+){spaced}\b", text) or (
            not alias[-1].isdigit() and re.search(rf"\b{spaced}\s+(?:[23]|{_TIER_WORDS})\b(?![.,:/]\d)", text)):
        return True
    if league in ("soccer_uefa_champs_league", "soccer_uefa_europa_league"):
        return bool(_CONFEDERATION.search(rest))
    if alias == "premier league" and _OTHER_PREMIER.search(words):
        return True
    if league.startswith("soccer_"):
        own = _OWN_COUNTRY.get(league, ())
        return any(country not in own for country in _COUNTRY.findall(rest))
    return False


def league_of(hint: Optional[str]) -> Tuple[Optional[str], bool]:
    """(the Odds API league key a slip's sport or league words name, whether they name another league of the
    same name the app doesn't cover): ("soccer_epl", False) for "England Premier League", (None, True) for
    "Brazil Serie A", "2. Bundesliga" or "NBA G League", (None, False) when they name no league."""
    short = _short(hint, MAX_FIELD)
    words = _words(short)
    if not words:
        return None, False
    if words in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[words], False
    ascii_text = unicodedata.normalize("NFKD", short or "").encode("ascii", "ignore").decode().lower()
    text = " ".join(re.sub(r"[^a-z0-9+./:]", " ", ascii_text).split())
    for alias in sorted(LEAGUE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", words):
            league = LEAGUE_ALIASES[alias]
            if _another_league(alias, league, words, text):
                return None, True
            return league, False
    return None, False


def league_for(hint: Optional[str]) -> Optional[str]:
    """The Odds API league key a slip's sport or league words name ("NBA", "England Premier League"), or None
    (also for another league of the same name: "Brazil Serie A", "NBA G League"; see league_of)."""
    return league_of(hint)[0]


def sport_prefixes(hint: Optional[str]) -> Tuple[str, ...]:
    """The league-key prefixes a sport named without its league covers ("soccer" -> ("soccer_",)), or ()."""
    words = _words(_short(hint, MAX_FIELD))
    for sport in sorted(SPORT_PREFIXES, key=len, reverse=True):
        if re.search(rf"\b{sport}\b", words):
            return SPORT_PREFIXES[sport]
    return ()


def estimated_chance(decimal: Decimal, kind: str) -> float:
    """The chance a leg wins when it isn't on the app's boards: (1 / odds) / (1 + the typical margin)."""
    margin = TYPICAL_MARGIN.get(kind, TYPICAL_MARGIN["other"])
    return float((1 / Decimal(decimal)) / (1 + Decimal(str(margin))))


# --- start times ---------------------------------------------------------------------------------

_ZONES = {"et": "America/New_York", "est": "America/New_York", "edt": "America/New_York",
          "ct": "America/Chicago", "cst": "America/Chicago", "cdt": "America/Chicago",
          "mt": "America/Denver", "mst": "America/Denver", "mdt": "America/Denver",
          "pt": "America/Los_Angeles", "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
          "gmt": "UTC", "utc": "UTC", "bst": "Europe/London", "uk": "Europe/London", "cet": "Europe/Paris",
          "cest": "Europe/Paris", "wat": "Africa/Lagos", "sast": "Africa/Johannesburg", "eat": "Africa/Nairobi"}
_MONTHS = {name: number for number, names in enumerate(
    (("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")), start=1) for name in names}
_WEEKDAYS = {name: number for number, names in enumerate(
    (("mon", "monday"), ("tue", "tues", "tuesday"), ("wed", "weds", "wednesday"), ("thu", "thur", "thurs", "thursday"),
     ("fri", "friday"), ("sat", "saturday"), ("sun", "sunday"))) for name in names}
_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_NAMES = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))


def _clock(text: str) -> Optional[Tuple[int, int]]:
    twelve = re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*([ap])\.?\s?m\b\.?", text)
    if twelve:
        hour, minute = int(twelve.group(1)), int(twelve.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None
        return (hour % 12) + (12 if twelve.group(3) == "p" else 0), minute
    plain = re.search(r"(?<![\d/.])(\d{1,2}):(\d{2})(?![\d/])", text)
    if plain and int(plain.group(1)) <= 23 and int(plain.group(2)) <= 59:
        return int(plain.group(1)), int(plain.group(2))
    return None


def _date(text: str, today: datetime.date, us_order: bool) -> Optional[datetime.date]:
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        try:
            return datetime.date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    month = day = year = None
    named = (re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})\.?(?:,?\s+(\d{{4}}))?\b", text)
             or re.search(rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", text))
    if named:
        first, second, year_text = named.groups()
        if first.isdigit():
            day, month = int(first), _MONTHS[second]
        else:
            month, day = _MONTHS[first], int(second)
        year = int(year_text) if year_text else None
    else:
        numeric = re.search(r"(?<![\d.:])(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?(?![\d:])", text)
        if not numeric:
            return None
        a, b = int(numeric.group(1)), int(numeric.group(2))
        if a > 12 or (b <= 12 and not us_order):
            day, month = a, b
        else:
            month, day = a, b
        if numeric.group(3):
            year = int(numeric.group(3)) + (2000 if len(numeric.group(3)) == 2 else 0)
    candidates = [year] if year else [today.year - 1, today.year, today.year + 1]
    best = None
    for option in candidates:
        try:
            date = datetime.date(option, month, day)
        except (ValueError, TypeError):
            continue
        if best is None or abs((date - today).days) < abs((best - today).days):
            best = date
    return best


def parse_start(text: Optional[str], now: datetime.datetime,
                tz: datetime.tzinfo = datetime.timezone.utc) -> Optional[datetime.datetime]:
    """A slip's start time ("Tue 7:30 PM", "26 Oct 19:45", "14/10 19:45", "Today 20:00 ET") as an aware time.

    Without a time zone on the slip it is the person's (``tz``). A weekday is the next one from today
    (yesterday counts, for a game in play); a date without a year is the nearest such date; a date without
    a time is midday. None when no date, weekday or time can be read.
    """
    value = (_short(text, MAX_FIELD) or "").lower()
    if not value:
        return None
    zone_word = re.search(r"\b(" + "|".join(_ZONES) + r")\b\.?\s*$", value)
    zone: datetime.tzinfo = ZoneInfo(_ZONES[zone_word.group(1)]) if zone_word else tz
    local_now = now.astimezone(zone)
    today = local_now.date()
    clock = _clock(value)
    date = _date(value, today, us_order=bool(re.search(r"\b[ap]\.?\s?m\b", value)) or zone_word is not None
                 and _ZONES.get(zone_word.group(1), "").startswith("America"))
    if date is None:
        weekday = re.search(rf"\b({_WEEKDAY_NAMES})\b", value)
        if re.search(r"\btomorrow\b", value):
            date = today + datetime.timedelta(days=1)
        elif re.search(r"\byesterday\b", value):
            date = today - datetime.timedelta(days=1)
        elif re.search(r"\b(?:today|tonight|live|now)\b", value):
            date = today
        elif weekday:
            ahead = (_WEEKDAYS[weekday.group(1)] - today.weekday()) % 7
            date = today + datetime.timedelta(days=ahead if ahead != 6 else -1)
        elif clock:
            date = today
    if date is None:
        return None
    hour, minute = clock if clock else (12, 0)
    return datetime.datetime(date.year, date.month, date.day, hour, minute, tzinfo=zone)


# --- screenshots ---------------------------------------------------------------------------------

def validate_image(data: bytes) -> str:
    """The media type of a screenshot by its first bytes (JPEG, PNG or WebP), or ValueError for anything else.
    The type the app sends is never trusted: these bytes decide."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 16:
        raise ValueError("That screenshot is empty or cut off: send a JPEG, PNG or WebP image.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"That screenshot is too large: up to {MAX_IMAGE_BYTES / 1_000_000:.1f} MB.")
    head = bytes(data[:12])
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Send the screenshot as a JPEG, PNG or WebP image.")


def decode_image(data: str) -> Tuple[bytes, str]:
    """A screenshot sent as base64 (a "data:image/...;base64," prefix allowed): (its bytes, its media type by
    its first bytes). ValueError when it isn't base64, is too large or isn't a JPEG, PNG or WebP."""
    if not isinstance(data, str):
        raise ValueError("Send the screenshot as base64 text.")
    text = data.strip()
    prefix = re.match(r"data:[\w.+/-]{1,64};base64,", text[:100])
    if prefix:
        text = text[prefix.end():]
    if len(text) > (MAX_IMAGE_BYTES * 4) // 3 + 8:
        raise ValueError(f"That screenshot is too large: up to {MAX_IMAGE_BYTES / 1_000_000:.1f} MB.")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("That screenshot isn't valid base64.") from None
    return raw, validate_image(raw)


# --- spotting a slip in chat ---------------------------------------------------------------------

# An odds-like token: American (+150, -110), decimal with 2-3 places (1.85, 1,85, 12.50), fractional (5/2)
# or evens. One decimal place is a line (-4.5, 224.5), not odds, and an amount after a currency isn't odds.
_ODDS_TOKEN = re.compile(
    r"(?<![\w.,/:$£€₦₹₵+\-])"
    r"(?P<odds>[+-]\d{3,5}(?![\d.,/])"
    r"|\d{1,3}\.\d{2,3}(?![\d.,/%])"
    r"|\d{1,3},\d{2}(?![\d.,/%])"
    r"|\d{1,3}/\d{1,3}(?![\d/.,])"
    r"|evs|evens)(?![A-Za-z])", re.I)
_SLIP_WORDS = re.compile(
    r"\b(?:stake|wager|to win|to return|returns?|payout|potential (?:win|winnings|return|returns|payout)|"
    r"possible win|odds|parlay|multi(?:ple)?|accumulator|acca|bet ?slip|round robin|same game|treble|"
    r"\d+[- ]?fold|\d+[- ]?leg)\b", re.I)


def _token_odds(token: str) -> Optional[Decimal]:
    """The odds an odds-like token stands for, or None when it only looks like odds (a date, an amount)."""
    fraction = re.fullmatch(r"(\d{1,3})/(\d{1,3})", token)
    if fraction and int(fraction.group(2)) not in _FRACTION_DENOMINATORS:
        return None
    odds = parse_odds(token)
    return odds if odds is not None and odds >= Decimal("1.01") else None


def odds_tokens(line: str) -> List["re.Match[str]"]:
    """The odds-like tokens in one line of text, in order."""
    return [match for match in _ODDS_TOKEN.finditer(line) if _token_odds(match.group("odds"))]


def looks_like_slip(text: str) -> bool:
    """Whether pasted text looks like a bet slip rather than ordinary chat: two or more odds-like tokens over
    two or more lines, or a slip's words (stake, to win, returns, odds, parlay, multi, accumulator, bet slip)
    with at least one. "Celtics -4.5 tonight?" alone is not a slip."""
    lines = [line for line in clean_text(text).split("\n") if line.strip()]
    counts = [len(odds_tokens(line)) for line in lines]
    tokens = sum(counts)
    if tokens >= 2 and sum(1 for count in counts if count) >= 2:
        return True
    return tokens >= 1 and bool(_SLIP_WORDS.search(clean_text(text)))


# --- the free reader -----------------------------------------------------------------------------

_BOOKMAKERS = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars", "bet365": "bet365",
    "espn bet": "ESPN BET", "fanatics": "Fanatics", "hard rock bet": "Hard Rock Bet", "betrivers": "BetRivers",
    "pointsbet": "PointsBet", "bovada": "Bovada", "betonline": "BetOnline", "mybookie": "MyBookie",
    "sportybet": "SportyBet", "betway": "Betway", "bet9ja": "Bet9ja", "betking": "BetKing", "msport": "MSport",
    "1xbet": "1xBet", "22bet": "22Bet", "betika": "Betika", "sportpesa": "SportPesa", "odibets": "Odibets",
    "hollywoodbets": "Hollywoodbets", "supabets": "Supabets", "paddy power": "Paddy Power",
    "william hill": "William Hill", "sky bet": "Sky Bet", "skybet": "Sky Bet", "ladbrokes": "Ladbrokes",
    "coral": "Coral", "betfred": "Betfred", "unibet": "Unibet", "betvictor": "BetVictor", "betfair": "Betfair",
    "boylesports": "BoyleSports", "betano": "Betano", "stake.com": "Stake", "pinnacle": "Pinnacle",
}
_BOOKMAKER = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(_BOOKMAKERS, key=len, reverse=True))
                        + r")\b", re.I)
_MONEY_LABELS = (
    ("total_stake_text", r"total\s+(?:stake|wager|cost|bet amount|amount staked|bet)"),
    ("stake_text", r"stake\s+per\s+bet|unit\s+stake|stake/bet|per\s+bet|stake|wager|risk|bet\s+amount|"
                   r"amount\s+staked|amount|bet(?=\s*[:=]?\s*(?:" + _CURRENCY_BEFORE + r"))"),
    ("potential_return_text", r"total\s+payout|potential\s+(?:win|winnings|returns?|payout)|possible\s+(?:win|"
                              r"winnings|payout)|est(?:\.|imated)?\s+(?:returns?|payout)|max(?:imum)?\s+(?:win|"
                              r"payout|returns?)|to\s+win|to\s+return|returns?|payout"),
)
_MONEY_LINE = re.compile(
    r"(?P<label>" + "|".join(f"(?P<{name}>{pattern})" for name, pattern in _MONEY_LABELS) + r")"
    r"\s*[:=\-]?\s*(?P<amount>(?:" + _CURRENCY_BEFORE + r")?\s?\d[\d,.]*(?:\s?(?:" + _CURRENCY_AFTER + r"))?)", re.I)
_NOISE = re.compile(
    r"^(?:bet\s*(?:id|ref|reference|no\.?|number|slip|placed|history|receipt|details|code|type)|booking\s*(?:code|id)|"
    r"ticket(?:\s*(?:id|no\.?|number))?|ref(?:erence)?(?:\s*(?:no\.?|number|id))?|receipt|cash\s*out|cashout|"
    r"share|rebet|re-bet|edit|remove|keep|accept|open bets|my bets|settled|unsettled|status|placed|date|"
    r"odds changed|live|in-play|boost(?:ed)?|profit boost|promo|reuse|copy|view|done|ok|close|x(?!\s*@)|total odds|"
    r"odds|potential bonus|max(?:imum)? bonus|bonus|tax|excise|withholding|net|balance|sportsbook|"
    r"account|deposit|login|sign in|cash out available|bet history|all|won|lost|pending|open|running|void|"
    # A ticket's price lines: boosted, combined or overall odds are the whole bet's, never a leg
    r"(?:price|odds|super|power|profit|win|acca|parlay)\s*boost(?:ed)?|superboost|power\s*price|enhanced|"
    r"was\b.*\bnow|combined|overall(?:\s+(?:odds|price))?|total\s+price|potential\s+odds|"
    r"(?:parlay|acca|accumulator|treble|double|multi|multiple|sgp|bet\s*builder)\s+(?:price|odds)|"
    r"selections?|your\s+(?:bets?|picks?|selections?)|betslip)\b",
    re.I)
_TYPE_CORE = {"parlay", "multi", "multiple", "accumulator", "acca", "treble", "double", "trixie", "patent", "yankee",
              "lucky", "canadian", "heinz", "goliath", "single", "singles", "robin", "system", "sgp", "teaser", "fold",
              "folds", "combo", "builder"}
_TYPE_OTHER = {"round", "super", "same", "game", "bet", "bets", "leg", "legs", "pick", "picks", "team", "teams",
               "selection", "selections", "by", "s", "x", "of", "your", "the", "way", "ticket", "slip", "odds",
               "total", "boost", "boosted", "sgp+", "plus", "straight"}
_EVENT_SPLIT = re.compile(r"\s+(?:vs\.?|v\.?|@)\s+", re.I)
_DASH_SPLIT = re.compile(r"\s+-\s+")  # "Chelsea - Man Utd" (Bet9ja, 1xBet, Unibet): the home side first
# A slip naming a same-game parlay or a bet builder: one price for all its legs
_SAME_GAME = re.compile(r"\b(?:sgp\+?|sgm|same[- ]game(?:\s+(?:parlay|multi))?|bet\s*builder|betbuilder|"
                        r"request\s*a\s*bet)\b", re.I)
# A market named before the pick ("1X2: Home", "1X2 - 1", "Over/Under: Over 2.5", "Spread: Celtics -3.5")
_LEAD_MARKET = re.compile(r"^(?P<label>[^:@]{1,30}?)\s*(?::|\s-\s)\s*(?P<pick>\S.*)$")
_ONE_X_TWO = {"1", "x", "2"}
_EVENT_AND_PICK = re.compile(r"^(?P<event>[^:|]+?\s(?:vs\.?|v\.?|@)\s[^:|]+?)\s*(?:\s-\s|:\s*|\|\s*|/\s+)(?P<pick>.+)$",
                             re.I)
_WEEKDAY_OR_MONTH = rf"(?:{_WEEKDAY_NAMES}|{_MONTH_NAMES}|today|tonight|tomorrow|live|now)"
_TIME_LINE = re.compile(rf"^(?:(?:{_WEEKDAY_OR_MONTH}|\d{{1,4}}(?:st|nd|rd|th)?|[ap]\.?m\.?|[:/.,\-]|\s|"
                        rf"(?:{'|'.join(_ZONES)})\b)+)$", re.I)
_LINE_ONLY = re.compile(r"^(?:[+-]\d{1,3}(?:\.\d{1,2})?|\d{1,4}\.\d|pk|pick'?em|[ou]\s?\d{1,4}(?:\.\d{1,2})?|"
                        r"(?:over|under)\s+\d{1,4}(?:\.\d{1,2})?)$", re.I)
_BULLET = re.compile(r"^(?:\(?\d{1,2}[.)]\s+|leg\s*\d{1,2}\s*[:.)-]?\s*|[•*·>\-]\s+)", re.I)
_PICK_LABEL = re.compile(r"^(?:pick|your pick|selection|bet|tip|prediction|market)\s*[:\-]\s*", re.I)
_ODDS_AT = re.compile(r"(?:@|\bat\b|\bodds\b:?)\s*$", re.I)
_TOTAL_WORDS = {"total", "match", "game", "goals", "points", "runs", "total goals", "total points", "match goals",
                "game total", "total runs", "full time", "ft", "totals", "match total", "over/under", "o/u"}
_HOME = {"home", "1", "home win", "home team", "w1"}
_AWAY = {"away", "2", "away win", "away team", "w2"}
HOME_WORDS, AWAY_WORDS = frozenset(_HOME), frozenset(_AWAY)  # a pick naming a side, not a team ("1", "Away")
_DRAW = {"draw", "the draw", "tie", "x", "draw (x)"}


def _is_type_line(line: str) -> bool:
    """A line naming the kind of bet ("3 Leg Parlay +596", "4 Fold Accumulator", "Singles", "Multi Bet (3)")."""
    text = line.lower()
    for match in reversed(odds_tokens(line)):
        text = text[:match.start()] + " " + text[match.end():]
    words = re.findall(r"[a-z+]+", text)
    core = [word for word in words if word in _TYPE_CORE]
    return bool(core) and all(word in _TYPE_CORE or word in _TYPE_OTHER for word in words)


def _round_robin_k(text: str) -> Optional[int]:
    """A round robin's legs per bet: "By 2s" first ("4 Team Round Robin (By 2s)" is by 2s, not 4), then
    "2's" or "2-Team Parlays", then "3 Picks" (which may be the number of legs: basic_parse drops a k that
    isn't below the legs it read)."""
    for pattern in (r"\bby\s*(\d)", r"(?<![\d.])(\d)\s*'?s\b",
                    r"(?<![\d.])(\d)\s*-?\s*(?:picks?|teams?|legs?)\s+(?:parlays?|bets?|multis?|combos?)\b",
                    r"(?<![\d.])(\d)\s*-?\s*(?:picks?|teams?|legs?)\b"):
        match = re.search(pattern, text, re.I)
        if match:
            k = int(match.group(1))
            return k if 2 <= k <= 7 else None
    return None


def _market_label(line: str) -> Optional[str]:
    """The market a whole line names ("Spread", "Full Time Result", "Goals Over/Under", "1X2"), or None."""
    text = " ".join(re.sub(r"[^a-z0-9/+ ]", " ", line.lower()).split())
    candidates = [text, re.sub(r"\s*\d+(?:\.\d+)?$", "", text), " ".join(re.sub(r"\(.*?\)", " ", line.lower()).split())]
    inner = re.search(r"\(([^)]*)\)", line)
    if inner:
        candidates.append(" ".join(re.sub(r"[^a-z0-9/+ ]", " ", inner.group(1).lower()).split()))
    for candidate in candidates:
        candidate = re.sub(r"[^a-z0-9/+ ]", " ", candidate).strip()
        candidate = " ".join(candidate.split())
        if candidate in _MARKET_LOOKUP and candidate not in ("other", "player", "line", "prop", "props"):
            return _MARKET_LOOKUP[candidate]
    return None


def _split_event(text: str) -> Optional[Tuple[str, str]]:
    """The two sides of a game as written ("Knicks @ Celtics" -> ("Knicks", "Celtics")), or None."""
    parts = _EVENT_SPLIT.split(text)
    if len(parts) != 2 or not all(re.search(r"[A-Za-z]{2}", part) for part in parts):
        return None
    if any(len(part) > 45 or len(part.split()) > 6 for part in parts):
        return None
    return parts[0].strip(" -:"), parts[1].strip(" -:")


def _dash_event(text: str, strict: bool = False) -> Optional[Tuple[str, str]]:
    """The two sides of a game written "Chelsea - Man Utd" (one spaced hyphen or dash), or None. ``strict``
    (a line of a pasted slip, not a game the reader named) also turns away halves that hold digits or name a
    market, a pick, a league or a sport, so "Arsenal - Match Result" and "England - Premier League" aren't
    games."""
    parts = _DASH_SPLIT.split(text.strip())
    if len(parts) != 2 or not all(re.search(r"[A-Za-z]{2}", part) for part in parts):
        return None
    if any(len(part) > 45 or len(part.split()) > 6 for part in parts):
        return None
    first, second = parts[0].strip(" -:"), parts[1].strip(" -:")
    if strict:
        for part in (first, second):
            lowered = part.lower()
            if (re.search(r"\d", part) or _market_label(part) or league_of(part) != (None, False)
                    or sport_prefixes(part) or _COUNTRY.fullmatch(_words(part))
                    or lowered in _HOME | _AWAY | _DRAW | {"over", "under", "yes", "no", "gg", "ng"}):
                return None
    return first, second


def event_sides(event_text: Optional[str]) -> Optional[Tuple[str, str, bool]]:
    """(first team, second team, whether the second is at home for sure): "Away @ Home" says so; "A v B",
    "A vs B" and "A - B" usually list the home side first, but not always."""
    if not event_text:
        return None
    sides = _split_event(event_text) or _dash_event(event_text)
    if sides is None:
        return None
    return sides[0], sides[1], bool(re.search(r"\s@\s", event_text))


def _money_fields(line: str) -> Dict[str, str]:
    """The money a line states, by field ("Wager: $10.00 To Win: $55.60" -> stake_text and
    potential_return_text), when the line starts with such a label; {} otherwise ("Arsenal to win 2.10" is a
    leg, not a return)."""
    found: Dict[str, str] = {}
    for match in _MONEY_LINE.finditer(line):
        if not found and match.start() > 0:
            return {}
        name = next(key for key, _ in _MONEY_LABELS if match.group(key))
        amount = match.group("amount").strip()
        if parse_amount(amount)[0] is not None:
            found.setdefault(name, amount)
    return found


def _is_time_line(line: str) -> bool:
    """A line that is only a date and/or time ("Tue 7:30 PM", "26 Oct 19:45", "14/10 19:45", "Today 20:00")."""
    if len(line) > 40 or not _TIME_LINE.fullmatch(line):
        return False
    return bool(re.search(r"\d:\d{2}|\d\s*[ap]\.?m\b|\d{1,2}/\d{1,2}", line, re.I)
                or re.search(rf"\b{_WEEKDAY_OR_MONTH}\b", line, re.I))


@dataclass
class _Pending:
    """What has been read since the last leg closed, waiting for its odds (block layouts)."""

    pick: Optional[str] = None
    event: Optional[str] = None
    label: Optional[str] = None
    start: Optional[str] = None
    sport: Optional[str] = None
    lines: List[str] = field(default_factory=list)
    picks: List[str] = field(default_factory=list)  # every pick line since the last leg, in order

    def empty(self) -> bool:
        return not (self.pick or self.event or self.label or self.start or self.sport)


@dataclass
class _Draft:
    """A leg read so far, before it becomes an ExtractedLeg."""

    pick: str
    odds_text: Optional[str]  # None for a leg of a same-game parlay, priced only as a whole
    event: Optional[str] = None
    label: Optional[str] = None
    start: Optional[str] = None
    sport: Optional[str] = None
    lines: List[str] = field(default_factory=list)
    tail: Optional[str] = None  # market words after the odds ("-110 Spread")


def _strip_market_words(text: str) -> Tuple[str, Optional[str]]:
    """A pick without the market words at its end or in brackets ("Arsenal (Match Result)", "Suns ML",
    "Celtics -3.5 Spread"): (the pick, the market those words name)."""
    market = None
    bracket = re.search(r"\s*\(([^)]*)\)\s*$", text)
    if bracket:
        named = _market_label(bracket.group(1))
        if named:
            market, text = named, text[:bracket.start()]
    words = text.split()
    for size in (4, 3, 2, 1):
        if len(words) > size:
            tail = " ".join(words[-size:])
            named = _market_label(tail)
            if named and not re.search(r"\d", tail) and tail.lower() not in ("over", "under", "win"):
                market = market or named
                words = words[:-size]
                break
    return " ".join(words).strip(" -:,"), market


def _read_pick(pick: str, label_market: Optional[str], sides: Optional[Tuple[str, str, bool]]
               ) -> Dict[str, Any]:
    """What a pick's words say: selection, market, line, prop stat and over/under."""
    if _market_label(pick) == "other":  # "Both Teams To Score", "Double Chance": the market is the pick
        return {"selection": pick.strip(), "market": "other", "line": None, "prop_stat": None, "over_under": None}
    text, word_market = _strip_market_words(pick)
    market_hint = word_market or label_market
    lowered = text.lower().strip()
    # Player ladders: "LeBron James 25+ Points" is over 24.5 (read before "Points" is taken as market words)
    ladder = re.search(r"^(?P<name>.+?)\s+(?P<count>\d{1,3})\+\s*(?P<stat>.+)$", pick.strip()) or \
        re.search(r"^(?P<name>.+?)\s+(?P<count>\d{1,3})\+\s*(?P<stat>.+)$", text)
    if ladder and prop_stat_of(ladder.group("stat")):
        return {"selection": ladder.group("name").strip(), "market": "player_prop",
                "line": int(ladder.group("count")) - 0.5, "prop_stat": prop_stat_of(ladder.group("stat")),
                "over_under": "over"}
    total = re.search(r"(?:^|\s)(?P<side>over|under|o|u)\s?(?P<line>\d{1,4}(?:\.\d{1,2})?)\b(?P<tail>.*)$", text, re.I)
    if total:
        name = text[:total.start()].strip(" -:,")
        side = "over" if total.group("side").lower().startswith("o") else "under"
        line = float(total.group("line"))
        stat = prop_stat_of(total.group("tail")) or (prop_stat_of(name) if name else None)
        if market_hint == "player_prop" and not stat:
            stat = prop_stat_of(pick)
        generic = not name or name.lower() in _TOTAL_WORDS or _market_label(name) in ("total", "moneyline")
        if not generic and (stat or market_hint == "player_prop"):
            player = re.sub(r"\b(?:" + "|".join(p.pattern for _, p in _STAT_WORDS) + r")", " ", name, flags=re.I)
            return {"selection": " ".join(player.split()) or name, "market": "player_prop", "line": line,
                    "prop_stat": stat, "over_under": side}
        if not generic:  # a team's own total ("Lakers Over 112.5"): not the game's total
            return {"selection": text, "market": "other", "line": line, "prop_stat": None, "over_under": side}
        return {"selection": side.capitalize(), "market": "total", "line": line, "prop_stat": None,
                "over_under": side}
    if lowered in _DRAW or re.fullmatch(r"(?:the\s+)?draw(?:\s*\(x\))?", lowered):
        return {"selection": "Draw", "market": "moneyline", "line": None, "prop_stat": None, "over_under": None}
    if lowered in _HOME or lowered in _AWAY:
        home = lowered in _HOME
        name = None
        if sides:
            first, second, second_home = sides
            # "A @ B": B is at home; "A v B": A is
            name = (second if second_home else first) if home else (first if second_home else second)
        return {"selection": name or ("Home" if home else "Away"), "market": "moneyline", "line": None,
                "prop_stat": None, "over_under": None}
    if lowered in ("1x", "x2", "12"):
        return {"selection": text, "market": "other", "line": None, "prop_stat": None, "over_under": None}
    spread = re.search(r"\s\(?(?P<line>[+-]\d{1,3}(?:\.\d{1,2})?)\)?$|\s(?P<pk>pk|pick'?em)$", text, re.I)
    if spread and re.search(r"[A-Za-z]", text[:spread.start()]) and market_hint not in ("total", "player_prop"):
        line = 0.0 if spread.group("pk") else float(spread.group("line"))
        return {"selection": text[:spread.start()].strip(" -:,"), "market": market_hint if market_hint == "other"
                else "spread", "line": line, "prop_stat": None, "over_under": None}
    to_win = re.search(r"\s+(?:to\s+win|win|ml|moneyline|money\s+line)$", text, re.I)
    if to_win:
        return {"selection": text[:to_win.start()].strip(" -:,"), "market": "moneyline", "line": None,
                "prop_stat": None, "over_under": None}
    plain_name = bool(re.fullmatch(r"[A-Za-z][A-Za-z.'&\- ]{1,60}", text)) and len(text.split()) <= 5
    if market_hint:
        market = market_hint
    else:
        market = "moneyline" if plain_name else "other"
    if market in ("spread", "total") and not re.search(r"\d", text):
        market = "other" if not plain_name else market
    return {"selection": text, "market": market, "line": None, "prop_stat": prop_stat_of(text)
            if market == "player_prop" else None, "over_under": None}


def _extracted(draft: _Draft) -> Optional[ExtractedLeg]:
    """The ExtractedLeg a draft stands for (None for a line that turned out to be no bet)."""
    pick = _PICK_LABEL.sub("", _BULLET.sub("", draft.pick.strip())).strip(" -:")
    event = draft.event
    split = _EVENT_AND_PICK.match(pick)
    if split:
        event = event or split.group("event").strip()
        pick = split.group("pick").strip(" -:")
    elif _split_event(pick) and not event:
        # "Knicks @ Celtics Over 224.5": the game, then the pick from the over/under on
        found = re.search(r"\s(?:over|under|o|u)\s?\d", pick, re.I)
        if found:
            event, pick = pick[:found.start()].strip(), pick[found.start():].strip()
    label_market = draft.label and _market_label(draft.label)
    if draft.tail:
        label_market = _market_label(draft.tail) or label_market
    lead = _LEAD_MARKET.match(pick)
    named = _market_label(lead.group("label")) if lead else None
    if lead and (named in ("moneyline", "spread", "total") or lead.group("pick").strip().lower() in _ONE_X_TWO):
        # "1X2: Home", "1X2 - 1", "Over/Under: Over 2.5": the market, then the pick (other markets keep their
        # words, "GG/NG - GG", so the pick still says what it is)
        label_market, pick = named or label_market, lead.group("pick").strip(" -:")
    if not re.search(r"[A-Za-z]", pick) and not (event and pick.lower() in _HOME | _AWAY):
        return None
    read = _read_pick(pick, label_market, event_sides(event))
    if read["market"] == "player_prop" and not read["prop_stat"] and draft.label:
        read["prop_stat"] = prop_stat_of(draft.label)
    hint = next((text for text in [*draft.lines, draft.sport or ""] if league_for(text)), None) or draft.sport
    raw = " · ".join(dict.fromkeys(line for line in draft.lines if line))
    return ExtractedLeg(raw=raw or pick, sport_hint=hint, event_text=event, selection=read["selection"],
                        market=read["market"], line=read["line"], prop_stat=read["prop_stat"],
                        over_under=read["over_under"], odds_text=draft.odds_text, start_text=draft.start)


def _leg_line(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """A line holding a whole leg: (the pick's text, its odds as written, market words after the odds), or
    None. The odds are the last odds-like token after "@" or, failing that, at the end of the line (where
    only market words may follow them: "-110 Spread")."""
    tokens = odds_tokens(line)
    if not tokens:
        return None
    after_at = [match for match in tokens if re.search(r"@\s*$", line[:match.start()])]
    chosen = after_at[-1] if after_at else tokens[-1]
    tail = line[chosen.end():].strip(" )]") or None
    if not after_at and tail and (re.search(r"\d", tail) or len(tail.split()) > 3 or not _market_label(tail)):
        return None
    before = _ODDS_AT.sub("", line[:chosen.start()].rstrip(" ([")).strip(" -:,|")
    if not re.search(r"[A-Za-z]{2}", before):
        lead = _LEAD_MARKET.match(before)
        pick = (lead.group("pick") if lead and _market_label(lead.group("label")) else before).strip().lower()
        if not (after_at and pick in _ONE_X_TWO):
            return None
    return before, chosen.group("odds"), tail


def _attach(target: Optional[_Draft], pending: _Pending, name: str, value: str, line: str) -> None:
    """A detail line (the game, a market, a time, a sport) belongs to the leg just read when nothing is
    waiting for the next one and that leg lacks it (layouts that list the details after the pick), else to
    the next leg (layouts that list them before)."""
    if target is not None and pending.empty() and getattr(target, name) is None:
        setattr(target, name, value)
        target.lines.append(line)
        return
    current = getattr(pending, name)
    if name == "sport" and current:
        value = f"{current} {value}"
    setattr(pending, name, value)
    pending.lines.append(line)


def basic_parse(text: str) -> Optional[Extraction]:
    """The free, code-only reader. One leg per line ("<selection> [market words/line] <odds>", the odds at the
    end or after "@"; a 1X2 slip's "1X2 - 1 @ 1.85" under "Chelsea - Man Utd"), or a bookmaker's block layout
    (the game, market and pick on their own lines and the odds on the next); a "Stake"/"Wager"/"Bet" line; a
    "Returns"/"To win"/"Payout" line. Boosted, combined and total price lines are skipped, and a same-game
    parlay's one price after its picks (two or more on a slip naming one, else three) is never one leg's
    odds: its legs are read without odds. Anything it can't place is skipped. None if no leg is read."""
    lines = [" ".join(line.split()) for line in clean_text(text).split("\n")]
    drafts: List[_Draft] = []
    pending = _Pending()
    money: Dict[str, str] = {}
    bet_type, k, system = "unknown", None, None
    book = _BOOKMAKER.search(clean_text(text))
    same_game = bool(_SAME_GAME.search(clean_text(text)))
    combined: Optional[str] = None  # one price for picks with no odds of their own (a same-game parlay)

    def close(draft: _Draft) -> None:
        nonlocal pending
        draft.event = draft.event or pending.event
        draft.label = draft.label or pending.label
        draft.start = draft.start or pending.start
        draft.sport = draft.sport or pending.sport
        draft.lines = pending.lines + draft.lines
        drafts.append(draft)
        pending = _Pending()

    def close_combined(odds_text: str) -> None:
        """Picks with no odds and then one price: a same-game parlay or bet builder, priced only as a whole
        (that price is never one leg's), so each pick is a leg without odds."""
        nonlocal pending, combined
        combined = combined or odds_text
        for pick in pending.picks:
            drafts.append(_Draft(pick=pick, odds_text=None, event=pending.event, start=pending.start,
                                 sport=pending.sport, lines=[pick]))
        pending = _Pending()

    for line in lines:
        if not line:
            continue
        last = drafts[-1] if drafts else None
        amounts = _money_fields(line)
        if amounts:
            for name, amount in amounts.items():
                money.setdefault(name, amount)
            continue
        if _is_type_line(line):
            if bet_type == "unknown":
                bet_type = bet_type_of(line)
                k = _round_robin_k(line) if bet_type == "round_robin" else None
                system = system_key(line) if bet_type == "system" else None
            continue
        if _NOISE.match(line) or (_BOOKMAKER.search(line) and len(line.split()) <= 4 and not odds_tokens(line)):
            continue
        leg = _leg_line(line)
        if leg is not None:
            pick, odds_text, tail = leg
            if pending.pick and not re.search(r"[A-Za-z]", re.sub(r"(?i)spread|total|moneyline|ml|over|under|"
                                                                  r"handicap|line", "", pick)):
                pick = f"{pending.pick} {pick}"  # "Boston Celtics" then "Spread -3.5 -110"
            elif pending.pick:  # a stray line before a whole-line leg isn't part of it
                pending.lines = [kept for kept in pending.lines if kept not in pending.pick]
            close(_Draft(pick=pick, odds_text=odds_text, tail=tail, lines=[line]))
            continue
        only = re.fullmatch(r"@?\s*(" + _ODDS_TOKEN.pattern + r")\s*", line, re.I)
        if only and _token_odds(only.group("odds")):
            if len(pending.picks) >= (2 if same_game else 3):
                close_combined(only.group("odds"))
            elif pending.pick:
                close(_Draft(pick=pending.pick, odds_text=only.group("odds"), lines=[line]))
            continue  # odds with no pick before them: a ticket's combined odds, or a boosted price
        if ((_split_event(line) or _dash_event(line, strict=True)) and not odds_tokens(line)
                and not _EVENT_AND_PICK.match(line)):
            _attach(last, pending, "event", line, line)
            continue
        if _is_time_line(line):
            _attach(last, pending, "start", line, line)
            continue
        if _market_label(line):
            if same_game and pending.picks:  # "Both Teams To Score" in a bet builder is one of its picks
                pending.picks.append(line)
            _attach(last, pending, "label", line, line)
            continue
        if (league_for(line) or sport_prefixes(line)) and len(line.split()) <= 6 and not _split_event(line):
            _attach(last, pending, "sport", line, line)
            continue
        if _LINE_ONLY.match(line) and pending.pick:
            pending.pick = f"{pending.pick} {line}"
            if pending.picks:
                pending.picks[-1] = pending.pick
            pending.lines.append(line)
            continue
        if re.search(r"[A-Za-z]{2}", line) and len(line) <= MAX_FIELD:
            if pending.pick and pending.lines and pending.lines[-1] == pending.pick:
                pending.lines.pop()
            pending.pick = line
            pending.picks.append(line)
            pending.lines.append(line)

    legs = [leg for leg in (_extracted(draft) for draft in drafts[:MAX_LEGS + 1]) if leg is not None]
    if not legs:
        return None
    if bet_type == "unknown" and len(legs) == 1:
        bet_type = "single"
    if bet_type == "unknown" and combined:
        bet_type = "multi"
    if k is not None and k >= len(legs):  # "4 Pick Round Robin" names its legs, not its bets' size
        k = None
    notes = (f"One price for the whole bet ({combined}): a same-game parlay or bet builder, so its legs have no "
             "odds of their own." if combined else None)
    currency = None
    for name in ("stake_text", "total_stake_text", "potential_return_text"):
        currency = currency or parse_amount(money.get(name))[1]
    return Extraction(bookmaker=_BOOKMAKERS[book.group(1).lower()] if book else None, bet_type=bet_type, k=k,
                      system=system, stake_text=money.get("stake_text"),
                      total_stake_text=money.get("total_stake_text"), currency=currency,
                      potential_return_text=money.get("potential_return_text"), legs=legs, notes=notes)


def odds_in_text(odds_text: Optional[str], text: Optional[str]) -> bool:
    """Whether odds as read appear in the pasted text as a token of their own (so "1.8" isn't found in
    "1.85"), ignoring case, spaces and the kind of minus sign."""
    if not odds_text or not text:
        return False
    wanted = clean_text(odds_text, MAX_SHORT).lower().replace(" ", "").lstrip("@")
    if not wanted:
        return False
    haystack = re.sub(r"(?<=[+\-/])\s+|\s+(?=/)", "", clean_text(text).lower())
    return re.search(rf"(?<![\d.,]){re.escape(wanted)}(?![\d])", haystack) is not None
