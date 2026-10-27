"""The app's per-person features: tickets, saved bets, rollover plans, player pages and settings.

DashboardService knows games (the odds feed, ESPN and the NBA model). The Lab
sits on top of it for everything that belongs to one person, who has one id
across the Telegram Mini App, the bot and a browser ("tg:<id>" or "web:<id>"):

- Tickets are always priced here, from the cached board. The app sends only
  which picks were chosen ({league, id, market, side}); every price and chance
  comes from service.games() and engine/tickets.py does the maths. A saved bet
  keeps the exact quote ("1.91") it was saved at, and the grader settles it
  from ESPN's final scores: bets.leg_outcome for each leg (a draw loses a
  team's three-way soccer moneyline), tickets.grade for the ticket. So only
  leagues ESPN scores can be saved, and a pick whose line moved is saved
  only once the person has seen the new line.
- "Analyse" is worked out in code: the priced ticket, facts about each leg and
  alternatives (drop the weakest leg, split into singles, a small round robin).
  Claude only writes words around those numbers, and only when an assistant
  is set up; the analysis is complete without it.
- Rollover plans go one day at a time. Each day's stake follows the chosen
  strategy from the money the plan actually has, nothing starts by itself, and
  a run that fails ends there with a suggestion to take a break.
- Responsible-gambling gates are code, never the model: nothing betting-related
  before the person confirms their age, nothing during a break they chose, and
  no stake past what their weekly budget has left. A person's money-moving
  writes (saving a bet, starting a plan, playing or stopping a day) run one
  at a time, so a double tap can't pass a check twice.
- The dashboard password is the owner's alone, and each browser on it makes up
  its own X-Player id, so for the gates every password browser is one person:
  a break, the 24-hour pause on picks after a warning sign, the weekly budget
  (over everything they all staked) and the break after a failed plan live in
  one shared row (Gates), and the plan rules (one at a time, a stopped plan's
  day still in play, repeated failures) count every password browser's plans,
  so a new X-Player id can't shed any of them. Age is confirmed in each
  browser, and slips, plans, chats and calls stay each browser's own.

- A pasted bet slip (engine/pasted.py reads it, or the AI does) is matched
  here to the games and prices the app knows, leg by leg, and priced at the
  person's own odds: a matched leg takes the board's margin-free chance (and
  the model's), a leg on another line, in another market or not on any board
  takes the person's price less a typical margin, and says so. The read is
  checked against the slip (the bookmaker's stated return, odds that couldn't
  be read, odds not in the pasted text) and the person confirms a doubtful
  one. Only what was read is kept, in the person's own rows; never a picture.

Odds quota is precious (about 500 requests a month, 3 per league fetch), so
only the leagues a person asks for are loaded, leagues out of season are
never fetched, and every board comes from the shared 2-hour cache. A pasted
slip may load at most three leagues not already cached, never below the odds
service's reserve.

Money is Decimal in whole cents, rounded once; chances are floats.
"""

from __future__ import annotations

import datetime
import difflib
import functools
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, available_timezones

from pydantic import ValidationError

from ..data.espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES, normalize_team
from ..data.espn_players import PlayerData, hit_rates
from ..data.leagues import POPULAR_LEAGUES
from ..data.odds_service import is_price_book
from ..data.store import MAX_CENTS, Store
from ..engine import bets, calls, pasted, rollover, target, tickets
from ..engine.odds_math import decimal_to_american, format_american, remove_vig
from ..engine.tickets import TicketLeg, chance_label, one_in
from .service import AI_LEAGUES, GAMES_TTL, CallRejected, DashboardService, NotFound

log = logging.getLogger(__name__)

CURRENCY = "$"
CENTS = Decimal("0.01")
DEFAULT_LEAGUES = ("basketball_nba",)
MAX_LEAGUES = 8
GAME_MARKETS = ("h2h", "spreads", "totals")
SIDES = ("home", "away", "draw", "over", "under")
LEAGUE_KEY = re.compile(r"[a-z0-9_]{2,64}")
EVENT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_STAKE = Decimal("100000")
MAX_TARGET = Decimal("10000000")
MAX_BUDGET = Decimal("1000000")
DAILY_AI_CENTS = 100  # AI_CHAT_DAILY_CENTS' default: a dollar of AI per person per day
COOLOFF_CHOICES = (24, 72, 168, 720, 2160)  # a day, 3 days, a week, 30 days, 90 days
PLAN_MIN_START = Decimal("1")
PLAN_RESTART_HOURS = 24  # after a failed run, a break before a new plan (the blueprint's "cool-off before a restart")
REPEATED_FAILURES = 2  # failed plans in a week that are logged for the owner's review
WEEK = 7 * 24 * 3600
GRADE_EVERY = 60  # seconds between grading runs started by page loads (the grader thread runs anyway)
# Per-person write locks, shared out by a hash of the id: memory stays bounded however many ids appear
WRITE_LOCKS = 64
SAME_GAME_WINDOW = 6 * 3600  # an odds-feed game and ESPN's "next game" this close are the same game
AGE_MESSAGE = "Confirm you're 18+ (21+ in the US) in Settings"
NO_CHAT = "AI chat isn't set up: add ANTHROPIC_API_KEY on the server"
NO_STORE = "This needs the server's state folder (STATE_DIRECTORY)."
PAUSE_SETTING = "picks_paused_until"  # users.settings: picks stay off until then after a warning sign in the chat
SPORTS_CHOSEN_SETTING = "sports_chosen"  # users.settings: favourite_sports were saved by the person
PLAN_FAILED_SETTING = "plan_failed_at"  # users.settings: when a plan's run last failed (the break before a new one)
# Times in the settings of the row holding the gates (see gates_id): each only ever moves later
GATE_SETTINGS = (PAUSE_SETTING, PLAN_FAILED_SETTING)
TIMEZONE_SETTING = "timezone"  # users.settings: the person's IANA time zone, sent by the app (the late-night note)
# The users row holding the break, picks pause, weekly budget and the break after a failed plan every password
# browser shares (see gates_id)
SHARED_WEB = "*web"
WEB_PREFIX = "web:"
GATE_FIELDS = ("cooloff_until", "weekly_budget_cents", "pending_budget_cents", "budget_raise_at")
BREAK_SUGGESTION = (
    "Most rollover runs end early, and starting another straight away is how losses grow. Take a break: "
    f"new plans open again in {PLAN_RESTART_HOURS} hours, and Settings has longer breaks if you want one."
)
# Player-prop stats the model projects with a measured spread (engine/calls.py); other combinations
# (points + rebounds, ...) get hit rates but no AI chance rather than a made-up one
PROP_STATS = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m", "stl": "stl", "blk": "blk",
              "tov": "tov", "pts+reb+ast": "pra"}
PROJECTED = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra")
TICKET_TITLES = {"single": "Singles", "multi": "Multi", "round_robin": "Round robin", "system": "System bet"}


# --- small helpers --------------------------------------------------------------------------------

def _amount(value: Any, what: str, most: Decimal, least: Decimal = CENTS) -> Decimal:
    """Money from a request as exact Decimal dollars: whole cents from ``least`` to ``most``."""
    if isinstance(value, bool):
        raise CallRejected(f"The {what} must be a number")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise CallRejected(f"The {what} must be a number") from None
    if not amount.is_finite() or not least <= amount <= most:
        raise CallRejected(f"The {what} must be from {_say_dollars(least)} to {_say_dollars(most)}")
    if amount != amount.quantize(CENTS):
        raise CallRejected(f"Enter the {what} in whole cents, e.g. 10.50")
    return amount


def _cents(value: Any) -> int:
    """Dollars (Decimal, a "10.00" string or a number) as whole cents, half up."""
    amount = value if isinstance(value, Decimal) else Decimal(str(value))
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _dollars(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(CENTS)


def _money(cents: Optional[int]) -> Optional[str]:
    """Cents as the API's money strings ("10.00")."""
    return None if cents is None else str(_dollars(cents))


def _say_dollars(amount: Decimal, currency: Optional[str] = CURRENCY) -> str:
    """$10, $1,000 or $10.50 for messages; another currency as a pasted slip wrote it (£10, NGN 1,000, or a
    bare 10 when the slip named none)."""
    amount = amount.quantize(CENTS)
    number = f"{int(amount):,}" if amount == amount.to_integral_value() else f"{amount:,.2f}"
    symbol = currency or ""
    if len(symbol) > 1 and symbol[-1].isalpha():  # a code or letters: "NGN 1,000", "KSh 200"
        return f"{symbol} {number}"
    return f"{symbol}{number}"


def _say(cents: int) -> str:
    return _say_dollars(_dollars(cents))


def _say_signed(value: Optional[str], currency: Optional[str] = CURRENCY) -> str:
    """An expected result like "-$1.20" or "+$0.40" from a money string."""
    amount = Decimal(value)
    sign = "-" if amount < 0 else "+" if amount > 0 else ""
    return f"{sign}{_say_dollars(abs(amount), currency)}"


def _floor_cents(amount: Decimal) -> Decimal:
    return amount.quantize(CENTS, rounding=ROUND_DOWN)


def _pct(p: float) -> str:
    """62%, 4.5%, 0.31%: whole percents from 10% up, more digits below."""
    pct = p * 100
    if pct >= 10 or pct == 0:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%"
    return f"{pct:.2g}%"


def _chance_words(p: float) -> str:
    """"1 in 6 (16%)" for a long shot, "62%" for something likely."""
    if p >= 0.5:
        return _pct(p)
    return f"1 in {one_in(p):,} ({_pct(p)})" if one_in(p) else "no chance"


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(text: Any) -> Optional[datetime.datetime]:
    try:
        moment = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def _odds(price: Any) -> Optional[Dict[str, Any]]:
    """Decimal odds as the app shows them: 2 decimals and the American price."""
    if price is None:
        return None
    value = Decimal(str(price))
    if value <= 1:
        return None
    return {"decimal": float(value.quantize(CENTS, rounding=ROUND_HALF_UP)),
            "american": format_american(decimal_to_american(float(value)))}


def _leg_key(league: str, event_id: str, market: str, side: str, point: Optional[float]) -> str:
    """The pricing engine's stable id for a pick, e.g. "basketball_nba:<event>:spreads:home:-3.5"."""
    key = f"{league}:{event_id}:{market}:{side}"
    return key if point is None else f"{key}:{point:g}"


def _ref_of(leg: Mapping[str, Any]) -> Dict[str, Any]:
    """The pick a priced or saved leg came from, as the app sends it back ({league, id, market, side, point})."""
    return {"league": leg["league"], "id": str(leg["event_id"]), "market": leg["market"], "side": leg["side"],
            "point": leg.get("point")}


def _ref(raw: Any) -> Dict[str, Any]:
    """A pick from a request (or from the AI's tools), checked. Prices and chances in it are ignored."""
    if not isinstance(raw, Mapping):
        raise CallRejected("Each pick needs league, id, market and side")
    league, event_id = str(raw.get("league") or ""), str(raw.get("id") or raw.get("event_id") or "")
    market, side = raw.get("market"), raw.get("side")
    if not LEAGUE_KEY.fullmatch(league) or not EVENT_ID.fullmatch(event_id):
        raise CallRejected("Each pick needs a league and a game id")
    if market not in GAME_MARKETS or side not in SIDES:
        raise CallRejected("Picks are moneyline (h2h), spread or total, on home, away, draw, over or under")
    point = raw.get("point")
    try:
        point = None if point is None else float(point)
    except (TypeError, ValueError):
        point = None
    if point is not None and not math.isfinite(point):
        point = None
    return {"league": league, "id": event_id, "market": market, "side": side, "point": point}


def _same_team(a: Optional[str], b: Optional[str]) -> bool:
    """The odds feed's and ESPN's names for one team ("Los Angeles Clippers", "LA Clippers")."""
    if not a or not b:
        return False
    x, y = normalize_team(a), normalize_team(b)
    return x == y or (bool(x) and bool(y) and x.split()[-1] == y.split()[-1])


def _three_way(card: Mapping[str, Any]) -> bool:
    """Whether the game's moneyline has a draw side (soccer): then a draw is its own outcome."""
    return "draw" in ((card.get("markets") or {}).get("h2h") or {})


def _outcome(leg: Mapping[str, Any], final: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    """bets.leg_outcome, except that a draw loses either team's three-way moneyline.

    A two-way moneyline (NFL, NBA) gives the stake back on a tie, but soccer's has a draw side and is
    priced over three outcomes with no push, so backing a team there loses on a draw. Bets saved before
    the game's market shape was recorded fall back to the league.
    """
    three_way = event.get("three_way")
    if three_way is None:
        three_way = str(leg.get("league") or "").startswith("soccer_")
    if (three_way and leg["market"] == "h2h" and leg["side"] in ("home", "away")
            and final["home"] == final["away"]):
        return "lost"
    return bets.leg_outcome(dict(leg), dict(final))


# --- rollover plans: one day at a time ------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    """Where a rollover plan stands after its settled days (money in cents)."""

    bankroll: int  # all the plan's money: what is banked plus what rides on the next day
    banked: int = 0  # set aside for good (bank half and the ladder)
    rolled: int = 0  # carried into the next day's stake (0: nothing carried)
    pot_wins: int = 0  # the ladder's open pot: wins so far
    days: int = 0  # settled days
    ended: Optional[str] = None  # why the run ended early: lost, below_half or broke


def _share(cents: int, share: float) -> int:
    return int((Decimal(cents) * Decimal(str(share))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def paused_until(user: Optional[Mapping[str, Any]], now: float) -> Optional[float]:
    """When picks come back on for a person after a strong warning sign (epoch seconds), or None if they're on.

    The pause lives in the settings of the row holding the person's gates (``PAUSE_SETTING``; see gates_id),
    so the chat and the Lab both honour it."""
    until = ((user or {}).get("settings") or {}).get(PAUSE_SETTING)
    return float(until) if isinstance(until, (int, float)) and not isinstance(until, bool) and until > now else None


def gates_id(uid: str) -> str:
    """The users row holding a person's break, picks pause and weekly budget: their own, except that every
    password browser ("web:<X-Player>") shares SHARED_WEB. The dashboard password is the owner's alone and
    the X-Player id is whatever the browser makes up, so a new one mustn't shed a break, a pause or a limit.
    Telegram users ("tg:<id>") keep their own."""
    return SHARED_WEB if uid.startswith(WEB_PREFIX) else uid


# An IANA zone's shape ("Europe/London", "America/Argentina/Buenos_Aires", "Etc/GMT+5", "UTC"): this leaves
# out the files beside the zones on some systems ("localtime", "posixrules", "posix/...", "right/...")
ZONE_NAME = re.compile(r"[A-Z][A-Za-z0-9_+-]*(?:/[A-Za-z0-9_+-]+)*")


@functools.lru_cache(maxsize=1)
def _zone_names() -> frozenset:
    return frozenset(available_timezones())


def check_timezone(name: Any) -> Optional[str]:
    """An IANA time zone name the app sent (Intl's resolvedOptions().timeZone), or None to clear it."""
    if name is None:
        return None
    zones = _zone_names()
    try:
        ok = (isinstance(name, str) and len(name) <= 64 and ZONE_NAME.fullmatch(name) is not None
              and bool(ZoneInfo(name)) and (not zones or name in zones))
    except (ValueError, KeyError, OSError):  # unknown (ZoneInfoNotFoundError is a KeyError) or malformed
        ok = False
    if not ok:
        raise CallRejected("Unknown time zone: send an IANA name such as Europe/London")
    return name


class Gates:
    """Where the responsible-gambling gates' state is read and written: the one source of truth for the Lab,
    the chat assistant and the bot. A break and the pause on picks can only be lengthened; the weekly budget
    follows Store.set_weekly_budget (lowering at once, raising after 24 hours). Needs ``store`` and ``_now``.
    """

    store: Store
    _now: Callable[[], float]

    def gates(self, uid: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The row holding the person's gates, with any budget raise that has waited 24 hours applied: their
        own account (None if unknown), or for a password browser the row all of them share (made on first
        use from their own rows, so no limit set before is lost)."""
        now = self._now() if now is None else now
        key = gates_id(uid)
        if key == uid:
            return self.store.user(uid, now=now)
        return self.store.shared_user(key, WEB_PREFIX, carry=GATE_SETTINGS, now=now)

    def break_until(self, uid: str, now: Optional[float] = None) -> Optional[float]:
        """When the person's break ends (epoch seconds), or None when they aren't on one."""
        now = self._now() if now is None else now
        until = (self.gates(uid, now) or {}).get("cooloff_until")
        return float(until) if until is not None and until > now else None

    def picks_paused_until(self, uid: str, now: Optional[float] = None) -> Optional[float]:
        """When picks come back on after a warning sign (epoch seconds), or None if they're on."""
        now = self._now() if now is None else now
        return paused_until(self.gates(uid, now), now)

    def pause_picks(self, uid: str, until: float) -> float:
        """Switch picks off until ``until`` (epoch seconds) after a strong warning sign, for every password
        browser at once when one said it; a pause already running longer is kept. The only place the pause
        is written (the chat assistant and the bot come here). Returns when picks come back on."""
        self.store.get_or_create_user(uid)
        key = gates_id(uid)
        if key != uid:
            self.gates(uid)
        stored = self.store.extend_setting(key, PAUSE_SETTING, until)
        return float(until) if stored is None else stored

    def with_gates(self, uid: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """The person's account with the gates' fields (break, budget, pause, when a plan last failed) from the
        row holding them."""
        if gates_id(uid) == uid:
            return user
        shared = self.gates(uid) or {}
        settings = dict(user.get("settings") or {})
        for key in GATE_SETTINGS:
            settings.pop(key, None)
            value = (shared.get("settings") or {}).get(key)
            if value is not None:
                settings[key] = value
        return {**user, **{field: shared.get(field) for field in GATE_FIELDS}, "settings": settings}

    def gate_plans(self, uid: str, status: Any = None, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """The plans the plan rules (one at a time, a stopped plan's day in play, repeated failures) look at:
        the person's own, or for a password browser every password browser's, since they are one person (each
        plan still belongs to, and is shown in, the browser that started it: see its "user")."""
        if gates_id(uid) != uid:
            return self.store.plans(status=status, limit=limit, members=WEB_PREFIX)
        return self.store.plans(uid, status=status, limit=limit)


def next_stake(plan: Mapping[str, Any], run: Run) -> int:
    """The next day's stake in cents, by the plan's strategy (engine/rollover.py's rules)."""
    start, strategy = plan["start_cents"], plan["strategy"]
    if strategy == "classic":
        return run.bankroll  # everything, every day
    if strategy == "bank_half":
        return run.rolled if run.days else start
    if strategy == "ladder":
        return run.rolled or _share(start, rollover.POT_SHARE)  # the open pot, or a fresh 10% one
    return _share(start, rollover.FLAT_SHARE)


def play_day(plan: Mapping[str, Any], run: Run, stake: int, returns: int) -> Tuple[Run, int, int]:
    """The plan after one settled day: (new run, banked that day, rolled into the next).

    ``returns`` is what the day's ticket paid back (0 lost, the stake on a push).
    """
    start, strategy = plan["start_cents"], plan["strategy"]
    bankroll = run.bankroll - stake + returns
    won, lost = returns > stake, returns < stake
    banked = rolled = 0
    pot_wins = run.pot_wins
    ended = None
    if strategy == "classic":
        rolled = returns
        ended = "lost" if lost else None
    elif strategy == "bank_half":
        banked = (returns - stake + 1) // 2 if won else 0  # half the day's profit, a half cent up
        rolled = returns - banked
        ended = "lost" if lost else None
    elif strategy == "ladder":
        if won:
            pot_wins += 1
            if pot_wins >= rollover.POT_WINS:
                banked, pot_wins = returns, 0  # the pot has had its wins: bank it
            else:
                rolled = returns
        elif lost:
            pot_wins = 0
        else:
            rolled = returns  # a push leaves the pot as it was
        if lost and Decimal(bankroll) < Decimal(start) * Decimal(str(rollover.LADDER_STOP)):
            ended = "below_half"
    elif bankroll < _share(start, rollover.FLAT_SHARE):
        ended = "broke"  # flat: can't cover another stake
    # Money banked earlier stays banked only while it isn't staked again: the ladder pays each fresh pot
    # from all the plan's money, so a pot that can't come out of the rest comes out of the banked part,
    # and banked never reads more than the money not riding on the day
    kept = min(run.banked, max(0, run.bankroll - stake))
    return Run(bankroll, kept + banked, rolled, pot_wins, run.days + 1, ended), banked, rolled


def replay(plan: Mapping[str, Any]) -> Run:
    """The plan's state from its settled days, worked out again from each day's stake and returns."""
    run = Run(bankroll=plan["start_cents"])
    for day in plan.get("days_played") or []:
        result = day.get("result")
        if not result or result.get("returns") is None:
            break
        run, _, _ = play_day(plan, run, day["stake_cents"], _cents(result["returns"]))
        if run.ended:
            break
    return run


# --- pasted slips: matching a slip's words to the boards ------------------------------------------

# Pasted slips (engine/pasted.py reads them; the Lab matches and prices them at the person's own odds)
NBA = "basketball_nba"
GAMES_CACHE_SECONDS = GAMES_TTL  # DashboardService.games keeps a league's cards this long (the "games" policy)
PASTE_LOADS = 3  # leagues not already cached that one pasted slip may load (each fetch costs odds quota)
FETCH_COST = 3  # quota one league's odds fetch costs (moneyline, spreads and totals)
SAME_START = 36 * 3600  # a slip's start time and a board's game this close are the same game
NEXT_GAME = 7 * 24 * 3600  # without a start time: the next game between those teams within a week
LIVE_SINCE = 12 * 3600  # a game that started this long ago may still be on a board, in play
TEAM_MATCH = 0.85  # how close a slip's team name must come to a board's (paste_team_score)
BETTER_BY = Decimal("0.02")  # the board's best price beats the person's by this much: say so
FAR_ABOVE_BOARD = Decimal("1.25")  # the person's odds this many times the board's best: a misread? (asked)
DEFAULT_PASTE_STAKE = Decimal("10")
UNKNOWN_START = "2999-01-01T00:00:00Z"  # a leg whose start isn't known is never counted as started
NO_SUCH_PASTE = "No such pasted slip"
PASTE_STATS = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m", "steals": "stl",
               "blocks": "blk", "turnovers": "tov", "pra": "pra"}
ESPN_PROP_STATS = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m", "stl": "stl", "blk": "blk",
                   "tov": "tov", "pra": "pts+reb+ast"}
STAT_WORDS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "stl": "steals",
              "blk": "blocks", "tov": "turnovers", "pra": "points + rebounds + assists"}
# Short names slips use that share no word with the odds feed's name
TEAM_ALIASES = {
    "man city": "manchester city", "man utd": "manchester united", "man united": "manchester united",
    "spurs": "tottenham hotspur", "wolves": "wolverhampton wanderers", "psg": "paris saint germain",
    "paris sg": "paris saint germain", "inter": "inter milan", "internazionale": "inter milan",
    "barca": "barcelona", "atletico": "atletico madrid", "atleti": "atletico madrid", "bayern": "bayern munich",
    "bvb": "borussia dortmund", "dortmund": "borussia dortmund", "gladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen", "juve": "juventus", "newcastle": "newcastle united",
    "west ham": "west ham united", "forest": "nottingham forest", "nottm forest": "nottingham forest",
    "villa": "aston villa", "palace": "crystal palace", "sixers": "philadelphia 76ers",
    "niners": "san francisco 49ers", "cavs": "cleveland cavaliers", "mavs": "dallas mavericks",
    "wolves nba": "minnesota timberwolves", "pels": "new orleans pelicans", "blazers": "portland trail blazers",
    "habs": "montreal canadiens", "leafs": "toronto maple leafs", "bolts": "tampa bay lightning",
}
# Words many teams share: never enough on their own to say which team ("City", "United", "State")
GENERIC_TEAM_WORDS = {"city", "united", "town", "county", "rovers", "athletic", "wanderers", "albion", "real",
                      "sporting", "club", "de", "the", "inter", "st", "saint", "state", "university", "college",
                      "team", "utd", "and", "los", "angeles", "new", "york", "san", "fc"}


def _word_matches(asked: str, named: str) -> Optional[str]:
    """How one word of a slip's team name matches one of a board's: "exact", "prefix" ("Man" for
    "Manchester", "BOS" for "Boston") or "close" (a typo), else None."""
    if asked == named:
        return "exact"
    if len(asked) >= 2 and named.startswith(asked):
        return "prefix"
    if len(asked) >= 4 and difflib.SequenceMatcher(None, asked, named).ratio() >= 0.85:
        return "close"
    return None


def _team_score(asked: str, named: str) -> float:
    if asked == named:
        return 1.0
    words, names = asked.split(), named.split()
    if any(word not in GENERIC_TEAM_WORDS for word in words):
        kinds = [next((kind for kind in (_word_matches(word, name) for name in names) if kind), None)
                 for word in words]
        if all(kind == "exact" for kind in kinds):
            return 0.95
        if all(kinds):
            return 0.92
        if words[-1] == names[-1] and words[-1] not in GENERIC_TEAM_WORDS:
            return 0.88  # the nickname ("NY Knicks", "Celtics")
    ratio = difflib.SequenceMatcher(None, asked, named).ratio()
    return ratio * 0.95 if ratio >= 0.9 else ratio * 0.5


def paste_team_score(asked: Optional[str], team: Optional[str]) -> float:
    """How well a team as a slip writes it ("BOS Celtics", "Man City", "Spurs", "Celtcs") names a board's team
    ("Boston Celtics"), from 0 to 1; TEAM_MATCH or more is the same team."""
    query, name = normalize_team(asked or ""), normalize_team(team or "")
    if not query or not name:
        return 0.0
    options = {query, TEAM_ALIASES.get(query, query)}
    return max(_team_score(option, name) for option in options)


def _person_name(name: Optional[str]) -> List[str]:
    words = normalize_team(name or "").split()
    return [word for word in words if word not in ("jr", "sr", "ii", "iii", "iv")]


def paste_player_score(asked: Optional[str], name: Optional[str]) -> float:
    """How well a player as a slip writes him ("LeBron James", "L. James", "Lebron") matches ESPN's name."""
    wanted, found = _person_name(asked), _person_name(name)
    if not wanted or not found:
        return 0.0
    if wanted == found:
        return 1.0
    if len(wanted) >= 2 and len(wanted[0]) == 1 and wanted[-1] == found[-1] and found[0].startswith(wanted[0]):
        return 0.9  # "L. James"
    return difflib.SequenceMatcher(None, " ".join(wanted), " ".join(found)).ratio()


def _is_draw(selection: Optional[str]) -> bool:
    return (selection or "").strip().lower() in ("draw", "the draw", "tie", "x")


def _leg_teams(leg: pasted.ExtractedLeg) -> List[str]:
    """The team names a leg gives to find its game: both sides of its event, else the team it picked."""
    sides = pasted.event_sides(leg.event_text)
    if sides:
        return [sides[0], sides[1]]
    picked = (leg.selection or "").strip()
    if leg.market in ("moneyline", "spread") and picked and not _is_draw(picked) and picked.lower() not in (
            pasted.HOME_WORDS | pasted.AWAY_WORDS | {"over", "under"}):
        return [picked]
    return []


def _margin_kind(leg: pasted.ExtractedLeg, league: Optional[str], card: Optional[Mapping[str, Any]]) -> str:
    """Which typical margin estimates this leg's chance (engine/pasted.TYPICAL_MARGIN)."""
    if leg.market == "moneyline":
        soccer = (league or "").startswith("soccer_") or pasted.sport_prefixes(leg.sport_hint) == ("soccer_",)
        three = soccer or _is_draw(leg.selection) or (card is not None and _three_way(card))
        return "moneyline_3way" if three else "moneyline"
    return leg.market if leg.market in pasted.TYPICAL_MARGIN else "other"


def _paste_entry(status: str, note: str, kind: str, league: Optional[str] = None,
                 card: Optional[Mapping[str, Any]] = None, now: Optional[float] = None) -> Dict[str, Any]:
    """One leg's match as stored with the slip: the game found (if any), its chances and where they came from."""
    start = card.get("start") if card else None
    begun = _parse_time(start) if start else None
    live = bool(card and (card.get("live") or (now is not None and begun is not None and begun.timestamp() <= now)))
    return {"status": status, "note": note, "kind": kind, "league": league, "ref": None,
            "game": f"{card['away']} @ {card['home']}" if card else None, "start": start, "live": live,
            "group": str(card["id"]) if card else None, "market_chance": None, "ai_chance": None, "push": 0.0,
            "chance_source": "estimated", "best_price": None, "better_price": False, "board": None}


def _margin_words(kind: str) -> str:
    return _pct(pasted.TYPICAL_MARGIN.get(kind, pasted.TYPICAL_MARGIN["other"]))


def _line_words(market: str, side: str, point: Optional[float]) -> str:
    """A line as the app writes it: "-4.5", "+3", "PK", "Over 224.5"."""
    if point is None:
        return "no line"
    if market in ("totals", "total"):
        return f"{side.title()} {point:g}"
    return "PK" if point == 0 else f"{point:+g}"


def _paste_label(leg: pasted.ExtractedLeg) -> str:
    """How a pasted leg reads on the card and the ticket: "Boston Celtics -3.5", "Over 231.5",
    "LeBron James over 27.5 points", "Draw"."""
    selection = leg.selection or leg.raw
    if leg.market == "spread" and leg.line is not None:
        return f"{selection} {_line_words('spreads', '', leg.line)}"[:MAX_LABEL]
    if leg.market == "total" and leg.line is not None:
        return f"{(leg.over_under or selection).title()} {leg.line:g}"[:MAX_LABEL]
    if leg.market == "player_prop" and leg.line is not None and leg.over_under:
        stat = f" {leg.prop_stat}" if leg.prop_stat else ""
        return f"{selection} {leg.over_under} {leg.line:g}{stat}"[:MAX_LABEL]
    if leg.market == "moneyline" and not _is_draw(selection) and not re.search(r"\bwin$", selection, re.I):
        return f"{selection} to win"[:MAX_LABEL]
    return selection[:MAX_LABEL]


MAX_LABEL = 90


# --- the lab --------------------------------------------------------------------------------------

class Lab(Gates):
    """Per-person features on top of the games service (see the module docstring).

    ``assistant`` is the Claude assistant (ai/assistant.py) or None; it needs
    ``explain(uid, fact_sheet)``, ``reply(uid, thread_id, message, request, slip)``, ``credits(uid)`` and,
    for pasted slips, ``read_slip(uid, thread_id=, text=, image=, media_type=)``.
    """

    def __init__(self, service: DashboardService, store: Optional[Store] = None, players: Optional[PlayerData] = None,
                 assistant: Any = None, public_url: Optional[str] = None, daily_cents: int = DAILY_AI_CENTS,
                 now: Optional[Callable[[], float]] = None) -> None:
        self.service = service
        self._store = store
        self._players = players
        self.assistant = assistant
        self.public_url = public_url
        self.daily_cents = daily_cents
        self._now = now or time.time
        # Its own threads: service.games() uses the service's pool inside, so sharing it could deadlock
        self.pool = ThreadPoolExecutor(max_workers=4)
        # The API runs requests on many threads: a person's writes take their lock (see _writing)
        self._write_locks = tuple(threading.RLock() for _ in range(WRITE_LOCKS))
        self._graded_at = 0.0
        # The games service runs the slip and plan grader inside its grading lease
        service.lab = self

    @property
    def store(self) -> Store:
        store = self._store if self._store is not None else self.service.store
        if store is None:
            raise CallRejected(NO_STORE)
        return store

    @property
    def players(self) -> PlayerData:
        if self._players is None:
            self._players = PlayerData(self.service.espn)
        return self._players

    def _writing(self, uid: str) -> threading.RLock:
        """The lock a person's money-moving writes hold from their checks to their save.

        A double tap sends two requests that run side by side; without it both could pass "the previous
        day has a result" or "the budget has room" before either saved. The dashboard is one process and
        the bot never saves bets or plans, so a lock here is enough. It is re-entrant (settling a plan's
        day takes it too, inside plan_day), and nothing holding it starts a grading run, which takes
        other people's locks. Password browsers share one budget, so they share one lock too.
        """
        return self._write_locks[hash(gates_id(uid)) % len(self._write_locks)]

    def _when(self, ts: float) -> str:
        local = datetime.datetime.fromtimestamp(ts, self.service.tz)
        return f"{local:%a %b} {local.day}, {local:%-I:%M %p %Z}"

    # --- people and the responsible-gambling gates -------------------------------------------------

    def user(self, uid: str) -> Dict[str, Any]:
        """The person's account, made on first use, with the gates (break, picks pause, weekly budget) read
        from the row holding them: for a password browser, the one every password browser shares."""
        try:
            return self.with_gates(uid, self.store.get_or_create_user(uid, now=self._now()))
        except ValueError as exc:
            raise CallRejected(str(exc)) from None

    def require_ready(self, user: Mapping[str, Any]) -> None:
        """Code's gate before anything betting-related: age confirmed and not on a break."""
        if user.get("age_confirmed_at") is None:
            raise CallRejected(AGE_MESSAGE)
        until = user.get("cooloff_until")
        if until is not None and until > self._now():
            raise CallRejected(f"You're on a break until {self._when(until)}")

    def ready(self, uid: str, picks: bool = False) -> Dict[str, Any]:
        """The person's account once the gates pass; ``picks`` adds the 24-hour pause after a warning sign
        in the chat, for anything that picks or plays bets (building, plans). Analysing still shows the
        numbers then, without the AI's words (the assistant refuses those)."""
        user = self.user(uid)
        self.require_ready(user)
        if picks:
            until = paused_until(user, self._now())
            if until:
                raise CallRejected(f"Picks are paused until {self._when(until)} after an earlier message about "
                                   "losses or money worries. The odds and the maths stay open.")
        return user

    def week_start(self, now: Optional[float] = None) -> float:
        """Monday 00:00 this week, local time: the weekly budget's week."""
        local = datetime.datetime.fromtimestamp(self._now() if now is None else now, self.service.tz)
        monday = (local - datetime.timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return monday.timestamp()

    def budget_used(self, uid: str, since: float) -> int:
        """Cents staked on bets since a time, by the person or, for a password browser, by every password
        browser (they share one budget). A paper plan's days are play money, so they don't count."""
        if gates_id(uid) != uid:
            return self.store.budget_staked_since(since, members=WEB_PREFIX)
        return self.store.budget_staked_since(since, user=uid)

    def budget(self, uid: str, user: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """{weekly, used, left, resets, pending} in money strings, or None without a budget."""
        weekly = user.get("weekly_budget_cents")
        if weekly is None:
            return None
        since = self.week_start()
        used = self.budget_used(uid, since)
        pending = None
        if user.get("pending_budget_cents") is not None:
            pending = {"amount": _money(user["pending_budget_cents"]), "effective_at": _iso(user.get("budget_raise_at"))}
        resets = datetime.datetime.fromtimestamp(since, self.service.tz) + datetime.timedelta(days=7)
        return {"weekly": _money(weekly), "used": _money(used), "left": _money(max(0, weekly - used)),
                "resets": _iso(resets.timestamp()), "pending": pending}

    def check_budget(self, uid: str, user: Mapping[str, Any], cents: int, what: str) -> None:
        """Refuse a stake past what this week's budget has left (and say how much is left)."""
        weekly = user.get("weekly_budget_cents")
        if weekly is None:
            return
        left = max(0, weekly - self.budget_used(uid, self.week_start()))
        if cents > left:
            raise CallRejected(f"{what} of {_say(cents)} is more than your weekly budget has left: "
                               f"{_say(left)} until Monday.")

    def credits(self, uid: str) -> Dict[str, Any]:
        """Today's AI allowance: {used_cents, daily_cents, left_cents}."""
        if self.assistant is not None:
            try:
                return self.assistant.credits(uid)
            except Exception:
                log.exception("Couldn't read the AI credits")
        store = self._store if self._store is not None else self.service.store
        used = store.ai_used_today(uid, now=self._now()) if store is not None else 0.0
        return {"used_cents": round(used, 2), "daily_cents": self.daily_cents,
                "left_cents": round(max(0.0, self.daily_cents - used), 2)}

    def me(self, uid: str) -> Dict[str, Any]:
        user = self.user(uid)
        until = user.get("cooloff_until")
        on_break = until is not None and until > self._now()
        return {
            "uid": uid,
            "name": user.get("name"),
            "is_telegram": uid.startswith("tg:"),
            "age_confirmed": user.get("age_confirmed_at") is not None,
            "odds_format": user.get("odds_format") or "decimal",
            "currency": user.get("currency") or "USD",
            "favourite_sports": list(user.get("favourite_sports") or []),
            # The person picked these themselves (so an NBA-only choice isn't read as the old NBA default)
            "sports_chosen": bool((user.get("settings") or {}).get(SPORTS_CHOSEN_SETTING)),
            "risk_tier": user.get("risk_tier"),
            "budget": self.budget(uid, user),
            "reality_check_minutes": user.get("reality_check_minutes"),
            "cooloff_until": _iso(until) if on_break else None,
            "on_break": on_break,
            "picks_paused_until": _iso(paused_until(user, self._now())),
            # A break, the picks pause and the budget (staked by all of them) cover every password browser
            "shared_limits": gates_id(uid) != uid,
            "timezone": (user.get("settings") or {}).get(TIMEZONE_SETTING),
            "credits": self.credits(uid),
            "public_url": self.public_url,
            "ai_enabled": self.assistant is not None,
        }

    def update_me(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Change settings. Only keys present in ``body`` change; a budget raise waits 24 hours. A password
        browser's budget is the one all password browsers share; its age confirmation is its own."""
        self.user(uid)
        store = self.store
        changes: Dict[str, Any] = {}
        if "timezone" in body:  # None clears it
            changes["settings"] = {TIMEZONE_SETTING: check_timezone(body["timezone"])}
        for field in ("name", "risk_tier", "reality_check_minutes"):  # None clears these
            if field in body:
                changes[field] = body[field]
        for field in ("odds_format", "favourite_sports"):
            if body.get(field) is not None:
                changes[field] = body[field]
        if "favourite_sports" in changes:
            sports = list(changes["favourite_sports"])
            if len(sports) > MAX_LEAGUES or not all(isinstance(s, str) and LEAGUE_KEY.fullmatch(s) for s in sports):
                raise CallRejected(f"Pick up to {MAX_LEAGUES} sports")
            changes["settings"] = {**changes.get("settings", {}), SPORTS_CHOSEN_SETTING: True}
        try:
            if changes:
                store.update_user(uid, changes, now=self._now())
            if body.get("weekly_budget") is not None:
                store.set_weekly_budget(gates_id(uid), _cents(_amount(body["weekly_budget"], "weekly budget",
                                                                      MAX_BUDGET, least=Decimal(0))), now=self._now())
            if body.get("confirm_age"):
                store.confirm_age(uid, now=self._now())
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return self.me(uid)

    def cooloff(self, uid: str, hours: int) -> Dict[str, Any]:
        """Start a break: picks, building and plans switch off until it ends (it can't be cut short). Taken in
        one password browser, it covers them all."""
        if hours not in COOLOFF_CHOICES:
            raise CallRejected("A break lasts 24 hours, 3 days, a week, 30 days or 90 days")
        self.user(uid)
        self.store.start_cooloff(gates_id(uid), hours, now=self._now())
        return self.me(uid)

    # --- the board: games and their prices ----------------------------------------------------------

    def sports(self) -> List[Dict[str, Any]]:
        """In-season leagues as one list: [{key, title, group, label, ai}]."""
        return [{"key": league["key"], "title": league["title"], "group": group["group"], "label": group["label"],
                 "ai": league["ai"]}
                for group in self.service.leagues() for league in group["leagues"]]

    def _leagues(self, requested: Optional[Iterable[str]], user: Mapping[str, Any]) -> List[str]:
        """The leagues asked for, else the person's favourites, else the NBA."""
        leagues = list(dict.fromkeys(requested or [])) or list(user.get("favourite_sports") or []) or list(DEFAULT_LEAGUES)
        if len(leagues) > MAX_LEAGUES:
            raise CallRejected(f"Pick up to {MAX_LEAGUES} leagues at a time")
        if not all(isinstance(key, str) and LEAGUE_KEY.fullmatch(key) for key in leagues):
            raise CallRejected("League keys look like basketball_nba")
        return leagues

    def _season(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """The odds feed's leagues by key (a free call, cached for hours), or None if it can't be read."""
        try:
            return {sport["key"]: sport for sport in self.service.odds.sports()}
        except Exception:
            return None

    @staticmethod
    def _out_of_season(info: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not info or not info.get("active"):
            return "Not in season right now, so there are no odds to show."
        if info.get("has_outrights"):
            return "Only futures are on offer for this league."
        return None

    def _unsettled(self, leagues: Iterable[str]) -> str:
        """The leagues among these that the grader can't settle (ESPN's final scores here cover only
        espn_client.LEAGUES), named for a message ("ATP Paris Masters and MMA"); "" when all can be.

        A saved bet on one would only ever be voided for want of a result, its loss never counted."""
        missing = sorted({league for league in leagues if league not in ESPN_LEAGUES})
        if not missing:
            return ""
        season = self._season() or {}
        names = [(season.get(league) or {}).get("title") or league for league in missing]
        return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"

    def _board(self, league: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """A league's cached board, or None and why. The odds feed's errors carry only an HTTP status;
        anything else (a connection error) may quote the request URL with its key, so it isn't repeated."""
        try:
            return self.service.games(league), None
        except RuntimeError as exc:
            return None, str(exc)[:200]
        except Exception as exc:
            log.warning("Loading the %s board failed (%s)", league, type(exc).__name__)
            return None, "Couldn't load this league right now; try again in a minute."

    def boards(self, leagues: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """Each league's board and, for leagues that couldn't be shown, why. Out-of-season leagues are
        never fetched (a fetch costs quota even when it returns nothing)."""
        season = self._season()
        errors: Dict[str, str] = {}
        wanted = []
        for league in leagues:
            problem = self._out_of_season(season.get(league)) if season is not None else None
            if problem:
                errors[league] = problem
            else:
                wanted.append(league)
        boards: Dict[str, Dict[str, Any]] = {}
        for league, (board, error) in zip(wanted, self.pool.map(self._board, wanted)):
            if board is None:
                errors[league] = error or "No odds right now."
            else:
                boards[league] = board
        return boards, errors

    def today(self, uid: str, leagues: Optional[Sequence[str]] = None, day: Optional[str] = None) -> Dict[str, Any]:
        """Games across the chosen leagues (default: favourites, else the NBA), soonest first.

        ``day`` (YYYY-MM-DD, local time) keeps only that day's games.
        """
        user = self.ready(uid)
        keys = self._leagues(leagues, user)
        wanted_day = None
        if day:
            try:
                wanted_day = datetime.date.fromisoformat(day)
            except ValueError:
                raise CallRejected("day must look like 2026-10-26") from None
        boards, errors = self.boards(keys)
        games = []
        for league, board in boards.items():
            for card in board["games"]:
                start = _parse_time(card["start"])
                if wanted_day and (start is None or start.astimezone(self.service.tz).date() != wanted_day):
                    continue
                games.append({**card, "league_title": board.get("title") or league})
        games.sort(key=lambda card: (card["start"], card["league"], card["id"]))
        return {"games": games, "errors": errors, "quota": dict(self.service.odds.quota),
                "odds_as_of": {league: board.get("odds_as_of") for league, board in boards.items()},
                "leagues": keys, "day": day}

    def _lookup(self, ref: Mapping[str, Any]) -> Optional[Tuple[TicketLeg, Dict[str, Any]]]:
        """The pick's current outcome on the cached board, as a TicketLeg with its game card.

        A board that can't be loaded is a RuntimeError with a safe reason (the API's 502), never the
        feed's own exception, whose message may quote the request URL with its key.
        """
        board, error = self._board(ref["league"])
        if board is None:
            raise RuntimeError(error or "No odds right now.")
        card = next((c for c in board["games"] if str(c["id"]) == ref["id"]), None)
        offer = ((card or {}).get("markets") or {}).get(ref["market"], {}).get(ref["side"])
        if not card or not offer or not offer.get("price"):
            return None
        try:
            return self._leg(ref["league"], card, ref["market"], ref["side"], offer), card
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Couldn't price %s: %s", ref, exc)
            return None

    def _leg(self, league: str, card: Mapping[str, Any], market: str, side: str, offer: Mapping[str, Any]) -> TicketLeg:
        point = offer.get("point")
        # Without a model prediction the card's "ai" is just the market's chance again, so it isn't the AI's
        modelled = card.get("prediction") is not None
        return TicketLeg(
            key=_leg_key(league, str(card["id"]), market, side, point), league=league, event_id=str(card["id"]),
            market=market, side=side, point=point, price=Decimal(str(offer["price"])), label=offer["label"],
            game=f"{card['away']} @ {card['home']}", start=card["start"],
            ai_prob=offer.get("ai") if modelled else None, market_prob=offer.get("fair"),
            push_prob=offer.get("push") or 0.0, book=offer.get("book"), group=str(card["id"]),
        )

    def leg_from_ref(self, ref: Mapping[str, Any]) -> Optional[TicketLeg]:
        """A pick ({league, id, market, side}) priced from the cached board, or None if it's gone."""
        season = self._season()
        checked = _ref(ref)
        if season is not None and self._out_of_season(season.get(checked["league"])):
            return None
        found = self._lookup(checked)
        return found[0] if found else None

    def _resolve(self, refs: Sequence[Any]) -> Tuple[List[Tuple[Dict[str, Any], TicketLeg, Dict[str, Any]]],
                                                      List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(found picks as (ref, leg, card), refs no longer offered, lines that moved since the person saw them)."""
        season = self._season()
        found, removed, moved = [], [], []
        for raw in refs:
            ref = _ref(raw)
            if season is not None and self._out_of_season(season.get(ref["league"])):
                removed.append(ref)
                continue
            hit = self._lookup(ref)
            if hit is None:
                removed.append(ref)
                continue
            leg, card = hit
            if ref["point"] is not None and leg.point is not None and ref["point"] != leg.point:
                moved.append({"ref": ref, "label": leg.label, "was": ref["point"], "now": leg.point})
            found.append((ref, leg, card))
        return found, removed, moved

    # --- tickets ------------------------------------------------------------------------------------

    @staticmethod
    def _ticket_args(body: Mapping[str, Any]) -> Tuple[str, Decimal, Optional[int], Optional[str], List[Any]]:
        ticket_type = body.get("type") or "multi"
        if ticket_type not in tickets.TICKET_TYPES:
            raise CallRejected(f"A ticket is one of {', '.join(tickets.TICKET_TYPES)}")
        stake = _amount(body.get("stake"), "stake", MAX_STAKE)
        k = body.get("k")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int)):
            raise CallRejected("k is a round robin's picks per bet, e.g. 2")
        system = body.get("system") or None
        refs = list(body.get("legs") or [])
        if not 1 <= len(refs) <= tickets.MAX_LEGS:
            raise CallRejected(f"A ticket has 1 to {tickets.MAX_LEGS} picks")
        return ticket_type, stake, k, system, refs

    @staticmethod
    def _priced(legs: Sequence[TicketLeg], ticket_type: str, stake: Decimal, k: Optional[int] = None,
                system: Optional[str] = None) -> Dict[str, Any]:
        try:
            return tickets.price_ticket(legs, ticket_type, stake, k=k, system=system)
        except ValueError as exc:
            raise CallRejected(str(exc)) from None

    def _price(self, body: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], TicketLeg, Dict[str, Any]]]]:
        ticket_type, stake, k, system, refs = self._ticket_args(body)
        return self._price_refs(refs, ticket_type, stake, k, system)

    def _price_refs(self, refs: Sequence[Any], ticket_type: str, stake: Decimal, k: Optional[int] = None,
                    system: Optional[str] = None) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], TicketLeg,
                                                                                         Dict[str, Any]]]]:
        """The picks priced at a stake already checked by the caller (a request's stake has a cap; a
        rollover plan's own stake may grow past it)."""
        found, removed, moved = self._resolve(refs)
        legs = [leg for _, leg, _ in found]
        priced = self._priced(legs, ticket_type, stake, k, system)
        if removed:
            priced["valid"] = False
            gone = "1 pick is" if len(removed) == 1 else f"{len(removed)} picks are"
            priced["reasons"] = list(priced["reasons"]) + [f"{gone} no longer offered and left out: remove "
                                                            f"{'it' if len(removed) == 1 else 'them'} from the slip"]
        priced["warnings"] = list(priced["warnings"]) + [
            f"The line moved on {m['label']}: it was {m['was']:g} when you picked it." for m in moved]
        unsettled = self._unsettled([leg.league for leg in legs])
        if unsettled:
            priced["warnings"].append(f"{unsettled} bets can't be settled automatically yet, so this ticket can be "
                                      "priced and analysed but not saved.")
        books = sorted({leg.book for leg in legs if leg.book})
        if priced["type"] != "single" and len(books) > 1:
            # Each leg is priced at its best bookmaker; one bookmaker's ticket usually pays a little less
            priced["warnings"].append(f"These are the best prices at different bookmakers ({', '.join(books)}); "
                                      "a ticket at one bookmaker may pay less.")
        as_of = {}
        for league in sorted({leg.league for leg in legs}):
            loaded = self.service.odds.odds_loaded_at(league)
            as_of[league] = _iso(loaded) if loaded else None
        priced.update(removed=removed, moved=moved, refs=[_ref_of(view) for view in priced["legs"]], odds_as_of=as_of)
        return priced, found

    def price(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        """A ticket priced from the cached odds: {type, stake, k?, system?, legs: [ref]} -> tickets.price_ticket
        plus ``removed`` (picks no longer offered), ``moved`` (lines that changed) and ``refs``.

        ``stake`` is per bet: each single, the multi, each bet of a round robin or system bet;
        ``total_stake`` in the result is what the whole ticket costs.
        """
        self.ready(uid)
        return self._price(body)[0]

    def analyse(self, uid: str, body: Mapping[str, Any], explain: bool = True) -> Dict[str, Any]:
        """The ticket priced, facts about each leg and alternatives, all from code; then Claude's words
        around those numbers when an assistant is set up (``explanation`` is None otherwise).

        ``explain=False`` skips Claude (the chat asks for the analysis as a tool and writes its own words).
        """
        self.ready(uid)
        priced, found = self._price(body)
        legs = [{"index": view["index"], "label": view["label"], "notes": self._notes(view, card)}
                for view, (_, _, card) in zip(priced["legs"], found)]
        alternatives = self._alternatives(priced, [leg for _, leg, _ in found])
        result: Dict[str, Any] = {"ticket": priced, "legs": legs, "alternatives": alternatives, "explanation": None}
        if explain and self.assistant is not None and priced["valid"]:
            written, note = None, None
            try:
                written = self.assistant.explain(uid, self.fact_sheet(priced, legs, alternatives))
            except CallRejected as exc:  # e.g. today's AI credits are used up
                note = str(exc)
            except Exception:
                log.exception("The AI explanation failed")
                note = "The AI's explanation isn't available right now; the numbers above are complete."
            if written:
                result["explanation"] = {**written, "alternatives": alternatives}
            if note:
                result["note"] = note
        result["credits"] = self.credits(uid)
        return result

    @staticmethod
    def fact_sheet(priced: Mapping[str, Any], legs: Sequence[Mapping[str, Any]],
                   alternatives: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Everything Claude may say about a ticket, as computed here: it explains these numbers and no others."""
        ticket_keys = ("type", "system", "k", "bets", "unit_stake", "total_stake", "odds", "max_payout", "max_profit",
                       "chance", "book_cut", "expected", "most_likely", "outcomes", "weakest_leg", "warnings", "valid",
                       "reasons")
        leg_keys = ("index", "label", "game", "league", "start", "odds", "ai_prob", "market_prob", "push_prob",
                    "fair_odds", "chance_label", "edge", "book_cut", "book")
        notes = {leg["index"]: leg["notes"] for leg in legs}
        return {
            "currency": CURRENCY,
            "ticket": {key: priced.get(key) for key in ticket_keys},
            "legs": [{**{key: view.get(key) for key in leg_keys}, "notes": notes.get(view["index"], [])}
                     for view in priced["legs"]],
            "alternatives": [{"title": alt["title"], "type": alt["ticket"]["type"], "k": alt["ticket"]["k"],
                              "unit_stake": alt["ticket"]["unit_stake"], "total_stake": alt["ticket"]["total_stake"],
                              "chance": alt["ticket"]["chance"], "expected": alt["ticket"]["expected"],
                              "most_likely": alt["ticket"]["most_likely"], "summary": alt["summary"]}
                             for alt in alternatives],
        }

    def _notes(self, leg: Mapping[str, Any], card: Optional[Mapping[str, Any]]) -> List[str]:
        """Plain facts about one leg (numbers from the board, the model and ESPN; nothing made up)."""
        notes: List[str] = []
        p, q = leg.get("ai_prob"), leg.get("market_prob")
        if leg.get("started"):
            notes.append("This game has started, so this pick can't go on a new ticket.")
        chance = p if p is not None else q
        if chance is not None:
            whose = "the AI's" if p is not None else "the market's"
            notes.append(f"{(leg.get('chance_label') or '').capitalize()}: wins about {_chance_words(chance)} "
                         f"by {whose} chance.")
        if p is not None and q is not None:
            # The AI's chance is outright and the market's is given no push: compare them like for like
            push = leg.get("push_prob") or 0.0
            ai = tickets.ai_given_no_push(p, push)
            gap = ai - q
            basis = " (both given no push)" if push > 0 else ""
            if abs(gap) > tickets.AI_GAP:
                notes.append(f"The AI ({_pct(ai)}) is {abs(gap) * 100:.0f} points {'above' if gap > 0 else 'below'} "
                             f"the market ({_pct(q)}){basis}: check the injury news before trusting the gap.")
            else:
                notes.append(f"The AI ({_pct(ai)}) and the market ({_pct(q)}) roughly agree{basis}.")
        elif q is not None:
            notes.append(f"No AI model for this pick: the market's {_pct(q)} is the only view.")
        odds = (leg.get("odds") or {}).get("decimal")
        cut = leg.get("book_cut")
        if cut is not None and odds is not None:
            where = f" at {leg['book']}" if leg.get("book") else ""
            if cut > 0:
                notes.append(f"At {odds:.2f}{where} the bookmaker keeps about {_pct(cut)} of money staked on this pick.")
            else:
                notes.append(f"The best price ({odds:.2f}{where}) beats the bookmakers' consensus: check it's still on offer.")
        if leg.get("edge") is not None:
            notes.append(f"Expected result at the AI's chance: {leg['edge'] * 100:+.1f}% of the stake.")
        if (leg.get("push_prob") or 0) >= 0.01:
            notes.append(f"About {_pct(leg['push_prob'])} chance of a push (the stake on this pick comes back).")
        if card:
            notes += self._card_notes(leg, card)
        return notes

    @staticmethod
    def _card_notes(leg: Mapping[str, Any], card: Mapping[str, Any]) -> List[str]:
        notes = []
        for side in ("away", "home"):
            names = (card.get("injuries") or {}).get(side) or []
            if names:
                more = f" and {len(names) - 4} more" if len(names) > 4 else ""
                notes.append(f"Injury report, {card[side]}: {', '.join(names[:4])}{more}.")
        for factor in (card.get("prediction") or {}).get("factors") or []:
            if "back-to-back" in factor or "missing" in factor:
                notes.append(f"Model factor: {factor}.")
        side = leg.get("side")
        if leg.get("market") in ("h2h", "spreads") and side in ("home", "away"):
            team = card[side]
            espn = card.get("espn_home_win")
            if espn is not None:
                notes.append(f"ESPN's matchup predictor gives {team} {_pct(espn if side == 'home' else 1 - espn)} to win.")
            for brief in card.get("analysts") or []:
                home_win = brief.get("home_win")
                if home_win is not None:
                    notes.append(f"{brief.get('label') or 'An analyst'} gives {team} "
                                 f"{_pct(home_win if side == 'home' else 1 - home_win)} to win.")
        return notes

    def _alternatives(self, priced: Mapping[str, Any], legs: Sequence[TicketLeg],
                      currency: Optional[str] = CURRENCY) -> List[Dict[str, Any]]:
        """Other ways to play the same picks, priced the same way: drop the weakest leg, split into singles
        (same total stake), and for 4+ legs a round robin by 2s and by 3s (same total stake)."""
        n = len(legs)
        if n < 2 or not priced.get("bets"):
            return []
        kind = priced["type"]
        unit, total = Decimal(priced["unit_stake"]), Decimal(priced["total_stake"])
        out = []
        weakest = priced.get("weakest_leg")
        if kind in ("multi", "single") and weakest is not None:
            rest = [leg for i, leg in enumerate(legs) if i != weakest]
            out.append(self._alternative(f"Drop the weakest leg ({legs[weakest].label})", rest, kind, unit, None,
                                         priced, currency))
        if kind != "single":
            each = _floor_cents(total / n)
            if each >= CENTS:
                out.append(self._alternative(f"Split into {n} singles of {_say_dollars(each, currency)}", legs,
                                             "single", each, None, priced, currency))
        if kind == "multi" and 4 <= n <= tickets.MAX_COMBO_LEGS:
            for k in (2, 3):
                count = math.comb(n, k)
                each = _floor_cents(total / count)
                if each >= CENTS:
                    out.append(self._alternative(f"Round robin by {k}s: {count} bets of "
                                                 f"{_say_dollars(each, currency)}", legs, "round_robin", each, k,
                                                 priced, currency))
        return [alt for alt in out if alt["ticket"]["valid"]]

    def _alternative(self, title: str, legs: Sequence[TicketLeg], kind: str, stake: Decimal, k: Optional[int],
                     base: Mapping[str, Any], currency: Optional[str] = CURRENCY) -> Dict[str, Any]:
        ticket = self._priced(legs, kind, stake, k)
        return {"title": title, "refs": [_ref_of(view) for view in ticket["legs"]], "ticket": ticket,
                "summary": _versus(ticket, base, currency)}

    def build(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        """"Stake X, want Y back": target.build_for_target over the chosen leagues' boards, with refs on every
        leg so the app can add them to the slip. The stake must fit the weekly budget; it is never raised."""
        user = self.ready(uid, picks=True)
        leagues = self._leagues(body.get("leagues"), user)
        stake = _amount(body.get("stake"), "stake", MAX_STAKE)
        goal = _amount(body.get("target"), "target", MAX_TARGET)
        max_legs = body.get("max_legs") or 3
        if isinstance(max_legs, bool) or not isinstance(max_legs, int) or not 1 <= max_legs <= 6:
            raise CallRejected("max_legs is 1 to 6")
        self.check_budget(uid, user, _cents(stake), "This stake")
        boards, errors = self.boards(leagues)
        candidates = []
        for league, board in boards.items():
            for card in board["games"]:
                if card.get("live"):
                    continue
                for market, sides in (card.get("markets") or {}).items():
                    for side, offer in sides.items():
                        if market not in GAME_MARKETS or not offer.get("price"):
                            continue
                        try:
                            candidates.append(self._leg(league, card, market, side, offer))
                        except (ValueError, KeyError, TypeError):
                            continue
        try:
            result = target.build_for_target(candidates, stake, goal, max_legs=max_legs, tier=body.get("tier"))
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        for ticket in result["tickets"]:
            ticket["refs"] = [_ref_of(view) for view in ticket["legs"]]
        result.update(leagues=leagues, errors=errors,
                      odds_as_of={league: board.get("odds_as_of") for league, board in boards.items()})
        return result

    # --- saved bets ---------------------------------------------------------------------------------

    def save_slip(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Save a bet, re-priced here, with each leg's exact quote and a share code."""
        with self._writing(uid):  # the gate, the budget left and the save, one request at a time
            user = self.ready(uid)
            priced, found = self._price(body)
            return self._save(uid, user, priced, found)

    def _save(self, uid: str, user: Mapping[str, Any], priced: Dict[str, Any],
              found: Sequence[Tuple[Dict[str, Any], TicketLeg, Dict[str, Any]]],
              plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save a priced ticket (the caller holds the person's write lock). It must be the bet the person
        saw: every pick still offered, on the line they picked, and in a league the grader can settle."""
        if priced["removed"]:
            raise CallRejected("Some picks are no longer offered: take them off the slip first.")
        if priced["moved"]:
            lines = "; ".join(f"{m['label']} now (it was {m['was']:g})" for m in priced["moved"])
            one = len(priced["moved"]) == 1
            raise CallRejected(f"The line has moved since you picked {'it' if one else 'them'}: {lines}. Nothing was "
                               f"saved: to take the new line, remove the pick and add it again from the game.")
        unsettled = self._unsettled(view["league"] for view in priced["legs"])
        if unsettled:
            raise CallRejected(f"{unsettled} bets can't be settled automatically yet, so they can't be saved: take "
                               "those picks off the slip to save the rest.")
        if not priced["valid"]:
            raise CallRejected(" ".join(priced["reasons"]) or "This ticket can't be saved.")
        total = _cents(priced["total_stake"])
        if total > MAX_CENTS:
            raise CallRejected(f"A stake of {_say(total)} is more than the app can record.")
        if not (plan and plan.get("mode") == "paper"):
            self.check_budget(uid, user, total, "This ticket")
        now = self._now()
        legs = [{**view, "quoted_at": self.service.odds.odds_loaded_at(view["league"]) or now} for view in priced["legs"]]
        # Team names for grading: ESPN's final score is found by date and both teams
        # and whether its moneyline has a draw side (a team's backer then loses on a draw)
        events = {str(card["id"]): {"league": leg.league, "home": card["home"], "away": card["away"],
                                    "three_way": _three_way(card)}
                  for _, leg, card in found}
        payload = {**priced, "events": events}
        if plan:
            payload["plan"] = plan
        try:
            row = self.store.save_slip(uid, priced["type"], legs, total, payload=payload, system=priced.get("system"),
                                       k=priced.get("k"), quoted_at=now, now=now)
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return self._slip_view(row)

    def _slip_view(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        payload = row.get("payload") or {}
        legs = []
        for leg in row.get("legs") or []:
            legs.append({
                "index": leg["idx"], "label": leg["label"], "game": leg.get("game"), "league": leg["league"],
                "event_id": leg["event_id"], "market": leg["market"], "side": leg["side"], "point": leg.get("point"),
                "price": leg["price"], "odds": _odds(leg["price"]), "start": leg.get("start"), "book": leg.get("book"),
                "ai_prob": leg.get("ai_prob"), "market_prob": leg.get("market_prob"), "result": leg.get("result"),
            })
        view = {
            "id": row["id"], "status": row["status"], "type": row["type"], "title": TICKET_TITLES.get(row["type"], row["type"]),
            "system": row.get("system"), "k": row.get("k"),
            "stake": _money(row["stake_cents"]), "unit_stake": payload.get("unit_stake"), "bets": payload.get("bets"),
            "created_at": _iso(row.get("created_at")), "updated_at": _iso(row.get("updated_at")),
            "share_code": row.get("share_code"),
            "odds": payload.get("odds"), "max_payout": payload.get("max_payout"), "chance": payload.get("chance"),
            "book_cut": payload.get("book_cut"), "expected": payload.get("expected"),
            "most_likely": payload.get("most_likely"),
            "legs": legs, "refs": [_ref_of(leg) for leg in legs],
            "result": row.get("result"), "plan": payload.get("plan"),
        }
        if "mine" in row:
            view["mine"] = row["mine"]
        return view

    def list_slips(self, uid: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        self.user(uid)
        self._grade_now()
        try:
            # Every one: the app's results (net, ROI) and Upcoming add up this list, so a cap would drop bets
            rows = self.store.slips(uid, status=status, limit=None)
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return [self._slip_view(row) for row in rows]

    def get_slip(self, uid: str, slip_id: int) -> Dict[str, Any]:
        row = self.store.slip(uid, slip_id)
        if row is None:
            raise NotFound("No such bet")
        return self._slip_view(row)

    def delete_slip(self, uid: str, slip_id: int) -> Dict[str, bool]:
        """Delete a bet before its games start; settled bets and a plan's days stay in the history."""
        store = self.store
        row = store.slip(uid, slip_id)
        if row is None:
            raise NotFound("No such bet")
        if not store.delete_slip(uid, slip_id, now=self._now()):
            if row["status"] in ("settled", "void"):
                raise CallRejected("Settled bets stay in your history.")
            if (row.get("payload") or {}).get("plan"):
                raise CallRejected("This bet is a day of a rollover plan, so it stays with the plan.")
            raise CallRejected("This bet's games have started, so it stays until it settles.")
        return {"ok": True}

    def slip_by_share(self, code: str, viewer: Optional[str] = None) -> Dict[str, Any]:
        """A shared slip as a request body the app can price again: {type, stake, k, system, legs: [ref]}."""
        row = self.store.shared_slip(code, viewer)
        if row is None:
            raise NotFound("No slip with that code")
        payload = row.get("payload") or {}
        unit = payload.get("unit_stake") or _money(row["stake_cents"])
        return {"type": row["type"], "stake": float(Decimal(unit)), "k": row.get("k"), "system": row.get("system"),
                "legs": [_ref_of(leg) for leg in row.get("legs") or []],
                "labels": [leg["label"] for leg in row.get("legs") or []],
                "share_code": row.get("share_code"), "mine": row.get("mine", False)}

    # --- grading saved bets and plan days (called from service.grade_due, inside its lease) ---------

    def _grade_now(self) -> None:
        """Settle finished bets before showing them (at most once a minute from page loads)."""
        now = self._now()
        if now - self._graded_at < GRADE_EVERY:
            return
        self._graded_at = now
        try:
            self.service.grade_due()
        except Exception:
            log.exception("Grading finished games failed")

    def grade_due(self, cutoff: str, give_up: str) -> int:
        """Settle saved bets whose games are over; returns how many were settled.

        ``cutoff``: legs that started after it are still being played. ``give_up``: a leg with no
        result this long after its start is voided (its stake rides on, like a push).
        """
        store = self._store if self._store is not None else self.service.store
        if store is None:
            return 0
        finals: Dict[Tuple[str, str], Dict[str, Any]] = {}
        settled = 0
        for slip in store.open_slips(started_before=cutoff):
            try:
                settled += self._grade_slip(store, slip, cutoff, give_up, finals)
            except Exception:
                log.exception("Couldn't grade bet %s", slip["id"])
        return settled

    def _grade_slip(self, store: Store, slip: Mapping[str, Any], cutoff: str, give_up: str,
                    finals: Dict[Tuple[str, str], Dict[str, Any]]) -> int:
        payload = slip.get("payload") or {}
        events = payload.get("events") or {}
        results: List[Optional[Dict[str, Any]]] = []
        for leg in slip["legs"]:
            if leg.get("result"):
                results.append(leg["result"])
                continue
            start = leg.get("start")
            if start is not None and start >= cutoff:
                results.append(None)  # still being played
                continue
            final = self._final(leg, events, finals) if leg["market"] in GAME_MARKETS else {"state": "pending"}
            if final["state"] == "final":
                event = events.get(str(leg["event_id"])) or {}
                results.append({"result": _outcome(leg, final, event), "home": final["home"], "away": final["away"]})
            elif final["state"] == "void":
                results.append({"result": "void", "reason": final.get("status") or "called off"})
            elif start is None or start < give_up:
                results.append({"result": "void", "reason": "no result"})
            else:
                results.append(None)
        grade = tickets.grade(results, payload)
        if grade["status"] == "pending":
            store.set_leg_results(slip["id"], results)
            return 0
        store.settle_slip(slip["id"], "void" if grade["status"] == "void" else "settled", grade, leg_results=results,
                          now=self._now())
        plan = payload.get("plan")
        if plan:
            self._settle_plan_day(slip["user"], plan["id"], plan["day"], grade)
        return 1

    def _final(self, leg: Mapping[str, Any], events: Mapping[str, Any],
               finals: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
        """The leg's game as ESPN has it (final, void or pending), looked up once per grading run."""
        key = (leg["league"], str(leg["event_id"]))
        if key not in finals:
            info = events.get(str(leg["event_id"])) or {}
            home, away = info.get("home"), info.get("away")
            if not (home and away) and " @ " in (leg.get("game") or ""):
                away, home = leg["game"].split(" @ ", 1)
            if not (home and away and leg.get("start")):
                finals[key] = {"state": "pending"}
            else:
                finals[key] = self.service.final({"league": leg["league"], "home": home, "away": away,
                                                  "start": leg["start"], "espn_id": info.get("espn_id")})
        return finals[key]

    # --- rollover plans -----------------------------------------------------------------------------

    def rollover_compare(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """The four strategies side by side (engine/rollover.compare) with the honest sentences."""
        start = _amount(body.get("start"), "start", MAX_STAKE)
        try:
            comparison = rollover.compare(start, body.get("days"), Decimal(str(body.get("min_odds"))),
                                          body.get("legs_per_day") or 1, currency=CURRENCY)
        except (ValueError, InvalidOperation) as exc:
            raise CallRejected(str(exc)) from None
        comparison["sentences"] = {
            "summary": comparison["summary"],
            "requirement": comparison["requirement"],
            "cost": {name: rollover.cost_sentence(start, Decimal(entry["market"]["expected_end"]), CURRENCY)
                     for name, entry in comparison["strategies"].items()},
        }
        return comparison

    def create_plan(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Start a plan (paper by default) after the comparison; one plan runs at a time, and a stopped
        plan's day still in play counts as running until its result is in."""
        with self._writing(uid):  # so two taps can't both find no plan running
            return self._create_plan(uid, body)

    def _create_plan(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        self.ready(uid, picks=True)
        store = self.store
        start = _amount(body.get("start"), "start", MAX_STAKE, least=PLAN_MIN_START)
        strategy = body.get("strategy") or rollover.DEFAULT_STRATEGY
        mode = body.get("mode") or "paper"
        if strategy not in rollover.STRATEGIES:
            raise CallRejected(f"strategy is one of {', '.join(rollover.STRATEGIES)}")
        # One person's plans: for a password browser, every password browser's (see gate_plans)
        running = self.gate_plans(uid, status="active", limit=1)
        if running:
            where = "" if running[0]["user"] == uid else " in another browser signed in with the password"
            raise CallRejected(f"You already have a plan running{where}: finish or stop it before starting another.")
        for stopped in self.gate_plans(uid, status="stopped", limit=10):
            # Stopping mid-day mustn't skip the break that day's loss would bring
            days = self._synced(stopped["user"], stopped).get("days_played") or []
            if days and days[-1].get("result") is None:
                where = "" if stopped["user"] == uid else " in another browser signed in with the password"
                raise CallRejected(f"Day {days[-1]['day']} of the plan you stopped{where} is still in play: a new "
                                   "plan can start once its result is in.")
        user = self.user(uid)  # again: settling that day may have just ended a run
        failed_at = (user.get("settings") or {}).get(PLAN_FAILED_SETTING)
        if failed_at and self._now() - failed_at < PLAN_RESTART_HOURS * 3600:
            raise CallRejected(f"Your last plan's run ended recently, so take a break: new plans open again "
                               f"{self._when(failed_at + PLAN_RESTART_HOURS * 3600)}.")
        cents = _cents(start)
        if mode == "tracked":
            self.check_budget(uid, user, cents, "This plan's start")
        elif user.get("weekly_budget_cents") is not None and cents > user["weekly_budget_cents"]:
            # Play money, but a paper plan still stays inside the real weekly limit
            raise CallRejected(f"This plan's start of {_say(cents)} is more than your weekly budget "
                               f"({_say(user['weekly_budget_cents'])}).")
        comparison = self.rollover_compare(body)
        try:
            plan = store.create_plan(uid, cents, body.get("days"), Decimal(str(body.get("min_odds"))),
                                     body.get("legs_per_day") or 1, strategy, payload=comparison, mode=mode,
                                     now=self._now())
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return self._plan_view(uid, plan, detail=True)

    def _plan(self, uid: str, plan_id: int) -> Dict[str, Any]:
        plan = self.store.plan(uid, plan_id)
        if plan is None:
            raise NotFound("No such plan")
        return self._synced(uid, plan)

    def _synced(self, uid: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """The plan with any day whose bet has settled filled in (in case grading stopped halfway)."""
        changed = False
        for day in plan.get("days_played") or []:
            if day.get("result") is None and day.get("slip_id"):
                slip = self.store.slip(uid, day["slip_id"])
                if slip and slip["status"] in ("settled", "void") and slip.get("result"):
                    try:
                        self._settle_plan_day(uid, plan["id"], day["day"], slip["result"])
                    except Exception:  # the plan must stay viewable (and stoppable) whatever went wrong
                        log.exception("Couldn't settle day %s of plan %s", day["day"], plan["id"])
                        continue
                    changed = True
        return self.store.plan(uid, plan["id"]) if changed else plan

    def list_plans(self, uid: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        self.user(uid)
        self._grade_now()
        try:
            plans = self.store.plans(uid, status=status, limit=None)  # every one, as for list_slips
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return [self._plan_view(uid, self._synced(uid, plan)) for plan in plans]

    def get_plan(self, uid: str, plan_id: int) -> Dict[str, Any]:
        self._grade_now()
        return self._plan_view(uid, self._plan(uid, plan_id), detail=True)

    def plan_day(self, uid: str, plan_id: int, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Play the plan's next day on the person's picks: the stake comes from the strategy and the plan's
        money, the day's bet is saved and linked, and a day needs the previous day's result first."""
        with self._writing(uid):  # from "the previous day has a result" to the day being recorded
            self._play_day(uid, plan_id, body)
        return self.get_plan(uid, plan_id)  # outside the lock: it may start a grading run

    def _play_day(self, uid: str, plan_id: int, body: Mapping[str, Any]) -> None:
        user = self.ready(uid, picks=True)
        plan = self._plan(uid, plan_id)
        if plan["status"] != "active":
            raise CallRejected(f"This plan is {plan['status']}: start a new one to play again.")
        days = plan.get("days_played") or []
        if days and days[-1].get("result") is None:
            raise CallRejected(f"Day {days[-1]['day']} is still in play: the next day opens once its result is in.")
        if plan["day"] >= plan["days"]:
            raise CallRejected("Every day of this plan has been played.")
        run = replay(plan)
        stake = next_stake(plan, run)
        if run.ended or stake < 1:
            raise CallRejected("This run has ended.")
        if stake > MAX_CENTS:
            raise CallRejected(f"This plan's stake of {_say(stake)} is more than the app can record, so it can't "
                               "play another day. Stopping it keeps the result.")
        refs = list(body.get("legs") or [])
        if not 1 <= len(refs) <= plan["legs_per_day"]:
            most = plan["legs_per_day"]
            raise CallRejected(f"This plan takes {'1 pick' if most == 1 else f'1 to {most} picks'} a day.")
        # The plan's own stake, not a request's: a winning run may stake more than the request cap allows
        priced, found = self._price_refs(refs, "single" if len(refs) == 1 else "multi", _dollars(stake))
        need = Decimal(plan["min_odds"])
        combined = math.prod((leg.price for _, leg, _ in found), start=Decimal(1))
        # Checked at the odds the person is shown (2 decimals, half up, as the priced ticket says), so a
        # multi at 1.995 shown as 2.00 meets a 2.00 plan rather than being refused at "2.00"
        shown = combined.quantize(CENTS, rounding=ROUND_HALF_UP)
        if found and not priced["removed"] and shown < need:
            raise CallRejected(f"Today's pick pays {shown}; this plan needs odds of {need} or more. "
                               f"Choose a pick at {need} or above.")
        # The plan's money after a win must stay within what the app can record
        if found and run.bankroll - stake + _cents(_dollars(stake) * combined) > MAX_CENTS:
            raise CallRejected(f"A win at these odds would take this plan past {_say(MAX_CENTS)}, more than the app "
                               f"can record. Choose shorter odds (still {need} or more), or stop the plan: stopping "
                               "keeps the result.")
        day = plan["day"] + 1
        slip = self._save(uid, user, priced, found,
                          plan={"id": plan["id"], "day": day, "mode": plan["mode"], "strategy": plan["strategy"]})
        recorded = self.store.record_plan_day(uid, plan["id"], stake, bankroll_cents=run.bankroll, slip_id=slip["id"],
                                              now=self._now())
        if recorded is None:
            self.store.delete_slip(uid, slip["id"], now=self._now())  # not linked yet, and its games haven't started
            raise CallRejected("This plan changed while the day was being saved: reload it and try again.")

    def stop_plan(self, uid: str, plan_id: int) -> Dict[str, Any]:
        """Stop a plan (a day already in play still settles and is recorded, and a loss on it still
        brings the break before a new plan)."""
        with self._writing(uid):
            plan = self._plan(uid, plan_id)
            if plan["status"] != "active" or not self.store.finish_plan(uid, plan_id, "stopped"):
                raise CallRejected(f"This plan is already {plan['status']}.")
        return self.get_plan(uid, plan_id)

    def _settle_plan_day(self, uid: str, plan_id: int, day_no: int, grade: Mapping[str, Any]) -> None:
        """Record a day's result and the money after it; end the plan when the run fails or every day is played.

        A day that was in play when the person stopped the plan counts too: if it ends the run, the break
        before a new plan applies as if the plan had still been running."""
        with self._writing(uid):  # the grader and a page load may both settle the same day
            store = self.store
            plan = store.plan(uid, plan_id)
            day = next((d for d in (plan or {}).get("days_played") or [] if d["day"] == day_no), None)
            if plan is None or day is None or day.get("result") is not None or grade.get("returns") is None:
                return
            before = replay(plan)
            run, banked, rolled = play_day(plan, before, day["stake_cents"], _cents(grade["returns"]))
            result = {"status": grade.get("status"), "returns": grade.get("returns"), "profit": grade.get("profit"),
                      "slip_id": day.get("slip_id")}
            # Past what the store can hold ($1bn; playing a day refuses a win that would get there): the
            # result is still recorded, the money columns stop at the cap and the plan stops there
            capped = max(run.bankroll, banked, rolled) > MAX_CENTS
            store.settle_plan_day(uid, plan_id, day_no, result, bankroll_cents=min(run.bankroll, MAX_CENTS),
                                  banked_cents=min(banked, MAX_CENTS), rolled_cents=min(rolled, MAX_CENTS))
            if run.ended and plan["status"] in ("active", "stopped"):
                self._fail_plan(uid, plan)
            elif plan["status"] == "active" and run.days >= plan["days"]:
                store.finish_plan(uid, plan_id, "done")
            elif plan["status"] == "active" and capped:
                store.finish_plan(uid, plan_id, "stopped")

    def _fail_plan(self, uid: str, plan: Mapping[str, Any]) -> None:
        """End a run that failed: no automatic restart, a break before the next plan, and repeated
        failures logged as a warning sign for the owner. A stopped plan keeps its status. The break and the
        count cover every password browser at once (see gate_plans); the warning is logged as the browser's."""
        store = self.store
        if plan["status"] == "active":
            store.finish_plan(uid, plan["id"], "failed")
        now = self._now()
        self.gates(uid, now)  # a password browser's shared row exists before it is written
        store.extend_setting(gates_id(uid), PLAN_FAILED_SETTING, now)
        recent = [p for p in self.gate_plans(uid, status=("failed", "stopped"), limit=20)
                  if p["created_at"] >= now - WEEK and (p["status"] == "failed" or replay(p).ended)]
        if len(recent) >= REPEATED_FAILURES:
            store.add_rg_event(uid, "failed_rollovers", {"plans": len(recent), "plan_id": plan["id"]},
                               action="break_suggested", now=now)

    def _plan_view(self, uid: str, plan: Mapping[str, Any], detail: bool = False) -> Dict[str, Any]:
        run = replay(plan)
        days = plan.get("days_played") or []
        waiting = bool(days) and days[-1].get("result") is None
        status = plan["status"]
        upcoming = None
        if status == "active" and not waiting and plan["day"] < plan["days"] and not run.ended:
            upcoming = {"day": plan["day"] + 1, "stake": _money(next_stake(plan, run)), "min_odds": plan["min_odds"],
                        "max_legs": plan["legs_per_day"]}
        played = []
        for day in days:
            entry = {"day": day["day"], "slip_id": day.get("slip_id"), "stake": _money(day["stake_cents"]),
                     "result": day.get("result"), "banked": _money(day.get("banked_cents") or 0),
                     "rolled": _money(day.get("rolled_cents") or 0), "created_at": _iso(day.get("created_at"))}
            if detail and day.get("slip_id"):
                slip = self.store.slip(uid, day["slip_id"])
                entry["slip"] = self._slip_view(slip) if slip else None
            played.append(entry)
        view = {
            "id": plan["id"], "status": status, "mode": plan["mode"], "strategy": plan["strategy"],
            "title": rollover.TITLES.get(plan["strategy"], plan["strategy"]), "rule": rollover.RULES.get(plan["strategy"]),
            "start": _money(plan["start_cents"]), "days": plan["days"], "min_odds": plan["min_odds"],
            "legs_per_day": plan["legs_per_day"], "created_at": _iso(plan.get("created_at")),
            "day": plan["day"], "bankroll": _money(run.bankroll), "banked": _money(run.banked),
            "profit": _money(run.bankroll - plan["start_cents"]),
            "in_play": _money(days[-1]["stake_cents"]) if waiting else None,
            "next": upcoming, "days_played": played,
            "message": self._plan_message(plan, run, waiting, upcoming),
        }
        if status == "failed" or (status == "stopped" and run.ended):
            view["suggestion"] = BREAK_SUGGESTION
        if detail:
            view["comparison"] = plan.get("payload")
        return view

    @staticmethod
    def _plan_message(plan: Mapping[str, Any], run: Run, waiting: bool, upcoming: Optional[Mapping[str, Any]]) -> str:
        days, status = plan["days"], plan["status"]
        start = plan["start_cents"]
        if run.ended == "lost" and plan["strategy"] == "bank_half":
            return f"Day {run.days} lost, so the run is over. {_say(run.banked)} banked along the way is kept."
        if run.ended == "lost":
            return f"Day {run.days} lost, so the run is over and the stake is gone."
        if run.ended == "below_half":
            return (f"The bankroll fell to {_say(run.bankroll)}, below half the {_say(start)} start, "
                    "so the plan stops as its rule says.")
        if run.ended == "broke":
            return f"The bankroll ({_say(run.bankroll)}) can't cover another flat stake, so the plan stops."
        if status == "done":
            return f"Finished: all {days} days played, ending with {_say(run.bankroll)} from {_say(start)}."
        if status == "stopped":
            return f"Stopped after {plan['day']} day{'s' if plan['day'] != 1 else ''} with {_say(run.bankroll)}."
        if waiting:
            return f"Day {plan['day']} is in play. The next day opens once its result is in."
        if upcoming:
            return (f"Day {upcoming['day']} of {days}: the plan's stake is {_say(_cents(upcoming['stake']))} on picks "
                    f"at {plan['min_odds']} or more. Nothing is placed until you choose the day's picks.")
        return f"Day {plan['day']} of {days}."

    # --- players ------------------------------------------------------------------------------------

    def player_search(self, q: str, league: Optional[str] = None) -> List[Dict[str, Any]]:
        found = self.players.search(q, league)
        return [{"id": p["id"], "name": p.get("name"), "team": p.get("team"), "team_abbr": p.get("team_abbr"),
                 "position": p.get("position"), "league": p.get("league"), "headshot": p.get("headshot")}
                for p in found.get("players") or []]

    def player(self, league: str, athlete_id: str) -> Dict[str, Any]:
        """A player page: ESPN's card, DraftKings' props for the next game with hit rates on each line, and
        for the NBA on game day the model's projection with its chance of going over each line.

        No gate here (the API route and the assistant apply their own before asking)."""
        players = self.players
        card = players.card(league, athlete_id)
        if card.get("name") is None:
            raise NotFound(card.get("reason") or "No such player")
        season = players.gamelog(league, athlete_id, last=100).get("rows") or card.get("games") or []
        props: List[Dict[str, Any]] = []
        props_info: Dict[str, Any] = {"as_of": None, "stale": False, "book": None, "reason": None}
        game = card.get("next_game") or {}
        if game.get("event_id"):
            found = players.props(league, game["event_id"])
            props = list((found.get("players") or {}).get(str(card["id"]), []))
            props_info = {"as_of": found.get("as_of"), "stale": found.get("stale", False), "book": found.get("book"),
                          "reason": found.get("reason") if not props else None}
            if not props and not props_info["reason"]:
                props_info["reason"] = "No props posted for this player yet."
        else:
            props_info["reason"] = "No upcoming game listed, so no props."
        row, projection, note = self._projection(league, card)
        return {
            **card,
            "props": [self._prop_view(prop, season, row) for prop in props],
            "props_info": props_info,
            "projection": projection,
            "projection_note": note,
        }

    def _projection(self, league: str, card: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]],
                                                                          Optional[Dict[str, Any]], Optional[str]]:
        """(the model's row for this player, its view, or why there is none). NBA only, on game day."""
        if league not in AI_LEAGUES:
            return None, None, "The AI projects NBA players for now."
        game = card.get("next_game") or {}
        when = _parse_time(game.get("date"))
        if when is None:
            return None, None, "No game listed for this player."
        today = datetime.datetime.fromtimestamp(self._now(), ESPN_DAY_TZ).date()
        if when.astimezone(ESPN_DAY_TZ).date() != today:
            return None, None, "The AI projects players on game day."
        try:
            board = self.service.games(league)
            match = next((c for c in board["games"]
                          if any(_same_team(c[side], card.get("team")) for side in ("home", "away"))
                          and abs((_parse_time(c["start"]) - when).total_seconds()) < SAME_GAME_WINDOW), None)
            if match is None:
                return None, None, "Today's game has no odds yet, so there's no projection."
            _, _, projections = self.service.detail(league, match["id"])
        except NotFound:
            return None, None, "Today's game is no longer on the board."
        except Exception as exc:
            log.warning("No projection for %s: %s", card.get("id"), type(exc).__name__)
            return None, None, "The projection isn't available right now."
        row = projections.get(str(card["id"]))
        if row is None:
            return None, None, "The model doesn't project this player today (out, or too few minutes)."
        minutes = row.get("minutes")
        stats = {}
        for stat in PROJECTED:
            mean = _mean(row, stat)
            stats[stat] = {"mean": round(mean, 1), "low": _quantile(stat, mean, minutes, 0.10),
                           "high": _quantile(stat, mean, minutes, 0.90)}
        view = {"event_id": match["id"], "game": f"{match['away']} @ {match['home']}", "start": match["start"],
                "minutes": minutes, "status": row.get("status"), "stats": stats,
                "method": "The NBA model's projection for this game (minutes, usage, pace and injuries); low and "
                          "high are the 10th and 90th percentiles of its spread."}
        return row, view, None

    @staticmethod
    def _prop_view(prop: Mapping[str, Any], season: Sequence[Mapping[str, Any]],
                   row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        view = {key: prop.get(key) for key in ("market", "stat", "name", "kind", "line", "open_line", "sides_known",
                                               "label", "updated")}
        view.update(over=_odds(prop.get("over_price")), under=_odds(prop.get("under_price")),
                    yes=_odds(prop.get("yes_price")))
        line = prop.get("line")
        view["hit_rates"] = hit_rates(season, prop["stat"], line) if prop.get("stat") and line is not None else None
        view["ai"] = None
        stat = PROP_STATS.get(prop.get("stat") or "")
        if row is not None and stat and line is not None:
            mean = _mean(row, stat)
            over, under, push = calls.stat_probabilities(stat, mean, float(line), row.get("minutes"))
            ai: Dict[str, Any] = {"mean": round(mean, 1), "over": round(over, 4), "under": round(under, 4),
                                  "push": round(push, 4), "over_one_in": one_in(over), "label": chance_label(over),
                                  "fair_over": round((1 - push) / over, 2) if over > 0 else None,
                                  "fair_under": round((1 - push) / under, 2) if under > 0 else None,
                                  "edge_over": None, "edge_under": None}
            for side, chance in (("over", over), ("under", under)):
                price = prop.get(f"{side}_price")
                if price is not None:
                    ai[f"edge_{side}"] = round(chance * float(price) + push - 1, 4)
            view["ai"] = ai
        return view

    # --- the AI chat (ai/assistant.py does the work) ----------------------------------------------

    def chat(self, uid: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        if self.assistant is None:
            raise CallRejected(NO_CHAT)
        self.user(uid)
        return self.assistant.reply(uid, thread_id=body.get("thread_id"), message=body.get("message"),
                                    request=body.get("request"), slip=body.get("slip"))

    def threads(self, uid: str) -> List[Dict[str, Any]]:
        self.user(uid)
        return [{**thread, "created_at": _iso(thread.get("created_at")), "last_at": _iso(thread.get("last_ts"))}
                for thread in self.store.threads(uid)]

    def messages(self, uid: str, thread_id: int) -> List[Dict[str, Any]]:
        if self.store.thread(uid, thread_id) is None:
            raise NotFound("No such chat")
        return [{**message, "at": _iso(message.get("ts"))} for message in self.store.messages(uid, thread_id)]

    # --- pasted bet slips: read, matched to the boards, priced at the person's own odds --------------

    def read_pasted(self, uid: str, extraction: Any, source: str, text: Optional[str] = None,
                    thread_id: Optional[int] = None) -> Dict[str, Any]:
        """Keep a slip someone pasted, as read (by the AI or by basic_parse), with each leg matched to the games
        and prices the app knows, and return its PastedView.

        The age and break gates apply (not the pause on picks: reading and the code's numbers stay open then,
        as on the Build screen). What was read is untrusted: it is checked and cut to size here, its odds
        and amounts are parsed by code, and at most PASTE_LOADS leagues not already cached are loaded for it.
        """
        user = self.ready(uid)
        if source not in pasted.SOURCES:
            raise CallRejected(f"source is one of {', '.join(pasted.SOURCES)}")
        slip = self._checked_slip(extraction)
        if thread_id is not None and self.store.thread(uid, thread_id) is None:
            raise NotFound("That chat doesn't exist")
        kept = pasted.clean_text(text)[:pasted.MAX_PASTE_CHARS] if text and str(text).strip() else None
        matches = self._match_slip(user, slip)
        try:
            pasted_id = self.store.save_pasted(uid, source, slip.model_dump(mode="json"), matches, text=kept,
                                               thread_id=thread_id, now=self._now())
        except ValueError as exc:
            raise CallRejected(str(exc)) from None
        return self._paste_state(self.store.pasted(uid, pasted_id), refresh=False)[2]

    def read_basic(self, uid: str, text: str, thread_id: Optional[int] = None) -> Dict[str, Any]:
        """The free, code-only reader (engine/pasted.basic_parse) and read_pasted(source="basic"). Refused with
        the format it reads when no bet is found."""
        self.ready(uid)
        if not isinstance(text, str) or not text.strip():
            raise CallRejected("Paste the text of your bet slip")
        if len(text) > pasted.MAX_PASTE_CHARS:
            raise CallRejected(f"Paste up to {pasted.MAX_PASTE_CHARS:,} characters of the slip")
        slip = pasted.basic_parse(text)
        if slip is None:
            raise CallRejected(pasted.BASIC_FORMAT_HELP)
        return self.read_pasted(uid, slip, "basic", text=text, thread_id=thread_id)

    def read_free(self, uid: str, text: str, thread_id: Optional[int] = None) -> Dict[str, Any]:
        """POST /api/pasted: the free reader, then the code's analysis: {pasted, analysis} and, after a warning
        sign in the pasted text, the support message ("support"). The chat's warning-sign rules run on pasted
        text too, first: a pasted note about losses pauses picks like a chat message does."""
        support = self._screen_paste(uid, text, thread_id)
        view = self.read_basic(uid, text, thread_id=thread_id)
        # No leg with odds of its own (a same-game parlay): what was read, and the view's warning says why
        priced = any(leg["odds"] is not None for leg in view["legs"])
        result: Dict[str, Any] = {"pasted": view, "analysis": self.analyse_pasted(uid, view["id"]) if priced else None}
        if support:
            result["support"] = support
        return result

    def chat_slip(self, uid: str, thread_id: Optional[int] = None, text: Optional[str] = None,
                  image: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/chat/slip: a pasted slip (text and/or a base64 screenshot) for the AI to read and talk
        about (assistant.read_slip). The screenshot is checked here (base64, size, a JPEG, PNG or WebP by its
        first bytes) and is never stored."""
        if self.assistant is None:
            raise CallRejected(NO_CHAT)
        text = text if isinstance(text, str) and text.strip() else None
        if text is not None and len(text) > pasted.MAX_PASTE_CHARS:
            raise CallRejected(f"Paste up to {pasted.MAX_PASTE_CHARS:,} characters of the slip")
        data = media_type = None
        if image:
            try:
                data, media_type = pasted.decode_image(image)
            except ValueError as exc:
                raise CallRejected(str(exc)) from None
        if text is None and data is None:
            raise CallRejected("Paste the text of your bet slip or add a screenshot of it")
        self.user(uid)
        return self.assistant.read_slip(uid, thread_id=thread_id, text=text, image=data, media_type=media_type)

    def pasted_view(self, uid: str, pasted_id: int) -> Dict[str, Any]:
        """A pasted slip again, its matched legs re-priced from the boards already cached (never a new load).
        Someone else's id is a NotFound before any gate, so nothing about it shows."""
        row = self._pasted_row(uid, pasted_id)
        self.ready(uid)
        return self._paste_state(row)[2]

    def analyse_pasted(self, uid: str, pasted_id: int, drop: Iterable[int] = (), stake: Any = None,
                       ticket_type: Optional[str] = None, k: Optional[int] = None,
                       system: Optional[str] = None) -> Dict[str, Any]:
        """A pasted slip priced at the person's own odds, all in code: the ticket (engine/tickets.price_ticket
        with each leg's chance from the board, the model or an estimate), facts about each leg and other ways
        to play it. ``drop`` holds 1-based legs to leave out, ``stake`` is per bet (default: the slip's own, or
        10) and ``ticket_type`` defaults to the slip's kind of bet. Legs whose odds couldn't be read are left
        out with a warning. While picks are paused the numbers stay, without alternatives."""
        row = self._pasted_row(uid, pasted_id)
        user = self.ready(uid)
        slip, entries, view = self._paste_state(row)
        count = len(view["legs"])
        dropped = self._paste_drop(drop, count)
        kept = [leg for leg in view["legs"] if leg["index"] not in dropped and leg["odds"] is not None]
        if not any(leg["odds"] is not None for leg in view["legs"]):
            raise CallRejected(pasted.NO_LEG_ODDS)
        if not kept:
            raise CallRejected("Keep at least one leg whose odds were read")
        kind, k, system, type_note = self._paste_ticket_type(slip, len(kept), ticket_type, k, system)
        unit, stake_note = self._paste_stake(stake, view)
        legs = [self._paste_ticket_leg(row["id"], slip.legs[leg["index"] - 1], entries[leg["index"] - 1], leg,
                                       slip.bookmaker) for leg in kept]
        priced = self._priced(legs, kind, unit, k, system)
        for leg_view, leg in zip(priced["legs"], kept):
            leg_view["slip_index"] = leg["index"]
            if leg_view.get("start") == UNKNOWN_START:
                leg_view["start"] = None
        warnings = [f"Leg {leg['index']} ({leg['label']}): its odds couldn't be read, so it is left out."
                    for leg in view["legs"] if leg["odds"] is None and leg["index"] not in dropped]
        priced["warnings"] = warnings + [note for note in (type_note, stake_note) if note] + list(priced["warnings"])
        currency = view["currency"]
        cards = self._paste_cards(entries)
        by_index = {leg["index"]: leg_view for leg_view, leg in zip(priced["legs"], kept)}
        notes = [{"index": leg["index"], "label": leg["label"], "in_ticket": leg["index"] in by_index,
                  "notes": self._paste_notes(leg, entries[leg["index"] - 1], by_index.get(leg["index"]),
                                             cards.get(str((entries[leg["index"] - 1].get("ref") or {}).get("id"))),
                                             leg["index"] in dropped)}
                 for leg in view["legs"]]
        until = paused_until(user, self._now())
        alternatives: List[Dict[str, Any]] = []
        if until is None and priced["valid"]:
            alternatives = self._paste_alternatives(priced, legs, kept, entries, dropped, currency)
        result = {"pasted": view, "ticket": priced, "legs": notes, "alternatives": alternatives,
                  "drop": sorted(dropped), "stake": _text_money(unit), "ticket_type": priced["type"], "k": priced["k"],
                  "system": priced["system"], "currency": currency, "paused_until": _iso(until)}
        if until is not None:
            result["pause_note"] = (f"Picks are paused until {self._when(until)} after an earlier message about "
                                    "losses or money worries. The odds and the maths stay open.")
        return result

    def pasted_fact_sheet(self, analysis: Mapping[str, Any]) -> Dict[str, Any]:
        """Everything Claude may say about a pasted slip's analysis: Lab.fact_sheet of its ticket (each leg's
        notes matched by the ticket's own leg index), in the slip's currency, plus what was read from the slip
        (odds as written, lines, the stake and the stated return) and how each leg matched."""
        ticket = analysis["ticket"]
        by_slip = {item["index"]: item["notes"] for item in analysis["legs"]}
        legs = [{"index": view["index"], "notes": by_slip.get(view.get("slip_index"), [])} for view in ticket["legs"]]
        sheet = self.fact_sheet(ticket, legs, analysis["alternatives"])
        view = analysis["pasted"]
        sheet["currency"] = view.get("currency") or ""
        sheet["pasted"] = {key: view.get(key) for key in (
            "bookmaker", "bet_type", "stake", "total_stake", "stated_return", "computed_return", "return_basis",
            "return_check", "needs_confirmation", "confirm_reasons", "warnings")}
        sheet["pasted"]["legs"] = [
            {"index": leg["index"], "label": leg["label"], "odds_as_written": leg["odds_text"], "odds": leg["odds"],
             "line": leg["line"], "status": leg["match"]["status"], "game": leg["match"]["game"],
             "note": leg["match"]["note"], "chance_source": leg["chance_source"],
             "market_chance": leg["market_chance"], "ai_chance": leg["ai_chance"], "best_price": leg["best_price"],
             "better_price": leg["better_price"]}
            for leg in view["legs"]]
        sheet.update(drop=list(analysis["drop"]), stake=analysis["stake"], paused=analysis.get("paused_until") is not None)
        return sheet

    # --- pasted slips: reading what was read -----------------------------------------------------------

    @staticmethod
    def _checked_slip(extraction: Any) -> pasted.Extraction:
        """The read checked again in code (cut to size, odds and markets normalised): refused when it has no
        leg or more than MAX_LEGS."""
        if isinstance(extraction, pasted.Extraction):
            extraction = extraction.model_dump()
        if not isinstance(extraction, Mapping):
            raise CallRejected("Nothing was read from that slip")
        try:
            slip = pasted.Extraction.model_validate(dict(extraction))
        except ValidationError:
            raise CallRejected("That slip couldn't be read: try pasting its text again") from None
        slip.legs = [leg for leg in slip.legs if (leg.selection or leg.raw).strip()]
        if not slip.legs:
            raise CallRejected("No bets were found on that slip")
        if len(slip.legs) > pasted.MAX_LEGS:
            raise CallRejected(f"That slip has more than {pasted.MAX_LEGS} legs; the app reads up to "
                               f"{pasted.MAX_LEGS}.")
        return slip

    def _pasted_row(self, uid: str, pasted_id: Any) -> Dict[str, Any]:
        if isinstance(pasted_id, bool) or not isinstance(pasted_id, int):
            raise NotFound(NO_SUCH_PASTE)
        row = self.store.pasted(uid, pasted_id)
        if row is None:
            raise NotFound(NO_SUCH_PASTE)
        return row

    def _screen_paste(self, uid: str, text: Any, thread_id: Optional[int]) -> Optional[str]:
        """The chat's warning-sign rules on pasted text (ai/assistant.py's code rules): a strong sign pauses picks
        for a day and is logged for the owner, as in the chat; returns the support message, or None."""
        try:
            from ..ai.assistant import note_warning_sign, risk_flags, support_message
        except ImportError:  # the assistant's module is part of the app; without it there are no rules to run
            log.exception("The warning-sign rules couldn't be loaded")
            return None
        flags = risk_flags(str(text or "")[:pasted.MAX_PASTE_CHARS])
        if not flags:
            return None
        note_warning_sign(self, uid, flags, "pasted_slip", str(text), thread_id=thread_id, now=self._now())
        return support_message(flags)

    # --- pasted slips: matching -------------------------------------------------------------------------

    def _person_tz(self, user: Mapping[str, Any]) -> datetime.tzinfo:
        name = (user.get("settings") or {}).get(TIMEZONE_SETTING)
        try:
            return ZoneInfo(name) if name else self.service.tz
        except (ValueError, KeyError, OSError):
            return self.service.tz

    def _saved_odds(self) -> Dict[str, float]:
        """When each league's odds were last saved by either process (no quota), or {} without a store."""
        store = getattr(self.service.odds, "store", None)
        try:
            return dict(store.odds_fetched_at()) if store is not None and hasattr(store, "odds_fetched_at") else {}
        except Exception:
            log.warning("Couldn't read when the odds were saved")
            return {}

    def _quota(self) -> Tuple[Optional[int], int]:
        odds = self.service.odds
        try:
            remaining = (odds.quota or {}).get("remaining")
        except Exception:
            remaining = None
        return remaining, int(getattr(odds, "reserve", 50) or 0)

    def _odds_cached(self, league: str, saved: Optional[Mapping[str, float]] = None) -> bool:
        """Whether a league's odds can be shown without a fetch (no quota spent): loaded within the odds
        service's cache time, saved that recently by the other process, or saved at all while the quota is
        below its reserve (saved odds are served then, never fetched)."""
        cards = getattr(self.service, "cache", None)
        age = cards.age(("games", league)) if cards is not None else None
        if age is not None and age < GAMES_CACHE_SECONDS:  # the games service's own copy of the board
            return True
        odds = self.service.odds
        ttl = float(getattr(odds, "odds_ttl", 7200) or 7200)
        now = time.time()  # the odds service's own clock
        loaded = odds.odds_loaded_at(league)
        if loaded is not None and now - loaded < ttl:
            return True
        fetched = (self._saved_odds() if saved is None else saved).get(league)
        if fetched is None:
            return False
        remaining, reserve = self._quota()
        return now - fetched < ttl or (remaining is not None and remaining < reserve)

    def _paste_loads(self) -> int:
        """How many uncached leagues one pasted slip may load: PASTE_LOADS, fewer when the quota would fall
        below the odds service's reserve."""
        remaining, reserve = self._quota()
        if remaining is None:
            return PASTE_LOADS
        return max(0, min(PASTE_LOADS, (remaining - reserve) // FETCH_COST))

    def _match_slip(self, user: Mapping[str, Any], slip: pasted.Extraction) -> List[Dict[str, Any]]:
        boards = _PasteBoards(self, user, self._paste_loads())
        tz = self._person_tz(user)
        now = self._now()
        matches = []
        for leg in slip.legs:
            odds = pasted.parse_odds(leg.odds_text)
            try:
                entry = self._match_leg(leg, odds, boards, now, tz)
            except Exception:
                log.exception("Couldn't match a pasted leg")
                entry = self._estimate(_paste_entry("not_found", "This leg couldn't be matched just now: its chance is "
                                                    "estimated from your odds.", _margin_kind(leg, None, None)), odds)
            if odds is None:
                entry["status"] = "no_odds"
                entry["note"] = "The odds for this leg couldn't be read, so it is left out of the numbers."
            matches.append(entry)
        return matches

    def _estimate(self, entry: Dict[str, Any], odds: Optional[Decimal]) -> Dict[str, Any]:
        """The person's price less a typical margin as the leg's chance (engine/pasted.estimated_chance)."""
        entry["ai_chance"] = None
        entry["push"] = 0.0
        entry["chance_source"] = "estimated"
        entry["market_chance"] = pasted.estimated_chance(odds, entry["kind"]) if odds is not None else None
        return entry

    def _match_leg(self, leg: pasted.ExtractedLeg, odds: Optional[Decimal], boards: "_PasteBoards", now: float,
                   tz: datetime.tzinfo) -> Dict[str, Any]:
        moment = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
        start = pasted.parse_start(leg.start_text, moment, tz)
        if leg.market == "player_prop":
            return self._match_prop(leg, odds, boards, now, start)
        kind = _margin_kind(leg, None, None)
        estimated = f"its chance is estimated from your odds less a typical {_margin_words(kind)} bookmaker margin"
        names = _leg_teams(leg)
        status = "unsupported" if leg.market == "other" else "not_found"
        if not names:
            return self._estimate(_paste_entry(status, f"The slip doesn't name this leg's game, so {estimated}.",
                                               kind), odds)
        leagues = boards.leagues_for(leg)
        for league in leagues:
            board = boards.get(league)
            card = self._paste_card(board, names, now, start) if board is not None else None
            if card is None:
                continue
            kind = _margin_kind(leg, league, card)
            if leg.market == "other":
                note = ("The app prices moneylines, spreads and totals only, so this market's chance is estimated "
                        f"from your odds less a typical {_margin_words(kind)} margin.")
                return self._estimate(_paste_entry("unsupported", note, kind, league, card, now), odds)
            return self._match_market(leg, odds, league, card, kind, now)
        why = boards.why_not(leagues)
        return self._estimate(_paste_entry(status, f"Not found on the app's boards{why}, so {estimated}.", kind,
                                           leagues[0] if len(leagues) == 1 else None), odds)

    def _paste_card(self, board: Mapping[str, Any], names: Sequence[str], now: float,
                    start: Optional[datetime.datetime]) -> Optional[Dict[str, Any]]:
        """The board's game a leg's team names point to: both teams when the slip names the game, else the one
        picked; within 36 hours of the slip's start time when it gives one, else the next game between them
        within a week (one in play counts). Ties go to the earliest start."""
        found = []
        for card in board.get("games") or []:
            begins = _parse_time(card.get("start"))
            if begins is None:
                continue
            if start is not None:
                if abs((begins - start).total_seconds()) > SAME_START:
                    continue
            elif not now - LIVE_SINCE <= begins.timestamp() <= now + NEXT_GAME:
                continue
            if len(names) >= 2:
                straight = (paste_team_score(names[0], card["away"]), paste_team_score(names[1], card["home"]))
                crossed = (paste_team_score(names[0], card["home"]), paste_team_score(names[1], card["away"]))
                pair = max(straight, crossed, key=lambda scores: (min(scores), sum(scores)))
                if min(pair) < TEAM_MATCH:
                    continue
                score = sum(pair) / 2
            else:
                score = max(paste_team_score(names[0], card["home"]), paste_team_score(names[0], card["away"]))
                if score < TEAM_MATCH:
                    continue
            found.append((-round(score, 3), begins, card))
        found.sort(key=lambda item: (item[0], item[1]))
        return found[0][2] if found else None

    @staticmethod
    def _team_side(selection: str, card: Mapping[str, Any]) -> Optional[str]:
        """Which side of the game a team picked on a slip is: "home", "away" or None when it is neither. A 1X2
        slip's "1" and "2" (and "Home win", "W2") name the home and away sides."""
        written = (selection or "").strip().lower()
        if written in pasted.HOME_WORDS:
            return "home"
        if written in pasted.AWAY_WORDS:
            return "away"
        home, away = paste_team_score(selection, card["home"]), paste_team_score(selection, card["away"])
        if max(home, away) < TEAM_MATCH:
            return None
        return "home" if home >= away else "away"

    def _match_market(self, leg: pasted.ExtractedLeg, odds: Optional[Decimal], league: str,
                      card: Mapping[str, Any], kind: str, now: float) -> Dict[str, Any]:
        """The leg's outcome on its game: moneyline -> h2h home/away/draw, spread -> the same team at the same
        point (sign and all), total -> over/under at the same point. Another line links to the game with its
        chance estimated."""
        game = f"{card['away']} @ {card['home']}"
        estimated = f"its chance is estimated from your odds less a typical {_margin_words(kind)} margin"
        if leg.market == "moneyline":
            market, side = "h2h", ("draw" if _is_draw(leg.selection) else self._team_side(leg.selection, card))
        elif leg.market == "spread":
            market, side = "spreads", self._team_side(leg.selection, card)
        else:
            market = "totals"
            side = leg.over_under or ("under" if (leg.selection or "").lower().startswith("u") else
                                      "over" if (leg.selection or "").lower().startswith("o") else None)
        if side is None:
            note = f"Couldn't tell which side of {game} \"{leg.selection}\" is, so {estimated}."
            return self._estimate(_paste_entry("not_found", note, kind, league, card, now), odds)
        offer = ((card.get("markets") or {}).get(market) or {}).get(side)
        if not offer or not offer.get("price"):
            what = {"h2h": "a draw" if side == "draw" else "a moneyline", "spreads": "a spread",
                    "totals": "a total"}[market]
            note = f"The board has {game} but not {what} for it right now, so {estimated}."
            return self._estimate(_paste_entry("not_found", note, kind, league, card, now), odds)
        point = offer.get("point")
        ref = {"league": league, "id": str(card["id"]), "market": market, "side": side, "point": point}
        if market != "h2h" and (leg.line is None or point is None or abs(float(point) - leg.line) > 1e-6):
            mine = "no line read" if leg.line is None else _line_words(market, side, leg.line)
            note = (f"The board has {_line_words(market, side, point)}; you have {mine}, so the chance at your line "
                    f"is estimated from your odds less a typical {_margin_words(kind)} margin.")
            entry = _paste_entry("line_differs", note, kind, league, card, now)
            entry["ref"] = ref
            entry["board"] = {"label": offer.get("label"), "point": point, "book": offer.get("book"),
                              "price": str(Decimal(str(offer["price"])).quantize(CENTS, rounding=ROUND_HALF_UP))}
            return self._estimate(entry, odds)
        entry = _paste_entry("matched", f"{game}, {self._when(_parse_time(card['start']).timestamp())}.", kind,
                             league, card, now)
        entry["ref"] = ref
        return self._board_chances(entry, card, offer, odds)

    def _board_chances(self, entry: Dict[str, Any], card: Mapping[str, Any], offer: Mapping[str, Any],
                       odds: Optional[Decimal]) -> Dict[str, Any]:
        """A matched leg's chances from the board (the market's margin-free chance, the model's when it has
        one) and the best price on offer."""
        modelled = card.get("prediction") is not None
        fair, ai = offer.get("fair"), offer.get("ai") if modelled else None
        if fair is None and ai is None:
            return self._estimate(entry, odds)
        entry["market_chance"] = fair
        entry["ai_chance"] = ai
        entry["push"] = (offer.get("push") or 0.0) if modelled else 0.0
        entry["chance_source"] = "model" if ai is not None else "market"
        best = Decimal(str(offer["price"]))
        # The board's best price is a price book's (data/odds_service.py), but a "better price" is only ever one a
        # person can bet at: never Pinnacle's or an exchange's
        if is_price_book(offer.get("book")):
            entry["best_price"] = {"decimal": str(best.quantize(CENTS, rounding=ROUND_HALF_UP)),
                                   "book": offer.get("book")}
            entry["better_price"] = odds is not None and best >= odds + BETTER_BY
        return entry

    def _match_prop(self, leg: pasted.ExtractedLeg, odds: Optional[Decimal], boards: "_PasteBoards", now: float,
                    start: Optional[datetime.datetime]) -> Dict[str, Any]:
        """A player prop: for the NBA, the player's next game, the model's chance of the over or under at the
        slip's line (on game day) and DraftKings' line on ESPN when it is the same line; otherwise estimated."""
        kind = "player_prop"
        estimated = f"its chance is estimated from your odds less a typical {_margin_words(kind)} margin"
        league, other = _PasteBoards.named_league(leg)
        prefixes = pasted.sport_prefixes(leg.sport_hint) or pasted.sport_prefixes(leg.raw)
        entry = _paste_entry("prop", "", kind, league)
        if other or league not in (None, NBA) or (league is None and prefixes and "basketball_" not in prefixes):
            entry["note"] = f"The app models NBA players only, so {estimated}."
            return self._estimate(entry, odds)
        stat = PASTE_STATS.get(leg.prop_stat or "")
        if stat is None or leg.line is None or leg.over_under is None:
            entry["note"] = ("The model covers NBA points, rebounds, assists, threes, steals, blocks, turnovers and "
                             f"points + rebounds + assists overs and unders, so {estimated}.")
            return self._estimate(entry, odds)
        try:
            player = self._paste_player(leg.selection)
        except Exception as exc:
            log.warning("Player search for a pasted prop failed (%s)", type(exc).__name__)
            entry["note"] = f"Player data isn't available right now, so {estimated}."
            return self._estimate(entry, odds)
        if player is None:
            entry["note"] = f"Couldn't find {leg.selection} among NBA players, so {estimated}."
            return self._estimate(entry, odds)
        entry["league"] = NBA
        entry["player"] = {"id": str(player["id"]), "name": player.get("name"), "team": player.get("team")}
        try:
            card = self.players.card(NBA, str(player["id"]))
        except Exception as exc:
            log.warning("Player card for a pasted prop failed (%s)", type(exc).__name__)
            card = {}
        game = card.get("next_game") or {}
        when = _parse_time(game.get("date"))
        if start is not None and when is not None and abs((when - start).total_seconds()) > SAME_START:
            entry["note"] = f"The slip's game isn't {player.get('name')}'s next game, so {estimated}."
            return self._estimate(entry, odds)
        entry.update(game=game.get("name"), start=_iso(when.timestamp()) if when else None,
                     live=bool(when and when.timestamp() <= now),
                     group=f"espn:{game['event_id']}" if game.get("event_id") else None)
        model_note = None
        row = None
        if card.get("name") and boards.get(NBA) is not None:
            row, projection, model_note = self._projection(NBA, card)
            if projection:
                entry.update(game=projection["game"], start=projection["start"], group=str(projection["event_id"]))
        elif card.get("name"):
            model_note = "The NBA odds aren't loaded right now, so there's no projection."
        if row is not None:
            mean = _mean(row, stat)
            over, under, push = calls.stat_probabilities(stat, mean, float(leg.line), row.get("minutes"))
            entry["ai_chance"] = over if leg.over_under == "over" else under
            entry["push"] = push
            entry["projection"] = {"stat": stat, "mean": round(mean, 1), "minutes": row.get("minutes")}
        self._prop_market(entry, card, game, stat, leg, odds)
        if entry["ai_chance"] is None and entry["market_chance"] is None:
            entry["note"] = f"{model_note or 'No projection or market line for this player.'} So {estimated}."
            return self._estimate(entry, odds)
        entry["chance_source"] = "model" if entry["ai_chance"] is not None else "market"
        parts = []
        if entry.get("projection"):
            parts.append(f"The NBA model projects {entry['projection']['mean']:g} {STAT_WORDS[stat]}")
        elif model_note:
            parts.append(model_note.rstrip("."))
        if entry["market_chance"] is not None:
            parts.append("DraftKings lists the same line")
        entry["note"] = "; ".join(parts) + "."
        return entry

    def _prop_market(self, entry: Dict[str, Any], card: Mapping[str, Any], game: Mapping[str, Any], stat: str,
                     leg: pasted.ExtractedLeg, odds: Optional[Decimal]) -> None:
        """The market's chance of a prop when DraftKings (on ESPN) lists the same line: its over and under
        prices with the margin taken out."""
        if not game.get("event_id") or not card.get("id"):
            return
        try:
            found = self.players.props(NBA, str(game["event_id"]))
        except Exception as exc:
            log.warning("Props for a pasted leg failed (%s)", type(exc).__name__)
            return
        for prop in (found.get("players") or {}).get(str(card["id"]), []):
            if prop.get("stat") != ESPN_PROP_STATS[stat] or prop.get("line") is None:
                continue
            if abs(float(prop["line"]) - float(leg.line)) > 1e-6 or not (prop.get("over_price") and prop.get("under_price")):
                continue
            over, under = remove_vig([float(prop["over_price"]), float(prop["under_price"])])
            entry["market_chance"] = over if leg.over_under == "over" else under
            price = Decimal(str(prop[f"{leg.over_under}_price"]))
            entry["best_price"] = {"decimal": str(price.quantize(CENTS, rounding=ROUND_HALF_UP)),
                                   "book": found.get("book") or "DraftKings"}
            entry["better_price"] = odds is not None and price >= odds + BETTER_BY
            return

    def _paste_player(self, name: str) -> Optional[Dict[str, Any]]:
        """The NBA player a slip names, by ESPN's search, when one name is a close enough match."""
        found = self.players.search(name, NBA).get("players") or []
        scored = [(paste_player_score(name, player.get("name")), player) for player in found if player.get("id")]
        scored = [(score, player) for score, player in scored if score >= TEAM_MATCH]
        return max(scored, key=lambda item: item[0])[1] if scored else None

    # --- pasted slips: the view and the numbers ------------------------------------------------------------

    def _paste_state(self, row: Mapping[str, Any], refresh: bool = True
                     ) -> Tuple[pasted.Extraction, List[Dict[str, Any]], Dict[str, Any]]:
        """(the slip as read, each leg's match, the PastedView). With ``refresh`` the matched legs take the
        boards' current prices when their league is cached; nothing is ever loaded for a view."""
        slip = pasted.Extraction.model_validate(row["extraction"])
        stored = list(row.get("matches") or [])
        now = self._now()
        saved = self._saved_odds() if refresh else {}
        boards: Dict[str, Optional[Dict[str, Any]]] = {}
        entries, legs = [], []
        for index, leg in enumerate(slip.legs, 1):
            odds = pasted.parse_odds(leg.odds_text)
            entry = dict(stored[index - 1]) if index - 1 < len(stored) else self._estimate(
                _paste_entry("not_found", "Not matched.", _margin_kind(leg, None, None)), odds)
            if refresh and entry.get("ref") and entry.get("status") in ("matched", "line_differs"):
                entry = self._paste_refresh(entry, leg, odds, boards, saved, now)
            entries.append(entry)
            legs.append(self._paste_leg_view(index, leg, odds, entry, row))
        view = {"id": row["id"], "thread_id": row.get("thread_id"), "created_at": _iso(row.get("created_at")),
                "source": row["source"], "bookmaker": slip.bookmaker, "bet_type": slip.bet_type, "k": slip.k,
                "system": slip.system, "legs": legs, "text": row.get("text"), "notes": slip.notes}
        view.update(self._paste_money(slip, legs))
        view.update(self._paste_checks(row, slip, legs, view))
        view.pop("stake_uneven", None)
        view.pop("stake_mismatch", None)
        view["matched_refs"] = [leg["match"]["ref"] for leg in legs
                                if leg["match"]["status"] == "matched" and not leg["match"]["live"]]
        leagues = sorted({leg["match"]["ref"]["league"] for leg in legs
                          if leg["match"]["ref"] and leg["match"]["status"] in ("matched", "line_differs")})
        view["odds_as_of"] = {league: _iso(self.service.odds.odds_loaded_at(league)) for league in leagues}
        return slip, entries, view

    def _paste_refresh(self, entry: Dict[str, Any], leg: pasted.ExtractedLeg, odds: Optional[Decimal],
                       boards: Dict[str, Optional[Dict[str, Any]]], saved: Mapping[str, float],
                       now: float) -> Dict[str, Any]:
        """A matched leg with the cached board's current price and chances (and a line that has moved since, or
        come to the slip's line, says so). A board that isn't cached keeps the match as it was read."""
        ref = entry["ref"]
        league = ref["league"]
        if league not in boards:
            boards[league] = self._board(league)[0] if self._odds_cached(league, saved) else None
        board = boards[league]
        card = next((c for c in (board or {}).get("games") or [] if str(c.get("id")) == ref["id"]), None)
        offer = (((card or {}).get("markets") or {}).get(ref["market"]) or {}).get(ref["side"])
        if not card or not offer or not offer.get("price"):
            return entry
        fresh = _paste_entry(entry["status"], entry["note"], entry["kind"], league, card, now)
        point = offer.get("point")
        fresh["ref"] = {**ref, "point": point}
        same = ref["market"] == "h2h" or (leg.line is not None and point is not None
                                          and abs(float(point) - leg.line) <= 1e-6)
        if same:
            fresh["status"] = "matched"
            if entry["status"] != "matched":
                fresh["note"] = f"{fresh['game']}, {self._when(_parse_time(card['start']).timestamp())}."
            return self._board_chances(fresh, card, offer, odds)
        mine = "no line read" if leg.line is None else _line_words(ref["market"], ref["side"], leg.line)
        moved = " now" if entry["status"] == "matched" else ""
        fresh["status"] = "line_differs"
        fresh["note"] = (f"The board{moved} has {_line_words(ref['market'], ref['side'], point)}; you have {mine}, so "
                         f"the chance at your line is estimated from your odds less a typical "
                         f"{_margin_words(entry['kind'])} margin.")
        fresh["board"] = {"label": offer.get("label"), "point": point, "book": offer.get("book"),
                          "price": str(Decimal(str(offer["price"])).quantize(CENTS, rounding=ROUND_HALF_UP))}
        return self._estimate(fresh, odds)

    @staticmethod
    def _paste_leg_view(index: int, leg: pasted.ExtractedLeg, odds: Optional[Decimal], entry: Mapping[str, Any],
                        row: Mapping[str, Any]) -> Dict[str, Any]:
        source = row["source"]
        verified = None if source == "basic" else (
            False if source == "image" else pasted.odds_in_text(leg.odds_text, row.get("text")))
        chance = lambda value: None if value is None else round(float(value), 4)  # noqa: E731
        return {
            "index": index, "raw": leg.raw, "label": _paste_label(leg), "selection": leg.selection,
            "market": leg.market, "line": leg.line, "prop_stat": leg.prop_stat, "over_under": leg.over_under,
            "event_text": leg.event_text, "start_text": leg.start_text, "sport_hint": leg.sport_hint,
            "odds_text": leg.odds_text, "odds": pasted.odds_view(odds) if odds is not None else None,
            "verified": verified,
            "match": {"status": entry["status"], "ref": entry.get("ref"), "game": entry.get("game"),
                      "start": entry.get("start"), "live": bool(entry.get("live")), "league": entry.get("league"),
                      "note": entry.get("note") or ""},
            "market_chance": chance(entry.get("market_chance")), "ai_chance": chance(entry.get("ai_chance")),
            "push": chance(entry.get("push")) or 0.0, "chance_source": entry.get("chance_source") or "estimated",
            "best_price": entry.get("best_price"), "better_price": bool(entry.get("better_price")),
        }

    @staticmethod
    def _slip_type(slip: pasted.Extraction, count: int) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
        """The slip's own kind of bet as engine/tickets prices it: (type, k, system, a note when it had to be
        assumed). "System 2/4" is every double of four legs (a round robin by 2s)."""
        kind, k, system, note = slip.bet_type, slip.k, None, None
        if kind == "system":
            system = pasted.system_key(slip.system)
            numbers = re.search(r"(\d)\s*/\s*(\d)", slip.system or "")
            if system is None and numbers:
                kind, k = "round_robin", int(numbers.group(1))
            elif system is None:
                kind, note = ("multi" if count > 1 else "single",
                              "The app doesn't know this system bet, so it is priced as a multi.")
        if kind == "round_robin":
            k = k or 2
        if kind == "unknown":
            kind = "multi" if count > 1 else "single"
            if count > 1:
                note = "The slip doesn't say what kind of bet it is, so it is priced as a multi (every leg must win)."
        return kind, k if kind == "round_robin" else None, system if kind == "system" else None, note

    def _paste_money(self, slip: pasted.Extraction, legs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """The slip's money as written (stake per bet, total, the bookmaker's stated return) and the code's
        own return at those odds and that stake."""
        stake, stake_currency = pasted.parse_amount(slip.stake_text)
        total, total_currency = pasted.parse_amount(slip.total_stake_text)
        stated, stated_currency = pasted.parse_amount(slip.potential_return_text)
        currency = slip.currency or stake_currency or total_currency or stated_currency
        kind, k, system, _ = self._slip_type(slip, len(legs))
        try:
            bets = len(tickets.combinations(kind, len(legs), k, system))
        except ValueError:
            bets = None
        uneven = False
        mismatch = None
        if stake is not None and total is not None and bets and abs(stake * bets - total) > CENTS:
            mismatch = {"stake": stake, "bets": bets, "total": total}  # a leg missed, or each-way: asked
        if stake is None and total is not None and bets:
            stake = _floor_cents(total / bets)
            uneven = stake * bets != total
        if total is None and stake is not None and bets:
            total = stake * bets
        payout = profit = None
        prices = [pasted.parse_odds(leg["odds_text"]) for leg in legs]
        if stake is not None and bets and all(price is not None for price in prices):
            payout = self._slip_payout(prices, kind, k, system, stake)
            profit = payout - stake * bets if payout is not None else None
        check, basis, computed = pasted.return_check(payout, stated), "payout", payout
        if check == "differs" and pasted.return_check(profit, stated) == "matches":
            check, basis, computed = "matches", "profit", profit  # US apps' "To win" is the profit
        return {"stake": _text_money(stake), "total_stake": _text_money(total), "currency": currency,
                "stated_return": _text_money(stated), "computed_return": _text_money(computed),
                "return_basis": basis, "return_check": check, "stake_uneven": uneven, "stake_mismatch": mismatch}

    def _slip_payout(self, prices: Sequence[Decimal], kind: str, k: Optional[int], system: Optional[str],
                     stake: Decimal) -> Optional[Decimal]:
        """What the slip pays if every leg wins, at the slip's odds and stake (engine/tickets' sums)."""
        if kind == "multi" or len(prices) == 1:
            return stake * math.prod(prices, start=Decimal(1))
        legs = [TicketLeg(key=f"leg{i}", league="pasted", event_id=f"leg{i}", market="pasted", side="pick", point=None,
                          price=price, label=f"Leg {i}", game="", start=UNKNOWN_START)
                for i, price in enumerate(prices, 1)]
        try:
            priced = tickets.price_ticket(legs, kind, stake, k=k, system=system)
        except ValueError:
            return None
        return Decimal(priced["max_payout"]) if priced.get("max_payout") is not None else None

    def _paste_checks(self, row: Mapping[str, Any], slip: pasted.Extraction, legs: Sequence[Mapping[str, Any]],
                      money: Mapping[str, Any]) -> Dict[str, Any]:
        """When the read is doubtful the person confirms it before any analysis: the bookmaker's return isn't
        the code's, some odds were unreadable, odds read from text aren't in that text, a matched leg's odds are
        far longer than the board's best (a combined or boosted price?), a multi has one leg, or the stake per
        bet doesn't add up to the total (a missed leg, or each-way). A slip with no odds on any leg (a
        same-game parlay) has nothing to confirm or price: a warning says why."""
        reasons, warnings = [], []
        symbol = money["currency"]
        if money["return_check"] == "differs":
            stated = _say_dollars(Decimal(money["stated_return"]), symbol)
            ours = _say_dollars(Decimal(money["computed_return"]), symbol) if money["computed_return"] else None
            reasons.append(f"Your bookmaker says {stated}; at the odds read we get {ours}: check the odds."
                           if ours else f"Your bookmaker says {stated}, which the odds read don't give: check them.")
        priced = [leg for leg in legs if leg["odds"] is not None]
        if not priced:  # nothing to price, so nothing to confirm: the reply says why (a same-game parlay)
            warnings.append(pasted.NO_LEG_ODDS)
        for leg in legs if priced else ():
            if leg["odds"] is None:
                reasons.append(f"Couldn't read the odds for leg {leg['index']} ({leg['label']}).")
            elif row["source"] == "text" and leg["verified"] is False:
                reasons.append(f"The odds read for leg {leg['index']} ({leg['odds_text']}) aren't in the text you "
                               "pasted.")
            elif self._far_above_board(leg):
                best = leg["best_price"]
                at = f" at {best['book']}" if best.get("book") else ""
                reasons.append(f"Leg {leg['index']}'s odds ({leg['odds_text']}) are far longer than any price on "
                               f"our board (best {best['decimal']}{at}): check them. Is it a combined or boosted "
                               "price?")
        if slip.bet_type in ("multi", "round_robin", "system") and len(legs) == 1:
            kind = {"multi": "a multi (parlay or accumulator)", "round_robin": "a round robin",
                    "system": "a system bet"}[slip.bet_type]
            reasons.append(f"Your slip looks like {kind}, but only one leg was read: check that none is missing.")
        mismatch = money.get("stake_mismatch")
        if mismatch:
            each, bets = mismatch["stake"], mismatch["bets"]
            spent = (f"{_say_dollars(each, symbol)} on each of the {bets} bets read is "
                     f"{_say_dollars(each * bets, symbol)}" if bets > 1 else
                     f"the stake read for its one bet is {_say_dollars(each, symbol)}")
            reasons.append(f"Your slip says {_say_dollars(mismatch['total'], symbol)} in all, but {spent}: a leg may "
                           "be missing, or it's an each-way bet.")
        estimated = [leg["index"] for leg in legs if leg["odds"] is not None and leg["chance_source"] == "estimated"]
        if len(estimated) == 1:
            warnings.append(f"Leg {estimated[0]} isn't matched to a price the app knows, so its chance is your odds "
                            "less a typical bookmaker margin (an estimate).")
        elif estimated:
            which = f"{', '.join(map(str, estimated[:-1]))} and {estimated[-1]}"
            warnings.append(f"Legs {which} aren't matched to prices the app knows, so their chances are your odds "
                            "less a typical bookmaker margin (estimates).")
        warnings += [f"Leg {leg['index']}'s game has started: its chances are from the board as it stands."
                     for leg in legs if leg["match"]["live"]]
        _, _, _, type_note = self._slip_type(slip, len(legs))
        if type_note:
            warnings.append(type_note)
        if money.get("stake_uneven"):
            warnings.append("The total stake doesn't split evenly into the slip's bets, so the stake per bet is "
                            "rounded down to the cent.")
        if len(legs) > tickets.MAX_LEGS:
            warnings.append(f"A slip with more than {tickets.MAX_LEGS} legs can be read but not priced in full: "
                            "leave some out to see the numbers.")
        return {"needs_confirmation": bool(reasons), "confirm_reasons": reasons, "warnings": warnings}

    @staticmethod
    def _far_above_board(leg: Mapping[str, Any]) -> bool:
        """A matched leg whose odds are far longer than the board's best price for it (FAR_ABOVE_BOARD times):
        more likely a misread, or a whole same-game parlay's or a boosted price, than a better bookmaker."""
        best = leg.get("best_price") or {}
        if leg["match"]["status"] != "matched" or leg["match"]["live"] or not best.get("decimal"):
            return False
        odds = pasted.parse_odds(leg["odds_text"])
        return odds is not None and odds >= Decimal(str(best["decimal"])) * FAR_ABOVE_BOARD

    # --- pasted slips: the analysis ---------------------------------------------------------------------------

    @staticmethod
    def _paste_drop(drop: Any, count: int) -> set:
        try:
            dropped = {int(index) for index in (drop or ()) if not isinstance(index, bool)}
        except (TypeError, ValueError):
            raise CallRejected("drop lists the legs to leave out by number, from 1") from None
        if any(not 1 <= index <= count for index in dropped):
            raise CallRejected(f"This slip's legs are numbered 1 to {count}")
        return dropped

    def _paste_ticket_type(self, slip: pasted.Extraction, count: int, ticket_type: Optional[str], k: Any,
                           system: Optional[str]) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
        if ticket_type is None:
            kind, slip_k, slip_system, note = self._slip_type(slip, count)
            return kind, k if k is not None else slip_k, system or slip_system, note
        if ticket_type not in tickets.TICKET_TYPES:
            raise CallRejected(f"A ticket is one of {', '.join(tickets.TICKET_TYPES)}")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int)):
            raise CallRejected("k is a round robin's picks per bet, e.g. 2")
        return ticket_type, k, system, None

    @staticmethod
    def _paste_stake(stake: Any, view: Mapping[str, Any]) -> Tuple[Decimal, Optional[str]]:
        """The stake per bet: the one asked for, else the slip's own, else 10 (said so)."""
        if stake is not None:
            return _amount(stake, "stake", MAX_STAKE), None
        if view.get("stake") is not None:
            own = Decimal(view["stake"])
            if CENTS <= own <= Decimal(MAX_CENTS) / 100:
                return own, None
        symbol = view.get("currency")
        return DEFAULT_PASTE_STAKE, (f"No stake was read from the slip, so it is priced at "
                                     f"{_say_dollars(DEFAULT_PASTE_STAKE, symbol)} a bet.")

    @staticmethod
    def _paste_ticket_leg(pasted_id: int, leg: pasted.ExtractedLeg, entry: Mapping[str, Any],
                          view: Mapping[str, Any], bookmaker: Optional[str]) -> TicketLeg:
        """A pasted leg for engine/tickets at the person's own odds, with its chance from the board, the model
        or the estimate. Legs from the same game share a group, so a same-game ticket is flagged."""
        ref = entry.get("ref") if entry.get("status") == "matched" else None
        price = pasted.parse_odds(leg.odds_text)
        index = view["index"]
        if ref:
            key = _leg_key(ref["league"], ref["id"], ref["market"], ref["side"], ref.get("point"))
            league, event_id, market, side, point = ref["league"], ref["id"], ref["market"], ref["side"], ref.get("point")
        else:
            key = f"pasted:{pasted_id}:{index}"
            league = entry.get("league") or "pasted"
            event_id = (entry.get("ref") or {}).get("id") or f"pasted-{pasted_id}-{index}"
            market = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}.get(
                leg.market, f"player_{PASTE_STATS.get(leg.prop_stat or '', 'prop')}" if leg.market == "player_prop"
                else "other")
            side, point = leg.over_under or "pick", leg.line
        ai = entry.get("ai_chance")
        push = float(entry.get("push") or 0.0)
        if ai is not None:
            push = min(push, 1.0 - float(ai))
        return TicketLeg(key=key, league=league, event_id=str(event_id), market=market, side=side, point=point,
                         price=price, label=view["label"], game=entry.get("game") or leg.event_text or "",
                         start=entry.get("start") or UNKNOWN_START, ai_prob=ai, market_prob=entry.get("market_chance"),
                         push_prob=push, book=bookmaker, group=entry.get("group") or str(event_id))

    def _paste_cards(self, entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """The game cards of matched legs from boards already cached (injuries, the model's factors), by id."""
        saved = self._saved_odds()
        cards: Dict[str, Dict[str, Any]] = {}
        for league in sorted({(entry.get("ref") or {}).get("league") for entry in entries if entry.get("ref")}):
            board = self._board(league)[0] if self._odds_cached(league, saved) else None
            for card in (board or {}).get("games") or []:
                cards[str(card["id"])] = card
        return cards

    def _paste_notes(self, leg: Mapping[str, Any], entry: Mapping[str, Any], priced: Optional[Mapping[str, Any]],
                     card: Optional[Mapping[str, Any]], dropped: bool) -> List[str]:
        """Plain facts about one pasted leg, as Lab._notes for the Build screen, plus a better price elsewhere,
        a line that differs from the board's and an estimated chance."""
        if leg["odds"] is None:
            return ["The odds for this leg couldn't be read, so it is left out of the numbers."]
        notes = ["Left out: you turned this leg off."] if dropped else []
        status = entry.get("status")
        if status == "matched" and priced is not None:
            notes += self._notes(priced, card)
        elif entry.get("note"):
            notes.append(entry["note"])
        chance = leg["ai_chance"] if leg["ai_chance"] is not None else leg["market_chance"]
        if status != "matched" and chance is not None:
            whose = {"model": "the model's", "market": "the market's"}.get(leg["chance_source"], "an estimated")
            notes.append(f"Wins about {_chance_words(chance)} by {whose} chance.")
        best = leg.get("best_price")
        if leg["better_price"] and best:
            where = f" at {best['book']}" if best.get("book") else ""
            notes.append(f"Better price: {best['decimal']}{where} (you have {leg['odds']['decimal']}).")
        if leg["match"]["live"] and status != "matched":
            notes.append("This game has started: the board's prices and chances are as it stands, not as they were "
                         "when you bet.")
        return notes

    def _paste_alternatives(self, priced: Mapping[str, Any], legs: Sequence[TicketLeg],
                            kept: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]],
                            dropped: set, currency: Optional[str]) -> List[Dict[str, Any]]:
        """Lab._alternatives at the person's odds, each also saying which of the slip's legs it keeps ("legs"),
        which to leave out to get it ("drop"), and the refs of those it can load into Build."""
        index_of: Dict[str, int] = {}
        for leg, view in zip(legs, kept):
            index_of.setdefault(leg.key, view["index"])
        everything = {view["index"] for view in kept}
        out = []
        for alt in self._alternatives(priced, legs, currency):
            indexes = [index_of.get(view["key"]) for view in alt["ticket"]["legs"]]
            alt["legs"] = indexes
            alt["drop"] = sorted((everything - set(indexes)) | dropped)
            alt["refs"] = [entries[index - 1]["ref"] for index in indexes
                           if index and entries[index - 1].get("status") == "matched"]
            for leg_view, index in zip(alt["ticket"]["legs"], indexes):
                leg_view["slip_index"] = index
                if leg_view.get("start") == UNKNOWN_START:
                    leg_view["start"] = None
            out.append(alt)
        return out


# --- module helpers that need the classes above ------------------------------------------------------

def _versus(alt: Mapping[str, Any], base: Mapping[str, Any], currency: Optional[str] = CURRENCY) -> str:
    """One line comparing an alternative with the person's ticket: chance of a profit and the expected result,
    both by the AI's chances when every leg has one, else by the market's."""
    basis = "ai" if alt["chance"]["profit"]["ai"] is not None and base["chance"]["profit"]["ai"] is not None else "market"
    parts = []
    p_alt, p_base = alt["chance"]["profit"][basis], base["chance"]["profit"][basis]
    if p_alt is not None and p_base is not None:
        parts.append(f"chance of a profit {_chance_words(p_alt)}, against {_chance_words(p_base)} for your ticket")
    e_alt, e_base = alt["expected"][basis], base["expected"][basis]
    if e_alt is not None and e_base is not None:
        parts.append(f"expected result {_say_signed(e_alt, currency)} against {_say_signed(e_base, currency)}")
    if not parts:
        return ""
    text = "; ".join(parts)
    return f"{text[0].upper()}{text[1:]} (by {'the AI' if basis == 'ai' else 'the market'}'s chances)."


def _mean(row: Mapping[str, Any], stat: str) -> float:
    if stat == "pra":
        return float(sum(row.get(key) or 0.0 for key in ("pts", "reb", "ast")))
    return float(row.get(stat) or 0.0)


def _quantile(stat: str, mean: float, minutes: Optional[float], share: float) -> int:
    """The smallest count the stat stays at or below with chance ``share`` (the model's own spread)."""
    for k in range(0, int(mean * 4) + 40):
        if calls.stat_probabilities(stat, mean, k + 0.5, minutes)[1] >= share:
            return k
    return int(mean * 4) + 40


class _PasteBoards:
    """The boards one pasted slip may look at: any league already cached (no quota), and at most ``loads``
    more loaded (each fetch costs odds quota), never a league out of season. Each is looked up once a slip.
    Without a league on the slip, cached boards come first (the person's favourites among them first), then
    the person's favourites (the NBA when they have none)."""

    def __init__(self, lab: Lab, user: Mapping[str, Any], loads: int) -> None:
        self.lab = lab
        self.loads = loads
        self.boards: Dict[str, Optional[Dict[str, Any]]] = {}
        self.missed: Dict[str, str] = {}  # league -> why its board wasn't looked at: season, quota or error
        self.season = lab._season()
        self.saved = lab._saved_odds()
        self.favourites = [key for key in user.get("favourite_sports") or [] if isinstance(key, str)] \
            or list(DEFAULT_LEAGUES)
        self._cached: Optional[List[str]] = None

    def _in_season(self, league: str) -> bool:
        return self.season is None or not self.lab._out_of_season(self.season.get(league))

    def cached(self) -> List[str]:
        """Leagues in season whose boards can be read without a fetch: favourites first, then popular ones."""
        if self._cached is None:
            keys = set(self.season or {}) | set(self.saved) | set(self.favourites)
            ranked = sorted(keys, key=lambda key: (key not in self.favourites,
                                                   POPULAR_LEAGUES.index(key) if key in POPULAR_LEAGUES
                                                   else len(POPULAR_LEAGUES), key))
            self._cached = [key for key in ranked if LEAGUE_KEY.fullmatch(key) and self._in_season(key)
                            and self.lab._odds_cached(key, self.saved)]
        return self._cached

    @staticmethod
    def named_league(leg: pasted.ExtractedLeg) -> Tuple[Optional[str], bool]:
        """(the league the leg's sport hint, else its own text, names; whether it names another league of that
        name the app doesn't cover, such as "Brazil Serie A" or "NBA G League")."""
        for text in (leg.sport_hint, leg.raw):
            league, other = pasted.league_of(text)
            if league or other:
                return league, other
        return None, False

    def leagues_for(self, leg: pasted.ExtractedLeg) -> List[str]:
        """The leagues to look for a leg's game in, in order: the one the slip names, else cached boards, then
        the person's favourites (only that sport's when the slip names a sport, its popular leagues if none).
        None for a league the app doesn't cover ("NBA G League", "Brazil Serie A"): its teams' namesakes in
        the big leagues ("Maine Celtics", "Botafogo") must not match, and no quota is spent looking."""
        league, other = self.named_league(leg)
        if other:
            return []
        if league:
            return [league]
        prefixes = pasted.sport_prefixes(leg.sport_hint) or pasted.sport_prefixes(leg.raw)
        cached = [key for key in self.cached() if not prefixes or key.startswith(prefixes)]
        rest = [key for key in self.favourites if (not prefixes or key.startswith(prefixes)) and key not in cached]
        if prefixes and not rest:
            rest = [key for prefix in prefixes for key in POPULAR_LEAGUES if key.startswith(prefix) and key not in cached]
        return cached + rest

    def get(self, league: str) -> Optional[Dict[str, Any]]:
        """A league's board, loading it only while this slip may still load one; None when it can't be had."""
        if league in self.boards:
            return self.boards[league]
        if not LEAGUE_KEY.fullmatch(league) or not self._in_season(league):
            self.missed[league] = "season"
            self.boards[league] = None
            return None
        if not self.lab._odds_cached(league, self.saved):
            if self.loads <= 0:
                self.missed[league] = "quota"
                return None
            self.loads -= 1
        board, _ = self.lab._board(league)
        if board is None:
            self.missed[league] = "error"
        self.boards[league] = board
        return board

    def why_not(self, leagues: Sequence[str]) -> str:
        """Why a game wasn't found, when boards were missing (" (...)"), else ""."""
        reasons = [self.missed.get(league) for league in leagues]
        if "quota" in reasons:
            whose = "some leagues'" if len(leagues) > 1 else "its league's"
            return f" ({whose} odds weren't loaded, to save the monthly odds allowance)"
        if not reasons or None in reasons:
            return ""  # every board that could be looked at was
        if "error" in reasons:
            return " (its league's odds couldn't be loaded just now)"
        return " (that league isn't in season)"


def _text_money(amount: Optional[Decimal]) -> Optional[str]:
    """Money as the API's strings ("10.00"), or None."""
    return None if amount is None else str(amount.quantize(CENTS, rounding=ROUND_HALF_UP))
