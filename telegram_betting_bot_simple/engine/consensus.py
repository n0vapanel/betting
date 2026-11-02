"""The markets brain: one blended chance per outcome from every source we can legitimately get.

Sources (each a ``Source`` dict) come in five kinds, with the report's starting weights ("Prediction markets
and a ten-book consensus..."; research_notes/Sports data sources and analysis UX/odds_and_predictions.md,
"Brain blend v3"):

    sharp             0.50  Pinnacle and the exchanges (Betfair, Matchbook), margin removed
    us_books          0.20  the median margin-free price of the US books
    prediction_market 0.20  Kalshi and Polymarket mids (liquid markets only)
    model_rating      0.05  ESPN's predictor, FPI/BPI, Elo, Torvik
    team_model        0.05  our own team model (NBA)

Market weight ~0.90 follows the evidence that a fresh closing line beats results-only models: stacked
weights of 0.07-0.19 on Elo or Massey slightly worsened log loss against the close, and the best fitted
blends against 538's RAPTOR gave the market 0.60-0.74 (Walker Harrison 2022). The weights are a starting
point, to be refitted per sport from the graded record.

``blend`` is a logarithmic opinion pool: ``logit p = sum(w_s * logit p_s)`` for two outcomes, and for three
(soccer's home/draw/away) the log-linear pool ``p_k proportional to prod_s p_sk ** w_s``, normalised; with
weights summing to 1 the two are the same formula. Rules:

- only sources with status "ok" count; weights are renormalised over the kinds present;
- sources of one kind share that kind's weight equally (Kalshi and Polymarket split 0.20);
- prediction markets get half weight within 15 minutes of the start (and after it): a study of 23 million
  Kalshi moneyline trades found calibration close to perfect mid-contract but "depart[ing] sharply as expiry
  approaches" (Moshrefi, arXiv 2607.14430);
- with no ok sharp, US-book or prediction-market source, the ok ratings get equal weights, the result is
  shrunk 25% toward the sport's home-win base rate (the notes say 20-30%) and confidence is "low";
- ``spread_pp`` is the largest max-min gap, in percentage points, across the ok sources on any outcome, and
  ``disagree`` is that gap reaching 5 points.

Code computes every number here; the AI only explains them. The blend is the app's best estimate of the
chance, and the realistic promise is to match the market and explain it, not to beat it.
"""

from __future__ import annotations

import datetime
import math
import statistics
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

KINDS = ("sharp", "us_books", "prediction_market", "model_rating", "team_model")
DEFAULT_WEIGHTS: Dict[str, float] = {"sharp": .50, "us_books": .20, "prediction_market": .20, "model_rating": .05,
                                     "team_model": .05}
MARKET_KINDS = ("sharp", "us_books", "prediction_market")
RATING_KINDS = ("model_rating", "team_model")
STATUSES = ("ok", "thin", "stale", "missing")
OUTCOMES = ("home", "away", "draw")

# Odds API bookmaker keys that anchor the consensus. Pinnacle's feed "may incur a delay" and
# football-data.co.uk found its public odds "systematically out of date" after July 2025, so the exchanges
# sit in the anchor beside it. These books feed the consensus only: nobody prices a ticket at them.
SHARP_BOOKS: Tuple[str, ...] = ("pinnacle", "betfair_ex_eu", "matchbook")
# Prediction markets that The Odds API can also list as bookmakers: they come in through their own sources
# (data/markets), so a book list that includes them must not count them a second time as "US books"
PREDICTION_MARKET_BOOKS: Tuple[str, ...] = ("kalshi", "polymarket")
BOOK_TITLES = {"pinnacle": "Pinnacle", "betfair_ex_eu": "Betfair", "matchbook": "Matchbook"}

NEAR_START_SECONDS = 15 * 60
NEAR_START_FACTOR = 0.5
DISAGREE_PP = 5.0
LOW_CONFIDENCE_SHRINK = 0.25
P_FLOOR = 1e-4  # chances are kept inside [P_FLOOR, 1 - P_FLOOR] before taking logs

# Home-win base rates for the low-confidence shrinkage (research_notes/.../venues_travel_weather.md, live
# standings): NFL 2025 53.9% (146-125-1, ties as half), NBA 2025-26 55.4% (679-546), NHL 2025-26 52.2% of
# games, MLB 2026 52.9% (1,284-1,145). Soccer 1X2: home 45%, draw 27%, away 28% (top-league norms; the
# notes' EPL sample split 45.5 / 24.3 / 30.2). Other sports (tennis, MMA, college) shrink toward an even
# split, as does any game at a neutral site.
BASE_RATES: Dict[str, Dict[str, float]] = {
    "americanfootball_nfl": {"home": 0.539, "away": 0.461},
    "basketball_nba": {"home": 0.554, "away": 0.446},
    "icehockey_nhl": {"home": 0.522, "away": 0.478},
    "baseball_mlb": {"home": 0.529, "away": 0.471},
    "soccer": {"home": 0.45, "draw": 0.27, "away": 0.28},
}
_SPORT_ALIASES = {"nfl": "americanfootball_nfl", "nba": "basketball_nba", "nhl": "icehockey_nhl",
                  "mlb": "baseball_mlb"}


class Source(TypedDict, total=False):
    """One opinion on a game: ``p`` is {"home", "away", "draw"?} chances that sum to 1."""

    name: str  # "Sharp", "US books", "Kalshi", "Polymarket", "ESPN", "Elo", "Torvik", "Model", ...
    kind: str  # one of KINDS
    p: Dict[str, float]
    as_of: Optional[str]  # ISO time
    status: str  # "ok" | "thin" | "stale" | "missing"
    url: Optional[str]
    detail: Optional[str]


class Blend(TypedDict):
    p: Dict[str, float]
    used: List[str]  # names of the sources with weight, in input order
    weights: Dict[str, float]  # name -> final weight (sums to 1)
    confidence: str  # "normal" | "low"
    spread_pp: float  # largest max-min gap across the ok sources on any outcome, in percentage points
    disagree: bool  # spread_pp >= 5
    as_of: Optional[str]  # newest as_of among the used sources
    n_sources: int
    base: Optional[Dict[str, float]]  # the base rate the low-confidence blend was shrunk toward
    near_start: bool  # prediction markets were at half weight


# --- small helpers ------------------------------------------------------------------------------------

def _ts(value: Any) -> Optional[float]:
    """Seconds since the epoch from a number, a datetime or an ISO string; None when unreadable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, datetime.datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
        return moment.timestamp()
    if isinstance(value, str) and value.strip():
        try:
            moment = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return _ts(moment)
    return None


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalised(p: Any, outcomes: Sequence[str]) -> Optional[Dict[str, float]]:
    """``p`` restricted to ``outcomes``, clamped away from 0 and 1 and rescaled to sum to 1; None when an
    outcome is missing or a value isn't a usable chance."""
    if not isinstance(p, Mapping):
        return None
    values = {}
    for outcome in outcomes:
        number = _number(p.get(outcome))
        if number is None or number < 0:
            return None
        values[outcome] = min(max(number, P_FLOOR), 1 - P_FLOOR)
    total = math.fsum(values.values())
    if total <= 0:
        return None
    return {outcome: value / total for outcome, value in values.items()}


def _outcomes_of(p: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(o for o in OUTCOMES if o in p)


def no_vig(prices: Mapping[str, Any]) -> Dict[str, float]:
    """Margin-free chances from one book's decimal prices for every outcome of a market, multiplicative
    (each implied chance 1/price divided by their sum). Empty when there are fewer than two outcomes or a
    price isn't a decimal price above 1."""
    if not isinstance(prices, Mapping) or len(prices) < 2:
        return {}
    implied = {}
    for outcome, price in prices.items():
        number = _number(price)
        if number is None or number <= 1:
            return {}
        implied[outcome] = 1 / number
    total = math.fsum(implied.values())
    return {outcome: value / total for outcome, value in implied.items()}


def _median_p(rows: Iterable[Mapping[str, float]], outcomes: Sequence[str]) -> Optional[Dict[str, float]]:
    rows = list(rows)
    if not rows:
        return None
    medians = {o: statistics.median(row[o] for row in rows) for o in outcomes}
    total = math.fsum(medians.values())
    return {o: v / total for o, v in medians.items()}


def _book_split(books: Mapping[str, Mapping[str, Any]], sharp: Iterable[str]
                ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Tuple[str, ...]]:
    """(sharp books' no-vig chances, other books' no-vig chances, the outcomes they share)."""
    sharp_keys = {str(key).lower() for key in sharp}
    devigged = {}
    for book, prices in (books or {}).items():
        if str(book).lower() in PREDICTION_MARKET_BOOKS:
            continue
        fair = no_vig(prices)
        if fair:
            devigged[str(book)] = fair
    if not devigged:
        return {}, {}, ()
    # Every book is compared on the same outcomes: the most common set (the larger one on a tie)
    sets = Counter(tuple(sorted(p)) for p in devigged.values())
    shared = max(sets, key=lambda s: (sets[s], len(s)))
    order = tuple(o for o in OUTCOMES if o in shared) + tuple(o for o in shared if o not in OUTCOMES)
    same = {book: p for book, p in devigged.items() if tuple(sorted(p)) == shared}
    sharp_rows = {book: p for book, p in same.items() if book.lower() in sharp_keys}
    other_rows = {book: p for book, p in same.items() if book.lower() not in sharp_keys}
    return sharp_rows, other_rows, order


def book_consensus(books: Mapping[str, Mapping[str, Any]], sharp: Iterable[str] = SHARP_BOOKS
                   ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """(sharp no-vig chances, US books' median no-vig chances) from {book key: {outcome: decimal price}}.

    Each book's margin is removed on its own (``no_vig``); the sharp anchor is the median across the sharp
    books present (Pinnacle, Betfair, Matchbook: with two it is their mean) and the US figure the median
    across every other book. Per-outcome medians are renormalised to sum to 1 (three-way markets). Either is
    None when no such book prices every outcome. Prediction markets listed as books are left out.
    """
    sharp_rows, other_rows, outcomes = _book_split(books, sharp)
    return _median_p(sharp_rows.values(), outcomes), _median_p(other_rows.values(), outcomes)


def make_source(name: str, kind: str, p: Optional[Mapping[str, Any]], as_of: Any = None, status: str = "ok",
                url: Optional[str] = None, detail: Optional[str] = None) -> Source:
    """A ``Source`` dict, checked: ``p`` keeps only home/away/draw and is renormalised; a missing or unusable
    ``p`` gives status "missing". ``as_of`` falls back to ``p["as_of"]`` (Ratings' dicts carry their own)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown source kind {kind!r}")
    if status not in STATUSES:
        raise ValueError(f"Unknown source status {status!r}")
    outcomes = _outcomes_of(p) if isinstance(p, Mapping) else ()
    chances = _normalised(p, outcomes) if len(outcomes) >= 2 else None
    if as_of is None and isinstance(p, Mapping):
        as_of = p.get("as_of")
    if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
        as_of = _iso(float(as_of))
    return {"name": name, "kind": kind, "p": chances or {}, "as_of": as_of,
            "status": status if chances else "missing", "url": url, "detail": detail}


def book_sources(books: Mapping[str, Mapping[str, Any]], as_of: Any = None, sharp: Iterable[str] = SHARP_BOOKS
                 ) -> List[Source]:
    """The "Sharp" and "US books" sources for one game's moneyline prices ({book: {outcome: decimal}})."""
    sharp_rows, other_rows, outcomes = _book_split(books, sharp)
    out: List[Source] = []
    if sharp_rows:
        names = " · ".join(BOOK_TITLES.get(book.lower(), book) for book in sharp_rows)
        out.append(make_source("Sharp", "sharp", _median_p(sharp_rows.values(), outcomes), as_of,
                               detail=f"no-vig: {names}"))
    if other_rows:
        count = len(other_rows)
        out.append(make_source("US books", "us_books", _median_p(other_rows.values(), outcomes), as_of,
                               detail=f"median no-vig of {count} book{'s' if count != 1 else ''}"))
    return out


def prediction_market_source(event: Mapping[str, Any], flip: bool = False) -> Optional[Source]:
    """A Kalshi or Polymarket ``MarketEvent``'s game-winner market as a source (None without one).

    Each outcome's quote ``p`` (the bid/ask mid) is normalised across the outcomes; the source is "thin"
    when any outcome's quote is (spread over 6 cents or under $1,000 of liquidity). ``flip`` swaps home and
    away, for an event whose sides are the other way round from the game's."""
    h2h = ((event or {}).get("markets") or {}).get("h2h") or {}
    if flip:
        h2h = {{"home": "away", "away": "home"}.get(side, side): quote for side, quote in h2h.items()}
    quotes = {side: quote for side, quote in h2h.items() if side in OUTCOMES and isinstance(quote, Mapping)}
    if not {"home", "away"} <= set(quotes):
        return None
    name = {"kalshi": "Kalshi", "polymarket": "Polymarket"}.get(str(event.get("source", "")).lower(),
                                                               str(event.get("source") or "Market"))
    thin = any(quote.get("thin") for quote in quotes.values())
    liquidity = [_number(quote.get("liquidity")) for quote in quotes.values()]
    known = [value for value in liquidity if value is not None]
    detail = f"mid price; ${min(known):,.0f} liquidity" if known else "mid price"
    return make_source(name, "prediction_market", {side: quote.get("p") for side, quote in quotes.items()},
                       event.get("as_of"), "thin" if thin else "ok", event.get("url"), detail)


def base_rate(sport: Optional[str], outcomes: Sequence[str] = ("home", "away"), neutral: bool = False
              ) -> Dict[str, float]:
    """The sport's home-win base rate over ``outcomes`` (an even split at a neutral site or when unknown).
    ``sport`` is an Odds API league key ("basketball_nba", "soccer_epl", ...) or "nba"/"nfl"/"nhl"/"mlb"."""
    outcomes = tuple(outcomes)
    even = {o: 1 / len(outcomes) for o in outcomes}
    if neutral:
        return even
    key = str(sport or "").lower()
    rate = BASE_RATES.get(key) or BASE_RATES.get(_SPORT_ALIASES.get(key, ""))
    if rate is None and key.startswith("soccer"):
        rate = BASE_RATES["soccer"]
    if rate is None or set(rate) != set(outcomes):
        return even
    return {o: rate[o] for o in outcomes}


def log_pool(chances: Sequence[Mapping[str, float]], weights: Sequence[float]) -> Dict[str, float]:
    """The log-linear opinion pool: p_k proportional to prod_s p_sk ** w_s, normalised. For two outcomes and
    weights summing to 1 this is exactly logit p = sum(w_s * logit p_s)."""
    outcomes = list(chances[0])
    logs = {o: math.fsum(w * math.log(min(max(p[o], P_FLOOR), 1 - P_FLOOR)) for p, w in zip(chances, weights))
            for o in outcomes}
    top = max(logs.values())
    raw = {o: math.exp(v - top) for o, v in logs.items()}
    total = math.fsum(raw.values())
    return {o: v / total for o, v in raw.items()}


def _near_start(now: Any, start: Any, seconds: Optional[float] = None) -> bool:
    begins = _ts(start)
    if begins is None:
        return False
    current = _ts(now)
    if current is None:
        current = time.time()
    return begins - current < (NEAR_START_SECONDS if seconds is None else seconds)


def _tuning(value: Optional[float], default: float, name: str, low: float, high: float) -> float:
    """A tuned parameter (learn/registry.py), or the module's constant when None; ValueError outside [low, high]."""
    if value is None:
        return default
    number = _number(value)
    if number is None or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value!r}")
    return number


def blend(sources: Iterable[Mapping[str, Any]], sport: Optional[str], weights: Optional[Mapping[str, float]] = None,
          now: Any = None, start: Any = None, *, neutral: bool = False, sharpness: float = 1.0,
          near_start_factor: Optional[float] = None, near_start_seconds: Optional[float] = None,
          low_confidence_shrink: Optional[float] = None) -> Optional[Blend]:
    """Blend the ok sources into one chance per outcome (see the module docstring). None with no ok source.

    ``weights`` overrides kind weights (merged over DEFAULT_WEIGHTS, e.g. a per-sport refit); ``now`` and
    ``start`` (epoch seconds, datetimes or ISO strings) drive the near-start rule; ``neutral`` makes the
    low-confidence shrinkage go toward an even split instead of the home base rate.

    The learning loop's tuned values (learn/registry.py) come in as keywords; each default is today's constant,
    so a call without them is unchanged. ``sharpness`` is the pool's extremising exponent ``a``:
    ``logit p = a x sum(w_s x logit p_s)`` (three outcomes: ``p_k`` proportional to ``prod_s p_sk ** (a x w_s)``);
    at 1 it is a no-op, otherwise the result carries a "sharpness" key. ``near_start_factor`` and
    ``near_start_seconds`` replace NEAR_START_FACTOR and NEAR_START_SECONDS, ``low_confidence_shrink``
    LOW_CONFIDENCE_SHRINK.
    """
    a = _number(sharpness)
    if a is None or not 0 < a <= 10:
        raise ValueError(f"sharpness must be a positive number, got {sharpness!r}")
    factor = _tuning(near_start_factor, NEAR_START_FACTOR, "near_start_factor", 0.0, 1.0)
    window = _tuning(near_start_seconds, NEAR_START_SECONDS, "near_start_seconds", 0.0, 86400.0)
    shrink = _tuning(low_confidence_shrink, LOW_CONFIDENCE_SHRINK, "low_confidence_shrink", 0.0, 1.0)
    ok = [s for s in sources if isinstance(s, Mapping) and s.get("status") == "ok" and s.get("kind") in KINDS]
    # The outcomes: three-way when a draw is priced by the markets (or, with no market, by any rating);
    # sources that don't price every outcome can't join the pool
    leading = [s for s in ok if s["kind"] in MARKET_KINDS] or ok
    outcomes: Tuple[str, ...] = ("home", "away", "draw") if any(
        isinstance(s.get("p"), Mapping) and _number(s["p"].get("draw")) is not None for s in leading
    ) else ("home", "away")
    pool = []
    for source in ok:
        chances = _normalised(source.get("p"), outcomes)
        if chances is not None:
            pool.append((source, chances))
    if not pool:
        return None

    near = False
    if any(source["kind"] in MARKET_KINDS for source, _ in pool):
        confidence = "normal"
        kind_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        near = _near_start(now, start, window) and any(source["kind"] == "prediction_market" for source, _ in pool)
        per_kind = Counter(source["kind"] for source, _ in pool)
        raw = []
        for source, _ in pool:
            weight = max(_number(kind_weights.get(source["kind"])) or 0.0, 0.0)
            if near and source["kind"] == "prediction_market":
                weight *= factor
            raw.append(weight / per_kind[source["kind"]])
    else:
        confidence = "low"
        raw = [1.0] * len(pool)
    total = math.fsum(raw)
    shares = [w / total for w in raw] if total > 0 else [1 / len(pool)] * len(pool)

    used = [(source, chances, w) for (source, chances), w in zip(pool, shares) if w > 0]
    p = log_pool([chances for _, chances, _ in used], [w if a == 1 else a * w for _, _, w in used])
    base = None
    if confidence == "low":
        base = base_rate(sport, outcomes, neutral)
        p = {o: (1 - shrink) * p[o] + shrink * base[o] for o in outcomes}

    gaps = [max(c[o] for _, c in pool) - min(c[o] for _, c in pool) for o in outcomes]
    spread = 100 * max(gaps)
    named: Dict[str, float] = {}
    for source, _, w in used:
        named[str(source.get("name"))] = named.get(str(source.get("name")), 0.0) + w
    stamps = [(stamp, source.get("as_of")) for source, _, _ in used
              if (stamp := _ts(source.get("as_of"))) is not None]
    if stamps:
        as_of = max(stamps, key=lambda pair: pair[0])[1]
    else:
        current = _ts(now)
        as_of = _iso(current if current is not None else time.time())
    out: Blend = {
        "p": {o: p[o] for o in outcomes},
        "used": list(named),
        "weights": named,
        "confidence": confidence,
        "spread_pp": round(spread, 2),
        "disagree": spread >= DISAGREE_PP - 1e-9,
        "as_of": as_of,
        "n_sources": len(used),
        "base": base,
        "near_start": near,
    }
    if a != 1:
        out["sharpness"] = a  # type: ignore[typeddict-unknown-key]
    return out


def score(p: Mapping[str, float], outcome: str) -> Optional[Dict[str, float]]:
    """Brier score and log loss of one forecast once ``outcome`` ("home", "away" or "draw") happened.

    Brier is half the squared distance summed over the outcomes, which for two outcomes is the usual
    (p_home - home_won)^2 the AI record already uses; log loss is -ln p(outcome), with p kept at least
    P_FLOOR. None when the forecast didn't price the outcome (an NFL tie on a two-way market)."""
    outcomes = _outcomes_of(p)
    if outcome not in outcomes or len(outcomes) < 2:
        return None
    chances = _normalised(p, outcomes)
    if chances is None:
        return None
    brier = 0.5 * math.fsum((chances[o] - (1.0 if o == outcome else 0.0)) ** 2 for o in outcomes)
    return {"brier": brier, "log_loss": -math.log(max(chances[outcome], P_FLOOR))}

