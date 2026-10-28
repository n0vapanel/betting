"""The soccer goals model: goal rates from the market's 1X2 and over/under 2.5, then a Dixon-Coles grid.

Soccer is the sport where the market does the most work. The research behind it (report, "Each sport gets its
own engine..."; computed from football-data.co.uk, 5,255 top-five-league matches) inverted de-vigged 1X2 and
over/under 2.5 closing odds into goal rates: implied means 1.557 home and 1.245 away against actual means of
1.543 and 1.283. A Dixon-Coles grid with rho = -0.13 then priced both teams to score with a Brier score of
0.2428 against 0.2476 for a constant, calibrated within a point in the middle but low at the high end, so
both-teams-to-score "yes" gets a +1.5-point nudge.

- ``rates_from_odds`` finds the home and away rates (lambda, mu) whose grid reproduces the four margin-free
  chances (home, draw, away, over 2.5) as closely as possible: least squares, solved by Levenberg-Marquardt on
  log rates. The fit is on the same Dixon-Coles grid the markets are read from, so the derived draw-no-bet and
  double chance agree with the 1X2 the person sees (typically within 0.5 points).
- ``grid`` is the score matrix P(home goals = x, away goals = y) for 0..max_goals each: independent Poissons
  times Dixon and Coles' (1997) low-score correction tau:
      tau(0,0) = 1 - lambda*mu*rho    tau(0,1) = 1 + lambda*rho
      tau(1,0) = 1 + mu*rho           tau(1,1) = 1 - rho          (1 elsewhere)
  The correction moves no probability in or out (its terms cancel), and the grid is renormalised for the
  scores past max_goals, so it sums to 1.
- ``markets`` reads every derived market off one grid: both teams to score (nudged), draw no bet (the draw is a
  push), double chance, team totals, game totals, clean sheets, expected goals and the top 6 correct scores.

One pair of rates prices all of these coherently. Pass the blend's 1X2 (engine/consensus.py) where there is
one, else the books' margin-free 1X2, and the books' margin-free over 2.5.

``candidates`` turns the derived markets into builder legs (engine/builder.py). A market no bookmaker prices
on the cached board is priced at its fair odds and flagged ``model_price``: the ticket then says "priced at
fair odds; check your bookmaker's price", and Safe never takes such a leg.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DC_RHO = -0.13
BTTS_NUDGE = 0.015
MAX_GOALS = 10
TEAM_TOTAL_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
TOTAL_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)
TOP_SCORES = 6
FIT_OK_PP = 1.0  # past this the two rates can't reproduce the prices well: the legs say so
MODEL_BOOK = "Fair odds"  # the "book" of a leg priced at the goals model's fair odds
_P_LOW, _P_HIGH = 0.01, 0.99  # chances the derived markets are kept inside
_RATE_LOW, _RATE_HIGH = 0.05, 6.0

Matrix = List[List[float]]


# --- the grid ----------------------------------------------------------------------------------------

def _pmf(rate: float, max_goals: int) -> List[float]:
    out = [math.exp(-rate)]
    for k in range(1, max_goals + 1):
        out.append(out[-1] * rate / k)
    return out


def grid(lam_home: float, lam_away: float, rho: float = DC_RHO, max_goals: int = MAX_GOALS) -> Matrix:
    """P(home scores x, away scores y) for x, y in 0..max_goals, rows by home goals; sums to 1."""
    if not (lam_home > 0 and lam_away > 0 and math.isfinite(lam_home) and math.isfinite(lam_away)):
        raise ValueError(f"Goal rates must be positive numbers, got {lam_home}, {lam_away}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be between -1 and 1, got {rho}")
    if not 1 <= int(max_goals) <= 30:
        raise ValueError(f"max_goals must be 1 to 30, got {max_goals}")
    max_goals = int(max_goals)
    home, away = _pmf(lam_home, max_goals), _pmf(lam_away, max_goals)
    cells = [[h * a for a in away] for h in home]
    tau = {(0, 0): 1 - lam_home * lam_away * rho, (0, 1): 1 + lam_home * rho,
           (1, 0): 1 + lam_away * rho, (1, 1): 1 - rho}
    for (x, y), factor in tau.items():
        cells[x][y] *= max(factor, 0.0)
    total = math.fsum(math.fsum(row) for row in cells)
    return [[value / total for value in row] for row in cells]


def outcome_probs(matrix: Matrix) -> Tuple[float, float, float]:
    """(home win, draw, away win) from a grid."""
    home = draw = away = 0.0
    for x, row in enumerate(matrix):
        for y, value in enumerate(row):
            if x > y:
                home += value
            elif x == y:
                draw += value
            else:
                away += value
    return home, draw, away


def over_prob(matrix: Matrix, line: float) -> float:
    """The chance the match total goes over ``line`` (a half line)."""
    return math.fsum(value for x, row in enumerate(matrix) for y, value in enumerate(row) if x + y > line)


def _fit_values(lam_home: float, lam_away: float, rho: float, max_goals: int) -> Tuple[float, float, float, float]:
    matrix = grid(lam_home, lam_away, rho, max_goals)
    home, draw, away = outcome_probs(matrix)
    return home, draw, away, over_prob(matrix, 2.5)


def _check_chance(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a chance, got {value!r}") from None
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be a chance between 0 and 1, got {value!r}")
    return number


def _solve2(a: float, b: float, c: float, d: float, e: float, f: float) -> Tuple[float, float]:
    """x, y from [[a, b], [c, d]] [x, y] = [e, f]."""
    det = a * d - b * c
    if abs(det) < 1e-300:
        return 0.0, 0.0
    return (e * d - b * f) / det, (a * f - c * e) / det


def rates_from_odds(p_home: float, p_draw: float, p_away: float, p_over_2_5: float, rho: float = DC_RHO,
                    max_goals: int = MAX_GOALS) -> Tuple[float, float]:
    """(home rate, away rate) whose Dixon-Coles grid best reproduces the margin-free 1X2 and over 2.5.

    The 1X2 chances are rescaled to sum to 1 first (so a price set with its margin still works). Least
    squares over the four chances, by Levenberg-Marquardt on the log rates from a coarse start; rates are kept
    between 0.05 and 6 goals. ``rho=0`` gives the plain independent-Poisson inversion.
    """
    h, d, a = (_check_chance(p_home, "p_home"), _check_chance(p_draw, "p_draw"), _check_chance(p_away, "p_away"))
    over = _check_chance(p_over_2_5, "p_over_2_5")
    total = h + d + a
    target = (h / total, d / total, a / total, over)

    def residuals(x: Tuple[float, float]) -> List[float]:
        values = _fit_values(math.exp(x[0]), math.exp(x[1]), rho, max_goals)
        return [m - t for m, t in zip(values, target)]

    def cost(r: Sequence[float]) -> float:
        return math.fsum(v * v for v in r)

    # A coarse start: the total that fits the over 2.5 on its own, split by the share that fits the 1X2 best
    low, high = 0.1, 12.0
    for _ in range(60):
        middle = (low + high) / 2
        under = math.exp(-middle) * (1 + middle + middle * middle / 2)
        if 1 - under < over:
            low = middle
        else:
            high = middle
    goals = (low + high) / 2
    start = min(((math.log(max(goals * s, _RATE_LOW)), math.log(max(goals * (1 - s), _RATE_LOW)))
                 for s in [i / 20 for i in range(1, 20)]), key=lambda x: cost(residuals(x)))

    x = start
    r = residuals(x)
    damping = 1e-3
    step = 1e-6
    lo, hi = math.log(_RATE_LOW), math.log(_RATE_HIGH)
    for _ in range(100):
        columns = []
        for i in range(2):
            up = list(x)
            down = list(x)
            up[i] += step
            down[i] -= step
            ru, rd = residuals(tuple(up)), residuals(tuple(down))
            columns.append([(u - v) / (2 * step) for u, v in zip(ru, rd)])
        j0, j1 = columns
        a11 = math.fsum(v * v for v in j0)
        a12 = math.fsum(u * v for u, v in zip(j0, j1))
        a22 = math.fsum(v * v for v in j1)
        g0 = math.fsum(u * v for u, v in zip(j0, r))
        g1 = math.fsum(u * v for u, v in zip(j1, r))
        improved = False
        for _ in range(20):
            dx, dy = _solve2(a11 * (1 + damping), a12, a12, a22 * (1 + damping), -g0, -g1)
            candidate = (min(max(x[0] + dx, lo), hi), min(max(x[1] + dy, lo), hi))
            rc = residuals(candidate)
            if cost(rc) < cost(r):
                moved = abs(candidate[0] - x[0]) + abs(candidate[1] - x[1])
                x, r = candidate, rc
                damping = max(damping / 3, 1e-9)
                improved = True
                break
            damping *= 4
        if not improved or moved < 1e-12:
            break
    return math.exp(x[0]), math.exp(x[1])


# --- the markets ------------------------------------------------------------------------------------

def _clamp(p: float) -> float:
    return min(max(p, _P_LOW), _P_HIGH)


def _line(line: float) -> str:
    return f"{line:g}"


def markets(matrix: Matrix, nudge: float = BTTS_NUDGE) -> Dict[str, Any]:
    """Every derived market from one grid. Team-total and total lines are string keys ("1.5"), so the dict
    survives a JSON round trip unchanged.

    btts: {"yes", "no", "raw"} with the nudge added to yes; dnb: {"home", "away"} as chances given no draw,
    with "push" the draw; double_chance: {"1X", "12", "X2"}; team_totals: {"home"|"away": {line: {"over",
    "under"}}}; totals: {line: {"over", "under"}}; correct_score: the top 6 [{"score": "1-0", "home", "away",
    "p"}]; plus "1x2", "clean_sheet" and "expected_goals".
    """
    home, draw, away = outcome_probs(matrix)
    size = len(matrix)
    home_marginal = [math.fsum(row) for row in matrix]
    away_marginal = [math.fsum(matrix[x][y] for x in range(size)) for y in range(len(matrix[0]))]
    both = math.fsum(value for x, row in enumerate(matrix) for y, value in enumerate(row) if x >= 1 and y >= 1)
    yes = _clamp(both + nudge)
    decided = home + away

    def totals_of(marginal: Sequence[float]) -> Dict[str, Dict[str, float]]:
        out = {}
        for line in TEAM_TOTAL_LINES:
            over = math.fsum(p for goals, p in enumerate(marginal) if goals > line)
            out[_line(line)] = {"over": over, "under": 1 - over}
        return out

    scores = sorted(((value, x, y) for x, row in enumerate(matrix) for y, value in enumerate(row)),
                    key=lambda item: (-item[0], item[1], item[2]))[:TOP_SCORES]
    return {
        "1x2": {"home": home, "draw": draw, "away": away},
        "btts": {"yes": yes, "no": 1 - yes, "raw": both},
        "dnb": {"home": home / decided if decided > 0 else 0.5, "away": away / decided if decided > 0 else 0.5,
                "push": draw},
        "double_chance": {"1X": home + draw, "12": home + away, "X2": draw + away},
        "team_totals": {"home": totals_of(home_marginal), "away": totals_of(away_marginal)},
        "totals": {_line(line): {"over": over_prob(matrix, line), "under": 1 - over_prob(matrix, line)}
                   for line in TOTAL_LINES},
        "clean_sheet": {"home": away_marginal[0], "away": home_marginal[0]},
        "expected_goals": {"home": math.fsum(x * p for x, p in enumerate(home_marginal)),
                           "away": math.fsum(y * p for y, p in enumerate(away_marginal))},
        "correct_score": [{"score": f"{x}-{y}", "home": x, "away": y, "p": value} for value, x, y in scores],
    }


def derive(p_home: float, p_draw: float, p_away: float, p_over_2_5: float, rho: float = DC_RHO,
           max_goals: int = MAX_GOALS, *, nudge: float = BTTS_NUDGE) -> Dict[str, Any]:
    """Rates, the fit against the inputs (model minus input, in points) and every derived market.

    ``rho`` and ``nudge`` default to DC_RHO and BTTS_NUDGE; the learning loop (learn/registry.py) passes its tuned
    values."""
    nudge = float(nudge)
    if not math.isfinite(nudge) or abs(nudge) > 0.2:
        raise ValueError(f"nudge must be a small chance, got {nudge!r}")
    lam_home, lam_away = rates_from_odds(p_home, p_draw, p_away, p_over_2_5, rho, max_goals)
    matrix = grid(lam_home, lam_away, rho, max_goals)
    derived = markets(matrix, nudge)
    total = p_home + p_draw + p_away
    fitted = (*outcome_probs(matrix), over_prob(matrix, 2.5))
    inputs = (p_home / total, p_draw / total, p_away / total, p_over_2_5)
    fit = {name: round((model - given) * 100, 3)
           for name, model, given in zip(("home", "draw", "away", "over_2_5"), fitted, inputs)}
    worst = max(abs(v) for v in fit.values())
    return {"rates": {"home": lam_home, "away": lam_away}, "rho": rho, "btts_nudge": nudge, "fit": fit,
            "max_error_pp": worst, "fit_ok": worst <= FIT_OK_PP, "markets": derived}


def fair_odds(chance: float, push: float = 0.0) -> Optional[float]:
    """The odds at which a leg breaks even: (1 - push) / chance."""
    return None if chance <= 0 else (1.0 - push) / chance


def fair_price(chance: float, push: float = 0.0) -> Optional[Decimal]:
    """The fair odds as a price: rounded down to the cent (never above the fair odds, so the rounding gives no leg an
    edge of its own), at least 1.01. The game page's "Goals model" shows the same price the Builder's legs use."""
    fair = fair_odds(chance, push)
    if fair is None:
        return None
    return max(Decimal(str(fair)).quantize(Decimal("0.01"), rounding=ROUND_DOWN), Decimal("1.01"))


# --- builder legs -------------------------------------------------------------------------------------

def _legs(derived: Mapping[str, Any], home: str, away: str) -> Iterable[Tuple[str, str, Optional[float], str,
                                                                            float, float, str]]:
    """(market, side, point, label, chance of a win outright, push chance, why) for every derived leg."""
    m = derived["markets"]
    rates = derived["rates"]
    btts = m["btts"]
    nudge = derived.get("btts_nudge", BTTS_NUDGE)
    nudge = BTTS_NUDGE if not isinstance(nudge, (int, float)) else float(nudge)
    if nudge > 0:
        yes_why, no_why = f", plus a {nudge * 100:g}-point calibration nudge", ", less the nudge"
    elif nudge < 0:
        yes_why, no_why = f", less a {-nudge * 100:g}-point calibration nudge", ", plus the nudge"
    else:
        yes_why = no_why = ""
    yield ("btts", "yes", None, "Both teams to score: Yes", btts["yes"], 0.0,
           f"Both score in {btts['raw']:.0%} of the model's scorelines{yes_why}")
    yield ("btts", "no", None, "Both teams to score: No", btts["no"], 0.0,
           f"At least one side is held scoreless in {1 - btts['raw']:.0%} of the model's scorelines{no_why}")
    one = m["1x2"]
    for side, team in (("home", home), ("away", away)):
        win = one[side]
        yield ("draw_no_bet", side, None, f"{team} (draw no bet)", win, one["draw"],
               f"{team} win {win:.0%}, draw {one['draw']:.0%} (stake back)")
    double = m["double_chance"]
    for side, label in (("1X", f"{home} or draw"), ("12", f"{home} or {away}"), ("X2", f"Draw or {away}")):
        yield ("double_chance", side, None, label, double[side], 0.0,
               f"Covers two of the three results: {double[side]:.0%}")
    for team_side, team in (("home", home), ("away", away)):
        lines = m["team_totals"][team_side]
        for line, chances in lines.items():
            for direction in ("over", "under"):
                yield ("team_totals", f"{team_side}_{direction}", float(line),
                       f"{team} {direction} {line} goals", chances[direction], 0.0,
                       f"{team} expected to score {rates[team_side]:.2f}")


def candidates(league: str, game_key: str, teams: Tuple[str, str], start: str, derived: Mapping[str, Any], *,
               ref_id: Optional[str] = None, prices: Optional[Mapping[Tuple[str, str, Optional[float]],
                                                                    Tuple[Any, str]]] = None,
               price_age_s: int = 0, flags: Iterable[str] = (), markets_wanted: Optional[Iterable[str]] = None,
               min_chance: float = 0.05) -> List[Any]:
    """Builder candidates (engine/builder.Candidate) for one game's derived markets.

    ``teams`` is (away, home), as in the builder. ``prices`` maps (market, side, point) to (decimal price,
    book) where a book prices the market on the cached board; every other leg is priced at its fair odds
    (rounded down to cents, at least 1.01), at the book "Fair odds", and flagged ``model_price``. ``flags`` are the
    game's own flags (e.g. "disagree", "low_conf"); ``price_age_s`` is the age of the prices the rates came
    from. Legs with a chance below ``min_chance`` (or above 1 - min_chance) are left out.
    """
    from .builder import Candidate
    away, home = teams
    wanted = set(markets_wanted) if markets_wanted is not None else None
    rates = derived["rates"]
    model_line = (f"Goals model: {home} {rates['home']:.2f}, {away} {rates['away']:.2f} expected goals, from the "
                  "1X2 and over/under 2.5 prices")
    worst = derived.get("max_error_pp")
    if worst is not None and worst > FIT_OK_PP:
        model_line += f" (it matches those prices only to within {worst:.1f} points)"
    out = []
    for market, side, point, label, chance, push, why in _legs(derived, home, away):
        if wanted is not None and market not in wanted:
            continue
        if not min_chance <= chance <= 1 - min_chance:
            continue
        leg_flags = set(flags)
        offer = (prices or {}).get((market, side, point))
        if offer is not None:
            price, book = Decimal(str(offer[0])), str(offer[1])
        else:
            price = fair_price(chance, push)
            book = MODEL_BOOK
            leg_flags.add("model_price")
        risk = ("Priced at fair odds by the goals model: check your bookmaker's price." if offer is None else None)
        out.append(Candidate(
            ref={"league": league, "id": str(ref_id or game_key), "market": market, "side": side, "point": point},
            league=league, game_key=game_key, teams=(away, home), start=start, price=price, book=book,
            chance=chance, chance_source="model", price_age_s=price_age_s, flags=leg_flags,
            why=[why, model_line], risk=risk, label=label, fair=None, push=push))
    return out
