"""The soccer goals model: rates from 1X2 + over/under 2.5, the Dixon-Coles grid and the derived markets."""

from __future__ import annotations

import datetime
import math
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.engine import soccer
from telegram_betting_bot_simple.engine.builder import Controls, build
from telegram_betting_bot_simple.engine.consensus import no_vig
from telegram_betting_bot_simple.engine.presets import PRESETS

NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.timezone.utc)
LATER = "2026-10-27T19:00:00Z"


def poisson(rate: float, k: int) -> float:
    return math.exp(-rate) * rate ** k / math.factorial(k)


# --- the grid -----------------------------------------------------------------------------------

@pytest.mark.parametrize("home, away, rho", [(1.557, 1.245, -0.13), (0.4, 0.3, -0.13), (3.2, 0.7, -0.13),
                                             (1.5, 1.5, 0.0), (2.0, 1.0, 0.1), (5.5, 4.5, -0.13)])
def test_the_grid_sums_to_one(home, away, rho) -> None:
    grid = soccer.grid(home, away, rho)
    assert len(grid) == soccer.MAX_GOALS + 1 and all(len(row) == soccer.MAX_GOALS + 1 for row in grid)
    assert math.fsum(math.fsum(row) for row in grid) == pytest.approx(1.0, abs=1e-12)
    assert all(value >= 0 for row in grid for value in row)


def test_dixon_coles_moves_only_the_four_low_scores() -> None:
    lam, mu, rho = 1.4, 1.1, -0.13
    dc, plain = soccer.grid(lam, mu, rho, max_goals=20), soccer.grid(lam, mu, 0.0, max_goals=20)
    # With 20 goals a side the truncation is negligible, so the cells are the textbook ones
    assert plain[2][1] == pytest.approx(poisson(lam, 2) * poisson(mu, 1), rel=1e-9)
    assert dc[0][0] == pytest.approx(plain[0][0] * (1 - lam * mu * rho), rel=1e-9)  # 0-0 up
    assert dc[0][1] == pytest.approx(plain[0][1] * (1 + lam * rho), rel=1e-9)  # 0-1 down
    assert dc[1][0] == pytest.approx(plain[1][0] * (1 + mu * rho), rel=1e-9)  # 1-0 down
    assert dc[1][1] == pytest.approx(plain[1][1] * (1 - rho), rel=1e-9)  # 1-1 up
    assert dc[2][2] == pytest.approx(plain[2][2], rel=1e-9) and dc[3][0] == pytest.approx(plain[3][0], rel=1e-9)
    # A negative rho means more draws than independent Poissons
    assert soccer.outcome_probs(dc)[1] > soccer.outcome_probs(plain)[1]


def test_bad_grid_inputs_raise() -> None:
    for args in [(0, 1.0), (1.0, -1), (float("nan"), 1.0)]:
        with pytest.raises(ValueError):
            soccer.grid(*args)
    with pytest.raises(ValueError):
        soccer.grid(1.0, 1.0, rho=2.0)
    with pytest.raises(ValueError):
        soccer.rates_from_odds(0.5, 0.3, 0.2, 1.2)
    with pytest.raises(ValueError):
        soccer.rates_from_odds(0.5, "x", 0.2, 0.5)


# --- the rates ----------------------------------------------------------------------------------

@pytest.mark.parametrize("home, away", [(1.557, 1.245), (2.6, 0.8), (0.7, 1.9), (1.0, 1.0), (0.45, 0.35),
                                        (3.4, 1.6)])
def test_rates_come_back_from_their_own_grid(home, away) -> None:
    grid = soccer.grid(home, away)
    p_home, p_draw, p_away = soccer.outcome_probs(grid)
    lam, mu = soccer.rates_from_odds(p_home, p_draw, p_away, soccer.over_prob(grid, 2.5))
    assert lam == pytest.approx(home, rel=1e-4) and mu == pytest.approx(away, rel=1e-4)


# Typical prices: an even league game, a Serie A home favourite, an away favourite, a low total, a clear favourite
MARKET_LINES = [((2.10, 3.40, 3.60), (1.90, 1.95)), ((1.95, 3.30, 4.20), (2.05, 1.80)),
                ((4.20, 3.60, 1.85), (1.80, 2.05)), ((2.45, 3.05, 3.30), (2.40, 1.57)),
                ((1.60, 4.00, 5.50), (1.70, 2.15))]


@pytest.mark.parametrize("one_x_two, over_under", MARKET_LINES)
def test_rates_reproduce_the_market_1x2_and_over_under_within_half_a_point(one_x_two, over_under) -> None:
    p = no_vig(dict(zip(("home", "draw", "away"), one_x_two)))
    over = no_vig(dict(zip(("over", "under"), over_under)))["over"]
    derived = soccer.derive(p["home"], p["draw"], p["away"], over)
    grid = soccer.grid(derived["rates"]["home"], derived["rates"]["away"])
    fitted = (*soccer.outcome_probs(grid), soccer.over_prob(grid, 2.5))
    for model, given in zip(fitted, (p["home"], p["draw"], p["away"], over)):
        assert abs(model - given) <= 0.005
    assert derived["max_error_pp"] <= 0.5 and derived["fit_ok"]
    m = derived["markets"]["1x2"]
    assert (m["home"], m["draw"], m["away"]) == pytest.approx(fitted[:3])


def test_a_poor_fit_is_reported_on_the_legs() -> None:
    # Two rates can't always reproduce four prices: a lopsided line misses by more than a point, and says so
    p = no_vig({"home": 1.25, "draw": 6.5, "away": 11.0})
    derived = soccer.derive(p["home"], p["draw"], p["away"], no_vig({"over": 1.44, "under": 2.80})["over"])
    assert 1.0 < derived["max_error_pp"] < 2.0 and not derived["fit_ok"]
    legs = soccer.candidates("soccer_epl", "g", ("Away", "Home"), LATER, derived)
    assert all("only to within 1.2 points" in leg.why[1] for leg in legs)


def test_a_1x2_with_its_margin_is_rescaled_first() -> None:
    with_margin = soccer.rates_from_odds(1 / 2.10, 1 / 3.40, 1 / 3.60, 0.5)
    without = soccer.rates_from_odds(*(v / (1 / 2.10 + 1 / 3.40 + 1 / 3.60) for v in (1 / 2.10, 1 / 3.40, 1 / 3.60)),
                                     0.5)
    assert with_margin == pytest.approx(without)


def test_the_report_average_match() -> None:
    # The report's implied means: 1.557 home, 1.245 away; expected goals read off the grid give them back
    m = soccer.markets(soccer.grid(1.557, 1.245))
    assert m["expected_goals"]["home"] == pytest.approx(1.557, abs=0.002)
    assert m["expected_goals"]["away"] == pytest.approx(1.245, abs=0.002)


# --- the markets --------------------------------------------------------------------------------

def test_btts_gets_the_one_and_a_half_point_nudge() -> None:
    grid = soccer.grid(1.557, 1.245)
    raw = math.fsum(v for x, row in enumerate(grid) for y, v in enumerate(row) if x and y)
    btts = soccer.markets(grid)["btts"]
    assert btts["raw"] == pytest.approx(raw)
    assert btts["yes"] == pytest.approx(raw + 0.015) and btts["no"] == pytest.approx(1 - raw - 0.015)
    assert soccer.markets(grid, nudge=0.0)["btts"]["yes"] == pytest.approx(raw)
    # Kept inside 1-99% whatever the rates
    tiny = soccer.markets(soccer.grid(0.05, 0.05))["btts"]
    assert tiny["yes"] == 0.015 + tiny["raw"] and 0.01 <= tiny["yes"] <= 0.99


def test_derived_markets_hang_together() -> None:
    grid = soccer.grid(1.8, 1.0)
    home, draw, away = soccer.outcome_probs(grid)
    m = soccer.markets(grid)
    assert m["dnb"]["home"] == pytest.approx(home / (home + away)) and m["dnb"]["push"] == pytest.approx(draw)
    assert m["dnb"]["home"] + m["dnb"]["away"] == pytest.approx(1.0)
    dc = m["double_chance"]
    assert dc["1X"] == pytest.approx(home + draw) and dc["X2"] == pytest.approx(draw + away)
    assert dc["1X"] + dc["12"] + dc["X2"] == pytest.approx(2.0)
    for team in ("home", "away"):
        lines = m["team_totals"][team]
        assert list(lines) == ["0.5", "1.5", "2.5", "3.5"]
        assert all(v["over"] + v["under"] == pytest.approx(1.0) for v in lines.values())
        overs = [v["over"] for v in lines.values()]
        assert overs == sorted(overs, reverse=True)
    assert m["clean_sheet"]["home"] == pytest.approx(1 - m["team_totals"]["away"]["0.5"]["over"])
    assert m["totals"]["2.5"]["over"] == pytest.approx(soccer.over_prob(grid, 2.5))
    scores = m["correct_score"]
    assert len(scores) == 6 and [s["p"] for s in scores] == sorted((s["p"] for s in scores), reverse=True)
    assert scores[0]["score"] == f"{scores[0]['home']}-{scores[0]['away']}"
    assert scores[0]["p"] == pytest.approx(max(v for row in grid for v in row))


# --- builder legs -------------------------------------------------------------------------------

def soccer_legs(prices=None, flags=()) -> list:
    p = no_vig({"home": 1.95, "draw": 3.30, "away": 4.20})
    over = no_vig({"over": 2.05, "under": 1.80})["over"]
    derived = soccer.derive(p["home"], p["draw"], p["away"], over)
    return soccer.candidates("soccer_epl", "epl1", ("Spurs", "Arsenal"), LATER, derived, prices=prices,
                             price_age_s=900, flags=flags), derived


def test_legs_without_a_book_price_are_priced_at_fair_odds_and_flagged() -> None:
    legs, derived = soccer_legs(prices={("btts", "yes", None): (Decimal("1.80"), "DraftKings")})
    by_ref = {(leg.market, leg.side, leg.point): leg for leg in legs}
    assert {leg.market for leg in legs} == {"btts", "draw_no_bet", "double_chance", "team_totals"}
    yes = by_ref[("btts", "yes", None)]
    assert yes.price == Decimal("1.80") and yes.book == "DraftKings" and "model_price" not in yes.flags
    assert yes.chance == pytest.approx(derived["markets"]["btts"]["yes"])
    no = by_ref[("btts", "no", None)]
    assert "model_price" in no.flags and no.book == soccer.MODEL_BOOK and no.chance_source == "model"
    assert float(no.price) == pytest.approx(1 / no.chance, abs=0.006) and abs(no.edge) < 0.01
    # Rounded down to the cent: a fair-odds price is never above the fair odds, so rounding gives no leg an edge
    fair_priced = [leg for leg in legs if "model_price" in leg.flags]
    assert fair_priced and all(float(leg.price) <= soccer.fair_odds(leg.chance, leg.push) + 1e-9
                               or leg.price == Decimal("1.01") for leg in fair_priced)
    assert all(leg.edge <= 1e-9 for leg in fair_priced if leg.price > Decimal("1.01"))
    dnb = by_ref[("draw_no_bet", "home", None)]
    one = derived["markets"]["1x2"]
    assert dnb.chance == pytest.approx(one["home"]) and dnb.push == pytest.approx(one["draw"])
    assert float(dnb.price) == pytest.approx((1 - one["draw"]) / one["home"], abs=0.006)
    assert by_ref[("double_chance", "1X", None)].label == "Arsenal or draw"
    assert by_ref[("team_totals", "home_over", 1.5)].label == "Arsenal over 1.5 goals"
    assert all(leg.teams == ("Spurs", "Arsenal") and leg.price_age_s == 900 for leg in legs)
    assert all(len(leg.why) == 2 and leg.why[1].startswith("Goals model:") for leg in legs)


def test_model_priced_legs_stay_out_of_safe_and_say_check_your_price() -> None:
    legs, _ = soccer_legs()
    safe = build(legs, PRESETS["safe"], NOW)
    assert not safe.tickets and safe.stats["removed"]["market"] == len(legs)  # Safe's markets: ML, spread, total
    # Safe with the soccer markets added is Custom, and still takes no fair-odds leg
    wider = build(legs, Controls.from_dict({"preset": "safe", "markets": ["h2h", "double_chance", "btts"]}), NOW)
    assert wider.label == "Custom" and not wider.tickets
    assert wider.near_misses and all("model_price" in miss["rules"] for miss in wider.near_misses)
    only_model = [miss for miss in wider.near_misses if miss["rules"] == ["model_price"]]
    assert only_model and "priced at fair odds by the goals model (excluded)" in only_model[0]["text"]
    risky = build(legs, Controls.from_dict({"preset": "risky", "markets": ["double_chance", "btts", "draw_no_bet"]}),
                  NOW)
    assert risky.tickets and risky.label == "Custom"
    ticket = risky.tickets[0]
    assert ticket["model_priced"] and any("Priced at fair odds; check your bookmaker's price" in n for n in ticket["notes"])
    assert all(leg["model_price"] and leg["book"] == soccer.MODEL_BOOK for leg in ticket["legs"])
    # Allowed books don't apply to a leg no book prices
    only = build(legs, Controls.from_dict({"preset": "risky", "markets": ["double_chance"], "books": ["fanduel"]}), NOW)
    assert only.tickets


def test_game_flags_carry_onto_every_leg() -> None:
    legs, _ = soccer_legs(flags=("disagree",))
    assert all("disagree" in leg.flags for leg in legs)
