"""Tests for the rollover risk planner.

The worked numbers come from the product blueprint's section "A ten-day
rollover at 2.0 finishes once in roughly 1,500 to 1,900 tries"; the planner
must reproduce them exactly (formulas) or within simulation noise (the
blueprint's 30-day strategy table came from 200,000 simulated runs).
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.engine.rollover import (
    STRATEGIES,
    bank_half_table,
    classic_table,
    compare,
    completion_chance,
    cost_sentence,
    day_probability,
    expected_multiple,
    honest_summary,
    kelly_fraction,
    outcomes,
    overstatement,
    required_daily_p,
    requirement_sentence,
    simulate,
    survival,
)

BANNED = ("lock", "guarantee", "sure", "risk-free", "can't lose", "win back")


def _ten_days_at_market() -> dict:
    return compare(Decimal("10"), 10, Decimal("2.00"), 1, 0.47)


# --- Formulas against the blueprint ---------------------------------------------------------------


def test_day_probability_compounds_the_cut_per_leg() -> None:
    assert day_probability(Decimal("2.0"), 1, cut=0.04) == pytest.approx(0.48)
    assert day_probability(Decimal("2.0"), 1, cut=0.06) == pytest.approx(0.47)
    assert day_probability(Decimal("2.0"), 2, cut=0.04) == pytest.approx(0.4608)
    assert day_probability(Decimal("2.0"), 1) == pytest.approx(0.4775)  # default 4.5% cut
    with pytest.raises(ValueError):
        day_probability(Decimal("1.0"), 1)
    with pytest.raises(ValueError):
        day_probability(Decimal("2.0"), 0)


@pytest.mark.parametrize("p, days, one_in, ev", [
    (0.48, 5, 39, -0.185), (0.48, 7, 170, -0.249), (0.48, 10, 1540, -0.335), (0.48, 14, 29015, -0.435),
    (0.47, 5, 44, -0.266), (0.47, 7, 197, -0.352), (0.47, 10, 1901, -0.461), (0.47, 14, 38961, -0.579),
    (0.52, 5, 26, 0.217), (0.52, 7, 97, 0.316), (0.52, 10, 692, 0.480), (0.52, 14, 9461, 0.732),
    (0.55, 5, 20, 0.611), (0.55, 7, 66, 0.949), (0.55, 10, 395, 1.59), (0.55, 14, 4314, 2.80),
])
def test_blueprint_completion_table(p: float, days: int, one_in: int, ev: float) -> None:
    assert round(1 / completion_chance(p, days)) == one_in
    assert expected_multiple(p, 2.0, days) - 1 == pytest.approx(ev, rel=0.003)  # the blueprint rounds to 3 figures


def test_survival_at_market_odds() -> None:
    curve = survival(0.47, 10)
    assert curve[:3] == pytest.approx([0.47, 0.2209, 0.103823])
    assert curve[4] == pytest.approx(0.023, abs=0.0005)
    assert 1 - curve[1] == pytest.approx(0.7791)  # about 78% are over by day 2
    assert sum(curve) == pytest.approx(0.886, abs=0.001)  # the average run wins fewer than one day


def test_required_daily_p_for_a_coin_flip_over_ten_days() -> None:
    needed = required_daily_p(0.5, 10)
    assert needed == pytest.approx(0.933, abs=0.0005)
    assert needed * 2.0 - 1 == pytest.approx(0.866, abs=0.001)  # an 87% edge per bet at 2.0
    assert required_daily_p(1.0, 5) == 1.0
    for bad_target, bad_days in ((0.0, 10), (1.5, 10), (0.5, 0)):
        with pytest.raises(ValueError):
            required_daily_p(bad_target, bad_days)


def test_small_ai_overconfidence_balloons_over_a_run() -> None:
    assert overstatement(0.52, 0.47, 10) == pytest.approx(2.75, abs=0.01)
    assert overstatement(0.52, 0.47, 30) == pytest.approx(20.8, abs=0.05)


def test_all_in_with_a_small_edge_is_many_times_kelly() -> None:
    assert 1 / kelly_fraction(0.52, 2.0) == pytest.approx(25)
    assert 1 / kelly_fraction(0.55, 2.0) == pytest.approx(10)
    assert kelly_fraction(0.47, 2.0) == 0.0


# --- compare(): the side-by-side planner -------------------------------------------------------------


def test_ten_day_classic_rollover_at_market_odds() -> None:
    result = _ten_days_at_market()
    classic = result["strategies"]["classic"]["market"]
    assert result["target"] == "10240.00"
    assert classic["complete_one_in"] == 1901
    assert classic["chance_complete"] == pytest.approx(0.47 ** 10, rel=1e-3)
    assert classic["chance_profit"] == classic["chance_complete"]
    assert classic["survival"][:2] == [0.47, 0.2209]
    assert classic["over_by_day_2"] == 0.7791
    assert classic["most_likely_end_day"] == 1
    assert classic["most_likely_end"] == {"day": 1, "probability": 0.53, "completes": False}
    assert classic["average_days_survived"] == 0.89
    assert classic["median_end"] == "0.00"
    assert classic["chance_half_or_less"] == pytest.approx(1 - 0.47 ** 10, abs=1e-4)
    # Expected end (pD)^N of the stake: 10 * 0.94^10 = 5.39, so the plan costs 4.61 on average.
    assert classic["expected_end"] == "5.39"
    assert classic["expected_cost"] == "4.61"
    assert classic["best_end"] == "10240.00"
    assert result["strategies"]["classic"]["ai"] is None
    assert result["p_market_estimated"] is False


def test_two_legs_a_day_compound_the_cut() -> None:
    p = day_probability(Decimal("2.0"), 2, cut=0.04)
    two_legs = compare(Decimal("100"), 10, Decimal("2.0"), 2, p)["strategies"]["classic"]["market"]
    one_leg = compare(Decimal("100"), 10, Decimal("2.0"), 1, 0.48)["strategies"]["classic"]["market"]
    assert two_legs["complete_one_in"] == 2317
    assert float(two_legs["expected_cost"]) / 100 == pytest.approx(0.558, abs=0.001)
    assert float(one_leg["expected_cost"]) / 100 == pytest.approx(0.335, abs=0.001)


@pytest.mark.parametrize("strategy, pot_basis, p, profit, median, half_or_less", [
    ("classic", "start", 0.47, 0.0, 0.0, 1.0),
    ("classic", "start", 0.53, 0.0, 0.0, 1.0),
    ("bank_half", "start", 0.47, 0.221, 0.0, 0.779),
    ("bank_half", "start", 0.53, 0.280, 0.50, 0.720),
    ("ladder", "bankroll", 0.47, 0.224, 0.48, 0.539),
    ("ladder", "bankroll", 0.53, 0.449, 0.93, 0.314),
    ("flat", "start", 0.47, 0.306, 0.90, 0.093),
    ("flat", "start", 0.53, 0.560, 1.10, 0.024),
])
def test_blueprint_thirty_day_strategy_table(strategy: str, pot_basis: str, p: float, profit: float,
                                             median: float, half_or_less: float) -> None:
    # The blueprint simulated its ladder with a pot of 10% of the current bankroll; the planner's
    # default (the spec) takes 10% of the start, so the blueprint row is checked with pot_basis="bankroll".
    result = outcomes(strategy, 30, 2.0, p, pot_basis=pot_basis)
    assert result.share(lambda value: value > 1 + 1e-9) == pytest.approx(profit, abs=0.005)
    assert result.median == pytest.approx(median, abs=0.01)
    assert result.share(lambda value: value <= 0.5 + 1e-9) == pytest.approx(half_or_less, abs=0.005)


def test_bank_half_is_exact_at_even_money() -> None:
    # At 2.0, two wins bank 1.25 (profit) and one win banks 0.5 (half), whatever happens later.
    for p in (0.47, 0.53):
        result = outcomes("bank_half", 30, 2.0, p)
        assert result.share(lambda value: value > 1) == pytest.approx(p ** 2)
        assert result.share(lambda value: value <= 0.5 + 1e-9) == pytest.approx(1 - p ** 2)


def test_ladder_stop_rule_keeps_most_of_the_bankroll() -> None:
    result = outcomes("ladder", 30, 2.0, 0.47)
    assert min(value for value, _ in result.finals) >= 0.4 - 1e-9  # stops before a pot can take it below 40%
    assert 0 < result.chance_complete < 1
    assert result.best == pytest.approx(1 + 15 * 0.1 * 3)  # 15 pots of 0.1 that each became 0.4


def test_flat_stakes_lose_least_at_market_odds() -> None:
    for days in (10, 30):
        result = compare(Decimal("10"), days, Decimal("2.0"), 1, 0.47)["strategies"]
        medians = {name: Decimal(result[name]["market"]["median_end"]) for name in STRATEGIES}
        assert max(medians, key=medians.get) == "flat"
        assert medians["flat"] >= medians["ladder"] >= medians["bank_half"] >= medians["classic"]
    strategies = _ten_days_at_market()["strategies"]
    costs = {name: Decimal(strategies[name]["market"]["expected_cost"]) for name in STRATEGIES}
    assert costs["flat"] < costs["ladder"] < costs["bank_half"] < costs["classic"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("p", [0.3, 0.47, 0.6])
def test_distributions_are_complete(strategy: str, p: float) -> None:
    result = outcomes(strategy, 20, 1.8, p)
    assert sum(chance for _, chance in result.finals) == pytest.approx(1.0)
    curve = result.survival
    assert all(later <= earlier + 1e-12 for earlier, later in zip(curve, curve[1:]))
    assert result.chance_complete == pytest.approx(curve[-1])


def test_classic_distribution_matches_the_formulas() -> None:
    result = outcomes("classic", 12, 1.9, 0.5)
    assert result.chance_complete == pytest.approx(completion_chance(0.5, 12))
    assert result.mean == pytest.approx(expected_multiple(0.5, 1.9, 12))
    assert result.survival == pytest.approx(survival(0.5, 12))


@pytest.mark.parametrize("strategy", ["classic", "bank_half"])
@pytest.mark.parametrize("p, days", [(0.47, 60), (0.3, 40), (0.4608, 60)])
def test_long_runs_keep_tiny_completion_chances_exact(strategy: str, p: float, days: int) -> None:
    # p ** days drops to 1e-20 and below; 1 - (chance of stopping) would cancel it to noise or 0.
    result = outcomes(strategy, days, 2.0, p)
    assert result.chance_complete == pytest.approx(p ** days, rel=1e-9)
    assert result.survival == pytest.approx(survival(p, days), rel=1e-9)
    assert result.survival == pytest.approx([p ** day for day in range(1, days + 1)], rel=1e-9)


@pytest.mark.parametrize("days, min_odds, legs, p_market", [
    (60, Decimal("2.0"), 1, 0.47), (60, Decimal("2.0"), 2, None), (40, Decimal("3.0"), 1, None),
])
def test_long_runs_agree_across_one_response(days: int, min_odds: Decimal, legs: int,
                                             p_market: float) -> None:
    result = compare(Decimal("10"), days, min_odds, legs, p_market)
    p = result["p_market"] if p_market is None else p_market
    classic = result["strategies"]["classic"]["market"]
    last_row = result["ladder_table"][-1]
    assert classic["chance_complete"] == pytest.approx(p ** days, rel=1e-3)  # the shown value has 4 figures
    assert classic["complete_one_in"] is not None
    assert classic["complete_one_in"] == last_row["market_one_in"]
    assert classic["chance_complete"] == last_row["market_chance"] == classic["chance_profit"]
    assert classic["survival"][-1] == classic["chance_complete"]
    bank_half = result["strategies"]["bank_half"]["market"]
    assert bank_half["complete_one_in"] == result["bank_half_table"][-1]["market_one_in"]


def test_vanishing_chances_have_no_one_in_number() -> None:
    # 1e-6 ** 52 is below the smallest normal float: 1 / p would overflow, so no "1 in X" is given.
    result = compare(Decimal("10"), 60, Decimal("2.0"), 1, 1e-6)
    classic = result["strategies"]["classic"]["market"]
    assert classic["complete_one_in"] is None
    assert result["ladder_table"][0]["market_one_in"] == 1000000
    assert result["ladder_table"][-1]["market_one_in"] is None
    assert "less than 1 in 10^300" in result["summary"]


def test_compare_is_deterministic_and_fast() -> None:
    assert _ten_days_at_market() == _ten_days_at_market()
    began = time.perf_counter()
    compare(Decimal("50"), 60, Decimal("1.5"), 2, None, 0.7)
    assert time.perf_counter() - began < 1.0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_seeded_simulation_agrees_with_exact_enumeration(strategy: str) -> None:
    runs = simulate(strategy, 10, 2.0, 0.47, runs=20000, seed=7)
    assert runs == simulate(strategy, 10, 2.0, 0.47, runs=20000, seed=7)
    assert runs != simulate(strategy, 10, 2.0, 0.47, runs=20000, seed=8)
    exact = outcomes(strategy, 10, 2.0, 0.47)
    # Shares only: the mean of a few thousand runs is dominated by rare complete runs.
    for test in (lambda value: value > 1 + 1e-9, lambda value: value <= 0.5 + 1e-9):
        simulated = sum(test(value) for value in runs) / len(runs)
        assert simulated == pytest.approx(exact.share(test), abs=0.015)


def test_ai_probability_is_shown_next_to_the_market() -> None:
    result = compare(Decimal("10"), 10, Decimal("2.00"), 1, 0.47, p_ai=0.52)
    ai = result["strategies"]["classic"]["ai"]
    assert ai["complete_one_in"] == 692
    assert result["overstatement"] == 2.75
    assert result["ladder_table"][9]["ai_one_in"] == 692
    assert any("overstates" in note for note in result["warnings"])
    assert ("At the AI's daily chance of 52% and odds of 2.00, staking everything is about 25 times the Kelly "
            "stake (the stake that grows a bankroll fastest; more only adds risk).") in result["warnings"]
    for name in STRATEGIES:
        assert result["strategies"][name]["ai"]["p_day"] == 0.52


def test_market_probability_is_estimated_when_missing() -> None:
    result = compare(Decimal("10"), 5, Decimal("2.00"), 2)
    assert result["p_market"] == pytest.approx(round(0.955 ** 2 / 2, 4))
    assert result["p_market_estimated"] is True
    assert any("estimated" in note for note in result["warnings"])
    assert any("2 legs" in note for note in result["warnings"])


def test_kelly_note_stands_alone_without_an_ai_edge() -> None:
    # No AI chance, or an AI at or below the market: no overstatement note comes before the Kelly note.
    for p_market, p_ai, source in ((0.55, None, "the market's"), (0.58, 0.55, "the AI's")):
        warnings = compare(Decimal("10"), 5, Decimal("2.0"), 1, p_market, p_ai)["warnings"]
        assert not any("overstates" in note or "Even if" in note for note in warnings)
        kelly = [note for note in warnings if "Kelly stake" in note]
        assert kelly == [f"At {source} daily chance of 55% and odds of 2.00, staking everything is about 10 times "
                         "the Kelly stake (the stake that grows a bankroll fastest; more only adds risk)."]
    # A tiny edge gives a large multiple, written with thousands separators.
    warnings = compare(Decimal("10"), 5, Decimal("2.0"), 1, 0.5001)["warnings"]
    assert any("about 5,000 times the Kelly stake" in note for note in warnings)


@pytest.mark.parametrize("kwargs", [
    {"days": 0}, {"days": 61}, {"min_odds": Decimal("1.1")}, {"legs_per_day": 0}, {"legs_per_day": 4},
    {"p_market": 0.0}, {"p_market": 1.0}, {"p_ai": 1.2}, {"start": Decimal("0")},
    {"start": Decimal("NaN")}, {"start": float("nan")}, {"start": Decimal("Infinity")}, {"start": "ten"},
    {"start": Decimal("1000000000.01")}, {"min_odds": Decimal("NaN")}, {"min_odds": float("nan")},
    {"min_odds": Decimal("Infinity")}, {"min_odds": float("inf")}, {"days": 10.0}, {"days": True},
    {"legs_per_day": 1.5}, {"legs_per_day": True}, {"p_market": float("nan")}, {"p_ai": float("inf")},
])
def test_compare_rejects_bad_inputs(kwargs: dict) -> None:
    args = {"start": Decimal("10"), "days": 10, "min_odds": Decimal("2.0"), "legs_per_day": 1, "p_market": 0.47}
    args.update(kwargs)
    with pytest.raises(ValueError):
        compare(**args)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        outcomes("martingale", 10, 2.0, 0.47)
    with pytest.raises(ValueError):
        outcomes("ladder", 10, 2.0, 0.47, pot_basis="budget")


# --- Day-by-day tables ----------------------------------------------------------------------------


def test_classic_table_if_every_day_wins() -> None:
    rows = classic_table(Decimal("10"), 10, Decimal("2.00"), 0.47)
    assert rows[0] == {"day": 1, "stake": "10.00", "returns": "20.00", "market_chance": 0.47,
                       "market_one_in": 2, "ai_chance": None, "ai_one_in": None}
    assert (rows[9]["stake"], rows[9]["returns"], rows[9]["market_one_in"]) == ("5120.00", "10240.00", 1901)
    # Exact decimal maths, rounded once: 10 * 1.91^3 = 69.68171.
    assert classic_table(Decimal("10"), 3, Decimal("1.91"), 0.5)[2]["returns"] == "69.68"


def test_bank_half_table_banks_half_of_each_profit() -> None:
    rows = bank_half_table(Decimal("10"), 10, Decimal("2.00"), 0.47)
    first, second = rows[0], rows[1]
    assert (first["stake"], first["returns"], first["bank"], first["banked_total"], first["roll"]) == (
        "10.00", "20.00", "5.00", "5.00", "15.00")
    assert (second["stake"], second["returns"], second["bank"], second["banked_total"], second["roll"]) == (
        "15.00", "30.00", "7.50", "12.50", "22.50")
    # If every day wins, the run ends with everything banked plus the last roll.
    best = compare(Decimal("10"), 10, Decimal("2.00"), 1, 0.47)["strategies"]["bank_half"]["market"]["best_end"]
    assert Decimal(rows[-1]["banked_total"]) + Decimal(rows[-1]["roll"]) == Decimal(best)


def test_money_is_returned_in_cents() -> None:
    result = compare(Decimal("12.34"), 7, Decimal("1.85"), 1, 0.5)
    for name in STRATEGIES:
        stats = result["strategies"][name]["market"]
        for key in ("median_end", "expected_end", "expected_cost", "best_end"):
            assert Decimal(stats[key]).as_tuple().exponent == -2
        assert Decimal(stats["expected_end"]) + Decimal(stats["expected_cost"]) == Decimal("12.34")


# --- Honest copy ----------------------------------------------------------------------------------


def test_honest_summary_sentence() -> None:
    assert _ten_days_at_market()["summary"] == (
        "To turn $10 into $10,240 in 10 days you need 10 wins in a row at 2.00. "
        "The market gives that about 1 in 1,900. Most likely the run ends on day 1 (53%).")
    assert honest_summary(Decimal("10"), 10, Decimal("2.00"), 0.47, p_ai=0.52) == (
        "To turn $10 into $10,240 in 10 days you need 10 wins in a row at 2.00. "
        "The market gives that about 1 in 1,900 and the AI about 1 in 690. Most likely the run ends on day 1 (53%).")
    assert honest_summary(Decimal("10"), 1, Decimal("1.50"), 0.6) == (
        "To turn $10 into $15 in 1 day you need 1 win at 1.50. "
        "The market gives that about a 60% chance. Most likely the run completes (60%).")
    assert honest_summary(Decimal("5.50"), 5, Decimal("2.0"), 0.48, legs_per_day=2, currency="£") == (
        "To turn £5.50 into £176 in 5 days you need 5 winning days in a row at 2.00, each built from 2 legs. "
        "The market gives that about 1 in 39 (2.5%). Most likely the run ends on day 1 (52%).")


def test_odds_show_the_same_everywhere() -> None:
    # 1.625 (American -160) rounds half up, as in tickets.py: never "1.63" in one place and "1.62" in another.
    result = compare(Decimal("10"), 10, Decimal("1.625"), 1, 0.58)
    assert result["min_odds"] == "1.63"
    assert "10 wins in a row at 1.63." in result["summary"]
    assert "per bet at 1.63." in result["requirement"]
    assert "at 1.62" not in result["summary"] + result["requirement"]


def test_requirement_and_cost_sentences() -> None:
    assert requirement_sentence(10, Decimal("2.00"), 0.47) == (
        "A coin-flip chance of finishing 10 days needs a 93% win chance every day, an edge of 87% per bet "
        "at 2.00. The market gives each day 47%.")
    # "an" before numbers read with a vowel sound: eighty-seven, eleven, eighteen, eight.
    assert "needs an 87% win chance" in requirement_sentence(5, Decimal("2.00"), 0.47)
    assert requirement_sentence(1, Decimal("2.00"), 0.47, target=0.8).startswith(
        "An 80% chance of finishing 1 day needs an 80% win chance every day")
    assert requirement_sentence(1, Decimal("20.0"), 0.04, target=0.11).startswith(
        "An 11% chance of finishing 1 day needs an 11% win chance")
    assert requirement_sentence(1, Decimal("20.0"), 0.04, target=0.18).startswith("An 18% chance")
    assert requirement_sentence(1, Decimal("20.0"), 0.04, target=0.08).startswith("An 8.0% chance")
    assert requirement_sentence(1, Decimal("20.0"), 0.04, target=0.12).startswith("A 12% chance")
    assert "without an edge" in requirement_sentence(1, Decimal("1.50"), 0.6)
    assert cost_sentence(Decimal("10"), Decimal("5.39")) == (
        "On average a run returns $5.39 of the $10 staked: an expected cost of $4.61.")
    assert "if the daily chance used is right" in cost_sentence(Decimal("10"), Decimal("14.80"))


def test_copy_never_uses_banned_phrases() -> None:
    result = compare(Decimal("10"), 10, Decimal("2.00"), 2, 0.47, p_ai=0.55)
    texts = [result["summary"], result["requirement"], *result["warnings"],
             *(entry["rule"] for entry in result["strategies"].values())]
    for text in texts:
        for phrase in BANNED:
            assert phrase not in text.lower(), (phrase, text)
