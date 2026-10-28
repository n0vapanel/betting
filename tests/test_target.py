"""Tests for the stake-and-target builder: the honest ceiling, the search, the scoring and the verdict.

Every candidate is synthetic, and every expected number is worked by hand in the comment next to it.
The cautious chance is p_c = market + 0.5 * (ai - market), both outright (the market's no-push chance times
1 - push); a leg's cautious return is p_c * odds + push.
"""

from __future__ import annotations

import datetime
import math
import time
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.engine.target import (
    CUT,
    POOL_SIZE,
    build_for_target,
    cautious_probability,
    no_edge_chance,
)
from telegram_betting_bot_simple.engine.tickets import TicketLeg

NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.timezone.utc)
LATER = "2026-10-25T23:00:00Z"
EARLIER = "2026-10-25T10:00:00Z"
BANNED = ("lock", "guarantee", "sure", "risk-free", "can't lose", "win back")


def leg(i, price, market=None, ai=None, push=0.0, event=None, group=None, start=LATER, key=None,
        book=None) -> TicketLeg:
    """Leg i of its own game unless ``event`` (or ``group``) ties it to another leg's game."""
    event = event or f"e{i}"
    return TicketLeg(key=key or f"basketball_nba:{event}:h2h:home:{i}", league="basketball_nba", event_id=event,
                     market="h2h", side="home", point=None, price=Decimal(str(price)), label=f"Team {i}",
                     game=f"Away {event} @ Home {event}", start=start, ai_prob=ai, market_prob=market,
                     push_prob=push, book=book, group=group)


def build(slate, stake, target_return, **kwargs):
    return build_for_target(slate, stake, target_return, now=NOW, **kwargs)


def labels(ticket):
    return [item["label"] for item in ticket["legs"]]


def combined(ticket) -> Decimal:
    return math.prod((Decimal(item["price"]) for item in ticket["legs"]), start=Decimal(1))


# $10 to get $35 back: odds of 3.50, so the window is 3.15 to 4.025
SLATE = [
    leg(1, 3.6, market=0.27, ai=0.29),  # p_c 0.28: 0.28 x 3.6 = 1.008, the only single in the window
    leg(2, 1.9, market=0.50, ai=0.54),  # p_c 0.52: 0.988
    leg(3, 1.9, market=0.50, ai=0.50),  # p_c 0.50: 0.95
    leg(4, 1.9, market=0.48, ai=0.51),  # p_c 0.495: 0.9405
    leg(5, 1.5, market=0.64, ai=0.68),  # p_c 0.66: 0.99
    leg(6, 1.5, market=0.64, ai=0.64),  # 0.96
    leg(7, 1.5, market=0.64, ai=0.64),  # 0.96
]


# --- the honest ceiling --------------------------------------------------------------------------

@pytest.mark.parametrize("multiple, legs, chance, one_in", [
    (2, 1, 0.4775, 2),  # 0.955 / 2: the blueprint's 47.8%
    (2, 4, 0.41590, 2),  # 0.955^4 / 2 = 0.8318 / 2: 41.6%
    (5, 2, 0.18241, 5),  # 0.912 / 5: 18.2%
    (10, 1, 0.0955, 10),  # 9.6%, 1 in 10
    (10, 4, 0.08318, 12),  # 8.3%, 1 in 12
    (100, 1, 0.00955, 105),  # 0.95%, 1 in 105
    (100, 6, 0.0075861, 132),  # 0.955^6 = 0.7586: 0.76%, 1 in 132
])
def test_no_edge_ceiling_matches_the_blueprint_table(multiple, legs, chance, one_in) -> None:
    assert CUT == 0.045
    assert no_edge_chance(multiple, legs) == pytest.approx(chance, rel=1e-3)
    assert round(1 / no_edge_chance(multiple, legs)) == one_in


def test_the_ceiling_comes_first_even_with_nothing_on_offer() -> None:
    out = build([], 20, 200)  # 10x the stake
    assert (out["stake"], out["target_return"], out["target_profit"]) == ("20.00", "200.00", "180.00")
    assert out["multiple"] == 10.0
    assert out["target_odds"] == {"decimal": 10.0, "american": "+900", "low": 9.0, "high": 11.5}
    assert (out["ceiling_chance"], out["ceiling_one_in"]) == (0.1, 10)  # Markov: 1 in M at fair odds
    # 0.955 / 10 = 1 in 10.5; 0.912 / 10 = 1 in 11.0; 0.871 / 10 = 1 in 11.5
    assert [row["one_in"] for row in out["no_edge"]] == [10, 11, 11]
    assert [row["legs"] for row in out["no_edge"]] == [1, 2, 3]
    assert out["tickets"] == [] and out["edge"] is False and out["expected_cost"] is None
    assert out["verdict"] == "Nothing on offer fits a 10.00 target right now. With no edge nothing beats 1 in 10."
    first = out["notes"][0]
    assert first.startswith("To turn $20.00 into $200.00 you need odds of 10.00.")
    assert "nothing beats 10% (1 in 10)" in first and "(1 in 11) with three legs" in first
    assert out["notes"][-1] == "No combination of the picks on offer lands between 9.00 and 11.50 odds."
    assert out["searched"] == {"candidates": 0, "usable": 0, "pool": 0, "combinations": 0}


def test_a_target_at_or_below_the_stake_needs_no_bet() -> None:
    for wanted in (10, 8):
        out = build([leg(1, 1.5, market=0.64)], 10, wanted)
        assert out["tickets"] == [] and out["verdict"].startswith("No bet needed:")
        assert out["ceiling_chance"] == 1.0 and out["target_odds"]["american"] is None


# --- the search and the scoring ------------------------------------------------------------------

def test_returns_the_best_single_double_and_treble_ranked_by_cautious_ev() -> None:
    out = build(SLATE, 10, 35)
    tickets = out["tickets"]
    assert [t["slot"] for t in tickets] == ["single", "double", "treble_or_more"]
    assert [t["rank"] for t in tickets] == [1, 2, 3]
    assert [t["name"] for t in tickets] == ["Single", "Double", "Treble"]
    assert labels(tickets[0]) == ["Team 1"]
    # Doubles at 1.9 x 1.9 = 3.61: Teams 2+3 = 0.988 x 0.95 = 0.9386 beat 2+4 (0.9292) and 3+4 (0.8935)
    assert sorted(labels(tickets[1])) == ["Team 2", "Team 3"]
    assert sorted(labels(tickets[2])) == ["Team 5", "Team 6", "Team 7"]  # 1.5^3 = 3.375
    # Cautious EV: +0.008; 0.9386 - 1; 0.99 x 0.96 x 0.96 - 1 = -0.087616
    assert [t["cautious"]["ev"] for t in tickets] == [0.008, -0.0614, -0.0876]
    assert [t["cautious"]["expected"] for t in tickets] == ["0.08", "-0.61", "-0.88"]
    assert [t["cautious"]["chance"] for t in tickets] == [0.28, 0.26, 0.2703]  # 0.66 x 0.64 x 0.64 = 0.2703
    assert [t["edge"] for t in tickets] == [True, False, False]
    # Each is price_ticket's own ticket at the stake given
    assert [t["type"] for t in tickets] == ["single", "multi", "multi"]
    assert all(t["valid"] and t["total_stake"] == "10.00" for t in tickets)
    assert [t["max_payout"] for t in tickets] == ["36.00", "36.10", "33.75"]
    assert [t["target_gap"] for t in tickets] == ["1.00", "1.10", "-1.25"]
    assert all(Decimal("3.15") <= combined(t) <= Decimal("4.025") for t in tickets)
    # The tier follows the AI's chance and always carries the loss chance: 0.29 is Risky, losing 71%
    assert tickets[0]["tier"] == {"key": "risky", "title": "Risky", "chance": 0.29, "loses": 0.71}
    assert out["edge"] is True and out["expected_cost"] is None
    assert out["verdict"] == ("Edge on paper: at the cautious chance the top ticket's expected result is +$0.08. "
                              "The AI's edge isn't proven, and the ticket still loses about 72% of the time.")
    assert out["searched"] == {"candidates": 7, "usable": 7, "pool": 7, "combinations": 5}


def test_cautious_chance_pulls_the_ai_halfway_to_the_market() -> None:
    assert cautious_probability(leg(1, 2.0, market=0.5, ai=0.6)) == pytest.approx(0.55)
    assert cautious_probability(leg(1, 2.0, market=0.5)) == pytest.approx(0.5)
    # No market chance: 0.955 / 2.0 = 0.4775 stands in, so 0.4775 + 0.5 x (0.55 - 0.4775) = 0.51375
    assert cautious_probability(leg(1, 2.0, ai=0.55)) == pytest.approx(0.51375)
    assert cautious_probability(leg(1, 2.0)) is None


def test_scoring_uses_the_cautious_chance_not_the_raw_ai() -> None:
    bold = leg(1, 1.8, market=0.50, ai=0.70)  # raw AI 0.70 x 1.8 = 1.26, cautious 0.60 x 1.8 = 1.08
    steady = leg(2, 1.85, market=0.58, ai=0.62)  # raw AI 1.147, cautious 0.60 x 1.85 = 1.11
    out = build([bold, steady], 10, 18)  # odds 1.80, window 1.62 to 2.07
    best = out["tickets"][0]
    assert labels(best) == ["Team 2"]
    assert best["cautious"]["ev"] == 0.11 and best["cautious"]["expected"] == "1.10"


def test_at_equal_cautious_ev_the_more_likely_ticket_wins() -> None:
    # 0.475 x 2.0 = 0.95 and 0.38 x 2.5 = 0.95: same expected return, but 47.5% beats 38%
    out = build([leg(2, 2.5, market=0.38), leg(1, 2.0, market=0.475)], 10, 22)  # window 1.98 to 2.53
    assert len(out["tickets"]) == 1 and labels(out["tickets"][0]) == ["Team 1"]


def test_no_edge_verdict_states_the_expected_cost() -> None:
    slate = [leg(1, 2.0, market=0.4775), leg(2, 2.1, market=0.45)]  # 0.955 and 0.945 per unit
    out = build(slate, 10, 20)
    assert [labels(t) for t in out["tickets"]] == [["Team 1"]]  # 2.0 x 2.1 = 4.2 is far past 2.30
    assert out["tickets"][0]["cautious"]["expected"] == "-0.45"  # 10 x (0.955 - 1)
    assert out["edge"] is False and out["expected_cost"] == "0.45"
    assert out["verdict"] == "No edge found: entertainment with an expected cost of $0.45."


def test_a_push_counts_as_stake_back_in_the_cautious_return() -> None:
    # The market's 0.50 is given no push: with a 6% push it wins 0.50 x 0.94 = 0.47 outright, which is
    # exactly the AI's chance, so p_c = 0.47 and the return is 0.47 x 1.95 + 0.06 (stake back) = 0.9765
    out = build([leg(1, 1.95, market=0.50, ai=0.47, push=0.06)], 10, Decimal("19.50"))
    ticket = out["tickets"][0]
    assert ticket["cautious"]["chance"] == 0.47 and ticket["cautious"]["ev"] == -0.0235
    assert ticket["cautious"]["expected"] == "-0.24" and out["expected_cost"] == "0.24"


def test_the_market_chance_on_a_line_that_can_push_counts_the_push_once() -> None:
    # The AI and the market agree (0.47 outright is 0.47 / 0.94 = 0.50 given no push), so there is no edge,
    # and the cost is what the ticket's own pricing says. Read as an outright chance, the market's 0.50 used
    # to count the push twice: "Edge on paper: +$0.06"
    out = build([leg(1, 1.95, market=0.50, ai=0.47, push=0.06)], 10, Decimal("19.50"))
    ticket = out["tickets"][0]
    assert out["edge"] is False and ticket["edge"] is False
    assert out["verdict"] == "No edge found: entertainment with an expected cost of $0.24."
    assert ticket["expected"]["ai"] == ticket["expected"]["market"] == "-0.24" == f"-{out['expected_cost']}"
    # The Safe note's market loss chance counts the push once too: 1 - (0.55 x 0.94 + 0.06) = 42.3%, as the
    # ticket's own pricing says (its market "any return" is 57.7%); read outright it was 1 - 0.55 - 0.06 = 39%
    out = build([leg(1, 1.6, market=0.55, ai=0.62, push=0.06)], 10, 16, tier="safe")
    assert out["notes"][-1] == ("Safe is a tier name, not a promise: by the AI's chance these tickets still lose "
                                "up to 40% of the time; by the market's chance the Single loses about 42%.")
    assert out["tickets"][0]["chance"]["any_return"]["market"] == pytest.approx(1 - 0.423, abs=1e-3)


def test_a_push_is_stake_back_not_a_loss_in_the_loss_chances() -> None:
    # The market's 0.50 given no push is 0.46 outright with an 8% push, so p_c = 0.46 + 0.5 x (0.60 - 0.46)
    # = 0.53; it pushes 8% of the time, so it loses 1 - 0.53 - 0.08 = 39%, not the 47% a push counted as a
    # loss would give
    out = build([leg(1, 2.0, market=0.50, ai=0.60, push=0.08)], 10, 20)
    assert out["verdict"].endswith("the ticket still loses about 39% of the time.")
    # By the AI's chance it loses 1 - 0.60 - 0.08 = 32%, not 40%
    assert out["tickets"][0]["tier"] == {"key": "safe", "title": "Safe", "chance": 0.6, "loses": 0.32}
    # A double loses only when some leg loses. The pushy leg's p_c is 0.45 + 0.5 x (0.50 - 0.45) = 0.475, so
    # 1 - (0.475 + 0.10) x 0.55 = 68% at the cautious chances; 1 - (0.50 + 0.10) x 0.60 = 64% at the AI's
    pushy, plain = leg(1, 1.9, market=0.50, ai=0.50, push=0.10), leg(2, 1.9, market=0.50, ai=0.60)
    out = build([pushy, plain], 10, Decimal("36.10"))  # (0.9025 + 0.10) x 1.045 = 1.048: an edge on paper
    assert out["verdict"].endswith("the ticket still loses about 68% of the time.")
    assert out["tickets"][0]["tier"]["loses"] == 0.64


def test_window_edges_are_inclusive() -> None:
    # Odds 2.00: the window is exactly 1.80 to 2.30
    slate = [leg(1, 1.79, market=0.53), leg(2, 1.8, market=0.53), leg(3, 2.3, market=0.415),
             leg(4, 2.31, market=0.41)]
    out = build(slate, 10, 20)
    assert out["searched"]["combinations"] == 2
    found = {label for t in out["tickets"] for label in labels(t)}
    assert found <= {"Team 2", "Team 3"}


def test_never_two_legs_from_one_game() -> None:
    slate = [
        leg(1, 1.9, market=0.5, ai=0.6, event="g1"),  # the two best legs share a game ...
        leg(2, 1.9, market=0.5, ai=0.6, event="g1", key="basketball_nba:g1:totals:over:224.5"),
        leg(3, 1.9, market=0.5, ai=0.5, event="g2"),
        leg(4, 1.9, market=0.5, ai=0.5, event="g3", group="g1"),  # ... and so does this one, by its group
    ]
    out = build(slate, 10, Decimal("36.10"))  # 1.9 x 1.9 = 3.61
    # Allowed doubles: 1+3, 2+3 and 3+4; never 1+2, 1+4 or 2+4
    assert out["searched"]["combinations"] == 3
    double = out["tickets"][0]
    assert "Team 3" in labels(double) and double["valid"]
    for ticket in out["tickets"]:
        groups = [item["group"] for item in ticket["legs"]]
        assert len(set(groups)) == len(groups)


def test_filters_started_out_of_range_and_unknown_legs() -> None:
    slate = [
        leg(1, 2.0, market=0.48),
        leg(2, 2.0, market=0.48, start=EARLIER),  # already started
        leg(3, 2.0, market=0.20),  # a market chance below 25%
        leg(4, 2.0, market=0.95),  # above 90%
        leg(5, 2.0),  # no chance at all
        leg(6, 2.0, market=0.48, start="tonight"),  # a start time that can't be checked
        leg(7, 2.0, market=0.25),  # the edges of the range are kept
        leg(8, 2.0, market=0.90),
    ]
    for now in (NOW, NOW.replace(tzinfo=None)):  # a naive time is read as UTC
        out = build_for_target(slate, 10, 20, now=now)
        assert out["searched"]["usable"] == 3
        used = {label for t in out["tickets"] for label in labels(t)}
        assert used and used <= {"Team 1", "Team 7", "Team 8"}
        assert ("Left out 5 picks: 2 already started, 2 with a market chance outside 25%-90%, "
                "1 with no chance yet.") in out["notes"]


def test_a_leg_with_only_an_ai_chance_uses_its_price_as_the_market() -> None:
    out = build([leg(1, 2.0, ai=0.55)], 10, 20)
    ticket = out["tickets"][0]
    assert ticket["cautious"]["chance"] == pytest.approx(0.51375, abs=1e-4)
    assert ticket["edge"] is True and out["verdict"].startswith("Edge on paper")
    assert "No margin-free market chance for Team 1: its price less a 4.5% cut stands in for it." in out["notes"]


def test_the_same_pick_at_two_books_keeps_the_best_price() -> None:
    key = "basketball_nba:e1:h2h:home"
    slate = [leg(1, 1.9, market=0.48, key=key, book="A"), leg(1, 2.0, market=0.48, key=key, book="B")]
    out = build(slate, 10, 20)
    assert out["searched"]["usable"] == 1
    assert out["tickets"][0]["legs"][0]["price"] == "2.0" and out["tickets"][0]["legs"][0]["book"] == "B"


def test_legs_that_cant_fit_the_window_stay_out_of_the_pool() -> None:
    # Odds 2.00: a leg at 3.0 is past the window however good it looks
    out = build([leg(1, 3.0, market=0.4, ai=0.6), leg(2, 2.0, market=0.48)], 10, 20)
    assert out["searched"]["pool"] == 1 and [labels(t) for t in out["tickets"]] == [["Team 2"]]
    # Odds 9.00 in at most two legs: 1.5 x 3.1 = 4.65 can't reach 8.10, so the 1.5 leg is left out
    slate = [leg(1, 3.0, market=0.32), leg(2, 3.1, market=0.31), leg(3, 1.5, market=0.64)]
    out = build(slate, 10, 90, max_legs=2)
    assert out["searched"]["pool"] == 2
    assert [sorted(labels(t)) for t in out["tickets"]] == [["Team 1", "Team 2"]]  # 3.0 x 3.1 = 9.3


def test_pool_keeps_the_forty_best_and_the_search_is_quick() -> None:
    good = [leg(i, 1.9, market=0.5, ai=0.5) for i in range(40)]  # 0.95 per unit
    weak = [leg(i, 1.9, market=0.5, ai=0.45) for i in range(40, 60)]  # p_c 0.475: 0.9025
    started = time.perf_counter()
    out = build(weak + good, 10, Decimal("130.32"), max_legs=4)  # 1.9^4 = 13.0321
    assert time.perf_counter() - started < 3.0
    assert POOL_SIZE == 40
    assert out["searched"]["usable"] == 60 and out["searched"]["pool"] == 40
    # Only four-leg tickets reach 11.73 to 14.99 (1.9^3 = 6.86), and every one of the 40 is tried
    assert out["searched"]["combinations"] == math.comb(40, 4)
    assert [t["slot"] for t in out["tickets"]] == ["treble_or_more"]
    ticket = out["tickets"][0]
    assert ticket["name"] == "Four-leg multi" and all(int(label.split()[1]) < 40 for label in labels(ticket))
    assert out["expected_cost"] == "1.85"  # 10 x (1 - 0.95^4) = 1.8549


def test_max_legs_caps_the_search_at_four() -> None:
    slate = [leg(i, 1.9, market=0.5) for i in range(6)]
    out = build(slate, 10, Decimal("130.32"), max_legs=6)
    assert out["max_legs"] == 4 and len(out["tickets"][0]["legs"]) == 4
    assert "Searched up to 4 legs: each extra leg adds the bookmaker's cut again." in out["notes"]
    assert build(slate, 10, Decimal("130.32"), max_legs=3)["tickets"] == []  # 1.9^3 = 6.86 is too short


def test_one_game_with_many_legs_cant_crowd_the_others_out_of_the_pool() -> None:
    # 45 strong legs from one game (p_c 0.53 x 1.9 = 1.007) would fill all 40 places on their own,
    # leaving only same-game pairs, which a ticket can't use. Each game keeps its 3 best instead.
    big = [leg(100 + i, 1.9, market=0.50, ai=0.56, event="big") for i in range(45)]
    others = [leg(i, 1.9, market=0.50, ai=0.54) for i in range(10)]  # p_c 0.52 x 1.9 = 0.988
    out = build(big + others, 10, Decimal("36.10"), max_legs=3)  # 1.9 x 1.9 = 3.61; window 3.25 to 4.15
    assert POOL_SIZE == 40 and out["searched"]["usable"] == 55 and out["searched"]["pool"] == 13  # 3 + 10
    # Doubles only (1.9^3 = 6.86 is too long): 3 big-game legs x 10 others + C(10, 2) = 30 + 45
    assert out["searched"]["combinations"] == 75
    double = out["tickets"][0]
    assert double["slot"] == "double" and double["valid"]
    assert len({item["group"] for item in double["legs"]}) == 2
    assert sum(item["group"] == "big" for item in double["legs"]) == 1  # the big game's best leg, once
    assert double["cautious"]["ev"] == pytest.approx(1.007 * 0.988 - 1, abs=1e-4)


def test_a_cut_pool_never_says_the_whole_slate_has_no_combination() -> None:
    big = [leg(100 + i, 1.9, market=0.50, ai=0.56, event="big") for i in range(45)]
    alone = build(big, 10, Decimal("36.10"), max_legs=3)  # every pair is from one game
    assert alone["tickets"] == [] and alone["searched"]["pool"] == 3
    assert alone["notes"][-1] == ("No combination of the 3 best picks searched (up to 3 a game) lands between "
                                  "3.25 and 4.15 odds.")
    others = [leg(i, 1.9, market=0.50, ai=0.54) for i in range(10)]
    safe = build(big + others, 10, Decimal("36.10"), max_legs=3, tier="safe")  # 0.56 x 0.54 = 30%: Risky
    assert safe["tickets"] == []
    assert safe["notes"][-1] == ("75 combinations of the 13 best picks searched (up to 3 a game) land near the "
                                 "target, but none has a Safe chance (60%+).")


def test_short_odds_target_offers_singles_only() -> None:
    slate = [leg(1, 1.08, market=0.88), leg(2, 1.07, market=0.89), leg(3, 1.07, market=0.89, ai=0.90)]
    out = build(slate, 100, 110)  # odds 1.10; a double at 1.07 x 1.08 = 1.156 would fit the window
    assert out["short_odds"] is True and out["max_legs"] == 1
    assert [labels(t) for t in out["tickets"]] == [["Team 3"]]  # 0.895 x 1.07 = 0.958 is the best single
    note = next(n for n in out["notes"] if n.startswith("Odds under 1.15"))
    # With no edge it loses 1 - 0.955 / 1.1 = 13% (1 in 8); a loss costs 1 / 0.10 = 10 wins
    assert "loses about 1 in 8" in note and "about 10 wins" in note


def test_lottery_like_target_still_shows_the_ceiling_and_searches() -> None:
    slate = [leg(i, 3.9, market=0.25) for i in range(3)]
    out = build(slate, 10, 600)  # 60x; 3.9^3 = 59.3 sits in 54 to 69
    assert out["lottery"] is True and (out["ceiling_one_in"], out["ceiling_chance"]) == (60, 0.0167)
    assert any(n.startswith("Lottery-like: 60 times the stake is at best a 1 in 60 shot") for n in out["notes"])
    ticket = out["tickets"][0]
    assert len(ticket["legs"]) == 3 and ticket["total_stake"] == "10.00"
    # 0.25 x 3.9 = 0.975 per leg: 0.975^3 = 0.92686, so 10 x -0.07314 = -0.73
    assert out["verdict"] == "No edge found: entertainment with an expected cost of $0.73."
    # 0.25^3 = 1.6%: below even the Crazy tier, and it says how often it loses
    assert ticket["tier"] == {"key": None, "title": None, "chance": 0.0156, "loses": 0.9844}


def test_never_raises_the_stake() -> None:
    out = build(SLATE, Decimal("7.50"), Decimal("26.25"))  # the same 3.50 odds on a smaller stake
    assert out["stake"] == "7.50" and out["tickets"]
    for ticket in out["tickets"]:
        assert ticket["unit_stake"] == ticket["total_stake"] == "7.50"
    assert out["tickets"][0]["max_payout"] == "27.00"  # 7.50 x 3.6
    # A target out of reach gets no ticket, never a bigger stake
    far = build(SLATE, 10, 5000, max_legs=4)
    assert far["stake"] == "10.00" and all(t["total_stake"] == "10.00" for t in far["tickets"])


# --- tiers ---------------------------------------------------------------------------------------

TIER_SLATE = [  # odds 1.50: the window is 1.35 to 1.725
    leg(1, 1.5, market=0.64, ai=0.66),  # AI 66%: Safe; p_c 0.65 x 1.5 = 0.975
    leg(2, 1.55, market=0.62, ai=0.60),  # AI 60%: Safe
    leg(3, 1.7, market=0.55, ai=0.58),  # AI 58%: Risky
    leg(4, 1.25, market=0.78, ai=0.78),  # 1.25 x 1.25 = 1.5625 at 0.78^2 = 61%: Safe
    leg(5, 1.25, market=0.78, ai=0.78),
]


def test_tier_keeps_only_tickets_in_its_chance_range() -> None:
    safe = build(TIER_SLATE, 10, 15, tier="safe")
    assert [t["slot"] for t in safe["tickets"]] == ["single", "double"]
    assert labels(safe["tickets"][0]) == ["Team 1"]
    assert all(t["tier"]["key"] == "safe" and t["tier"]["loses"] <= 0.40 for t in safe["tickets"])
    # By the market's chance these lose 36% (Team 1) and 1 - 0.78^2 = 39% (the double): within 40%, so no more
    assert safe["notes"][-1] == ("Safe is a tier name, not a promise: by the AI's chance these tickets still lose "
                                 "up to 40% of the time.")
    risky = build(TIER_SLATE, 10, 15, tier="Risky")
    assert [labels(t) for t in risky["tickets"]] == [["Team 3"]] and risky["tier"] == "risky"


def test_a_tier_that_cant_meet_the_target_says_so() -> None:
    crazy = build(TIER_SLATE, 10, 15, tier="crazy")
    assert crazy["tickets"] == []
    assert any(n.startswith("Crazy tickets win less than 25% of the time") for n in crazy["notes"])
    assert "4 combinations land near the target, but none has a Crazy chance (3%-25%)." in crazy["notes"]
    safe = build(TIER_SLATE, 10, 100, tier="safe")
    assert any(n.startswith("A Safe ticket needs a 60%+ chance, but at 10.00 odds the honest ceiling is "
                            "10% (1 in 10)") and n.endswith("don't fit together.") for n in safe["notes"])


def test_a_ticket_in_the_tier_past_the_ceiling_says_only_the_ai_puts_it_there() -> None:
    # At 1.70 the honest ceiling is 1 / 1.7 = 59%, yet the AI's 62% puts this single in Safe
    out = build([leg(1, 1.7, market=0.56, ai=0.62)], 10, 17, tier="safe")
    assert [t["tier"]["key"] for t in out["tickets"]] == ["safe"]
    assert not any("don't fit together" in note for note in out["notes"])
    assert ("A Safe ticket needs a 60%+ chance, but at 1.70 odds the honest ceiling is 59% (1 in 2): only the "
            "AI's chance, which the market doesn't share, puts these tickets in the tier, and it isn't proven."
            ) in out["notes"]
    # The cap is the AI's view; the market's 1 - 0.56 = 44% reads worse, so it is quoted too
    assert out["notes"][-1] == ("Safe is a tier name, not a promise: by the AI's chance these tickets still lose "
                                "up to 40% of the time; by the market's chance the Single loses about 44%.")
    # ... and the verdict's 41% is the cautious chance's: 1 - 0.59
    assert out["verdict"].endswith("the ticket still loses about 41% of the time.")


# --- inputs and wording --------------------------------------------------------------------------

@pytest.mark.parametrize("stake, wanted, kwargs", [
    (0, 20, {}), (-5, 20, {}), (10, 0, {}), ("abc", 20, {}), ("NaN", 20, {}), (10, "Infinity", {}),
    (10, 20, {"max_legs": 0}), (10, 20, {"tier": "yolo"}),
    ("0.004", "0.24", {}), ("1e-400", 1, {}),  # under a cent
    ("10.005", 20, {}), (10, "20.001", {}),  # fractions of a cent
    (1, "1e26", {}), ("1e10", "2e10", {}),  # past $1,000,000,000
    (1, 20_000_000, {}), ("0.01", 1_000_000, {}),  # more than 10,000,000 times the stake
])
def test_bad_inputs_raise(stake, wanted, kwargs) -> None:
    with pytest.raises(ValueError):
        build(SLATE, stake, wanted, **kwargs)


def test_money_at_the_limits_is_accepted() -> None:
    assert (build([], "10.000", "20.0")["stake"], build([], "0.01", "0.02")["stake"]) == ("10.00", "0.01")
    far = build([], 1, 10_000_000)  # exactly 10,000,000 times the stake
    assert far["ceiling_one_in"] == 10_000_000 and far["target_odds"]["american"] == "+999999900"


def test_tiny_chances_never_read_as_zero() -> None:
    # 1 / 10,000 = 0.01%; 0.955 / 10,000 = 0.00955% and 0.871 / 10,000 = 0.0087%, to two significant figures
    first = build([], 1, 10_000)["notes"][0]
    assert "nothing beats 0.010% (1 in 10,000)" in first
    assert "about 0.0095% (1 in 10,471) with one leg" in first and "0.0087% (1 in 11,481) with three legs" in first
    assert "0.0% (" not in first


def test_messages_never_use_banned_words() -> None:
    runs = [build(SLATE, 10, 35), build([], 20, 200), build([leg(1, 2.0, market=0.4775)], 10, 20),
            build([leg(1, 1.07, market=0.89)], 100, 110), build([leg(i, 3.9, market=0.25) for i in range(3)], 10, 600),
            build(TIER_SLATE, 10, 15, tier="safe"), build(TIER_SLATE, 10, 15, tier="crazy"),
            build(TIER_SLATE, 10, 100, tier="safe"), build([leg(1, 2.0, ai=0.55)], 10, 20), build([], 10, 10),
            build([leg(1, 1.7, market=0.56, ai=0.62)], 10, 17, tier="safe"), build([], 1, 10_000),
            build([leg(100 + i, 1.9, market=0.5, ai=0.56, event="big") for i in range(45)], 10, Decimal("36.10"))]
    for out in runs:
        for text in [out["verdict"], *out["notes"]]:
            assert not any(word in text.lower() for word in BANNED), text


def test_each_slot_keeps_its_best_ticket_by_cautious_ev_not_the_likeliest() -> None:
    # $10 to $30: two doubles land in the window. Team 1 + Team 2 (cautious EV +9.0%) beats Team 3 + Team 4 (+7.5%),
    # though the second is likelier: the Builder's search must not trade cautious EV for chance
    slate = [leg(1, 1.8, market=0.58, ai=0.58), leg(2, 1.8, market=0.58, ai=0.58),
             leg(3, 1.7, market=0.61, ai=0.61), leg(4, 1.7, market=0.61, ai=0.61)]
    result = build(slate, 10, 30, max_legs=2)
    double = next(t for t in result["tickets"] if t["slot"] == "double")
    assert labels(double) == ["Team 1", "Team 2"] and double["cautious"]["ev"] == pytest.approx(0.0899, abs=1e-4)
    assert "+$0.90" in result["verdict"]


@pytest.mark.parametrize("seed", range(12))
def test_each_slot_matches_brute_force_on_a_small_pool(seed) -> None:
    """Every slot's ticket has the best cautious expected return of all the tickets of its sizes in the window."""
    import itertools
    import random

    from telegram_betting_bot_simple.engine import target as module
    rng = random.Random(seed)
    slate = []
    for i in range(10):
        market = rng.uniform(0.3, 0.7)
        ai = min(0.9, max(0.25, market + rng.uniform(-0.04, 0.06)))
        slate.append(leg(i, round(rng.uniform(0.93, 0.98) / market, 2), market=round(market, 4), ai=round(ai, 4)))
    stake, wanted = 10, rng.choice((25, 40, 60, 90))
    result = build(slate, stake, wanted, max_legs=4)
    low, high = wanted / stake * float(module.WINDOW_LOW), wanted / stake * float(module.WINDOW_HIGH)
    best = {}
    for n in range(1, 5):
        for combo in itertools.combinations(slate, n):
            odds = math.prod(float(x.price) for x in combo)
            if not low - 1e-9 <= odds <= high + 1e-9:
                continue
            ev = math.prod(cautious_probability(x) * float(x.price) for x in combo) - 1
            slot = module._slot(n)
            best[slot] = max(best.get(slot, -math.inf), ev)
    got = {t["slot"]: t["cautious"]["ev"] for t in result["tickets"]}
    assert set(got) == set(best)
    for slot, ev in best.items():
        assert got[slot] == pytest.approx(ev, abs=1e-4), slot
