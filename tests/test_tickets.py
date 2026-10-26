"""Tests for the one ticket pricing function: singles, multis, round robins, system bets and grading.

Every expected number below is worked by hand in the comment next to it.
"""

from __future__ import annotations

import datetime
import math
import time
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.engine.tickets import (
    EXTRA_LEG_WARNING,
    MAX_LEGS,
    SYSTEMS,
    TICKET_TYPES,
    TicketLeg,
    chance_label,
    combinations,
    grade,
    one_in,
    price_ticket,
    settle_factor,
)

NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.timezone.utc)
LATER = "2026-10-25T23:00:00Z"


def leg(i, price, ai=None, market=None, push=0.0, event=None, group=None, start=LATER) -> TicketLeg:
    """Leg i of its own game unless ``event`` puts it in another leg's game. The price goes in as
    given (string, float or Decimal), so TicketLeg's own conversion and checks are what's tested."""
    event = event or f"e{i}"
    return TicketLeg(key=f"basketball_nba:{event}:h2h:home:{i}", league="basketball_nba", event_id=event,
                     market="h2h", side="home", point=None, price=price, label=f"Team {i}",
                     game=f"Away {event} @ Home {event}", start=start, ai_prob=ai, market_prob=market,
                     push_prob=push, group=group)


def outcome(ticket, result):
    return next(o for o in ticket["outcomes"] if o["result"] == result)


# --- shapes ------------------------------------------------------------------------------------

def test_ticket_types_and_simple_combinations() -> None:
    assert TICKET_TYPES == ("single", "multi", "round_robin", "system")
    assert combinations("single", 3) == [(0,), (1,), (2,)]
    assert combinations("multi", 3) == [(0, 1, 2)]
    with pytest.raises(ValueError):
        combinations("parlay", 3)
    with pytest.raises(ValueError):
        combinations("multi", 0)


def test_system_bet_counts() -> None:
    counts = {"trixie": 4, "patent": 7, "yankee": 11, "lucky15": 15, "canadian": 26, "super_yankee": 26,
              "lucky31": 31, "heinz": 57, "lucky63": 63, "super_heinz": 120, "goliath": 247}
    assert set(SYSTEMS) == set(counts)
    for name, bets in counts.items():
        system = SYSTEMS[name]
        assert system.bets == bets == len(combinations("system", system.legs, system=name)), name
        n = system.legs
        # Full cover: every double and up, 2^n - n - 1 bets; Patent and the Luckys add the n singles
        assert bets == (2 ** n - 1 if 1 in system.sizes else 2 ** n - n - 1), name
    assert combinations("system", 3, system="trixie") == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    # Names are forgiving about spaces, case and underscores
    assert len(combinations("system", 4, system="Lucky 15")) == 15
    assert len(combinations("system", 5, system="super yankee")) == 26
    with pytest.raises(ValueError):
        combinations("system", 3, system="yankee")  # a Yankee needs 4 legs
    with pytest.raises(ValueError):
        combinations("system", 3, system="flag")


def test_round_robin_combinations() -> None:
    # By 2s over 4 legs: C(4, 2) = 6 doubles; by 3s: C(4, 3) = 4 trebles; 2s are the default
    assert combinations("round_robin", 4, k=2) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(combinations("round_robin", 4, k=3)) == 4
    assert len(combinations("round_robin", 4)) == 6
    with pytest.raises(ValueError):
        combinations("round_robin", 4, k=4)  # that's just the multi
    with pytest.raises(ValueError):
        combinations("round_robin", 2, k=2)


# --- settling ----------------------------------------------------------------------------------

def test_settle_factor() -> None:
    price = Decimal("1.91")
    assert settle_factor("won", price) == Decimal("1.91")
    assert settle_factor("lost", price) == 0
    for result in ("push", "void", "cancelled", "canceled"):
        assert settle_factor(result, price) == 1
    with pytest.raises(ValueError):
        settle_factor("pending", price)


def test_dead_heat() -> None:
    # Two picks tied for the one place paid: half the stake wins at 5.0, so the factor is 5.0 * 1 / 2
    assert settle_factor("dead_heat", Decimal("5.0"), places_paid=1, tied=2) == Decimal("2.5")
    assert settle_factor("Dead heat", Decimal("4.5"), places_paid=2, tied=3) == Decimal("3")  # 4.5 * 2 / 3
    assert settle_factor("dead_heat", Decimal("4.5"), places_paid=3, tied=2) == Decimal("4.5")  # all paid in full
    with pytest.raises(ValueError):
        settle_factor("dead_heat", Decimal("4.5"), tied=0)
    # In a multi at 2.0 x dead-heated 5.0: 10 * 2.0 * 2.5 = 50
    ticket = price_ticket([leg(0, "2.0"), leg(1, "5.0")], "multi", Decimal(10), now=NOW)
    graded = grade(["won", {"result": "dead_heat", "places_paid": 1, "tied": 2}], ticket)
    assert graded == {"status": "won", "returns": "50.00", "profit": "40.00", "total_stake": "10.00",
                      "paid_bets": 1, "bets": 1, "results": ["won", "dead_heat"]}


# --- multis ------------------------------------------------------------------------------------

def test_three_leg_multi_at_1_91() -> None:
    ticket = price_ticket([leg(i, "1.91", ai=0.5, market=0.5) for i in range(3)], "multi", Decimal(10), now=NOW)
    assert ticket["valid"] and ticket["reasons"] == []
    assert ticket["type"] == "multi" and ticket["bets"] == 1
    assert ticket["unit_stake"] == ticket["total_stake"] == "10.00"
    # 1.91^3 = 6.967871, which is +596.79 American
    assert ticket["odds"] == {"decimal": 6.97, "american": "+597"}
    assert ticket["max_payout"] == "69.68" and ticket["max_profit"] == "59.68"  # 10 * 6.967871 = 69.67871
    chance = ticket["chance"]
    assert chance["headline"] == "every leg wins"
    assert chance["ai"] == chance["market"] == 0.125  # 0.5^3
    assert chance["ai_one_in"] == 8 and chance["label"] == "long shot"
    # The book keeps 1 - Q * D = 1 - 0.125 * 6.967871 = 12.9%, so 10 staked expects back 8.71
    assert ticket["book_cut"] == 0.129
    assert ticket["expected"] == {"ai": "-1.29", "market": "-1.29", "ai_roi": -0.129, "market_roi": -0.129}
    assert ticket["most_likely"] == {"result": "lose all", "probability": 0.875, "basis": "ai", "profit": "-10.00"}
    assert outcome(ticket, "profit")["min_return"] == outcome(ticket, "profit")["max_return"] == "69.68"
    assert EXTRA_LEG_WARNING in ticket["warnings"]
    # Whelan: legs at 50% are worth combining two at a time at most
    assert any("worth combining" in w for w in ticket["warnings"])
    first = ticket["legs"][0]
    assert first["price"] == "1.91" and first["odds"] == {"decimal": 1.91, "american": "-110"}
    assert first["fair_odds"] == 2.0 and first["edge"] == -0.045 and first["book_cut"] == 0.045
    assert first["chance_label"] == "coin flip" and first["started"] is False


def test_push_in_a_multi_drops_the_leg() -> None:
    legs = [leg(0, "1.91", ai=0.5, market=0.5), leg(1, "1.91", ai=0.45, market=0.5, push=0.10),
            leg(2, "1.91", ai=0.5, market=0.5)]
    ticket = price_ticket(legs, "multi", Decimal(10), now=NOW)
    graded = grade(["won", "push", "won"], ticket)
    assert graded["status"] == "won"
    assert graded["returns"] == "36.48" and graded["profit"] == "26.48"  # 10 * 1.91 * 1 * 1.91 = 36.481
    chance = ticket["chance"]
    assert chance["ai"] == 0.1125  # every leg wins: 0.5 * 0.45 * 0.5
    assert chance["any_return"]["ai"] == chance["profit"]["ai"] == 0.1375  # 0.5 * (0.45 + 0.10) * 0.5
    # The market's 0.5 is given no push, so leg 1 wins 0.5 * 0.9 = 0.45 outright: 0.5 * (0.45 + 0.10) * 0.5
    assert chance["market"] == 0.1125 and chance["any_return"]["market"] == 0.1375
    # A push still pays: the profit outcomes run from 36.48 (leg 1 pushes) to 69.68 (all win)
    assert outcome(ticket, "profit")["min_return"] == "36.48"
    assert outcome(ticket, "profit")["max_return"] == "69.68"
    # Expected, push-aware: 10 * 0.955 * (0.45 * 1.91 + 0.10) * 0.955 - 10 = -1.249 under both views here
    assert ticket["expected"]["ai"] == "-1.25" and ticket["expected"]["market"] == "-1.25"
    assert ticket["book_cut"] == 0.129  # pushes are left out of the cut: 1 - 0.955^3
    assert ticket["most_likely"]["result"] == "lose all" and ticket["most_likely"]["probability"] == 0.8625
    assert "A push on Team 1 drops it from the ticket and lowers the payout." in ticket["warnings"]
    assert ticket["legs"][1]["edge"] == -0.0405  # 0.45 * 1.91 + 0.10 - 1


def test_backing_both_sides_of_a_line_that_can_push_never_shows_a_profit() -> None:
    # Market 0.5 a side from a de-vigged two-way price, 5% push, both at 1.91: the push comes
    # out of both sides, so each wins 0.475 outright and expects 0.475 * 1.91 + 0.05 - 1 = -0.04275,
    # which is -(1 - push) x the 4.5% cut. 10 a side: -0.4275 and -0.855 together.
    one = price_ticket([leg(0, "1.91", market=0.5, push=0.05)], "single", Decimal(10), now=NOW)
    assert one["book_cut"] == 0.045
    assert one["expected"]["market"] == "-0.43"
    assert one["expected"]["market_roi"] == pytest.approx(-(1 - 0.05) * 0.045, abs=1e-4)
    assert one["chance"]["market"] == 0.475 and one["chance"]["market_one_in"] == 2
    assert outcome(one, "profit")["market"] == outcome(one, "lose all")["market"] == 0.475
    assert outcome(one, "stake back")["market"] == 0.05
    both = price_ticket([leg(0, "1.91", market=0.5, push=0.05), leg(1, "1.91", market=0.5, push=0.05, event="e0")],
                        "single", Decimal(10), now=NOW)
    assert both["expected"]["market"] == "-0.86"  # -0.855, half up
    assert both["book_cut"] == 0.045


def test_ten_singles_still_show_a_one_in() -> None:
    # Ten singles at 2.0 that can't push are listed in full (2^10 outcomes): a profit needs 6+ wins,
    # (210 + 120 + 45 + 10 + 1) / 1024 = 0.377
    ten = price_ticket([leg(i, "2.0", ai=0.5) for i in range(10)], "single", Decimal(1), now=NOW)
    assert ten["chance"]["headline"] == "profit" and ten["chance"]["ai"] == 0.377
    assert ten["chance"]["ai_one_in"] == 3 and ten["chance"]["label"] == "underdog"
    assert ten["chance"]["every_leg"]["ai"] == 0.001 and ten["most_likely"] is not None
    # Twelve are still quick
    began = time.perf_counter()
    twelve = price_ticket([leg(i, "1.91", ai=0.5, market=0.5) for i in range(12)], "single", Decimal(1), now=NOW)
    assert time.perf_counter() - began < 1.0 and twelve["chance"]["ai_one_in"] is not None
    # Ten that can push are too many to list (3^10), so the headline falls back to every pick winning
    pushy = price_ticket([leg(i, "2.0", ai=0.45, push=0.1) for i in range(10)], "single", Decimal(1), now=NOW)
    assert pushy["chance"]["headline"] == "every leg wins" and pushy["outcomes"] == []
    assert pushy["chance"]["ai_one_in"] == one_in(0.45 ** 10) == 2937
    assert pushy["chance"]["label"] == "lottery ticket"
    assert any("Too many picks" in w for w in pushy["warnings"])


def test_multi_outcomes_with_pushes_match_the_closed_form() -> None:
    # (win, push) per leg: (0.5, 0.1) at 2.0, (0.4, 0.2) at 2.5, (0.3, 0.1) at 3.0
    legs = [leg(0, "2.0", ai=0.5, push=0.1), leg(1, "2.5", ai=0.4, push=0.2), leg(2, "3.0", ai=0.3, push=0.1)]
    ticket = price_ticket(legs, "multi", Decimal(10), now=NOW)
    assert ticket["chance"]["every_leg"]["ai"] == 0.06  # 0.5 * 0.4 * 0.3
    assert ticket["chance"]["any_return"]["ai"] == 0.144  # no leg loses: 0.6 * 0.6 * 0.4
    assert outcome(ticket, "stake back")["ai"] == 0.002  # every leg pushes: 0.1 * 0.2 * 0.1
    assert ticket["chance"]["profit"]["ai"] == 0.142  # 0.144 - 0.002
    assert outcome(ticket, "lose all")["ai"] == 0.856
    assert outcome(ticket, "stake back")["min_return"] == "10.00"
    assert outcome(ticket, "profit")["min_return"] == "20.00"  # the 2.0 wins and the rest push
    assert outcome(ticket, "profit")["max_return"] == "150.00"  # 10 * 2 * 2.5 * 3
    # Expected factors 0.5*2+0.1 = 1.1, 0.4*2.5+0.2 = 1.2, 0.3*3+0.1 = 1.0: 10 * 1.32 - 10
    assert ticket["expected"]["ai"] == "3.20"
    assert ticket["expected"]["market"] is None and ticket["book_cut"] is None  # no market chances given
    assert ticket["chance"]["market"] is None


def test_book_cut_compounds_per_leg() -> None:
    # A 4.5% cut per leg (0.5 * 1.91 = 0.955): five legs keep 1 - 0.955^5 = 20.56%
    five = price_ticket([leg(i, "1.91", market=0.5) for i in range(5)], "multi", Decimal(10), now=NOW)
    assert five["book_cut"] == 0.2056
    assert five["expected"]["market"] == "-2.06"  # 10 * 0.794359 - 10
    single = price_ticket([leg(0, "1.91", market=0.5)], "single", Decimal(10), now=NOW)
    assert single["book_cut"] == 0.045
    # Blueprint: at a 5% cut (0.5 * 1.9 = 0.95) a Lucky 15 keeps 10.3% and a Goliath 18.8%
    # Lucky 15: 1 - ((1 + 0.95)^4 - 1) / 15; Goliath: 1 - ((1 + 0.95)^8 - 1 - 8 * 0.95) / 247
    lucky15 = price_ticket([leg(i, "1.9", market=0.5) for i in range(4)], "system", Decimal(1), system="lucky15", now=NOW)
    assert lucky15["book_cut"] == round(1 - (1.95 ** 4 - 1) / 15, 4) == 0.1027
    goliath = price_ticket([leg(i, "1.9", market=0.5) for i in range(8)], "system", Decimal(1), system="goliath", now=NOW)
    assert goliath["book_cut"] == round(1 - (1.95 ** 8 - 1 - 8 * 0.95) / 247, 4) == 0.1884


# --- round robins and systems ------------------------------------------------------------------

def test_round_robin_by_2_over_4_legs() -> None:
    legs = [leg(i, "1.9", ai=0.5, market=0.5) for i in range(4)]
    ticket = price_ticket(legs, "round_robin", Decimal(10), k=2, now=NOW)
    assert ticket["valid"] and ticket["type"] == "round_robin" and ticket["k"] == 2
    assert ticket["bets"] == 6 and ticket["unit_stake"] == "10.00" and ticket["total_stake"] == "60.00"
    assert ticket["odds"] is None
    assert ticket["max_payout"] == "216.60" and ticket["max_profit"] == "156.60"  # 6 doubles * 3.61 * 10
    chance = ticket["chance"]
    # Blueprint: by 2s, the chance of any profit rises from 6.25% (the 4-leg multi) to 31.3%.
    # Two wins pay one double (36.10 of 60), so a profit needs 3 or 4 wins: (4 + 1) / 16
    assert chance["headline"] == "profit" and chance["ai"] == chance["market"] == 0.3125
    assert chance["ai_one_in"] == 3 and chance["label"] == "underdog"
    assert chance["every_leg"]["ai"] == 0.0625
    assert chance["any_return"]["ai"] == 0.6875  # 2+ wins: 11 / 16
    assert ticket["most_likely"] == {"result": "some back", "probability": 0.375, "basis": "ai", "profit": "-23.90"}
    assert outcome(ticket, "profit")["min_return"] == "108.30"  # 3 wins: 3 doubles * 36.10
    assert ticket["book_cut"] == 0.0975  # 1 - 0.95^2
    assert ticket["expected"]["market"] == "-5.85"  # 60 * 0.9025 - 60
    multi = price_ticket(legs, "multi", Decimal(10), now=NOW)
    assert multi["chance"]["profit"]["ai"] == 0.0625
    # 2s are the default
    assert price_ticket(legs, "round_robin", Decimal(10), now=NOW)["bets"] == 6


def test_trixie_hand_worked() -> None:
    # 2.0, 3.0 and 4.0 at 50%, 30% and 20%, 1 a bet: doubles 6 + 8 + 12, treble 24
    legs = [leg(0, "2.0", ai=0.5), leg(1, "3.0", ai=0.3), leg(2, "4.0", ai=0.2)]
    ticket = price_ticket(legs, "system", Decimal(1), system="trixie", now=NOW)
    assert ticket["system"] == "trixie" and ticket["bets"] == 4 and ticket["total_stake"] == "4.00"
    assert ticket["max_payout"] == "50.00" and ticket["max_profit"] == "46.00"
    # Any two wins beat the 4 staked: 0.5*0.3*0.8 + 0.5*0.7*0.2 + 0.5*0.3*0.2 + 0.5*0.3*0.2 = 0.25
    assert ticket["chance"]["ai"] == 0.25 and ticket["chance"]["any_return"]["ai"] == 0.25
    assert ticket["most_likely"] == {"result": "lose all", "probability": 0.75, "basis": "ai", "profit": "-4.00"}
    assert outcome(ticket, "profit")["min_return"] == "6.00"
    # Expected factors 1.0, 0.9, 0.8: doubles 0.9 + 0.8 + 0.72, treble 0.72, so 3.14 back for 4
    assert ticket["expected"]["ai"] == "-0.86"
    assert ticket["weakest_leg"] == 2


def test_patent_adds_the_singles() -> None:
    legs = [leg(0, "2.0", ai=0.5), leg(1, "3.0", ai=0.3), leg(2, "4.0", ai=0.2)]
    ticket = price_ticket(legs, "system", Decimal(1), system="patent", now=NOW)
    assert ticket["bets"] == 7 and ticket["max_payout"] == "59.00"  # singles 9 + doubles 26 + treble 24
    assert ticket["expected"]["ai"] == "-1.16"  # 2.7 + 2.42 + 0.72 = 5.84 back for 7
    assert ticket["chance"]["any_return"]["ai"] == 0.72  # 1 - 0.5 * 0.7 * 0.8
    assert ticket["chance"]["profit"]["ai"] == 0.25  # one win returns 2, 3 or 4 of 7; two wins return 11+
    # Exactly one win: 0.28 + 0.12 + 0.07 = 0.47, the most likely result, with a spread of returns
    assert ticket["most_likely"] == {"result": "some back", "probability": 0.47, "basis": "ai", "profit": None}
    assert outcome(ticket, "some back")["min_return"] == "2.00"
    assert outcome(ticket, "some back")["max_return"] == "4.00"


def test_singles() -> None:
    legs = [leg(0, "2.0", ai=0.5), leg(1, "3.0", ai=0.4)]
    ticket = price_ticket(legs, "single", Decimal(10), now=NOW)
    assert ticket["bets"] == 2 and ticket["total_stake"] == "20.00" and ticket["odds"] is None
    assert ticket["max_payout"] == "50.00"
    assert ticket["chance"]["headline"] == "profit" and ticket["chance"]["ai"] == 0.4  # the 3.0 wins
    assert outcome(ticket, "stake back")["ai"] == 0.3  # only the 2.0 wins: 0.5 * 0.6
    assert outcome(ticket, "lose all")["ai"] == 0.3
    assert [o["result"] for o in ticket["outcomes"]] == ["lose all", "stake back", "profit"]
    assert ticket["most_likely"]["result"] == "profit"
    assert ticket["expected"]["ai"] == "2.00"  # 10 * (1.0 + 1.2) - 20
    assert EXTRA_LEG_WARNING not in ticket["warnings"]
    # One single (or a one-leg multi) wins when its leg does
    one = price_ticket([leg(0, "2.0", ai=0.5)], "multi", Decimal(10), now=NOW)
    assert one["type"] == "single" and one["odds"]["decimal"] == 2.0
    assert one["chance"]["headline"] == "every leg wins" and one["chance"]["ai"] == 0.5


def test_goliath_lists_every_outcome_quickly() -> None:
    legs = [leg(i, "1.90909", ai=0.5, market=0.5, push=0.05) for i in range(8)]
    began = time.perf_counter()
    ticket = price_ticket(legs, "system", Decimal(1), system="goliath", now=NOW)
    assert time.perf_counter() - began < 1.0
    assert ticket["bets"] == 247 and ticket["valid"]
    assert sum(o["ai"] for o in ticket["outcomes"]) == pytest.approx(1.0, abs=1e-3)
    assert ticket["chance"]["every_leg"]["ai"] == round(0.5 ** 8, 4)
    # A 12-leg multi stops at each first lost leg, so it is cheap too
    twelve = price_ticket([leg(i, "1.91", ai=0.5, push=0.05) for i in range(12)], "multi", Decimal(1), now=NOW)
    assert twelve["valid"] and twelve["chance"]["ai_one_in"] == 4096  # 2^12


# --- chances and labels ------------------------------------------------------------------------

def test_chance_labels_and_one_in() -> None:
    assert chance_label(0.60) == "favourite"
    assert chance_label(0.5999) == chance_label(0.40) == "coin flip"
    assert chance_label(0.3999) == chance_label(0.20) == "underdog"
    assert chance_label(0.1999) == chance_label(0.05) == "long shot"
    assert chance_label(0.0499) == "lottery ticket"
    assert chance_label(None) is None
    assert one_in(0.125) == 8 and one_in(0.9) == 1
    assert one_in(0.47 ** 10) == 1901  # the blueprint's 10-day rollover at 2.0
    assert one_in(0) is None and one_in(None) is None


def test_weakest_leg_falls_back_to_the_market() -> None:
    legs = [leg(0, "1.5", ai=0.6), leg(1, "2.2", ai=0.4), leg(2, "1.4", ai=0.7)]
    assert price_ticket(legs, "multi", Decimal(5), now=NOW)["weakest_leg"] == 1
    legs = [leg(0, "1.5", ai=0.6), leg(1, "2.9", market=0.3), leg(2, "1.9", ai=0.5)]
    ticket = price_ticket(legs, "multi", Decimal(5), now=NOW)
    assert ticket["weakest_leg"] == 1
    # Without an AI chance on every leg there is no AI view of the ticket; legs 0 and 2 have no
    # market chance either, so the warning mustn't promise the market's view
    assert ticket["chance"]["ai"] is None and ticket["expected"]["ai"] is None
    assert ticket["chance"]["market"] is None
    assert "No AI chance for Team 1: the AI's view of the ticket isn't shown." in ticket["warnings"]


def test_market_only_and_no_chances_at_all() -> None:
    legs = [leg(i, "1.91", market=0.5) for i in range(2)]
    ticket = price_ticket(legs, "multi", Decimal(10), now=NOW)
    assert ticket["most_likely"] == {"result": "lose all", "probability": 0.75, "basis": "market", "profit": "-10.00"}
    assert "No AI chance for Team 0: only the market's view is shown." in ticket["warnings"]
    blind = price_ticket([leg(i, "1.91") for i in range(2)], "multi", Decimal(10), now=NOW)
    assert blind["chance"]["ai"] is None and blind["chance"]["market"] is None
    assert blind["most_likely"] is None and blind["book_cut"] is None
    assert blind["max_payout"] == "36.48"
    # No chance at all: no view is shown, and the warning says so rather than promising the market's
    assert "No chance available for Team 0: the ticket's chances can't be shown." in blind["warnings"]
    assert not any("only the market's view" in w for w in blind["warnings"])
    # A market chance on one leg only: neither view of the ticket can be shown
    mixed = price_ticket([leg(0, "1.91", market=0.5), leg(1, "1.91")], "multi", Decimal(10), now=NOW)
    assert mixed["chance"]["market"] is None
    assert "No AI chance for Team 0: the AI's view of the ticket isn't shown." in mixed["warnings"]


def test_ai_far_from_the_market_is_flagged() -> None:
    ticket = price_ticket([leg(0, "2.0", ai=0.62, market=0.5)], "single", Decimal(10), now=NOW)
    assert "The AI is far from the market on Team 0: check the injury news." in ticket["warnings"]
    view = ticket["legs"][0]
    assert view["edge"] == 0.24 and view["fair_odds"] == 1.61  # 0.62 * 2 - 1, 1 / 0.62
    assert view["chance_label"] == "favourite"


def test_push_aware_fair_odds() -> None:
    # 45% win, 10% push at 2.0: 0.45 * 2 + 0.10 = 1.0 back per unit, so the fair price is 0.9 / 0.45 = 2.0
    view = price_ticket([leg(0, "2.0", ai=0.45, push=0.10)], "single", Decimal(10), now=NOW)["legs"][0]
    assert view["edge"] == 0.0 and view["fair_odds"] == 2.0


# --- validation --------------------------------------------------------------------------------

def test_same_game_legs_are_refused_in_combined_tickets() -> None:
    legs = [leg(0, "1.91", ai=0.5), leg(1, "1.91", ai=0.5, event="e0"), leg(2, "2.1", ai=0.45)]
    for ticket_type in ("multi", "round_robin"):
        ticket = price_ticket(legs, ticket_type, Decimal(10), now=NOW)
        assert not ticket["valid"]
        assert ticket["reasons"] == ["Team 0 and Team 1 are from the same game, so their results are linked; "
                                     "pick one (same-game tickets aren't supported)"]
    # A player prop carries its game as the group even with its own event id
    prop = leg(3, "1.87", ai=0.5, event="prop-77", group="e2")
    assert not price_ticket([legs[0], legs[2], prop], "multi", Decimal(10), now=NOW)["valid"]
    # Singles are separate bets, so they're allowed, with a warning that the chances treat them as unrelated
    singles = price_ticket(legs, "single", Decimal(10), now=NOW)
    assert singles["valid"]
    assert any("same game" in w for w in singles["warnings"])


def test_started_legs_are_refused() -> None:
    legs = [leg(0, "1.91", ai=0.5), leg(1, "1.91", ai=0.5, start="2026-10-25T11:00:00Z")]
    for ticket_type in ("multi", "single"):
        ticket = price_ticket(legs, ticket_type, Decimal(10), now=NOW)
        assert not ticket["valid"] and ticket["reasons"] == ["Team 1 has already started"]
        assert ticket["legs"][1]["started"] is True and ticket["legs"][0]["started"] is False
    # A later clock: both have started
    later = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc)
    assert len(price_ticket(legs, "multi", Decimal(10), now=later)["reasons"]) == 2


def test_leg_limits_and_shapes() -> None:
    def reasons(legs, ticket_type, **kwargs):
        ticket = price_ticket(legs, ticket_type, Decimal(1), now=NOW, **kwargs)
        assert not ticket["valid"] and ticket["bets"] == 0 and ticket["max_payout"] is None
        return ticket["reasons"]

    many = [leg(i, "1.5") for i in range(MAX_LEGS + 1)]
    assert reasons([], "multi") == ["Add at least one pick"]
    assert reasons(many, "multi") == [f"A ticket can have at most {MAX_LEGS} picks"]
    assert price_ticket(many[:MAX_LEGS], "multi", Decimal(1), now=NOW)["valid"]
    assert reasons(many[:9], "round_robin") == ["Round robins and system bets can have at most 8 picks"]
    assert reasons(many[:2], "round_robin") == ["A round robin needs at least 3 picks"]
    assert reasons(many[:4], "round_robin", k=4) == ["A round robin of 4 picks can be by 2s up to 3s"]
    assert reasons(many[:3], "system", system="yankee") == ["A Yankee needs exactly 4 picks"]
    duplicate = [leg(0, "1.5"), leg(0, "1.5")]
    assert "Team 0 is on the ticket more than once" in price_ticket(duplicate, "single", Decimal(1), now=NOW)["reasons"]
    # A zero stake still prices the ticket but can't be placed
    zero = price_ticket([leg(0, "1.5", ai=0.7)], "single", Decimal(0), now=NOW)
    assert zero["reasons"] == ["Enter a stake above zero"] and zero["chance"]["ai"] == 0.7
    assert zero["expected"]["ai"] == "0.00"
    with pytest.raises(ValueError):
        price_ticket([leg(0, "1.5")], "single", Decimal(-1), now=NOW)
    with pytest.raises(ValueError):
        price_ticket([leg(0, "1.5")], "parlay", Decimal(1), now=NOW)
    with pytest.raises(ValueError):
        price_ticket([leg(i, "1.5") for i in range(3)], "system", Decimal(1), system="flag", now=NOW)


def test_legs_check_their_numbers() -> None:
    with pytest.raises(ValueError):
        leg(0, "1.0")  # odds of 1.0 pay nothing
    with pytest.raises(ValueError):
        leg(0, "1.9", ai=1.2)
    with pytest.raises(ValueError):
        leg(0, "2.0", ai=0.8, push=0.5)  # a win and a push can't add up to more than certain
    assert leg(0, "2.0", ai=0.6, push=0.4).push_prob == 0.4  # exactly certain is fine
    # The market's chance is given no push, so 0.9 with a 20% push is 0.72 win, 0.2 push, 0.08 lose
    assert leg(0, "2.0", market=0.9, push=0.2).market_prob == 0.9
    # A missing push (None, as callers pass for a moneyline) is no chance of one, not a crash later
    no_push = leg(0, "2.0", ai=0.5, push=None)
    assert no_push.push_prob == 0.0
    assert price_ticket([no_push], "single", Decimal(10), now=NOW)["legs"][0]["edge"] == 0.0
    for price in (float("inf"), "Infinity", "NaN", "abc"):
        with pytest.raises(ValueError):
            leg(0, price)
    assert leg(0, "1.9").group == "e0"  # a game market's group is its game


def test_bad_stakes_raise_value_error() -> None:
    # Typed stakes that aren't finite numbers are bad input (a 400), never a decimal.InvalidOperation (a 500)
    for stake in ("NaN", "Infinity", "-Infinity", float("nan"), float("inf"), "abc", ""):
        with pytest.raises(ValueError):
            price_ticket([leg(0, "1.5", ai=0.7)], "single", stake, now=NOW)
    assert price_ticket([leg(0, "1.5", ai=0.7)], "single", " 10 ", now=NOW)["total_stake"] == "10.00"


def test_money_is_exact_decimal_rounded_once() -> None:
    # As a float, 2.005 is 2.00499999..., which would round down to 2.00
    single = price_ticket([leg(0, 2.005)], "single", "1", now=NOW)
    assert single["legs"][0]["price"] == "2.005" and single["max_payout"] == "2.01"
    # Three 1.91s for 0.10: 0.6967871 shows as 0.70
    assert price_ticket([leg(i, 1.91) for i in range(3)], "multi", Decimal("0.10"), now=NOW)["max_payout"] == "0.70"


# --- grading -----------------------------------------------------------------------------------

def _trixie():
    legs = [leg(0, "2.0", ai=0.5), leg(1, "3.0", ai=0.3), leg(2, "4.0", ai=0.2)]
    return price_ticket(legs, "system", Decimal(1), system="trixie", now=NOW)


def test_grade_system_bets() -> None:
    ticket = _trixie()
    # Only the 2.0 x 3.0 double comes in: 6 back for 4
    graded = grade(["won", "won", "lost"], ticket)
    assert (graded["status"], graded["returns"], graded["profit"], graded["paid_bets"]) == ("partial", "6.00", "2.00", 1)
    assert grade(["won", "lost", "lost"], ticket)["status"] == "lost"
    # The void leg counts as 1: doubles 2 + 8 + 4 and treble 8
    graded = grade(["won", "void", "won"], ticket)
    assert (graded["status"], graded["returns"], graded["profit"]) == ("won", "22.00", "18.00")
    graded = grade(["push", "void", {"result": "cancelled"}], ticket)
    assert (graded["status"], graded["returns"], graded["profit"]) == ("void", "4.00", "0.00")


def test_grade_singles_and_round_robins() -> None:
    singles = price_ticket([leg(0, "2.0"), leg(1, "3.0")], "single", Decimal(10), now=NOW)
    graded = grade(["won", "lost"], singles)
    assert (graded["status"], graded["returns"], graded["profit"]) == ("partial", "20.00", "0.00")
    robin = price_ticket([leg(i, "2.0") for i in range(3)], "round_robin", Decimal(5), k=2, now=NOW)
    # Doubles 0+1 (2 * 1), 0+2 (2 * 2), 1+2 (1 * 2): 8 units of 5
    graded = grade(["won", "push", "won"], robin)
    assert (graded["status"], graded["returns"], graded["profit"]) == ("won", "40.00", "25.00")


def test_grade_waits_for_open_legs_unless_the_ticket_is_already_lost() -> None:
    multi = price_ticket([leg(i, "1.91") for i in range(3)], "multi", Decimal(10), now=NOW)
    assert grade(["lost", None, "won"], multi)["status"] == "lost"
    assert grade(["won", "pending", "won"], multi)["status"] == "pending"
    robin = price_ticket([leg(i, "2.0") for i in range(3)], "round_robin", Decimal(5), now=NOW)
    assert grade(["won", None, "won"], robin) == {"status": "pending", "returns": None, "profit": None,
                                                 "total_stake": "15.00", "paid_bets": None, "bets": 3,
                                                 "results": ["won", None, "won"]}
    assert grade(["lost", "lost", None], robin)["status"] == "lost"  # every double holds a lost leg
    with pytest.raises(ValueError):
        grade(["won"], robin)
    with pytest.raises(ValueError):
        grade([], price_ticket([], "multi", Decimal(1), now=NOW))


def test_grading_uses_the_exact_price() -> None:
    # Odds of 1.90909 show as 1.91, but 100 on a win returns 190.909 -> 190.91, not 191.00
    ticket = price_ticket([leg(0, "1.90909")], "single", Decimal(100), now=NOW)
    assert ticket["legs"][0]["odds"]["decimal"] == 1.91
    assert grade(["won"], ticket)["returns"] == "190.91"
    assert math.isclose(float(ticket["max_payout"]), 190.91)
