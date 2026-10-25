"""The Safe / Risky / Crazy ticket builder."""

from __future__ import annotations

import pytest

from telegram_betting_bot_simple.engine import bets
from telegram_betting_bot_simple.engine.bets import Candidate, Ticket, build_all, build_tier, whelan_max_legs


def leg(game, p, price, fair=None, market="h2h", side="home", books=("Book A", "Book B")) -> Candidate:
    return Candidate(event_id=game, game=f"Away {game} @ Home {game}", market=market, side=side, point=None,
                     label=f"{game} {side}", probability=p, fair=p if fair is None else fair,
                     prices={book: price for book in books})


def test_whelan_leg_counts() -> None:
    assert whelan_max_legs(0.30) == 1  # never combine legs below a third
    assert whelan_max_legs(0.50) == 2 and whelan_max_legs(0.60) == 3 and whelan_max_legs(0.80) == 7


def test_ticket_maths_drops_pushed_legs() -> None:
    a, b = leg("g1", 0.6, 1.8), leg("g2", 0.5, 2.0)
    b.push = 0.03  # a whole-number line
    ticket = Ticket([a, b], "Book A")
    assert ticket.decimal == pytest.approx(3.6) and ticket.probability == pytest.approx(0.30)
    assert ticket.ev == pytest.approx((0.6 * 1.8) * (0.5 * 2.0 + 0.03) - 1)
    assert ticket.cautious_ev < ticket.ev


def test_tiers_follow_the_ai_chance_of_the_whole_ticket() -> None:
    slate = [
        leg("g1", 0.80, 1.22, fair=0.79),   # big favourite
        leg("g2", 0.76, 1.27, fair=0.76),
        leg("g3", 0.55, 1.75, fair=0.55),   # near coin flips
        leg("g4", 0.52, 1.85, fair=0.51),
        leg("g5", 0.50, 1.95, fair=0.49),
        leg("g6", 0.20, 4.60, fair=0.20),   # long shot
        leg("g7", 0.05, 15.0, fair=0.05),   # too long: never used
    ]
    tiers = {tier.key: tier for tier in bets.TIERS}
    for key, (low, high) in {"safe": (0.60, 1.01), "risky": (0.25, 0.60), "crazy": (0.03, 0.25)}.items():
        tickets = build_tier(tiers[key], slate)
        assert tickets, key
        for ticket in tickets:
            assert low <= ticket.probability < high
            assert all(l.event_id != "g7" for l in ticket.legs)
            assert len({l.event_id for l in ticket.legs}) == len(ticket.legs)  # one leg per game
    safe = build_tier(tiers["safe"], slate)
    assert safe[0].legs[0].event_id in {"g1", "g2"} and all(len(t.legs) <= 2 for t in safe)
    # No two tickets in a tier share more than one leg, and no leg is in all three
    for tier_key in ("risky", "crazy"):
        tickets = build_tier(tiers[tier_key], slate)
        keys = [{l.key for l in t.legs} for t in tickets]
        assert all(len(a & b) <= 1 for i, a in enumerate(keys) for b in keys[i + 1:])


def test_legs_the_ai_rates_below_the_market_are_never_used() -> None:
    slate = [leg("g1", 0.78, 1.25, fair=0.82), leg("g2", 0.62, 1.55, fair=0.62)]
    safe = build_tier(bets.TIERS[0], slate)
    assert safe and all(l.event_id != "g1" for t in safe for l in t.legs)


def test_safe_skips_legs_far_from_the_market_and_crazy_caps_legs() -> None:
    slate = [leg("g1", 0.80, 1.40, fair=0.70), leg("g2", 0.70, 1.40, fair=0.70)]
    assert all(l.event_id != "g1" for t in build_tier(bets.TIERS[0], slate) for l in t.legs)
    many = [leg(f"g{i}", 0.52, 1.9) for i in range(9)]
    crazy = build_tier(bets.TIERS[2], many)
    assert crazy and all(len(t.legs) <= 5 for t in crazy)


def test_tickets_are_priced_at_one_book() -> None:
    a = leg("g1", 0.75, 1.30, books=("Only A",))
    b = leg("g2", 0.80, 1.25, books=("Only B",))
    safe = build_tier(bets.TIERS[0], [a, b])
    assert all(len(t.legs) == 1 for t in safe)  # no single bookmaker offers both


def test_views_are_honest() -> None:
    out = build_all([leg("g1", 0.80, 1.22, fair=0.80), leg("g2", 0.76, 1.27, fair=0.76),
                     leg("g3", 0.20, 4.6, fair=0.20)])
    safe = out["safe"]["bets"][0]
    assert safe["badge"] in {"For fun", "Close to fair"}  # never "Value" until the AI has a record
    assert safe["expected_cost_per_10"] == pytest.approx(max(0, -safe["ev"]) * 10, abs=0.01)
    assert "loses about" in safe["why"]
    crazy = out["crazy"]["bets"][0]
    assert "1 in 5" in crazy["why"] and crazy["one_in"] == 5
    assert out["risky"]["empty"] is None or out["risky"]["bets"] == []


def test_legs_and_tickets_are_graded_from_final_scores() -> None:
    final = {"home": 110, "away": 104}
    spread = {"market": "spreads", "side": "home", "point": -6.0, "odds": {"decimal": 1.9}}
    assert bets.leg_outcome(spread, final) == "push"
    assert bets.leg_outcome({**spread, "point": -5.5}, final) == "won"
    assert bets.leg_outcome({**spread, "side": "away", "point": 5.5}, final) == "lost"
    assert bets.leg_outcome({"market": "totals", "side": "under", "point": 214.5, "odds": {"decimal": 1.9}}, final) == "won"
    assert bets.leg_outcome({"market": "h2h", "side": "away", "point": None, "odds": {"decimal": 3.0}}, final) == "lost"

    win = {"market": "h2h", "side": "home", "point": None, "odds": {"decimal": 1.5}}
    status, result = bets.grade_ticket([win, spread], [final, final])
    assert status == "won" and result["profit"] == pytest.approx(0.5)  # the pushed leg drops out
    status, result = bets.grade_ticket([win, {**spread, "point": -7.5}], [final, final])
    assert status == "lost" and result["profit"] == -1
    assert bets.grade_ticket([spread], [None])[0] == "void"  # called off

    rows = [{"tier": "safe", "status": "won", "payload": {"probability": 0.7}, "result": {"profit": 0.4}},
            {"tier": "safe", "status": "lost", "payload": {"probability": 0.65}, "result": {"profit": -1.0}},
            {"tier": "crazy", "status": "pending", "payload": {"probability": 0.1}, "result": None}]
    safe, risky, crazy = bets.tier_record(rows)
    assert (safe["tickets"], safe["won"], safe["actual"]) == (2, 1, 0.5)
    assert safe["predicted"] == pytest.approx(0.675) and safe["roi"] == pytest.approx(-0.3)
    assert risky["tickets"] == 0 and crazy["tickets"] == 0
