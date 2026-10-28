"""The Builder (engine/builder.py) and its presets (engine/presets.py)."""

from __future__ import annotations

import datetime
import itertools
import json
import math
import random
import time
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.ai.explain import banned_phrases
from telegram_betting_bot_simple.engine import bets
from telegram_betting_bot_simple.engine.builder import (
    MARKETS,
    Candidate,
    Controls,
    age_text,
    build,
    edge_of,
    market_catalog,
    reach_chance,
    restake,
    target_odds_for,
)
from telegram_betting_bot_simple.engine.presets import PRESETS, WEIGHTS, describe, label_for, preset
from telegram_betting_bot_simple.engine.tickets import TicketLeg, price_ticket

NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.timezone.utc)
BANNED = ("lock", "guarantee", "sure", "risk-free", "can't lose", "win back")


def start(hours: float = 7) -> str:
    return (NOW + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def cand(game: str, side: str = "home", price=1.9, chance: float = 0.5, market: str = "h2h", point=None,
         league: str = "basketball_nba", book: str = "DraftKings", hours: float = 7, teams=None, **extra) -> Candidate:
    extra.setdefault("fair", chance)
    return Candidate(ref={"league": league, "id": game, "market": market, "side": side, "point": point},
                     league=league, game_key=game, teams=teams or (f"Away {game}", f"Home {game}"),
                     start=start(hours), price=Decimal(str(price)), book=book, chance=chance, **extra)


def slate(games: int, seed: int, per_game: int = 4, margin: float = 0.045) -> list:
    """A random board: each game's moneyline at a 4.5% cut, and a total near 50/50."""
    rng = random.Random(seed)
    out = []
    for g in range(games):
        home = rng.uniform(0.12, 0.88)
        hours = 2 + (g % 30)
        for side, p in (("home", home), ("away", 1 - home)):
            price = round((1 - margin) / p, 2)
            if price > 1.01:
                out.append(cand(f"g{g}", side, price, p, hours=hours))
        if per_game >= 4:
            over = rng.uniform(0.44, 0.56)
            for side, p in (("over", over), ("under", 1 - over)):
                out.append(cand(f"g{g}", side, round((1 - margin) / p, 2), p, market="totals", point=220.5,
                                hours=hours))
    return out


def keys(ticket) -> set:
    return {leg["key"] for leg in ticket["legs"]}


def texts(result) -> list:
    out = list(result.notes) + [miss["text"] for miss in result.near_misses]
    if result.blocked:
        out.append(result.blocked["message"])
    for ticket in result.tickets:
        out += ticket["notes"] + [leg["risk"] for leg in ticket["legs"]]
    return out


# --- presets --------------------------------------------------------------------------------------

def test_presets_reproduce_the_old_tier_windows() -> None:
    for tier in bets.TIERS:
        controls = PRESETS[tier.key]
        assert controls.joint_min == tier.p_low
        assert controls.joint_max == (None if tier.p_high > 1 else tier.p_high)
        assert (controls.odds_min, controls.odds_max) == (tier.odds_low, tier.odds_high)
        assert controls.legs is None and controls.legs_max == tier.max_legs
        assert (controls.min_chance, controls.multi_leg_min) == (tier.single_min, tier.multi_leg_min)
        assert controls.whelan and controls.whelan_slack == tier.whelan_slack
        assert controls.caution_pp == 2.0 and controls.markets == {"h2h", "spreads", "totals"}
        assert label_for(controls) == tier.title and controls.preset == tier.key
    assert PRESETS["safe"].exclude_disagree and PRESETS["safe"].exclude_low_conf and PRESETS["safe"].exclude_model_price
    assert not PRESETS["risky"].exclude_model_price and not PRESETS["crazy"].exclude_model_price
    assert set(describe()) == {"safe", "risky", "crazy"} and "36x" in describe()["crazy"]["detail"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_each_preset_builds_inside_its_old_window(seed) -> None:
    board = slate(40, seed)
    for tier in bets.TIERS:
        result = build(board, PRESETS[tier.key], NOW)
        assert result.label == tier.title and result.tickets, tier.key
        for ticket in result.tickets:
            n = len(ticket["legs"])
            joint = math.prod(leg["inputs"]["chance"] for leg in ticket["legs"])
            odds = math.prod(float(leg["price"]) for leg in ticket["legs"])
            assert tier.p_low <= joint < tier.p_high
            assert tier.odds_low - 1e-9 <= odds <= tier.odds_high + 1e-9
            assert n <= tier.max_legs and n <= bets.whelan_max_legs(joint ** (1 / n)) + tier.whelan_slack
            assert len({leg["event_id"] for leg in ticket["legs"]}) == n
            assert ticket["label"] == tier.title and ticket["valid"]
            if n > 1:
                assert all(leg["inputs"]["chance"] >= tier.multi_leg_min for leg in ticket["legs"])


def test_crazy_says_expect_to_lose_most_of_these() -> None:
    result = build(slate(30, 4), PRESETS["crazy"], NOW)
    assert any("expect to lose most of these" in note for note in result.notes)
    assert all(any("expect to lose most of these" in note for note in t["notes"]) for t in result.tickets)


def test_safe_never_takes_low_confidence_disagreement_or_fair_odds_legs() -> None:
    board = [cand("a", chance=0.80, price=1.20, flags={"low_conf"}),
             cand("b", chance=0.80, price=1.21, flags={"disagree"}),
             cand("c", chance=0.80, price=1.22, flags={"model_price"}, book="Fair odds"),
             cand("d", chance=0.70, price=1.36)]
    result = build(board, PRESETS["safe"], NOW)
    used = {leg["event_id"] for t in result.tickets for leg in t["legs"]}
    assert used == {"d"}
    reasons = {miss["ref"]["id"]: miss["text"] for miss in result.near_misses}
    assert "low confidence (not in Safe)" in reasons["a"]
    assert "disagree on this game (not in Safe)" in reasons["b"]
    assert "goals model (not in Safe)" in reasons["c"]
    # Turning those rules off is a Custom run, and then they can be used
    custom = build(board, PRESETS["safe"].replace(exclude_low_conf=False, exclude_disagree=False,
                                                  exclude_model_price=False), NOW)
    assert custom.label == "Custom" and {"a", "b", "c"} & {leg["event_id"] for t in custom.tickets for leg in t["legs"]}


def test_custom_appears_when_a_risk_control_changes() -> None:
    safe = preset("safe")
    assert label_for(safe) == "Safe"
    assert label_for(safe.replace(leagues=["soccer_epl"], tickets=5, books={"fanduel"}, tz="Europe/London")) == "Safe"
    for change in ({"legs": 2}, {"min_chance": 0.7}, {"markets": {"h2h"}}, {"max_price_age_s": 3600},
                   {"target_odds": 1.5}, {"caution_pp": 3.0}):
        assert label_for(safe.replace(**change)) == "Custom", change
    assert label_for(Controls()) == "Custom"
    kept = preset("crazy", keep=safe.replace(leagues=["basketball_nba"], tickets=1))
    assert kept.leagues == ["basketball_nba"] and kept.tickets == 1 and label_for(kept) == "Crazy"


# --- controls -------------------------------------------------------------------------------------

def test_controls_round_trip_through_json() -> None:
    controls = Controls.from_dict({"preset": "risky", "leagues": ["basketball_nba", "soccer_epl"], "legs": 5,
                                   "target_odds": 10, "markets": ["h2h", "btts"], "books": ["fanduel", "draftkings"],
                                   "window": {"from": "2026-10-27T12:00:00Z", "to": "2026-10-28T12:00:00+00:00"},
                                   "kickoff": {"earliest": "18:00", "latest": None}, "tz": "Europe/London"})
    assert controls.preset == "risky" and controls.min_chance == PRESETS["risky"].min_chance
    assert controls.window == {"from": "2026-10-27T12:00:00Z", "to": "2026-10-28T12:00:00Z"}
    data = json.loads(json.dumps(controls.to_dict()))
    assert data["markets"] == ["h2h", "btts"] and data["books"] == ["draftkings", "fanduel"]
    again = Controls.from_dict(data)
    assert again == controls and again.cache_key() == controls.cache_key()
    assert controls.replace(legs=4).cache_key() != controls.cache_key()
    assert Controls.from_dict({"preset": "safe"}) == PRESETS["safe"]
    assert Controls.from_dict({"preset": "custom", "legs": 2}).preset is None
    assert Controls.from_dict({"legs": 2, "unknown": 1}).legs == 2  # unknown fields are ignored


@pytest.mark.parametrize("data, words", [
    ({"legs": 13}, "legs"), ({"legs": 0}, "legs"), ({"tickets": 6}, "tickets"), ({"min_chance": 55}, "0.55 for 55%"),
    ({"markets": ["player_props"]}, "Stage 6"), ({"markets": ["corners"]}, "Stage 6"), ({"markets": []}, "at least"),
    ({"markets": ["parlay"]}, "Unknown market"), ({"preset": "yolo"}, "Unknown preset"),
    ({"window": {"from": "soon"}}, "ISO"), ({"window": {"from": "2026-10-28T00:00:00Z", "to": "2026-10-27T00:00:00Z"}},
                                            "end after"),
    ({"kickoff": {"earliest": "25:00"}}, "18:30"), ({"tz": "Mars/Base"}, "time zone"), ({"target_odds": 1.0}, "target"),
    ({"leg_odds_min": 3, "leg_odds_max": 2}, "leg_odds_min"), ({"exclude_thin": "yes"}, "true or false"),
    ({"leagues": "nba"}, "list"), ({"leagues": ["NBA; drop"]}, "league"), ({"tolerance_pct": 95}, "tolerance"),
    ({"joint_min": 0.6, "joint_max": 0.5}, "joint_min"), ({"max_price_age_s": 10}, "max_price_age_s"),
])
def test_bad_controls_raise_with_a_reason(data, words) -> None:
    with pytest.raises(ValueError, match=words):
        Controls.from_dict(data)


def test_helpers() -> None:
    assert target_odds_for(10, 100) == 10.0 and target_odds_for("2.50", "5") == 2.0
    assert age_text(30) == "under 1 min" and age_text(40 * 60) == "40 min" and age_text(3 * 3600) == "3 h"
    assert age_text(3 * 3600 + 40 * 60) == "3 h 40 min"
    assert edge_of(0.5, 0.0, Decimal("1.91")) == pytest.approx(0.5 - 1 / 1.91)
    assert edge_of(0.45, 0.1, 2.0) == pytest.approx(0.45 - 0.9 / 2.0)
    catalog = market_catalog()
    assert set(MARKETS) <= set(catalog) and not catalog["player_props"]["available"]
    assert "Stage 6" in catalog["corners"]["reason"] and catalog["h2h"]["available"]
    with pytest.raises(ValueError):
        cand("x", chance_source="ai")
    with pytest.raises(ValueError):
        cand("x", price=1.0)


# --- the rules and their near misses --------------------------------------------------------------

GOOD = [cand("ok1", chance=0.58, price=1.75), cand("ok2", chance=0.56, price=1.84)]


@pytest.mark.parametrize("leg, change, rule, words", [
    (cand("x", book="Pinnacle"), {}, "books", "Home x @ 1.90: not offered at your books"),
    (cand("x", book="FanDuel"), {"books": {"draftkings"}}, "books", "not offered at your books"),
    (cand("x", side="home", market="spreads", point=-4.5, price=1.91, label="Celtics −4.5", price_age_s=13200), {},
     "max_price_age", "Celtics −4.5 @ 1.91: price 3 h 40 min old (max 3 h)"),
    (cand("x", price=1.15, chance=0.83), {"leg_odds_min": 1.2}, "leg_odds_min", "odds 1.15 below your minimum 1.20"),
    (cand("x", price=5.5, chance=0.17), {"leg_odds_max": 4.0}, "leg_odds_max", "odds 5.50 above your maximum 4.00"),
    (cand("x", chance=0.52), {"min_chance": 0.55}, "min_chance", "chance 52% below your minimum 55%"),
    (cand("x", chance=0.50, price=1.91), {"min_edge": -0.02}, "min_edge", "edge -2.4 pts below your minimum -2.0"),
    (cand("x", flags={"disagree"}), {"exclude_disagree": True}, "disagree", "the sources disagree on this game (excluded)"),
    (cand("x", flags={"thin"}), {"exclude_thin": True}, "thin", "thin market only (excluded)"),
    (cand("x", flags={"low_conf"}), {"exclude_low_conf": True}, "low_conf", "low confidence (excluded)"),
    (cand("x", flags={"model_price"}, book="Fair odds"), {"exclude_model_price": True}, "model_price",
     "priced at fair odds by the goals model (excluded)"),
])
def test_every_leg_rule_names_itself_in_the_near_misses(leg, change, rule, words) -> None:
    # Strict: closest effort would fill the five tickets asked for with the rule loosened (tests/test_best_effort.py)
    result = build(GOOD + [leg], Controls(legs=1, tickets=5, best_effort=False).replace(**change), NOW)
    assert result.tickets and all(leg.key not in keys(t) for t in result.tickets)
    miss = next(m for m in result.near_misses if m["key"] == leg.key)
    assert miss["rule"] == rule and words in miss["text"]
    assert result.stats["removed"][rule] == 1


@pytest.mark.parametrize("leg, change, rule", [
    (cand("x", league="soccer_epl"), {"leagues": ["basketball_nba"]}, "league"),
    (cand("x", "over", market="totals", point=220.5), {"markets": {"h2h", "spreads"}}, "market"),
    (cand("x", hours=30), {"window": {"from": start(0), "to": start(24)}}, "window"),
    (cand("x", hours=1), {"window": {"from": start(2)}}, "window"),
    (cand("x", hours=-1), {}, "started"),
    (cand("x", flags={"live"}), {}, "started"),
    (cand("x", chance=0.0), {}, "no_chance"),
])
def test_scope_rules_are_counted_not_listed(leg, change, rule) -> None:
    board = [cand("ok1", chance=0.55, price=1.75, hours=7)]
    result = build(board + [leg], Controls(legs=1).replace(**change), NOW)
    assert result.stats["removed"][rule] == 1
    assert all(m["key"] != leg.key for m in result.near_misses)
    assert all(leg.key not in keys(t) for t in result.tickets)


def test_the_kick_off_window_is_local_time() -> None:
    # 23:30 UTC on 27 October is 19:30 in New York (still on summer time)
    late = Candidate(ref={"league": "basketball_nba", "id": "ny", "market": "h2h", "side": "home"},
                     league="basketball_nba", game_key="ny", teams=("A", "B"), start="2026-10-27T23:30:00Z",
                     price=Decimal("1.9"), book="DraftKings", chance=0.5)
    base = Controls(legs=1, tz="America/New_York")
    assert build([late], base.replace(kickoff={"earliest": "20:00"}), NOW).stats["removed"]["kickoff"] == 1
    assert build([late], base.replace(kickoff={"latest": "20:00"}), NOW).tickets
    assert build([late], base.replace(kickoff={"earliest": "19:00", "latest": "19:45"}), NOW).tickets
    # A window past midnight wraps: 19:30 is inside 18:00-02:00 and outside 02:00-18:00
    assert build([late], base.replace(kickoff={"earliest": "18:00", "latest": "02:00"}), NOW).tickets
    assert not build([late], base.replace(kickoff={"earliest": "02:00", "latest": "18:00"}), NOW).tickets
    assert not build([late], base.replace(kickoff={"earliest": "20:00"}, tz=None), NOW).stats["removed"].get("kickoff")


def test_sharp_books_never_price_a_ticket_and_allowed_books_pick_the_best_allowed() -> None:
    leg = cand("x", chance=0.5, price=1.90, book="DraftKings",
               prices={"DraftKings": "1.90", "FanDuel": "1.95", "Pinnacle": "2.05", "betfair_ex_eu": "2.10"})
    best = build([leg], Controls(legs=1), NOW).tickets[0]["legs"][0]
    assert (best["price"], best["book"]) == ("1.95", "FanDuel")
    mine = build([leg], Controls(legs=1, books={"draftkings"}), NOW).tickets[0]["legs"][0]
    assert (mine["price"], mine["book"]) == ("1.90", "DraftKings")
    # The same pick passed once per book: the best price wins, and Pinnacle is never it
    twice = [cand("y", price=1.90, book="DraftKings"), cand("y", price=1.97, book="BetMGM"),
             cand("y", price=2.10, book="Pinnacle")]
    ticket = build(twice, Controls(legs=1), NOW).tickets
    assert len(ticket) == 1 and (ticket[0]["legs"][0]["price"], ticket[0]["legs"][0]["book"]) == ("1.97", "BetMGM")
    assert any("Only 1 ticket sharing no leg fits the rules (you asked for 3)" in note
               for note in build(twice, Controls(legs=1), NOW).notes)
    assert all(leg["book"] != "Pinnacle" for t in build(slate(10, 9), Controls(), NOW).tickets for leg in t["legs"])


# --- the search -----------------------------------------------------------------------------------

def brute_force(board, controls):
    """Every ticket of 1-4 legs, ranked: the rules and the score written out again, independently."""
    weights = WEIGHTS.get(controls.preset, WEIGHTS[None])
    caution = controls.caution_pp / 100
    legs = [leg for leg in board if controls.min_chance is None or leg.chance >= controls.min_chance]
    if controls.target_odds:
        tol = controls.tolerance_pct / 100
        low, high = controls.target_odds * (1 - tol), controls.target_odds * (1 + tol)
    else:
        low, high = controls.odds_min or 1.0, controls.odds_max or math.inf
    sizes = [controls.legs] if controls.legs else range(1, controls.legs_max + 1)
    ranked = []
    for n in sizes:
        for combo in itertools.combinations(legs, n):
            if len({leg.game_key for leg in combo}) < n:
                continue
            if n > 1 and controls.multi_leg_min and min(leg.chance for leg in combo) < controls.multi_leg_min:
                continue
            odds = math.prod(float(leg.price) for leg in combo)
            joint = math.prod(leg.chance for leg in combo)
            if not low - 1e-9 <= odds <= high + 1e-9:
                continue
            if not controls.target_odds:
                if controls.joint_min and joint < controls.joint_min - 1e-9:
                    continue
                if controls.joint_max and joint >= controls.joint_max:
                    continue
            if controls.whelan and controls.legs is None and n > 1 and \
                    n > bets.whelan_max_legs(joint ** (1 / n)) + controls.whelan_slack:
                continue
            close = 0.0
            if controls.target_odds:
                gap = math.log(odds / controls.target_odds)
                close = abs(gap) / (math.log(controls.target_odds / low) if gap < 0
                                    else math.log(high / controls.target_odds))
            ret = sum(math.log(max((leg.chance - caution) * float(leg.price) + leg.push, 1e-9)) for leg in combo)
            score = -weights.target * close + weights.ret * ret + weights.chance * math.log(joint) - 0.001 * n
            ranked.append((score, tuple(sorted(leg.key for leg in combo))))
    ranked.sort(reverse=True)
    chosen, used = [], set()
    for score, combo in ranked:
        if used & set(combo):
            continue
        chosen.append((score, combo))
        used |= set(combo)
        if len(chosen) == controls.tickets:
            break
    return chosen


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("controls", [
    Controls(legs_max=4), Controls(target_odds=6.0, tolerance_pct=20), Controls(legs=3, target_odds=5.0),
    Controls(legs_max=4, whelan=False, target_odds=12.0), Controls(legs=2, tickets=5),
    Controls.from_dict({"preset": "risky"}), Controls.from_dict({"preset": "crazy", "legs_max": 4}),
    Controls.from_dict({"preset": "safe", "exclude_thin": False}),
], ids=["auto", "target6", "3legs", "nowhelan", "2legs-5tickets", "risky", "crazy4", "safe"])
def test_the_exhaustive_search_matches_brute_force(seed, controls) -> None:
    board = slate(9, seed)
    expected = brute_force(board, controls)
    result = build(board, controls, NOW)
    got = [(t["score"], tuple(sorted(keys(t)))) for t in result.tickets]
    assert [combo for _, combo in got] == [combo for _, combo in expected]
    for (score, _), (want, _) in zip(got, expected):
        assert score == pytest.approx(want, abs=2e-6)
    if not expected:
        assert result.blocked and result.blocked["message"].startswith("No ")


def test_the_beam_finds_a_ten_leg_ticket_near_the_target() -> None:
    board = slate(40, 11)
    controls = Controls(legs=10, target_odds=20.0, tolerance_pct=10, tickets=2)
    result = build(board, controls, NOW)
    assert len(result.tickets) == 2
    for ticket in result.tickets:
        assert len(ticket["legs"]) == 10 and len({leg["event_id"] for leg in ticket["legs"]}) == 10
        assert 18.0 - 1e-9 <= ticket["joint"]["odds"] <= 22.0 + 1e-9
        assert ticket["odds"]["decimal"] == pytest.approx(ticket["joint"]["odds"], abs=0.01) and ticket["valid"]
        # A 10-leg ticket that is really fair odds of ~20 wins about 1 in 20 less ten cuts
        assert 0.02 < ticket["reach_target"]["chance"] < 0.06
    assert not keys(result.tickets[0]) & keys(result.tickets[1])


def test_the_beam_is_close_to_the_best_five_leg_ticket() -> None:
    board = slate(12, 5, per_game=2)
    controls = Controls(legs=5, target_odds=12.0, tolerance_pct=25, tickets=1)
    expected = brute_force(board, controls)
    got = build(board, controls, NOW).tickets[0]
    assert expected and got["score"] >= expected[0][0] - 0.02


def test_auto_beyond_four_legs_uses_the_beam() -> None:
    # Leg odds capped at 1.45 can't reach 20x in four legs: Auto has to go long, with the Whelan cap lifted
    board = slate(40, 3)
    controls = Controls(legs_max=12, target_odds=20.0, leg_odds_max=1.45, whelan=False, tickets=1)
    ticket = build(board, controls, NOW).tickets[0]
    assert len(ticket["legs"]) >= 9 and all(float(leg["price"]) <= 1.45 for leg in ticket["legs"])
    assert 18.0 <= ticket["joint"]["odds"] <= 22.0


@pytest.mark.parametrize("controls", [Controls(tickets=5, legs_max=3), Controls(legs=4, tickets=5, target_odds=8),
                                      Controls(legs=7, tickets=4, target_odds=15, tolerance_pct=20)])
def test_tickets_share_no_leg(controls) -> None:
    result = build(slate(40, 21), controls, NOW)
    assert len(result.tickets) == controls.tickets
    for a, b in itertools.combinations(result.tickets, 2):
        assert not keys(a) & keys(b)
    assert [t["rank"] for t in result.tickets] == list(range(1, controls.tickets + 1))
    assert [t["score"] for t in result.tickets] == sorted((t["score"] for t in result.tickets), reverse=True)


def test_swaps_keep_the_rules() -> None:
    board = slate(30, 8)
    controls = Controls.from_dict({"preset": "risky", "tickets": 2, "legs": 2})
    result = build(board, controls, NOW)
    by_key = {leg.key: leg for leg in board}
    taken = set().union(*(keys(t) for t in result.tickets))
    swaps = 0
    for ticket in result.tickets:
        for i, leg in enumerate(ticket["legs"]):
            assert len(leg["swaps"]) <= 3
            for swap in leg["swaps"]:
                swaps += 1
                assert swap["key"] not in taken
                others = [by_key[other["key"]] for j, other in enumerate(ticket["legs"]) if j != i]
                new = others + [by_key[swap["key"]]]
                assert len({x.game_key for x in new}) == len(new)
                odds = math.prod(float(x.price) for x in new)
                joint = math.prod(x.chance for x in new)
                assert swap["odds"] == pytest.approx(odds, abs=0.01) and swap["joint"] == pytest.approx(joint, abs=1e-4)
                assert controls.odds_min <= odds <= controls.odds_max
                assert controls.joint_min <= joint < controls.joint_max
                assert all(x.chance >= controls.multi_leg_min for x in new)
            scores = [swap["score_change"] for swap in leg["swaps"]]
            assert scores == sorted(scores, reverse=True)
    assert swaps


def test_reach_target_maths_against_price_ticket() -> None:
    legs = [(0.5, 0.0, 2.0), (0.45, 0.10, 1.9)]
    ticket_legs = [TicketLeg(key=f"k{i}", league="nba", event_id=f"e{i}", market="spreads", side="home", point=-3.0,
                             price=Decimal(str(d)), label=f"L{i}", game=f"g{i}", start=start(), ai_prob=c, push_prob=p)
                   for i, (c, p, d) in enumerate(legs)]
    priced = price_ticket(ticket_legs, "multi", 10, now=NOW)
    # Only every leg winning pays 3.8x; a push on the second leaves 2.0x, a profit, and both pushing is stake back
    assert reach_chance(legs, 3.0) == pytest.approx(priced["chance"]["every_leg"]["ai"], abs=1e-4)
    assert reach_chance(legs, 1.5) == pytest.approx(priced["chance"]["profit"]["ai"], abs=1e-4)
    assert reach_chance(legs, 1.5) == pytest.approx(0.5 * 0.45 + 0.5 * 0.10)
    assert reach_chance(legs, None) == pytest.approx(0.5 * 0.45)
    # On a built ticket with no pushes, reaching the target's low end is every leg winning
    result = build(slate(20, 2), Controls(legs=3, target_odds=6.0, tickets=3), NOW)
    for ticket in result.tickets:
        assert ticket["reach_target"]["odds"] == pytest.approx(5.4) and ticket["reach_target"]["target"] == 6.0
        at_target = ticket["reach_target"]["at_target"]
        assert at_target == (ticket["reach_target"]["chance"] if ticket["joint"]["odds"] >= 6.0 else 0.0)
        assert ticket["reach_target"]["chance"] == pytest.approx(ticket["chance"]["every_leg"]["ai"], abs=1e-4)
        assert ticket["reach_target"]["chance"] == ticket["joint"]["chance"]
    # A push leg in a built ticket: a push leaves too little to reach the target, so it doesn't count
    push = cand("p", market="spreads", point=-3.0, chance=0.45, push=0.08, price=2.0)
    other = cand("q", chance=0.55, price=1.80)
    built = build([push, other], Controls(legs=2, target_odds=3.6, tolerance_pct=5), NOW).tickets[0]
    assert built["reach_target"]["chance"] == pytest.approx(0.45 * 0.55, abs=1e-4)
    assert built["chance"]["every_leg"]["ai"] == pytest.approx(0.45 * 0.55, abs=1e-4)


def test_the_performance_bound() -> None:
    board = slate(50, 99)[:200]
    assert len(board) == 200
    for controls in (Controls(legs=12, target_odds=40.0, tolerance_pct=15, tickets=3),
                     Controls(legs_max=12, target_odds=40.0, whelan=False, tickets=3),
                     Controls(legs_max=4, tickets=5)):
        began = time.perf_counter()
        result = build(board, controls, NOW)
        took = time.perf_counter() - began
        assert result.tickets
        assert took < 4.5, f"{controls}: {took:.2f}s"  # the target is 1.5 s on the dev machine; this bound is generous


# --- blocked and notes ----------------------------------------------------------------------------
# What blocked a run is the strict run's story (best_effort False): with closest effort on, a run with any eligible
# leg always builds (tests/test_best_effort.py)

def test_blocked_says_which_rule_and_by_how_many() -> None:
    board = [cand(f"g{i}", chance=0.60, price=1.58) for i in range(4)]
    board += [cand(f"h{i}", chance=0.50, price=1.90) for i in range(8)]
    result = build(board, Controls(legs=6, target_odds=20.0, min_chance=0.55, best_effort=False), NOW)
    assert not result.tickets
    assert result.blocked["message"] == "No 6-leg ticket near 20.0x: only 4 games pass 'min chance 55%'."
    assert result.blocked["rules"] == [{"rule": "min_chance", "label": "min chance 55%", "removed": 8}]
    assert result.blocked["passed"] == {"legs": 4, "games": 4} and "min chance 55%" in result.blocked["relax"]


def test_blocked_when_the_target_is_out_of_reach_or_whelan_rules_it_out() -> None:
    short = [cand(f"g{i}", chance=0.8, price=1.2) for i in range(6)]
    out = build(short, Controls.from_dict({"preset": "safe", "target_odds": 10.0, "best_effort": False}), NOW)
    assert "the longest 3 legs that pass reach only 1.73x" in out.blocked["message"]
    whelan = build(slate(30, 6), Controls(legs_max=6, target_odds=40.0, best_effort=False), NOW)
    assert not whelan.tickets and "'the Whelan leg cap' ruled out" in whelan.blocked["message"]
    assert "set the number of legs to lift it" in whelan.blocked["message"]
    # Setting the number of legs lifts it, and the ticket says so
    lifted = build(slate(30, 6), Controls(legs=6, target_odds=40.0), NOW)
    assert lifted.tickets and any("Whelan cap was lifted" in note for note in lifted.tickets[0]["notes"])
    empty = build([], Controls(), NOW, unpriced={"soccer_epl": "No prices loaded for EPL; asked for them in the background."})
    assert empty.blocked["message"].startswith("No prices are loaded")
    assert empty.near_misses[0]["rule"] == "no_prices" and "EPL" in empty.near_misses[0]["text"]
    many = build([cand(f"x{i}", chance=0.3, price=3.0) for i in range(20)], Controls(min_chance=0.5), NOW,
                 unpriced={"soccer_epl": "No prices loaded for EPL"})
    # A league with no prices doesn't take a leg's place: 1 league plus the 12 legs
    assert len(many.near_misses) == 13 and many.near_misses[0]["rule"] == "no_prices"
    assert sum(miss["rule"] != "no_prices" for miss in many.near_misses) == 12
    nothing = build([cand("x", chance=0.3, price=3.0)], Controls(min_chance=0.5, best_effort=False), NOW)
    assert nothing.blocked["message"] == "No ticket: no leg passes the rules ('min chance 50%' ruled out 1)."


def test_a_target_replaces_the_presets_band() -> None:
    # "5 legs across NBA and EPL at about 10x, Risky": Risky's leg rules, the person's legs and target
    board = slate(30, 12)
    result = build(board, Controls.from_dict({"preset": "risky", "legs": 5, "target_odds": 10.0}), NOW)
    assert result.label == "Custom" and result.tickets
    assert all(len(t["legs"]) == 5 and 9.0 <= t["joint"]["odds"] <= 11.0 for t in result.tickets)
    assert any("band don't apply" in note for note in result.notes)


def test_one_leg_per_game_even_when_same_game_legs_are_allowed() -> None:
    board = slate(6, 14)
    result = build(board, Controls(avoid_same_game=False, max_per_game=2, legs=3, tickets=2), NOW)
    assert all(len({leg["event_id"] for leg in t["legs"]}) == 3 for t in result.tickets)
    assert any("one leg per game" in note for note in result.notes)


def test_team_and_league_caps() -> None:
    # The same team plays twice in three days
    board = [cand("g1", teams=("Lakers", "Celtics"), chance=0.62, price=1.6),
             cand("g2", teams=("Celtics", "Knicks"), chance=0.62, price=1.6, hours=50),
             cand("g3", teams=("Bulls", "Heat"), chance=0.6, price=1.6, league="basketball_wnba")]
    one = build(board, Controls(legs=2, tickets=3), NOW)
    assert all({"g1", "g2"} - {leg["event_id"] for leg in t["legs"]} for t in one.tickets)
    two = build(board, Controls(legs=2, tickets=3, max_per_team=2), NOW)
    assert any({leg["event_id"] for leg in t["legs"]} == {"g1", "g2"} for t in two.tickets)
    league = build(board, Controls(legs=2, tickets=3, max_per_team=2, max_per_league=1, best_effort=False), NOW)
    assert league.tickets and all(len({leg["league"] for leg in t["legs"]}) == 2 for t in league.tickets)


def test_ideas_are_only_ideas() -> None:
    board = [cand("g", "home", 1.40, 0.68), cand("g", "over", 1.91, 0.5, market="totals", point=220.5),
             cand("h", "home", 1.90, 0.5)]
    result = build(board, Controls(prefer_correlation=True), NOW)
    assert result.ideas and "Idea only" in result.ideas[0]["text"]
    assert all(len({leg["event_id"] for leg in t["legs"]}) == len(t["legs"]) for t in result.tickets)
    assert not build(board, Controls(), NOW).ideas


def test_results_are_json_ready_and_deterministic() -> None:
    board = slate(25, 31)
    first = build(board, Controls.from_dict({"preset": "crazy"}), NOW).to_dict()
    second = build(list(reversed(board)), Controls.from_dict({"preset": "crazy"}), NOW).to_dict()
    json.dumps(first)  # Decimals are strings already
    assert [keys(t) for t in first["tickets"]] == [keys(t) for t in second["tickets"]]
    ticket = first["tickets"][0]
    leg = ticket["legs"][0]
    assert {"ref", "chance", "edge", "ev", "age", "why", "risk", "swaps", "flags"} <= set(leg)
    assert {"reach_target", "joint", "cautious", "label", "notes", "refs", "book_cut", "expected"} <= set(ticket)
    assert first["stats"]["candidates"] == len(board) and first["stats"]["combos"] > 0 and first["stats"]["ms"] >= 0
    assert first["label"] == "Crazy" and first["preset"] == "crazy" and first["controls"]["preset"] == "crazy"


def test_restake_prices_the_same_tickets_at_another_stake() -> None:
    board = slate(20, 41)
    controls = Controls(legs=2, target_odds=4.0, tickets=2)
    shared = build(board, controls, NOW).to_dict()
    mine = restake(json.loads(json.dumps(shared)), "25", now=NOW)
    direct = build(board, controls, NOW, stake=25).to_dict()
    assert mine["stake"] == "25.00" and shared["stake"] == "10.00"
    for got, want in zip(mine["tickets"], direct["tickets"]):
        for field in ("unit_stake", "max_payout", "max_profit", "expected", "chance", "book_cut", "odds"):
            assert got[field] == want[field], field
        assert [leg["swaps"] for leg in got["legs"]] == [leg["swaps"] for leg in want["legs"]]
        assert [leg["edge"] for leg in got["legs"]] == [leg["edge"] for leg in want["legs"]]


def test_messages_never_use_banned_words() -> None:
    board = slate(20, 50) + [cand("x", flags={"disagree"}), cand("y", flags={"model_price"}, book="Fair odds"),
                             cand("z", price_age_s=20000)]
    runs = [build(board, PRESETS[name], NOW) for name in PRESETS]
    runs += [build(board, Controls(legs=8, target_odds=500.0), NOW), build([], Controls(), NOW),
             build(board, Controls(min_chance=0.99), NOW), build(board, Controls(legs_max=6, target_odds=40.0), NOW),
             build(board, Controls(prefer_correlation=True, avoid_same_game=False, exclude_started=False,
                                   exclude_disagree=True), NOW)]
    for result in runs:
        for text in texts(result) + [idea["text"] for idea in result.ideas]:
            assert not banned_phrases(text), text
            assert not any(word in text.lower() for word in BANNED), text


def test_a_fair_odds_leg_is_ranked_as_a_bookmaker_would_price_it() -> None:
    """A goals-model leg at fair odds carries no bookmaker's cut: ranked at that price it would beat every leg a
    person can bet at. The search values it with a typical 5% margin off (its shown price stays the fair odds)."""
    from telegram_betting_bot_simple.engine.builder import MODEL_MARGIN
    book = cand("a", chance=0.58, price=1.66)  # a 3.7% cut on this side
    model = cand("b", chance=0.58, price=1.72, flags={"model_price"}, book="Fair odds", market="btts", side="yes",
                 chance_source="model", fair=None)
    controls = Controls.from_dict({"preset": "risky", "markets": ["h2h", "btts"], "legs": 1, "tickets": 2})
    result = build([model, book], controls, NOW)
    assert [t["legs"][0]["key"] for t in result.tickets] == [book.key, model.key]
    assert result.tickets[1]["legs"][0]["price"] == "1.72"  # shown at its fair odds
    cautious = result.tickets[1]["cautious"]["return"]
    assert cautious == pytest.approx((0.58 - 0.02) * 1.72 / (1 + MODEL_MARGIN), abs=1e-4)
    # Without the flag the same price would have come first
    plain = cand("b", chance=0.58, price=1.72, market="btts", side="yes", chance_source="model", fair=None)
    assert build([plain, book], controls, NOW).tickets[0]["legs"][0]["key"] == plain.key


def test_ticket_chances_and_notes_are_the_ones_a_run_puts_on_its_tickets() -> None:
    from telegram_betting_bot_simple.engine.builder import ticket_chances, ticket_notes
    legs = [cand(f"g{n}", chance=0.55, price=1.80, book="DraftKings" if n else "FanDuel") for n in range(4)]
    controls = Controls.from_dict({"preset": "crazy", "legs": 4, "target_odds": 10.0, "tolerance_pct": 15,
                                   "min_chance": None})
    ticket = build(legs, controls, NOW).tickets[0]
    outcomes = [(leg["inputs"]["chance"], leg["inputs"]["push"], float(leg["price"])) for leg in ticket["legs"]]
    again = ticket_chances(outcomes, 10.0, 15)
    assert again["joint"] == ticket["joint"] and again["reach_target"] == ticket["reach_target"]
    assert ticket["reach_target"]["odds"] == pytest.approx(8.5)
    notes = ticket_notes([(leg["label"], leg["flags"], leg["book"]) for leg in ticket["legs"]],
                         ticket["joint"]["chance"], crazy=True)
    assert notes == [n for n in ticket["notes"] if not n.startswith("You asked for")]
    assert "Long shot: expect to lose most of these." in notes


# --- fixes after review ---------------------------------------------------------------------------

def dnb(game: str, chance: float, push: float, **extra) -> Candidate:
    """A draw-no-bet leg priced at the goals model's fair odds, (1 - push) / chance."""
    return cand(game, market="draw_no_bet", side="home", chance=chance, push=push,
                price=round((1 - push) / chance, 2), fair=None, flags={"model_price"}, book="Fair odds",
                chance_source="model", **extra)


def test_a_push_is_stake_back_in_the_joint_block_the_notes_and_the_band() -> None:
    from telegram_betting_bot_simple.engine.builder import ticket_chances, ticket_notes
    a, b = dnb("a", 0.4575, 0.2841), dnb("b", 0.5213, 0.2677)
    ticket = build([a, b], Controls(markets={"draw_no_bet"}, legs=2), NOW).tickets[0]
    joint = ticket["joint"]
    # Priced at fair, the ticket's fair odds are its odds (every leg at (1 - push) / chance), not 1 / every leg winning
    assert joint["fair_odds"] == pytest.approx(joint["odds"], abs=0.02) and joint["fair_odds"] < 2.3
    assert joint["chance"] == pytest.approx(0.4575 * 0.5213, abs=1e-4)
    not_lose = (0.4575 + 0.2841) * (0.5213 + 0.2677)
    assert joint["lose"] == pytest.approx(1 - not_lose, abs=1e-4)
    assert joint["profit"] == pytest.approx(not_lose - 0.2841 * 0.2677, abs=1e-4)
    assert joint["profit"] == pytest.approx(ticket["chance"]["profit"]["ai"], abs=1e-4)
    # Every leg wins only 24% of the time, but it wins money 51% of the time: not a long shot
    assert not any("expect to lose most" in note for note in ticket["notes"])
    assert any(note.startswith("Priced at fair odds; check your bookmaker's price") for note in ticket["notes"])
    # The joint band is on the chance of winning money: 51% is inside 25-60%
    banded = build([a, b], Controls(markets={"draw_no_bet"}, legs=2, joint_min=0.25, joint_max=0.60), NOW)
    assert [keys(t) for t in banded.tickets] == [{a.key, b.key}]
    assert not build([a, b], Controls(markets={"draw_no_bet"}, legs=2, joint_min=0.55, best_effort=False), NOW).tickets
    # With no push nothing changes: fair odds 1 / chance, and under 1 in 4 is a long shot
    assert ticket_chances([(0.2, 0.0, 4.5)])["joint"]["fair_odds"] == 5.0
    assert "Long shot: expect to lose most of these." in ticket_notes([("x", set(), "DraftKings")], 0.2)
    assert "Long shot: expect to lose most of these." not in ticket_notes([("x", set(), "DraftKings")], 0.3)


def test_a_fair_odds_leg_has_no_edge_to_measure() -> None:
    bettable = [cand(f"g{i}", chance=0.53, price=1.91) for i in range(4)]  # +0.6 pts against the price
    fair = [cand(f"m{i}", chance=0.58, price=1.72, flags={"model_price"}, book="Fair odds", market="btts",
                 side="yes", chance_source="model", fair=None) for i in range(4)]
    result = build(bettable + fair, Controls(markets={"h2h", "btts"}, legs=2, min_edge=-0.01, best_effort=False), NOW)
    assert result.tickets and not any(leg["model_price"] for t in result.tickets for leg in t["legs"])
    miss = next(m for m in result.near_misses if m["key"] == fair[0].key)
    assert miss["rule"] == "min_edge" and "no bookmaker's price to measure an edge against" in miss["reason"]
    # Without the rule a fair-odds leg shows no edge; a bettable leg shows its points
    shown = build([fair[0], bettable[0]], Controls(markets={"h2h", "btts"}, legs=1, tickets=2), NOW)
    edges = {t["legs"][0]["key"]: t["legs"][0]["edge"] for t in shown.tickets}
    assert edges[fair[0].key] is None and edges[bettable[0].key] == pytest.approx(0.53 - 1 / 1.91, abs=1e-4)
    swaps = build(fair[:2] + bettable[:2], Controls(markets={"h2h", "btts"}, legs=1, tickets=1), NOW).tickets[0]
    assert any(s["edge"] is None for s in swaps["legs"][0]["swaps"])


def test_presets_keep_the_old_tiers_value_rules() -> None:
    from telegram_betting_bot_simple.engine.presets import VALUE_RULES, value_rules
    for tier in bets.TIERS:
        assert (VALUE_RULES[tier.key]["min_ev"], VALUE_RULES[tier.key]["max_book_cut"]) == (tier.ev_floor,
                                                                                            tier.edge_cap)
    assert VALUE_RULES["safe"]["three_leg_min"] == 0.85 and VALUE_RULES["safe"]["max_gap"] == bets.SAFE_MAX_GAP
    assert value_rules("Custom") == {} and value_rules("Safe") == VALUE_RULES["safe"]
    # Four heavy favourites at 82% priced 1.12 (an 8% cut each): the old Safe tier lays off, and so does Safe here
    favourites = [cand(f"h{i}", chance=0.82, price=1.12) for i in range(4)]
    old = [bets.Candidate(event_id=f"h{i}", game=f"A{i} @ H{i}", market="h2h", side="home", point=None,
                          label=f"H{i}", probability=0.82, fair=0.82, prices={"DraftKings": 1.12}) for i in range(4)]
    assert bets.build_tier(bets.TIERS[0], old) == []
    strict = PRESETS["safe"].replace(best_effort=False)  # still Safe: strict or not keeps the preset's name
    safe = build(favourites, strict, NOW)
    assert safe.label == "Safe" and safe.tickets == [] and safe.stats["rejected"].get("min_ev")
    assert any("(Safe)" in rule["label"] for rule in safe.blocked["ticket_rules"])
    # Any edit makes it Custom, the person's own rules: then they build (and show their expected loss)
    assert build(favourites, PRESETS["safe"].replace(min_chance=0.61), NOW).tickets
    # Safe takes 3 legs only when every leg is 85%+, and no leg 8+ points from the books
    trio = [cand("a", chance=0.95, price=1.04), cand("b", chance=0.95, price=1.04), cand("c", chance=0.73, price=1.36)]
    three = build(trio, PRESETS["safe"].replace(legs=3), NOW)
    assert three.label == "Custom" and three.tickets  # Custom: the 3-leg floor doesn't apply
    trio_safe = build(trio, strict, NOW)
    assert trio_safe.tickets and all(len(t["legs"]) <= 2 for t in trio_safe.tickets)
    assert trio_safe.stats["rejected"].get("three_leg")
    far = build([cand("x", chance=0.75, price=1.33, fair=0.65)], PRESETS["safe"], NOW)
    assert far.near_misses[0]["rule"] == "max_gap" and "10 pts above the books'" in far.near_misses[0]["reason"]


def test_the_pool_reaches_a_target_the_favourites_cant() -> None:
    # 60 legs: more than the exhaustive search's 40, and a target that needs prices well above the favourites'
    small = slate(15, 3)
    for controls in (Controls(legs=2, target_odds=5.0), Controls(legs=3, target_odds=10.0, tickets=1)):
        expected = brute_force(small, controls)
        result = build(small, controls, NOW)
        assert len(result.tickets) == len(expected) == controls.tickets
        assert result.tickets[0]["score"] >= expected[0][0] - 0.03
    # 160 legs: three tickets sharing no leg, from the exhaustive search and from the beam
    board = slate(40, 3)
    for n, target in ((3, 10.0), (6, 60.0), (8, 300.0)):
        result = build(board, Controls(legs=n, target_odds=target), NOW)
        assert len(result.tickets) == 3, (n, target)
        for ticket in result.tickets:
            assert len(ticket["legs"]) == n and 0.9 * target - 1e-6 <= ticket["joint"]["odds"] <= 1.1 * target + 1e-6


def test_blocked_names_the_shape_rule_that_cut_every_branch() -> None:
    capped = build(slate(20, 5), Controls(legs=3, max_per_league=1, best_effort=False), NOW)  # every leg is NBA
    assert not capped.tickets and "1 per league" in capped.blocked["message"]
    assert any("1 per league" in words for words in capped.blocked["relax"])
    assert "fewer legs than 3" in capped.blocked["relax"]
    beam_capped = build(slate(30, 5), Controls(legs=8, max_per_league=3, best_effort=False), NOW)
    assert not beam_capped.tickets and "3 per league" in beam_capped.blocked["message"]
    risky8 = build(slate(40, 3), Controls.from_dict({"preset": "risky", "legs": 8, "best_effort": False}), NOW)
    assert not risky8.tickets and "'joint chance 25%+'" in risky8.blocked["message"]
    assert "joint chance 25%+" in risky8.blocked["relax"] and "max price age 3 h" not in risky8.blocked["relax"]
    exact = build(slate(40, 3), Controls(legs=8, target_odds=60.0, tolerance_pct=0, best_effort=False), NOW)
    assert not exact.tickets and "none of the" in exact.blocked["message"]
    assert exact.blocked["relax"] == ["a wider tolerance than ±0%"]


def test_try_relaxing_follows_what_blocked_the_run() -> None:
    # Out of reach: more legs or a lower target; a stale favourite's rule can't raise the reach, so it isn't named
    board = slate(20, 7) + [cand("old", chance=0.8, price=1.2, price_age_s=20000)]
    out = build(board, Controls.from_dict({"preset": "risky", "legs": 3, "target_odds": 500.0, "best_effort": False}),
                NOW)
    assert "reach only" in out.blocked["message"]
    assert out.blocked["relax"][:2] == ["more legs than 3", "a lower target"]
    assert "max price age 3 h" not in out.blocked["relax"]
    # A long price the same rule removed could raise it: then it is named
    longer = build(board + [cand("old2", chance=0.3, price=30.0, price_age_s=20000)],
                   Controls(legs=3, target_odds=500.0, best_effort=False), NOW)
    assert "max price age 3 h" in longer.blocked["relax"]
    # Auto stops at its own number of legs, and says so
    auto = build(board, Controls.from_dict({"preset": "safe", "target_odds": 10.0, "best_effort": False}), NOW)
    assert "more legs (Auto stops at 3)" in auto.blocked["relax"]
    # Even the shortest legs pay too much: fewer legs or a higher target
    short = build(board, Controls(legs=4, target_odds=1.3, best_effort=False), NOW)
    assert short.blocked["relax"][:2] == ["fewer legs than 4", "a higher target"]
    # ... counted one leg a game: a game's second short price can't go on the same ticket
    eight = [cand(f"r{i}", chance=0.78, price=1.25) for i in range(10)]
    eight.append(cand("r0", side="over", market="totals", point=2.5, chance=0.95, price=1.02))
    band = build(eight, Controls.from_dict({"preset": "risky", "legs": 8, "best_effort": False}), NOW)
    assert "even the shortest legs that pass pay 4.86x" in band.blocked["message"]
    assert band.blocked["relax"][:2] == ["fewer legs than 8", "a higher ticket-odds band"]


def test_near_misses_are_the_closest_to_passing_at_most_two_a_game() -> None:
    board = [cand(f"g{i}", chance=0.59, price=1.6) for i in range(3)]  # a point below the minimum
    board += [cand("old", side=side, market=market, point=point, chance=0.7, price=1.4, price_age_s=72000)
              for side, market, point in (("home", "h2h", None), ("away", "h2h", None), ("over", "totals", 220.5),
                                          ("under", "totals", 220.5))]  # 20 h old, one game
    result = build(board, Controls(min_chance=0.6), NOW)
    rules = [miss["rule"] for miss in result.near_misses]
    assert rules[:3] == ["min_chance"] * 3 and rules.count("max_price_age") == 2
    assert all("game_key" not in miss for miss in result.near_misses)
    # Every rule that ruled legs out shows its closest leg, however many closer legs another rule removed
    crowd = [cand(f"c{i}", chance=0.55, price=1.75) for i in range(20)]
    crowd.append(cand("d", chance=0.7, price=1.38, flags={"disagree"}))
    safe = build(crowd, PRESETS["safe"], NOW)
    assert len(safe.near_misses) == 12 and [m["rule"] for m in safe.near_misses].count("disagree") == 1
