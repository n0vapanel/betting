"""Closest effort (engine/builder.py, engine/presets.py): the Builder always builds the closest tickets it can.

The owner asked Ask AI for "a 2 leg NBA ticket around 3.5 odds with 80% chance" and got a refusal. These tests hold
the engine to the contract in SPEC_V3_BESTEFFORT ("Contract 1"): the target chance and closest-effort controls, the
labelled options when both targets are given, the rounds that loosen the person's rules one step at a time (and the
rules never loosened), the per-ticket relaxed words, strict mode exactly as before, the likeliest winners
(top_picks) and the performance bound. No network: the engine fetches nothing."""

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
from telegram_betting_bot_simple.engine import builder as engine
from telegram_betting_bot_simple.engine.builder import (
    OPTION_LABELS,
    Candidate,
    Controls,
    build,
    restake,
    ticket_fit,
    top_picks,
)
from telegram_betting_bot_simple.engine.presets import PRESETS, SCOPE_FIELDS, describe, label_for, preset

NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.timezone.utc)
BANNED = ("lock", "guarantee", "sure", "risk-free", "can't lose", "win back")
SAFE_EXCLUDED = {"disagree", "thin", "low_conf", "model_price"}


def start(hours: float = 7) -> str:
    return (NOW + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def cand(game: str, side: str = "home", price=1.9, chance: float = 0.5, market: str = "h2h", point=None,
         league: str = "basketball_nba", book: str = "DraftKings", hours: float = 7, teams=None, **extra) -> Candidate:
    extra.setdefault("fair", chance)
    return Candidate(ref={"league": league, "id": game, "market": market, "side": side, "point": point},
                     league=league, game_key=game, teams=teams or (f"Away {game}", f"Home {game}"),
                     start=start(hours), price=Decimal(str(price)), book=book, chance=chance, **extra)


def slate(games: int, seed: int, margin: float = 0.045) -> list:
    """test_builder's random board (the Stage 3 performance world): moneylines at a 4.5% cut and a total."""
    rng = random.Random(seed)
    out = []
    for g in range(games):
        home = rng.uniform(0.12, 0.88)
        hours = 2 + (g % 30)
        for side, p in (("home", home), ("away", 1 - home)):
            price = round((1 - margin) / p, 2)
            if price > 1.01:
                out.append(cand(f"g{g}", side, price, p, hours=hours))
        over = rng.uniform(0.44, 0.56)
        for side, p in (("over", over), ("under", 1 - over)):
            out.append(cand(f"g{g}", side, round((1 - margin) / p, 2), p, market="totals", point=220.5, hours=hours))
    return out


# --- a realistic NBA night --------------------------------------------------------------------------------------
# Twelve games tonight, each with a moneyline, a spread set near 50/50 and a total, at three books with a cut near
# 4.5%; the brain's blend a point or two off the books' margin-free chance. One game the sources disagree on, one
# thin market (Safe never takes either).
# (id, away, home, home chance by the books, the blend's home chance, home spread, total, the blend's over chance,
#  flags)
NBA_NIGHT = [
    ("nba1", "Hornets", "Timberwolves", 0.83, 0.84, -12.5, 218.5, 0.49, set()),
    ("nba2", "Raptors", "Spurs", 0.76, 0.77, -7.5, 229.5, 0.52, set()),
    ("nba3", "Pelicans", "Grizzlies", 0.47, 0.49, 1.5, 233.5, 0.51, set()),
    ("nba4", "Wizards", "Celtics", 0.88, 0.87, -14.5, 224.5, 0.48, set()),
    ("nba5", "Jazz", "Nuggets", 0.79, 0.80, -9.5, 231.5, 0.53, set()),
    ("nba6", "Kings", "Suns", 0.58, 0.60, -2.5, 227.5, 0.50, set()),
    ("nba7", "Bulls", "Knicks", 0.68, 0.69, -5.5, 221.5, 0.47, set()),
    ("nba8", "Heat", "Bucks", 0.62, 0.63, -3.5, 219.5, 0.54, set()),
    ("nba9", "Magic", "Cavaliers", 0.66, 0.65, -4.5, 214.5, 0.50, set()),
    ("nba10", "Lakers", "Warriors", 0.55, 0.56, -1.5, 235.5, 0.52, {"disagree"}),
    ("nba11", "Pistons", "Thunder", 0.86, 0.86, -13.5, 226.5, 0.51, set()),
    ("nba12", "Trail Blazers", "Rockets", 0.71, 0.73, -6.5, 223.5, 0.49, {"thin"}),
]
NIGHT_BOOKS = {"DraftKings": 0.0, "FanDuel": 0.006, "BetMGM": -0.004}  # each book's cut a little off 4.5%


def nba_night() -> list:
    out = []
    for i, (gid, away, home, fair, blend, spread, total, over, flags) in enumerate(NBA_NIGHT):
        legs = [("h2h", "home", None, fair, blend, home), ("h2h", "away", None, 1 - fair, 1 - blend, away),
                ("spreads", "home", spread, 0.5, 0.5 + (blend - fair) * 0.8, f"{home} {spread:+g}"),
                ("spreads", "away", -spread, 0.5, 0.5 - (blend - fair) * 0.8, f"{away} {-spread:+g}"),
                ("totals", "over", total, 0.5, over, f"Over {total:g}"),
                ("totals", "under", total, 0.5, 1 - over, f"Under {total:g}")]
        for market, side, point, fair_p, chance, label in legs:
            prices = {book: Decimal(str(round((1 - 0.045 - extra) / fair_p, 2))) for book, extra in NIGHT_BOOKS.items()}
            book = max(prices, key=lambda name: (prices[name], name))
            out.append(Candidate(ref={"league": "basketball_nba", "id": gid, "market": market, "side": side,
                                      "point": point},
                                 league="basketball_nba", game_key=gid, teams=(away, home), start=start(7 + (i % 4) / 2),
                                 price=prices[book], book=book, prices=prices, chance=round(chance, 4),
                                 chance_source="blend", fair=fair_p, flags=set(flags), price_age_s=600, label=label,
                                 why=[f"{label}: Markets say {chance:.0%}"]))
    return out


TRANSCRIPT = {"preset": "safe", "leagues": ["basketball_nba"], "legs": 2, "target_odds": 3.5, "target_chance": 0.8}


def keys(ticket) -> frozenset:
    return frozenset(leg["key"] for leg in ticket["legs"])


def two_leg_tickets(board, floor: float = 0.0, flags=SAFE_EXCLUDED):
    """Every 2-leg ticket of the board's bettable legs (one leg a game, no team twice), each leg at its best price
    book's price and at least ``floor`` likely: (odds, chance, keys)."""
    legs = []
    for c in board:
        if c.flags & set(flags) or c.chance < floor - 1e-9:
            continue
        offers = dict(c.prices or {c.book: c.price})
        legs.append((c.key, c.game_key, float(max(offers.values())), c.chance, {t.lower() for t in c.teams}))
    out = []
    for a, b in itertools.combinations(legs, 2):
        if a[1] != b[1] and not a[4] & b[4]:
            out.append((a[2] * b[2], a[3] * b[3], frozenset((a[0], b[0]))))
    return out


def distance(odds: float, chance: float, target_odds=3.5, target_chance=0.8, tolerance=0.10) -> float:
    return abs(math.log(odds / target_odds)) / math.log(1 + tolerance) + abs(chance - target_chance) / 0.10


# --- the controls -------------------------------------------------------------------------------------------------

def test_the_new_controls_are_validated_saved_and_in_the_presets() -> None:
    controls = Controls.from_dict({"preset": "risky", "target_odds": 3.5, "target_chance": 0.8, "best_effort": False})
    assert controls.target_chance == 0.8 and controls.best_effort is False
    data = json.loads(json.dumps(controls.to_dict()))
    assert data["target_chance"] == 0.8 and data["best_effort"] is False
    assert Controls.from_dict(data) == controls and controls.replace(target_chance=0.7).cache_key() != controls.cache_key()
    defaults = Controls().to_dict()
    assert defaults["target_chance"] is None and defaults["best_effort"] is True
    for name, values in PRESETS.items():
        assert values.target_chance is None and values.best_effort is True, name
        assert describe()[name]["controls"]["target_chance"] is None
        assert describe()[name]["controls"]["best_effort"] is True
    for bad, words in (({"target_chance": 80}, "0.8 for 80%"), ({"target_chance": 0}, "target_chance"),
                       ({"target_chance": 1.0}, "target_chance"), ({"target_chance": True}, "target_chance"),
                       ({"target_chance": "most"}, "target_chance"), ({"best_effort": "yes"}, "true or false"),
                       ({"best_effort": None}, "best_effort")):
        with pytest.raises(ValueError, match=words):
            Controls.from_dict(bad)
    assert Controls(target_chance=0.01).target_chance == 0.01 and Controls(target_chance="0.99").target_chance == 0.99


def test_custom_detection_a_target_chance_is_custom_strict_keeps_the_name() -> None:
    safe = preset("safe")
    assert "best_effort" in SCOPE_FIELDS and "target_chance" not in SCOPE_FIELDS
    assert label_for(safe.replace(best_effort=False)) == "Safe"  # strict or not changes no rule a ticket is held to
    assert label_for(safe.replace(target_chance=0.6)) == "Custom"
    # Switching presets keeps the person's closest-effort switch, as it keeps their leagues
    kept = preset("crazy", keep=safe.replace(best_effort=False))
    assert kept.best_effort is False and label_for(kept) == "Crazy"


def test_ticket_fit_is_the_ticket_against_the_ask() -> None:
    fit = ticket_fit(Decimal("1.85") * Decimal("1.90"), 0.2652, 3.5, 0.8, 10)
    # 3.515 exactly: rounded half up as price_ticket shows it, never a float's 3.51
    assert fit == {"odds": 3.52, "chance": 0.2652, "target_odds": 3.5, "target_chance": 0.8, "odds_gap_pct": 0.4,
                   "chance_gap_pp": -53.5, "meets": {"odds": True, "chance": False}}
    assert ticket_fit(1.22, 0.75, None, None) is None
    only = ticket_fit(4.2, 0.2, target_odds=3.5, tolerance_pct=10)
    assert only["meets"] == {"odds": False, "chance": None} and only["chance_gap_pp"] is None
    assert only["odds_gap_pct"] == 20.0
    assert ticket_fit(1.5, 0.8, target_chance=0.8)["meets"] == {"odds": None, "chance": True}


# --- the transcript: 2 NBA legs, about 3.5x, an 80% chance, under Safe ------------------------------------------

def test_the_transcript_request_returns_three_labelled_options_with_the_engines_own_numbers() -> None:
    board = nba_night()
    result = build(board, Controls.from_dict(TRANSCRIPT), NOW)
    assert result.blocked is None and result.label == "Custom"
    assert [(o["key"], o["label"], o["ticket_index"]) for o in result.options] == [
        ("your_odds", "Likeliest at your odds", 0), ("your_chance", "Closest to your chance", 1),
        ("closest", "Best balance", 2)]
    assert len(result.tickets) == 3 and len({keys(t) for t in result.tickets}) == 3
    for ticket in result.tickets:
        fit = ticket["fit"]
        odds = math.prod(float(leg["price"]) for leg in ticket["legs"])
        chance = math.prod(leg["inputs"]["chance"] for leg in ticket["legs"])
        # The fit holds the ticket's own numbers: the odds price_ticket shows and the chance every leg wins
        assert fit["odds"] == ticket["odds"]["decimal"] and fit["chance"] == ticket["joint"]["chance"]
        assert fit["target_odds"] == 3.5 and fit["target_chance"] == 0.8
        assert fit["odds_gap_pct"] == round((odds / 3.5 - 1) * 100, 1)
        assert fit["chance_gap_pp"] == round((chance - 0.8) * 100, 1)
        assert fit["meets"]["chance"] is False  # nothing on this night is 80% for two legs
        assert len(ticket["legs"]) == 2 and all(leg["league"] == "basketball_nba" for leg in ticket["legs"])
        assert not any(set(leg["flags"]) & SAFE_EXCLUDED for leg in ticket["legs"])  # never loosened
    your_odds, your_chance, closest = result.tickets
    # Likeliest at your odds: inside 3.15-3.85, the highest chance, with Safe's per-leg floor loosened only as far
    # as it had to go (72% for a multi's legs, 5 points a round)
    assert your_odds["fit"]["meets"]["odds"] is True
    assert [e["words"] for e in your_odds["relaxed"]] == ["per-leg chance lowered from 72% to 52%"]
    assert your_odds["relaxed"][0]["rule"] == "multi_leg_min" and your_odds["relaxed"][0]["from"] == 0.72
    in_window = [t for t in two_leg_tickets(board, floor=0.52) if 3.15 - 1e-9 <= t[0] <= 3.85 + 1e-9]
    assert your_odds["joint"]["chance"] == pytest.approx(max(t[1] for t in in_window), abs=1e-4)
    # Closest to your chance: the likeliest two legs under Safe's own rules, nothing loosened
    assert your_chance["relaxed"] == []
    safe_pairs = two_leg_tickets(board, floor=0.72)
    assert your_chance["joint"]["chance"] == pytest.approx(max(t[1] for t in safe_pairs), abs=1e-4)
    assert your_chance["fit"]["chance_gap_pp"] > your_odds["fit"]["chance_gap_pp"]
    # Best balance: the smallest d of any ticket the rules never loosened allow (Safe's flags stay out), here
    # nearer 3.5x than the option nearest the odds, with the floor it needed named
    best = min(distance(o, c) for o, c, _ in two_leg_tickets(board))
    assert distance(closest["joint"]["odds"], closest["joint"]["chance"]) == pytest.approx(best, abs=1e-6)
    assert best < distance(your_odds["joint"]["odds"], your_odds["joint"]["chance"])
    assert abs(closest["fit"]["odds_gap_pct"]) < abs(your_odds["fit"]["odds_gap_pct"])
    assert [e["words"] for e in closest["relaxed"]] == ["per-leg chance lowered from 72% to 47%"]
    assert any("closest for its aim" in note for note in result.notes)
    # The rounds that got there are recorded; the run's own story (near misses, counts) stays the strict one's
    rounds = result.stats["relaxation"]
    assert [r["objective"] for r in rounds] == ["your_odds"] * 4 + ["closest"] and rounds[3]["found"] == 1
    # Best balance's search is recorded too: from where the options got to, every rule closest effort may loosen off
    # (so the step its ticket names, 47%, is one a recorded round took)
    closest_round = rounds[-1]
    assert closest_round["last"] is True and closest_round["searched"] and closest_round["found"] == 1
    assert [(c["rule"], c["from"], c["to"]) for c in closest_round["changes"]] == [("min_chance", 0.4, None),
                                                                                  ("multi_leg_min", 0.52, None)]
    json.dumps(result.to_dict())
    assert result.to_dict()["options"] == result.options


def test_strict_says_what_it_says_today_for_the_transcript() -> None:
    result = build(nba_night(), Controls.from_dict({**TRANSCRIPT, "best_effort": False}), NOW)
    assert result.tickets == [] and result.options == [] and "relaxation" not in result.stats
    assert result.blocked["message"].startswith("No 2-leg ticket near 3.5x at an 80%+ chance")


@pytest.mark.parametrize("target_chance", [0.6, 0.9])
def test_closest_to_your_chance_is_exact_on_a_big_board(target_chance) -> None:
    # 200 legs: more than a search pool holds, so the pool must be chosen for the chance, not the target's price
    board = slate(50, 99)[:200]
    result = build(board, Controls(legs=2, target_odds=12.0, target_chance=target_chance), NOW)
    option = next(o for o in result.options if o["key"] == "your_chance")
    ticket = result.tickets[option["ticket_index"]]
    pairs = two_leg_tickets(board)
    reaching = [t for t in pairs if t[1] >= target_chance]
    if reaching:  # the chance, at the highest odds
        assert ticket["fit"]["meets"]["chance"] is True
        assert ticket["joint"]["odds"] == pytest.approx(max(t[0] for t in reaching), abs=1e-6)
    else:  # the highest chance there is
        assert ticket["joint"]["chance"] == pytest.approx(max(t[1] for t in pairs), abs=1e-4)
    assert result.tickets[0]["fit"]["meets"]["odds"] is True  # likeliest at your odds: inside 10.8-13.2


def test_options_that_pick_the_same_ticket_are_merged_under_the_first_label() -> None:
    board = [cand("g1", price=1.85, chance=0.52), cand("g2", price=1.90, chance=0.51),
             cand("g3", price=1.10, chance=0.88), cand("g4", price=1.12, chance=0.86)]
    result = build(board, Controls(legs=2, target_odds=3.5, target_chance=0.8), NOW)
    # g1 + g2 is the only ticket in the odds window and also the smallest d: one ticket, labelled for the odds
    assert [(o["key"], o["ticket_index"]) for o in result.options] == [("your_odds", 0), ("your_chance", 1)]
    assert [keys(t) for t in result.tickets] == [frozenset({board[0].key, board[1].key}),
                                                 frozenset({board[2].key, board[3].key})]


def test_both_targets_met_is_a_normal_run_and_every_ticket_meets_both() -> None:
    result = build(nba_night(), Controls(leagues=["basketball_nba"], legs=2, target_odds=1.5, target_chance=0.6), NOW)
    assert result.options == [] and len(result.tickets) == 3
    assert all(t["fit"]["meets"] == {"odds": True, "chance": True} and t["relaxed"] == [] for t in result.tickets)


def test_both_targets_met_after_loosening_a_rule_is_still_a_normal_run() -> None:
    # 1.21 x 1.24 = 1.50 at 0.80 x 0.78 = 62%, but a minimum edge of +1 point leaves both legs out
    board = [cand("a", price=1.21, chance=0.80), cand("b", price=1.24, chance=0.78), cand("c", price=3.0, chance=0.34)]
    result = build(board, Controls(legs=2, target_odds=1.5, target_chance=0.6, min_edge=0.01, tickets=1), NOW)
    assert result.options == [] and keys(result.tickets[0]) == {board[0].key, board[1].key}
    assert result.tickets[0]["fit"]["meets"] == {"odds": True, "chance": True}
    assert [e["rule"] for e in result.tickets[0]["relaxed"]] == ["min_edge"]


# --- a target chance alone ----------------------------------------------------------------------------------------

def test_a_target_chance_alone_returns_the_highest_odds_at_or_above_it() -> None:
    board = nba_night()
    result = build(board, Controls(leagues=["basketball_nba"], legs=2, target_chance=0.5, tickets=1), NOW)
    ticket = result.tickets[0]
    reaching = [t for t in two_leg_tickets(board, flags=()) if t[1] >= 0.5]
    assert ticket["joint"]["chance"] >= 0.5 and ticket["fit"]["meets"]["chance"] is True
    assert ticket["joint"]["odds"] == pytest.approx(max(t[0] for t in reaching), abs=1e-6)
    assert ticket["fit"]["target_odds"] is None and ticket["fit"]["meets"]["odds"] is None
    # Auto: singles too; every ticket reaches the chance, the best odds first
    auto = build(board, Controls.from_dict({"preset": "risky", "target_chance": 0.6}), NOW)
    assert auto.label == "Custom" and len(auto.tickets) == 3
    assert all(t["joint"]["chance"] >= 0.6 for t in auto.tickets)
    assert any("band don't apply" in note for note in auto.notes)  # Risky's own 25-60% band is replaced


def test_a_target_chance_no_ticket_reaches_gives_the_highest_chance_there_is() -> None:
    board = nba_night()
    result = build(board, Controls(leagues=["basketball_nba"], legs=2, target_chance=0.95, tickets=1), NOW)
    ticket = result.tickets[0]
    assert ticket["joint"]["chance"] == pytest.approx(max(t[1] for t in two_leg_tickets(board, flags=())), abs=1e-4)
    assert ticket["fit"]["meets"]["chance"] is False and ticket["relaxed"] == []
    assert any("No ticket reaches a 95% chance" in note for note in result.notes)
    strict = build(board, Controls(leagues=["basketball_nba"], legs=2, target_chance=0.95, best_effort=False), NOW)
    assert strict.tickets == [] and "95%+ chance" in strict.blocked["message"]


def test_a_target_chance_loosens_the_rules_that_bind_before_settling_for_the_likeliest() -> None:
    # Leg odds of 1.30+ keep out the short favourites a 70% double needs: the rounds loosen the leg-odds band until a
    # ticket reaches the chance (Timberwolves + Thunder, 72%), rather than the likeliest at the rules (50%)
    board = nba_night()
    controls = Controls(leagues=["basketball_nba"], legs=2, target_chance=0.7, leg_odds_min=1.3, tickets=1)
    result = build(board, controls, NOW)
    ticket = result.tickets[0]
    assert ticket["fit"]["meets"]["chance"] is True and ticket["joint"]["chance"] == pytest.approx(0.7224, abs=1e-4)
    assert [leg["label"] for leg in ticket["legs"]] == ["Timberwolves", "Thunder"]
    assert [e["words"] for e in ticket["relaxed"]] == ["leg odds minimum lowered from 1.30 to 1.04"]
    assert not any("No ticket reaches" in note for note in result.notes)
    rounds = result.stats["relaxation"]
    assert [c["rule"] for r in rounds for c in r["changes"]] == ["leg_odds_min"] and rounds[-1]["found"] == 1
    # With both targets, "Closest to your chance" loosens the band the same way
    both = build(board, controls.replace(target_odds=3.5, tickets=3), NOW)
    option = next(o for o in both.options if o["key"] == "your_chance")
    assert keys(both.tickets[option["ticket_index"]]) == keys(ticket)
    assert both.tickets[option["ticket_index"]]["relaxed"] == ticket["relaxed"]
    # Out of reach even with the band loosened: the highest chance there is with it loosened (75%, the closest to
    # 95%, not 50% at 1.30+), the band named; no rule binds without it: the likeliest two at 1.30+ have nothing to say
    far = build(board, controls.replace(target_chance=0.95), NOW)
    assert far.tickets[0]["joint"]["chance"] == pytest.approx(max(t[1] for t in two_leg_tickets(board, flags=())),
                                                             abs=1e-4)
    assert [e["words"] for e in far.tickets[0]["relaxed"]] == ["leg odds minimum lowered from 1.30 to 1.04"]
    assert "No ticket reaches a 95% chance: these are the likeliest there are." in far.notes
    # A minimum edge that keeps the 90% favourites out: dropping it reaches 81% at 1.17x, not 30% at 3.61x
    favourites = [cand("fav1", price=1.08, chance=0.90), cand("fav2", price=1.08, chance=0.90),
                  cand("g3", price=1.90, chance=0.55), cand("g4", price=1.90, chance=0.55),
                  cand("g5", price=1.95, chance=0.54)]
    for asked in (Controls(legs=2, target_chance=0.8, min_edge=0.0, tickets=1),
                  Controls(legs=2, target_odds=3.5, target_chance=0.8, min_edge=0.0)):
        run = build(favourites, asked, NOW)
        index = next((o["ticket_index"] for o in run.options if o["key"] == "your_chance"), 0)
        assert keys(run.tickets[index]) == {favourites[0].key, favourites[1].key}
        assert run.tickets[index]["fit"]["meets"]["chance"] is True
        assert [e["words"] for e in run.tickets[index]["relaxed"]] == ["minimum edge (+0.0 pts) dropped"]


def test_every_option_keeps_the_number_of_legs_asked() -> None:
    # Safe, 12 legs at exactly 1000x and 50%: "Likeliest at your odds" spends the combination budget, and every option
    # still keeps 12 legs (12-leg tickets exist; the last round tries the asked number first)
    board = slate(50, 99)[:200]
    asked = {"preset": "safe", "legs": 12, "target_odds": 1000.0, "target_chance": 0.5, "tolerance_pct": 0}
    result = build(board, Controls.from_dict(asked), NOW)
    assert [o["key"] for o in result.options] == ["your_odds", "your_chance", "closest"]
    assert all(len(t["legs"]) == 12 for t in result.tickets)
    assert not any(e["rule"] == "legs" for t in result.tickets for e in t["relaxed"])
    # Closest to your chance: no 12 legs reach 50%, so the likeliest 12 legs under Safe's own rules, as the chance
    # alone gives
    your_chance = result.tickets[result.options[1]["ticket_index"]]
    alone = build(board, Controls.from_dict({**asked, "target_odds": None, "tickets": 1}), NOW).tickets[0]
    assert your_chance["relaxed"] == [] and keys(your_chance) == keys(alone)
    assert your_chance["joint"]["chance"] == pytest.approx(0.1124, abs=1e-4)
    assert abs(result.tickets[0]["fit"]["odds_gap_pct"]) < 1  # likeliest at your odds: near 1000x at 12 legs


# --- the rounds ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("board, controls, steps, words", [
    # 1. the per-leg chance floors, 5 points a round
    ([cand(f"g{i}", chance=0.6, price=1.55) for i in range(3)], Controls(min_chance=0.7, legs=1),
     [("min_chance", 0.7, 0.65), ("min_chance", 0.65, 0.6)], ["per-leg chance lowered from 70% to 60%"]),
    # 2. the leg-odds band, 25% wider a round, twice, then dropped
    ([cand(f"g{i}", chance=0.62, price=1.5) for i in range(3)], Controls(leg_odds_min=2.0, legs=1),
     [("leg_odds_min", 2.0, 1.6), ("leg_odds_min", 1.6, 1.28)], ["leg odds minimum lowered from 2.00 to 1.28"]),
    ([cand(f"g{i}", chance=0.36, price=2.5) for i in range(3)], Controls(leg_odds_max=1.5, legs=1),
     [("leg_odds_max", 1.5, 1.875), ("leg_odds_max", 1.875, 2.3438), ("leg_odds_max", 2.3438, None)],
     ["leg odds maximum (1.50) dropped"]),
    # 3. the minimum edge
    ([cand(f"g{i}", chance=0.5, price=1.9) for i in range(3)], Controls(min_edge=0.05, legs=1),
     [("min_edge", 0.05, None)], ["minimum edge (+5.0 pts) dropped"]),
    # 4. the joint chance band (no target)
    ([cand(f"g{i}", chance=0.6, price=1.6) for i in range(3)], Controls(joint_min=0.9, legs=1),
     [("joint_min", 0.9, None)], ["joint chance floor (90%+) dropped"]),
    # 5. the value rules: the Whelan cap on Auto
    (slate(30, 6), Controls(legs_max=6, target_odds=40.0), [("whelan", True, False)], ["the Whelan leg cap lifted"]),
    # 6. the odds window: 10, 20, 35, 60%
    ([cand(f"g{i}", chance=0.5, price=1.9) for i in range(4)], Controls(legs=2, target_odds=2.7),
     [("tolerance_pct", 10.0, 20.0), ("tolerance_pct", 20.0, 35.0)], ["odds window widened to ±35%"]),
    # 7. an exact number of legs, last: 1 then 2 either way
    ([cand(f"g{i}", chance=0.5, price=1.9) for i in range(2)], Controls(legs=4),
     [("legs", 4, [3, 5]), ("legs", [3, 5], [2, 6])], ["legs changed from 4 to 2"]),
])
def test_every_relaxation_step_is_recorded(board, controls, steps, words) -> None:
    result = build(board, controls, NOW)
    rounds = result.stats["relaxation"]
    assert [(c["rule"], c["from"], c["to"]) for r in rounds for c in r["changes"]] == steps
    assert [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert all(r["found"] == 0 for r in rounds[:-1]) and rounds[-1]["found"] >= 1  # the first round that finds
    assert all(c["words"] for r in rounds for c in r["changes"])
    assert result.tickets and all([e["words"] for e in t["relaxed"]] == words for t in result.tickets)
    assert result.blocked is None and any("relaxed" in note for note in result.notes)
    strict = build(board, controls.replace(best_effort=False), NOW)
    assert not strict.tickets and strict.blocked and "relaxation" not in strict.stats


def test_the_rounds_follow_the_order_and_loosen_only_what_binds() -> None:
    # Safe's favourites at 82% priced 1.12: Safe's value rules rule them out (the old tier lays off). Closest effort
    # loosens the rule that binds; the run keeps Safe's name, and each ticket says what was loosened for it
    favourites = [cand(f"h{i}", chance=0.82, price=1.12) for i in range(4)]
    result = build(favourites, PRESETS["safe"], NOW)
    assert result.label == "Safe" and result.tickets
    order = [c["rule"] for r in result.stats["relaxation"] for c in r["changes"]]
    assert order == sorted(order, key=lambda rule: engine.RELAX_ORDER.index(
        {"joint_min": "joint", "three_leg": "value", "min_ev": "value", "book_cut": "value"}[rule]))
    for ticket in result.tickets:
        assert ticket["label"] == "Safe" and "Safe's expected-result floor (-5%) dropped" in [
            e["words"] for e in ticket["relaxed"]]
    # Per-ticket words name only what that ticket needed: Safe with leg odds of 3.00+ loosens the leg-odds band;
    # the singles it then finds are 82% likely, so the per-leg floor isn't named even though a round lowered it
    board = slate(50, 99)
    wide = build(board, PRESETS["safe"].replace(leg_odds_min=3.0, tickets=5), NOW)
    assert len(wide.tickets) == 5
    assert all([e["rule"] for e in t["relaxed"]] == ["leg_odds_min"] for t in wide.tickets)


def test_a_rule_that_rules_out_combinations_in_the_window_binds_before_more_legs() -> None:
    # Risky, about 40x, legs 60%+ on Auto: once the floor lets long enough legs in, the Whelan cap rules out every
    # 3-leg combination landing at 30-50x. That round's counts make the Whelan cap the binding rule: the floor
    # isn't lowered any further, and the ticket names both
    result = build(slate(50, 99)[:200], Controls.from_dict({"preset": "risky", "target_odds": 40.0, "min_chance": 0.6,
                                                            "tolerance_pct": 25, "tickets": 1}), NOW)
    rounds = result.stats["relaxation"]
    searched = [r for r in rounds if r["searched"]]
    assert searched[0]["found"] == 0 and [c["rule"] for c in rounds[-1]["changes"]] == ["whelan"]
    assert len(searched) == 2 and rounds[-1]["found"] == 1
    assert [e["words"] for e in result.tickets[0]["relaxed"]] == ["per-leg chance lowered from 60% to 25%",
                                                                  "the Whelan leg cap lifted"]


def test_the_last_round_takes_the_nearest_odds_there_are() -> None:
    board = [cand(f"g{i}", chance=0.8, price=1.2) for i in range(3)]
    # A single at 1000x: no round lands, so the last round drops the window and keeps the number of legs asked
    result = build(board, Controls(legs=1, target_odds=1000.0), NOW)
    last = result.stats["relaxation"][-1]
    assert last["last"] is True and last["found"] == 3
    assert [(c["rule"], c["from"], c["to"]) for c in last["changes"]] == [("tolerance_pct", 60.0, None),
                                                                         ("legs", 3, 1)]
    for ticket in result.tickets:
        assert len(ticket["legs"]) == 1 and ticket["fit"]["meets"]["odds"] is False
        assert [e["words"] for e in ticket["relaxed"]] == ["odds window dropped: the nearest odds there are"]
    # Four legs from three games: the asked number can't be met, so the last round tries 1 either way, then 2
    result = build(board, Controls(legs=4, target_odds=1000.0), NOW)
    last_rounds = [r for r in result.stats["relaxation"] if r.get("last")]
    assert [(r["searched"], r["found"]) for r in last_rounds] == [(False, 0), (True, 1)]
    assert [c["to"] for c in last_rounds[0]["changes"]] == [None, 4] and last_rounds[1]["changes"][0]["to"] == [3, 5]
    ticket = result.tickets[0]
    assert len(ticket["legs"]) == 3 and ticket["odds"]["decimal"] == pytest.approx(1.73)
    assert [e["words"] for e in ticket["relaxed"]] == ["odds window dropped: the nearest odds there are",
                                                       "legs changed from 4 to 3"]


def test_a_partly_filled_run_is_topped_up_after_the_strict_tickets() -> None:
    board = [cand("a", chance=0.7, price=1.36), cand("b", chance=0.66, price=1.45), cand("c", chance=0.61, price=1.57)]
    controls = Controls(min_chance=0.65, legs=1, tickets=3)
    strict = build(board, controls.replace(best_effort=False), NOW)
    result = build(board, controls, NOW)
    assert [keys(t) for t in result.tickets[:2]] == [keys(t) for t in strict.tickets]
    assert all(t["relaxed"] == [] for t in result.tickets[:2])
    assert keys(result.tickets[2]) == {board[2].key}
    assert [e["words"] for e in result.tickets[2]["relaxed"]] == ["per-leg chance lowered from 65% to 60%"]
    assert [t["rank"] for t in result.tickets] == [1, 2, 3]
    # The near misses and the counts are the person's own rules': the leg that was let in is still listed
    assert result.near_misses == strict.near_misses and result.stats["removed"] == strict.stats["removed"]


# --- what is never loosened -----------------------------------------------------------------------------------------

def test_the_never_relax_rules_hold() -> None:
    good = [cand("ok", chance=0.45, price=2.1)]
    never = [
        cand("started", chance=0.9, price=1.05, hours=-1),  # under way
        cand("live", chance=0.9, price=1.05, flags={"live"}),
        cand("stale", chance=0.9, price=1.05, price_age_s=20000),  # older than the price-age limit
        cand("sharp", chance=0.9, price=1.05, book="Pinnacle"),  # never a sharp book
        cand("exchange", chance=0.9, price=1.05, book="Betfair Exchange"),
        cand("other", chance=0.9, price=1.05, league="soccer_epl"),  # out of the leagues chosen
        cand("total", "over", chance=0.9, price=1.05, market="totals", point=220.5),  # a market not chosen
        cand("ok", "over", chance=0.9, price=1.05, market="spreads", point=-1.5),  # the same game again
        cand("late", chance=0.9, price=1.05, hours=60),  # outside the time window
        cand("odd", chance=0.9, price=1.05, flags={"disagree"}),
        cand("thin", chance=0.9, price=1.05, flags={"thin"}),
    ]
    controls = Controls(leagues=["basketball_nba"], markets={"h2h", "spreads"},
                        window={"from": start(0), "to": start(24)}, exclude_disagree=True, exclude_thin=True,
                        legs=3, target_odds=1.2, target_chance=0.9, min_chance=0.6, tickets=3)
    result = build(good + never, controls, NOW)
    used = {leg["event_id"] for t in result.tickets for leg in t["legs"]}
    assert used == {"ok"} and all(len(t["legs"]) == 1 for t in result.tickets)  # one leg a game
    # Only never-loosened rules stand in the way: nothing at all is built
    nothing = build(never[:5] + never[8:], controls, NOW)
    assert nothing.tickets == [] and nothing.blocked and nothing.options == []
    # The team cap is the person's own too: the Celtics play twice, never on one ticket
    twice = [cand("g1", teams=("Lakers", "Celtics"), chance=0.62, price=1.6),
             cand("g2", teams=("Celtics", "Knicks"), chance=0.62, price=1.6, hours=30)]
    capped = build(twice, Controls(legs=2, tickets=1), NOW)
    assert all(len(t["legs"]) == 1 for t in capped.tickets) and capped.tickets


@pytest.mark.parametrize("seed", range(6))
def test_a_run_never_ends_empty_while_an_eligible_leg_exists(seed) -> None:
    rng = random.Random(seed)
    board = slate(rng.randint(1, 12), seed)
    for leg in board:
        if rng.random() < 0.2:
            leg.flags.add(rng.choice(["disagree", "thin", "live"]))
        if rng.random() < 0.1:
            leg.price_age_s = 20000
    for _ in range(25):
        data = {"legs": rng.choice([None, 1, 2, 3, 5, 9, 12]), "tickets": rng.randint(1, 5),
                "target_odds": rng.choice([None, 1.3, 3.5, 40.0, 5000.0]),
                "target_chance": rng.choice([None, None, 0.3, 0.8, 0.97]),
                "min_chance": rng.choice([None, 0.5, 0.95]), "min_edge": rng.choice([None, 0.0, 0.05]),
                "leg_odds_min": rng.choice([None, 2.5]), "exclude_disagree": True, "exclude_thin": rng.random() < 0.5}
        controls = Controls.from_dict({"preset": rng.choice(["safe", "risky", "crazy", None]), **data})
        result = build(board, controls, NOW)
        eligible = build(board, controls.replace(
            min_chance=None, multi_leg_min=None, leg_odds_min=None, leg_odds_max=None, min_edge=None, joint_min=None,
            joint_max=None, odds_min=None, odds_max=None, target_odds=None, target_chance=None, legs=1, legs_max=1,
            best_effort=False, preset=None), NOW).tickets
        assert bool(result.tickets) == bool(eligible), data
        for ticket in result.tickets:
            assert not {"live", "disagree"} & {f for leg in ticket["legs"] for f in leg["flags"]}
            assert all(leg["price_age_s"] <= controls.max_price_age_s for leg in ticket["legs"])
            assert len({leg["event_id"] for leg in ticket["legs"]}) == len(ticket["legs"])


# --- strict mode ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_strict_mode_is_unchanged_and_closest_effort_only_adds(seed) -> None:
    board = slate(25, seed)
    for data in ({"preset": "safe"}, {"preset": "risky"}, {"preset": "crazy", "legs": 5},
                 {"legs": 3, "target_odds": 6.0}, {"preset": "safe", "legs": 4}, {"min_chance": 0.8, "legs": 2},
                 {"legs": 6, "target_odds": 40.0, "tolerance_pct": 5}):
        strict = build(board, Controls.from_dict({**data, "best_effort": False}), NOW).to_dict()
        effort = build(board, Controls.from_dict(data), NOW).to_dict()
        assert strict["options"] == [] and "relaxation" not in strict["stats"]
        assert all(t["relaxed"] == [] for t in strict["tickets"])
        n = len(strict["tickets"])
        # The strict tickets come first, unchanged; closest effort only adds after them (or nothing)
        for got, want in zip(effort["tickets"][:n], strict["tickets"]):
            assert keys(got) == keys(want) and got["odds"] == want["odds"] and got["relaxed"] == []
        if n == strict["controls"]["tickets"]:
            strict["stats"].pop("ms"), effort["stats"].pop("ms")
            strict["controls"].pop("best_effort"), effort["controls"].pop("best_effort")
            assert strict == effort  # nothing to add: the very same result
        else:
            assert effort["blocked"] is None or not effort["tickets"]
            assert strict["near_misses"] == effort["near_misses"] and strict["stats"]["removed"] == effort["stats"][
                "removed"]


def test_strict_tickets_carry_the_new_fields_empty() -> None:
    result = build(slate(20, 4), Controls(legs=2, target_odds=4.0, best_effort=False), NOW)
    for ticket in result.tickets:
        assert ticket["relaxed"] == [] and ticket["fit"]["target_odds"] == 4.0 and ticket["fit"]["meets"]["odds"]
    plain = build(slate(20, 4), Controls(legs=2, best_effort=False), NOW)
    assert all(t["fit"] is None and t["relaxed"] == [] for t in plain.tickets) and plain.options == []


# --- the likeliest winners ------------------------------------------------------------------------------------------

def test_top_picks_ranks_by_chance_one_a_game_at_a_bettable_price() -> None:
    board = nba_night()
    picks = top_picks(board, n=5)
    assert len(picks) == 5 and len({p["game"] for p in picks}) == 5
    chances = [p["chance"] for p in picks]
    assert chances == sorted(chances, reverse=True)
    moneylines = sorted((c for c in board if c.market == "h2h"), key=lambda c: -c.chance)
    best_per_game = []
    for c in moneylines:
        if c.game_key not in {b.game_key for b in best_per_game}:
            best_per_game.append(c)
    assert [p["key"] for p in picks] == [c.key for c in best_per_game[:5]]
    first = picks[0]
    assert set(first) >= {"ref", "label", "game", "start", "chance", "chance_source", "price", "book", "edge", "why"}
    # 1.09 at DraftKings and BetMGM: the leg's own book wins a tie, as in the Builder
    assert first["label"] == "Celtics" and first["price"] == "1.09" and first["book"] == "DraftKings"
    assert first["edge"] == pytest.approx(0.87 - 1 / 1.09, abs=1e-4) and first["why"]
    json.dumps(picks)
    assert len(top_picks(board)) == 10 and len(top_picks(board, n=20)) == 12  # one a game
    assert all(p["ref"]["market"] == "totals" for p in top_picks(board, markets=("totals",)))
    assert all(float(p["price"]) >= 1.5 for p in top_picks(board, min_price=1.5))


def test_top_picks_breaks_ties_by_edge_and_never_takes_an_unsafe_price() -> None:
    tie = [cand("a", chance=0.7, price=1.38), cand("b", chance=0.7, price=1.42), cand("c", chance=0.7, price=1.40)]
    assert [p["ref"]["id"] for p in top_picks(tie)] == ["b", "c", "a"]
    unsafe = [cand("live", chance=0.95, price=1.05, flags={"live"}),
              cand("fair", chance=0.94, price=1.06, flags={"model_price"}, book="Fair odds"),
              cand("sharp", chance=0.93, price=1.07, book="Pinnacle"),
              cand("started", chance=0.92, price=1.08, hours=-2),
              cand("stale", chance=0.91, price=1.09, price_age_s=20000),
              cand("ok", chance=0.6, price=1.6, prices={"Pinnacle": Decimal("1.7"), "FanDuel": Decimal("1.62")})]
    picks = top_picks(unsafe, now=NOW, max_price_age_s=3 * 3600)
    assert [p["ref"]["id"] for p in picks] == ["ok"] and picks[0]["price"] == "1.62" and picks[0]["book"] == "FanDuel"
    # Without the time and age the caller filters those itself (dashboard/lab.py does)
    assert [p["ref"]["id"] for p in top_picks(unsafe)] == ["started", "stale", "ok"]


# --- performance, shapes, words -------------------------------------------------------------------------------------

def test_the_performance_bound_holds_with_relaxation() -> None:
    board = slate(50, 99)[:200]  # the Stage 3 performance world: 200 candidates
    assert len(board) == 200
    for data in ({"preset": "safe", "legs": 12, "target_odds": 40.0},
                 {"preset": "safe", "legs": 12},
                 {"preset": "safe", "legs": 12, "target_odds": 40.0, "target_chance": 0.5},
                 {"preset": "safe", "legs": 4, "target_odds": 25.0, "target_chance": 0.6},
                 {"legs": 12, "target_odds": 1_000_000.0},
                 {"preset": "risky", "legs": 8, "tickets": 5},
                 {"legs": 7, "target_odds": 200.0, "target_chance": 0.3, "tolerance_pct": 0},
                 # The heaviest found: five tickets of up to 12 legs at exactly 1,000,000x with long legs only. The
                 # rounds stop before they would go past the combination budget, and once it is spent the last
                 # round's top-ups are narrow searches
                 {"tickets": 5, "legs_max": 12, "target_odds": 1_000_000.0, "tolerance_pct": 0, "leg_odds_min": 3.0},
                 {"preset": "crazy", "tickets": 5, "legs_max": 12, "target_odds": 1_000_000.0, "tolerance_pct": 0,
                  "leg_odds_min": 3.0},
                 {"tickets": 5, "legs_max": 12, "target_odds": 1_000_000.0, "tolerance_pct": 0, "min_edge": 0.02},
                 {"preset": "safe", "legs": 12, "target_odds": 1000.0, "target_chance": 0.5, "tolerance_pct": 0}):
        began = time.perf_counter()
        result = build(board, Controls.from_dict(data), NOW)
        took = time.perf_counter() - began
        assert result.tickets and (result.stats.get("relaxation") or result.options), data
        assert took < 2.5, f"{data}: {took:.2f}s"
        if data.get("legs_max") == 12:
            assert len(result.tickets) == 5, data  # the budget bounds the work, not the tickets asked for


def test_a_loosened_tickets_swaps_never_break_a_rule_further_than_its_words() -> None:
    # Best balance searches with every floor off, so its swaps once could go below the floor the ticket's words state
    # (here a 16.8% leg under "per-leg chance lowered from 75% to 20%")
    rng = random.Random(48)
    board = slate(rng.randint(4, 30), 48)
    controls = Controls.from_dict({"preset": "crazy", "legs": 2, "target_odds": 20.0, "target_chance": 0.5,
                                   "min_chance": 0.75})
    result = build(board, controls, NOW)
    closest = result.tickets[result.options[-1]["ticket_index"]]
    assert result.options[-1]["key"] == "closest"
    assert [e["words"] for e in closest["relaxed"]] == ["per-leg chance lowered from 75% to 20%"]
    swaps = [s for leg in closest["legs"] for s in leg["swaps"]]
    assert swaps and all(s["chance"] >= 0.20 for s in swaps)
    # Across many runs: no swap takes a leg below the per-leg floor, or under the leg-odds minimum, its ticket names
    checked = 0
    for seed in range(60):
        rng = random.Random(seed)
        board = slate(rng.randint(4, 30), seed)
        data = {"preset": rng.choice(["safe", "risky", "crazy", None]), "legs": rng.choice([1, 2, 3, 4]),
                "target_odds": rng.choice([None, 2.0, 3.5, 8.0, 20.0]),
                "target_chance": rng.choice([None, None, 0.5, 0.8]), "min_chance": rng.choice([None, 0.6, 0.75]),
                "leg_odds_min": rng.choice([None, None, 1.6])}
        controls = Controls.from_dict({k: v for k, v in data.items() if v is not None})
        for ticket in build(board, controls, NOW).tickets:
            n = len(ticket["legs"])
            floor = max(controls.min_chance or 0.0, (controls.multi_leg_min or 0.0) if n >= 2 else 0.0)
            named = {e["rule"]: e["to"] for e in ticket["relaxed"]}
            floor_to = next((named[r] or 0.0 for r in ("min_chance", "multi_leg_min") if r in named), floor)
            odds_to = named.get("leg_odds_min", controls.leg_odds_min) if controls.leg_odds_min else None
            for i, leg in enumerate(ticket["legs"]):
                others = [other["chance"] for j, other in enumerate(ticket["legs"]) if j != i]
                for swap in leg["swaps"]:
                    checked += 1
                    assert min(others + [swap["chance"]]) >= floor_to - 1e-9, (seed, data, swap)
                    if odds_to:
                        assert float(swap["price"]) >= odds_to - 1e-9, (seed, data, swap)
    assert checked > 300


def test_results_with_options_are_json_ready_deterministic_and_restake() -> None:
    board = nba_night()
    first = build(board, Controls.from_dict(TRANSCRIPT), NOW).to_dict()
    again = build(list(reversed(board)), Controls.from_dict(TRANSCRIPT), NOW).to_dict()
    json.dumps(first)
    assert [keys(t) for t in first["tickets"]] == [keys(t) for t in again["tickets"]]
    assert first["options"] == again["options"]
    mine = restake(json.loads(json.dumps(first)), "25", now=NOW)
    assert mine["options"] == first["options"] and mine["stake"] == "25.00"
    for got, want in zip(mine["tickets"], first["tickets"]):
        assert got["fit"] == want["fit"] and got["relaxed"] == want["relaxed"]


def test_closest_effort_words_never_use_banned_claims() -> None:
    board = nba_night() + slate(15, 8)
    runs = [build(board, Controls.from_dict(TRANSCRIPT), NOW), build(board, PRESETS["safe"].replace(legs=6), NOW),
            build(board, Controls(legs=12, target_odds=5000.0, min_edge=0.03, leg_odds_max=2.0), NOW),
            build(board, Controls(target_chance=0.99, legs=4), NOW),
            build(board, Controls.from_dict({"preset": "crazy", "legs": 2, "joint_min": 0.2}), NOW),
            build([cand(f"h{i}", chance=0.82, price=1.12) for i in range(4)], PRESETS["safe"], NOW)]
    words = []
    for result in runs:
        words += list(result.notes) + [o["label"] for o in result.options]
        words += [e["words"] for t in result.tickets for e in t["relaxed"]]
        words += [c["words"] for r in result.stats.get("relaxation", []) for c in r["changes"]]
    assert len(words) > 20
    for text in words:
        assert not banned_phrases(text), text
        assert not any(word in text.lower() for word in BANNED), text
    assert set(OPTION_LABELS) == set(engine.OPTION_KEYS)


def test_relaxed_words_name_only_rules_the_ticket_breaks_at_the_tightest_step() -> None:
    board = nba_night() + slate(20, 11)
    two_games = [cand(f"g{i}", chance=0.5, price=1.9) for i in range(2)]
    cases = [(board, Controls.from_dict(TRANSCRIPT)), (board, PRESETS["safe"].replace(legs=5, target_odds=8.0)),
             (board, Controls(legs=3, target_odds=60.0, min_chance=0.55)), (board, PRESETS["risky"].replace(legs=9)),
             (board, Controls(min_chance=0.9, leg_odds_min=1.8, legs=2)),
             (board, Controls(leg_odds_min=1.6, legs=2, target_odds=2.4)),
             (board, Controls(legs=2, target_odds=1000.0)), (two_games, Controls(legs=4, min_chance=0.6))]
    seen = set()
    for board, controls in cases:
        for ticket in build(board, controls, NOW).tickets:
            chances = [leg["inputs"]["chance"] for leg in ticket["legs"]]
            prices = [float(leg["price"]) for leg in ticket["legs"]]
            odds = math.prod(prices)
            for entry in ticket["relaxed"]:
                rule, old, new = entry["rule"], entry["from"], entry["to"]
                seen.add(rule)
                if rule in ("min_chance", "multi_leg_min"):
                    assert min(chances) < old and (new is None or new <= min(chances) < new + 0.05 + 1e-9)
                elif rule == "leg_odds_min":
                    assert min(prices) < old and (new is None or new <= min(prices))
                elif rule == "tolerance_pct":
                    target = controls.target_odds
                    assert not target * (1 - old / 100) <= odds <= target * (1 + old / 100)
                    assert new is None or target * (1 - new / 100) - 1e-6 <= odds <= target * (1 + new / 100) + 1e-6
                elif rule == "legs":
                    assert old == controls.legs and new == len(ticket["legs"]) != old
    assert {"multi_leg_min", "min_chance", "leg_odds_min", "tolerance_pct", "legs"} <= seen
