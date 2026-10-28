"""Safe, Risky and Crazy as builder presets: each fills every control (engine/builder.py).

They start from the old fixed tiers of engine/bets.py (research_notes, sections 5.2-5.7):

    Preset  Joint chance  Ticket odds  Legs (Auto)  Leg chance: single / in a multi  Whelan cap
    Safe    60%+          1.15-1.80    1-3          60% / 72%                        as is
    Risky   25-60%        1.50-4.50    1-3          25% / 45%                        as is
    Crazy   3-25%         3.40-36      1-5          8% / 34%                         up to 2 legs past it

plus a cautious chance of the blend minus 2 points and moneylines, spreads and totals only. Safe also never
takes a leg the sources disagree on, a leg whose chance comes only from ratings (low confidence), a thin
market or a leg the goals model prices at fair odds. Crazy says "expect to lose most of these".

While a run is exactly a preset, the old tier's rules on a ticket's value apply too (VALUE_RULES, from
engine/bets.py's Tier): the expected-result floor at the brain's chances (Safe -5%, Risky -10%, Crazy -15%), the
cap on the books' compounded cut (9%, 14%, 22%), and for Safe every leg of a 3-leg ticket at 85%+ and no leg whose
chance is more than 8 points from the books'. So Safe on the Build screen refuses what the AI bets' Safe refuses.
A Custom run is the person's own rules: these don't apply, and the ticket still shows its expected result and cut.

Editing any control that shapes the tickets relabels the run "Custom" (``label_for``). The scope (leagues, time
and kick-off windows, started games), the allowed books, the number of tickets, the time zone, the
correlation ideas and the closest-effort switch (``best_effort``: strict or not changes no rule a ticket that
meets them is held to; a ticket closest effort loosened a rule for says so in its ``relaxed``) don't change the
risk, so they keep the preset's name. A target chance (``target_chance``), like a target odds, makes it Custom;
the presets set none and leave closest effort on.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

from .builder import DEFAULT_MARKETS, Controls, Weights

TITLES: Dict[str, str] = {"safe": "Safe", "risky": "Risky", "crazy": "Crazy"}
SUBTITLES: Dict[str, str] = {
    "safe": "More likely, small payout. It still loses some of the time.",
    "risky": "Balanced: a real chance and a real payout.",
    "crazy": "Long shot, big payout: expect to lose most of these.",
}

_COMMON = dict(markets=set(DEFAULT_MARKETS), caution_pp=2.0, whelan=True, tickets=3, avoid_same_game=True,
               max_per_game=1, max_per_team=1, max_price_age_s=3 * 3600)

PRESETS: Dict[str, Controls] = {
    "safe": Controls(preset="safe", joint_min=0.60, joint_max=None, odds_min=1.15, odds_max=1.80, legs_max=3,
                     min_chance=0.60, multi_leg_min=0.72, whelan_slack=0, exclude_disagree=True,
                     exclude_low_conf=True, exclude_model_price=True, exclude_thin=True, **_COMMON),
    "risky": Controls(preset="risky", joint_min=0.25, joint_max=0.60, odds_min=1.50, odds_max=4.50, legs_max=3,
                      min_chance=0.25, multi_leg_min=0.45, whelan_slack=0, **_COMMON),
    "crazy": Controls(preset="crazy", joint_min=0.03, joint_max=0.25, odds_min=3.40, odds_max=36.0, legs_max=5,
                      min_chance=0.08, multi_leg_min=0.34, whelan_slack=2, **_COMMON),
}

# How each preset ranks tickets: closeness to a target, the cautious expected return, the joint chance.
# Safe leans hard on the chance of winning, Crazy hardly at all (it wants the best return on a long shot).
WEIGHTS: Dict[Optional[str], Weights] = {
    "safe": Weights(target=0.15, ret=1.0, chance=0.6),
    "risky": Weights(target=0.15, ret=1.0, chance=0.25),
    "crazy": Weights(target=0.15, ret=1.0, chance=0.05),
    None: Weights(target=0.15, ret=1.0, chance=0.2),
}

# The fields that don't change the risk: changing them keeps the preset's name
SCOPE_FIELDS: Tuple[str, ...] = ("leagues", "window", "kickoff", "exclude_started", "books", "tickets", "preset",
                                 "tz", "prefer_correlation", "best_effort")
PRESET_FIELDS: Tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Controls) if f.name not in SCOPE_FIELDS)


def _value_rules() -> Dict[str, Dict[str, float]]:
    from .bets import SAFE_MAX_GAP, TIERS
    rules = {tier.key: {"min_ev": tier.ev_floor, "max_book_cut": tier.edge_cap} for tier in TIERS}
    rules["safe"].update(three_leg_min=SAFE_THREE_LEG_MIN, max_gap=SAFE_MAX_GAP)
    return rules


SAFE_THREE_LEG_MIN = 0.85  # engine/bets.py's Safe: 3 legs only if every leg is 85%+
VALUE_RULES: Dict[str, Dict[str, float]] = _value_rules()


def value_rules(label: str) -> Dict[str, float]:
    """The old tier's rules on a ticket's value for a run labelled ``label`` ("Safe", "Risky", "Crazy"); none for
    "Custom"."""
    key = next((key for key, title in TITLES.items() if title == label), None)
    return dict(VALUE_RULES.get(key) or {})


def preset(name: str, keep: Optional[Controls] = None) -> Controls:
    """A fresh copy of a preset's controls; with ``keep``, that run's scope, books and ticket count stay."""
    key = str(name or "").strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose safe, risky or crazy")
    chosen = dataclasses.replace(PRESETS[key], markets=set(PRESETS[key].markets))
    if keep is not None:
        chosen = dataclasses.replace(chosen, **{f: getattr(keep, f) for f in SCOPE_FIELDS if f != "preset"})
    return chosen


def label_for(controls: Controls) -> str:
    """"Safe", "Risky" or "Crazy" while every risk-shaping control equals that preset's, else "Custom"."""
    base = PRESETS.get(controls.preset or "")
    if base is None:
        return "Custom"
    if all(getattr(controls, name) == getattr(base, name) for name in PRESET_FIELDS):
        return TITLES[controls.preset]
    return "Custom"


def weights_for(controls: Controls) -> Weights:
    """The ranking weights of the preset these controls started from (Custom keeps its starting preset's)."""
    return WEIGHTS.get(controls.preset, WEIGHTS[None])


def describe() -> Dict[str, Any]:
    """The presets for GET /api/builder/presets: each one's title, subtitle, a one-line detail, its controls and
    its value rules (VALUE_RULES, with a line of words for them)."""
    out = {}
    for key, controls in PRESETS.items():
        band = (f"{controls.joint_min:.0%}+" if controls.joint_max is None
                else f"{controls.joint_min:.0%}-{controls.joint_max:.0%}")
        value = VALUE_RULES[key]
        words = (f"Expected result {value['min_ev']:+.0%} or better and a bookmaker's cut up to "
                 f"{value['max_book_cut']:.0%}")
        if "three_leg_min" in value:
            words += (f"; 3 legs only if each is {value['three_leg_min']:.0%}+, and no leg more than "
                      f"{value['max_gap'] * 100:.0f} points from the books")
        out[key] = {"title": TITLES[key], "subtitle": SUBTITLES[key],
                    "detail": (f"Chance of the whole ticket {band}, up to {controls.legs_max} legs, paying "
                               f"{controls.odds_min:.2f}x-{controls.odds_max:g}x."),
                    "controls": controls.to_dict(), "value_rules": dict(value), "value": words + "."}
    return out
