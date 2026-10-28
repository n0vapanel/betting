"""Help's "Rollover explained" numbers are worked out by engine/rollover.py, so they can't drift from it.

The page's tables live in dashboard/static/js/views/help/content.js (STRATEGIES, ROLLOVER_TABLE, SURVIVAL)
and help.js says what each cell means. This reads both files as text and works every number out again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from telegram_betting_bot_simple.engine import rollover

HELP = Path(__file__).resolve().parent.parent / "telegram_betting_bot_simple" / "dashboard" / "static" / "js" / "views"
CONTENT = (HELP / "help" / "content.js").read_text(encoding="utf-8")
PAGE = (HELP / "help.js").read_text(encoding="utf-8")
EPS = 1e-9  # the planner's own slack for "in profit" and "at half or less" (rollover._EPS)
STRING = r'"(?:[^"\\]|\\.)*"'


def block(name: str) -> str:
    """The text of ``export const NAME = ...;`` in content.js."""
    match = re.search(rf"^export const {name} = (.*?);$", CONTENT, re.M | re.S)
    assert match, f"{name} is missing from content.js"
    return match.group(1)


def strategies() -> List[Dict[str, Any]]:
    entry = re.compile(rf"\{{\s*title:\s*({STRING}),\s*market:\s*(\[[^\]]*\]),\s*edge:\s*(\[[^\]]*\])\s*\}}")
    rows = [{"title": json.loads(title), "market": json.loads(market), "edge": json.loads(edge)}
            for title, market, edge in entry.findall(block("STRATEGIES"))]
    assert len(rows) == block("STRATEGIES").count("title:"), "an entry in STRATEGIES didn't parse"
    return rows


def which(title: str) -> str:
    """The engine's strategy a row's title describes."""
    lowered = title.lower()
    for words, key in (("classic", "classic"), ("bank half", "bank_half"), ("ladder", "ladder"), ("flat", "flat")):
        if words in lowered:
            return key
    raise AssertionError(f"No strategy for {title!r}")


def the_page_says() -> Dict[str, float]:
    """The run the strategies are worked out for, and the two daily chances, as help.js states them."""
    run = re.search(r"(\d+)-day runs, one bet a day at (\d+\.\d+)", PAGE)
    market = re.search(r"At market odds \((\d+(?:\.\d+)?)% a day\)", PAGE)
    edge = re.search(r"With a real edge \((\d+(?:\.\d+)?)%\)", PAGE)
    assert run and market and edge, "help.js no longer says which run and chances the table is for"
    return {"days": int(run.group(1)), "odds": float(run.group(2)), "market": float(market.group(1)) / 100,
            "edge": float(edge.group(1)) / 100}


def check_share(cell: str, share: float) -> None:
    """A chance written as "22.1%", or "≈0%" / "≈100%" for one that only rounds to it."""
    if cell == "≈0%":
        assert 0 < share < 0.0005, (cell, share)
    elif cell == "≈100%":
        assert 0.9995 < share < 1, (cell, share)
    else:
        assert cell == f"{share * 100:.1f}%", (cell, share)


def check_end(cell: str, value: float) -> None:
    """A typical end as a multiple of the start: "0" for nothing left, else two decimals ("0.40")."""
    if cell == "0":
        assert abs(value) < EPS, (cell, value)
    else:
        assert cell == f"{value:.2f}", (cell, value)


def test_the_strategy_table_is_the_engines() -> None:
    said = the_page_says()
    assert (said["days"], said["odds"]) == (30, 2.0)
    rows = strategies()
    assert sorted(which(row["title"]) for row in rows) == sorted(rollover.STRATEGIES)  # every strategy, once
    for row in rows:
        for column in ("market", "edge"):
            result = rollover.outcomes(which(row["title"]), said["days"], said["odds"], said[column])
            profit, typical, half = row[column]
            # help.js: "{profit} chance of a profit · typical end {typical} · {half} end at half or less"
            check_share(profit, result.share(lambda value: value > 1 + EPS))
            check_end(typical, result.median)
            check_share(half, result.share(lambda value: value <= 0.5 + EPS))


def test_the_strategy_titles_state_the_engines_rules() -> None:
    titles = {which(row["title"]): row["title"] for row in strategies()}
    assert f"pot of {rollover.POT_SHARE:.0%} of the start" in titles["ladder"]
    assert f"at most {rollover.POT_WINS} wins" in titles["ladder"]
    assert rollover.LADDER_STOP == 0.5 and "below half the start" in titles["ladder"]
    assert f"Flat {rollover.FLAT_SHARE:.0%} of the start" in titles["flat"]
    assert rollover.BANK_SHARE == 0.5 and "bank half of each day's profit" in titles["bank_half"]
    assert "pots are 10% of the start" in CONTENT and rollover.POT_SHARE == 0.10


def _one_in(cell: str) -> int:
    match = re.fullmatch(r"1 in ([\d,]+)", cell)
    assert match, cell
    return int(match.group(1).replace(",", ""))


def _percent(cell: str) -> float:
    match = re.fullmatch(r"([+−-])(\d+(?:\.\d+)?)%", cell)
    assert match, cell
    return (-1 if match.group(1) in "−-" else 1) * float(match.group(2))


def test_the_classic_rollover_table_is_the_engines() -> None:
    table = block("ROLLOVER_TABLE")
    columns = [(int(days), multiple) for days, multiple in re.findall(r'\["(\d+) days", "([\d,]+)×"\]', table)]
    assert len(columns) == 4
    for days, multiple in columns:
        assert int(multiple.replace(",", "")) == 2 ** days  # what $1 grows to at 2.00
    rows = re.findall(rf'label: "(\d+)%", note: {STRING}, cells: (\[\[.*?\]\])\s*\}}', table)
    assert len(rows) == 4
    for label, cells in rows:
        p = int(label) / 100
        for (days, _), (chance, average) in zip(columns, json.loads(cells)):
            assert _one_in(chance) == round(1 / rollover.completion_chance(p, days)), (label, days, chance)
            mean = (rollover.expected_multiple(p, 2.0, days) - 1) * 100
            shown = _percent(average)
            decimals = len(average.split(".")[1].rstrip("%")) if "." in average else 0
            assert abs(mean - shown) <= 0.5 * 10 ** -decimals + 1e-9, (label, days, average, mean)


def test_the_survival_bars_are_the_engines() -> None:
    bars = re.findall(r'\{ day: (\d+), share: ([\d.]+), text: "([\d.]+)%" \}', block("SURVIVAL"))
    assert [int(day) for day, _, _ in bars] == [1, 2, 3, 4, 5]
    curve = rollover.survival(0.47, 5)
    for (day, share, text), alive in zip(bars, curve):
        assert float(share) == pytest.approx(alive, abs=0.0005), day
        assert text == f"{alive * 100:.1f}".rstrip("0").rstrip("."), day
