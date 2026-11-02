"""Small fixes, the bot's and the scan's side (the app's side: tests/test_p0_fix_frontend.py "small fixes").

1. A scan row whose edge the scan withheld (engine/scan.py: over EDGE_CAP, ``edge_suspect`` and no number) says why:
   on a scan ranked by edge the bot marks it "edge hidden" where its edge would be and the card says why once.
2. "Last, First" names: the bot's finder button cleans names by the finder's own rule ("Porter Jr., Michael" is
   "Michael Porter Jr."), and the scan reads two names in a row as one where the board says so ("James", "LeBron").
"""

from __future__ import annotations

from typing import Any, Dict, List

from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.engine import scan as engine

NBA = "basketball_nba"
EDGE_HIDDEN = "Edge hidden: over +50% usually means a stale or wrong price."
LONG_PRICE = {"kind": "milestone", "line": 24.5, "label": "25+", "over": {"decimal": 2.79, "american": "+179"},
              "under": None, "yes": None, "book": "DraftKings", "stale": False, "matches": True}


def scan_row(i: int, name: str = "", withheld: bool = False, **over: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rank": i + 1, "athlete_id": str(100 + i), "name": name or f"Player {i}", "team": "BOS", "opponent": "MIA",
        "league": NBA, "kind": "count", "threshold": 25, "projection": 24.1, "chance": 0.55, "model": "nba",
        "hits": {"l10": [6, 10], "season": [30, 50]}, "line": LONG_PRICE, "why": ["6 of his last 10 at 25+"],
    }
    if withheld:
        out["edge_suspect"] = True  # the scan's row: no edge number at all
    else:
        out["edge"] = 0.068
    out.update(over)
    return out


def scan_card(players: List[Dict[str, Any]], **over: Any) -> Dict[str, Any]:
    """players_scan's card for each player's own posted line (no number asked): ranked by edge, so rows show it."""
    out = {"type": "player_scan", "source": "players_scan", "title": "Points: each player's posted line · NBA · tonight",
           "leagues": [NBA], "stat": "pts", "threshold": None, "side": "over", "sort": None, "day": "today",
           "players": players, "close": [], "more": 0, "games": 10, "games_started": 0, "names": []}
    out.update(over)
    return out


# --- 1. a withheld edge says why ----------------------------------------------------------------------------------

def test_a_withheld_edge_is_marked_on_its_row_and_said_once_and_a_normal_row_says_neither() -> None:
    rows = [scan_row(i) for i in range(5)] + [scan_row(5, "Long Price", withheld=True)]
    text = bot_chat.card_text(scan_card(rows))
    assert text.count(EDGE_HIDDEN) == 1 and f"<i>{EDGE_HIDDEN}</i>" in text.split("\n")
    assert "Also: Long Price (24.1, 55%, edge hidden)" in text  # where "edge +6.8%" would be
    assert text.count("edge +6.8%") == 5 and text.count("edge hidden") == 1
    # A numbered row: the mark on its line, beside the price
    numbered = bot_chat.card_text(scan_card([scan_row(0), scan_row(1, "Long Price", withheld=True)]))
    assert "   24.1 proj · 55% · 6/10 L10 · 30/50 season · 25+ 2.79 (DraftKings) · edge hidden" in numbered.split("\n")
    assert numbered.count(EDGE_HIDDEN) == 1
    # Normal rows only: neither, and the text is as before
    plain = bot_chat.card_text(scan_card([scan_row(i) for i in range(6)]))
    assert "hidden" not in plain and plain.count("edge +6.8%") == 6
    # A scan that shows no row's edge (a number asked: ranked by chance) has none to withhold, nor does a ruled-out
    # player's row
    assert "hidden" not in bot_chat.card_text(scan_card(rows, threshold=25))
    out = scan_row(1, "Long Price", withheld=True, ruled_out=True, status="Out")
    assert "hidden" not in bot_chat.card_text(scan_card([scan_row(0), out], names=["Long Price"]))


def worst_case(name: str, withheld: bool = True) -> Dict[str, Any]:
    """An answer around the biggest scan card: the most rows a scan returns, a status, the form model and a long why on
    every row, every note a card carries; a long answer and every head and foot line around it."""
    rows = [scan_row(i, name.format(i=i), withheld=withheld, status="Day-To-Day", model="form",
                     why=["Reached 25+ in 6 of his last 10 games, and 30 of 50 this season at home"])
            for i in range(engine.MAX_ROWS)]
    card = scan_card(rows, sort="edge", more=9999, no_games_today=["americanfootball_nfl", "icehockey_nhl"],
                     games_started=9, loading={"games_ready": 3, "games_total": 10, "failed": 7})
    return {"thread_id": 1, "disclosure": "AI analyst: an opinion, never a guarantee. " * 2,
            "credits": {"left_cents": 80, "daily_cents": 100},
            "reply": {"text": "Words from the analyst. " * 300, "cards": [card], "chips": []}}


ASKED = "who has the best edges on their posted lines tonight " * 3
LONGEST = "D'Angelo Alexander-Walker {i:02d}"  # longer than any NBA player's name, and its apostrophe escapes


def test_the_worst_case_scan_with_every_edge_withheld_still_fits_one_message() -> None:
    result = worst_case(LONGEST)
    [card] = result["reply"]["cards"]
    text = bot_chat.card_text(card)
    # What saying why costs: a character a row ("edge hidden" where "edge +6.8%" was) and the note's one line
    [before] = worst_case(LONGEST, withheld=False)["reply"]["cards"]
    assert len(text) - len(bot_chat.card_text(before)) == engine.MAX_ROWS + len(f"\n<i>{EDGE_HIDDEN}</i>")
    out = bot_chat.reply_text(result, asked=ASKED)
    assert len(out) <= bot_chat.MAX_TEXT
    assert text in out  # the card whole, not dropped to make room: every row marked, the note once
    assert text.count("edge hidden") == engine.MAX_ROWS and out.count(EDGE_HIDDEN) == 1
    # Names no board has: the message still fits (the answer's words, then the card, give way as before)
    assert len(bot_chat.reply_text(worst_case("x" * 60 + " {i}"), asked=ASKED)) <= bot_chat.MAX_TEXT


# --- 2. "Last, First" names ----------------------------------------------------------------------------------------

BOARD = ["LeBron James", "Bronny James", "Stephen Curry", "Seth Curry", "Michael Porter Jr.", "Kevin Porter Jr.",
         "Michael Smith", "Jordan Clarkson", "DeAndre Jordan"]
PLAYERS = [{"athlete_id": str(i), "name": name, "team": "LAL"} for i, name in enumerate(BOARD)]


def found_names(names: List[str]) -> Dict[str, List[str]]:
    found, missing = engine.match_names(PLAYERS, names)
    assert missing == []
    return {asked: [BOARD[int(i)] for i in ids] for asked, ids in found.items()}


def test_the_bots_finder_button_cleans_last_first_names_by_the_finders_rule() -> None:
    for value, names in (("Porter Jr., Michael", ["Michael Porter Jr."]), ("James, LeBron", ["James", "LeBron"]),
                         ("LeBron James, Stephen Curry", ["LeBron James", "Stephen Curry"]), ("Curry", ["Curry"]),
                         ("Michael", ["Michael"]), ("Michael Porter Jr., Curry", ["Michael Porter Jr.", "Curry"])):
        assert bot_chat._finder_names([value]) == names, value
        assert bot_chat._finder_names(names) == names, value  # the rule is idempotent
    card = scan_card([scan_row(0)], threshold=20, names=["Porter Jr., Michael", "James, LeBron"])
    assert bot_chat.finder_hash(card).endswith("&player=Michael+Porter+Jr.&player=James&player=LeBron")


def test_the_scan_reads_two_names_in_a_row_as_one_last_first_name_where_the_board_says_so() -> None:
    # What the finder and the bot's button send for the brief's names, and what the scan finds for them
    assert found_names(["Michael Porter Jr."]) == {"Michael Porter Jr.": ["Michael Porter Jr."]}
    assert found_names(["James", "LeBron"]) == {"LeBron James": ["LeBron James"]}  # before: "James" two players
    assert found_names(["LeBron James", "Stephen Curry"]) == {"LeBron James": ["LeBron James"],
                                                              "Stephen Curry": ["Stephen Curry"]}
    assert found_names(["Curry"]) == {"Curry": ["Stephen Curry", "Seth Curry"]}  # as before: ambiguous
    assert found_names(["Michael"]) == {"Michael": ["Michael Porter Jr.", "Michael Smith"]}  # as before
    assert found_names(["Porter Jr.", "Michael"]) == {"Michael Porter Jr.": ["Michael Porter Jr."]}  # an old address
    # Not one name: each alone is a single player ("Clarkson", "Jordan": DeAndre), "First, Last" order, two words
    assert found_names(["Clarkson", "Jordan"]) == {"Clarkson": ["Jordan Clarkson"], "Jordan": ["DeAndre Jordan"]}
    assert found_names(["LeBron", "James"]) == {"LeBron": ["LeBron James"], "James": ["LeBron James", "Bronny James"]}
    assert found_names(["Curry", "Michael Smith"]) == {"Curry": ["Stephen Curry", "Seth Curry"],
                                                      "Michael Smith": ["Michael Smith"]}
    # The scan's notes: "James, LeBron" is LeBron James, with no "more than one James" note
    named = engine.sort_names(PLAYERS, [], ["James", "LeBron"])
    assert named == {"found": {"LeBron James": ["0"]}, "missing": [], "ambiguous": [], "started": []}
