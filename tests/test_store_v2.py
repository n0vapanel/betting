"""Version 2 of the store: accounts, saved slips, rollover plans, chat, AI credits and warning signs.

Every test uses a fresh SQLite file under tmp_path and passes ``now`` explicitly, so the 24-hour budget
rule, cool-offs and "has this game started" checks run without waiting or touching the clock.
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
from contextlib import closing
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.data import store as store_module
from telegram_betting_bot_simple.data.store import SHARE_ALPHABET, SHARE_CODE_LENGTH, Store
from telegram_betting_bot_simple.engine import rollover, tickets
from telegram_betting_bot_simple.engine.tickets import TicketLeg

HOUR = 3600.0
# 03:00 UTC on 15 January 2027 is still the evening of the 14th in New York
NOW = datetime.datetime(2027, 1, 15, 3, 0, tzinfo=datetime.timezone.utc).timestamp()
FUTURE, PAST = "2030-01-01T00:00:00Z", "2020-01-01T00:00:00Z"


@pytest.fixture()
def store(tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    for user_id, name in (("tg:1", "Sam"), ("tg:2", "Alex")):
        store.get_or_create_user(user_id, name, now=NOW)
    return store


def leg(event: str = "g1", side: str = "home", price="1.91", start: str = FUTURE, **extra):
    """A leg shaped like a priced ticket's leg."""
    return {"key": f"basketball_nba:{event}:h2h:{side}", "league": "basketball_nba", "event_id": event,
            "market": "h2h", "side": side, "point": None, "price": price, "label": f"{side} in {event}",
            "game": "Clippers @ Thunder", "start": start, "ai_prob": 0.55, "market_prob": 0.5, **extra}


def save(store: Store, user: str = "tg:1", legs=None, stake: int = 1000, status: str = "saved", **extra):
    return store.save_slip(user, "multi" if len(legs or [1]) > 1 else "single", legs or [leg()], stake,
                           status=status, now=extra.pop("now", NOW), **extra)


# --- the schema --------------------------------------------------------------------------------


def test_new_tables_sit_beside_the_old_ones(tmp_path) -> None:
    path = str(tmp_path / "betting.db")
    store = Store(path)
    store.save_odds_snapshot("basketball_nba", [{"id": "g1"}], 1.0)
    store.get_or_create_user("tg:1", "Sam", now=NOW)
    reopened = Store(path)  # a second process (or a restart) on the same file
    assert reopened.odds_snapshot("basketball_nba") == (1.0, [{"id": "g1"}])
    assert reopened.user("tg:1")["name"] == "Sam"
    with closing(sqlite3.connect(path)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert {"odds_snapshots", "predictions", "analyses", "users", "slips", "slip_legs", "plans", "plan_days",
            "chat_threads", "chat_messages", "ai_credits", "ai_holds", "rg_events"} <= tables


def test_the_stores_limits_match_the_engines() -> None:
    """The store checks what it keeps against the engine's rules, so a stored slip or plan can be re-priced."""
    assert store_module.TICKET_TYPES == tickets.TICKET_TYPES
    assert store_module.MAX_SLIP_LEGS == tickets.MAX_LEGS
    assert store_module.PLAN_STRATEGIES == rollover.STRATEGIES
    assert store_module.PLAN_DAYS == (rollover.MIN_DAYS, rollover.MAX_DAYS)
    assert store_module.PLAN_LEGS_PER_DAY == (rollover.MIN_LEGS, rollover.MAX_LEGS)
    assert store_module.PLAN_MIN_ODDS == rollover.MIN_ODDS


# --- users ---------------------------------------------------------------------------------------


def test_users_are_made_once_and_keep_a_chosen_name(store) -> None:
    sam = store.user("tg:1")
    assert sam["name"] == "Sam" and sam["created_at"] == NOW
    assert (sam["odds_format"], sam["currency"], sam["favourite_sports"], sam["settings"]) == ("decimal", "USD", [], {})
    assert sam["age_confirmed_at"] is None and sam["weekly_budget_cents"] is None and sam["cooloff_until"] is None
    # Telegram's name on a later visit doesn't overwrite theirs, and the account isn't recreated
    again = store.get_or_create_user("tg:1", "Samuel", now=NOW + HOUR)
    assert again["name"] == "Sam" and again["created_at"] == NOW
    # A name fills a blank one
    assert store.get_or_create_user("web:abc123", now=NOW)["name"] is None
    assert store.get_or_create_user("web:abc123", "  Jo ", now=NOW)["name"] == "Jo"
    assert store.user("tg:404") is None
    with pytest.raises(ValueError):
        store.get_or_create_user("  ")


def test_a_blank_name_is_no_name_so_a_real_one_can_fill_it(store) -> None:
    assert store.get_or_create_user("web:x", "   ", now=NOW)["name"] is None  # not "", which COALESCE keeps
    assert store.get_or_create_user("web:x", "Jo", now=NOW)["name"] == "Jo"
    assert store.update_user("web:x", {"name": "  "})["name"] is None
    assert store.get_or_create_user("web:x", "Jo", now=NOW)["name"] == "Jo"


def test_settings_are_checked_and_merged(store) -> None:
    updated = store.update_user("tg:1", {
        "odds_format": "american", "currency": "gbp", "favourite_sports": ["basketball_nba", "icehockey_nhl",
                                                                          "basketball_nba"],
        "risk_tier": "safe", "reality_check_minutes": 30, "settings": {"tz": "Europe/London", "theme": "dark"},
    })
    assert updated["odds_format"] == "american" and updated["currency"] == "GBP"
    assert updated["favourite_sports"] == ["basketball_nba", "icehockey_nhl"]
    assert updated["risk_tier"] == "safe" and updated["reality_check_minutes"] == 30
    # settings merge in; None removes a key
    merged = store.update_user("tg:1", {"settings": {"theme": None, "lang": "en"}})["settings"]
    assert merged == {"tz": "Europe/London", "lang": "en"}
    assert store.update_user("tg:1", {"reality_check_minutes": None})["reality_check_minutes"] is None
    assert store.update_user("tg:404", {"odds_format": "decimal"}) is None


@pytest.mark.parametrize("changes", [
    {"odds_format": "moneyline"}, {"reality_check_minutes": 45}, {"reality_check_minutes": True},
    {"risk_tier": "yolo"}, {"currency": "dollars"}, {"favourite_sports": "nba"}, {"settings": ["dark"]},
    # Limits the AI (or a quick settings form) can't touch: each has its own rule
    {"weekly_budget_cents": 1}, {"age_confirmed_at": 1.0}, {"cooloff_until": None}, {"id": "tg:9"},
])
def test_bad_or_protected_settings_are_refused(store, changes) -> None:
    with pytest.raises(ValueError):
        store.update_user("tg:1", changes)
    assert store.user("tg:1")["odds_format"] == "decimal"


def test_a_budget_lowers_at_once_and_raises_a_day_later(store) -> None:
    first = store.set_weekly_budget("tg:1", 5000, now=NOW)  # onboarding: applies at once
    assert first["weekly_budget_cents"] == 5000 and first["pending_budget_cents"] is None
    assert store.set_weekly_budget("tg:1", 3000, now=NOW)["weekly_budget_cents"] == 3000  # lower: at once

    raised = store.set_weekly_budget("tg:1", 8000, now=NOW)
    assert raised["weekly_budget_cents"] == 3000
    assert raised["pending_budget_cents"] == 8000 and raised["budget_raise_at"] == NOW + 24 * HOUR
    assert store.user("tg:1", now=NOW + 23 * HOUR)["weekly_budget_cents"] == 3000
    later = store.user("tg:1", now=NOW + 24 * HOUR)
    assert later["weekly_budget_cents"] == 8000 and later["pending_budget_cents"] is None
    assert later["budget_raise_at"] is None

    # Lowering while a raise waits cancels the raise
    t = NOW + 30 * HOUR
    store.set_weekly_budget("tg:1", 10000, now=t)
    lowered = store.set_weekly_budget("tg:1", 2000, now=t + HOUR)
    assert lowered["weekly_budget_cents"] == 2000 and lowered["pending_budget_cents"] is None
    assert store.user("tg:1", now=t + 48 * HOUR)["weekly_budget_cents"] == 2000


def test_asking_for_more_restarts_a_raise_and_asking_for_less_keeps_its_clock(store) -> None:
    store.set_weekly_budget("tg:1", 2000, now=NOW)
    store.set_weekly_budget("tg:1", 4000, now=NOW)
    smaller = store.set_weekly_budget("tg:1", 3500, now=NOW + 10 * HOUR)
    assert smaller["pending_budget_cents"] == 3500 and smaller["budget_raise_at"] == NOW + 24 * HOUR
    bigger = store.set_weekly_budget("tg:1", 6000, now=NOW + 12 * HOUR)
    assert bigger["pending_budget_cents"] == 6000 and bigger["budget_raise_at"] == NOW + 36 * HOUR
    assert store.user("tg:1", now=NOW + 30 * HOUR)["weekly_budget_cents"] == 2000


@pytest.mark.parametrize("cents", [-1, 10.5, "1000", True, None, 10 ** 19])
def test_a_budget_is_whole_cents(store, cents) -> None:
    with pytest.raises(ValueError):
        store.set_weekly_budget("tg:1", cents, now=NOW)


def test_budget_and_age_need_an_account(store) -> None:
    assert store.set_weekly_budget("tg:404", 1000, now=NOW) is None
    assert store.confirm_age("tg:404", now=NOW) is False


def test_age_confirmation_keeps_the_first_time(store) -> None:
    assert store.confirm_age("tg:1", now=NOW) is True
    assert store.confirm_age("tg:1", now=NOW + HOUR) is True
    assert store.user("tg:1")["age_confirmed_at"] == NOW


def test_a_cooloff_can_be_extended_but_never_shortened(store) -> None:
    assert store.start_cooloff("tg:1", 48, now=NOW) == NOW + 48 * HOUR
    assert store.cooloff_active("tg:1", now=NOW + 47 * HOUR)
    assert not store.cooloff_active("tg:1", now=NOW + 49 * HOUR)
    assert store.start_cooloff("tg:1", 24, now=NOW) == NOW + 48 * HOUR  # shorter: ignored
    assert store.start_cooloff("tg:1", 24 * 30, now=NOW) == NOW + 30 * 24 * HOUR
    assert store.user("tg:1")["cooloff_until"] == NOW + 30 * 24 * HOUR
    assert not store.cooloff_active("tg:2", now=NOW) and not store.cooloff_active("tg:404", now=NOW)
    assert store.start_cooloff("tg:404", 24, now=NOW) is None
    for hours in (1, 24 * 365, float("nan"), True):
        with pytest.raises(ValueError):
            store.start_cooloff("tg:2", hours, now=NOW)


# --- slips ---------------------------------------------------------------------------------------


def test_slips_keep_the_exact_quote_and_get_a_share_code(store) -> None:
    legs = [
        leg("g1", price=Decimal("1.910")),
        leg("g2", price=2.5, quoted_at=NOW - 60),
        TicketLeg("basketball_nba:g3:totals:over:224.5", "basketball_nba", "g3", "totals", "over", 224.5,
                  Decimal("3.40"), "Over 224.5", "Magic @ Knicks", FUTURE, 0.3, 0.28, 0.01, "DraftKings"),
    ]
    payload = {"type": "multi", "max_payout": "56.65", "legs": [{"price": Decimal("1.910")}]}
    slip = store.save_slip("tg:1", "multi", legs, 1000, payload=payload, quoted_at=NOW - 300, now=NOW)
    assert slip["user"] == "tg:1" and slip["status"] == "saved" and slip["type"] == "multi"
    assert slip["stake_cents"] == 1000 and slip["created_at"] == slip["updated_at"] == NOW
    assert [l["price"] for l in slip["legs"]] == ["1.910", "2.5", "3.40"]
    assert [l["idx"] for l in slip["legs"]] == [0, 1, 2]
    assert [l["quoted_at"] for l in slip["legs"]] == [NOW - 300, NOW - 60, NOW - 300]
    third = slip["legs"][2]
    assert (third["market"], third["side"], third["point"], third["book"]) == ("totals", "over", 224.5, "DraftKings")
    assert third["push_prob"] == 0.01 and third["result"] is None
    assert slip["payload"]["legs"][0]["price"] == "1.910"  # Decimals in the payload stay exact too
    code = slip["share_code"]
    assert len(code) == SHARE_CODE_LENGTH and set(code) <= set(SHARE_ALPHABET)
    assert save(store)["share_code"] != code


@pytest.mark.parametrize("legs, stake, status", [
    ([leg(price="1.00")], 1000, "saved"),
    ([leg(price="abc")], 1000, "saved"),
    ([leg(price=float("inf"))], 1000, "saved"),
    ([leg(label="")], 1000, "saved"),
    ([{k: v for k, v in leg().items() if k != "event_id"}], 1000, "saved"),
    ([], 1000, "saved"),
    ([leg()], 0, "saved"),        # a bet needs a stake (a draft doesn't)
    ([leg()], 10.5, "draft"),
    ([leg()], 1000, "settled"),   # settling has its own method
])
def test_bad_slips_are_refused(store, legs, stake, status) -> None:
    with pytest.raises(ValueError):
        store.save_slip("tg:1", "single", legs, stake, status=status, now=NOW)
    assert store.slips("tg:1") == []


@pytest.mark.parametrize("ticket_type, legs, extra", [
    ("parlay-of-doom", [leg()], {}),
    ("", [leg()], {}),
    ("single", [leg(ai_prob=7.5)], {}),
    ("single", [leg(market_prob=-0.1)], {}),
    ("single", [leg(push_prob=1.5)], {}),
    ("single", [leg(ai_prob=float("nan"))], {}),
    ("single", [leg(market_prob=Decimal("NaN"))], {}),
    ("single", [leg(ai_prob="0.5")], {}),
    ("single", [leg(ai_prob=True)], {}),
    ("single", [leg(start=None)], {}),        # a bet needs every game's start time...
    ("single", [leg(start="  ")], {}),
    ("single", [leg(start="tomorrow")], {}),
    ("round_robin", [leg("g1"), leg("g2"), leg("g3")], {"k": 0}),
    ("round_robin", [leg("g1"), leg("g2"), leg("g3")], {"k": 10 ** 19}),
])
def test_slips_the_engine_could_not_reprice_are_refused(store, ticket_type, legs, extra) -> None:
    with pytest.raises(ValueError):
        store.save_slip("tg:1", ticket_type, legs, 1000, now=NOW, **extra)
    assert store.slips("tg:1") == []


def test_start_times_are_kept_in_the_one_form_the_started_checks_compare(store) -> None:
    # ...a draft may still be missing one
    draft = save(store, legs=[leg(start=None)], stake=0, status="draft")
    assert draft["legs"][0]["start"] is None
    # ESPN's minutes-only form and an offset would sort wrongly against "now" as text
    slip = save(store, legs=[leg("g1", start="2030-01-01T00:30Z"), leg("g2", start="2030-01-01T02:00:00+02:00"),
                             leg("g3", start="2030-01-01T00:00:00")])
    assert [l["start"] for l in slip["legs"]] == ["2030-01-01T00:30:00Z", "2030-01-01T00:00:00Z",
                                                   "2030-01-01T00:00:00Z"]  # no zone given: UTC


def test_slips_are_listed_and_fetched_only_by_their_owner(store) -> None:
    draft = save(store, stake=0, status="draft", now=NOW)
    first = save(store, now=NOW + 1)
    second = save(store, legs=[leg("g1"), leg("g2", side="away")], now=NOW + 2)
    other = save(store, user="tg:2", now=NOW + 3)
    assert [s["id"] for s in store.slips("tg:1")] == [second["id"], first["id"], draft["id"]]
    assert [s["id"] for s in store.slips("tg:1", status="saved")] == [second["id"], first["id"]]
    assert [s["id"] for s in store.slips("tg:1", status=["draft", "settled"])] == [draft["id"]]
    assert len(store.slips("tg:1", limit=1)) == 1
    assert len(store.slips("tg:1")[0]["legs"]) == 2
    assert store.slip("tg:1", other["id"]) is None and store.slip("tg:2", other["id"])["id"] == other["id"]

    # Anyone with the share code can view it; only the owner sees whose it is
    shared = store.shared_slip(" " + second["share_code"].lower() + " ")
    assert shared["id"] == second["id"] and shared["mine"] is False and shared["user"] is None
    assert len(shared["legs"]) == 2
    assert store.shared_slip(second["share_code"], viewer="tg:2")["user"] is None
    mine = store.shared_slip(second["share_code"], viewer="tg:1")
    assert mine["mine"] is True and mine["user"] == "tg:1"
    assert store.shared_slip("NOPE2345") is None and store.shared_slip("") is None


def test_a_draft_is_edited_in_place_then_saved_as_a_bet(store) -> None:
    draft = save(store, stake=0, status="draft", now=NOW)
    edited = store.save_slip("tg:1", "multi", [leg("g1"), leg("g2")], 0, status="draft", slip_id=draft["id"],
                             now=NOW + 60)
    assert (edited["id"], edited["share_code"]) == (draft["id"], draft["share_code"])
    assert len(edited["legs"]) == 2 and edited["created_at"] == NOW and edited["updated_at"] == NOW + 60
    saved = store.save_slip("tg:1", "multi", [leg("g1"), leg("g2", price="2.10")], 500, slip_id=draft["id"],
                            payload={"total_stake": "5.00"}, now=NOW + 120)
    assert saved["status"] == "saved" and saved["created_at"] == NOW + 120  # the bet counts from now
    assert [l["price"] for l in saved["legs"]] == ["1.91", "2.10"] and saved["payload"] == {"total_stake": "5.00"}
    # Only a draft can be replaced, and only by its owner
    assert store.save_slip("tg:1", "single", [leg()], 500, slip_id=draft["id"], now=NOW) is None
    other = save(store, user="tg:2", stake=0, status="draft")
    assert store.save_slip("tg:1", "single", [leg()], 500, slip_id=other["id"], now=NOW) is None
    assert len(store.slip("tg:2", other["id"])["legs"]) == 1


def test_deleting_a_slip_never_rewrites_history(store) -> None:
    draft = save(store, legs=[leg(start=PAST)], stake=0, status="draft")
    assert store.delete_slip("tg:1", draft["id"], now=NOW) is True  # a draft was never a bet
    assert store.slip("tg:1", draft["id"]) is None

    pending = save(store)
    assert store.delete_slip("tg:2", pending["id"], now=NOW) is False  # not theirs
    assert store.delete_slip("tg:1", pending["id"], now=NOW) is True  # nothing has started yet
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute("SELECT COUNT(*) FROM slip_legs WHERE slip_id = ?", (pending["id"],)).fetchone()[0] == 0

    under_way = save(store, legs=[leg("g1"), leg("g2", start=PAST)])
    assert store.delete_slip("tg:1", under_way["id"], now=NOW) is False
    settled = save(store)
    store.settle_slip(settled["id"], "settled", {"status": "lost"}, now=NOW)
    assert store.delete_slip("tg:1", settled["id"], now=NOW) is False
    in_a_plan = save(store)
    plan = store.create_plan("tg:1", 1000, 5, "2.00", 1, "ladder", now=NOW)
    store.record_plan_day("tg:1", plan["id"], 100, 1000, slip_id=in_a_plan["id"], now=NOW)
    assert store.delete_slip("tg:1", in_a_plan["id"], now=NOW) is False
    assert store.delete_slip("tg:1", 99999, now=NOW) is False
    assert {s["id"] for s in store.slips("tg:1")} == {under_way["id"], settled["id"], in_a_plan["id"]}


def test_a_bet_with_no_start_time_on_record_counts_as_started(store) -> None:
    """Saving a bet now needs every start time; a row without one (an older save) fails closed."""
    unknown = save(store, legs=[leg("g1"), leg("g2")])
    with closing(sqlite3.connect(store.path)) as db, db:
        db.execute("UPDATE slip_legs SET start = NULL WHERE slip_id = ? AND idx = 1", (unknown["id"],))
    assert store.delete_slip("tg:1", unknown["id"], now=NOW) is False
    assert store.delete_slip("tg:1", unknown["id"], now=NOW + 1e8) is False  # long after: the loss stays
    assert [s["id"] for s in store.open_slips(started_before="2027-01-15T03:00:00Z")] == [unknown["id"]]


def test_slips_settle_from_the_graders_queue(store) -> None:
    started = save(store, legs=[leg("g1", start=PAST), leg("g2")], now=NOW)
    later = save(store, legs=[leg("g3")], now=NOW + 1)
    draft = save(store, legs=[leg("g4", start=PAST)], stake=0, status="draft")
    now_iso = "2027-01-15T03:00:00Z"
    assert [s["id"] for s in store.open_slips(started_before=now_iso)] == [started["id"]]
    assert [s["id"] for s in store.open_slips()] == [started["id"], later["id"]]

    store.set_leg_results(started["id"], ["won", None])  # the second game is still to play
    assert [l["result"] for l in store.slip("tg:1", started["id"])["legs"]] == ["won", None]

    heat = {"result": "dead_heat", "places_paid": 1, "tied": 2}
    grade = {"status": "won", "returns": "18.24", "profit": "8.24"}
    assert store.settle_slip(started["id"], "settled", grade, leg_results=["won", heat], now=NOW + 5) is True
    settled = store.slip("tg:1", started["id"])
    assert settled["status"] == "settled" and settled["result"] == grade and settled["updated_at"] == NOW + 5
    assert [l["result"] for l in settled["legs"]] == ["won", heat]
    assert [s["id"] for s in store.open_slips()] == [later["id"]]
    assert store.settle_slip(later["id"], "void", {"status": "void", "returns": "10.00"}, now=NOW) is True
    assert store.slip("tg:1", later["id"])["status"] == "void"

    assert store.settle_slip(draft["id"], "settled", {}, now=NOW) is False  # a draft was never a bet
    assert store.settle_slip(99999, "settled", {}, now=NOW) is False
    with pytest.raises(ValueError):
        store.settle_slip(later["id"], "won", {}, now=NOW)


def test_a_priced_ticket_is_saved_then_graded_from_what_was_stored(store) -> None:
    """The main path: price_ticket's legs and output saved as they are, then graded from the stored payload."""
    legs = [TicketLeg(f"basketball_nba:{event}:h2h:home", "basketball_nba", event, "h2h", "home", None,
                      Decimal(price), f"Home in {event}", "Clippers @ Thunder", FUTURE, 0.55, 0.5)
            for event, price in (("g1", "1.91"), ("g2", "2.10"), ("g3", "1.80"))]
    now = datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc)
    priced = tickets.price_ticket(legs, "system", Decimal("1.00"), system="trixie", now=now)
    assert priced["valid"] and priced["bets"] == 4 and priced["total_stake"] == "4.00"
    slip = store.save_slip("tg:1", priced["type"], priced["legs"], 400, payload=priced, system=priced["system"],
                           k=priced["k"], now=NOW)
    assert [l["price"] for l in slip["legs"]] == ["1.91", "2.10", "1.80"]
    assert (slip["type"], slip["system"], slip["k"]) == ("system", "trixie", None)

    store.set_leg_results(slip["id"], ["won", "push", "lost"])
    stored = store.slip("tg:1", slip["id"])
    graded = tickets.grade([l["result"] for l in stored["legs"]], stored["payload"])
    # Only the double of legs 1 and 2 pays, and the push drops out of it: 1.00 x 1.91
    assert (graded["status"], graded["returns"], graded["profit"], graded["paid_bets"]) == (
        "partial", "1.91", "-2.09", 1)
    assert store.settle_slip(slip["id"], "settled", graded, now=NOW + HOUR) is True
    assert store.slip("tg:1", slip["id"])["result"]["returns"] == "1.91" and store.open_slips() == []


def test_stakes_count_against_the_budget_from_when_the_bet_was_saved(store) -> None:
    save(store, stake=1000, now=NOW)
    save(store, stake=250, now=NOW + HOUR)
    save(store, stake=0, status="draft", now=NOW)
    voided = save(store, stake=400, now=NOW)
    store.settle_slip(voided["id"], "void", {"status": "void"}, now=NOW)
    settled = save(store, stake=300, now=NOW)
    store.settle_slip(settled["id"], "settled", {"status": "lost"}, now=NOW)
    save(store, user="tg:2", stake=5000, now=NOW)
    assert store.staked_since("tg:1", NOW) == 1000 + 250 + 300
    assert store.staked_since("tg:1", NOW + 1) == 250
    assert store.staked_since("tg:404", 0) == 0


def test_the_budget_counts_real_stakes_by_one_person_or_every_password_browser(store) -> None:
    save(store, stake=1000, now=NOW)
    save(store, stake=700, now=NOW, payload={"plan": {"id": 1, "mode": "paper"}})  # play money
    save(store, stake=200, now=NOW, payload={"plan": {"id": 2, "mode": "tracked"}, "note": "paper"})
    for user, stake in (("web:browser-a1", 300), ("web:browser-b2", 40), ("webby:x", 9), ("tg:2", 5000)):
        store.get_or_create_user(user, now=NOW)
        save(store, user=user, stake=stake, now=NOW)
    save(store, user="web:browser-a1", stake=60, now=NOW, payload={"plan": {"mode": "paper"}})
    assert store.budget_staked_since(NOW, user="tg:1") == 1000 + 200
    assert store.budget_staked_since(NOW, members="web:") == 300 + 40  # not "webby:", not a paper day
    assert store.budget_staked_since(NOW + 1, members="web:") == 0
    with pytest.raises(ValueError):
        store.budget_staked_since(NOW, user="tg:1", members="web:")


# --- plans ---------------------------------------------------------------------------------------


def test_plans_are_recorded_one_day_at_a_time(store) -> None:
    comparison = {"classic": {"chance_complete": 0.0012}, "ladder": {"median_end": "9.40"}}
    plan = store.create_plan("tg:1", 1000, 3, Decimal("2.00"), 1, "ladder", payload=comparison, now=NOW)
    assert (plan["status"], plan["mode"], plan["day"], plan["bankroll_cents"]) == ("active", "paper", 0, 1000)
    assert plan["min_odds"] == "2.00" and plan["payload"] == comparison and plan["days_played"] == []
    assert plan["days"] == 3 and plan["legs_per_day"] == 1 and plan["strategy"] == "ladder"

    slip = save(store, stake=100)
    day1 = store.record_plan_day("tg:1", plan["id"], 100, 1100, result="won", banked_cents=100,
                                 slip_id=slip["id"], now=NOW + HOUR)
    assert day1 == {"plan_id": plan["id"], "day": 1, "slip_id": slip["id"], "stake_cents": 100, "result": "won",
                    "banked_cents": 100, "rolled_cents": 0, "created_at": NOW + HOUR}
    day2 = store.record_plan_day("tg:1", plan["id"], 100, 1100, rolled_cents=100, now=NOW + 25 * HOUR)
    assert day2["day"] == 2 and day2["result"] is None  # the result can wait for the games
    assert store.settle_plan_day("tg:1", plan["id"], 2, "lost", 1000, rolled_cents=0) is True

    fetched = store.plan("tg:1", plan["id"])
    assert fetched["day"] == 2 and fetched["bankroll_cents"] == 1000
    assert [(d["day"], d["result"]) for d in fetched["days_played"]] == [(1, "won"), (2, "lost")]
    assert store.record_plan_day("tg:1", plan["id"], 100, 900, now=NOW)["day"] == 3
    assert store.record_plan_day("tg:1", plan["id"], 100, 900, now=NOW) is None  # every day recorded

    assert store.finish_plan("tg:1", plan["id"], "failed") is True
    assert store.finish_plan("tg:1", plan["id"], "done") is False  # finished once
    assert store.plan("tg:1", plan["id"])["status"] == "failed"
    # A day still in play when the plan ended is recorded all the same
    assert store.settle_plan_day("tg:1", plan["id"], 3, "lost", 800) is True
    assert store.settle_plan_day("tg:1", plan["id"], 9, "lost", 800) is False


def test_plans_belong_to_their_owner(store) -> None:
    mine = store.create_plan("tg:1", 1000, 10, "1.50", 2, "flat", mode="tracked", now=NOW)
    stopped = store.create_plan("tg:1", 500, 5, "2", 1, "classic", now=NOW + 1)
    store.finish_plan("tg:1", stopped["id"], "stopped")
    theirs = store.create_plan("tg:2", 1000, 10, "2.00", 1, "bank_half", now=NOW)
    assert [p["id"] for p in store.plans("tg:1")] == [stopped["id"], mine["id"]]
    assert [p["id"] for p in store.plans("tg:1", status="active")] == [mine["id"]]
    assert store.plans("tg:1", status="active")[0]["mode"] == "tracked"
    assert store.plan("tg:2", mine["id"]) is None
    assert store.record_plan_day("tg:2", mine["id"], 50, 1000, now=NOW) is None
    assert store.record_plan_day("tg:1", stopped["id"], 50, 500, now=NOW) is None  # not active
    their_slip = save(store, user="tg:2")
    assert store.record_plan_day("tg:1", mine["id"], 50, 1000, slip_id=their_slip["id"], now=NOW) is None
    # Every password browser's plans, for the rules that cover them all (each plan keeps its own user)
    web_a = store.create_plan("web:browser-a1", 1000, 5, "2", 1, "classic", now=NOW + 2)
    web_b = store.create_plan("web:browser-b2", 1000, 5, "2", 1, "classic", now=NOW + 3)
    assert [(p["id"], p["user"]) for p in store.plans(members="web:")] == [
        (web_b["id"], "web:browser-b2"), (web_a["id"], "web:browser-a1")]
    assert [p["id"] for p in store.plans(members="web:", status="active", limit=1)] == [web_b["id"]]
    for bad in ({}, {"user": "tg:1", "members": "web:"}):
        with pytest.raises(ValueError, match="a user or an id prefix"):
            store.plans(**bad)
    assert store.settle_plan_day("tg:2", mine["id"], 1, "won", 1000) is False
    assert store.finish_plan("tg:2", mine["id"], "stopped") is False
    assert store.plan("tg:1", mine["id"])["day"] == 0 and store.plan("tg:2", theirs["id"])["status"] == "active"


def test_settling_a_day_keeps_what_it_leaves_out_and_never_rewinds_the_bankroll(store) -> None:
    plan = store.create_plan("tg:1", 1000, 5, "2.00", 1, "bank_half", now=NOW)
    store.record_plan_day("tg:1", plan["id"], 1000, 2000, banked_cents=500, rolled_cents=1000, now=NOW)
    assert store.settle_plan_day("tg:1", plan["id"], 1, "won", 2000) is True
    day1 = store.plan("tg:1", plan["id"])["days_played"][0]
    assert (day1["result"], day1["banked_cents"], day1["rolled_cents"]) == ("won", 500, 1000)
    assert store.settle_plan_day("tg:1", plan["id"], 1, "won", 2000, banked_cents=0) is True
    assert store.plan("tg:1", plan["id"])["days_played"][0]["banked_cents"] == 0

    # Day 2 is recorded before day 1's result is in; settling day 1 late keeps day 2's bankroll
    later = store.create_plan("tg:1", 1000, 5, "2.00", 1, "classic", now=NOW)
    store.record_plan_day("tg:1", later["id"], 1000, 2000, now=NOW)
    store.record_plan_day("tg:1", later["id"], 2000, 4000, now=NOW + HOUR)
    assert store.settle_plan_day("tg:1", later["id"], 1, "won", 2000) is True
    assert store.plan("tg:1", later["id"])["bankroll_cents"] == 4000
    assert store.settle_plan_day("tg:1", later["id"], 2, "lost", 0) is True
    assert store.plan("tg:1", later["id"])["bankroll_cents"] == 0


def test_the_planners_limits_are_accepted(store) -> None:
    lowest = store.create_plan("tg:1", 1, 1, "1.2", 1, "classic", now=NOW)
    assert (lowest["days"], lowest["min_odds"], lowest["legs_per_day"]) == (1, "1.2", 1)
    assert store.create_plan("tg:1", 10 ** 11, 60, Decimal("1000"), 3, "flat", now=NOW)["days"] == 60


@pytest.mark.parametrize("args, kwargs", [
    ((0, 5, "2.00", 1, "ladder"), {}),
    ((10 ** 19, 5, "2.00", 1, "ladder"), {}),
    ((1000, 0, "2.00", 1, "ladder"), {}),
    ((1000, 61, "2.00", 1, "ladder"), {}),
    ((1000, 10 ** 19, "2.00", 1, "ladder"), {}),
    ((1000, 5, "1", 1, "ladder"), {}),
    ((1000, 5, "1.01", 1, "ladder"), {}),
    ((1000, 5, "1.19", 1, "ladder"), {}),
    ((1000, 5, "2.00", 0, "ladder"), {}),
    ((1000, 5, "2.00", 4, "ladder"), {}),
    ((1000, 5, "2.00", 99, "ladder"), {}),
    ((1000, 5, "2.00", 1, ""), {}),
    ((1000, 5, "2.00", 1, "yolo"), {}),
    ((1000, 5, "2.00", 1, "ladder"), {"mode": "real money"}),
])
def test_bad_plans_are_refused(store, args, kwargs) -> None:
    with pytest.raises(ValueError):
        store.create_plan("tg:1", *args, now=NOW, **kwargs)
    with pytest.raises(ValueError):
        store.finish_plan("tg:1", 1, "active")


def test_two_processes_never_record_the_same_day_twice(store) -> None:
    other_process = Store(store.path)
    plan = store.create_plan("tg:1", 1000, 5, "2.00", 1, "ladder", now=NOW)
    assert store.record_plan_day("tg:1", plan["id"], 100, 1000, now=NOW)["day"] == 1
    assert other_process.record_plan_day("tg:1", plan["id"], 100, 1000, now=NOW)["day"] == 2
    assert [d["day"] for d in store.plan("tg:1", plan["id"])["days_played"]] == [1, 2]


# --- chat ----------------------------------------------------------------------------------------


def test_chat_threads_keep_their_messages_in_order(store) -> None:
    older = store.create_thread("tg:1", "Tonight's NBA", now=NOW)
    newer = store.create_thread("tg:1", now=NOW + 10)
    assert older["title"] == "Tonight's NBA" and newer["title"] is None and older["summary"] is None
    blocks = [{"type": "text", "text": "Thunder -3.5 has about a 1 in 2 chance."}, {"type": "tool_use", "id": "t1"}]
    assert store.add_message("tg:1", older["id"], "user", "Is Thunder -3.5 good?", now=NOW + 20) == 0
    assert store.add_message("tg:1", older["id"], "assistant", blocks, tokens=812, cost=0.0123, now=NOW + 21) == 1
    assert store.add_message("tg:1", older["id"], "user", "And the total?", now=NOW + 22) == 2

    messages = store.messages("tg:1", older["id"])
    assert [(m["idx"], m["role"]) for m in messages] == [(0, "user"), (1, "assistant"), (2, "user")]
    assert messages[1]["content"] == blocks and messages[1]["tokens"] == 812 and messages[1]["cost"] == 0.0123
    assert [m["idx"] for m in store.messages("tg:1", older["id"], last=2)] == [1, 2]

    # Most recently active first, with a message count
    listed = store.threads("tg:1")
    assert [(t["id"], t["messages"], t["last_ts"]) for t in listed] == [(older["id"], 3, NOW + 22),
                                                                         (newer["id"], 0, NOW + 10)]
    assert store.update_thread("tg:1", older["id"], summary="Asked about Thunder -3.5") is True
    assert store.thread("tg:1", older["id"])["summary"] == "Asked about Thunder -3.5"
    assert store.update_thread("tg:1", older["id"]) is True

    # Nobody else can read, write to or rename someone's thread
    assert store.thread("tg:2", older["id"]) is None and store.threads("tg:2") == []
    assert store.messages("tg:2", older["id"]) == []
    assert store.add_message("tg:2", older["id"], "user", "hi", now=NOW) is None
    assert store.update_thread("tg:2", older["id"], title="mine now") is False
    assert store.update_thread("tg:2", older["id"]) is False
    for bad in ({"role": "", "content": "hi"}, {"role": "user", "content": None},
                {"role": "user", "content": "hi", "tokens": -1}, {"role": "user", "content": "hi", "tokens": 10 ** 19},
                {"role": "user", "content": "hi", "cost": float("inf")}):
        with pytest.raises(ValueError):
            store.add_message("tg:1", older["id"], now=NOW, **bad)
    assert len(store.messages("tg:1", older["id"])) == 3


# --- AI credits ----------------------------------------------------------------------------------


def test_ai_credits_add_up_per_person_per_day(store) -> None:
    assert Store.credit_day(NOW) == "2027-01-14"  # still the 14th in New York
    assert store.ai_used_today("tg:1", now=NOW) == 0.0
    assert store.add_ai_usage("tg:1", 1.25, now=NOW) == 1.25
    assert store.add_ai_usage("tg:1", Decimal("0.5"), now=NOW + HOUR) == 1.75
    assert store.add_ai_usage("tg:1", 3, day="2027-01-15") == 3.0
    assert store.add_ai_usage("tg:2", 0.4, now=NOW) == 0.4
    assert store.ai_used_today("tg:1", now=NOW) == 1.75
    assert store.ai_used_today("tg:1", day="2027-01-15") == 3.0
    # Infinity would lock them out of the AI for the day; NaN would fail inside SQLite
    for cents in (-1, "2", True, None, float("inf"), float("nan"), Decimal("NaN"), Decimal("sNaN"), 10 ** 400):
        with pytest.raises(ValueError):
            store.add_ai_usage("tg:1", cents, now=NOW)
    assert store.ai_used_today("tg:1", now=NOW) == 1.75


def test_an_ai_hold_is_one_call_at_a_time_across_processes(tmp_path) -> None:
    path = str(tmp_path / "betting.db")
    bot, dashboard = Store(path), Store(path)  # two processes on one file
    caps = {"web:a": 100, "*web": 50, "*all": 1000}
    token, cents = dashboard.reserve_ai("web:a", caps, 40, 5, now=NOW)
    assert cents == 40
    assert bot.reserve_ai("web:a", caps, 40, 5, now=NOW) is None  # busy, whichever process asks
    # Another password browser gets only what the shared allowance has left after the hold
    other = bot.reserve_ai("web:b", {"web:b": 100, "*web": 50, "*all": 1000}, 40, 5, now=NOW)
    assert other[1] == 10
    assert bot.ai_left("web:c", {"web:c": 100, "*web": 50, "*all": 1000}, now=NOW) == 0
    assert bot.ai_left("tg:1", {"tg:1": 100, "*all": 1000}, now=NOW) == 100  # "*all" has 950 left
    assert dashboard.ai_left("web:a", caps, now=NOW) == 40  # a person's own hold isn't taken off
    # The real cost moves from held to used in one write; the person stays busy until released
    dashboard.settle_ai("web:a", token, 12.5, ["*web", "*all"], now=NOW)
    assert [bot.ai_used_today(who, now=NOW) for who in ("web:a", "*web", "*all")] == [12.5, 12.5, 12.5]
    assert bot.ai_left("web:c", {"web:c": 100, "*web": 50}, now=NOW) == 50 - 12.5 - 10
    assert bot.reserve_ai("web:a", caps, 40, 5, now=NOW) is None
    dashboard.release_ai("web:a", token)
    again, cents = bot.reserve_ai("web:a", caps, 40, 5, now=NOW)
    assert cents == 50 - 12.5 - 10
    dashboard.release_ai("web:a", token)  # a late release of the old call never frees the new one
    assert dashboard.reserve_ai("web:a", caps, 40, 5, now=NOW) is None
    # A hold left by a process that died lapses: it stops counting, and the person may start again
    later = NOW + store_module.AI_HOLD_SECONDS + 1
    assert bot.ai_left("web:c", {"web:c": 100, "*web": 50}, now=later) == 50 - 12.5
    assert bot.reserve_ai("web:a", caps, 40, 5, now=later)[1] == 37.5
    # A call still running renews its hold before each request; one that has lapsed can't be renewed
    held, _ = dashboard.reserve_ai("web:d", {"web:d": 100}, 40, 5, now=NOW)
    middle = NOW + store_module.AI_HOLD_SECONDS - 1
    assert dashboard.renew_ai("web:d", held, now=middle) and not dashboard.renew_ai("web:d", "stale", now=middle)
    assert bot.reserve_ai("web:d", {"web:d": 100}, 40, 5, now=later) is None  # still busy past the first lapse
    assert not dashboard.renew_ai("web:d", held, now=middle + store_module.AI_HOLD_SECONDS)
    # Under the least a call needs, nothing is held, but the person still counts as busy
    dashboard.add_ai_usage("tg:2", 98, now=NOW)
    low = dashboard.reserve_ai("tg:2", {"tg:2": 100, "*all": 1000}, 40, 5, now=NOW)
    assert low[1] == 2 and dashboard.reserve_ai("tg:2", {"tg:2": 100}, 40, 5, now=NOW) is None
    assert bot.ai_left("tg:1", {"tg:1": 100, "*all": 1000}, now=NOW) == 100


def test_racing_ai_holds_never_pass_a_shared_cap(tmp_path) -> None:
    path = str(tmp_path / "betting.db")
    stores = [Store(path) for _ in range(3)]
    start = threading.Barrier(12)
    got: dict = {}

    def call(i: int, uid: str) -> None:
        start.wait(5)
        got[i] = stores[i % 3].reserve_ai(uid, {uid: 100, "*web": 100, "*all": 1000}, 40, 5, now=NOW)

    # Twelve password browsers at once against a 100-cent shared allowance: 40 + 40 + 20, then nothing
    threads = [threading.Thread(target=call, args=(i, f"web:racer-{i}")) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert sorted(cents for _, cents in got.values()) == [0] * 9 + [20, 40, 40]
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT SUM(cents) FROM ai_holds").fetchone()[0] == 100
    # The same person from twelve places at once: exactly one call starts
    start, got = threading.Barrier(12), {}
    threads = [threading.Thread(target=call, args=(i, "tg:7")) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(got) == 12 and sum(hold is not None for hold in got.values()) == 1


def test_a_shared_row_holds_the_gates_and_is_no_ones_account(store) -> None:
    for uid in ("web:a1", "web:b2"):
        store.get_or_create_user(uid, now=NOW)
    store.set_weekly_budget("web:a1", 4000, now=NOW)
    store.set_weekly_budget("web:b2", 2500, now=NOW - 30 * HOUR)
    store.set_weekly_budget("web:b2", 3000, now=NOW - 25 * HOUR)  # waited its 24 hours: it counts
    store.start_cooloff("web:b2", 24, now=NOW - 48 * HOUR)  # long over
    store.update_user("web:a1", {"settings": {"pause": NOW + HOUR, "theme": "dark"}})
    store.update_user("web:b2", {"settings": {"pause": NOW + 2 * HOUR}})
    shared = store.shared_user("*web", "web:", carry=("pause",), now=NOW)
    assert (shared["weekly_budget_cents"], shared["pending_budget_cents"]) == (3000, None)
    assert shared["cooloff_until"] == NOW - 24 * HOUR and shared["settings"] == {"pause": NOW + 2 * HOUR}
    # Made once: later changes to a browser's own row don't move it
    store.set_weekly_budget("web:a1", 100, now=NOW)
    assert store.shared_user("*web", "web:", carry=("pause",), now=NOW)["weekly_budget_cents"] == 3000
    # The pause only ever lengthens, in one write
    assert store.extend_setting("*web", "pause", NOW + HOUR) == NOW + 2 * HOUR
    assert store.extend_setting("*web", "pause", NOW + 5 * HOUR) == NOW + 5 * HOUR
    assert store.extend_setting("tg:404", "pause", NOW) is None
    for bad in ("*web", "*anything"):
        with pytest.raises(ValueError, match="reserved"):
            store.get_or_create_user(bad)
    with pytest.raises(ValueError):
        store.shared_user("web:x", "web:")


# --- responsible gambling ------------------------------------------------------------------------


def test_warning_signs_are_logged_and_reviewed_once(store) -> None:
    chasing = store.add_rg_event("tg:1", "chasing", {"text": "need to win back"}, action="support_message", now=NOW)
    late = store.add_rg_event("tg:1", "late_night", now=NOW + HOUR)
    theirs = store.add_rg_event("tg:2", "stake_jump", {"from": 1000, "to": 5000}, action="picks_off_24h", now=NOW)
    assert [e["id"] for e in store.rg_events()] == [late, theirs, chasing]
    first = store.rg_events("tg:1", kind="chasing")[0]
    assert first["detail"] == {"text": "need to win back"} and first["action"] == "support_message"
    assert first["ts"] == NOW and first["reviewed_at"] is None
    assert [e["id"] for e in store.rg_events(since=NOW + 1)] == [late]

    assert store.review_rg_event(chasing, now=NOW + 2 * HOUR) is True
    assert store.review_rg_event(chasing, now=NOW + 3 * HOUR) is False  # the first review's time is kept
    assert store.review_rg_event(99999, now=NOW) is False
    assert store.rg_events("tg:1", kind="chasing")[0]["reviewed_at"] == NOW + 2 * HOUR
    assert [e["id"] for e in store.rg_events(reviewed=False)] == [late, theirs]
    assert [e["id"] for e in store.rg_events(reviewed=True)] == [chasing]
    with pytest.raises(ValueError):
        store.add_rg_event("tg:1", "", now=NOW)
    # Every password browser's, for the warnings that cover them all
    web = [store.add_rg_event(uid, "chasing", now=NOW + i) for i, uid in enumerate(("web:a1", "web:b2"))]
    assert [e["id"] for e in store.rg_events(members="web:")] == web[::-1]
    assert [e["id"] for e in store.rg_events(members="web:", kind="chasing", since=NOW + 1)] == [web[1]]
