"""The shared short-term memory (data/cache.py): one load per key for everyone in a process, and for the bot and the
dashboard through one SQLite file; stale-while-revalidate; negative caching; LRU; the prune; the TTL policy; stats.

No network: every loader is a fake, and time is a fake clock unless a test needs real threads to overlap.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from decimal import Decimal

import pytest

from telegram_betting_bot_simple.data import cache as cache_module
from telegram_betting_bot_simple.data.cache import (POLICY, Policy, SecretInKey, TTLCache, all_stats, forbid_secrets,
                                                     freshest, key_text, stats_line, ttl_for)
from telegram_betting_bot_simple.data.store import Store

GAME = ("espn", "summary", "basketball_nba", "401810001")


class Clock:
    """A fake clock (epoch seconds) whose sleep moves time on."""

    def __init__(self, now=1_793_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Loader:
    """A fake fetch: counts its calls, returns its values in turn (the last one again), raises an exception value."""

    def __init__(self, *values, gate=None):
        self.values = list(values) or [{"home": 101, "away": 99}]
        self.calls = 0
        self.gate = gate  # an Event the load waits for (to keep it "in flight")
        self.started = threading.Event()

    def __call__(self):
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5), "the test never let the load finish"
        value = self.values[min(self.calls, len(self.values)) - 1]
        if isinstance(value, BaseException):
            raise value
        return value


def counts(cache, kind="espn"):
    return cache.stats()["kinds"].get(kind) or dict.fromkeys(cache_module.STAT_FIELDS, 0)


def wait_until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "betting.db")


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch):
    monkeypatch.setattr(cache_module, "_SECRETS", frozenset())


# --- one process ------------------------------------------------------------------------------------------

def test_two_people_asking_for_the_same_game_cause_one_load():
    cache = TTLCache(clock=Clock())
    gate = threading.Event()
    load = Loader(gate=gate)
    results = []
    first = threading.Thread(target=lambda: results.append(cache.get(GAME, 60, load)))
    first.start()
    assert load.started.wait(5)
    second = threading.Thread(target=lambda: results.append(cache.get(GAME, 60, load)))
    second.start()
    gate.set()
    first.join(5)
    second.join(5)
    assert results == [{"home": 101, "away": 99}] * 2 and load.calls == 1
    assert counts(cache)["misses"] == 1 and counts(cache)["hits"] == 1 and counts(cache)["loads"] == 1


def test_twenty_threads_at_once_cause_one_load():
    cache = TTLCache(clock=Clock())
    start = threading.Barrier(20)

    calls = []

    def loader():
        calls.append(1)
        time.sleep(0.05)  # long enough for every thread to arrive while it runs
        return {"home": 101}

    results = []

    def person():
        start.wait(5)
        results.append(cache.get(GAME, 60, loader))

    threads = [threading.Thread(target=person) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(results) == 20 and all(r == {"home": 101} for r in results) and len(calls) == 1
    assert counts(cache)["misses"] == 1 and counts(cache)["hits"] == 19


def test_drop_in_api_loaded_at_age_invalidate():
    clock = Clock()
    cache = TTLCache(clock=clock)
    assert cache  # an empty cache is still truthy: the clients take ``cache or TTLCache()``
    assert cache.get("k", 60, lambda: "fresh") == "fresh"
    assert cache.get("k", 60, lambda: "ignored") == "fresh"
    clock.now += 10
    assert cache.loaded_at("k") == clock.now - 10 and cache.age("k") == 10
    cache.invalidate("k")
    assert cache.loaded_at("k") is None and cache.age("k") is None
    assert cache.get("k", 60, lambda: "again") == "again"
    # The odds tests poke an expired copy straight into _entries as (monotonic, wall, value): still understood
    cache._entries["old"] = (clock.now - 7 * 3600, clock.now - 7 * 3600, ["old"])
    assert cache.get("old", 6 * 3600, lambda: ["new"]) == ["new"]


# --- the bot and the dashboard on one SQLite file -----------------------------------------------------------

def test_the_bot_and_the_dashboard_share_one_load(db):
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    bot = TTLCache(Store(db), name="bot", clock=clock)  # another process: its own Store, the same file
    dashboard_load, bot_load = Loader({"home": 101}), Loader({"home": 999})
    assert dashboard.get(GAME, 60, dashboard_load) == {"home": 101}
    clock.now += 20
    value, fetched_at = bot.get_with_time(GAME, 60, bot_load)
    assert value == {"home": 101} and fetched_at == clock.now - 20 and bot_load.calls == 0
    assert bot.age(GAME) == 20  # as old as the dashboard's fetch, not new
    assert counts(bot)["shared_hits"] == 1 and counts(bot)["loads"] == 0
    clock.now += 50  # both copies are past 60 s: the bot refreshes, the dashboard then takes the bot's copy
    bot_load.values = [{"home": 102}]
    assert bot.get(GAME, 60, bot_load) == {"home": 102} and bot_load.calls == 1
    assert dashboard.get(GAME, 60, dashboard_load) == {"home": 102} and dashboard_load.calls == 1


def test_the_second_process_waits_for_the_first_ones_load(db):
    dashboard = TTLCache(Store(db), name="dashboard", lease_poll=0.01)
    bot = TTLCache(Store(db), name="bot", lease_poll=0.01)
    gate = threading.Event()
    dashboard_load, bot_load = Loader({"home": 101}, gate=gate), Loader({"home": 999})
    results = {}
    first = threading.Thread(target=lambda: results.setdefault("dashboard", dashboard.get(GAME, 60, dashboard_load)))
    first.start()
    assert dashboard_load.started.wait(5)  # the dashboard holds the lease and is loading
    second = threading.Thread(target=lambda: results.setdefault("bot", bot.get(GAME, 60, bot_load)))
    second.start()
    wait_until(lambda: counts(bot)["lease_waits"] == 1)  # the bot found the lease taken and is polling the table
    gate.set()
    first.join(5)
    second.join(5)
    assert results == {"dashboard": {"home": 101}, "bot": {"home": 101}}
    assert dashboard_load.calls == 1 and bot_load.calls == 0
    assert counts(bot)["shared_hits"] == 1 and counts(bot)["misses"] == 0


def test_a_value_saved_just_before_the_claim_is_used_not_loaded_again(db):
    """The other process saves its value and lets go of the lease between this one's look at the table and its
    claim: the claim looks once more and uses that value (two real processes did load some keys twice without it)."""
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)

    class Racy(Store):
        armed = True

        def cache_lease(self, *args):
            if self.armed:
                self.armed = False
                dashboard.get(GAME, 60, Loader({"home": 101}))  # the dashboard's load finishes right now
            return super().cache_lease(*args)

    bot = TTLCache(Racy(db), name="bot", clock=clock)
    load = Loader({"home": 999})
    assert bot.get(GAME, 60, load) == {"home": 101} and load.calls == 0
    assert counts(bot)["shared_hits"] == 1 and counts(bot)["misses"] == 0
    assert Store(db).cache_lease(key_text(GAME), "a third", 30, clock.now)  # the bot let its claim go again


def test_when_the_holder_fails_the_waiting_process_loads_itself(db):
    dashboard = TTLCache(Store(db), name="dashboard", lease_poll=0.01)
    bot = TTLCache(Store(db), name="bot", lease_poll=0.01)
    gate = threading.Event()
    failing, bot_load = Loader(RuntimeError("ESPN returned HTTP 503"), gate=gate), Loader({"home": 101})
    errors = []

    def dashboard_reads():
        try:
            dashboard.get(GAME, 60, failing)
        except RuntimeError as exc:
            errors.append(exc)

    first = threading.Thread(target=dashboard_reads)
    first.start()
    assert failing.started.wait(5)
    results = []
    second = threading.Thread(target=lambda: results.append(bot.get(GAME, 60, bot_load)))
    second.start()
    wait_until(lambda: counts(bot)["lease_waits"] == 1)
    gate.set()  # the dashboard's load fails and its lease is released: the bot takes it and loads
    first.join(5)
    second.join(5)
    assert len(errors) == 1 and results == [{"home": 101}] and bot_load.calls == 1


def test_a_wait_that_runs_out_loads_itself_or_serves_the_last_good_copy(db):
    clock = Clock()
    store = Store(db)
    bot = TTLCache(Store(db), name="bot", clock=clock, sleep=clock.sleep, lease_wait=2)
    other = "dashboard:elsewhere"
    assert store.cache_lease(key_text(GAME), other, 30, clock.now)  # a load that never ends
    load = Loader({"home": 101}, {"home": 102})
    assert bot.get(GAME, 60, load) == {"home": 101} and load.calls == 1  # waited 2 s, then loaded itself
    assert clock.now >= 1_793_000_002
    clock.now += 61
    assert store.cache_lease(key_text(GAME), other, 30, clock.now)
    assert bot.get(GAME, 60, load) == {"home": 101} and load.calls == 1  # had a copy: served it after the wait


# --- stale-while-revalidate ------------------------------------------------------------------------------

def test_stale_while_revalidate_serves_the_old_copy_at_once_and_refreshes_once():
    clock = Clock()
    cache = TTLCache(clock=clock)
    assert cache.get(GAME, 20, Loader({"home": 1}), stale=20) == {"home": 1}
    clock.now += 25  # past the TTL, inside the stale window
    gate = threading.Event()
    refresh = Loader({"home": 2}, gate=gate)
    for _ in range(5):
        assert cache.get(GAME, 20, refresh, stale=20) == {"home": 1}  # at once, while the refresh is held up
    assert refresh.started.wait(5)
    gate.set()
    assert cache.drain(5)
    assert refresh.calls == 1
    assert cache.get(GAME, 20, refresh, stale=20) == {"home": 2} and refresh.calls == 1
    row = counts(cache)
    assert row["stale_hits"] == 5 and row["refreshes"] == 1 and row["hits"] == 1 and row["loads"] == 2
    clock.now += 41  # past TTL + stale: the reader waits for the load
    assert cache.get(GAME, 20, Loader({"home": 3}), stale=20) == {"home": 3}


def test_a_failed_background_refresh_keeps_the_copy_and_backs_off(caplog):
    clock = Clock()
    cache = TTLCache(clock=clock)
    cache.get(GAME, 20, Loader({"home": 1}), stale=60)
    clock.now += 25
    broken = Loader(RuntimeError("ESPN returned HTTP 502 for https://site.api.espn.com/x?event=1"))
    with caplog.at_level(logging.WARNING):
        assert cache.get(GAME, 20, broken, stale=60) == {"home": 1}
        assert cache.drain(5)
    assert broken.calls == 1 and "serving the cached copy" in caplog.text and "?event" not in caplog.text
    assert cache.get(GAME, 20, broken, stale=60) == {"home": 1}
    assert cache.drain(5) and broken.calls == 1  # backing off: no second refresh within 30 s
    assert cache.failing(GAME)[0] == 1


def test_a_refresh_takes_the_other_process_copy_instead_of_fetching(db):
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    bot = TTLCache(Store(db), name="bot", clock=clock)
    bot.get(GAME, 20, Loader({"home": 1}), stale=20)
    clock.now += 25
    dashboard_load = Loader({"home": 2})
    assert dashboard.get(GAME, 20, dashboard_load, stale=20) == {"home": 1}  # the bot's copy, stale: served ...
    assert dashboard.drain(5) and dashboard_load.calls == 1  # ... while the dashboard refreshed it
    refresh = Loader({"home": 3})
    assert bot.get(GAME, 20, refresh, stale=20) == {"home": 1}  # stale in the bot's memory: served at once
    assert bot.drain(5) and refresh.calls == 0  # the refresh found the dashboard's fresh copy in the table
    assert bot.get(GAME, 20, refresh, stale=20) == {"home": 2}


# --- failures ---------------------------------------------------------------------------------------------

def test_a_failure_with_nothing_cached_is_remembered_for_error_ttl():
    clock = Clock()
    cache = TTLCache(clock=clock)
    broken = Loader(RuntimeError("down"), {"home": 101})
    with pytest.raises(RuntimeError):
        cache.get(GAME, 60, broken, error_ttl=30)
    with pytest.raises(RuntimeError, match="down"):
        cache.get(GAME, 60, broken, error_ttl=30)  # re-raised without trying
    assert broken.calls == 1 and counts(cache)["negative_hits"] == 1 and counts(cache)["load_errors"] == 1
    clock.now += 31
    assert cache.get(GAME, 60, broken, error_ttl=30) == {"home": 101} and broken.calls == 2
    assert cache.failing(GAME) is None


def test_failures_in_a_row_double_the_wait_only_with_a_cap_and_error_ttl_zero_retries():
    clock = Clock()
    cache = TTLCache(clock=clock)
    broken = Loader(RuntimeError("down"))
    cap = TTLCache.ERROR_CAP  # the markets Keeper's rule: 30 s doubling to 15 minutes
    for wait in (30, 60):  # 30 s after the first failure, 60 s after the second
        with pytest.raises(RuntimeError):
            cache.get(GAME, 60, broken, error_ttl=30, backoff_cap=cap)
        assert cache.failing(GAME)[1] == wait
        clock.now += wait
    assert cache.failing(GAME) is None and broken.calls == 2
    with pytest.raises(RuntimeError):
        cache.get(GAME, 60, broken, error_ttl=30, backoff_cap=cap)  # the third in a row: 4 x 30 s
    assert cache.failing(GAME) == (3, 120)
    for _ in range(10):
        clock.now += cache.failing(GAME)[1]
        with pytest.raises(RuntimeError):
            cache.get(GAME, 60, broken, error_ttl=30, backoff_cap=cap)
    assert cache.failing(GAME)[1] == cap
    other = Loader(RuntimeError("down"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            cache.get("no-memory", 60, other, error_ttl=0)
    assert other.calls == 3


def test_without_a_cap_a_failure_is_remembered_for_error_ttl_however_many_in_a_row():
    # The spec's error_ttl: a live summary whose source is back is fetched within error_ttl, not 15 minutes later
    clock = Clock()
    cache = TTLCache(clock=clock)
    key = ("espn", "summary", "basketball_nba", "401")
    broken = Loader(*([RuntimeError("down")] * 8), {"home": 1})
    tries = []
    for _ in range(8):
        with pytest.raises(RuntimeError):
            cache.get(key, 20, broken, stale=20)
        tries.append(cache.failing(key))
        clock.now += 30
    assert tries == [(n, 30) for n in range(1, 9)] and broken.calls == 8
    assert cache.get(key, 20, broken, stale=20) == {"home": 1}  # back: answered 30 s after the last failure
    # With a last good copy too: the copy is served meanwhile and the source asked again every error_ttl
    clock.now += 41
    down = Loader(RuntimeError("down"))
    for _ in range(4):
        assert cache.get(key, 20, down) == {"home": 1}
        assert cache.failing(key)[1] == 30
        clock.now += 30
    assert down.calls == 4


def test_a_failure_with_a_last_good_value_serves_it(caplog):
    clock = Clock()
    cache = TTLCache(clock=clock)
    assert cache.get(GAME, 60, Loader({"home": 101})) == {"home": 101}
    clock.now += 61
    broken = Loader(RuntimeError("ESPN returned HTTP 503"))
    with caplog.at_level(logging.WARNING):
        assert cache.get(GAME, 60, broken) == {"home": 101}
    assert "serving the cached copy" in caplog.text
    # Until the failure's wait (30 s) is over every reader gets the copy at once, without trying again: a down
    # source isn't hammered and nobody queues behind a failing fetch
    for _ in range(5):
        assert cache.get(GAME, 60, broken) == {"home": 101}
    assert broken.calls == 1 and counts(cache)["stale_hits"] == 5
    clock.now += 31  # the wait is over: the next reader tries once; the second failure in a row waits 30 s again
    assert cache.get(GAME, 60, broken) == {"home": 101} and broken.calls == 2
    assert cache.get(GAME, 60, broken) == {"home": 101} and broken.calls == 2
    assert cache.failing(GAME) == (2, 30)
    # backoff=False: the next reader past the TTL tries again at once (each waiting for the fetch)
    assert cache.get(GAME, 60, broken, backoff=False) == {"home": 101} and broken.calls == 3
    assert cache.failing(GAME) == (3, 30)
    # The source is back: once the wait is over the next reader gets a fresh copy
    clock.now += 31
    assert cache.get(GAME, 60, Loader({"home": 102})) == {"home": 102} and cache.failing(GAME) is None


def test_a_copy_newer_than_the_failure_is_judged_on_its_own(db):
    clock = Clock()
    bot = TTLCache(Store(db), name="bot", clock=clock)
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    with pytest.raises(RuntimeError):
        bot.get(GAME, 60, Loader(RuntimeError("down")))
    clock.now += 1
    dashboard.get(GAME, 60, Loader({"home": 101}))  # the other process got through
    assert bot.get(GAME, 60, Loader(RuntimeError("down"))) == {"home": 101}  # remembered failure, but a copy now


# --- memory bounds ----------------------------------------------------------------------------------------

def test_lru_eviction_and_lock_cleanup():
    cache = TTLCache(max_entries=3, clock=Clock())
    for key in "abc":
        cache.get(key, 60, lambda key=key: key.upper())
    assert cache.get("a", 60, lambda: pytest.fail("cached")) == "A"  # "a" is now the most recently used
    cache.get("d", 60, lambda: "D")
    assert cache.loaded_at("b") is None and cache.stats()["entries"] == 3  # the least recently used went
    assert set(cache._key_locks) <= {"a", "c", "d"}
    with pytest.raises(RuntimeError):
        cache.get("x", 60, Loader(RuntimeError("down")))
    assert "x" not in cache._key_locks  # a failed key keeps no lock
    assert cache.stats()["total"]["evictions"] == 1
    big = TTLCache(max_entries=100, clock=Clock())
    for n in range(5000):
        big.get(("player", n), 60, lambda n=n: n)
    assert big.stats()["entries"] == 100 and len(big._key_locks) <= 100
    assert big.stats()["kinds"]["player"]["evictions"] == 4900


# --- what goes to the shared table --------------------------------------------------------------------------

def test_values_that_would_not_come_back_the_same_stay_in_memory(db):
    store = Store(db)
    cache = TTLCache(store, clock=Clock())
    unchanged = {"a": [1, 2.5, None, True, "x"], "b": {"c": float("nan")}}
    cases = {("v", "tuple"): (1, 2), ("v", "set"): {1, 2}, ("v", "int keys"): {1: "a"},
             ("v", "decimal"): Decimal("1.91"), ("v", "date"): datetime.date(2026, 10, 27),
             ("v", "nested tuple"): {"a": [(1, 2)]}}
    for key, value in cases.items():
        assert cache.get(key, 60, lambda value=value: value) == value
        assert store.cache_get(key_text(key)) is None, key
    assert cache.get(("v", "json"), 60, lambda: unchanged) is unchanged
    assert store.cache_get(key_text(("v", "json")))
    cache.get(("v", "lossy"), 60, lambda: (1, 2), persist=True)  # the keeper's rule: tuples saved as lists
    other = TTLCache(store, clock=Clock())
    assert other.get(("v", "lossy"), 60, lambda: pytest.fail("shared")) == [1, 2]
    cache.get(("v", "private"), 60, lambda: {"a": 1}, persist=False)
    assert store.cache_get(key_text(("v", "private"))) is None
    weird = ("v", object())  # a key with no stable text: memory only
    assert cache.get(weird, 60, lambda: {"a": 1}) == {"a": 1} and cache.get(weird, 60, lambda: 2) == {"a": 1}


def test_shared_rows_are_gzip_json_with_their_times(db):
    import gzip
    import json

    clock = Clock()
    store = Store(db)
    cache = TTLCache(store, clock=clock)
    cache.get(GAME, 20, lambda: {"home": 101}, stale=20, kind="summary")
    fetched_at, expires_at, stale_until, blob = store.cache_get(key_text(GAME))
    assert (fetched_at, expires_at, stale_until) == (clock.now, clock.now + 20, clock.now + 40)
    assert json.loads(gzip.decompress(blob)) == {"home": 101}
    assert store.cache_summary()["kinds"] == {"summary": 1}


def test_the_prune(db):
    clock = Clock()
    store = Store(db)
    cache = TTLCache(store, clock=clock)
    start = clock.now
    cache.get(("a",), 60, lambda: 1, stale=60)  # stale_until start + 120
    cache.get(("b",), 3600, lambda: 2)  # stale_until start + 3600
    assert store.cache_lease(key_text(("c",)), "someone", 30, clock.now)
    clock.now = start + 120 + 86400 + 1
    assert cache.prune() == 1
    assert store.cache_get(key_text(("a",))) is None and store.cache_get(key_text(("b",))) is not None
    assert store.cache_lease(key_text(("c",)), "me", 30, clock.now)  # the lapsed lease went too
    writes = TTLCache(store, clock=clock)
    writes.PRUNE_WRITES, writes.PRUNE_EVERY = 3, 30 * 86400  # by writes only
    writes.get(("d",), 60, lambda: 4)  # its first write prunes what an earlier run left
    clock.now += 2 * 86400
    writes.get(("e",), 60, lambda: 5)
    writes.get(("f",), 60, lambda: 6)
    assert store.cache_get(key_text(("b",))) is not None  # two writes since the last prune: not yet
    writes.get(("g",), 60, lambda: 7)  # the third: pruned
    assert store.cache_get(key_text(("b",))) is None and store.cache_get(key_text(("d",))) is None
    assert store.cache_get(key_text(("g",))) is not None


def test_a_broken_store_leaves_the_memory_tier_working(caplog):
    class Broken:
        def __getattr__(self, name):
            if name.startswith("cache_"):
                def fail(*args, **kwargs):
                    raise OSError("disk I/O error")
                return fail
            raise AttributeError(name)

    cache = TTLCache(Broken(), clock=Clock())
    load = Loader({"home": 101})
    with caplog.at_level(logging.WARNING):
        assert cache.get(GAME, 60, load) == {"home": 101} and cache.get(GAME, 60, load) == {"home": 101}
    assert load.calls == 1 and "using memory only" in caplog.text


# --- invalidation -----------------------------------------------------------------------------------------

def test_a_status_change_invalidates_the_games_computed_entries_for_everyone(db):
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    bot = TTLCache(Store(db), name="bot", clock=clock)
    card, consensus, other = ("game_card", "nba", "401"), ("consensus", "nba", "401"), ("game_card", "nba", "4010")
    for key in (card, consensus, other):
        dashboard.get(key, 60, lambda key=key: {"built": key[0], "status": "pre"}, kind="game_card")
    assert dashboard.invalidate_prefix(("game_card", "nba", "401"), ("consensus", "nba", "401")) == 2
    rebuild = Loader({"built": "game_card", "status": "live"})
    assert bot.get(card, 60, rebuild) == {"built": "game_card", "status": "live"}  # rebuilt once ...
    assert dashboard.get(card, 60, rebuild) == {"built": "game_card", "status": "live"} and rebuild.calls == 1
    assert bot.get(other, 60, lambda: pytest.fail("4010 is another game")) == {"built": "game_card", "status": "pre"}
    assert Store(db).cache_get(key_text(consensus)) is None
    dashboard.invalidate(other)
    assert Store(db).cache_get(key_text(other)) is None


# --- keys -------------------------------------------------------------------------------------------------

def test_key_text_is_stable_json():
    day = datetime.date(2026, 10, 27)
    assert key_text(("espn", "scoreboard", "basketball_nba", day)) == \
        '["espn","scoreboard","basketball_nba",{"date":"2026-10-27"}]'
    assert key_text(("a", 1, 1.0, 2.5, None, True, ("x",))) == '["a",1,1,2.5,null,true,["x"]]'
    assert key_text("k") == '"k"' and key_text(frozenset({"b", "a"})) == key_text(frozenset({"a", "b"}))
    with pytest.raises(TypeError):
        key_text(("a", ["list"]))


def test_a_key_with_a_secret_is_refused_and_secrets_never_reach_the_log(db, caplog):
    secret = "0123456789abcdef0123456789abcdef"
    forbid_secrets(secret, "", "short")
    cache = TTLCache(Store(db), clock=Clock())
    with pytest.raises(SecretInKey) as refused:
        cache.get(("odds", f"https://api.the-odds-api.com/v4/sports?apiKey={secret}"), 60, Loader())
    assert secret not in str(refused.value)
    clock = Clock()
    memory = TTLCache(clock=clock)
    memory._entries[("odds", "sports")] = (clock() - 7200, clock() - 7200, [])  # an expired copy
    broken = Loader(RuntimeError(f"failed: /v4/sports?apiKey={secret} and {secret}"))
    with caplog.at_level(logging.INFO):
        assert memory.get(("odds", "sports"), 60, broken) == []
    assert "serving the cached copy" in caplog.text and secret not in caplog.text


# --- the policy table -----------------------------------------------------------------------------------------

def test_the_policy_table():
    assert ttl_for("summary", "live") == ttl_for("summary", "in") == ttl_for("boxscore", "live") == (20, 20)
    assert ttl_for("summary", "pre") == (300, 600) and ttl_for("summary", "post") == (86400, 0)
    assert ttl_for("player", "gameday") == (3600, 3600) and ttl_for("player") == ttl_for("gamelog") == (6 * 3600, 3600)
    assert ttl_for("splits") == (6 * 3600, 6 * 3600)
    assert ttl_for("injuries") == ttl_for("news") == (600, 600)
    assert ttl_for("standings") == (3600, 0) and ttl_for("roster") == (3 * 3600, 0)
    assert ttl_for("markets") == (300, 300) and ttl_for("markets", "live") == (60, 300)
    assert ttl_for("ratings") == (12 * 3600, 12 * 3600) and ttl_for("predictor") == (3600, 0)
    from telegram_betting_bot_simple.config import Config
    fields = Config.model_fields
    for state, field in (("live", "slate_live_seconds"), ("today", "slate_today_seconds"),
                         ("idle", "slate_idle_seconds"), ("future", "slate_future_seconds")):
        assert ttl_for("scoreboard", state).ttl == fields[field].default  # the slate's own intervals
    assert ttl_for("summary", "postponed") == ttl_for("summary")  # an unknown state: the kind's default
    with pytest.raises(KeyError):
        ttl_for("sumary")
    assert freshest(ttl_for("markets", "live"), ttl_for("summary", "pre")) == Policy(60, 300)
    assert all(None in table and all(p.ttl > 0 and p.stale >= 0 for p in table.values()) for table in POLICY.values())


def test_get_for_uses_the_policy_and_counts_the_kind():
    clock = Clock()
    cache = TTLCache(clock=clock)
    load = Loader({"home": 1}, {"home": 2})
    assert cache.get_for("summary", GAME, load, state="live") == {"home": 1}
    clock.now += 25  # past 20 s, inside the 20 s stale window
    assert cache.get_for("summary", GAME, load, state="live") == {"home": 1}
    assert cache.drain(5) and load.calls == 2
    assert counts(cache, "summary")["stale_hits"] == 1 and counts(cache, "summary")["refreshes"] == 1


# --- stats ----------------------------------------------------------------------------------------------

def test_stats_count_every_read(db):
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    bot = TTLCache(Store(db), name="bot-stats", clock=clock)
    dashboard.get(GAME, 60, Loader(), kind="summary")             # miss, load
    dashboard.get(GAME, 60, Loader(), kind="summary")             # hit
    bot.get(GAME, 60, Loader(), kind="summary")                   # shared hit
    with pytest.raises(RuntimeError):
        bot.get(("x",), 60, Loader(RuntimeError("down")), kind="news")  # miss, load error
    with pytest.raises(RuntimeError):
        bot.get(("x",), 60, Loader(), kind="news")                # negative hit
    clock.now += 70
    bot.get(GAME, 60, Loader(), stale=30, kind="summary")         # stale hit, background refresh
    assert bot.drain(5)
    summary = dashboard.stats()["kinds"]["summary"]
    assert (summary["misses"], summary["loads"], summary["hits"]) == (1, 1, 1)
    bot_stats = bot.stats(table=True)
    assert bot_stats["kinds"]["summary"]["shared_hits"] == 1 and bot_stats["kinds"]["summary"]["stale_hits"] == 1
    assert bot_stats["kinds"]["summary"]["refreshes"] == 1 and bot_stats["kinds"]["summary"]["loads"] == 1
    news = bot_stats["kinds"]["news"]
    assert (news["misses"], news["load_errors"], news["negative_hits"]) == (1, 1, 1)
    assert bot_stats["total"]["loads"] == 2 and bot_stats["shared"] and bot_stats["table"]["rows"] == 1
    assert bot_stats["entries"] == 1 and bot_stats["failing"] == 1
    everything = all_stats()
    assert "bot-stats" in [c["name"] for c in everything["caches"]]
    assert everything["kinds"]["news"]["negative_hits"] >= 1
    line = stats_line(bot_stats)
    assert line.startswith("cache: 4 reads,") and "1 misses" in line and "2 loads (1 failed" in line


# --- computed results built while a source failed ---------------------------------------------------------

def test_a_partial_computed_result_is_shared_for_partial_keep_seconds_then_rebuilt(db):
    clock = Clock()
    dashboard = TTLCache(Store(db), name="dashboard", clock=clock)
    bot = TTLCache(Store(db), name="bot", clock=clock)
    card = ("game_card", "basketball_nba", "g1", "sig")
    build = Loader({"card": 1, "partial": True}, {"card": 2, "partial": False})
    partial = lambda value: value["partial"]  # noqa: E731
    for _ in range(3):  # people in the dashboard meanwhile share the one build ...
        assert dashboard.get_computed(card, 60, build, partial)["card"] == 1
    assert bot.get_computed(card, 60, build, partial)["card"] == 1  # ... and so does the bot, through the table
    assert build.calls == 1
    clock.now += cache_module.PARTIAL_KEEP  # then the next reader rebuilds it: whole again
    assert bot.get_computed(card, 60, build, partial)["card"] == 2 and build.calls == 2
    # The dashboard's partial copy is as old: dropped too, and the bot's whole one comes from the table, unbuilt
    assert dashboard.get_computed(card, 60, build, partial)["card"] == 2 and build.calls == 2
    whole = Loader({"card": 3, "partial": False})
    other = ("game_card", "basketball_nba", "g2", "sig")
    dashboard.get_computed(other, 60, whole, partial)
    clock.now += 30
    assert dashboard.get_computed(other, 60, whole, partial)["card"] == 3 and whole.calls == 1  # whole: its TTL
