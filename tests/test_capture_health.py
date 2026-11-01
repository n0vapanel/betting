"""v4 P1, contract C (p1:prices): the Brain progress page's "Capture" health block and its card.

learn/progress.py ``capture_health`` reads raw.db (capture/raw_store.py) on a read-only connection with plain SQL,
and never creates it. It reports:
- coverage per sport over the last 7 days: finals with a raw final stored within 36 h of the end, out of all finals
  (void jobs aren't finals, games still being played aren't yet, and finals still inside their 36 h are "pending");
- raw.db's size and the disk's free space;
- the boards the nightly sweep couldn't read (capture_state "sweep:<day>"): their games are left out unseen, so the
  coverage of their sport and the total is unknown, with an alarm;
- the day's requests and the backfill's latest night (in its own time zone, capture/backfill.py backfill_tz);
- forward capture's HTTP error share (http_ok/http_error; or the stored answers when there are no counters), its
  alarm on the two UTC days together or either alone, and the backfill's own share (backfill_http_*) and alarm;
- the older-games backfill per league and season (backfill_progress);
- the latest errors;
- alarms for those, a payload-shape change in any league, and the disk at 70% or more (or under the floor);
- and every query on capture_jobs and raw_payloads goes by an index (EXPLAIN QUERY PLAN shows no SCAN).

The card (dashboard/static/js/views/brain/progress.js ``captureHtml``) is drawn by the app's own ES modules in Node,
as tests/test_p0_fix_frontend.py does: the whole Brain page from the real progress() payload, and cases that check
the escaping, the words and the empty states. Where Playwright and Chromium are installed, the card is also laid out in
Chromium at 360 px to check nothing overflows. No network, no server.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sqlite3
import subprocess
from contextlib import closing
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Tuple

import pytest

from telegram_betting_bot_simple.capture import raw_store as raw_store_module
from telegram_betting_bot_simple.capture.raw_store import DISK_FLOOR_SHARE, Job, RawStore, disk_room, min_free_bytes
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import progress, registry

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "telegram_betting_bot_simple" / "dashboard" / "static"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
NODE = shutil.which("node")
NOW = datetime.datetime(2026, 11, 1, 18, 0, tzinfo=datetime.timezone.utc).timestamp()
HOUR, DAY, GB = 3600.0, 86400.0, 1 << 30  # GB as the page counts it: 1024-based, as df -h
TODAY, YESTERDAY = "2026-11-01", "2026-10-31"
NBA, NFL, NHL, MLB, EPL = "basketball_nba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb", "soccer_epl"
UTC = datetime.timezone.utc


class Usage(NamedTuple):
    total: int
    used: int
    free: int


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("RAW_DB_PATH", "CAPTURE_ENABLED", "CAPTURE_PER_DAY", "CAPTURE_BACKFILL_PER_NIGHT",
                 "CAPTURE_BACKFILL_SEASONS"):
        monkeypatch.delenv(name, raising=False)
    progress._shape_cache.clear()
    registry.bind(None)
    yield
    registry.bind(None)


@pytest.fixture(autouse=True)
def disk(monkeypatch) -> Dict[str, Usage]:
    """The disk raw.db is on, as shutil.disk_usage reports it to the block and to capture/raw_store.py's disk_room:
    plenty of room (10% full) whatever this machine's disk is, unless a test says otherwise."""
    state = {"usage": Usage(int(100e9), int(10e9), int(90e9))}
    monkeypatch.setattr(raw_store_module.shutil, "disk_usage", lambda path: state["usage"])
    monkeypatch.setattr(progress.shutil, "disk_usage", lambda path: state["usage"])
    return state


def summary(**extra: Any) -> bytes:
    return json.dumps({"header": {"id": "1"}, "boxscore": {"teams": []}, **extra}).encode()


def put(raw: RawStore, key: str, league: str, at: float, *, source: str = "espn_site", endpoint: str = "summary",
        state: str = "final", status: int = 200, body: Optional[bytes] = None,
        params: Optional[Mapping[str, Any]] = None) -> int:
    return raw.put(game_key=key, league=league, source=source, endpoint=endpoint, params=params or {},
                   body=summary(key=key, at=at) if body is None else body, state=state, http_status=status,
                   horizon="final" if state == "final" else "t24", fetched_at=at)


def job(raw: RawStore, key: str, league: str, start: float, *, state: str = "todo", kinds: Tuple[str, ...] = (),
        error: Optional[str] = None, updated: Optional[float] = None) -> None:
    raw.save_job(Job(key, league, key.rsplit(":", 1)[-1], start, start + 4 * HOUR, state=state, kinds_done=list(kinds),
                     last_error=error), now=start + 5 * HOUR if updated is None else updated)


def health(path: Path, now: float = NOW, **kwargs: Any) -> Dict[str, Any]:
    block = progress.capture_health(str(path), now, **kwargs)
    json.dumps(block)  # plain JSON
    return block


# --- the block -----------------------------------------------------------------------------------------------------

def test_no_raw_db_yet_is_nothing_stored_and_none_is_created(tmp_path, disk) -> None:
    block = health(tmp_path / "raw.db")
    assert block["stored"] is False and block["size"] is None and block["alarms"] == []
    assert block["coverage"]["total"] == {"finals": 0, "covered": 0, "missed": 0, "pending": 0, "share": None}
    assert list(tmp_path.iterdir()) == []  # never created
    assert progress.capture_health(None, NOW)["stored"] is False


def test_coverage_counts_finals_stored_within_36_hours_of_the_end_per_sport(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    start = NOW - 2 * DAY
    job(raw, "espn:1", NBA, start, state="done")
    put(raw, "espn:1", NBA, start + 4 * HOUR)  # 30 min after its end: covered
    job(raw, "espn:2", NBA, NOW - 3 * DAY, state="done")
    put(raw, "espn:2", NBA, NOW - 3 * DAY + 3.5 * HOUR + 37 * HOUR)  # 37 h after its end: late
    job(raw, "espn:3", NHL, NOW - 4 * DAY, state="gave_up")  # never stored
    job(raw, "espn:4", EPL, NOW - 10 * HOUR)  # ended 7 h ago, not stored yet: still inside its 36 h
    job(raw, "espn:5", MLB, NOW - HOUR)  # still being played: not a final yet
    job(raw, "espn:6", NBA, NOW - 2 * DAY, state="void")  # called off: not a final
    job(raw, "static:greco1899:csvs", "mma_mixed_martial_arts", NOW - DAY)  # not a game
    # A tennis match's final is its day board, stored under the board's own key: the job says it got it
    job(raw, "espn-match:9", "tennis_atp", NOW - 2 * DAY, state="done", kinds=("final:scoreboard",),
        updated=NOW - 2 * DAY + 7 * HOUR)
    job(raw, "espn:7", NFL, NOW - 8 * DAY, state="done")  # before the week
    put(raw, "espn:7", NFL, NOW - 8 * DAY + 5 * HOUR)
    # Only a pre-start copy and a 404 at the final: not covered, and still inside its 36 h
    job(raw, "espn:8", NFL, NOW - 20 * HOUR, state="partial")
    put(raw, "espn:8", NFL, NOW - 21 * HOUR, state="pre")
    put(raw, "espn:8", NFL, NOW - 14 * HOUR, status=404, body=b'{"code": 404}')
    block = health(tmp_path / "raw.db")
    rows = {row["sport"]: row for row in block["coverage"]["sports"]}
    assert [row["sport"] for row in block["coverage"]["sports"]] == ["basketball", "football", "hockey", "soccer",
                                                                     "tennis"]
    assert rows["basketball"] == {"sport": "basketball", "label": "Basketball", "finals": 2, "covered": 1, "missed": 1,
                                  "pending": 0, "share": 0.5, "unread": 0}
    assert (rows["hockey"]["missed"], rows["hockey"]["share"]) == (1, 0.0)
    assert (rows["soccer"]["pending"], rows["soccer"]["share"]) == (1, None)
    assert (rows["football"]["finals"], rows["football"]["pending"]) == (1, 1)
    assert (rows["tennis"]["covered"], rows["tennis"]["share"]) == (1, 1.0)
    assert block["coverage"]["total"] == {"finals": 6, "covered": 2, "missed": 2, "pending": 2, "share": 0.5}
    assert (block["coverage"]["days"], block["coverage"]["within_hours"], block["coverage"]["target"]) == (7, 36, 0.98)
    # Pending finals turn into misses once their 36 h pass
    later = health(tmp_path / "raw.db", NOW + 2 * DAY)
    assert {row["sport"]: row["missed"] for row in later["coverage"]["sports"]}["soccer"] == 1


def test_raw_db_size_and_the_disk_alarm_at_70_percent(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    disk["usage"] = Usage(100 * GB, 69 * GB, 31 * GB)
    files = sum(os.path.getsize(p) for p in tmp_path.iterdir() if p.name.startswith("raw.db"))  # with its WAL files
    block = health(tmp_path / "raw.db")
    assert block["size"] == {"raw_bytes": files, "disk_total": 100 * GB, "disk_free": 31 * GB, "disk_full": 0.69,
                             "floor_bytes": 10 * GB, "under_floor": False, "full_until": None, "full_minutes": 5,
                             "paused": False}
    assert block["alarms"] == []
    disk["usage"] = Usage(100 * GB, 66 * GB, 30 * GB)  # reserved blocks count as used: 70% is (total - free) / total
    block = health(tmp_path / "raw.db")
    assert block["size"]["disk_full"] == 0.7
    assert block["alarms"] == [{"kind": "disk", "text": f"The disk is 70% full: 30.0 GB free, raw.db is "
                                                        f"{progress._bytes_text(block['size']['raw_bytes'])}."}]


def test_under_the_disk_floor_the_alarm_says_capture_and_the_backfill_are_paused(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    disk["usage"] = Usage(20 * GB, int(18.6 * GB), int(1.4 * GB))  # the floor: max(2 GiB, 10% of 20 GiB) = 2 GiB
    block = health(tmp_path / "raw.db")
    assert (block["size"]["floor_bytes"], block["size"]["paused"]) == (2 * GB, True)
    assert block["alarms"] == [{"kind": "disk", "text": "The disk is 93% full: 1.4 GB free. Capture and the backfill are "
                                                        "paused until 2.0 GB is free."}]
    # It fires below the 70% rule too: a small disk under the 2 GiB floor
    disk["usage"] = Usage(4 * GB, int(2.4 * GB), int(1.6 * GB))
    block = health(tmp_path / "raw.db")
    assert block["size"]["disk_full"] == 0.6 and block["alarms"] == [
        {"kind": "disk", "text": "The disk is 60% full: 1.6 GB free. Capture and the backfill are paused until 2.0 GB "
                                 "is free."}]
    # A big disk's floor is its 10%; CAPTURE_MIN_FREE_GB raises the 2 GiB
    disk["usage"] = Usage(500 * GB, int(460.5 * GB), int(39.5 * GB))
    assert health(tmp_path / "raw.db")["alarms"][0]["text"].endswith("paused until 50.0 GB is free.")
    disk["usage"] = Usage(20 * GB, int(16.5 * GB), int(3.5 * GB))
    block = health(tmp_path / "raw.db", settings={"min_free_gb": 4})
    assert block["size"]["floor_bytes"] == 4 * GB and block["alarms"][0]["text"].endswith("paused until 4.0 GB is free.")


@pytest.mark.parametrize("free, full", [(2.5, "88%"), (3.0, "85%")])  # above the floor
def test_above_the_disk_floor_nothing_says_paused(tmp_path, disk, free, full) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    disk["usage"] = Usage(20 * GB, int((20 - free) * GB), int(free * GB))
    block = health(tmp_path / "raw.db")
    assert block["size"]["paused"] is False
    [alarm] = block["alarms"]  # still over 70% full: the plain disk alarm
    assert alarm["text"].startswith(f"The disk is {full} full: {free:.1f} GB free, raw.db is ")
    assert "paused" not in json.dumps(block["alarms"])
    disk["usage"] = Usage(20 * GB, 10 * GB, 10 * GB)
    assert health(tmp_path / "raw.db")["alarms"] == []


@pytest.mark.parametrize("total_gb", [4, 20, 100, 500])
@pytest.mark.parametrize("gb", [None, 1, 2, 5])
def test_paused_is_the_workers_own_rule(tmp_path, disk, total_gb, gb) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    for free_share in (0.01, 0.05, 0.0999, 0.1001, 0.2, 0.6):
        free = int(total_gb * GB * free_share)
        disk["usage"] = Usage(total_gb * GB, total_gb * GB - free, free)
        size = health(tmp_path / "raw.db", settings={"min_free_gb": gb})["size"]
        assert size["paused"] is (not disk_room(str(tmp_path / "raw.db"), min_free_bytes(gb)))
        assert size["floor_bytes"] == int(max(min_free_bytes(gb), DISK_FLOOR_SHARE * total_gb * GB))


def test_the_floor_setting_comes_from_the_config(tmp_path, disk, monkeypatch) -> None:
    store = world(tmp_path)
    disk["usage"] = Usage(20 * GB, int(16.5 * GB), int(3.5 * GB))
    learner = SimpleNamespace(config=SimpleNamespace(capture_min_free_gb=5, raw_db_path=None))
    assert progress._capture(store, learner, NOW)["size"]["paused"] is True
    assert progress._capture(store, None, NOW)["size"]["paused"] is False  # the default 2 GiB
    monkeypatch.setenv("CAPTURE_MIN_FREE_GB", "3.75")
    assert progress._capture(store, None, NOW)["size"]["floor_bytes"] == int(3.75 * GB)


def full_disk(raw: RawStore, monkeypatch) -> None:
    """SQLite's own "database or disk is full" on a write to ``raw`` (its page count capped, as
    tests/test_capture_store.py does it): the store's real pause starts (capture/raw_store.py full_pause)."""
    with closing(sqlite3.connect(raw.path)) as db:
        cap = db.execute("PRAGMA page_count").fetchone()[0] + 1
    real = RawStore._connect

    def capped(self):
        db = real(self)
        db.execute(f"PRAGMA max_page_count = {cap}")
        return db

    monkeypatch.setattr(RawStore, "_connect", capped)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        raw.put(game_key="espn:full", league=NBA, source="espn_site", endpoint="summary", params={},
                body=os.urandom(200_000), state="final", http_status=200, fetched_at=NOW)
    monkeypatch.setattr(RawStore, "_connect", real)


def test_sqlites_disk_full_pause_says_so_even_with_free_space(tmp_path, disk, monkeypatch) -> None:
    """After SQLite refused a write as "database or disk is full" the worker pauses (capture/raw_store.py full_pause)
    though the free blocks look fine (inodes, a quota): the alarm says why, not "until X GB is free"."""
    monkeypatch.setattr(raw_store_module, "_FULL_UNTIL", {})  # the process-wide pauses, this test's own
    raw = RawStore(str(tmp_path / "raw.db"), clock=lambda: NOW)
    put(raw, "espn:1", NBA, NOW - HOUR)
    full_disk(raw, monkeypatch)
    block = health(tmp_path / "raw.db")
    assert (block["size"]["paused"], block["size"]["under_floor"], block["size"]["full_until"]) == (
        True, False, progress.iso(NOW + raw_store_module.FULL_PAUSE))
    assert [a["text"] for a in block["alarms"] if a["kind"] == "disk"] == [
        "raw.db can't store more: SQLite said the database or disk is full, though the disk shows 83.8 GB free (check "
        "its inodes or quota). Capture and the backfill are paused, trying again every 5 minutes."]
    # Under the floor as well: the floor's words (room comes back only with free space)
    disk["usage"] = Usage(20 * GB, int(18.6 * GB), int(1.4 * GB))
    assert [a["text"] for a in health(tmp_path / "raw.db")["alarms"] if a["kind"] == "disk"] == [
        "The disk is 93% full: 1.4 GB free. Capture and the backfill are paused until 2.0 GB is free."]
    # Five minutes on, the pause is over: no alarm, nothing paused
    disk["usage"] = Usage(int(100e9), int(10e9), int(90e9))
    block = health(tmp_path / "raw.db", NOW + raw_store_module.FULL_PAUSE + 1)
    assert block["size"]["paused"] is False and block["size"]["full_until"] is None
    assert not [a for a in block["alarms"] if a["kind"] == "disk"]


def test_the_days_requests_and_the_backfills_last_night(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("capture", TODAY, 812)
    raw.spend("prices", TODAY, 140)
    raw.spend("capture", YESTERDAY, 2999)
    raw.spend("backfill", "2026-10-30", 4100)
    raw.spend("backfill", YESTERDAY, 5980)
    block = health(tmp_path / "raw.db", settings={"per_day": 3000, "per_night": 6000, "seasons": 3, "tz": UTC})
    # The night of 1 November (01:00-06:00) has passed with nothing spent: the latest is the night of 31 October
    assert block["requests"] == {"day": TODAY, "capture": 812, "prices": 140, "per_day": 3000,
                                 "backfill_night": YESTERDAY, "backfill": 5980, "per_night": 6000,
                                 "night": TODAY, "night_open": False}
    raw.spend("backfill", TODAY, 3000)
    assert (health(tmp_path / "raw.db", settings={"tz": UTC})["requests"]["backfill_night"],
            health(tmp_path / "raw.db", settings={"tz": UTC})["requests"]["backfill"]) == (TODAY, 3000)


@pytest.mark.parametrize("timezone, night, night_open, spent", [
    ("UTC", "2026-11-02", True, 200),  # 04:00 UTC: tonight's window, running
    ("America/New_York", "2026-11-01", False, 100),  # 23:00 the evening before in New York: last night was the 1st
])
def test_the_backfills_night_is_keyed_in_its_own_time_zone(tmp_path, disk, timezone, night, night_open,
                                                           spent) -> None:
    """The night comes from the learner's config by the backfill's own rule (capture/backfill.py backfill_tz:
    TIMEZONE), as the worker builds the backfill: the same UTC moment is a different night in New York."""
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("backfill", "2026-11-01", 100)
    raw.spend("backfill", "2026-11-02", 200)
    now = datetime.datetime(2026, 11, 2, 4, 0, tzinfo=UTC).timestamp()
    config = SimpleNamespace(timezone=timezone, raw_db_path=str(tmp_path / "raw.db"))
    block = progress._capture(Store(str(tmp_path / "betting.db")), SimpleNamespace(config=config), now)
    assert (block["requests"]["night"], block["requests"]["night_open"]) == (night, night_open)
    assert (block["requests"]["backfill_night"], block["requests"]["backfill"]) == (night, spent)


@pytest.mark.parametrize("window", [None, (22, 4)])  # the backfill's own (01:00-06:00), and one across midnight
@pytest.mark.parametrize("timezone", ["UTC", "America/New_York", "Europe/London"])
def test_the_night_is_the_backfills_own_night_at_every_hour(tmp_path, monkeypatch, timezone, window) -> None:
    """backfill_night is RawBackfill.night and in_window, hour by hour across New York's and London's DST changes."""
    from zoneinfo import ZoneInfo
    from telegram_betting_bot_simple.capture import backfill as raw_backfill
    from telegram_betting_bot_simple.capture.http import PoliteHttp
    monkeypatch.setattr(raw_backfill, "_projected", True)  # no size projection to log
    if window is not None:
        monkeypatch.setattr(raw_backfill, "WINDOW_HOURS", window)
    zone = ZoneInfo(timezone)
    clock = {"t": 0.0}
    backfill = raw_backfill.RawBackfill(RawStore(str(tmp_path / "raw.db")), PoliteHttp(clock=lambda: clock["t"]),
                                        lambda: clock["t"], tz=zone, window=raw_backfill.WINDOW_HOURS)
    start = datetime.datetime(2026, 10, 24, tzinfo=UTC).timestamp()
    for hour in range(16 * 24):  # 24 Oct to 8 Nov 2026: both DST changes, each hour
        clock["t"] = start + hour * 3600 + 1800
        assert progress.backfill_night(clock["t"], zone) == (backfill.night(), backfill.in_window()), clock["t"]




def test_the_block_asks_the_backfills_own_time_zone_rule() -> None:
    """capture/backfill.py backfill_tz itself: TIMEZONE from the config (New York without one), server-local time
    (None) for a name that isn't a zone."""
    from zoneinfo import ZoneInfo
    assert progress.backfill_tz_of(SimpleNamespace(timezone="Europe/London")) == ZoneInfo("Europe/London")
    assert progress.backfill_tz_of(None) == ZoneInfo("America/New_York")
    assert progress.backfill_tz_of(SimpleNamespace(timezone="Not/A_Zone")) is None


def test_the_http_error_share_comes_from_the_request_counters(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("http_ok", YESTERDAY, 900)
    raw.spend("http_error", YESTERDAY, 20)
    raw.spend("http_ok", TODAY, 50)
    raw.spend("http_error", TODAY, 5)
    raw.spend("http_error", "2026-10-29", 5000)  # before yesterday: not counted
    raw.spend("backfill_http_ok", YESTERDAY, 6000)  # the backfill's own: never in forward capture's share
    raw.spend("backfill_http_error", TODAY, 7)
    block = health(tmp_path / "raw.db")
    assert block["http"] == {"since": "2026-10-31T00:00:00Z", "requests": 975, "errors": 25, "share": 0.0256,
                             "counted": "requests",
                             "days": {"yesterday": {"requests": 920, "errors": 20, "share": 0.0217},
                                      "today": {"requests": 55, "errors": 5, "share": 0.0909}}}
    assert not [a for a in block["alarms"] if a["kind"] == "http"]
    raw.spend("http_error", TODAY, 350)
    block = health(tmp_path / "raw.db")
    assert block["http"]["share"] == 0.283
    assert [a for a in block["alarms"] if a["kind"] == "http"] == [
        {"kind": "http", "text": "28% of capture requests failed today and yesterday (UTC): 375 of 1,325."}]


def test_a_high_share_of_few_requests_is_no_alarm(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("http_ok", TODAY, 20)
    raw.spend("http_error", TODAY, 20)
    block = health(tmp_path / "raw.db")
    assert block["http"]["share"] == 0.5 and block["alarms"] == []


def test_without_request_counters_the_stored_answers_of_the_last_day_stand_in(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    for n in range(40):
        put(raw, f"espn:{n}", NBA, NOW - HOUR - n)
    for n in range(20):
        put(raw, f"espn:{100 + n}", NBA, NOW - 2 * HOUR, status=503, body=f'{{"n": {n}}}'.encode())
    put(raw, "espn:old", NBA, NOW - 3 * DAY, status=500, body=b"{}")  # older than a day
    block = health(tmp_path / "raw.db")
    assert block["http"] == {"since": progress.iso(NOW - DAY), "requests": 60, "errors": 20, "share": 0.3333,
                             "counted": "stored answers", "days": None}
    assert [a["text"] for a in block["alarms"] if a["kind"] == "http"] == [
        "33% of answers stored failed in the last 24 hours: 20 of 60."]


# --- F1: forward capture's alarm, the backfill's apart (review c11 a) ---------------------------------------------------

NOW_C11 = datetime.datetime(2026, 11, 3, 18, 0, tzinfo=UTC).timestamp()  # the reviewer's repro: 3 November, 18:00 UTC


def test_a_forward_outage_during_a_backfill_night_alarms(tmp_path, disk) -> None:
    """The review's c11 case: a backfill night of 6,000 requests, then every forward request failing (450 today).
    Counted apart (backfill_http_*), the backfill can't dilute forward capture's share."""
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("backfill_http_ok", "2026-11-02", 6000)
    raw.spend("http_error", "2026-11-03", 450)
    block = health(tmp_path / "raw.db", NOW_C11)
    assert (block["http"]["requests"], block["http"]["errors"], block["http"]["share"]) == (450, 450, 1.0)
    assert [a["text"] for a in block["alarms"] if a["kind"] == "http"] == [
        "100% of capture requests failed today and yesterday (UTC): 450 of 450."]
    assert block["backfill_http"] == {"since": "2026-11-02T00:00:00Z", "requests": 6000, "errors": 0, "share": 0.0}


def test_a_fresh_outage_alarms_on_its_own_day_even_when_the_two_days_dont(tmp_path, disk) -> None:
    """With the counts mixed as before the split (the backfill's night in http_ok), today's own share still alarms:
    a fresh outage is never diluted by yesterday."""
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.spend("http_ok", "2026-11-02", 6000)
    raw.spend("http_error", "2026-11-03", 450)
    block = health(tmp_path / "raw.db", NOW_C11)
    assert block["http"]["share"] == 0.0698  # the two days together: under the 25%
    assert [a["text"] for a in block["alarms"] if a["kind"] == "http"] == [
        "100% of capture requests failed today (UTC): 450 of 450."]
    # Yesterday's outage, today fine: yesterday's own share
    raw2 = RawStore(str(tmp_path / "raw2.db"))
    raw2.spend("http_ok", "2026-11-02", 20)
    raw2.spend("http_error", "2026-11-02", 100)
    raw2.spend("http_ok", "2026-11-03", 400)
    assert [a["text"] for a in health(tmp_path / "raw2.db", NOW_C11)["alarms"] if a["kind"] == "http"] == [
        "83% of capture requests failed yesterday (UTC): 100 of 120."]
    # Each day alone still needs HTTP_ALARM_MIN requests
    raw3 = RawStore(str(tmp_path / "raw3.db"))
    raw3.spend("http_ok", "2026-11-02", 3000)
    raw3.spend("http_error", "2026-11-03", progress.HTTP_ALARM_MIN - 1)
    assert not [a for a in health(tmp_path / "raw3.db", NOW_C11)["alarms"] if a["kind"] == "http"]


def test_the_backfills_own_share_and_alarm(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    assert health(tmp_path / "raw.db", NOW_C11)["backfill_http"] is None  # no backfill request in the two days
    raw.spend("backfill_http_ok", "2026-11-02", 140)
    raw.spend("backfill_http_error", "2026-11-03", 59)
    raw.spend("backfill_http_error", "2026-10-31", 5000)  # before yesterday: not counted
    block = health(tmp_path / "raw.db", NOW_C11)
    assert block["backfill_http"] == {"since": "2026-11-02T00:00:00Z", "requests": 199, "errors": 59, "share": 0.2965}
    assert block["alarms"] == [] and block["http"]["requests"] == 0  # under its 200 requests; forward untouched
    raw.spend("backfill_http_error", "2026-11-03", 1)
    block = health(tmp_path / "raw.db", NOW_C11)
    assert block["alarms"] == [{"kind": "backfill", "text": "30% of the older-games backfill's requests failed today "
                                                            "and yesterday (UTC): 60 of 200."}]
    raw.spend("backfill_http_ok", "2026-11-03", 50)  # 60 of 250: 24%, under the 25%
    assert health(tmp_path / "raw.db", NOW_C11)["alarms"] == []


# --- F6: the boards the sweep couldn't read, and every league's shape (review c11 b, c) ---------------------------------

def sweep(raw: RawStore, day: str, *, read: Tuple[str, ...] = (NBA,), failed: Tuple[str, ...] = (), tries: int = 1,
          last: bool = False) -> None:
    """A day's sweep as capture/worker.py keeps it in capture_state."""
    raw.set_state(f"sweep:{day}", {"leagues": sorted(read), "made": 3, "tries": tries, "failed": list(failed),
                                   "next": NOW_C11 + 1800})
    if last:
        raw.set_state("sweep:last", day)


def test_a_board_the_sweep_couldnt_read_makes_coverage_unknown_and_alarms(tmp_path, disk) -> None:
    """The review's c11 case: one NBA job, covered; the EPL board was never read, so no EPL job exists and the EPL's
    finals are missing unseen. Coverage can't be claimed: the soccer row and the total are unknown, with an alarm."""
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.add_job("espn:2", NBA, "2", NOW_C11 - 30 * HOUR, NOW_C11 - 20 * HOUR, kinds_done=["final:summary"],
                state="done", now=NOW_C11 - 20 * HOUR)
    sweep(raw, "2026-11-02", failed=(EPL,), tries=3, last=True)
    block = health(tmp_path / "raw.db", NOW_C11)
    coverage = block["coverage"]
    assert coverage["unread"] == [{"league": EPL, "day": "2026-11-02", "tries": 3, "final": True}]
    rows = {row["sport"]: row for row in coverage["sports"]}
    assert rows["basketball"]["share"] == 1.0 and rows["basketball"]["unread"] == 0
    assert rows["soccer"] == {"sport": "soccer", "label": "Soccer", "finals": 0, "covered": 0, "missed": 0, "pending": 0,
                              "share": None, "unread": 1}
    assert coverage["total"] == {"finals": 1, "covered": 1, "missed": 0, "pending": 0, "share": None}
    assert [a["text"] for a in block["alarms"] if a["kind"] == "boards"] == [
        "The sweep couldn't read 1 ESPN board: soccer_epl (2026-11-02). Its finals aren't counted, so coverage is "
        "unknown."]


def test_unread_boards_still_being_tried_say_so_and_old_or_odd_sweeps_are_read_safely(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    sweep(raw, "2026-11-02", failed=(EPL, NHL), tries=1)  # yesterday's: the sweep tries again in half an hour
    sweep(raw, "2026-10-31", failed=(NFL,), tries=1)  # an older day's: its sweep is over
    sweep(raw, "2026-10-20", failed=(MLB,), tries=3)  # before the coverage window: not shown
    raw.set_state("sweep:2026-10-30", ["not", "a", "sweep"])  # an odd value: skipped
    raw.set_state("sweep:2026-10-29", {"failed": "soccer_epl", "tries": "x"})
    block = health(tmp_path / "raw.db", NOW_C11)
    # Days in order; an older day's sweep is over (final), yesterday's still tries again
    assert block["coverage"]["unread"] == [
        {"league": NFL, "day": "2026-10-31", "tries": 1, "final": True},
        {"league": NHL, "day": "2026-11-02", "tries": 1, "final": False},
        {"league": EPL, "day": "2026-11-02", "tries": 1, "final": False}]
    assert [a["text"] for a in block["alarms"] if a["kind"] == "boards"] == [
        "The sweep couldn't read 3 ESPN boards yet: americanfootball_nfl (2026-10-31), icehockey_nhl (2026-11-02), "
        "soccer_epl (2026-11-02). Their finals aren't counted, so coverage is unknown."]
    sweep(raw, "2026-11-02", read=(NBA, NHL, EPL), failed=(), tries=2, last=True)  # read on the second try
    sweep(raw, "2026-10-31", read=(NBA, NFL), failed=(), tries=2)
    block = health(tmp_path / "raw.db", NOW_C11)
    assert block["coverage"]["unread"] == [] and not [a for a in block["alarms"] if a["kind"] == "boards"]


def test_one_leagues_changed_summary_alarms_behind_a_newer_league_row(tmp_path, disk) -> None:
    """The review's c11 case: the EPL's final summary lost its box score; the NBA's, stored after it, is fine."""
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", EPL, NOW_C11 - 600, body=json.dumps({"header": {}}).encode())
    put(raw, "espn:2", NBA, NOW_C11 - 300, body=json.dumps({"header": {}, "boxscore": {}}).encode())
    block = health(tmp_path / "raw.db", NOW_C11)
    assert {(s["league"], s["ok"]) for s in block["shapes"] if s["endpoint"] == "summary"} == {(EPL, False), (NBA, True)}
    assert [a["text"] for a in block["alarms"] if a["kind"] == "shape"] == [
        "ESPN game summaries changed shape: boxscore is missing from the newest answer (soccer_epl, 2026-11-03)."]


def test_one_change_in_many_leagues_is_one_alarm_listing_them(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    leagues = ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a", "soccer_germany_bundesliga",
               "soccer_france_ligue_one"]
    for n, league in enumerate(leagues):
        put(raw, f"espn:{n}", league, NOW_C11 - 600 + n, body=json.dumps({"header": {}, "n": n}).encode())
    put(raw, "espn:9", NBA, NOW_C11 - 60, body=json.dumps({"header": {}}).encode())
    texts = [a["text"] for a in health(tmp_path / "raw.db", NOW_C11)["alarms"] if a["kind"] == "shape"]
    assert texts == ["ESPN game summaries changed shape: boxscore is missing from the newest answer (basketball_nba, "
                     "soccer_epl, soccer_france_ligue_one and 3 more, 2026-11-03)."]


# --- F5: the health block reads by index (review c03) -----------------------------------------------------------------

SCANNED = re.compile(r"\bSCAN (capture_jobs|raw_payloads|capture_state|j|p)\b")


def test_every_query_on_jobs_and_payloads_goes_by_an_index(tmp_path, disk, monkeypatch) -> None:
    """EXPLAIN QUERY PLAN of every statement the block runs (traced on its own connection) shows no full scan of
    capture_jobs or raw_payloads: coverage by capture_jobs(start), the latest errors by capture_jobs(updated_at) WHERE
    last_error IS NOT NULL (capture/raw_store.py's indexes), payloads by rowid and their game key."""
    raw = RawStore(str(tmp_path / "raw.db"))
    for n in range(300):
        start = NOW - (n % 200) * 4 * HOUR
        job(raw, f"espn:{n}", NBA if n % 2 else EPL, start, state="done" if n % 5 else "gave_up",
            error="summary: HTTP 503" if n % 7 == 0 else None, updated=start + 5 * HOUR)
        put(raw, f"espn:{n}", NBA if n % 2 else EPL, start + 4 * HOUR, status=200 if n % 11 else 503)
    raw.spend("http_ok", TODAY, 300)
    raw.spend("backfill_http_ok", TODAY, 300)
    raw.spend("backfill", YESTERDAY, 300)
    sweep(raw, "2026-10-31", failed=(NHL,), tries=3)
    raw.set_cursor(NBA, 2026, {"week": 1}, games_seen=3, games_stored=3, now=NOW)
    statements: List[str] = []
    real = progress._connect_ro

    def traced(path: str):
        db = real(path)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(progress, "_connect_ro", traced)
    block = health(tmp_path / "raw.db")
    assert block["stored"] and block["coverage"]["total"]["finals"] and block["errors"] and block["coverage"]["unread"]
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert any("FROM capture_jobs j" in sql for sql in selects) and any("FROM capture_jobs WHERE" in sql
                                                                       for sql in selects)
    with progress._connect_ro(str(tmp_path / "raw.db")) as db:
        plans = {sql: [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + sql)] for sql in selects}
    scans = {sql[:90]: plan for sql, plan in plans.items() if any(SCANNED.search(step) for step in plan)}
    assert scans == {}, scans
    jobs = [plan for sql, plan in plans.items() if "capture_jobs" in sql]
    assert all(any("USING INDEX capture_jobs_" in step for step in plan) for plan in jobs), jobs


def test_backfill_progress_per_league_and_season_in_wave_order(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    raw.set_cursor(NHL, 2026, {"week": 3}, games_seen=50, games_stored=48, now=NOW - DAY)
    raw.set_cursor("soccer_epl", 2025, {"week": 30}, done=True, games_seen=380, games_stored=380, now=NOW - 2 * DAY)
    raw.set_cursor(NBA, 2025, {"week": 0}, done=True, games_seen=1230, games_stored=1229, now=NOW - 3 * DAY)
    raw.set_cursor(NBA, 2026, {"week": 12}, games_seen=210, games_stored=200, now=NOW - HOUR)
    raw.set_cursor("soccer_zzz_other", 2026, None, now=NOW)
    block = health(tmp_path / "raw.db", settings={"seasons": 3})
    assert [b["league"] for b in block["backfill"]] == [NBA, "soccer_epl", NHL, "soccer_zzz_other"]
    nba = block["backfill"][0]
    assert [s["season"] for s in nba["seasons"]] == [2026, 2025]
    assert (nba["done"], nba["begun"], nba["planned"], nba["games_seen"], nba["games_stored"]) == (1, 2, 4, 1440, 1429)
    assert nba["updated_at"] == progress.iso(NOW - HOUR)
    assert nba["seasons"][1] == {"season": 2025, "done": True, "games_seen": 1230, "games_stored": 1229,
                                 "updated_at": progress.iso(NOW - 3 * DAY)}
    assert health(tmp_path / "raw.db")["backfill"][0]["planned"] is None  # the number of seasons unknown


def test_the_latest_errors_newest_first_and_cut_to_size(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    long = "Unreachable: " + "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=1 " * 6
    job(raw, "espn:1", NBA, NOW - DAY, state="todo", error=long, updated=NOW - 3 * HOUR)
    job(raw, "espn:2", NHL, NOW - DAY, state="gave_up", error="HTTP 404", updated=NOW - 5 * HOUR)
    job(raw, "espn:3", NHL, NOW - DAY, state="done", updated=NOW - HOUR)  # no error
    put(raw, "espn:4", EPL, NOW - 2 * HOUR, source="espn_core", endpoint="core_plays", status=500, body=b"down",
        params={"page": 2, "limit": 300})
    block = health(tmp_path / "raw.db")
    assert [(e["game_key"], e["what"]) for e in block["errors"]] == [
        ("espn:4", "ESPN play-by-play pages"), ("espn:1", "capture job, retrying"), ("espn:2", "capture job, given up")]
    assert block["errors"][0] == {"at": progress.iso(NOW - 2 * HOUR), "league": EPL, "game_key": "espn:4",
                                  "what": "ESPN play-by-play pages", "error": "HTTP 500"}
    assert len(block["errors"][1]["error"]) <= progress.ERROR_CHARS and block["errors"][1]["error"].endswith("…")
    for n in range(8):
        job(raw, f"espn:9{n}", NBA, NOW - DAY, error=f"error {n}", updated=NOW - 10 * 60 + n)
    assert len(health(tmp_path / "raw.db")["errors"]) == progress.ERRORS_SHOWN


def test_a_payload_shape_change_is_an_alarm(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    real = (FIXTURES / "capture" / "sum_nba.json").read_bytes()
    put(raw, "espn:1", NBA, NOW - 5 * HOUR, body=real)
    put(raw, "espn:2", NFL, NOW - 3 * HOUR, state="pre", body=(FIXTURES / "capture" / "sum_nfl.json").read_bytes())
    put(raw, "static:espn_site:scoreboard:basketball_nba:20261031", NBA, NOW - 4 * HOUR, endpoint="scoreboard",
        state="static", body=(FIXTURES / "capture" / "sb_nba.json").read_bytes())
    put(raw, "espn:3", EPL, NOW - 4 * HOUR, source="espn_core", endpoint="core_plays", params={"page": 1, "limit": 300},
        body=(FIXTURES / "capture" / "core_epl_plays.json").read_bytes())
    put(raw, "espn:1", NBA, NOW - 6 * HOUR, source="espn_propbets", endpoint="propbets", state="pre",
        params={"limit": 500, "page": 1}, body=(FIXTURES / "espn" / "props_nba_401909866.json").read_bytes())
    put(raw, "static:kalshi:KXEPLCORNERS", EPL, NOW - 6 * HOUR, source="kalshi", endpoint="events", state="pre",
        params={"status": "open", "limit": 200, "page": 1},
        body=(FIXTURES / "markets" / "kalshi_epl_total.json").read_bytes())
    put(raw, "static:nflverse:pbp_2026", NFL, NOW - 6 * HOUR, source="nflverse", endpoint="pbp_season", state="static",
        body=b"\x1f\x8b not json, and never checked")
    block = health(tmp_path / "raw.db")
    assert block["alarms"] == []
    assert {(s["source"], s["endpoint"]): s["ok"] for s in block["shapes"]} == {
        ("espn_site", "summary"): True, ("espn_site", "scoreboard"): True, ("espn_core", "core_plays"): True,
        ("espn_propbets", "propbets"): True, ("kalshi", "events"): True}
    # ESPN renames the box score in the newest final summary; DraftKings' props come back as an HTML page
    put(raw, "espn:9", NBA, NOW - HOUR, body=real.replace(b'"boxscore"', b'"boxScore"'))
    put(raw, "espn:1", NBA, NOW - HOUR, source="espn_propbets", endpoint="propbets", state="pre",
        params={"limit": 500, "page": 1}, body=b"<html>Access denied</html>")
    block = health(tmp_path / "raw.db")
    assert [a for a in block["alarms"] if a["kind"] == "shape"] == [
        {"kind": "shape", "text": "ESPN game summaries changed shape: boxscore is missing from the newest answer "
                                  f"({NBA}, 2026-11-01)."},
        {"kind": "shape", "text": "ESPN DraftKings props changed shape: the newest answer isn't a JSON object any "
                                  f"more ({NBA}, 2026-11-01)."}]


def test_a_stored_answer_is_decoded_once_however_often_the_page_is_opened(tmp_path, disk, monkeypatch) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    decoded: List[str] = []
    real = progress._decode_body
    monkeypatch.setattr(progress, "_decode_body", lambda blob, codec: (decoded.append(codec), real(blob, codec))[1])
    for _ in range(3):
        assert health(tmp_path / "raw.db")["shapes"][0]["ok"] is True
    assert decoded == ["zlib6"]


def test_raw_db_is_only_read(tmp_path, disk) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    put(raw, "espn:1", NBA, NOW - HOUR)
    job(raw, "espn:1", NBA, NOW - DAY, state="done", error="x")
    raw.spend("http_ok", TODAY, 3)
    with open(tmp_path / "raw.db", "rb") as handle:
        before = handle.read()
    health(tmp_path / "raw.db")
    with open(tmp_path / "raw.db", "rb") as handle:
        assert handle.read() == before
    with progress._connect_ro(str(tmp_path / "raw.db")) as db, pytest.raises(Exception):
        db.execute("DELETE FROM capture_jobs")


def test_an_unreadable_raw_db_is_reported_not_raised(tmp_path, disk) -> None:
    (tmp_path / "raw.db").write_bytes(b"this is not a database" * 100)
    disk["usage"] = Usage(100 * GB, 90 * GB, 10 * GB)
    block = health(tmp_path / "raw.db")
    assert block["stored"] is False and block["unreadable"] == "DatabaseError"
    assert [a["kind"] for a in block["alarms"]] == ["disk"]  # the disk still counts
    empty = tmp_path / "empty" / "raw.db"
    empty.parent.mkdir()
    empty.write_bytes(b"")
    assert health(empty)["stored"] is False


def test_raw_db_is_found_where_the_capture_puts_it(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple.capture.raw_store import resolve_raw_db_path
    (tmp_path / "state").mkdir()
    store = Store(str(tmp_path / "state" / "betting.db"))
    assert progress.raw_db_path(store, None) == str(tmp_path / "state" / "raw.db") == resolve_raw_db_path(None, store)
    unset = SimpleNamespace(config=SimpleNamespace(raw_db_path=None))
    assert progress.raw_db_path(store, unset) == str(tmp_path / "state" / "raw.db")
    monkeypatch.setenv("RAW_DB_PATH", "/data/raw.db")
    assert progress.raw_db_path(store, None) == "/data/raw.db"
    learner = SimpleNamespace(config=SimpleNamespace(raw_db_path="/srv/capture/raw.db"))
    assert progress.raw_db_path(store, learner) == "/srv/capture/raw.db"  # the same rule as the worker's
    monkeypatch.delenv("RAW_DB_PATH")
    assert progress.raw_db_path(SimpleNamespace(), None) is None


def test_sports_of_the_58_paths(tmp_path) -> None:
    from telegram_betting_bot_simple.data.espn_client import LEAGUES, SCOREBOARD_ONLY
    sports = {progress.capture_sport(key) for key in [*LEAGUES, *SCOREBOARD_ONLY]}
    assert sports == {"basketball", "football", "hockey", "baseball", "soccer", "tennis", "mma"}
    assert progress.capture_sport("tennis_atp_paris") == "tennis" and progress.capture_sport("cricket_odi") == "other"


def world(tmp_path: Path) -> Store:
    """A betting.db with raw.db beside it, as on the server: a week of captured finals, the backfill under way, two
    errors and a full disk."""
    store = Store(str(tmp_path / "betting.db"))
    raw = RawStore(str(tmp_path / "raw.db"))
    for n in range(40):
        start = NOW - (1 + n % 6) * DAY
        job(raw, f"espn:{n}", NBA if n % 2 else EPL, start, state="done")
        put(raw, f"espn:{n}", NBA if n % 2 else EPL, start + 4 * HOUR, body=summary(n=n))
    job(raw, "espn:99", NHL, NOW - 3 * DAY, state="gave_up", error="Unreachable: site.api.espn.com: timed out",
        updated=NOW - HOUR)
    raw.spend("capture", TODAY, 812)
    raw.spend("prices", TODAY, 140)
    raw.spend("http_ok", TODAY, 800)
    raw.spend("http_error", TODAY, 12)
    raw.spend("backfill", YESTERDAY, 5980)
    raw.spend("backfill_http_ok", YESTERDAY, 5900)
    raw.spend("backfill_http_error", YESTERDAY, 80)
    raw.set_cursor(NBA, 2026, {"week": 12}, games_seen=210, games_stored=200, now=NOW - HOUR)
    raw.set_cursor(NBA, 2025, {"week": 30}, done=True, games_seen=1230, games_stored=1229, now=NOW - 3 * DAY)
    return store


def test_the_progress_page_carries_the_capture_block(tmp_path, disk) -> None:
    store = world(tmp_path)
    disk["usage"] = Usage(80 * GB, 60 * GB, 20 * GB)
    page = progress.progress(store, None, now=NOW)
    block = page["capture"]
    assert block["stored"] is True and block["enabled"] is True
    assert block["coverage"]["total"] == {"finals": 41, "covered": 40, "missed": 1, "pending": 0,
                                          "share": round(40 / 41, 4)}
    assert block["requests"]["per_day"] == 3000 and block["requests"]["per_night"] == 6000
    assert block["backfill"][0]["planned"] == 4 and [a["kind"] for a in block["alarms"]] == ["disk"]
    assert list(page)[-2:] == ["health", "capture"]
    json.dumps(page)
    # The bot's /brain and the Ask AI tool don't change
    assert "capture" not in progress.tool_facts(store, None, now=NOW)


def test_the_page_never_depends_on_the_capture(tmp_path, disk, monkeypatch, caplog) -> None:
    store = world(tmp_path)
    monkeypatch.setattr(progress, "capture_health", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    page = progress.progress(store, None, now=NOW)
    assert page["capture"]["unreadable"] == "error" and page["capture"]["stored"] is False
    assert page["health"] and "the capture block failed" in caplog.text


def test_the_settings_come_from_the_learners_config(tmp_path, disk, monkeypatch) -> None:
    store = world(tmp_path)
    learner = SimpleNamespace(config=SimpleNamespace(capture_enabled=False, capture_per_day=1200,
                                                     capture_backfill_per_night=400, capture_backfill_seasons=2,
                                                     raw_db_path=None))
    block = progress._capture(store, learner, NOW)
    assert block["enabled"] is False and block["stored"] is True
    assert (block["requests"]["per_day"], block["requests"]["per_night"], block["backfill"][0]["planned"]) == (1200, 400,
                                                                                                             3)
    monkeypatch.setenv("CAPTURE_ENABLED", "0")
    monkeypatch.setenv("CAPTURE_PER_DAY", "900")
    block = progress._capture(store, None, NOW)
    assert block["enabled"] is False and block["requests"]["per_day"] == 900


# --- the card, drawn by the app's own modules in Node -----------------------------------------------------------------

HOOKS = r"""
import { readFileSync } from "node:fs";

let STATIC = "";

export async function initialize(data) {
  STATIC = data.static;
}

function exportNames(source) {
  const names = new Set();
  for (const m of source.matchAll(/export\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/g)) names.add(m[1]);
  for (const m of source.matchAll(/export\s+(?:const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)) names.add(m[1]);
  for (const m of source.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const part of m[1].split(",")) {
      const name = part.trim().split(/\s+as\s+/).pop().trim();
      if (/^[A-Za-z_$][\w$]*$/.test(name) && name !== "default") names.add(name);
    }
  }
  return [...names];
}

export async function load(url, context, nextLoad) {
  const path = url.split("?")[0];
  if (!STATIC || !path.startsWith(STATIC) || !path.endsWith(".js")) return nextLoad(url, context);
  if (path === `${STATIC}js/app.js`) {
    const names = exportNames(readFileSync(new URL(path), "utf8"));
    const source = ["const S = globalThis.__appStub;", ...names.map((n) => `export const ${n} = S.get(${JSON.stringify(n)});`)];
    return { format: "module", shortCircuit: true, source: source.join("\n") };
  }
  return nextLoad(url, { ...context, format: "module" });
}
"""

# Reads {static, hooks, titles, cases} on stdin, writes {results}: a "page" case draws the whole Brain page from its
# payload (GET /api/brain/progress answered by a stubbed fetch) and returns the HTML and the page's styles; a "card"
# case returns captureHtml(normalizeProgress(payload)).
DRIVER = r"""
import * as nodeModule from "node:module";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(0, "utf8"));
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
if (typeof nodeModule.register !== "function") {
  finish({ skip: `node ${process.version} has no module.register` });
  await new Promise(() => {});
}
const STATIC = pathToFileURL(input.static.replace(/\/*$/, "/")).href;
nodeModule.register(pathToFileURL(input.hooks).href, { data: { static: STATIC } });

const noop = () => {};
function storage() {
  const items = new Map();
  return {
    getItem: (k) => (items.has(String(k)) ? items.get(String(k)) : null), setItem: (k, v) => { items.set(String(k), String(v)); },
    removeItem: (k) => { items.delete(String(k)); }, clear: () => items.clear(), key: (i) => [...items.keys()][i] ?? null,
    get length() { return items.size; },
  };
}
function element(tag) {
  return {
    tagName: String(tag).toUpperCase(), id: "", style: {}, dataset: {}, children: [], textContent: "", innerHTML: "",
    className: "", isConnected: true, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    setAttribute: noop, getAttribute: () => null, removeAttribute: noop, hasAttribute: () => false,
    appendChild(child) { this.children.push(child); return child; }, append: noop, prepend: noop, remove: noop,
    addEventListener: noop, removeEventListener: noop, querySelector: () => null, querySelectorAll: () => [],
    contains: () => false, focus: noop, blur: noop, closest: () => null, matches: () => false,
  };
}
const byId = new Map();
const doc = {
  head: element("head"), body: element("body"), documentElement: element("html"), activeElement: null, title: "",
  getElementById: (id) => byId.get(id) || null, createElement: element, createElementNS: (_, tag) => element(tag),
  querySelector: () => null, querySelectorAll: () => [], addEventListener: noop, removeEventListener: noop,
  visibilityState: "visible", hidden: false,
};
doc.head.appendChild = (child) => { if (child?.id) byId.set(child.id, child); return child; };
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.addEventListener = noop;
globalThis.removeEventListener = noop;
globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
globalThis.location = { hash: "#/bets?tab=record&view=brain", href: "http://test.local/", origin: "http://test.local", pathname: "/", search: "", host: "test.local" };
globalThis.history = { state: null, length: 1, replaceState: noop, pushState: noop, back: noop };
globalThis.matchMedia = () => ({ matches: false, media: "", addEventListener: noop, removeEventListener: noop, addListener: noop, removeListener: noop });
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.innerWidth = 360;
globalThis.innerHeight = 800;

const TITLES = input.titles || {};
const OVERRIDES = { leagueTitle: (key) => TITLES[key], replaceAddress: noop };
globalThis.__appStub = { get: (name) => (name in OVERRIDES ? OVERRIDES[name] : noop) };

const fetched = [];
let answer = {};
globalThis.fetch = async (path) => {
  fetched.push(String(path));
  const body = JSON.stringify(answer);
  return { status: 200, ok: true, headers: new Headers(), json: async () => JSON.parse(body) };
};

const mod = (rel) => import(new URL(rel, STATIC).href);

async function pageCase(spec) {
  const page = await mod("js/views/brain/progress.js");
  answer = spec.payload;
  fetched.length = 0;
  const box = { innerHTML: "", isConnected: true, querySelector: () => null, querySelectorAll: () => [], addEventListener: noop,
    removeEventListener: noop, contains: () => false };
  await page.renderBrain(box, { query: new URLSearchParams() });
  return { html: box.innerHTML, css: byId.get("brain-styles")?.textContent || "", fetched: [...fetched] };
}

async function cardCase(spec) {
  const page = await mod("js/views/brain/progress.js");
  const model = await mod("js/views/brain/model.js");
  return { html: page.captureHtml(model.normalizeProgress(spec.payload)) };
}

const results = [];
for (const spec of input.cases || []) {
  try {
    results.push(spec.kind === "page" ? await pageCase(spec) : await cardCase(spec));
  } catch (error) {
    results.push({ error: String(error?.stack || error) });
  }
}
finish({ results });
"""

TITLES = {NBA: "NBA", NHL: "NHL", EPL: "Premier League", "soccer_uefa_europa_conference_league":
          "UEFA Europa Conference League"}


@pytest.fixture(scope="module")
def draw(tmp_path_factory):
    """draw(*cases) -> each case's parsed HTML (with .css: the page's own styles, for a "page" case)."""
    if not NODE:
        pytest.skip("node isn't installed: these tests draw the app's ES modules in Node")
    folder = tmp_path_factory.mktemp("brain")
    (folder / "hooks.mjs").write_text(HOOKS, encoding="utf-8")
    (folder / "driver.mjs").write_text(DRIVER, encoding="utf-8")

    def run(*cases: Dict[str, Any]) -> List[Any]:
        payload = json.dumps({"static": str(STATIC), "hooks": str(folder / "hooks.mjs"), "titles": TITLES,
                              "cases": list(cases)})
        env = {**os.environ, "TZ": "America/New_York", "NODE_OPTIONS": ""}
        done = subprocess.run([NODE, str(folder / "driver.mjs")], input=payload, capture_output=True, text=True,
                              encoding="utf-8", timeout=120, cwd=folder, env=env)
        assert done.returncode == 0, f"node failed ({done.returncode}):\n{done.stderr[-3000:]}"
        out = json.loads(done.stdout)
        if out.get("skip"):
            pytest.skip(out["skip"])
        drawn = []
        for case, result in zip(cases, out["results"]):
            assert "error" not in result, f"the {case['kind']} case threw in Node:\n{result['error']}"
            page = parse(result["html"])
            page.css = result.get("css", "")
            page.fetched = result.get("fetched", [])
            drawn.append(page)
        return drawn

    return run


_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())


class El:
    def __init__(self, tag: str, attrs: Mapping[str, Optional[str]], parent: Optional["El"] = None) -> None:
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: List[Any] = []
        self.html = self.css = ""
        self.fetched: List[str] = []

    @property
    def classes(self) -> set:
        return set((self.attrs.get("class") or "").split())

    def walk(self) -> Iterator["El"]:
        for child in self.children:
            if isinstance(child, El):
                yield child
                yield from child.walk()

    def text(self) -> str:
        parts: List[str] = []

        def add(node: "El") -> None:
            for child in node.children:
                if isinstance(child, str):
                    parts.append(child)
                elif child.attrs.get("aria-hidden") != "true":
                    add(child)

        add(self)
        return " ".join("".join(parts).split())

    def find(self, tag: Optional[str] = None, cls: Optional[str] = None, **attrs: str) -> List["El"]:
        return [el for el in self.walk() if (tag is None or el.tag == tag) and (cls is None or cls in el.classes)
                and all(el.attrs.get(k.replace("_", "-")) == v for k, v in attrs.items())]

    def one(self, tag: Optional[str] = None, cls: Optional[str] = None, **attrs: str) -> "El":
        found = self.find(tag, cls, **attrs)
        assert len(found) == 1, f"{len(found)} elements match {tag}.{cls} {attrs}"
        return found[0]


class _Builder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = El("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        el = El(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(el)
        if tag not in _VOID:
            self.stack.append(el)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.stack[-1].children.append(El(tag, attrs, self.stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def parse(html: str) -> El:
    builder = _Builder()
    builder.feed(html)
    builder.close()
    builder.root.html = html
    return builder.root


def tiles(card: El) -> Dict[str, Tuple[str, str]]:
    """The card's number tiles: {label: (value, small print)}."""
    out = {}
    for pair in card.find("div"):
        if pair.parent is not None and "bp-nums" in pair.parent.classes:
            dd = pair.one("dd")
            small = dd.find("small")
            out[pair.one("dt").text()] = (" ".join(c for c in dd.children if isinstance(c, str)).strip(),
                                          small[0].text() if small else "")
    return out


def card_case(capture: Any) -> Dict[str, Any]:
    return {"kind": "card", "payload": {"capture": capture}}


def capture_block(**over: Any) -> Dict[str, Any]:
    block = {
        "enabled": True, "stored": True, "as_of": "2026-11-01T18:00:00Z", "last_stored": "2026-11-01T17:55:00Z",
        "coverage": {"days": 7, "within_hours": 36, "target": 0.98,
                     "sports": [{"sport": "basketball", "label": "Basketball", "finals": 412, "covered": 409, "missed": 1,
                                 "pending": 2, "share": 0.99756},
                                {"sport": "soccer", "label": "Soccer", "finals": 300, "covered": 288, "missed": 12,
                                 "pending": 0, "share": 0.96}],
                     "total": {"finals": 712, "covered": 697, "missed": 13, "pending": 2, "share": 0.98169}},
        "size": {"raw_bytes": 4_213_000_000, "disk_total": 80 * GB, "disk_free": 55 * GB, "disk_full": 0.3125},
        "requests": {"day": TODAY, "capture": 812, "prices": 140, "per_day": 3000, "backfill": 5980,
                     "backfill_night": YESTERDAY, "per_night": 6000},
        "http": {"since": "2026-10-31T00:00:00Z", "requests": 1520, "errors": 31, "share": 0.0204, "counted": "requests"},
        "backfill": [{"league": NBA, "seasons": [{"season": 2026, "done": False, "games_seen": 210, "games_stored": 200},
                                                 {"season": 2025, "done": True, "games_seen": 1230, "games_stored": 1229}],
                      "done": 1, "begun": 2, "planned": 4, "games_seen": 1440, "games_stored": 1429,
                      "updated_at": "2026-11-01T17:00:00Z"}],
        "errors": [{"at": "2026-11-01T17:00:00Z", "league": NHL, "game_key": "espn:401", "what": "capture job, given up",
                    "error": "Unreachable: site.api.espn.com: timed out"}],
        "shapes": [], "alarms": []}
    block.update(over)
    return block


def test_the_whole_page_draws_the_card_from_the_real_payload(draw, tmp_path, disk) -> None:
    store = world(tmp_path)
    disk["usage"] = Usage(80 * GB, 60 * GB, 20 * GB)
    payload = json.loads(json.dumps(progress.progress(store, None, now=NOW)))
    [page] = draw({"kind": "page", "payload": payload})
    assert page.fetched == ["/api/brain/progress"]
    card = page.one("section", id="bp-capture")
    assert "card" in card.classes and "bp-cap" in card.classes  # the page's own card style
    assert card.one("h2").text() == "Raw capture" and card.one("span", "badge").text() == "1 alarm"
    assert [n.text() for n in card.find("div", "notice")] == [a["text"] for a in payload["capture"]["alarms"]]
    numbers = tiles(card)
    assert numbers["Finals stored within 36 h"] == ("97.5%", "40 of 41 finals, last 7 days · below the 98% target")
    assert numbers["Requests today"] == ("812", "of 3,000 a day · 140 for prices")
    assert numbers["Failed requests"] == ("1.4%", "12 of 812 requests, today and yesterday (UTC)")
    # The night of 1 November has passed with nothing spent: the latest night is named by its date, not "last night"
    assert numbers["Backfill, night of Oct 31"] == ("5,980", "of 6,000 requests")
    assert numbers["Failed backfill requests"] == ("1.3%", "80 of 5,980 requests, today and yesterday (UTC)")
    assert numbers["raw.db"][1] == "disk 75% full, 20.0 GB free"
    rows = {li.find("span")[0].text(): li.one("b").text() for li in card.find("li") if li.parent and "bp-kv" in li.parent.classes}
    assert rows["Basketball"].startswith("100%") and rows["Soccer"].startswith("100%")
    assert rows["Ice hockey"] == "0.0% · 0 of 1"
    fold = card.find("details", "bp-cap-fold")
    assert fold[0].one("summary").text() == "Older games: 1 league · 1 of 4 seasons done · 1,429 games"
    assert "NBA" in fold[0].text() and "2026 200 so far · 2025 done" in fold[0].text()
    assert fold[1].one("summary").text() == "Latest errors (1)" and "timed out" in fold[1].text()
    assert ".bp .bp-cap" in page.css  # the card's rules come with the page's own styles


def test_the_card_says_capture_is_paused_after_sqlites_disk_full(draw, tmp_path, disk, monkeypatch) -> None:
    """The real pause (a write refused as "database or disk is full", capture/raw_store.py full_pause) through
    progress() to the card: the raw.db tile says capture is paused and the alarm says why, with room on the disk."""
    monkeypatch.setattr(raw_store_module, "_FULL_UNTIL", {})
    store = world(tmp_path)
    full_disk(RawStore(str(tmp_path / "raw.db"), clock=lambda: NOW), monkeypatch)
    payload = json.loads(json.dumps(progress.progress(store, None, now=NOW)))
    [page] = draw({"kind": "page", "payload": payload})
    card = page.one("section", id="bp-capture")
    assert tiles(card)["raw.db"][1] == "disk 10% full, 83.8 GB free · capture paused"
    assert [n.text() for n in card.find("div", "notice")] == [
        "raw.db can't store more: SQLite said the database or disk is full, though the disk shows 83.8 GB free (check "
        "its inodes or quota). Capture and the backfill are paused, trying again every 5 minutes."]
    assert card.one("span", "badge").text() == "1 alarm"


HOSTILE = '<img src=x onerror="alert(1)"><script>alert(2)</script>&amp;'


def test_every_string_is_escaped(draw) -> None:
    block = capture_block(
        alarms=[{"kind": "shape", "text": f"ESPN changed shape: {HOSTILE}"}, {"kind": "<b>", "text": HOSTILE}],
        errors=[{"at": "2026-11-01T17:00:00Z", "league": HOSTILE, "game_key": HOSTILE, "what": HOSTILE, "error": HOSTILE}],
        backfill=[{"league": HOSTILE, "seasons": [{"season": 2026, "done": False, "games_stored": 3}], "done": 0,
                   "begun": 1, "planned": 4, "games_stored": 3, "updated_at": None}])
    block["coverage"]["sports"][0]["label"] = HOSTILE
    block["coverage"]["unread"] = [{"league": HOSTILE, "day": "2026-11-02", "tries": 3, "final": True}]
    [card] = draw(card_case(block))
    assert not card.find("img") and not card.find("script")
    assert all(el.tag in {"section", "div", "h2", "span", "svg", "path", "circle", "line", "polyline", "rect", "p", "dl",
                          "dt", "dd", "small", "h3", "ul", "li", "b", "details", "summary"} for el in card.walk())
    text = card.text()
    assert text.count('<img src=x onerror="alert(1)">') >= 6 and "<script>alert(2)</script>&amp;" in text
    assert HOSTILE in card.one("p", "bp-cap-unread").text()
    assert "onerror" not in json.dumps([el.attrs for el in card.walk()])


@pytest.mark.parametrize("share, shown", [(0.99999, "99.9%"), (0.98, "98.0%"), (1.0, "100%"), (0.9755, "97.5%"),
                                          (None, "–")])
def test_coverage_is_never_rounded_up(draw, share, shown) -> None:
    block = capture_block()
    block["coverage"]["total"]["share"] = share
    [card] = draw(card_case(block))
    assert tiles(card)["Finals stored within 36 h"][0] == shown


def test_coverage_below_the_target_says_so_in_words(draw) -> None:
    block = capture_block()
    [ok] = draw(card_case(block))
    assert "target" not in tiles(ok)["Finals stored within 36 h"][1]
    assert ok.one("span", "badge").text() == "No alarms" and "good" in ok.one("span", "badge").classes
    block["coverage"]["total"].update(share=0.9, covered=90, missed=10)
    [low] = draw(card_case(block))
    assert tiles(low)["Finals stored within 36 h"] == ("90.0%", "90 of 100 finals, last 7 days · below the 98% target")


def test_a_paused_capture_says_so_in_the_raw_db_tile(draw) -> None:
    block = capture_block()
    block["size"].update(disk_total=20 * GB, disk_free=int(1.4 * GB), disk_full=0.93, floor_bytes=2 * GB, paused=True)
    block["alarms"] = [{"kind": "disk", "text": "The disk is 93% full: 1.4 GB free. Capture and the backfill are paused "
                                                "until 2.0 GB is free."}]
    [paused, running] = draw(card_case(block), card_case(capture_block()))
    assert tiles(paused)["raw.db"] == ("3.9 GB", "disk 93% full, 1.4 GB free · capture paused")
    assert [n.text() for n in paused.find("div", "notice")] == [block["alarms"][0]["text"]]
    assert "paused" not in tiles(running)["raw.db"][1] and tiles(running)["raw.db"][1] == "disk 31% full, 55.0 GB free"


def test_unread_boards_show_and_coverage_says_it_is_unknown(draw) -> None:
    block = capture_block()
    block["coverage"]["total"]["share"] = None
    block["coverage"]["sports"][1].update(share=None, unread=1)  # soccer
    block["coverage"]["sports"].append({"sport": "hockey", "label": "Ice hockey", "finals": 0, "covered": 0,
                                        "missed": 0, "pending": 0, "share": None, "unread": 1})
    block["coverage"]["unread"] = [{"league": EPL, "day": "2026-11-02", "tries": 3, "final": True},
                                   {"league": NHL, "day": "2026-11-02", "tries": 1, "final": False}]
    [card] = draw(card_case(block))
    assert tiles(card)["Finals stored within 36 h"] == ("–", "unknown: 2 boards unread · 697 of 710 finals it could see")
    rows = {li.find("span")[0].text(): li.one("b").text() for li in card.find("li") if li.parent and "bp-kv" in li.parent.classes}
    assert rows["Soccer"] == "unknown · 1 board unread · 288 of 300 seen"
    assert rows["Ice hockey"] == "unknown · 1 board unread" and rows["Basketball"].startswith("99.7%")
    # The ESPN day as sent ("2026-11-02"), never shifted by the viewer's time zone (Node runs in New York here)
    assert card.one("p", "bp-cap-unread").text() == (
        "The sweep couldn't read these boards, so their games aren't counted: Premier League (Nov 2, after 3 tries) · "
        "NHL (Nov 2, trying again).")


@pytest.mark.parametrize("night, backfill_night, night_open, label", [
    ("2026-11-02", "2026-11-02", True, "Backfill tonight so far"),
    ("2026-11-02", "2026-11-02", False, "Backfill last night"),
    ("2026-11-02", "2026-10-30", False, "Backfill, night of Oct 30"),
])
def test_the_backfills_night_and_own_failed_share(draw, night, backfill_night, night_open, label) -> None:
    block = capture_block(backfill_http={"since": "2026-10-31T00:00:00Z", "requests": 6000, "errors": 120,
                                         "share": 0.02})
    block["requests"].update(night=night, backfill_night=backfill_night, night_open=night_open)
    [card] = draw(card_case(block))
    numbers = tiles(card)
    assert numbers[label] == ("5,980", "of 6,000 requests")
    assert numbers["Failed backfill requests"] == ("2.0%", "120 of 6,000 requests, today and yesterday (UTC)")
    assert numbers["Failed requests"] == ("2.0%", "31 of 1,520 requests, today and yesterday (UTC)")
    [none] = draw(card_case(capture_block(backfill_http=None)))
    assert "Failed backfill requests" not in tiles(none)


def test_alarms_are_warnings_in_words_disk_first(draw) -> None:
    block = capture_block(alarms=[{"kind": "http", "text": "31% of capture requests failed."},
                                  {"kind": "shape", "text": "ESPN game summaries changed shape."},
                                  {"kind": "disk", "text": "The disk is 71% full."}])
    [card] = draw(card_case(block))
    notices = card.find("div", "notice")
    assert [n.text() for n in notices] == ["The disk is 71% full.", "ESPN game summaries changed shape.",
                                           "31% of capture requests failed."]
    assert all("warn" in n.classes for n in notices)
    badge = card.one("span", "badge")
    assert badge.text() == "3 alarms" and "warn" in badge.classes


@pytest.mark.parametrize("block, words", [
    (capture_block(stored=False), "Nothing captured yet"),
    (capture_block(stored=False, enabled=False), "Raw capture is off (CAPTURE_ENABLED=0), so nothing is stored."),
    (capture_block(stored=False, unreadable="DatabaseError"), "raw.db couldn't be read just now"),
])
def test_the_empty_states(draw, block, words) -> None:
    [card] = draw(card_case(block))
    assert words in card.one("p", "bp-empty").text() and not card.find("dl")


def test_no_capture_block_means_no_card_and_a_capture_turned_off_says_so(draw) -> None:
    [none, off] = draw({"kind": "card", "payload": {}}, card_case(capture_block(enabled=False)))
    assert none.html == ""
    assert "Raw capture is off (CAPTURE_ENABLED=0): these are the numbers from when it last ran." in off.text()


@pytest.mark.parametrize("bad", ["capture", 7, [], {"coverage": [], "size": "big", "requests": None,
                                                     "backfill": {"x": 1}, "errors": "oops",
                                                     "alarms": [None, 3, {"text": 5}], "stored": "yes"},
                                 {"stored": True, "coverage": {"total": {"share": "all"}, "sports": [{"finals": "x"}]},
                                  "http": {"requests": "many"}, "size": {"disk_full": 7}}])
def test_a_malformed_block_never_breaks_the_page(draw, bad) -> None:
    [card] = draw(card_case(bad))
    if card.html:
        assert card.one("section", id="bp-capture") and "undefined" not in card.html and "NaN" not in card.html


def test_the_card_never_paints_words_in_the_warning_hue() -> None:
    css = (STATIC / "js" / "views" / "brain" / "styles.js").read_text(encoding="utf-8")
    rules = [line for line in css.splitlines() if ".bp-cap" in line]
    assert rules and not [r for r in rules if re.search(r"(?<![\w-])color\s*:\s*var\(\s*--warn\s*\)", r)]
    assert all("overflow-wrap: anywhere" in r for r in rules if "bp-meta" in r or "errors p" in r)


# --- the card at 360 px, laid out in Chromium (Playwright) ------------------------------------------------------------

LAYOUT = r"""
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json" };

(async () => {
  let browser;
  try {
    browser = await chromium.launch();
  } catch (error) {
    finish({ skip: `Chromium can't start: ${String(error).split("\n")[0]}` });
    return;
  }
  const results = [];
  for (const scheme of ["light", "dark"]) {
    const page = await browser.newPage({ viewport: { width: 360, height: 800 }, deviceScaleFactor: 1, colorScheme: scheme });
    const problems = [];
    page.on("pageerror", (error) => problems.push(String(error)));
    await page.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.hostname !== "test.local") return route.fulfill({ status: 404, body: "" });
      if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: input.html });
      if (url.pathname === "/js/app.js") return route.fulfill({ contentType: "text/javascript", body: input.appStub });
      if (url.pathname === "/api/brain/progress") return route.fulfill({ contentType: "application/json", body: JSON.stringify(input.payload) });
      const file = path.join(input.static, decodeURIComponent(url.pathname));
      if (!file.startsWith(input.static) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "" });
      return route.fulfill({ contentType: TYPES[path.extname(file)] || "application/octet-stream", body: fs.readFileSync(file) });
    });
    await page.goto("http://test.local/");
    await page.waitForFunction(() => window.__done === true || typeof window.__failed === "string", null, { timeout: 30000 });
    const measured = await page.evaluate(() => {
      if (window.__failed) return { failed: window.__failed };
      const card = document.getElementById("bp-capture");
      if (!card) return { failed: "no #bp-capture" };
      const box = card.getBoundingClientRect();
      const out = [];
      for (const el of card.querySelectorAll("*")) {
        const r = el.getBoundingClientRect();
        if (!r.width && !r.height) continue;
        if (r.left < box.left - 0.5 || r.right > box.right + 0.5) {
          out.push({ tag: el.tagName, cls: String(el.getAttribute("class") || ""), left: r.left, right: r.right, text: (el.textContent || "").slice(0, 80) });
        }
        if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflowX === "visible" && el.clientWidth) {
          out.push({ tag: el.tagName, cls: String(el.getAttribute("class") || ""), scroll: el.scrollWidth, client: el.clientWidth, text: (el.textContent || "").slice(0, 80) });
        }
      }
      return { width: window.innerWidth, page: document.documentElement.scrollWidth, card: { left: box.left, right: box.right },
        open: card.querySelectorAll("details[open]").length, notices: card.querySelectorAll(".notice").length, overflow: out };
    });
    results.push({ scheme, problems, ...measured });
    await page.close();
  }
  await browser.close();
  finish({ results });
})().catch((error) => finish({ error: String(error && error.stack || error) }));
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/styles.css"></head>
<body><div class="shell"><main id="view" class="main"><div id="box"></div></main></div>
<script>const TITLES = %s; window.__appStub = { get: (name) => name === "leagueTitle" ? (key) => TITLES[key] : () => undefined };</script>
<script type="module">
import { renderBrain } from "/js/views/brain/progress.js";
try {
  await renderBrain(document.getElementById("box"));
  for (const fold of document.querySelectorAll("#bp-capture details")) fold.open = true;
  window.__done = true;
} catch (error) {
  window.__failed = String(error && error.stack || error);
}
</script></body></html>"""


def node_path() -> Optional[str]:
    """NODE_PATH with the global modules folder, when Node can require playwright through it; else None."""
    if not NODE:
        return None
    folders = [p for p in os.environ.get("NODE_PATH", "").split(os.pathsep) if p]
    folders.append(str(Path(NODE).resolve().parent.parent / "lib" / "node_modules"))
    joined = os.pathsep.join(folders)
    found = subprocess.run([NODE, "-e", "require.resolve('playwright')"], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "NODE_PATH": joined})
    return joined if found.returncode == 0 else None


def app_stub() -> str:
    source = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    names = set(re.findall(r"export\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)", source))
    names |= set(re.findall(r"export\s+(?:const|let|var|class)\s+([A-Za-z_$][\w$]*)", source))
    for group in re.findall(r"export\s*\{([^}]*)\}", source):
        names |= {part.strip().split(" as ")[-1].strip() for part in group.split(",") if part.strip()}
    return "\n".join(f"export const {name} = window.__appStub.get({json.dumps(name)});" for name in sorted(names)
                     if re.fullmatch(r"[A-Za-z_$][\w$]*", name) and name != "default")


def worst_case() -> Dict[str, Any]:
    """A capture block with the longest words it can carry: URLs, game keys, a long league, every alarm."""
    long_league = "soccer_uefa_europa_conference_league"
    url = ("https://sports.core.api.espn.com/v2/sports/soccer/leagues/uefa.europa.conf/events/401234567/competitions/"
           "401234567/plays?limit=300&page=3")
    block = capture_block(
        alarms=[{"kind": "disk", "text": "The disk is 91% full: 7.2 GB free, raw.db is 61.4 GB."},
                {"kind": "shape", "text": "ESPN play-by-play pages changed shape: items, pageCount are missing from the "
                                          f"newest answer ({long_league}, 2026-11-01)."},
                {"kind": "http", "text": "31% of capture requests failed today and yesterday (UTC): 1,412 of 4,553."}],
        errors=[{"at": "2026-11-01T17:00:00Z", "league": long_league,
                 "game_key": f"static:espn_site:scoreboard:{long_league}:20261031", "what": "capture job, retrying",
                 "error": f"Unreachable: {url}: URLError: <urlopen error [Errno 111] Connection refused>"}] * 5,
        backfill=[{"league": league, "seasons": [{"season": 2026 - n, "done": n > 0, "games_stored": 123456}
                                                 for n in range(4)],
                   "done": 3, "begun": 4, "planned": 4, "games_stored": 493824, "updated_at": "2026-11-01T17:00:00Z"}
                  for league in (NBA, NFL, EPL, long_league, "soccer_germany_bundesliga", NHL)])
    block["coverage"]["sports"].append({"sport": "football", "label": "American football", "finals": 123456,
                                        "covered": 123000, "missed": 400, "pending": 56, "share": None, "unread": 12})
    block["coverage"]["unread"] = [{"league": league, "day": "2026-11-02", "tries": 3, "final": n % 2 == 0}
                                   for n, league in enumerate([long_league, "soccer_conmebol_copa_sudamericana",
                                                               "soccer_uefa_champs_league_women", NFL, NHL] * 3)]
    block["coverage"]["total"]["share"] = None
    block["alarms"].append({"kind": "boards", "text": "The sweep couldn't read 15 ESPN boards yet: "
                                                      f"{long_league} (2026-11-02), " * 4 + "and 11 more. Their "
                                                      "finals aren't counted, so coverage is unknown."})
    block["requests"].update(capture=12345678, prices=1234567, per_day=12345678, night="2026-11-02",
                             backfill_night="2026-10-30", backfill=12345678, per_night=12345678)
    block["backfill_http"] = {"since": "2026-10-31T00:00:00Z", "requests": 12345678, "errors": 1234567,
                              "share": 0.1}
    return block


def test_the_card_fits_a_360_px_phone(tmp_path, disk) -> None:
    joined = node_path()
    if not joined:
        pytest.skip("Playwright for Node isn't installed: the 360 px layout check needs Chromium")
    store = world(tmp_path)
    disk["usage"] = Usage(80 * GB, 60 * GB, 20 * GB)
    real = json.loads(json.dumps(progress.progress(store, None, now=NOW)))
    for payload in (real, {**real, "capture": worst_case()}):
        script = tmp_path / "layout.cjs"
        script.write_text(LAYOUT, encoding="utf-8")
        data = json.dumps({"static": str(STATIC), "html": PAGE % json.dumps(TITLES), "appStub": app_stub(),
                           "payload": payload})
        done = subprocess.run([NODE, str(script)], input=data, capture_output=True, text=True, encoding="utf-8",
                              timeout=180, cwd=tmp_path, env={**os.environ, "NODE_PATH": joined, "NODE_OPTIONS": ""})
        assert done.returncode == 0, done.stderr[-3000:]
        out = json.loads(done.stdout)
        if out.get("skip"):
            pytest.skip(out["skip"])
        assert "error" not in out, out.get("error")
        for result in out["results"]:
            assert not result.get("failed"), result
            assert result["problems"] == [], result["problems"]
            assert result["width"] == 360 and result["page"] <= 360, result
            assert result["overflow"] == [], result["overflow"]
            assert result["open"] == 2 and result["notices"] == len(payload["capture"]["alarms"])
