"""The raw capture store and its door to the network (v4 P1, plan section 3.3): capture/raw_store.py RawStore and
capture/http.py PoliteHttp. No network: PoliteHttp's one send is replaced by a fake, and urllib is never reached."""

from __future__ import annotations

import datetime
import email.message
import gzip
import hashlib
import io
import json
import os
import sqlite3
import threading
import urllib.error
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

from telegram_betting_bot_simple.capture import http as capture_http
from telegram_betting_bot_simple.capture.http import (BACKOFF_CAP, BREAKER, MAX_WAIT, Backoff, HttpResponse,
                                                      PoliteHttp, Refused, Unreachable)
from telegram_betting_bot_simple.capture.raw_store import (BUDGET_KINDS, JOB_STATES, Job, RawStore,
                                                           canonical_params, resolve_raw_db_path)

NBA = "basketball_nba"


@pytest.fixture
def raw(tmp_path) -> RawStore:
    return RawStore(str(tmp_path / "state" / "raw.db"))


@pytest.fixture(autouse=True)
def no_shared_door():
    """No test leaves the process a door (capture.http.set_shared): a later test's HistHttp would wait on it."""
    capture_http.set_shared(None)
    yield
    capture_http.set_shared(None)


def put(raw: RawStore, body: bytes = b'{"a": 1}', **over: Any) -> int:
    kwargs: Dict[str, Any] = dict(game_key="espn:401", league=NBA, source="espn_site", endpoint="summary", params={},
                                  body=body, state="final", http_status=200, horizon="final", fetched_at=1000.0)
    kwargs.update(over)
    return raw.put(**kwargs)


# --- the schema: section 3.3 exactly, in a file of its own ---------------------------------------------------------

PLAN_RAW = ["id", "game_key", "league", "source", "endpoint", "params", "state", "horizon", "fetched_at", "http_status",
            "sha256", "bytes_raw", "codec", "body"]
PLAN_JOBS = ["game_key", "league", "espn_id", "start", "due_at", "state", "attempts", "kinds_done", "last_error",
             "updated_at"]


def test_the_tables_are_the_plans_in_a_wal_file_of_their_own(tmp_path, raw) -> None:
    with closing(sqlite3.connect(raw.path)) as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(raw_payloads)")] == PLAN_RAW
        assert [row[1] for row in db.execute("PRAGMA table_info(capture_jobs)")] == PLAN_JOBS
        indexes = {row[1]: [col[2] for col in db.execute(f"PRAGMA index_info({row[1]})")]
                   for row in db.execute("PRAGMA index_list(raw_payloads)")}
        assert indexes["raw_by_game"] == ["game_key", "source", "endpoint", "fetched_at"]
        assert indexes["raw_payloads_fetched"] == ["fetched_at", "http_status"]  # the health reads' (beyond 3.3)
        unique = [cols for name, cols in indexes.items() if name.startswith("sqlite_autoindex")]
        assert unique == [["game_key", "source", "endpoint", "params", "sha256"]]
        jobs = {row[1]: ([col[2] for col in db.execute(f"PRAGMA index_info({row[1]})")], bool(row[4]))
                for row in db.execute("PRAGMA index_list(capture_jobs)") if not row[1].startswith("sqlite_autoindex")}
        # the plan's, and the health block's two (F5): by start, and the newest errors (a partial index)
        assert jobs == {"capture_jobs_due": (["state", "due_at"], False), "capture_jobs_start": (["start"], False),
                        "capture_jobs_errors": (["updated_at"], True)}
        partial = db.execute("SELECT sql FROM sqlite_master WHERE name = 'capture_jobs_errors'").fetchone()[0]
        assert "WHERE last_error IS NOT NULL" in partial
        assert [row[1] for row in db.execute("PRAGMA table_info(raw_seen)")] == ["payload_id", "seen_at", "horizon",
                                                                                "state"]
        assert [col[2] for col in db.execute("PRAGMA index_info(raw_seen_by_payload)")] == ["payload_id", "seen_at"]
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        defaults = {row[1]: row[4] for row in db.execute("PRAGMA table_info(capture_jobs)")}
        assert (defaults["state"], defaults["attempts"], defaults["kinds_done"]) == ("'todo'", "0", "'[]'")
        assert {row[1]: row[4] for row in db.execute("PRAGMA table_info(raw_payloads)")}["codec"] == "'zlib6'"
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"raw_payloads", "capture_jobs", "capture_budget", "backfill_progress", "capture_state", "raw_seen"}
    # betting.db never gets these tables (raw blobs never bloat it)
    from telegram_betting_bot_simple.data.store import Store
    store = Store(str(tmp_path / "betting.db"))
    with closing(store._connect()) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not names & tables


def test_raw_db_sits_beside_betting_db_unless_raw_db_path_says_otherwise(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    assert resolve_raw_db_path(None, str(state)) == str(state / "raw.db")
    assert resolve_raw_db_path(None, str(state / "betting.db")) == str(state / "raw.db")
    store = SimpleNamespace(path=str(state / "betting.db"))
    assert resolve_raw_db_path(SimpleNamespace(raw_db_path=None), store) == str(state / "raw.db")
    assert resolve_raw_db_path(SimpleNamespace(raw_db_path=None, state_dir=str(state))) == str(state / "raw.db")
    assert resolve_raw_db_path(SimpleNamespace(raw_db_path="/data/elsewhere.db"), store) == "/data/elsewhere.db"
    monkeypatch.setenv("RAW_DB_PATH", str(tmp_path / "env.db"))
    assert resolve_raw_db_path(None, store) == str(tmp_path / "env.db")
    # The worker's name for it is the same function
    from telegram_betting_bot_simple.capture.worker import raw_db_file
    monkeypatch.delenv("RAW_DB_PATH")
    assert raw_db_file(SimpleNamespace(raw_db_path=None), store) == str(state / "raw.db")


# --- put: deduplicated on the body's sha256 ----------------------------------------------------------------------------

def test_the_same_body_is_kept_once_and_a_corrected_one_is_a_new_row(raw) -> None:
    first = put(raw, b'{"score": 101}', fetched_at=1000.0)
    again = put(raw, b'{"score": 101}', fetched_at=2000.0)
    assert again == first and raw.payload_count() == 1
    corrected = put(raw, b'{"score": 102}', fetched_at=3000.0)
    assert corrected != first and raw.payload_count() == 2
    latest = raw.latest("espn:401", "espn_site", "summary")
    assert latest.id == corrected and latest.json() == {"score": 102} and latest.fetched_at == 3000.0
    # insert says whether a row was added
    assert raw.insert(game_key="espn:401", league=NBA, source="espn_site", endpoint="summary", params={},
                      body=b'{"score": 102}', state="final", http_status=200) == (corrected, False)
    # the same bytes for another game, source, endpoint or params are rows of their own
    assert len({put(raw, b'{"score": 101}', game_key="espn:402"), put(raw, b'{"score": 101}', source="espn_core"),
                put(raw, b'{"score": 101}', endpoint="core_status"),
                put(raw, b'{"score": 101}', params={"page": 2})}) == 4


def test_params_are_canonical_json_with_their_types_kept(raw) -> None:
    assert canonical_params({"page": 1, "limit": 300}) == canonical_params({"limit": 300, "page": 1}) == \
        '{"limit":300,"page":1}'
    assert canonical_params(None) == canonical_params({}) == "{}"
    assert canonical_params({"dates": "20261031", "skip": None}) == '{"dates":"20261031"}'
    assert canonical_params({"page": 1}) != canonical_params({"page": "1"})  # the fetchers fix the types
    one = put(raw, b'{"p": 1}', source="espn_core", endpoint="core_plays", params={"page": 1, "limit": 300})
    assert put(raw, b'{"p": 1}', source="espn_core", endpoint="core_plays", params={"limit": 300, "page": 1}) == one
    found = raw.latest("espn:401", "espn_core", "core_plays", {"limit": 300, "page": 1})
    assert found.id == one and found.params == {"limit": 300, "page": 1}
    assert raw.latest("espn:401", "espn_core", "core_plays", {"page": 2, "limit": 300}) is None


def test_bodies_are_zlib_compressed_unless_they_already_are_gzip_or_zip(raw) -> None:
    text = json.dumps({"plays": [{"id": n, "text": "Corner, Liverpool."} for n in range(400)]}).encode()
    pid = put(raw, text)
    gz = gzip.compress(b"play_id,desc\n1,pass\n")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("a.csv", "x,y\n1,2\n")
    with closing(sqlite3.connect(raw.path)) as db:
        def row(pid: int) -> Tuple[Any, ...]:
            return db.execute("SELECT codec, bytes_raw, sha256, length(body) FROM raw_payloads WHERE id = ?",
                              (pid,)).fetchone()
        codec, size, digest, stored = row(pid)
        assert (codec, size, digest) == ("zlib6", len(text), hashlib.sha256(text).hexdigest())
        assert stored < len(text) / 5  # JSON compresses many times over
        gid = put(raw, gz, game_key="static:nflverse:pbp:2026", league="americanfootball_nfl", source="nflverse",
                  endpoint="pbp_season", params={"season": 2026}, state="static", horizon=None)
        zid = put(raw, zipped.getvalue(), game_key="static:x:y", source="mcp", endpoint="csv", state="static",
                  horizon=None)
        assert row(gid)[0] == "identity" and row(gid)[3] == len(gz)
        assert row(zid)[0] == "identity"
    payload = raw.latest("static:nflverse:pbp:2026", "nflverse", "pbp_season", {"season": 2026})
    assert payload.raw_bytes() == gz  # as fetched, byte for byte
    assert payload.text() == "play_id,desc\n1,pass\n"
    assert raw.latest("espn:401", "espn_site", "summary").raw_bytes() == text
    with pytest.raises(ValueError):
        raw.latest("static:x:y", "mcp", "csv").text()


def test_text_reads_utf8_with_a_bom_and_falls_back_to_windows_1252(raw) -> None:
    put(raw, "﻿Div,HomeTeam\nE0,Arsenal\n".encode("utf-8"), game_key="static:fd:a", source="football_data",
        endpoint="csv", state="static", horizon=None)
    put(raw, "Div,HomeTeam\nSP1,Alavés\n".encode("cp1252"), game_key="static:fd:b", source="football_data",
        endpoint="csv", state="static", horizon=None)
    assert raw.latest("static:fd:a", "football_data", "csv").text().startswith("Div,HomeTeam")
    assert "Alavés" in raw.latest("static:fd:b", "football_data", "csv").text()


def test_put_refuses_unknown_states_and_horizons(raw) -> None:
    with pytest.raises(ValueError):
        put(raw, state="finished")
    with pytest.raises(ValueError):
        put(raw, horizon="t48")
    with pytest.raises(ValueError):
        put(raw, game_key="")
    for horizon in ("open", "t24", "t90", "close", "final", None):
        put(raw, json.dumps({"h": horizon}).encode(), state="pre", horizon=horizon)


# --- reads --------------------------------------------------------------------------------------------------------------

def test_latest_takes_only_2xx_answers_unless_asked(raw) -> None:
    good = put(raw, b'{"ok": 1}', fetched_at=1000.0)
    bad = put(raw, b'{"code": 503}', http_status=503, horizon=None, fetched_at=2000.0)
    assert raw.latest("espn:401", "espn_site", "summary").id == good
    assert raw.latest("espn:401", "espn_site", "summary", ok_only=False).id == bad
    assert raw.latest("espn:401", "espn_site", "summary", {}).id == good
    assert raw.latest("espn:999", "espn_site", "summary") is None


def test_a_bundle_is_the_newest_payload_per_source_endpoint_and_params(raw) -> None:
    put(raw, b'{"v": 1}', fetched_at=1000.0)
    put(raw, b'{"v": 2}', fetched_at=2000.0)
    for page in (3, 1, 2, 10):
        put(raw, json.dumps({"page": page}).encode(), source="espn_core", endpoint="core_plays",
            params={"page": page, "limit": 300}, fetched_at=1500.0 + page)
    put(raw, b'{"code": 500}', source="espn_core", endpoint="core_plays", params={"page": 4, "limit": 300},
        http_status=500, horizon=None)
    put(raw, b'{"x": 1}', game_key="espn:402")
    bundle = raw.bundle("espn:401")
    assert bundle.game_key == "espn:401" and bundle.league == NBA and len(bundle) == 5 and bool(bundle)
    assert bundle.get("espn_site", "summary").json() == {"v": 2}
    assert [p.json()["page"] for p in bundle.pages("espn_core", "core_plays")] == [1, 2, 3, 10]
    assert bundle.get("espn_core", "core_plays", {"page": 2, "limit": 300}).json() == {"page": 2}
    assert ("espn_core", "core_plays") in bundle and ("mlb_statsapi", "feed_live") not in bundle
    assert bundle.endpoints() == {("espn_site", "summary"), ("espn_core", "core_plays")}
    assert len(raw.bundle("espn:401", ok_only=False)) == 6  # the error page too, when asked
    assert not raw.bundle("espn:999")


def test_iter_games_by_league_and_fetch_time(raw) -> None:
    put(raw, b"1", game_key="espn:1", fetched_at=100.0)
    put(raw, b"2", game_key="espn:2", league="icehockey_nhl", fetched_at=200.0)
    put(raw, b"3", game_key="espn:3", fetched_at=300.0)
    put(raw, b"4", game_key="espn:1", fetched_at=400.0)
    assert list(raw.iter_games()) == ["espn:1", "espn:2", "espn:3"]
    assert list(raw.iter_games(league=NBA)) == ["espn:1", "espn:3"]
    assert list(raw.iter_games(since=150.0)) == ["espn:2", "espn:3", "espn:1"]  # espn:1's payload at 400
    assert list(raw.iter_games(since=150.0, until=300.0)) == ["espn:2"]
    assert list(raw.iter_games(league="soccer_epl")) == []


def test_has_final_needs_a_2xx_payload_taken_once_the_game_was_final(raw) -> None:
    put(raw, b'{"pre": 1}', state="pre", horizon="t90")
    assert not raw.has_final("espn:401")
    put(raw, b'{"err": 1}', http_status=404, horizon=None)
    assert not raw.has_final("espn:401")
    put(raw, b'{"final": 1}')
    assert raw.has_final("espn:401") and not raw.has_final("espn:401", "espn_core", "core_plays")


# --- budgets: atomic counters --------------------------------------------------------------------------------------------

def test_spend_counts_against_a_limit_and_never_passes_it(raw) -> None:
    day = datetime.date(2026, 11, 1)
    assert raw.spend("capture", day, 2, limit=3) and raw.spent("capture", "2026-11-01") == 2
    assert not raw.spend("capture", day, 2, limit=3) and raw.spent("capture", day) == 2  # all or nothing
    assert raw.spend("capture", day, 1, limit=3) and not raw.spend("capture", day, 1, limit=3)
    assert raw.spend("backfill", day) and raw.spend("prices", day, 5) and raw.spent("prices", day) == 5
    assert raw.spent("backfill", "2026-11-02") == 0
    assert raw.spend("http_ok", day) and raw.spend("http_error", day)
    with pytest.raises(ValueError):
        raw.spend("odds_api", day)
    with pytest.raises(ValueError):
        raw.spend("capture", "yesterday")
    assert raw.spend("backfill_http_ok", day) and raw.spend("backfill_http_error", day)
    assert {row["kind"] for row in raw.counts()} == {"capture", "backfill", "prices", "http_ok", "http_error",
                                                     "backfill_http_ok", "backfill_http_error"}
    assert set(BUDGET_KINDS) == {"capture", "backfill", "prices", "http_ok", "http_error", "backfill_http_ok",
                                 "backfill_http_error"}


def test_two_processes_never_both_take_the_last_request(tmp_path) -> None:
    path = str(tmp_path / "raw.db")
    stores = [RawStore(path), RawStore(path)]
    taken: List[bool] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def take(store: RawStore) -> None:
        start.wait()
        for _ in range(20):
            got = store.spend("capture", "2026-11-01", 1, limit=50)
            with lock:
                taken.append(got)

    threads = [threading.Thread(target=take, args=(stores[n % 2],)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert taken.count(True) == 50 and stores[0].spent("capture", "2026-11-01") == 50


def test_old_counters_are_dropped(raw) -> None:
    raw.spend("capture", "2025-01-01")
    raw.spend("capture", "2026-11-01")
    assert [row["day"] for row in raw.counts(["capture"])] == ["2026-11-01"]
    assert raw.counts(["capture"], since_day="2026-11-02") == []


# --- the backfill's cursors ------------------------------------------------------------------------------------------------

def test_cursors_and_counts_survive_a_restart(tmp_path) -> None:
    path = str(tmp_path / "raw.db")
    raw = RawStore(path)
    assert raw.get_cursor(NBA, 2025) is None and raw.progress() == []
    raw.set_cursor(NBA, 2025, {"window": "2026-04-13"}, add_seen=12, add_stored=10, now=10.0)
    raw.set_cursor(NBA, 2025, {"window": "2026-04-06"}, add_seen=5, add_stored=5, now=20.0)
    raw.set_cursor("americanfootball_nfl", 2025, {"week": 9}, games_seen=3, games_stored=1, now=30.0)
    again = RawStore(path)  # a restarted process
    assert again.get_cursor(NBA, 2025) == {"window": "2026-04-06"}
    assert again.progress(NBA) == [{"league": NBA, "season": 2025, "cursor": {"window": "2026-04-06"}, "done": False,
                                    "games_seen": 17, "games_stored": 15, "updated_at": 20.0}]
    again.set_cursor(NBA, 2025, None, done=True, now=40.0)
    row = again.progress(NBA)[0]
    assert row["done"] and row["cursor"] is None and row["games_seen"] == 17
    again.set_cursor(NBA, 2025, {"x": 1}, now=50.0)  # done stays as it was when not given
    assert again.progress(NBA)[0]["done"]
    again.set_cursor(NBA, 2024, {"window": "2025-06-01"}, now=60.0)
    assert [(r["league"], r["season"]) for r in again.progress()] == [
        ("americanfootball_nfl", 2025), (NBA, 2025), (NBA, 2024)]


# --- jobs -------------------------------------------------------------------------------------------------------------------

def test_jobs_are_one_per_game_and_change_in_one_transaction(raw) -> None:
    assert raw.add_job("espn:1", NBA, "1", 1000.0, 500.0, now=1.0)
    assert not raw.add_job("espn:1", NBA, "1", 9999.0, 9999.0)  # one per game: the first stays
    assert raw.add_job("espn:2", NBA, "2", 2000.0, 300.0)
    assert raw.add_job("espn:3", NBA, "3", 3000.0, 900.0, state="done", kinds_done=["final:summary"])
    assert [job.game_key for job in raw.due_jobs(600.0)] == ["espn:2", "espn:1"]  # earliest due first
    assert raw.next_due() == 300.0
    job = raw.job("espn:1")
    assert (job.start, job.due_at, job.state, job.attempts, job.kinds_done) == (1000.0, 500.0, "todo", 0, [])

    def advance(job: Optional[Job]) -> Job:
        assert job is not None
        job.kinds_done += ["pre:t90:summary", "pre:t90:summary"]
        job.state, job.attempts, job.last_error = "partial", 1, "x" * 900
        return job

    raw.modify_job("espn:1", advance, now=5.0)
    job = raw.job("espn:1")
    assert job.kinds_done == ["pre:t90:summary"] and job.state == "partial" and len(job.last_error) == 500
    assert raw.modify_job("espn:404", lambda job: None) is None and raw.job("espn:404") is None
    assert raw.job_counts() == {"todo": 1, "partial": 1, "done": 1, "gave_up": 0, "void": 0}
    assert [job.game_key for job in raw.jobs(states=["todo", "partial"])] == ["espn:1", "espn:2"]
    assert [job.game_key for job in raw.jobs(since=1500.0, until=3000.0)] == ["espn:2"]
    assert raw.jobs(prefix="espn-match:") == []
    with pytest.raises(ValueError):
        raw.modify_job("espn:1", lambda job: Job(job.game_key, NBA, "1", 1.0, 1.0, state="finished"))
    assert raw.job("espn:1").state == "partial"  # a failed change writes nothing
    assert set(JOB_STATES) == {"todo", "partial", "done", "gave_up", "void"}


def test_the_health_helpers(raw) -> None:
    put(raw, b'{"a": 1}')
    put(raw, b'{"code": 404}', http_status=404, horizon=None, fetched_at=2000.0)
    put(raw, b'{"code": 503}', http_status=503, horizon=None, fetched_at=3000.0, game_key="espn:402")
    raw.add_job("espn:403", NBA, "403", 10.0, 10.0)
    raw.modify_job("espn:403", lambda job: Job(**{**job.__dict__, "last_error": "summary: HTTP 500"}), now=2500.0)
    assert raw.http_counts(since=0) == {"2xx": 1, "3xx": 0, "4xx": 1, "5xx": 1}
    assert raw.http_counts(since=2500.0) == {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 1}
    assert raw.http_counts() == {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}  # by default the last 7 days only
    errors = raw.recent_errors(limit=5, since=0)
    assert [e["at"] for e in errors] == [3000.0, 2500.0, 2000.0]
    assert raw.recent_errors(limit=5) == []  # (these are from 1970)
    assert errors[1]["error"] == "summary: HTTP 500" and errors[0]["status"] == 503
    assert raw.size_bytes() > 0
    raw.set_state("static:greco1899", {"at": 5.0, "ok": True})
    assert raw.get_state("static:greco1899") == {"at": 5.0, "ok": True} and raw.get_state("nope", 7) == 7


# --- PoliteHttp: per-host token buckets ---------------------------------------------------------------------------------------

class FakeClock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now
        self.slept: List[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Recorder:
    """Stands in for PoliteHttp._send: records (time, url, headers) and answers from ``answers`` (url substring ->
    HttpResponse, or an exception to raise); 200 {} otherwise."""

    def __init__(self, clock: FakeClock, answers: Optional[Dict[str, Any]] = None) -> None:
        self.clock = clock
        self.answers = answers or {}
        self.sent: List[Tuple[float, str, Dict[str, str]]] = []

    def __call__(self, request: Any) -> HttpResponse:
        self.sent.append((self.clock(), request.full_url, dict(request.header_items())))
        for part, answer in self.answers.items():
            if part in request.full_url:
                if isinstance(answer, BaseException):
                    raise answer
                return answer
        return HttpResponse(200, b"{}", {})


def polite(clock: FakeClock, answers: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[PoliteHttp, Recorder]:
    http = PoliteHttp(clock=clock, sleep=clock.sleep, **kwargs)
    recorder = Recorder(clock, answers)
    http._send = recorder  # type: ignore[method-assign]
    return http, recorder


HOSTS = {  # host -> (rate, burst) of section 3.3
    "site.api.espn.com": (1.0, 3), "sports.core.api.espn.com": (1.0, 3), "site.web.api.espn.com": (0.5, 1),
    "statsapi.mlb.com": (0.5, 1), "raw.githubusercontent.com": (0.2, 1), "github.com": (0.2, 1),
    "football-data.co.uk": (0.5, 1),
}


def assert_within_rate(times: List[float], rate: float, burst: int) -> None:
    """No window of the request times holds more than burst + rate x its length."""
    times = sorted(times)
    for i in range(len(times)):
        for j in range(i, len(times)):
            assert j - i + 1 <= burst + rate * (times[j] - times[i]) + 1e-6, (i, j, times[i], times[j])


def test_no_host_goes_above_its_rate(tmp_path) -> None:
    clock = FakeClock()
    http, sent = polite(clock)
    hosts = list(HOSTS)
    for n in range(200):  # interleaved, as the worker and the backfill would ask
        host = hosts[(n * 3) % len(hosts)] if n % 5 else "site.api.espn.com"
        http.get(f"https://{host}/path/{n}")
    by_host: Dict[str, List[float]] = {}
    for at, url, _headers in sent.sent:
        by_host.setdefault(url.split("/")[2], []).append(at)
    assert set(by_host) == set(HOSTS)
    for host, times in by_host.items():
        rate, burst = HOSTS[host]
        assert_within_rate(times, rate, burst)


def test_espn_s_site_and_core_hosts_burst_three_then_go_one_a_second() -> None:
    for host in ("site.api.espn.com", "sports.core.api.espn.com"):
        clock = FakeClock()
        http, sent = polite(clock)
        for n in range(8):
            http.get(f"https://{host}/{n}")
        times = [at - sent.sent[0][0] for at, _url, _h in sent.sent]
        assert times == pytest.approx([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        clock.now += 60.0  # an idle minute refills the burst, never more than 3
        start = clock.now
        for n in range(4):
            http.get(f"https://{host}/again/{n}")
        assert [at - start for at, _url, _h in sent.sent[-4:]] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    clock = FakeClock()
    http, sent = polite(clock)
    for n in range(100):
        http.get(f"https://site.api.espn.com/{n}")
    assert sent.sent[-1][0] - sent.sent[0][0] == pytest.approx(97.0)  # sustained: 1 a second after the burst


def test_the_github_hosts_share_one_bucket_and_other_hosts_get_half_a_request_a_second() -> None:
    clock = FakeClock()
    http, sent = polite(clock)
    for url in ("https://github.com/nflverse/a.csv.gz", "https://raw.githubusercontent.com/x/b.csv",
                "https://objects.githubusercontent.com/c", "https://release-assets.githubusercontent.com/d"):
        http.get(url)
    times = [at for at, _url, _h in sent.sent]
    assert [b - a for a, b in zip(times, times[1:])] == pytest.approx([5.0, 5.0, 5.0])  # 0.2 a second, together
    for n in range(3):
        http.get(f"https://api.elections.kalshi.com/trade-api/v2/events?n={n}")
    kalshi = [at for at, url, _h in sent.sent if "kalshi" in url]
    assert [b - a for a, b in zip(kalshi, kalshi[1:])] == pytest.approx([2.0, 2.0])
    assert PoliteHttp(rates={"example.test": (4.0, 2)}).rates["example.test"] == (4.0, 2)
    with pytest.raises(ValueError):
        PoliteHttp(rates={"example.test": 0})


def test_requests_carry_the_stable_user_agent_and_the_query() -> None:
    clock = FakeClock()
    http, sent = polite(clock)
    http.get("https://site.api.espn.com/s", {"event": 401, "skip": None})
    http.get("https://site.api.espn.com/s?x=1", {"page": 2}, headers={"User-Agent": "curl/8.5"})
    (_, url, headers), (_, url2, headers2) = sent.sent
    assert url == "https://site.api.espn.com/s?event=401" and url2 == "https://site.api.espn.com/s?x=1&page=2"
    assert headers["User-agent"] == capture_http.USER_AGENT and headers["Accept-encoding"] == "gzip"
    assert headers2["User-agent"] == "curl/8.5"
    assert http.requests == {"site.api.espn.com": 2}
    assert http.clock is clock  # the sources stamp fetched_at with it


# --- PoliteHttp: back-off and the breaker ------------------------------------------------------------------------------------

def test_a_failing_host_backs_off_exponentially_then_breaks_and_a_success_resets_it() -> None:
    clock = FakeClock()
    http, sent = polite(clock, {"down": HttpResponse(503, b"busy", {})})
    url = "https://statsapi.mlb.com/down"
    assert http.get(url).status == 503  # returned, never raised
    assert http.backing_off("statsapi.mlb.com") and http.backing_off(url)
    assert not http.backing_off("site.api.espn.com")
    waits = []
    for _ in range(4):
        before = clock.now
        http.get(url)
        waits.append(round(clock.now - before, 6))
    assert waits == [5.0, 10.0, 20.0, 40.0]  # exponential, slept (the bucket's 2 s lies under each)
    assert http.backoff_until("statsapi.mlb.com") == pytest.approx(clock.now + 80.0)
    sent_before, slept_before = len(sent.sent), len(clock.slept)
    with pytest.raises(Backoff) as raised:
        http.get(url)  # 80 s to go: more than MAX_WAIT, so nothing is sent and nothing is slept
    assert isinstance(raised.value, Unreachable) and len(sent.sent) == sent_before and 80.0 > MAX_WAIT
    assert len(clock.slept) == slept_before and raised.value.until == pytest.approx(clock.now + 80.0)
    for failures in (6, 7):
        clock.now = http.backoff_until("statsapi.mlb.com")
        http.get(url)
        assert http.backoff_until("statsapi.mlb.com") == pytest.approx(clock.now + 5.0 * 2 ** (failures - 1))
    clock.now = http.backoff_until("statsapi.mlb.com")
    http.get(url)  # the 8th failure in a row: the breaker keeps the host closed for the cap
    assert http.backoff_until("statsapi.mlb.com") == pytest.approx(clock.now + BACKOFF_CAP) and BREAKER == 8
    clock.now += BACKOFF_CAP
    sent.answers.clear()
    assert http.get(url).status == 200 and not http.backing_off("statsapi.mlb.com")
    before = clock.now
    http.get(url)
    assert clock.now - before == pytest.approx(2.0)  # only the bucket's pace again


def test_429_honours_retry_after_up_to_the_cap_and_404_is_no_failure() -> None:
    clock = FakeClock()
    http, _sent = polite(clock, {"limited": HttpResponse(429, b"", {"retry-after": "120"}),
                                 "forever": HttpResponse(429, b"", {"retry-after": "999999"}),
                                 "missing": HttpResponse(404, b"{}", {})})
    http.get("https://site.web.api.espn.com/limited")
    assert http.backoff_until("site.web.api.espn.com") == pytest.approx(clock.now + 120.0)
    http.get("https://sports.core.api.espn.com/forever")
    assert http.backoff_until("sports.core.api.espn.com") == pytest.approx(clock.now + BACKOFF_CAP)
    for _ in range(10):
        assert http.get("https://site.api.espn.com/missing").status == 404
    assert not http.backing_off("site.api.espn.com")


def test_connection_errors_raise_unreachable_and_count_as_failures() -> None:
    clock = FakeClock()
    http, _sent = polite(clock, {"reset": ConnectionResetError("reset by peer"), "timeout": TimeoutError("slow"),
                                 "dns": urllib.error.URLError("no such host")})
    seen: List[Tuple[str, Optional[int]]] = []
    http.on_result = lambda url, status: seen.append((url.rsplit("/", 1)[1], status))
    for name in ("reset", "timeout", "dns"):
        with pytest.raises(Unreachable):
            http.get(f"https://site.api.espn.com/{name}")
        clock.now += 3600
    http.get("https://site.api.espn.com/fine")
    assert seen == [("reset", None), ("timeout", None), ("dns", None), ("fine", 200)]


# --- PoliteHttp: guards (the budgets) ------------------------------------------------------------------------------------------

def test_a_guard_sees_every_request_first_and_its_refusal_sends_nothing() -> None:
    clock = FakeClock()
    http, sent = polite(clock)
    asked: List[str] = []

    def allow(url: str) -> None:
        asked.append(url)
        if len(asked) > 2:
            raise Refused("budget")

    with http.guard(allow):
        http.get("https://site.api.espn.com/1")
        http.get("https://site.api.espn.com/2")
        with pytest.raises(Refused):
            http.get("https://site.api.espn.com/3")
        inner: List[str] = []
        with http.guard(inner.append):  # an inner guard replaces the outer one for its block
            http.get("https://site.api.espn.com/4")
        assert inner == ["https://site.api.espn.com/4"]
    http.get("https://site.api.espn.com/5")  # no guard outside the block
    assert [url[-1] for _t, url, _h in sent.sent] == ["1", "2", "4", "5"] and len(asked) == 3
    # another thread's requests aren't this block's
    other: List[str] = []
    with http.guard(lambda url: (_ for _ in ()).throw(Refused("no"))):
        thread = threading.Thread(target=lambda: other.append(http.get("https://site.api.espn.com/6").status))
        thread.start()
        thread.join()
    assert other == [200]


# --- PoliteHttp: the real send (urllib, faked below the opener) --------------------------------------------------------------------

class FakeAnswer(io.BytesIO):
    def __init__(self, status: int, body: bytes, headers: Dict[str, str]) -> None:
        super().__init__(body)
        self.status = status
        self.headers = email.message.Message()
        for key, value in headers.items():
            self.headers[key] = value

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeAnswer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FakeOpener:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.opened: List[Tuple[str, float]] = []

    def open(self, request: Any, timeout: float = 0.0) -> Any:
        self.opened.append((request.full_url, timeout))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def test_gzip_transfer_encoding_is_undone_on_every_status_and_a_gz_file_is_left_alone() -> None:
    payload = json.dumps({"events": []}).encode()
    http = PoliteHttp(clock=FakeClock(), sleep=lambda s: None, timeout=7.0)
    http._opener = FakeOpener(FakeAnswer(200, gzip.compress(payload), {"Content-Encoding": "gzip"}))
    answer = http.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")
    assert (answer.status, answer.body) == (200, payload) and answer.headers["content-encoding"] == "gzip"
    assert http._opener.opened[0][1] == 7.0
    # ESPN gzips its 400s too (asked or not): HTTPError's body is decoded the same way, and returned, not raised
    error = b'{"code":400,"message":"Failed to get events endpoint."}'
    headers = email.message.Message()
    headers["Content-Encoding"] = "gzip"
    http._opener = FakeOpener(urllib.error.HTTPError("https://site.api.espn.com/x", 400, "Bad Request", headers,
                                                     io.BytesIO(gzip.compress(error))))
    answer = http.get("https://site.api.espn.com/x")
    assert (answer.status, answer.body) == (400, error)
    # nflverse's .csv.gz: no Content-Encoding, so the file stays a gzip file, byte for byte
    csv_gz = gzip.compress(b"play_id,desc\n1,run\n")
    http._opener = FakeOpener(FakeAnswer(200, csv_gz, {"Content-Type": "application/octet-stream"}))
    assert http.get("https://github.com/nflverse/nflverse-data/releases/download/pbp/x.csv.gz").body == csv_gz
    # a body that says gzip but isn't stays as it came
    http._opener = FakeOpener(FakeAnswer(200, b"not gzip", {"Content-Encoding": "gzip"}))
    assert http.get("https://example.test/odd").body == b"not gzip"
    # a connection error below urllib is Unreachable
    http._opener = FakeOpener(urllib.error.URLError(OSError("network is down")))
    with pytest.raises(Unreachable):
        http.get("https://example.test/down")


def test_the_proxy_and_ca_bundle_come_from_the_environment(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "ca.pem"
    bundle.write_text("not read here")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert capture_http.ca_bundle() == (str(bundle), None)
    assert capture_http.ca_bundle({"SSL_CERT_FILE": str(tmp_path / "missing.pem")}) == (None, None)
    for name in ("HTTPS_PROXY", "https_proxy"):  # urllib reads either (the lower-case one wins)
        monkeypatch.setenv(name, "http://proxy.test:3128")
    contexts: List[Any] = []
    monkeypatch.setattr(capture_http.ssl, "create_default_context",
                        lambda cafile=None, capath=None: contexts.append((cafile, capath)) or
                        __import__("ssl").SSLContext(__import__("ssl").PROTOCOL_TLS_CLIENT))
    opener = capture_http.build_opener()
    proxies = [h for h in opener.handlers if isinstance(h, capture_http.urllib.request.ProxyHandler)]
    assert proxies and proxies[0].proxies.get("https") == "http://proxy.test:3128"
    assert contexts == [(str(bundle), None)]


def test_no_capture_module_but_the_door_imports_an_http_library() -> None:
    package = Path(capture_http.__file__).resolve().parent
    import re
    pattern = re.compile(r"^\s*(?:import|from)\s+(requests|httpx|urllib3|aiohttp|http\.client|urllib\.request|socket)\b",
                         re.M)
    talking = sorted(str(path.relative_to(package)) for path in package.rglob("*.py")
                     if pattern.search(path.read_text(encoding="utf-8")))
    assert talking == ["http.py"]


# --- the disk floor: raw.db never fills the disk betting.db is on ------------------------------------------------------------

def test_disk_room_needs_the_floor_and_a_tenth_of_the_disk_free(tmp_path, monkeypatch) -> None:
    from collections import namedtuple
    from telegram_betting_bot_simple.capture import raw_store
    Usage = namedtuple("Usage", "total used free")
    seen: List[str] = []

    def usage(free: float, total: float = 100e9):
        def read(folder: str):
            seen.append(folder)
            return Usage(total, total - free, free)
        return read

    monkeypatch.setattr(raw_store.shutil, "disk_usage", usage(free=20e9))
    assert raw_store.disk_room(str(tmp_path / "raw.db"), 2 * 1024 ** 3) and seen[-1] == str(tmp_path)
    monkeypatch.setattr(raw_store.shutil, "disk_usage", usage(free=9.9e9))  # under 10% of the disk
    assert not raw_store.disk_room(str(tmp_path / "raw.db"), 2 * 1024 ** 3)
    monkeypatch.setattr(raw_store.shutil, "disk_usage", usage(free=1.5 * 1024 ** 3, total=10e9))  # under 2 GiB
    assert not raw_store.disk_room(str(tmp_path / "raw.db"), 2 * 1024 ** 3)
    monkeypatch.setattr(raw_store.shutil, "disk_usage", usage(free=1.5 * 1024 ** 3, total=10e9))
    assert raw_store.disk_room(str(tmp_path / "raw.db"), 1 * 1024 ** 3)
    # a folder not made yet is judged by the nearest one that exists
    monkeypatch.setattr(raw_store.shutil, "disk_usage", usage(free=50e9))
    assert raw_store.disk_room(str(tmp_path / "not" / "yet" / "raw.db"), 0) and seen[-1] == str(tmp_path)

    def broken(folder: str):
        raise OSError("can't read the disk")

    monkeypatch.setattr(raw_store.shutil, "disk_usage", broken)
    assert raw_store.disk_room(str(tmp_path / "raw.db"), 2 * 1024 ** 3)  # unreadable: never stops the capture
    assert raw_store.DISK_FLOOR_SHARE == 0.10


def test_room_reads_the_disk_at_most_once_a_minute(tmp_path, monkeypatch) -> None:
    from collections import namedtuple
    from telegram_betting_bot_simple.capture import raw_store
    Usage = namedtuple("Usage", "total used free")
    free = {"now": 1e9}
    reads: List[int] = []

    def read(folder: str):
        reads.append(1)
        return Usage(100e9, 100e9 - free["now"], free["now"])

    monkeypatch.setattr(raw_store.shutil, "disk_usage", read)
    clock = FakeClock()
    raw = RawStore(str(tmp_path / "raw.db"), min_free_gb=2, clock=clock)
    assert raw.min_free == 2 * 1024 ** 3
    assert not raw.room() and len(reads) == 1
    free["now"] = 50e9
    clock.now += 30
    assert not raw.room() and len(reads) == 1  # cached
    clock.now += 31
    assert raw.room() and len(reads) == 2
    assert RawStore(str(tmp_path / "raw.db"), min_free_gb="0.5").min_free == 0.5 * 1024 ** 3
    monkeypatch.setenv("CAPTURE_MIN_FREE_GB", "5")
    assert RawStore(str(tmp_path / "raw.db")).min_free == 5 * 1024 ** 3
    monkeypatch.setenv("CAPTURE_MIN_FREE_GB", "lots")
    assert RawStore(str(tmp_path / "raw.db")).min_free == 2 * 1024 ** 3


def test_the_health_reads_never_scan_the_payload_table(raw) -> None:
    """http_counts and recent_errors read raw_payloads by its fetch-time index (the backfill takes it to ~100k rows,
    4 GB): EXPLAIN QUERY PLAN shows a SEARCH, never a SCAN of raw_payloads."""
    import re as re_module
    for n in range(300):
        put(raw, json.dumps({"n": n}).encode(), game_key=f"espn:{n}", fetched_at=1000.0 + n,
            http_status=200 if n % 7 else 503)
    statements: List[str] = []
    real_connect = raw._connect

    def traced() -> sqlite3.Connection:
        db = real_connect()
        db.set_trace_callback(statements.append)
        return db

    raw._connect = traced  # type: ignore[method-assign]
    assert raw.http_counts(since=1200.0)["5xx"] == len([n for n in range(200, 300) if not n % 7])
    assert len(raw.recent_errors(limit=3, since=0)) == 3
    selects = [text for text in statements if text.lstrip().upper().startswith("SELECT") and "raw_payloads" in text]
    assert len(selects) == 2
    with closing(sqlite3.connect(raw.path)) as db:
        for text in selects:
            plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + text)]
            assert any(step.startswith("SEARCH raw_payloads USING") and "raw_payloads_fetched" in step for step in plan)
            assert not any(re_module.match(r"SCAN raw_payloads\b", step) for step in plan), plan


def test_a_backing_off_host_costs_no_budget() -> None:
    clock = FakeClock()
    http, sent = polite(clock, {"down": HttpResponse(503, b"", {})})
    spent: List[str] = []
    with http.guard(spent.append):
        for _ in range(5):
            http.get("https://statsapi.mlb.com/down")  # 5, 10, 20, 40 s waits: then 80 s
        assert len(spent) == 5
        for _ in range(3):
            with pytest.raises(Backoff):
                http.get("https://statsapi.mlb.com/down")
    assert len(spent) == 5 and len(sent.sent) == 5  # nothing sent, nothing counted


# --- sightings: the newest answer, and every horizon a body served (F3) ---------------------------------------------------

SIGHT = dict(game_key="espn:401", league=NBA, source="espn_site", endpoint="summary", params={}, http_status=200,
             state="pre")


def test_a_body_that_comes_back_is_the_newest_answer_again(raw) -> None:
    """A, then B, then A (a reverted correction, a line that moves and moves back): latest() and bundle() give the
    second A, with its sighting time; no new row is stored for it."""
    a, b = b'{"header": {"v": "A"}}', b'{"header": {"v": "B"}}'
    assert raw.insert(body=a, horizon="t24", fetched_at=1000.0, **SIGHT) == (1, True)
    assert raw.insert(body=b, horizon="t90", fetched_at=2000.0, **SIGHT) == (2, True)
    assert raw.insert(body=a, horizon="close", fetched_at=3000.0, **SIGHT) == (1, False)
    latest = raw.latest("espn:401", "espn_site", "summary")
    assert latest.raw_bytes() == a and latest.seen_at == 3000.0 and latest.fetched_at == 1000.0
    assert latest.horizon == "t24"  # as first kept: the sighting carries the later horizon
    (bundled,) = raw.bundle("espn:401").payloads
    assert bundled.raw_bytes() == a and bundled.seen_at == 3000.0
    assert raw.payload_count() == 2
    assert raw.sightings(1) == [{"seen_at": 1000.0, "horizon": "t24", "state": "pre"},
                                {"seen_at": 3000.0, "horizon": "close", "state": "pre"}]
    assert raw.sightings(2) == [{"seen_at": 2000.0, "horizon": "t90", "state": "pre"}] and raw.sightings(99) == []
    # a body seen once: seen_at is its fetch time
    assert raw.latest("espn:401", "espn_site", "summary", horizon="t90").seen_at == 2000.0


def test_a_horizon_query_finds_a_snapshot_identical_to_an_earlier_one(raw) -> None:
    props = dict(SIGHT, source="espn_propbets", endpoint="propbets", params={"limit": 500, "page": 1})
    raw.insert(body=b'{"items": [1]}', horizon="t90", fetched_at=5000.0, **props)
    raw.insert(body=b'{"items": [1]}', horizon="close", fetched_at=9000.0, **props)  # unchanged at close
    close = raw.latest("espn:401", "espn_propbets", "propbets", {"limit": 500, "page": 1}, horizon="close")
    t90 = raw.latest("espn:401", "espn_propbets", "propbets", horizon="t90")
    assert close is not None and t90 is not None and close.id == t90.id and close.seen_at == 9000.0
    assert raw.latest("espn:401", "espn_propbets", "propbets", horizon="t24") is None
    # the newest of two bodies seen at one horizon wins
    raw.insert(body=b'{"items": [2]}', horizon="close", fetched_at=9500.0, **props)
    assert raw.latest("espn:401", "espn_propbets", "propbets", horizon="close").raw_bytes() == b'{"items": [2]}'
    assert raw.latest("espn:401", "espn_propbets", "propbets", horizon="t90").raw_bytes() == b'{"items": [1]}'


def test_has_final_counts_a_body_seen_again_once_the_game_was_final(raw) -> None:
    board = dict(SIGHT, game_key="static:espn_site:scoreboard:basketball_nba:20261030", source="espn_site",
                 endpoint="scoreboard", params={"dates": "20261030"})
    raw.insert(body=b'{"events": []}', state="live", horizon=None, fetched_at=10.0,
               **{k: v for k, v in board.items() if k != "state"})
    assert not raw.has_final(board["game_key"], "espn_site", "scoreboard")
    raw.insert(body=b'{"events": []}', state="final", horizon="final", fetched_at=20.0,
               **{k: v for k, v in board.items() if k != "state"})
    assert raw.has_final(board["game_key"], "espn_site", "scoreboard") and raw.payload_count() == 1


OLD_SCHEMA = """
CREATE TABLE raw_payloads (id INTEGER PRIMARY KEY, game_key TEXT NOT NULL, league TEXT NOT NULL, source TEXT NOT NULL,
    endpoint TEXT NOT NULL, params TEXT NOT NULL, state TEXT NOT NULL, horizon TEXT, fetched_at REAL NOT NULL,
    http_status INTEGER NOT NULL, sha256 TEXT NOT NULL, bytes_raw INTEGER NOT NULL, codec TEXT NOT NULL DEFAULT 'zlib6',
    body BLOB NOT NULL, UNIQUE (game_key, source, endpoint, params, sha256));
CREATE INDEX raw_by_game ON raw_payloads (game_key, source, endpoint, fetched_at);
CREATE TABLE capture_jobs (game_key TEXT PRIMARY KEY, league TEXT NOT NULL, espn_id TEXT, start REAL NOT NULL,
    due_at REAL NOT NULL, state TEXT NOT NULL DEFAULT 'todo', attempts INTEGER NOT NULL DEFAULT 0,
    kinds_done TEXT NOT NULL DEFAULT '[]', last_error TEXT, updated_at REAL);
CREATE INDEX capture_jobs_due ON capture_jobs (state, due_at);
"""


def test_a_raw_db_from_before_the_sightings_opens_and_works(tmp_path) -> None:
    import zlib
    path = str(tmp_path / "raw.db")
    with closing(sqlite3.connect(path)) as db:
        db.executescript(OLD_SCHEMA)
        body = b'{"header": {"v": "old"}}'
        db.execute("INSERT INTO raw_payloads (game_key, league, source, endpoint, params, state, horizon, fetched_at, "
                   "http_status, sha256, bytes_raw, codec, body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   ("espn:7", NBA, "espn_site", "summary", "{}", "final", "final", 100.0, 200,
                    hashlib.sha256(body).hexdigest(), len(body), "zlib6", zlib.compress(body, 6)))
        db.commit()
    raw = RawStore(path)  # additive: the new tables and indexes are made, the rows stay
    old = raw.latest("espn:7", "espn_site", "summary")
    assert old.json() == {"header": {"v": "old"}} and old.seen_at == old.fetched_at == 100.0
    assert raw.latest("espn:7", "espn_site", "summary", horizon="final").id == old.id  # its own horizon counts
    assert raw.has_final("espn:7") and raw.sightings(old.id) == [{"seen_at": 100.0, "horizon": "final",
                                                                  "state": "final"}]
    assert raw.insert(game_key="espn:7", league=NBA, source="espn_site", endpoint="summary", params={},
                      body=b'{"header": {"v": "old"}}', state="final", http_status=200, horizon="final",
                      fetched_at=500.0) == (old.id, False)
    assert raw.latest("espn:7", "espn_site", "summary").seen_at == 500.0
    with closing(sqlite3.connect(path)) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    assert {"raw_seen", "raw_seen_by_payload", "capture_jobs_start", "capture_jobs_errors", "capture_budget"} <= names


def test_the_health_blocks_job_reads_use_their_indexes(raw) -> None:
    """F5: coverage by start and the newest errors read capture_jobs through an index, never a scan of the table."""
    for n in range(200):
        raw.add_job(f"espn:{n}", NBA, str(n), 1000.0 + n, 2000.0 + n, now=3000.0 + n)
    raw.modify_job("espn:7", lambda job: Job(**{**job.__dict__, "last_error": "summary: HTTP 500"}), now=4000.0)
    shapes = [("SELECT j.league, j.start, j.state FROM capture_jobs j WHERE j.start >= ? AND j.start < ? AND "
               "j.state != 'void'", (1050.0, 1100.0), "capture_jobs_start"),
              ("SELECT updated_at, game_key, last_error FROM capture_jobs WHERE last_error IS NOT NULL AND "
               "updated_at >= ? ORDER BY updated_at DESC LIMIT 10", (0.0,), "capture_jobs_errors")]
    with closing(sqlite3.connect(raw.path)) as db:
        for sql, args, index in shapes:
            plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + sql, args)]
            assert any(step.startswith("SEARCH") and index in step for step in plan), plan
            assert not any(step.startswith("SCAN") for step in plan), plan
    assert [e["game_key"] for e in raw.recent_errors(since=0)] == ["espn:7"]


# --- a full disk SQLite reports, though free blocks say there is room (F8) -------------------------------------------------

def test_sqlite_saying_the_disk_is_full_turns_room_off_for_five_minutes(tmp_path, monkeypatch, caplog) -> None:
    from collections import namedtuple
    from telegram_betting_bot_simple.capture import raw_store
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda folder: Usage(100e9, 10e9, 90e9))  # blocks: plenty
    clock = FakeClock()
    raw = RawStore(str(tmp_path / "raw.db"), clock=clock)
    put(raw, b'{"first": 1}')
    with closing(sqlite3.connect(raw.path)) as db:
        cap = db.execute("PRAGMA page_count").fetchone()[0] + 1
    real = RawStore._connect

    def capped(self):  # SQLite's own limit: a write past it fails as "database or disk is full" (SQLITE_FULL)
        db = real(self)
        db.execute(f"PRAGMA max_page_count = {cap}")
        return db

    monkeypatch.setattr(RawStore, "_connect", capped)
    assert raw.room()
    caplog.set_level("WARNING", logger="telegram_betting_bot_simple.capture.raw_store")
    for n in range(3):
        with pytest.raises(sqlite3.OperationalError, match="full"):
            put(raw, os.urandom(200_000), game_key=f"espn:{n}")  # incompressible: needs many pages
        assert not raw.room()
    full = [r for r in caplog.records if "database or disk is full" in r.getMessage()]
    assert len(full) == 1  # logged once
    # the health block's read (its own connection, same process): the pause and when it ends
    ends = raw_store.full_pause(raw.path, now=clock.now)
    assert ends is not None and clock.now < ends <= clock.now + raw_store.FULL_PAUSE
    assert raw_store.full_pause(str(tmp_path / "other.db"), now=clock.now) is None
    assert raw_store.full_pause(str(tmp_path / "." / "raw.db"), now=clock.now) == ends  # by its absolute path
    clock.now += raw_store.FULL_PAUSE - 1
    assert not raw.room() and raw_store.full_pause(raw.path, now=clock.now) == ends
    clock.now += 2
    assert raw.room()  # five minutes on: the disk is read again (and its free blocks say there is room)
    assert raw_store.full_pause(raw.path, now=clock.now) is None
    assert raw_store.full_pause(None, now=clock.now) is None  # never raises
    assert raw_store.disk_is_full(sqlite3.OperationalError("database or disk is full"))
    assert not raw_store.disk_is_full(sqlite3.OperationalError("database is locked"))


# --- one deadline per GET (F9) ------------------------------------------------------------------------------------------------

class Trickle(FakeAnswer):
    """An answer that comes a byte at a time, each byte taking ``every`` seconds on the fake clock, read as urllib's
    HTTPResponse reads: ``read1`` returns what one socket read got (a byte), ``read(n)`` keeps reading until it has n
    bytes or the end (a buffered read: however slowly they come, no socket timeout fires)."""

    def __init__(self, clock: FakeClock, every: float, size: int = 200) -> None:
        super().__init__(200, b"x" * size, {})
        self.clock, self.every = clock, every

    def read1(self, n: int = -1) -> bytes:  # type: ignore[override]
        self.clock.now += self.every
        return super().read(1)

    def read(self, n: int = -1) -> bytes:  # type: ignore[override]
        got = b""
        while n < 0 or len(got) < n:
            byte = self.read1(1)
            if not byte:
                break
            got += byte
        return got


def test_a_host_that_trickles_its_answer_past_the_deadline_is_unreachable() -> None:
    clock = FakeClock()
    http = PoliteHttp(clock=clock, sleep=clock.sleep)
    assert http.deadline == 60.0 == 3 * http.timeout
    seen: List[Optional[int]] = []
    http.on_result = lambda url, status: seen.append(status)
    http._opener = FakeOpener(Trickle(clock, every=5.0))  # 200 bytes at 5 s each: 1000 s, every read in time
    before = clock.now
    with pytest.raises(Unreachable, match="deadline"):
        http.get("https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter/1")
    assert 60.0 < clock.now - before <= 65.0  # stopped at the deadline, not after 1000 s
    assert seen == [None] and http.backing_off("site.web.api.espn.com")  # a failure for the back-off
    # the same trickle inside the deadline is fine
    clock.now += 3600
    http._opener = FakeOpener(Trickle(clock, every=0.1, size=50))
    assert http.get("https://site.web.api.espn.com/x").body == b"x" * 50
    # an error's body (urllib's HTTPError) is held to the same deadline: Unreachable, not a 503 read for 1000 s
    http._opener = FakeOpener(urllib.error.HTTPError("https://site.web.api.espn.com/y", 503, "Service Unavailable",
                                                     email.message.Message(), Trickle(clock, every=5.0)))
    before = clock.now
    with pytest.raises(Unreachable, match="deadline"):
        http.get("https://site.web.api.espn.com/y")
    assert 60.0 < clock.now - before <= 65.0 and seen[-1] is None
    assert PoliteHttp(timeout=5.0).deadline == 15.0 and PoliteHttp(deadline=9.0).deadline == 9.0


def test_the_active_guard_is_the_blocks_own() -> None:
    http = PoliteHttp(clock=FakeClock(), sleep=lambda s: None)
    mine = lambda url: None  # noqa: E731
    assert http.active_guard() is None
    with http.guard(mine):
        assert http.active_guard() is mine
    assert http.active_guard() is None


# --- one bucket per host for the whole process (FIX2 Major 1); the guard after the token's wait (FIX2 Minor 2) ---------

SITE_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def within_bucket(times: Sequence[float], rate: float, burst: int) -> None:
    """Every stretch of the requests (any two and those sent between them) keeps to a ``rate``/``burst`` token
    bucket."""
    times = sorted(times)
    for i in range(len(times)):
        for j in range(i, len(times)):
            assert j - i + 1 <= burst + rate * (times[j] - times[i]) + 1e-6, (i, j, times[j] - times[i])


def test_acquire_takes_a_token_as_get_does_and_spends_counts_and_settles_nothing() -> None:
    clock = FakeClock()
    http, sent = polite(clock)
    results: List[Optional[int]] = []
    http.on_result = lambda url, status: results.append(status)
    asked: List[str] = []
    with http.guard(asked.append):  # acquire asks no guard
        for _ in range(4):
            http.acquire(SITE_SUMMARY + "?event=1")
    assert clock.slept == [1.0]  # the burst of 3, then a second for the 4th token
    assert sent.sent == [] and asked == [] and results == [] and http.requests == {}
    http.get(SITE_SUMMARY)  # the bucket's own tokens were taken: the next one is a second away
    assert clock.slept == [1.0, 1.0] and len(sent.sent) == 1 and results == [200]
    # a host backing off for longer than MAX_WAIT: Backoff, and no token taken
    http._settle("statsapi.mlb.com", True, retry_after=MAX_WAIT + 30)
    with pytest.raises(Backoff):
        http.acquire("https://statsapi.mlb.com/api/v1/schedule")
    assert http._buckets["statsapi.mlb.com"].tokens == 1.0
    with pytest.raises(ValueError):
        http.acquire("not a url")


def test_the_processs_door_paces_the_learn_backfill_with_forward_capture_and_set_shared_none_clears_it() -> None:
    """Major 1: with the capture's PoliteHttp as the process's door, HistHttp takes a token from its bucket just before
    each request. Two forward requests and one of the learn backfill's, over and over on one host, keep to that host's
    bucket (1 a second, a burst of 3) over every stretch. With no door, or once set_shared(None) cleared it, HistHttp's
    requests and waits are its own pace's alone, exactly as before."""
    from telegram_betting_bot_simple.learn.sources.http import TIMEOUT, HistHttp
    clock = FakeClock()
    http, sent = polite(clock)
    learned: List[Tuple[float, str, Dict[str, Any], float]] = []

    def transport(url: str, params: Mapping[str, Any], timeout: float) -> Tuple[int, bytes]:
        learned.append((clock(), url, dict(params), timeout))
        return 200, b"{}"

    hist = HistHttp(pace=0.5, clock=clock, sleep=clock.sleep, transport=transport)
    for n in range(4):  # no door
        hist.json(SITE_SUMMARY, {"event": n})
    first = learned[0][0]
    assert [(at - first, url, params, timeout) for at, url, params, timeout in learned] == [
        (0.5 * n, SITE_SUMMARY, {"event": n}, TIMEOUT) for n in range(4)]
    assert clock.slept == [0.5, 0.5, 0.5] and sent.sent == []
    capture_http.set_shared(http)
    assert capture_http.shared() is http
    clock.now += 60.0  # a full bucket
    for n in range(15):
        http.get(SITE_SUMMARY, {"event": f"f{n}a"})
        http.get(SITE_SUMMARY, {"event": f"f{n}b"})
        hist.json(SITE_SUMMARY, {"event": f"l{n}"})
    assert len(sent.sent) == 30 and len(learned) == 19 and http.requests == {"site.api.espn.com": 30}
    within_bucket([at for at, _url, _headers in sent.sent] + [at for at, *_rest in learned[4:]], rate=1.0, burst=3)
    capture_http.set_shared(None)
    assert capture_http.shared() is None
    slept = len(clock.slept)
    for n in range(3):  # the door's bucket is empty now: only the backfill's own pace is waited for
        hist.json(SITE_SUMMARY, {"event": f"x{n}"})
    assert clock.slept[slept:] == [0.5, 0.5, 0.5]
    capture_http.shared_acquire(SITE_SUMMARY)  # no door: nothing, and no error
    assert clock.slept[slept:] == [0.5, 0.5, 0.5]


def test_shared_acquire_never_raises() -> None:
    clock = FakeClock()
    http, _sent = polite(clock)
    capture_http.set_shared(http)
    http._settle("site.api.espn.com", True, retry_after=MAX_WAIT + 30)  # the capture's host backs off for long
    capture_http.shared_acquire(SITE_SUMMARY)  # no token to give: the other door's request goes on
    capture_http.shared_acquire("not a url")
    assert clock.slept == []


def test_the_guard_is_asked_once_after_the_token_wait_just_before_the_request_goes_out() -> None:
    """Minor 2 (the final check's c16): a backfill request whose guard passed before its wait for a token went out
    even when a forward job fell due during that wait. The guard is now asked once, after the wait: the request is
    refused, nothing is sent, and the refusal takes no token (the forward request after it goes at once)."""
    clock = FakeClock()
    http, sent = polite(clock)
    due: List[float] = []
    asked: List[float] = []

    def sleep(seconds: float) -> None:  # a forward job falls due while a request waits for its token
        clock.sleep(seconds)
        due.append(clock.now)

    def backfill_guard(url: str) -> None:
        asked.append(clock.now)
        if due:
            raise Refused("forward")

    http.sleep = sleep  # type: ignore[method-assign]
    start = clock.now
    with http.guard(backfill_guard):
        for _ in range(3):  # the burst: no wait
            http.get(SITE_SUMMARY)
        with pytest.raises(Refused, match="forward"):
            http.get(SITE_SUMMARY)  # waits a second for its token; the job fell due meanwhile
    assert len(sent.sent) == 3 and asked == [start, start, start, start + 1.0]
    http.get(SITE_SUMMARY)  # the forward request: the token waited for is still there
    assert len(sent.sent) == 4 and sent.sent[-1][0] == start + 1.0 and due == [start + 1.0]


# --- RAW_DB_PATH never puts raw blobs where they don't belong (M3) -------------------------------------------------------

def test_raw_db_path_naming_betting_db_or_another_database_falls_back_to_the_default(tmp_path, monkeypatch,
                                                                                    caplog) -> None:
    from types import SimpleNamespace
    from telegram_betting_bot_simple.capture import raw_store
    from telegram_betting_bot_simple.data.store import Store
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    monkeypatch.setattr(raw_store, "_PATH_WARNED", set())
    state = tmp_path / "state"
    state.mkdir()
    betting = Store(str(state / "betting.db"))
    default = str(state / "raw.db")
    (state / "notes.db").write_text("not a database")
    import shutil
    shutil.copy(betting.path, state / "copy.db")
    RawStore(str(state / "mine.db"))
    caplog.set_level("WARNING", logger="telegram_betting_bot_simple.capture.raw_store")
    cases = {str(state / "betting.db"): default, str(state / "copy.db"): default, str(state / "notes.db"): default,
             str(state): default, str(state / "mine.db"): str(state / "mine.db"),
             str(tmp_path / "new" / "raw.db"): str(tmp_path / "new" / "raw.db"), str(state / "empty.db"): None}
    (state / "empty.db").write_bytes(b"")
    cases[str(state / "empty.db")] = str(state / "empty.db")
    for given, expected in cases.items():
        assert resolve_raw_db_path(SimpleNamespace(raw_db_path=given), betting) == expected, given
    warned = [r.getMessage() for r in caplog.records if "RAW_DB_PATH is ignored" in r.getMessage()]
    assert len(warned) == 4 and all(default in text for text in warned)
    assert "it is betting.db" in warned[0] and "another database" in warned[1] and "SQLite" in warned[2]
    resolve_raw_db_path(SimpleNamespace(raw_db_path=str(state / "betting.db")), betting)
    assert len([r for r in caplog.records if "RAW_DB_PATH is ignored" in r.getMessage()]) == 4  # warned once each
    monkeypatch.setenv("RAW_DB_PATH", str(state / "betting.db"))  # the environment's too
    assert resolve_raw_db_path(None, betting) == default
