"""The learning loop end to end over the real modules (the spec's "Done when"): the loop on a fake clock with no
network, from the backfill (the captured fixtures) and games logged through the slate's own path and graded by the
real grader, through nightly scores, a Monday gate that promotes a planted improvement and holds on noise, a nightly
rollback of a planted regression, the AI's notes and proposals through a fake Claude (batches), a fake DeepSeek and
none, Pause and the owner's Undo through the HTTP routes, to the progress page, the bot's /brain and the Ask AI tool
reading the same numbers from the store. Also: byte-identical outputs through the wiring with an empty registry (the
code before the learning loop's, pinned), and the integration fixes (cool-off after an Undo or a rollback, the
tripwire judged against the defaults' blend, clearing safe mode, the backfill's count of games).

Only the outside world is faked (tests/learn_world.py): HTTP for the backfill, ESPN for the grader, the AI clients."""

from __future__ import annotations

import datetime
import json
import sqlite3
import types
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import brain_store as bs
from telegram_betting_bot_simple.learn import evaluate as ev
from telegram_betting_bot_simple.learn import loop, registry, tune
from telegram_betting_bot_simple.learn.history import Backfill, HistoryReader
from tests import learn_sim as sim
from tests import learn_world as world
from tests.learn_wiring_cases import wiring_outputs
from tests.test_auth import BOT_TOKEN, make_init_data

OWNER, FRIEND = 42, 7
DAY = world.DAY
GOLDEN = Path(__file__).parent / "fixtures" / "learn_eval" / "wiring_golden.json"
ALLOWED_HOSTS = {"www.football-data.co.uk", "football-data.co.uk", "raw.githubusercontent.com",
                 "sports.core.api.espn.com", "site.api.espn.com", "statsapi.mlb.com"}


def client_for(app: Any, uid: int = None, password: str = None) -> TestClient:
    client = TestClient(app)
    if uid is not None:
        init = make_init_data({"id": uid, "first_name": "Sam"}, auth_date=int(datetime.datetime.now().timestamp()))
        assert client.post("/api/tg-login", json={"init_data": init}).status_code == 200
    if password is not None:
        assert client.post("/api/login", json={"password": password}).status_code == 200
    return client


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Any:
    """The whole timeline, once: each test below reads what it left (``run.seen``, the store, the routes)."""
    store = Store(str(tmp_path_factory.mktemp("learn") / "betting.db"))
    registry.bind(store)
    t = world.LearningTimeline(store)
    app = create_app(t.service, "s3cret", login_delay=0, bot_token=BOT_TOKEN, telegram_ids=[OWNER, FRIEND],
                     learner=t.learner, owner_id=OWNER)
    t.app = app
    t.owner = client_for(app, OWNER)
    t.friend = client_for(app, FRIEND)
    t.routes: List[Dict[str, Any]] = []

    def undo(version: int) -> Dict[str, Any]:
        refused = t.friend.post("/api/brain/undo", json={"version": version})
        response = t.owner.post("/api/brain/undo", json={"version": version})
        t.routes.append({"route": "undo", "friend": refused.status_code, "owner": response.status_code})
        return response.json()

    def pause(paused: bool) -> Dict[str, Any]:
        refused = t.friend.post("/api/brain/pause", json={"paused": paused})
        response = t.owner.post("/api/brain/pause", json={"paused": paused})
        t.routes.append({"route": "pause", "paused": paused, "friend": refused.status_code,
                         "owner": response.status_code})
        return response.json()

    t.undo_with, t.pause_with = undo, pause
    t.world()
    t.after_world = {"graded_before_tick": len(store.consensus_rows(status="graded"))}
    t.first_week()
    with closing(store._connect()) as db:
        t.after_first = {"champions": {k: v["version"] for k, v in bs.champions(db).items()},
                         "trials": bs.trials(db, limit=None), "note": bs.get_state(db, "note:latest"),
                         "jobs": list(t.learner._jobs())}
    t.collect()
    with closing(store._connect()) as db:
        t.after_collect = {k: bs.get_state(db, k) for k in ("note:latest", "proposals:latest", "changelog",
                                                            "memo:latest", "ai_last")}
        t.after_collect["challengers"] = bs.versions(db, status="challenger")
    t.regression(days=21)
    t.pause()
    t.resume()
    t.undo()
    t.now = t.clock.t
    yield t
    registry.bind(None)


# --- 1. the world: logged through the slate's path, graded by the grader, captured into the history ---------------

def test_games_are_logged_with_the_live_version_and_graded_by_the_real_grader(run) -> None:
    assert run.after_world["graded_before_tick"] == 0  # nothing is graded until the learner's night runs the grader
    first_night = run.seen["first"]["ran"][0]
    assert first_night["kind"] == "night" and first_night["steps"]["grade"]["result"]["graded"] == sum(
        run.seen["world"].values())
    with closing(run.store._connect()) as db:
        live = db.execute("SELECT COUNT(*) FROM hist_games WHERE source = 'live'").fetchone()[0]
        versions = dict(db.execute("SELECT params_version, COUNT(*) FROM consensus_log WHERE league = ? "
                                   "GROUP BY params_version", (world.NBA,)).fetchall())
    assert live >= sum(run.seen["world"].values())  # every graded game kept in the history's shape
    nba = run.after_first["champions"]["nba"]
    assert versions.get(0, 0) >= run.seen["world"]["nba"] and versions.get(nba, 0) > 0  # logged under the champion
    assert run.seen["versions_logged"].get(str(nba), 0) == versions[nba]


def test_the_backfill_reads_only_the_captures_inside_its_window_and_budget(run) -> None:
    report = run.seen["first"]["ran"][0]["steps"]["backfill"]["result"]
    assert report["ran"] is True and report["requests"] > 0 and report["games"] > 0
    assert report["requests"] <= run.per_night and report["stopped"] in ("done", "budget")
    assert set(run.transport.hosts()) <= ALLOWED_HOSTS
    with closing(run.store._connect()) as db:
        sources = dict(db.execute("SELECT source, COUNT(*) FROM hist_games GROUP BY source").fetchall())
    assert {"football-data", "nflverse", "espn"} <= set(sources)
    # Every night's requests were made between 01:00 and 06:00 (the fake clock moved 2 s a request)
    for tick in [run.seen["first"]] + run.seen["regression"]["ticks"]:
        for r in tick["ran"]:
            if r["kind"] == "night":
                result = r["steps"]["backfill"]["result"]
                assert result.get("stopped") != "outside the 01:00-06:00 window"
    for league in ("soccer_epl", "soccer_usa_mls", "americanfootball_nfl", "basketball_nba", "baseball_mlb"):
        for record in HistoryReader(run.store).games(league):
            assert all(line["as_of"] < record["start_ts"] for line in record["lines"]), record["game_key"]


# --- 2. the first Monday: a planted improvement promoted, noise held -------------------------------------------

def test_the_monday_gate_promotes_the_planted_improvement_and_holds_on_noise(run) -> None:
    gate = run.seen["first"]["ran"][1]["steps"]["gate"]["result"]
    assert run.seen["first"]["ran"][1]["kind"] == "week" and gate["week"] == "2026-W36"
    assert {p["sport"] for p in gate["promoted"]} == {"nba", "mlb"}
    trials = run.after_first["trials"]
    nhl = [t for t in trials if t["sport"] == "nhl"]
    assert nhl and all(t["decision"] in ("held", "rejected") for t in nhl)  # noise: no change
    for t in trials:
        if t["decision"] == "promoted":
            assert t["eval_n"] >= tune.MIN_EVAL and t["hi"] <= tune.NONINF_MARGIN and t["source"] == "refit"
            assert "within the allowed margin" in t["reason"] and "-0.0000 " not in t["reason"]
    promoted = next(t for t in trials if t["decision"] == "promoted" and t["sport"] == "nba")
    for knob, new in promoted["new"].items():  # one cycle's step at most
        assert abs(new - promoted["old"][knob]) <= registry.KNOBS[knob].step + 1e-9


def test_the_owner_hears_the_week_in_three_lines(run) -> None:
    summary = run.seen["first"]["ran"][1]["steps"]["telegram"]["result"]
    lines = summary["text"].split("\n")
    assert summary["sent"] is True and len(lines) == 3
    promoted = len(run.seen["first"]["ran"][1]["steps"]["gate"]["result"]["promoted"])
    assert lines[0].startswith("Brain, week 2026-W36: ") and f"{promoted} promoted, 0 rolled back" in lines[0]
    assert lines[2] == "Details: https://bets.example.com/#/bets?tab=brain"
    assert run.sent[0] == summary["text"]


# --- 3. the AI: Claude in batches, judged forward only ----------------------------------------------------------

def test_claude_answers_in_batches_and_every_text_is_checked(run) -> None:
    assert {j["kind"] for j in run.after_first["jobs"]} == {"note", "proposals", "changelog", "memo"}
    assert run.after_first["note"]["source"] == "template"  # the template note shows until the batch is back
    collected = {c["kind"]: c["status"] for c in run.seen["collect"]["ai"]}
    assert collected == {"note": "done", "proposals": "done", "changelog": "done", "memo": "done"}
    note = run.after_collect["note:latest"]
    assert note["source"] == "ai:claude" and note["model"] == "claude-opus-5-5" and "log loss of" in note["text"]
    texts = run.after_collect["changelog"]
    assert texts and all(entry["source"] == "ai:claude" for entry in texts.values())
    assert run.after_collect["memo:latest"]["source"] == "ai:claude"
    batch = run.claude.client.messages.batches.created[0][0]["params"]
    assert batch["output_config"]["format"]["type"] == "json_schema" and "tool_choice" not in batch
    assert "fallbacks" not in batch  # refused on Batches
    calls, dollars = run.store.analyst_spend("brain:claude")
    assert calls >= 4 and 0 < dollars * 100 <= 500


def test_an_ai_proposal_is_frozen_and_judged_only_on_games_after_it(run) -> None:
    latest = run.after_collect["proposals:latest"]
    assert latest["provider"] == "claude" and len(latest["proposals"]) == 1
    outcome = latest["proposals"][0]["outcome"]
    assert "version" in outcome
    row = next(v for v in run.after_collect["challengers"] if v["version"] == outcome["version"])
    assert row["source"] == "ai:claude" and row["effective_from"] == latest["at"]
    with closing(run.store._connect()) as db:
        shadow = bs.get_state(db, f"shadow:{outcome['version']}")
        starts = [r["start"] for r in ev.consensus_records(db) if r["sport"] == "nba"
                  and r["start"] > row["effective_from"]]
    assert shadow and shadow["n"] <= len(starts) and shadow["needed"] == tune.MIN_EVAL
    assert ev.leakage_problems(proposals=[{"id": "p", "created_at": row["effective_from"], "judged_on": starts}]) == []


# --- 4. a regression is rolled back; DeepSeek, then templates --------------------------------------------------

def test_a_planted_regression_is_rolled_back_by_the_nightly_watch(run) -> None:
    out = run.seen["regression"]
    assert out["rolled_back_on"] is not None
    nba = run.after_first["champions"]["nba"]
    with closing(run.store._connect()) as db:
        assert bs.get_version(db, nba)["status"] == "rolled_back"
        trial = next(t for t in bs.trials(db, limit=None) if t["decision"] == "rolled_back" and t["sport"] == "nba")
        after = db.execute("SELECT DISTINCT params_version FROM consensus_log WHERE league = ? AND logged_at > ?",
                           (world.NBA, trial["created_at"])).fetchall()
    assert trial["version"] == nba and ("change detector" in trial["reason"] or "Worse than" in trial["reason"])
    # The first Monday promoted two NBA changes one on top of the other: watched and rolled back together
    retired = trial["details"]["versions"]
    assert len(retired) == 2 and retired[0] == nba
    with closing(run.store._connect()) as db:
        assert all(bs.get_version(db, v)["status"] == "rolled_back" for v in retired)
        first_promoted = bs.get_version(db, nba)["effective_from"]  # the first Monday's run
        nba_rows = bs.trials(db, sport="nba", limit=None)
    # No new NBA change was stacked on the watched one before the watch could judge it: the Monday in between held
    # the candidates that would have gone live (waiting their turn, or frozen as the calibration drifted)
    waited = [t for t in nba_rows if (t["details"] or {}).get("would_have") == "promoted"
              and first_promoted < t["created_at"] < trial["created_at"]]
    assert waited and all(t["decision"] == "held" for t in waited)
    assert not [t for t in nba_rows if t["decision"] == "promoted" and first_promoted < t["created_at"]
                < trial["created_at"]]
    assert registry.current(world.NBA, run.store).version != nba
    assert after == [(0,)]  # games logged after the rollback are back on the parent (the defaults)
    assert any(text.startswith("Brain: NBA rolled back automatically") for text in run.sent)


def test_the_provider_follows_the_keys_at_run_time(run) -> None:
    deepseek_calls, cost = run.store.analyst_spend("brain:deepseek")
    assert deepseek_calls >= 2 and cost > 0
    body = run.deepseek.client.bodies[0]
    assert body["response_format"] == {"type": "json_object"}
    weeks = [r for tick in run.seen["regression"]["ticks"] for r in tick["ran"] if r["kind"] == "week"]
    assert weeks and weeks[0]["steps"]["proposals"]["result"]["ai"]["provider"] == "deepseek"
    nights = [r for tick in run.seen["regression"]["ticks"] for r in tick["ran"] if r["kind"] == "night"]
    assert nights[-1]["steps"]["note"]["result"]["ai"]["status"] == "skipped"  # no key: the template note
    assert run.nobody.status()["reason"] == "AI notes: off (no key)"


# --- 5. Pause, resume and the owner's Undo through the routes --------------------------------------------------

def test_pause_freezes_the_values_and_stops_fetches_and_ai_calls(run) -> None:
    paused = run.seen["pause"]
    assert paused["state"]["paused"] is True
    week = next(r for r in paused["report"]["ran"] if r["kind"] == "week")
    night = next(r for r in paused["report"]["ran"] if r["kind"] == "night")
    assert week["steps"]["gate"]["result"]["skipped"] == "paused" and not week["steps"]["gate"]["result"]["promoted"]
    assert week["steps"]["proposals"]["result"] == {"skipped": "paused"}
    assert night["steps"]["backfill"]["result"] == {"skipped": "paused"}
    assert paused["before"] == paused["after"]  # no request and no AI call while paused
    assert "learning is paused" in week["steps"]["telegram"]["result"]["text"]
    assert run.seen["resume"]["paused"] is False
    assert [r for r in run.routes if r["route"] == "pause"] == [
        {"route": "pause", "paused": True, "friend": 403, "owner": 200},
        {"route": "pause", "paused": False, "friend": 403, "owner": 200}]


def test_the_owners_undo_reverts_the_version_and_the_page_says_so(run) -> None:
    undo = run.seen["undo"]
    assert undo["result"] == {"undone": undo["version"], "now_live": undo["parent"]}
    assert [r for r in run.routes if r["route"] == "undo"] == [{"route": "undo", "friend": 403, "owner": 200}]
    assert registry.current(world.MLB, run.store).version == undo["parent"]
    page = run.owner.get("/api/brain/progress").json()
    card = next(c for c in page["changes"] if c["version"] == undo["version"] and c["decision"] == "promoted")
    assert card["rolled_back"]["decision"] == "undone" and card["undo"] is False
    undone = next(c for c in page["changes"] if c["version"] == undo["version"] and c["decision"] == "undone")
    assert "undone by the owner" in undone["text"]
    mlb = next(s for s in page["sports"] if s["sport"] == "mlb")
    assert ("undone", undo["version"]) in {(m["kind"], m["version"]) for m in mlb["markers"]}
    assert any(text.startswith("Undone by the owner: brain version") for text in run.sent)
    # Undoing it again is refused; a friend sees no controls
    assert run.owner.post("/api/brain/undo", json={"version": undo["version"]}).status_code == 409
    seen = run.friend.get("/api/brain/progress").json()
    assert seen["owner"] is False and not any(c["undo"] for c in seen["changes"])
    assert TestClient(run.app).get("/api/brain/progress").status_code == 401


# --- 6. the page, the bot and the Ask AI tool read the same numbers from the store ------------------------------

def test_the_progress_page_shows_the_stores_numbers(run) -> None:
    page = run.owner.get("/api/brain/progress").json()
    json.dumps(page)
    # The three sports with games logged, then soccer: scored on the history's matches only (goals, counts)
    assert [s["sport"] for s in page["sports"]] == ["nba", "nhl", "mlb", "soccer"]
    soccer = page["sports"][3]
    assert soccer["headline"] is None and {t["target"] for t in soccer["targets"]} >= {"goals", "btts"}
    assert all(t["status"] for t in soccer["targets"] if t["target"] in ("corners", "cards"))  # record only
    with closing(run.store._connect()) as db:
        records = ev.consensus_records(db, until=run.now)
        graded = dict(db.execute("SELECT league, COUNT(*) FROM consensus_log WHERE status = 'graded' AND kind = "
                                 "'last' GROUP BY league").fetchall())
        scores = {(r["sport"], r["week"]): r for r in bs.scores(db, target="result")}
    for block in page["sports"][:3]:
        rows = [r for r in records if r["sport"] == block["sport"]]
        fresh = loop.sport_headline(rows)
        h = block["headline"]
        assert h["n"] == fresh["n"] == 200 and h["word"] == fresh["word"]
        assert h["gap"] == pytest.approx(fresh["gap"], abs=1e-6) and h["text"].startswith(block["label"])
        assert len(block["series"]) >= 10  # the first Monday scored the weeks already graded (14 here)
        for point in block["series"]:  # every point is a stored weekly score
            stored = scores[(block["sport"], point["week"])]
            assert point["n"] == stored["n"] and point["gap"] == pytest.approx(stored["gap"], abs=1e-6)
        assert block["calibration"]["words"] and block["calibration"]["n"]
    nba = page["sports"][0]
    assert {m["kind"] for m in nba["markers"]} >= {"promoted", "rolled_back"}
    back = next(c for c in page["changes"] if c["decision"] == "rolled_back" and c["sport"] == "nba")
    assert "Blend weight of sharp books 0.5" in back["text"] or back["text_source"] != "template"
    assert page["health"]["graded"] == {registry.sport_key(k): v for k, v in graded.items()}
    assert {"football-data", "nflverse", "espn"} <= set(page["health"]["backfill"])
    assert all(page["health"]["runs"][k]["last"] for k in ("night", "week", "month"))
    assert page["experiments"] and all(e["needed"] == tune.MIN_EVAL for e in page["experiments"])
    assert page["held"]["tested"] >= 1 and page["held"]["words"].startswith("This week:")
    assert page["ai"]["proposals"]["made"] >= 2 and page["ai"]["cost"]["spent_cents"] > 0
    assert page["learning"]["paused"] is False and page["safe_mode"] is None


def test_the_bot_and_the_ask_ai_tool_read_the_same_numbers(run) -> None:
    from telegram_betting_bot_simple.ai.tools import Toolbox
    from telegram_betting_bot_simple.bot.handlers import brain_text
    from telegram_betting_bot_simple.dashboard.lab import Lab
    from zoneinfo import ZoneInfo

    page = run.owner.get("/api/brain/progress").json()
    text = brain_text(run.service)
    for block in page["sports"]:
        if block["headline"]:
            assert block["headline"]["text"] in text
    assert "Last change (" in text
    run.service.learner = run.learner
    lab = Lab(run.service, run.store)
    tool = Toolbox(lab, f"tg:{OWNER}", ZoneInfo("America/New_York")).run("brain_progress", {})
    assert tool.ok and [s["sport"] for s in tool.content["sports"]] == ["NBA", "NHL", "MLB", "Soccer"]
    assert [(s["headline"] or {}).get("gap") for s in tool.content["sports"]] == [
        (s["headline"] or {}).get("gap") for s in page["sports"]]
    assert tool.content["source"].startswith("The learning loop")


# --- 7. byte-identical defaults through the wiring ---------------------------------------------------------------

def test_an_empty_registry_leaves_every_output_byte_identical(tmp_path) -> None:
    """slate, consensus, soccer, context and NBA outputs through the app's wiring, with the registry bound to a real
    Store holding no versions, equal the code before the learning loop's (b9e3111) byte for byte."""
    golden = json.loads(GOLDEN.read_text())
    store = Store(str(tmp_path / "empty.db"))
    registry.bind(store)
    try:
        (tmp_path / "world").mkdir()
        now = wiring_outputs(store, tmp_path / "world")
    finally:
        registry.bind(None)
    for key in ("slate", "soccer", "nba"):
        assert now[key] == golden[key], key
    for key, value in golden["world"].items():
        assert now["world"][key] == value, key


def test_a_live_version_does_change_those_outputs(tmp_path) -> None:
    """The check above is not empty: once a version is live, the NBA blends and the NBA model move."""
    golden = json.loads(GOLDEN.read_text())
    store = Store(str(tmp_path / "tuned.db"))
    values = registry.defaults("nba")
    values.update({"consensus.weight.sharp": 0.45, "consensus.weight.us_books": 0.25, "nba.market_weight": 0.87,
                   "nba.espn_weight": 0.08})
    with closing(store._connect()) as db:
        version = bs.insert_version(db, "nba", values, status="challenger", now=1.0, knob="test", gate="nudge")
        bs.promote(db, version, 2.0)
    registry.bind(store)
    try:
        (tmp_path / "world").mkdir()
        now = wiring_outputs(store, tmp_path / "world")
    finally:
        registry.bind(None)
    assert now["slate"][:2] != golden["slate"][:2]  # the NBA cases come first
    assert now["slate"][3:] == golden["slate"][3:]  # other sports untouched
    assert now["nba"] != golden["nba"]


# --- 8. the integration fixes ------------------------------------------------------------------------------------

def test_an_undone_setting_waits_before_it_is_moved_again() -> None:
    db = sqlite3.connect(":memory:", isolation_level=None)
    bs.init_brain_schema(db)
    rows = sim.consensus_world(4000, seed=11, noise=world.PLANTED)
    now = rows[-1]["start"] + 3 * DAY
    first = tune.run_week(db, now=now, corpus=ev.Corpus(consensus=rows), families=["weights"])
    version = first["promoted"][0]["version"]
    assert tune.undo(db, version, now=now + 3600) == {"undone": version, "now_live": 0}
    again = tune.run_week(db, now=now + 7 * DAY, corpus=ev.Corpus(consensus=rows), families=["weights"])
    assert again["promoted"] == []
    held = next(t for t in bs.trials(db, limit=None) if t["details"].get("would_have") == "promoted")
    assert held["decision"] == "held" and held["reason"].startswith("Not moved again yet: the owner took it back")
    later = tune.run_week(db, now=now + (tune.COOL_OFF_WEEKS * 7 + 1) * DAY, corpus=ev.Corpus(consensus=rows),
                          families=["weights"])
    assert [p["sport"] for p in later["promoted"]] == ["nba"]  # after the wait it may move again


def test_the_tripwire_judges_the_live_blend_against_the_defaults_blend(tmp_path) -> None:
    store = Store(str(tmp_path / "betting.db"))
    now = 2_000_000_000.0
    # Sources that disagree a lot (the data): the defaults' blend is far from the sharp price too
    rows = sim.consensus_world(60, seed=5, start=now - DAY / 2, per_day=200, noise={"sharp": 0.9})
    for r in rows:
        r["logged_at"] = now - 3600
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
        version = bs.insert_version(db, "nba", {**registry.defaults("nba"), "consensus.near_start_factor": 0.6},
                                    status="challenger", now=now - 10)
        bs.promote(db, version, now - 5)
    quiet = tune.check_tripwire(store, now=now)
    assert not quiet["fired"] and not quiet["safe_mode"]
    without = tune.tripwire(ev.consensus_records(sqlite3.connect(store.path), status=None))
    assert without["fired"]  # the old rule would have switched every sport to the defaults
    # A broken live blend still trips it
    with closing(store._connect()) as db:
        db.execute("UPDATE consensus_log SET blend = ?", (json.dumps({"p": {"home": 0.99, "away": 0.01}}),))
    tripped = tune.check_tripwire(store, now=now)
    assert tripped["fired"] and tripped["safe_mode"]


def test_the_owner_clears_safe_mode_through_the_route(tmp_path) -> None:
    store = Store(str(tmp_path / "betting.db"))
    sent: List[str] = []
    learn = loop.Learner(store, clock=lambda: 1_900_000_000.0, notify=lambda t: sent.append(t) or True,
                         backfill=lambda now: {})
    with closing(store._connect()) as db:
        bs.set_safe_mode(db, True, "Tripwire: a test.", 1_899_999_000.0, {})
    app = create_app(types.SimpleNamespace(store=store, lab=object(), slate=None), "s3cret", login_delay=0,
                     bot_token=BOT_TOKEN, telegram_ids=[OWNER, FRIEND], learner=learn, owner_id=OWNER)
    assert TestClient(app).post("/api/brain/clear-safe-mode").status_code == 401
    assert client_for(app, FRIEND).post("/api/brain/clear-safe-mode").status_code == 403
    owner = client_for(app, password="s3cret")
    assert owner.get("/api/brain/progress").json()["safe_mode"]["reason"] == "Tripwire: a test."
    assert owner.post("/api/brain/clear-safe-mode").json() == {"cleared": True}
    assert owner.get("/api/brain/progress").json()["safe_mode"] is None
    assert owner.post("/api/brain/clear-safe-mode").json() == {"cleared": False}
    assert sent == ["Safe mode cleared by the owner: the brain's live settings serve again."]
    with closing(store._connect()) as db:
        assert bs.trials(db, limit=1)[0]["decision"] == "cleared"


def test_a_backfill_stopped_by_its_budget_still_counts_the_games_it_wrote(tmp_path) -> None:
    store = Store(str(tmp_path / "betting.db"))
    clock = world.Clock(world.at(2026, 10, 28, 2))
    report = Backfill(store, TTLCache(), clock, http=world.hist_http(clock, world.HistTransport()), tz=world.UTC,
                      per_night=12).run_night()
    with closing(store._connect()) as db:
        written = db.execute("SELECT COUNT(*) FROM hist_games").fetchone()[0]
    assert report["stopped"] == "budget" and report["games"] == written > 0


def test_change_texts_name_each_value_with_its_own_setting() -> None:
    text = loop.template_change({"sport": "nba", "decision": "rolled_back", "knob": "x",
                                 "old": {"consensus.weight.us_books": 0.218, "consensus.weight.sharp": 0.484},
                                 "new": {"consensus.weight.us_books": 0.2, "consensus.weight.sharp": 0.5},
                                 "reason": "A steady drift."})
    assert text == ("NBA: back to the version before it (Blend weight of sharp books 0.5; Blend weight of US books "
                    "0.2). A steady drift.")
    assert tune.stats.signed(-0.000037) == "-0.000037" and tune.stats.signed(0.0004) == "+0.0004"


def test_the_owner_notifier_sends_through_the_bot_and_never_raises() -> None:
    from telegram.error import NetworkError

    from telegram_betting_bot_simple.bot.notify import MAX_MESSAGE, owner_notifier

    sent: List[Dict[str, Any]] = []

    class Bot:
        async def send_message(self, **kwargs: Any) -> None:
            sent.append(kwargs)

    class Down:
        async def send_message(self, **kwargs: Any) -> None:
            raise NetworkError("unreachable")

    assert owner_notifier(None, "42") is None and owner_notifier("123:abc", None) is None
    assert owner_notifier("123:abc", "42", bot=Bot())("x" * 5000) is True
    assert sent[0]["chat_id"] == "42" and len(sent[0]["text"]) == MAX_MESSAGE
    assert owner_notifier("123:abc", "42", bot=Down())("hello") is False
    assert loop.telegram_notifier("123:abc", "42") is not None  # the learner's sender is this one


def test_the_dashboard_builds_the_learner_from_the_settings(tmp_path) -> None:
    """dashboard/__main__.py's learner: LEARN_* settings, the owner's chat for the summary, the store and cache."""
    from telegram_betting_bot_simple.config import Config
    from telegram_betting_bot_simple.dashboard.__main__ import build_learner

    config = Config(TELEGRAM_BOT_TOKEN="123456:test", LEARN_ENABLED="0", LEARN_AI="off", LEARN_BACKFILL_PER_NIGHT="300",
                    LEARN_AI_MONTHLY_CENTS="250", TELEGRAM_CHANNEL_ID="42", PUBLIC_URL="https://bets.example.com",
                    _env_file=None)
    service = types.SimpleNamespace(store=Store(str(tmp_path / "betting.db")), cache=TTLCache())
    learner = build_learner(config, service)
    assert learner.store is service.store and learner.cache is service.cache and learner.service is service
    assert learner.enabled is False and learner.paused()["by"] == "LEARN_ENABLED" and learner.per_night == 300
    assert learner.analyst.choice().provider is None and learner.analyst.monthly_cents == 250
    assert learner.notify is not None and learner.page_url == "https://bets.example.com/#/bets?tab=brain"
    assert config.owner_telegram_id == 42
