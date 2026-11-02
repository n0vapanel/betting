"""A learning world over the REAL modules, for the integration test (tests/test_learn_integration.py) and the e2e
launcher: no network, no fake app code. Only the outside world is faked:

- ``HistTransport``: the history backfill's HTTP door (learn/sources/http.py ``HistHttp(transport=...)``) answering
  from the trimmed captures under tests/fixtures/learn (404 for anything not captured);
- ``FakeEspn``: ESPN as the grader sees it (each finished game's score, the captured EPL summary for soccer);
- ``SmartClaude`` / ``SmartDeepSeek``: Anthropic's and DeepSeek's clients, wrapped by the real ai/claude.py
  ClaudeAnalyst and ai/deepseek.py DeepSeekAnalyst inside the real learn/analyst.py BrainAnalyst. They answer every
  kind from the fact sheet they were sent (numbers copied from it, evidence ids from it), so the real checks pass;
  Claude also has the Message Batches API.

Games are logged the way the slate logs them (data/slate.py ``brain_params`` -> ``market_detail`` ->
``SlateBuilder._consensus_row`` -> ``Store.log_consensus`` at the time of the last refresh before the start) and
graded by the real DashboardService grader (``grade_due``), which also captures each game into the history tables.
"""

from __future__ import annotations

import datetime
import json
import re
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from telegram_betting_bot_simple.data import slate as slate_mod
from telegram_betting_bot_simple.engine import consensus

FIXTURES = Path(__file__).parent / "fixtures" / "learn"
UTC = datetime.timezone.utc
_dumps = json.dumps
EMPTY_LIST = json.dumps({"count": 0, "pageIndex": 1, "pageCount": 1, "items": []}).encode()
FD_FILES = {"mmz4281/2526/E0.csv": "fd_E0_2526.csv", "mmz4281/1920/D1.csv": "fd_D1_1920.csv",
            "mmz4281/2627/E0.csv": "fd_E0_2627.csv", "/new/USA.csv": "fd_new_USA.csv",
            "nfldata/master/data/games.csv": "nfl_games.csv"}


class Clock:
    """A settable clock (epoch seconds); ``sleep`` moves it on (the backfill's 2-second pace)."""

    def __init__(self, t: float) -> None:
        self.t = float(t)
        self.slept = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += seconds


class HistTransport:
    """The backfill's requests, answered from the captures (``transport(url, params, timeout) -> (status, bytes)``).
    Every request is recorded with its host."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        self.calls.append((url, dict(params or {})))
        for pattern, name in FD_FILES.items():
            if url.endswith(pattern):
                return 200, (FIXTURES / name).read_bytes()
        if "football-data.co.uk" in url:
            return 404, b"Not Found"
        if "sports.core.api.espn.com" in url and url.endswith("/events"):
            return 200, (FIXTURES / "espn_nba_events.json").read_bytes() if "/nba/" in url else EMPTY_LIST
        if url.endswith("/odds"):
            return 200, (FIXTURES / ("espn_mlb_odds.json" if "/mlb/" in url else "espn_nba_odds.json")).read_bytes()
        if url.endswith("/summary"):
            return 200, (FIXTURES / "espn_nba_summary.json").read_bytes()
        if url.endswith("/scoreboard"):
            if "/mlb/" in url and (params or {}).get("dates") == "20250703":
                return 200, (FIXTURES / "espn_mlb_scoreboard_20250703.json").read_bytes()
            return 200, b'{"events": []}'
        if "statsapi.mlb.com" in url:
            season = str((params or {}).get("season"))
            return 200, (FIXTURES / "mlb_season_2025.json").read_bytes() if season == "2025" else b'{"dates": []}'
        return 404, b"{}"

    def hosts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for url, _ in self.calls:
            host = re.sub(r"^https?://([^/]+)/.*$", r"\1", url)
            out[host] = out.get(host, 0) + 1
        return out


def hist_http(clock: Clock, transport: HistTransport) -> Any:
    """The real HistHttp (pace, retries, allow/on_request hooks) over the captures, on the fake clock."""
    from telegram_betting_bot_simple.learn.sources.http import HistHttp
    return HistHttp(clock=clock, sleep=clock.sleep, transport=transport)


class FakeEspn:
    """ESPN for the grader: ``game_result`` of the games this world finished, the captured EPL summary for soccer
    games (corners, cards, shots), no summary otherwise."""

    def __init__(self) -> None:
        self.results: Dict[str, Tuple[int, int]] = {}
        self.summaries = 0
        self._epl = json.loads((FIXTURES / "espn_epl_summary.json").read_text())

    def finish(self, espn_id: str, home: int, away: int) -> None:
        self.results[str(espn_id)] = (int(home), int(away))

    def game_result(self, league: str, event_id: str) -> Dict[str, Any]:
        found = self.results.get(str(event_id))
        if found is None:
            return {"state": "pending"}
        return {"state": "final", "home": found[0], "away": found[1]}

    def summary(self, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        self.summaries += 1
        return self._epl if str(league).startswith("soccer") else None

    def find_game(self, *args: Any, **kwargs: Any) -> None:
        return None

    def box_score(self, *args: Any, **kwargs: Any) -> None:
        return None

    def closing_line(self, *args: Any, **kwargs: Any) -> None:
        return None


def log_games(store: Any, records: Iterable[Mapping[str, Any]], espn: FakeEspn) -> Dict[str, int]:
    """Log each simulated game as the slate does at its last refresh before the start (the live parameter
    version's blend, ``params_version`` kept), and tell ESPN its final score. ``records`` are learn_sim
    ``consensus_world`` rows (their sources, start, logged_at and outcome; their own blend is not used).
    Returns {version: games logged with it}."""
    versions: Dict[str, int] = {}
    for r in records:
        league = r["league"]
        params = slate_mod.brain_params(league, store)
        detail = slate_mod.market_detail(league, r["sources"], r["start"], r["logged_at"], params=params)
        espn_id = str(r["game_key"]).split(":", 1)[1]
        game = {"league": league, "odds_id": f"o-{espn_id}", "espn_id": espn_id, "home": {"name": f"Home {espn_id}"},
                "away": {"name": f"Away {espn_id}"}, "start": consensus._iso(r["start"])}
        row = slate_mod.SlateBuilder._consensus_row(game, detail, params.version)
        store.log_consensus([row], now=r["logged_at"])
        home_won = r["outcome"] == "home"
        espn.finish(espn_id, 101 if home_won else 99, 99 if home_won else 101)
        versions[str(params.version)] = versions.get(str(params.version), 0) + 1
    return versions


# --- the AI providers' clients (wrapped by the REAL analysts) ------------------------------------------------------

def _facts(user: str) -> Dict[str, Any]:
    text = user.split("Facts (json):\n", 1)[-1]
    text = text.split("\n\nYour previous answer", 1)[0]
    try:
        return json.loads(text)
    except ValueError:
        return {}


def answer_for(kind: str, facts: Mapping[str, Any], proposal: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """An answer of ``kind`` written only from ``facts`` (numbers and ids copied from it), as a careful analyst
    would: the worst miss's log loss, one bounded proposal on a knob the registry lists, plain change texts."""
    misses = facts.get("misses") or []
    if kind == "note":
        if not misses:
            return {"note": "No graded games to look at this week.", "hypotheses": []}
        worst = misses[0]
        return {"note": (f"The worst miss had a log loss of {worst['brain_ll']}; the market's was "
                         f"{worst.get('market_ll')}. Most misses are the games' own uncertainty."),
                "hypotheses": [{"text": "Late prices may lag team news in the last hour.",
                                "evidence": [worst["id"]]}]}
    if kind == "proposals":
        registry_rows = facts.get("registry") or {}
        if not misses or not registry_rows or proposal is None:
            return {"proposals": [], "none_reason": "Nothing in the facts supports a change this week."}
        sport = proposal.get("sport") or next(iter(registry_rows))
        return {"proposals": [{"id": f"{facts.get('week')}-{sport}-1", "knob": proposal["knob"], "sport": sport,
                               "direction": proposal.get("direction", "increase"), "size": proposal["size"],
                               "hypothesis": "The US books' prices carry news the sharp feed is slow to show.",
                               "evidence": [misses[0]["id"]], "test": "paired log loss on later games",
                               "expect": "lower log loss on games after the proposal"}], "none_reason": None}
    if kind == "changelog":
        texts = []
        for change in facts.get("changes") or []:
            n = change.get("eval_n")
            texts.append({"id": change["id"], "text": (f"Checked on {n} games after the week's refit." if n else
                                                       "Recorded by the weekly check.")})
        return {"texts": texts}
    tried = facts.get("candidates_tried", 0)
    return {"memo": f"This month the brain tried {tried} candidates; most weeks end with no change.",
            "data_ideas": ["Keep the closing lines of every league the app follows."]}


class _Usage:
    def __init__(self, tokens_in: int, tokens_out: int) -> None:
        self.input_tokens, self.output_tokens = tokens_in, tokens_out
        self.cache_creation_input_tokens = self.cache_read_input_tokens = 0
        self.iterations = None
        self.server_tool_use = None


class _Message:
    def __init__(self, text: str, model: str = "claude-opus-5-5") -> None:
        self.content = [types.SimpleNamespace(type="text", text=text)]
        self.usage = _Usage(max(1, len(text) // 3) + 4000, 1500)
        self.model, self.stop_reason = model, "end_turn"


class _Batches:
    def __init__(self, owner: "SmartClaude") -> None:
        self.owner = owner
        self.created: List[List[Dict[str, Any]]] = []
        self.status = "ended"  # a test can hold batches "in_progress"

    def create(self, requests: Sequence[Mapping[str, Any]]) -> Any:
        self.created.append([dict(r) for r in requests])
        return types.SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id: str) -> Any:
        return types.SimpleNamespace(processing_status=self.status)

    def results(self, batch_id: str) -> List[Any]:
        requests = self.created[int(batch_id.split("_")[1]) - 1]
        out = []
        for r in requests:
            text = self.owner.reply(r["params"])
            out.append(types.SimpleNamespace(custom_id=r["custom_id"], result=types.SimpleNamespace(
                type="succeeded", message=_Message(text))))
        return out


class SmartClaude:
    """Stands in for anthropic.Anthropic (``beta.messages.create`` and ``messages.batches``): answers from the fact
    sheet in the request. ``proposal`` is the knob it proposes ({"knob", "size", "sport", "direction"}) or None."""

    KINDS = {"note": "note", "proposals": "proposals", "changelog": "texts", "memo": "memo"}

    def __init__(self, proposal: Optional[Mapping[str, Any]] = None, batches: bool = True) -> None:
        self.proposal = dict(proposal) if proposal else None
        self.requests: List[Dict[str, Any]] = []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))
        self.messages = types.SimpleNamespace(batches=_Batches(self)) if batches else types.SimpleNamespace()

    @staticmethod
    def kind_of(request: Mapping[str, Any]) -> str:
        schema = ((request.get("output_config") or {}).get("format") or {}).get("schema") or {}
        props = set((schema.get("properties") or {}))
        if "proposals" in props:
            return "proposals"
        if "texts" in props:
            return "changelog"
        if "memo" in props:
            return "memo"
        return "note"

    def reply(self, request: Mapping[str, Any]) -> str:
        self.requests.append(dict(request))
        user = request["messages"][-1]["content"]
        return json.dumps(answer_for(self.kind_of(request), _facts(user), self.proposal))

    def _create(self, **request: Any) -> Any:
        return _Message(self.reply(request))


class _DSResponse:
    status_code = 200

    def __init__(self, body: Dict[str, Any]) -> None:
        self.body = body
        self.text = json.dumps(body)

    def json(self) -> Dict[str, Any]:
        return self.body


class SmartDeepSeek:
    """Stands in for DeepSeekAnalyst's requests session (JSON mode): answers from the fact sheet it was sent."""

    def __init__(self, proposal: Optional[Mapping[str, Any]] = None) -> None:
        self.proposal = dict(proposal) if proposal else None
        self.bodies: List[Dict[str, Any]] = []

    def post(self, url: str, json: Any = None, timeout: Any = None, headers: Any = None) -> Any:  # noqa: A002
        self.bodies.append(json)
        system = json["messages"][0]["content"]
        user = json["messages"][-1]["content"]
        kind = ("proposals" if '"proposals"' in system else "changelog" if '"texts"' in system else
                "memo" if '"memo"' in system else "note")
        text = _dumps(answer_for(kind, _facts(user), self.proposal))
        return _DSResponse({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": max(1, len(user) // 3), "completion_tokens": 1200},
                            "model": "deepseek-v4-pro"})



def analyst(store: Any, provider: Optional[str], clock: Callable[[], float], *, tz: Any = UTC,
            proposal: Optional[Mapping[str, Any]] = None, batches: bool = True, cents: int = 500) -> Any:
    """The REAL BrainAnalyst with a fake client underneath: provider "claude", "deepseek" or None (no key: the
    templates). The client is at ``.client``."""
    from telegram_betting_bot_simple.ai.claude import ClaudeAnalyst
    from telegram_betting_bot_simple.ai.deepseek import DeepSeekAnalyst
    from telegram_betting_bot_simple.learn import analyst as an

    if provider == "claude":
        client = SmartClaude(proposal, batches=batches)
        keys = {"anthropic_api_key": "sk-ant-test"}
        factory = lambda choice: an.ClaudeBrain(ClaudeAnalyst("sk-ant-test", client=client),  # noqa: E731
                                                batches=batches)
    elif provider == "deepseek":
        client = SmartDeepSeek(proposal)
        keys = {"deepseek_api_key": "sk-test"}
        factory = lambda choice: an.DeepSeekBrain(DeepSeekAnalyst("sk-test", session=client))  # noqa: E731
    else:
        client, keys, factory = None, {}, None
    brain = an.BrainAnalyst(store, mode="auto", keys=keys, monthly_cents=cents, factory=factory, clock=clock, tz=tz,
                            batches=batches)
    brain.client = client
    return brain


__all__ = ["Clock", "HistTransport", "hist_http", "FakeEspn", "log_games", "SmartClaude", "SmartDeepSeek", "analyst",
           "answer_for", "UTC"]


# --- the "Done when" run: the loop on a fake clock over several weeks ------------------------------------------------

DAY = 86400.0
PLANTED = {"prediction_market": 1.5}  # the prediction market's chance is far noisier than assumed: a refit finds it
REGRESSION = {"prediction_market": 0.02, "sharp": 2.5, "us_books": 2.5}  # later it turns sharp and the books don't:
# the weight taken from it now hurts, by enough (about +0.004 log loss a game) for the first planned look of the
# rollback watch (200 games, a 98.3% range) to see it
NBA, MLB, NHL = "basketball_nba", "baseball_mlb", "icehockey_nhl"


def at(*parts: int) -> float:
    return datetime.datetime(*parts, tzinfo=UTC).timestamp()


class LearningTimeline:
    """The learning loop end to end on a fake clock, over the REAL Store, grader, backfill, evaluator, tuner, registry,
    analyst and learner (only HTTP, ESPN and the AI clients are fakes; see the module's docstring). Each step moves
    the clock and ticks the learner the way the dashboard's thread does; ``seen`` keeps what each step returned.

    1. ``world``: 100 days of graded NBA and MLB games where the US books are the better feed (a planted
       improvement) and NHL games that are pure noise, logged through the slate's path and graded by the grader.
    2. ``first_week``: Tuesday 01:00 after the last game: the night (grade, the backfill from the captures inside its
       01:00-06:00 window, scores, the AI note as a Claude batch), the Monday run (the gate: NBA and MLB promoted, NHL
       held on noise; AI proposals; change texts; the owner's Telegram summary) and the month's memo.
    3. ``collect``: eleven minutes on, the Claude batches are back: the note, a proposal frozen as a challenger
       judged only on later games, the change texts and the memo, each checked.
    4. ``regression``: day by day, new games (the US books gone bad for the NBA), the nightly watch, until the NBA
       change is rolled back; the Monday run in between asks DeepSeek instead (the provider is chosen at run time);
       from ``templates_from`` no AI is linked (template notes).
    5. ``pause`` / ``resume``: the owner pauses (the Monday run then scores but changes nothing, no backfill, no AI)
       and switches learning back on.
    6. ``undo``: the owner takes the live MLB change back.

    ``undo`` and ``pause`` default to the learner's own methods; the integration test hands in the HTTP routes."""

    def __init__(self, store: Any, *, per_night: int = 400, nba_games: int = 1600, mlb_games: int = 1200,
                 nhl_games: int = 600, end: float = None, notify: Optional[Callable[[str], bool]] = None) -> None:
        from telegram_betting_bot_simple.dashboard.service import DashboardService
        from telegram_betting_bot_simple.data.cache import TTLCache
        from telegram_betting_bot_simple.learn import loop
        from zoneinfo import ZoneInfo

        self.store = store
        self.end = at(2026, 9, 7) if end is None else end  # a Monday: the last games are the day before
        self.clock = Clock(self.end + DAY + 3600)  # Tuesday 01:00
        self.espn = FakeEspn()
        self.transport = HistTransport()
        self.cache = TTLCache()
        self.service = DashboardService(None, self.espn, ZoneInfo("America/New_York"), store=store)
        self.sent: List[str] = []
        self.sizes = {"nba": nba_games, "mlb": mlb_games, "nhl": nhl_games}
        self.per_night = per_night
        self.proposal = {"knob": "consensus.weight.prediction_market", "size": 0.05, "sport": "nba",
                         "direction": "decrease"}
        self.claude = analyst(store, "claude", self.clock, proposal=self.proposal)
        self.deepseek = analyst(store, "deepseek", self.clock, proposal={
            "knob": "consensus.weight.prediction_market", "size": 0.05, "sport": "mlb", "direction": "decrease"})
        self.nobody = analyst(store, None, self.clock)
        self.learner = loop.Learner(store, service=self.service, cache=self.cache, clock=self.clock, tz=UTC,
                                    analyst=self.claude, notify=notify or (lambda text: self.sent.append(text) or True),
                                    backfill=self.backfill, public_url="https://bets.example.com")
        self.seen: Dict[str, Any] = {"versions_logged": {}}
        self.undo_with: Callable[[int], Mapping[str, Any]] = lambda version: self.learner.undo(version)
        self.pause_with: Callable[[bool], Mapping[str, Any]] = lambda paused: self.learner.set_paused(paused)

    # -- the outside world --

    def backfill(self, now: float) -> Dict[str, Any]:
        """The REAL nightly backfill (learn/history.py) over the captures, on the fake clock (its 2-second pace
        moves the clock)."""
        from telegram_betting_bot_simple.learn.history import Backfill
        return Backfill(self.store, self.cache, self.clock, http=hist_http(self.clock, self.transport), tz=UTC,
                        per_night=self.per_night).run_night()

    def log(self, records: Sequence[Mapping[str, Any]]) -> None:
        for version, n in log_games(self.store, sorted(records, key=lambda r: r["logged_at"]), self.espn).items():
            self.seen["versions_logged"][version] = self.seen["versions_logged"].get(version, 0) + n

    def tick(self, when: Optional[float] = None) -> Dict[str, Any]:
        if when is not None:
            self.clock.t = when
        return self.learner.tick()

    # -- the steps --

    def world(self) -> Dict[str, Any]:
        from tests import learn_sim as sim

        days = 100
        out = {}
        for key, league, per_day, noise, seed, prefix in (
                ("nba", NBA, self.sizes["nba"] // days, PLANTED, 11, "n"),
                ("mlb", MLB, self.sizes["mlb"] // days, PLANTED, 13, "b"),
                ("nhl", NHL, self.sizes["nhl"] // days, None, 12, "h")):
            out[key] = sim.consensus_world(per_day * days, league=league, seed=seed, noise=noise, per_day=per_day,
                                           start=self.end - days * DAY, game_prefix=prefix)
        self.log(out["nba"] + out["mlb"] + out["nhl"])
        self.seen["world"] = {k: len(v) for k, v in out.items()}
        return self.seen["world"]

    def first_week(self) -> Dict[str, Any]:
        self.seen["first"] = self.tick(self.end + DAY + 3600)
        return self.seen["first"]

    def collect(self) -> Dict[str, Any]:
        from telegram_betting_bot_simple.learn import loop
        self.seen["collect"] = self.tick(self.clock.t + loop.POLL_SECONDS + 60)
        return self.seen["collect"]

    def day(self, d: int, nba_noise: Mapping[str, float]) -> Dict[str, Any]:
        """Day ``d`` after the first week's run: that day's games (logged with whatever version is live, graded
        once over), then the night's tick at 02:36 the next day."""
        from tests import learn_sim as sim

        begin = self.end + (2 + d) * DAY
        games = (sim.consensus_world(24, league=NBA, seed=100 + d, noise=nba_noise, per_day=24, start=begin,
                                     game_prefix=f"nd{d}-")
                 + sim.consensus_world(16, league=MLB, seed=200 + d, noise=PLANTED, per_day=16, start=begin,
                                       game_prefix=f"bd{d}-")
                 + sim.consensus_world(6, league=NHL, seed=300 + d, per_day=6, start=begin, game_prefix=f"hd{d}-"))
        self.log(games)
        return self.tick(begin + DAY + 2.6 * 3600)

    def regression(self, days: int = 14, deepseek_from: int = 4, templates_from: int = 7) -> Dict[str, Any]:
        """Days of games with the US books gone bad for the NBA, until the nightly watch rolls the NBA change back.
        From day ``deepseek_from`` DeepSeek is the linked AI (the Monday run asks it), from ``templates_from`` none."""
        out: Dict[str, Any] = {"ticks": [], "rolled_back_on": None}
        for d in range(days):
            if d == deepseek_from:
                self.learner.analyst = self.deepseek
            if d == templates_from:
                self.learner.analyst = self.nobody
            report = self.day(d, REGRESSION)
            out["ticks"].append(report)
            night = next((r for r in report["ran"] if r["kind"] == "night"), None)
            rolled = ((night or {}).get("steps", {}).get("scores", {}).get("result") or {}).get("rolled_back")
            if rolled and out["rolled_back_on"] is None:
                out["rolled_back_on"] = d
            if out["rolled_back_on"] is not None and d > out["rolled_back_on"] and d >= templates_from:
                break  # a day of games after the rollback, logged on the version before it
        self.seen["regression"] = out
        return out

    def pause(self) -> Dict[str, Any]:
        """The owner pauses learning; the next Monday's run then scores but changes nothing (and nothing is
        fetched or asked)."""
        from telegram_betting_bot_simple.learn import loop

        self.learner.analyst = self.claude
        before = {"transport": len(self.transport.calls), "claude": len(self.claude.client.requests),
                  "batches": len(self.claude.client.messages.batches.created)}
        state = dict(self.pause_with(True))
        _, due = loop.period("week", self.clock.t + 7 * DAY, UTC)
        report = self.tick(due + 60)
        after = {"transport": len(self.transport.calls), "claude": len(self.claude.client.requests),
                 "batches": len(self.claude.client.messages.batches.created)}
        self.seen["pause"] = {"state": state, "report": report, "before": before, "after": after}
        return self.seen["pause"]

    def resume(self) -> Dict[str, Any]:
        self.seen["resume"] = dict(self.pause_with(False))
        return self.seen["resume"]

    def undo(self, sport: str = "mlb") -> Dict[str, Any]:
        from contextlib import closing

        from telegram_betting_bot_simple.learn import brain_store as bs
        with closing(self.store._connect()) as db:
            row = bs.champion(db, sport)
        if row is None:
            return {"error": f"no live change for {sport}"}
        self.seen["undo"] = {"version": row["version"], "parent": row.get("parent") or 0,
                             "result": dict(self.undo_with(int(row["version"])))}
        return self.seen["undo"]

    def run_all(self, *, undo: bool = True) -> Dict[str, Any]:
        self.world()
        self.first_week()
        self.collect()
        self.regression()
        self.pause()
        self.resume()
        if undo:
            self.undo()
        return self.seen


__all__ += ["LearningTimeline", "at", "DAY", "PLANTED", "REGRESSION"]
