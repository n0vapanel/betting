"""The app's outputs through its wiring (not the engine alone) on fixed inputs, for the learning loop's "byte-identical
defaults" check: with an empty registry (a real Store holding no brain versions), every one of these must equal what
the code before the learning loop (b9e3111) produced. tests/fixtures/learn_eval/wiring_golden.json holds that code's
outputs, made by running this same module against a copy of b9e3111 (the module uses only calls both versions have).

- slate: ``data/slate.market_detail`` (the "Markets say" blend) for two- and three-way games, near the start, with
  thin, stale and prediction-market sources and ratings only;
- the Stage 4 world of tests/test_context_wiring (the real SlateBuilder, ContextService, DashboardService over a
  real Store, no network): the served slate, every game's context block, the consensus_log and context_log rows
  the slate and the context service wrote, the goals model of the EPL game and the totals legs' adjusted chances;
- soccer: ``DashboardService.soccer_derived`` over saved boards (rho and the both-teams-to-score nudge);
- NBA: ``nba_model.predict_game`` with ``DashboardService.params`` (the market and ESPN weights) and the method note.
"""

from __future__ import annotations

import datetime
import gzip
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

VOLATILE = ("generated_at", "logged_at", "fetched_at", "graded_at", "updated_at", "etag", "built_at", "as_of_wall")


def _clean(value: Any) -> Any:
    """Drop wall-clock stamps (the only thing allowed to differ between two runs)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def dump(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, default=str)


def _src(name: str, kind: str, p: Dict[str, float], as_of: str, status: str = "ok") -> Dict[str, Any]:
    return {"name": name, "kind": kind, "p": p, "as_of": as_of, "status": status}


def slate_cases() -> List[Dict[str, Any]]:
    start = datetime.datetime(2026, 10, 30, 23, 0, tzinfo=datetime.timezone.utc).timestamp()
    iso = lambda ts: datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa
    two = {"home": 0.62, "away": 0.38}
    three = {"home": 0.47, "draw": 0.27, "away": 0.26}
    cases = []
    for league, p in (("basketball_nba", two), ("icehockey_nhl", {"home": 0.55, "away": 0.45}),
                      ("americanfootball_nfl", {"home": 0.71, "away": 0.29}), ("soccer_epl", three),
                      ("baseball_mlb", {"home": 0.51, "away": 0.49})):
        for minutes in (600, 10):  # hours before, and inside the near-start window
            now = start - minutes * 60
            at = iso(now - 120)
            shifted = {k: max(0.02, min(0.98, v + (0.03 if k == "home" else -0.03 / (len(p) - 1))))
                       for k, v in p.items()}
            sources = [_src("Pinnacle", "sharp", p, at), _src("US books", "us_books", shifted, at),
                       _src("Kalshi", "prediction_market", {k: round(v, 3) for k, v in shifted.items()}, at),
                       _src("Polymarket", "prediction_market", p, at, "thin"),
                       _src("ESPN", "model_rating", {k: round((v + 1 / len(p)) / 2, 4) for k, v in p.items()}, at),
                       _src("Model", "team_model", shifted, at, "stale")]
            cases.append({"league": league, "start": start, "now": now, "sources": sources})
        cases.append({"league": league, "start": start, "now": start - 7200,
                      "sources": [_src("ESPN", "model_rating", p, iso(start - 7300))]})  # ratings only
    cases.append({"league": "mma_mixed_martial_arts", "start": start, "now": start - 3600,
                  "sources": [_src("Pinnacle", "sharp", two, iso(start - 3700))], "neutral": True})
    return cases


def slate_outputs(store: Any) -> List[str]:
    from telegram_betting_bot_simple.data import slate as slate_mod

    tuned = "params" in inspect.signature(slate_mod.market_detail).parameters
    out = []
    for case in slate_cases():
        kwargs: Dict[str, Any] = {"neutral": bool(case.get("neutral"))}
        if tuned:  # the call the slate makes now: the sport's live parameters (the defaults here)
            kwargs["params"] = slate_mod.brain_params(case["league"], store)
        out.append(dump(slate_mod.market_detail(case["league"], case["sources"], case["start"], case["now"],
                                                **kwargs)))
    return out


def context_world_outputs(tmp_path: Path) -> Dict[str, Any]:
    """The Stage 4 world (tests/test_context_wiring.World): slate, context blocks, logs, goals model, legs."""
    import requests

    from telegram_betting_bot_simple.data import context as context_mod
    from telegram_betting_bot_simple.data.odds_service import configure_books
    from tests import test_context_wiring as cw

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no network here")

    real_request = requests.Session.request
    requests.Session.request = refuse  # type: ignore[method-assign]
    configure_books()
    try:
        world = cw.World(tmp_path)
        world.round()
        world.later()
        out: Dict[str, Any] = {}
        current = world.slate.current()
        body = json.loads(gzip.decompress(current.gzip) if getattr(current, "gzip", None) else current.raw)
        out["slate"] = dump(body)
        out["blocks"] = {f"{g[0]}:{g[1]}": dump(world.context.for_game(g[0], g[1])) for g in cw.GAMES}
        out["consensus_log"] = dump(sorted(world.store.consensus_rows(), key=lambda r: (r["league"], r["game_key"])))
        rows = world.store.context_rows() if hasattr(world.store, "context_rows") else []
        out["context_log"] = dump(sorted(rows, key=lambda r: (r.get("league"), r.get("game_key"))))
        epl = next(g for g in cw.GAMES if g[0] == cw.EPL)
        saved = world.store.odds_snapshot(cw.EPL)
        game = next(g for g in saved[1] if g["id"] == epl[1])
        out["goals_model"] = dump(world.service.soccer_derived(cw.EPL, game))
        legs = {}
        for g in cw.GAMES:
            for side in ("over", "under"):
                block = world.context.for_game(g[0], g[1])
                adjustment = next((a for a in block.get("adjustments") or [] if a.get("kind") == "total"), None)
                if adjustment:
                    total = (g[12] or (None,))[0]
                    legs[f"{g[1]}:{side}"] = context_mod.adjusted_chance(g[0], adjustment, side, total, 0.5)
        out["legs"] = dump(legs)
        return out
    finally:
        requests.Session.request = real_request  # type: ignore[method-assign]
        configure_books()


def soccer_outputs(store: Any) -> List[str]:
    from telegram_betting_bot_simple.dashboard.service import DashboardService
    from telegram_betting_bot_simple.data.odds_service import normalize_event

    service = DashboardService(None, None, ZoneInfo("America/New_York"), store=store)
    out = []
    for i, (h, d, a, over, under) in enumerate(((1.55, 4.2, 6.0, 1.85, 2.0), (2.6, 3.3, 2.9, 2.2, 1.7),
                                                (5.5, 4.0, 1.6, 1.62, 2.35), (2.05, 3.1, 3.9, 2.6, 1.5))):
        books = [{"key": key, "title": title, "last_update": "2026-10-30T12:00:00Z", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Home FC", "price": round(h * k, 2)},
                                        {"name": "Draw", "price": round(d * k, 2)},
                                        {"name": "Away FC", "price": round(a * k, 2)}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": round(over * k, 2), "point": 2.5},
                                           {"name": "Under", "price": round(under * k, 2), "point": 2.5}]}]}
            for key, title, k in (("pinnacle", "Pinnacle", 1.0), ("draftkings", "DraftKings", 0.98),
                                  ("fanduel", "FanDuel", 0.97))]
        game = normalize_event({"id": f"s{i}", "sport_key": "soccer_epl", "sport_title": "EPL",
                                "commence_time": "2026-10-31T15:00:00Z", "home_team": "Home FC",
                                "away_team": "Away FC", "bookmakers": books})
        out.append(dump(service.soccer_derived("soccer_epl", game)))
        # With the blend's 1X2, as the game route passes it
        out.append(dump(service.soccer_derived("soccer_epl", game, {"home": 0.5, "draw": 0.27, "away": 0.23})))
    return out


def nba_outputs(store: Any) -> List[str]:
    from telegram_betting_bot_simple.dashboard.service import DashboardService
    from telegram_betting_bot_simple.engine import nba_model

    service = DashboardService(None, None, ZoneInfo("America/New_York"), store=store)
    params = service.params
    out = [dump(params), service._method_note()]
    for home, away, espn, market in (
            ((112.0, 108.0), (109.0, 113.0), 0.61,
             {"spreads": {"home": {"point": -4.5}}, "totals": {"over": {"point": 226.5}}}),
            ((118.0, 104.0), (104.0, 118.0), None,
             {"h2h": {"home": {"fair": 0.81}}, "totals": {"over": {"point": 231.0}}}),
            ((110.0, 111.0), (111.0, 110.0), 0.47, {})):
        teams = [nba_model.TeamInputs(name, nba_model.TeamRating(pts[0], pts[1], 30.0), 0.0, [], False)
                 for name, pts in (("Home", home), ("Away", away))]
        out.append(dump(nba_model.predict_game(teams[0], teams[1], 112.0, market, espn, params)))
    return out


def wiring_outputs(store: Any, tmp_path: Path) -> Dict[str, Any]:
    started = time.time()
    out = {"slate": slate_outputs(store), "soccer": soccer_outputs(store), "nba": nba_outputs(store),
           "world": context_world_outputs(tmp_path)}
    out["_seconds"] = round(time.time() - started, 2)
    return out
