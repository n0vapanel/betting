"""The player scan (engine/scan.py) and its boards (data/player_board.py): v3's "who's projected for 20+ tonight".

Nothing here touches the network: ESPN's answers come from tests/fixtures/scan/ (trimmed captures from 28 October
2026) through a patched HTTP layer, so the real EspnClient, PlayerData and shared TTLCache run underneath.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import re
import threading
import time
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
import requests

from telegram_betting_bot_simple.ai.explain import BANNED
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.data import espn_players
from telegram_betting_bot_simple.data.cache import POLICY, TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.player_board import PlayerBoard, board_for, parse_events
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.engine import calls, nba_model
from telegram_betting_bot_simple.engine import scan as engine

FIX = Path(__file__).parent / "fixtures" / "scan"
NBA, NFL, NHL, MLB = engine.NBA, engine.NFL, engine.NHL, engine.MLB
NOW = datetime.datetime(2026, 10, 28, 16, 0, tzinfo=datetime.timezone.utc).timestamp()  # before the first tip
ESPN_HOSTS = {"site.api.espn.com", "site.web.api.espn.com", "sports.core.api.espn.com"}


def load(name: str) -> Any:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def log_rows(sport: str, *names: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in names:
        rows += espn_players._parse_gamelog(sport, load(name))["rows"]
    return rows


SGA_ROWS = log_rows("basketball", "nba_gamelog_4278073", "nba_gamelog_4278073_2026")  # this season's 4, then last's


def props_of(name: str, sport: str) -> Dict[str, List[Dict[str, Any]]]:
    return espn_players._props_from_json(espn_players._props_to_json(espn_players._parse_props(sport, load(name)["items"])))


# --- a fake web: ESPN's answers from the fixtures ------------------------------------------------------------

class Web:
    """Patches requests' Session.request: answers ESPN URLs from ``routes`` (a function url, params -> JSON or None
    for a 404), counts requests in flight (``peak``), records each with the cache loader's kind it ran inside, and
    sleeps ``delay`` seconds per request (a recorded latency)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, routes, delay: float = 0.0) -> None:
        self.routes = routes
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []
        self.flying = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.local = threading.local()
        web = self

        def request(session, method, url, params=None, **kwargs):
            with web.lock:
                web.flying += 1
                web.peak = max(web.peak, web.flying)
                frames = getattr(web.local, "kinds", [])
                web.calls.append({"url": str(url), "params": dict(params or {}), "kind": frames[-1] if frames else None})
            try:
                if web.delay:
                    time.sleep(web.delay)
                body = web.routes(str(url), dict(params or {}))
            finally:
                with web.lock:
                    web.flying -= 1
            response = requests.Response()
            response.status_code = 200 if body is not None else 404
            response._content = json.dumps(body if body is not None else {}).encode()
            response.url = str(url)
            return response

        monkeypatch.setattr(requests.sessions.Session, "request", request)

        def wrap(real):
            def read(cache, key, ttl, loader, **options):
                kind = options.get("kind")

                def marked():
                    kinds = web.local.__dict__.setdefault("kinds", [])
                    kinds.append(kind)
                    try:
                        return loader()
                    finally:
                        kinds.pop()
                return real(cache, key, ttl, marked, **options)
            return read

        monkeypatch.setattr(TTLCache, "get", wrap(TTLCache.get))
        monkeypatch.setattr(TTLCache, "get_with_time", wrap(TTLCache.get_with_time))

    def hosts(self) -> set:
        return {re.sub(r"^https?://([^/]+).*$", r"\1", call["url"]) for call in self.calls}


def fixture_routes(url: str, params: Dict[str, Any]) -> Optional[Any]:
    """The captured ESPN answers: NHL and NFL boards (scoreboards, Edmonton's and Buffalo's rosters, their props and
    a few game logs); anything else is an empty answer."""
    sport_path = re.search(r"/sports/(\w+)/(?:leagues/)?([\w.]+)/", url)
    sport, league = (sport_path.group(1), sport_path.group(2)) if sport_path else ("", "")
    short = {"nba": "nba", "nhl": "nhl", "nfl": "nfl", "mlb": "mlb", "eng.1": "epl"}.get(league, league)
    if url.endswith("/scoreboard"):
        dates = {"nba": "20261028", "nhl": "20261028", "nfl": "20261101", "mlb": "20261028", "epl": "20261031"}
        if params.get("dates") == dates.get(short):
            return load(f"{short}_scoreboard_{dates[short]}")
        return {"events": []}
    if url.endswith("/summary"):
        name = f"{short}_summary_{params.get('event')}"
        return load(name) if (FIX / f"{name}.json").exists() else {"injuries": []}
    match = re.search(r"/teams/(\d+)/roster$", url)
    if match:
        name = f"{short}_roster_{match.group(1)}"
        return load(name) if (FIX / f"{name}.json").exists() else {"athletes": []}
    match = re.search(r"/athletes/(\d+)/gamelog$", url)
    if match:
        suffix = f"_{params['season']}" if params.get("season") else ""
        name = f"{short}_gamelog_{match.group(1)}{suffix}"
        return load(name) if (FIX / f"{name}.json").exists() else {}
    match = re.search(r"/events/(\d+)/competitions/\d+/odds/100/propBets$", url)
    if match:
        name = f"{short}_props_{match.group(1)}"
        return load(name) if (FIX / f"{name}.json").exists() else {"items": [], "pageCount": 1}
    return {}


class Service:
    """The DashboardService surface a board reads, without the NBA model (no odds)."""

    def __init__(self, espn: EspnClient, cache: TTLCache, detail=None, odds=None) -> None:
        self.espn = espn
        self.cache = cache
        self.odds = odds
        self._detail = detail

    def detail(self, league, event_id):
        if self._detail is None:
            raise LookupError(event_id)
        return self._detail(league, event_id)


def make_board(now: float = NOW, **options: Any) -> PlayerBoard:
    cache = TTLCache(name="scan-test")
    espn = EspnClient(cache, clock=lambda: now)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(now, datetime.timezone.utc))
    service = Service(espn, cache, options.pop("detail", None), options.pop("odds", None))
    return PlayerBoard(service, players, cache, now=lambda: now, **options)


def no_banned(texts: List[str]) -> None:
    for text in texts:
        for name, pattern in BANNED:
            assert not pattern.search(text), (name, text)


# --- the lingo and the stat table --------------------------------------------------------------------------

SPEC_ALIASES = {
    NBA: {"pts": ["points", "pts", "scorers", "buckets"], "reb": ["rebounds", "boards"], "ast": ["assists", "dimes"],
          "fg3m": ["threes", "3s", "3PM", "triples"], "stl": ["steals"], "blk": ["blocks"], "stocks": ["stl+blk"],
          "pra": ["PRA", "pts+reb+ast"], "pr": ["P+R"], "pa": ["pts+ast"], "ra": ["reb+ast"],
          "dd": ["double-double"], "td": ["triple-double"]},
    NFL: {"pass_yds": ["passing yards"], "pass_td": ["passing TDs"], "rush_yds": ["rushing yards"],
          "rec_yds": ["receiving yards"], "rec": ["receptions", "catches"], "rush_rec_yds": ["rush+rec yards"],
          "anytime_td": ["anytime TD", "ATTD", "scores a TD"]},
    NHL: {"goals": ["goals"], "assists": ["assists"], "points": ["points"],
          "sog": ["shots", "shots on goal", "SOG"], "saves": ["saves"], "anytime_goal": ["anytime goal"]},
    MLB: {"hits": ["hits"], "tb": ["total bases"], "hr": ["home runs", "dingers"], "rbi": ["RBI"], "runs": ["runs"],
          "hrr": ["H+R+RBI"], "k": ["strikeouts", "Ks", "pitcher strikeouts"]},
    "soccer_epl": {"goals": ["goals", "anytime scorer"], "shots": ["shots"], "sot": ["shots on target"],
                   "assists": ["assists"]},
}


def test_the_stat_table_has_every_league_stat_and_alias() -> None:
    assert engine.STATS[engine.WNBA] is engine.STATS[NBA]
    for league, stats in SPEC_ALIASES.items():
        assert set(stats) <= set(engine.STATS[league]), league
        for key, aliases in stats.items():
            spec = engine.STATS[league][key]
            assert spec["label"] and isinstance(spec["unit"], str) and spec["markets"] is not None
            for alias in aliases + [key]:
                assert engine.resolve_stat(league, alias) == key, (league, alias)
    # Every soccer league the app shows shares the table; the NBA's stats with a measured spread name calls' key
    assert engine.STATS["soccer_spain_la_liga"] is engine.STATS["soccer_epl"]
    assert {key for key, spec in engine.STATS[NBA].items() if spec["nba"]} == {"pts", "reb", "ast", "fg3m", "stl",
                                                                              "blk", "pra"}
    assert all(engine.STATS[NBA][key]["nba"] in calls.STAT_DISPERSION for key in ("pts", "reb", "ast", "pra"))
    menu = engine.stats_menu(NHL)[NHL]
    assert {"key": "sog", "label": "shots on goal", "short": "shots on goal", "unit": "SOG", "yes": False,
            "threshold": None} in menu


@pytest.mark.parametrize("text, threshold, side", [
    ("players projected to score at least 20 points today", 20, "over"),
    ("20+ points", 20, "over"),
    ("over 19.5 points", 20, "over"),
    ("20 or more points", 20, "over"),
    ("guys with 20 points tonight", 20, "over"),
    ("more than 20 points", 21, "over"),
    ("under 9.5 rebounds", 10, "under"),
    ("9 or fewer rebounds", 10, "under"),
    ("u9.5 boards", 10, "under"),
    ("o24.5 points", 25, "over"),
    ("1+ goals premier league", 1, "over"),
    ("anytime goal scorers nhl", 1, "over"),
])
def test_parse_ask_reads_the_number_and_side(text: str, threshold: int, side: str) -> None:
    ask = engine.parse_ask(text)
    assert (ask["threshold"], ask["side"]) == (threshold, side)
    assert ask["stat"] is not None


def test_parse_ask_routes_the_lingo() -> None:
    ask = engine.parse_ask("give me players that are projected to score at least 20 points today")
    assert (ask["stat"], ask["threshold"], ask["side"], ask["day"]) == ("pts", 20, "over", "today")
    assert ask["leagues"][0] == NBA and ask["stats"][NHL] == "points"  # the NBA first for points
    assert engine.parse_ask("who's going for 10+ boards tonight")["stats"] == {NBA: "reb", engine.WNBA: "reb"}
    assert engine.parse_ask("who's hitting 3+ threes")["stat"] == "fg3m"
    td = engine.parse_ask("anytime TD scorers this week")
    assert (td["stats"], td["threshold"]) == ({NFL: "anytime_td"}, 1)  # not the NBA's "scorers"
    sog = engine.parse_ask("SOG 3+")
    assert (sog["leagues"], sog["stat"], sog["threshold"]) == ([NHL], "sog", 3)  # no soccer unless named
    assert engine.parse_ask("shots on target 2+ premier league")["stats"] == {"soccer_epl": "sot"}
    ks = engine.parse_ask("Ks tonight")
    assert (ks["stats"], ks["threshold"]) == ({MLB: "k"}, None)
    pra = engine.parse_ask("PRA overs tonight")
    assert (pra["stat"], pra["threshold"]) == ("pra", None)  # each player's posted line
    assert engine.parse_ask("double-double tomorrow")["stat"] == "dd"
    assert engine.parse_ask("double-double tomorrow")["day"] == "tomorrow"
    assert engine.parse_ask("give me 3 locks")["stat"] is None  # top picks, not a scan
    assert engine.parse_ask("250+ passing yards")["stats"] == {NFL: "pass_yds"}
    assert engine.parse_ask("7+ strikeouts")["stats"] == {MLB: "k"}


def test_make_ask_checks_and_labels() -> None:
    assert engine.make_ask(NBA, "points", 20)["label"] == "20+ points"
    assert engine.make_ask(NBA, "boards", "9.5", "under")["label"] == "under 9.5 rebounds"
    assert engine.make_ask(NBA, "boards", "9.5", "under")["threshold"] == 10
    assert engine.make_ask(NFL, "ATTD")["label"] == "anytime TD"
    dd = engine.make_ask(NBA, "double-double", 7, "under")
    assert (dd["threshold"], dd["side"], dd["label"]) == (2, "over", "double-double")
    assert engine.make_ask(NBA, "pra")["label"] == "PRA overs (posted lines)"
    assert engine.make_ask(MLB, "Ks", 7)["line"] == 6.5
    for bad in [("basketball_euroleague", "pts", 20), (NBA, "touchdowns", 1), (NBA, "pts", "lots"), (NBA, "pts", -3),
                (NBA, "pts", float("nan"))]:
        with pytest.raises(ValueError):
            engine.make_ask(*bad)
    with pytest.raises(ValueError):
        engine.make_ask(NBA, "pts", 20, "sideways")
    with pytest.raises(ValueError):
        engine.make_ask(NBA, "pts", 20, "over", "next week")


# --- hit rates and averages against hand counts -----------------------------------------------------------

def test_history_matches_hand_counts() -> None:
    pts = engine.STATS[NBA]["pts"]
    # SGA, newest first: this season 26, 34, 17, 32; then last season's playoffs 35, 15, 32, 19, 26, 30, ...
    hits, avg, values = engine.history(SGA_ROWS, pts, 20)
    assert values[:10] == [26, 34, 17, 32, 35, 15, 32, 19, 26, 30]
    assert hits == {"l5": [4, 5], "l10": [7, 10], "season": [3, 4]}
    assert avg == {"l5": 28.8, "l10": 26.6, "season": round(109 / 4, 1)}
    # An under counts the games below the number: rebounds 3, 6, 2, 6 | 4, 1, 2, 4, 2, 4 under 5.5
    hits, avg, _ = engine.history(SGA_ROWS, engine.STATS[NBA]["reb"], 6, "under")
    assert hits == {"l5": [3, 5], "l10": [8, 10], "season": [2, 4]}
    # The same counts as the player card's hit rates against the same line
    card = espn_players.hit_rates(SGA_ROWS, "pts", 19.5)
    assert [card["hits"]["last5"], card["hits"]["last10"], card["hits"]["season"]] == [4, 7, 3]


def test_history_sums_the_columns_a_row_has_and_respects_needs() -> None:
    attd = engine.STATS[NFL]["anytime_td"]
    cook = log_rows("football", "nfl_gamelog_4379399")  # rushing TDs 1, 2, 1, 1, 1, 0; no receiving TDs
    hits, avg, values = engine.history(cook, attd, 1)
    assert values == [1, 2, 1, 1, 1, 0]
    assert hits == {"l5": [5, 5], "l10": [5, 6], "season": [5, 6]} and avg["l5"] == 1.2
    receiver = [{"season": "2026", "stats": {"REC TD": 1, "REC YDS": 40}}]  # no rushing columns at all
    assert engine.history(receiver, attd, 1)[0]["l10"] == [1, 1]
    hits_stat = engine.STATS[MLB]["hits"]
    pitcher = {"season": "2026", "stats": {"IP": 6.0, "H": 5, "K": 8}}
    batter = {"season": "2026", "stats": {"AB": 4, "H": 2, "2B": 1, "3B": 0, "HR": 1, "R": 1, "RBI": 2}}
    assert engine.log_value(pitcher, hits_stat) is None and engine.log_value(batter, hits_stat) == 2
    assert engine.log_value(batter, engine.STATS[MLB]["tb"]) == 2 + 1 + 0 + 0 + 1 * 3  # H + 2B + 2x3B + 3xHR = 6
    assert engine.log_value(pitcher, engine.STATS[MLB]["k"]) == 8
    dd = engine.STATS[NBA]["dd"]
    assert engine.log_value({"stats": {"pts": 24, "reb": 11, "ast": 9, "stl": 1, "blk": 0}}, dd) == 2
    assert engine.log_value({"stats": {"pts": 24, "reb": 11, "ast": 10, "stl": 1, "blk": 0}}, dd) == 3


# --- the form model against hand values -------------------------------------------------------------------

def test_form_mean_is_recency_weighted_and_shrunk() -> None:
    w = [1.0, 0.5 ** 0.2, 0.5 ** 0.4]
    weighted = (30 * w[0] + 20 * w[1] + 10 * w[2]) / sum(w)
    assert engine.form_mean([30, 20, 10], [30, 20, 10]) == pytest.approx((3 * weighted + 5 * 20) / 8, abs=1e-9)
    assert engine.form_mean([30, 20, 10], [30, 20, 10]) == pytest.approx(20.345473, abs=1e-5)
    # Ten games of 30 after a season averaging 20: (10 x 30 + 5 x 20) / 15
    assert engine.form_mean([30] * 10, [30] * 10 + [10] * 10) == pytest.approx(400 / 15)
    assert engine.form_mean([], []) is None


def test_count_and_yards_chances_against_hand_values() -> None:
    assert engine.count_chance(2.0, 1, 1.0) == pytest.approx(1 - math.exp(-2))  # anytime: 1 - P(0), Poisson
    assert engine.count_chance(2.0, 3, 2.0) == pytest.approx(0.3125)  # NB r=2, p=0.5: 1 - (.25 + .25 + .1875)
    assert engine.count_chance(2.0, 3, 2.0, "under") == pytest.approx(0.6875)
    assert engine.count_chance(0.0, 1, 1.0) == 0.0
    assert engine.dispersion([0, 2, 4]) == pytest.approx(2.0)
    assert engine.dispersion([2, 2, 2]) == 1.0  # never below a Poisson
    assert engine.dispersion([0, 10, 0, 10]) == engine.MAX_DISPERSION
    assert engine.dispersion([5, 7]) == 1.0  # too few games to fit
    assert engine.yards_chance(80, 75, 20) == pytest.approx(NormalDist().cdf(0.275))
    assert engine.yards_chance(80, 75, 20, "under") == pytest.approx(1 - NormalDist().cdf(0.275))
    assert engine.yards_sd([50, 50, 50], 120) == pytest.approx(18.0)  # 15% of the mean
    assert engine.yards_sd([10, 10], 40) == 10.0  # at least 10 yards
    assert engine.yards_sd([40, 60, 80], 60) == pytest.approx(20.0)  # the log's own spread


def test_form_projection_picks_the_model_by_kind() -> None:
    shakir = log_rows("football", "nfl_gamelog_4373678")
    yards = engine.form_projection(shakir, engine.STATS[NFL]["rec_yds"])
    values = [34, 80, 82, 6, 38, 50]
    assert set(yards) == {"mean", "games", "sd"} and yards["games"] == 6
    assert yards["sd"] == pytest.approx(engine.yards_sd(values, yards["mean"]))
    draisaitl = log_rows("hockey", "nhl_gamelog_3114727")
    sog = engine.form_projection(draisaitl, engine.STATS[NHL]["sog"])
    assert set(sog) == {"mean", "games", "dispersion"} and sog["dispersion"] >= 1.0
    chance = engine.form_chance(sog, engine.STATS[NHL]["sog"], 3)
    assert chance == pytest.approx(engine.count_chance(sog["mean"], 3, sog["dispersion"]))
    assert engine.form_projection([], engine.STATS[NHL]["sog"]) is None


# --- the scan: NBA rows are the player card's own numbers ----------------------------------------------------

SGA_ROW = {"id": "4278073", "name": "Shai Gilgeous-Alexander", "position": "G", "status": "", "minutes": 34.3,
           "pts": 31.2, "reb": 4.6, "ast": 6.1, "fg3m": 2.0, "stl": 1.4, "blk": 0.8, "tov": 2.2, "out": False,
           "side": "away"}


def board_view(players: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    view = {"league": NBA, "day": "2026-10-28", "as_of": "2026-10-28T16:00:00Z",
            "games": {"401909891": {"event_id": "401909891", "label": "OKC @ DAL", "start": "2026-10-29T01:30:00Z",
                                    "outs": {}, "props": {"book": "DraftKings", "as_of": "2026-10-28T15:59:00Z",
                                                          "stale": False}}},
            "players": players, "games_ready": 1, "games_total": 1, "games_started": 0}
    view.update(extra)
    return view


def nba_player(row: Dict[str, Any], rows: List[Dict[str, Any]], props: List[Dict[str, Any]], **extra: Any
               ) -> Dict[str, Any]:
    player = {"athlete_id": row["id"], "name": row["name"], "team": "OKC", "opponent": "DAL", "home": False,
              "position": row["position"], "event_id": "401909891", "status": row.get("status") or None,
              "projection": {"means": {k: row[k] for k in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")},
                             "minutes": row["minutes"]}, "rows": rows, "props": props}
    player.update(extra)
    return player


def test_nba_rows_carry_the_player_cards_own_chance() -> None:
    props = props_of("nba_props_401909891", "basketball")["4278073"]
    player = nba_player(SGA_ROW, SGA_ROWS, props)
    for stat, threshold, prop_stat in (("pts", 20, "pts"), ("pts", 31, "pts"), ("pra", 40, "pts+reb+ast")):
        result = engine.scan(board_view([player]), engine.make_ask(NBA, stat, threshold))
        row = (result["players"] + result["close"])[0]
        line = threshold - 0.5
        card = Lab._prop_view({"market": "x", "stat": prop_stat, "kind": "over_under", "line": line,
                               "over_price": Decimal("1.90")}, SGA_ROWS, SGA_ROW)
        assert row["chance"] == card["ai"]["over"], (stat, threshold)
        assert row["model"] == "nba" and row["chance_source"] == "nba" and row["minutes"] == 34.3
        assert row["projection"] == card["ai"]["mean"]
        assert row["model_label"] == "NBA model"
    # The 20+ milestone DraftKings posts: the scan's edge is the card's edge on the same price
    rung = next(p for p in props if p["kind"] == "milestone" and p["market"] == "player_points" and p["line"] == 19.5)
    row = engine.scan(board_view([player]), engine.make_ask(NBA, "pts", 20))["players"][0]
    card = Lab._prop_view(rung, SGA_ROWS, SGA_ROW)
    assert row["edge"] == card["ai"]["edge_over"] and row["line"]["matches"] is True
    assert row["line"]["label"] == "20+" and row["line"]["over"] == card["over"]
    assert row["why"][0] == "7 of last 10 at 20+" and row["why"][1] == "averaging 28.8 points over the last 5"
    assert len(row["why"]) <= 3
    # Combos the card gives no chance for use the model's mean with his own spread; double-doubles combine the
    # per-stat chances of 10+
    pr = engine.scan(board_view([player]), engine.make_ask(NBA, "pr", 34))
    pr_row = (pr["players"] + pr["close"])[0]
    assert pr_row["chance_source"] == "nba+form" and pr_row["projection"] == round(31.2 + 4.6, 1)
    dd = engine.scan(board_view([player]), engine.make_ask(NBA, "dd"))
    dd_row = (dd["players"] + dd["close"])[0]
    p10 = [calls.stat_probabilities(s, SGA_ROW[s], 9.5, 34.3)[0] for s in engine.DOUBLE_PARTS]
    at_least_two = 1 - math.prod(1 - p for p in p10) - sum(
        p * math.prod(1 - q for j, q in enumerate(p10) if j != i) for i, p in enumerate(p10))
    assert dd_row["chance"] == round(at_least_two, 4) and dd_row["chance_source"] == "nba"


# --- candidates, ranking and close --------------------------------------------------------------------------

def form_player(athlete_id: str, name: str, pts: List[float], status: Optional[str] = None, **extra: Any
                ) -> Dict[str, Any]:
    rows = [{"season": "2026-27", "stats": {"pts": value, "reb": 5, "ast": 4}} for value in pts]
    player = {"athlete_id": athlete_id, "name": name, "team": "DAL", "opponent": "OKC", "home": True,
              "position": "G", "event_id": "401909891", "status": status, "projection": None, "rows": rows,
              "props": []}
    player.update(extra)
    return player


def test_candidates_ranking_close_and_status() -> None:
    players = [
        form_player("1", "Star", [31, 28, 35, 30, 27, 33, 29, 26, 34, 30]),
        form_player("2", "Second", [22, 19, 24, 21, 18, 23, 20, 22, 25, 17]),
        form_player("3", "Streaky", [14, 12, 21, 13, 15, 12, 14, 11, 13, 16]),  # reached 20 once: a candidate
        form_player("4", "Bench", [6, 8, 5, 7, 9, 4, 6, 8, 7, 5]),  # never near: not a candidate
        form_player("5", "Hurt", [30, 32, 29, 31, 30, 28, 33, 30, 29, 31], status="Out"),
        form_player("6", "Doubtful", [30, 32, 29, 31, 30, 28, 33, 30, 29, 31], status="Doubtful"),
        form_player("7", "Iffy", [25, 24, 26, 23, 27, 25, 24, 26, 25, 24], status="Questionable"),
        form_player("8", "Gtd", [23, 22, 24, 21, 25, 23, 22, 24, 23, 22], status="Day-To-Day"),
    ]
    result = engine.scan(board_view(players), engine.make_ask(NBA, "pts", 20))
    names = [row["name"] for row in result["players"]]
    assert names == ["Star", "Iffy", "Gtd", "Second"]  # ranked by chance
    assert [row["rank"] for row in result["players"]] == [1, 2, 3, 4]
    chances = [row["chance"] for row in result["players"]]
    assert chances == sorted(chances, reverse=True) and all(row["model"] == "form" for row in result["players"])
    assert all(row["model_label"] == "from recent form" for row in result["players"])
    assert [row["name"] for row in result["close"]] == ["Streaky"] and result["close"][0]["rank"] == 5
    every = result["players"] + result["close"]
    assert not {"Hurt", "Doubtful", "Bench"} & {row["name"] for row in every}  # ruled out or not in range
    iffy = next(row for row in every if row["name"] == "Iffy")
    gtd = next(row for row in every if row["name"] == "Gtd")
    assert (iffy["status"], iffy["status_tag"], gtd["status_tag"]) == ("Questionable", "Q", "GTD")
    assert "loading" not in result and result["more"] == 0
    # n caps the list; the rest that qualified are counted, and close stays the rows just below
    capped = engine.scan(board_view(players), engine.make_ask(NBA, "pts", 20), n=2)
    assert [row["name"] for row in capped["players"]] == ["Star", "Iffy"] and capped["more"] == 2
    assert [row["name"] for row in capped["close"]] == ["Streaky"]
    # Under: the strongest unders among players in range of the number
    unders = engine.scan(board_view(players), engine.make_ask(NBA, "pts", 20, "under"))
    assert unders["players"][0]["name"] == "Streaky" and unders["players"][0]["why"][0].startswith("under 19.5 in 9 of")
    for row in result["players"] + unders["players"]:
        no_banned(row["why"])
    # Each row has the contract's keys
    assert {"rank", "athlete_id", "name", "team", "opponent", "home", "game", "event_id", "start", "status",
            "projection", "chance", "chance_source", "model", "hits", "avg", "why"} <= set(result["players"][0])


def test_yes_asks_use_their_own_bar_and_the_yes_price() -> None:
    attd = [{"market": "player_anytime_td", "kind": "yes", "line": None, "yes_price": Decimal("2.10"),
             "over_price": None, "under_price": None, "label": None}]
    cook = log_rows("football", "nfl_gamelog_4379399")
    shakir = log_rows("football", "nfl_gamelog_4373678")
    view = board_view([
        {"athlete_id": "4379399", "name": "James Cook III", "team": "BUF", "opponent": "BAL", "home": True,
         "position": "RB", "event_id": "401909891", "status": None, "projection": None, "rows": cook, "props": attd},
        {"athlete_id": "4373678", "name": "Khalil Shakir", "team": "BUF", "opponent": "BAL", "home": True,
         "position": "WR", "event_id": "401909891", "status": None, "projection": None, "rows": shakir, "props": []},
    ], league=NFL)
    result = engine.scan(view, engine.make_ask(NFL, "anytime_td"))
    top = result["players"][0]
    assert top["name"] == "James Cook III" and top["chance"] >= engine.YES_BAR
    assert top["edge"] == round(top["chance"] * 2.10 - 1, 4) and top["line"]["yes"]["american"] == "+110"
    assert top["why"][:2] == ["a TD in 5 of last 6", "1.2 TDs a game over the last 5"]
    assert [row["name"] for row in result["close"]] == ["Khalil Shakir"]  # 1 of 6: under the 30% bar


# --- edges and posted lines ---------------------------------------------------------------------------------

def test_edges_only_on_the_matching_price_and_stale_lines_flagged() -> None:
    def prop(kind, line, over=None, under=None, label=None):
        return {"market": "player_points", "kind": kind, "line": line, "over_price": over, "under_price": under,
                "yes_price": None, "label": label}

    def row_for(props, threshold=20, side="over", stale=False):
        view = board_view([nba_player(SGA_ROW, SGA_ROWS, props)])
        view["games"]["401909891"]["props"]["stale"] = stale
        result = engine.scan(view, engine.make_ask(NBA, "pts", threshold, side))
        return (result["players"] + result["close"])[0]

    over = calls.stat_probabilities("pts", 31.2, 19.5, 34.3)[0]
    milestone = row_for([prop("milestone", 19.5, Decimal("1.25"), label="20+")])
    assert milestone["edge"] == round(round(over, 4) * 1.25 - 1, 4)
    matching_ou = row_for([prop("over_under", 19.5, Decimal("1.30"), Decimal("3.40"))])
    assert matching_ou["edge"] == round(round(over, 4) * 1.30 - 1, 4)
    under = row_for([prop("over_under", 30.5, Decimal("1.87"), Decimal("1.95"))], threshold=31, side="under")
    assert under["edge"] == round(under["chance"] * 1.95 - 1, 4)
    other_rung = row_for([prop("over_under", 30.5, Decimal("1.87"), Decimal("1.95"))])  # 30.5 isn't "20+"
    assert "edge" not in other_rung and other_rung["line"]["matches"] is False and other_rung["line"]["line"] == 30.5
    unpriced = row_for([prop("over_under", 19.5)])  # ESPN's DraftKings items often carry no side labels
    assert "edge" not in unpriced and unpriced["line"]["over"] is None
    stale = row_for([prop("milestone", 19.5, Decimal("1.25"), label="20+")], stale=True)
    assert stale["line"]["stale"] is True and stale["line"]["book"] == "DraftKings"
    assert milestone["line"]["stale"] is False


def test_no_number_scans_each_players_posted_line_by_edge() -> None:
    players = [
        form_player("1", "Value", [27] * 10, props=[{"market": "player_points", "kind": "over_under", "line": 22.5,
                                                    "over_price": Decimal("1.91"), "under_price": Decimal("1.91")}]),
        form_player("2", "Fair", [24] * 10, props=[{"market": "player_points", "kind": "over_under", "line": 23.5,
                                                   "over_price": Decimal("1.91"), "under_price": Decimal("1.91")}]),
        form_player("3", "NoLine", [40] * 10),
    ]
    result = engine.scan(board_view(players), engine.make_ask(NBA, "pts", None))
    assert [row["name"] for row in result["players"]] == ["Value", "Fair"]
    assert [row["threshold"] for row in result["players"]] == [23, 24]
    assert result["players"][0]["edge"] > result["players"][1]["edge"]
    assert result["ask"]["label"] == "points overs (posted lines)"


def test_a_partial_board_says_whats_loading() -> None:
    view = board_view([form_player("1", "Star", [31] * 10)], games_ready=3, games_total=10)
    result = engine.scan(view, engine.make_ask(NBA, "pts", 20))
    assert result["loading"] == {"games_ready": 3, "games_total": 10}
    assert result["players"][0]["name"] == "Star"  # what's ready is answered


# --- the boards: ESPN through the shared cache, at most 4 at once -----------------------------------------------

def test_espn_grouped_rosters_are_read() -> None:
    cache = TTLCache()
    espn = EspnClient(cache)
    espn._get = lambda url, params=None: load("nhl_roster_6") if "hockey" in url else load("nfl_roster_2")
    hockey = espn.roster(NHL, "6")
    assert len(hockey) == 29 and {p["group"] for p in hockey} == {"Centers", "Left Wings", "Right Wings", "Defense",
                                                                    "Goalies"}
    draisaitl = next(p for p in hockey if p["id"] == "3114727")
    assert draisaitl == {"id": "3114727", "name": "Leon Draisaitl", "position": "C", "status": "", "group": "Centers"}
    football = EspnClient(TTLCache())
    football._get = lambda url, params=None: load("nfl_roster_2")
    allen = next(p for p in football.roster(NFL, "2") if p["id"] == "3918298")
    assert (allen["position"], allen["group"]) == ("QB", "offense")
    flat = EspnClient(TTLCache())
    flat._get = lambda url, params=None: load("nba_roster_6")
    kyrie = next(p for p in flat.roster(NBA, "6") if p["id"] == "6442")
    assert kyrie == {"id": "6442", "name": "Kyrie Irving", "position": "G", "status": "Day-To-Day"}  # no group key


def test_parse_events_reads_the_scoreboard() -> None:
    games = parse_events(load("nba_scoreboard_20261028")["events"])
    assert len(games) == 12 and games[0]["state"] == "pre"
    dal = next(g for g in games if g["event_id"] == "401909891")
    assert dal["home"] == {"id": "6", "abbr": "DAL", "name": "Dallas Mavericks"} and dal["away"]["abbr"] == "OKC"
    assert dal["start_text"] == "2026-10-29T01:30:00Z"


def test_a_form_board_builds_through_the_cache_four_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    web = Web(monkeypatch, fixture_routes, delay=0.005)
    board = make_board()
    assert board.wait(NHL, "today", 30)
    result = board.scan(NHL, "SOG", 3)
    top = result["players"][0]
    assert top["name"] == "Leon Draisaitl" and top["model"] == "form" and top["hits"]["l10"] == [8, 10]
    assert top["line"]["label"] == "3+" and top["line"]["matches"] and "edge" in top
    assert result["games"] == 4 and "loading" not in result and result["no_games"] is False
    goal = board.scan(NHL, "anytime goal")["players"][0]
    assert goal["name"] == "Leon Draisaitl" and goal["line"]["kind"] == "yes" and goal["why"][0].startswith("a goal in")
    # Every request went to ESPN, inside a cache loader with a policy kind; never The Odds API; at most 4 at once
    assert web.hosts() <= ESPN_HOSTS
    assert web.calls and all(call["kind"] in POLICY for call in web.calls)
    assert {call["kind"] for call in web.calls} <= {"scoreboard", "summary", "roster", "gamelog", "props"}
    assert 1 < web.peak <= 4
    # A second scan is warm: nothing more is fetched
    before = len(web.calls)
    board.scan(NHL, "points", 2)
    assert len(web.calls) == before
    # The NFL: Buffalo's grouped roster, its props players as the pool; the 1 November board
    assert board.wait(NFL, "2026-11-01", 30)
    nfl = board.snapshot(NFL, "2026-11-01")
    buf = [p for p in nfl["players"] if p["team"] == "BUF"]
    assert {p["position"] for p in buf} <= {"QB", "RB", "WR", "TE", "FB"} and buf
    assert all(p["status"] not in ("Out", "Injured Reserve") for p in buf)


def test_a_partial_board_answers_at_once_with_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def slow(url, params):
        if "/gamelog" in url and "3114727" not in url:
            release.wait(10)
        return fixture_routes(url, params)

    Web(monkeypatch, slow)
    board = make_board(wait=0.3)
    started = time.monotonic()
    result = board.scan(NHL, "sog", 3)
    assert time.monotonic() - started < 3
    assert result["loading"]["games_total"] == 4 and result["loading"]["games_ready"] < 4
    release.set()
    assert board.wait(NHL, "today", 30)
    assert "loading" not in board.scan(NHL, "sog", 3)


def test_props_and_injuries_refresh_until_tip_off(monkeypatch: pytest.MonkeyPatch) -> None:
    Web(monkeypatch, fixture_routes)
    clock = [NOW]
    board = make_board(now=clock[0])
    board._now = lambda: clock[0]
    reads: List[str] = []
    props, summary = board.players.props, board.espn.summary
    board.players.props = lambda league, event_id: reads.append(f"props {event_id}") or props(league, event_id)
    board.espn.summary = lambda league, event_id: reads.append(f"summary {event_id}") or summary(league, event_id)
    assert board.wait(NHL, "today", 30)
    first = len(reads)
    assert first == 8  # each of the 4 games' props and injury report, once
    board.snapshot(NHL)
    assert len(reads) == first  # not due yet
    clock[0] += engine_minutes(31)
    board.snapshot(NHL)  # due: read again in the background (through the cache, which keeps its own TTLs)
    deadline = time.monotonic() + 10
    while len(reads) < 2 * first and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sorted(reads[first:]) == sorted(reads[:first])
    # After tip-off a game drops out of the board (and is never refreshed)
    clock[0] = parse_events(load("nhl_scoreboard_20261028")["events"])[-1]["start"] + 60
    view = board.snapshot(NHL)
    assert view["games_total"] == 0 and view["games_started"] == 4 and view["players"] == []


def engine_minutes(minutes: float) -> float:
    return minutes * 60.0


def test_nba_board_uses_the_model_only_when_the_odds_cost_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the odds board fresh, the NBA game's rows come from service.detail (the player card's own); stale odds
    are never loaded for it (The Odds API's quota is untouched): the rows fall back to the form model."""
    Web(monkeypatch, fixture_routes)

    class Odds:
        odds_ttl = 7200
        quota = {"remaining": 400}
        reserve = 50

        def __init__(self, fetched_at):
            self.fetched_at = fetched_at
            self.board_reads = 0

        def odds_loaded_at(self, league):
            return self.fetched_at

        def saved_games(self, league):
            return (self.fetched_at, [{"id": "odds-dal", "home": "Dallas Mavericks", "away": "Oklahoma City Thunder",
                                       "commence_time": "2026-10-29T01:30:00Z"}])

        def league_games(self, league, refresh=False):
            self.board_reads += 1
            raise AssertionError("the scan never loads an odds board")

    asked = []

    def detail(league, odds_id):
        asked.append(odds_id)
        return {}, {"teams": {}}, {"4278073": {**SGA_ROW, "status": ""}}

    fresh = Odds(time.time() - 600)
    board = make_board(detail=detail, odds=fresh)
    board.espn.find_game = lambda league, home, away, start: {"event_id": "401909891", "home_id": "6", "away_id": "25"}
    assert board.wait(NBA, "today", 30)
    rows = board.scan(NBA, "points", 20)["players"]
    sga = next(row for row in rows if row["athlete_id"] == "4278073")
    assert asked == ["odds-dal"] and sga["model"] == "nba"
    assert sga["chance"] == round(calls.stat_probabilities("pts", 31.2, 19.5, 34.3)[0], 4)
    assert sga["hits"]["l10"] == [7, 10]  # his game log, through PlayerData
    stale = Odds(time.time() - 3 * 3600)
    asked.clear()
    board2 = make_board(detail=detail, odds=stale)
    assert board2.wait(NBA, "today", 30)
    assert asked == [] and stale.board_reads == 0
    form_rows = board2.scan(NBA, "points", 20)
    assert all(row["model"] == "form" for row in form_rows["players"] + form_rows["close"])


def test_board_for_shares_one_board_and_the_slate_prewarms_it() -> None:
    class Svc:
        espn = None
        cache = None

    service = Svc()
    board = board_for(service, players=object())
    assert board_for(service) is board and service.player_board is board
    board.close()

    asked = []

    class FakeBoard:
        def prewarm(self, rows):
            asked.append(rows)

    clock = [1000.0]
    slate = SlateBuilder(odds=None, espn=None, player_board=FakeBoard(), now=lambda: clock[0], players_every=300)
    games = [{"league": NBA, "start": "2026-10-28T23:00:00Z", "status": "pre"},
             {"league": NHL, "start": "2026-10-28T22:00:00Z", "status": "live"}]
    slate._prewarm_players(games, clock[0])
    slate._prewarm_players(games, clock[0] + 60)  # throttled
    assert asked == [[{"league": NBA, "start": "2026-10-28T23:00:00Z", "status": "pre"}]]
    slate._prewarm_players(games, clock[0] + 301)
    assert len(asked) == 2
    # Without a board of its own, the slate asks the service's Lab for the process's board (made on first use)

    class Lab:
        made = 0

        @property
        def player_board(self):
            Lab.made += 1
            return FakeBoard()

    class WithLab:
        lab = Lab()

    asked.clear()
    from_lab = SlateBuilder(odds=None, espn=None, service=WithLab(), now=lambda: clock[0])
    from_lab._prewarm_players(games, clock[0])
    from_lab._prewarm_players(games, clock[0] + 10)  # throttled before the Lab is asked again
    assert len(asked) == 1 and Lab.made == 1
    SlateBuilder(odds=None, espn=None, service=object(), now=lambda: clock[0])._prewarm_players(games, clock[0])
    assert len(asked) == 1  # no board anywhere: nothing happens


def test_prewarm_starts_game_day_boards_nba_first(monkeypatch: pytest.MonkeyPatch) -> None:
    Web(monkeypatch, fixture_routes)
    board = make_board()
    soon = "2026-10-28T23:00:00Z"
    rows = [{"league": NHL, "start": soon, "status": "pre"},
            {"league": NBA, "start": soon, "status": "pre"},
            {"league": "soccer_epl", "start": soon, "status": "pre"},  # built on its first scan
            {"league": NFL, "start": "2026-11-01T18:00:00Z", "status": "pre"}]  # 4 days off
    assert board.prewarm(rows) == ["basketball_nba:2026-10-28"]  # the NBA first, one cold build at a time
    assert board.wait(NBA, "today", 30)
    assert board.prewarm(rows) == ["basketball_nba:2026-10-28", "icehockey_nhl:2026-10-28"]  # then the NHL
    assert board.wait(NHL, "today", 30)
    assert [entry["league"] for entry in board.status()] == [NBA, NHL]


# --- the recorded slate: a cold 10-game NBA build and a warm scan ----------------------------------------------

LATENCY = load("latency")["seconds"]  # seconds per request while capturing, through this container's proxy
RECORDED = sorted(LATENCY.values())[len(LATENCY) // 2]  # the median
TEST_DELAY = 0.02  # the replay's per-request delay


class NbaWorld:
    """Ten of the 28 October 2026 NBA games, replayed from the captured responses with ids swapped: every team's
    roster (Dallas's and Oklahoma City's as captured, the rest from Dallas's with new ids), each player's season
    averages and game logs (Shai Gilgeous-Alexander's, scaled), the summary's injury report and DraftKings' props
    (the captured game's, re-pointed at each game's leading players). The real DashboardService projects them."""

    def __init__(self) -> None:
        self.events = load("nba_scoreboard_20261028")["events"][:10]
        self.teams: Dict[str, Dict[str, Any]] = {}
        for event in self.events:
            for side in event["competitions"][0]["competitors"]:
                self.teams[side["team"]["id"]] = side["team"]
        self.rosters: Dict[str, List[Dict[str, Any]]] = {}
        self.factor: Dict[str, float] = {}
        template = load("nba_roster_6")["athletes"]
        captured = {"6": template, "25": load("nba_roster_25")["athletes"]}
        for team_id, team in self.teams.items():
            athletes = copy.deepcopy(captured.get(team_id) or template)
            if team_id not in captured:
                for index, athlete in enumerate(athletes):
                    athlete["id"] = str(7_000_000 + int(team_id) * 100 + index)
                    athlete["displayName"] = f"{team['abbreviation']} Player {index + 1}"
            self.rosters[team_id] = athletes
            healthy = [a for a in athletes if not a.get("injuries")]
            for rank, athlete in enumerate(healthy + [a for a in athletes if a.get("injuries")]):
                self.factor[athlete["id"]] = max(0.15, 1.0 - 0.09 * rank)
        today = datetime.datetime.now(ZoneInfo("America/New_York")).date()
        self.label = nba_model.season_label(today)
        start = int(self.label[:4])
        self.previous = f"{start - 1}-{start % 100:02d}"
        self.current_log = load("nba_gamelog_4278073")
        self.previous_log = load("nba_gamelog_4278073_2026")
        self.props = load("nba_props_401909891")

    def game(self, event_id: str) -> Dict[str, Any]:
        return next(e for e in self.events if e["id"] == event_id)

    def leaders(self, event: Dict[str, Any]) -> List[str]:
        ids = []
        for side in event["competitions"][0]["competitors"]:
            healthy = [a["id"] for a in self.rosters[side["team"]["id"]] if not a.get("injuries")]
            ids += healthy[:2]
        return ids

    def stats(self, athlete_id: str) -> Dict[str, Any]:
        f = self.factor.get(athlete_id, 0.3)
        labels = ["GP", "GS", "MIN", "FG", "FG%", "3PT", "3P%", "FT", "FT%", "OR", "DR", "REB", "AST", "BLK", "STL",
                  "PF", "TO", "PTS"]

        def line(gp, scale):
            return [str(gp), str(gp), f"{34 * f * scale:.1f}", "9.0-18.0", "50.0", f"{2.0 * f:.1f}-5.0", "36.0",
                    "6.0-7.0", "85.0", "0.6", "3.7", f"{6.0 * f:.1f}", f"{5.5 * f:.1f}", f"{0.8 * f:.1f}",
                    f"{1.3 * f:.1f}", "2.0", f"{2.4 * f:.1f}", f"{30.0 * f * scale:.1f}"]

        return {"categories": [{"name": "averages", "labels": labels, "statistics": [
            {"season": {"year": 0, "displayName": self.previous}, "stats": line(70, 1.0)},
            {"season": {"year": 0, "displayName": self.label}, "stats": line(4, 1.02)}]}]}

    def gamelog(self, athlete_id: str, season: Optional[str]) -> Dict[str, Any]:
        f = self.factor.get(athlete_id, 0.3)
        body = copy.deepcopy(self.previous_log if season else self.current_log)
        labels = body["labels"]
        for stype in body["seasonTypes"]:
            for category in stype["categories"]:
                for event in category["events"]:
                    stats = event["stats"]
                    for name in ("PTS", "REB", "AST", "MIN"):
                        i = labels.index(name)
                        stats[i] = str(round(float(stats[i]) * f))
        return body

    def summary(self, event_id: str) -> Dict[str, Any]:
        event = self.game(event_id)
        injuries = []
        for side in event["competitions"][0]["competitors"]:
            team_id = side["team"]["id"]
            injuries.append({"team": {"id": team_id}, "injuries": [
                {"status": a["injuries"][0]["status"], "athlete": {"id": a["id"], "displayName": a["displayName"]}}
                for a in self.rosters[team_id] if a.get("injuries")]})
        return {"header": {"id": event_id, "competitions": event["competitions"]}, "injuries": injuries}

    def propbets(self, event_id: str) -> Dict[str, Any]:
        body = copy.deepcopy(self.props)
        mapping = dict(zip(["4278073", "5041939", "6442", "4222252"], self.leaders(self.game(event_id))))
        for item in body["items"]:
            ref = item["athlete"]["$ref"]
            old = ref.split("/athletes/")[1].split("?")[0]
            item["athlete"]["$ref"] = ref.replace(f"/athletes/{old}", f"/athletes/{mapping.get(old, old)}")
        return body

    def standings(self) -> Dict[str, Any]:
        entries = []
        for team_id, team in self.teams.items():
            strength = int(team_id) % 7
            entries.append({"team": {"id": team_id, "displayName": team["displayName"]}, "stats": [
                {"name": "wins", "value": 2 + strength % 3}, {"name": "losses", "value": 2},
                {"name": "avgPointsFor", "value": 112.0 + strength}, {"name": "avgPointsAgainst", "value": 114.0 - strength}]})
        return {"children": [{"standings": {"entries": entries}}]}

    def routes(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        if url.endswith("/scoreboard"):
            return {"events": self.events} if params.get("dates") == "20261028" else {"events": []}
        if url.endswith("/summary"):
            return self.summary(str(params.get("event")))
        if url.endswith("/standings"):
            return self.standings()
        match = re.search(r"/teams/(\d+)/roster$", url)
        if match:
            return {"athletes": self.rosters.get(match.group(1), [])}
        match = re.search(r"/athletes/(\d+)/(gamelog|stats)$", url)
        if match:
            if match.group(2) == "stats":
                return self.stats(match.group(1))
            return self.gamelog(match.group(1), params.get("season"))
        match = re.search(r"/events/(\d+)/competitions/\d+/odds/100/propBets$", url)
        if match:
            return self.propbets(match.group(1))
        return {}

    def odds_games(self) -> List[Dict[str, Any]]:
        from telegram_betting_bot_simple.data.odds_service import normalize_event
        games = []
        for event in self.events:
            sides = {c["homeAway"]: c["team"]["displayName"] for c in event["competitions"][0]["competitors"]}
            bookmakers = [{"key": "draftkings", "title": "DraftKings", "markets": [
                {"key": "h2h", "outcomes": [{"name": sides["home"], "price": 1.80}, {"name": sides["away"], "price": 2.05}]},
                {"key": "spreads", "outcomes": [{"name": sides["home"], "price": 1.91, "point": -2.5},
                                                {"name": sides["away"], "price": 1.91, "point": 2.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 226.5},
                                               {"name": "Under", "price": 1.91, "point": 226.5}]}]}]
            games.append(normalize_event({"id": f"odds-{event['id']}", "sport_key": NBA, "sport_title": "NBA",
                                          "commence_time": event["date"].replace("Z", ":00Z")
                                          if len(event["date"]) == 17 else event["date"],
                                          "home_team": sides["home"], "away_team": sides["away"],
                                          "bookmakers": bookmakers}))
        return games


class WorldOdds:
    """The odds service's no-credit surface: a board saved ten minutes ago (league_games serves it from memory)."""

    odds_ttl = 7200
    quota = {"remaining": 400, "used": 100}
    reserve = 50
    store = None

    def __init__(self, games: List[Dict[str, Any]]) -> None:
        self.games = games
        self.at = time.time() - 600

    def odds_loaded_at(self, league):
        return self.at if league == NBA else None

    def saved_games(self, league):
        return (self.at, copy.deepcopy(self.games)) if league == NBA else None

    def league_games(self, league, refresh=False):
        assert not refresh
        return copy.deepcopy(self.games) if league == NBA else []

    def consensus_inputs(self, league):
        return {}

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]


def _cold_build(monkeypatch: pytest.MonkeyPatch, world: "NbaWorld", delay: float):
    from telegram_betting_bot_simple.dashboard.service import DashboardService

    web = Web(monkeypatch, world.routes, delay=delay)
    cache = TTLCache(name="recorded-slate")
    espn = EspnClient(cache, clock=lambda: NOW)
    service = DashboardService(WorldOdds(world.odds_games()), espn, ZoneInfo("America/New_York"), cache=cache)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    board = PlayerBoard(service, players, cache, now=lambda: NOW)
    started = time.monotonic()
    assert board.wait(NBA, "today", 120)
    return board, web, time.monotonic() - started


def test_recorded_slate_cold_build_and_warm_scan(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A cold 10-game NBA slate builds 4-wide in under 3 minutes at the latency recorded while capturing, and a warm
    scan answers in under 300 ms. The slate is built twice from cold: with no delay (the work that isn't waiting on
    ESPN) and with TEST_DELAY per request; the time at the recorded latency is the first plus the waiting scaled by
    RECORDED / TEST_DELAY."""
    world = NbaWorld()
    _, _, overhead = _cold_build(monkeypatch, world, 0.0)
    board, web, cold = _cold_build(monkeypatch, world, TEST_DELAY)
    requests_made = len(web.calls)
    projected = overhead + max(0.0, cold - overhead) * RECORDED / TEST_DELAY
    result = board.scan(NBA, "points", 20)
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        warm = board.scan(NBA, "points", 20)
        timings.append(time.perf_counter() - t0)
    warm_ms = 1000 * sorted(timings)[len(timings) // 2]
    with capsys.disabled():
        print(f"\nrecorded slate: 10 NBA games, {requests_made} ESPN requests, peak {web.peak} at once; cold build "
              f"{overhead:.2f}s of work + {cold - overhead:.2f}s waiting at {TEST_DELAY * 1000:.0f} ms/request -> "
              f"{projected:.0f}s at the recorded {RECORDED * 1000:.0f} ms/request; warm scan {warm_ms:.1f} ms "
              f"(median of 20)")
    assert web.peak <= 4
    assert web.hosts() <= ESPN_HOSTS
    assert projected < 180, projected
    assert warm_ms < 300, warm_ms
    # The real NBA model projected the rotation (no game's rows fell back to the form model)
    rows = result["players"] + result["close"]
    assert rows and all(row["model"] == "nba" for row in rows)
    assert result["games"] == 10 and "loading" not in result and len({row["event_id"] for row in rows}) > 3
    assert warm["players"] == result["players"]
    assert not any(call["url"].startswith("https://api.the-odds-api.com") for call in web.calls)
    no_banned([reason for row in rows for reason in row["why"]])


def test_reasons_read_well_without_a_log_for_one_a_game_and_never_carry_a_price() -> None:
    """A projection with no game log behind it still says why it's there; "1 TD a game", not "1 TDs"; a posted line's
    price is the row's own (shown in the person's odds format), never written into a reason."""
    spec = engine.STATS[NBA]["pts"]
    empty = {"threshold": 20, "hits": {"l5": [0, 0], "l10": [0, 0], "season": [0, 0]},
             "avg": {"l5": None, "l10": None, "season": None}}
    assert engine.why(empty, spec, "over") == ["no games logged this season or last"]
    td = engine.STATS[NFL]["anytime_td"]
    one = {"threshold": 1, "hits": {"l5": [3, 5], "l10": [5, 10], "season": [5, 10]},
           "avg": {"l5": 1.0, "l10": 0.6, "season": 0.6},
           "line": {"kind": "yes", "line": None, "yes": {"decimal": 2.1, "american": "+110"}, "matches": True}}
    assert engine.why(one, td, "over") == ["a TD in 5 of last 10", "1 TD a game over the last 5", "anytime TD posted"]
    milestone = dict(empty, hits={"l5": [4, 5], "l10": [8, 10], "season": [8, 10]},
                     avg={"l5": 26.0, "l10": 25.0, "season": 25.0},
                     line={"kind": "milestone", "line": 19.5, "label": "20+", "matches": True,
                           "over": {"decimal": 1.17, "american": "-588"}})
    reasons = engine.why(milestone, spec, "over")
    assert reasons[-1] == "20+ posted" and not any("588" in reason for reason in reasons)
