"""Stage 2 sources (data/markets): Polymarket, Kalshi, ratings and aliases, on trimmed live captures.

The fixtures in tests/fixtures/markets/ are small cuts of answers captured live on 2026-10-27 (field names and
values unchanged; only events, markets and fields the parsers don't read were dropped). No test touches the
network: every client gets a fake fetcher, and requests.get is made to fail loudly.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics

import pytest

from telegram_betting_bot_simple.data.markets import (COVERAGE, TARGET_LEAGUES, Kalshi, Polymarket, Ratings,
                                                      make_quote, match_events, missing, normalise, team_key)
from telegram_betting_bot_simple.data.markets import aliases, common, kalshi as kalshi_mod
from telegram_betting_bot_simple.data.markets import polymarket as poly_mod, ratings as ratings_mod
from telegram_betting_bot_simple.data.markets.common import HttpError, Keeper, complement, is_thin, parse_time
from telegram_betting_bot_simple.data.cache import key_text
from telegram_betting_bot_simple.data.store import Store

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "markets")
NOW = parse_time("2026-10-27T01:15:00Z")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read() if name.endswith((".html", ".tsv")) else json.load(fh)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(common.requests, "get", refuse)


class Clock:
    """A fake wall clock whose sleep moves time on."""

    def __init__(self, now=NOW):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Fetch:
    """A fake HTTP getter: answers by URL substring, counts calls, records when they happened."""

    def __init__(self, routes, clock=None):
        self.routes = routes  # list of (substring, answer or callable(params) -> (status, body))
        self.calls = []
        self.clock = clock

    def __call__(self, url, params=None, text=False):
        self.calls.append((url, dict(params or {}), self.clock() if self.clock else None))
        for part, answer in self.routes:
            if part in url:
                result = answer(params or {}) if callable(answer) else answer
                if isinstance(result, Exception):
                    raise result
                return result
        return 404, None


# --- the thin rule, quotes and normalisation ---------------------------------------------------------------

def test_thin_rule():
    assert not is_thin(0.42, 0.48, None)  # a 6-cent spread is still tradable ...
    assert is_thin(0.42, 0.49, None)  # ... 7 cents is thin
    assert is_thin(None, 0.40, None) and is_thin(0.40, None, 5000)  # one side missing
    assert is_thin(0.50, 0.51, 999.0) and not is_thin(0.50, 0.51, 1000.0)  # liquidity under $1,000
    assert not is_thin(0.50, 0.51, None)  # Kalshi reports no liquidity: judged by spread alone


def test_quote_mid_last_and_fallback():
    assert make_quote(0.42, 0.43, 0.5, 26000, 100)["p"] == pytest.approx(0.425)  # midpoint when both exist
    assert make_quote(None, 0.04, 0.03)["p"] == 0.03  # else the last price
    quote = make_quote(None, None, None, None, None, "0.5")  # else the venue's shown price, thin
    assert quote["p"] == 0.5 and quote["thin"]
    assert make_quote(0.0, 1.0, 0.0) is None  # 0 and 1 are "no order", not prices
    crossed = make_quote(0.60, 0.55, 0.57)  # a crossed book: the bid is dropped, the last trade is the price
    assert crossed["bid"] is None and crossed["p"] == 0.57 and crossed["thin"]
    assert make_quote(0.60, 0.55, None) is None  # an ask alone is no chance
    other = complement(make_quote(0.18, 0.19, 0.19, 28246.66))
    assert other["p"] == pytest.approx(0.815) and other["bid"] == pytest.approx(0.81) and other["ask"] == pytest.approx(0.82)


def test_normalise_scales_exclusive_outcomes_to_one():
    quotes = {"home": make_quote(0.53, 0.54), "away": make_quote(0.23, 0.24), "draw": make_quote(0.24, 0.25)}
    out = normalise(quotes)
    assert sum(q["p"] for q in out.values()) == pytest.approx(1.0)
    assert out["home"]["p"] == pytest.approx(0.535 / 1.015)  # mids 0.535 + 0.235 + 0.245 = 1.015
    assert out["home"]["bid"] == 0.53  # the venue's own numbers are kept


# --- Polymarket ------------------------------------------------------------------------------------------

def test_polymarket_moneylines_two_way():
    events = {e["id"]: e for e in poly_mod.parse_moneylines("basketball_nba", fixture("polymarket_nba_moneyline.json"), NOW)}
    game = events["nba-bkn-bos-2026-10-27"]
    assert (game["home"], game["away"], game["home_abbr"], game["away_abbr"]) == ("Celtics", "Nets", "bos", "bkn")
    assert (game["home_key"], game["away_key"]) == ("BOS", "BKN")
    h2h = game["markets"]["h2h"]
    assert h2h["away"]["p"] == pytest.approx(0.185) and h2h["home"]["p"] == pytest.approx(0.815)  # mid 0.18/0.19
    assert h2h["home"]["p"] + h2h["away"]["p"] == pytest.approx(1.0) and "draw" not in h2h
    assert h2h["home"]["liquidity"] == pytest.approx(28230.24, abs=0.01) and not h2h["home"]["thin"]
    assert game["start"] == "2026-10-27T23:30:00Z" and game["date"] == "2026-10-27"
    assert game["url"] == "https://polymarket.com/event/nba-bkn-bos-2026-10-27" and game["as_of"] == "2026-10-27T01:15:00Z"
    assert game["source"] == "polymarket" and game["league"] == "basketball_nba"
    assert events["nba-min-cle-2026-10-26"]["live"] is True and events["nba-was-cle-2026-10-28"]["away_key"] == "WSH"


def test_polymarket_soccer_three_way_is_normalised():
    events = {e["id"]: e for e in poly_mod.parse_moneylines("soccer_epl", fixture("polymarket_epl_moneyline.json"), NOW)}
    game = events["epl-che-mun-2026-10-31"]
    assert (game["home"], game["away"]) == ("Chelsea FC", "Manchester United FC")  # soccer lists home first
    h2h = game["markets"]["h2h"]
    assert set(h2h) == {"home", "away", "draw"}
    assert sum(q["p"] for q in h2h.values()) == pytest.approx(1.0, abs=1e-3)
    total = 0.535 + 0.245 + 0.225  # the three Yes midpoints
    assert h2h["home"]["p"] == pytest.approx(0.535 / total, abs=1e-4)
    assert h2h["draw"]["p"] == pytest.approx(0.245 / total, abs=1e-4)


def test_polymarket_leaves_out_futures_doubles_and_other_tours():
    assert poly_mod.parse_moneylines("americanfootball_nfl", fixture("polymarket_futures.json"), NOW) == []
    assert poly_mod.parse_moneylines("tennis_atp", fixture("polymarket_futures.json"), NOW) == []
    tennis = fixture("polymarket_tennis_moneyline.json")
    atp = poly_mod.parse_moneylines("tennis_atp", tennis, NOW)
    wta = poly_mod.parse_moneylines("tennis_wta", tennis, NOW)
    assert [e["id"] for e in atp] == ["atp-carabel-coria-2026-10-26"]  # no ITF, no doubles
    assert [e["id"] for e in wta] == ["wta-fernand-sakatsu-2026-10-26"]
    assert atp[0]["home"] == "Camilo Ugo Carabelli" and atp[0]["start"] == "2026-10-27T17:30:00Z"
    fights = poly_mod.parse_moneylines("mma_mixed_martial_arts", fixture("polymarket_ufc_moneyline.json"), NOW)
    assert {e["home"] for e in fights} == {"Renato Moicano", "Joe Kropschot"}


def test_polymarket_old_open_games_are_left_behind():
    later = NOW + 86400  # the 26 October games started more than 12 hours before this
    ids = [e["id"] for e in poly_mod.parse_moneylines("basketball_nba", fixture("polymarket_nba_moneyline.json"), later)]
    assert "nba-min-cle-2026-10-26" not in ids and "nba-was-cle-2026-10-28" in ids


def test_polymarket_main_lines():
    lines = poly_mod.parse_lines("basketball_nba", fixture("polymarket_nba_lines.json"))
    spread, total = lines["nba-chi-bos-2026-10-26"]["spreads"], lines["nba-chi-bos-2026-10-26"]["totals"]
    # closest to 50/50 among the lines that aren't thin: Celtics -17.5 at 0.51/0.55, O/U 237.5 at 0.51/0.53
    assert spread["point"] == -17.5 and spread["home"]["p"] == pytest.approx(0.53) and spread["away"]["p"] == pytest.approx(0.47)
    assert total["point"] == 237.5 and total["over"]["p"] == pytest.approx(0.52) and total["under"]["p"] == pytest.approx(0.48)
    thin = lines["nba-ind-mia-2026-10-30"]["spreads"]  # a 0.01/0.71 book: shown, labelled thin
    assert thin["home"]["thin"] and thin["point"] == -3.5
    epl = poly_mod.parse_lines("soccer_epl", fixture("polymarket_epl_lines.json"))  # under "...-more-markets"
    assert set(epl) == {"epl-ast-ful-2026-10-31"}
    assert epl["epl-ast-ful-2026-10-31"]["spreads"]["point"] == -1.5
    assert epl["epl-ast-ful-2026-10-31"]["totals"]["point"] == 2.5


def polymarket_routes(moneylines, lines=()):
    def answer(params):
        if params.get("offset"):
            return 200, []
        kind = params.get("sports_market_types")
        return 200, list(moneylines if kind == "moneyline" else lines)
    return [("gamma-api.polymarket.com/markets", answer)]


def test_polymarket_client_merges_lines_and_caches():
    clock = Clock()
    fetch = Fetch(polymarket_routes(fixture("polymarket_epl_moneyline.json"), fixture("polymarket_epl_lines.json")), clock)
    source = Polymarket(fetch=fetch, clock=clock, lines=True)
    events = source.events("soccer_epl")
    assert {e["id"] for e in events} == {"epl-che-mun-2026-10-31", "epl-liv-ars-2026-11-01", "epl-ast-ful-2026-10-31"}
    villa = next(e for e in events if e["id"] == "epl-ast-ful-2026-10-31")
    assert villa["markets"]["spreads"]["point"] == -1.5 and villa["markets"]["totals"]["point"] == 2.5  # lines merged
    assert "spreads" not in next(e for e in events if e["id"] == "epl-che-mun-2026-10-31")["markets"]
    url, params, _ = fetch.calls[0]
    assert params["tag_id"] == 306 and params["sports_market_types"] == "moneyline" and params["closed"] == "false"
    assert params["end_date_min"] == "2026-10-26T01:15:00Z" and params["limit"] == 100
    assert fetch.calls[1][1]["sports_market_types"] == ["spreads", "totals"] and fetch.calls[1][1]["liquidity_num_min"] == 1000
    calls = len(fetch.calls)
    clock.now += 4 * 60
    assert source.events("soccer_epl") == events and len(fetch.calls) == calls  # within 5 minutes: cached
    clock.now += 2 * 60
    latest = source.events("soccer_epl")
    assert len(fetch.calls) == calls + 1  # moneylines again; lines keep their 15 minutes
    assert {e["as_of"] for e in latest} == {"2026-10-27T01:21:00Z"}
    assert next(e for e in latest if e["id"] == "epl-ast-ful-2026-10-31")["markets"]["spreads"] == villa["markets"]["spreads"]
    assert source.events("soccer_bogus_league") == [] and source.cached_events("soccer_epl") == latest


def test_polymarket_fetches_no_lines_by_default_and_tennis_only_liquid_matches(caplog):
    """Nothing reads a venue's spreads or totals before Stage 6: by default only the moneylines are downloaded
    (the lines were over half the bytes). Tennis asks only for matches with $1,000 of liquidity (most of the tag is
    ITF), and a fetch that fills every page is logged as possibly cut short."""
    clock = Clock()
    fetch = Fetch(polymarket_routes(fixture("polymarket_epl_moneyline.json"), fixture("polymarket_epl_lines.json")), clock)
    events = Polymarket(fetch=fetch, clock=clock).events("soccer_epl")
    assert len(fetch.calls) == 1 and fetch.calls[0][1]["sports_market_types"] == "moneyline"
    assert all("spreads" not in e["markets"] for e in events) and "liquidity_num_min" not in fetch.calls[0][1]
    tennis = Fetch(polymarket_routes(fixture("polymarket_tennis_moneyline.json")), clock)
    Polymarket(fetch=tennis, clock=clock).events("tennis_atp")
    assert tennis.calls[0][1]["liquidity_num_min"] == poly_mod.TENNIS_LIQUIDITY == 1000
    full = [{"slug": f"x-{n}"} for n in range(poly_mod.PAGE)]
    pages = Fetch([("gamma-api.polymarket.com/markets", lambda params: (200, full))], clock)
    source = Polymarket(fetch=pages, clock=clock)
    with caplog.at_level("WARNING"):
        assert source.events("basketball_nba") == []
        clock.now += 6 * 60
        source.events("basketball_nba")
    assert len(pages.calls) == 2 * poly_mod.MAX_PAGES
    assert caplog.text.count("filled all 8 pages") == 1  # once an hour per tag


def test_polymarket_live_league_refreshes_every_minute():
    clock = Clock()
    fetch = Fetch(polymarket_routes(fixture("polymarket_nba_moneyline.json")), clock)
    source = Polymarket(fetch=fetch, clock=clock)
    source.events("basketball_nba")  # min-cle is live
    calls = len(fetch.calls)
    clock.now += 61
    source.events("basketball_nba")
    assert len(fetch.calls) > calls


def test_tennis_tours_share_one_call():
    clock = Clock()
    fetch = Fetch(polymarket_routes(fixture("polymarket_tennis_moneyline.json")), clock)
    source = Polymarket(fetch=fetch, clock=clock)
    assert [e["id"] for e in source.events("tennis_atp_paris_masters")] == ["atp-carabel-coria-2026-10-26"]
    assert [e["id"] for e in source.events("tennis_wta")] == ["wta-fernand-sakatsu-2026-10-26"]
    assert len(fetch.calls) == 1 and fetch.calls[0][1]["tag_id"] == 864  # one moneyline call, no lines call


# --- Kalshi ---------------------------------------------------------------------------------------------

def test_kalshi_games_orientation_start_and_normalisation():
    events = {e["id"]: e for e in kalshi_mod.parse_games("basketball_nba", fixture("kalshi_nba_game.json"), NOW)}
    game = events["KXNBAGAME-26OCT26MINCLE"]
    assert (game["away"], game["home"], game["away_abbr"], game["home_abbr"]) == ("Minnesota", "Cleveland", "MIN", "CLE")
    assert game["start"] == "2026-10-26T23:00:00Z"  # the milestone's start ...
    raw = next(e for e in fixture("kalshi_nba_game.json")["events"] if e["event_ticker"] == "KXNBAGAME-26OCT26MINCLE")
    assert raw["markets"][0]["occurrence_datetime"] == "2026-10-27T02:00:00Z"  # ... never occurrence_datetime
    assert game["live"] is True and game["date"] == "2026-10-26"
    h2h = game["markets"]["h2h"]
    assert h2h["home"]["p"] + h2h["away"]["p"] == pytest.approx(1.0)
    mids = {m["ticker"][-3:]: (float(m["yes_bid_dollars"]) + float(m["yes_ask_dollars"])) / 2 for m in raw["markets"]}
    assert h2h["away"]["p"] == pytest.approx(mids["MIN"] / (mids["MIN"] + mids["CLE"]), abs=1e-4)
    assert h2h["home"]["liquidity"] is None and h2h["home"]["volume"] > 0
    no_milestone = events["KXNBAGAME-26OCT27BKNBOS"]
    assert no_milestone["start"] is None and no_milestone["date"] == "2026-10-27"  # no real start: left as None
    assert events["KXNBAGAME-26OCT27PORLAL"]["home_key"] == "LAL"  # "Los Angeles L"
    assert game["url"] == "https://kalshi.com/markets/kxnbagame"


def test_kalshi_soccer_tie_and_home_first():
    events = {e["id"]: e for e in kalshi_mod.parse_games("soccer_epl", fixture("kalshi_epl_game.json"), NOW)}
    game = events["KXEPLGAME-26OCT31CFCMUN"]
    assert (game["home"], game["away"], game["home_abbr"]) == ("Chelsea", "Manchester United", "CFC")
    assert set(game["markets"]["h2h"]) == {"home", "away", "draw"}
    assert sum(q["p"] for q in game["markets"]["h2h"].values()) == pytest.approx(1.0, abs=1e-3)


def test_kalshi_ticker_with_start_time_and_other_sports():
    mlb = kalshi_mod.parse_games("baseball_mlb", fixture("kalshi_mlb_game.json"), NOW)
    assert [(e["id"], e["away_abbr"], e["home_abbr"], e["date"]) for e in mlb] == [
        ("KXMLBGAME-26OCT262000MILTB", "MIL", "TB", "2026-10-26"),
        ("KXMLBGAME-26OCT272000MILTB", "MIL", "TB", "2026-10-27")]
    nfl = {e["id"]: e for e in kalshi_mod.parse_games("americanfootball_nfl", fixture("kalshi_nfl_game.json"), NOW)}
    assert nfl["KXNFLGAME-26NOV01INDJAC"]["home_key"] == "JAX"  # Kalshi's JAC is ESPN's JAX
    assert nfl["KXNFLGAME-26NOV01LACLAR"]["home_key"] == "LAR"
    atp = kalshi_mod.parse_games("tennis_atp_paris_masters", fixture("kalshi_atp_match.json"), NOW)
    assert {(e["home"], e["away"]) for e in atp} == {("Flavio Cobolli", "Grigor Dimitrov"), ("Karen Khachanov", "Carlos Alcaraz")}
    ufc = kalshi_mod.parse_games("mma_mixed_martial_arts", fixture("kalshi_ufc_fight.json"), NOW)
    assert {e["home"] for e in ufc} == {"Zach Reese", "Nick Klein"}


def test_kalshi_lines():
    games = kalshi_mod.parse_games("basketball_nba", fixture("kalshi_nba_game.json"), NOW)
    lines = kalshi_mod.parse_lines("basketball_nba", fixture("kalshi_nba_spread.json"), fixture("kalshi_nba_total.json"), games)
    phi_ind = lines["KXNBAGAME-26OCT28PHIIND"]  # PHI (away) "wins by over 1.5" at 0.49/0.52 -> home IND +1.5
    assert phi_ind["spreads"]["point"] == 1.5
    assert phi_ind["spreads"]["away"]["p"] == pytest.approx(0.505) and phi_ind["spreads"]["home"]["p"] == pytest.approx(0.495)
    assert phi_ind["totals"]["point"] == 237.5 and phi_ind["totals"]["over"]["p"] == pytest.approx(0.515)
    epl_games = kalshi_mod.parse_games("soccer_epl", fixture("kalshi_epl_game.json"), NOW)
    epl = kalshi_mod.parse_lines("soccer_epl", fixture("kalshi_epl_spread.json"), fixture("kalshi_epl_total.json"), epl_games)
    cfc = epl["KXEPLGAME-26OCT31CFCMUN"]
    assert cfc["spreads"]["point"] == -1.5 and cfc["spreads"]["home"]["p"] == pytest.approx(0.305)
    # over 3.5 at 0.39/0.41 is nearer 50/50 than over 2.5 at 0.61/0.63
    assert cfc["totals"]["point"] == 3.5 and cfc["totals"]["over"]["p"] == pytest.approx(0.40)


def kalshi_routes(answers):
    """Kalshi routes by series ticker: answers maps series -> body or a list of (status, body) to play in turn."""
    def answer(params):
        value = answers.get(params.get("series_ticker"))
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        return (200, value) if value is not None else (200, {"cursor": "", "events": [], "milestones": []})
    return [("api.elections.kalshi.com/trade-api/v2/events", answer)]


def test_kalshi_paces_requests_one_a_second():
    clock = Clock()
    fetch = Fetch(kalshi_routes({"KXNBAGAME": fixture("kalshi_nba_game.json"), "KXNBASPREAD": fixture("kalshi_nba_spread.json"),
                                 "KXNBATOTAL": fixture("kalshi_nba_total.json")}), clock)
    source = Kalshi(fetch=fetch, clock=clock, sleep=clock.sleep, lines=True)
    events = source.events("basketball_nba")
    times = [when for _url, _params, when in fetch.calls]
    assert len(times) == 3 and all(b - a >= 1.0 for a, b in zip(times, times[1:]))
    phi_ind = next(e for e in events if e["id"] == "KXNBAGAME-26OCT28PHIIND")
    assert phi_ind["markets"]["spreads"]["point"] == 1.5 and "totals" in phi_ind["markets"]
    params = fetch.calls[0][1]
    assert params == {"series_ticker": "KXNBAGAME", "status": "open", "with_nested_markets": "true",
                      "with_milestones": "true", "limit": 200}


def test_kalshi_backs_off_on_too_many_requests():
    clock = Clock()
    limited = (429, fixture("kalshi_too_many_requests.json"))
    fetch = Fetch(kalshi_routes({"KXEPLGAME": [limited, (200, fixture("kalshi_too_many_requests.json")), (200, fixture("kalshi_epl_game.json"))]}), clock)
    source = Kalshi(fetch=fetch, clock=clock, sleep=clock.sleep)
    games = source.keeper.get(("h2h", "soccer_epl"), 300, lambda: source.fetch_games("soccer_epl"), empty=[])
    assert len(games) == 2
    assert [s for s in clock.sleeps if s >= 2] == [2.0, 4.0]  # 429, then a 200 carrying the error, then fine
    times = [when for _u, _p, when in fetch.calls]
    assert times[1] - times[0] >= 2.0 and times[2] - times[1] >= 4.0


def test_kalshi_gives_up_after_the_back_offs_and_keeps_nothing_raised(caplog):
    clock = Clock()
    fetch = Fetch(kalshi_routes({"KXUFCFIGHT": [(429, fixture("kalshi_too_many_requests.json"))]}), clock)
    source = Kalshi(fetch=fetch, clock=clock, sleep=clock.sleep)
    with caplog.at_level(logging.WARNING):
        assert source.events("mma_mixed_martial_arts") == []
    assert len(fetch.calls) == 5 and [s for s in clock.sleeps if s >= 2] == [2.0, 4.0, 8.0, 16.0]
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "HttpError" in message and "http" not in message.replace("HttpError", "")  # only the error's type
    with pytest.raises(HttpError):
        source.request("/events", {"series_ticker": "KXUFCFIGHT"})


def test_kalshi_rests_after_running_out_of_retries():
    """One throttled spell costs one back-off: for REST seconds every request fails at once, without asking."""
    clock = Clock()
    answers = {"KXUFCFIGHT": [(429, fixture("kalshi_too_many_requests.json"))], "KXNBAGAME": fixture("kalshi_nba_game.json")}
    fetch = Fetch(kalshi_routes(answers), clock)
    source = Kalshi(fetch=fetch, clock=clock, sleep=clock.sleep)
    assert not source.resting()
    assert source.events("mma_mixed_martial_arts") == [] and len(fetch.calls) == 5 and source.resting()
    assert source.events("basketball_nba") == [] and len(fetch.calls) == 5  # resting: not asked, no back-off
    assert sum(s for s in clock.sleeps if s >= 2) == 30.0
    clock.now += kalshi_mod.REST
    assert not source.resting() and len(source.events("basketball_nba")) > 0 and len(fetch.calls) == 6


# --- the keeper: last good copy, persistence, back-off ----------------------------------------------------

def test_failures_keep_the_last_good_copy(caplog):
    clock = Clock()
    state = {"fail": False}

    def answer(params):
        if state["fail"]:
            return RuntimeError("boom https://gamma-api.polymarket.com/markets?tag_id=306")
        return polymarket_routes(fixture("polymarket_epl_moneyline.json"))[0][1](params)

    fetch = Fetch([("gamma-api", answer)], clock)
    source = Polymarket(fetch=fetch, clock=clock)
    good = source.events("soccer_epl")
    assert good
    state["fail"] = True
    clock.now += 16 * 60
    with caplog.at_level(logging.WARNING):
        assert source.events("soccer_epl") == good  # the last good copy, as_of unchanged
    assert all("gamma-api" not in r.getMessage() and "boom" not in r.getMessage() for r in caplog.records)
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    moneyline_calls = lambda: sum(1 for _u, params, _t in fetch.calls if params.get("sports_market_types") == "moneyline")  # noqa: E731
    calls = moneyline_calls()
    clock.now += 10
    assert source.events("soccer_epl") == good
    assert moneyline_calls() == calls  # not retried for 30 s after a failure ...
    clock.now += 31
    assert source.events("soccer_epl") == good
    assert moneyline_calls() == calls + 1  # ... then tried again (and failed again: still the old copy)


def test_a_restart_starts_from_the_saved_copy(tmp_path):
    store = Store(str(tmp_path / "markets.db"))
    clock = Clock()
    fetch = Fetch(polymarket_routes(fixture("polymarket_epl_moneyline.json")), clock)
    first = Polymarket(store=store, fetch=fetch, clock=clock).events("soccer_epl")
    assert first and store.cache_get(key_text(("markets", "polymarket", "h2h", 306)))  # the shared cache table
    restarted_fetch = Fetch([("gamma-api", RuntimeError("down"))], clock)
    clock.now += 60
    restarted = Polymarket(store=store, fetch=restarted_fetch, clock=clock)
    assert restarted.cached_events("soccer_epl") == first  # without fetching
    assert restarted.events("soccer_epl") == first and restarted_fetch.calls == []  # still fresh: no fetch
    clock.now += 3600
    assert restarted.events("soccer_epl") == first  # stale, the fetch fails: the saved copy
    assert restarted_fetch.calls


def test_keeper_never_raises_and_returns_empty_without_a_copy():
    keeper = Keeper("test", None, Clock())

    def broken():
        raise ValueError("nope")

    assert keeper.get("k", 60, broken, empty=[]) == []
    assert keeper.peek("k") is None


def test_keeper_lease_lets_one_process_fetch(tmp_path):
    store = Store(str(tmp_path / "lease.db"))
    clock = Clock()
    a, b = Keeper("lease", store, clock), Keeper("lease", Store(str(tmp_path / "lease.db")), clock)  # two processes
    assert a.get("x", 60, lambda: [1], empty=[]) == [1]
    clock.now += 120
    name = key_text(("markets", "lease", "x"))
    assert store.cache_lease(name, a.cache.holder, Keeper.LEASE, clock.now)  # a is refreshing x right now ...
    assert b.get("x", 60, lambda: [2], empty=[]) == [1]  # ... so b serves the shared copy instead of fetching too
    store.cache_release(name, a.cache.holder)
    assert b.get("x", 60, lambda: [2], empty=[]) == [2]  # the lease is free: b refreshes it
    assert a.get("x", 60, lambda: [3], empty=[]) == [2]  # and a picks up b's copy from the table, without a fetch


def test_keeper_is_a_thin_wrapper_over_the_shared_cache(tmp_path):
    """One rule set: the keeper's copies live in a TTLCache (memory and the shared table), are counted in its stats
    under the source's name, and a Stage 2 copy saved as a slates blob is the starting point after the upgrade."""
    store = Store(str(tmp_path / "keeper.db"))
    clock = Clock()
    store.save_slate("markets:ratings:power:basketball_nba", b'{"BOS": {"rating": 5.5}}', "etag", NOW - 600)
    keeper = Keeper("ratings", store, clock)
    assert keeper.peek(("power", "basketball_nba")) == (NOW - 600, {"BOS": {"rating": 5.5}})
    assert keeper.get(("power", "basketball_nba"), 3600, lambda: pytest.fail("still fresh"), empty={}) == {
        "BOS": {"rating": 5.5}}
    assert keeper.get("teams", 60, lambda: {"BOS": "Boston"}, empty={}) == {"BOS": "Boston"}
    assert store.cache_get(key_text(("markets", "ratings", "teams")))  # saved for the other process and a restart
    counts = keeper.cache.stats()["kinds"]["ratings"]
    assert counts["hits"] == 1 and counts["misses"] == 1 and counts["loads"] == 1
    when = keeper.put(("predictors", "basketball_nba"), {"401": [NOW, {"home": 0.6}]})
    restarted = Keeper("ratings", Store(str(tmp_path / "keeper.db")), clock)
    assert restarted.peek(("predictors", "basketball_nba")) == (when, {"401": [NOW, {"home": 0.6}]})


def test_keeper_backs_off_failures_doubling_to_fifteen_minutes(caplog):
    clock = Clock()
    keeper = Keeper("test", None, clock)
    calls = []

    def broken():
        calls.append(clock.now)
        raise HttpError(503, "Venue")

    with caplog.at_level(logging.WARNING):
        for _ in range(400):  # a read every 15 seconds for 100 minutes
            assert keeper.get("k", 60, broken, empty=[]) == []
            clock.now += 15
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert gaps[:7] == [30, 60, 120, 240, 480, 900, 900] and max(gaps) == 15 * 60
    assert caplog.text.count("refresh failed (HttpError)") == len(calls)  # each real failure once, not each read


# --- aliases and matching -------------------------------------------------------------------------------

@pytest.mark.parametrize("league,spellings,key", [
    ("basketball_nba", ["was", "WAS", "WSH", "Washington", "Wizards", "Washington Wizards"], "WSH"),
    ("basketball_nba", ["gsw", "GS", "Golden State Warriors", "Warriors"], "GS"),
    ("basketball_nba", ["nyk", "NY", "New York Knicks", "New York"], "NY"),
    ("basketball_nba", ["LA Clippers", "Los Angeles C", "lac"], "LAC"),
    ("icehockey_nhl", ["cal", "CGY", "Calgary Flames"], "CGY"),
    ("icehockey_nhl", ["las", "VGK", "Vegas", "Golden Knights"], "VGK"),
    ("icehockey_nhl", ["lak", "LA", "Los Angeles", "LA Kings"], "LA"),
    ("icehockey_nhl", ["mon", "MTL", "Montreal", "Montréal Canadiens"], "MTL"),
    ("americanfootball_nfl", ["JAC", "JAX", "Jacksonville"], "JAX"),
    ("americanfootball_nfl", ["la", "LAR", "Los Angeles R", "Rams"], "LAR"),
    ("baseball_mlb", ["CWS", "CHW", "Chicago W", "White Sox"], "CHW"),
])
def test_team_key_folds_venue_spellings(league, spellings, key):
    assert {team_key(league, s) for s in spellings} == {key}


def test_team_key_names_outside_the_tables():
    assert team_key("basketball_nba", "Seattle SuperSonics") is None
    assert team_key("soccer_epl", "Man Utd") == team_key("soccer_epl", "Manchester United FC") == "manchester united"
    assert team_key("soccer_uefa_champs_league", "FC Internazionale Milano") == team_key("soccer_uefa_champs_league", "Inter")
    assert team_key("soccer_usa_mls", "New York RB") == team_key("soccer_usa_mls", "New York Red Bulls")
    assert team_key("soccer_usa_mls", "CF Montréal") == team_key("soccer_usa_mls", "Montreal") == "montreal"
    assert team_key("americanfootball_ncaaf", "Kennesaw St.") == team_key("americanfootball_ncaaf", "Kennesaw State")
    assert team_key("basketball_ncaab", "St. John's") == team_key("basketball_ncaab", "Saint John's")
    assert team_key("basketball_ncaab", "UConn") == team_key("basketball_ncaab", "Connecticut")
    assert team_key("tennis_atp", "Jannik Sinner") == "jannik sinner" and team_key("soccer_epl", "") is None


def test_tables_have_no_spelling_for_two_teams():
    for league, table in aliases.TABLES.items():
        seen = {}
        for canonical, spellings in table.items():
            for spelling in [canonical, *spellings]:
                assert seen.setdefault(aliases._plain(spelling), canonical) == canonical, (league, spelling)


def slate_game(game_id, start, home, away):
    return {"id": game_id, "league": "basketball_nba", "start": start, "home": home, "away": away}


def test_match_events_with_abbreviation_mismatch():
    events = poly_mod.parse_moneylines("basketball_nba", fixture("polymarket_nba_moneyline.json"), NOW)
    games = [
        slate_game("basketball_nba:g1", "2026-10-28T23:00:00Z", {"name": "Cleveland Cavaliers", "short": "Cavaliers", "abbr": "CLE"},
                   {"name": "Washington Wizards", "short": "Wizards", "abbr": "WSH"}),
        slate_game("basketball_nba:g2", "2026-10-28T00:00:00Z", {"name": "New York Knicks", "abbr": "NY"},
                   {"name": "Detroit Pistons", "abbr": "DET"}),
        slate_game("basketball_nba:g3", "2026-11-05T00:00:00Z", {"name": "Boston Celtics", "abbr": "BOS"},
                   {"name": "Brooklyn Nets", "abbr": "BKN"}),  # no event that week
    ]
    matched = match_events("basketball_nba", games, events)
    assert matched["basketball_nba:g1"]["id"] == "nba-was-cle-2026-10-28"  # Polymarket "was" = ESPN "WSH"
    assert matched["basketball_nba:g2"]["id"] == "nba-det-nyk-2026-10-27"  # "nyk" = "NY"
    assert "basketball_nba:g3" not in matched and not matched["basketball_nba:g1"].get("swapped")


def test_match_events_nhl_codes_differ_on_every_venue():
    events = poly_mod.parse_moneylines("icehockey_nhl", fixture("polymarket_nhl_moneyline.json"), NOW)
    assert {(e["away_abbr"], e["home_abbr"]) for e in events} == {("lak", "nyr"), ("tor", "cal")}
    games = [{"id": "kings", "start": "2026-10-26T23:00:00Z", "home": {"name": "New York Rangers", "abbr": "NYR"},
              "away": {"name": "Los Angeles Kings", "abbr": "LA"}},
             {"id": "flames", "start": "2026-10-27T00:30:00Z", "home": {"name": "Calgary Flames", "abbr": "CGY"},
              "away": {"name": "Toronto Maple Leafs", "abbr": "TOR"}}]
    matched = match_events("icehockey_nhl", games, events)
    assert matched["kings"]["id"] == "nhl-lak-nyr-2026-10-26" and matched["flames"]["id"] == "nhl-tor-cal-2026-10-26"
    assert matched["flames"]["home_key"] == "CGY" and matched["flames"]["swapped"] is False


def test_match_events_turns_a_reversed_event_round():
    event = poly_mod.parse_moneylines("soccer_epl", fixture("polymarket_epl_lines.json"), NOW)  # no moneylines here
    assert event == []
    events = Polymarket(fetch=Fetch(polymarket_routes(fixture("polymarket_epl_moneyline.json"), [])), clock=Clock()).events("soccer_epl")
    che = next(e for e in events if e["id"] == "epl-che-mun-2026-10-31")
    che = {**che, "markets": {**che["markets"], "spreads": {"home": make_quote(0.30, 0.31), "away": make_quote(0.69, 0.70), "point": -1.5}}}
    game = {"id": "g", "start": "2026-10-31T12:30:00Z", "home": "Manchester United", "away": "Chelsea"}  # reversed
    out = match_events("soccer_epl", [game], [che])["g"]
    assert out["swapped"] is True and out["home"] == "Manchester United FC" and out["away"] == "Chelsea FC"
    assert out["markets"]["h2h"]["home"]["p"] == che["markets"]["h2h"]["away"]["p"]
    assert out["markets"]["h2h"]["draw"] == che["markets"]["h2h"]["draw"]
    assert out["markets"]["spreads"]["point"] == 1.5 and out["markets"]["spreads"]["home"]["p"] == pytest.approx(0.695)
    assert che["markets"]["spreads"]["point"] == -1.5  # the original is untouched


def test_match_events_double_header_goes_to_the_nearest_start():
    def event(ticker, start, bid):
        quote = make_quote(bid, bid + 0.01)
        return {"source": "kalshi", "league": "baseball_mlb", "id": ticker, "home": "Boston", "away": "New York Y",
                "home_abbr": "BOS", "away_abbr": "NYY", "start": start, "date": "2026-07-04", "as_of": "x",
                "markets": {"h2h": {"home": quote, "away": complement(quote)}}}

    events = [event("KXMLBGAME-26JUL041910NYYBOS", "2026-07-04T23:10:00Z", 0.55),
              event("KXMLBGAME-26JUL041305NYYBOS", "2026-07-04T17:05:00Z", 0.50)]
    games = [{"id": "late", "start": "2026-07-04T23:10:00Z", "home": {"name": "Boston Red Sox", "abbr": "BOS"},
              "away": {"name": "New York Yankees", "abbr": "NYY"}},
             {"id": "early", "start": "2026-07-04T17:05:00Z", "home": {"name": "Boston Red Sox", "abbr": "BOS"},
              "away": {"name": "New York Yankees", "abbr": "NYY"}}]
    matched = match_events("baseball_mlb", games, events)
    assert matched["early"]["id"].endswith("1305NYYBOS") and matched["late"]["id"].endswith("1910NYYBOS")
    # one game of the pair postponed: the remaining event isn't pulled onto the other game 6 hours away
    only = match_events("baseball_mlb", [games[0]], [events[1]])
    assert only == {}


def test_match_events_by_date_when_the_venue_has_no_start():
    events = kalshi_mod.parse_games("basketball_nba", fixture("kalshi_nba_game.json"), NOW)
    bos = next(e for e in events if e["id"] == "KXNBAGAME-26OCT27BKNBOS")
    assert bos["start"] is None
    game = slate_game("g", "2026-10-27T23:30:00Z", {"name": "Boston Celtics", "abbr": "BOS"}, {"name": "Brooklyn Nets", "abbr": "BKN"})
    assert match_events("basketball_nba", [game], [bos])["g"]["id"] == bos["id"]
    far = slate_game("g", "2026-11-03T23:30:00Z", game["home"], game["away"])
    assert match_events("basketball_nba", [far], [bos]) == {}


def test_match_events_names_and_players():
    kalshi_epl = kalshi_mod.parse_games("soccer_epl", fixture("kalshi_epl_game.json"), NOW)
    game = {"id": "e", "start": "2026-10-31T12:30:00Z", "home": {"name": "Chelsea", "short": "Chelsea", "abbr": "CHE"},
            "away": {"name": "Manchester United", "short": "Man United", "abbr": "MAN"}}
    assert match_events("soccer_epl", [game], kalshi_epl)["e"]["id"] == "KXEPLGAME-26OCT31CFCMUN"
    atp = kalshi_mod.parse_games("tennis_atp", fixture("kalshi_atp_match.json"), NOW)
    match = {"id": "t", "start": "2026-10-27T13:40:00Z", "home": {"name": "Carlos Alcaraz", "short": "C. Alcaraz"},
             "away": {"name": "Karen Khachanov", "short": "K. Khachanov"}}
    out = match_events("tennis_atp", [match], atp)["t"]
    assert out["id"] == "KXATPMATCH-26OCT25KHAALC" and out["swapped"] and out["home"] == "Carlos Alcaraz"
    college = {"id": "c", "start": "2026-10-28T23:00:00Z", "home": {"name": "Middle Tennessee Blue Raiders", "short": "Middle Tennessee"},
               "away": {"name": "Kennesaw State Owls", "short": "Kennesaw St"}}
    event = {"id": "KXNCAAFGAME-26OCT28KENNMTU", "home": "Middle Tennessee", "away": "Kennesaw St.", "start": "2026-10-28T23:00:00Z",
             "markets": {"h2h": {}}}
    assert match_events("americanfootball_ncaaf", [college], [event])["c"]["id"] == event["id"]


def test_match_events_needs_both_teams():
    events = kalshi_mod.parse_games("basketball_nba", fixture("kalshi_nba_game.json"), NOW)
    game = slate_game("g", "2026-10-26T23:00:00Z", {"name": "Cleveland Cavaliers", "abbr": "CLE"}, {"name": "Chicago Bulls", "abbr": "CHI"})
    assert match_events("basketball_nba", [game], events) == {}


# --- ratings --------------------------------------------------------------------------------------------

class FakeEspn:
    def __init__(self, summaries):
        self.summaries = summaries
        self.calls = 0

    def summary(self, league, event_id):
        self.calls += 1
        value = self.summaries[event_id]
        if isinstance(value, Exception):
            raise value
        return value


def test_espn_predictor():
    espn = FakeEspn({"401909877": fixture("espn_nba_summary_predictor.json"), "401873026": fixture("espn_nfl_summary_predictor.json")})
    clock = Clock()
    ratings = Ratings(espn, clock=clock)
    nba = ratings.espn_predictor("basketball_nba", "401909877")
    assert nba == {"home": pytest.approx(0.84), "away": pytest.approx(0.16), "as_of": "2026-10-27T01:15:00Z"}
    nfl = ratings.espn_predictor("americanfootball_nfl", 401873026)  # 61.9 / 37.8, ties left out
    assert nfl["home"] == pytest.approx(61.9 / 99.7) and nfl["home"] + nfl["away"] == pytest.approx(1.0)
    assert espn.calls == 2
    clock.now += 30 * 60
    ratings.espn_predictor("basketball_nba", "401909877")
    assert espn.calls == 2  # cached for an hour
    assert ratings.espn_predictor("icehockey_nhl", "1") is None and ratings.espn_predictor("basketball_nba", None) is None
    assert ratings_mod.parse_predictor({"predictor": None}) is None


def test_espn_predictor_follows_the_header_when_ids_disagree():
    summary = fixture("espn_nba_summary_predictor.json")
    summary["header"]["competitions"][0]["competitors"][0]["homeAway"] = "away"
    summary["header"]["competitions"][0]["competitors"][1]["homeAway"] = "home"
    assert ratings_mod.parse_predictor(summary) == {"home": pytest.approx(0.16), "away": pytest.approx(0.84)}


def test_espn_predictor_survives_a_restart(tmp_path):
    store = Store(str(tmp_path / "pred.db"))
    clock = Clock()
    Ratings(FakeEspn({"401909877": fixture("espn_nba_summary_predictor.json")}), store=store, clock=clock).espn_predictor("basketball_nba", "401909877")
    down = FakeEspn({"401909877": RuntimeError("down")})
    clock.now += 10 * 60
    restarted = Ratings(down, store=store, clock=clock)
    assert restarted.espn_predictor("basketball_nba", "401909877")["home"] == pytest.approx(0.84)
    assert down.calls == 0


def espn_routes():
    return [("college-football/seasons", (200, fixture("espn_cfb_powerindex.json"))),
            ("nba/seasons", (200, fixture("espn_nba_powerindex.json"))),
            ("basketball/nba/teams", (200, fixture("espn_nba_teams.json"))),
            ("college-football/teams", (200, fixture("espn_cfb_teams.json"))),
            ("barttorvik.com/2027_team_results.json", (200, fixture("torvik_2027_team_results.json"))),
            ("tennisabstract.com/reports/atp_elo_ratings.html", (200, fixture("tennis_abstract_atp_elo.html"))),
            ("eloratings.net/World.tsv", (200, fixture("eloratings_world.tsv"))),
            ("eloratings.net/en.teams.tsv", (200, fixture("eloratings_teams.tsv")))]


def test_power_bpi_and_the_normal_formula():
    clock = Clock()
    fetch = Fetch(espn_routes(), clock)
    ratings = Ratings(None, fetch=fetch, clock=clock)
    power = ratings.power("basketball_nba")
    assert set(power) == {"BOS", "BKN", "OKC", "NY", "DET"}
    assert power["OKC"] == {**power["OKC"], "rating": 8.616, "kind": "BPI", "as_of": "2026-10-26T23:54:00Z"}
    assert "seasons/2027/powerindex" in fetch.calls[0][0]  # 2026-27 is ESPN's season 2027
    bos, bkn = power["BOS"]["rating"], power["BKN"]["rating"]
    game = ratings.game_prob("basketball_nba", {"name": "Boston Celtics", "abbr": "BOS"}, "Brooklyn Nets")
    expected = 0.5 * (1 + math.erf(((bos - bkn + 2.0) / 12.0) / math.sqrt(2)))  # Phi((BPI_h - BPI_a + 2.0) / 12)
    assert game["home"] == pytest.approx(expected) and game["away"] == pytest.approx(1 - expected)
    assert game["source"] == "BPI" and "draw" not in game and game["margin"] == pytest.approx(bos - bkn + 2.0, abs=0.01)
    neutral = ratings.game_prob("basketball_nba", "BOS", "BKN", neutral=True)
    assert neutral["home"] == pytest.approx(0.5 * (1 + math.erf(((bos - bkn) / 12.0) / math.sqrt(2))))
    calls = len(fetch.calls)
    clock.now += 11 * 3600
    ratings.power("basketball_nba")
    assert len(fetch.calls) == calls  # 12-hour TTL
    assert ratings.game_prob("basketball_nba", "BOS", "Seattle SuperSonics") is None
    assert ratings.game_prob("icehockey_nhl", "BOS", "TOR") is None and ratings.power("icehockey_nhl") == {}


def test_power_fpi_college_names():
    ratings = Ratings(None, fetch=Fetch(espn_routes()), clock=Clock())
    power = ratings.power("americanfootball_ncaaf")
    assert {"ohio state", "michigan", "kennesaw state", "middle tennessee"} <= set(power)
    game = ratings.game_prob("americanfootball_ncaaf", "Middle Tennessee", "Kennesaw St.")
    margin = power["middle tennessee"]["rating"] - power["kennesaw state"]["rating"] + 2.5
    assert game["source"] == "FPI" and game["home"] == pytest.approx(0.5 * (1 + math.erf(margin / 15.5 / math.sqrt(2))))


def test_torvik_matchup_formula():
    ratings = Ratings(None, fetch=Fetch(espn_routes()), clock=Clock())
    rows = {r[1]: r for r in fixture("torvik_2027_team_results.json")}
    game = ratings.game_prob("basketball_ncaab", {"name": "Duke Blue Devils", "short": "Duke"}, {"name": "UConn Huskies", "short": "UConn"})
    duke, uconn = rows["Duke"], rows["Connecticut"]
    avg_eff = statistics.fmean(r[4] for r in rows.values())
    avg_tempo = statistics.fmean(r[44] for r in rows.values())
    possessions = duke[44] * uconn[44] / avg_tempo
    margin = (duke[4] * uconn[6] / avg_eff - uconn[4] * duke[6] / avg_eff) * possessions / 100 + 3.0
    assert game["source"] == "Torvik" and game["margin"] == pytest.approx(margin, abs=0.01)
    assert game["home"] == pytest.approx(0.5 * (1 + math.erf(margin / 11.0 / math.sqrt(2))))
    assert ratings.game_prob("basketball_ncaab", "St. John's Red Storm", "Saint Mary's")["source"] == "Torvik"


def test_tennis_elo_with_surface():
    ratings = Ratings(None, fetch=Fetch(espn_routes()), clock=Clock())
    power = ratings.power("tennis_atp_paris_masters")
    sinner, alcaraz = power["jannik sinner"], power["carlos alcaraz"]
    assert sinner["as_of"] == "2026-10-26T00:00:00Z" and sinner["hard"] == 2234.3 and alcaraz["rating"] == 2149.8
    overall = ratings.game_prob("tennis_atp", "J. Sinner", "Carlos Alcaraz")
    assert overall["home"] == pytest.approx(1 / (1 + 10 ** (-(2296.9 - 2149.8) / 400)))
    hard = ratings.game_prob("tennis_atp", "Jannik Sinner", "C. Alcaraz", surface="Hard")
    d = (0.5 * 2296.9 + 0.5 * 2234.3) - (0.5 * 2149.8 + 0.5 * 2086.2)
    assert hard["home"] == pytest.approx(1 / (1 + 10 ** (-d / 400))) and "hard" in hard["detail"]


def test_national_team_elo_with_draw():
    ratings = Ratings(None, fetch=Fetch(espn_routes()), clock=Clock())
    game = ratings.game_prob("soccer_uefa_nations_league", "Spain", "England")
    we = 1 / (10 ** (-(2287 - 2146 + 100) / 400) + 1)
    draw = 4 * 0.27 * we * (1 - we)
    assert game["draw"] == pytest.approx(draw) and game["home"] == pytest.approx(we - draw / 2)
    assert game["home"] + game["away"] + game["draw"] == pytest.approx(1.0)
    assert game["home"] + game["draw"] / 2 == pytest.approx(we)  # eloratings.net's expected result kept exact
    usa = ratings.game_prob("soccer_fifa_world_cup", "USA", "Korea Republic", neutral=True)
    assert usa and usa["home"] > usa["away"]
    assert ratings.game_prob("soccer_epl", "Chelsea", "Arsenal") is None  # club soccer: no rating source


def test_ratings_keep_the_last_good_copy(tmp_path):
    store = Store(str(tmp_path / "ratings.db"))
    clock = Clock()
    Ratings(None, store=store, fetch=Fetch(espn_routes()), clock=clock).power("basketball_nba")
    clock.now += 13 * 3600
    down = Fetch([("espn", (503, None))], clock)
    restarted = Ratings(None, store=store, fetch=down, clock=clock)
    assert restarted.power("basketball_nba")["OKC"]["rating"] == 8.616 and down.calls  # tried, kept the old copy
    assert restarted.game_prob("basketball_nba", "OKC", "DET", fetch=False)["source"] == "BPI"


def test_cached_reads_never_fetch():
    fetch = Fetch([])
    assert Polymarket(fetch=fetch, clock=Clock()).cached_events("basketball_nba") == []
    assert Kalshi(fetch=fetch, clock=Clock(), sleep=lambda s: None).cached_events("basketball_nba") == []
    ratings = Ratings(None, fetch=fetch, clock=Clock())
    assert ratings.cached_power("basketball_nba") == {} and ratings.game_prob("basketball_nba", "BOS", "NY", fetch=False) is None
    assert fetch.calls == []


# --- coverage -------------------------------------------------------------------------------------------

def test_coverage_of_the_leagues_asked_for():
    assert missing("polymarket") == [] and missing("kalshi") == []
    for league in ("basketball_nba", "americanfootball_nfl", "americanfootball_ncaaf", "basketball_ncaab",
                   "tennis_atp", "tennis_wta"):
        assert COVERAGE[league]["rating"]
    assert set(missing("rating")) >= {"icehockey_nhl", "baseball_mlb", "soccer_epl", "mma_mixed_martial_arts"}
    assert set(TARGET_LEAGUES) - set(missing("predictor")) == {"basketball_nba", "americanfootball_nfl", "americanfootball_ncaaf"}
    assert Polymarket().covers("tennis_atp_vienna_open") and Kalshi().covers("tennis_wta_tokyo")
    assert Kalshi().covers("soccer_japan_j_league") and Polymarket().covers("soccer_japan_j_league")
    assert not Kalshi().covers("soccer_england_league2") and Polymarket().covers("soccer_england_league2")
