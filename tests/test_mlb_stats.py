"""data/mlb_stats.py: MLB StatsAPI's schedule with announced weather, through the shared cache. The answers are
captured schedules (a July 2026 day with weather, a World Series day before MLB announced any); no network."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from telegram_betting_bot_simple.data.cache import Policy, TTLCache, ttl_for
from telegram_betting_bot_simple.data.mlb_stats import (HYDRATE, SCHEDULE, MlbStats, parse_schedule, parse_weather,
                                                          parse_wind, roof_closed)

FIXTURES = Path(__file__).parent / "fixtures" / "context"
UTC = datetime.timezone.utc


def schedule(day: str):
    return json.loads((FIXTURES / f"mlb_schedule_{day}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("text, expected", [
    ("14 mph, Out To RF", (14, "out", "RF")),
    ("9 mph, Out To CF", (9, "out", "CF")),
    ("10 mph, Out To LF", (10, "out", "LF")),
    ("8 mph, In From LF", (8, "in", "LF")),
    ("7 mph, In From CF", (7, "in", "CF")),
    ("11 mph, In From RF", (11, "in", "RF")),
    ("5 mph, L To R", (5, "cross", None)),
    ("6 mph, R To L", (6, "cross", None)),
    ("Calm", (0, "calm", None)),
    ("0 mph, Calm", (0, "calm", None)),
    ("0 mph, None", (0, "calm", None)),
    ("6 mph, Varies", (6, "varies", None)),
    ("Varies", (None, "varies", None)),
    ("12 mph, None", (12, None, None)),
    ("12 MPH, OUT TO RF", (12, "out", "RF")),
    ("", (None, None, None)),
    (None, (None, None, None)),
    ("breezy", (None, None, None)),
])
def test_parse_wind(text, expected):
    assert parse_wind(text) == expected


def test_roof_closed():
    assert roof_closed("Roof Closed") and roof_closed("Dome") and roof_closed(" dome ")
    assert not roof_closed("Sunny") and not roof_closed("Partly Cloudy") and not roof_closed(None)


def test_parse_weather():
    assert parse_weather({"condition": "Sunny", "temp": "76", "wind": "15 mph, Out To RF"}) == {
        "condition": "Sunny", "temp_f": 76, "wind_mph": 15, "wind_text": "15 mph, Out To RF", "wind_kind": "out",
        "field": "RF", "roof_closed": False}
    closed = parse_weather({"condition": "Roof Closed", "temp": "68", "wind": "0 mph, None"})
    assert closed["roof_closed"] is True and closed["wind_kind"] == "calm" and closed["wind_mph"] == 0
    assert parse_weather({}) is None and parse_weather(None) is None


def test_a_captured_summer_day():
    games = parse_schedule(schedule("2026-07-18"))
    assert len(games) == 16
    first = games[0]
    assert set(first) == {"game_pk", "start", "status", "detailed_state", "double_header", "game_number", "day_night",
                          "home", "away", "venue", "weather"}
    assert set(first["weather"]) == {"condition", "temp_f", "wind_mph", "wind_text", "wind_kind", "field",
                                     "roof_closed"}
    by_pk = {game["game_pk"]: game for game in games}
    cleveland = by_pk[824414]
    assert cleveland["home"] == {"name": "Cleveland Guardians", "id": 114} and cleveland["venue"] == {
        "mlb_id": 5, "name": "Progressive Field", "roof": "open"}
    assert cleveland["weather"]["wind_kind"] == "out" and cleveland["weather"]["field"] == "RF"
    assert cleveland["double_header"] == "S" and by_pk[824412]["game_number"] == 2  # a split doubleheader
    toronto = by_pk[822790]
    assert toronto["weather"]["roof_closed"] is True and toronto["venue"]["roof"] == "retractable"
    assert by_pk[823763]["weather"]["wind_kind"] == "in" and by_pk[823763]["weather"]["field"] == "CF"
    assert {game["weather"]["roof_closed"] for game in games} == {True, False}
    kinds = {game["weather"]["wind_kind"] for game in games}
    assert {"out", "in", "cross", "calm"} <= kinds


def test_a_game_before_mlb_announces_weather():
    (game,) = parse_schedule(schedule("2026-10-27"))
    assert game["weather"] is None and game["venue"]["roof"] == "dome" and game["status"] == "Preview"
    assert game["start"] == "2026-10-28T00:00:00Z"
    assert parse_schedule({}) == [] and parse_schedule({"dates": [{"games": [{"gamePk": None}, "x"]}]}) == []


class Session:
    def __init__(self, answers):
        self.answers, self.calls, self.status = answers, [], 200

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))

        class Response:
            status_code = self.status
            ok = self.status == 200

            @staticmethod
            def json():
                return self.answers.get(params["date"], {"dates": []})
        return Response()


def test_games_reads_through_the_cache_by_the_days_state():
    now = [datetime.datetime(2026, 7, 18, 16, 0, tzinfo=UTC).timestamp()]  # noon in New York
    clock = lambda: now[0]  # noqa: E731
    session = Session({"2026-07-18": schedule("2026-07-18"), "2026-07-17": {"dates": []}})
    mlb = MlbStats(TTLCache(clock=clock), session=session)
    assert len(mlb.games(datetime.date(2026, 7, 18))) == 16
    assert session.calls == [(SCHEDULE, {"sportId": 1, "date": "2026-07-18", "hydrate": HYDRATE})]
    assert HYDRATE == "venue(location,fieldInfo),weather"
    assert mlb.day_state(datetime.date(2026, 7, 18)) == "today"
    assert mlb.day_state(datetime.date(2026, 7, 17)) == "past" and mlb.day_state(datetime.date(2026, 7, 19)) == "future"
    now[0] += 9 * 60
    mlb.games("2026-07-18")
    assert len(session.calls) == 1  # 10 minutes on the day itself
    now[0] += 12 * 60 + 10 * 60  # past the 10 minutes and their 10 stale ones
    mlb.games(datetime.datetime(2026, 7, 18, 20, 0, tzinfo=UTC))
    assert len(session.calls) == 2
    mlb.games("2026-07-17")
    now[0] += 5 * 3600
    mlb.games("2026-07-17")
    assert len(session.calls) == 3  # another day's schedule is kept 6 hours
    assert ttl_for("mlb_schedule", "today") == Policy(600, 600)
    assert ttl_for("mlb_schedule", "past") == ttl_for("mlb_schedule", "future") == Policy(6 * 3600, 0)


def test_a_failure_is_remembered_briefly():
    now = [datetime.datetime(2026, 7, 18, 16, 0, tzinfo=UTC).timestamp()]
    session = Session({})
    session.status = 503
    mlb = MlbStats(TTLCache(clock=lambda: now[0]), session=session)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        mlb.games("2026-07-18")
    with pytest.raises(RuntimeError):
        mlb.games("2026-07-18")
    assert len(session.calls) == 1  # not retried within the minute
