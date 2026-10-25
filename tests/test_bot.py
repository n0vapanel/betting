"""Unit tests for the bot's session management, tier classification and handlers.

These tests focus on components that do not require network access or a
Telegram environment: the SessionManager used to track user state
across messages, the TierClassifier that assigns predictions to
confidence tiers, and how handlers wire the odds feed, pick leagues and
format games.
"""

from __future__ import annotations

import asyncio
import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from telegram.ext import Application, CallbackQueryHandler

from telegram_betting_bot_simple.bot.handlers import format_games, setup_handlers, split_message
from telegram_betting_bot_simple.data.leagues import MAX_LEAGUES, pick_leagues
from telegram_betting_bot_simple.bot.session_manager import SessionManager
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.engine.tier_classifier import TierClassifier

def test_session_manager_reuse() -> None:
    mgr = SessionManager()
    chat_id = 123
    s1 = mgr.get_session(chat_id)
    s1.sport = "nba"
    # Getting the session again should return the same instance
    s2 = mgr.get_session(chat_id)
    assert s1 is s2
    assert s2.sport == "nba"
    # Resetting the session should create a new blank session
    mgr.reset_session(chat_id)
    s3 = mgr.get_session(chat_id)
    assert s3 is not s1
    assert s3.sport is None

def test_tier_classifier_thresholds() -> None:
    classifier = TierClassifier(threshold_middle=0.6, threshold_surefire=0.75)
    # Surefire: probability above surefire threshold
    outcome, tier = classifier.classify({"Team A": 0.8, "Team B": 0.2})
    assert outcome == "Team A"
    assert tier == "Surefire"
    # Middle: between middle and surefire
    outcome, tier = classifier.classify({"A": 0.65, "B": 0.35})
    assert tier == "Middle"
    # Hot: below middle threshold
    outcome, tier = classifier.classify({"A": 0.55, "B": 0.45})
    assert tier == "Hot"

def _application(**env: str) -> Application:
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", **env)
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    return application

def test_setup_handlers_wires_odds_client_from_config() -> None:
    application = _application(THE_ODDS_API_KEY="odds", TIMEZONE="Europe/London")
    assert application.bot_data["odds_client"].api_key == "odds"
    assert application.bot_data["timezone"] == ZoneInfo("Europe/London")
    # Without a key the bot still starts, and tells users how to add one
    assert _application().bot_data["odds_client"] is None

def test_pick_leagues_keeps_in_season_game_leagues_popular_first() -> None:
    sports = [
        {"key": "soccer_spain_segunda_division", "group": "Soccer", "title": "La Liga 2", "active": True, "has_outrights": False},
        {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": True, "has_outrights": False},
        {"key": "soccer_fifa_world_cup", "group": "Soccer", "title": "FIFA World Cup", "active": False, "has_outrights": False},
        {"key": "soccer_epl_winner", "group": "Soccer", "title": "EPL Winner", "active": True, "has_outrights": True},
        {"key": "basketball_nba", "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
    ]
    assert [league["key"] for league in pick_leagues(sports, "Soccer")] == ["soccer_epl", "soccer_spain_segunda_division"]
    many = [
        {"key": f"soccer_{i:02d}", "group": "Soccer", "title": f"League {i:02d}", "active": True, "has_outrights": False}
        for i in range(30)
    ]
    assert len(pick_leagues(many, "Soccer")) == MAX_LEAGUES

def test_format_games_lists_soonest_first_with_american_odds() -> None:
    events = [
        {
            "home_team": "Arsenal", "away_team": "Chelsea", "start_time": "2026-10-26T15:00:00Z",
            "markets": {"h2h": {"Arsenal": 110.0, "Chelsea": 220.0, "Draw": 240.0}}, "bookmaker": "FanDuel",
        },
        {
            "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics", "start_time": "2026-10-25T23:30:00Z",
            "markets": {"h2h": {"Boston Celtics": 130.0, "Los Angeles Lakers": -150.0}}, "bookmaker": "DraftKings",
        },
        {"home_team": "Home", "away_team": "Away", "start_time": "2026-10-25T11:00:00Z", "markets": {}, "bookmaker": None},
    ]
    now = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.timezone.utc)
    blocks = format_games(events, "Test League", ZoneInfo("America/New_York"), now=now)
    assert blocks == [
        "Test League: next 3 games (moneyline odds)",
        "🔴 Live now\nAway @ Home\nOdds not posted yet",
        "Sun Oct 25, 7:30 PM EDT\nBoston Celtics @ Los Angeles Lakers\n"
        "Boston Celtics +130 · Los Angeles Lakers -150 (DraftKings)",
        "Mon Oct 26, 11:00 AM EDT\nChelsea @ Arsenal\nChelsea +220 · Arsenal +110 · Draw +240 (FanDuel)",
    ]

def test_split_message_respects_telegram_limit() -> None:
    assert split_message(["a", "b"]) == ["a\nb"]
    assert split_message(["a", "b"], separator="\n\n") == ["a\n\nb"]
    # A busy soccer day: hundreds of lines won't fit in one 4096-character message
    lines = ["Here are your games:"] + [
        f"• Team {i} @ Team {i + 1} → Team {i} (Hot, 51%)" for i in range(300)
    ]
    messages = split_message(lines)
    assert len(messages) > 1
    assert all(len(m.encode("utf-16-le")) // 2 <= 4096 for m in messages)
    # Every line arrives once, whole and in order
    assert "\n".join(messages).split("\n") == lines

def test_taps_on_expired_menus_get_an_answer() -> None:
    handlers = _application().handlers[0]
    button_handlers = [h for h in handlers if isinstance(h, CallbackQueryHandler)]
    # The catch-all must come last, so it only sees taps no menu step claimed
    fallback = button_handlers[-1]
    assert fallback.pattern is None
    assert all(h.pattern is not None for h in button_handlers[:-1])

    answers = []

    class FakeQuery:
        async def answer(self, text=None, show_alert=False):
            answers.append((text, show_alert))

    asyncio.run(fallback.callback(SimpleNamespace(callback_query=FakeQuery()), None))
    assert answers == [("This menu has expired. Send /start to begin again.", True)]
