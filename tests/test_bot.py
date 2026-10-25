"""Unit tests for the bot's session management, tier classification and handlers.

These tests focus on components that do not require network access or a
Telegram environment: the SessionManager used to track user state
across messages, the TierClassifier that assigns predictions to
confidence tiers, and the bot's screens, buttons and Single/Multi flow
(run against the shared fake games service).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from telegram.ext import Application, CallbackQueryHandler

from telegram_betting_bot_simple.bot import handlers
from telegram_betting_bot_simple.bot.handlers import (
    game_card_text,
    parse_pick,
    pick_data,
    setup_handlers,
    short_name,
    slip_text,
)
from telegram_betting_bot_simple.bot.session_manager import SessionManager
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.leagues import MAX_LEAGUES, pick_leagues
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

def test_pick_buttons_fit_telegram_and_round_trip() -> None:
    game_id = "4c1f9e2b7d3a5e6f8a9b0c1d2e3f4a5b"  # The Odds API ids are 32 hex characters
    data = pick_data("soccer_spain_segunda_division", game_id, "spreads", "away")
    assert len(data.encode()) <= 64
    assert parse_pick(data) == {"league": "soccer_spain_segunda_division", "id": game_id[:16],
                                "market": "spreads", "side": "away"}
    assert parse_pick("p:nba:abc:zz") is None

def test_short_names() -> None:
    assert short_name("Oklahoma City Thunder") == "Thunder"
    assert short_name("Portland Trail Blazers") == "Trail Blazers"
    assert short_name("Manchester United", "soccer_epl") == "Manchester United"

def test_game_card_shows_prediction_injuries_and_leaders(service) -> None:
    card = service.game("basketball_nba", "g1")
    card["injuries"] = {"away": ["Bradley Beal (Out)", "James Harden (Day-To-Day)"], "home": []}
    card["players"] = {"home": [
        {"name": "Shai Gilgeous-Alexander", "out": False, "pts": 31.2, "reb": 5.1, "ast": 6.4},
        {"name": "Nikola Topic", "out": True, "pts": 0.0, "reb": 0.0, "ast": 0.0},
    ], "away": []}
    text = game_card_text(card, "7:00 PM EDT")
    assert "Los Angeles Clippers @ Oklahoma City Thunder" in text
    assert "AI prediction:</b> Thunder" in text
    assert "Spread Thunder -13.5: covers" in text
    assert "Clippers: Bradley Beal" in text and "Harden" not in text  # only players ruled out
    assert "Shai Gilgeous-Alexander: 31.2 pts, 5.1 reb, 6.4 ast" in text
    assert "Topic" not in text.split("Projected leaders")[1]

def test_setup_handlers_shares_the_dashboard_service() -> None:
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", THE_ODDS_API_KEY="odds")
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    assert isinstance(application.bot_data["service"], DashboardService)
    buttons = [h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler)]
    assert buttons[-1].pattern is None  # catch-all for expired menus comes last
    assert all(h.pattern is not None for h in buttons[:-1])
    no_key = Application.builder().token(config.telegram_token).build()
    setup_handlers(no_key, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    assert no_key.bot_data["service"] is None

def test_multi_flow_builds_a_slip_priced_at_one_bookmaker(service) -> None:
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    application.bot_data["service"] = service
    context = SimpleNamespace(application=application, user_data={})
    screens = []

    class FakeQuery:
        def __init__(self, data):
            self.data = data
            self.answers = []

        async def answer(self, text=None, show_alert=False):
            self.answers.append(text)

        async def edit_message_text(self, text, reply_markup=None, **kwargs):
            screens.append({"text": text, "buttons": [[(b.text, b.callback_data) for b in row]
                                                      for row in reply_markup.inline_keyboard]})

    def tap(handler, data):
        query = FakeQuery(data)
        asyncio.run(handler(SimpleNamespace(callback_query=query, message=None), context))
        return query

    tap(handlers.mode_selected, "mode:multi")
    assert "Multi (parlay)" in screens[-1]["text"]
    tap(handlers.sport_selected, "sport:Basketball")  # only the NBA in season: straight to games
    games = screens[-1]
    assert "Clippers @ Thunder" in games["text"]
    tap(handlers.game_selected, games["buttons"][0][0][1])
    thunder_ml = screens[-1]["buttons"][0][1]  # moneyline row: away, home
    assert thunder_ml[0].startswith("Thunder ML")
    assert tap(handlers.pick_selected, thunder_ml[1]).answers == ["Added to slip ✓ (1 picks)"]
    assert screens[-1]["buttons"][0][1][0].startswith("✓ Thunder ML")
    tap(handlers.game_selected, games["buttons"][1][0][1])
    tap(handlers.pick_selected, screens[-1]["buttons"][0][0][1])  # Magic ML
    tap(handlers.show_slip, "slip")
    slip = screens[-1]["text"]
    assert "2-leg multi" in slip
    assert "at FanDuel" in slip  # 1.11 x 3.60 beats DraftKings' 1.10 x 3.50
    assert "payout <b>$39.96</b>" in slip  # $10 default stake
    tap(handlers.stake_selected, "stake:25")
    assert "payout <b>$99.90</b>" in screens[-1]["text"]

def test_single_mode_shows_the_payout_of_one_pick(service) -> None:
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    application.bot_data["service"] = service
    context = SimpleNamespace(application=application, user_data={"mode": "single", "stake": 10})
    screens = []

    class FakeQuery:
        data = pick_data("basketball_nba", "g2", "h2h", "away")

        async def answer(self, text=None, show_alert=False):
            pass

        async def edit_message_text(self, text, reply_markup=None, **kwargs):
            screens.append(text)

    asyncio.run(handlers.pick_selected(SimpleNamespace(callback_query=FakeQuery(), message=None), context))
    assert "Single: Orlando Magic to win" in screens[-1]
    assert "returns <b>$36.00</b>" in screens[-1]  # best price 3.60 at FanDuel

def test_slip_text_lists_reasons() -> None:
    text = slip_text({"legs": [], "reasons": ["Add at least 2 picks from different games for a multi"]}, [])
    assert "⚠️ Add at least 2 picks" in text

def test_taps_on_expired_menus_get_an_answer() -> None:
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    fallback = [h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler)][-1]

    answers = []

    class FakeQuery:
        async def answer(self, text=None, show_alert=False):
            answers.append((text, show_alert))

    asyncio.run(fallback.callback(SimpleNamespace(callback_query=FakeQuery()), None))
    assert answers == [("This menu has expired. Send /start to begin again.", True)]
