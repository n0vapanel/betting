"""Unit tests for the bot's session management, tier classification and handler wiring.

These tests focus on components that do not require network access or a
Telegram environment: the SessionManager used to track user state
across messages, the TierClassifier that assigns predictions to
confidence tiers, and how handlers build the engine and format results.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from telegram.ext import Application, CallbackQueryHandler, ConversationHandler

from telegram_betting_bot_simple.bot.handlers import format_prediction, setup_handlers, split_message
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

def test_setup_handlers_passes_api_keys_to_engine() -> None:
    config = Config(
        TELEGRAM_BOT_TOKEN="123456:test",
        API_SPORTS_KEY="sports",
        THE_ODDS_API_KEY="odds",
        NEWS_API_KEY="news",
    )
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    engine = application.bot_data["pred_engine"]
    assert (engine.api_sports_key, engine.odds_api_key, engine.news_api_key) == ("sports", "odds", "news")

def test_format_prediction() -> None:
    prediction = {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "winner": "Lakers",
        "tier": "Middle",
        "probabilities": {"Lakers": 0.62, "Celtics": 0.38},
    }
    assert format_prediction(prediction) == "Celtics @ Lakers → Lakers (Middle, 62%)"

def test_split_message_respects_telegram_limit() -> None:
    assert split_message(["a", "b"]) == ["a\nb"]
    # A busy soccer day: hundreds of fixtures won't fit in one 4096-character message
    lines = ["Here are your predictions:"] + [
        f"• Team {i} @ Team {i + 1} → Team {i} (Hot, 51%)" for i in range(300)
    ]
    messages = split_message(lines)
    assert len(messages) > 1
    assert all(len(m.encode("utf-16-le")) // 2 <= 4096 for m in messages)
    # Every line arrives once, whole and in order
    assert "\n".join(messages).split("\n") == lines

def test_taps_on_expired_menus_get_an_answer() -> None:
    config = Config(TELEGRAM_BOT_TOKEN="123456:test")
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    handlers = application.handlers[0]
    # The fallback must come after the conversation, so it only sees taps the conversation didn't claim
    conv_index = next(i for i, h in enumerate(handlers) if isinstance(h, ConversationHandler))
    fallback = next(h for h in handlers[conv_index + 1:] if isinstance(h, CallbackQueryHandler))

    answers = []

    class FakeQuery:
        async def answer(self, text=None, show_alert=False):
            answers.append((text, show_alert))

    asyncio.run(fallback.callback(SimpleNamespace(callback_query=FakeQuery()), None))
    assert answers == [("This menu has expired. Send /start to begin again.", True)]
