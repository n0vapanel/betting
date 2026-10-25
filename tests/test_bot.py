"""Unit tests for the bot's session management, tier classification and handler wiring.

These tests focus on components that do not require network access or a
Telegram environment: the SessionManager used to track user state
across messages, the TierClassifier that assigns predictions to
confidence tiers, and how handlers build the engine and format results.
"""

from __future__ import annotations

from telegram.ext import Application

from telegram_betting_bot_simple.bot.handlers import format_prediction, setup_handlers
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
