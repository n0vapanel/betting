"""Unit tests for the bot's session management and tier classification.

These tests focus on components that do not require network access or a
Telegram environment: the SessionManager used to track user state
across messages and the TierClassifier that assigns predictions to
confidence tiers.
"""

from __future__ import annotations

from telegram_betting_bot_simple.bot.session_manager import SessionManager
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
