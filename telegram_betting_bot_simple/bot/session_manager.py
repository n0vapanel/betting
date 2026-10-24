"""Session management for Telegram betting bot.

This module defines simple classes to track user interactions across multiple
steps of the prediction workflow.  A `UserSession` stores the user's
selected sport, match date and prediction type (single vs. multi).  The
`SessionManager` maintains a mapping of chat identifiers to session
instances, ensuring that each user can navigate the conversation without
losing context.  Sessions can be reset between conversations.

The conversation flow typically proceeds through the following stages:
  1. Sport selection – the user chooses a sport such as NBA or soccer.
  2. Date entry – the user provides a date (defaulting to today).
  3. Prediction type – the user selects whether to receive individual
     predictions or a multi‑bet suggestion.
  4. Prediction delivery – the bot generates predictions using the engine.

The session manager helps tie these selections together across multiple
messages or callback queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional


@dataclass
class UserSession:
    """Represents the state of a user interacting with the bot.

    Attributes:
        sport: A short code identifying the selected sport (e.g. ``"nba"``).
        match_date: The date for which the user wants predictions.  If
            ``None``, today's date will be used by the prediction engine.
        prediction_type: Either ``"single"`` for individual game predictions
            or ``"multi"`` for a combined parlay recommendation.
    """

    sport: Optional[str] = None
    match_date: Optional[date] = None
    prediction_type: Optional[str] = None


class SessionManager:
    """Manages user sessions keyed by chat ID.

    A ``SessionManager`` holds a dictionary mapping Telegram chat IDs to
    ``UserSession`` instances.  It exposes helper methods for retrieving
    sessions (creating them on demand) and clearing them when a user
    cancels or completes a conversation.  Using a single manager allows
    multiple conversations to run concurrently without interfering with
    each other's state.
    """

    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}

    def get_session(self, chat_id: int) -> UserSession:
        """Return the existing session for ``chat_id`` or create a new one."""
        return self._sessions.setdefault(chat_id, UserSession())

    def reset_session(self, chat_id: int) -> None:
        """Reset the session associated with ``chat_id`` to a new instance."""
        self._sessions[chat_id] = UserSession()
