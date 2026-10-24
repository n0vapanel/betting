"""Factory function to create and configure the Telegram bot application."""

from __future__ import annotations

from telegram.ext import Application

from ..config import Config
from .handlers import setup_handlers


def create_application(config: Config) -> Application:
    """Create a Telegram `Application` and register handlers.

    Args:
        config: The application configuration containing at least the
            Telegram bot token.

    Returns:
        A fully configured `Application` instance ready to be started.
    """
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    return application
