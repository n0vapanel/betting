# telegram_betting_bot_simple/main.py
from __future__ import annotations

import logging
import os
from telegram.ext import Application, PersistenceInput, PicklePersistence

from .config import Config, load_config
from .bot.handlers import setup_handlers

def build_application(config: Config) -> Application:
    """The bot with its handlers and the shared games service."""
    builder = Application.builder().token(config.telegram_token)
    if config.state_dir:
        # Keeps each user's mode, slip and stake across restarts. Not bot_data: it holds the live
        # games service, which is rebuilt on every start (a saved copy would replace it on startup)
        builder = builder.persistence(PicklePersistence(
            os.path.join(config.state_dir, "bot_state.pickle"), store_data=PersistenceInput(bot_data=False),
        ))
    application = builder.build()
    setup_handlers(application, config)
    if application.bot_data["service"] is None:
        logging.warning("THE_ODDS_API_KEY is not set, so the bot can't show games or odds")
    return application

def main() -> None:
    # Initializes the bot, polls until Ctrl+C / SIGTERM, then shuts down cleanly.
    build_application(load_config()).run_polling()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # httpx logs every request URL at INFO, and those URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
