# telegram_betting_bot_simple/main.py
from __future__ import annotations

import logging
import os
from telegram.ext import Application, PicklePersistence

from .config import load_config
from .bot.handlers import setup_handlers

def main() -> None:
    config = load_config()
    builder = Application.builder().token(config.telegram_token)
    if config.state_dir:
        # Keeps each user's mode, slip and stake across restarts
        builder = builder.persistence(PicklePersistence(os.path.join(config.state_dir, "bot_state.pickle")))
    application = builder.build()
    setup_handlers(application, config)
    # Initializes the bot, polls until Ctrl+C / SIGTERM, then shuts down cleanly.
    application.run_polling()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # httpx logs every request URL at INFO, and those URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
