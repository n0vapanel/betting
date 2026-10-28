# telegram_betting_bot_simple/main.py
"""Start the Telegram bot: python -m telegram_betting_bot_simple.main

When PUBLIC_URL points at the dashboard over HTTPS, the bot's menu button (left of the message box)
opens it as a Telegram Mini App, so friends (those in TELEGRAM_ALLOWED_IDS) get the full app without a
password. Without it the menu button goes back to Telegram's plain one, so a button left over from an
earlier address doesn't lead to a dead page.
"""
from __future__ import annotations

import logging
import os
from telegram import MenuButtonDefault, MenuButtonWebApp, WebAppInfo
from telegram.error import TelegramError
from telegram.ext import Application, PersistenceInput, PicklePersistence

from .config import Config, load_config
from .bot.handlers import setup_handlers

async def set_menu_button(application: Application) -> None:
    """Point the bot's menu button at the Mini App, or back to the plain one (runs once the bot has connected).

    Telegram keeps the button between restarts, so it is reset when PUBLIC_URL is removed or changed.
    """
    config: Config = application.bot_data["config"]
    if config.mini_app_url:
        button = MenuButtonWebApp("Open app", WebAppInfo(config.mini_app_url))
    else:
        if config.public_url:
            logging.warning("PUBLIC_URL must start with https:// for Telegram to open it; no app button")
        button = MenuButtonDefault()
    try:
        await application.bot.set_chat_menu_button(menu_button=button)
    except TelegramError as exc:
        # The bot works without it, so a refusal (e.g. a URL Telegram won't open) only gets logged
        logging.warning("Couldn't set the bot's menu button: %s", exc)

def build_application(config: Config) -> Application:
    """The bot with its handlers and the shared games service."""
    builder = Application.builder().token(config.telegram_token).post_init(set_menu_button)
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
