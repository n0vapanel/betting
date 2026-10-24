# telegram_betting_bot_simple/main.py
from __future__ import annotations

import asyncio
import logging
from telegram.ext import Application

from .config import load_config
from .bot.handlers import setup_handlers

async def main() -> None:
    config = load_config()
    application = Application.builder().token(config.telegram_token).build()
    setup_handlers(application, config)
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    await application.updater.idle()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
