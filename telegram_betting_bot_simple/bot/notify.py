"""Messages from the app to its owner's Telegram chat (TELEGRAM_CHANNEL_ID), sent by the bot.

The learning loop (learn/loop.py) tells the owner about each change to the brain (a promotion, a rollback, an Undo,
safe mode, the Pause switch) and sends a three-line weekly summary with a link to the Brain progress page. The
message goes through python-telegram-bot's ``Bot`` with the app's own token, from whichever thread calls it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from telegram import Bot
from telegram.error import TelegramError

log = logging.getLogger(__name__)

MAX_MESSAGE = 4000  # Telegram allows 4,096 characters


def owner_notifier(token: Optional[str], chat_id: Optional[str], *, bot: Any = None
                   ) -> Optional[Callable[[str], bool]]:
    """A sender of plain-text messages to the owner's chat, or None without a token or chat id. It never raises
    (a message that can't be sent is logged, without the token) and returns whether Telegram took it."""
    if not token or not chat_id:
        return None

    def send(text: str) -> bool:
        async def go() -> None:
            if bot is not None:
                await bot.send_message(chat_id=chat_id, text=text[:MAX_MESSAGE], disable_web_page_preview=True)
                return
            # A Bot made for this message: started and shut down here (its HTTP client closed after), as
            # python-telegram-bot asks for a Bot used outside an Application
            async with Bot(token) as sender:
                await sender.send_message(chat_id=chat_id, text=text[:MAX_MESSAGE], disable_web_page_preview=True)

        try:
            asyncio.run(go())
            return True
        except (TelegramError, OSError, RuntimeError, ValueError) as exc:
            log.warning("Couldn't send the brain's message to the owner's chat (%s)", type(exc).__name__)
            return False

    return send
