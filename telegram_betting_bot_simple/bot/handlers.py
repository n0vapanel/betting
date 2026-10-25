"""Command and callback handlers for the Telegram betting bot.

The bot lets a user browse upcoming games and their odds.  `/start`
shows a sport picker; picking a sport lists its leagues that are in
season (or jumps straight to the games when only one is); picking a
league lists its next games with the latest moneyline odds from The
Odds API.  It also defines `/help` and a catch‑all for unknown commands.

Every step is driven by the button's callback data, so menus keep
working after the bot restarts.  The Odds API client and display
timezone live in ``application.bot_data``; API calls run in a worker
thread so a slow response doesn't stall other chats.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from ..config import Config
from ..data.apis import TheOddsApiClient
from ..data.leagues import SPORTS, pick_leagues

MAX_GAMES = 10

# Telegram rejects text messages longer than this, counted in UTF-16 code units.
TELEGRAM_MESSAGE_LIMIT = 4096


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the sport picker."""
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"sport:{group}") for label, group in SPORTS[i:i + 2]]
        for i in range(0, len(SPORTS), 2)
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_message = (
        "Welcome to the Sports Betting Bot!\n"
        "Pick a sport to see upcoming games and odds."
    )
    if update.message:
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide a list of available commands and brief instructions."""
    help_text = (
        "Available commands:\n"
        "/start – browse upcoming games and odds\n"
        "/help – display this help message\n\n"
        "After /start, pick a sport, then a league, to see its next games "
        "with the latest moneyline odds."
    )
    if update.message:
        await update.message.reply_text(help_text)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any unknown commands."""
    if update.message:
        await update.message.reply_text(
            "Sorry, I didn't understand that command. Type /help for assistance."
        )


async def expired_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer taps on buttons from older versions of the menu."""
    if update.callback_query:
        try:
            await update.callback_query.answer(
                "This menu has expired. Send /start to begin again.", show_alert=True
            )
        except BadRequest:
            pass  # Tapped too long ago for Telegram to accept an answer


async def sport_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List the sport's leagues in season, or its games if only one league is."""
    query = update.callback_query
    await query.answer()
    group = query.data.removeprefix("sport:")
    client: Optional[TheOddsApiClient] = context.application.bot_data["odds_client"]
    if client is None:
        await query.message.reply_text(
            "Showing games needs a The Odds API key (free at the-odds-api.com). "
            "Add THE_ODDS_API_KEY to the bot's .env and restart it."
        )
        return
    sports = await asyncio.to_thread(client.get_sports)
    if not sports:
        await query.message.reply_text(
            "Couldn't reach the odds feed. Check THE_ODDS_API_KEY, then try /start again."
        )
        return
    leagues = pick_leagues(sports, group)
    if not leagues:
        await query.message.reply_text(
            f"No {group} leagues are in season right now. Send /start to pick another sport."
        )
        return
    if len(leagues) == 1:
        await send_games(query.message, context, leagues[0]["key"], leagues[0]["title"])
        return
    context.chat_data["league_titles"] = {league["key"]: league["title"] for league in leagues}
    keyboard = [
        [
            InlineKeyboardButton(league["title"], callback_data=f"league:{league['key']}")
            for league in leagues[i:i + 2]
        ]
        for i in range(0, len(leagues), 2)
    ]
    await query.message.reply_text(
        f"Pick a {group} league:", reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def league_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List the league's next games with their odds."""
    query = update.callback_query
    await query.answer()
    key = query.data.removeprefix("league:")
    # Titles are remembered from the league picker; a restart forgets them
    title = context.chat_data.get("league_titles", {}).get(key, key)
    await send_games(query.message, context, key, title)


async def send_games(
    message: Message, context: ContextTypes.DEFAULT_TYPE, key: str, title: str
) -> None:
    """Reply with the next games of league ``key`` and their moneyline odds."""
    client: TheOddsApiClient = context.application.bot_data["odds_client"]
    events = await asyncio.to_thread(
        client.get_odds, key, regions="us", markets="h2h", odds_format="american"
    )
    if not events:
        await message.reply_text(
            f"No upcoming {title} games with odds right now. Send /start to look at another league."
        )
        return
    blocks = format_games(events, title, context.application.bot_data["timezone"])
    blocks.append("Send /start to look at another league.")
    for text in split_message(blocks, separator="\n\n"):
        await message.reply_text(text)


def format_games(
    events: List[Dict[str, Any]],
    title: str,
    tz: ZoneInfo,
    now: Optional[datetime.datetime] = None,
) -> List[str]:
    """Return a header plus one text block per game (soonest first) with its moneyline odds."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    timed = []
    for event in events:
        start = event.get("start_time")
        if start:
            # fromisoformat() only accepts a trailing "Z" from Python 3.11
            timed.append((datetime.datetime.fromisoformat(start.replace("Z", "+00:00")), event))
    timed.sort(key=lambda pair: pair[0])
    games = timed[:MAX_GAMES]

    count = "next game" if len(games) == 1 else f"next {len(games)} games"
    blocks = [f"{title}: {count} (moneyline odds)"]
    for start, event in games:
        local = start.astimezone(tz)
        when = (
            "🔴 Live now" if start <= now
            else f"{local:%a %b} {local.day}, {local.hour % 12 or 12}:{local:%M %p %Z}"
        )
        home, away = event.get("home_team"), event.get("away_team")
        prices = event.get("markets", {}).get("h2h", {})
        # Away, home, then anything else the market offers (e.g. "Draw" in soccer)
        names = [away, home] + [name for name in prices if name not in (away, home)]
        odds = " · ".join(f"{name} {prices[name]:+.0f}" for name in names if name in prices)
        if odds and event.get("bookmaker"):
            odds = f"{odds} ({event['bookmaker']})"
        blocks.append(f"{when}\n{away} @ {home}\n{odds or 'Odds not posted yet'}")
    return blocks


def split_message(
    lines: List[str], limit: int = TELEGRAM_MESSAGE_LIMIT, separator: str = "\n"
) -> List[str]:
    """Join ``lines`` into as few messages as fit Telegram's length limit, never splitting a line."""
    messages: List[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}{separator}{line}" if current else line
        if current and len(candidate.encode("utf-16-le")) // 2 > limit:
            messages.append(current)
            current = line
        else:
            current = candidate
    if current:
        messages.append(current)
    return messages


def setup_handlers(application: Application, config: Config) -> None:
    """Register command and button handlers with the application.

    Creates The Odds API client (when a key is configured) and stores it,
    with the display timezone, in ``application.bot_data``.
    """
    application.bot_data["config"] = config
    application.bot_data["odds_client"] = (
        TheOddsApiClient(config.odds_api_key) if config.odds_api_key else None
    )
    application.bot_data["timezone"] = ZoneInfo(config.timezone)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(sport_selected, pattern=r"^sport:"))
    application.add_handler(CallbackQueryHandler(league_selected, pattern=r"^league:"))
    # Any other button comes from an older version of the menu
    application.add_handler(CallbackQueryHandler(expired_menu))
    application.add_handler(CommandHandler("help", help_command))
    # Catch all other commands with the unknown command handler
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
