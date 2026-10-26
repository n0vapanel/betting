"""AI chat in Telegram: plain messages to the bot go to the assistant (ai/assistant.py).

The chat is private: only people in TELEGRAM_ALLOWED_IDS (the owner and
friends) can use it, since every message costs money on the owner's API
account; anyone else gets a short note pointing them to /start, where
browsing games stays open to all.

While the assistant works, the person sees what it is doing: a message
draft (Telegram's streaming preview, Bot API 9.3+) that starts as Telegram's
own "Thinking…" and is updated at most once a second with progress lines
("Finding NBA games…", "Pricing 3 legs…"). A draft is a 30-second preview, so
it is sent again every 20 seconds while a slow step runs, and the answer
always arrives as a real message. If drafts aren't available, a "Thinking…"
message is edited in place instead and becomes the answer.

The answer carries the assistant's text, its cards (tickets, plans, players)
written out by code, the follow-up chips as buttons, "Add to slip" for each
ticket the bot's slip can hold (only the person adds legs, by tapping) and
"📱 Open in app" when PUBLIC_URL is an HTTPS address. The bot's slip is one
parlay, so singles, round robins and system bets are added in the app, where
they keep their shape. Buttons hold the thread, the answer's place in it and
a position (well under Telegram's 64 bytes); what they mean is read back
from that stored answer, so a button under an older answer acts on that
answer, and they keep working after a restart.
"""

from __future__ import annotations

import asyncio
import html
import logging
import random
import time
from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.constants import ChatType, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from ..ai.assistant import build_assistant
from ..ai.explain import money, one_in_text, pct
from ..config import Config
from ..dashboard.service import CallRejected, NotFound

THINKING = "🤔 Thinking…"
DRAFT_EVERY = 1.0  # seconds between progress updates (Telegram rate-limits edits)
DRAFT_KEEPALIVE = 20.0  # a draft is shown for about 30 seconds, so a quiet one is sent again this often
PROGRESS_LINES = 4
SESSION_GAP = 6 * 3600  # a chat quiet this long starts a new thread, with the AI disclosure again
MAX_TEXT = 4096  # Telegram's message limit
MIN_TEXT_ROOM = 600  # characters kept for the AI's words before a card is dropped to make room
MAX_CARDS = 3
MAX_CARD_LEGS = 8
PRIVATE_MESSAGE = "AI chat is private for now; tap /start to browse games."
NO_AI_MESSAGE = "The AI chat isn't switched on here (the server needs ANTHROPIC_API_KEY). Tap /start to browse games."
ASK_PROMPT = ("💬 <b>Ask AI</b>\nType your question as a normal message, for example:\n"
              "• <i>Who does the AI favour in the NBA tonight?</i>\n"
              "• <i>NBA, $10 to get about $50 back, max 3 legs</i>\n"
              "• <i>What are the real odds of a 5-day rollover at 2.0?</i>\n\n"
              "Our code works out every number; the AI explains them.")
AGE_BUTTON = "✅ I'm 18+ (21+ in the US)"
EXPIRED = "This button has expired: type your question instead."
NOT_ADDABLE = ("This ticket is {title}: the bot's slip holds one parlay, so it would become a different bet. "
               "Add it in the app (📱 Open in app), where it keeps its shape.")


def build_chat_assistant(config: Config, service: Any) -> Optional[Any]:
    """The assistant for the bot's chat: the one the games service's Lab already has (services.py builds
    it), else a new one; None without ANTHROPIC_API_KEY or without the games service."""
    if service is None or not config.anthropic_api_key:
        return None
    lab = getattr(service, "lab", None)
    if lab is not None and getattr(lab, "assistant", None) is not None:
        return lab.assistant
    if getattr(service, "store", None) is None:
        return None
    if lab is None:
        try:
            from ..dashboard.lab import Lab
        except ImportError:
            logging.warning("The AI chat needs dashboard/lab.py; it is switched off")
            return None
        lab = Lab(service)
    assistant = build_assistant(config, lab, service.store)
    lab.assistant = lab.assistant or assistant
    return assistant


def _uid(update: Update) -> str:
    return f"tg:{update.effective_user.id}"


def _allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    config = context.application.bot_data.get("config")
    user = update.effective_user
    return bool(user and config and user.id in config.telegram_user_ids)


# --- writing the answer -----------------------------------------------------------------------------

def addable(card: Dict[str, Any]) -> bool:
    """Whether the bot's slip (one parlay) can hold a ticket as it is: a multi, or a single leg."""
    refs = card.get("refs") or []
    return card.get("type") == "ticket" and bool(refs) and (card.get("ticket_type") == "multi" or len(refs) == 1)


def _ticket_text(card: Dict[str, Any]) -> str:
    esc = html.escape
    ticket = card.get("ticket") or {}
    lines = [f"🧾 <b>{esc(str(card.get('title') or 'Ticket'))}</b>"]
    for leg in (ticket.get("legs") or [])[:MAX_CARD_LEGS]:
        odds = (leg.get("odds") or {}).get("decimal")
        chance = f" · AI {pct(leg['ai_prob'])}" if leg.get("ai_prob") is not None else ""
        lines.append(f"• {esc(str(leg.get('label')))} {odds:.2f}{chance}" if isinstance(odds, (int, float))
                     else f"• {esc(str(leg.get('label')))}")
    stake, payout = ticket.get("total_stake"), ticket.get("max_payout")
    if stake and payout:
        odds = (ticket.get("odds") or {}).get("decimal")
        at = f" (odds {odds:.2f})" if isinstance(odds, (int, float)) else f" ({ticket.get('bets')} bets)"
        lines.append(f"Stake {money(stake)} → up to {money(payout)}{at}")
    chance = ticket.get("chance") or {}
    views = []
    if chance.get("ai_one_in"):
        views.append(f"AI {one_in_text(chance['ai_one_in'])} ({pct(chance.get('ai'))})")
    if chance.get("market_one_in"):
        views.append(f"market {one_in_text(chance['market_one_in'])} ({pct(chance.get('market'))})")
    if views:
        what = "Chance of a profit" if chance.get("headline") == "profit" else "Chance it wins"
        lines.append(f"{what}: " + " · ".join(views))
    expected = ticket.get("expected") or {}
    money_bits = []
    cut = ticket.get("book_cut")
    if cut is not None:
        # Each leg at its best bookmaker can beat the consensus; then there is no cut to show
        money_bits.append(f"Bookmaker keeps ~{pct(cut, 1)}" if cut >= 0 else
                          f"Best prices beat the market's view by {pct(-cut, 1)}")
    basis = "ai" if expected.get("ai") is not None else "market"
    if expected.get(basis) is not None:
        money_bits.append(f"expected result {money(expected[basis])} ({'AI' if basis == 'ai' else 'market'})")
    if money_bits:
        line = " · ".join(money_bits)
        lines.append(line[0].upper() + line[1:])
    most = ticket.get("most_likely") or {}
    if most.get("result") and most.get("probability") is not None:
        lines.append(f"Most likely: {esc(str(most['result']))} ({pct(most['probability'])})")
    if ticket.get("reasons"):
        lines.append(f"⚠️ {esc(str(ticket['reasons'][0]))}")
    if card.get("refs") and not addable(card):
        lines.append("📱 Singles, round robins and system bets go on the app's slip (the bot's slip is one parlay).")
    return "\n".join(lines)


def _plan_text(card: Dict[str, Any]) -> str:
    esc = html.escape
    compare = card.get("compare") or {}
    lines = [f"📈 <b>{esc(str(card.get('title') or 'Rollover plan'))}</b>"]
    if compare.get("summary"):
        lines.append(esc(str(compare["summary"])))
    for entry in (compare.get("strategies") or {}).values():
        stats = entry.get("market") or {}
        if stats:
            lines.append(f"• {esc(str(entry.get('title')))}: finishes {pct(stats.get('chance_complete'))}, "
                         f"profit {pct(stats.get('chance_profit'))}, expected cost {money(stats.get('expected_cost'))}")
    return "\n".join(lines)


def _player_text(card: Dict[str, Any]) -> str:
    esc = html.escape
    about = ", ".join(str(part) for part in (card.get("team"), card.get("position")) if part)
    lines = [f"👤 <b>{esc(str(card.get('title')))}</b>" + (f" ({esc(about)})" if about else "")]
    if card.get("status") or card.get("injury"):
        lines.append(f"Status: {esc(str(card.get('injury') or card.get('status')))}")
    last10 = card.get("last10") or {}
    shown = [f"{key} {value:g}" for key, value in list(last10.items())[:5] if isinstance(value, (int, float))]
    if shown:
        games = (card.get("sample") or {}).get("last10")
        lines.append(f"Last {games or 10}: " + ", ".join(shown))
    return "\n".join(lines)


def card_text(card: Dict[str, Any]) -> str:
    """One card from the assistant as Telegram HTML (the numbers are the code's, as priced)."""
    kind = card.get("type")
    if kind == "ticket":
        return _ticket_text(card)
    if kind == "plan":
        return _plan_text(card)
    if kind == "player":
        return _player_text(card)
    return ""


def reply_text(result: Dict[str, Any], asked: Optional[str] = None) -> str:
    """The whole answer as Telegram HTML, cut to fit one message."""
    reply = result.get("reply") or {}
    head = []
    if result.get("disclosure"):
        head.append(f"<i>🤖 {html.escape(result['disclosure'])}</i>")
    if asked:
        head.append(f"<i>› {html.escape(asked)}</i>")
    cards = [card_text(card) for card in (reply.get("cards") or [])[:MAX_CARDS]]
    credits = result.get("credits") or {}
    foot = []
    if credits.get("daily_cents"):
        foot.append(f"<i>AI analyst · {credits.get('left_cents', 0):g}¢ of {credits['daily_cents']}¢ "
                    "of today's AI credits left</i>")
    body = str(reply.get("text") or "")
    for keep in range(len(cards), -1, -1):  # the cards are the code's numbers: cut the text before them
        room = MAX_TEXT - len("\n\n".join(head + [""] + cards[:keep] + foot))
        if room >= MIN_TEXT_ROOM or len(html.escape(body)) <= room or (keep == 0 and room > 0):
            return "\n\n".join(head + [_fit(body, room)] + cards[:keep] + foot)
    return _fit(body, MAX_TEXT)


def _fit(text: str, room: int) -> str:
    """``text`` escaped for HTML and cut (at a character, never inside an entity) to ``room`` characters."""
    if len(html.escape(text)) <= room:
        return html.escape(text)
    cut = text[: max(0, room - 1)]
    while cut and len(html.escape(cut)) > room - 1:
        cut = cut[: -max(1, len(cut) // 20)]
    return html.escape(cut.rstrip()) + "…"


def reply_markup(result: Dict[str, Any], config: Optional[Config], private: bool) -> Optional[InlineKeyboardMarkup]:
    """The answer's buttons. Each names the thread and this answer's place in it (``message_idx``), so a tap
    acts on this answer even after newer ones arrive."""
    reply = result.get("reply") or {}
    thread, at = result.get("thread_id"), result.get("message_idx")
    rows: List[List[InlineKeyboardButton]] = []
    if thread is not None and at is not None:
        cards = (reply.get("cards") or [])[:MAX_CARDS]
        tickets = [i for i, card in enumerate(cards) if card.get("type") == "ticket"]
        for n, index in enumerate(tickets, 1):
            if not addable(cards[index]):
                continue  # added in the app instead (its card says so)
            label = "➕ Add to slip" if len(tickets) == 1 else f"➕ Add ticket {n} to slip"
            rows.append([InlineKeyboardButton(label, callback_data=f"ai:add:{thread}:{at}:{index}")])
        chips = [InlineKeyboardButton(chip[:60], callback_data=f"ai:{thread}:{at}:{i}")
                 for i, chip in enumerate((reply.get("chips") or [])[:4])]
        rows += [chips[i:i + 2] for i in range(0, len(chips), 2)]
    url = getattr(config, "mini_app_url", None)
    if url and private:  # Telegram refuses Mini App buttons outside private chats
        rows.append([InlineKeyboardButton("📱 Open in app", web_app=WebAppInfo(f"{url}/#ask"))])
    return InlineKeyboardMarkup(rows) if rows else None


# --- progress while the assistant works -----------------------------------------------------------

class Progress:
    """What the assistant is doing, shown as a draft (or an edited "Thinking…" message) until the answer."""

    def __init__(self, bot: Any, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.draft_id = random.randint(1, 2 ** 31 - 1)  # any non-zero id; the same id animates the updates
        self.lines: List[str] = []
        self.dirty = False
        self.message: Any = None
        self.task: Optional[asyncio.Task] = None
        self.last_sent = time.monotonic()

    async def start(self) -> None:
        try:
            await self.bot.send_message_draft(self.chat_id, self.draft_id, "")  # Telegram shows "Thinking…"
        except (TelegramError, AttributeError, TypeError) as exc:
            logging.info("No message draft (%s); showing a Thinking… message instead", exc)
            self.message = await self.bot.send_message(self.chat_id, THINKING)
        self.last_sent = time.monotonic()
        self.task = asyncio.create_task(self._pump())

    def push(self, line: str) -> None:
        if line and (not self.lines or self.lines[-1] != line):
            self.lines.append(line)
            self.dirty = True

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(DRAFT_EVERY)
            # A draft disappears about 30 seconds after its last update: a long model call with no new
            # line would leave the chat looking dead, so the same draft is sent again
            quiet = self.message is None and time.monotonic() - self.last_sent >= DRAFT_KEEPALIVE
            if self.dirty or quiet:
                self.dirty = False
                await self._show()

    async def _show(self) -> None:
        text = "\n".join(self.lines[-PROGRESS_LINES:])
        self.last_sent = time.monotonic()
        try:
            if self.message is None:
                await self.bot.send_message_draft(self.chat_id, self.draft_id, text)
            else:
                await self.message.edit_text(f"{THINKING}\n{text}")
        except TelegramError:
            pass  # progress is a nicety; the answer still comes

    async def finish(self, text: str, markup: Optional[InlineKeyboardMarkup]) -> None:
        if self.task:
            self.task.cancel()
        if self.message is not None:
            try:
                await self.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=markup,
                                             disable_web_page_preview=True)
                return
            except TelegramError:
                logging.warning("Couldn't turn the Thinking… message into the answer; sending it instead")
        await self.bot.send_message(self.chat_id, text, parse_mode=ParseMode.HTML, reply_markup=markup,
                                    disable_web_page_preview=True)


# --- handlers --------------------------------------------------------------------------------------

def _thread(context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """The person's current chat thread, or None to start a new one after a quiet spell."""
    thread = context.user_data.get("ai_thread")
    if thread and time.time() - context.user_data.get("ai_last", 0) < SESSION_GAP:
        return int(thread)
    return None


async def run_turn(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str,
                   thread_id: Optional[int] = None, asked: Optional[str] = None) -> None:
    """Send one message to the assistant and post its answer, with progress while it works."""
    assistant = context.application.bot_data.get("assistant")
    chat = update.effective_chat
    user = update.effective_user
    uid = _uid(update)
    name = (user.first_name or user.username or "Player")[:24]
    await asyncio.to_thread(assistant.store.get_or_create_user, uid, name)
    progress = Progress(context.bot, chat.id)
    await progress.start()
    loop = asyncio.get_running_loop()

    def on_step(line: str) -> None:  # called from the worker thread
        loop.call_soon_threadsafe(progress.push, line)

    thread_id = thread_id if thread_id is not None else _thread(context)
    try:
        try:
            result = await asyncio.to_thread(assistant.reply, uid, thread_id, text, None, None, on_step)
        except NotFound:  # the thread is gone (a new database): start a fresh one
            result = await asyncio.to_thread(assistant.reply, uid, None, text, None, None, on_step)
    except CallRejected as exc:
        await progress.finish(html.escape(str(exc)), None)
        return
    except Exception:
        logging.exception("The AI chat failed")
        await progress.finish("Something went wrong on my side. Try again in a minute.", None)
        return
    context.user_data["ai_thread"] = result["thread_id"]
    context.user_data["ai_last"] = time.time()
    config = context.application.bot_data.get("config")
    markup = reply_markup(result, config, chat.type == ChatType.PRIVATE)
    if (result.get("reply") or {}).get("kind") == "gate":
        account = await asyncio.to_thread(assistant.store.user, uid)
        if account and not account.get("age_confirmed_at"):
            age = [InlineKeyboardButton(AGE_BUTTON, callback_data="ai:age")]
            markup = InlineKeyboardMarkup([age] + list(markup.inline_keyboard if markup else []))
    await progress.finish(reply_text(result, asked), markup)


async def chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A plain text message to the bot: the AI chat for the owner's friends, a pointer to /start for others."""
    message = update.effective_message
    if message is None or not message.text or update.effective_user is None:
        return
    if not _allowed(update, context):
        await message.reply_text(PRIVATE_MESSAGE)
        return
    if context.application.bot_data.get("assistant") is None:
        await message.reply_text(NO_AI_MESSAGE)
        return
    await run_turn(update, context, message.text)


async def _stored_reply(context: ContextTypes.DEFAULT_TYPE, uid: str, thread_id: int, index: int) -> Dict[str, Any]:
    """The answer a button was sent under: message ``index`` of one of the person's own threads, or {} when
    it isn't there or isn't an answer (the button has expired)."""
    assistant = context.application.bot_data.get("assistant")
    rows = await asyncio.to_thread(assistant.store.messages, uid, thread_id)
    row = next((row for row in rows if row.get("idx") == index), None)
    if row and row["role"] == "assistant" and isinstance(row["content"], dict):
        return row["content"]
    return {}


async def _check_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    query = update.callback_query
    if not _allowed(update, context):
        await query.answer(PRIVATE_MESSAGE, show_alert=True)
        return False
    if context.application.bot_data.get("assistant") is None:
        await query.answer(NO_AI_MESSAGE, show_alert=True)
        return False
    return True


async def chip_tapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A follow-up chip: sent to the assistant as if the person had typed it."""
    query = update.callback_query
    if not await _check_button(update, context):
        return
    _, thread, at, index = query.data.split(":")
    stored = await _stored_reply(context, _uid(update), int(thread), int(at))
    chips = stored.get("chips") or []
    if int(index) >= len(chips):
        await query.answer(EXPIRED, show_alert=True)
        return
    await query.answer()
    await run_turn(update, context, chips[int(index)], thread_id=int(thread), asked=chips[int(index)])


async def add_to_slip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Put a ticket's legs on the bot's multi slip: the person's own tap, never the AI's.

    Only the ticket under that button, and only as the bet it is: the bot's slip is one parlay, so N singles,
    round robins and system bets are pointed to the app instead of turning into a parlay here.
    """
    query = update.callback_query
    if not await _check_button(update, context):
        return
    _, _, thread, at, index = query.data.split(":")
    stored = await _stored_reply(context, _uid(update), int(thread), int(at))
    cards = stored.get("cards") or []
    card = cards[int(index)] if int(index) < len(cards) else None
    refs = [ref for ref in (card or {}).get("refs") or []
            if ref.get("market") in ("h2h", "spreads", "totals") and ref.get("id")]
    if not card or not refs:
        await query.answer("This ticket isn't available any more.", show_alert=True)
        return
    if not addable(card):
        await query.answer(NOT_ADDABLE.format(title=str(card.get("title") or "not a parlay"))[:200], show_alert=True)
        return
    picks = context.user_data.setdefault("picks", [])
    added = 0
    for ref in refs:
        pick = {"league": ref["league"], "id": ref["id"], "market": ref["market"], "side": ref["side"]}
        if not any(p.get("league") == pick["league"] and p.get("id", "")[:16] == pick["id"][:16]
                   and p.get("market") == pick["market"] and p.get("side") == pick["side"] for p in picks):
            picks.append(pick)
            added += 1
    context.user_data["mode"] = "multi"
    await query.answer(f"Added {added} leg(s) to your slip ✓ ({len(picks)} in all, priced as one parlay). "
                       "Send /slip to see it.", show_alert=True)


async def confirm_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not await _check_button(update, context):
        return
    assistant = context.application.bot_data.get("assistant")
    await asyncio.to_thread(assistant.store.get_or_create_user, _uid(update))
    await asyncio.to_thread(assistant.store.confirm_age, _uid(update))
    await query.answer("Thanks!")
    try:
        await query.edit_message_text("✅ Thanks. Ask me anything: type a message.")
    except TelegramError:
        pass


async def ask_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The home screen's "Ask AI" button: how to use the chat (or why it's closed)."""
    query = update.callback_query
    if not _allowed(update, context):
        await query.answer(PRIVATE_MESSAGE, show_alert=True)
        return
    if context.application.bot_data.get("assistant") is None:
        await query.answer(NO_AI_MESSAGE, show_alert=True)
        return
    await query.answer()
    rows = [[InlineKeyboardButton("« Home", callback_data="home")]]
    url = getattr(context.application.bot_data.get("config"), "mini_app_url", None)
    if url and update.effective_chat and update.effective_chat.type == ChatType.PRIVATE:
        rows.insert(0, [InlineKeyboardButton("📱 Open in app", web_app=WebAppInfo(f"{url}/#ask"))])
    try:
        await query.edit_message_text(ASK_PROMPT, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(rows))
    except TelegramError:
        pass


def setup_chat(application: Application, config: Config) -> None:
    """Register the AI chat: free text in private chats, its buttons and the home screen's Ask AI button.

    Call before the catch-all for expired buttons (it must stay last).
    """
    application.bot_data["assistant"] = build_chat_assistant(config, application.bot_data.get("service"))
    application.add_handler(CallbackQueryHandler(ask_ai, pattern=r"^ask$"))
    application.add_handler(CallbackQueryHandler(confirm_age, pattern=r"^ai:age$"))
    # ai:add:<thread>:<answer>:<card> and ai:<thread>:<answer>:<chip>; older two-number buttons fall
    # through to the catch-all, which says the menu has expired
    application.add_handler(CallbackQueryHandler(add_to_slip, pattern=r"^ai:add:\d{1,12}:\d{1,6}:\d$"))
    # A chat turn takes seconds: block=False lets the bot keep serving everyone else's taps meanwhile
    application.add_handler(CallbackQueryHandler(chip_tapped, pattern=r"^ai:\d{1,12}:\d{1,6}:\d$", block=False))
    # New messages only: editing a sent message shouldn't start a second (paid) turn
    text = filters.UpdateType.MESSAGE & filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    application.add_handler(MessageHandler(text, chat_message, block=False))
