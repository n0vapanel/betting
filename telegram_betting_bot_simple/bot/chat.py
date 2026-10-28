"""AI chat in Telegram: plain messages to the bot go to the assistant (ai/assistant.py).

The chat is private: only people in TELEGRAM_ALLOWED_IDS (the owner and
friends) can use it, since every message costs money on the owner's API
account; anyone else gets a short note pointing them to /start, where
browsing games stays open to all. A message with a warning sign (chasing
losses, betting to pay debts, distress) gets the support message and the
helplines from anyone, AI or no AI: code's own rules spot it, for free.

While the assistant works, the person sees what it is doing: a message
draft (Telegram's streaming preview, Bot API 9.3+) that starts as Telegram's
own "Thinking…" and is updated at most once a second with progress lines
("Finding NBA games…", "Pricing 3 legs…"). A draft is a 30-second preview, so
it is sent again every 20 seconds while a slow step runs, and the answer
always arrives as a real message. If drafts aren't available, a "Thinking…"
message is edited in place instead and becomes the answer.

A bet slip can be pasted here too: a photo of it, a screenshot sent as an
image file, or its text (a message that looks like a slip). The assistant
reads it (ai/slip_reader.py), the Lab matches and prices it, and the answer
starts with what was read, then the code's numbers and the AI's words; a
doubtful read asks "Is that right?" first. Without the AI (no API key) the
free code-only reader still reads pasted text. Images are never stored.

The answer carries the assistant's text, its cards (tickets, plans, players,
pasted slips) written out by code, the follow-up chips as buttons, "Add to
slip" for each ticket the bot's slip can hold (only the person adds legs, by
tapping) and "📱 Open in app" when PUBLIC_URL is an HTTPS address. The bot's slip is one
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

from ..ai.assistant import (
    HELPLINE,
    Assistant,
    build_assistant,
    note_warning_sign,
    pasted_template,
    reading_text,
    risk_flags,
    support_message,
)
from ..ai.explain import (FORM_WORDS, first_reason, money, one_in_text, pct, relaxed_texts, scan_loading_text,
                          scan_matchup)
from ..ai.slip_reader import MAX_IMAGE_BYTES
from ..ai.tools import pasted_card, pasted_label
from ..config import Config
from ..dashboard.service import CallRejected, NotFound
from ..engine.pasted import NO_LEG_ODDS

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
              "• <i>Players projected for 20+ points tonight</i>\n"
              "• <i>NBA, $10 to get about $50 back, max 3 legs</i>\n"
              "• <i>What are the real odds of a 5-day rollover at 2.0?</i>\n\n"
              "Our code works out every number; the AI explains them.")
NO_AI_IMAGE = ("Reading a screenshot needs the AI chat, which isn't switched on here. Paste the slip's text "
               "instead and the free reader will read it.")
TOO_BIG = "That image is too big to read: send it as a photo (Telegram shrinks it), or crop the screenshot."
NOT_AN_IMAGE = "Send the slip as a photo, or as a JPEG, PNG or WebP screenshot."
# The most an image may be before it is downloaded at all; Claude reads up to MAX_IMAGE_BYTES of it, so a
# bigger screenshot file is turned away (a photo is shrunk by Telegram and always fits)
MAX_DOWNLOAD = 5 * 1024 * 1024
PHOTO_EDGE = 1600  # the largest photo size Telegram offers up to this many pixels on the long side is read
FREE_READER_NOTE = "The AI chat isn't switched on here, so the app's free reader (code only, no AI) read it."
LEG_MARKS = {"matched": "✓", "line_differs": "≈", "not_found": "?", "prop": "👤", "unsupported": "?",
             "no_odds": "⚠️"}
AGE_BUTTON = "✅ I'm 18+ (21+ in the US)"
FREE_AGE = "Before I read slips, please confirm you're 18 or older (21+ in the US)."
FREE_AGE_DONE = "✅ Thanks. Send the slip's text again and the free reader will read it."
FREE_CHECK = ("If that's what your slip says, tap ✅ Yes, that's right for the numbers; if not, send the corrected "
              "text.")
CONFIRM_BUTTON = "✅ Yes, that's right"
CHECK_FIRST = "🔎 The numbers come once you've checked what was read."
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


def fit_text(fit: Any) -> Optional[str]:
    """A closest-effort ticket's fit to what was asked, as the app shows it: "3.52x · 31% (you asked 3.5x · 80%)"
    (engine/builder.py's fit block: the ticket's odds and chance, and the targets given)."""
    if not isinstance(fit, dict):
        return None
    have = [f"{fit['odds']:.2f}x" if isinstance(fit.get("odds"), (int, float)) else None,
            pct(fit["chance"]) if isinstance(fit.get("chance"), (int, float)) else None]
    asked = [f"{fit['target_odds']:g}x" if isinstance(fit.get("target_odds"), (int, float)) else None,
             pct(fit["target_chance"]) if isinstance(fit.get("target_chance"), (int, float)) else None]
    have, asked = [bit for bit in have if bit], [bit for bit in asked if bit]
    if not have:
        return None
    return " · ".join(have) + (f" (you asked {' · '.join(asked)})" if asked else "")


def _ticket_text(card: Dict[str, Any]) -> str:
    esc = html.escape
    ticket = card.get("ticket") or {}
    lines = [f"🧾 <b>{esc(str(card.get('title') or 'Ticket'))}</b>"]
    # A Builder ticket picked by closest effort: how close it gets, and the rules relaxed to get there
    fit = fit_text(card.get("fit"))
    if fit:
        lines.append(f"🎯 {esc(fit)}")
    relaxed = relaxed_texts(card.get("relaxed"))
    if relaxed:
        lines.append(f"<i>Relaxed: {esc('; '.join(relaxed))}</i>")
    for leg in (ticket.get("legs") or [])[:MAX_CARD_LEGS]:
        odds = (leg.get("odds") or {}).get("decimal")
        chance = f" · AI {pct(leg['ai_prob'])}" if leg.get("ai_prob") is not None else ""
        lines.append(f"• {esc(str(leg.get('label')))} {odds:.2f}{chance}" if isinstance(odds, (int, float))
                     else f"• {esc(str(leg.get('label')))}")
    lines += _ticket_numbers(ticket)
    if card.get("refs") and not addable(card):
        lines.append("📱 Singles, round robins and system bets go on the app's slip (the bot's slip is one parlay).")
    return "\n".join(lines)


def _ticket_numbers(ticket: Dict[str, Any], currency: str = "$") -> List[str]:
    """A priced ticket's money and chances, a line each (the code's numbers, as priced)."""
    esc = html.escape
    lines = []
    stake, payout = ticket.get("total_stake"), ticket.get("max_payout")
    if stake and payout:
        odds = (ticket.get("odds") or {}).get("decimal")
        at = f" (odds {odds:.2f})" if isinstance(odds, (int, float)) else f" ({ticket.get('bets')} bets)"
        lines.append(f"Stake {esc(money(stake, currency))} → up to {esc(money(payout, currency))}{at}")
    chance = ticket.get("chance") or {}
    views = []
    if chance.get("ai_one_in"):
        views.append(f"AI {_chance_view(chance['ai_one_in'], chance.get('ai'))}")
    if chance.get("market_one_in"):
        views.append(f"market {_chance_view(chance['market_one_in'], chance.get('market'))}")
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
        money_bits.append(f"expected result {esc(money(expected[basis], currency))} "
                          f"({'AI' if basis == 'ai' else 'market'})")
    if money_bits:
        line = " · ".join(money_bits)
        lines.append(line[0].upper() + line[1:])
    most = ticket.get("most_likely") or {}
    if most.get("result") and most.get("probability") is not None:
        lines.append(f"Most likely: {esc(str(most['result']))} ({pct(most['probability'])})")
    if ticket.get("reasons"):
        lines.append(f"⚠️ {esc(str(ticket['reasons'][0]))}")
    return lines


def _pasted_text(card: Dict[str, Any]) -> str:
    """A pasted slip's card: each leg as read with how it matched, then the code's numbers at the person's
    own odds and the return check. A doubtful read the person hasn't confirmed shows only what was read (the
    numbers come once they've checked it; in the pause on picks they show, as in the app). Everything pasted
    is escaped."""
    esc = html.escape
    view = card.get("pasted") or {}
    analysis = card.get("analysis") or {}
    asking = bool(view.get("needs_confirmation") and not view.get("confirmed") and not analysis.get("paused_until"))
    legs = view.get("legs") or []
    currency = str(view.get("currency") or "$")
    unit = currency if len(currency) <= 1 else f"{currency} "
    kind = str(view.get("bet_type") or "").replace("_", " ")
    head = ["Your slip"] + [str(part) for part in (view.get("bookmaker"), kind if kind != "unknown" else None) if part]
    if view.get("stake"):
        head.append(f"{unit}{view['stake']} stake")
    lines = [f"📋 <b>{esc(' · '.join(head))}</b>"]
    dropped = set(analysis.get("drop") or [])
    for leg in legs[:MAX_CARD_LEGS]:
        match = leg.get("match") or {}
        bits = [f"{LEG_MARKS.get(str(match.get('status')), '?')} {esc(pasted_label(leg))}",
                esc(str(leg.get("odds_text") or "odds unread"))]
        chance = leg.get("market_chance") if leg.get("market_chance") is not None else leg.get("ai_chance")
        if chance is not None and not asking:  # a doubtful read shows what was read only
            bits.append(pct(chance) + (" (estimated)" if leg.get("chance_source") == "estimated" else ""))
        best = leg.get("best_price") or {}
        if leg.get("better_price") and best.get("decimal") and not asking:
            bits.append(f"better price {esc(str(best['decimal']))}" + (f" at {esc(str(best['book']))}"
                                                                       if best.get("book") else ""))
        line = "• " + " · ".join(bits)
        lines.append(f"<s>{line}</s> (left out)" if leg.get("index") in dropped else line)
    if len(legs) > MAX_CARD_LEGS:
        lines.append(f"… and {len(legs) - MAX_CARD_LEGS} more legs (📱 open the app to see them all)")
    if asking:
        lines.append(CHECK_FIRST)
        return "\n".join(lines)
    lines += _ticket_numbers(analysis.get("ticket") or {}, unit)
    stated, computed = view.get("stated_return"), view.get("computed_return")
    if view.get("return_check") == "matches" and stated:
        lines.append(f"✓ Matches your bookmaker's {esc(unit)}{esc(str(stated))}")
    elif view.get("return_check") == "differs" and stated and computed:
        lines.append(f"⚠️ Your bookmaker says {esc(unit)}{esc(str(stated))}; we get {esc(unit)}{esc(str(computed))}: "
                     "check the odds")
    for warning in (view.get("warnings") or [])[:2]:
        lines.append(f"⚠️ {esc(str(warning))}")
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


MAX_PICKS_SHOWN = 5


def _chance_view(one_in: Any, p: Any) -> str:
    """A ticket's chance as "1 in 3 (29%)" under even, else just "66%" ("1 in X" is 1/p rounded, so anything from about
    40% to 66% is "1 in 2", which beside 66% reads as 50%)."""
    if isinstance(p, (int, float)) and not isinstance(p, bool) and p >= 0.5:
        return pct(p)
    return f"{one_in_text(one_in)} ({pct(p)})"


def _picks_text(card: Dict[str, Any]) -> str:
    """The likeliest winners (brain_top_picks' card), ranked: each pick's chance, price and bookmaker and why, in a
    few words (the first reason that isn't its chance again). Each has its own "Add pick" button."""
    esc = html.escape
    lines = [f"🏆 <b>{esc(str(card.get('title') or 'Most likely winners'))}</b>"]
    for i, pick in enumerate((card.get("picks") or [])[:MAX_PICKS_SHOWN], 1):
        bits = [f"{pick.get('rank') or i}. {esc(str(pick.get('label') or 'A pick'))}"]
        if isinstance(pick.get("chance"), (int, float)):
            bits.append(pct(pick["chance"]))
        if pick.get("price"):
            try:
                price = f"{float(pick['price']):.2f}"  # "1.7" reads as 1.70, as the app shows it
            except (TypeError, ValueError):
                price = str(pick["price"])
            bits.append(esc(price) + (f" at {esc(str(pick['book']))}" if pick.get("book") else ""))
        lines.append(" · ".join(bits))
        first = first_reason(pick.get("why"))
        if first:
            lines.append(f"   <i>{esc(first)}</i>")
    return "\n".join(lines)


MAX_SCAN_SHOWN = 5  # a player scan's rows written out (the app's card has them all)
MAX_SCAN_VIEW = 3  # "View player" buttons under a scan: its top rows
SCAN_PLAYER_GONE = "This player isn't available any more."


def _decimal_price(price: Any) -> Optional[str]:
    """A price as decimal odds to 2 places ("1.87"), from 1.87, "1.87" or {"decimal": 1.87}; None without one."""
    if isinstance(price, dict):
        price = price.get("decimal")
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    return f"{value:.2f}" if value > 1 else None


def scan_line_text(line: Any) -> Optional[str]:
    """A scan row's posted line and price: "o24.5 1.87 · u24.5 1.95 (DraftKings)", "20+ 1.62" for a milestone,
    "yes 2.10" for a yes/no prop; "(may be out of date)" when the line couldn't refresh."""
    if not isinstance(line, dict):
        return None
    number = line.get("line")
    bits = []
    over, under, yes = _decimal_price(line.get("over")), _decimal_price(line.get("under")), _decimal_price(line.get("yes"))
    shown = f"{number:g}" if isinstance(number, (int, float)) and not isinstance(number, bool) else None
    if line.get("kind") == "milestone" and isinstance(number, (int, float)) and (over or yes):
        reached = line.get("label") or f"{number + 0.5:g}+"  # a 2.5 milestone is "3+"
        bits.append(f"{reached} {over or yes}")
    else:
        if over and shown:
            bits.append(f"o{shown} {over}")
        if under and shown:
            bits.append(f"u{shown} {under}")
        if yes and not bits:
            bits.append(f"yes {yes}")
    if not bits:
        return None
    text = " · ".join(bits) + (f" ({line['book']})" if line.get("book") else "")
    return text + (" (may be out of date)" if line.get("stale") else "")


def _scan_row_text(row: Dict[str, Any], number: int, under: bool) -> List[str]:
    esc = html.escape
    head = f"{row.get('rank') or number}. <b>{esc(str(row.get('name') or 'A player'))}</b>"
    matchup = scan_matchup(row)
    if matchup:
        head += f" ({esc(matchup)})"
    status = str(row.get("status") or "")
    if status and status.lower() not in ("active", "healthy", "available"):
        head += f" · {esc(status)}"
    bits = []
    if isinstance(row.get("projection"), (int, float)):
        bits.append(f"{row['projection']:.1f} proj")
    if isinstance(row.get("chance"), (int, float)):
        bits.append(pct(row["chance"]) + (" to stay under" if under else ""))
    hits = row.get("hits") or {}
    for window, name in (("l10", "L10"), ("season", "season")):
        pair = hits.get(window) if isinstance(hits, dict) else None
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[1]:
            bits.append(f"{pair[0]:g}/{pair[1]:g} {name}")
    line = scan_line_text(row.get("line"))
    if line:
        bits.append(esc(line))
    lines = [head]
    if bits:
        lines.append("   " + " · ".join(bits))
    first = first_reason(row.get("why"))
    notes = [esc(first)] if first else []
    if row.get("model") == "form":
        notes.append(FORM_WORDS)
    if notes:
        lines.append(f"   <i>{' · '.join(notes)}</i>")
    return lines


def _scan_text(card: Dict[str, Any]) -> str:
    """A player scan (players_scan's card), as a numbered list strongest first: each player's matchup, status, the
    projection, the chance, the L10 and season hits and the posted line and price, a short why, and "from recent
    form" for the form model's rows; then who is close and, while the board builds, how many games are loaded.
    The first rows get "View player" buttons (reply_markup); props aren't added to the slip."""
    esc = html.escape
    under = card.get("side") == "under"
    lines = [f"📊 <b>{esc(str(card.get('title') or 'Player scan'))}</b>"]
    players = [row for row in card.get("players") or [] if isinstance(row, dict)]
    for number, row in enumerate(players[:MAX_SCAN_SHOWN], 1):
        lines += _scan_row_text(row, number, under)
    if len(players) > MAX_SCAN_SHOWN:
        lines.append(f"… and {len(players) - MAX_SCAN_SHOWN} more (📱 open the app to see them all)")
    close = [row for row in card.get("close") or [] if isinstance(row, dict)]
    if close:
        names = []
        for row in close[:5]:
            bit = esc(str(row.get("name") or "A player"))
            numbers = [f"{row['projection']:.1f}" if isinstance(row.get("projection"), (int, float)) else None,
                       pct(row["chance"]) if isinstance(row.get("chance"), (int, float)) else None]
            numbers = [part for part in numbers if part]
            names.append(bit + (f" ({', '.join(numbers)})" if numbers else ""))
        lines.append(("Also close: " if players else "Closest: ") + "; ".join(names))
    if not players and not close:
        lines.append("Nobody qualifies yet.")
    loading = scan_loading_text(card.get("loading"))
    if loading:
        lines.append(f"<i>⏳ {esc(loading)}</i>")
    return "\n".join(lines)


def card_text(card: Dict[str, Any]) -> str:
    """One card from the assistant as Telegram HTML (the numbers are the code's, as priced)."""
    kind = card.get("type")
    if kind == "ticket":
        return _ticket_text(card)
    if kind == "picks":
        return _picks_text(card)
    if kind == "player_scan":
        return _scan_text(card)
    if kind == "plan":
        return _plan_text(card)
    if kind == "player":
        return _player_text(card)
    if kind == "pasted":
        return _pasted_text(card)
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
    # Under a ticket, a plan, a pasted slip or the likeliest winners, the helpline, as the app shows it on every screen
    shown = (reply.get("cards") or [])[:MAX_CARDS]
    if any(card.get("type") in ("ticket", "plan", "pasted", "picks", "player_scan") for card in shown) \
            and HELPLINE not in body_text(result):
        foot.append(f"<i>{html.escape(HELPLINE)}</i>")
    if credits.get("daily_cents"):
        foot.append(f"<i>AI analyst · {credits.get('left_cents', 0):g}¢ of {credits['daily_cents']}¢ "
                    "of today's AI credits left</i>")
    body = str(reply.get("text") or "")
    for keep in range(len(cards), -1, -1):  # the cards are the code's numbers: cut the text before them
        room = MAX_TEXT - len("\n\n".join(head + [""] + cards[:keep] + foot))
        if room >= MIN_TEXT_ROOM or len(html.escape(body)) <= room or (keep == 0 and room > 0):
            return "\n\n".join(head + [_fit(body, room)] + cards[:keep] + foot)
    return _fit(body, MAX_TEXT)


def body_text(result: Dict[str, Any]) -> str:
    return str((result.get("reply") or {}).get("text") or "")


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
        for index, card in enumerate(cards):
            if card.get("type") != "picks":
                continue
            # The likeliest winners: one "Add pick" button each (the person's tap, as for a ticket), three a row
            buttons = [InlineKeyboardButton(f"➕ Add pick {pick.get('rank') or n}",
                                            callback_data=f"ai:pick:{thread}:{at}:{index}:{n - 1}")
                       for n, pick in enumerate((card.get("picks") or [])[:MAX_PICKS_SHOWN], 1)]
            rows += [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        for index, card in enumerate(cards):
            if card.get("type") != "player_scan":
                continue
            # A player scan: "View player" for its top rows (the player's page, read here; props aren't slip legs)
            for n, row in enumerate((card.get("players") or [])[:MAX_SCAN_VIEW], 1):
                if isinstance(row, dict) and row.get("athlete_id") and row.get("league"):
                    name = str(row.get("name") or "player")[:40]
                    rows.append([InlineKeyboardButton(f"👤 View {name}",
                                                      callback_data=f"ai:pl:{thread}:{at}:{index}:{n - 1}")])
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
                   thread_id: Optional[int] = None, asked: Optional[str] = None,
                   image: Optional[bytes] = None) -> None:
    """Send one message (or a slip's screenshot, with its caption as ``text``) to the assistant and post its
    answer, with progress while it works. A message that looks like a slip is read as one by reply()."""
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

    def ask(thread: Optional[int]) -> Dict[str, Any]:
        if image is not None:
            return assistant.read_slip(uid, thread, text=text or None, image=image, on_step=on_step)
        return assistant.reply(uid, thread, text, None, None, on_step)

    try:
        try:
            result = await asyncio.to_thread(ask, thread_id)
        except NotFound:  # the thread is gone (a new database): start a fresh one
            result = await asyncio.to_thread(ask, None)
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
    """A plain text message to the bot: the AI chat for the owner's friends, a pointer to /start for others.

    A warning sign gets the support message from anyone, before either pointer to the games."""
    message = update.effective_message
    if message is None or not message.text or update.effective_user is None:
        return
    allowed = _allowed(update, context)
    assistant = context.application.bot_data.get("assistant")
    if not allowed or assistant is None:
        flags = risk_flags(message.text)
        if flags:
            await _support_without_ai(update, context, message.text, flags)
        elif allowed and _lab(context) is not None and Assistant.looks_pasted(message.text.strip()):
            await _free_slip(update, context, message.text.strip())
        else:
            await message.reply_text(PRIVATE_MESSAGE if not allowed else NO_AI_MESSAGE)
        return
    await run_turn(update, context, message.text)


def _lab(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """The Lab (dashboard/lab.py) the bot shares with the app, or None without the games service."""
    assistant = context.application.bot_data.get("assistant")
    return getattr(assistant, "lab", None) or getattr(context.application.bot_data.get("service"), "lab", None)


async def _free_slip(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """A pasted slip with no AI chat here (no API key): the free code-only reader reads it, and the answer is
    the code's own (what was read, the numbers, the return check and the helpline)."""
    lab = _lab(context)
    uid = _uid(update)
    user = update.effective_user
    message = update.effective_message

    def read() -> Dict[str, Any]:
        lab.store.get_or_create_user(uid, (user.first_name or user.username or "Player")[:24])
        if not lab.user(uid).get("age_confirmed_at"):
            return {"age": True}
        view = lab.read_basic(uid, text)
        words = [reading_text(view), FREE_READER_NOTE]
        if not any(leg.get("odds") for leg in view["legs"]):  # a same-game parlay: one price for the whole bet
            words.append(NO_LEG_ODDS)
            return {"reply": {"text": "\n\n".join(words), "cards": [pasted_card(view, None, "read_basic")],
                              "chips": []}}
        paused = lab.picks_paused_until(uid)
        if view.get("needs_confirmation") and not paused:  # what was read and why it's doubtful, no numbers yet
            words += [Assistant._doubts(view), FREE_CHECK]
            return {"reply": {"text": "\n\n".join(words), "cards": [pasted_card(view, None, "read_basic")],
                              "chips": []}, "confirm": view["id"]}
        return _free_answer(lab, uid, lab.analyse_pasted(uid, view["id"]), paused)

    try:
        result = await asyncio.to_thread(read)
    except CallRejected as exc:
        await message.reply_text(str(exc))
        return
    except Exception:
        logging.exception("The free slip reader failed")
        await message.reply_text("Something went wrong reading that slip. Try again in a minute.")
        return
    if result.get("age"):  # the bot has no Settings: the button confirms it here, without the AI chat
        await message.reply_text(FREE_AGE, reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton(AGE_BUTTON, callback_data="paste:age")]]))
        return
    await _send_free(update, context, result)


def _free_answer(lab: Any, uid: str, analysis: Dict[str, Any], paused: Any, confirmed: bool = False) -> Dict[str, Any]:
    """The free reader's answer once a slip can be priced: what was read, the code's numbers and words (no AI),
    and during the pause on picks no suggestions."""
    view = analysis.get("pasted") or {}
    if confirmed:  # the person said the read is right: the card stops asking
        view = {**view, "needs_confirmation": False, "confirmed": True}
        analysis = {**analysis, "pasted": view}
    words = f"{reading_text(view)}\n\n{FREE_READER_NOTE}"
    if paused:  # the numbers stay open; nothing that reads as a suggestion
        analysis = {**analysis, "alternatives": []}
        words += "\n\nPicks are paused after an earlier message about losses or money worries."
    else:
        words += f"\n\n{pasted_template(view, analysis)}"
    return {"reply": {"text": words, "cards": [pasted_card(view, analysis, "read_basic")], "chips": []}}


async def _send_free(update: Update, context: ContextTypes.DEFAULT_TYPE, result: Dict[str, Any]) -> None:
    """The free reader's answer as a message: with ✅ Yes, that's right when the read needs checking first."""
    chat = update.effective_chat
    config = context.application.bot_data.get("config")
    markup = reply_markup(result, config, chat is not None and chat.type == ChatType.PRIVATE)
    if result.get("confirm") is not None:
        confirm = [InlineKeyboardButton(CONFIRM_BUTTON, callback_data=f"paste:ok:{int(result['confirm'])}")]
        markup = InlineKeyboardMarkup([confirm] + list(markup.inline_keyboard if markup else []))
    await update.effective_message.reply_text(reply_text(result), parse_mode=ParseMode.HTML,
                                              disable_web_page_preview=True, reply_markup=markup)


async def free_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """✅ Yes, that's right under a doubtful free read (no AI chat here): the code's numbers for that slip, the
    person's own (another person's id is not found)."""
    query = update.callback_query
    lab = _lab(context)
    if not _allowed(update, context) or lab is None:
        await query.answer(PRIVATE_MESSAGE if not _allowed(update, context) else NO_AI_MESSAGE, show_alert=True)
        return
    uid = _uid(update)

    def priced() -> Dict[str, Any]:
        analysis = lab.analyse_pasted(uid, int(query.data.split(":")[2]))
        return _free_answer(lab, uid, analysis, lab.picks_paused_until(uid), confirmed=True)

    try:
        result = await asyncio.to_thread(priced)
    except (CallRejected, NotFound) as exc:
        await query.answer(str(exc)[:200] if isinstance(exc, CallRejected) else EXPIRED, show_alert=True)
        return
    except Exception:
        logging.exception("The free slip reader failed on a confirmed read")
        await query.answer("Something went wrong. Try again in a minute.", show_alert=True)
        return
    await query.answer()
    try:
        await query.edit_message_reply_markup(None)  # the read is confirmed: the button has done its job
    except TelegramError:
        pass
    await _send_free(update, context, result)


async def free_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The age button under the free reader's answer (no AI chat here): the person confirms, then sends the
    slip again."""
    query = update.callback_query
    lab = _lab(context)
    if not _allowed(update, context) or lab is None:
        await query.answer(PRIVATE_MESSAGE if not _allowed(update, context) else NO_AI_MESSAGE, show_alert=True)
        return
    user = update.effective_user
    await asyncio.to_thread(lab.store.get_or_create_user, _uid(update), (user.first_name or user.username or "Player")[:24])
    await asyncio.to_thread(lab.store.confirm_age, _uid(update))
    await query.answer("Thanks!")
    try:
        await query.edit_message_text(FREE_AGE_DONE)
    except TelegramError:
        pass


def _pick_image(message: Any) -> Any:
    """The image to read from a photo or an image file: the largest photo size up to PHOTO_EDGE pixels on the
    long side (or the smallest, if all are bigger), or the document. Raises CallRejected when it isn't one we
    can read (too big, or not an image)."""
    photos = list(getattr(message, "photo", None) or [])
    if photos:
        fits = [size for size in photos if max(size.width, size.height) <= PHOTO_EDGE]
        picked = max(fits, key=lambda size: size.width * size.height) if fits else \
            min(photos, key=lambda size: size.width * size.height)
    else:
        picked = getattr(message, "document", None)
        if picked is None or not str(getattr(picked, "mime_type", "") or "").startswith("image/"):
            raise CallRejected(NOT_AN_IMAGE)
    if (getattr(picked, "file_size", None) or 0) > min(MAX_DOWNLOAD, MAX_IMAGE_BYTES):
        raise CallRejected(TOO_BIG)
    return picked


async def chat_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A photo or an image file sent to the bot: a slip to read, for the owner's friends. The caption runs
    the warning-sign rules for anyone (the support message, AI or no AI); the image is read, never kept."""
    message = update.effective_message
    if message is None or update.effective_user is None:
        return
    caption = (getattr(message, "caption", None) or "").strip()
    allowed = _allowed(update, context)
    assistant = context.application.bot_data.get("assistant")
    if not allowed or assistant is None:
        flags = risk_flags(caption)
        if flags:
            await _support_without_ai(update, context, caption, flags)
        else:
            await message.reply_text(PRIVATE_MESSAGE if not allowed else NO_AI_IMAGE if _lab(context) else NO_AI_MESSAGE)
        return
    if risk_flags(caption):  # the support message first, from the caption alone: the image isn't fetched
        await run_turn(update, context, caption)
        return
    try:
        picked = _pick_image(message)
        data = bytes(await (await picked.get_file()).download_as_bytearray())
    except CallRejected as exc:
        await message.reply_text(str(exc))
        return
    except TelegramError:
        logging.warning("Couldn't download a slip image from Telegram", exc_info=True)
        await message.reply_text("I couldn't download that image from Telegram: try sending it again.")
        return
    await run_turn(update, context, caption, image=data)


async def _support_without_ai(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, flags: List[str]) -> None:
    """A warning sign from someone the AI chat doesn't answer (not on the allowlist, or no API key): the same
    support message and helplines, and the same pause on the AI's picks (logged for the owner to review)."""
    assistant = context.application.bot_data.get("assistant")
    # The Lab writes the pause (dashboard/lab.py's Gates), as it does for the chat and the app
    lab = getattr(assistant, "lab", None) or getattr(context.application.bot_data.get("service"), "lab", None)
    if lab is not None:
        try:
            await asyncio.to_thread(note_warning_sign, lab, _uid(update), flags, "rules, bot without the AI chat",
                                    text)
        except Exception:
            logging.exception("Couldn't record a warning sign")
    await update.effective_message.reply_text(support_message(flags))


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


async def add_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Put one of the likeliest winners on the bot's slip: the person's own tap, never the AI's."""
    query = update.callback_query
    if not await _check_button(update, context):
        return
    _, _, thread, at, index, number = query.data.split(":")
    stored = await _stored_reply(context, _uid(update), int(thread), int(at))
    cards = stored.get("cards") or []
    card = cards[int(index)] if int(index) < len(cards) else None
    picks_ = (card or {}).get("picks") or [] if (card or {}).get("type") == "picks" else []
    ref = (picks_[int(number)].get("ref") or {}) if int(number) < len(picks_) else {}
    if ref.get("market") not in ("h2h", "spreads", "totals") or not ref.get("id"):
        await query.answer("This pick isn't available any more.", show_alert=True)
        return
    slip = context.user_data.setdefault("picks", [])
    pick = {"league": ref["league"], "id": ref["id"], "market": ref["market"], "side": ref["side"]}
    if not any(p.get("league") == pick["league"] and p.get("id", "")[:16] == pick["id"][:16]
               and p.get("market") == pick["market"] and p.get("side") == pick["side"] for p in slip):
        slip.append(pick)
    context.user_data["mode"] = "multi"
    await query.answer(f"Added to your slip ✓ ({len(slip)} in all, priced as one parlay). Send /slip to see it.",
                       show_alert=True)


def player_page_text(player: Dict[str, Any]) -> str:
    """A player's page (Lab.player) as Telegram HTML: who, status, next game, the NBA model's projection on game day
    and the posted props with their hit rates and the model's chance of the over."""
    esc = html.escape
    about = ", ".join(str(part) for part in (player.get("team"), player.get("position")) if part)
    lines = [f"👤 <b>{esc(str(player.get('name') or 'Player'))}</b>" + (f" ({esc(about)})" if about else "")]
    if player.get("injury") or (player.get("status") and str(player["status"]).lower() != "active"):
        lines.append(f"Status: {esc(str(player.get('injury') or player.get('status')))}")
    game = player.get("next_game") or {}
    if isinstance(game, dict) and (game.get("name") or game.get("opponent")):
        lines.append(f"Next: {esc(str(game.get('name') or game.get('opponent')))}")
    projection = player.get("projection") or {}
    stats = projection.get("stats") if isinstance(projection, dict) else None
    if isinstance(stats, dict):
        shown = [f"{key} {value['mean']:g}" for key, value in stats.items()
                 if key in ("pts", "reb", "ast", "fg3m", "pra") and isinstance(value, dict)
                 and isinstance(value.get("mean"), (int, float))]
        if shown:
            lines.append("Projected: " + ", ".join(shown))
    for prop in (player.get("props") or [])[:6]:
        if not isinstance(prop, dict):
            continue
        bits = [esc(str(prop.get("label") or prop.get("name") or prop.get("market") or "Prop"))]
        price = scan_line_text({"line": prop.get("line"), "over": prop.get("over"), "under": prop.get("under"),
                                "yes": prop.get("yes"), "kind": prop.get("kind")})
        if price:
            bits.append(esc(price))
        rates = prop.get("hit_rates") or {}
        if isinstance(rates, dict) and rates.get("last10"):
            bits.append(f"L10 {esc(str(rates['last10']))}")
        ai = prop.get("ai") or {}
        if isinstance(ai, dict) and isinstance(ai.get("over"), (int, float)):
            bits.append(f"AI {pct(ai['over'])} over")
        lines.append("• " + " · ".join(bits))
    info = player.get("props_info") or {}
    if not player.get("props") and isinstance(info, dict) and info.get("reason"):
        lines.append(f"<i>{esc(str(info['reason']))}</i>")
    lines.append("<i>Player props are for reading here; the slip takes game picks.</i>")
    return "\n".join(lines)


async def view_player(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """"View player" under a player scan: that row's player page, sent as a message (behind the player pages'
    gates: age confirmed, not on a break)."""
    query = update.callback_query
    if not await _check_button(update, context):
        return
    _, _, thread, at, index, number = query.data.split(":")
    stored = await _stored_reply(context, _uid(update), int(thread), int(at))
    cards = stored.get("cards") or []
    card = cards[int(index)] if int(index) < len(cards) else None
    rows = (card or {}).get("players") or [] if (card or {}).get("type") == "player_scan" else []
    row = rows[int(number)] if int(number) < len(rows) and isinstance(rows[int(number)], dict) else {}
    lab = _lab(context)
    if not row.get("athlete_id") or not row.get("league") or lab is None:
        await query.answer(SCAN_PLAYER_GONE, show_alert=True)
        return
    await query.answer()

    def page() -> Dict[str, Any]:
        lab.ready(_uid(update))
        return lab.player(str(row["league"]), str(row["athlete_id"]))

    try:
        text = player_page_text(await asyncio.to_thread(page))
    except (CallRejected, NotFound) as exc:
        text = html.escape(str(exc) or SCAN_PLAYER_GONE)
    except Exception:
        logging.exception("Couldn't load a player page in the bot")
        text = "Couldn't load that player right now: try again in a minute."
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


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
    # The free reader's buttons (no AI chat needed): the age, and "Yes, that's right" under a doubtful read
    application.add_handler(CallbackQueryHandler(free_age, pattern=r"^paste:age$"))
    application.add_handler(CallbackQueryHandler(free_confirm, pattern=r"^paste:ok:\d{1,12}$", block=False))
    # ai:add:<thread>:<answer>:<card> and ai:<thread>:<answer>:<chip>; older two-number buttons fall
    # through to the catch-all, which says the menu has expired
    application.add_handler(CallbackQueryHandler(add_to_slip, pattern=r"^ai:add:\d{1,12}:\d{1,6}:\d$"))
    application.add_handler(CallbackQueryHandler(add_pick, pattern=r"^ai:pick:\d{1,12}:\d{1,6}:\d:\d$"))
    application.add_handler(CallbackQueryHandler(view_player, pattern=r"^ai:pl:\d{1,12}:\d{1,6}:\d:\d{1,2}$",
                                                 block=False))
    # A chat turn takes seconds: block=False lets the bot keep serving everyone else's taps meanwhile
    application.add_handler(CallbackQueryHandler(chip_tapped, pattern=r"^ai:\d{1,12}:\d{1,6}:\d$", block=False))
    # New messages only: editing a sent message shouldn't start a second (paid) turn
    text = filters.UpdateType.MESSAGE & filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    application.add_handler(MessageHandler(text, chat_message, block=False))
    # A slip's photo, or a screenshot sent as an image file
    images = filters.UpdateType.MESSAGE & (filters.PHOTO | filters.Document.IMAGE) & filters.ChatType.PRIVATE
    application.add_handler(MessageHandler(images, chat_photo, block=False))
