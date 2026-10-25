"""Command and callback handlers for the Telegram betting bot.

Flow: /start → Single or Multi → sport → league → games → a game's card
(the model's predicted score, win/cover/over chances, best odds,
injuries and projected top players) with one button per pick. In Single
mode a pick shows that bet's payout; in Multi mode picks go on a slip
(marked ✓) whose combined odds come from the best single bookmaker.
Each screen edits the same message.

Buttons carry what they need (league, game id prefix, pick), so menus
keep working after a restart, and prices are always re-read from the
shared odds cache, never taken from a button. The games, predictions and
slip pricing come from the same service as the web dashboard, called in
a worker thread so a slow load doesn't stall other chats.
"""

from __future__ import annotations

import asyncio
import html
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
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
from ..data.leagues import SPORTS, pick_leagues
from ..services import build_service

MAX_GAMES = 10
STAKES = (5, 10, 25, 50)
# Pick buttons encode market and side in one letter each to stay under Telegram's 64-byte limit
MARKET_CODES = {"h": "h2h", "s": "spreads", "t": "totals"}
SIDE_CODES = {"h": "home", "a": "away", "d": "draw", "o": "over", "u": "under"}
GAME_ID_CHARS = 16
TWO_WORD_NICKNAMES = ("Trail Blazers", "Red Sox", "White Sox", "Blue Jays", "Maple Leafs", "Golden Knights",
                      "Red Wings", "Blue Jackets")


# --- formatting ------------------------------------------------------------------

def short_name(name: str, league: str = "") -> str:
    """Team nickname ("Oklahoma City Thunder" -> "Thunder"); soccer clubs keep their full name."""
    if league.startswith("soccer_"):
        return name
    for nickname in TWO_WORD_NICKNAMES:
        if name.endswith(f" {nickname}"):
            return nickname
    return name.split()[-1] if " " in name else name


def pct(probability: Optional[float]) -> str:
    return "–" if probability is None else f"{round(probability * 100)}%"


def money(amount: float) -> str:
    return f"${amount:,.2f}"


def pick_data(league: str, game_id: str, market: str, side: str) -> str:
    """Callback data for a pick, e.g. "p:basketball_nba:4c1f…:sh" (at most 64 bytes)."""
    m = next(code for code, name in MARKET_CODES.items() if name == market)
    s = next(code for code, name in SIDE_CODES.items() if name == side)
    return f"p:{league}:{game_id[:GAME_ID_CHARS]}:{m}{s}"


def parse_pick(data: str) -> Optional[Dict[str, str]]:
    try:
        _, league, game_id, codes = data.split(":")
        return {"league": league, "id": game_id, "market": MARKET_CODES[codes[0]], "side": SIDE_CODES[codes[1]]}
    except (ValueError, KeyError, IndexError):
        return None


def same_pick(a: Dict[str, str], b: Dict[str, str]) -> bool:
    return (a["league"] == b["league"] and a["id"][:GAME_ID_CHARS] == b["id"][:GAME_ID_CHARS]
            and a["market"] == b["market"] and a["side"] == b["side"])


def offer_button_text(card: Dict[str, Any], market: str, side: str, offer: Dict[str, Any]) -> str:
    league = card.get("league", "")
    team = {"home": card["home"], "away": card["away"]}.get(side)
    if market == "h2h":
        name = f"{short_name(team, league)} ML" if team else "Draw"
    elif market == "spreads":
        name = f"{short_name(team, league)} {offer['point']:+g}"
    else:
        name = f"{side.title()} {offer['point']:g}"
    return f"{name} {offer['american']} · {pct(offer['ai'])}"


def game_card_text(card: Dict[str, Any], odds_as_of: Optional[str]) -> str:
    """A game's prediction, odds context, injuries and projected leaders (HTML)."""
    league = card.get("league", "")
    home, away = card["home"], card["away"]
    esc = html.escape
    lines = [f"<b>{esc(away)} @ {esc(home)}</b>", "🔴 Live (pre-game prediction)" if card["live"] else esc(card["when"]), ""]
    p = card.get("prediction")
    markets = card["markets"]
    if p:
        fav, dog = (home, away) if p["home_win"] >= 0.5 else (away, home)
        fav_score = p["home_score"] if fav == home else p["away_score"]
        dog_score = p["away_score"] if fav == home else p["home_score"]
        lines.append(f"🤖 <b>AI prediction:</b> {esc(short_name(fav, league))} {fav_score:.0f} – "
                     f"{dog_score:.0f} {esc(short_name(dog, league))} (total {p['total']:.1f})")
        lines.append(f"Win chance: {esc(short_name(home, league))} {pct(p['home_win'])} · "
                     f"{esc(short_name(away, league))} {pct(1 - p['home_win'])}")
        spread = markets.get("spreads", {}).get("home")
        if spread:
            lines.append(f"Spread {esc(short_name(home, league))} {spread['point']:+g}: covers {pct(spread['ai'])}")
        over = markets.get("totals", {}).get("over")
        if over:
            lines.append(f"Total {over['point']:g}: over {pct(over['ai'])} · under {pct(1 - over['ai'])}")
        sources = p.get("sources", {})
        others = []
        if "market" in sources:
            others.append(f"market {pct(sources['market']['home_win_prob'])}")
        if "espn" in sources:
            others.append(f"ESPN {pct(sources['espn']['home_win_prob'])}")
        if "model" in sources:
            others.append(f"our team model {pct(sources['model']['home_win_prob'])}")
        if others:
            lines.append(f"{esc(short_name(home, league))} win by source: {', '.join(others)}")
    else:
        h2h = markets.get("h2h", {})
        if h2h:
            lines.append("📊 <b>Market chances</b> (no bookmaker margin): " + " · ".join(
                f"{esc(short_name(card.get(side, 'Draw'), league) if side != 'draw' else 'Draw')} {pct(offer['fair'])}"
                for side, offer in h2h.items()))
        if card.get("espn_home_win") is not None:
            lines.append(f"ESPN predictor: {esc(short_name(home, league))} {pct(card['espn_home_win'])}")

    injuries = card.get("injuries") or {}
    out = {side: [i for i in injuries.get(side, []) if "(Out)" in i or "(Doubtful)" in i] for side in ("away", "home")}
    if any(out.values()):
        lines += ["", "🩹 <b>Out</b>"]
        for side in ("away", "home"):
            if out[side]:
                names = ", ".join(i.rsplit(" (", 1)[0] for i in out[side])
                lines.append(f"{esc(short_name(card[side], league))}: {esc(names)}")

    players = card.get("players")
    if players:
        lines += ["", "⭐ <b>Projected leaders</b>"]
        for side in ("away", "home"):
            top = [pl for pl in players.get(side, []) if not pl["out"]]
            top = sorted(top, key=lambda pl: pl["pts"], reverse=True)[:3]
            if top:
                lines.append(f"<i>{esc(short_name(card[side], league))}</i>")
                lines += [f"{esc(pl['name'])}: {pl['pts']:.1f} pts, {pl['reb']:.1f} reb, {pl['ast']:.1f} ast"
                          for pl in top]
    lines += ["", f"Odds{f' as of {esc(odds_as_of)}' if odds_as_of else ''}: best price across bookmakers. "
                  "Button % = AI chance of winning."]
    return "\n".join(lines)


def slip_text(slip: Dict[str, Any], picks: List[Dict[str, Any]]) -> str:
    """A multi slip: legs, combined odds at the best bookmaker, payout and the model's view (HTML)."""
    esc = html.escape
    count = len(slip.get("legs", []))
    lines = [f"🧾 <b>Bet slip: {count}-leg multi</b>", ""]
    for i, leg in enumerate(slip.get("legs", []), 1):
        lines.append(f"{i}. {esc(leg['label'])} {leg['odds']['american']} — {esc(leg['game'])}")
    if slip.get("odds"):
        at_book = f" at {esc(slip['book'])}" if slip.get("book") else ""
        others = ", ".join(f"{esc(b['book'])} {b['american']}" for b in slip.get("books", [])[1:3])
        compare = f" ({others})" if others else ""
        lines += [
            "",
            f"Odds <b>{slip['odds']['american']}</b>{at_book}{compare}",
            f"Stake {money(slip['stake'])} → payout <b>{money(slip['returns'])}</b> (profit {money(slip['profit'])})",
        ]
        if slip.get("probability") is not None:
            lines.append(f"AI chance all legs win: {pct(slip['probability'])} · expected value {slip['ev'] * 100:+.1f}%")
        if slip.get("book_edge") is not None:
            lines.append(f"Bookmaker edge on this multi: {slip['book_edge'] * 100:.1f}%")
    if slip.get("reasons"):
        lines += [""] + [f"⚠️ {esc(reason)}" for reason in slip["reasons"]]
    return "\n".join(lines)


def single_text(slip: Dict[str, Any]) -> str:
    """One single bet: odds, payout for the stake and the model's view (HTML)."""
    esc = html.escape
    leg = slip["legs"][0]
    lines = [
        f"🎯 <b>Single: {esc(leg['label'])}</b>",
        esc(leg["game"]),
        "",
        f"Odds <b>{leg['odds']['american']}</b> (best price)",
        f"Stake {money(leg['stake'])} → returns <b>{money(leg['returns'])}</b> (profit {money(leg['returns'] - leg['stake'])})",
    ]
    if leg.get("probability") is not None:
        lines.append(f"AI chance it wins: {pct(leg['probability'])} · expected value {leg['ev'] * 100:+.1f}%")
    if slip.get("reasons"):
        lines += [""] + [f"⚠️ {esc(reason)}" for reason in slip["reasons"]]
    return "\n".join(lines)


# --- screens ---------------------------------------------------------------------

def _picks(context: ContextTypes.DEFAULT_TYPE) -> List[Dict[str, str]]:
    return context.user_data.setdefault("picks", [])


def _mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("mode", "single")


def _slip_button(context: ContextTypes.DEFAULT_TYPE) -> List[InlineKeyboardButton]:
    if _mode(context) != "multi":
        return []
    return [InlineKeyboardButton(f"🧾 Bet slip ({len(_picks(context))})", callback_data="slip")]


async def _show(update: Update, text: str, rows: List[List[InlineKeyboardButton]]) -> None:
    """Edit the screen message in place (or send it, for commands)."""
    markup = InlineKeyboardMarkup(rows)
    query = update.callback_query
    if query:
        try:
            await query.edit_message_text(text, reply_markup=markup, parse_mode=ParseMode.HTML,
                                          disable_web_page_preview=True)
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                raise
    elif update.message:
        await update.message.reply_text(text, reply_markup=markup, parse_mode=ParseMode.HTML,
                                        disable_web_page_preview=True)


async def _no_service(update: Update) -> None:
    await _show(update, "Showing games needs a The Odds API key (free at the-odds-api.com). "
                        "Add THE_ODDS_API_KEY to the bot's .env and restart it.", [])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask for Single or Multi."""
    rows = [[InlineKeyboardButton("🎯 Single game", callback_data="mode:single"),
             InlineKeyboardButton("🧩 Multi (parlay)", callback_data="mode:multi")]]
    if _picks(context):
        rows.append([InlineKeyboardButton(f"🧾 Bet slip ({len(_picks(context))})", callback_data="slip")])
    await _show(update, (
        "<b>SportsBet AI</b>\nHow do you want to play?\n\n"
        "🎯 <b>Single</b>: one game at a time, with the AI's prediction and each pick's payout.\n"
        "🧩 <b>Multi</b>: combine picks from different games into one parlay."
    ), rows)


async def mode_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data["mode"] = "multi" if query.data == "mode:multi" else "single"
    await show_sports(update, context)


async def show_sports(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = [
        [InlineKeyboardButton(label, callback_data=f"sport:{group}") for label, group in SPORTS[i:i + 2]]
        for i in range(0, len(SPORTS), 2)
    ]
    rows.append([InlineKeyboardButton("« Single / Multi", callback_data="home")] + _slip_button(context))
    mode = "Multi (parlay)" if _mode(context) == "multi" else "Single game"
    await _show(update, f"<b>{mode}</b>\nPick a sport:", rows)


async def go_home(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await start(update, context)


async def sport_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List the sport's leagues in season, or its games if only one league is."""
    query = update.callback_query
    await query.answer()
    group = query.data.removeprefix("sport:")
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    try:
        sports = await asyncio.to_thread(service.odds.sports)
    except Exception:
        sports = []
    if not sports:
        await _show(update, "Couldn't reach the odds feed. Check THE_ODDS_API_KEY, then try /start again.", [])
        return
    leagues = pick_leagues(sports, group)
    back = [InlineKeyboardButton("« Sports", callback_data="sports")] + _slip_button(context)
    if not leagues:
        await _show(update, f"No {html.escape(group)} leagues are in season right now.", [back])
        return
    if len(leagues) == 1:
        await show_games(update, context, leagues[0]["key"])
        return
    rows = [
        [InlineKeyboardButton(league["title"], callback_data=f"league:{league['key']}") for league in leagues[i:i + 2]]
        for i in range(0, len(leagues), 2)
    ]
    rows.append(back)
    await _show(update, f"Pick a {html.escape(group)} league:", rows)


async def sports_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await show_sports(update, context)


async def league_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await show_games(update, context, query.data.removeprefix("league:"))


async def show_games(update: Update, context: ContextTypes.DEFAULT_TYPE, league: str) -> None:
    """The league's next games, each a button, with the AI's pick in the text."""
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    context.user_data["league"] = league
    back = [InlineKeyboardButton("« Sports", callback_data="sports")] + _slip_button(context)
    try:
        board = await asyncio.to_thread(service.games, league)
    except Exception:
        await _show(update, "Couldn't load games right now. Try again in a minute.", [back])
        return
    games = board["games"][:MAX_GAMES]
    if not games:
        await _show(update, "No upcoming games with odds right now.", [back])
        return
    as_of = f" · odds as of {html.escape(board['odds_as_of'])}" if board.get("odds_as_of") else ""
    lines = [f"<b>{html.escape(board.get('title') or league)}</b>: next {len(games)} games{as_of}", ""]
    rows = []
    for card in games:
        away, home = short_name(card["away"], league), short_name(card["home"], league)
        when = "LIVE" if card["live"] else card["when"].split(", ", 1)[-1]
        p = card.get("prediction")
        pick = f" — AI: {html.escape(short_name(p['pick'], league))} {pct(p['pick_prob'])}" if p else ""
        lines.append(f"{html.escape(when)} · {html.escape(away)} @ {html.escape(home)}{pick}")
        rows.append([InlineKeyboardButton(f"{away} @ {home} · {when}", callback_data=f"g:{league}:{card['id'][:GAME_ID_CHARS]}")])
    rows.append(back)
    lines += ["", "Tap a game for the full prediction and odds."]
    await _show(update, "\n".join(lines), rows)


async def game_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, league, game_id = query.data.split(":", 2)
    await show_game(update, context, league, game_id)


async def show_game(update: Update, context: ContextTypes.DEFAULT_TYPE, league: str, game_id: str) -> None:
    """A game's card with one button per pick (✓ when it's on the multi slip)."""
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    back = [InlineKeyboardButton("« Games", callback_data=f"league:{league}")] + _slip_button(context)
    try:
        board = await asyncio.to_thread(service.games, league)
        match = next((g for g in board["games"] if g["id"].startswith(game_id)), None)
        card = await asyncio.to_thread(service.game, league, match["id"]) if match else None
    except Exception:
        card, board = None, {}
    if not card:
        await _show(update, "That game isn't offered any more.", [back])
        return
    chosen = _picks(context) if _mode(context) == "multi" else []
    rows = []
    for market in ("h2h", "spreads", "totals"):
        row = []
        for side, offer in card["markets"].get(market, {}).items():
            pick = {"league": league, "id": card["id"], "market": market, "side": side}
            mark = "✓ " if any(same_pick(pick, c) for c in chosen) else ""
            row.append(InlineKeyboardButton(mark + offer_button_text(card, market, side, offer),
                                            callback_data=pick_data(league, card["id"], market, side)))
        if row:
            rows.append(row)
    rows.append(back)
    await _show(update, game_card_text(card, board.get("odds_as_of")), rows)


async def pick_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Single mode: show the bet's payout. Multi mode: add or remove the pick on the slip."""
    query = update.callback_query
    pick = parse_pick(query.data)
    if pick is None:
        await query.answer("This menu has expired. Send /start to begin again.", show_alert=True)
        return
    if _mode(context) == "multi":
        picks = _picks(context)
        existing = next((c for c in picks if same_pick(pick, c)), None)
        if existing:
            picks.remove(existing)
            await query.answer(f"Removed from slip ({len(picks)} left)")
        else:
            picks.append(pick)
            await query.answer(f"Added to slip ✓ ({len(picks)} picks)")
        await show_game(update, context, pick["league"], pick["id"])
        return
    await query.answer()
    context.user_data["single"] = pick
    await show_single(update, context)


def _stake_rows(context: ContextTypes.DEFAULT_TYPE) -> List[List[InlineKeyboardButton]]:
    current = context.user_data.get("stake", 10)
    return [[InlineKeyboardButton(f"{'• ' if amount == current else ''}${amount}", callback_data=f"stake:{amount}")
             for amount in STAKES]]


async def _full_pick(service: Any, pick: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Expand a pick's game id prefix to the full id the pricing service expects."""
    board = await asyncio.to_thread(service.games, pick["league"])
    match = next((g for g in board["games"] if g["id"].startswith(pick["id"][:GAME_ID_CHARS])), None)
    return {**pick, "id": match["id"]} if match else None


async def show_single(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = context.application.bot_data.get("service")
    pick = context.user_data.get("single")
    if service is None or not pick:
        await start(update, context)
        return
    context.user_data["view"] = "single"
    back = [InlineKeyboardButton("« Game", callback_data=f"g:{pick['league']}:{pick['id'][:GAME_ID_CHARS]}"),
            InlineKeyboardButton("« Games", callback_data=f"league:{pick['league']}")]
    full = await _full_pick(service, pick)
    if not full:
        await _show(update, "That pick isn't offered any more.", [back])
        return
    stake = context.user_data.get("stake", 10)
    slip = await asyncio.to_thread(service.price_slip, [full], "single", 0, {0: stake})
    if not slip.get("legs"):
        await _show(update, "That pick isn't offered any more.", [back])
        return
    await _show(update, single_text(slip), _stake_rows(context) + [back])


async def show_slip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query:
        await query.answer()
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    context.user_data["view"] = "slip"
    league = context.user_data.get("league")
    back = [InlineKeyboardButton("« Games", callback_data=f"league:{league}") if league
            else InlineKeyboardButton("« Sports", callback_data="sports")]
    picks = _picks(context)
    if not picks:
        await _show(update, "🧾 Your slip is empty. In Multi mode, open a game and tap picks to add them.",
                    [[InlineKeyboardButton("🧩 Multi mode", callback_data="mode:multi")]])
        return
    full = [p for p in [await _full_pick(service, pick) for pick in picks] if p]
    if len(full) < len(picks):
        context.user_data["picks"] = [p for p in picks if any(same_pick(p, f) for f in full)]
    slip = await asyncio.to_thread(service.price_slip, full, "multi", context.user_data.get("stake", 10), {})
    rows = _stake_rows(context) + [[InlineKeyboardButton("🗑 Clear slip", callback_data="clear")] + back]
    await _show(update, slip_text(slip, full), rows)


async def stake_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data["stake"] = int(query.data.removeprefix("stake:"))
    if context.user_data.get("view") == "single":
        await show_single(update, context)
    else:
        await show_slip(update, context)


async def clear_slip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer("Slip cleared")
    context.user_data["picks"] = []
    await show_slip(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide a list of available commands and brief instructions."""
    help_text = (
        "Available commands:\n"
        "/start – choose Single or Multi and browse games\n"
        "/slip – show your multi bet slip\n"
        "/help – display this help message\n\n"
        "Pick a sport and league, then open a game to see the AI's predicted score, win, spread and "
        "total chances, injuries and projected top players. In Multi mode, tap picks to build a parlay."
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


def setup_handlers(application: Application, config: Config) -> None:
    """Register command and button handlers, sharing the dashboard's games and pricing service."""
    application.bot_data["config"] = config
    application.bot_data["service"] = build_service(config)
    application.bot_data["timezone"] = ZoneInfo(config.timezone)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("slip", show_slip))
    application.add_handler(CallbackQueryHandler(mode_selected, pattern=r"^mode:(single|multi)$"))
    application.add_handler(CallbackQueryHandler(go_home, pattern=r"^home$"))
    application.add_handler(CallbackQueryHandler(sports_selected, pattern=r"^sports$"))
    application.add_handler(CallbackQueryHandler(sport_selected, pattern=r"^sport:"))
    application.add_handler(CallbackQueryHandler(league_selected, pattern=r"^league:"))
    application.add_handler(CallbackQueryHandler(game_selected, pattern=r"^g:"))
    application.add_handler(CallbackQueryHandler(pick_selected, pattern=r"^p:"))
    application.add_handler(CallbackQueryHandler(show_slip, pattern=r"^slip$"))
    application.add_handler(CallbackQueryHandler(stake_selected, pattern=r"^stake:\d+$"))
    application.add_handler(CallbackQueryHandler(clear_slip, pattern=r"^clear$"))
    # Any other button comes from an older version of the menu
    application.add_handler(CallbackQueryHandler(expired_menu))
    application.add_handler(CommandHandler("help", help_command))
    # Catch all other commands with the unknown command handler
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
