"""Command and callback handlers for the Telegram betting bot.

Flow: /start → Single or Multi → sport → league → games → a game's card
(the model's predicted score, win/cover/over chances, best odds,
injuries and projected top players) with one button per pick. In Single
mode a pick shows that bet's payout; in Multi mode picks go on a slip
(marked ✓) whose combined odds come from the best single bookmaker.
On NBA games 🔮 lets anyone make a call (winner and how sure they are),
scored after the game next to the AI's score; /start also leads to the
AI's Safe/Risky/Crazy bets, your calls and the leaderboard, and (when
PUBLIC_URL is set) to the web app opened as a Telegram Mini App. Each
screen edits the same message. Plain text messages go to the AI chat
(bot/chat.py), which is private to the owner's friends.

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
    WebAppInfo,
)
from telegram.constants import ChatType, ParseMode
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
from ..engine import calls
from ..services import build_service
from .chat import setup_chat

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

    analysts = [a for a in (card.get("analysts") or {}).get("analysts", []) if a.get("forecast")]
    if analysts:
        lines += ["", "🧠 <b>AI analysts</b>"]
        for analyst in analysts:
            forecast = analyst["forecast"]
            home_pick = forecast["home_win_prob"] >= 0.5
            chance = forecast["home_win_prob"] if home_pick else 1 - forecast["home_win_prob"]
            first = (forecast.get("summary") or "").split(". ")[0].rstrip(".")
            lines.append(f"{esc(analyst['label'])}: {esc(short_name(home if home_pick else away, league))} "
                         f"{pct(chance)}" + (f": {esc(first[:200])}." if first else ""))

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
                        "Run the setup script on the server again to add it, or add THE_ODDS_API_KEY "
                        "to the bot's .env and restart it.", [])


def _app_rows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> List[List[InlineKeyboardButton]]:
    """An "Open app" button for the Mini App when PUBLIC_URL is set.

    Telegram only allows Mini App buttons in private chats (a group would refuse the whole message).
    """
    config = context.application.bot_data.get("config")
    url = getattr(config, "mini_app_url", None)
    chat = getattr(update, "effective_chat", None)
    if not url or chat is None or chat.type != ChatType.PRIVATE:
        return []
    return [[InlineKeyboardButton("📱 Open app", web_app=WebAppInfo(url))]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The home screen: browse games (Single or Multi), AI bets, your calls and the leaderboard."""
    rows = _app_rows(update, context)
    rows += [[InlineKeyboardButton("🎯 Single game", callback_data="mode:single"),
              InlineKeyboardButton("🧩 Multi (parlay)", callback_data="mode:multi")],
             [InlineKeyboardButton("🎲 AI bets: Safe · Risky · Crazy", callback_data="bets")],
             [InlineKeyboardButton("💬 Ask AI", callback_data="ask")],
             [InlineKeyboardButton("🔮 My calls", callback_data="mycalls"),
              InlineKeyboardButton("🏆 Leaderboard", callback_data="board")]]
    if _picks(context):
        rows.append([InlineKeyboardButton(f"🧾 Bet slip ({len(_picks(context))})", callback_data="slip")])
    await _show(update, (
        "<b>SportsBet AI</b>\nWhat do you want to do?\n\n"
        "🎯 <b>Single</b>: one game at a time, with the AI's prediction and each pick's payout.\n"
        "🧩 <b>Multi</b>: combine picks from different games into one parlay.\n"
        "🎲 <b>AI bets</b>: bets the AI builds at three risk levels.\n"
        "💬 <b>Ask AI</b>: type any question as a message; our code works out the numbers and the AI explains them.\n"
        "🔮 <b>Calls</b>: open an NBA game and make your own call; after the game you and the AI are "
        "both scored, so you can see who reads the game better."
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
    if card.get("prediction") and not card["live"]:
        rows.append([InlineKeyboardButton("🔮 Make your call", callback_data=f"call:{league}:{card['id'][:GAME_ID_CHARS]}")])
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


# --- calls: predict a winner and get scored against the AI -------------------------

CONFIDENCES = (50, 60, 70, 80, 90, 100)


def _points(probability: float) -> float:
    """FiveThirtyEight forecasting-game points for the chance given to the team that won."""
    return calls.forecast_points(probability)


def _player(update: Update) -> tuple:
    user = update.effective_user
    name = (user.first_name or user.username or "Player")[:24] if user else "Player"
    return f"tg:{user.id if user else 0}", name


async def _find_card(service: Any, league: str, game_id: str) -> Optional[Dict[str, Any]]:
    board = await asyncio.to_thread(service.games, league)
    return next((g for g in board["games"] if g["id"].startswith(game_id)), None)


async def call_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pick the winner of a game."""
    query = update.callback_query
    await query.answer()
    _, league, game_id = query.data.split(":", 2)
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    card = await _find_card(service, league, game_id)
    back = [InlineKeyboardButton("« Game", callback_data=f"g:{league}:{game_id}")]
    if not card or not card.get("prediction") or card["live"]:
        await _show(update, "Calls on this game are closed (it has started, or the AI has no prediction).", [back])
        return
    p = card["prediction"]
    esc = html.escape
    text = (f"🔮 <b>Your call: {esc(card['away'])} @ {esc(card['home'])}</b>\n{esc(card['when'])}\n\n"
            f"The AI gives {esc(short_name(card['home'], league))} {pct(p['home_win'])} and "
            f"{esc(short_name(card['away'], league))} {pct(1 - p['home_win'])}.\n\nWho wins?")
    rows = [[InlineKeyboardButton(short_name(card["away"], league), callback_data=f"cw:{league}:{game_id}:a"),
             InlineKeyboardButton(short_name(card["home"], league), callback_data=f"cw:{league}:{game_id}:h")],
            back]
    await _show(update, text, rows)


async def call_winner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """How sure are you? Each button shows the points it wins or loses."""
    query = update.callback_query
    await query.answer()
    _, league, game_id, side = query.data.split(":", 3)
    service = context.application.bot_data.get("service")
    card = await _find_card(service, league, game_id) if service else None
    back = [InlineKeyboardButton("« Game", callback_data=f"g:{league}:{game_id}")]
    if not card or not card.get("prediction") or card["live"]:
        await _show(update, "Calls on this game are closed.", [back])
        return
    team = card["home"] if side == "h" else card["away"]
    ai = card["prediction"]["home_win"] if side == "h" else 1 - card["prediction"]["home_win"]
    esc = html.escape
    lines = [f"🔮 <b>{esc(team)} to win</b>", f"{calls.agreement(ai)}: it gives them {pct(ai)}.", "", "How sure are you?", ""]
    lines += [f"{c}%: {_points(c / 100):+.1f} if right, {_points(1 - c / 100):+.1f} if wrong" for c in CONFIDENCES]
    rows = [[InlineKeyboardButton(f"{c}%", callback_data=f"cc:{league}:{game_id}:{side}:{c}") for c in CONFIDENCES[:3]],
            [InlineKeyboardButton(f"{c}%", callback_data=f"cc:{league}:{game_id}:{side}:{c}") for c in CONFIDENCES[3:]],
            back]
    await _show(update, "\n".join(lines), rows)


async def call_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save the call and show the AI's view of it."""
    query = update.callback_query
    _, league, game_id, side, confidence = query.data.split(":", 4)
    service = context.application.bot_data.get("service")
    card = await _find_card(service, league, game_id) if service else None
    back = [InlineKeyboardButton("« Games", callback_data=f"league:{league}"),
            InlineKeyboardButton("🔮 My calls", callback_data="mycalls")]
    if not card:
        await query.answer()
        await _show(update, "That game isn't offered any more.", [back])
        return
    player, name = _player(update)
    call = {"winner": "home" if side == "h" else "away", "confidence": int(confidence) / 100}
    try:
        saved = await asyncio.to_thread(service.save_call, player, name, league, card["id"], call)
    except Exception as exc:
        await query.answer()
        await _show(update, f"Couldn't save your call: {html.escape(str(exc))}", [back])
        return
    await query.answer("Call saved ✓")
    team = card["home"] if side == "h" else card["away"]
    c = call["confidence"]
    esc = html.escape
    text = (f"✅ <b>Call saved: {esc(team)} at {confidence}%</b>\n{esc(card['away'])} @ {esc(card['home'])}\n\n"
            f"{esc(saved['ai'].get('verdict', ''))}\n\n"
            f"If {esc(short_name(team, league))} win: {_points(c):+.1f} points. If they lose: {_points(1 - c):+.1f}.\n"
            "It's scored after the game, next to the AI's score on the same game. "
            "Tap the game's 🔮 again before tip-off to change it.")
    await _show(update, text, [back])


async def my_calls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query:
        await query.answer()
    service = context.application.bot_data.get("service")
    back = [InlineKeyboardButton("🏆 Leaderboard", callback_data="board"), InlineKeyboardButton("« Home", callback_data="home")]
    if service is None:
        await _no_service(update)
        return
    player, _ = _player(update)
    rows = await asyncio.to_thread(service.my_calls, player)
    if not rows:
        await _show(update, "🔮 No calls yet. Open an NBA game and tap <b>🔮 Make your call</b>.", [back])
        return
    esc = html.escape
    graded = [r for r in rows if r["status"] == "graded"]
    you = sum(r["user_points"] for r in graded)
    ai = sum(r["ai_points"] for r in graded)
    if not graded:
        headline = "Calls are scored after each game, next to the AI's score on the same games."
    elif you >= ai:
        headline = f"You beat the AI by {you - ai:.1f} points on the {len(graded)} game(s) you called."
    else:
        headline = f"The AI is ahead of you by {ai - you:.1f} points on the {len(graded)} game(s) you called."
    lines = ["🔮 <b>My calls</b>", f"You {you:+.1f} pts · the AI {ai:+.1f} on the same games", headline, ""]
    for r in rows[:12]:
        call = r["call"]
        team = r["home"] if call.get("winner") == "home" else r["away"]
        what = f"{short_name(team, r['league'])} {pct(call.get('confidence'))}" if call.get("winner") else "player lines"
        if r["status"] == "graded":
            res = r["result"]
            status = f"{res['away']}–{res['home']} · you {r['user_points']:+.1f}, AI {r['ai_points']:+.1f}"
        elif r["status"] == "void":
            status = "void"
        else:
            status = "⏳ pending"
        lines.append(f"{esc(short_name(r['away'], r['league']))} @ {esc(short_name(r['home'], r['league']))}: "
                     f"{esc(what)} — {esc(status)}")
    await _show(update, "\n".join(lines), [back])


async def leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query:
        await query.answer()
    service = context.application.bot_data.get("service")
    back = [InlineKeyboardButton("🔮 My calls", callback_data="mycalls"), InlineKeyboardButton("« Home", callback_data="home")]
    if service is None:
        await _no_service(update)
        return
    rows = await asyncio.to_thread(service.leaderboard)
    if not rows:
        await _show(update, "🏆 No graded calls yet: calls are scored after each game.", [back])
        return
    esc = html.escape
    lines = ["🏆 <b>Leaderboard</b>", "Points, and vs AI: your points minus the AI's on the same games (copying the AI scores 0)", ""]
    for i, r in enumerate(rows[:15], 1):
        lines.append(f"{i}. {esc(r['name'])}: <b>{r['points']:+.1f}</b> · vs AI {r['vs_ai']:+.1f} · {r['calls']} calls")
    await _show(update, "\n".join(lines), [back])


# --- AI bets: safe, risky, crazy ----------------------------------------------------------

TIER_ICONS = {"safe": "🛡", "risky": "⚖️", "crazy": "🚀"}


def _leg_line(leg: Dict[str, Any]) -> str:
    chances = f"AI {pct(leg['probability'])}"
    if leg.get("fair") is not None:
        chances += f", bookmakers {pct(leg['fair'])}"
    return f"{html.escape(leg['label'])} {leg['odds']['american']} ({chances})"


def bets_text(data: Dict[str, Any], tier: Optional[str], league_title: str) -> str:
    esc = html.escape
    tiers = data["tiers"]
    lines = [f"🎲 <b>AI bets · {esc(league_title)}</b>", esc(data.get("note", "")), ""]
    for key in ([tier] if tier else ("safe", "risky", "crazy")):
        info = tiers[key]
        lines.append(f"{TIER_ICONS[key]} <b>{esc(info['title'])}</b>: {esc(info['rule'])}")
        bets_list = info["bets"] if tier else info["bets"][:1]
        if not bets_list:
            lines += [f"   {esc(info.get('empty') or 'Nothing fits right now.')}", ""]
            continue
        for bet in bets_list:
            kind = "Single" if len(bet["legs"]) == 1 else f"{len(bet['legs'])}-leg multi"
            lines.append(f"<b>{kind}</b> at {esc(bet['book'] or '')}: {bet['odds']['american']} · {esc(bet['badge'])}")
            lines += [f"   • {_leg_line(leg)}" for leg in bet["legs"]]
            market = f", bookmakers {pct(bet['fair_probability'])}" if bet.get("fair_probability") is not None else ""
            lines.append(f"   Chance: AI {pct(bet['probability'])}{market} · $10 returns {money(10 * bet['odds']['decimal'])}"
                         f" · expected {bet['ev'] * 10:+.2f} per $10")
            lines.append(f"   <i>{esc(bet['why'])}</i>")
        lines.append("")
    return "\n".join(lines)


async def show_bets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    tier = query.data.split(":", 1)[1] if ":" in query.data else None
    service = context.application.bot_data.get("service")
    if service is None:
        await _no_service(update)
        return
    league = context.user_data.get("league") or "basketball_nba"
    try:
        data = await asyncio.to_thread(service.ai_bets, league)
        title = next((s["title"] for s in await asyncio.to_thread(service.odds.sports) if s.get("key") == league), league)
    except Exception:
        await _show(update, "Couldn't build bets right now. Try again in a minute.",
                    [[InlineKeyboardButton("« Home", callback_data="home")]])
        return
    rows = [[InlineKeyboardButton(f"{TIER_ICONS[key]} {data['tiers'][key]['title']}", callback_data=f"bets:{key}")
             for key in ("safe", "risky", "crazy")],
            [InlineKeyboardButton("« Home", callback_data="home")]]
    await _show(update, bets_text(data, tier, title), rows)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide a list of available commands and brief instructions."""
    help_text = (
        "Available commands:\n"
        "/start – browse games, AI bets, your calls and the leaderboard\n"
        "/slip – show your multi bet slip\n"
        "/help – display this help message\n\n"
        "Pick a sport and league, then open a game to see the AI's predicted score, win, spread and "
        "total chances, injuries and projected top players. In Multi mode, tap picks to build a parlay. "
        "On NBA games, tap 🔮 to make your own call: after the game you and the AI are both scored."
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
    application.add_handler(CallbackQueryHandler(call_start, pattern=r"^call:[a-z0-9_]+:\w+$"))
    application.add_handler(CallbackQueryHandler(call_winner, pattern=r"^cw:[a-z0-9_]+:\w+:[ah]$"))
    application.add_handler(CallbackQueryHandler(call_confirm, pattern=r"^cc:[a-z0-9_]+:\w+:[ah]:(50|60|70|80|90|100)$"))
    application.add_handler(CallbackQueryHandler(my_calls, pattern=r"^mycalls$"))
    application.add_handler(CallbackQueryHandler(leaderboard, pattern=r"^board$"))
    application.add_handler(CallbackQueryHandler(show_bets, pattern=r"^bets(:(safe|risky|crazy))?$"))
    # The AI chat: plain messages, its buttons and the Ask AI button (before the catch-all below)
    setup_chat(application, config)
    # Any other button comes from an older version of the menu
    application.add_handler(CallbackQueryHandler(expired_menu))
    application.add_handler(CommandHandler("help", help_command))
    # Catch all other commands with the unknown command handler
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
