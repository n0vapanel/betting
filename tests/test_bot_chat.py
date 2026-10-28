"""The bot's AI chat: who may use it, progress while it works, the answer's buttons and what they do.

Runs the real assistant with the fake Claude client and fake Lab from test_assistant.py, and a fake
Telegram bot that records drafts, messages and edits (no network).
"""

from __future__ import annotations

import asyncio
import datetime
import html
import json
import time
from types import SimpleNamespace

import pytest
from telegram import CallbackQuery, Chat, InlineKeyboardMarkup, Message, Update, User
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, MessageHandler

from telegram_betting_bot_simple.ai.assistant import (
    DISCLOSURE,
    DISTRESS_EXTRA,
    HELPLINE,
    SUPPORT_MESSAGE,
    Assistant,
    note_warning_sign,
)
from telegram_betting_bot_simple.bot import chat, handlers
from telegram_betting_bot_simple.bot.chat import AGE_BUTTON, NO_AI_MESSAGE, PRIVATE_MESSAGE, THINKING
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.lab import paused_until
from telegram_betting_bot_simple.data.store import Store

from test_assistant import NOW, NY, FakeClaude, FakeLab, message, route, text, tool_use

CONFIG = Config(TELEGRAM_BOT_TOKEN="123456:test", TELEGRAM_ALLOWED_IDS="42", PUBLIC_URL="https://bets.example.com")


class FakeMessage:
    def __init__(self, text="", chat_id=42):
        self.text, self.chat_id = text, chat_id
        self.replies, self.edits, self.markup, self.reply_markups = [], [], None, []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        self.reply_markups.append(kwargs.get("reply_markup"))

    async def edit_text(self, text, parse_mode=None, reply_markup=None, **kwargs):
        self.edits.append(text)
        self.text, self.markup = text, reply_markup


class FakeBot:
    def __init__(self, drafts_work=True):
        self.drafts_work = drafts_work
        self.drafts, self.sent = [], []

    async def send_message_draft(self, chat_id, draft_id, text=None, **kwargs):
        if not self.drafts_work:
            raise BadRequest("Method not found")
        self.drafts.append((chat_id, draft_id, text))
        return True

    async def send_message(self, chat_id, text, parse_mode=None, reply_markup=None, **kwargs):
        sent = FakeMessage(text, chat_id)
        sent.markup = reply_markup
        self.sent.append(sent)
        return sent


class SlowLab(FakeLab):
    """Takes a moment per call, so the progress updates have time to show."""

    def build(self, uid, body):
        time.sleep(0.15)
        return super().build(uid, body)


class SlowClaude(FakeClaude):
    """A model call that takes a while with no progress line meanwhile (a long Opus round)."""

    def _create(self, **request):
        if request["model"] != "claude-haiku-5-5":
            time.sleep(0.4)
        return super()._create(**request)


@pytest.fixture()
def store(tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    store.get_or_create_user("tg:42", "Sam", now=NOW)
    store.confirm_age("tg:42", now=NOW)
    return store


def build_turn():
    return FakeClaude(
        router=[route("build", leagues=["basketball_nba"], stake=10, target=40, max_legs=3)],
        chat=[message([tool_use("ticket_build", {"leagues": ["basketball_nba"], "stake": 10, "target": 40,
                                                 "max_legs": 3})], stop="tool_use"),
              message([text("Two tickets fit your $10 to about $40 back. Both are long shots.")])],
    )


def setup(store, service, claude, bot=None, lab_class=FakeLab):
    application = Application.builder().token(CONFIG.telegram_token).build()
    handlers.setup_handlers(application, CONFIG)  # no odds key here: no games service, no assistant
    application.bot_data["assistant"] = Assistant(claude, lab_class(service, store), store, tz=NY, clock=lambda: NOW)
    return application, SimpleNamespace(application=application, user_data={}, bot=bot or FakeBot())


def typed(text, user_id=42):
    sent = FakeMessage(text)
    return SimpleNamespace(effective_message=sent, effective_user=SimpleNamespace(id=user_id, first_name="Sam",
                                                                                 username=None),
                           effective_chat=SimpleNamespace(id=user_id, type="private"), callback_query=None), sent


def tapped(data, user_id=42):
    answers = []

    async def answer(text=None, show_alert=False):
        answers.append((text, show_alert))

    edits = []

    async def edit_message_text(text, **kwargs):
        edits.append(text)

    async def edit_message_reply_markup(markup=None, **kwargs):
        edits.append(("markup", markup))

    query = SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text, answers=answers, edits=edits,
                            edit_message_reply_markup=edit_message_reply_markup)
    update = SimpleNamespace(callback_query=query, effective_message=FakeMessage(),
                             effective_user=SimpleNamespace(id=user_id, first_name="Sam", username=None),
                             effective_chat=SimpleNamespace(id=user_id, type="private"))
    return update, query


def buttons(markup: InlineKeyboardMarkup):
    return [(b.text, b.callback_data, b.web_app.url if b.web_app else None) for row in markup.inline_keyboard for b in row]


# --- who may chat ----------------------------------------------------------------------------------------

def test_strangers_get_the_private_chat_message_and_no_ai(service, store) -> None:
    claude = FakeClaude()
    _, context = setup(store, service, claude)
    update, sent = typed("who wins tonight?", user_id=7)
    asyncio.run(chat.chat_message(update, context))
    assert sent.replies == [PRIVATE_MESSAGE] and not claude.requests and not context.bot.sent


def test_without_an_api_key_the_chat_says_so(service, store) -> None:
    _, context = setup(store, service, FakeClaude())
    context.application.bot_data["assistant"] = None
    update, sent = typed("hi")
    asyncio.run(chat.chat_message(update, context))
    assert sent.replies == [NO_AI_MESSAGE]


def test_a_warning_sign_gets_the_support_message_from_anyone_ai_or_not(service, store) -> None:
    claude = FakeClaude()
    _, context = setup(store, service, claude)
    # Someone off the allowlist: no AI, but code's rules see the warning sign and answer it, with no pointer to games
    update, sent = typed("I lost my rent money on parlays and I want to kill myself", user_id=777)
    asyncio.run(chat.chat_message(update, context))
    [reply] = sent.replies
    assert reply == SUPPORT_MESSAGE + "\n\n" + DISTRESS_EXTRA and "/start" not in reply
    assert not claude.requests
    event = store.rg_events("tg:777")[0]
    assert event["action"] == "support_message,picks_off_24h" and event["detail"]["source"].startswith("rules")
    assert paused_until(store.user("tg:777"), time.time())  # the bot's AI bets honour it too
    # Without an API key, the same for the owner's friends
    context.application.bot_data["assistant"] = None
    update, sent = typed("I'm in debt and need to win my money back tonight")
    asyncio.run(chat.chat_message(update, context))
    assert sent.replies == [SUPPORT_MESSAGE] and "GAMBLER" in sent.replies[0]
    # Ordinary messages still get the usual pointers
    for user_id, expected in ((777, PRIVATE_MESSAGE), (42, NO_AI_MESSAGE)):
        update, sent = typed("who wins tonight?", user_id=user_id)
        asyncio.run(chat.chat_message(update, context))
        assert sent.replies == [expected]


# --- a turn ------------------------------------------------------------------------------------------------

def test_a_message_shows_progress_then_the_answer_with_cards_chips_and_the_app(service, store, monkeypatch) -> None:
    monkeypatch.setattr(chat, "DRAFT_EVERY", 0.02)
    _, context = setup(store, service, build_turn(), lab_class=SlowLab)
    update, _ = typed("NBA, $10 to get about $40 back, max 3 legs")
    asyncio.run(chat.chat_message(update, context))

    bot = context.bot
    assert bot.drafts[0][2] == ""  # an empty draft: Telegram's own "Thinking…" straight away
    assert len({draft_id for _, draft_id, _ in bot.drafts}) == 1  # one draft, updated in place
    assert any("Building tickets…" in (text or "") for _, _, text in bot.drafts)
    [answer] = bot.sent
    assert answer.text.startswith(f"<i>🤖 {html.escape(DISCLOSURE)}</i>")
    assert "Two tickets fit your $10 to about $40 back." in answer.text
    assert "🧾 <b>Single</b>" in answer.text and "Chance it wins: AI 1 in" in answer.text
    assert "of today's AI credits left" in answer.text
    assert f"<i>{html.escape(HELPLINE)}</i>" in answer.text  # under tickets, the helpline (as in the app)
    labels = buttons(answer.markup)
    thread = context.user_data["ai_thread"]
    # thread : this answer's place in the thread (message 1, after the question) : card or chip
    assert ("➕ Add ticket 1 to slip", f"ai:add:{thread}:1:0", None) in labels
    assert ("Explain the risk", f"ai:{thread}:1:1", None) in labels
    assert labels[-1] == ("📱 Open in app", None, "https://bets.example.com/#ask")
    assert all(len(data.encode()) <= 64 for _, data, _ in labels if data)


def test_without_drafts_a_thinking_message_becomes_the_answer(service, store) -> None:
    claude = FakeClaude(router=[route("chat")], chat=[message([text("Hello! Ask me about tonight's games.")])])
    _, context = setup(store, service, claude, bot=FakeBot(drafts_work=False))
    update, _ = typed("hello")
    asyncio.run(chat.chat_message(update, context))
    [thinking] = context.bot.sent
    assert thinking.edits and "Hello! Ask me about tonight&#x27;s games." in thinking.text
    assert THINKING not in thinking.text


def test_a_quiet_spell_starts_a_new_thread_with_the_disclosure_again(service, store) -> None:
    claude = FakeClaude(router=[route("chat"), route("chat"), route("chat")],
                        chat=[message([text("Hi.")]), message([text("Hi again.")]), message([text("Welcome back.")])])
    _, context = setup(store, service, claude)
    for words in ("hi", "hi again"):
        asyncio.run(chat.chat_message(typed(words)[0], context))
    first = context.user_data["ai_thread"]
    assert html.escape(DISCLOSURE) in context.bot.sent[0].text
    assert html.escape(DISCLOSURE) not in context.bot.sent[1].text  # same thread: disclosed once
    context.user_data["ai_last"] = time.time() - chat.SESSION_GAP - 1
    asyncio.run(chat.chat_message(typed("back")[0], context))
    assert context.user_data["ai_thread"] != first and html.escape(DISCLOSURE) in context.bot.sent[2].text


# --- the answer's buttons ------------------------------------------------------------------------------------

def test_a_chip_is_sent_as_if_typed(service, store) -> None:
    claude = build_turn()
    _, context = setup(store, service, claude)
    asyncio.run(chat.chat_message(typed("NBA, $10 to get about $40 back")[0], context))
    thread = context.user_data["ai_thread"]
    claude.router.append(route("explain"))
    claude.chat.append(message([text("Most of the time a ticket like this loses.")]))
    update, query = tapped(f"ai:{thread}:1:1")  # after "Compare these tickets"
    asyncio.run(chat.chip_tapped(update, context))
    assert query.answers == [(None, False)]
    router_input = [r for r in claude.requests if r["model"] == "claude-haiku-5-5"][-1]["messages"][0]["content"]
    assert router_input.endswith("Latest message:\nExplain the risk")
    assert "<i>› Explain the risk</i>" in context.bot.sent[-1].text
    for data in (f"ai:{thread}:1:7", f"ai:{thread}:0:0", f"ai:{thread}:9:0"):  # no such chip, a question, no message
        stale, query = tapped(data)
        asyncio.run(chat.chip_tapped(stale, context))
        assert "expired" in query.answers[0][0], data


def test_add_to_slip_puts_the_ticket_on_the_bots_slip_only_when_tapped(service, store) -> None:
    _, context = setup(store, service, build_turn())
    asyncio.run(chat.chat_message(typed("NBA, $10 to get about $40 back")[0], context))
    assert not context.user_data.get("picks")  # the AI never adds legs by itself
    update, query = tapped(f"ai:add:{context.user_data['ai_thread']}:1:0")
    asyncio.run(chat.add_to_slip(update, context))
    assert context.user_data["mode"] == "multi" and len(context.user_data["picks"]) == 1
    assert context.user_data["picks"][0]["league"] == "basketball_nba"
    assert query.answers[0][0].startswith("Added 1 leg(s) to your slip") and "one parlay" in query.answers[0][0]


def _proposal(ticket, title):
    return [message([tool_use("slip_propose", {**ticket, "title": title})], stop="tool_use"),
            message([text(f"Here is {title}.")])]


def test_a_button_under_an_older_answer_acts_on_that_answer(service, store) -> None:
    a = {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"}
    b = {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "away"}
    claude = FakeClaude(router=[route("build", leagues=["basketball_nba"], stake=10, target=40)] * 2,
                        chat=_proposal({"type": "single", "stake": 10, "legs": [a]}, "ticket A")
                        + _proposal({"type": "single", "stake": 10, "legs": [b]}, "ticket B"))
    _, context = setup(store, service, claude)
    asyncio.run(chat.chat_message(typed("NBA $10 to get $40 back")[0], context))
    asyncio.run(chat.chat_message(typed("another one")[0], context))
    under_a = next(data for label, data, _ in buttons(context.bot.sent[0].markup) if label.startswith("➕"))
    under_b = next(data for label, data, _ in buttons(context.bot.sent[1].markup) if label.startswith("➕"))
    assert under_a != under_b
    update, query = tapped(under_a)  # the person scrolls up and taps "Add to slip" under ticket A
    asyncio.run(chat.add_to_slip(update, context))
    assert [(p["id"], p["side"]) for p in context.user_data["picks"]] == [("g1", "home")]  # A's leg, not B's
    # A chip under the first answer sends that answer's chip, even though a newer answer exists
    claude.router.append(route("explain"))
    claude.chat.append(message([text("Most of the time a single like this loses.")]))
    first_chip = next(data for label, data, _ in buttons(context.bot.sent[0].markup) if label == "Explain the risk")
    update, query = tapped(first_chip)
    asyncio.run(chat.chip_tapped(update, context))
    assert "<i>› Explain the risk</i>" in context.bot.sent[-1].text


def test_singles_and_system_bets_aren_t_squeezed_into_the_bot_s_parlay_slip(service, store) -> None:
    legs = [{"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"},
            {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "home"}]
    claude = FakeClaude(router=[route("build", leagues=["basketball_nba"], stake=10, target=40)],
                        chat=_proposal({"type": "single", "stake": 10, "split": True, "legs": legs}, "two singles"))
    _, context = setup(store, service, claude)
    asyncio.run(chat.chat_message(typed("NBA $10 to get $40 back, as singles")[0], context))
    answer = context.bot.sent[0]
    assert not [label for label, _, _ in buttons(answer.markup) if label.startswith("➕")]
    assert "go on the app's slip" in answer.text and "📱 Open in app" in [label for label, _, _ in buttons(answer.markup)]
    # Even a button made by hand can't turn the two singles into a parlay
    update, query = tapped(f"ai:add:{context.user_data['ai_thread']}:1:0")
    asyncio.run(chat.add_to_slip(update, context))
    assert not context.user_data.get("picks") and "holds one parlay" in query.answers[0][0]
    assert len(query.answers[0][0]) <= 200  # Telegram's limit for an alert


def test_a_slow_model_call_keeps_the_draft_alive(service, store, monkeypatch) -> None:
    monkeypatch.setattr(chat, "DRAFT_EVERY", 0.02)
    monkeypatch.setattr(chat, "DRAFT_KEEPALIVE", 0.1)
    claude = SlowClaude(router=[route("chat")], chat=[message([text("Hello!")])])
    _, context = setup(store, service, claude)
    asyncio.run(chat.chat_message(typed("hello")[0], context))
    texts = [text for _, _, text in context.bot.drafts]
    # No new progress line arrives during the slow call, so the same draft is sent again to keep it showing
    assert any(a == b for a, b in zip(texts, texts[1:])), texts
    assert context.bot.sent[-1].text.endswith("of today's AI credits left</i>")


def test_the_age_gate_offers_a_button_that_confirms(service, tmp_path) -> None:
    store = Store(str(tmp_path / "fresh.db"))
    claude = FakeClaude()
    _, context = setup(store, service, claude)
    asyncio.run(chat.chat_message(typed("hi")[0], context))
    labels = buttons(context.bot.sent[0].markup)
    assert labels[0] == (AGE_BUTTON, "ai:age", None) and not claude.requests
    update, query = tapped("ai:age")
    asyncio.run(chat.confirm_age(update, context))
    assert store.user("tg:42")["age_confirmed_at"] is not None and query.edits


def test_a_stranger_s_button_taps_do_nothing(service, store) -> None:
    _, context = setup(store, service, build_turn())
    for handler, data in ((chat.chip_tapped, "ai:1:1:0"), (chat.add_to_slip, "ai:add:1:1:0"), (chat.confirm_age, "ai:age")):
        update, query = tapped(data, user_id=7)
        asyncio.run(handler(update, context))
        assert query.answers == [(PRIVATE_MESSAGE, True)]
    assert store.user("tg:7") is None


# --- registration and the home screen ----------------------------------------------------------------------

def test_the_chat_is_registered_before_the_catch_all_and_home_has_ask_ai(service, store) -> None:
    application, context = setup(store, service, FakeClaude())
    registered = application.handlers[0]
    buttons_ = [h for h in registered if isinstance(h, CallbackQueryHandler)]
    assert buttons_[-1].pattern is None  # expired menus still come last
    patterns = [h.pattern.pattern for h in buttons_[:-1]]
    assert r"^ask$" in patterns and r"^ai:\d{1,12}:\d{1,6}:\d$" in patterns and r"^ai:add:\d{1,12}:\d{1,6}:\d$" in patterns
    tap = CallbackQuery("1", User(42, "Sam", False), "chat", data="ai:3:1")
    old = [h for h in buttons_ if h.check_update(Update(9, callback_query=tap))]
    assert old == [buttons_[-1]]  # a button from before answers were numbered: "this menu has expired"
    [text] = [h for h in registered if isinstance(h, MessageHandler) and h.callback is chat.chat_message]
    assert text.block is False  # a slow AI turn doesn't hold up other people's taps
    chat_, sam = Chat(42, "private"), User(42, "Sam", False)
    sent = Message(1, datetime.datetime.now(datetime.timezone.utc), chat_, from_user=sam, text="hi")
    edited = Message(1, datetime.datetime.now(datetime.timezone.utc), chat_, from_user=sam, text="hi!")
    assert text.check_update(Update(1, message=sent))
    assert not text.check_update(Update(2, edited_message=edited))  # an edit doesn't start a second turn

    screens = []

    async def edit_message_text(text, reply_markup=None, **kwargs):
        screens.append((text, buttons(reply_markup)))

    async def answer(*args, **kwargs):
        pass

    query = SimpleNamespace(data="home", answer=answer, edit_message_text=edit_message_text)
    update = SimpleNamespace(callback_query=query, message=None, effective_chat=SimpleNamespace(type="private"),
                             effective_user=SimpleNamespace(id=42))
    asyncio.run(handlers.go_home(update, context))
    assert ("💬 Ask AI", "ask", None) in screens[-1][1]
    asyncio.run(chat.ask_ai(update, context))
    assert screens[-1][0].startswith("💬 <b>Ask AI</b>")
    assert ("📱 Open in app", None, "https://bets.example.com/#ask") in screens[-1][1]


def test_long_answers_fit_one_telegram_message_and_are_escaped() -> None:
    card = {"type": "ticket", "title": "3-leg multi", "ticket": {"legs": [], "total_stake": "10.00",
                                                                  "max_payout": "60.00", "odds": {"decimal": 6.0}}}
    result = {"thread_id": 1, "credits": {"left_cents": 80, "daily_cents": 100},
              "reply": {"text": "<b>& " * 3000, "cards": [card] * 3, "chips": []}}
    out = chat.reply_text(result)
    assert len(out) <= chat.MAX_TEXT and "&lt;b&gt;&amp;" in out and out.count("🧾") >= 1
    assert "<b>&" not in out.replace("<b>3-leg multi</b>", "")
    json.dumps(out)  # plain text, nothing odd left behind


def test_a_ticket_card_shows_the_code_s_numbers_and_never_a_negative_cut() -> None:
    ticket = {"legs": [{"label": "New York Knicks to win", "odds": {"decimal": 1.38}, "ai_prob": 0.71}],
              "total_stake": "10.00", "max_payout": "13.80", "odds": {"decimal": 1.38}, "book_cut": -0.05,
              "chance": {"headline": "every leg wins", "ai": 0.71, "market": 0.7, "ai_one_in": 1, "market_one_in": 1},
              "expected": {"ai": "-0.20", "market": "-0.34"}, "most_likely": {"result": "profit", "probability": 0.71}}
    text = chat.card_text({"type": "ticket", "title": "Single: New York Knicks to win", "ticket": ticket})
    assert "• New York Knicks to win 1.38 · AI 71%" in text
    assert "Stake $10.00 → up to $13.80 (odds 1.38)" in text
    assert "Best prices beat the market's view by 5.0% · expected result -$0.20 (AI)" in text
    assert "keeps" not in text


# --- the bot's AI bets and betting screens follow the same account gates ------------------------------------

def test_the_bot_s_ai_bets_honour_the_pause_the_break_and_the_age_check(service, store) -> None:
    service.store = store
    application, context = setup(store, service, FakeClaude())
    application.bot_data["service"] = service
    shown = []

    def tap(handler, data, user_id=42):
        async def answer(*args, **kwargs):
            pass

        async def edit_message_text(text, reply_markup=None, **kwargs):
            shown.append((text, buttons(reply_markup) if reply_markup else []))

        query = SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text)
        update = SimpleNamespace(callback_query=query, message=None, effective_chat=SimpleNamespace(id=user_id, type="private"),
                                 effective_user=SimpleNamespace(id=user_id, first_name="Sam", username=None))
        asyncio.run(handler(update, context))
        return shown[-1]

    # tg:42 has confirmed their age and has no pause or break: the AI's bets show
    assert "🛡 <b>Safe</b>" in tap(handlers.show_bets, "bets")[0]
    # After a warning sign in the chat, picks are paused: no bets, the helpline instead (browsing stays open)
    note_warning_sign(application.bot_data["assistant"].lab, "tg:42", ["chasing"], "rules", "I need to win my money back")
    text, labels = tap(handlers.show_bets, "bets:safe")
    assert text.startswith("⏸ Picks are paused until") and "GAMBLER" in text and "Safe" not in text
    assert "Clippers @ Thunder" in tap(handlers.league_selected, "league:basketball_nba")[0]
    # A break the person chose: no AI bets, no payouts, no slip
    store.get_or_create_user("tg:43")
    store.confirm_age("tg:43")
    store.start_cooloff("tg:43", 72)
    assert tap(handlers.show_bets, "bets", user_id=43)[0].startswith("⏸ You're on a break until")
    pick = handlers.pick_data("basketball_nba", "g2", "h2h", "away")
    assert tap(handlers.pick_selected, pick, user_id=43)[0].startswith("⏸ You're on a break until")
    assert tap(handlers.show_slip, "slip", user_id=43)[0].startswith("⏸ You're on a break until")
    # No age confirmed yet: the age button, and once confirmed the bets
    text, labels = tap(handlers.show_bets, "bets", user_id=44)
    assert "18 or older" in text and labels[0] == (AGE_BUTTON, "age:bets", None)
    assert "🛡 <b>Safe</b>" in tap(handlers.confirm_age_for_bets, "age:bets", user_id=44)[0]
    assert store.user("tg:44")["age_confirmed_at"] is not None


# --- a pasted slip: photos, image files and slip text ----------------------------------------------------

from test_lab import Players  # noqa: E402
from test_slip_reader import NOW as PASTE_NOW  # noqa: E402
from test_slip_reader import PNG, SLIP_TEXT, SlipClaude, answer_turn, extraction, read  # noqa: E402

from telegram_betting_bot_simple.dashboard import lab as lab_module  # noqa: E402

from telegram_betting_bot_simple.ai.assistant import CONFIRM_CHIP, FIX_CHIP  # noqa: E402
from telegram_betting_bot_simple.engine.pasted import NO_LEG_ODDS  # noqa: E402


class FakeFile:
    def __init__(self, data):
        self.data = data

    async def download_as_bytearray(self):
        return bytearray(self.data)


class FakeImage:
    """A PhotoSize or a Document: its size, and the bytes get_file() hands back."""

    def __init__(self, width=0, height=0, data=PNG, file_size=None, mime_type=None):
        self.width, self.height, self.data, self.mime_type = width, height, data, mime_type
        self.file_size = len(data) if file_size is None else file_size
        self.fetched = False

    async def get_file(self):
        self.fetched = True
        return FakeFile(self.data)


def sent_image(photos=(), document=None, caption=None, user_id=42):
    sent = FakeMessage()
    sent.text, sent.photo, sent.document, sent.caption = None, list(photos), document, caption
    update = SimpleNamespace(effective_message=sent, effective_user=SimpleNamespace(id=user_id, first_name="Sam",
                                                                                    username=None),
                             effective_chat=SimpleNamespace(id=user_id, type="private"), callback_query=None)
    return update, sent


@pytest.fixture()
def paste_store(tmp_path):
    store = Store(str(tmp_path / "paste.db"))
    store.get_or_create_user("tg:42", "Sam", now=PASTE_NOW)
    store.confirm_age("tg:42", now=PASTE_NOW)
    return store


def paste_setup(store, service, claude):
    """The bot with the real Assistant over the real Lab (the shared fake board, fake ESPN player data), both on
    the pasted-slip tests' clock, hours before the board's games."""
    application = Application.builder().token(CONFIG.telegram_token).build()
    handlers.setup_handlers(application, CONFIG)
    lab = lab_module.Lab(service, store, players=Players(game_date="2030-01-01T00:00:00Z"), now=lambda: PASTE_NOW)
    application.bot_data["assistant"] = Assistant(claude, lab, store, tz=NY, clock=lambda: PASTE_NOW)
    return application, SimpleNamespace(application=application, user_data={}, bot=FakeBot())


def test_a_photo_of_a_slip_is_read_from_the_largest_size_up_to_1600_px(service, paste_store) -> None:
    claude = answer_turn()
    _, context = paste_setup(paste_store, service, claude)
    sizes = [FakeImage(90, 160), FakeImage(720, 1280), FakeImage(1440, 2560)]
    update, _ = sent_image(sizes)
    asyncio.run(chat.chat_photo(update, context))
    assert [size.fetched for size in sizes] == [False, True, False]
    [request] = claude.reads()
    assert request["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    [answer] = context.bot.sent
    assert "I read your slip as a 3-leg multi at DraftKings" in answer.text
    assert "📋 <b>Your slip · DraftKings · multi · $10.00 stake</b>" in answer.text
    assert "✓ New York Knicks to win · 1.36" in answer.text and "👤 LeBron James over 27.5 points · +120" in answer.text
    assert "✓ Matches your bookmaker&#x27;s $33.21" in answer.text or "✓ Matches your bookmaker's $33.21" in answer.text
    assert f"<i>{html.escape(HELPLINE)}</i>" in answer.text  # a ticket's numbers come with the helpline
    labels = buttons(answer.markup)
    assert ("What if I drop the weakest leg?", f"ai:{context.user_data['ai_thread']}:1:0", None) in labels
    assert labels[-1] == ("📱 Open in app", None, "https://bets.example.com/#ask")
    # Every size bigger than 1600 px: the smallest is read
    claude.reader.append(read())
    claude.chat.append(message([text("The weakest leg is LeBron James at +120.")]))
    big = [FakeImage(1700, 2000), FakeImage(2400, 3000)]
    asyncio.run(chat.chat_photo(sent_image(big)[0], context))
    assert [size.fetched for size in big] == [True, False]


def test_an_image_file_is_read_too_but_not_one_too_big_or_not_an_image(service, paste_store) -> None:
    claude = answer_turn()
    _, context = paste_setup(paste_store, service, claude)
    asyncio.run(chat.chat_photo(sent_image(document=FakeImage(mime_type="image/png"))[0], context))
    assert claude.reads() and "I read your slip as" in context.bot.sent[-1].text
    for document, expected in ((FakeImage(mime_type="image/png", file_size=4_000_000), chat.TOO_BIG),
                               (FakeImage(mime_type="application/pdf"), chat.NOT_AN_IMAGE)):
        update, sent = sent_image(document=document)
        asyncio.run(chat.chat_photo(update, context))
        assert sent.replies == [expected] and not document.fetched
    # A file that says it is an image but isn't one: refused by its bytes
    update, sent = sent_image(document=FakeImage(mime_type="image/png", data=b"GIF89a" + b"\x00" * 40))
    asyncio.run(chat.chat_photo(update, context))
    assert "JPEG, PNG or WebP" in context.bot.sent[-1].text and len(claude.reads()) == 1


def test_slip_text_sent_to_the_bot_is_read_as_a_slip(service, paste_store) -> None:
    claude = answer_turn()
    _, context = paste_setup(paste_store, service, claude)
    asyncio.run(chat.chat_message(typed(SLIP_TEXT)[0], context))
    assert claude.reads() and not [r for r in claude.requests if r["model"] == "claude-haiku-5-5"
                                   and r not in claude.reads()]  # no router call: code spotted the slip
    assert "I read your slip as a 3-leg multi" in context.bot.sent[-1].text


def test_a_doubtful_read_asks_in_telegram_and_the_chip_confirms(service, paste_store) -> None:
    doubtful = extraction()
    doubtful["legs"][0]["odds_text"] = "1.63"
    claude = SlipClaude(reader=[read(doubtful)])
    _, context = paste_setup(paste_store, service, claude)
    asyncio.run(chat.chat_message(typed(SLIP_TEXT)[0], context))
    thread = context.user_data["ai_thread"]
    labels = buttons(context.bot.sent[-1].markup)
    assert (CONFIRM_CHIP, f"ai:{thread}:1:0", None) in labels and (FIX_CHIP, f"ai:{thread}:1:1", None) in labels
    assert not claude.chat_requests()
    # What was read and why it's doubtful, and none of the code's numbers until the person has checked it
    asking = context.bot.sent[-1].text
    assert "✓ New York Knicks to win · 1.63" in asking and chat.CHECK_FIRST in asking
    for number in ("Stake $", "Chance it wins", "expected result", "Best prices", "%"):
        assert number not in asking, number
    claude.chat.append(message([text("The weakest leg is LeBron James at +120.")]))
    update, query = tapped(f"ai:{thread}:1:0")
    asyncio.run(chat.chip_tapped(update, context))
    assert "The weakest leg is LeBron James at +120." in context.bot.sent[-1].text
    assert "Stake $10.00 → up to" in context.bot.sent[-1].text and chat.CHECK_FIRST not in context.bot.sent[-1].text


def test_strangers_photos_get_the_private_message_and_a_caption_warning_sign_gets_support(service, paste_store) -> None:
    claude = SlipClaude()
    _, context = paste_setup(paste_store, service, claude)
    image = FakeImage(720, 1280)
    update, sent = sent_image([image], user_id=7)
    asyncio.run(chat.chat_photo(update, context))
    assert sent.replies == [PRIVATE_MESSAGE] and not image.fetched
    update, sent = sent_image([image], caption="need this to win my money back", user_id=7)
    asyncio.run(chat.chat_photo(update, context))
    assert sent.replies == [SUPPORT_MESSAGE] and not claude.requests and not image.fetched
    # From the owner's friends, the caption's warning sign is answered before any read
    update, _ = sent_image([image], caption="I lost my rent money, need to win it back")
    asyncio.run(chat.chat_photo(update, context))
    assert html.escape(SUPPORT_MESSAGE[:40]) in context.bot.sent[-1].text and not claude.requests
    assert not image.fetched


def test_without_the_ai_the_free_reader_still_reads_slip_text(service, paste_store) -> None:
    _, context = paste_setup(paste_store, service, SlipClaude())
    lab = context.application.bot_data["assistant"].lab
    context.application.bot_data["assistant"] = None
    context.application.bot_data["service"] = SimpleNamespace(lab=lab)
    update, sent = typed(SLIP_TEXT)
    asyncio.run(chat.chat_message(update, context))
    [reply] = sent.replies
    assert "I read your slip as a 3-leg multi" in reply and html.escape(chat.FREE_READER_NOTE) in reply
    assert "📋 <b>Your slip" in reply and "matches ours" in reply
    # A screenshot needs the AI; ordinary chat still gets the usual pointer
    update, sent = sent_image([FakeImage(720, 1280)])
    asyncio.run(chat.chat_photo(update, context))
    assert sent.replies == [chat.NO_AI_IMAGE]
    update, sent = typed("hi")
    asyncio.run(chat.chat_message(update, context))
    assert sent.replies == [NO_AI_MESSAGE]


# --- review fixes: the free reader's doubtful reads and age gate, same-game parlays, currencies ----------------

def free_bot(service, store):
    """The bot with no AI chat (no API key), its free reader over the real Lab."""
    _, context = paste_setup(store, service, SlipClaude())
    lab = context.application.bot_data["assistant"].lab
    context.application.bot_data["assistant"] = None
    context.application.bot_data["service"] = SimpleNamespace(lab=lab)
    return context, lab


def test_without_the_ai_a_doubtful_read_asks_first_and_the_button_gives_the_numbers(service, paste_store) -> None:
    context, lab = free_bot(service, paste_store)
    update, sent = typed(SLIP_TEXT.replace("Payout: $33.21", "Payout: $40.00"))
    asyncio.run(chat.chat_message(update, context))
    [reply] = sent.replies
    assert "I read your slip as a 3-leg multi" in reply and "- Your bookmaker says $40; at the odds read" in reply
    assert html.escape(chat.FREE_CHECK) in reply and chat.CHECK_FIRST in reply
    assert "Stake $10.00 →" not in reply and "On average" not in reply and "Chance it wins" not in reply
    label, callback, _ = buttons(sent.reply_markups[0])[0]
    pasted_id = int(callback.split(":")[2])
    assert (label, callback) == (chat.CONFIRM_BUTTON, f"paste:ok:{pasted_id}") and lab.store.pasted("tg:42", pasted_id)
    # A stranger's tap is refused (the Lab finds only the person's own slip anyway); theirs gets the numbers
    other, query = tapped(f"paste:ok:{pasted_id}", user_id=7)
    asyncio.run(chat.free_confirm(other, context))
    assert query.answers == [(PRIVATE_MESSAGE, True)]
    update, query = tapped(f"paste:ok:{pasted_id}")
    asyncio.run(chat.free_confirm(update, context))
    [numbers] = update.effective_message.replies
    assert "Stake $10.00 → up to $33.24" in numbers and chat.CHECK_FIRST not in numbers
    assert "Your bookmaker says $40.00; we get $33.24: check the odds on your slip." in numbers
    assert ("markup", None) in query.edits  # the button has done its job
    assert chat.CONFIRM_BUTTON not in json.dumps(buttons(update.effective_message.reply_markups[0]))


def test_without_the_ai_the_age_gate_comes_with_a_button_that_works(service, tmp_path) -> None:
    store = Store(str(tmp_path / "young.db"))
    context, lab = free_bot(service, store)
    update, sent = typed(SLIP_TEXT)
    asyncio.run(chat.chat_message(update, context))
    assert sent.replies == [chat.FREE_AGE]
    assert buttons(sent.reply_markups[0]) == [(AGE_BUTTON, "paste:age", None)]
    assert not store.pasted_for_thread("tg:42", None)
    update, query = tapped("paste:age")
    asyncio.run(chat.free_age(update, context))
    assert query.answers == [("Thanks!", False)] and query.edits == [chat.FREE_AGE_DONE]
    assert store.user("tg:42")["age_confirmed_at"]
    update, sent = typed(SLIP_TEXT)
    asyncio.run(chat.chat_message(update, context))
    assert "I read your slip as a 3-leg multi" in sent.replies[0]


def test_without_the_ai_a_same_game_parlay_is_read_but_not_priced(service, paste_store) -> None:
    context, _ = free_bot(service, paste_store)
    update, sent = typed("DraftKings\nSGP\nKnicks @ Thunder\nJalen Brunson 25+ Points\nKnicks Moneyline\n+350\n"
                         "Wager $10.00\nTo Win $35.00")
    asyncio.run(chat.chat_message(update, context))
    [reply] = sent.replies
    assert "I read your slip as a 2-leg multi at DraftKings" in reply and html.escape(NO_LEG_ODDS) in reply
    assert "Stake $10.00 →" not in reply and "odds unread" in reply


def test_a_slips_currency_can_never_put_markup_in_the_bots_message(service, paste_store) -> None:
    evil = '<a href="https://evil.example/x">$</a>'
    claude = SlipClaude(reader=[read(extraction(currency=evil))], chat=[message([text("Noted.")])])
    _, context = paste_setup(paste_store, service, claude)
    asyncio.run(chat.chat_message(typed(SLIP_TEXT)[0], context))
    answer = context.bot.sent[-1].text
    assert "evil.example" not in answer and "Stake $10.00 → up to $33.24" in answer  # the stake's own "$"
    # Whatever a card holds, the money in it is escaped
    card = {"type": "pasted", "pasted": {"currency": "A&B<i>", "stake": "10.00", "legs": []},
            "analysis": {"ticket": {"total_stake": "10.00", "max_payout": "20.00", "odds": {"decimal": 2.0},
                                    "expected": {"market": "-1.00"}, "book_cut": 0.1}}}
    text_ = chat.card_text(card)
    assert "<i>" not in text_ and "A&amp;B&lt;i&gt; 10.00" in text_ and "A&amp;B&lt;i&gt; 1.00" in text_
