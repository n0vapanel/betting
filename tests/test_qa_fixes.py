"""The QA sweep's fix round (scratchpad brief FIX_QA.md): one test per item, each failing without its fix.

B (a crafted request never gets a 500):
  1. a JSON body with NaN (or Infinity) is a 422 that never repeats it; a NaN can't be kept in the builder's state
  2. JSON nested thousands deep is a 422 without the nesting repeated
  3. a row id too large for SQLite is a 422 (path ids and a body's thread id); lab_call turns an OverflowError into one
M:
  4. long text with no spaces doesn't push the Help search or the Player finder sideways at 360 and 390 px
  5. Make your call checks the name by the server's rule, says so plainly, sends "’" as "'" and keeps the name only
     once the server took it
  6. /brain says the brain-vs-market gap as better or worse, never as a (negative) log loss
F:
  7. the every-game lines: heading and note agree with "Priced picks", "(1 game)", the headline fits the record, and
     each count ("3 pending") stays whole on a phone
  8. a malformed g: button gets the expired-menu answer, not a ValueError
  9. no API credits: one plain sentence, without the credits-left tail
  10. AI bets open on a league with a model, can switch league, and the note matches the league shown
  11. the player-name searches take 60 characters and say plainly when the server refuses a longer one
  12. New chat on an empty chat says it's already one; Select all / Clear and "All sports" say when they'd change
      nothing; the rollover chart has no pointer cursor

The screens run in Chromium (Playwright for Node, as tests/test_capture_health.py does), the app's own modules from
this tree's static folder with js/app.js stubbed and every /api/ request answered here: no server, no network. Skipped
when Node or Playwright isn't installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from telegram.ext import Application

from telegram_betting_bot_simple.ai.explain import NO_CREDITS
from telegram_betting_bot_simple.bot import chat as bot_chat, handlers
from telegram_betting_bot_simple.bot.handlers import setup_handlers
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.app import MAX_ROW_ID, create_app
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import every_game, loop
from telegram_betting_bot_simple.learn.progress import brain_lines, progress
from tests import test_capture_health
from tests.test_lab import ME, UID, Clock, Odds, Players

STATIC = test_capture_health.STATIC
NBA = "basketball_nba"
MMA = "mma_mixed_martial_arts"


@pytest.fixture()
def lab(service, tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    service.odds = Odds()
    service.store = store
    lab = Lab(service, store, players=Players(), public_url="https://bets.example.com", now=Clock())
    lab.update_me(UID, {"confirm_age": True})
    return lab


@pytest.fixture()
def client(lab):
    client = TestClient(create_app(lab.service, "s3cret", login_delay=0), raise_server_exceptions=False)
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(ME)
    return client


def send(client: TestClient, method: str, path: str, raw: str):
    return client.request(method, path, content=raw.encode(), headers={"Content-Type": "application/json"})


# --- 1 · NaN in a body ------------------------------------------------------------------------------------------------

NAN_BODIES = [
    ("POST", "/api/slip", '{"mode": NaN}'),
    ("POST", "/api/slip", '{"stake": NaN}'),
    ("POST", "/api/slip", '{"stakes": {"0": NaN}}'),
    ("POST", "/api/slip", '{"stakes": {"0": Infinity}}'),
    ("POST", "/api/calls", '{"league": NaN, "id": "g1", "player": "browser-1234", "name": "Sam", "call": {}}'),
    ("PUT", "/api/me", '{"odds_format": NaN}'),
    ("POST", "/api/me/cooloff", '{"hours": NaN}'),
    ("POST", "/api/tickets/price", '{"type": "multi", "stake": NaN, "legs": []}'),
    ("POST", "/api/plans", '{"start": NaN, "days": 3}'),
    ("POST", "/api/builder/run", '{"controls": {"target_odds": -Infinity}}'),
]


@pytest.mark.parametrize("method,path,raw", NAN_BODIES)
def test_a_nan_in_a_body_is_a_422_that_never_repeats_it(client, method, path, raw) -> None:
    # (an Infinity stake for a real pick was a 500 too, from the reply: stakes are now bounded as stake is)
    response = send(client, method, path, raw)
    assert response.status_code == 422, response.text
    assert not re.search(r"nan|infinity", response.text, re.I), response.text
    assert all({"type", "loc", "msg"} <= set(error) for error in response.json()["detail"])


def test_a_422_still_repeats_an_input_json_can_carry(client) -> None:
    [error] = send(client, "POST", "/api/slip", '{"mode": "both"}').json()["detail"]
    assert error["loc"] == ["body", "mode"] and error["input"] == "both"


def test_the_builders_kept_state_can_never_hold_a_nan(client) -> None:
    # It used to be taken and kept in users.settings as sent (JSON itself has no NaN)
    assert send(client, "PUT", "/api/me", '{"builder": {"ui": {"x": NaN}}}').status_code == 422
    assert send(client, "POST", "/api/builder/run", '{"controls": {"ui": {"x": Infinity}}}').status_code == 422
    assert client.get("/api/me").status_code == 200
    assert client.put("/api/me", json={"builder": {"ui": {"x": 1.5, "y": "a"}}}).status_code == 200  # still kept


# --- 2 · JSON nested thousands deep -----------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["[" * 2000 + "]" * 2000, '{"picks": ' + "[" * 3000 + "]" * 3000 + "}",
                                 '{"mode": ' + '{"a": ' * 1500 + "1" + "}" * 1500 + "}"])
def test_json_nested_thousands_deep_is_a_short_422(client, raw) -> None:
    response = send(client, "POST", "/api/slip", raw)
    assert response.status_code == 422 and len(response.text) < 1000, response.text[:300]


# --- 3 · a row id too large for SQLite --------------------------------------------------------------------------------

HUGE = "99999999999999999999"
ID_ROUTES = [("GET", f"/api/slips/{HUGE}"), ("DELETE", f"/api/slips/{HUGE}"), ("GET", f"/api/plans/{HUGE}"),
             ("POST", f"/api/plans/{HUGE}/stop"), ("POST", f"/api/plans/{HUGE}/day"), ("GET", f"/api/pasted/{HUGE}"),
             ("POST", f"/api/pasted/{HUGE}/analyse"), ("GET", f"/api/chat/{HUGE}"),
             ("DELETE", f"/api/calls/{HUGE}?player=browser-1234")]


@pytest.mark.parametrize("method,path", ID_ROUTES)
def test_a_huge_row_id_is_a_422_not_a_crash(client, method, path) -> None:
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code == 422, response.text
    assert client.request(method, path.replace(HUGE, str(MAX_ROW_ID)),
                          json={} if method == "POST" else None).status_code in (400, 404, 422)


def test_a_huge_thread_id_in_a_body_is_a_422(client) -> None:
    for path, body in (("/api/chat", {"message": "hi"}), ("/api/pasted", {"text": "Lakers ML -150"}),
                       ("/api/chat/slip", {"text": "Lakers ML -150"})):
        response = client.post(path, json={**body, "thread_id": int(HUGE)})
        assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", "thread_id"], path


def test_lab_call_turns_an_overflow_into_a_422(lab, client, monkeypatch) -> None:
    def overflow(*_: Any, **__: Any) -> Any:
        raise OverflowError("Python int too large to convert to SQLite INTEGER")
    monkeypatch.setattr(Lab, "get_slip", overflow)  # the app's own Lab, made in create_app
    response = client.get("/api/slips/1")
    assert response.status_code == 422 and response.json() == {"detail": "A number in this request is too large"}


# --- 6 · /brain's gap ---------------------------------------------------------------------------------------------

def test_brain_says_the_gap_as_better_or_worse_never_as_a_negative_log_loss() -> None:
    qa = {"n": 2, "gap": -0.4479, "lo": -0.5513, "hi": -0.3446, "word": "too few games to say yet"}
    text = loop.headline_text("nhl", qa)
    assert text == ("NHL, last 2 games: too few games to say yet (brain vs market: 0.4479 better on log loss, "
                    "95% range 0.3446 to 0.5513 better).")
    worse = loop.headline_text("nba", {"n": 300, "gap": 0.012, "lo": 0.004, "hi": 0.02, "word": "behind the sharp market"})
    assert worse.endswith("(brain vs market: 0.0120 worse on log loss, 95% range 0.0040 to 0.0200 worse).")
    memo = loop.template_memo({"scores": [{"sport": "nba", "week": "2026-W44", "n": 50, "gap": -0.0123}]})
    for said in (text, worse, memo):
        assert not re.search(r"log loss:? ?[-−]", said), said
    assert "NBA, week 2026-W44, 50 games: brain vs market 0.0123 better on log loss." in memo


# --- 7 · the every-game lines -----------------------------------------------------------------------------------------

def _block(**over: Any) -> Dict[str, Any]:
    out = {"predicted": 4, "graded": 1, "pending": 3, "void": 0, "scored": 1, "hits": 1, "hit_rate": 1.0,
           "brain_brier": 0.2, "brain_log_loss": 0.5, "market_n": 1, "market_brier": 0.24, "market_log_loss": 0.6,
           "market_brain_brier": 0.2}
    out.update(over)
    return out


def test_every_game_lines_say_one_game_and_many_games() -> None:
    assert every_game.block_text("live", _block()).endswith("Brier 0.200 vs the line's 0.240 (1 game).")
    assert every_game.block_text("retro", _block(market_n=12)).endswith("(12 games).")


def test_brain_heading_and_note_agree_with_the_priced_picks_line(tmp_path) -> None:
    from tests.test_every_game import graded_store
    store = graded_store(tmp_path)
    lines = brain_lines(store)
    record = lines[-1]
    assert "Priced picks" in record and "not picks" in record and "priced picks are kept apart" in record
    note = progress(store)["every_game"]["note"]
    assert "learning records, not picks" in note and "Priced picks" in note and "scored apart" in note


def test_brains_headline_fits_when_only_the_every_game_record_has_games(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple.learn import progress as progress_module
    from tests.test_every_game import graded_store
    store = graded_store(tmp_path)
    monkeypatch.setattr(progress_module, "_headlines", lambda db, store: {})  # no sport graded against the market
    lines = brain_lines(store)
    assert lines[0] == ("No headline against the sharp market yet: no graded game has its price. The record below "
                        "is scored against each game's own line.")
    assert "vs the line's" in lines[-1]
    empty = brain_lines(Store(str(tmp_path / "empty.db")))
    assert empty[0] == "No graded games with a market price yet, so there is nothing to score."


def test_each_every_game_count_stays_whole_on_a_phone(tmp_path) -> None:
    from tests.test_every_game import graded_store
    payload = json.loads(json.dumps(progress(graded_store(tmp_path))))
    [result] = browser([{"kind": "brain", "width": 390, "payload": payload}])
    assert result["counts"], result
    assert all(c["nowrap"] and c["lines"] == 1 for c in result["counts"]), result["counts"]
    assert any(re.fullmatch(r"\d[\d,]* pending", c["text"]) for c in result["counts"]), result["counts"]


# --- 8 · a malformed g: button ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("data", ["g:basketball_nba", "g:", "g:basketball_nba:"])
def test_a_malformed_game_button_gets_the_expired_menu_answer(service, data) -> None:
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    application.bot_data["service"] = service
    answers: List[Any] = []

    class FakeQuery:
        def __init__(self) -> None:
            self.data = data

        async def answer(self, text=None, show_alert=False):
            answers.append((text, show_alert))

        async def edit_message_text(self, *_: Any, **__: Any) -> None:
            raise AssertionError("no screen for a stale button")

    update = SimpleNamespace(callback_query=FakeQuery(), message=None, effective_user=None)
    asyncio.run(handlers.game_selected(update, SimpleNamespace(application=application, user_data={})))
    assert answers == [("This menu has expired. Send /start to begin again.", True)]


# --- 9 · no API credits ---------------------------------------------------------------------------------------------

def test_no_api_credits_is_one_plain_sentence_without_the_credits_tail(service, tmp_path) -> None:
    from tests.test_assistant import NOW, FakeClaude, FakeLab, make
    from tests.test_small_fixes import _refusal
    store = Store(str(tmp_path / "betting.db"))
    store.get_or_create_user("tg:42", "Sam", now=NOW)
    store.confirm_age("tg:42", now=NOW)
    refusal = _refusal("Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.")
    claude = FakeClaude(router=[refusal], chat=[refusal])
    result = make(store, FakeLab(service, store), claude).reply("tg:42", message="Who wins tonight?")
    assert result["reply"]["text"] == f"{NO_CREDITS}."
    assert result["credits"]["daily_cents"]  # the allowance is still known, just not shown under this
    shown = bot_chat.reply_text(result)
    assert shown.count("The AI is paused") == 1 and "can't answer right now" not in shown
    assert "of today's AI credits left" not in shown and "¢" not in shown
    other = {**result, "reply": {**result["reply"], "text": "The Thunder are favourites tonight."}}
    assert "of today's AI credits left" in bot_chat.reply_text(other)  # every other answer keeps it


# --- 10 · the bot's AI bets -------------------------------------------------------------------------------------------

def test_ai_bets_open_on_a_league_with_a_model_and_can_switch_league(service) -> None:
    sports = [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
              {"key": "icehockey_nhl", "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False},
              {"key": MMA, "group": "Mixed Martial Arts", "title": "MMA", "active": True, "has_outrights": False},
              {"key": "golf_masters_winner", "group": "Golf", "title": "Masters", "active": True, "has_outrights": True}]
    service.odds.sports = lambda: sports
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    application.bot_data["service"] = service
    context = SimpleNamespace(application=application, user_data={"league": MMA})
    screens: List[Dict[str, Any]] = []

    class FakeQuery:
        def __init__(self, data: str) -> None:
            self.data = data

        async def answer(self, text=None, show_alert=False):
            pass

        async def edit_message_text(self, text, reply_markup=None, **kwargs):
            screens.append({"text": text, "buttons": [[(b.text, b.callback_data) for b in row]
                                                      for row in reply_markup.inline_keyboard]})

    user = SimpleNamespace(id=42, first_name="Sam", username=None)

    def tap(handler, data: str) -> Dict[str, Any]:
        asyncio.run(handler(SimpleNamespace(callback_query=FakeQuery(data), message=None, effective_user=user),
                            context))
        return screens[-1]

    first = tap(handlers.show_bets, "bets")
    assert "AI bets · NBA" in first["text"] and "on NBA moneylines" in first["text"]  # not MMA, the last browsed
    switch = [b for row in first["buttons"] for b in row if b[1].startswith("betsl:")]
    # The league last browsed first, then the popular ones in season; no futures, never the one shown
    assert switch == [("MMA", f"betsl:{MMA}"), ("NHL", "betsl:icehockey_nhl")]
    mma = tap(handlers.bets_league_selected, f"betsl:{MMA}")
    assert "AI bets · MMA" in mma["text"] and "No AI model for this league yet" in mma["text"]
    assert "NBA" not in mma["text"].split("\n", 1)[1]  # the note no longer talks about NBA moneylines
    assert ("NBA", f"betsl:{NBA}") in [b for row in mma["buttons"] for b in row]
    assert "AI bets · MMA" in tap(handlers.show_bets, "bets:safe")["text"]  # a level keeps the league picked
    assert "AI bets · NBA" in tap(handlers.show_bets, "bets")["text"]  # opening again starts on a model's league


# --- the screens in Chromium (4, 5, 7, 11, 12) ------------------------------------------------------------------------

BROWSER = r"""
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json" };
const STUB = `(() => {
  const STUB = {
    me: () => ({ ai_enabled: true, age_confirmed: true, name: "" }), priced: () => ({}), layout: () => "phone",
    onLayoutChange: () => () => {}, leagueTitle: (k) => k, sportEmoji: () => "", legKey: (r) => JSON.stringify(r ?? null),
    loadSports: async () => [], loadMe: async () => ({}), inTelegram: () => false, session: {}, favouriteSports: () => [],
    pinnedSports: () => [],
    slip: { has: () => false, legs: () => [], add() {}, remove() {}, toggle() {}, onChange: () => () => {} },
  };
  window.__appStub = { get: (name) => (name in STUB ? STUB[name] : () => undefined) };
})();`;
const page = (script) => `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/styles.css"></head>
<body><div id="desk" class="desk"><div class="shell"><main id="view" class="main"><div id="box"></div></main></div></div>
<div id="toasts" aria-live="polite"></div>
<script>${STUB}</script>
<script type="module">
try {
${script}
  window.__done = true;
} catch (error) { window.__failed = String(error && error.stack || error); }
</script></body></html>`;

// Each case: a page script, the /api/ answers, then what's measured
const wide = (sel) => `{
  const out = [];
  for (const el of document.querySelectorAll(${JSON.stringify(sel)})) {
    const over = el.scrollWidth > el.clientWidth + 1 && el.clientWidth && getComputedStyle(el).overflowX === "visible";
    if (over || el.getBoundingClientRect().right > innerWidth + 0.5) out.push(el.tagName + "." + el.className + ": " + (el.textContent || "").trim().slice(0, 40));
  }
  return out;
}`;

async function run(browser, spec) {
  const api = { calls: [], answers: spec.api || {} };
  const p = await browser.newPage({ viewport: { width: spec.width || 390, height: 800 }, deviceScaleFactor: 1 });
  const problems = [];
  p.on("pageerror", (error) => problems.push(String(error)));
  await p.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.hostname !== "test.local") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: page(SCRIPTS[spec.kind](spec)) });
    if (url.pathname === "/js/app.js") return route.fulfill({ contentType: "text/javascript", body: input.appStub });
    if (url.pathname.startsWith("/api/")) {
      api.calls.push({ method: request.method(), path: url.pathname, body: request.postData() });
      const answer = api.answers[`${request.method()} ${url.pathname}`] || { status: 200, body: {} };
      return route.fulfill({ status: answer.status, contentType: "application/json", body: JSON.stringify(answer.body) });
    }
    const file = path.join(input.static, decodeURIComponent(url.pathname));
    if (!file.startsWith(input.static) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ contentType: TYPES[path.extname(file)] || "application/octet-stream", body: fs.readFileSync(file) });
  });
  await p.goto("http://test.local/");
  await p.waitForFunction(() => window.__done === true || typeof window.__failed === "string", null, { timeout: 30000 });
  const failed = await p.evaluate(() => window.__failed || null);
  const out = failed ? { failed } : await MEASURE[spec.kind](p, spec, api);
  await p.close();
  return { kind: spec.kind, width: spec.width, problems, ...out };
}

const SCRIPTS = {
  help: (s) => `const { render } = await import("/js/views/help.js");
  document.getElementById("box").className = "view view-help";
  await render(document.getElementById("box"), { query: new URLSearchParams(${JSON.stringify(s.query)}) });`,
  finder: (s) => `const { render } = await import("/js/views/finder.js");
  document.getElementById("box").className = "view view-players";
  await render(document.getElementById("box"), { query: new URLSearchParams(${JSON.stringify(s.query)}), signal: new AbortController().signal });
  await new Promise((r) => setTimeout(r, 500));`,
  call: () => `const legacy = await import("/js/legacy.js");
  const { prefs } = await import("/js/api.js");
  window.__mods = { legacy, prefs };`,
  ctree: () => `const { checkTree, bindCheckTree } = await import("/js/ui.js");
  window.__mods = { checkTree, bindCheckTree };`,
  tree: () => `const { createTree } = await import("/js/views/today/tree.js");
  window.__mods = { createTree };`,
  ask: () => `const ask = await import("/js/views/ask.js");
  document.getElementById("box").className = "view view-ask";
  await ask.render(document.getElementById("box"), { query: new URLSearchParams(""), signal: new AbortController().signal });
  window.__mods = { ask };`,
  chart: () => `const { survivalChart, bindSurvivalChart } = await import("/js/views/build/chart.js");
  const { ensureStyles } = await import("/js/views/build/styles.js");
  ensureStyles();
  const box = document.getElementById("box");
  box.className = "view view-build";
  const series = [{ label: "Classic", curve: [0.8, 0.6, 0.4] }, { label: "Flat", curve: [0.9, 0.85, 0.8] }];
  box.innerHTML = survivalChart(series);
  bindSurvivalChart(box.querySelector(".bx-chart"), series);`,
  search: () => `const ui = await import("/js/ui.js");
  window.__mods = { ui };`,
  brain: () => `const { renderBrain } = await import("/js/views/brain/progress.js");
  await renderBrain(document.getElementById("box"));`,
};

const MEASURE = {
  help: async (p) => ({ scroll: await p.evaluate(() => document.documentElement.scrollWidth), wide: await p.evaluate(`(() => ${wide("#box *")})()`) }),
  finder: async (p) => ({ scroll: await p.evaluate(() => document.documentElement.scrollWidth), wide: await p.evaluate(`(() => ${wide("#box *")})()`),
    text: await p.evaluate(() => document.getElementById("box").textContent.replace(/\s+/g, " ")) }),
  call: async (p, spec, api) => {
    const steps = [];
    const accepted = await p.evaluate((names) => names.map((n) => window.__mods.legacy.callName(n)), spec.names);
    await p.evaluate((game) => { window.__mods.legacy.openCallSheet(game); }, spec.game);
    await p.waitForSelector("[data-call-save]");
    await p.waitForTimeout(500); // the first judge (250 ms after opening) has answered
    for (const step of spec.steps) {
      api.answers["POST /api/calls"] = step.answer;
      const before = api.calls.filter((c) => c.method === "POST" && c.path === "/api/calls").length;
      await p.evaluate((name) => { document.querySelector("[data-name]").value = name; }, step.name);
      await p.click("[data-call-save]");
      await p.waitForTimeout(300);
      const sent = api.calls.filter((c) => c.method === "POST" && c.path === "/api/calls").slice(before).map((c) => JSON.parse(c.body).name);
      steps.push({ name: step.name, sent, error: await p.evaluate(() => document.querySelector("[data-call-error]")?.textContent ?? null),
        kept: await p.evaluate(() => window.__mods.prefs.name()), open: await p.evaluate(() => Boolean(document.querySelector("[data-call-save]"))) });
    }
    return { accepted, steps, hint: await p.evaluate(() => window.__mods.legacy.CALL_NAME_HINT) };
  },
  ctree: async (p) => p.evaluate(() => {
    const box = document.getElementById("box");
    const groups = [{ key: "g", label: "G", items: [{ key: "a", label: "A" }, { key: "b", label: "B" }] }];
    box.innerHTML = window.__mods.checkTree({ name: "t", groups, checked: new Set(["a", "b"]), allLabel: "All sports" });
    let changes = 0;
    window.__mods.bindCheckTree(box, { checked: new Set(["a", "b"]), onChange: () => { changes += 1; } });
    const button = (act) => box.querySelector(`[data-ctree-act="${act}"]`);
    const state = (did) => ({ did, all: button("all").getAttribute("aria-disabled"), none: button("none").getAttribute("aria-disabled"), changes,
      cursor: getComputedStyle(button("all")).cursor });
    const out = [state("start, both checked")];
    button("all").click(); out.push(state("Select all again"));
    button("none").click(); out.push(state("Clear"));
    button("none").click(); out.push(state("Clear again"));
    box.querySelector('[data-ctree-box="item"][value="a"]').click(); out.push(state("one box"));
    button("all").click(); out.push(state("Select all"));
    return { states: out };
  }),
  tree: async (p) => p.evaluate(() => {
    const tree = window.__mods.createTree({ onSelect: () => {} });
    document.getElementById("box").appendChild(tree.el);
    const prepared = { groups: [{ key: "Basketball", label: "Basketball", leagues: ["basketball_nba", "basketball_wnba"] }],
      leagues: new Map([["basketball_nba", { key: "basketball_nba", title: "NBA" }], ["basketball_wnba", { key: "basketball_wnba", title: "WNBA" }]]),
      order: ["basketball_nba", "basketball_wnba"], week: new Map() };
    const counts = new Map([["basketball_nba", { count: 3, live: 0 }], ["basketball_wnba", { count: 2, live: 0 }]]);
    const draw = (selection) => {
      tree.update({ prepared, counts, selection, pinned: [], presets: [], activePreset: null, liveCount: 0 });
      return tree.el.querySelector("[data-only-all]").getAttribute("aria-pressed");
    };
    return { all: draw(null), some: draw(new Set(["basketball_nba"])), every: draw(new Set(["basketball_nba", "basketball_wnba"])) };
  }),
  ask: async (p) => {
    await p.click("[data-new-chat]");
    await p.waitForTimeout(200);
    const toasts = await p.evaluate(() => [...document.querySelectorAll("#toasts .toast")].map((t) => t.textContent.trim()));
    const states = await p.evaluate(() => {
      const { alreadyNew } = window.__mods.ask;
      const empty = { busy: false, loading: false, messages: [], threadId: null, attach: null, requestOpen: null };
      return [alreadyNew(empty), alreadyNew({ ...empty, messages: [{ role: "user", text: "hi" }] }), alreadyNew({ ...empty, threadId: 4 }),
        alreadyNew({ ...empty, attach: { name: "slip.png" } }), alreadyNew({ ...empty, requestOpen: false }), alreadyNew({ ...empty, loading: true })];
    });
    return { toasts, states };
  },
  chart: async (p) => p.evaluate(() => Object.fromEntries(["figure", "svg.g-plot", "rect.g-hit", "path", "text"].map((sel) => {
    const el = document.querySelector(sel);
    return [sel, el ? getComputedStyle(el).cursor : null];
  }))),
  search: async (p) => p.evaluate(() => {
    const { searchFailed, SEARCH_MAX } = window.__mods.ui;
    return { max: SEARCH_MAX, refused: searchFailed({ status: 422, message: "String should have at most 60 characters" }),
      other: searchFailed({ status: 502, message: "ESPN didn't answer." }) };
  }),
  brain: async (p) => {
    await p.waitForSelector("#bp-every-game .bp-nw", { timeout: 15000 });
    return { counts: await p.evaluate(() => [...document.querySelectorAll("#bp-every-game .bp-nw")].map((el) => ({
      text: el.textContent, nowrap: getComputedStyle(el).whiteSpace === "nowrap",
      lines: new Set([...el.getClientRects()].map((r) => Math.round(r.top))).size }))) };
  },
};

(async () => {
  let browser;
  try {
    browser = await chromium.launch();
  } catch (error) {
    finish({ skip: `Chromium can't start: ${String(error).split("\n")[0]}` });
    return;
  }
  const results = [];
  for (const spec of input.cases) {
    try {
      results.push(await run(browser, spec));
    } catch (error) {
      results.push({ kind: spec.kind, error: String(error && error.stack || error) });
    }
  }
  await browser.close();
  finish({ results });
})().catch((error) => finish({ error: String(error && error.stack || error) }));
"""


def app_stub() -> str:
    """js/app.js as the screens import it: every export, each from window.__appStub (set in the page)."""
    return test_capture_health.app_stub()


def browser(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run each case's page in Chromium; skips when Node, Playwright or Chromium is missing."""
    joined = test_capture_health.node_path()
    if not joined:
        pytest.skip("Playwright for Node isn't installed: these checks need Chromium")
    for case in cases:  # the brain case reads its payload from /api/brain/progress
        if case["kind"] == "brain":
            case["api"] = {"GET /api/brain/progress": {"status": 200, "body": case.pop("payload")}}
    data = json.dumps({"static": str(STATIC), "appStub": app_stub(), "cases": cases})
    done = subprocess.run([test_capture_health.NODE, "-e", BROWSER], input=data, capture_output=True, text=True,
                          encoding="utf-8", timeout=300, env={**os.environ, "NODE_PATH": joined, "NODE_OPTIONS": ""})
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout)
    if out.get("skip"):
        pytest.skip(out["skip"])
    assert "error" not in out, out.get("error")
    for result in out["results"]:
        assert not result.get("error") and not result.get("failed"), result
        assert result["problems"] == [], result
    return out["results"]


# --- 4 · long text with no spaces -------------------------------------------------------------------------------------

LONG = "W" * 60


def test_long_text_never_pushes_help_or_the_finder_sideways_on_a_phone() -> None:
    scan = {"players": [], "close": [], "names": [LONG], "missing_names": [LONG], "ambiguous_names": [],
            "started_names": [], "games": 5, "games_started": 0, "leagues": [NBA], "loading": None}
    cases = []
    for width in (360, 390):
        cases.append({"kind": "help", "width": width, "query": f"q={LONG}"})
        cases.append({"kind": "finder", "width": width, "api": {"GET /api/players/scan": {"status": 200, "body": scan}},
                      "query": f"league={NBA}&stat=pts&at=20&day=today&player={LONG}"})
    for result in browser(cases):
        assert result["scroll"] <= result["width"] and result["wide"] == [], result
        if result["kind"] == "finder":  # the long name is on screen: its note and the empty state say it
            assert f"I couldn't find {LONG}" in result["text"] and f"Nothing to show for {LONG}" in result["text"]


# --- 5 · Make your call's name ----------------------------------------------------------------------------------------

NAMES = ["Sam", "Zoë", "José María", "Саша", "O'Neil-Smith Jr.", "user_7", "A" * 24, "Sam’s", "Sam 🏀", "Sam & Co",
         "A" * 25, "x<y", "", "   "]
GAME = {"league": NBA, "id": "g1", "home": "Oklahoma City Thunder", "away": "Los Angeles Clippers",
        "prediction": {"home_win": 0.7, "sources": {}}, "players": {"home": [], "away": []}}
REFUSED = {"status": 422, "body": {"detail": [{"type": "string_pattern_mismatch", "loc": ["body", "name"],
                                               "msg": "String should match pattern '^[\\w .'-]{1,24}$'"}]}}


@pytest.fixture(scope="module")
def call_sheet() -> Dict[str, Any]:
    steps = [{"name": "Sam 🏀", "answer": {"status": 200, "body": {"ok": True}}},
             {"name": "Sam & Co", "answer": {"status": 200, "body": {"ok": True}}},
             {"name": "A" * 25, "answer": {"status": 200, "body": {"ok": True}}},
             {"name": "Sam’s", "answer": REFUSED},
             {"name": "Sam’s", "answer": {"status": 200, "body": {"ok": True}}}]
    [result] = browser([{"kind": "call", "names": NAMES, "game": GAME, "steps": steps,
                         "api": {"GET /api/calls": {"status": 200, "body": []},
                                 "POST /api/judge": {"status": 200, "body": {"verdict": "", "props": []}}}}])
    return result


def test_the_call_name_is_checked_by_the_servers_own_rule(call_sheet, client) -> None:
    accepted = dict(zip(NAMES, call_sheet["accepted"]))
    assert accepted["Sam’s"] == "Sam's" and accepted["Sam 🏀"] is None and accepted["A" * 25] is None
    for name, sent in accepted.items():  # what the screen sends, the server takes; what it holds back, it refuses
        if not name.strip():
            continue  # a blank name is asked for ("Add your name for the leaderboard.") before any rule
        body = {"league": NBA, "id": "nope", "player": "browser-1234", "name": sent if sent is not None else name,
                "call": {"winner": "home", "confidence": 0.6}}
        response = client.post("/api/calls", json=body)
        refused_name = response.status_code == 422 and any(e["loc"] == ["body", "name"]
                                                           for e in response.json()["detail"])
        assert refused_name == (sent is None), (name, sent, response.text)


def test_a_bad_call_name_is_said_plainly_and_kept_only_once_the_server_took_it(call_sheet) -> None:
    hint = call_sheet["hint"]
    assert hint == "Use letters, numbers, spaces and . ' - (up to 24)"
    emoji, ampersand, long, refused, saved = call_sheet["steps"]
    for step in (emoji, ampersand, long):
        assert step["sent"] == [] and step["error"] == hint and step["kept"] == "" and step["open"], step
    assert refused["sent"] == ["Sam's"] and refused["error"] == hint and refused["kept"] == "", refused
    assert saved["sent"] == ["Sam's"] and saved["kept"] == "Sam's" and not saved["open"], saved


# --- 11 · the player-name searches ------------------------------------------------------------------------------------

def test_player_name_searches_take_60_characters_and_say_why_a_longer_one_fails() -> None:
    [result] = browser([{"kind": "search"}])
    assert result["max"] == 60
    assert result["refused"] == "Player names are searched by up to 60 characters: try a shorter one."
    assert result["other"] == "ESPN didn't answer."
    for view in ("today.js", "game.js"):
        source = (STATIC / "js" / "views" / view).read_text(encoding="utf-8")
        field = re.search(r'<input id="(?:game-)?player-q"[^>]*>', source)
        assert field and 'maxlength="${SEARCH_MAX}"' in field.group(0), view
        assert "notice(searchFailed(error)" in source and "notice(error.message" not in source, view


# --- 12 · controls that did nothing -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def controls() -> Dict[str, Dict[str, Any]]:
    results = browser([{"kind": "ctree"}, {"kind": "tree"}, {"kind": "chart"},
                       {"kind": "ask", "api": {"GET /api/chat/threads": {"status": 200, "body": []},
                                               "GET /api/me": {"status": 200, "body": {}}}}])
    return {result["kind"]: result for result in results}


def test_new_chat_on_an_empty_chat_says_it_already_is_one(controls) -> None:
    ask = controls["ask"]
    assert ask["toasts"] == ["This is already a new chat: ask away"]
    assert ask["states"] == [True, False, False, False, False, False]


def test_select_all_and_clear_say_when_they_would_change_nothing(controls) -> None:
    states = {s["did"]: (s["all"], s["none"], s["changes"]) for s in controls["ctree"]["states"]}
    assert states == {"start, both checked": ("true", "false", 0), "Select all again": ("true", "false", 0),
                      "Clear": ("false", "true", 1), "Clear again": ("false", "true", 1),
                      "one box": ("false", "false", 2), "Select all": ("true", "false", 3)}
    assert controls["ctree"]["states"][0]["cursor"] != "pointer"


def test_all_sports_is_pressed_while_every_sport_is_shown(controls) -> None:
    tree = controls["tree"]
    assert (tree["all"], tree["some"], tree["every"]) == ("true", "false", "false")


def test_the_rollover_chart_has_no_pointer_cursor(controls) -> None:
    assert controls["chart"]["svg.g-plot"] and "pointer" not in controls["chart"].values(), controls["chart"]
