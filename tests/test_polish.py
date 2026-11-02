"""The QA re-crawl's polish round (scratchpad brief FIX_POLISH.md): one test per item, each failing without its fix.

  1. tap targets of 44 x 44 px: My bets' "All" period, Help's info button in a long heading, the header's screen name
     on a short screen ("FAQ"), Settings' "How the app works, in 60 seconds" row
  2. the Ask game screen: Tab reaches every control without falling to <body>, and a box drawn again while the
     focus is in it (the request bar when the sports list arrives) keeps the focus on the same control
  3. controls that did nothing: the league tree's name buttons are pressed while their leagues are the whole
     selection and a tap then says so, and in the Builder a league or sport with no games is off and says "No
     games"; Today's "Save" of a view is off until there's a name, and saving hands the focus back to "Save view"

Each runs at 360 px (a touch phone) and 1280 px in Chromium (Playwright for Node, as tests/test_qa_fixes.py does): the
app's own modules and styles from this tree's static folder, js/app.js stubbed, every /api/ request answered here. No
server, no network. Skipped when Node or Playwright isn't installed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List

import pytest

from tests import test_capture_health

STATIC = test_capture_health.STATIC
WIDTHS = (360, 1280)

BROWSER = r"""
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json" };
const stub = (extra) => `(() => {
  const STUB = {
    me: () => ({ ai_enabled: true, age_confirmed: true, name: "" }), priced: () => ({}), layout: { mode: "phone" },
    onLayoutChange: () => () => {}, leagueTitle: (k) => k, sportEmoji: () => "", legKey: (r) => JSON.stringify(r ?? null),
    loadSports: async () => [], loadMe: async () => ({}), inTelegram: false, session: {}, favouriteSports: () => [],
    pinnedSports: () => [],
    slip: { count: 0, has: () => false, legs: () => [], add() {}, remove() {}, toggle() {}, onChange: () => () => {} },
  };
  ${extra || ""}
  window.__appStub = { get: (name) => (name in STUB ? STUB[name] : () => undefined) };
})();`;
const page = (spec, script) => `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/styles.css"></head>
<body>${spec.kind === "ask" ? '<a id="start" href="#start">Start</a>' : ""}${spec.kind === "title" ? input.header : ""}
<div id="desk" class="desk"><div class="shell"><main id="view" class="main"><div id="box"></div></main></div></div>
<div id="toasts" aria-live="polite"></div>
<script>${stub(STUBS[spec.kind] || "")}</script>
<script type="module">
window.tap = (el) => {  // the area a finger hits: the box and an absolutely placed ::before / ::after over it
  const r = el.getBoundingClientRect();
  let w = r.width, h = r.height;
  for (const p of ["::before", "::after"]) {
    const cs = getComputedStyle(el, p);
    if (cs.content === "none" || cs.content === "normal" || cs.position !== "absolute" || getComputedStyle(el).position === "static") continue;
    const n = (k) => (cs[k] === "auto" ? null : parseFloat(cs[k]));
    const [t, b, l, rt] = [n("top"), n("bottom"), n("left"), n("right")];
    if (t !== null && b !== null) h = Math.max(h, r.height - t - b);
    if (l !== null && rt !== null) w = Math.max(w, r.width - l - rt);
  }
  return { w: Math.round(w * 10) / 10, h: Math.round(h * 10) / 10, text: (el.textContent || "").trim().slice(0, 40), label: el.getAttribute("aria-label") || "" };
};
window.toasts = () => [...document.querySelectorAll("#toasts .toast")].map((t) => t.textContent.trim());
try {
${script}
  window.__done = true;
} catch (error) { window.__failed = String(error && error.stack || error); }
</script></body></html>`;

const STUBS = {
  // The sports list arrives when the test says so: the request bar is drawn again then
  ask: `STUB.loadSports = () => new Promise((resolve) => { window.__sports = () => resolve([]); });`,
};

const TREE = `const { createTree } = await import("/js/views/today/tree.js");
  const { ensureStyles } = await import("/js/views/today/styles.js");
  ensureStyles();
  const league = (key, title) => [key, { key, title, odds: "fresh" }];
  window.__prepared = {
    groups: [{ key: "Ice Hockey", label: "Ice Hockey", leagues: ["icehockey_nhl"] },
      { key: "Soccer", label: "Soccer", leagues: ["soccer_epl", "soccer_efl_champ"] },
      { key: "Mixed Martial Arts", label: "MMA", leagues: ["mma_mixed_martial_arts"] }],
    leagues: new Map([league("icehockey_nhl", "NHL"), league("soccer_epl", "EPL"), league("soccer_efl_champ", "Championship"), league("mma_mixed_martial_arts", "MMA")]),
    order: ["icehockey_nhl", "soccer_epl", "soccer_efl_champ", "mma_mixed_martial_arts"], week: new Map() };
  window.__counts = new Map([["icehockey_nhl", { count: 5, live: 0 }], ["soccer_epl", { count: 1, live: 0 }]]);
  window.__mods = { createTree };`;

const SCRIPTS = {
  bets: () => `const { resultsHtml } = await import("/js/views/bets/slips.js");
  const { ensureStyles } = await import("/js/views/bets/styles.js");
  ensureStyles();
  const box = document.getElementById("box");
  box.className = "view view-bets";
  box.innerHTML = resultsHtml([], "30d");`,
  help: () => `const { render } = await import("/js/views/help.js");
  const box = document.getElementById("box");
  box.className = "view view-help";
  await render(box, { topic: "ai", query: new URLSearchParams("") });`,
  // A screen's name in the header, as app.js drawTitle writes it
  title: () => `document.getElementById("title").innerHTML = '<span class="page-name">FAQ</span>';`,
  settings: () => `const { render } = await import("/js/views/settings.js");
  const box = document.getElementById("box");
  box.className = "view view-settings";
  await render(box, { query: new URLSearchParams(""), signal: new AbortController().signal });`,
  tree: () => TREE,
  save: () => TREE,
  ask: () => `const ask = await import("/js/views/ask.js");
  const box = document.getElementById("box");
  box.className = "view view-ask";
  await ask.render(box, { query: new URLSearchParams("game=basketball_nba:g1&name=Knicks%20at%20Celtics"), signal: new AbortController().signal });
  await new Promise((r) => setTimeout(r, 300));`,
};

// The controls Tab stops at, in the page's order (no positive tabindex in the app)
const TABBABLES = `[...document.querySelectorAll("a[href], button, input, select, textarea, summary, [tabindex]")]
  .filter((e) => e.tabIndex >= 0 && !e.matches(":disabled") && e.getClientRects().length && !e.closest("[hidden], [inert]"))`;
const WHO = `(() => { const e = document.activeElement; if (!e || e === document.body) return "BODY";
  return e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + [...e.attributes].filter((a) => a.name.startsWith("data-")).map((a) => "[" + a.name + "=" + a.value + "]").join("")
    + " " + (e.getAttribute("aria-label") || e.textContent || "").trim().replace(/\s+/g, " ").slice(0, 30); })()`;

const MEASURE = {
  bets: async (p) => ({ targets: await p.evaluate(() => [...document.querySelectorAll('[data-seg="mb-period"]')].map(window.tap)) }),
  help: async (p) => ({ targets: await p.evaluate(() => [...document.querySelectorAll("#box .info-btn")].map(window.tap)) }),
  title: async (p) => ({ targets: await p.evaluate(() => [window.tap(document.getElementById("title"))]) }),
  settings: async (p) => ({ targets: await p.evaluate(() => [...document.querySelectorAll('#box a[href="#/help/start"]')].map(window.tap)) }),
  tree: async (p, spec) => p.evaluate((pickEmpty) => {
    const picked = [];
    const tree = window.__mods.createTree({ views: false, pins: false, countWords: "in your time window", only: "Only", pickEmpty,
      onSelect: (selection) => picked.push(selection ? [...selection] : null) });
    document.getElementById("box").appendChild(tree.el);
    // The sport's own button was tapped: its one league is the whole selection
    tree.update({ prepared: window.__prepared, counts: window.__counts, selection: new Set(["icehockey_nhl"]), pinned: [], presets: [], activePreset: null, liveCount: 0 });
    const read = (el) => ({ pressed: el.getAttribute("aria-pressed"), off: el.getAttribute("aria-disabled"), label: el.getAttribute("aria-label"),
      title: el.getAttribute("title"), cursor: getComputedStyle(el).cursor, color: getComputedStyle(el).color });
    const names = {};
    for (const el of tree.el.querySelectorAll("[data-only]")) names[el.dataset.only] = read(el);
    for (const el of tree.el.querySelectorAll("[data-only-group]")) names[`group:${el.dataset.onlyGroup}`] = read(el);
    const taps = {};
    for (const key of ["icehockey_nhl", "group:Ice Hockey", "mma_mixed_martial_arts", "soccer_efl_champ", "soccer_epl", "group:Mixed Martial Arts"]) {
      const before = { count: picked.length, toasts: window.toasts().length };
      tree.el.querySelector(key.startsWith("group:") ? `[data-only-group="${key.slice(6)}"]` : `[data-only="${key}"]`).click();
      taps[key] = { selected: picked.length > before.count ? picked[picked.length - 1] : "nothing", toasts: window.toasts().slice(before.toasts) };
    }
    const enabled = getComputedStyle(tree.el.querySelector('[data-only="soccer_epl"]')).color;
    return { names, taps, enabled };
  }, spec.pickEmpty),
  save: async (p) => {
    await p.evaluate(() => {
      window.__saved = [];
      const presets = [];
      const draw = () => tree.update({ prepared: window.__prepared, counts: window.__counts, selection: null, pinned: [], presets, activePreset: "all", liveCount: 0 });
      const tree = window.__mods.createTree({ onSelect() {}, onSavePreset(name) { window.__saved.push(name); presets.push({ name, leagues: null, status: "all" }); draw(); } });
      document.getElementById("box").appendChild(tree.el);
      draw();
    });
    const state = (did) => p.evaluate(({ did, who }) => {
      const form = document.querySelector("[data-preset-form]");
      const save = form.querySelector('button[type="submit"]');
      return { did, open: !form.hidden, off: save.getAttribute("aria-disabled"), noValidate: form.noValidate, saved: [...window.__saved],
        focus: eval(who), expanded: document.querySelector("[data-preset-save]").getAttribute("aria-expanded"), toasts: window.toasts() };
    }, { did, who: WHO });
    const out = [];
    await p.click("[data-preset-save]");
    out.push(await state("opened"));
    // A tap on Save while it's off (force: Playwright itself waits for an aria-disabled button to come on)
    await p.click('[data-preset-form] button[type="submit"]', { force: true });
    out.push(await state("Save with no name"));
    await p.fill("[data-preset-form] input", "   ");
    await p.click('[data-preset-form] button[type="submit"]', { force: true });
    out.push(await state("Save with spaces"));
    await p.fill("[data-preset-form] input", "Evening");
    out.push(await state("named"));
    await p.click('[data-preset-form] button[type="submit"]');
    out.push(await state("saved by a tap"));
    await p.click("[data-preset-save]");
    await p.keyboard.type("Late");
    await p.keyboard.press("Enter");
    out.push(await state("saved with Enter"));
    return { states: out };
  },
  ask: async (p) => {
    const fine = await p.evaluate(() => matchMedia("(pointer: fine)").matches);
    const landed = await p.evaluate(WHO); // where the game's link put the focus
    const all = await p.evaluate(`${TABBABLES}.length`);
    const afterComposer = await p.evaluate(`(() => { const list = ${TABBABLES}; const i = list.findIndex((e) => e.id === "ask-text"); return i < 0 ? -1 : list.length - 1 - i; })()`);
    await p.focus("#start");
    const seq = [];
    let redrawn = null;
    for (let i = 0; i < all - 1; i++) {
      await p.keyboard.press("Tab");
      const now = await p.evaluate(WHO);
      seq.push(now);
      if (!redrawn && await p.evaluate(() => Boolean(document.activeElement.closest("[data-request]")))) {
        // The sports list arrives while the focus is in the request bar: the bar is drawn again
        const before = await p.evaluateHandle(() => document.activeElement);
        await p.evaluate(() => window.__sports());
        await p.waitForTimeout(100);
        redrawn = { focus: now, after: await p.evaluate(WHO), replaced: await p.evaluate((el) => !el.isConnected, before) };
        await before.dispose();
      }
    }
    // From the composer (where the link leaves the focus on a keyboard) Tab goes through every control after it
    let fromComposer = [];
    if (afterComposer > 0) {
      await p.focus("#ask-text");
      for (let i = 0; i < afterComposer; i++) {
        await p.keyboard.press("Tab");
        fromComposer.push(await p.evaluate(WHO));
      }
    }
    return { fine, landed, all, seq, redrawn, afterComposer, fromComposer };
  },
};

async function run(browser, spec) {
  const api = { calls: [], answers: spec.api || {} };
  const phone = spec.width < 600;
  const context = await browser.newContext({ viewport: { width: spec.width, height: 800 }, deviceScaleFactor: 1, isMobile: phone, hasTouch: phone });
  const p = await context.newPage();
  const problems = [];
  p.on("pageerror", (error) => problems.push(String(error)));
  await p.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.hostname !== "test.local") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: page(spec, SCRIPTS[spec.kind](spec)) });
    if (url.pathname === "/js/app.js") return route.fulfill({ contentType: "text/javascript", body: input.appStub });
    if (url.pathname.startsWith("/api/")) {
      api.calls.push({ method: request.method(), path: url.pathname });
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
  await context.close();
  return { kind: spec.kind, width: spec.width, pickEmpty: spec.pickEmpty, problems, ...out };
}

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
      results.push({ kind: spec.kind, width: spec.width, error: String(error && error.stack || error) });
    }
  }
  await browser.close();
  finish({ results });
})().catch((error) => finish({ error: String(error && error.stack || error) }));
"""


def header() -> str:
    """The app's header, as index.html has it (the title link's text is the screen's name)."""
    source = (STATIC / "index.html").read_text(encoding="utf-8")
    found = re.search(r'<header class="app-header">.*?</header>', source, re.S)
    assert found, "index.html has no app header"
    return found.group(0)


def browser(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run each case's page in Chromium; skips when Node, Playwright or Chromium is missing."""
    joined = test_capture_health.node_path()
    if not joined:
        pytest.skip("Playwright for Node isn't installed: these checks need Chromium")
    data = json.dumps({"static": str(STATIC), "appStub": test_capture_health.app_stub(), "header": header(), "cases": cases})
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


@pytest.fixture(scope="module")
def results() -> Dict[Any, Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for width in WIDTHS:
        cases += [{"kind": kind, "width": width} for kind in ("bets", "help", "title", "settings", "save")]
        cases += [{"kind": "tree", "width": width, "pickEmpty": False}, {"kind": "tree", "width": width, "pickEmpty": True}]
        cases.append({"kind": "ask", "width": width, "api": {"GET /api/chat/threads": {"status": 200, "body": []},
                                                              "GET /api/me": {"status": 200, "body": {}}}})
    out = browser(cases)
    return {(r["kind"], r["width"], r.get("pickEmpty")): r for r in out}


# --- 1 · tap targets ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("kind, what", [("bets", 'My bets\' periods ("All")'), ("help", "Help's info buttons (AI analysts)"),
                                        ("title", 'the header\'s screen name ("FAQ")'),
                                        ("settings", 'Settings\' "How the app works, in 60 seconds"')])
def test_each_tap_target_is_at_least_44_px_square(results, kind, what, width) -> None:
    targets = results[(kind, width, None)]["targets"]
    assert targets, f"{what}: nothing measured at {width} px"
    small = [t for t in targets if t["w"] < 44 or t["h"] < 44]
    assert small == [], f"{what} at {width} px: {small}"


def test_the_ai_analysts_info_button_is_measured_in_its_long_heading(results) -> None:
    for width in WIDTHS:  # the button the re-crawl found squeezed to 26 px by its heading
        labels = [t["label"] for t in results[("help", width, None)]["targets"]]
        assert "What is AI analysts?" in labels, labels


# --- 2 · the Ask game screen keeps the keyboard focus -------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_tab_through_the_ask_game_screen_never_lands_on_body(results, width) -> None:
    ask = results[("ask", width, None)]
    assert ask["all"] > 20, ask
    # Every control after the start link, each once, never <body>: the request bar drawn again on the way included
    assert "BODY" not in ask["seq"], ask["seq"]
    assert len(set(ask["seq"])) == ask["all"] - 1, ask["seq"]
    redrawn = ask["redrawn"]
    assert redrawn and redrawn["replaced"], ask  # the control did go: the bar's HTML was written again
    assert redrawn["after"] == redrawn["focus"], redrawn  # the same control has the focus again


def test_the_game_link_leaves_the_focus_in_the_composer_on_a_keyboard_only(results) -> None:
    desk, phone = results[("ask", 1280, None)], results[("ask", 360, None)]
    assert desk["fine"] and desk["landed"].startswith("textarea#ask-text"), desk["landed"]
    assert not phone["fine"] and phone["landed"] == "BODY", phone["landed"]  # no keyboard raised on a phone
    # From there Tab reaches every control after the composer before the page's end (QA's <body> after 6)
    assert desk["afterComposer"] >= 1 and "BODY" not in desk["fromComposer"], desk["fromComposer"]


# --- 3 · controls that did nothing --------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_a_name_button_is_pressed_while_its_leagues_are_the_whole_selection_and_a_tap_says_so(results, width) -> None:
    for pick_empty in (False, True):  # the Builder and Today
        names = results[("tree", width, pick_empty)]["names"]
        assert names["icehockey_nhl"]["pressed"] == "true" and names["group:Ice Hockey"]["pressed"] == "true", names
        assert names["icehockey_nhl"]["cursor"] == "default", names["icehockey_nhl"]
        assert names["soccer_epl"]["pressed"] == "false" and names["group:Soccer"]["pressed"] == "false", names
        # A tap on one says why nothing changes (Today's onSelect still runs: it closes the drawer on a phone)
        taps = results[("tree", width, pick_empty)]["taps"]
        assert taps["icehockey_nhl"] == {"selected": ["icehockey_nhl"], "toasts": ["NHL is already the only league picked"]}, taps
        assert taps["group:Ice Hockey"] == {"selected": ["icehockey_nhl"], "toasts": ["Ice Hockey is already the only sport picked"]}, taps


@pytest.mark.parametrize("width", WIDTHS)
def test_in_the_builder_a_league_or_sport_with_no_games_is_off_and_says_so(results, width) -> None:
    tree = results[("tree", width, False)]
    names, taps = tree["names"], tree["taps"]
    for key, title in (("mma_mixed_martial_arts", "MMA"), ("soccer_efl_champ", "Championship"), ("group:Mixed Martial Arts", "MMA")):
        name = names[key]
        assert name["off"] == "true" and name["cursor"] == "default" and name["color"] != tree["enabled"], (key, name)
        assert name["label"].startswith(f"Only {title}: No games in your time window"), name["label"]
        assert name["title"].startswith("No games in your time window"), name["title"]
    for key in ("icehockey_nhl", "soccer_epl", "group:Soccer", "group:Ice Hockey"):
        assert names[key]["off"] is None, (key, names[key])
    assert taps["mma_mixed_martial_arts"] == {"selected": "nothing", "toasts": ["No MMA games in your time window"]}, taps
    assert taps["soccer_efl_champ"] == {"selected": "nothing", "toasts": ["No Championship games in your time window"]}, taps
    assert taps["group:Mixed Martial Arts"] == {"selected": "nothing", "toasts": ["No MMA games in your time window"]}, taps
    assert taps["soccer_epl"] == {"selected": ["soccer_epl"], "toasts": []}, taps


@pytest.mark.parametrize("width", WIDTHS)
def test_on_today_a_league_with_no_games_still_opens(results, width) -> None:
    tree = results[("tree", width, True)]
    assert all(n["off"] is None for n in tree["names"].values()), tree["names"]
    assert tree["taps"]["soccer_efl_champ"] == {"selected": ["soccer_efl_champ"], "toasts": []}, tree["taps"]
    assert tree["taps"]["mma_mixed_martial_arts"] == {"selected": ["mma_mixed_martial_arts"], "toasts": []}, tree["taps"]


@pytest.mark.parametrize("width", WIDTHS)
def test_save_view_is_off_until_there_is_a_name_and_saving_keeps_the_focus(results, width) -> None:
    states = {s["did"]: s for s in results[("save", width, None)]["states"]}
    opened = states["opened"]
    assert opened["open"] and opened["off"] == "true" and opened["noValidate"], opened
    assert opened["focus"].startswith("input"), opened  # "Save view" puts the focus in the name box
    for did in ("Save with no name", "Save with spaces"):  # nothing to save: no silent browser bubble, still off
        s = states[did]
        assert s["open"] and s["off"] == "true" and s["saved"] == [] and s["focus"].startswith("input"), s
    assert states["named"]["off"] == "false", states["named"]
    for did, saved in (("saved by a tap", ["Evening"]), ("saved with Enter", ["Evening", "Late"])):
        s = states[did]
        assert s["saved"] == saved and not s["open"], s
        assert s["focus"].startswith("button[data-preset-save") and s["expanded"] == "false", s  # never <body>
