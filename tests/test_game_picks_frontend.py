"""Game picks, the screens: the "Suggested picks" section on the game page (views/game/picks.js, drawn by
views/game.js right after the action bar), by the app's own ES modules.

Contract B of scratchpad/build/SPEC_GAME_PICKS.md:
  - The section loads from GET /api/game/picks?league=&id= once the page has drawn: the page never waits for it, and a
    skeleton holds its place until the answer arrives. It sits right after the action bar.
  - Each game pick: its market title and label; "Brain 64% · Market 58%" side by side (the brain's alone without a
    market chance); a neutral "Brain higher than market" chip when ``brain_higher`` (never "edge" or "value" on a game
    pick); the best price and its book; a "Why?" toggle over its ``why`` lines; and "Add to slip", through the store's
    slip.add with the leg the Builder's and Ask AI's cards add, showing the added state. The slip prices soccer's
    goals-model markets at fair odds, so such a pick shows its price with the button off and a line saying why.
  - The player props: "Likeliest" and "Best edges", the P0 scan rows (views/players/scan.js rowItemHtml: a posted line
    shows its edge), and "See all in the Player finder" (Best edges (all stats) for the league and the game's day:
    props.day, else the game's start on ESPN's day, US Eastern).
  - The states: paused (the pause note with a link to Settings, no picks: no request when /api/me already says so; a
    400 is shown in the server's own words), unpriced (``unpriced.code`` "on_way": "Prices for this game are on their
    way…" with a spinner, the route asked again by the section itself at 15 s and 45 s (fake timers here), then the
    server's reason; any other code, such as the month's odds allowance at its reserve or today's used up, the server's
    own words at once, never "check back"; checked against the real Lab._game_unpriced too), the game started (no request when the page shows it live), nothing
    at all, player props still loading (top-level ``loading`` or ``props.loading``), and always the line "No pick is
    certain. Only you add bets to your slip."
  - Every string is escaped.

How (the node harness pattern of tests/test_p0_fix_frontend.py): Node imports the modules from this tree's
dashboard/static with a small stand-in for the browser (document, localStorage, location), js/app.js replaced by a
stub exporting the same names (a loader hook) whose store names are the REAL js/store.js (so "Add to slip" goes through
the real slip.add and lands in the real slip), and fetch stubbed: GET /api/game/picks, /api/game and /api/me answered
with the case's JSON, every other path a 404. No browser, no server, no network. The HTML comes back here and is read
with html.parser: what a screen reader hears (aria-hidden parts left out) and what's on screen (sr-only parts left out).
Skipped when Node (with module.register: 18.19+ or 20.6+) is missing.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import pytest

from telegram_betting_bot_simple.dashboard.lab import PRICE_REASONS, Lab

STATIC = Path(__file__).resolve().parents[1] / "telegram_betting_bot_simple" / "dashboard" / "static"
NODE = shutil.which("node")
NBA, EPL, TENNIS = "basketball_nba", "soccer_epl", "tennis_atp_paris"
GAME_ID = "3f2a9c41b7e04d55a1c8e0f6d2b49a77"
START = "2030-01-01T00:30:00Z"  # far ahead: never under way
SECTION = "section#suggested-picks"
DISCLOSURE = "No pick is certain. Only you add bets to your slip."
ON_THEIR_WAY = "Prices for this game are on their way…"
UNPRICED_PLAIN = "Prices for this game aren't in yet."
STARTED = "This game has started: picks are for games still to come."
EMPTY = "No picks for this game yet."
LOADING = "Player props are still loading."
NO_SLIP = "The slip takes moneylines, spreads and totals."
SETTINGS = "#/settings?focus=limits"

# --- drawing the screens in Node --------------------------------------------------------------------------------------

# Loader hooks: the static folder's .js files are ES modules (no package.json says so), and js/app.js (the shell:
# router, header, slip sheet) becomes a stub exporting the same names, so a screen's module loads on its own.
HOOKS = r"""
import { readFileSync } from "node:fs";

let STATIC = "";

export async function initialize(data) {
  STATIC = data.static;
}

function exportNames(source) {
  const names = new Set();
  for (const m of source.matchAll(/export\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/g)) names.add(m[1]);
  for (const m of source.matchAll(/export\s+(?:const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)) names.add(m[1]);
  for (const m of source.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const part of m[1].split(",")) {
      const name = part.trim().split(/\s+as\s+/).pop().trim();
      if (/^[A-Za-z_$][\w$]*$/.test(name) && name !== "default") names.add(name);
    }
  }
  return [...names];
}

export async function load(url, context, nextLoad) {
  const path = url.split("?")[0];
  if (!STATIC || !path.startsWith(STATIC) || !path.endsWith(".js")) return nextLoad(url, context);
  if (path === `${STATIC}js/app.js`) {
    const names = exportNames(readFileSync(new URL(path), "utf8"));
    const source = ["const S = globalThis.__appStub;", ...names.map((n) => `export const ${n} = S.get(${JSON.stringify(n)});`)];
    return { format: "module", shortCircuit: true, source: source.join("\n") };
  }
  return nextLoad(url, { ...context, format: "module" });
}
"""

# Reads {static, hooks, cases} on stdin, writes {results} on stdout. A section case gives sectionHtml() for a state; a
# controller case runs gamePicks() on a stand-in page (start, the route's answers, clicks, the slip changing) and gives
# a snapshot after each step; a page case renders views/game.js itself; a shape case gives the slip legs a game pick,
# Ask AI's pickLeg and the Builder's slipLeg make; a call case gives what a module's function returns (JSON).
DRIVER = r"""
import * as nodeModule from "node:module";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(0, "utf8"));
// A pipe's write is asynchronous: exit only once it's all out
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
if (typeof nodeModule.register !== "function") {
  finish({ skip: `node ${process.version} has no module.register` });
  await new Promise(() => {});
}
const STATIC = pathToFileURL(input.static.replace(/\/*$/, "/")).href;
nodeModule.register(pathToFileURL(input.hooks).href, { data: { static: STATIC } });

// A browser, as far as these modules reach for one when they load and while they draw
const noop = () => {};
function storage() {
  const items = new Map();
  return {
    getItem: (k) => (items.has(String(k)) ? items.get(String(k)) : null), setItem: (k, v) => { items.set(String(k), String(v)); },
    removeItem: (k) => { items.delete(String(k)); }, clear: () => items.clear(), key: (i) => [...items.keys()][i] ?? null,
    get length() { return items.size; },
  };
}
const byId = new Map();
function element(tag) {
  return {
    tagName: String(tag).toUpperCase(), id: "", style: {}, dataset: {}, children: [], textContent: "", innerHTML: "",
    className: "", isConnected: true, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    setAttribute: noop, getAttribute: () => null, removeAttribute: noop, hasAttribute: () => false,
    appendChild(child) { this.children.push(child); if (child?.id) byId.set(child.id, child); return child; },
    append: noop, prepend: noop, remove: noop, replaceChildren: noop,
    addEventListener: noop, removeEventListener: noop, querySelector: () => null, querySelectorAll: () => [],
    contains: () => false, focus: noop, blur: noop, closest: () => null, matches: () => false, scrollIntoView: noop,
  };
}
const doc = {
  head: element("head"), body: element("body"), documentElement: element("html"), activeElement: null, title: "",
  getElementById: (id) => byId.get(id) || null, createElement: element, createElementNS: (_, tag) => element(tag),
  querySelector: () => null, querySelectorAll: () => [], addEventListener: noop, removeEventListener: noop,
  visibilityState: "visible", hidden: false,
};
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.addEventListener = noop;
globalThis.removeEventListener = noop;
globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
globalThis.location = { hash: "", href: "http://test.local/", origin: "http://test.local", pathname: "/", search: "", host: "test.local" };
globalThis.history = { state: null, length: 1, replaceState: noop, pushState: noop, back: noop };
globalThis.matchMedia = () => ({ matches: false, media: "", addEventListener: noop, removeEventListener: noop, addListener: noop, removeListener: noop });
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.innerWidth = 390;
globalThis.innerHeight = 844;
globalThis.scrollTo = noop;
globalThis.Element = class Element {};
globalThis.CSS = { escape: (s) => String(s).replace(/[^\w-]/g, (c) => `\\${c}`) };

const mod = (rel) => import(new URL(rel, STATIC).href);

// js/app.js's stub: the REAL store (slip, me, setMe, onBreak, legKey, oddsFormat, leagueTitle, ...) and the shell's
// layout and title as no-ops. Filled before any module that imports app.js loads.
const store = await mod("js/store.js");
const LAYOUT = { mode: "phone" };
const OVERRIDES = {
  ...store, layout: LAYOUT, onLayoutChange: () => noop, setTitle: noop, refreshMainButton: noop, replaceAddress: noop,
};
globalThis.__appStub = { get: (name) => (name in OVERRIDES ? OVERRIDES[name] : noop) };

// fetch: each path's answers in order (the last one repeats); ``hold`` keeps an answer back until a "release" step
const fetched = [];
let routes = {};
const held = [];
const routeOf = (url) => (url.startsWith("/api/game/picks") ? "picks" : url.startsWith("/api/game?") ? "game"
  : url.startsWith("/api/me") ? "me" : "other");
globalThis.fetch = async (path, options = {}) => {
  const url = String(path);
  fetched.push(url);
  const queue = routes[routeOf(url)] || [];
  const answer = (queue.length > 1 ? queue.shift() : queue[0]) || { status: 404, body: { detail: "Not Found" } };
  const reply = () => {
    if (options.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const status = answer.status ?? 200;
    const body = JSON.stringify(answer.body ?? null);
    return { status, ok: status < 400, headers: new Headers(), json: async () => JSON.parse(body) };
  };
  if (answer.hold) return new Promise((resolve, reject) => held.push(() => { try { resolve(reply()); } catch (e) { reject(e); } }));
  return reply();
};

const tick = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0)); };

function reset(spec) {
  routes = JSON.parse(JSON.stringify(spec.routes || {}));
  held.length = 0;
  fetched.length = 0;
  store.slip.clear();
  store.setMe({ uid: "web:test", age_confirmed: true, ...(spec.me || {}) });
  LAYOUT.mode = spec.layout || "phone";
  drainToasts();
}

/** The toasts shown since the last call: their words. */
function drainToasts() {
  const host = byId.get("toasts");
  if (!host) return [];
  const words = host.children.map((el) => String(el.innerHTML).replace(/<[^>]*>/g, "").trim());
  host.children.length = 0;
  return words;
}

/** A page root as far as the section and the game page reach for one: the section's slot (data-gp) is a window
    onto the page's html, so a redraw of the slot shows in the page. Listeners are kept for the steps to call. */
function pageRoot() {
  const OPEN = "<div data-gp>";
  const CLOSE = "</section>";
  const listeners = { click: [], toggle: [], input: [] };
  const root = {
    isConnected: true, innerHTML: "", listeners, dataset: {}, classList: { add: noop, remove: noop, toggle: noop },
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    removeEventListener(type, fn) { listeners[type] = (listeners[type] || []).filter((f) => f !== fn); },
    querySelector(sel) { return sel === "[data-gp]" && root.innerHTML.includes(OPEN) ? slot : null; },
    querySelectorAll: () => [], contains: () => false,
  };
  const bounds = () => {
    const a = root.innerHTML.indexOf(OPEN) + OPEN.length;
    const b = root.innerHTML.indexOf(CLOSE, a) + CLOSE.length;
    return [a, b];
  };
  const slot = {
    get innerHTML() { const [a, b] = bounds(); return root.innerHTML.slice(a, b); },
    set innerHTML(html) { const [a, b] = bounds(); root.innerHTML = root.innerHTML.slice(0, a) + html + root.innerHTML.slice(b); },
    contains: () => false, querySelector: () => null,
  };
  return root;
}

const target = (sel, dataset = {}) => ({ closest: (s) => (s === sel ? { dataset } : null) });

/** Fake timers for the section's own tries: set/clear as the controller takes them, advance(ms) runs what falls due. */
function fakeClock() {
  const clock = { now: 0, timers: [], scheduled: [], next: 1 };
  clock.set = (fn, ms) => {
    const id = clock.next++;
    clock.timers.push({ id, fn, at: clock.now + ms });
    clock.scheduled.push(ms);
    return id;
  };
  clock.clear = (id) => { clock.timers = clock.timers.filter((t) => t.id !== id); };
  clock.advance = async (ms) => {
    const until = clock.now + ms;
    for (;;) {
      const due = clock.timers.filter((t) => t.at <= until).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      clock.timers = clock.timers.filter((t) => t !== due);
      clock.now = due.at;
      due.fn();
      await tick();
    }
    clock.now = until;
  };
  return clock;
}

function legOf(i) {
  return { ref: { league: "basketball_nba", id: `other${i}`, market: "h2h", side: "home", point: null }, label: `Other ${i}`,
    game: `Away ${i} @ Home ${i}`, start: "2030-01-01T00:30:00Z", league: "basketball_nba", price: 1.9 };
}

async function steps(spec, root, extra) {
  const snaps = [];
  const clock = extra.clock;
  const snap = (at) => snaps.push({ at, html: root.innerHTML, legs: JSON.parse(JSON.stringify(store.slip.legs)),
    toasts: drainToasts(), fetched: [...fetched], pending: clock ? clock.timers.map((t) => t.at - clock.now) : [],
    scheduled: clock ? [...clock.scheduled] : [] });
  for (const step of spec.steps || []) {
    if (step.do === "settle") await tick();
    else if (step.do === "release") { held.splice(0).forEach((go) => go()); await tick(); }
    else if (step.do === "snap") snap(step.at || "");
    else if (step.do === "click") {
      const event = { target: step.target === "add" ? target("[data-gp-add]", { gpAdd: String(step.i ?? 0) }) : target("[data-gp-retry]"), preventDefault: noop };
      for (const fn of [...root.listeners.click]) fn(event);
      await tick();
    } else if (step.do === "toggle") {
      for (const fn of [...root.listeners.toggle]) fn({ target: { dataset: { gpWhy: step.key }, open: step.open !== false } });
    } else if (step.do === "slip") {
      if (step.op === "add") store.slip.add(step.leg);
      else if (step.op === "remove") store.slip.remove(step.leg.ref);
      else if (step.op === "clear") store.slip.clear();
      else if (step.op === "fill") for (let i = 0; i < step.n; i++) store.slip.add(legOf(i));
      await tick();
    } else if (step.do === "me") store.setMe({ ...store.me(), ...step.me });
    else if (step.do === "stop") extra.stop();
    else if (step.do === "abort") { extra.abort(); await tick(); }
    else if (step.do === "advance") await clock.advance(step.ms);
  }
  return snaps;
}

async function sectionCase(spec) {
  const picks = await mod("js/views/game/picks.js");
  reset(spec);
  const v = spec.view || {};
  const view = { ...v, game: spec.game, data: v.answer === undefined ? null : picks.readPicks(v.answer), open: new Set(v.open || []) };
  return { html: picks.sectionHtml(view) };
}

async function controllerCase(spec) {
  const picks = await mod("js/views/game/picks.js");
  reset(spec);
  const root = pageRoot();
  const controller = new AbortController();
  const clock = fakeClock();
  const gp = picks.gamePicks(spec.game, { signal: controller.signal, clock });
  root.innerHTML = `<div class="action-bar"></div><div data-gp>${gp.html()}</div><div data-cx="line"></div>`;
  const first = { at: "drawn", html: root.innerHTML, legs: [], toasts: [], fetched: [...fetched], pending: [], scheduled: [] };
  gp.start(root);
  const snaps = await steps(spec, root, { stop: () => gp.stop(), abort: () => controller.abort(), clock });
  gp.stop();
  controller.abort();
  return { snaps: [first, ...snaps] };
}

async function pageCase(spec) {
  const view = await mod("js/views/game.js");
  reset(spec);
  routes.game = [{ status: 200, body: spec.game }];
  const root = pageRoot();
  const controller = new AbortController();
  await view.render(root, { league: spec.game.league, id: spec.game.id, query: new URLSearchParams(spec.query || ""), signal: controller.signal });
  const first = { at: "rendered", html: root.innerHTML, legs: [], toasts: drainToasts(), fetched: [...fetched] };
  const snaps = await steps(spec, root, { stop: () => view.onLeave?.(), abort: () => controller.abort() });
  view.onLeave?.();
  controller.abort();
  return { snaps: [first, ...snaps] };
}

async function shapeCase(spec) {
  const picks = await mod("js/views/game/picks.js");
  const cards = await mod("js/views/ask/cards.js");
  const common = await mod("js/views/build/common.js");
  const model = await mod("js/views/today/model.js");
  reset(spec);
  const { pick, game } = spec;
  const gameText = `${game.away} @ ${game.home}`;
  const price = Number(pick.price);
  return {
    leg: picks.gamePickLeg(pick, game) ?? null,
    ask: cards.pickLeg({ ...pick, game: gameText, start: game.start }) ?? null,
    builder: common.slipLeg({ label: pick.label, game: gameText, start: game.start, league: game.league,
      odds: { decimal: price, american: model.american(price) }, ai_prob: pick.chance, market_prob: pick.market_chance,
      adjusted: pick.adjusted }, pick.ref),
  };
}

async function callCase(spec) {
  const module = await mod(spec.module);
  return { value: module[spec.fn](...(spec.args || [])) ?? null };
}

const KINDS = { section: sectionCase, controller: controllerCase, page: pageCase, shape: shapeCase, call: callCase };
const results = [];
for (const spec of input.cases || []) {
  try {
    results.push(await KINDS[spec.kind](spec));
  } catch (error) {
    results.push({ error: String(error?.stack || error) });
  }
}
finish({ results });
"""


@pytest.fixture(scope="module")
def draw(tmp_path_factory):
    """draw(*cases) -> each case's result: a section's parsed html, a controller's or page's snapshots (each with its
    parsed html), a shape's legs, or what a called function returned."""
    if not NODE:
        pytest.skip("node isn't installed: these tests draw the app's ES modules in Node")
    folder = tmp_path_factory.mktemp("game_picks")
    (folder / "hooks.mjs").write_text(HOOKS, encoding="utf-8")
    (folder / "driver.mjs").write_text(DRIVER, encoding="utf-8")

    def run(*cases: Dict[str, Any]) -> List[Any]:
        payload = json.dumps({"static": str(STATIC), "hooks": str(folder / "hooks.mjs"), "cases": list(cases)})
        env = {**os.environ, "TZ": "America/New_York", "NODE_OPTIONS": ""}
        done = subprocess.run([NODE, str(folder / "driver.mjs")], input=payload, capture_output=True, text=True,
                              encoding="utf-8", timeout=120, cwd=folder, env=env)
        assert done.returncode == 0, f"node failed ({done.returncode}):\n{done.stderr[-3000:]}"
        out = json.loads(done.stdout)
        if out.get("skip"):
            pytest.skip(out["skip"])
        drawn = []
        for case, result in zip(cases, out["results"]):
            assert "error" not in result, f"the {case['kind']} case threw in Node:\n{result['error']}"
            if case["kind"] == "section":
                drawn.append(parse(result["html"]))
            elif case["kind"] in ("controller", "page"):
                drawn.append([Snap(s) for s in result["snaps"]])
            elif case["kind"] == "call":
                drawn.append(result.get("value"))
            else:
                drawn.append(result)
        return drawn

    return run


class Snap:
    """One step of a controller or page case: the page's html (parsed), the slip's legs, the toasts and the paths
    fetched so far."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.at = raw["at"]
        self.page = parse(raw["html"])
        self.legs = raw["legs"]
        self.toasts = raw["toasts"]
        self.fetched = raw["fetched"]
        self.pending = raw.get("pending", [])  # the section's tries still to come (ms from now, fake clock)
        self.scheduled = raw.get("scheduled", [])  # every try it has set up so far (ms after the answer before it)

    @property
    def section(self) -> "El":
        return self.page.one(SECTION)

    def picks_fetched(self) -> List[str]:
        return [path for path in self.fetched if path.startswith("/api/game/picks")]


# --- reading the HTML (as tests/test_p0_fix_frontend.py reads it) ----------------------------------------------------

_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())
_PART = re.compile(r'([a-z][a-z0-9]*)|\.([\w-]+)|#([\w-]+)|\[([\w-]+)(?:="?([^\]"]*)"?)?\]')


class El:
    """An element of the drawn HTML: its tag, attributes, parent and children (elements and text)."""

    def __init__(self, tag: str, attrs: Mapping[str, Optional[str]], parent: Optional["El"] = None) -> None:
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: List[Any] = []
        self.html = ""

    @property
    def classes(self) -> set:
        return set((self.attrs.get("class") or "").split())

    def walk(self) -> Iterator["El"]:
        for child in self.children:
            if isinstance(child, El):
                yield child
                yield from child.walk()

    def text(self, *, heard: bool = False, seen: bool = False) -> str:
        """The text, whitespace collapsed: ``heard`` leaves out aria-hidden parts (what a screen reader says),
        ``seen`` leaves out sr-only parts (what's on screen)."""
        parts: List[str] = []

        def add(node: "El") -> None:
            for child in node.children:
                if isinstance(child, str):
                    parts.append(child)
                elif not ((heard and child.attrs.get("aria-hidden") == "true") or (seen and "sr-only" in child.classes)):
                    add(child)

        add(self)
        return " ".join("".join(parts).split())

    def select(self, selector: str) -> List["El"]:
        """Elements matching a simple CSS selector, in document order: tag, .class, #id, [attr] and [attr=value]
        parts, joined by descendant (space) and child (>) combinators; a comma separates alternatives."""
        chains = []
        for group in selector.split(","):
            steps, joint = [], " "
            for token in group.replace(">", " > ").split():
                if token == ">":
                    joint = ">"
                    continue
                parts = list(_PART.finditer(token))
                assert "".join(m.group(0) for m in parts) == token, f"selector part {token!r} isn't supported"
                steps.append((joint, parts))
                joint = " "
            chains.append(steps)
        return [el for el in self.walk() if any(_chain(el, steps, len(steps) - 1) for steps in chains)]

    def one(self, selector: str) -> "El":
        found = self.select(selector)
        assert len(found) == 1, f"{len(found)} elements match {selector!r}"
        return found[0]

    def texts(self, selector: str, **how: bool) -> List[str]:
        return [el.text(**how) for el in self.select(selector)]


def _matches(el: El, parts: List[re.Match]) -> bool:
    for m in parts:
        tag, cls, ident, attr, value = m.groups()
        if tag and el.tag != tag:
            return False
        if cls and cls not in el.classes:
            return False
        if ident and el.attrs.get("id") != ident:
            return False
        if attr and (attr not in el.attrs or (value is not None and el.attrs.get(attr) != value)):
            return False
    return True


def _chain(el: Optional[El], steps: List[Tuple[str, List[re.Match]]], i: int) -> bool:
    joint, parts = steps[i]
    if el is None or not _matches(el, parts):
        return False
    if i == 0:
        return True
    if joint == ">":
        return _chain(el.parent, steps, i - 1)
    up = el.parent
    while up is not None:
        if _chain(up, steps, i - 1):
            return True
        up = up.parent
    return False


class _Builder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = El("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        el = El(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(el)
        if tag not in _VOID:
            self.stack.append(el)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.stack[-1].children.append(El(tag, attrs, self.stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def parse(html: str) -> El:
    builder = _Builder()
    builder.feed(html)
    builder.close()
    builder.root.html = html
    return builder.root


def picks_of(scope: El) -> List[El]:
    return scope.select("ol.gp-list > li.gp-pick")


def states(scope: El) -> List[str]:
    """The section's one-line states (prices on their way or not in, started, nothing, props loading), as heard: each
    line's words (its spinner, when it has one, is left out)."""
    out = []
    for line in scope.select("p.gp-state"):
        words = [c for c in line.children if isinstance(c, El) and c.tag == "span" and "gp-spin" not in c.classes]
        out.append(words[0].text(heard=True))
    return out


def disclosure(scope: El) -> str:
    return scope.one("p.gp-disclosure").text()


# --- contract-shaped fixtures: Lab.game_picks' answer and the game page's game ---------------------------------------

TITLES = {"h2h": "Moneyline", "spreads": "Spread / handicap", "totals": "Totals", "btts": "Both teams to score"}


def pick(market: str = "h2h", side: str = "home", point: Optional[float] = None, label: str = "Boston Celtics",
         chance: float = 0.6412, market_chance: Optional[float] = 0.5801, price: Any = "1.72", book: str = "FanDuel",
         brain_higher: bool = True, why: Optional[List[str]] = None, league: str = NBA, event: str = GAME_ID,
         **over: Any) -> Dict[str, Any]:
    """One of Lab.game_picks' picks (engine/builder.py game_picks)."""
    out = {"ref": {"league": league, "id": event, "market": market, "side": side, "point": point},
           "key": f"{league}:{event}:{market}:{side}:{point}", "market": market, "market_title": TITLES.get(market, market),
           "label": label, "chance": chance, "market_chance": market_chance, "price": price, "book": book,
           "edge": 0.1029, "brain_higher": brain_higher,
           "why": ["Markets say 64% for Boston", "Home court, two days' rest", "The books moved toward Boston"]
           if why is None else why}
    out.update(over)
    return out


PICKS = [
    pick(),
    pick("spreads", "home", -4.5, "Boston Celtics -4.5", 0.5523, 0.5214, "1.91", "DraftKings", True),
    pick("totals", "over", 224.5, "Over 224.5", 0.5312, None, 1.95, "BetMGM", False),
]
NAMES = ["Jayson Tatum", "Jaylen Brown", "LeBron James", "Anthony Davis", "Derrick White", "Austin Reaves"]


def price(decimal: float, american: str) -> Dict[str, Any]:
    return {"decimal": decimal, "american": american}


def posted(at: float, over: Tuple[float, str] = (1.87, "-115"), under: Tuple[float, str] = (1.95, "-105"),
           **more: Any) -> Dict[str, Any]:
    return {"kind": "over_under", "line": at, "label": str(at), "market": "player_points", "over": price(*over),
            "under": price(*under), "yes": None, "book": "DraftKings", "as_of": START, "stale": False, "matches": True,
            **more}


def prop(i: int, stat: str = "pts", stat_label: str = "points", side: str = "over", at: float = 24.5,
         chance: float = 0.62, edge: Optional[float] = 0.051, **over: Any) -> Dict[str, Any]:
    """A props row: the P0 scan's row (engine/scan.py), as the posted-line scan and scan_edges give it."""
    out = {
        "rank": 11 + i, "athlete_id": str(4000 + i), "name": NAMES[i % len(NAMES)], "team": "BOS", "opponent": "LAL",
        "home": True, "game": "Los Angeles Lakers @ Boston Celtics", "event_id": "401700001", "start": START,
        "status": None, "status_tag": None, "minutes": 34.5, "threshold": int(at + 0.5), "kind": "count",
        "projection": 26.1, "chance": chance, "chance_source": "nba", "model": "nba", "model_label": "NBA model",
        "hits": {"l5": [4, 5], "l10": [7, 10], "season": [20, 30]}, "avg": {"l5": 27.0, "l10": 26.0, "season": 25.1},
        "line": posted(at), "edge": edge, "why": ["7 of last 10 over"], "league": NBA, "addable": False,
        "stat": stat, "stat_label": stat_label, "side": side,
    }
    out.update(over)
    return out


LIKELIEST = [prop(0, chance=0.71, edge=0.012), prop(1, "reb", "rebounds", "over", 6.5, 0.66, -0.013),
             prop(2, "ast", "assists", "under", 8.5, 0.61, 0.004)]
EDGES = [prop(3, "pra", "PRA", "over", 44.5, 0.58, 0.093), prop(4, "fg3m", "threes", "over", 2.5, 0.55, 0.071),
         prop(5, "pts", "points", "under", 19.5, 0.57, 0.052)]


def answer(picks: Optional[List[Dict[str, Any]]] = None, likeliest: Optional[List[Dict[str, Any]]] = None,
           edges: Optional[List[Dict[str, Any]]] = None, day: Optional[str] = "today", **over: Any) -> Dict[str, Any]:
    """Lab.game_picks' answer (GET /api/game/picks)."""
    props: Dict[str, Any] = {"likeliest": LIKELIEST if likeliest is None else likeliest,
                             "edges": EDGES if edges is None else edges}
    if day is not None:
        props["day"] = day
    out = {"picks": PICKS if picks is None else picks, "props": props,
           "game": {"label": "LAL @ BOS", "start": START, "started": False},
           "as_of": "2029-12-31T23:50:00Z"}
    out.update(over)
    return out


def game(**over: Any) -> Dict[str, Any]:
    """GET /api/game's game, as far as the page draws it: no AI model (an NBA game without a prediction)."""
    offer = lambda p, am, fair, label, point=None: {  # noqa: E731
        "price": p, "american": am, "fair": fair, "ai": fair, "books": 6, "label": label, "point": point}
    out = {
        "id": GAME_ID, "league": NBA, "home": "Boston Celtics", "away": "Los Angeles Lakers", "start": START,
        "live": False, "prediction": None, "soccer": False, "context": None, "players": None, "analysts": None,
        "espn_home_win": 0.6, "injuries": {"home": [], "away": []}, "books": [],
        "markets": {
            "h2h": {"home": offer(1.72, "-139", 0.5801, "Boston Celtics"), "away": offer(2.2, "+120", 0.4199, "Los Angeles Lakers")},
            "spreads": {"home": offer(1.91, "-110", 0.5214, "Boston Celtics -4.5", -4.5),
                        "away": offer(1.91, "-110", 0.4786, "Los Angeles Lakers +4.5", 4.5)},
            "totals": {"over": offer(1.95, "-105", 0.5, "Over 224.5", 224.5), "under": offer(1.87, "-115", 0.5, "Under 224.5", 224.5)},
        },
    }
    out.update(over)
    return out


GAME = game()
PAGE_GAME = {key: GAME[key] for key in ("id", "league", "home", "away", "start", "live")}
HOSTILE = '<img src=x onerror="window.__xss=1">'
QUOTES = "O'Neal \"Shaq\" & <b>Co</b>"


def ok(body: Any, **more: Any) -> Dict[str, Any]:
    return {"status": 200, "body": body, **more}


def refused(status: int, detail: str) -> Dict[str, Any]:
    return {"status": status, "body": {"detail": detail}}


def section_case(view: Mapping[str, Any], *, me: Optional[Mapping[str, Any]] = None,
                 game_: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "section", "game": dict(game_ or PAGE_GAME), "view": dict(view), "me": dict(me or {})}


def ready(body: Any, **view: Any) -> Dict[str, Any]:
    return section_case({"status": "ready", "answer": body, **view})


def controller_case(picks_answers: List[Dict[str, Any]], steps: List[Dict[str, Any]], *,
                    me: Optional[Mapping[str, Any]] = None, game_: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "controller", "game": dict(game_ or PAGE_GAME), "routes": {"picks": picks_answers},
            "steps": steps, "me": dict(me or {})}


SETTLE, SNAP = {"do": "settle"}, {"do": "snap"}


def snap(at: str) -> Dict[str, Any]:
    return {"do": "snap", "at": at}


def click_add(i: int = 0) -> Dict[str, Any]:
    return {"do": "click", "target": "add", "i": i}


# The slate's answers to a request for a league's odds (data/slate.py Slate._queue_odds: code -> (queued, words))
SLATE_ANSWERS = {
    "queued": (True, "Loading the latest odds."),
    "loading": (True, "The latest odds are on their way."),
    "fresh": (False, "The odds are up to date."),
    "throttled": (False, "Asked for less than 2 hours ago."),
    "no_games": (False, "No games coming up in this league."),
    "reserve": (False, "The month's odds allowance is down to its reserve."),
    "daily_cap": (False, "Today's odds loads are used up; saved odds stay on show."),
    "not_in_season": (False, "No odds market for this league right now."),
    "no_listing": (False, "The list of leagues in season isn't available right now."),
    "error": (False, "Couldn't ask for odds right now."),
}


def lab_unpriced(code: Optional[str] = None, *, slate: bool = True) -> Dict[str, str]:
    """game_picks' ``unpriced`` as the backend words it (the real Lab._game_unpriced) for the slate's answer ``code``
    (``slate`` False: a process without the slate, the bot's)."""
    class Slate:
        def request_odds(self, league: str) -> Dict[str, Any]:
            queued, words = SLATE_ANSWERS[code]
            return {"queued": queued, "reason": words, "code": code}

    return Lab._game_unpriced(SimpleNamespace(service=SimpleNamespace(slate=Slate() if slate else None)), NBA, {})


ASKED_REASON = lab_unpriced("queued")  # the slate queued the board: on its way
RESERVE_REASON = lab_unpriced("reserve")  # the month's odds allowance is at its reserve: not coming
# The code the route gives (PRICE_REASONS' key, dashboard/lab.py _price_why) with fast-odds' words
ON_WAY = {"reason": f"No prices are loaded for this game yet; {PRICE_REASONS['on_way']}", "code": "on_way"}


def said(reason: str) -> str:
    """A reason as the section says it: a sentence."""
    return f"{reason[0].upper()}{reason[1:]}{'' if reason.endswith(('.', '!', '?', '…')) else '.'}"


def unpriced_answer(unpriced: Any) -> Dict[str, Any]:
    return answer(picks=[], likeliest=[], edges=[], unpriced=unpriced)

PAUSED_UNTIL = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=20)).isoformat()
PAUSE_400 = ("Picks are paused until 9:41 PM Sun 1 Nov after an earlier message about losses or money worries. "
             "The odds and the maths stay open.")


# --- the section on the page --------------------------------------------------------------------------------------------

def around(html: str) -> Tuple[str, str]:
    """A page's html before the section's slot (data-gp) and after the section."""
    start = html.index("<div data-gp>")
    end = html.index("</section>", start) + len("</section>")
    return html[:start], html[end:]


def test_the_section_sits_right_after_the_action_bar_and_the_page_never_waits_for_it(draw) -> None:
    (snaps,) = draw({"kind": "page", "game": GAME, "routes": {"picks": [ok(answer(), hold=True)]},
                     "steps": [SETTLE, snap("waiting"), {"do": "release"}, snap("answered")]})
    rendered, waiting, answered = snaps
    # render() came back while the picks were still on their way: the page is drawn, the section a skeleton
    assert rendered.picks_fetched() == [f"/api/game/picks?league={NBA}&id={GAME_ID}"], rendered.fetched
    for step in (rendered, waiting):
        section = step.section
        assert section.one("h2").text() == "Suggested picks"
        assert section.one(".skel-wrap").attrs.get("aria-busy") == "true"
        assert not picks_of(section) and disclosure(section) == DISCLOSURE
    assert rendered.page.select(".game-hero") and rendered.page.one(".action-bar a").text() == "Ask AI about this game"
    # Right after the action bar ("Ask AI about this game"), before the line the rest of the page starts with
    marks = []
    for el in rendered.page.walk():
        if "action-bar" in el.classes:
            marks.append("bar")
        elif "data-gp" in el.attrs:
            marks.append("slot")
        elif el.tag == "section" and el.attrs.get("id") == "suggested-picks":
            marks.append("picks")
        elif el.attrs.get("data-cx") == "line":
            marks.append("line")
    assert marks == ["bar", "slot", "picks", "line"], marks
    assert rendered.page.one("div[data-gp] > section.gp").attrs.get("id") == "suggested-picks"
    # The answer replaced the skeleton in place; the page around it is as it was, and the route was asked once
    section = answered.section
    assert not section.select(".skel-wrap")
    assert [li.one(".gp-label").text() for li in picks_of(section)] == ["Boston Celtics", "Boston Celtics -4.5", "Over 224.5"]
    assert around(answered.page.html) == around(rendered.page.html)
    assert answered.picks_fetched() == rendered.picks_fetched()


def test_a_live_game_and_a_paused_profile_never_ask_the_route(draw) -> None:
    live, paused = draw(
        {"kind": "page", "game": game(live=True), "routes": {"picks": [ok(answer())]}, "steps": [SETTLE, SNAP]},
        {"kind": "page", "game": GAME, "routes": {"picks": [ok(answer())]}, "steps": [SETTLE, SNAP],
         "me": {"picks_paused_until": PAUSED_UNTIL}})
    for snaps in (live, paused):
        for step in snaps:
            assert not step.picks_fetched(), step.fetched
            assert not picks_of(step.section) and not step.section.select("button[data-gp-add]")
            assert disclosure(step.section) == DISCLOSURE
    assert states(live[-1].section) == [STARTED]
    note = paused[-1].section.one(".gp-pause")
    assert note.text().startswith("Picks are paused until ") and "losses or money worries" in note.text(), note.text()
    assert note.one("a").attrs.get("href") == SETTINGS and note.one("a").text() == "Settings"


# --- each game pick ---------------------------------------------------------------------------------------------------

def test_each_pick_shows_market_label_both_chances_price_book_why_and_add(draw) -> None:
    (page,) = draw(ready(answer()))
    first, spread, total = picks_of(page.one(SECTION))
    assert first.one(".gp-market").text() == "Moneyline" and first.one(".gp-label").text() == "Boston Celtics"
    # Side by side, as the owner asked: "Brain 64% · Market 58%" on screen, a pause between them for a screen reader
    assert first.one(".gp-chances").text(seen=True) == "Brain 64% · Market 58%"
    assert first.one(".gp-chances").text(heard=True) == "Brain 64%, Market 58%"
    assert first.one(".gp-price").text() == "1.72 at FanDuel"
    assert spread.one(".gp-market").text() == "Spread / handicap" and spread.one(".gp-price").text() == "1.91 at DraftKings"
    assert spread.one(".gp-chances").text(seen=True) == "Brain 55% · Market 52%"
    # A number price (not the engine's string) reads the same
    assert total.one(".gp-price").text() == "1.95 at BetMGM"
    why = first.one("details.gp-why")
    assert why.one("summary").text() == "Why?" and "open" not in why.attrs
    assert why.texts("li") == PICKS[0]["why"]
    add = first.one("button[data-gp-add]")
    assert add.text() == "Add to slip" and add.attrs.get("aria-pressed") == "false"
    assert add.attrs.get("aria-label") == "Add Boston Celtics to your slip" and "btn-primary" in add.classes
    assert page.one(SECTION).one("ol.gp-list").attrs.get("aria-label") == "The brain's pick for each market"
    assert page.one(SECTION).texts("h3.gp-caption") == ["Game picks", "Player props"]


def test_without_a_market_chance_only_the_brains_is_shown(draw) -> None:
    lone = pick("totals", "over", 224.5, "Over 224.5", 0.5312, None, "1.95", "BetMGM", brain_higher=True)
    (page,) = draw(ready(answer(picks=[lone])))
    (row,) = picks_of(page)
    assert row.one(".gp-chances").text(seen=True) == "Brain 53%" and row.one(".gp-chances").text(heard=True) == "Brain 53%"
    assert row.one(".gp-nums").text() == "Brain 53%"
    # No market chance, no comparison: the chip needs both numbers
    assert not row.select(".gp-higher")


def test_brain_higher_is_a_neutral_chip_and_a_game_pick_never_says_edge_or_value(draw) -> None:
    (page,) = draw(ready(answer()))
    first, spread, total = picks_of(page)
    for row in (first, spread):
        chip = row.one(".gp-higher")
        assert chip.text() == "Brain higher than market" and chip.classes == {"badge", "muted", "gp-higher"}
        assert not chip.select("svg")  # no mark, no colour of its own: just the words
    assert not total.select(".gp-higher")  # brain_higher false
    # Owner decision 4: models side by side, never an "edge" or "value" call on a game pick (the props keep the scan's)
    words = page.one("ol.gp-list").text(heard=True).lower() + " ".join(
        el.attrs.get("aria-label", "") for el in page.one("ol.gp-list").walk()).lower()
    assert "edge" not in words and "value" not in words, words


def test_prices_follow_the_persons_odds_format(draw) -> None:
    (page,) = draw(section_case({"status": "ready", "answer": answer(picks=[pick(price="1.72"), pick(
        "spreads", "away", 4.5, "Los Angeles Lakers +4.5", 0.45, 0.48, "2.50", "Caesars", False)])},
        me={"odds_format": "american"}))
    first, second = picks_of(page)
    assert first.one(".gp-price").text() == "-139 at FanDuel"
    assert second.one(".gp-price").text() == "+150 at Caesars"


def test_why_reads_the_persons_units_and_says_at_most_three_lines(draw) -> None:
    units = {"metric": ["Wind 19 km/h across", "12°C at tip-off", "a", "b"],
             "imperial": ["Wind 12 mph across", "54°F at tip-off", "c", "d"]}
    many = pick(why=["one", "two", "three", "four", "five"])
    auto, metric, many_page, none = draw(
        ready(answer(picks=[pick(why_units=units)])),
        section_case({"status": "ready", "answer": answer(picks=[pick(why_units=units)])}, me={"units": "metric"}),
        ready(answer(picks=[many])),
        ready(answer(picks=[pick(why=[])])))
    # "auto" is imperial for the NBA (views/context/units.js), as Ask AI's card reads why_units
    assert picks_of(auto)[0].one("details.gp-why").texts("li") == ["Wind 12 mph across", "54°F at tip-off", "c"]
    assert picks_of(metric)[0].one("details.gp-why").texts("li") == ["Wind 19 km/h across", "12°C at tip-off", "a"]
    assert picks_of(many_page)[0].one("details.gp-why").texts("li") == ["one", "two", "three"]
    # Nothing to say: no toggle
    assert not picks_of(none)[0].select("details.gp-why")


def test_a_pick_with_no_price_a_person_can_take_is_never_shown(draw) -> None:
    (page,) = draw(ready(answer(picks=[pick(price=None), pick("spreads", price="1.00"), pick("totals", price="abc"),
                                       pick("h2h", "away", label="Los Angeles Lakers", price="2.20")], likeliest=[],
                                edges=[])))
    assert [row.one(".gp-label").text() for row in picks_of(page)] == ["Los Angeles Lakers"]


# --- Add to slip --------------------------------------------------------------------------------------------------------

def test_add_to_slip_puts_the_builders_and_ask_ais_leg_in_the_real_slip_and_shows_it(draw) -> None:
    (snaps,) = draw(controller_case([ok(answer())], [
        SETTLE, snap("ready"), click_add(0), snap("added"), click_add(1), snap("both"), click_add(0), snap("removed")]))
    _, ready_, added, both, removed = snaps
    assert ready_.legs == [] and ready_.toasts == []
    (leg,) = added.legs
    # The leg the Builder's and Ask AI's cards add (the store adds key and added), every field the server's
    assert {k: v for k, v in leg.items() if k not in ("key", "added")} == {
        "ref": {"league": NBA, "id": GAME_ID, "market": "h2h", "side": "home", "point": None},
        "label": "Boston Celtics", "game": "Los Angeles Lakers @ Boston Celtics", "start": START, "league": NBA,
        "price": 1.72, "american": "-139", "ai": 0.6412, "fair": 0.5801}
    assert leg["key"] == f"{NBA}:{GAME_ID}:h2h:home" and isinstance(leg["added"], (int, float))
    assert added.toasts == ["Added Boston Celtics to your slip"]
    button = picks_of(added.section)[0].one("button[data-gp-add]")
    assert button.text() == "In your slip" and button.attrs.get("aria-pressed") == "true"
    assert button.attrs.get("aria-label") == "Boston Celtics: in your slip. Tap to take it out"
    assert "btn-secondary" in button.classes
    # The others still offer theirs
    assert [b.text() for b in added.section.select("button[data-gp-add]")] == ["In your slip", "Add to slip", "Add to slip"]
    assert [l["ref"]["market"] for l in both.legs] == ["h2h", "spreads"]
    assert both.legs[1]["ref"]["point"] == -4.5 and both.legs[1]["price"] == 1.91 and both.legs[1]["fair"] == 0.5214
    # Tapped again: off the slip, and the button says so
    assert [l["ref"]["market"] for l in removed.legs] == ["spreads"]
    assert removed.toasts == ["Took Boston Celtics off your slip"]
    assert [b.text() for b in removed.section.select("button[data-gp-add]")] == ["Add to slip", "In your slip", "Add to slip"]


def test_the_leg_has_the_same_shape_the_builder_and_ask_ai_add(draw) -> None:
    cases = [pick(), pick("spreads", "home", -4.5, "Boston Celtics -4.5", 0.5523, 0.5214, "1.91"),
             pick("totals", "under", 224.5, "Under 224.5", 0.47, None, 1.87)]
    found = draw(*({"kind": "shape", "pick": p, "game": PAGE_GAME} for p in cases))
    for p, shape in zip(cases, found):
        leg, ask, builder = shape["leg"], shape["ask"], shape["builder"]
        # Ask AI's pickLeg: every field it adds, the same
        assert {k: leg[k] for k in ask} == ask, (leg, ask)
        # The Builder's slipLeg (american, ai and fair as well): the very same leg
        assert leg == builder, (leg, builder)


def test_an_adjusted_pick_keeps_adjusted_in_its_slip_leg_as_the_builders_does(draw) -> None:
    # A totals pick whose chance a forecast or roof change moved (chance_source "adjusted", adjusted set), and the same
    # selection unmoved
    adjusted = pick("totals", "over", 224.5, "Over 224.5", 0.5612, 0.5214, "1.95", "BetMGM", True,
                    adjusted=True, chance_source="adjusted")
    plain = pick("totals", "over", 224.5, "Over 224.5", 0.5312, 0.5214, "1.95", "BetMGM", False, chance_source="books")
    moved, unmoved, snaps = draw(
        {"kind": "shape", "pick": adjusted, "game": PAGE_GAME}, {"kind": "shape", "pick": plain, "game": PAGE_GAME},
        controller_case([ok(answer(picks=[PICKS[0], adjusted], likeliest=[], edges=[]))], [SETTLE, click_add(0), click_add(1), SNAP]))
    # The Builder's leg for the same selection carries adjusted: the game pick's leg is that very leg
    assert moved["builder"].get("adjusted") is True and moved["leg"] == moved["builder"], (moved["leg"], moved["builder"])
    assert "adjusted" not in unmoved["leg"] and unmoved["leg"] == unmoved["builder"]
    # Through the real slip: the kept leg says so, and the slip's own word for its chance is "Adj" (ui.js legChanceWord)
    legs = {leg["ref"]["market"]: leg for leg in snaps[-1].legs}
    assert legs["totals"].get("adjusted") is True and "adjusted" not in legs["h2h"], legs
    word_moved, word_plain = draw(
        {"kind": "call", "module": "js/ui.js", "fn": "legChanceWord", "args": [legs["totals"]]},
        {"kind": "call", "module": "js/ui.js", "fn": "legChanceWord", "args": [legs["h2h"]]})
    assert (word_moved, word_plain) == ("Adj", "AI")


def test_the_slip_button_follows_the_slip_changed_elsewhere(draw) -> None:
    same = {"ref": {"league": NBA, "id": GAME_ID, "market": "h2h", "side": "home", "point": None},
            "label": "Boston Celtics", "game": "Los Angeles Lakers @ Boston Celtics", "start": START, "league": NBA,
            "price": 1.72}
    other_side = {**same, "ref": {**same["ref"], "side": "away"}, "label": "Los Angeles Lakers", "price": 2.2}
    (snaps,) = draw(controller_case([ok(answer())], [
        SETTLE, {"do": "slip", "op": "add", "leg": same}, snap("odds button"),
        {"do": "slip", "op": "add", "leg": other_side}, snap("swapped"), {"do": "slip", "op": "clear"}, snap("cleared")]))
    _, tapped, swapped, cleared = snaps
    first = lambda s: picks_of(s.section)[0].one("button[data-gp-add]").text()  # noqa: E731
    assert first(tapped) == "In your slip"
    # The other side of the market swaps it out (the slip's rule): Boston is no longer in
    assert first(swapped) == "Add to slip" and [l["label"] for l in swapped.legs] == ["Los Angeles Lakers"]
    assert first(cleared) == "Add to slip" and cleared.legs == []


def test_a_full_slip_says_so_and_adds_nothing(draw) -> None:
    (snaps,) = draw(controller_case([ok(answer())], [SETTLE, {"do": "slip", "op": "fill", "n": 12}, click_add(0), SNAP]))
    full = snaps[-1]
    assert len(full.legs) == 12 and all(l["ref"]["id"] != GAME_ID for l in full.legs)
    assert full.toasts == ["Your slip is full: remove a pick first"]
    assert picks_of(full.section)[0].one("button[data-gp-add]").text() == "Add to slip"


def test_a_soccer_market_pick_shows_its_price_with_the_slip_button_off(draw) -> None:
    epl = {**PAGE_GAME, "league": EPL, "id": "epl1", "home": "Arsenal", "away": "Chelsea"}
    btts = pick("btts", "yes", None, "Both teams to score: yes", 0.58, 0.55, "1.80", "Bet365", True, league=EPL, event="epl1")
    h2h = pick(label="Arsenal", league=EPL, event="epl1")
    (snaps,) = draw(controller_case([ok(answer(picks=[h2h, btts], likeliest=[], edges=[]))],
                                    [SETTLE, SNAP, click_add(1), SNAP], game_=epl))
    _, drawn, tapped = snaps
    _, soccer = picks_of(drawn.section)
    assert soccer.one(".gp-market").text() == "Both teams to score" and soccer.one(".gp-price").text() == "1.80 at Bet365"
    button = soccer.one(".gp-add button")
    assert "disabled" in button.attrs and "data-gp-add" not in button.attrs and button.text() == "Add to slip"
    assert soccer.one(".gp-noslip").text() == NO_SLIP
    # A tap that reaches the handler anyway adds nothing (the slip would price it at the goals model's fair odds)
    assert tapped.legs == [] and tapped.toasts == []
    # The moneyline beside it is the slip's
    assert picks_of(drawn.section)[0].one("button[data-gp-add]").text() == "Add to slip"


def test_a_pick_for_another_game_never_reaches_the_slip(draw) -> None:
    stray = pick(label="Somebody else", event="another-game")
    (snaps,) = draw(controller_case([ok(answer(picks=[stray], likeliest=[], edges=[]))], [SETTLE, click_add(0), SNAP]))
    row = picks_of(snaps[-1].section)[0]
    assert "disabled" in row.one(".gp-add button").attrs and not row.select(".gp-noslip")
    assert snaps[-1].legs == []


def test_an_open_why_stays_open_when_the_slip_redraws_the_section(draw) -> None:
    (snaps,) = draw(controller_case([ok(answer())], [
        SETTLE, {"do": "toggle", "key": "spreads", "open": True}, click_add(0), snap("redrawn"),
        {"do": "toggle", "key": "spreads", "open": False}, click_add(0), snap("closed")]))
    redrawn, closed = snaps[-2], snaps[-1]
    opened = [("open" in row.one("details.gp-why").attrs) for row in picks_of(redrawn.section)]
    assert opened == [False, True, False]
    assert not any("open" in row.one("details.gp-why").attrs for row in picks_of(closed.section))


# --- the player props -------------------------------------------------------------------------------------------------

def test_props_are_the_scans_own_rows_likeliest_then_best_edges_with_their_edge(draw) -> None:
    (page,) = draw(ready(answer()))
    props = page.one(SECTION).one(".gp-props")
    assert props.one("h3.gp-caption").text() == "Player props"
    assert props.texts("h4.gp-sub") == ["Likeliest", "Best edges"]
    likeliest, edges = props.select("ol.ps-list")
    assert likeliest.attrs.get("aria-label") == "Likeliest" and edges.attrs.get("aria-label") == "Best edges"
    assert [li.one(".ps-name").text() for li in likeliest.select("li.ps-item")] == NAMES[:3]
    assert [li.one(".ps-name").text() for li in edges.select("li.ps-item")] == NAMES[3:6]
    # Ranked 1-3 within each list, whatever the scan's own rank
    assert [li.one(".ps-rank").text() for li in edges.select("li.ps-item")] == ["1", "2", "3"]
    # Each row says its own stat and side, and a posted line shows its edge, as in the scan
    first = edges.select("li.ps-item")[0]
    assert first.one(".ps-stat").text() == "PRA (P+R+A)"
    assert first.one(".ps-line-main").text() == "DraftKings O 44.5 1.87"
    nums = {pair.one("dt").text(): pair.one("dd").text(heard=True) for pair in first.select("dl.ps-nums > div")}
    assert nums["Edge"] == "+9.3%" and nums["Chance"] == "58%", nums
    under = edges.select("li.ps-item")[2]
    assert under.one(".ps-stat").text() == "Points" and under.one(".ps-line-main").text() == "DraftKings U 19.5 1.95"
    assert likeliest.select("li.ps-item")[1].one(".ps-stat").text() == "Rebounds"
    assert likeliest.select("li.ps-item")[1].texts("dl.ps-nums > div > dt") == ["Proj.", "Chance", "Edge", "L5", "L10", "Season"]
    assert likeliest.select("li.ps-item")[1].select("dl.ps-nums > div")[2].one("dd").text(heard=True) == "−1.3%"
    # View player, as on the scan's rows; no slip button (the slip takes game picks)
    assert first.one(".ps-actions a").attrs.get("href") == f"#/player/{NBA}/4003"
    assert not props.select("button[data-gp-add], [data-scan-add]")


def test_props_rows_are_drawn_by_the_scans_own_row_function(draw) -> None:
    rows = [{**EDGES[0], "rank": 1}, {**EDGES[1], "rank": 2}]
    page, *own = draw(ready(answer(likeliest=[], edges=EDGES[:2])), *({
        "kind": "call", "module": "js/views/players/scan.js", "fn": "rowItemHtml",
        "args": [row, i, {"view": {"edges": True, "league": NBA}}]} for i, row in enumerate(rows)))
    # The scan's own markup, character for character (the finder's and Ask AI's card's rows)
    for html in own:
        assert html and html in page.html


def test_a_stale_props_line_never_shows_its_edge(draw) -> None:
    stale = prop(3, "pra", "PRA", "over", 44.5, 0.58, 0.093, line=posted(44.5, stale=True))
    (page,) = draw(ready(answer(likeliest=[], edges=[stale])))
    item = page.one(".gp-props li.ps-item")
    nums = {pair.one("dt").text(): pair.one("dd").text(heard=True) for pair in item.select("dl.ps-nums > div")}
    assert nums["Edge"] == "–" and "Old line" in item.one(".ps-line").text()


def test_props_keep_to_three_rows_each_and_an_empty_list_is_left_out(draw) -> None:
    more = [prop(i % 6, chance=0.7 - i / 100) for i in range(5)]
    page, only_edges = draw(ready(answer(likeliest=more, edges=more)), ready(answer(likeliest=[], edges=EDGES[:1])))
    for ol in page.select(".gp-props ol.ps-list"):
        assert len(ol.select("li.ps-item")) == 3
    assert only_edges.texts(".gp-props h4.gp-sub") == ["Best edges"]


def test_see_all_opens_the_finders_best_edges_for_the_league_and_day(draw) -> None:
    today, tomorrow, nfl = draw(
        ready(answer(day="today")), ready(answer(day="tomorrow")),
        section_case({"status": "ready", "answer": answer(day="today")},
                     game_={**PAGE_GAME, "league": "americanfootball_nfl"}))
    link = today.one("a.gp-finder")
    assert link.text() == "See all in the Player finder"
    assert link.attrs.get("href") == f"#/players?league={NBA}&stat=pts&at=line&day=today&sort=edges"
    assert tomorrow.one("a.gp-finder").attrs.get("href") == f"#/players?league={NBA}&stat=pts&at=line&day=tomorrow&sort=edges"
    assert nfl.one("a.gp-finder").attrs.get("href") == \
        "#/players?league=americanfootball_nfl&stat=pass_yds&at=line&day=today&sort=edges"


def test_without_props_day_the_finder_day_is_the_games_start_on_espns_day(draw) -> None:
    eastern = ZoneInfo("America/New_York")
    now = datetime.datetime.now(datetime.timezone.utc)
    soon = (now + datetime.timedelta(hours=3)).replace(microsecond=0)
    far = (now + datetime.timedelta(days=4)).replace(microsecond=0)
    expected = {0: "today", 1: "tomorrow"}.get((soon.astimezone(eastern).date() - now.astimezone(eastern).date()).days)
    near, later, tennis = draw(
        section_case({"status": "ready", "answer": answer(day=None)}, game_={**PAGE_GAME, "start": soon.isoformat()}),
        section_case({"status": "ready", "answer": answer(day=None)}, game_={**PAGE_GAME, "start": far.isoformat()}),
        section_case({"status": "ready", "answer": answer(day="today")}, game_={**PAGE_GAME, "league": TENNIS}))
    assert near.one("a.gp-finder").attrs.get("href").endswith(f"&day={expected}&sort=edges")
    # A game on neither day: the finder can't show it (the props still show)
    assert not later.select("a.gp-finder") and later.select(".gp-props li.ps-item")
    # A league the scan doesn't cover: no finder link
    assert not tennis.select("a.gp-finder")


def test_the_servers_props_day_wins_over_the_games_start(draw) -> None:
    soon = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)).replace(microsecond=0)
    near = {**PAGE_GAME, "start": soon.isoformat()}
    said_none, said_tomorrow = draw(
        section_case({"status": "ready", "answer": answer(day=None) | {"props": {"likeliest": LIKELIEST, "edges": [], "day": None}}},
                     game_=near),
        section_case({"status": "ready", "answer": answer(day="tomorrow")}, game_=near))
    # props.day null (the server: neither day, or no scan for the league): no link, whatever the start says
    assert not said_none.select("a.gp-finder") and said_none.select(".gp-props li.ps-item")
    assert said_tomorrow.one("a.gp-finder").attrs.get("href").endswith("&day=tomorrow&sort=edges")


BOARD_DAYS = [  # (asked at, the game's start, its board day)
    ("2026-03-07T04:30:00Z", "2026-03-07T04:00:00Z", "today"),      # 11:00 PM EST Mar 6, asked at 11:30 PM
    ("2026-03-07T04:30:00Z", "2026-03-08T03:00:00Z", "tomorrow"),   # 10:00 PM EST Mar 7 (the night before DST)
    ("2026-03-07T04:30:00Z", "2026-03-08T05:30:00Z", None),         # 12:30 AM EST Mar 8: the day after tomorrow
    ("2026-11-01T03:30:00Z", "2026-11-01T23:00:00Z", "tomorrow"),   # asked 11:30 PM EDT Oct 31, game 6 PM EST Nov 1
    ("2026-11-01T03:30:00Z", "2026-11-01T03:59:00Z", "today"),      # 11:59 PM EDT Oct 31
    ("2026-11-01T03:30:00Z", "2026-10-31T16:00:00Z", "today"),      # earlier the same day (started: shown as such)
    ("2026-11-01T03:30:00Z", "not a time", None),
    ("2026-11-01T03:30:00Z", None, None),
]


def test_board_day_is_espns_us_eastern_day(draw) -> None:
    ms = lambda iso: datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000  # noqa: E731
    found = draw(*({"kind": "call", "module": "js/views/game/picks.js", "fn": "boardDay", "args": [start, ms(now)]}
                   for now, start, _ in BOARD_DAYS))
    assert found == [day for _, _, day in BOARD_DAYS]


# --- the states ---------------------------------------------------------------------------------------------------------

def test_paused_by_the_route_shows_the_servers_own_words_with_a_link_to_settings_and_no_picks(draw) -> None:
    (snaps,) = draw(controller_case([refused(400, PAUSE_400)], [SETTLE, SNAP]))
    section = snaps[-1].section
    note = section.one(".gp-pause")
    assert note.one("span").text() == PAUSE_400
    link = note.one("a.text-link")
    assert link.attrs.get("href") == SETTINGS and link.text() == "Settings"
    assert not picks_of(section) and not section.select(".gp-props, button[data-gp-add], .ps-item")
    assert disclosure(section) == DISCLOSURE
    assert snaps[-1].picks_fetched() == [f"/api/game/picks?league={NBA}&id={GAME_ID}"]


def waiting(section: El) -> bool:
    """The section is waiting for prices on their way: one line, with the spinner and its words, first."""
    lines = section.select(".gp-body > p.gp-state.gp-wait")
    return len(lines) == 1 and bool(lines[0].select("span.gp-spin")) and states(section)[:1] == [ON_THEIR_WAY]


def test_prices_on_their_way_are_asked_for_again_at_15_and_45_seconds_then_the_reason(draw) -> None:
    (snaps,) = draw(controller_case([ok(unpriced_answer(ON_WAY))], [
        SETTLE, snap("first"), {"do": "advance", "ms": 14_999}, snap("not yet"), {"do": "advance", "ms": 1}, snap("try 1"),
        {"do": "advance", "ms": 29_999}, snap("not yet 2"), {"do": "advance", "ms": 1}, snap("try 2"),
        {"do": "advance", "ms": 120_000}, snap("later")]))
    _, first, not_yet, try1, not_yet2, try2, later = snaps
    path = f"/api/game/picks?league={NBA}&id={GAME_ID}"
    # The first answer: on their way, with the spinner; never "check back"; the first try set for 15 s
    assert waiting(first.section) and first.section.one("span.gp-spin").attrs.get("aria-hidden") == "true"
    assert first.section.one("p.gp-wait").attrs.get("role") == "status"
    assert "check back" not in first.section.text().lower() and not first.section.select("[data-gp-retry]")
    assert first.picks_fetched() == [path] and first.pending == [15_000]
    assert not_yet.picks_fetched() == [path] and waiting(not_yet.section)
    # At 15 s the section asks again by itself (only its slot is redrawn), and sets the last try 30 s on (45 s in)
    assert try1.picks_fetched() == [path, path] and try1.pending == [30_000] and waiting(try1.section)
    assert around(try1.page.html) == around(first.page.html)
    assert len(not_yet2.picks_fetched()) == 2
    # Still unpriced after the last try: the server's own reason, no spinner, no more tries
    assert try2.picks_fetched() == [path] * 3 and try2.pending == [] and try2.scheduled == [15_000, 30_000]
    assert states(try2.section) == [said(ON_WAY["reason"])] and not try2.section.select(".gp-spin, [data-gp-retry]")
    assert later.picks_fetched() == [path] * 3 and later.pending == []
    for step in snaps:
        assert disclosure(step.section) == DISCLOSURE


def test_prices_that_arrive_on_a_try_are_shown_and_the_waiting_ends(draw) -> None:
    (snaps,) = draw(controller_case([ok(unpriced_answer(ON_WAY)), ok(answer())],
                                    [SETTLE, {"do": "advance", "ms": 15_000}, snap("priced"), {"do": "advance", "ms": 120_000}, SNAP]))
    priced, later = snaps[-2], snaps[-1]
    assert [row.one(".gp-label").text() for row in picks_of(priced.section)][0] == "Boston Celtics"
    assert not priced.section.select(".gp-spin, p.gp-wait") and priced.pending == []
    assert len(later.picks_fetched()) == 2 and later.scheduled == [15_000]


def test_any_other_reason_is_the_servers_own_words_at_once_with_no_tries(draw) -> None:
    codes = {"arriving": PRICE_REASONS["arriving"], "reserve": PRICE_REASONS["reserve"],
             "daily_cap": PRICE_REASONS["daily_cap"], "no_games": PRICE_REASONS["no_games"]}
    found = draw(*(controller_case([ok(unpriced_answer({"reason": f"No prices are loaded for this game yet; {words}",
                                                         "code": code}))],
                                   [SETTLE, snap("first"), {"do": "advance", "ms": 120_000}, snap("later")])
                   for code, words in codes.items()))
    for (code, words), snaps in zip(codes.items(), found):
        first, later = snaps[-2], snaps[-1]
        assert states(first.section) == [f"No prices are loaded for this game yet; {words}."], (code, states(first.section))
        assert not first.section.select(".gp-spin, [data-gp-retry]") and first.pending == [], code
        assert len(later.picks_fetched()) == 1 and later.scheduled == [], code
        assert "on their way" not in first.section.text() and "check back" not in first.section.text().lower()
    # The two the owner would meet, word for word
    reserve, daily = found[1][-1].section, found[2][-1].section
    assert states(reserve) == ["No prices are loaded for this game yet; the month's odds allowance is at its reserve."]
    assert states(daily) == ["No prices are loaded for this game yet; today's odds allowance is used up."]


def test_the_tries_stop_when_the_section_goes_or_the_page_is_left(draw) -> None:
    stopped, aborted = draw(
        controller_case([ok(unpriced_answer(ON_WAY))], [SETTLE, snap("waiting"), {"do": "stop"},
                                                        {"do": "advance", "ms": 120_000}, snap("after")]),
        controller_case([ok(unpriced_answer(ON_WAY))], [SETTLE, snap("waiting"), {"do": "abort"},
                                                        {"do": "advance", "ms": 120_000}, snap("after")]))
    for snaps in (stopped, aborted):
        waiting_, after = snaps[-2], snaps[-1]
        assert waiting_.pending == [15_000] and waiting(waiting_.section)
        # No try is left set, none is made, and nothing is drawn after the section went
        assert after.pending == [] and len(after.picks_fetched()) == 1
        assert after.page.html == waiting_.page.html


def test_a_try_that_fails_counts_and_the_waiting_goes_on(draw) -> None:
    (snaps,) = draw(controller_case(
        [ok(unpriced_answer(ON_WAY)), refused(502, "The server had a problem (502). Try again in a minute."),
         ok(unpriced_answer(ON_WAY))],
        [SETTLE, {"do": "advance", "ms": 15_000}, snap("failed try"), {"do": "advance", "ms": 30_000}, snap("last")]))
    failed, last = snaps[-2], snaps[-1]
    # The failed try leaves the waiting line (not an error), and the last try is still set
    assert waiting(failed.section) and not failed.section.select(".gp-error") and failed.pending == [30_000]
    assert len(last.picks_fetched()) == 3 and states(last.section) == [said(ON_WAY["reason"])] and last.pending == []


def test_the_raw_slate_codes_and_a_server_without_codes_are_read_too(draw) -> None:
    cases = {
        "queued": ({"reason": "No prices are loaded for this game yet; the latest odds are loading in the background",
                    "code": "queued"}, True),
        "loading": ({"reason": "No prices are loaded for this game yet; the latest odds are on their way", "code": "loading"}, True),
        # No code (an older server): the reason, read loosely
        "words on way": ({"reason": f"No prices are loaded for this game yet; {PRICE_REASONS['on_way']}"}, True),
        "words backend on way": ({"reason": ASKED_REASON["reason"]}, True),  # the backend's own words, code left off
        "words backend reserve": ({"reason": RESERVE_REASON["reason"]}, False),
        "words loading": ({"reason": "No prices are loaded for this game yet; the latest odds are loading in the background"}, True),
        "words arriving": ({"reason": f"No prices are loaded for this game yet; {PRICE_REASONS['arriving']}"}, False),
    }
    pages = draw(*(controller_case([ok(unpriced_answer(unpriced))], [SETTLE, SNAP]) for unpriced, _ in cases.values()))
    for (key, (unpriced, on_way)), snaps in zip(cases.items(), pages):
        section, step = snaps[-1].section, snaps[-1]
        if on_way:
            assert waiting(section) and step.pending == [15_000], (key, states(section))
        else:
            assert states(section) == [said(unpriced["reason"])] and step.pending == [], (key, states(section))


def test_every_answer_the_backend_gives_reads_right(draw) -> None:
    # The real Lab._game_unpriced over each of the slate's answers and a process with no slate: only the board asked for
    # or on its way (queued, loading) is waited for; every other one is the server's own words at once
    reasons = {code: lab_unpriced(code) for code in SLATE_ANSWERS}
    reasons["no slate"] = lab_unpriced(slate=False)
    pages = draw(*(ready(unpriced_answer(unpriced)) for unpriced in reasons.values()))
    for (key, unpriced), page in zip(reasons.items(), pages):
        section = page.one(SECTION)
        if key in ("queued", "loading"):
            assert waiting(section), (key, unpriced, states(section))
        else:
            assert states(section) == [said(unpriced["reason"])], (key, unpriced, states(section))
            assert not section.select(".gp-spin, [data-gp-retry]") and "on their way" not in states(section)[0], key
        assert "check back" not in section.text().lower() and disclosure(section) == DISCLOSURE


def test_unpriced_without_a_reason_says_only_that_prices_arent_in(draw) -> None:
    empty, flag = draw(ready(unpriced_answer({})), ready(unpriced_answer(True)))
    for page in (empty, flag):
        assert states(page.one(SECTION)) == [UNPRICED_PLAIN] and not page.select("[data-gp-retry], .gp-spin")


def test_props_loading_offers_check_again_unless_the_section_is_already_waiting(draw) -> None:
    loading = {"games_ready": 0, "games_total": 0}
    reserve, on_way, priced = draw(
        ready(answer(picks=[], likeliest=[], edges=[], unpriced=RESERVE_REASON, loading=loading)),
        ready(answer(picks=[], likeliest=[], edges=[], unpriced=ON_WAY, loading=loading)),
        ready(answer(likeliest=[], edges=[], loading=loading)))
    for page in (reserve, priced):
        (button,) = page.select("button[data-gp-retry]")
        assert button.parent.parent.attrs.get("class") == "gp-props" and button.text() == "Check again"
    # While the section waits for prices, its own tries refresh the props too: no button
    assert states(on_way.one(SECTION)) == [ON_THEIR_WAY, LOADING] and not on_way.select("[data-gp-retry]")


def test_check_again_is_busy_while_the_answer_is_on_its_way(draw) -> None:
    loading = {"games_ready": 3, "games_total": 9}
    (snaps,) = draw(controller_case(
        [ok(answer(likeliest=[], edges=[], loading=loading)), ok(answer(), hold=True)],
        [SETTLE, {"do": "click", "target": "retry"}, snap("busy"), {"do": "click", "target": "retry"}, snap("still"),
         {"do": "release"}, snap("done")]))
    busy, still, done = snaps[-3:]
    button = busy.section.one("button[data-gp-retry]")
    assert button.text() == "Checking…" and "disabled" in button.attrs and button.attrs.get("aria-busy") == "true"
    assert len(still.picks_fetched()) == 2  # a second tap while busy asks nothing more
    assert done.section.select(".gp-props li.ps-item") and not done.section.select("[data-gp-retry]")


def test_a_game_the_server_says_has_started_shows_no_picks_and_no_props(draw) -> None:
    started = answer(game={"label": "LAL @ BOS", "start": START, "started": True})  # props sent anyway
    (page,) = draw(ready(started))
    section = page.one(SECTION)
    assert states(section) == [STARTED]
    assert not picks_of(section) and not section.select(".gp-props, .ps-item, a.gp-finder, button")
    assert disclosure(section) == DISCLOSURE


def test_nothing_at_all_says_no_picks_yet(draw) -> None:
    nothing, not_found, empty_body = draw(
        ready(answer(picks=[], likeliest=[], edges=[])),
        controller_case([refused(404, "Game not found")], [SETTLE, SNAP]),
        ready({}))
    assert states(nothing.one(SECTION)) == [EMPTY]
    assert states(not_found[-1].section) == [EMPTY]
    assert states(empty_body.one(SECTION)) == [EMPTY]
    for section in (nothing.one(SECTION), not_found[-1].section):
        assert not section.select(".gp-props, .gp-list") and disclosure(section) == DISCLOSURE


def test_props_still_loading_say_so_with_whatever_rows_are_ready(draw) -> None:
    loading = {"games_ready": 3, "games_total": 9}
    top, inside, with_rows, no_picks = draw(
        ready(answer(likeliest=[], edges=[], loading=loading)),
        ready({**answer(likeliest=[], edges=[]), "props": {"likeliest": [], "edges": [], "loading": loading}}),
        ready(answer(likeliest=LIKELIEST[:1], edges=[], loading=loading)),
        ready(answer(picks=[], likeliest=[], edges=[], loading=loading)))
    for page in (top, inside, with_rows):
        props = page.one(SECTION).one(".gp-props")
        assert states(props) == [LOADING]
        assert props.one("p.gp-state button[data-gp-retry]").text() == "Check again"
        assert picks_of(page.one(SECTION))  # the game picks don't wait for the props
    assert [li.one(".ps-name").text() for li in with_rows.select(".gp-props li.ps-item")] == [NAMES[0]]
    # Loading with no game picks: only that line (not "No picks for this game yet.")
    assert states(no_picks.one(SECTION)) == [LOADING]


def test_an_error_offers_try_again(draw) -> None:
    (snaps,) = draw(controller_case([refused(500, "The server had a problem (500). Try again in a minute."), ok(answer())],
                                    [SETTLE, snap("error"), {"do": "click", "target": "retry"}, snap("again")]))
    error, again = snaps[-2], snaps[-1]
    box = error.section.one(".gp-error")
    assert box.one(".notice").text() == "Couldn't load the suggested picks. The server had a problem (500). Try again in a minute."
    assert box.one("button[data-gp-retry]").text() == "Try again"
    assert disclosure(error.section) == DISCLOSURE
    assert picks_of(again.section) and not again.section.select(".gp-error")


def test_the_disclosure_line_is_there_in_every_state(draw) -> None:
    pages = draw(
        section_case({"status": "loading"}), section_case({"status": "paused", "pause": "Picks are paused."}),
        section_case({"status": "started"}), section_case({"status": "error", "error": "No."}),
        ready(answer()), ready(answer(picks=[], unpriced={"reason": "x"})), ready({}))
    for page in pages:
        section = page.one(SECTION)
        assert section.one("h2").text() == "Suggested picks"
        assert disclosure(section) == DISCLOSURE
        # The last thing in the section
        assert [el for el in section.walk() if el.tag in ("p", "div", "ol", "details")][-1].classes == {"note", "gp-disclosure"}


def test_after_the_page_goes_nothing_is_drawn_and_the_slip_is_not_followed(draw) -> None:
    (snaps,) = draw(controller_case([ok(answer(), hold=True)], [
        {"do": "stop"}, {"do": "release"}, snap("late answer"),
        {"do": "slip", "op": "fill", "n": 1}, snap("slip changed")]))
    drawn, late, changed = snaps
    for step in (late, changed):
        assert step.page.html == drawn.page.html  # still the skeleton: nothing painted after stop()


def test_the_request_is_cancelled_with_the_page(draw) -> None:
    (snaps,) = draw(controller_case([ok(answer(), hold=True)], [{"do": "abort"}, {"do": "release"}, SNAP]))
    assert snaps[-1].section.select(".skel-wrap") and not picks_of(snaps[-1].section)


# --- escaping -----------------------------------------------------------------------------------------------------------

def test_every_string_is_escaped(draw) -> None:
    hostile_pick = pick(label=HOSTILE, market_title=QUOTES, book=HOSTILE, why=[HOSTILE, QUOTES])
    hostile_prop = prop(0, name=HOSTILE, team=QUOTES, opponent=HOSTILE, stat="weird", stat_label=HOSTILE,
                        line=posted(24.5, book=HOSTILE), why=[HOSTILE], athlete_id='1" onclick="x')
    page, paused, failed, unpriced = draw(
        ready(answer(picks=[hostile_pick], likeliest=[hostile_prop], edges=[])),
        controller_case([refused(400, f"Picks are paused until {HOSTILE}")], [SETTLE, SNAP]),
        controller_case([refused(502, HOSTILE)], [SETTLE, SNAP]),
        ready(answer(picks=[], likeliest=[], edges=[], unpriced={"reason": HOSTILE})))
    for scope in (page.one(SECTION), paused[-1].section, failed[-1].section, unpriced.one(SECTION)):
        # No element from outside: no image, no handler, no bold "Co" from the quotes' markup
        assert not scope.select("img"), scope.html
        assert not [el for el in scope.walk() if el.tag == "b" and el.text() == "Co"]
        for el in scope.walk():
            assert not any(name.startswith("on") for name in el.attrs), el.attrs
    row = picks_of(page)[0]
    assert row.one(".gp-label").text() == HOSTILE and row.one(".gp-market").text() == QUOTES
    assert row.one(".gp-price").text() == f"1.72 at {HOSTILE}"
    assert row.one("details.gp-why").texts("li") == [HOSTILE, QUOTES]
    assert row.one("button[data-gp-add]").attrs.get("aria-label") == f"Add {HOSTILE} to your slip"
    item = page.one(".gp-props li.ps-item")
    assert item.one(".ps-name").text() == HOSTILE and HOSTILE in item.one(".ps-line").text()
    assert not item.select(".ps-actions a")  # an athlete id the route can't take: no player link at all
    assert paused[-1].section.one(".gp-pause span").text() == f"Picks are paused until {HOSTILE}"
    assert HOSTILE in failed[-1].section.one(".gp-error .notice").text()
    assert states(unpriced.one(SECTION)) == [f"{HOSTILE}."]
