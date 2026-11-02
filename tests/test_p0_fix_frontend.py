"""P0 fix phase, the screens: Ask AI's player_scan card and the Player finder, drawn by the app's own ES modules.

The P0 frontend fixes had no test in the repo suite, only Playwright scripts kept in scratch (persona review m6). These
pin them:
  - m12: the finder's Number label is heard once ("Number: each player's posted line"), not with the words twice.
  - m4 (web): the first five rows, then the rest that qualify under "Also:", and how many more qualify.
  - m5 (web): a yes/no stat (anytime TD, double-double) shows only its chance, never a projection.
  - m6 (web): rows ranked by edge (each player's posted line) show the edge beside the chance.
And the screens' side of the P0 fix contracts, worded as the chat's answer and the bot word them:
  - 1 (F3): a note for each name the scan couldn't rank: not found ("I couldn't find {name} in tonight's games."),
    more than one match ("More than one player matches {name}: {c1}, {c2} or {c3}. Which one?") and a game already
    started ("{player}'s game ({game}) has already started, so it isn't in the scan."), every name escaped.
  - 2 (F2): the trend chip says what the last 5 were compared with: "L5 28.4 vs 24.1 season" or "... last season".
  - 3 (M5/F6): ``games`` counts the games still to come: "Every game tonight has started, so nobody is left to rank."
    only when none is left, else "{N} game(s) tonight {have|has} already started: their players aren't listed."
  - 4 (m1): the "more" line is the result's own ``more``, never counted again, worded by mode ("{N} more qualify",
    "{N} more players are ranked", "{N} more posted lines have an edge": open the Player finder).
And the final check's V4: the names the finder sends keep digits, suffixes and parentheses ("Marvin Bagley 3rd",
"Michael Porter, Jr." as one player, "Marcus Sutton (MIA)") by the rule the bot's finder button shares, and markup still
never survives.

How: Node imports js/views/ask/cards.js and js/views/finder.js from this tree's dashboard/static with a small stand-in
for the browser (document, localStorage, location), js/app.js (the shell) replaced by a stub exporting the same names
(a loader hook), and GET /api/players/scan answered by a stubbed fetch with the case's scan. The HTML comes back here
and is read with html.parser: what a screen reader hears (aria-hidden parts left out) and what's on screen (sr-only
parts left out). No browser, no server, no network. Skipped when Node (with module.register: 18.19+ or 20.6+) is
missing. The static folder is the one next to this file's tests/ folder, so the same file run from an older tree's
tests/ draws that tree's screens.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from urllib.parse import quote

import pytest

from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.engine import scan as engine

STATIC = Path(__file__).resolve().parents[1] / "telegram_betting_bot_simple" / "dashboard" / "static"
NODE = shutil.which("node")
NBA, NFL, NHL = "basketball_nba", "americanfootball_nfl", "icehockey_nhl"

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

# Reads {static, hooks, cases} on stdin, writes {results} on stdout. A card case gives its html (after a Refresh with
# ``refresh``, the fresh scan laid over the card as Ask AI does it) and the query its Refresh asks; a finder case
# gives the html of its three boxes (controls, summary, results), the scans it asked for and the addresses it set; a
# call case gives what a module's function returns for the arguments (JSON).
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
function element(tag) {
  return {
    tagName: String(tag).toUpperCase(), id: "", style: {}, dataset: {}, children: [], textContent: "", innerHTML: "",
    className: "", isConnected: true, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    setAttribute: noop, getAttribute: () => null, removeAttribute: noop, hasAttribute: () => false,
    appendChild(child) { this.children.push(child); return child; }, append: noop, prepend: noop, remove: noop,
    addEventListener: noop, removeEventListener: noop, querySelector: () => null, querySelectorAll: () => [],
    contains: () => false, focus: noop, blur: noop, closest: () => null, matches: () => false,
  };
}
const byId = new Map();
const doc = {
  head: element("head"), body: element("body"), documentElement: element("html"), activeElement: null, title: "",
  getElementById: (id) => byId.get(id) || null, createElement: element, createElementNS: (_, tag) => element(tag),
  querySelector: () => null, querySelectorAll: () => [], addEventListener: noop, removeEventListener: noop,
  visibilityState: "visible", hidden: false,
};
doc.head.appendChild = (child) => { if (child?.id) byId.set(child.id, child); return child; };
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

// js/app.js's stub: the slip, the layout ("phone": cards; wider: the table) and the address the screens use
const replaced = [];
const LAYOUT = { mode: "phone" };
const OVERRIDES = {
  slip: { has: () => false, legs: () => [], add: noop, remove: noop, toggle: noop, onChange: () => noop },
  layout: LAYOUT, onLayoutChange: () => noop, replaceAddress: (hash) => { replaced.push(hash); }, setTitle: noop,
  legKey: (ref) => JSON.stringify(ref ?? null),
};
globalThis.__appStub = { get: (name) => (name in OVERRIDES ? OVERRIDES[name] : noop) };

// GET /api/players/scan answered with the case's scan (a fresh copy each time)
const fetched = [];
let answer = {};
globalThis.fetch = async (path) => {
  fetched.push(String(path));
  const body = JSON.stringify(answer);
  return { status: 200, ok: true, headers: new Headers(), json: async () => JSON.parse(body) };
};

const mod = (rel) => import(new URL(rel, STATIC).href);

async function finderCase(spec) {
  const finder = await mod("js/views/finder.js");
  LAYOUT.mode = spec.layout || "desk";
  answer = spec.scan || {};
  fetched.length = 0;
  replaced.length = 0;
  const hash = spec.hash || "#/players";
  location.hash = hash;
  const boxes = {};
  const box = (sel) => (boxes[sel] ||= { innerHTML: "", contains: () => false, querySelector: () => null, querySelectorAll: () => [], addEventListener: noop, dataset: {} });
  const root = { isConnected: true, innerHTML: "", addEventListener: noop, removeEventListener: noop, querySelector: box, querySelectorAll: () => [], contains: () => false };
  const controller = new AbortController();
  await finder.render(root, { query: new URLSearchParams(hash.split("?")[1] || ""), signal: controller.signal });
  finder.onLeave?.();
  controller.abort();
  return {
    controls: boxes["[data-pf-controls]"]?.innerHTML || "", summary: boxes["[data-pf-summary]"]?.innerHTML || "",
    results: boxes["[data-pf-results]"]?.innerHTML || "", fetched: [...fetched], replaced: [...replaced],
  };
}

async function callCase(spec) {
  const module = await mod(spec.module);
  return { value: module[spec.fn](...(spec.args || [])) };
}

async function cardCase(spec) {
  const cards = await mod("js/views/ask/cards.js");
  const scan = await mod("js/views/players/scan.js");
  const card = spec.refresh ? cards.refreshedScan(spec.card, spec.refresh) : spec.card;
  return { html: cards.cardHtml(card, spec.key || "0:0", spec.options || {}), refreshQuery: scan.scanQuery(cards.scanAsk(card)), card };
}

const results = [];
for (const spec of input.cases || []) {
  try {
    results.push(spec.kind === "finder" ? await finderCase(spec) : spec.kind === "call" ? await callCase(spec) : await cardCase(spec));
  } catch (error) {
    results.push({ error: String(error?.stack || error) });
  }
}
finish({ results });
"""


@pytest.fixture(scope="module")
def draw(tmp_path_factory):
    """draw(*cases) -> each case drawn: a card's parsed html (with .refresh_query), a finder's parsed boxes, or what
    a called function returned."""
    if not NODE:
        pytest.skip("node isn't installed: these tests draw the app's ES modules in Node")
    folder = tmp_path_factory.mktemp("screens")
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
            if case["kind"] == "call":
                drawn.append(result.get("value"))
            elif case["kind"] == "card":
                page = parse(result["html"])
                page.refresh_query, page.card = result["refreshQuery"], result["card"]
                drawn.append(page)
            else:
                drawn.append(SimpleNamespace(controls=parse(result["controls"]), summary=parse(result["summary"]),
                                             results=parse(result["results"]), fetched=result["fetched"],
                                             replaced=result["replaced"]))
        return drawn

    return run


def card_case(card: Mapping[str, Any], **extra: Any) -> Dict[str, Any]:
    return {"kind": "card", "card": dict(card), **extra}


def finder_case(hash_: str, scan_: Mapping[str, Any], layout: str = "desk") -> Dict[str, Any]:
    return {"kind": "finder", "hash": hash_, "scan": dict(scan_), "layout": layout}


# --- reading the HTML ----------------------------------------------------------------------------------------------

_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())
_PART = re.compile(r'([a-z][a-z0-9]*)|\.([\w-]+)|#([\w-]+)|\[([\w-]+)(?:="?([^\]"]*)"?)?\]')


class El:
    """An element of the drawn HTML: its tag, attributes, parent and children (elements and text)."""

    def __init__(self, tag: str, attrs: Mapping[str, Optional[str]], parent: Optional["El"] = None) -> None:
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: List[Any] = []
        self.html = ""
        self.refresh_query = ""  # a card's: the scan its Refresh asks for, and the card it draws
        self.card: Dict[str, Any] = {}

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


def numbers(item: El) -> List[Tuple[str, str]]:
    """A card row's numbers block, in order: [(label, value)], e.g. [("Proj.", "31.2"), ("Chance", "82%"), ...]."""
    return [(pair.one("dt").text(), pair.one("dd").text(heard=True)) for pair in item.select("dl.ps-nums > div")]


def row_names(scope: El, selector: str = "li.ps-item") -> List[str]:
    return [item.select(".ps-name, .pf-name")[0].text() for item in scope.select(selector)]


def table_bodies(page: El) -> List[El]:
    return page.select("table.pf-table > tbody")


def table_rows(body: El) -> List[El]:
    return [tr for tr in body.select("tbody > tr") if "pf-group" not in tr.classes]


def notes(page: El) -> List[str]:
    """The notes a card or the finder shows above its rows (the board's state, the names it couldn't rank)."""
    return page.texts("p.ps-alert")


# --- contract-shaped fixtures: Lab.player_scan's result and players_scan's card -----------------------------------------

TIP = "2030-01-01T00:30:00Z"
NAMES = ["Luka Dončić", "Shai Gilgeous-Alexander", "Jayson Tatum", "Nikola Jokić", "Anthony Edwards", "De'Aaron Fox",
         "Tyrese Haliburton", "Donovan Mitchell", "Jalen Brunson", "Devin Booker", "Kevin Durant", "Trae Young",
         "Paolo Banchero", "Scottie Barnes", "Franz Wagner", "Cade Cunningham", "Zion Williamson"]
HOSTILE = ['<img src=x onerror="window.__xss=1">', "O'Neal \"Shaq\" O’Brien", "Đorđević"]


def price(decimal: float, american: str) -> Dict[str, Any]:
    return {"decimal": decimal, "american": american}


def milestone(at: int = 20, over: Tuple[float, str] = (1.4, "-250"), **more: Any) -> Dict[str, Any]:
    return {"kind": "milestone", "line": at - 0.5, "label": f"{at}+", "market": "player_points_alternate",
            "over": price(*over), "under": None, "yes": None, "book": "DraftKings", "as_of": TIP, "stale": False,
            "matches": True, **more}


def posted(at: float, over: Tuple[float, str], under: Tuple[float, str], **more: Any) -> Dict[str, Any]:
    return {"kind": "over_under", "line": at, "label": str(at), "market": "player_points_rebounds_assists",
            "over": price(*over), "under": price(*under), "yes": None, "book": "DraftKings", "as_of": TIP,
            "stale": False, "matches": True, **more}


def yes_line(over: Tuple[float, str] = (2.3, "+130")) -> Dict[str, Any]:
    return {"kind": "yes", "line": None, "label": None, "market": "player_anytime_td", "over": None, "under": None,
            "yes": price(*over), "book": "DraftKings", "as_of": TIP, "stale": False, "matches": True}


def row(i: int, **over: Any) -> Dict[str, Any]:
    out = {
        "rank": i + 1, "athlete_id": str(4000 + i), "name": NAMES[i % len(NAMES)], "team": "DAL", "opponent": "PHX",
        "home": True, "game": "Phoenix Suns @ Dallas Mavericks", "event_id": "nba1", "start": TIP, "status": None,
        "status_tag": None, "minutes": 34.5, "threshold": 20, "kind": "count", "projection": round(30 - i * 0.7, 1),
        "chance": round(0.86 - i * 0.03, 4), "chance_source": "nba", "model": "nba", "model_label": "NBA model",
        "hits": {"l5": [4, 5], "l10": [8, 10], "season": [30, 40]}, "avg": {"l5": 28.4, "l10": 27.0, "season": 24.1},
        "line": milestone(), "edge": round(0.142 - i * 0.01, 4), "why": ["9 of last 10 at 20+"], "league": NBA,
        "addable": False,
    }
    out.update(over)
    return out


def scan(**over: Any) -> Dict[str, Any]:
    """Lab.player_scan's result (GET /api/players/scan): 20+ points tonight in the NBA unless told otherwise."""
    out = {"mode": "scan", "ask": "20+ points", "stat": "pts", "stat_label": "points", "threshold": 20, "side": "over",
           "day": "today", "leagues": [NBA], "players": [], "close": [], "more": 0, "games": 10, "games_started": 0,
           "error": None, "as_of": TIP, "missing_names": [], "names": []}
    out.update(over)
    return out


def card(found: Mapping[str, Any], **extra: Any) -> Dict[str, Any]:
    """players_scan's card (ai/tools.py scan_card): the scan's fields, its title, sort and names."""
    out = {"type": "player_scan", "source": "players_scan", "title": "Projected for 20+ points · NBA · tonight",
           "slip_note": "Player props are for reading here; the slip takes game picks.", "sort": None, "names": []}
    out.update(found)
    out.update(extra)
    return out


QUALIFIERS = [row(i) for i in range(12)]  # 5 first, then 7 under "Also:"
CLOSE = [row(i, chance=round(0.41 - (i - 12) * 0.02, 4), projection=18.2) for i in (12, 13, 14)]
POSTED = scan(ask="PRA overs (posted lines)", stat="pra", stat_label="PRA", threshold=None, players=[
    row(0, threshold=45, projection=44.1, chance=0.49, edge=0.081, line=posted(44.5, (2.2, "+120"), (1.67, "-150"))),
    row(1, threshold=39, projection=40.3, chance=0.58, edge=0.023, line=posted(38.5, (1.76, "-132"), (2.05, "+105"))),
    row(2, threshold=33, projection=33.9, chance=0.53, edge=-0.01, line=posted(32.5, (1.87, "-115"), (1.95, "-105"))),
])
ANYTIME_TD = scan(ask="anytime TD", stat="anytime_td", stat_label="anytime TD", threshold=1, day="tomorrow", leagues=[NFL],
                  games=8, players=[row(i, league=NFL, kind="yes_no", projection=None, threshold=1,
                                        chance=round(0.55 - i * 0.05, 2), line=yes_line(), edge=round(0.08 - i * 0.02, 2))
                                    for i in range(4)]
                  # an older server's row: no kind, and a projection
                  + [{key: value for key, value in row(4, league=NFL, threshold=1, chance=0.33, projection=0.4,
                                                       line=None, edge=None).items() if key != "kind"}])

FINDER_20 = "#/players?league=basketball_nba&stat=pts&at=20&day=today"
FINDER_PRA_LINE = "#/players?league=basketball_nba&stat=pra&at=line&day=today"
FINDER_TD = "#/players?league=americanfootball_nfl&stat=anytime_td&at=1&day=tomorrow"


# --- m6: the P0 frontend fixes (each of these fails on c5deb8f's screens) ----------------------------------------------

def test_m12_the_finders_number_label_is_heard_once(draw) -> None:
    line, over, under = draw(finder_case(FINDER_PRA_LINE, POSTED),
                             finder_case(FINDER_20, scan(players=QUALIFIERS[:3])),
                             finder_case("#/players?league=basketball_nba&stat=reb&at=12&day=today&side=under",
                                         scan(ask="under 11.5 rebounds", stat="reb", threshold=12, side="under")))
    label = line.controls.one("#pf-at-label")
    # c5deb8f: "Number: each player's posted line each player's posted line" (an sr-only copy and the shown words)
    assert label.text(heard=True) == "Number: each player's posted line", label.text(heard=True)
    assert label.text(seen=True) == "Number each player's posted line"
    assert over.controls.one("#pf-at-label").text(heard=True) == "Number: at least"
    assert under.controls.one("#pf-at-label").text(heard=True) == "Number: fewer than"
    # The stepper is named by that label
    assert line.controls.one("span.stepper").attrs.get("aria-labelledby") == "pf-at-label"


def test_m4_the_card_names_five_then_the_rest_under_also_and_how_many_more(draw) -> None:
    (page,) = draw(card_case(card(scan(players=QUALIFIERS, close=CLOSE, more=14))))
    first, *_ = page.select("ol.ps-list")
    # c5deb8f: one list of all 12, and no word of the 14 more
    assert row_names(first) == [r["name"] for r in QUALIFIERS[:5]], row_names(first)
    assert page.one("p.ps-also-head").text() == "Also:"
    also = page.one("ol.ps-also")
    assert also.attrs.get("start") == "6" and also.attrs.get("aria-label") == "Also: 7 more who qualify"
    assert row_names(also) == [r["name"] for r in QUALIFIERS[5:]]
    # The close ones stay collapsed, after the qualifiers
    close = page.one("details.ps-close")
    assert "open" not in close.attrs and close.one("summary").text() == "Also close (3)"
    assert row_names(close) == [r["name"] for r in CLOSE]
    more = page.one("p.ps-more").text()
    assert "14 more qualify" in more and "open the Player finder" in more, more


def test_m4_the_finder_splits_at_five_with_also_and_says_how_many_more(draw) -> None:
    found = scan(players=QUALIFIERS, close=CLOSE, more=14)
    desk, phone = draw(finder_case(FINDER_20, found), finder_case(FINDER_20, found, layout="phone"))
    bodies = table_bodies(desk.results)
    # c5deb8f: every qualifier in the first tbody, and no "Also:" group
    top_names = [tr.one(".pf-name").text() for tr in table_rows(bodies[0])]
    assert top_names == [r["name"] for r in QUALIFIERS[:5]], top_names
    assert [body.attrs.get("class") for body in bodies] == [None, "pf-also", "pf-close"]
    _, also, close = bodies
    head = also.one("tr.pf-also-head th")
    assert head.text(seen=True) == "Also:" and head.text(heard=True) == "Also: 7 more who qualify"
    assert [tr.one(".pf-name").text() for tr in table_rows(also)] == [r["name"] for r in QUALIFIERS[5:]]
    assert [tr.one(".pf-rank").text() for tr in table_rows(also)] == [str(n) for n in range(6, 13)]
    assert "pf-close" in close.classes and close.one("button.pf-toggle").text() == "Also close (3)"
    assert "14 more qualify beyond the 12 shown: raise the number, or filter by player, to narrow the list." in \
        desk.results.texts("p.ps-note")
    # Phones: the same split as cards
    first, *_ = phone.results.select("ol.ps-list")
    assert row_names(first) == [r["name"] for r in QUALIFIERS[:5]]
    assert row_names(phone.results.one("ol.ps-also")) == [r["name"] for r in QUALIFIERS[5:]]
    assert phone.results.one("p.ps-also-head").text() == "Also:"


def test_m5_a_yes_no_stat_shows_only_its_chance(draw) -> None:
    page, desk, phone = draw(card_case(card(ANYTIME_TD, title="Strongest anytime TD · NFL · tomorrow")),
                             finder_case(FINDER_TD, ANYTIME_TD), finder_case(FINDER_TD, ANYTIME_TD, layout="phone"))
    for where, scope in (("card", page), ("finder (phone)", phone.results)):
        items = scope.select("li.ps-item")
        assert len(items) == 5, where
        for item in items:
            # c5deb8f: "Proj. –" on each row, and "Proj. 0.4" on the older server's
            assert [label for label, _ in numbers(item)] == ["Chance", "L5", "L10", "Season"], (where, numbers(item))
        assert [dict(numbers(item))["Chance"] for item in items] == ["55%", "50%", "45%", "40%", "33%"], where
    heads = desk.results.texts("table.pf-table thead th")
    assert "Proj." not in heads and "Chance" in heads, heads
    assert not desk.results.select("td.pf-n")
    assert [tr.one("td.pf-c").text() for tr in table_rows(table_bodies(desk.results)[0])] == ["55%", "50%", "45%", "40%", "33%"]


def test_m6_rows_ranked_by_edge_show_the_edge_beside_the_chance(draw) -> None:
    posted_card = card(POSTED, title="Strongest PRA overs (posted lines) · NBA · tonight")
    page, phone, desk = draw(card_case(posted_card), finder_case(FINDER_PRA_LINE, POSTED, layout="phone"),
                             finder_case(FINDER_PRA_LINE, POSTED))
    edges = ["+8.1%", "+2.3%", "−1.0%"]
    for where, scope in (("card", page), ("finder (phone)", phone.results)):
        items = scope.select("li.ps-item")
        # Ranked by edge, as the server sent them: 49% above 58%, and the edge says why
        assert [dict(numbers(item))["Chance"] for item in items] == ["49%", "58%", "53%"], where
        for item, edge in zip(items, edges):
            labels = [label for label, _ in numbers(item)]
            # c5deb8f: no Edge beside the chance (Proj., Chance, L5, L10, Season)
            assert labels == ["Proj.", "Chance", "Edge", "L5", "L10", "Season"], (where, labels)
            assert dict(numbers(item))["Edge"] == edge, (where, numbers(item))
            assert item.text(seen=True).count(edge) == 1, (where, item.text(seen=True))  # said once, not twice
    # Wider screens: the posted line ranks by edge, and the Edge column says it
    rows = table_rows(table_bodies(desk.results)[0])
    assert [tr.one("td.pf-e").text() for tr in rows] == edges
    assert [tr.one("td.pf-c").text() for tr in rows] == ["49%", "58%", "53%"]
    assert [th.text() for th in desk.results.select("thead th[aria-sort]")] == ["Edge"]


# --- contract 1 (F3): the names the scan couldn't rank -------------------------------------------------------------------

NAMED = scan(ask="3+ threes", stat="fg3m", stat_label="threes", threshold=3,
             names=["Luka", "Jalen", "Draisaitl", *HOSTILE],
             players=[row(0, threshold=3, projection=3.4, chance=0.62, qualifies=True)],
             missing_names=[HOSTILE[0], HOSTILE[1]],
             ambiguous_names=[{"name": "Jalen", "candidates": ["Jalen Brunson", "Jalen Green", "Jalen Williams"]},
                              {"name": HOSTILE[2], "candidates": ["Luka Đorđević", "Nikola Đorđević"]}],
             started_names=[{"name": "Draisaitl", "player": "Leon Draisaitl", "game": "Edmonton Oilers @ Calgary Flames"},
                            {"name": "<b>Bold</b>", "player": "<b>Bold</b> O'Hara", "game": "<i>Ottawa</i> @ Boston"}])
NAME_NOTES = [
    'I couldn\'t find <img src=x onerror="window.__xss=1"> in tonight\'s games.',
    "I couldn't find O'Neal \"Shaq\" O’Brien in tonight's games.",
    "More than one player matches Jalen: Jalen Brunson, Jalen Green or Jalen Williams. Which one?",
    "More than one player matches Đorđević: Luka Đorđević or Nikola Đorđević. Which one?",
    "Leon Draisaitl's game (Edmonton Oilers @ Calgary Flames) has already started, so it isn't in the scan.",
    "<b>Bold</b> O'Hara's game (<i>Ottawa</i> @ Boston) has already started, so it isn't in the scan.",
]
FINDER_NAMED = "#/players?league=basketball_nba&stat=fg3m&at=3&day=today&player=Luka&player=Jalen&player=Draisaitl"


def _escaped(page: El) -> None:
    """No name became markup: each note's words are text only, no <img> anywhere, and the html has them escaped."""
    for note in page.select("p.ps-alert"):
        assert [child.tag for child in note.children if isinstance(child, El)] == ["svg", "span"], note.text()
        assert not [child for child in note.one("span").children if isinstance(child, El)], note.text()
    assert not page.select("img")
    assert "&lt;img src=x onerror=&quot;window.__xss=1&quot;&gt;" in page.html


def test_the_card_has_a_note_for_each_name_it_couldnt_rank_in_every_surfaces_words(draw) -> None:
    named_card = card(NAMED, title="Luka Dončić · 3+ threes · NBA · tonight", names=NAMED["names"])
    tomorrow = card({**NAMED, "day": "tomorrow", "ambiguous_names": [], "started_names": []},
                    title="3+ threes · NBA · tomorrow", names=NAMED["names"])
    nobody = card({**NAMED, "players": []}, names=NAMED["names"])
    bare = card({**NAMED, "missing_names": [], "ambiguous_names": [{"name": "Jalen", "candidates": []}],
                 "started_names": [{"name": "Kris", "player": "Kris Dunn", "game": ""}]}, names=["Luka", "Jalen", "Kris"])
    page, later, empty, plain = draw(card_case(named_card), card_case(tomorrow), card_case(nobody), card_case(bare))
    # prefix tree: only the missing names, as "... isn't in tonight's games, or the name didn't match a player."
    assert notes(page) == NAME_NOTES, notes(page)
    _escaped(page)
    assert row_names(page) == ["Luka Dončić"]
    assert notes(later) == [note.replace("tonight's", "tomorrow's") for note in NAME_NOTES[:2]]
    # Nobody found: the notes say why, with no "No player qualifies yet." under them
    assert notes(empty) == NAME_NOTES and not empty.select("li.ps-item")
    assert "No player qualifies yet." not in empty.text()
    _escaped(empty)
    # No candidates or no game label sent: the same sentences without them
    assert notes(plain) == ["More than one player matches Jalen. Which one?",
                            "Kris Dunn's game has already started, so it isn't in the scan."], notes(plain)


def test_the_finder_has_a_note_for_each_name_it_couldnt_rank_in_every_surfaces_words(draw) -> None:
    desk, phone, nobody = draw(finder_case(FINDER_NAMED, NAMED), finder_case(FINDER_NAMED, NAMED, layout="phone"),
                               finder_case(FINDER_NAMED, {**NAMED, "players": []}))
    for page in (desk.results, phone.results):
        assert notes(page) == NAME_NOTES, notes(page)
        _escaped(page)
    # Nobody to show: the same notes above "Nothing to show for ...", with "Clear the name"
    assert notes(nobody.results) == NAME_NOTES, notes(nobody.results)
    _escaped(nobody.results)
    empty = nobody.results.one("div.empty")
    assert empty.one("h3").text() == "Nothing to show for Luka, Jalen and Draisaitl tonight"
    assert empty.one("button").text() == "Clear the name" and not empty.select("p")


def test_a_refresh_redraws_the_name_notes_from_the_fresh_scan_and_keeps_the_names(draw) -> None:
    named_card = card(NAMED, names=["Luka", "Jalen", "Draisaitl"])
    resolved = scan(ask="3+ threes", stat="fg3m", stat_label="threes", threshold=3, names=["Luka", "Jalen", "Draisaitl"],
                    players=[row(0, threshold=3, qualifies=True), row(8, threshold=3, qualifies=False)])
    started = {**resolved, "started_names": [{"name": "Jalen", "player": "Jalen Brunson", "game": "NYK @ BOS"}]}
    fresh, later = draw(card_case(named_card, refresh=resolved), card_case(named_card, refresh=started))
    # prefix tree: the old ambiguous and started names stay on the card after a Refresh that no longer says them
    assert "ambiguous_names" not in fresh.card and "started_names" not in fresh.card, sorted(fresh.card)
    assert notes(fresh) == [], notes(fresh)
    assert row_names(fresh) == ["Luka Dončić", "Jalen Brunson"]
    assert notes(later) == ["Jalen Brunson's game (NYK @ BOS) has already started, so it isn't in the scan."]
    # Refresh asks for the same players again
    assert "player=Luka&player=Jalen&player=Draisaitl" in fresh.refresh_query, fresh.refresh_query


# --- contract 2 (F2): the trend chip's baseline ------------------------------------------------------------------------

def _trend_rows(sign: int) -> List[Dict[str, Any]]:
    return [
        row(0, trend={"l5": round(24.1 + sign * 4.3, 1), "season": 24.1, "diff": sign * 4.3, "vs": "season"}),
        row(1, trend={"l5": round(15.2 + sign * 14.8, 1), "season": 15.2, "diff": sign * 14.8, "vs": "last_season"}),
        row(2),  # no trend: no chip
    ]


def _chips(page: El) -> List[Tuple[str, str]]:
    """Each trend chip as (on screen, heard)."""
    return [(chip.text(seen=True), chip.text(heard=True)) for chip in page.select("span.ps-trend")]


def test_the_trend_chip_says_whether_the_last_5_met_this_season_or_last_season(draw) -> None:
    hot = card(scan(players=_trend_rows(1), sort="hot"), sort="hot", title="Hot: points, last 5 against the season · NBA · tonight")
    cold = card(scan(players=_trend_rows(-1), sort="cold"), sort="cold", title="Cold: points · NBA · tonight")
    plain = card(scan(players=_trend_rows(1)))
    hot_scan = scan(players=_trend_rows(1), sort="hot")
    hot_page, cold_page, plain_page, desk, phone = draw(
        card_case(hot), card_case(cold), card_case(plain),
        finder_case(FINDER_20 + "&sort=hot", hot_scan), finder_case(FINDER_20 + "&sort=hot", hot_scan, layout="phone"))
    want_hot = [("L5 28.4 vs 24.1 season", "Last 5 games 28.4 on average, 24.1 over the rest of this season: up 4.3"),
                ("L5 30 vs 15.2 last season", "Last 5 games 30 on average, 15.2 last season: up 14.8")]
    # prefix tree: "L5 30 vs 15.2 season" for a baseline that is last season's
    assert _chips(hot_page) == want_hot, _chips(hot_page)
    assert _chips(cold_page) == [
        ("L5 19.8 vs 24.1 season", "Last 5 games 19.8 on average, 24.1 over the rest of this season: down 4.3"),
        ("L5 0.4 vs 15.2 last season", "Last 5 games 0.4 on average, 15.2 last season: down 14.8")], _chips(cold_page)
    # Each chip sits on its own row; the row without a trend has none; the footnote says what the baseline is
    assert [len(item.select("span.ps-trend")) for item in hot_page.select("li.ps-item")] == [1, 1, 0]
    assert any("or against last season's, early in a season" in note for note in hot_page.texts("p.ps-note"))
    # A plain scan never shows the chip; the finder's table and cards show the same chips
    assert _chips(plain_page) == []
    assert _chips(desk.results) == want_hot and _chips(phone.results) == want_hot


# --- contract 3 (M5/F6): games already started ------------------------------------------------------------------------

def test_the_started_notes_read_games_as_the_games_still_to_come(draw) -> None:
    some = card(scan(players=QUALIFIERS[:3], games=4, games_started=6))
    one = card(scan(players=QUALIFIERS[:3], games=9, games_started=1))
    every = card(scan(players=[], games=0, games_started=9))
    some_empty = card(scan(players=[], ask="45+ points", threshold=45, games=4, games_started=6))
    none = card(scan(players=QUALIFIERS[:3], games=10, games_started=0))
    tomorrow = card(scan(players=[], day="tomorrow", games=0, games_started=3), title="Projected for 20+ points · tomorrow")
    pages = draw(*(card_case(c) for c in (some, one, every, some_empty, none, tomorrow)))
    # prefix tree: "6 games have already started: ..." (no day) and "All 9 of tonight's games have started: nobody
    # is left to scan tonight."
    assert notes(pages[0]) == ["6 games tonight have already started: their players aren't listed."], notes(pages[0])
    assert notes(pages[1]) == ["1 game tonight has already started: their players aren't listed."], notes(pages[1])
    assert notes(pages[2]) == ["Every game tonight has started, so nobody is left to rank."], notes(pages[2])
    assert "No player qualifies yet." not in pages[2].text()
    # 6 started and 4 still to come, nobody at 45+: both are said
    assert notes(pages[3]) == ["6 games tonight have already started: their players aren't listed."]
    assert "No player qualifies yet." in pages[3].texts("p.small")
    assert notes(pages[4]) == []
    assert notes(pages[5]) == ["Every game tomorrow has started, so nobody is left to rank."]


def test_the_finder_says_the_started_games_as_every_surface_does(draw) -> None:
    rows, every, some_empty, tomorrow = draw(
        finder_case(FINDER_20, scan(players=QUALIFIERS[:3], games=8, games_started=2)),
        finder_case(FINDER_20, scan(players=[], games=0, games_started=9)),
        finder_case("#/players?league=basketball_nba&stat=pts&at=45&day=today",
                    scan(players=[], ask="45+ points", threshold=45, games=4, games_started=6)),
        finder_case("#/players?league=basketball_nba&stat=pts&at=20&day=tomorrow",
                    scan(players=[], day="tomorrow", games=0, games_started=3)))
    assert notes(rows.results) == ["2 games tonight have already started: their players aren't listed."], notes(rows.results)
    # Every game started: the empty state is the note itself, with tomorrow's games one tap away
    empty = every.results.one("div.empty")
    assert empty.one("h3").text() == "Every game tonight has started, so nobody is left to rank.", empty.text()
    assert empty.one("button").text() == "Tomorrow's games" and notes(every.results) == []
    # 6 started and 4 to come, nobody at 45+: the note sits above "Nobody is projected ..."
    assert notes(some_empty.results) == ["6 games tonight have already started: their players aren't listed."]
    assert some_empty.results.one("div.empty h3").text() == "Nobody is projected for 45+ points yet"
    assert tomorrow.results.one("div.empty h3").text() == "Every game tomorrow has started, so nobody is left to rank."
    assert not tomorrow.results.select("div.empty button")


# --- contract 4 (m1): "N more", the result's own number in every surface's words ---------------------------------------

EDGES = {"mode": "edges", "ask": "posted lines by edge", "stat": None, "stat_label": None, "threshold": None, "side": None,
         "day": "today", "leagues": [NBA], "games": 10, "games_started": 0, "as_of": TIP, "more": 28, "close": [],
         "players": [row(i, stat=stat, stat_label=label, side="over", edge=round(0.12 - i * 0.01, 3))
                     for i, (stat, label) in enumerate([("pra", "PRA"), ("reb", "rebounds"), ("fg3m", "threes"),
                                                        ("pts", "points"), ("ast", "assists"), ("pr", "P+R")])]}


def test_the_cards_more_line_is_the_results_own_number_worded_by_mode(draw) -> None:
    ten = [row(i) for i in range(10)]
    cases = [
        card(scan(players=ten, more=229)),
        card(scan(players=ten, more=229, sort="hot"), sort="hot", title="Hot: points · NBA · tonight"),
        card(scan(players=ten, more=229, sort="cold"), sort="cold", title="Cold: points · NBA · tonight"),
        card(EDGES, sort="edge", title="Best edges on posted lines · NBA · tonight"),
        card(scan(players=ten[:3], more=1)),
        card(scan(players=ten[:3], more=1, sort="hot"), sort="hot"),
        card({**EDGES, "more": 1}, sort="edge"),
        card(scan(players=[row(i) for i in range(25)], more=3)),
        card(scan(players=ten, more=0)),
    ]
    pages = draw(*(card_case(c) for c in cases))
    lines = [page.texts("p.ps-more") for page in pages]
    # prefix tree: "Beyond these 10, 229 more qualify: open the Player finder to see them all."
    assert lines == [
        ["229 more qualify: open the Player finder."],
        ["229 more players are ranked: open the Player finder."],
        ["229 more players are ranked: open the Player finder."],
        ["28 more posted lines have an edge: open the Player finder."],
        ["1 more qualifies: open the Player finder."],
        ["1 more player is ranked: open the Player finder."],
        ["1 more posted line has an edge: open the Player finder."],
        ["3 more qualify: open the Player finder."],  # never counted again from the 25 rows on the card
        [],
    ], lines
    # The line comes before "Open in Player finder", which opens the same scan
    link = pages[1].one("div.ask-card-actions a")
    assert link.text() == "Open in Player finder" and "sort=hot" in link.attrs.get("href", "")


def test_the_finders_more_line_counts_the_same_way(draw) -> None:
    hot = scan(players=[row(i) for i in range(8)], more=180, sort="hot")
    desk, edges = draw(finder_case(FINDER_20 + "&sort=hot", hot), finder_case(FINDER_20 + "&sort=edges", EDGES))
    assert "180 more players are ranked beyond the 8 shown." in desk.results.texts("p.ps-note")
    assert "28 more posted lines have an edge beyond the 6 shown." in edges.results.texts("p.ps-note")


# --- guards: the finder's address and Ask AI's Refresh keep the new state ----------------------------------------------

@pytest.mark.parametrize("hash_, query", [
    ("#/players?league=basketball_nba&stat=pts&at=20&day=today&sort=hot&player=Luka+Don%C4%8Di%C4%87&player=%C4%90or%C4%91evi%C4%87",
     ["stat=pts", "threshold=20", "sort=hot", "player=Luka+Don%C4%8Di%C4%87", "player=%C4%90or%C4%91evi%C4%87"]),
    ("#/players?league=icehockey_nhl&stat=sog&at=3&day=tomorrow&sort=cold", ["league=icehockey_nhl", "stat=sog", "day=tomorrow", "sort=cold"]),
    ("#/players?league=basketball_nba&stat=pts&at=20&day=today&sort=edges&player=Luka", ["sort=edge"]),
    ("#/players?league=basketball_nba&stat=pra&at=line&day=today&sort=chance", ["stat=pra"]),
    ("#/players?league=basketball_nba&stat=reb&at=12&day=today&side=under&sort=projection", ["side=under", "threshold=12"]),
])
def test_the_finders_address_round_trips(draw, hash_, query) -> None:
    (first,) = draw(finder_case(hash_, scan()))
    tidy = first.replaced[-1] if first.replaced else hash_
    (again,) = draw(finder_case(tidy, scan()))
    assert again.replaced == [], f"{tidy} was tidied again to {again.replaced}"
    asked = again.fetched[0]
    assert all(part in asked for part in query), asked
    if "sort=edges" in hash_:
        # Best edges covers everyone: the name goes, and no stat is asked
        assert "player=" not in tidy and "player=" not in asked and "stat=" not in asked
    else:
        assert all(f"player={name}" in tidy for name in re.findall(r"player=([^&]+)", hash_))


def test_ask_ais_refresh_asks_for_the_same_scan(draw) -> None:
    named = card(scan(names=["Luka", "Đorđević"], players=[row(0)], sort="hot"), sort="hot", names=["Luka", "Đorđević"])
    overs = card({**EDGES, "side": "over", "ask": "overs by edge"}, sort="edge", names=["Luka"])
    plain = card(scan(players=[row(0)], side="under", threshold=12, stat="reb", ask="under 11.5 rebounds"))
    pages = draw(card_case(named), card_case(overs), card_case(plain))
    assert pages[0].refresh_query.endswith("&sort=hot&player=Luka&player=%C4%90or%C4%91evi%C4%87"), pages[0].refresh_query
    assert "sort=edge" in pages[1].refresh_query and "side=over" in pages[1].refresh_query
    assert "player=Luka" in pages[1].refresh_query and "stat=" not in pages[1].refresh_query
    assert "stat=reb" in pages[2].refresh_query and "threshold=12" in pages[2].refresh_query and "side=under" in pages[2].refresh_query


# --- V4 (final check): the names the finder sends keep digits, suffixes and parentheses --------------------------------
# The rule the finder (scan.js cleanNames) shares with the bot's finder button (bot/chat.py _finder_names): split on
# commas; in each piece every character that isn't a letter (any script), a digit, a space or one of . ' ’ - ( ) becomes
# a space; spaces collapsed, trimmed, at most 60 characters; a piece that is only a suffix (Jr, Sr, II, III, IV, 2nd,
# 3rd, any case, a period or not) joins the name before it; no empty pieces or repeats (ignoring case); at most 5.

SCAN_JS = "js/views/players/scan.js"
# The route's own pattern for a name (dashboard/app.py ScanName), read from its source; this one when it isn't there
_APP = (STATIC.parent / "app.py").read_text(encoding="utf-8") if (STATIC.parent / "app.py").exists() else ""
_ROUTE = re.search(r'ScanName = Annotated\[str, Field\([^)]*pattern=r"([^"]+)"', _APP)
ROUTE_NAME = re.compile(_ROUTE.group(1) if _ROUTE else r"^[\w .,'’()\-]{1,60}$")

V4_NAMES = [
    ("Marvin Bagley 3rd", ["Marvin Bagley 3rd"]),
    ("Michael Porter, Jr.", ["Michael Porter Jr."]),
    ("Marcus Sutton (MIA)", ["Marcus Sutton (MIA)"]),
    ("Robert Williams, III, Kelly Oubre Jr", ["Robert Williams III", "Kelly Oubre Jr"]),
    ("Tim Hardaway, sr, Gary Payton, ii, Marvin Bagley, 3RD.", ["Tim Hardaway sr", "Gary Payton ii", "Marvin Bagley 3RD."]),
    ("Lonnie Walker, IV, Jaime Jaquez, 2nd", ["Lonnie Walker IV", "Jaime Jaquez 2nd"]),
    (["Michael Porter", "Jr."], ["Michael Porter Jr."]),  # one name per address entry: a lone suffix still joins
    ("Jr., Luka", ["Jr.", "Luka"]),  # no name before it: it stands alone
    ("Michael Porter Jr., Michael Porter, Jr.", ["Michael Porter Jr."]),  # a repeat once joined
    ("steph, Steph, STEPH", ["steph"]),
    ("a, b, c, d, e, f", ["a", "b", "c", "d", "e"]),
    ("Luka_Doncic\tJr", ["Luka Doncic Jr"]),
    ("Dončić, Dončić", ["Dončić"]),  # accents composed (NFC) before anything else
    ("Ðorđević, 李 小龍, Ömer Yurtseven 2", ["Ðorđević", "李 小龍", "Ömer Yurtseven 2"]),
    ("x" * 70, ["x" * 60]),
    ("x" * 57 + ", Jr.", ["x" * 57 + " Jr"]),  # a joined name is cut to 60 again: the route takes no more
    (" , ,  ", []),
]
V4_HOSTILE = [
    ('<img src=x onerror="window.__xss=1">', ["img src x onerror window. xss 1"]),
    ('"><svg/onload=alert(1)>', ["svg onload alert(1)"]),
    ("O'Neal' onfocus=", ["O'Neal' onfocus"]),
    ("<script>alert('x')</script>", ["script alert('x') script"]),
    ("javascript:alert(1)//", ["javascript alert(1)"]),
]


def clean_case(value: Any) -> Dict[str, Any]:
    return {"kind": "call", "module": SCAN_JS, "fn": "cleanNames", "args": [value]}


def test_v4_names_keep_digits_suffixes_and_parentheses_by_the_shared_rule(draw) -> None:
    got = draw(*(clean_case(value) for value, _ in V4_NAMES))
    # Before: "Marvin Bagley rd", ["Michael Porter", "Jr."] and "Marcus Sutton MIA"
    assert got == [want for _, want in V4_NAMES], [(value, out) for (value, _), out in zip(V4_NAMES, got)]
    # Whatever it keeps, the route takes; and cleaning what it gives changes nothing
    assert all(ROUTE_NAME.match(name) for names in got for name in names), got
    again = draw(*(clean_case(names) for names in got))
    assert again == got


def test_v4_the_finder_sends_keeps_and_shows_such_names(draw) -> None:
    head = "#/players?league=basketball_nba&stat=pts&at=20&day=today"
    names = "&player=Marvin+Bagley+3rd&player=Michael+Porter+Jr.&player=Marcus+Sutton+%28MIA%29"
    (first,) = draw(finder_case(f"{head}&player=Marvin+Bagley+3rd&player=Michael+Porter%2C+Jr.&player=Marcus+Sutton+%28MIA%29", scan()))
    # "Michael Porter, Jr." in one entry of the address is one player
    assert first.replaced == [head + names], first.replaced
    (again,) = draw(finder_case(head + names, scan()))
    assert again.replaced == [], again.replaced  # the address round-trips
    assert again.fetched[0].endswith(names), again.fetched[0]  # and the route is asked for those three
    assert again.controls.one("#pf-player").attrs.get("value") == "Marvin Bagley 3rd, Michael Porter Jr., Marcus Sutton (MIA)"
    assert "Marvin Bagley 3rd, Michael Porter Jr. and Marcus Sutton (MIA)" in again.summary.text()


def test_v4_hostile_names_are_still_dropped_or_escaped(draw) -> None:
    got = draw(*(clean_case(value) for value, _ in V4_HOSTILE))
    # Before: digits and parentheses went too ("svg onload alert"); markup never survives either way
    assert got == [want for _, want in V4_HOSTILE], got
    hostile = [value for value, _ in V4_HOSTILE[:2]]
    hash_ = "#/players?league=basketball_nba&stat=pts&at=20&day=today" + "".join(f"&player={quote(v)}" for v in hostile)
    cleaned = [V4_HOSTILE[0][1][0], V4_HOSTILE[1][1][0]]
    (page,) = draw(finder_case(hash_, scan(players=[], names=cleaned, missing_names=cleaned)))
    # The address and the scan carry the cleaned names only
    assert page.replaced[-1].endswith("&player=img+src+x+onerror+window.+xss+1&player=svg+onload+alert%281%29"), page.replaced
    assert page.fetched[0].endswith("&player=img+src+x+onerror+window.+xss+1&player=svg+onload+alert%281%29"), page.fetched
    # Shown as text: the box, the summary, the notes; no element carries an event handler or comes from a name
    assert page.controls.one("#pf-player").attrs.get("value") == ", ".join(cleaned)
    assert notes(page.results) == [f"I couldn't find {name} in tonight's games." for name in cleaned]
    for box in (page.controls, page.summary, page.results):
        assert not box.select("img, script, iframe"), box.html[:300]
        assert not [el.tag for el in box.walk() if any(attr.startswith("on") for attr in el.attrs)]


def test_v4_ask_ais_refresh_and_finder_link_keep_such_names(draw) -> None:
    names = ["Marvin Bagley 3rd", "Michael Porter Jr.", "Marcus Sutton (MIA)"]
    (page,) = draw(card_case(card(scan(names=names, players=[row(0, name="Marvin Bagley III", qualifies=True)]), names=names)))
    query = "&player=Marvin+Bagley+3rd&player=Michael+Porter+Jr.&player=Marcus+Sutton+%28MIA%29"
    # Before: "player=Marvin+Bagley+rd" and "Marcus+Sutton+MIA"
    assert page.refresh_query.endswith(query), page.refresh_query
    assert page.one("div.ask-card-actions a").attrs.get("href", "").endswith(query)


# --- small fixes 1: a row whose edge the server withheld says why, wherever its edge is drawn ---------------------------

EDGE_HIDDEN = "Edge hidden: over +50% usually means a stale or wrong price."


def suspect(i: int, **over: Any) -> Dict[str, Any]:
    """A row whose edge the server withheld (engine/scan.py: over EDGE_CAP, ``edge_suspect`` and no ``edge``)."""
    return {**{key: value for key, value in row(i, **over).items() if key != "edge"}, "edge_suspect": True}


def test_small_a_withheld_edge_says_why_on_its_row_and_a_normal_row_doesnt(draw) -> None:
    long_price = milestone(25, (2.79, "+179"))
    by_edge = scan(**{**POSTED, "players": POSTED["players"] + [suspect(3, threshold=25, chance=0.64, line=long_price)]})
    at_20 = scan(players=[row(0), suspect(1, chance=0.64, line=long_price), row(2)])
    posted_card = card(by_edge, title="Strongest PRA overs (posted lines) · NBA · tonight")
    page, phone, desk, page_20, desk_20 = draw(
        card_case(posted_card), finder_case(FINDER_PRA_LINE, by_edge, layout="phone"),
        finder_case(FINDER_PRA_LINE, by_edge), card_case(card(at_20)), finder_case(FINDER_20, at_20))
    for where, items, hidden in (("card, by edge", page.select("li.ps-item"), NAMES[3]),
                                 ("finder (phone), by edge", phone.results.select("li.ps-item"), NAMES[3]),
                                 ("finder (desk), by edge", table_rows(table_bodies(desk.results)[0]), NAMES[3]),
                                 ("card, 20+", page_20.select("li.ps-item"), NAMES[1]),
                                 ("finder (desk), 20+", table_rows(table_bodies(desk_20.results)[0]), NAMES[1])):
        names = [item.select(".ps-name, .pf-name")[0].text() for item in items]
        assert len(names) >= 3 and hidden in names, (where, names)
        # Once, on the withheld row's posted line, in the page's own muted style; never on a row with its edge
        for name, item in zip(names, items):
            assert item.text().count(EDGE_HIDDEN) == (name == hidden), (where, name, item.text())
            assert name == hidden or "Edge hidden" not in item.text(), (where, name)
        line = items[names.index(hidden)].select(".ps-line, .pf-line")[0]
        assert [el.classes for el in line.walk() if el.text() == EDGE_HIDDEN] == [{"muted"}], (where, line.text())
    # The rows with an edge still show it as before (m6), and the withheld one's Edge says nothing
    assert [tr.one("td.pf-e").text() for tr in table_rows(table_bodies(desk.results)[0])] == \
        ["+8.1%", "+2.3%", "−1.0%", "–"]
    assert [dict(numbers(item))["Edge"] for item in page.select("li.ps-item")] == ["+8.1%", "+2.3%", "−1.0%", "–"]


# --- small fixes 2: "Last, First" names in the Player finder, the web's and the bot's one rule --------------------------

LAST_FIRST = [
    ("Porter Jr., Michael", ["Michael Porter Jr."]),
    ("James, LeBron", ["James", "LeBron"]),  # one word after the comma: the scan reads it on the board (match_names)
    ("LeBron James, Stephen Curry", ["LeBron James", "Stephen Curry"]),
    ("Curry", ["Curry"]),
    ("Michael", ["Michael"]),
    # Only a last name and its suffix, then one word: anything else splits as before
    ("Porter, Jr., Michael", ["Michael Porter Jr."]),
    ("bagley iii, marvin", ["marvin bagley iii"]),
    ("Michael Porter Jr., Curry", ["Michael Porter Jr.", "Curry"]),
    ("Porter Jr., Michael Smith", ["Porter Jr.", "Michael Smith"]),
    ("y" * 56 + " Jr., Bob", ["Bob " + "y" * 56]),  # cut to 60 and trimmed, as a joined suffix is
]
# What the scan finds for those names on a board (the brief's "Result"): the same on both sides, since the names are
LAST_FIRST_BOARD = ["LeBron James", "Bronny James", "Stephen Curry", "Seth Curry", "Michael Porter Jr.",
                    "Kevin Porter Jr.", "Michael Smith"]
LAST_FIRST_FOUND = [
    {"Michael Porter Jr.": ["Michael Porter Jr."]},
    {"LeBron James": ["LeBron James"]},  # not "James" (two players) and "LeBron"
    {"LeBron James": ["LeBron James"], "Stephen Curry": ["Stephen Curry"]},
    {"Curry": ["Stephen Curry", "Seth Curry"]},  # as before: ambiguous
    {"Michael": ["Michael Porter Jr.", "Michael Smith"]},  # as before: ambiguous
]


def test_small_last_first_names_follow_one_rule_on_the_web_and_in_the_bot(draw) -> None:
    cases = LAST_FIRST + V4_NAMES + V4_HOSTILE
    web = draw(*(clean_case(value) for value, _ in cases))
    # Parity: the finder's cleanNames and the bot's finder button (_finder_names), character for character
    assert web == [bot_chat._finder_names(value if isinstance(value, list) else [value]) for value, _ in cases]
    assert web == [want for _, want in cases], [(value, out) for (value, _), out in zip(cases, web)]
    assert draw(*(clean_case(names) for names in web)) == web  # and cleaning what it gives changes nothing
    board = [{"athlete_id": str(i), "name": name} for i, name in enumerate(LAST_FIRST_BOARD)]
    for names, want in zip(web, LAST_FIRST_FOUND):
        found, missing = engine.match_names(board, names)
        assert {asked: [LAST_FIRST_BOARD[int(i)] for i in ids] for asked, ids in found.items()} == want and not missing


def test_small_the_finder_sends_last_first_names_as_one(draw) -> None:
    head = "#/players?league=basketball_nba&stat=pts&at=20&day=today"
    (page,) = draw(finder_case(f"{head}&player={quote('Porter Jr., Michael')}&player={quote('LeBron James, Stephen Curry')}",
                               scan()))
    names = "&player=Michael+Porter+Jr.&player=LeBron+James&player=Stephen+Curry"
    assert page.replaced == [head + names], page.replaced  # "Porter Jr., Michael" in the address is one player
    (again,) = draw(finder_case(head + names, scan()))
    assert again.replaced == [] and again.fetched[0].endswith(names), again.fetched
    assert again.controls.one("#pf-player").attrs.get("value") == "Michael Porter Jr., LeBron James, Stephen Curry"
