"""Help in plain English: "it's for the public and they should be able to easily understand everything".

Help's home page is fourteen short sections in a fixed order (dashboard/static/js/views/help/content.js SECTIONS),
each with an example and ending with the same line (CLOSER), folded so the page isn't one wall of text. A search
finds each section by its heading and the brief's keywords; no promise word appears anywhere in Help; the page fits a
360 px phone in the light and the dark theme; and the bot's /help is short and points to the app's Help.

The page checks draw the real screen in Chromium (Playwright for Node, as tests/test_qa_fixes.py does) and skip when
Node, Playwright or Chromium is missing.
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
from telegram.constants import ChatType

from telegram_betting_bot_simple.bot import handlers
from tests import test_capture_health

STATIC = test_capture_health.STATIC
VIEWS = STATIC / "js" / "views"
CONTENT = (VIEWS / "help" / "content.js").read_text(encoding="utf-8")
PAGE = (VIEWS / "help.js").read_text(encoding="utf-8")
UI = (STATIC / "js" / "ui.js").read_text(encoding="utf-8")

HEADINGS = ["Getting started", "Reading odds and chances", "Today", "A game's page", "Player finder", "The Builder",
            "Your slip and your bets", "Ask AI", "The Brain page", "Make your call", "Safety tools", "Data and privacy",
            "Glossary", "FAQ"]
CLOSER = "Predictions are estimates and can be wrong. Only bet what you can afford to lose."
# The section a search finds first, for the brief's keywords
FINDS = {"edge": "odds", "push": "slip", "Brier": "brain", "break": "safety", "Last, First": "players"}
# Words that promise a win, which Help never uses. The FAQ's question "Is this a sure thing?" is the one place "sure"
# appears, and its answer is no.
BANNED = re.compile(r"\bguarantee\w*|\bsure\b|\blocks?\b|\bcan(?:'|’)?t lose\b|\bcannot lose\b"
                    r"|\bbeat(?:ing)? the bookies\b|\brisk[- ]?free\b", re.I)
ALLOWED = "Is this a sure thing?"


def _block(source: str, opener: str) -> str:
    """From ``opener`` to the first line that closes it ("];" or "};" at the start of a line)."""
    start = source.index(opener)
    end = re.compile(r"^[\]}];$", re.M).search(source, start).end()
    return source[start:end]


def _without_comments(source: str) -> str:
    """JavaScript with its // comments out (a URL's "https://" has no space before the slashes, so it stays)."""
    return re.sub(r"(?m)^\s*//.*$|\s//\s.*$", "", source)


def sections() -> List[Dict[str, str]]:
    block = _block(CONTENT, "export const SECTIONS = [")
    found = re.findall(r'^    key: "([a-z]+)", title: ("(?:[^"\\]|\\.)*")', block, re.M)
    return [{"key": key, "title": json.loads(title)} for key, title in found]


# --- the words -------------------------------------------------------------------------------------------------------

def test_the_sections_are_the_fourteen_asked_for_in_order_each_with_an_example_and_links() -> None:
    found = sections()
    assert [s["title"] for s in found] == HEADINGS
    assert len({s["key"] for s in found}) == len(HEADINGS)
    block = _block(CONTENT, "export const SECTIONS = [")
    for field in ("blurb:", "words:", "parts:", "example:", "links:"):
        assert block.count(field) == len(HEADINGS), field
    assert f'export const CLOSER = "{CLOSER}";' in CONTENT
    assert "CLOSER" in PAGE and "SECTIONS" in PAGE  # help.js draws them (the browser checks below see the result)


def test_every_help_link_in_the_sections_opens_a_real_entry_or_guide() -> None:
    block = _block(CONTENT, "export const SECTIONS = [")
    topics = set(re.findall(r'\{ key: "([a-z_]+)", title: ', _block(CONTENT, "export const TOPICS = [")))
    glossary = set(re.findall(r"^  ([a-z_0-9]+): \{", _block(UI, "export const GLOSSARY = {"), re.M))
    new_terms = set(re.findall(r"^  ([a-z_0-9]+): \{", _block(CONTENT, "const NEW_TERMS = {"), re.M))
    links = re.findall(r'"#/help/([a-z0-9_-]+)"', block)
    assert links and topics and glossary and new_terms
    assert set(links) <= topics | glossary | new_terms | {"glossary"}, set(links) - topics - glossary - new_terms


def test_no_word_in_help_promises_a_win() -> None:
    styles, sheet = ((VIEWS / "help" / name).read_text(encoding="utf-8") for name in ("styles.js", "sheet.js"))
    text = "\n".join(_without_comments(source) for source in (CONTENT, PAGE, styles, sheet,
                                                             _block(UI, "export const GLOSSARY = {")))
    assert text.count(ALLOWED) == 1  # the FAQ's question, once
    said = [match.group(0) for match in BANNED.finditer(text.replace(ALLOWED, ""))]
    assert said == [], said
    assert not BANNED.search(handlers.HELP_TEXT)


# --- the bot's /help -------------------------------------------------------------------------------------------------

class _Message:
    def __init__(self) -> None:
        self.sent: List[Any] = []

    async def reply_text(self, text: str, **options: Any) -> None:
        self.sent.append((text, options))


def _help(url: Any, chat: str) -> Any:
    update = SimpleNamespace(message=_Message(), effective_chat=SimpleNamespace(type=chat))
    config = SimpleNamespace(mini_app_url=url)
    context = SimpleNamespace(application=SimpleNamespace(bot_data={"config": config}))
    asyncio.run(handlers.help_command(update, context))
    [(text, options)] = update.message.sent
    return text, options.get("reply_markup")


def test_the_bot_s_help_is_short_and_points_to_the_app_s_help() -> None:
    text = handlers.HELP_TEXT
    assert len(text) <= 4096
    assert "Help" in text.split("/help", 1)[1] and "📱 Open app" in text  # where the full Help is
    for command in ("/start", "/slip", "/brain", "/help"):
        assert command in text
    assert CLOSER in text and "1-800-GAMBLER" in text and "not a bookmaker" in text
    sent, markup = _help("https://app.example", ChatType.PRIVATE)
    assert sent == text
    [[button]] = markup.inline_keyboard  # the app opens from the message itself
    assert button.text == "📱 Open app" and button.web_app.url == "https://app.example"
    assert _help(None, ChatType.PRIVATE)[1] is None  # no Mini App set up: the words alone
    assert _help("https://app.example", ChatType.GROUP)[1] is None  # Telegram refuses Mini App buttons in groups


# --- the page in Chromium --------------------------------------------------------------------------------------------

BROWSER = r"""
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const finish = (out) => process.stdout.write(JSON.stringify(out), () => process.exit(0));
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json" };
const STUB = `window.__appStub = { get: (name) => ({
  me: () => ({ ai_enabled: true, age_confirmed: true, name: "" }), priced: () => ({}),
})[name] || (() => undefined) };`;
const page = (spec) => `<!doctype html><html lang="en" data-theme="${spec.theme}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/styles.css"></head>
<body><div id="desk" class="desk"><div class="shell"><main id="view" class="main"><div id="box" class="view view-help"></div></main></div></div>
<script>${STUB}</script>
<script type="module">
try {
  const { render } = await import("/js/views/help.js");
  await render(document.getElementById("box"), { query: new URLSearchParams(${JSON.stringify(spec.query)}) });
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  window.__done = true;
} catch (error) { window.__failed = String(error && error.stack || error); }
</script></body></html>`;

function measure(kind) {
  const box = document.getElementById("box");
  if (kind === "page") for (const fold of box.querySelectorAll("details")) fold.open = true;
  const wide = [];
  for (const el of box.querySelectorAll("*")) {
    const over = el.scrollWidth > el.clientWidth + 1 && el.clientWidth && getComputedStyle(el).overflowX === "visible";
    if (over || el.getBoundingClientRect().right > innerWidth + 0.5) wide.push(`${el.tagName}.${el.className}: ${(el.textContent || "").trim().slice(0, 40)}`);
  }
  const secs = [...box.querySelectorAll(".hp-sec")];
  const brain = box.querySelector("#hp-s-brain");
  return {
    scroll: document.documentElement.scrollWidth, wide,
    headings: secs.map((d) => d.querySelector("summary h2").textContent.trim()),
    closers: secs.map((d) => d.querySelector(".hp-sec-body").lastElementChild.textContent.trim()),
    examples: secs.map((d) => d.querySelectorAll(".example").length),
    toc: box.querySelectorAll(".hp-toc [data-sec-link]").length,
    open: secs.filter((d) => d.open).map((d) => d.dataset.sec),
    found: [...box.querySelectorAll(".hp-found-secs [data-sec-link]")].map((a) => a.dataset.secLink),
    brainTop: brain ? brain.getBoundingClientRect().top : null,
    background: getComputedStyle(document.body).backgroundColor,
  };
}

(async () => {
  let browser;
  try {
    browser = await chromium.launch();
  } catch (error) {
    finish({ skip: `Chromium couldn't start: ${String(error).split("\n")[0]}` });
    return;
  }
  const results = [];
  for (const spec of input.cases) {
    const p = await browser.newPage({ viewport: { width: spec.width, height: 800 }, deviceScaleFactor: 1, colorScheme: spec.theme });
    const problems = [];
    p.on("pageerror", (error) => problems.push(String(error)));
    await p.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.hostname !== "test.local") return route.fulfill({ status: 404, body: "" });
      if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: page(spec) });
      if (url.pathname === "/js/app.js") return route.fulfill({ contentType: "text/javascript", body: input.appStub });
      const file = path.join(input.static, decodeURIComponent(url.pathname));
      if (!file.startsWith(input.static) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "" });
      return route.fulfill({ contentType: TYPES[path.extname(file)] || "application/octet-stream", body: fs.readFileSync(file) });
    });
    await p.goto("http://test.local/");
    await p.waitForFunction(() => window.__done === true || typeof window.__failed === "string", null, { timeout: 30000 });
    const failed = await p.evaluate(() => window.__failed || null);
    const out = failed ? { failed } : await p.evaluate(measure, spec.kind);
    results.push({ ...spec, problems, ...out });
    await p.close();
  }
  await browser.close();
  finish({ results });
})().catch((error) => finish({ error: String(error && error.stack || error) }));
"""


@pytest.fixture(scope="module")
def drawn() -> List[Dict[str, Any]]:
    joined = test_capture_health.node_path()
    if not joined:
        pytest.skip("Playwright for Node isn't installed: these checks need Chromium")
    cases = [{"kind": "page", "theme": theme, "width": 360, "query": ""} for theme in ("light", "dark")]
    cases += [{"kind": "search", "theme": "light", "width": 360, "query": f"q={words}", "words": words}
              for words in [*FINDS, *HEADINGS]]
    cases.append({"kind": "open", "theme": "dark", "width": 360, "query": "s=brain"})
    data = json.dumps({"static": str(STATIC), "appStub": test_capture_health.app_stub(), "cases": cases})
    done = subprocess.run([test_capture_health.NODE, "-e", BROWSER], input=data, capture_output=True, text=True,
                          encoding="utf-8", timeout=300, env={**os.environ, "NODE_PATH": joined, "NODE_OPTIONS": ""})
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout)
    if out.get("skip"):
        pytest.skip(out["skip"])
    assert "error" not in out, out.get("error")
    for result in out["results"]:
        assert not result.get("failed") and result["problems"] == [], result
    return out["results"]


def _cases(drawn: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [case for case in drawn if case["kind"] == kind]


def test_every_section_is_on_the_page_in_order_with_its_example_and_the_same_last_line(drawn) -> None:
    for case in _cases(drawn, "page"):
        assert case["headings"] == HEADINGS
        assert case["closers"] == [CLOSER] * len(HEADINGS)  # each section ends with the same short line
        assert case["examples"] == [1] * len(HEADINGS)
        assert case["toc"] == len(HEADINGS)  # "On this page" links each one


def test_the_page_fits_a_360_px_phone_in_the_light_and_the_dark_theme_with_every_section_open(drawn) -> None:
    pages = _cases(drawn, "page")
    assert sorted(case["theme"] for case in pages) == ["dark", "light"]
    assert pages[0]["background"] != pages[1]["background"]  # the two themes really are different
    for case in pages + _cases(drawn, "search"):
        assert case["scroll"] <= case["width"] and case["wide"] == [], (case["theme"], case["query"], case["wide"])


def test_a_search_finds_the_right_section_first(drawn) -> None:
    searches = {case["words"]: case["found"] for case in _cases(drawn, "search")}
    for words, key in FINDS.items():
        assert searches[words][:1] == [key], (words, searches[words])
    for heading, section in zip(HEADINGS, sections()):  # every heading is searchable
        assert searches[heading][:1] == [section["key"]], (heading, searches[heading])


def test_a_link_to_one_section_opens_it_and_scrolls_to_it(drawn) -> None:
    [case] = _cases(drawn, "open")
    assert case["open"] == ["brain"]  # the others stay folded
    assert 0 <= case["brainTop"] <= 120, case["brainTop"]  # scrolled to it (72 px stays clear for the header)


def test_the_browser_sign_in_speaks_to_the_public_not_the_owner() -> None:
    """The sign-in screen is for anyone with access now: no "owner and friends", the 18+ line and the helpline."""
    from pathlib import Path
    app = (Path(__file__).resolve().parent.parent / "telegram_betting_bot_simple" / "dashboard" / "static" / "js" /
           "app.js").read_text(encoding="utf-8")
    assert "owner and friends" not in app and "from your server setup" not in app
    assert "for adults (18+)" in app and "Predictions are estimates and can be wrong" in app
    assert "1-800-GAMBLER" in app and 'id="sign-in"' in app and 'name="password"' in app
