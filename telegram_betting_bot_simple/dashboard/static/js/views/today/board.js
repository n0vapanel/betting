// The board: a dense, sortable table on tablets and desktops (840 px and up) and cards on phones, from the
// same filtered games. One delegated click handler on the board's root does rows, folds, sorting and
// "open this league"; price buttons are left to the shell's own delegated handler (ui.js), and links to
// games to the shell's inspector handler (layout.js), so a click is handled exactly once.
//
//   boardParts({mode, games, prepared, state, ai, asked, pinned, inspected, canPrefetch, cols})
//                             -> {table, ai, head, items: [{key, html}], body}: the rows (and separators) in order,
//                             each with a key ("g:<game id>" or "s:<section>"), and the selector of the element
//                             they go in (".tb-body" or ".td-cards"). A row's HTML is written when first read
//                             (a long board's first paint writes only its first screenful)
//   frameHtml(parts, inner)   the table (or the card list) around ``inner`` (some of the items' HTML)
//   patchItems(body, items, drawn) -> {html, parsed}: bring a drawn board up to date in place (a refresh;
//                             ``drawn``: Map key -> the HTML each row was drawn with): unchanged rows keep
//                             their nodes, so the page never shrinks, the reader keeps their place and focus
//                             stays put; only changed rows are parsed again
//   keyOf(node)               an item's key from its node (data-game, data-detail or data-sep)
//   bindBoard(root, {onSort, onOpenLeague, onToggle, onOpenDetail}) -> unbind
//   syncHeader(root)          keep the sticky header's sideways scroll with the rows' (call after drawing)
//
// Table semantics: role="table" with rowgroups, rows, column headers (aria-sort on the sorted one) and a
// row header (the time) per row; a separator's or a fold-out's one cell spans the ``cols`` columns shown
// (aria-colspan; Today keeps it right as the board's width hides or shows columns). The first column stays
// put when the table scrolls sideways; rows use content-visibility so a long all-sports day draws only what
// is on screen.

import { esc, ago } from "../../ui.js";
import { MARKETS, rowHtml, detailHash } from "./row.js";
import { cardHtml } from "./card.js";

const HEAD = [
  { cls: "tb-when", label: "Time", sort: "time" },
  { cls: "tb-lg", label: "League", sort: "league" },
  { cls: "tb-match", label: "Away @ Home" },
  ...MARKETS.map((m) => ({ cls: `tb-${m.key}`, label: m.label })),
  { cls: "tb-mkt", label: "Market", short: "Mkt", sort: "market", title: "Market chance: the bookmakers' chance, margin removed (sorts by the favourite's)" },
  { cls: "tb-ai", label: "AI", sort: "ai", ai: true, title: "The AI's chance (NBA for now): sort by the AI's favourite" },
  { cls: "tb-edge", label: "Edge", sort: "gap", ai: true, title: "AI minus market, in points: sort by the biggest gap" },
  { cls: "tb-badges", label: "Notes" },
  { cls: "tb-asof", label: "Odds as of" },
];

function headHtml(sort, ai) {
  return `<div class="tb-head" role="rowgroup"><div class="tb-hrow" role="row">${HEAD.filter((h) => ai || !h.ai).map((h) => {
    const sorted = h.sort && sort.key === h.sort;
    const dir = sorted ? (sort.dir === -1 ? (h.sort === "time" || h.sort === "league" ? "descending" : "ascending") : (h.sort === "time" || h.sort === "league" ? "ascending" : "descending")) : "none";
    const text = h.short ? `<span class="h-full">${esc(h.label)}</span><span class="h-short" aria-hidden="true">${esc(h.short)}</span>` : esc(h.label);
    const inner = h.sort
      ? `<button type="button" class="tb-sort" data-sort="${h.sort}"${h.title ? ` title="${esc(h.title)}"` : ""}${h.short ? ` aria-label="${esc(h.label)}"` : ""}>${text}<span class="tb-arrow" aria-hidden="true">${sorted ? (dir === "ascending" ? "▲" : "▼") : ""}</span></button>`
      : `<span${h.title ? ` title="${esc(h.title)}"` : ""}>${esc(h.label)}</span>`;
    return `<div class="tb-c ${h.cls}" role="columnheader"${h.sort ? ` aria-sort="${dir}"` : ""}>${inner}</div>`;
  }).join("")}</div></div>`;
}

const DAY = new Intl.DateTimeFormat(undefined, { weekday: "long", month: "short", day: "numeric" });

/** Separators: the person's pinned leagues first, a day per heading in a search, a league per heading. */
function sectionOf(game, state, pinned, prepared, multiDay) {
  if (state.sort.key === "league") return prepared.leagues.get(game.league)?.title || game.league;
  if (state.sort.key !== "time") return null;
  const parts = [];
  if (pinned.length) parts.push(pinned.includes(game.league) ? "Your sports" : "More sports");
  if (multiDay) parts.push(DAY.format(new Date(game._start)));
  return parts.join(" · ") || null;
}

/** Template whitespace between tags (newlines and indentation) is dropped: a long board is ~25% lighter. */
const tight = (html) => html.replace(/\n\s*/g, "");

/**
 * The board's parts (see the header). Separators: the person's pinned leagues first, a day per heading in a
 * search, a league per heading when sorted by league.
 */
export function boardParts({ mode, games, prepared, state, ai, asked, pinned, inspected, canPrefetch, cols = ai ? 11 : 9 }) {
  const multiDay = state.search.trim().length > 0;
  const table = mode !== "phone";
  const items = [];
  const seen = new Set();
  let last = null;
  for (const game of games) {
    const section = sectionOf(game, state, pinned, prepared, multiDay);
    if (section && section !== last) {
      let key = `s:${section}`;
      for (let n = 2; seen.has(key); n += 1) key = `s:${section} ${n}`;
      seen.add(key);
      const sep = esc(key.slice(2));
      items.push({
        key,
        html: table
          ? `<div class="tb-sep" role="row" data-sep="${sep}"><div class="tb-sep-c" role="cell" aria-colspan="${esc(cols)}"><span>${esc(section)}</span></div></div>`
          : `<h2 class="day-head td-sep" data-sep="${sep}">${esc(section)}</h2>`,
      });
      last = section;
    }
    const entry = prepared.leagues.get(game.league);
    const options = {
      entry, ai, asked, withDay: multiDay, open: inspected && detailHash(game) === inspected,
      prefetchable: canPrefetch(game, entry), expanded: state.expanded.has(game.id), showLeague: true, cols,
    };
    let html = null;
    items.push({ key: `g:${game.id}`, get html() { return (html ??= tight(table ? rowHtml(game, options) : cardHtml(game, options))); } });
  }
  return { table, ai, head: table ? tight(headHtml(state.sort, ai)) : "", items, body: table ? ".tb-body" : ".td-cards" };
}

/** The table (or the phone's card list) around ``inner``: some or all of the items' HTML. */
export function frameHtml({ table, ai, head }, inner = "") {
  return table
    ? `<div class="tb${ai ? " has-ai" : ""}" role="table" aria-label="Games"><div class="tb-headwrap">${head}</div><div class="tb-bodywrap"><div class="tb-body" role="rowgroup">${inner}</div></div></div>`
    : `<div class="td-cards stack">${inner}</div>`;
}

/** An item's key from its node: a game's row or card, a game's fold-out row, or a separator. */
export function keyOf(node) {
  const d = node.dataset;
  if (!d) return null;
  if (d.game != null) return `g:${d.game}`;
  if (d.detail != null) return `g:${d.detail}`;
  if (d.sep != null) return `s:${d.sep}`;
  return null;
}

// "12 min ago" moves on by itself: a row whose HTML differs only there keeps its node (its text is updated)
const STAMP = /(<time class="td-asof"[^>]*>)[^<]*<\/time>/g;
const unstamped = (html) => html.replace(STAMP, "$1</time>");

function byKey(nodes) {
  const out = new Map();
  for (const node of nodes) {
    const key = keyOf(node);
    if (!key) continue;
    const list = out.get(key);
    if (list) list.push(node);
    else out.set(key, [node]);
  }
  return out;
}

/**
 * Bring the drawn ``body`` (".tb-body" / ".td-cards") up to date with ``items`` in place. ``drawn``: Map
 * key -> the HTML each key was drawn with. Rows whose HTML is the same keep their nodes; changed and new
 * ones are parsed in one go; rows no longer shown go; the order follows ``items`` (moving nodes only when it
 * changed). Returns {html: Map key -> html now drawn, parsed: how many items were drawn anew}.
 */
export function patchItems(body, items, drawn) {
  const nodes = byKey(body.children);
  const kept = new Set();
  const plan = [];
  const html = new Map();
  for (const item of items) {
    let found = nodes.get(item.key) || null;
    const before = drawn.get(item.key);
    if (found && before !== item.html) {
      if (before != null && unstamped(before) === unstamped(item.html)) {
        for (const node of found) for (const t of node.querySelectorAll("time.td-asof")) t.textContent = ago(t.getAttribute("datetime"));
      } else {
        found = null;
      }
    }
    if (found) for (const node of found) kept.add(node);
    plan.push({ item, nodes: found });
    html.set(item.key, item.html);
  }
  for (const node of [...body.children]) if (!kept.has(node)) node.remove();
  const fresh = plan.filter((p) => !p.nodes);
  if (fresh.length) {
    const template = document.createElement("template");
    template.innerHTML = fresh.map((p) => p.item.html).join("");
    const parsed = byKey(template.content.children);
    for (const p of fresh) p.nodes = parsed.get(p.item.key) || [];
  }
  let cursor = body.firstElementChild;
  for (const p of plan) {
    for (const node of p.nodes) {
      if (node === cursor) cursor = cursor.nextElementSibling;
      else body.insertBefore(node, cursor);
    }
  }
  return { html, parsed: fresh.length };
}

/** The sticky header follows the rows when the table scrolls sideways. */
export function syncHeader(root) {
  const body = root.querySelector(".tb-bodywrap");
  const head = root.querySelector(".tb-headwrap");
  if (!body || !head) return;
  // Read the rows' scroll after the frame is laid out, not in the middle of drawing it
  requestAnimationFrame(() => { if (body.scrollLeft) head.scrollLeft = body.scrollLeft; });
  if (body.dataset.synced) return;
  body.dataset.synced = "1";
  body.addEventListener("scroll", () => { head.scrollLeft = body.scrollLeft; }, { passive: true });
}

/** One delegated click handler for the board (see the header). Returns a function that removes it. */
export function bindBoard(root, { onSort, onOpenLeague, onToggle, onOpenDetail }) {
  const click = (event) => {
    const t = event.target;
    if (!(t instanceof Element)) return;
    // Prices (ui.js), ⓘ (ui.js) and links (layout.js / the router) handle themselves
    if (t.closest("[data-leg-key], [data-explain], a[href]")) return;
    const sort = t.closest("[data-sort]");
    if (sort) {
      onSort(sort.dataset.sort);
      return;
    }
    const league = t.closest("[data-open-league]");
    if (league) {
      onOpenLeague(league.dataset.openLeague);
      return;
    }
    const expand = t.closest("[data-expand]");
    if (expand) {
      onToggle(expand.dataset.expand);
      return;
    }
    if (t.closest("button, input, select, label, .odds")) return;
    const row = t.closest(".tb-row");
    if (!row) return;
    if (row.dataset.route) onOpenDetail(row.dataset.route, row);
    else onToggle(row.dataset.game);
  };
  root.addEventListener("click", click);
  return () => root.removeEventListener("click", click);
}
