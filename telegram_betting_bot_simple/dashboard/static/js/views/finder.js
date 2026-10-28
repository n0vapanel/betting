// Player finder (#/players): today's or tomorrow's players ranked for a stat and a number, from the player scan
// (GET /api/players/scan; js/views/players/scan.js has the row pieces Ask AI's card shares).
//
// Reached from Build ("Find players"), the player search ("Find a player" on Today, a game's player search) and
// Ask AI's player_scan card ("Open in Player finder"). The controls: league, stat (the scan's stats with friendly
// names), "At least" a number (or "Fewer than", an under; or each player's posted line), today or tomorrow, and the
// sort (chance, projection, L10 hits, edge). The rows: a dense table with sortable headers on wider screens, cards
// on phones; "Also close" stays collapsed until opened.
//
// The address carries the scan, so a link opens the same one:
//   #/players?league=basketball_nba&stat=pts&at=20&day=today[&side=under][&sort=projection]   (at=line: posted lines)
// A board still loading answers with what's ready ("8 of 10 games loaded"): the page asks again every few seconds
// until it's complete, and Refresh asks at once. A scan is code only: no AI allowance and no Odds API requests.
// Every number shown is the server's; text from outside is escaped.

import { api, isQuiet } from "../api.js";
import { setTitle, replaceAddress, layout, onLayoutChange } from "../app.js";
import { esc, segmented, chip, notice, skeleton, emptyState, errorState, ago, explain, timeLabel } from "../ui.js";
import { icon } from "../icons.js";
import { copyText } from "../legacy.js";
import {
  SCAN_LEAGUES, DEFAULT_LEAGUE, MAX_THRESHOLD, MAX_ROWS, scanLeague, statsFor, statOf, scanQuery, finderHash, askText,
  whenText, rowItemHtml, chanceHtml, lineHtml, statusChip, modelChip, hitsText, hitShare, numText, edgeText, matchupText,
  playerHref, loadingText, ensureScanStyles,
} from "./players/scan.js";

export const title = "Player finder";

const SORTS = [
  { value: "chance", label: "Chance" },
  { value: "projection", label: "Projection" },
  { value: "l10", label: "L10 hits" },
  { value: "edge", label: "Edge" },
];
const SORT_KEYS = SORTS.map((s) => s.value);
const POLL_FIRST_MS = 3000; // a loading board is asked again after this, then a little later each time
const POLL_MAX_MS = 10000;
const POLL_FOR_MS = 4 * 60 * 1000; // a cold board builds in under 3 minutes: stop asking after this
const TYPE_MS = 450; // typing a number waits this long before scanning

let leaving = null;

export function onLeave() {
  leaving?.();
  leaving = null;
}

// --- the ask in the address ------------------------------------------------------------------------------------

/** The scan the address asks for, every part checked: {league, stat, at, line, side, day, sort}. */
export function readAsk(query) {
  const get = (k) => (query?.get ? query.get(k) : null);
  const league = scanLeague(get("league")) ? get("league") : DEFAULT_LEAGUE;
  const stats = statsFor(league);
  const stat = statOf(league, get("stat")) ? get("stat") : stats[0].key;
  const spec = statOf(league, stat);
  const rawAt = String(get("at") ?? "").trim().toLowerCase();
  let line = rawAt === "line";
  let at = Number(rawAt);
  if (!line && !(Number.isFinite(at) && at > 0 && at <= MAX_THRESHOLD)) at = spec.start;
  at = Math.ceil(at - 1e-9);
  if (spec.yes) {
    at = spec.start;
    line = false;
  }
  return {
    league, stat, at, line,
    side: !spec.yes && get("side") === "under" ? "under" : "over",
    day: get("day") === "tomorrow" ? "tomorrow" : "today",
    // Each player's posted line ranks by edge first, as the server does; a number by chance
    sort: SORT_KEYS.includes(get("sort")) ? get("sort") : line ? "edge" : "chance",
  };
}

function hashOf(ask) {
  return finderHash({ league: ask.league, stat: ask.stat, threshold: ask.line ? null : ask.at, side: ask.side, day: ask.day, sort: ask.sort });
}

function stepOf(n) {
  return n >= 100 ? 10 : n >= 40 ? 5 : 1;
}

// --- sorting ---------------------------------------------------------------------------------------------------

/** Rows in the chosen order, highest first (an under's projection lowest first); ties keep the server's order. */
export function sortRows(rows, sort, side = "over") {
  const keyed = rows.map((row, i) => ({ row, i }));
  const num = (v, missing) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? missing : Number(v));
  const value = {
    chance: (r) => num(r.chance, -1),
    projection: (r) => (side === "under" ? -num(r.projection, Infinity) : num(r.projection, -Infinity)),
    l10: (r) => hitShare(r.hits, "l10") * 1000 + Math.min(999, num(r.hits?.l10?.[1], 0)),
    edge: (r) => num(r.edge, -Infinity),
  }[sort] || ((r) => num(r.chance, -1));
  keyed.sort((a, b) => (value(b.row) - value(a.row)) || (a.i - b.i));
  return keyed.map((k) => k.row);
}

// --- drawing -----------------------------------------------------------------------------------------------------

function leagueOptions(current) {
  const group = (name, label) => `<optgroup label="${esc(label)}">${SCAN_LEAGUES.filter((l) => l.group === name).map((l) =>
    `<option value="${esc(l.key)}"${l.key === current ? " selected" : ""}>${esc(l.title)}</option>`).join("")}</optgroup>`;
  return group("us", "US sports") + group("soccer", "Soccer");
}

function statOptions(league, current) {
  return statsFor(league).map((s) => `<option value="${esc(s.key)}"${s.key === current ? " selected" : ""}>${esc(s.name)}</option>`).join("");
}

function numberHtml(ask) {
  const spec = statOf(ask.league, ask.stat);
  if (spec.yes) {
    return `<div class="pf-field pf-at"><span class="label">Number</span>
      <p class="pf-fixed">${icon("check", { size: 16 })}<span>${esc(spec.name)}: a yes or no bet</span></p></div>`;
  }
  const off = ask.line;
  const word = ask.side === "under" ? "Fewer than" : "At least";
  return `<div class="pf-field pf-at"><span class="label" id="pf-at-label">Number<span class="sr-only">: ${off ? "each player's posted line" : word.toLowerCase()}</span>${off ? ` <span class="muted small">each player's posted line</span>` : ""}</span>
    <div class="pf-at-row">
      ${segmented("pf-side", [{ value: "over", label: "At least" }, { value: "under", label: "Fewer than" }], ask.side, { label: "Over or under" })}
      <span class="stepper${off ? " off" : ""}" role="group" aria-labelledby="pf-at-label">
        <button type="button" data-pf-step="-1" aria-label="Lower the number"${off || ask.at <= 1 ? " disabled" : ""}>${icon("minus", { size: 18 })}</button>
        <input class="pf-num" data-pf="at" type="number" inputmode="numeric" min="1" max="${MAX_THRESHOLD}" step="1" value="${esc(ask.at)}"
          aria-label="${esc(`${word} this many ${spec.name.toLowerCase()}`)}"${off ? " disabled" : ""}>
        <button type="button" data-pf-step="1" aria-label="Raise the number"${off || ask.at >= MAX_THRESHOLD ? " disabled" : ""}>${icon("plus", { size: 18 })}</button>
      </span>
      ${chip("Posted line", { active: off, attrs: 'data-pf-line title="Each player\'s own posted line, ranked by edge"', iconName: "target" })}
    </div></div>`;
}

function controlsHtml(ask) {
  return `<div class="pf-grid">
    <label class="pf-field"><span class="label">League</span><select class="input" data-pf="league">${leagueOptions(ask.league)}</select></label>
    <label class="pf-field"><span class="label">Stat</span><select class="input" data-pf="stat">${statOptions(ask.league, ask.stat)}</select></label>
    ${numberHtml(ask)}
    <div class="pf-field"><span class="label" id="pf-day-label">Day</span>${segmented("pf-day", [{ value: "today", label: "Today" }, { value: "tomorrow", label: "Tomorrow" }], ask.day, { label: "Day" })}</div>
    <div class="pf-field pf-sort-field"><span class="label">Sort by</span>${segmented("pf-sort", SORTS, ask.sort, { label: "Sort by" })}</div>
  </div>`;
}

/** The summary above the rows: the ask in words, when, the loading line, Refresh and Copy link. */
function summaryHtml(state) {
  const scan = state.scan;
  const ask = state.ask;
  const league = scanLeague(ask.league)?.title || "";
  const words = scan ? askText(scan) : "";
  const loading = scan ? loadingText(scan.loading) : "";
  const polling = Boolean(scan?.loading && state.pollUntil > Date.now());
  return `<div class="pf-summary">
    <div class="grow"><b>${esc(words ? `${words[0].toUpperCase()}${words.slice(1)}` : "Players")}</b> <span class="muted">· ${esc(league)} · ${esc(whenText(ask))}</span>
      ${scan?.as_of ? `<span class="muted small">· ESPN data from ${esc(ago(scan.as_of))}</span>` : ""}
      ${loading ? `<p class="ps-loading" role="status">${icon("clock", { size: 14 })}<span>${esc(loading)}${polling ? " · updating" : ""}</span></p>` : ""}</div>
    <div class="pf-tools">
      <button type="button" class="btn btn-secondary small" data-pf-refresh${state.busy ? " disabled aria-busy=\"true\"" : ""}>${icon("refresh", { size: 16 })}${state.busy ? "Loading…" : "Refresh"}</button>
      <button type="button" class="btn btn-secondary small" data-pf-copy>${icon("share", { size: 16 })}Copy link</button>
    </div>
  </div>`;
}

function sortHead(label, key, ask, { cls = "" } = {}) {
  const on = ask.sort === key;
  return `<th scope="col" class="${cls}"${on ? ' aria-sort="descending"' : ""}><button type="button" class="pf-sort${on ? " on" : ""}" data-pf-sortby="${key}"
    aria-label="${esc(`Sort by ${label}${on ? " (sorted)" : ""}`)}">${esc(label)}${on ? icon("down", { size: 14 }) : ""}</button></th>`;
}

// The row's reasons the table's own columns don't already show (its hits and its posted line)
const SHOWN_WHY = /^(?:under \S+ |a .+ |\S+ )?in \d+ of last|^\d+ of last|\bposted\b/i;

function tableRow(row, i, ask) {
  const href = playerHref(row);
  const why = (Array.isArray(row.why) ? row.why : []).filter((w) => w && !SHOWN_WHY.test(String(w))).slice(0, 2);
  const meta = [matchupText(row), row.start ? timeLabel(row.start) : "", row.minutes != null ? `${numText(row.minutes)} min` : ""].filter(Boolean).join(" · ");
  const edge = edgeText(row.edge);
  return `<tr>
    <td class="num pf-rank">${esc(row.rank || i + 1)}</td>
    <th scope="row" class="pf-player">
      <div class="pf-who">${href ? `<a class="pf-name" href="${esc(href)}">${esc(row.name || "Player")}</a>` : `<b class="pf-name">${esc(row.name || "Player")}</b>`}${statusChip(row)}${modelChip(row)}</div>
      <div class="muted small">${esc(meta)}</div>
      ${why.length ? `<div class="small pf-why">${why.map(esc).join(" · ")}</div>` : ""}
    </th>
    <td class="num pf-n">${esc(numText(row.projection))}</td>
    <td class="pf-c">${chanceHtml(row.chance)}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "l5"))}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "l10"))}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "season"))}</td>
    <td class="pf-line small">${lineHtml(row, ask.side, { edge: false })}</td>
    <td class="num pf-e${Number(row.edge) > 0 ? " pos" : ""}">${edge ? esc(edge) : '<span class="muted">–</span>'}</td>
  </tr>`;
}

function tableHtml(players, close, state) {
  const ask = state.ask;
  const head = `<thead><tr>
    <th scope="col" class="pf-rank">#</th><th scope="col">Player</th>
    ${sortHead("Proj.", "projection", ask, { cls: "pf-n" })}${sortHead("Chance", "chance", ask, { cls: "pf-c" })}
    <th scope="col" class="pf-h">L5</th>${sortHead("L10", "l10", ask, { cls: "pf-h" })}<th scope="col" class="pf-h">Season</th>
    <th scope="col">Posted line</th>${sortHead("Edge", "edge", ask, { cls: "pf-e" })}
  </tr></thead>`;
  const closeRows = close.length ? `<tbody class="pf-close">
    <tr class="pf-group"><td colspan="9"><button type="button" class="pf-toggle" data-pf-close aria-expanded="${state.closeOpen}">${icon("chevron", { size: 16 })}Also close (${close.length})</button></td></tr>
    ${state.closeOpen ? close.map((row, i) => tableRow(row, players.length + i, ask)).join("") : ""}</tbody>` : "";
  return `<div class="table-wrap pf-table-wrap"><table class="pf-table">
    <caption class="sr-only">${esc(`Players for ${askText(state.scan)} ${whenText(ask)}, sorted by ${SORTS.find((s) => s.value === ask.sort)?.label || "chance"}`)}</caption>
    ${head}<tbody>${players.map((row, i) => tableRow(row, i, ask)).join("")}</tbody>${closeRows}</table></div>`;
}

function cardsHtml(players, close, state) {
  const side = state.ask.side;
  return `${players.length ? `<ol class="ps-list" aria-label="${esc(`Players for ${askText(state.scan)}`)}">${players.map((row, i) => rowItemHtml(row, i, { side })).join("")}</ol>` : ""}
    ${close.length ? `<details class="ps-close" data-pf-details${state.closeOpen ? " open" : ""}><summary>${icon("chevron", { size: 16 })}Also close (${close.length})</summary>
      <ol class="ps-list" aria-label="Also close">${close.map((row, i) => rowItemHtml(row, players.length + i, { side })).join("")}</ol></details>` : ""}`;
}

function resultsHtml(state) {
  const scan = state.scan;
  if (state.error) return state.error;
  if (!scan) return skeleton("lines", 6);
  const ask = state.ask;
  const league = ask.league;
  const tag = (row) => ({ league, ...row });
  const players = sortRows((scan.players || []).filter((r) => r && typeof r === "object").map(tag), ask.sort, ask.side);
  const close = sortRows((scan.close || []).filter((r) => r && typeof r === "object").map(tag), ask.sort, ask.side);
  const leagueName = scanLeague(league)?.title || "";
  if (!players.length && !close.length) {
    if (scan.loading) return `${notice(loadingText(scan.loading) || "Reading the day's games…", "info")}${skeleton("lines", 4)}`;
    const noGames = (scan.leagues_without_games || []).length || (!scan.leagues?.length && !scan.games);
    if (noGames) {
      return emptyState({
        iconName: "today", title: `No ${leagueName} games ${whenText(ask)}`,
        body: "Try another league, or tomorrow's games.",
        actions: ask.day === "today" ? [{ label: "Tomorrow's games", attrs: "data-pf-tomorrow", primary: true }] : [],
      });
    }
    return emptyState({
      iconName: "search", title: `Nobody is projected for ${askText(scan)} yet`,
      body: ask.line ? "No posted lines for this stat yet: try a number instead." : "Try a lower number, another stat or tomorrow's games.",
      actions: !ask.line && ask.at > 1 && !statOf(league, ask.stat)?.yes ? [{ label: `Try ${Math.max(1, ask.at - stepOf(ask.at) * 2)}+`, attrs: "data-pf-lower", primary: true }] : [],
    });
  }
  const lead = players.length ? "" : `<p class="small pf-lead">Nobody is projected for ${esc(askText(scan))} yet: the closest are below.</p>`;
  const more = Number(scan.more) > 0 ? `<p class="ps-note">${esc(`${scan.more} more qualify beyond the ${players.length} shown.`)}</p>` : "";
  const list = layout.mode === "phone" ? cardsHtml(players, close, { ...state, closeOpen: state.closeOpen || !players.length })
    : tableHtml(players, close, { ...state, closeOpen: state.closeOpen || !players.length });
  const forms = [...players, ...close].some((r) => r.model === "form");
  return `${lead}${list}${more}
    <p class="ps-note">L5, L10 and Season count the games at the number. ${forms ? "\"Recent form\" rows are projected from the game log (recent games weigh most). " : ""}Edge is the chance × the posted price − 1, shown only when the price is for exactly this number. Lines: DraftKings via ESPN. Player props are for reading here: the slip takes game picks. ${explain("player_finder", { label: "How the Player finder works" })}</p>`;
}

// --- styles ---------------------------------------------------------------------------------------------------

const CSS = `
@media (min-width: 840px) { body[data-route="players"] .shell > * { max-width: 1280px; } }
.view-players .pf-controls { display: grid; gap: 12px; }
.view-players .pf-grid { display: grid; gap: 14px 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.view-players .pf-grid > .pf-field:nth-child(n+3) { grid-column: 1 / -1; }
@media (min-width: 560px) { .view-players .pf-grid { gap: 14px; } }
@media (min-width: 760px) { .view-players .pf-grid > .pf-field:nth-child(n+4) { grid-column: auto; } }
@media (min-width: 1100px) { .view-players .pf-grid { grid-template-columns: minmax(120px, 0.7fr) minmax(170px, 1fr) minmax(410px, 2.2fr) minmax(170px, 0.8fr); }
  .view-players .pf-grid > .pf-field:nth-child(n+3) { grid-column: auto; }
  .view-players .pf-grid > .pf-sort-field { grid-column: 1 / -1; } }
.view-players .pf-field { display: grid; gap: 0; min-width: 0; align-content: start; }
.view-players .pf-field .label { margin-bottom: 6px; }
.view-players .pf-at-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0; }
.view-players .pf-at-row .chip { min-height: 40px; }
.view-players .pf-at-row .seg { flex: 0 1 auto; }
.view-players .pf-at-row .seg button { padding: 0 10px; }
.view-players .stepper.off { opacity: 0.5; }
.view-players .pf-num { width: 64px; min-height: var(--tap); border: 0; background: transparent; text-align: center; font-weight: 800; font-size: 16px;
  color: var(--text); font-variant-numeric: tabular-nums; -moz-appearance: textfield; }
.view-players .pf-num::-webkit-outer-spin-button, .view-players .pf-num::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.view-players .pf-num:focus { outline: 2px solid var(--accent); outline-offset: -2px; }
.view-players .stepper button:disabled { opacity: 0.45; }
.view-players .pf-fixed { display: flex; align-items: center; gap: 6px; min-height: var(--tap); margin: 0; color: var(--text-2); font-size: 14px; }
.view-players .pf-sort-field .seg { max-width: 560px; }
.view-players .pf-sort-field .seg button { padding: 0 6px; font-size: 13.5px; }
.view-players .pf-summary { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 14px 0 10px; }
.view-players .pf-summary .grow { flex: 1 1 260px; min-width: 0; overflow-wrap: anywhere; }
.view-players .pf-summary .ps-loading { margin-top: 4px; }
.view-players .pf-tools { display: flex; gap: 8px; flex-wrap: wrap; }
.view-players .pf-lead { margin: 0 0 8px; }
.view-players .ps-note { margin-top: 10px; }
.view-players .pf-results .empty { margin-top: 8px; }
.view-players .pf-table-wrap { border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
.view-players .pf-table { font-size: 13.5px; }
.view-players .pf-table th, .view-players .pf-table td { padding: 8px 8px; vertical-align: top; text-align: left; border-bottom: 1px solid var(--line); }
.view-players .pf-table thead th { font-size: 12px; color: var(--text-2); font-weight: 650; text-transform: uppercase; letter-spacing: 0.02em; white-space: nowrap; vertical-align: bottom; }
.view-players .pf-table tbody th[scope="row"] { text-transform: none; letter-spacing: 0; font-size: 13.5px; font-weight: 400; color: var(--text); white-space: normal; min-width: 200px; }
.view-players .pf-table .num { font-variant-numeric: tabular-nums; font-weight: 700; white-space: nowrap; }
.view-players .pf-rank { width: 32px; color: var(--text-2); }
.view-players .pf-n, .view-players .pf-h, .view-players .pf-e { text-align: right; }
.view-players .pf-table thead .pf-n, .view-players .pf-table thead .pf-h, .view-players .pf-table thead .pf-e { text-align: right; }
.view-players .pf-c { min-width: 120px; }
.view-players .pf-e.pos { color: var(--good-ink); }
.view-players .pf-line { min-width: 150px; }
.view-players .pf-line .badge { margin-top: 2px; }
.view-players .pf-who { display: flex; flex-wrap: wrap; align-items: center; gap: 2px 8px; }
.view-players .pf-name { display: inline-flex; align-items: center; min-height: var(--tap); margin: -10px 0 -8px; font-weight: 700; font-size: 14.5px; color: var(--text); }
.view-players a.pf-name:hover { color: var(--info); }
.view-players .pf-why { color: var(--text-2); margin-top: 2px; line-height: 1.4; }
.view-players .pf-sort { display: inline-flex; align-items: center; gap: 2px; min-height: var(--tap); margin: -12px -4px -8px; padding: 0 4px; border-radius: 8px;
  color: inherit; font: inherit; text-transform: inherit; letter-spacing: inherit; }
.view-players .pf-sort.on { color: var(--text); }
.view-players .pf-sort:hover { color: var(--text); background: var(--card-2); }
.view-players .pf-group td { padding: 0 8px; background: var(--card-2); }
.view-players .pf-toggle { display: inline-flex; align-items: center; gap: 6px; min-height: var(--tap); font-weight: 650; color: var(--text-2); }
.view-players .pf-toggle .icon { transition: transform var(--speed); }
.view-players .pf-toggle[aria-expanded="true"] .icon { transform: rotate(90deg); }
.view-players .pf-table tbody tr:last-child th, .view-players .pf-table tbody tr:last-child td { border-bottom-color: var(--line); }
.view-players .pf-table tbody:last-child tr:last-child th, .view-players .pf-table tbody:last-child tr:last-child td { border-bottom: 0; }
@media (max-width: 1199px) { .view-players .pf-why { display: none; } }
`;

function ensureStyles() {
  ensureScanStyles();
  if (document.getElementById("finder-styles")) return;
  const style = document.createElement("style");
  style.id = "finder-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}

// --- the screen -----------------------------------------------------------------------------------------------

/** A refusal from the server (age, a break) as a way forward, or a plain error. */
function failureHtml(error) {
  const message = error?.message || "";
  if (/18\+|21\+|confirm/i.test(message)) {
    return emptyState({ iconName: "shield", title: "Confirm your age first", body: message, actions: [{ label: "Open settings", href: "#/settings", primary: true }] });
  }
  if (/break/i.test(message)) {
    return emptyState({ iconName: "pause", title: "You're on a break", body: message, actions: [{ label: "Responsible play", href: "#/help/responsible", primary: true }] });
  }
  return errorState(error);
}

export async function render(root, params) {
  ensureStyles();
  setTitle(title);
  const signal = params.signal;
  const state = { ask: readAsk(params.query), scan: null, error: "", busy: false, closeOpen: false, pollUntil: 0 };
  let token = 0;
  let controller = null;
  let pollTimer = null;
  let typeTimer = null;
  let pollWait = POLL_FIRST_MS;

  root.innerHTML = `<div class="page-head"><h1>Player finder</h1>
      <p>Who's projected to reach a number: how often each player got there (last 5, last 10, this season), tonight's projection, the chance and the posted line. Free: no AI allowance, no odds requests.</p></div>
    <section class="card section pf-controls" aria-label="What to find" data-pf-controls></section>
    <div data-pf-summary></div>
    <section class="pf-results" aria-label="Players" data-pf-results></section>`;
  const $ = (sel) => root.querySelector(sel);

  const drawControls = () => {
    const box = $("[data-pf-controls]");
    const active = document.activeElement;
    const key = box.contains(active)
      ? (active.dataset.pf ? `[data-pf="${active.dataset.pf}"]` : active.dataset.seg ? `[data-seg="${active.dataset.seg}"][data-value="${active.dataset.value}"]`
        : active.matches("[data-pf-line]") ? "[data-pf-line]" : active.dataset.pfStep ? `[data-pf-step="${active.dataset.pfStep}"]` : null)
      : null;
    box.innerHTML = controlsHtml(state.ask);
    if (key) box.querySelector(key)?.focus({ preventScroll: true });
  };
  const drawSummary = () => {
    const box = $("[data-pf-summary]");
    const focused = box.contains(document.activeElement) ? (document.activeElement.matches("[data-pf-copy]") ? "[data-pf-copy]" : "[data-pf-refresh]") : null;
    box.innerHTML = summaryHtml(state);
    if (focused) box.querySelector(focused)?.focus({ preventScroll: true });
  };
  const drawResults = () => {
    const box = $("[data-pf-results]");
    const focused = box.contains(document.activeElement) ? document.activeElement : null;
    const sortKey = focused?.dataset?.pfSortby;
    const toggle = focused?.matches?.("[data-pf-close]");
    box.innerHTML = resultsHtml(state);
    if (sortKey) box.querySelector(`[data-pf-sortby="${sortKey}"]`)?.focus({ preventScroll: true });
    else if (toggle) box.querySelector("[data-pf-close]")?.focus({ preventScroll: true });
  };
  const drawAll = () => {
    drawControls();
    drawSummary();
    drawResults();
  };

  const stopPolling = () => {
    clearTimeout(pollTimer);
    pollTimer = null;
  };

  /** Scan with the current controls; ``quiet`` keeps the rows on screen while asking (polling, Refresh). */
  async function load({ quiet = false } = {}) {
    stopPolling();
    controller?.abort();
    controller = new AbortController();
    const mine = ++token;
    const ask = { ...state.ask };
    state.busy = true;
    state.error = "";
    if (!quiet) {
      state.scan = null;
      drawResults();
    }
    drawSummary();
    const abort = () => controller?.abort();
    signal.addEventListener("abort", abort, { once: true });
    let data = null;
    let failure = null;
    try {
      data = await api(scanQuery({ league: ask.league, stat: ask.stat, threshold: ask.line ? null : ask.at, side: ask.side, day: ask.day, n: MAX_ROWS }), { signal: controller.signal });
    } catch (error) {
      failure = error;
    }
    signal.removeEventListener("abort", abort);
    if (mine !== token || !root.isConnected) return;
    state.busy = false;
    if (failure) {
      if (isQuiet(failure)) return;
      state.scan = null;
      state.error = failureHtml(failure);
      state.pollUntil = 0;
    } else {
      state.scan = data || {};
      if (state.scan.loading) {
        if (!state.pollUntil) state.pollUntil = Date.now() + POLL_FOR_MS;
        if (Date.now() < state.pollUntil) {
          pollTimer = setTimeout(() => load({ quiet: true }), pollWait);
          pollWait = Math.min(POLL_MAX_MS, Math.round(pollWait * 1.5));
        }
      } else {
        state.pollUntil = 0;
        pollWait = POLL_FIRST_MS;
      }
    }
    drawSummary();
    drawResults();
  }

  /** The controls changed: the address follows (no new history entry) and a new scan runs. */
  const changed = ({ controls = true } = {}) => {
    state.pollUntil = 0;
    pollWait = POLL_FIRST_MS;
    state.closeOpen = false;
    replaceAddress(hashOf(state.ask));
    if (controls) drawControls();
    load();
  };
  const resorted = () => {
    replaceAddress(hashOf(state.ask));
    drawControls();
    drawResults();
  };

  const setAt = (value) => {
    const n = Math.max(1, Math.min(MAX_THRESHOLD, Math.ceil(Number(value) - 1e-9)));
    if (!Number.isFinite(n) || n === state.ask.at) return false;
    state.ask.at = n;
    return true;
  };

  root.addEventListener("change", (event) => {
    const el = event.target;
    if (el.matches('[data-pf="league"]')) {
      const league = scanLeague(el.value) ? el.value : DEFAULT_LEAGUE;
      const keep = statOf(league, state.ask.stat);
      state.ask = readAsk(new URLSearchParams({ league, stat: keep ? state.ask.stat : "", at: keep && !keep.yes ? (state.ask.line ? "line" : String(state.ask.at)) : "",
        side: keep ? state.ask.side : "over", day: state.ask.day, sort: state.ask.sort }));
      changed();
    } else if (el.matches('[data-pf="stat"]')) {
      const spec = statOf(state.ask.league, el.value);
      if (!spec) return;
      state.ask = readAsk(new URLSearchParams({ league: state.ask.league, stat: spec.key, at: state.ask.line && !spec.yes ? "line" : String(spec.start),
        side: spec.yes ? "over" : state.ask.side, day: state.ask.day, sort: state.ask.sort }));
      changed();
    } else if (el.matches('[data-pf="at"]')) {
      clearTimeout(typeTimer);
      if (setAt(el.value)) changed();
      else el.value = String(state.ask.at);
    }
  });
  root.addEventListener("input", (event) => {
    const el = event.target;
    if (!el.matches('[data-pf="at"]')) return;
    clearTimeout(typeTimer);
    typeTimer = setTimeout(() => {
      const value = Number(el.value);
      if (Number.isFinite(value) && value > 0 && value <= MAX_THRESHOLD && setAt(value)) changed({ controls: false });
    }, TYPE_MS);
  });
  root.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches('[data-pf="at"]')) {
      clearTimeout(typeTimer);
      if (setAt(event.target.value)) changed({ controls: false });
    }
  });
  root.addEventListener("click", (event) => {
    const t = event.target;
    const seg = t.closest("[data-seg]");
    if (seg && root.contains(seg)) {
      const value = seg.dataset.value;
      if (seg.dataset.seg === "pf-day" && value !== state.ask.day) {
        state.ask.day = value === "tomorrow" ? "tomorrow" : "today";
        changed();
      } else if (seg.dataset.seg === "pf-side" && value !== state.ask.side) {
        state.ask.side = value === "under" ? "under" : "over";
        state.ask.line = false;
        changed();
      } else if (seg.dataset.seg === "pf-sort" && value !== state.ask.sort && SORT_KEYS.includes(value)) {
        state.ask.sort = value;
        resorted();
      }
      return;
    }
    const sortBy = t.closest("[data-pf-sortby]");
    if (sortBy) {
      const value = sortBy.dataset.pfSortby;
      if (SORT_KEYS.includes(value) && value !== state.ask.sort) {
        state.ask.sort = value;
        resorted();
      }
      return;
    }
    const step = t.closest("[data-pf-step]");
    if (step) {
      const dir = Number(step.dataset.pfStep);
      const size = dir > 0 ? stepOf(state.ask.at) : stepOf(Math.max(1, state.ask.at - 1));
      if (setAt(state.ask.at + dir * size)) changed();
      return;
    }
    if (t.closest("[data-pf-line]")) {
      state.ask.line = !state.ask.line;
      if (state.ask.line) state.ask.side = "over";
      // Posted lines rank by edge, a number by chance (unless another sort was chosen)
      if (state.ask.line && state.ask.sort === "chance") state.ask.sort = "edge";
      else if (!state.ask.line && state.ask.sort === "edge") state.ask.sort = "chance";
      changed();
      return;
    }
    if (t.closest("[data-pf-refresh]")) {
      load({ quiet: Boolean(state.scan) });
      return;
    }
    if (t.closest("[data-pf-copy]")) {
      const link = `${location.origin}${location.pathname}${hashOf(state.ask)}`;
      copyText(link, "Link");
      return;
    }
    if (t.closest("[data-pf-close]")) {
      state.closeOpen = !state.closeOpen;
      drawResults();
      return;
    }
    if (t.closest("[data-pf-tomorrow]")) {
      state.ask.day = "tomorrow";
      changed();
      return;
    }
    if (t.closest("[data-pf-lower]")) {
      if (setAt(Math.max(1, state.ask.at - stepOf(state.ask.at) * 2))) changed();
    }
  });
  root.addEventListener("toggle", (event) => {
    if (event.target.matches?.("[data-pf-details]")) state.closeOpen = event.target.open;
  }, true);

  // Phones get cards, wider screens the table: redraw the rows when the layout changes
  const offLayout = onLayoutChange(() => {
    if (root.isConnected && state.scan) drawResults();
  });
  leaving = () => {
    stopPolling();
    clearTimeout(typeTimer);
    controller?.abort();
    offLayout?.();
  };

  // A tidy address (every part checked) without a new history entry
  if (location.hash !== hashOf(state.ask)) replaceAddress(hashOf(state.ask));
  drawAll();
  await load();
}
