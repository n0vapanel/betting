// Player finder (#/players): today's or tomorrow's players ranked for a stat and a number, from the player scan
// (GET /api/players/scan; js/views/players/scan.js has the row pieces Ask AI's card shares).
//
// Reached from Build ("Find players"), the player search ("Find a player" on Today, a game's player search) and
// Ask AI's player_scan card ("Open in Player finder"). The controls: league, stat (the scan's stats with friendly
// names), "At least" a number (or "Fewer than", an under; or each player's posted line), today or tomorrow, a player
// name filter (one name, or up to five separated by commas), and the sort: the list's own (chance, projection, L10
// hits, edge: sorted on the screen) or the server's rankings, each a new scan: "Hot (last 5 vs season)" and "Cold"
// rank everyone with 5 games logged this season by the stat's average over those 5 against the rest of the season
// (or last season's, early in a season: the chip says "last season"), and "Best edges (all stats)" ranks every
// stat's posted lines by edge. The rows: a dense table with sortable headers on wider screens,
// cards on phones; the first five, then the rest that qualify under "Also:"; "Also close" stays collapsed until
// opened; how many more qualify beyond the rows shown.
//
// The address carries the scan, so a link opens the same one:
//   #/players?league=basketball_nba&stat=pts&at=20&day=today[&side=under][&sort=projection|hot|cold|edges][&player=...]
//   (at=line: posted lines; player repeats, up to 5 names)
// A board still loading answers with what's ready ("8 of 10 games loaded"): the page asks again every few seconds
// until it's complete, and Refresh asks at once; each name it couldn't rank (not found, more than one match, game
// already started), games still missing data, games already started and ESPN's scoreboard not answering are said
// above the rows, worded as Ask AI's card, the chat and the bot word them. A scan is code only: no AI allowance and
// no Odds API requests.
// Every number shown is the server's; text from outside is escaped.

import { api, isQuiet } from "../api.js";
import { setTitle, replaceAddress, layout, onLayoutChange } from "../app.js";
import { esc, segmented, chip, notice, skeleton, emptyState, errorState, ago, explain, timeLabel } from "../ui.js";
import { icon } from "../icons.js";
import { copyText } from "../legacy.js";
import {
  SCAN_LEAGUES, DEFAULT_LEAGUE, MAX_THRESHOLD, MAX_ROWS, MAX_NAMES, MAX_NAME, TOP, TREND_SORTS, EDGES, scanLeague,
  statsFor, statOf, scanQuery, finderHash, askText, whenText, rowItemHtml, chanceHtml, lineHtml, statusChip, flagChip,
  trendChip, modelChip, hitsText, hitShare, numText, edgeText, shownEdge, matchupText, playerHref, loadingText,
  statusText, stillLoading, rowListsHtml, alsoLabel, moreText, nameNotes, boardNotes, allStarted, startedText, notesHtml,
  cleanNames, namesText, rowStatName, ensureScanStyles,
} from "./players/scan.js";

export const title = "Player finder";

// The list's own sorts (on the screen: no new request)
const SORTS = [
  { value: "chance", label: "Chance" },
  { value: "projection", label: "Projection" },
  { value: "l10", label: "L10 hits" },
  { value: "edge", label: "Edge" },
];
// The server's rankings (each a new scan): the trend of everyone with 5 games logged, or every stat's posted lines
const RANKS = [
  { value: "hot", label: "Hot (last 5 vs season)" },
  { value: "cold", label: "Cold" },
  { value: EDGES, label: "Best edges (all stats)" },
];
const LIST_SORTS = SORTS.map((s) => s.value);
const SORT_KEYS = [...LIST_SORTS, ...RANKS.map((s) => s.value)];
const POLL_FIRST_MS = 3000; // a loading board is asked again after this, then a little later each time
const POLL_MAX_MS = 10000;
const POLL_FOR_MS = 4 * 60 * 1000; // a cold board builds in under 3 minutes: stop asking after this
const TYPE_MS = 450; // typing a number waits this long before scanning
const NAME_MS = 800; // typing a name waits a little longer (a half-typed name matches nobody)

let leaving = null;

export function onLeave() {
  leaving?.();
  leaving = null;
}

/** Whether a sort is one of the server's rankings (hot, cold, best edges): a new scan, not a new order. */
const ranked = (sort) => !LIST_SORTS.includes(sort);

// --- the ask in the address ------------------------------------------------------------------------------------

/** The scan the address asks for, every part checked: {league, stat, at, line, side, day, sort, names}. */
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
  // Each player's posted line ranks by edge first, as the server does; a number by chance
  const sort = SORT_KEYS.includes(get("sort")) ? get("sort") : line ? "edge" : "chance";
  return {
    league, stat, at, line,
    side: !spec.yes && get("side") === "under" ? "under" : "over",
    day: get("day") === "tomorrow" ? "tomorrow" : "today",
    sort,
    // Best edges covers every player: a name filter goes with the other sorts
    names: sort === EDGES ? [] : cleanNames(query?.getAll ? query.getAll("player") : []),
  };
}

function hashOf(ask) {
  return finderHash({ league: ask.league, stat: ask.stat, threshold: ask.line ? null : ask.at, side: ask.side, day: ask.day, sort: ask.sort, names: ask.names });
}

/** The ask as an address's query, with some parts changed (readAsk checks them all again). */
function askQuery(ask, change = {}) {
  const next = { ...ask, ...change };
  const q = new URLSearchParams({ league: next.league, stat: next.stat ?? "", at: next.at ?? "", side: next.side, day: next.day, sort: next.sort });
  for (const name of next.names || []) q.append("player", name);
  return q;
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
    edge: (r) => num(shownEdge(r), -Infinity),
  }[sort] || ((r) => num(r.chance, -1));
  keyed.sort((a, b) => (value(b.row) - value(a.row)) || (a.i - b.i));
  return keyed.map((k) => k.row);
}

/** How the rows were asked for (rowItemHtml's ``view``, and what the table shows). */
function viewOf(ask) {
  const edges = ask.sort === EDGES;
  const trend = TREND_SORTS.includes(ask.sort) ? ask.sort : "";
  return {
    // each player's posted line ranks by edge (hot and cold rank by the trend, whatever the number)
    edges, list: !ranked(ask.sort), trend, names: !edges && ask.names.length > 0, posted: !edges && !trend && ask.line,
    league: ask.league, yesNo: !edges && Boolean(statOf(ask.league, ask.stat)?.yes),
  };
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

function statHtml(ask) {
  if (ask.sort === EDGES) {
    return `<label class="pf-field"><span class="label">Stat</span><select class="input" data-pf="stat" disabled><option selected>All stats</option></select></label>`;
  }
  return `<label class="pf-field"><span class="label">Stat</span><select class="input" data-pf="stat">${statOptions(ask.league, ask.stat)}</select></label>`;
}

function numberHtml(ask) {
  const spec = statOf(ask.league, ask.stat);
  if (ask.sort === EDGES) {
    return `<div class="pf-field pf-at"><span class="label">Number</span>
      <p class="pf-fixed">${icon("target", { size: 16 })}<span>Each player's posted line, every stat</span></p></div>`;
  }
  if (spec.yes) {
    return `<div class="pf-field pf-at"><span class="label">Number</span>
      <p class="pf-fixed">${icon("check", { size: 16 })}<span>${esc(spec.name)}: a yes or no bet</span></p></div>`;
  }
  const off = ask.line;
  const word = ask.side === "under" ? "Fewer than" : "At least";
  // Screen readers hear the label once: "Number: at least", or "Number: each player's posted line" (shown as well)
  return `<div class="pf-field pf-at"><span class="label" id="pf-at-label">Number${off
    ? `<span class="sr-only">:</span> <span class="muted small">each player's posted line</span>` : `<span class="sr-only">: ${word.toLowerCase()}</span>`}</span>
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

/** The name filter: one name or several separated by commas (off for Best edges, which covers every player). */
function playerHtml(ask) {
  const edges = ask.sort === EDGES;
  return `<div class="pf-field pf-player-field"><label class="label" for="pf-player">Player</label>
    <div class="pf-search">${icon("search", { size: 16 })}
      <input class="input" id="pf-player" data-pf="player" type="text" inputmode="search" enterkeyhint="search" autocomplete="off" autocapitalize="words"
        spellcheck="false" maxlength="${MAX_NAMES * (MAX_NAME + 2)}" placeholder="${edges ? "Every player" : "Any player"}" value="${esc(ask.names.join(", "))}"
        aria-describedby="pf-player-hint"${edges ? " disabled" : ""}>
      <button type="button" class="pf-clear" data-pf-clear-name aria-label="Clear the player name">${icon("close", { size: 18 })}</button>
    </div>
    <span class="muted small pf-hint" id="pf-player-hint">${edges ? "Best edges covers every player." : "A name, or up to 5 with commas."}</span></div>`;
}

function controlsHtml(ask) {
  return `<div class="pf-grid">
    <label class="pf-field"><span class="label">League</span><select class="input" data-pf="league">${leagueOptions(ask.league)}</select></label>
    ${statHtml(ask)}
    ${numberHtml(ask)}
    <div class="pf-field"><span class="label" id="pf-day-label">Day</span>${segmented("pf-day", [{ value: "today", label: "Today" }, { value: "tomorrow", label: "Tomorrow" }], ask.day, { label: "Day" })}</div>
    ${playerHtml(ask)}
    <div class="pf-field pf-sort-field"><span class="label">Sort by</span>
      <div class="pf-sorts">${segmented("pf-sort", SORTS, ask.sort, { label: "Sort the list by" })}${segmented("pf-sort", RANKS, ask.sort, { label: "Or rank everyone by" })}</div></div>
  </div>`;
}

/** The ask in words for the summary: "20+ points", "Hot (last 5 vs season) · 20+ points", "3+ threes · Steph Curry". */
function summaryWords(state) {
  const ask = state.ask;
  if (ask.sort === EDGES) return "Best edges (all stats)";
  const words = state.scan ? askText(state.scan) : "";
  const trend = ask.sort === "hot" ? "Hot (last 5 vs season)" : ask.sort === "cold" ? "Cold (last 5 vs season)" : "";
  const parts = [trend, words ? `${trend ? words : `${words[0].toUpperCase()}${words.slice(1)}`}` : "", namesText(ask.names)].filter(Boolean);
  return parts.join(" · ") || "Players";
}

/** The summary above the rows: the ask in words, when, the loading line, Refresh and Copy link. */
function summaryHtml(state) {
  const scan = state.scan;
  const ask = state.ask;
  const league = scanLeague(ask.league)?.title || "";
  const status = scan ? statusText(scan) : "";
  const polling = Boolean(scan && stillLoading(scan.loading) && state.pollUntil > Date.now());
  return `<div class="pf-summary">
    <div class="grow"><b>${esc(summaryWords(state))}</b> <span class="muted">· ${esc(league)} · ${esc(whenText(ask))}</span>
      ${scan?.as_of ? `<span class="muted small">· ESPN data from ${esc(ago(scan.as_of))}</span>` : ""}
      ${status ? `<p class="ps-loading" role="status">${icon("clock", { size: 14 })}<span>${esc(status)}${polling ? " · updating" : ""}</span></p>` : ""}</div>
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

/** The table's columns for a view: edges mode adds each row's stat; a yes/no stat has no projection column. */
function columnsOf(view) {
  const stat = view.edges;
  const proj = !view.yesNo;
  return { stat, proj, count: 8 + (stat ? 1 : 0) + (proj ? 1 : 0) };
}

function tableRow(row, i, ask, view, cols) {
  const href = playerHref(row);
  const why = (Array.isArray(row.why) ? row.why : []).filter((w) => w && !SHOWN_WHY.test(String(w))).slice(0, 2);
  const meta = [matchupText(row), row.start ? timeLabel(row.start) : "", row.minutes != null ? `${numText(row.minutes)} min` : ""].filter(Boolean).join(" · ");
  const edge = shownEdge(row);
  const yesNo = row.kind ? row.kind === "yes_no" : view.yesNo;
  const side = view.edges ? (row.side === "under" ? "under" : "over") : ask.side;
  return `<tr>
    <td class="num pf-rank">${esc(row.rank || i + 1)}</td>
    <th scope="row" class="pf-player">
      <div class="pf-who">${href ? `<a class="pf-name" href="${esc(href)}">${esc(row.name || "Player")}</a>` : `<b class="pf-name">${esc(row.name || "Player")}</b>`}${statusChip(row)}${flagChip(row, { names: view.names, yesNo })}${modelChip(row)}${view.trend ? trendChip(row) : ""}</div>
      <div class="muted small">${esc(meta)}</div>
      ${why.length ? `<div class="small pf-why">${why.map(esc).join(" · ")}</div>` : ""}
    </th>
    ${cols.stat ? `<td class="pf-stat">${esc(rowStatName(row, ask.league) || "–")}</td>` : ""}
    ${cols.proj ? `<td class="num pf-n">${yesNo ? '<span class="muted">–</span>' : esc(numText(row.projection))}</td>` : ""}
    <td class="pf-c">${chanceHtml(row.chance)}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "l5"))}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "l10"))}</td>
    <td class="num pf-h">${esc(hitsText(row.hits, "season"))}</td>
    <td class="pf-line small">${lineHtml(row, side, { edge: false })}</td>
    <td class="num pf-e${edge > 0 ? " pos" : ""}">${edge !== null ? esc(edgeText(edge)) : '<span class="muted">–</span>'}</td>
  </tr>`;
}

/** How the table is ordered, in words (its caption). */
function orderWords(ask) {
  if (ask.sort === EDGES) return "biggest edge first";
  if (ask.sort === "hot") return "hottest first (last 5 against the season)";
  if (ask.sort === "cold") return "coldest first (last 5 against the season)";
  return `sorted by ${SORTS.find((s) => s.value === ask.sort)?.label || "chance"}`;
}

function tableHtml(players, close, state, view) {
  const ask = state.ask;
  const cols = columnsOf(view);
  // The list's own sorts are header buttons; the server's rankings keep their order
  const head = (label, key, cls) => (view.list ? sortHead(label, key, ask, { cls }) : `<th scope="col" class="${cls}">${esc(label)}</th>`);
  const thead = `<thead><tr>
    <th scope="col" class="pf-rank">#</th><th scope="col">Player</th>
    ${cols.stat ? '<th scope="col" class="pf-stat">Stat</th>' : ""}
    ${cols.proj ? head("Proj.", "projection", "pf-n") : ""}${head("Chance", "chance", "pf-c")}
    <th scope="col" class="pf-h">L5</th>${head("L10", "l10", "pf-h")}<th scope="col" class="pf-h">Season</th>
    <th scope="col">Posted line</th>${head("Edge", "edge", "pf-e")}
  </tr></thead>`;
  const split = !view.names && players.length > TOP;
  const top = split ? players.slice(0, TOP) : players;
  const rest = split ? players.slice(TOP) : [];
  const also = rest.length ? `<tbody class="pf-also">
    <tr class="pf-group pf-also-head"><th scope="rowgroup" colspan="${cols.count}">Also:<span class="sr-only"> ${esc(alsoLabel(rest.length, { sort: ask.sort, edges: view.edges }).replace(/^Also: /, ""))}</span></th></tr>
    ${rest.map((row, i) => tableRow(row, top.length + i, ask, view, cols)).join("")}</tbody>` : "";
  const closeRows = close.length ? `<tbody class="pf-close">
    <tr class="pf-group"><td colspan="${cols.count}"><button type="button" class="pf-toggle" data-pf-close aria-expanded="${state.closeOpen}">${icon("chevron", { size: 16 })}Also close (${close.length})</button></td></tr>
    ${state.closeOpen ? close.map((row, i) => tableRow(row, players.length + i, ask, view, cols)).join("") : ""}</tbody>` : "";
  return `<div class="table-wrap pf-table-wrap"><table class="pf-table">
    <caption class="sr-only">${esc(`Players for ${askText(state.scan)} ${whenText(ask)}${view.names ? ` (${namesText(ask.names)})` : ""}, ${orderWords(ask)}`)}</caption>
    ${thead}<tbody>${top.map((row, i) => tableRow(row, i, ask, view, cols)).join("")}</tbody>${also}${closeRows}</table></div>`;
}

function cardsHtml(players, close, state, view) {
  const ask = state.ask;
  const item = (row, i) => rowItemHtml(row, i, { side: ask.side, view });
  return `${rowListsHtml(players, item, { label: `Players for ${askText(state.scan)}`, also: alsoLabel(players.length - TOP, { sort: ask.sort, edges: view.edges }), split: !view.names })}
    ${close.length ? `<details class="ps-close" data-pf-details${state.closeOpen ? " open" : ""}><summary>${icon("chevron", { size: 16 })}Also close (${close.length})</summary>
      <ol class="ps-list" aria-label="Also close">${close.map((row, i) => item(row, players.length + i)).join("")}</ol></details>` : ""}`;
}

/** Whether an empty list's own message says it all (loading, ESPN down, every game started): no notes above it. */
function emptySaysAll(scan) {
  return Boolean((stillLoading(scan.loading) && !scan.error) || scan.error || allStarted(scan));
}

/**
 * What to say when there are no rows: still loading, ESPN down, every game started ("Every game tonight has started,
 * so nobody is left to rank.", as every surface says it), the names it couldn't rank (their notes sit above), or none.
 */
function emptyHtml(scan, ask, view, names = []) {
  const leagueName = scanLeague(ask.league)?.title || "";
  const when = whenText(ask);
  if (stillLoading(scan.loading) && !scan.error) return `${notice(loadingText(scan.loading) || "Reading the day's games…", "info")}${skeleton("lines", 4)}`;
  if (scan.error) {
    return emptyState({
      iconName: "alert", title: `Couldn't read ${when === "tomorrow" ? "tomorrow" : "tonight"}'s ${leagueName} games`,
      body: "ESPN's scoreboard didn't answer just now. Try again in a minute.", actions: [{ label: "Refresh", attrs: "data-pf-refresh", primary: true }],
    });
  }
  if (allStarted(scan)) {
    return emptyState({
      iconName: "clock", title: startedText(scan, { when }),
      actions: ask.day === "today" ? [{ label: "Tomorrow's games", attrs: "data-pf-tomorrow", primary: true }] : [],
    });
  }
  if (view.names) {
    // Each name's note (not found, more than one match, game already started) is above; an older server sends none
    return emptyState({
      iconName: "search", title: `Nothing to show for ${namesText(ask.names)} ${when}`,
      body: names.length ? "" : "Check the spelling, or try tomorrow's games.", actions: [{ label: "Clear the name", attrs: "data-pf-clear-name", primary: true }],
    });
  }
  const noGames = (scan.leagues_without_games || []).length || (!scan.leagues?.length && !scan.games);
  if (noGames) {
    return emptyState({
      iconName: "today", title: `No ${leagueName} games ${when}`,
      body: "Try another league, or tomorrow's games.",
      actions: ask.day === "today" ? [{ label: "Tomorrow's games", attrs: "data-pf-tomorrow", primary: true }] : [],
    });
  }
  if (view.edges) {
    return emptyState({
      iconName: "target", title: "No posted line has an edge right now",
      body: "Lines move through the day: try again later, or tomorrow's games.",
      actions: ask.day === "today" ? [{ label: "Tomorrow's games", attrs: "data-pf-tomorrow", primary: true }] : [],
    });
  }
  if (view.trend) {
    return emptyState({ iconName: "trend", title: "Nobody has 5 games logged this season yet",
      body: "Hot and cold compare this season's last 5 games with the rest of the season (or with last season's, early in a season): try another stat or league." });
  }
  return emptyState({
    iconName: "search", title: `Nobody is projected for ${askText(scan)} yet`,
    body: ask.line ? "No posted lines for this stat yet: try a number instead." : "Try a lower number, another stat or tomorrow's games.",
    actions: !ask.line && ask.at > 1 && !statOf(ask.league, ask.stat)?.yes ? [{ label: `Try ${Math.max(1, ask.at - stepOf(ask.at) * 2)}+`, attrs: "data-pf-lower", primary: true }] : [],
  });
}

function resultsHtml(state) {
  const scan = state.scan;
  if (state.error) return state.error;
  if (!scan) return skeleton("lines", 6);
  const ask = state.ask;
  const view = viewOf(ask);
  const league = ask.league;
  const tag = (row) => ({ league, ...row });
  const order = (rows) => (view.list ? sortRows(rows, ask.sort, ask.side) : rows);
  const players = order((scan.players || []).filter((r) => r && typeof r === "object").map(tag));
  const close = order((scan.close || []).filter((r) => r && typeof r === "object").map(tag));
  const when = whenText(ask);
  // Each name it couldn't rank, then the board's state, worded as the chat's answer and the bot word them
  const names = nameNotes(scan, when);
  const board = boardNotes(scan, { when });
  if (!players.length && !close.length) {
    const above = emptySaysAll(scan) ? "" : notesHtml([...names, ...board]);
    return `${above ? `<div class="pf-notes">${above}</div>` : ""}${emptyHtml(scan, ask, view, names)}`;
  }
  const notes = notesHtml([...names, ...board]);
  const lead = players.length ? "" : `<p class="small pf-lead">Nobody is projected for ${esc(askText(scan))} yet: the closest are below.</p>`;
  const more = moreText(scan, { shown: players.length, finder: true, sort: ask.sort });
  const list = layout.mode === "phone" ? cardsHtml(players, close, { ...state, closeOpen: state.closeOpen || !players.length }, view)
    : tableHtml(players, close, { ...state, closeOpen: state.closeOpen || !players.length }, view);
  const forms = [...players, ...close].some((r) => r.model === "form");
  return `${notes ? `<div class="pf-notes">${notes}</div>` : ""}${lead}${list}${more ? `<p class="ps-note">${esc(more)}</p>` : ""}
    <p class="ps-note">L5, L10 and Season count the games at ${view.edges ? "each line" : "the number"}. ${view.trend ? "Hot and cold rank by the stat's average over this season's last 5 games against the rest of the season (or against last season's, early in a season). " : ""}${forms ? "\"Recent form\" rows are projected from the game log (recent games weigh most). " : ""}Edge is the chance × the posted price − 1, shown only when the price is for exactly this number and the line isn't old. Lines: DraftKings via ESPN. Player props are for reading here: the slip takes game picks. ${explain("player_finder", { label: "How the Player finder works" })}</p>`;
}

// --- styles ---------------------------------------------------------------------------------------------------

const CSS = `
@media (min-width: 840px) { body[data-route="players"] .shell > * { max-width: 1280px; } }
.view-players .pf-controls { display: grid; gap: 12px; }
.view-players .pf-grid { display: grid; gap: 14px 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.view-players .pf-grid > .pf-field:nth-child(n+3) { grid-column: 1 / -1; }
@media (min-width: 560px) { .view-players .pf-grid { gap: 14px; } }
@media (min-width: 760px) { .view-players .pf-grid > .pf-field:nth-child(n+4) { grid-column: auto; }
  .view-players .pf-grid > .pf-field.pf-sort-field { grid-column: 1 / -1; } }
@media (min-width: 1100px) { .view-players .pf-grid { grid-template-columns: minmax(120px, 0.7fr) minmax(170px, 1fr) minmax(410px, 2.2fr) minmax(170px, 0.8fr); }
  .view-players .pf-grid > .pf-field:nth-child(n+3) { grid-column: auto; }
  .view-players .pf-grid > .pf-field.pf-player-field { grid-column: 1 / 3; }
  .view-players .pf-grid > .pf-field.pf-sort-field { grid-column: 3 / -1; } }
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
.view-players select:disabled { opacity: 0.6; }
.view-players .pf-search { position: relative; display: flex; align-items: center; min-width: 0; }
.view-players .pf-search > .icon { position: absolute; left: 12px; color: var(--text-3); pointer-events: none; }
.view-players .pf-search .input { padding-left: 36px; padding-right: 46px; }
.view-players .pf-search .input:disabled { opacity: 0.6; }
.view-players .pf-clear { position: absolute; right: 1px; top: 50%; transform: translateY(-50%); width: var(--tap); height: var(--tap); display: grid; place-items: center;
  border-radius: 12px; color: var(--text-2); }
.view-players .pf-clear:hover { color: var(--text); }
.view-players .pf-search .input:placeholder-shown ~ .pf-clear, .view-players .pf-search .input:disabled ~ .pf-clear { display: none; }
.view-players .pf-hint { margin-top: 4px; }
.view-players .pf-sorts { display: flex; flex-wrap: wrap; gap: 8px; min-width: 0; }
.view-players .pf-sort-field .seg { flex: 0 1 auto; max-width: 100%; flex-wrap: wrap; }
.view-players .pf-sort-field .seg button { flex: 1 1 auto; padding: 0 8px; font-size: 13.5px; }
@media (max-width: 559px) { .view-players .pf-sort-field .seg { flex: 1 1 100%; } }
.view-players .pf-summary { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 14px 0 10px; }
.view-players .pf-summary .grow { flex: 1 1 260px; min-width: 0; overflow-wrap: anywhere; }
.view-players .pf-summary .ps-loading { margin-top: 4px; }
.view-players .pf-tools { display: flex; gap: 8px; flex-wrap: wrap; }
.view-players .pf-notes { display: grid; gap: 4px; margin: 0 0 10px; }
.view-players .pf-notes .ps-note { margin-top: 0; }
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
.view-players .pf-stat { font-weight: 650; min-width: 90px; }
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
.view-players .pf-group td, .view-players .pf-group th { padding: 0 8px; background: var(--card-2); }
.view-players .pf-also-head th { padding: 8px; font-size: 13.5px; font-weight: 700; color: var(--text-2); text-transform: none; letter-spacing: 0; }
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
        : active.matches("[data-pf-line]") ? "[data-pf-line]" : active.dataset.pfStep ? `[data-pf-step="${active.dataset.pfStep}"]`
          : active.matches("[data-pf-clear-name]") ? '[data-pf="player"]' : null)
      : null;
    box.innerHTML = controlsHtml(state.ask);
    const again = key ? box.querySelector(key) : null;
    if (again && !again.disabled) {
      again.focus({ preventScroll: true });
      if (again.matches('[data-pf="player"]')) again.setSelectionRange(again.value.length, again.value.length);
    }
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
      data = await api(scanQuery({
        league: ask.league, stat: ask.stat, threshold: ask.line ? null : ask.at, side: ask.side, day: ask.day, n: MAX_ROWS,
        sort: ranked(ask.sort) ? ask.sort : "", names: ask.names,
      }), { signal: controller.signal });
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
      if (stillLoading(state.scan.loading)) {
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
  /** The name box's words as the filter; false when they name the same players. */
  const setNames = (value) => {
    const names = cleanNames(value);
    if (names.join("\n") === state.ask.names.join("\n")) return false;
    state.ask.names = names;
    return true;
  };
  /** A new sort: the list's own sorts reorder the rows on the screen; the server's rankings (and leaving one) scan again. */
  const setSort = (value) => {
    if (!SORT_KEYS.includes(value) || value === state.ask.sort) return;
    const before = state.ask.sort;
    state.ask.sort = value;
    if (value === EDGES) state.ask.names = []; // Best edges covers every player
    if (ranked(value) || ranked(before)) changed();
    else resorted();
  };

  root.addEventListener("change", (event) => {
    const el = event.target;
    if (el.matches('[data-pf="league"]')) {
      const league = scanLeague(el.value) ? el.value : DEFAULT_LEAGUE;
      const keep = statOf(league, state.ask.stat);
      state.ask = readAsk(askQuery(state.ask, { league, stat: keep ? state.ask.stat : "", at: keep && !keep.yes ? (state.ask.line ? "line" : String(state.ask.at)) : "",
        side: keep ? state.ask.side : "over" }));
      changed();
    } else if (el.matches('[data-pf="stat"]')) {
      const spec = statOf(state.ask.league, el.value);
      if (!spec) return;
      state.ask = readAsk(askQuery(state.ask, { stat: spec.key, at: state.ask.line && !spec.yes ? "line" : String(spec.start), side: spec.yes ? "over" : state.ask.side }));
      changed();
    } else if (el.matches('[data-pf="at"]')) {
      clearTimeout(typeTimer);
      if (setAt(el.value)) changed();
      else el.value = String(state.ask.at);
    } else if (el.matches('[data-pf="player"]')) {
      clearTimeout(typeTimer);
      if (setNames(el.value)) changed({ controls: false });
    }
  });
  root.addEventListener("input", (event) => {
    const el = event.target;
    if (el.matches('[data-pf="player"]')) {
      clearTimeout(typeTimer);
      typeTimer = setTimeout(() => {
        if (setNames(el.value)) changed({ controls: false });
      }, NAME_MS);
      return;
    }
    if (!el.matches('[data-pf="at"]')) return;
    clearTimeout(typeTimer);
    typeTimer = setTimeout(() => {
      const value = Number(el.value);
      if (Number.isFinite(value) && value > 0 && value <= MAX_THRESHOLD && setAt(value)) changed({ controls: false });
    }, TYPE_MS);
  });
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    if (event.target.matches('[data-pf="at"]')) {
      clearTimeout(typeTimer);
      if (setAt(event.target.value)) changed({ controls: false });
    } else if (event.target.matches('[data-pf="player"]')) {
      event.preventDefault();
      clearTimeout(typeTimer);
      if (setNames(event.target.value)) changed({ controls: false });
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
      } else if (seg.dataset.seg === "pf-sort") {
        setSort(value);
      }
      return;
    }
    const sortBy = t.closest("[data-pf-sortby]");
    if (sortBy) {
      setSort(sortBy.dataset.pfSortby);
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
    if (t.closest("[data-pf-clear-name]")) {
      // The name box empty again (and the scan without the filter); the caret goes back to the box
      clearTimeout(typeTimer);
      const box = $('[data-pf="player"]');
      if (box) box.value = "";
      if (setNames("")) changed();
      $('[data-pf="player"]')?.focus({ preventScroll: true });
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
