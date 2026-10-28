// The player scan's pieces, shared by Ask AI's player_scan card (ask/cards.js) and the Player finder (finder.js).
//
// GET /api/players/scan?league=&stat=&threshold=&side=&day=&n= (Lab.player_scan over engine/scan.py) ranks today's
// or tomorrow's players for a stat and a number ("20+ points"): {ask ("20+ points"), stat, stat_label, threshold,
// side, day, leagues, players: [row], close: [row], more, games, loading?: {games_ready, games_total}, as_of,
// leagues_without_games?}. Each row: {rank, athlete_id, name, team, opponent, home, game, event_id, start, status,
// status_tag ("Q", "GTD", "P"), minutes?, threshold, projection, chance, chance_source, model ("nba" | "form"),
// model_label, hits: {l5: [made, games], l10, season}, avg: {l5, l10, season}, line?: {kind ("over_under" |
// "milestone" | "yes"), line, label, over, under, yes ({decimal, american}), book, as_of, stale, matches}, edge?,
// why: [short reasons], league, addable}. ESPN's game logs and DraftKings lines through the shared cache: a scan
// never spends Odds API quota and costs no AI allowance.
//
// Every number shown is the server's: these functions only lay rows out. A row offers "Add to slip" only when the
// server says the slip takes it (``addable`` with a game-pick ``ref``); a player prop shows its price instead.
//
// SCAN_TABLES mirrors engine/scan.py's STATS (the keys, in the engine's order, with friendly names, a short name
// and the number the finder starts at); a yes/no stat ("anytime TD", "double-double") has its own fixed number.

import { esc, pct, priceText, timeLabel } from "../../ui.js";
import { icon } from "../../icons.js";

// The stats a league's scan takes (engine/scan.py STATS): {key, name, short, start, yes}
const SCAN_TABLES = {
  "basketball": [
    { "key": "pts", "name": "Points", "short": "PTS", "start": 20, "yes": false },
    { "key": "reb", "name": "Rebounds", "short": "REB", "start": 10, "yes": false },
    { "key": "ast", "name": "Assists", "short": "AST", "start": 8, "yes": false },
    { "key": "fg3m", "name": "Threes (3PM)", "short": "3PM", "start": 3, "yes": false },
    { "key": "stl", "name": "Steals", "short": "STL", "start": 2, "yes": false },
    { "key": "blk", "name": "Blocks", "short": "BLK", "start": 2, "yes": false },
    { "key": "stocks", "name": "Steals + blocks", "short": "STL+BLK", "start": 3, "yes": false },
    { "key": "pra", "name": "PRA (P+R+A)", "short": "PRA", "start": 30, "yes": false },
    { "key": "pr", "name": "Pts + reb (P+R)", "short": "P+R", "start": 25, "yes": false },
    { "key": "pa", "name": "Pts + ast (P+A)", "short": "P+A", "start": 25, "yes": false },
    { "key": "ra", "name": "Reb + ast (R+A)", "short": "R+A", "start": 12, "yes": false },
    { "key": "dd", "name": "Double-double", "short": "DD", "start": 2, "yes": true },
    { "key": "td", "name": "Triple-double", "short": "TD", "start": 3, "yes": true }
  ],
  "football": [
    { "key": "pass_yds", "name": "Passing yards", "short": "Pass yds", "start": 250, "yes": false },
    { "key": "pass_td", "name": "Passing TDs", "short": "Pass TD", "start": 2, "yes": false },
    { "key": "rush_yds", "name": "Rushing yards", "short": "Rush yds", "start": 75, "yes": false },
    { "key": "rec_yds", "name": "Receiving yards", "short": "Rec yds", "start": 75, "yes": false },
    { "key": "rec", "name": "Receptions", "short": "Rec", "start": 5, "yes": false },
    { "key": "rush_rec_yds", "name": "Rush + rec yards", "short": "Rush+rec", "start": 100, "yes": false },
    { "key": "anytime_td", "name": "Anytime TD", "short": "ATTD", "start": 1, "yes": true }
  ],
  "hockey": [
    { "key": "goals", "name": "Goals", "short": "G", "start": 1, "yes": false },
    { "key": "assists", "name": "Assists", "short": "A", "start": 1, "yes": false },
    { "key": "points", "name": "Points", "short": "PTS", "start": 1, "yes": false },
    { "key": "sog", "name": "Shots on goal", "short": "SOG", "start": 3, "yes": false },
    { "key": "saves", "name": "Saves", "short": "SV", "start": 25, "yes": false },
    { "key": "anytime_goal", "name": "Anytime goal", "short": "Goal", "start": 1, "yes": true }
  ],
  "baseball": [
    { "key": "hits", "name": "Hits", "short": "H", "start": 1, "yes": false },
    { "key": "tb", "name": "Total bases (TB)", "short": "TB", "start": 2, "yes": false },
    { "key": "hr", "name": "Home runs", "short": "HR", "start": 1, "yes": false },
    { "key": "rbi", "name": "RBIs", "short": "RBI", "start": 1, "yes": false },
    { "key": "runs", "name": "Runs", "short": "R", "start": 1, "yes": false },
    { "key": "hrr", "name": "H+R+RBI", "short": "H+R+RBI", "start": 2, "yes": false },
    { "key": "k", "name": "Strikeouts (Ks)", "short": "K", "start": 6, "yes": false }
  ],
  "soccer": [
    { "key": "goals", "name": "Goals", "short": "G", "start": 1, "yes": false },
    { "key": "shots", "name": "Shots", "short": "Shots", "start": 2, "yes": false },
    { "key": "sot", "name": "Shots on target", "short": "SOT", "start": 1, "yes": false },
    { "key": "assists", "name": "Assists", "short": "A", "start": 1, "yes": false }
  ]
};

// The leagues the scan covers (engine/scan.py LEAGUE_ORDER), with the table each one reads
export const SCAN_LEAGUES = [
  { key: "basketball_nba", title: "NBA", table: "basketball", group: "us" },
  { key: "basketball_wnba", title: "WNBA", table: "basketball", group: "us" },
  { key: "americanfootball_nfl", title: "NFL", table: "football", group: "us" },
  { key: "icehockey_nhl", title: "NHL", table: "hockey", group: "us" },
  { key: "baseball_mlb", title: "MLB", table: "baseball", group: "us" },
  { key: "soccer_epl", title: "Premier League", table: "soccer", group: "soccer" },
  { key: "soccer_spain_la_liga", title: "La Liga", table: "soccer", group: "soccer" },
  { key: "soccer_italy_serie_a", title: "Serie A", table: "soccer", group: "soccer" },
  { key: "soccer_germany_bundesliga", title: "Bundesliga", table: "soccer", group: "soccer" },
  { key: "soccer_france_ligue_one", title: "Ligue 1", table: "soccer", group: "soccer" },
  { key: "soccer_usa_mls", title: "MLS", table: "soccer", group: "soccer" },
  { key: "soccer_uefa_champs_league", title: "Champions League", table: "soccer", group: "soccer" },
  { key: "soccer_uefa_europa_league", title: "Europa League", table: "soccer", group: "soccer" },
  { key: "soccer_efl_champ", title: "Championship", table: "soccer", group: "soccer" },
  { key: "soccer_netherlands_eredivisie", title: "Eredivisie", table: "soccer", group: "soccer" },
  { key: "soccer_portugal_primeira_liga", title: "Primeira Liga", table: "soccer", group: "soccer" },
  { key: "soccer_mexico_ligamx", title: "Liga MX", table: "soccer", group: "soccer" },
  { key: "soccer_spl", title: "Scottish Premiership", table: "soccer", group: "soccer" },
  { key: "soccer_brazil_campeonato", title: "Brasileirão", table: "soccer", group: "soccer" },
  { key: "soccer_argentina_primera_division", title: "Argentina Primera", table: "soccer", group: "soccer" },
];
export const DEFAULT_LEAGUE = "basketball_nba";
export const SCAN_DAYS = ["today", "tomorrow"];
export const MAX_THRESHOLD = 999; // the route takes numbers above 0 and at most 999
export const MAX_ROWS = 25; // the route's n

/** A league of the scan ({key, title, table}), or null. */
export function scanLeague(key) {
  return SCAN_LEAGUES.find((l) => l.key === key) || null;
}

/** The stats a league's scan takes, in the engine's order ([] for a league it doesn't cover). */
export function statsFor(league) {
  return SCAN_TABLES[scanLeague(league)?.table] || [];
}

/** One stat of a league ({key, name, short, start, yes}), or null. */
export function statOf(league, key) {
  return statsFor(league).find((s) => s.key === key) || null;
}

// --- reading a scan ---------------------------------------------------------------------------------------

/** [made, games] of a row's hits in a window ("l5", "l10", "season"), or null. */
export function hitsPair(hits, window) {
  const pair = hits?.[window];
  if (!Array.isArray(pair) || pair.length !== 2) return null;
  const [made, games] = pair.map(Number);
  return Number.isFinite(made) && Number.isFinite(games) && games > 0 ? [made, games] : null;
}

/** "9/10" for a window's hits; "–" with no games. */
export function hitsText(hits, window) {
  const pair = hitsPair(hits, window);
  return pair ? `${pair[0]}/${pair[1]}` : "–";
}

/** Hits as words for a screen reader: "9 of last 10", "30 of 45 this season". */
function hitsWords(hits, window) {
  const pair = hitsPair(hits, window);
  if (!pair) return "no games";
  return window === "season" ? `${pair[0]} of ${pair[1]} this season` : `${pair[0]} of last ${pair[1]}`;
}

/** A share of games (0..1) for sorting by L10 hits; -1 with none. */
export function hitShare(hits, window = "l10") {
  const pair = hitsPair(hits, window);
  return pair ? pair[0] / pair[1] : -1;
}

/** A number as the scan gives it (one decimal at most): "31.2", "7". */
export function numText(value) {
  const n = Number(value);
  if (value === null || value === undefined || value === "" || !Number.isFinite(n)) return "–";
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

/** "+4.2%" / "−1.3%": the row's edge (chance × the posted price − 1), or "". */
export function edgeText(edge) {
  const v = Number(edge) * 100;
  if (edge === null || edge === undefined || !Number.isFinite(v)) return "";
  const text = Math.abs(v) < 0.05 ? "0.0" : Math.abs(v).toFixed(1);
  return `${v >= 0.05 ? "+" : v <= -0.05 ? "−" : ""}${text}%`;
}

/** "DAL v PHX": the player's team v the opponent. */
export function matchupText(row) {
  const team = row?.team || "";
  const opponent = row?.opponent || "";
  if (team && opponent) return `${team} v ${opponent}`;
  return String(team || opponent || row?.game || "");
}

/** The ask in words: the scan's own ("20+ points"), else from its fields. */
export function askText(scan) {
  if (typeof scan?.ask === "string" && scan.ask.trim()) return scan.ask.trim();
  if (scan?.ask?.label) return String(scan.ask.label);
  const label = scan?.stat_label || scan?.stat || "the stat";
  const n = Number(scan?.threshold);
  if (scan?.threshold == null || !Number.isFinite(n)) return `${label} on the posted line`;
  return scan?.side === "under" ? `under ${n - 0.5} ${label}` : `${n}+ ${label}`;
}

/** "tonight" or "tomorrow". */
export function whenText(scan) {
  return scan?.day === "tomorrow" ? "tomorrow" : "tonight";
}

/** The posted line's pieces: {label ("20+", "O 24.5", "Yes"), price (the person's format), book, stale}, or null. */
export function lineParts(line, side = "over") {
  if (!line || typeof line !== "object") return null;
  const under = side === "under";
  let label;
  let price;
  if (line.kind === "yes" || (line.yes && !line.over && !line.under)) {
    label = "Yes";
    price = line.yes;
  } else if (line.kind === "milestone") {
    label = line.label || (line.line != null ? `${Number(line.line) + 0.5}+` : "Milestone");
    price = line.over;
  } else {
    const at = line.line != null ? numText(line.line) : "";
    label = `${under ? "U" : "O"} ${at}`.trim();
    price = under ? line.under || null : line.over || null;
  }
  return { label, price: price ? priceText(price) : "", book: line.book || "", stale: Boolean(line.stale), exact: line.matches !== false };
}

/** The slip leg a row offers (only a game pick the server says the slip takes), or null. */
export function rowLeg(row) {
  const ref = row?.addable === true ? row.ref : null;
  if (!ref?.league || ref.id == null || !["h2h", "spreads", "totals"].includes(ref.market)) return null;
  const price = Number(row.price);
  return {
    ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
    label: row.label || row.name, game: row.game, start: row.start, league: ref.league,
    price: Number.isFinite(price) ? price : undefined,
  };
}

/** The player page for a row ("" when its id isn't one the route takes). */
export function playerHref(row) {
  const league = row?.league;
  const id = String(row?.athlete_id ?? "");
  return league && /^[a-z0-9_]{2,64}$/.test(league) && /^[0-9]{1,12}$/.test(id) ? `#/player/${encodeURIComponent(league)}/${id}` : "";
}

/** GET /api/players/scan's query for an ask ({league?, stat, threshold?, side?, day?, n?}). */
export function scanQuery({ league = null, stat, threshold = null, side = "over", day = "today", n = 10 } = {}) {
  const q = new URLSearchParams();
  if (league) q.set("league", league);
  q.set("stat", String(stat || ""));
  if (threshold !== null && threshold !== undefined && threshold !== "") q.set("threshold", String(threshold));
  if (side === "under") q.set("side", "under");
  q.set("day", day === "tomorrow" ? "tomorrow" : "today");
  q.set("n", String(Math.max(1, Math.min(MAX_ROWS, Number(n) || 10))));
  return `/api/players/scan?${q.toString()}`;
}

/** The Player finder's address for an ask: #/players?league=&stat=&at=<n|line>&day=[&side=under][&sort=]. */
export function finderHash({ league = DEFAULT_LEAGUE, stat = "pts", threshold = null, side = "over", day = "today", sort = "" } = {}) {
  const q = new URLSearchParams();
  q.set("league", league || DEFAULT_LEAGUE);
  q.set("stat", stat || "pts");
  q.set("at", threshold === null || threshold === undefined || threshold === "" ? "line" : String(threshold));
  q.set("day", day === "tomorrow" ? "tomorrow" : "today");
  if (side === "under") q.set("side", "under");
  if (sort && sort !== "chance") q.set("sort", sort);
  return `#/players?${q.toString()}`;
}

// --- drawing rows -----------------------------------------------------------------------------------------

const STATUS_KIND = { Q: "warn", GTD: "warn", P: "muted" };

/** The injury chip ("Q", "GTD", "P") with the full status for screen readers and on hover; "" when healthy. */
export function statusChip(row) {
  const tag = row?.status_tag;
  if (!tag) return "";
  const words = String(row.status || tag);
  return `<span class="badge ${STATUS_KIND[tag] || "muted"} ps-status" title="${esc(words)}"><span aria-hidden="true">${esc(tag)}</span><span class="sr-only">${esc(words)}</span></span>`;
}

/** "Recent form" (the form model) or "NBA model". */
export function modelChip(row) {
  if (row?.model === "form") return `<span class="ps-model form" title="${esc(row.model_label || "from recent form")}">${icon("trend", { size: 13 })}Recent form</span>`;
  if (row?.model === "nba") return `<span class="ps-model nba" title="${esc(row.model_label || "NBA model")}">${icon("bot", { size: 13 })}NBA model</span>`;
  return "";
}

/** The chance with a small bar (the width is the chance; the number is always written). */
export function chanceHtml(chance) {
  const p = Number(chance);
  const width = Number.isFinite(p) ? Math.max(0, Math.min(100, p * 100)) : 0;
  return `<span class="ps-chance"><b class="num">${esc(pct(chance))}</b><span class="ps-bar" aria-hidden="true"><i style="width:${width.toFixed(1)}%"></i></span></span>`;
}

/** "DraftKings 20+ −250 · Edge +4.2%" with an old-line chip, or "No line posted" (``edge`` false leaves the edge out). */
export function lineHtml(row, side, { edge: withEdge = true } = {}) {
  const parts = lineParts(row?.line, side);
  const edge = withEdge ? edgeText(row?.edge) : "";
  if (!parts) return `<span class="muted">No line posted</span>`;
  return `<span class="ps-line-main">${parts.book ? `<span class="muted">${esc(parts.book)}</span> ` : ""}<b>${esc(parts.label)}</b>${parts.price ? ` <b class="num">${esc(parts.price)}</b>` : ""}</span>${
    edge ? ` <span class="ps-edge${Number(row.edge) > 0 ? " pos" : ""}">Edge <b class="num">${esc(edge)}</b></span>` : ""}${
    parts.stale ? ` <span class="badge warn">${icon("clock", { size: 12 })}Old line</span>` : ""}`;
}

/** The tip time and minutes: "Tonight 7:30 PM · 34 min". */
function whenLine(row) {
  return [matchupText(row), row?.start ? timeLabel(row.start) : "", row?.minutes != null ? `${numText(row.minutes)} min` : ""].filter(Boolean).join(" · ");
}

/**
 * One row as a card item (Ask AI's card and the finder on phones): rank, name and status, team v opponent and the
 * tip time, the projection, the chance with its bar, the hits (L5, L10, season), the posted line and price, the
 * model, a short why, and "View player" (plus "Add to slip" when the row allows it).
 * ``addButton(row, i)`` draws the slip button for an addable row (the caller's own handler).
 */
export function rowItemHtml(row, i, { side = "over", addButton = null } = {}) {
  const href = playerHref(row);
  const why = (Array.isArray(row.why) ? row.why : []).filter(Boolean).slice(0, 3);
  const leg = rowLeg(row);
  const rank = row.rank || i + 1;
  const name = esc(row.name || "Player");
  return `<li class="ps-item">
    <span class="ps-rank" aria-label="${esc(`Number ${rank}`)}">${esc(rank)}</span>
    <div class="ps-body">
      <div class="ps-head"><b class="ps-name">${name}</b>${statusChip(row)}${modelChip(row)}</div>
      <div class="muted small ps-when">${esc(whenLine(row))}</div>
      <dl class="ps-nums">
        <div><dt>Proj.</dt><dd class="num">${esc(numText(row.projection))}</dd></div>
        <div class="ps-nums-chance"><dt>Chance</dt><dd>${chanceHtml(row.chance)}</dd></div>
        <div><dt>L5</dt><dd class="num" aria-label="${esc(hitsWords(row.hits, "l5"))}">${esc(hitsText(row.hits, "l5"))}</dd></div>
        <div><dt>L10</dt><dd class="num" aria-label="${esc(hitsWords(row.hits, "l10"))}">${esc(hitsText(row.hits, "l10"))}</dd></div>
        <div><dt>Season</dt><dd class="num" aria-label="${esc(hitsWords(row.hits, "season"))}">${esc(hitsText(row.hits, "season"))}</dd></div>
      </dl>
      <div class="small ps-line">${lineHtml(row, side)}</div>
      ${why.length ? `<p class="small ps-why">${why.map(esc).join(" · ")}</p>` : ""}
    </div>
    <div class="ps-actions">${href ? `<a class="btn btn-secondary small" href="${esc(href)}" aria-label="${esc(`View ${row.name || "player"}`)}">${icon("user", { size: 16 })}View player</a>` : ""}${leg && addButton ? addButton(row, i) : ""}</div>
  </li>`;
}

/** The loading line: "8 of 10 games loaded" ("Reading today's games…" before any count), or "". */
export function loadingText(loading) {
  if (!loading || typeof loading !== "object") return "";
  const ready = Number(loading.games_ready);
  const total = Number(loading.games_total);
  if (!Number.isFinite(total) || total <= 0) return "Reading the day's games…";
  if (Number.isFinite(ready) && ready >= total) return "";
  return `${Number.isFinite(ready) ? ready : 0} of ${total} games loaded`;
}

// --- styles (added once; both Ask AI and the finder use them) ----------------------------------------------------

const CSS = `
.ps-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.ps-item {
  display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; align-items: start; gap: 4px 12px; padding: 10px 12px;
  border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); min-width: 0;
}
.ps-rank {
  width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; font-size: 13.5px;
  background: var(--accent-soft); color: var(--text); font-variant-numeric: tabular-nums;
}
.ps-body { display: grid; gap: 4px; min-width: 0; }
.ps-head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; min-width: 0; }
.ps-name { font-size: 15px; min-width: 0; overflow-wrap: anywhere; }
.ps-when { overflow-wrap: anywhere; }
.ps-status { padding: 1px 7px; }
.ps-model { display: inline-flex; align-items: center; gap: 3px; font-size: 12px; font-weight: 600; color: var(--text-2); white-space: nowrap; }
.ps-model .icon { flex: none; }
.ps-model.form { color: var(--violet); }
.ps-nums { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 2px 0 0; }
.ps-nums > div { flex: 0 0 auto; min-width: 40px; }
.ps-nums dt { font-size: 11.5px; color: var(--text-3); font-weight: 650; text-transform: uppercase; letter-spacing: 0.02em; }
.ps-nums dd { margin: 0; font-weight: 750; font-size: 15px; font-variant-numeric: tabular-nums; }
.ps-nums .ps-nums-chance { flex: 0 1 150px; min-width: 112px; }
.ps-chance { display: inline-flex; align-items: center; gap: 6px; min-width: 0; width: 100%; }
.ps-chance b { flex: none; min-width: 34px; }
.ps-bar { position: relative; flex: 1 1 auto; height: 6px; min-width: 24px; max-width: 120px; border-radius: 999px; background: var(--card-3); overflow: hidden; }
.ps-bar i { position: absolute; inset: 0 auto 0 0; border-radius: inherit; background: var(--accent); }
.ps-line { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; color: var(--text); min-width: 0; }
.ps-line-main { min-width: 0; overflow-wrap: anywhere; }
.ps-edge { color: var(--text-2); white-space: nowrap; }
.ps-edge.pos b { color: var(--good-ink); }
.ps-why { color: var(--text-2); margin: 0; line-height: 1.45; overflow-wrap: anywhere; }
.ps-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.ps-actions:empty { display: none; }
.ps-actions .btn { flex: 0 1 auto; white-space: nowrap; }
/* a narrow card: the buttons go under the numbers, full width */
@media (max-width: 559px) {
  .ps-item { grid-template-columns: 28px minmax(0, 1fr); }
  .ps-nums .ps-nums-chance { flex: 1 1 calc(100% - 76px); } /* the projection and the chance, then the three hit windows */
  .ps-actions { grid-column: 2; margin-top: 2px; }
  .ps-actions .btn { flex: 1 1 140px; }
}
.ps-loading { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin: 0; font-size: 13.5px; color: var(--text-2); }
.ps-loading > .icon { flex: none; }
.ps-loading .chip { margin-left: auto; }
.ps-close > summary { cursor: pointer; display: flex; align-items: center; gap: 6px; min-height: var(--tap); font-weight: 650; font-size: 14px; color: var(--text-2); list-style: none; }
.ps-close > summary::-webkit-details-marker { display: none; }
.ps-close > summary .icon { transition: transform var(--speed); }
.ps-close[open] > summary .icon { transform: rotate(90deg); }
.ps-close[open] > summary { margin-bottom: 6px; }
.ps-note { margin: 0; font-size: 12.5px; color: var(--text-2); line-height: 1.45; }
`;

export function ensureScanStyles() {
  if (document.getElementById("scan-styles")) return;
  const style = document.createElement("style");
  style.id = "scan-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
