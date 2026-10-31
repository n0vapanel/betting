// The player scan's pieces, shared by Ask AI's player_scan card (ask/cards.js) and the Player finder (finder.js).
//
// GET /api/players/scan?league=&stat=&threshold=&side=&day=&n=[&sort=][&player=...] (Lab.player_scan over
// engine/scan.py) ranks today's or tomorrow's players for a stat and a number ("20+ points"): {mode ("scan" |
// "edges"), ask ("20+ points"), stat, stat_label, threshold, side, day, leagues, players: [row], close: [row], more,
// games (the games still to come), games_started, error?, loading?: {games_ready, games_total, failed}, as_of,
// leagues_without_games?, and for a name filter missing_names? ([name]: matched nobody), ambiguous_names? ([{name,
// candidates}]: matched several players) and started_names? ([{name, player, game}]: that player's game has started)}.
// Each row: {rank, athlete_id, name, team, opponent, home, game, event_id, start, status, status_tag ("Q", "GTD",
// "P"), minutes?, threshold, kind ("count" | "yards" | "yes_no"), projection (null for a yes/no stat), chance,
// chance_source, model ("nba" | "form"), model_label, hits: {l5: [made, games], l10, season}, avg: {l5, l10, season},
// trend?: {l5, season, diff, vs ("season" | "last_season")}, line?: {kind ("over_under" | "milestone" | "yes"), line,
// label, over, under, yes ({decimal, american}), book, as_of, stale, matches}, edge?, why: [short reasons], league,
// addable, qualifies? (shown for a name filter's rows) and ruled_out? (a named player who is out)}. ``sort`` hot or
// cold ranks by the trend (this season's last 5 games' average against the rest of this season's, or against last
// season's while this season is short); edges mode (``sort=edge`` with no stat) ranks every stat's posted lines by
// edge, each row with its own stat, stat_label and side. ESPN's game logs and DraftKings lines through the shared
// cache: a scan never spends Odds API quota and costs no AI allowance.
//
// Every number shown is the server's: these functions only lay rows out. A row offers "Add to slip" only when the
// server says the slip takes it (``addable`` with a game-pick ``ref``); a player prop shows its price instead. What
// each piece shows: a yes/no stat only its chance (no projection); rows ranked by edge (posted lines, edges mode)
// their edge beside the chance; hot and cold their trend chip ("L5 28.4 vs 24.1 season", or "… last season"); a name
// filter whether each row qualifies, and a note for each name it couldn't rank (not found, more than one match, game
// already started); and the board's own state (games still loading or missing data, games already started, ESPN
// down). The notes and the "N more" line read as the chat's answer and the bot say them. An edge is never shown on
// an old line, and an under is never shown a milestone's over-only price.
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
export const MAX_NAMES = 5; // the route takes ``player`` up to 5 times,
export const MAX_NAME = 60; // each at most 60 characters
export const TOP = 5; // rows listed first; the rest that qualify follow under "Also:" (as the chat's answer says them)
export const TREND_SORTS = ["hot", "cold"]; // the server ranks everyone by the last 5 against the season
export const EDGES = "edges"; // the finder's "Best edges (all stats)": the route's sort=edge with no stat

// A piece that is only a name's suffix: it belongs to the name before it ("Michael Porter, Jr.")
const NAME_SUFFIX = /^(?:jr|sr|ii|iii|iv|2nd|3rd)\.?$/i;
/** Text cut to ``max`` characters (code points, as the route counts them), then trimmed. */
const cutText = (text, max) => Array.from(text).slice(0, max).join("").trim();

/**
 * Player names as the route takes them, by the rule the bot's finder button shares (bot/chat.py _finder_names):
 * accents composed first (NFC, as the route's pattern wants them); commas separate names ("steph, lebron"); in each
 * piece every character that isn't a letter (any alphabet), a digit, a space or one of . ' ’ - ( ) becomes a space;
 * spaces collapsed and trimmed, at most MAX_NAME characters; a piece that is only a suffix (Jr, Sr, II, III, IV, 2nd
 * or 3rd, any case, with a period or not) joins the name before it ("Michael Porter, Jr." is "Michael Porter Jr.",
 * still at most MAX_NAME); then no empty names, no repeats (ignoring case) and at most MAX_NAMES. So "Marvin Bagley
 * 3rd" and "Marcus Sutton (MIA)" stay whole, and markup never survives ("<img onerror=…>" is "img onerror").
 */
export function cleanNames(values) {
  const pieces = [];
  for (const value of Array.isArray(values) ? values : [values]) {
    for (const part of String(value ?? "").normalize("NFC").split(",")) {
      const name = cutText(part.replace(/[^\p{L}\p{Nd} .'’()-]/gu, " ").replace(/ +/g, " ").trim(), MAX_NAME);
      if (!name) continue;
      if (pieces.length && NAME_SUFFIX.test(name)) pieces[pieces.length - 1] = cutText(`${pieces.at(-1)} ${name}`, MAX_NAME);
      else pieces.push(name);
    }
  }
  const out = [];
  for (const name of pieces) {
    if (!out.some((n) => n.toLowerCase() === name.toLowerCase())) out.push(name);
  }
  return out.slice(0, MAX_NAMES);
}

/** "Steph Curry", "Steph Curry and LeBron", "A, B and C". */
export function namesText(names) {
  const list = (names || []).map(String);
  return list.length > 1 ? `${list.slice(0, -1).join(", ")} and ${list.at(-1)}` : list[0] || "";
}

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

/** The edge a row may show: its own, but none on an old (stale) line, whose price may have moved; null when none. */
export function shownEdge(row) {
  const edge = row?.edge;
  if (edge === null || edge === undefined || edge === "" || !Number.isFinite(Number(edge))) return null;
  return row?.line?.stale ? null : Number(edge);
}

/**
 * Whether a row is a yes/no stat (anytime TD, anytime goal, double-double): the server's ``kind``, else the stat's
 * own table entry (an older server). Such a row has a chance and no projection worth showing.
 */
export function isYesNo(row, scan = null) {
  if (row?.kind) return row.kind === "yes_no";
  const league = row?.league || scan?.leagues?.[0] || null;
  return Boolean(league && statOf(league, row?.stat || scan?.stat)?.yes);
}

/** An edges-mode row's stat in words ("Points", "PRA (P+R+A)"): the finder's own name, else the server's label. */
export function rowStatName(row, league = null) {
  const spec = statOf(row?.league || league, row?.stat);
  const label = String(row?.stat_label || row?.stat || "");
  return spec?.name || (label ? `${label[0].toUpperCase()}${label.slice(1)}` : "");
}

/** "DAL v PHX": the player's team v the opponent. */
export function matchupText(row) {
  const team = row?.team || "";
  const opponent = row?.opponent || "";
  if (team && opponent) return `${team} v ${opponent}`;
  return String(team || opponent || row?.game || "");
}

/** The ask in words: the scan's own ("20+ points"), else from its fields; edges mode is "best edges (all stats)". */
export function askText(scan) {
  if (scan?.mode === "edges") return "best edges (all stats)";
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

/**
 * The posted line's pieces: {label ("20+", "O 24.5", "Yes"), price (the person's format), book, stale}, or null. A
 * milestone ("12+") and a yes bet are over-only prices, so an under ask never shows one as its line (null).
 */
export function lineParts(line, side = "over") {
  if (!line || typeof line !== "object") return null;
  const under = side === "under";
  const overOnly = line.kind === "milestone" || line.kind === "yes" || (line.yes && !line.over && !line.under);
  if (under && overOnly) return null;
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

/**
 * GET /api/players/scan's query for an ask ({league?, stat, threshold?, side?, day?, n?, sort?, names?}). ``sort``
 * "hot" or "cold" asks the server to rank by the trend; "edges" is edges mode (sort=edge with no stat, number or
 * side: every stat's posted lines); the other sorts are the screen's own and aren't sent. ``names`` go as ``player``
 * (in edges mode too: Ask AI's best-edges card for a named player, "best props for Luka", refreshes his lines only; the
 * finder's own Best edges covers everyone and sends none). ``edgeSide`` ("over" | "under") keeps edges mode to one
 * side (Ask AI's "any good overs" card); left out, edges mode is both sides.
 */
export function scanQuery({ league = null, stat, threshold = null, side = "over", day = "today", n = 10, sort = "", names = [], edgeSide = "" } = {}) {
  const q = new URLSearchParams();
  const edges = sort === EDGES;
  if (league) q.set("league", league);
  if (!edges) q.set("stat", String(stat || ""));
  if (!edges && threshold !== null && threshold !== undefined && threshold !== "") q.set("threshold", String(threshold));
  if (!edges && side === "under") q.set("side", "under");
  if (edges && (edgeSide === "over" || edgeSide === "under")) q.set("side", edgeSide);
  q.set("day", day === "tomorrow" ? "tomorrow" : "today");
  q.set("n", String(Math.max(1, Math.min(MAX_ROWS, Number(n) || 10))));
  if (edges) q.set("sort", "edge");
  else if (TREND_SORTS.includes(sort)) q.set("sort", sort);
  for (const name of cleanNames(names)) q.append("player", name);
  return `/api/players/scan?${q.toString()}`;
}

/**
 * The Player finder's address for an ask: #/players?league=&stat=&at=<n|line>&day=[&side=under][&sort=][&player=...]
 * (``sort`` left out only for a number ranked by chance; each player's posted line always names its sort, so a
 * posted line ranked by chance reads back as it was).
 */
export function finderHash({ league = DEFAULT_LEAGUE, stat = "pts", threshold = null, side = "over", day = "today", sort = "", names = [] } = {}) {
  const q = new URLSearchParams();
  const line = threshold === null || threshold === undefined || threshold === "";
  q.set("league", league || DEFAULT_LEAGUE);
  q.set("stat", stat || "pts");
  q.set("at", line ? "line" : String(threshold));
  q.set("day", day === "tomorrow" ? "tomorrow" : "today");
  if (side === "under") q.set("side", "under");
  if (sort && (line || sort !== "chance")) q.set("sort", sort);
  for (const name of cleanNames(names)) q.append("player", name);
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

/**
 * A name filter's row: "Out" (his status) for a player ruled out, else whether he qualifies for the ask (projected to
 * get there, or 50% likely or more; 30% for a yes/no stat). "" for a ranked row.
 */
export function flagChip(row, { names = false, yesNo = false } = {}) {
  if (row?.ruled_out) return `<span class="badge danger ps-flag">${esc(String(row.status || "Ruled out"))}</span>`;
  if (!names || typeof row?.qualifies !== "boolean") return "";
  const bar = yesNo ? "30%" : "50%";
  return row.qualifies
    ? `<span class="badge good ps-flag" title="${esc(`Projected to get there, or ${bar} likely or more`)}">Qualifies</span>`
    : `<span class="badge muted ps-flag" title="${esc(`Not projected to get there, and under ${bar} likely`)}">Doesn't qualify</span>`;
}

/**
 * Hot and cold: "L5 28.4 vs 24.1 season" (the stat's average over this season's last 5 games against the rest of
 * this season's), or "L5 28.4 vs 24.1 last season" when the server compared them with last season's average
 * (``trend.vs`` "last_season": this season is still short), with an arrow for the way it moved and the same in words
 * for screen readers; "" without a trend.
 */
export function trendChip(row) {
  const trend = row?.trend;
  const l5 = Number(trend?.l5);
  const season = Number(trend?.season);
  if (!trend || trend.l5 == null || trend.season == null || !Number.isFinite(l5) || !Number.isFinite(season)) return "";
  const diff = trend.diff != null && Number.isFinite(Number(trend.diff)) ? Number(trend.diff) : l5 - season;
  const way = diff > 0 ? "up" : diff < 0 ? "down" : "";
  const last = trend.vs === "last_season";
  const words = `Last 5 games ${numText(l5)} on average, ${numText(season)} ${last ? "last season" : "over the rest of this season"}${way ? `: ${way} ${numText(Math.abs(diff))}` : ""}`;
  return `<span class="ps-trend${way === "up" ? " hot" : way === "down" ? " cold" : ""}" title="${esc(words)}">${way ? icon(way, { size: 13 }) : ""}<span aria-hidden="true">L5 ${esc(numText(l5))} vs ${esc(numText(season))} ${last ? "last season" : "season"}</span><span class="sr-only">${esc(words)}</span></span>`;
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

/**
 * "DraftKings 20+ −250 · Edge +4.2%" with an old-line chip, or "No line posted" ("No under line posted" when only an
 * over-only price is posted for an under). ``edge`` false leaves the edge out; ``stat`` leads with the row's own stat
 * (edges mode: "PRA (P+R+A) DraftKings O 44.5 −110"). An old line never shows an edge.
 */
export function lineHtml(row, side, { edge: withEdge = true, stat = "" } = {}) {
  const parts = lineParts(row?.line, side);
  const value = withEdge ? shownEdge(row) : null;
  const edge = edgeText(value);
  const what = stat ? `<b class="ps-stat">${esc(stat)}</b>` : "";
  if (!parts) return `${what}<span class="muted">${side === "under" && row?.line ? "No under line posted" : "No line posted"}</span>`;
  return `${what}<span class="ps-line-main">${parts.book ? `<span class="muted">${esc(parts.book)}</span> ` : ""}<b>${esc(parts.label)}</b>${parts.price ? ` <b class="num">${esc(parts.price)}</b>` : ""}</span>${
    edge ? ` <span class="ps-edge${value > 0 ? " pos" : ""}">Edge <b class="num">${esc(edge)}</b></span>` : ""}${
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
 * ``addButton(row, i)`` draws the slip button for an addable row (the caller's own handler). ``view`` says how the
 * rows were asked for: ``posted`` (each player's posted line) and ``edges`` (every stat's posted lines, each row with
 * its own stat and side) are ranked by edge, so the edge sits with the numbers; ``trend`` ("hot" | "cold") adds the
 * trend chip; ``names`` (a name filter) says whether each row qualifies; ``yesNo`` marks a yes/no stat for a server
 * that doesn't send each row's ``kind`` (a yes/no row shows its chance and no projection).
 */
export function rowItemHtml(row, i, { side = "over", addButton = null, view = {} } = {}) {
  const href = playerHref(row);
  const why = (Array.isArray(row.why) ? row.why : []).filter(Boolean).slice(0, 3);
  const leg = rowLeg(row);
  const rank = row.rank || i + 1;
  const name = esc(row.name || "Player");
  const yesNo = row.kind ? row.kind === "yes_no" : Boolean(view.yesNo) || isYesNo(row);
  const mySide = view.edges ? (row.side === "under" ? "under" : "over") : side;
  const byEdge = Boolean(view.posted || view.edges);
  const edge = shownEdge(row);
  const hits = (label, window) => `<div><dt>${label}</dt><dd class="num" aria-label="${esc(hitsWords(row.hits, window))}">${esc(hitsText(row.hits, window))}</dd></div>`;
  return `<li class="ps-item">
    <span class="ps-rank" aria-label="${esc(`Number ${rank}`)}">${esc(rank)}</span>
    <div class="ps-body">
      <div class="ps-head"><b class="ps-name">${name}</b>${statusChip(row)}${flagChip(row, { names: view.names, yesNo })}${modelChip(row)}${view.trend ? trendChip(row) : ""}</div>
      <div class="muted small ps-when">${esc(whenLine(row))}</div>
      <dl class="ps-nums${yesNo ? " no-proj" : ""}${byEdge ? " with-edge" : ""}">
        ${yesNo ? "" : `<div><dt>Proj.</dt><dd class="num">${esc(numText(row.projection))}</dd></div>`}
        <div class="ps-nums-chance"><dt>Chance</dt><dd>${chanceHtml(row.chance)}</dd></div>
        ${byEdge ? `<div class="ps-nums-edge"><dt>Edge</dt><dd class="num${edge > 0 ? " pos" : ""}">${esc(edgeText(edge) || "–")}</dd></div>` : ""}
        ${hits("L5", "l5")}${hits("L10", "l10")}${hits("Season", "season")}
      </dl>
      <div class="small ps-line">${lineHtml(row, mySide, { edge: !byEdge, stat: view.edges ? rowStatName(row, view.league) : "" })}</div>
      ${why.length ? `<p class="small ps-why">${why.map(esc).join(" · ")}</p>` : ""}
    </div>
    <div class="ps-actions">${href ? `<a class="btn btn-secondary small" href="${esc(href)}" aria-label="${esc(`View ${row.name || "player"}`)}">${icon("user", { size: 16 })}View player</a>` : ""}${leg && addButton ? addButton(row, i) : ""}</div>
  </li>`;
}

/**
 * Ranked rows as lists of card items: the first TOP, then the rest under "Also:" (they qualify too: the chat's answer
 * names the first five and the rest the same way). ``item(row, i)`` draws one; ``split`` false keeps one list (a
 * name filter's rows, which needn't qualify). ``label`` names the first list, ``also`` the second, for screen readers.
 */
export function rowListsHtml(rows, item, { label = "Players", also = "Also: more who qualify", split = true } = {}) {
  if (!rows.length) return "";
  const first = split ? rows.slice(0, TOP) : rows;
  const rest = split ? rows.slice(TOP) : [];
  return `<ol class="ps-list" aria-label="${esc(label)}">${first.map((row, i) => item(row, i)).join("")}</ol>${rest.length ? `
    <p class="ps-also-head" aria-hidden="true">Also:</p>
    <ol class="ps-list ps-also" start="${first.length + 1}" aria-label="${esc(also)}">${rest.map((row, i) => item(row, first.length + i)).join("")}</ol>` : ""}`;
}

/** The "Also:" list's name for screen readers: "Also: 5 more who qualify" (hot and cold: "5 more ranked"). */
export function alsoLabel(count, { sort = "", edges = false } = {}) {
  if (edges) return `Also: ${count} more posted line${count === 1 ? "" : "s"} with an edge`;
  if (TREND_SORTS.includes(sort)) return `Also: ${count} more ranked`;
  return `Also: ${count} more who qualif${count === 1 ? "ies" : "y"}`;
}

/**
 * How many more there are beyond the rows the result returned (the result's own ``more``, as it came: never counted
 * again here), in the words every surface uses: "229 more qualify: open the Player finder." (Ask AI's card; hot and
 * cold: "229 more players are ranked: ..."; edges mode: "28 more posted lines have an edge: ..."), or in the finder
 * itself "14 more qualify beyond the 25 shown: ...". "" when there are none.
 */
export function moreText(scan, { shown = 0, finder = false, sort = "" } = {}) {
  const more = Math.floor(Number(scan?.more));
  if (!Number.isFinite(more) || more <= 0) return "";
  const edges = scan?.mode === "edges" || sort === EDGES;
  const trend = !edges && TREND_SORTS.includes(sort);
  const what = edges ? `${more} more posted line${more === 1 ? " has" : "s have"} an edge`
    : trend ? `${more} more player${more === 1 ? " is" : "s are"} ranked`
    : `${more} more qualif${more === 1 ? "ies" : "y"}`;
  if (!finder) return `${what}: open the Player finder.`;
  return `${what} beyond the ${shown} shown${edges || trend ? "." : ": raise the number, or filter by player, to narrow the list."}`;
}

/** "Jalen Brunson", "Jalen Brunson or Jalen Green", "A, B or C". */
function orText(list) {
  return list.length > 1 ? `${list.slice(0, -1).join(", ")} or ${list.at(-1)}` : list[0] || "";
}

const nameOf = (value) => String(value ?? "").replace(/\s+/g, " ").trim();

/**
 * What a name filter couldn't rank, one note per name in the words every surface uses: a name that matched nobody
 * ("I couldn't find Zion in tonight's games."), one that matched several players ("More than one player matches
 * Jalen: Jalen Brunson, Jalen Green or Jalen Williams. Which one?"), and one whose game has already started ("Leon
 * Draisaitl's game (EDM @ CGY) has already started, so it isn't in the scan."). [{kind, text}], from the result's
 * ``missing_names``, ``ambiguous_names`` ({name, candidates}) and ``started_names`` ({name, player, game}).
 */
export function nameNotes(scan, when = "tonight") {
  const day = when === "tomorrow" ? "tomorrow's" : "tonight's";
  const list = (key) => (Array.isArray(scan?.[key]) ? scan[key] : []).slice(0, MAX_NAMES);
  const notes = [];
  for (const name of list("missing_names").map(nameOf).filter(Boolean)) {
    notes.push({ kind: "info", text: `I couldn't find ${name} in ${day} games.` });
  }
  for (const entry of list("ambiguous_names")) {
    const name = nameOf(entry?.name);
    const candidates = (Array.isArray(entry?.candidates) ? entry.candidates : []).map(nameOf).filter(Boolean).slice(0, 5);
    if (name) notes.push({ kind: "info", text: `More than one player matches ${name}${candidates.length ? `: ${orText(candidates)}` : ""}. Which one?` });
  }
  for (const entry of list("started_names")) {
    const player = nameOf(entry?.player) || nameOf(entry?.name);
    const game = nameOf(entry?.game);
    if (player) notes.push({ kind: "info", text: `${player}'s game${game ? ` (${game})` : ""} has already started, so it isn't in the scan.` });
  }
  return notes;
}

/** Whether every game of the day has started (``games`` counts the games still to come): nobody is left to rank. */
export function allStarted(scan) {
  const started = Math.floor(Number(scan?.games_started));
  return Number.isFinite(started) && started > 0 && !(Math.floor(Number(scan?.games)) > 0);
}

/**
 * Games already started, in the words every surface uses (``games_started``; ``games`` counts the games still to
 * come): "Every game tonight has started, so nobody is left to rank." when none is left, else "2 games tonight have
 * already started: their players aren't listed."; "" when none has started.
 */
export function startedText(scan, { when = "tonight" } = {}) {
  const started = Math.floor(Number(scan?.games_started));
  if (!Number.isFinite(started) || started <= 0) return "";
  const day = when === "tomorrow" ? "tomorrow" : "tonight";
  return allStarted(scan) ? `Every game ${day} has started, so nobody is left to rank.`
    : `${started} game${started === 1 ? "" : "s"} ${day} ${started === 1 ? "has" : "have"} already started: their players aren't listed.`;
}

/**
 * The board's own state, as notes: ESPN's scoreboard didn't answer (some games may be missing), and games already
 * started (startedText). [{kind, text}].
 */
export function boardNotes(scan, { when = "tonight" } = {}) {
  const notes = [];
  const day = when === "tomorrow" ? "tomorrow" : "tonight";
  if (scan?.error) notes.push({ kind: "warn", text: `ESPN's scoreboard didn't answer just now, so some of ${day}'s games may be missing. Refresh in a minute.` });
  const started = startedText(scan, { when });
  if (started) notes.push({ kind: "info", text: started });
  return notes;
}

/** Notes as short lines with an icon (escaped). */
export function notesHtml(notes) {
  return (notes || []).filter((n) => n?.text).map((n) =>
    `<p class="ps-alert ${n.kind === "warn" ? "warn" : "info"}">${icon(n.kind === "warn" ? "alert" : "info", { size: 14 })}<span>${esc(n.text)}</span></p>`).join("");
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

/** Games whose roster, game log or projection failed and are still missing: "2 games still missing data", or "". */
export function failedText(loading) {
  const failed = Math.floor(Number(loading?.failed));
  return Number.isFinite(failed) && failed > 0 ? `${failed} game${failed === 1 ? "" : "s"} still missing data` : "";
}

/** Whether the board is still building (its games still being read, or fewer ready than there are). */
export function stillLoading(loading) {
  if (!loading || typeof loading !== "object") return false;
  const ready = Number(loading.games_ready);
  const total = Number(loading.games_total);
  return !Number.isFinite(total) || total <= 0 || !(Number.isFinite(ready) && ready >= total);
}

/**
 * The board's progress in words: "8 of 10 games loaded", "2 games still missing data", both, or "". With ESPN's
 * scoreboard down (``error``) there is nothing being read, so "Reading the day's games…" isn't said.
 */
export function statusText(scan) {
  const loading = scan?.loading;
  const reading = loadingText(loading);
  const total = Number(loading?.games_total);
  const words = scan?.error && !(Number.isFinite(total) && total > 0) ? "" : reading;
  return [words, failedText(loading)].filter(Boolean).join(" · ");
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
.ps-nums .ps-nums-edge dd.pos { color: var(--good-ink); }
.ps-trend { display: inline-flex; align-items: center; gap: 3px; font-size: 12px; font-weight: 650; white-space: nowrap; color: var(--text-2); }
.ps-trend .icon { flex: none; }
.ps-trend.hot { color: var(--good-ink); }
.ps-trend.cold { color: var(--danger-ink); }
.ps-flag { padding: 1px 7px; }
/* On a row's tinted card the badge inks are mixed a little toward the text colour, so "Qualifies" and "Out" keep
   4.5:1 in Telegram's own palettes too (a webview without color-mix keeps the badge's ink) */
@supports (color: color-mix(in srgb, red 50%, blue)) {
  .ps-flag.good { color: color-mix(in srgb, var(--good-ink) 80%, var(--text)); }
  .ps-flag.danger { color: color-mix(in srgb, var(--danger-ink) 80%, var(--text)); }
}
.ps-stat { font-weight: 750; }
/* a narrow card: the buttons go under the numbers, full width */
@media (max-width: 559px) {
  .ps-item { grid-template-columns: 28px minmax(0, 1fr); }
  .ps-nums .ps-nums-chance { flex: 1 1 calc(100% - 76px); } /* the projection and the chance, then the three hit windows */
  .ps-nums.no-proj .ps-nums-chance { flex-basis: 100%; } /* a yes/no stat: the chance, then the hits */
  /* ranked by edge: the edge leads the second line (edge, L5, L10, season) on every row */
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
.ps-also-head { margin: 12px 0 6px; font-size: 14px; font-weight: 700; color: var(--text-2); }
/* the board's notes (not .ps-note: that's each list's footnote) */
.ps-alert { display: flex; align-items: flex-start; gap: 6px; margin: 0; font-size: 12.5px; color: var(--text-2); line-height: 1.45; }
.ps-alert > .icon { flex: none; margin-top: 2px; }
.ps-alert.warn > .icon { color: var(--warn-ink); }
.ps-alert.info > .icon { color: var(--info-ink); }
`;

export function ensureScanStyles() {
  if (document.getElementById("scan-styles")) return;
  const style = document.createElement("style");
  style.id = "scan-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
