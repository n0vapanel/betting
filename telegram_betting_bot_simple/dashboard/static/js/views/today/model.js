// Today's data, without the page: the slate (GET /api/slate) indexed once, then filtered and sorted in
// the browser by league, day, status and search. No DOM here, so every rule can be checked in node.
//
// The numbers are the server's: prices, the bookmakers' margin-free chances, the markets brain's blend and the
// NBA model's chances come in the slate. This module only compares them on the same footing (the AI's chance
// minus the bookmakers', for the same outcome of the same game) and writes them in the person's odds format.
//
//   prepare(slate)                     -> {leagues: Map, order: [keys], groups, games, week: Map(league -> games
//                                         in the seven days)} (memoised per slate)
//   dayStrip(now)                      -> [{key: "YYYY-MM-DD", label, long, offset}] today and the next 6 days
//   view(prepared, {leagues, day, status, search, sort, pinned}) -> {games, counts, days, live, total, searchAll}
//   bookChance(game)                   -> the bookmakers' margin-free chance per outcome (never the blend), or null
//   edges(game) / gap(game)            -> AI minus the bookmakers' chance per side (or null) / the biggest gap
//   legFor(game, market, side)         -> the slip leg an odds button adds (ref names the odds event id)
//   american(decimal)                  -> "+150" / "-200"
//   normalizeGame(raw)                 -> a slate game in the full shape (the trimmed slate, or an older copy
//                                         saved on this device as it was: both are read)
//   marketsOf(game)                    -> the game's "Markets say" (its mkt block) checked, or null
//
// The slate is trimmed to stay small (about 3 KB gzipped per 60 games): a game's ``espn_id``/``odds_id`` are
// left out when its ``id`` holds them ("<league>:<odds id>" or "<league>:espn:<ESPN id>"), ESPN logos come as
// their path after https://a.espncdn.com/, and a team's full ``name`` only when ``short`` or ``abbr`` is
// missing. normalizeGame puts the ids, the logo addresses and a name back; a copy of the older, untrimmed
// slate (still in a device's cache after an update) passes through unchanged.
//
// "Markets say": each game's ``chance`` is the markets brain's blend (src "blend"; "market" in older slates)
// and ``mkt`` is its compact summary: {p: {home, away, draw?}, fav, conf: "normal"|"low", dis: bool, spr:
// the sources' widest gap in points, src: [[short name, chance of fav, status], ...], as_of}.
//
// One name, one number: "Mkt" (a price button's chance), a slip leg's market chance and Edge are the
// bookmakers' margin-free chance (``fair``: {home, away, draw?}, sent next to the blend; in an older slate,
// ``chance`` itself), the same figure the game page's buttons and the priced slip show. "Markets" is the blend.
// A slate with the blend but no ``fair`` shows no "Mkt" chance and no Edge rather than pass the blend off as
// the bookmakers'.

export const STATUSES = ["all", "upcoming", "live", "finished"];
export const SORTS = ["time", "league", "gap", "dis", "market", "ai"];
export const DAYS = 7;

const STATUS_OF = { pre: "upcoming", live: "live", final: "finished", postponed: "finished" };
const STATUS_ORDER = { live: 0, pre: 1, postponed: 2, final: 3 };

/** "2026-10-26" for a Date, in this device's own calendar. */
export function dayKey(date) {
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

let dayNames = null; // two formatters made once (toLocaleDateString builds one per call)

/** Today and the next six days: [{key, offset, label: "Today" | "Tomorrow" | "Wed 28", long}]. */
export function dayStrip(now = new Date()) {
  dayNames ||= {
    short: new Intl.DateTimeFormat([], { weekday: "short" }),
    long: new Intl.DateTimeFormat([], { weekday: "long", month: "short", day: "numeric" }),
  };
  const out = [];
  for (let offset = 0; offset < DAYS; offset += 1) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + offset, 12);
    const label = offset === 0 ? "Today" : offset === 1 ? "Tomorrow" : `${dayNames.short.format(d)} ${d.getDate()}`;
    out.push({ key: dayKey(d), offset, label, long: dayNames.long.format(d) });
  }
  return out;
}

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

const ESPN_IMAGES = "https://a.espncdn.com/";
const LOGO_PATH = /^[A-Za-z0-9][A-Za-z0-9._~\/%-]{0,250}(\?[A-Za-z0-9=&._%-]{0,80})?$/;

/**
 * A logo address from the slate: ESPN's path ("i/teamlogos/nba/500/bos.png") gets its host back; a full
 * https address (an older slate) is kept as it is; anything else is dropped (row.js still checks the host).
 */
export function logoUrl(value) {
  if (typeof value !== "string" || !value) return null;
  if (/^https:\/\//i.test(value)) return value;
  if (value.includes("..") || value.includes("//") || !LOGO_PATH.test(value)) return null;
  return ESPN_IMAGES + value;
}

function fullTeam(team) {
  if (!team || typeof team !== "object") return team;
  const logo = logoUrl(team.logo);
  const name = typeof team.name === "string" && team.name ? team.name : team.short || team.abbr || "";
  if (logo === team.logo && name === team.name) return team;
  const out = { ...team, name };
  if (logo) out.logo = logo;
  else delete out.logo;
  return out;
}

/**
 * A slate game in the full shape: ``odds_id`` and ``espn_id`` from its ``id`` when they were left out, logo
 * addresses rebuilt and a ``name`` for each side (the full name when sent, else the short name or the
 * abbreviation). A game already in the full shape (an older cached slate) comes back as it is.
 */
export function normalizeGame(raw) {
  if (!raw || typeof raw !== "object" || typeof raw.id !== "string" || typeof raw.league !== "string") return raw;
  const out = { ...raw };
  const prefix = `${raw.league}:`;
  const rest = raw.id.startsWith(prefix) ? raw.id.slice(prefix.length) : "";
  if (rest.startsWith("espn:")) {
    if (out.espn_id == null && rest.length > 5) out.espn_id = rest.slice(5);
  } else if (rest && out.odds_id == null) {
    out.odds_id = rest;
  }
  out.home = fullTeam(raw.home);
  out.away = fullTeam(raw.away);
  return out;
}

const SIDES = ["home", "away", "draw"];
const SOURCE_STATUSES = ["ok", "thin", "stale", "missing"];

/**
 * The game's "Markets say" summary, checked: {p: {home, away, draw?}, fav, conf, dis, spread (points),
 * sources: [{name, p (chance of fav), status}], asOf}, or null when the slate has none (an older slate, or no
 * source has an opinion yet).
 */
export function marketsOf(game) {
  const m = game?.mkt;
  if (!m || typeof m !== "object" || !m.p || typeof m.p !== "object") return null;
  const p = {};
  for (const side of SIDES) if (num(m.p[side]) !== null) p[side] = m.p[side];
  if (p.home === undefined || p.away === undefined) return null;
  let fav = SIDES.includes(m.fav) && p[m.fav] !== undefined ? m.fav : null;
  if (!fav) for (const side of SIDES) if (p[side] !== undefined && (!fav || p[side] > p[fav])) fav = side;
  const sources = [];
  for (const entry of Array.isArray(m.src) ? m.src : []) {
    if (!Array.isArray(entry) || typeof entry[0] !== "string" || num(entry[1]) === null) continue;
    sources.push({ name: entry[0].slice(0, 24), p: entry[1], status: SOURCE_STATUSES.includes(entry[2]) ? entry[2] : "ok" });
  }
  return {
    p, fav, conf: m.conf === "low" ? "low" : "normal", dis: m.dis === true, spread: num(m.spr), sources,
    asOf: typeof m.as_of === "string" ? m.as_of : null,
  };
}

/** How far the sources are apart on a game, in points (null without a "Markets say"). */
export function disagreement(game) {
  const m = marketsOf(game);
  return m ? (m.spread ?? (m.dis ? 5 : 0)) : null;
}

/** A best price [price, book, point?] from the slate, checked: {price, book, point} or null. */
export function offer(game, market, side) {
  const entry = game?.odds?.[market]?.[side];
  if (!Array.isArray(entry)) return null;
  const price = num(entry[0]);
  if (price === null || price <= 1) return null;
  const point = market === "h2h" ? null : num(entry[2]);
  if (market !== "h2h" && point === null) return null;
  return { price, book: typeof entry[1] === "string" ? entry[1] : "", point };
}

/** American odds for a decimal price: 2.50 -> "+150", 1.50 -> "-200" (formatting only). */
export function american(decimal) {
  const d = Number(decimal);
  if (!Number.isFinite(d) || d <= 1) return "";
  const value = d >= 2 ? Math.round((d - 1) * 100) : Math.round(-100 / (d - 1));
  return value > 0 ? `+${value}` : String(value);
}

/**
 * The bookmakers' margin-free chance of each outcome, {home, away, draw?}, or null: the slate's ``fair``, else
 * ``chance`` when it is the market's own (an older slate). Never the blend (``chance`` with src "blend").
 */
export function bookChance(game) {
  const f = game?.fair;
  if (f && typeof f === "object" && num(f.home) !== null && num(f.away) !== null) return f;
  const c = game?.chance;
  if (c && typeof c === "object" && c.src !== "blend" && num(c.home) !== null && num(c.away) !== null) return c;
  return null;
}

/**
 * The AI's chance minus the market's for each side of the moneyline, in probability units, or null.
 * Like with like: both are the chance that side wins this game, the market's the bookmakers' with their
 * margin taken out (bookChance; never the blend). A three-way market (a draw) is compared only when the AI
 * gives a draw chance too.
 */
export function edges(game) {
  const ai = game?.ai;
  const market = bookChance(game);
  if (!ai || !market) return null;
  const sides = num(market.draw) !== null ? ["home", "away", "draw"] : ["home", "away"];
  const out = {};
  for (const side of sides) {
    const a = num(ai[side]);
    const m = num(market[side]);
    if (a === null || m === null) return null;
    out[side] = a - m;
  }
  return out;
}

/** The biggest AI-vs-market gap on a game (0 to 1), or null without both. */
export function gap(game) {
  const e = edges(game);
  if (!e) return null;
  return Math.max(...Object.values(e).map(Math.abs));
}

/** The side the AI rates above the market (its best edge), {side, edge}, or null. */
export function bestEdge(game) {
  const e = edges(game);
  if (!e) return null;
  let best = null;
  for (const [side, value] of Object.entries(e)) if (!best || value > best.edge) best = { side, edge: value };
  return best;
}

/** The favourite by the market's chance: {side, chance} or null. */
export function favourite(game, source = "chance") {
  const c = game?.[source];
  if (!c) return null;
  let best = null;
  for (const side of ["home", "away", "draw"]) {
    const p = num(c[side]);
    if (p !== null && (!best || p > best.chance)) best = { side, chance: p };
  }
  return best;
}

const SHORT_NAMES = /^(basketball_(nba|wnba)|americanfootball_(nfl|cfl)|baseball_mlb|icehockey_nhl|(tennis|mma|boxing)_.+)$/;

/**
 * A team's name for tight spaces: the nickname in the US pro leagues ("Celtics"), where it is unique, and a
 * player's short name in one-on-one sports ("J. Sinner", "Pereira"); the full name elsewhere ("Hull City"
 * and "Coventry City" would both be "City").
 */
export function teamShort(team, league = "") {
  if (!team) return "";
  if (SHORT_NAMES.test(league) && team.short) return team.short;
  return team.name || team.short || "";
}

/** How a pick reads on the slip ("Boston Celtics -3.5", "Over 224.5"), as dashboard/service.py writes it. */
export function outcomeLabel(game, market, side, point) {
  const team = side === "home" ? game.home?.name : side === "away" ? game.away?.name : side === "draw" ? "Draw" : side;
  const signed = (p) => `${p > 0 ? "+" : ""}${p}`;
  if (market === "h2h") return side === "draw" ? "Draw" : `${team} to win`;
  if (market === "spreads" && point !== null) return `${team} ${signed(point)}`;
  if (market === "totals" && point !== null) return `${side === "over" ? "Over" : "Under"} ${point}`;
  return String(team || side);
}

/**
 * The slip leg a price button adds, or null when the game has no price there or no odds event id (the
 * server prices a pick from its odds board, which knows games by that id). Prices and chances in it are for
 * drawing the slip at once; the server looks every pick up again when it prices the slip.
 */
export function legFor(game, market, side) {
  const o = offer(game, market, side);
  if (!o || !game.odds_id) return null;
  const modelled = market === "h2h" && game.ai && num(game.ai[side]) !== null;
  return {
    ref: { league: game.league, id: String(game.odds_id), market, side, point: o.point },
    label: outcomeLabel(game, market, side, o.point),
    game: `${game.away?.name || "Away"} @ ${game.home?.name || "Home"}`,
    start: game.start,
    league: game.league,
    price: o.price,
    american: american(o.price),
    ai: modelled ? game.ai[side] : null,
    fair: market === "h2h" ? num(bookChance(game)?.[side]) : null, // the bookmakers', as the priced slip has it
  };
}

// --- indexing -------------------------------------------------------------------------------------------

let memo = { slate: null, prepared: null, today: "" };

/**
 * Index a slate once: leagues by key (in the server's order), groups by sport, and each game with what the
 * filters need (its local day, a lower-case search text, its status bucket and its biggest gap).
 */
export function prepare(slate, now = new Date()) {
  const today = dayKey(now);
  if (memo.slate === slate && memo.today === today && memo.prepared) return memo.prepared;
  const leagues = new Map();
  const order = [];
  for (const entry of Array.isArray(slate?.leagues) ? slate.leagues : []) {
    if (!entry || typeof entry.key !== "string" || leagues.has(entry.key)) continue;
    leagues.set(entry.key, entry);
    order.push(entry.key);
  }
  const games = [];
  for (const sent of Array.isArray(slate?.games) ? slate.games : []) {
    if (!sent || typeof sent.id !== "string" || typeof sent.league !== "string") continue;
    const raw = normalizeGame(sent);
    const start = Date.parse(raw.start);
    if (!Number.isFinite(start)) continue;
    if (!leagues.has(raw.league)) { // a game in a league the list doesn't name: list it anyway
      const entry = { key: raw.league, title: raw.league, group: "Other", label: "Other", count: 0, live: 0, odds: "none" };
      leagues.set(raw.league, entry);
      order.push(raw.league);
    }
    const status = STATUS_OF[raw.status] ? raw.status : "pre";
    // A game still live from last night belongs on Today, not on the day it started
    const day = status === "live" && dayKey(new Date(start)) < today ? today : dayKey(new Date(start));
    const words = [raw.home?.name, raw.home?.short, raw.home?.abbr, raw.away?.name, raw.away?.short, raw.away?.abbr,
      leagues.get(raw.league)?.title, raw.venue?.name, raw.venue?.city].filter(Boolean).join(" ");
    games.push({ ...raw, status, _start: start, _day: day, _bucket: STATUS_OF[status], _text: fold(words), _gap: gap(raw), _dis: disagreement(raw) });
  }
  const groups = [];
  const byGroup = new Map();
  for (const key of order) {
    const entry = leagues.get(key);
    const name = entry.group || "Other";
    let group = byGroup.get(name);
    if (!group) {
      group = { key: name, label: entry.label || name, leagues: [] };
      byGroup.set(name, group);
      groups.push(group);
    }
    group.leagues.push(key);
  }
  const week = new Map();
  for (const g of games) week.set(g.league, (week.get(g.league) || 0) + 1);
  const prepared = { leagues, order, groups, games, week, generatedAt: slate?.generated_at || null, asOf: slate?.as_of || null };
  memo = { slate, prepared, today };
  return prepared;
}

/** Lower case without accents, for search ("Atlético" matches "atletico"). */
export function fold(text) {
  return String(text || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/** The emoji at the start of a group label ("🏀 Basketball" -> "🏀"), or "". */
export function groupIcon(label) {
  const m = /^(\p{Extended_Pictographic}️?)/u.exec(String(label || ""));
  return m ? m[1] : "";
}

/** "Basketball" from "🏀 Basketball". */
export function groupName(label) {
  return String(label || "").replace(/^\p{Extended_Pictographic}️?\s*/u, "") || String(label || "");
}

// --- filtering and sorting ----------------------------------------------------------------------------

/**
 * The board for a view:
 *   options = {leagues: Set | null (all), day: "YYYY-MM-DD", status: "all"|"upcoming"|"live"|"finished",
 *              search: "", sort: {key: "time"|"league"|"gap"|"dis"|"market"|"ai", dir: 1|-1}, pinned: [league keys]}
 * A search looks across all seven days (people search for a team, not a date). Returns
 *   {games, counts: Map(league -> {count, live}) for the day, days: Map(day -> count) for the leagues,
 *    live (games live in the leagues), total (games on the day before the status filter), searchAll}
 */
export function view(prepared, { leagues = null, day = "", status = "all", search = "", sort = { key: "time", dir: 1 }, pinned = [] } = {}) {
  const query = fold(search).trim();
  const terms = query ? query.split(/\s+/) : [];
  const inLeagues = (g) => !leagues || leagues.has(g.league);
  const counts = new Map();
  const days = new Map();
  let live = 0;
  let total = 0;
  const games = [];
  for (const g of prepared.games) {
    if (g._day === day) {
      const c = counts.get(g.league) || { count: 0, live: 0 };
      c.count += 1;
      if (g.status === "live") c.live += 1;
      counts.set(g.league, c);
    }
    if (!inLeagues(g)) continue;
    days.set(g._day, (days.get(g._day) || 0) + 1);
    if (g.status === "live") live += 1;
    if (!terms.length && g._day !== day) continue;
    if (terms.length && !terms.every((t) => g._text.includes(t))) continue;
    if (!terms.length) total += 1;
    if (status !== "all" && g._bucket !== status) continue;
    games.push(g);
  }
  if (terms.length) total = games.length;
  return { games: sortGames(games, sort, pinned, prepared), counts, days, live, total, searchAll: terms.length > 0 };
}

/**
 * Sort a board. "time": live first, then upcoming by start, then finished (pinned leagues first when the
 * person has some); "league": by the league list's order (pinned first), then time; "gap", "dis" (how far the
 * markets' sources are apart), "market", "ai": biggest first (games without that number last), ties by time.
 * ``dir`` -1 reverses the main key.
 */
export function sortGames(games, sort = { key: "time", dir: 1 }, pinned = [], prepared = null) {
  const key = SORTS.includes(sort?.key) ? sort.key : "time";
  const dir = sort?.dir === -1 ? -1 : 1;
  const pinRank = new Map(pinned.map((k, i) => [k, i]));
  const leagueRank = new Map((prepared?.order || []).map((k, i) => [k, i]));
  const pin = (g) => (pinRank.has(g.league) ? 0 : 1);
  const byTime = (a, b) => (STATUS_ORDER[a.status] - STATUS_ORDER[b.status]) || (a._start - b._start) || a.id.localeCompare(b.id);
  const value = (g) => {
    if (key === "gap") return g._gap;
    if (key === "dis") return g._dis ?? disagreement(g);
    if (key === "market") return favourite(g, "chance")?.chance ?? null;
    if (key === "ai") return favourite(g, "ai")?.chance ?? null;
    return null;
  };
  const list = [...games];
  if (key === "time") {
    list.sort((a, b) => (pin(a) - pin(b)) || dir * byTime(a, b));
  } else if (key === "league") {
    const rank = (g) => (pinRank.has(g.league) ? pinRank.get(g.league) - 1000 : leagueRank.get(g.league) ?? 999);
    list.sort((a, b) => dir * (rank(a) - rank(b)) || byTime(a, b));
  } else {
    list.sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      if (va === null && vb === null) return byTime(a, b);
      if (va === null) return 1;
      if (vb === null) return -1;
      return dir * (vb - va) || byTime(a, b);
    });
  }
  return list;
}

/** Leagues with a price board worth asking for: in the view, odds stale or missing, games still to start. */
export function leaguesNeedingOdds(prepared, games) {
  const want = new Set();
  for (const g of games) {
    if (g.status !== "pre") continue;
    const entry = prepared.leagues.get(g.league);
    if (entry && (entry.odds === "stale" || entry.odds === "none")) want.add(g.league);
  }
  return [...want];
}

/** The tri-state of a set of keys against the selection: "on", "off" or "mixed". */
export function triState(keys, selected) {
  if (!selected) return keys.length ? "on" : "off";
  let on = 0;
  for (const key of keys) if (selected.has(key)) on += 1;
  return on === 0 ? "off" : on === keys.length ? "on" : "mixed";
}
