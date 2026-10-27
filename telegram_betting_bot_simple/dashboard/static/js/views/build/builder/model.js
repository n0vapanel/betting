// The Builder's controls, without the page: what a preset fills, when a run is "Custom", how the controls
// travel (POST /api/builder/run, PUT /api/me {builder}) and how they are kept on this device. No DOM here.
//
// The controls are engine/builder.py's Controls as its to_dict() sends them (GET /api/builder/presets), the
// same keys both ways. Chances and edges are shares (0.55 for 55%); the form shows them in % and points.
// Nothing here works out a betting number: the target odds from a stake and "win about" is the person's own
// two amounts divided, and every chance, price and payout on screen is the server's.
//
//   FIELDS                       the control keys the API takes (dashboard/app.py BuilderControls), in order
//   SCOPE_FIELDS                 keys that don't change the risk (engine/presets.py): editing them keeps the name
//   FIXED                        controls the engine doesn't vary yet (one leg per game, no started games)
//   cleanControls(raw, base)     a full controls object: base's values, overwritten by raw's valid ones
//   labelFor(c, presets, scope)  "Safe" | "Risky" | "Crazy" while every risk field equals c.preset's, else "Custom"
//   changedFields(c, presets, scope) the risk fields that differ from the preset (for "Custom · 2 changes")
//   withPreset(name, c, presets, scope) the preset's risk fields, keeping c's scope (leagues, window, books...)
//   windowFor(ui, now)           {from, to} ISO for Today / next 24 h / 3 days / custom (null when custom is unset)
//   runBody(c, ui, now)          the controls as POST /api/builder/run takes them (window worked out, tz set)
//   savedBody(c, ui, now)        the same plus the app's small "ui" state, for PUT /api/me {builder}
//   restore(local, server, ...)  the newer of this device's copy and the server's, as {controls, ui}
//   targetFrom(stake, back)      "5.00" from $10 → $50 (the person's own amounts), or null
//   ageWords(seconds)            "3 h", "40 min"

export const FIELDS = [
  "leagues", "window", "kickoff", "exclude_started", "legs", "target_odds", "tolerance_pct", "leg_odds_min", "leg_odds_max",
  "min_chance", "min_edge", "max_per_game", "max_per_team", "max_per_league", "markets", "avoid_same_game",
  "prefer_correlation", "max_price_age_s", "exclude_disagree", "exclude_thin", "books", "tickets", "preset", "caution_pp",
  "legs_max", "odds_min", "odds_max", "joint_min", "joint_max", "multi_leg_min", "whelan", "whelan_slack",
  "exclude_low_conf", "exclude_model_price", "tz",
];

// engine/presets.py SCOPE_FIELDS (the presets route sends its own list as defaults.scope_fields)
export const SCOPE_FIELDS = ["leagues", "window", "kickoff", "exclude_started", "books", "tickets", "preset", "tz", "prefer_correlation"];

// Fixed until a bookmaker's own same-game price is loaded: the engine keeps one leg per game and never uses a game
// already under way, whatever these say (engine/builder.py only adds a note). The form shows them switched off with
// the reason; every controls object carries these values, so they never make a run "Custom".
export const FIXED = { max_per_game: 1, avoid_same_game: true, exclude_started: true };

export const PRESET_KEYS = ["safe", "risky", "crazy"];
export const PRESET_TITLES = { safe: "Safe", risky: "Risky", crazy: "Crazy" };
export const PRESET_ICONS = { safe: "shield", risky: "scale", crazy: "zap" };

// What the presets route returns when the server has none yet (engine/builder.py Controls() defaults)
export const DEFAULTS = {
  leagues: null, window: null, kickoff: null, exclude_started: true, legs: null, target_odds: null, tolerance_pct: 10,
  leg_odds_min: null, leg_odds_max: null, min_chance: null, min_edge: null, max_per_game: 1, max_per_team: 1,
  max_per_league: null, markets: ["h2h", "spreads", "totals"], avoid_same_game: true, prefer_correlation: false,
  max_price_age_s: 10800, exclude_disagree: false, exclude_thin: false, books: null, tickets: 3, preset: null,
  caution_pp: 2, legs_max: 4, odds_min: null, odds_max: null, joint_min: null, joint_max: null, multi_leg_min: null,
  whelan: true, whelan_slack: 0, exclude_low_conf: false, exclude_model_price: false, tz: null,
};

export const MARKET_ORDER = ["h2h", "spreads", "totals", "team_totals", "btts", "draw_no_bet", "double_chance", "player_props", "corners", "cards"];

export const WINDOWS = [
  { value: "today", label: "Today" },
  { value: "24h", label: "Next 24 h" },
  { value: "3d", label: "3 days" },
  { value: "custom", label: "Custom" },
];

export const AGES = [1800, 3600, 7200, 10800, 21600, 43200, 86400];
export const MAX_WINDOW_DAYS = 8; // dashboard/lab.py MAX_WINDOW_DAYS: the slate holds today and the next six days
export const MAX_LEGS = 12;
export const MAX_TICKETS = 5;

const LEAGUE = /^[a-z0-9_]{2,64}$/;
const BOOK = /^[A-Za-z0-9_ .-]{2,40}$/;
const CLOCK = /^([01]\d|2[0-3]):[0-5]\d$/;
const MONEY = /^\d{1,6}(\.\d{1,2})?$/;

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const num = (v, lo, hi) => (isNum(v) && v >= lo && v <= hi ? v : undefined);
const int = (v, lo, hi) => (Number.isInteger(v) && v >= lo && v <= hi ? v : undefined);
const bool = (v) => (typeof v === "boolean" ? v : undefined);

/** One field's value from outside (the server, this device, a URL): the value when it fits, else undefined. */
function cleanField(key, v) {
  if (v === null) {
    // null means "none" / "all" / "Auto" for these; the rest need a value
    return ["leagues", "window", "kickoff", "legs", "target_odds", "leg_odds_min", "leg_odds_max", "min_chance", "min_edge",
      "max_per_league", "books", "preset", "odds_min", "odds_max", "joint_min", "joint_max", "multi_leg_min", "tz"].includes(key) ? null : undefined;
  }
  switch (key) {
    case "leagues": {
      if (!Array.isArray(v)) return undefined;
      const list = [...new Set(v.filter((k) => typeof k === "string" && LEAGUE.test(k)))].slice(0, 200);
      return list.length ? list : null;
    }
    case "books": {
      if (!Array.isArray(v)) return undefined;
      const list = [...new Set(v.filter((k) => typeof k === "string" && BOOK.test(k)))].slice(0, 20);
      return list.length ? list : null;
    }
    case "markets": {
      if (!Array.isArray(v)) return undefined;
      const list = MARKET_ORDER.filter((m) => v.includes(m));
      return list.length ? list : undefined;
    }
    case "window": {
      if (!v || typeof v !== "object") return undefined;
      const from = typeof v.from === "string" && Number.isFinite(Date.parse(v.from)) ? v.from : null;
      const to = typeof v.to === "string" && Number.isFinite(Date.parse(v.to)) ? v.to : null;
      return from && to ? { from, to } : null;
    }
    case "kickoff": {
      if (!v || typeof v !== "object") return undefined;
      const earliest = CLOCK.test(v.earliest || "") ? v.earliest : null;
      const latest = CLOCK.test(v.latest || "") ? v.latest : null;
      return earliest || latest ? { earliest, latest } : null;
    }
    case "preset": return PRESET_KEYS.includes(v) ? v : v === "custom" ? null : undefined;
    case "tz": return typeof v === "string" && v.length >= 1 && v.length <= 64 ? v : undefined;
    case "legs": return int(v, 1, MAX_LEGS);
    case "legs_max": return int(v, 1, MAX_LEGS);
    case "tickets": return int(v, 1, MAX_TICKETS);
    case "max_per_game": case "max_per_team": case "max_per_league": return int(v, 1, MAX_LEGS);
    case "whelan_slack": return int(v, 0, MAX_LEGS);
    case "max_price_age_s": return int(v, 60, 7 * 86400);
    case "target_odds": return num(v, 1.01, 1e6);
    case "tolerance_pct": return num(v, 0, 90);
    case "leg_odds_min": case "leg_odds_max": return num(v, 1.01, 1000);
    case "odds_min": return num(v, 1, 1e6);
    case "odds_max": return num(v, 1.01, 1e6);
    case "min_chance": case "joint_min": case "multi_leg_min": return num(v, 0, 1);
    case "joint_max": return num(v, 0, 1.01);
    case "min_edge": return num(v, -1, 1);
    case "caution_pp": return num(v, 0, 20);
    case "exclude_started": case "avoid_same_game": case "prefer_correlation": case "exclude_disagree": case "exclude_thin":
    case "whelan": case "exclude_low_conf": case "exclude_model_price":
      return bool(v);
    default: return undefined;
  }
}

/** A full controls object (every FIELDS key): ``base``'s values, overwritten by ``raw``'s that are valid. */
export function cleanControls(raw, base = DEFAULTS) {
  const out = {};
  for (const key of FIELDS) {
    const fallback = base && key in base ? base[key] : DEFAULTS[key];
    const value = raw && typeof raw === "object" && key in raw ? cleanField(key, raw[key]) : undefined;
    out[key] = value === undefined ? copy(fallback) : value;
  }
  return { ...out, ...FIXED };
}

function copy(v) {
  if (Array.isArray(v)) return [...v];
  if (v && typeof v === "object") return { ...v };
  return v;
}

/** Equal as controls: numbers to 1e-9, lists as sets, objects by their fields, null and undefined alike. */
export function same(a, b) {
  if (a == null || b == null) return a == null && b == null;
  if (isNum(a) && isNum(b)) return Math.abs(a - b) < 1e-9;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x) => b.includes(x));
  if (typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    return [...keys].every((k) => same(a[k], b[k]));
  }
  return a === b;
}

const riskFields = (scope = SCOPE_FIELDS) => FIELDS.filter((key) => !scope.includes(key) && !(key in FIXED));

/** The risk fields where ``c`` differs from its preset (all of them when it has none). */
export function changedFields(c, presets, scope = SCOPE_FIELDS) {
  const base = presets?.[c.preset];
  if (!base) return riskFields(scope);
  return riskFields(scope).filter((key) => !same(c[key], base[key]));
}

/** "Safe", "Risky" or "Crazy" while every risk field equals that preset's (engine/presets.py label_for), else "Custom". */
export function labelFor(c, presets, scope = SCOPE_FIELDS) {
  if (!presets?.[c.preset]) return "Custom";
  return changedFields(c, presets, scope).length ? "Custom" : PRESET_TITLES[c.preset];
}

/** A preset's controls with ``c``'s scope kept (engine/presets.py preset(name, keep)). */
export function withPreset(name, c, presets, scope = SCOPE_FIELDS) {
  const base = presets?.[name];
  if (!base) return c;
  const out = cleanControls(base, DEFAULTS);
  for (const key of scope) if (key !== "preset" && c && key in c) out[key] = copy(c[key]);
  out.preset = name;
  return out;
}

// --- the time window --------------------------------------------------------------------------------------

const iso = (ms) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");

/** The end of the local day of ``now`` (23:59:59.999 local). */
function endOfDay(now) {
  const d = new Date(now);
  d.setHours(23, 59, 59, 999);
  return d.getTime();
}

/**
 * {from, to} for the window chosen: "today" (now to the end of today, local time), "24h", "3d" (72 hours),
 * or "custom" (ui.from / ui.to, ISO). null when a custom window is unset or impossible.
 */
export function windowFor(ui, now = Date.now()) {
  const mode = ui?.win || "today";
  if (mode === "24h") return { from: iso(now), to: iso(now + 24 * 3600e3) };
  if (mode === "3d") return { from: iso(now), to: iso(now + 72 * 3600e3) };
  if (mode === "custom") {
    const from = Date.parse(ui?.from || "");
    const to = Date.parse(ui?.to || "");
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return null;
    return { from: iso(from), to: iso(to) };
  }
  // Late in the evening "today" is nearly over: keep at least the next 3 hours so a run still finds games
  return { from: iso(now), to: iso(Math.max(endOfDay(now), now + 3 * 3600e3)) };
}

/** Why a custom window can't be used, or "" (it must end after it starts, at most 8 days long, not be over). */
export function windowProblem(ui, now = Date.now()) {
  if (ui?.win !== "custom") return "";
  const from = Date.parse(ui?.from || "");
  const to = Date.parse(ui?.to || "");
  if (!Number.isFinite(from) || !Number.isFinite(to)) return "Pick when the window starts and ends.";
  if (to <= from) return "The window must end after it starts.";
  if (to - from > MAX_WINDOW_DAYS * 86400e3) return `The window can be at most ${MAX_WINDOW_DAYS} days long.`;
  if (to <= now) return "That window is over: pick Today, the next 24 hours or 3 days.";
  return "";
}

/** The person's time zone for the kick-off window (IANA name), or null. */
export function localZone() {
  try {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return typeof zone === "string" && zone.length <= 64 ? zone : null;
  } catch {
    return null;
  }
}

/** The controls as POST /api/builder/run takes them: only FIELDS, the window worked out, the time zone set. */
export function runBody(c, ui, now = Date.now()) {
  const body = {};
  for (const key of FIELDS) {
    const v = c[key];
    if (v === undefined) continue;
    body[key] = copy(v);
  }
  body.window = windowFor(ui, now);
  if (!body.window) delete body.window;
  body.tz = localZone() || c.tz || null;
  if (body.tz === null) delete body.tz;
  // The API takes null for "all books" and "every league"; an empty list would be refused
  if (Array.isArray(body.books) && !body.books.length) body.books = null;
  if (Array.isArray(body.leagues) && !body.leagues.length) body.leagues = null;
  return body;
}

// The "ui" part the server keeps with the controls (dashboard/app.py: at most 12 short values)
const UI_KEYS = ["win", "from", "to", "stake", "back", "at"];

export function cleanUi(raw) {
  const ui = { win: "today", from: null, to: null, stake: null, back: null, at: null };
  if (!raw || typeof raw !== "object") return ui;
  if (WINDOWS.some((w) => w.value === raw.win)) ui.win = raw.win;
  for (const key of ["from", "to"]) if (typeof raw[key] === "string" && Number.isFinite(Date.parse(raw[key])) && raw[key].length <= 64) ui[key] = raw[key];
  for (const key of ["stake", "back"]) if (typeof raw[key] === "string" && MONEY.test(raw[key]) && Number(raw[key]) >= 0.01) ui[key] = raw[key];
  if (typeof raw.at === "string" && Number.isFinite(Date.parse(raw.at))) ui.at = raw.at;
  return ui;
}

/** The body for PUT /api/me {builder}: the run's controls plus the app's own small state. */
export function savedBody(c, ui, now = Date.now()) {
  const body = runBody(c, ui, now);
  const kept = {};
  for (const key of UI_KEYS) if (ui?.[key] != null) kept[key] = String(ui[key]).slice(0, 64);
  kept.at = iso(now);
  body.ui = kept;
  return body;
}

/**
 * The person's last controls: the newer of this device's copy ({controls, ui}) and the server's
 * (GET /api/me "builder": the controls with "label" and "ui"). null when neither is usable.
 */
export function restore(local, server, presets) {
  const pick = [];
  if (local && typeof local === "object" && local.controls) pick.push({ controls: local.controls, ui: local.ui, at: Date.parse(local.ui?.at || "") || 0 });
  if (server && typeof server === "object") pick.push({ controls: server, ui: server.ui, at: Date.parse(server.ui?.at || "") || 0 });
  if (!pick.length) return null;
  pick.sort((a, b) => b.at - a.at);
  const chosen = pick[0];
  const base = presets?.[chosen.controls.preset] || DEFAULTS;
  return { controls: cleanControls(chosen.controls, base), ui: cleanUi(chosen.ui) };
}

// --- small conversions -----------------------------------------------------------------------------------

/** A money amount typed by the person ("10", "$12.50", "1,000") as the API's text, or null. */
export function cleanMoney(value) {
  const text = String(value ?? "").trim().replace(/^\$/, "").replace(/,/g, "");
  if (!MONEY.test(text)) return null;
  const n = Number(text);
  return n >= 0.01 && n <= 100000 ? text : null;
}

/** Target odds from the person's stake and "win about" (the total back): back ÷ stake to 2 places, or null. */
export function targetFrom(stake, back) {
  const s = Number(stake);
  const b = Number(back);
  if (!(s > 0) || !(b > 0)) return null;
  const t = Math.round((b / s) * 100) / 100;
  return t >= 1.01 && t <= 1e6 ? t : null;
}

/** "Win about" from a stake and target odds: stake × odds, to the cent (the person's own amounts). */
export function backFrom(stake, target) {
  const s = Number(stake);
  const t = Number(target);
  if (!(s > 0) || !(t > 1)) return null;
  return (Math.round(s * t * 100) / 100).toFixed(2).replace(/\.00$/, "");
}

/** "3 h", "40 min", "1 h 30 min" for a number of seconds. */
export function ageWords(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return "under 1 min";
  const minutes = Math.floor(s / 60);
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

/** A number for an input box: 10 → "10", 1.5 → "1.5", null → "". */
export function numText(v, places = 2) {
  if (!isNum(v)) return "";
  return String(Number(v.toFixed(places)));
}

/** A share as points for an input box: 0.55 → "55", -0.02 → "-2". */
export function pointsText(v) {
  return isNum(v) ? String(Number((v * 100).toFixed(1))) : "";
}

/**
 * A number typed into a box: {ok, value} where value is null for an empty box (when ``optional``).
 * ``lo``/``hi`` bound it; ``scale`` turns % or points into a share; ``integer`` wants a whole number.
 */
export function parseNumber(text, { lo = -Infinity, hi = Infinity, optional = true, scale = 1, integer = false } = {}) {
  const raw = String(text ?? "").trim().replace(/,/g, ".").replace(/^\+/, "").replace(/[x×%]$/i, "").trim().replace(/^−/, "-");
  if (!raw) return optional ? { ok: true, value: null } : { ok: false, value: null };
  if (!/^-?\d+(\.\d+)?$/.test(raw)) return { ok: false, value: null };
  const n = Number(raw);
  if (!Number.isFinite(n) || n < lo || n > hi || (integer && !Number.isInteger(n))) return { ok: false, value: null };
  return { ok: true, value: scale === 1 ? n : Math.round((n / scale) * 1e6) / 1e6 };
}

/** "1.15–1.80x" for an odds band, "60%+" or "25–60%" for a chance band (formatting the preset's own numbers). */
export function bandText(lo, hi, kind = "odds") {
  if (kind === "odds") {
    if (lo == null && hi == null) return "";
    if (hi == null) return `${Number(lo).toFixed(2)}x+`;
    return `${Number(lo ?? 1).toFixed(2)}–${Number(hi) >= 10 ? Math.round(hi) : Number(hi).toFixed(2)}x`;
  }
  const p = (v) => `${Math.round(v * 100)}`;
  if (lo == null && hi == null) return "";
  if (hi == null || hi >= 1) return `${p(lo ?? 0)}%+`;
  return `${p(lo ?? 0)}–${p(hi)}%`;
}

/**
 * The slate's games in the window that haven't started, counted per league: Map(key -> {count, live: 0}). Games
 * under way never go into a ticket (their prices are from before the start), whatever "exclude started" says.
 */
export function windowCounts(prepared, window, now = Date.now()) {
  const counts = new Map();
  if (!prepared) return counts;
  const from = Date.parse(window?.from || "");
  const to = Date.parse(window?.to || "");
  for (const g of prepared.games) {
    if (g.status !== "pre" || g._start <= now) continue;
    if (Number.isFinite(from) && g._start < from) continue;
    if (Number.isFinite(to) && g._start > to) continue;
    const c = counts.get(g.league) || { count: 0, live: 0 };
    c.count += 1;
    counts.set(g.league, c);
  }
  return counts;
}
