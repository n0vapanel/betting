// Shared client state: who is signed in (GET /api/me), the leagues in season, and the bet slip.
//
// app.js re-exports all of this, so screens import it from "../app.js". It lives in its own module
// because ui.js needs it too (an odds button shows whether its pick is in the slip, in the person's
// odds format) and ui.js must not import app.js.
//
// The slip only ever holds which picks were tapped ({league, id, market, side, point}) plus what the
// button showed, for drawing it before the server answers. Prices, chances and payouts always come
// back from POST /api/tickets/price: the server looks every pick up again in its cached odds.

import { api, prefs } from "./api.js";

// --- who you are ---------------------------------------------------------------------------------

const DEFAULT_ME = {
  uid: null, name: null, is_telegram: false, age_confirmed: false, odds_format: "decimal", currency: "USD",
  favourite_sports: [], risk_tier: null, budget: null, reality_check_minutes: null, cooloff_until: null,
  on_break: false, credits: null, public_url: null, ai_enabled: false,
};

const listeners = { me: new Set(), slip: new Set(), priced: new Set() };
const emit = (kind, value) => listeners[kind].forEach((fn) => { try { fn(value); } catch (e) { console.warn(e); } });

export const session = {
  me: null, // GET /api/me, or DEFAULT_ME while it's unknown
  sports: null, // GET /api/sports: [{key, title, group, label, ai}]
  quota: null, // the odds feed's {remaining, used}, from the last /api/today
};

/** The person's settings and limits (never null: defaults until /api/me answers). */
export function me() {
  return session.me || { ...DEFAULT_ME, odds_format: prefs.get("format") === "american" ? "american" : "decimal" };
}

/** Load (or reload) GET /api/me. A server without the route yet gets the defaults. */
export async function loadMe() {
  try {
    setMe(await api("/api/me"));
  } catch (error) {
    if (error.status === 404) setMe({ ...DEFAULT_ME, age_confirmed: true, odds_format: me().odds_format });
    else throw error;
  }
  return session.me;
}

export function setMe(value) {
  session.me = { ...DEFAULT_ME, ...value };
  prefs.set("format", session.me.odds_format);
  emit("me", session.me);
}

/** PUT /api/me with only the fields that change; returns the new profile. */
export async function updateMe(changes) {
  setMe(await api("/api/me", { method: "PUT", body: changes }));
  return session.me;
}

export function onMeChange(fn) {
  listeners.me.add(fn);
  return () => listeners.me.delete(fn);
}

/**
 * On a break right now (on_break, until cooloff_until). The server refuses pricing, analysing and saving
 * a slip then (as well as picks and plans), so the slip's Analyse and Save are off and its old price goes.
 */
export function onBreak(profile = me()) {
  if (!profile.on_break) return false;
  if (!profile.cooloff_until) return true;
  const until = Date.parse(profile.cooloff_until);
  return Number.isFinite(until) && until > Date.now();
}

/** The leagues shown before anyone has picked any (Today, Settings, the sports sheet, Build and Ask AI). */
export const DEFAULT_SPORTS = ["basketball_nba"];

/** The person's favourite leagues, or the default ones until they save some. */
export function favouriteSports(profile = me()) {
  const list = profile.favourite_sports || [];
  return list.length ? [...list] : [...DEFAULT_SPORTS];
}

/** "decimal" or "american". */
export function oddsFormat() {
  return me().odds_format === "american" ? "american" : "decimal";
}

/** The leagues in season (cached for the visit). */
export async function loadSports() {
  if (!session.sports) session.sports = await api("/api/sports");
  return session.sports;
}

export function leagueInfo(key) {
  return (session.sports || []).find((s) => s.key === key) || null;
}

const SPORT_EMOJI = {
  basketball: "🏀", americanfootball: "🏈", baseball: "⚾", icehockey: "🏒", soccer: "⚽", tennis: "🎾",
  mma: "🥊", boxing: "🥊", golf: "⛳", cricket: "🏏", rugbyleague: "🏉", rugbyunion: "🏉", aussierules: "🏉",
};

/** "🏀" for basketball_nba (from the league key's sport prefix). */
export function sportEmoji(key) {
  return SPORT_EMOJI[String(key || "").split("_")[0]] || "🏟️";
}

/** "NBA" for basketball_nba (falls back to the key until /api/sports has loaded). */
export function leagueTitle(key) {
  const info = leagueInfo(key);
  if (info) return info.title;
  const parts = String(key || "").split("_");
  return parts.length > 1 ? parts.slice(1).join(" ").toUpperCase() : String(key || "");
}

// --- the bet slip ----------------------------------------------------------------------------------

const SLIP_KEY = "sportsbet-slip-v1";
export const MAX_LEGS = 12;
const TYPES = ["single", "multi", "round_robin", "system"];

/** A pick's identity without its line, so a moved line is still the same pick. */
export function legKey(ref) {
  return `${ref.league}:${ref.id}:${ref.market}:${ref.side}`;
}

function loadSlip() {
  try {
    const saved = JSON.parse(localStorage.getItem(SLIP_KEY) || "null");
    if (saved && Array.isArray(saved.legs)) {
      return {
        legs: saved.legs.filter((leg) => leg?.ref?.league && leg.ref.id).slice(0, MAX_LEGS),
        type: TYPES.includes(saved.type) ? saved.type : "multi",
        stake: /^\d{1,6}(\.\d{1,2})?$/.test(String(saved.stake)) ? String(saved.stake) : "10",
        k: Number.isInteger(saved.k) ? saved.k : null,
        system: typeof saved.system === "string" ? saved.system : null,
      };
    }
  } catch {
    /* a broken copy starts a fresh slip */
  }
  return { legs: [], type: "multi", stake: "10", k: null, system: null };
}

let state = loadSlip();

function saveSlip() {
  try {
    localStorage.setItem(SLIP_KEY, JSON.stringify(state));
  } catch {
    /* private browsing: the slip lasts for this visit */
  }
}

function changed() {
  saveSlip();
  emit("slip", slip);
}

// Another tab changed the slip: follow it
window.addEventListener("storage", (event) => {
  if (event.key === SLIP_KEY) {
    state = loadSlip();
    emit("slip", slip);
  }
});

/**
 * The slip store.
 *   slip.legs                -> [{ref, key, label, game, start, league, price, american, ai, fair}]
 *   slip.has(refOrKey)       -> boolean
 *   slip.add(leg) / slip.remove(refOrKey) / slip.toggle(leg) -> toggle returns true when added
 *   slip.clear()
 *   slip.set({type, stake, k, system})
 *   slip.body()              -> the request body for /api/tickets/price, /analyse, /api/slips
 *   slip.onChange(fn)        -> unsubscribe function
 * A leg needs at least {ref: {league, id, market, side, point}}; label/game/price are only for drawing.
 * Adding the other side of a market already in the slip (e.g. the away team after the home team)
 * swaps it, the way sportsbooks do.
 */
export const slip = {
  get legs() {
    return state.legs;
  },
  get count() {
    return state.legs.length;
  },
  get type() {
    return state.type;
  },
  get stake() {
    return state.stake;
  },
  get k() {
    return state.k;
  },
  get system() {
    return state.system;
  },
  has(refOrKey) {
    const key = typeof refOrKey === "string" ? refOrKey : legKey(refOrKey);
    return state.legs.some((leg) => leg.key === key);
  },
  add(leg) {
    const ref = cleanRef(leg.ref);
    const key = legKey(ref);
    if (slip.has(key)) return true;
    const sameMarket = (l) => l.ref.league === ref.league && l.ref.id === ref.id && l.ref.market === ref.market;
    const legs = state.legs.filter((l) => !sameMarket(l));
    if (legs.length >= MAX_LEGS) return false;
    legs.push({ ...leg, ref, key, added: Date.now() });
    state = { ...state, legs };
    changed();
    return true;
  },
  remove(refOrKey) {
    const key = typeof refOrKey === "string" ? refOrKey : legKey(refOrKey);
    state = { ...state, legs: state.legs.filter((leg) => leg.key !== key) };
    changed();
  },
  /** Add or remove; returns true when the pick is now in the slip, false when removed, null when full. */
  toggle(leg) {
    if (slip.has(leg.ref)) {
      slip.remove(leg.ref);
      return false;
    }
    return slip.add(leg) ? true : null;
  },
  clear() {
    state = { ...state, legs: [] };
    changed();
  },
  set({ type, stake, k, system } = {}) {
    const next = { ...state };
    if (type !== undefined && TYPES.includes(type)) next.type = type;
    if (stake !== undefined) next.stake = String(stake);
    if (k !== undefined) next.k = k;
    if (system !== undefined) next.system = system;
    state = next;
    changed();
  },
  refs() {
    return state.legs.map((leg) => ({ ...leg.ref }));
  },
  body() {
    const body = { type: state.type, stake: state.stake, legs: slip.refs() };
    if (state.type === "round_robin") body.k = state.k || 2;
    if (state.type === "system" && state.system) body.system = state.system;
    return body;
  },
  onChange(fn) {
    listeners.slip.add(fn);
    return () => listeners.slip.delete(fn);
  },
};

function cleanRef(ref) {
  const point = ref.point === undefined || ref.point === null || ref.point === "" ? null : Number(ref.point);
  return { league: String(ref.league), id: String(ref.id), market: ref.market, side: ref.side,
    point: Number.isFinite(point) ? point : null };
}

/** A slip leg from a game card's offer: what an odds button adds. */
export function legFromOffer(card, market, side, offer) {
  return {
    ref: { league: card.league, id: String(card.id), market, side, point: offer.point ?? null },
    label: offer.label || `${side} ${market}`,
    game: `${card.away} @ ${card.home}`,
    start: card.start,
    league: card.league,
    price: offer.price,
    american: offer.american,
    ai: card.prediction ? offer.ai : null, // without a model the card's "ai" is the market again
    fair: offer.fair,
  };
}

// --- pricing (POST /api/tickets/price) ----------------------------------------------------------------

let pricing = { body: null, result: null, error: null, busy: false, controller: null };

/** The last pricing of the current slip: {busy, result, error}. result is the server's priced ticket. */
export function priced() {
  return { busy: pricing.busy, result: pricing.result, error: pricing.error };
}

export function onPriced(fn) {
  listeners.priced.add(fn);
  return () => listeners.priced.delete(fn);
}

/**
 * Price the slip on the server (skipped when nothing changed). Resolves to the priced ticket or null.
 * Errors land in priced().error (an ApiError) and are not thrown.
 */
export async function priceSlip({ force = false } = {}) {
  // An empty slip, or a break (the server prices nothing then): drop the old price without asking
  if (!state.legs.length || onBreak()) {
    pricing.controller?.abort();
    pricing = { body: null, result: null, error: null, busy: false, controller: null };
    emit("priced", priced());
    return null;
  }
  const body = slip.body();
  const text = JSON.stringify(body);
  if (!force && text === pricing.body && (pricing.result || pricing.busy)) return pricing.result;
  pricing.controller?.abort();
  const controller = new AbortController();
  pricing = { ...pricing, body: text, busy: true, error: null, controller };
  emit("priced", priced());
  try {
    const result = await api("/api/tickets/price", { method: "POST", body, signal: controller.signal });
    if (pricing.controller !== controller) return pricing.result;
    pricing = { ...pricing, result, busy: false, controller: null };
  } catch (error) {
    if (error.name === "AbortError" || pricing.controller !== controller) return pricing.result;
    pricing = { ...pricing, result: null, error, busy: false, controller: null };
  }
  emit("priced", priced());
  return pricing.result;
}
