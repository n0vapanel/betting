// "Suggested picks" on the game page (views/game.js draws it right after the action bar): the brain's pick for each
// of the game's markets and the game's best player props. GET /api/game/picks?league=&id= (Lab.game_picks over
// engine/builder.py game_picks) is asked once the page has drawn: the page never waits for it, and a skeleton holds
// the section's place until the answer arrives.
//
// The answer: {picks: [{ref, key, market, market_title, label, chance, market_chance, price, book, edge, brain_higher,
// why, why_units?}], props: {likeliest: [row], edges: [row], day?}, game: {label, start, started}, unpriced?:
// {reason}, loading?: {games_ready, games_total, failed?}, as_of}. A pick is the Builder's candidate with the highest
// chance in its market (the main line for spreads and totals), priced at the best book a person can bet at: never a
// sharp book or an exchange, never the goals model's fair odds, never a stale price, never once the game has started.
// A props row is the player scan's (views/players/scan.js draws it, as the Player finder and Ask AI's card do).
//
// Each pick shows its market and label; "Brain 64% · Market 58%" side by side (the brain's alone without a market
// chance); a neutral "Brain higher than market" chip when the server says so (never an "edge" or "value" on a game
// pick: owner decision 4); the best price and its book; a "Why?" toggle over its reasons; and "Add to slip"
// (slip.add with the leg the Builder's and Ask AI's cards add, shown as "In your slip" once there: a second tap takes
// it out). The slip takes moneylines, spreads and totals only (it prices soccer's goals-model markets at fair odds,
// which nobody can bet), so a soccer market's pick shows its price and book with the button off and a line saying so.
// The props: "Likeliest" and "Best edges" (up to 3 each, the scan's rows: a posted line shows its edge, as in the
// scan) and "See all in the Player finder" (its Best edges (all stats) for the league and the game's day).
//
// The states: picks paused (the pause note and a link to Settings, no picks: no request is made when /api/me already
// says so, and a 400 refusal is shown in the server's own words), prices not in yet (while the league's board is on
// its way, "Prices for this game are on their way…" with a spinner, and the section asks again by itself at about 15 s
// and 45 s, then gives the server's reason; any other reason, such as the month's odds allowance at its reserve or
// today's used up, in the server's own words at once), the game under way (no request when the page already shows
// it live), nothing at all, player props still loading;
// and always the line "No pick is certain. Only you add bets to your slip." Every number shown is the server's; every
// string from outside is escaped.

import { api, qs, isQuiet } from "../../api.js";
import { slip } from "../../app.js";
import { esc, pct, priceText, skeleton, notice, section, toast } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { gateHint, picksBlock } from "../build/common.js";
import { american } from "../today/model.js";
import { unitsFor } from "../context/units.js";
import { rowItemHtml, scanLeague, statsFor, finderHash, EDGES, ensureScanStyles } from "../players/scan.js";

export const TITLE = "Suggested picks";
export const SECTION_ID = "suggested-picks";
export const DISCLOSURE = "No pick is certain. Only you add bets to your slip.";
export const WORDS = {
  onTheirWay: "Prices for this game are on their way…", // while the board is asked for or loading (with a spinner)
  unpricedPlain: "Prices for this game aren't in yet.", // no reason given: nothing said about asking

  started: "This game has started: picks are for games still to come.",
  empty: "No picks for this game yet.",
  loading: "Player props are still loading.",
  noSlip: "The slip takes moneylines, spreads and totals.",
};
export const SETTINGS_HREF = "#/settings?focus=limits";
// The markets the slip takes (Ask AI's pickLeg and the scan's rowLeg take the same three)
export const SLIP_MARKETS = ["h2h", "spreads", "totals"];
// engine/builder.py MARKET_TITLES, for a pick that comes without its title
const MARKET_TITLES = {
  h2h: "Moneyline", spreads: "Spread / handicap", totals: "Totals", team_totals: "Team totals",
  btts: "Both teams to score", draw_no_bet: "Draw no bet", double_chance: "Double chance",
};
export const PROPS_SHOWN = 3; // Lab.game_picks sends up to 3 of each
// A game with no prices (``unpriced: {reason, code}``): ``code`` is the reason's key (dashboard/lab.py _price_why:
// on_way, arriving, reserve, daily_cap, no_games). Only "on_way" (the league's board asked for, loading or on its way;
// the slate's own "queued" and "loading" read the same) is waited for: "Prices for this game are on their way…" with a
// spinner, and the route asked again by the section itself after RETRY_MS[0], then RETRY_MS[1] later (about 15 s and
// 45 s in). Still unpriced after the last try, or any other code (the month's allowance at its reserve, today's used
// up, no games, not arrived), the server's own reason is shown at once. A server that sends no code: its reason read
// loosely ("on its way", "on their way", "loading").
const ON_WAY = ["on_way", "queued", "loading"];
const ON_WAY_WORDS = /\bon (?:its|their) way\b|\bloading\b/i;
export const RETRY_MS = [15000, 30000];
const DAY_TZ = "America/New_York"; // ESPN's day, as the player board counts it (data/espn_client.py ESPN_DAY_TZ)

const num = (value) => {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

/** A pick's decimal price when it is one a person can take (above 1), else null. */
function priceOf(pick) {
  const price = num(pick?.price);
  return price !== null && price > 1 ? price : null;
}

/**
 * The slip leg a game pick adds: the shape the Builder's (build/common.js slipLeg) and Ask AI's (ask/cards.js
 * pickLeg) cards add, {ref: {league, id, market, side, point}, label, game, start, league, price, american, ai, fair,
 * adjusted?} (``adjusted``: its chance moved for a forecast or roof change, so the slip says "Adj", as it does for the
 * Builder's leg and the page's own price button). Null when the slip can't take it: not a moneyline, spread or total,
 * no usable price, or a pick for another game.
 */
export function gamePickLeg(pick, game = null) {
  const ref = pick?.ref;
  if (!ref || typeof ref !== "object" || !ref.league || ref.id === null || ref.id === undefined || ref.id === "") return null;
  if (!SLIP_MARKETS.includes(ref.market) || typeof ref.side !== "string" || !ref.side) return null;
  if (game && (String(ref.league) !== String(game.league) || String(ref.id) !== String(game.id))) return null;
  const price = priceOf(pick);
  if (price === null) return null;
  return {
    ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
    label: pick.label,
    game: game ? `${game.away} @ ${game.home}` : pick.game,
    start: game ? game.start : pick.start,
    league: ref.league,
    price,
    american: american(price) || undefined,
    ai: num(pick.chance),
    fair: num(pick.market_chance),
    ...(pick.adjusted ? { adjusted: true } : {}), // as the Builder's leg: "Adj" on the slip
  };
}

/** Whether the slip takes a pick's market at all (a soccer goals-model market it doesn't). */
function slipTakes(pick) {
  return SLIP_MARKETS.includes(pick?.ref?.market ?? pick?.market);
}

/** The answer, read defensively: {picks, likeliest, edges, loading, started, unpriced, day}. */
export function readPicks(data) {
  const raw = data && typeof data === "object" ? data : {};
  const props = raw.props && typeof raw.props === "object" ? raw.props : {};
  const rows = (list) => (Array.isArray(list) ? list : []).filter((row) => row && typeof row === "object").slice(0, PROPS_SHOWN);
  const loading = [raw.loading, props.loading].find((value) => value && typeof value === "object") || null;
  const unpriced = raw.unpriced === true || (raw.unpriced && typeof raw.unpriced === "object") ? raw.unpriced : null;
  return {
    // A pick with no price a person can take is never shown
    picks: (Array.isArray(raw.picks) ? raw.picks : []).filter((pick) => pick && typeof pick === "object" && priceOf(pick) !== null),
    likeliest: rows(props.likeliest),
    edges: rows(props.edges),
    loading,
    started: raw.game?.started === true,
    unpriced,
    // The board's day ("today" | "tomorrow"); null when the server says the game is on neither day (or its league has
    // no scan); undefined from a server that doesn't say (then the game's start decides: boardDay)
    day: props.day === "today" || props.day === "tomorrow" ? props.day : "day" in props ? null : undefined,
  };
}

/** "today" or "tomorrow" for a game's start on ESPN's day (US Eastern), else null (or for a start it can't read). */
export function boardDay(start, now = Date.now()) {
  const at = Date.parse(start || "");
  if (!Number.isFinite(at)) return null;
  try {
    const format = new Intl.DateTimeFormat("en-US", { timeZone: DAY_TZ, year: "numeric", month: "2-digit", day: "2-digit" });
    const ymd = (ms) => {
      const parts = Object.fromEntries(format.formatToParts(new Date(ms)).map((p) => [p.type, p.value]));
      return `${parts.year}-${parts.month}-${parts.day}`;
    };
    const today = ymd(now);
    const [y, m, d] = today.split("-").map(Number);
    const tomorrow = new Date(Date.UTC(y, m - 1, d + 1)).toISOString().slice(0, 10);
    const day = ymd(at);
    return day === today ? "today" : day === tomorrow ? "tomorrow" : null;
  } catch {
    return null;
  }
}

/**
 * A game with no prices ({reason, code}) as the section reads it: {onWay, text}. ``onWay``: the prices are on their way
 * (code "on_way", or the slate's "queued" or "loading"; with no code, the reason says so), so the section waits for
 * them; ``text``: the server's own reason as a sentence (shown at once for any other code, and after the last try),
 * or, with no reason at all, only that the prices aren't in yet.
 */
export function unpricedSay(unpriced) {
  const code = typeof unpriced?.code === "string" ? unpriced.code.trim().toLowerCase() : "";
  const reason = typeof unpriced?.reason === "string" ? unpriced.reason.replace(/\s+/g, " ").trim() : "";
  const onWay = code ? ON_WAY.includes(code) : ON_WAY_WORDS.test(reason.replace(/’/g, "'"));
  if (!reason) return { onWay, text: WORDS.unpricedPlain };
  const sentence = `${reason[0].toUpperCase()}${reason.slice(1)}`;
  return { onWay, text: /[.!?…]$/.test(sentence) ? sentence : `${sentence}.` };
}

/** The Player finder's Best edges (all stats) for a league and day, or "" for a league or day it doesn't cover. */
export function finderLink(league, day) {
  const stat = statsFor(league)[0]?.key;
  if (!scanLeague(league) || !stat || (day !== "today" && day !== "tomorrow")) return "";
  return finderHash({ league, stat, threshold: null, day, sort: EDGES });
}

/** A pick's reasons in the person's units (why_units, as Ask AI's card reads them), at most 3. */
function whyLines(pick, league) {
  const units = pick?.why_units && typeof pick.why_units === "object" ? pick.why_units[unitsFor(league)] : null;
  const lines = Array.isArray(units) && units.length ? units : pick?.why;
  return (Array.isArray(lines) ? lines : [lines])
    .filter((line) => line !== null && line !== undefined && String(line).trim())
    .map((line) => String(line))
    .slice(0, 3);
}

/** The key a pick's "Why?" toggle is remembered by (one pick a market), so a redraw keeps it open. */
const whyKey = (pick, i) => String(pick?.market || pick?.ref?.market || i).replace(/[^\w-]/g, "_");

// --- drawing -------------------------------------------------------------------------------------------------

/** "Add to slip", "In your slip", or the button off (with why, for a market the slip doesn't take). */
function addButton(pick, i, leg) {
  const label = String(pick?.label || "this pick");
  if (!leg) {
    return `<button type="button" class="btn btn-secondary small" disabled>${icon("plus", { size: 16 })}Add to slip</button>${
      slipTakes(pick) ? "" : `<span class="gp-noslip">${esc(WORDS.noSlip)}</span>`}`;
  }
  return slip.has(leg.ref)
    ? `<button type="button" class="btn btn-secondary small" data-gp-add="${i}" aria-pressed="true" aria-label="${esc(`${label}: in your slip. Tap to take it out`)}">${icon("check", { size: 16 })}In your slip</button>`
    : `<button type="button" class="btn btn-primary small" data-gp-add="${i}" aria-pressed="false" aria-label="${esc(`Add ${label} to your slip`)}">${icon("plus", { size: 16 })}Add to slip</button>`;
}

/** One game pick: market and label, the brain's and the market's chances, the price and book, Why? and the slip. */
function pickHtml(pick, i, v) {
  const league = pick.ref?.league || v.game.league;
  const price = priceOf(pick);
  const brain = num(pick.chance);
  const market = num(pick.market_chance);
  const higher = pick.brain_higher === true && market !== null;
  const why = whyLines(pick, league);
  const key = whyKey(pick, i);
  const title = pick.market_title || MARKET_TITLES[pick.market] || MARKET_TITLES[pick.ref?.market] || "";
  return `<li class="gp-pick">
    <div class="gp-main">
      ${title ? `<span class="gp-market">${esc(title)}</span>` : ""}
      <b class="gp-label">${esc(pick.label || title || "Pick")}</b>
      <div class="gp-nums"><span class="gp-chances">Brain <b class="num">${esc(pct(brain))}</b>${market !== null
        ? `<span class="gp-dot" aria-hidden="true"> · </span><span class="sr-only">, </span>Market <b class="num">${esc(pct(market))}</b>` : ""}</span>${
        higher ? `<span class="badge muted gp-higher">Brain higher than market</span>` : ""}</div>
      <div class="gp-price"><b class="num">${esc(priceText({ decimal: price, american: american(price) }))}</b>${pick.book ? ` <span class="muted">at ${esc(pick.book)}</span>` : ""}</div>
    </div>
    <div class="gp-add">${addButton(pick, i, gamePickLeg(pick, v.game))}</div>
    ${why.length ? `<details class="gp-why" data-gp-why="${esc(key)}"${v.open.has(key) ? " open" : ""}><summary>${icon("chevron", { size: 16 })}Why?</summary>
      <ul>${why.map((line) => `<li>${esc(line)}</li>`).join("")}</ul></details>` : ""}
  </li>`;
}

/** "Check again" (asks the route once more), or its busy state while that answer is on its way. */
function againButton(busy) {
  return `<button type="button" class="chip gp-again" data-gp-retry${busy ? ' disabled aria-busy="true"' : ""}>${icon("refresh", { size: 16 })}${busy ? "Checking…" : "Check again"}</button>`;
}

/** A one-line state ("No picks for this game yet."), announced politely when it arrives. */
function stateLine(iconName, text, { again = false, busy = false } = {}) {
  return `<p class="gp-state" role="status">${icon(iconName, { size: 16 })}<span>${esc(text)}</span>${again ? againButton(busy) : ""}</p>`;
}

/** "Prices for this game are on their way…" with a small spinner, while the section waits for them by itself. */
function waitingLine() {
  return `<p class="gp-state gp-wait" role="status"><span class="gp-spin" aria-hidden="true"></span><span>${esc(WORDS.onTheirWay)}</span></p>`;
}

const caption = (text) => `<h3 class="gp-caption">${esc(text)}</h3>`;

/** The pause note, in the server's words or /api/me's, with the way to Settings; never a pick. */
function pauseHtml(text) {
  return `<div class="notice info gp-pause" role="note">${icon("pause", { size: 18 })}<div><span>${esc(text || "Picks are paused for now.")}</span>
    <a class="text-link" href="${esc(SETTINGS_HREF)}">Settings${icon("chevron", { size: 16 })}</a></div></div>`;
}

function errorHtml(v) {
  return `<div class="gp-error">${notice(`Couldn't load the suggested picks. ${v.error || ""}`.trim(), "warn")}
    <button type="button" class="btn btn-secondary small gp-retry" data-gp-retry${v.busy ? ' disabled aria-busy="true"' : ""}>${icon("refresh", { size: 16 })}${v.busy ? "Trying again…" : "Try again"}</button></div>`;
}

/** The player props: Likeliest and Best edges (the scan's own rows), still loading, and the finder for the day. */
function propsHtml(d, v, { again = true } = {}) {
  const { likeliest, edges, loading } = d;
  if (!likeliest.length && !edges.length && !loading) return "";
  const league = v.game.league;
  const list = (rows, label) => (rows.length
    ? `<h4 class="gp-sub">${esc(label)}</h4><ol class="ps-list" aria-label="${esc(label)}">${rows.map((row, i) =>
      rowItemHtml({ ...row, league: row.league || league, rank: i + 1 }, i, { view: { edges: true, league } })).join("")}</ol>`
    : "");
  const href = finderLink(league, d.day === undefined ? boardDay(v.game.start) : d.day);
  return `<div class="gp-props">${caption("Player props")}
    ${loading ? stateLine("clock", WORDS.loading, { again, busy: v.busy }) : ""}
    ${list(likeliest, "Likeliest")}${list(edges, "Best edges")}
    ${href ? `<a class="text-link gp-finder" href="${esc(href)}">${icon("users", { size: 16 })}See all in the Player finder${icon("chevron", { size: 16 })}</a>` : ""}
  </div>`;
}

function bodyHtml(v) {
  if (v.status === "loading") return skeleton("lines", 3);
  if (v.status === "paused") return pauseHtml(v.pause);
  if (v.status === "started") return stateLine("clock", WORDS.started);
  if (v.status === "error") return errorHtml(v);
  const d = v.data || readPicks(null);
  // Under way: nothing priced is shown (the props' lines are prices too)
  if (d.started) return stateLine("clock", WORDS.started);
  const parts = [];
  // No prices: on their way (the section asks again by itself until it gives up), else the server's own reason
  const unpriced = d.unpriced ? unpricedSay(d.unpriced) : null;
  const waiting = Boolean(unpriced?.onWay && !v.gaveUp);
  if (unpriced) parts.push(waiting ? waitingLine() : stateLine("info", unpriced.text));
  if (d.picks.length) parts.push(`${caption("Game picks")}<ol class="gp-list" aria-label="The brain's pick for each market">${d.picks.map((pick, i) => pickHtml(pick, i, v)).join("")}</ol>`);
  const props = propsHtml(d, v, { again: !waiting }); // while waiting, the section's own tries refresh the props too
  if (props) parts.push(props);
  return parts.join("") || stateLine("info", WORDS.empty);
}

/**
 * The whole section for a state: {game, status ("loading" | "paused" | "started" | "error" | "ready"), data
 * (readPicks), pause (the note's words), error, busy, open (the "Why?" toggles left open), gaveUp (prices on their way
 * were asked for again RETRY_MS.length times and still aren't in: the server's reason now)}.
 */
export function sectionHtml(v) {
  const view = { open: new Set(), busy: false, gaveUp: false, ...v };
  return section(TITLE, `<div class="gp-body">${bodyHtml(view)}</div><p class="note gp-disclosure">${esc(DISCLOSURE)}</p>`,
    { id: SECTION_ID, className: "gp" });
}

// --- the section on a page -----------------------------------------------------------------------------------

/** The element to focus again after a redraw, as a selector ("" for none of the section's own). */
function focusKey(el) {
  const add = el?.closest?.("[data-gp-add]");
  if (add) return `[data-gp-add="${add.dataset.gpAdd}"]`;
  if (el?.closest?.("[data-gp-retry]")) return "[data-gp-retry]";
  const why = el?.closest?.("[data-gp-why]");
  if (why && el.closest("summary")) return `[data-gp-why="${why.dataset.gpWhy}"] > summary`;
  return "";
}

/**
 * The section's controller for one game page (or the inspector's copy of it):
 *   const picks = gamePicks(game, {signal})
 *   picks.html()        the section as it stands (the page draws it into <div data-gp>, also on a redraw)
 *   picks.start(root)   listen on the page's root (Add to slip, Check again, the Why? toggles), follow the slip, and
 *                       ask the route (unless the game is live or picks are paused: then no request is made)
 *   picks.stop()        stop following the slip and stop waiting for prices (the page's cleanup)
 * ``signal`` (the page's) cancels the request and the waiting when the page goes. Prices on their way are asked for
 * again by the section itself (RETRY_MS: only the section is redrawn); ``clock`` ({set(fn, ms), clear(id)}, the page's
 * timers by default) is for tests.
 */
export function gamePicks(game, { signal = null, clock = null } = {}) {
  const v = { game, status: "loading", data: null, pause: "", error: "", busy: false, open: new Set(), gaveUp: false };
  if (game?.live) {
    v.status = "started";
  } else {
    const block = picksBlock();
    if (block) {
      v.status = "paused";
      v.pause = block.text;
    }
  }
  const timers = clock || { set: (fn, ms) => setTimeout(fn, ms), clear: (id) => clearTimeout(id) };
  let root = null;
  let seq = 0;
  let stopped = false;
  let offSlip = null;
  let timer = null; // the next try while prices are on their way
  let tries = 0; // the tries made for them so far
  ensureStyles();

  const html = () => sectionHtml(v);

  function paint() {
    if (stopped || !root || root.isConnected === false) return;
    const box = root.querySelector("[data-gp]");
    if (!box) return;
    const active = typeof document !== "undefined" ? document.activeElement : null;
    const again = active && box.contains?.(active) ? focusKey(active) : "";
    box.innerHTML = html();
    if (again) box.querySelector?.(again)?.focus?.({ preventScroll: true });
  }

  /**
   * After each answer: prices on their way are asked for again after RETRY_MS[tries]; once every try is made and they
   * still aren't in, the server's reason is shown (gaveUp). Anything else (prices, another reason, a pause, an error)
   * ends the waiting.
   */
  function schedule() {
    timers.clear(timer);
    timer = null;
    const d = v.status === "ready" ? v.data : null;
    const onWay = Boolean(d && !d.started && d.unpriced && unpricedSay(d.unpriced).onWay);
    v.gaveUp = onWay && tries >= RETRY_MS.length;
    if (!onWay || v.gaveUp || stopped) return;
    timer = timers.set(() => {
      timer = null;
      tries += 1;
      load({ quiet: true });
    }, RETRY_MS[tries]);
  }

  /** Ask the route. ``quiet`` (the section's own tries): a failed try keeps what is shown and counts as a try. */
  async function load({ quiet = false } = {}) {
    const mine = ++seq;
    try {
      const data = await api(`/api/game/picks${qs({ league: game.league, id: game.id })}`, { signal });
      if (mine !== seq || stopped) return;
      v.status = "ready";
      v.data = readPicks(data);
    } catch (error) {
      if (mine !== seq || stopped || isQuiet(error)) return;
      const hint = gateHint(error);
      if (error?.status === 400 && (hint?.kind === "paused" || hint?.kind === "break")) {
        v.status = "paused"; // the server's own words: never a pick
        v.pause = error.message;
      } else if (quiet) {
        // A try that failed: what is shown stays (the next try, or the server's reason after the last)
      } else if (error?.status === 404) {
        v.status = "ready"; // the game went, or a server without the route yet: nothing to suggest
        v.data = readPicks(null);
      } else {
        v.status = "error";
        v.error = error?.message || "";
      }
    }
    v.busy = false;
    schedule();
    paint();
  }

  function again() {
    if (v.busy || stopped) return;
    v.busy = true;
    paint();
    load();
  }

  function stop() {
    stopped = true;
    timers.clear(timer);
    timer = null;
    offSlip?.();
    offSlip = null;
    root?.removeEventListener?.("click", onClick);
    root?.removeEventListener?.("toggle", onToggle, true);
  }

  function onClick(event) {
    const target = event.target;
    const add = target?.closest?.("[data-gp-add]");
    if (add) {
      // The person's tap only: on the slip, or off it again
      if (v.status !== "ready" || v.data?.started) return;
      const pick = v.data?.picks?.[Number(add.dataset.gpAdd)];
      const leg = pick ? gamePickLeg(pick, game) : null;
      if (!leg) return;
      if (slip.has(leg.ref)) {
        slip.remove(leg.ref);
        haptic.select();
        toast(`Took ${leg.label || "the pick"} off your slip`, { kind: "info" });
      } else if (slip.add(leg)) {
        haptic.select();
        toast(`Added ${leg.label || "the pick"} to your slip`, { kind: "good" });
      } else {
        toast("Your slip is full: remove a pick first", { kind: "warn" });
      }
      return;
    }
    if (target?.closest?.("[data-gp-retry]")) again();
  }

  function onToggle(event) {
    const box = event.target;
    const key = box?.dataset?.gpWhy;
    if (!key) return;
    if (box.open) v.open.add(key);
    else v.open.delete(key);
  }

  return {
    html,
    state: () => v,
    start(el) {
      if (root || stopped) return;
      root = el;
      root.addEventListener("click", onClick);
      root.addEventListener("toggle", onToggle, true); // "toggle" doesn't bubble
      offSlip = slip.onChange(() => paint()); // an odds button, the slip sheet or another tab changed it
      signal?.addEventListener?.("abort", stop, { once: true }); // the page went: no more tries
      if (v.status === "loading") load();
    },
    stop,
  };
}

// --- styles (added once; every rule is scoped to the section) ----------------------------------------------------

const CSS = `
.gp .gp-body { display: grid; gap: 0; min-width: 0; }
.gp .gp-body > .notice:first-child { margin-top: 0; }
.gp-caption { margin: 14px 0 8px; font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-3); }
.gp-body > .gp-caption:first-child, .gp-body > .gp-props:first-child > .gp-caption:first-child { margin-top: 0; }
.gp-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.gp-pick {
  display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; gap: 4px 12px; padding: 10px 12px;
  border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); min-width: 0;
}
.gp-main { display: grid; gap: 3px; min-width: 0; }
.gp-market { font-size: 12.5px; font-weight: 650; color: var(--text-2); overflow-wrap: anywhere; }
.gp-label { font-size: 15px; min-width: 0; overflow-wrap: anywhere; }
.gp-nums { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: 13.5px; color: var(--text-2); }
.gp-nums b { color: var(--text); font-size: 14.5px; }
.gp-chances { white-space: nowrap; }
.gp-higher { white-space: normal; }
.gp-price { font-size: 13.5px; min-width: 0; overflow-wrap: anywhere; }
.gp-price b { font-size: 14.5px; }
.gp-add { align-self: center; display: grid; gap: 4px; justify-items: end; min-width: 0; }
.gp-add .btn { white-space: nowrap; }
.gp-noslip { font-size: 12px; color: var(--text-2); text-align: right; max-width: 190px; line-height: 1.35; }
.gp-why { grid-column: 1 / -1; min-width: 0; }
.gp-why > summary {
  cursor: pointer; display: inline-flex; align-items: center; gap: 4px; min-height: var(--tap); padding-right: 8px;
  font-weight: 650; font-size: 13.5px; color: var(--text-2); list-style: none;
}
.gp-why > summary::-webkit-details-marker { display: none; }
.gp-why > summary .icon { flex: none; transition: transform var(--speed); }
.gp-why[open] > summary .icon { transform: rotate(90deg); }
.gp-why ul { margin: 0 0 4px; padding-left: 20px; color: var(--text-2); font-size: 13px; line-height: 1.45; }
.gp-why li { overflow-wrap: anywhere; }
.gp-state { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 0; font-size: 14px; color: var(--text-2); line-height: 1.45; }
.gp-state > .icon { flex: none; }
.gp-state > span { flex: 1 1 200px; min-width: 0; }
/* prices on their way: a small spinner (still, for people who ask for reduced motion: styles.css stops animations) */
.gp-state > .gp-spin {
  flex: none; width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--line-strong);
  border-top-color: var(--text-2); animation: gp-spin 0.9s linear infinite;
}
@keyframes gp-spin { to { transform: rotate(360deg); } }
.gp-props { min-width: 0; }
.gp-body > * + .gp-props { margin-top: 2px; }
.gp-props > .gp-state { margin-bottom: 8px; }
.gp-sub { margin: 10px 0 6px; font-size: 14px; font-weight: 700; color: var(--text-2); }
.gp-props > .gp-state + .gp-sub, .gp-props > .gp-caption + .gp-sub { margin-top: 0; }
.gp-finder { margin-top: 6px; gap: 6px; }
.gp-error { display: grid; gap: 8px; justify-items: start; }
.gp-error > .notice { margin-top: 0; justify-self: stretch; }
.gp .gp-disclosure { margin-top: 12px; }
/* a narrow card: the slip button goes under the pick, full width */
@media (max-width: 559px) {
  .gp-pick { grid-template-columns: minmax(0, 1fr); }
  .gp-add { justify-items: stretch; }
  .gp-add .btn { width: 100%; }
  .gp-noslip { text-align: left; max-width: none; }
}
/* the inspector's column is narrow at any window width: the same layout as a phone, for the picks and the scan rows */
.inspector .gp-pick { grid-template-columns: minmax(0, 1fr); }
.inspector .gp-add { justify-items: stretch; }
.inspector .gp-add .btn { width: 100%; }
.inspector .gp-noslip { text-align: left; max-width: none; }
.inspector .gp .ps-item { grid-template-columns: 28px minmax(0, 1fr); }
.inspector .gp .ps-actions { grid-column: 2; margin-top: 2px; }
.inspector .gp .ps-actions .btn { flex: 1 1 140px; }
`;

export function ensureStyles() {
  ensureScanStyles(); // the props rows are the scan's own
  if (typeof document === "undefined" || document.getElementById("game-picks-styles")) return;
  const style = document.createElement("style");
  style.id = "game-picks-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
