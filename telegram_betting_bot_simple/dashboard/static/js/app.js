// SportsBet AI: the app shell (router, header, tab bar, bet slip, sign-in and onboarding).
//
// No build step: plain ES modules served from /static. /static/app.js imports this file and calls boot().
//
// ── Screens ("views") ─────────────────────────────────────────────────────────────────────────────
// Each screen is a module in js/views/ that exports:
//   render(root, params)   required. Fill ``root`` (an empty <div> inside <main>); may be async.
//                          params = {...route parts, query: URLSearchParams, signal: AbortSignal, path}
//                          ``signal`` aborts when the person leaves the screen: pass it to api() and
//                          check ``root.isConnected`` after every await before touching the page.
//   mainButton(params)     optional. {text, onClick, disabled?} for Telegram's bottom main button (only
//                          shown inside Telegram; in a browser the screen draws its own button). Call
//                          refreshMainButton() when its text or state changes.
//   onLeave()              optional. Clean up timers and listeners outside ``root``.
//   title                  optional string (or set it with setTitle()) for the header on sub-pages.
// Routes (hash based, so a link is just href="#/..."):
//   #/today  #/game/<league>/<id>  #/player/<league>/<id>  #/build  #/build/rollover  #/ask
//   #/bets  #/help  #/help/<topic>  #/settings        (?query parts are passed in params.query)
// Deep links in use: #/ask?game=<league>:<id>  #/ask?player=<league>:<id>&name=..  #/build?analyse=1
//   #/game/<league>/<id>?section=analysts  #/bets?tab=calls|leaders|record|slips
//   #/build/rollover?start=10&days=5&min_odds=2.00&legs_per_day=1 (the planner takes these once, then tidies them)
// One-shot requests: "analyse the slip" costs AI credits, so it must run once per tap and never again
// from Back, Forward or a reload. The shell keeps it out of the address: analyseSlip() remembers the
// request in memory and opens plain #/build, and a #/build?analyse=1 link is tidied to #/build before the
// screen draws. Either way Build sees params.query.get("analyse") === "1" on that one render only. Don't
// put the flag back into the address (navigate("#/build?analyse=1")); call analyseSlip().
//
// ── Shared helpers screens use ────────────────────────────────────────────────────────────────────
// From this file:   navigate(hash), goBack(), replaceAddress(hash), setTitle(text), refreshMainButton(), rerender(),
//                   analyseSlip(), openSlipSheet(), openOnboarding(), openSportsSheet(), setQuota(quota), signOut(),
//                   applyTheme(choice), inTelegram, and re-exported from store.js: slip, priceSlip, priced,
//                   onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports, loadSports, leagueTitle,
//                   sportEmoji, legKey, legFromOffer, session
// From ../ui.js:    esc, pct, chanceText, money, signedMoney, priceText, pointText, shortName, timeLabel,
//                   dayHeading, ago, chip, linkChip, segmented, badge, injuryBadge, section, notice,
//                   skeleton, emptyState, errorState, toast, sheet, confirmSheet, explain, GLOSSARY,
//                   openExplainer, oddsButton, oddsGrid, gameCard, splitBar, winChances, analystChips, barChart
// From ../api.js:   api(path, {method, body, signal}), get/post/put/del, qs, ApiError, isQuiet, getPlayerId, prefs
// From ../icons.js: icon(name, {size, label}), logo()
// From ../legacy.js: the first dashboard's pieces (calls, analysts, AI bets tiers, leaderboard, AI record)
// From ../telegram.js: haptic, setVerticalSwipes, openLink, setSecondaryButton
//
// Rules the shell keeps for every screen: numbers come from the server (never worked out here); errors
// are shown in the server's own words; every leg is added by the person's own tap; the helpline footer
// is on every screen.

import { api, setInitData, onUnauthorized, passwordLogin, telegramLogin, logout, isQuiet, prefs } from "./api.js";
import * as tg from "./telegram.js";
import {
  slip, priceSlip, priced, onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports, loadSports,
  leagueTitle, sportEmoji, legKey, legFromOffer, session,
} from "./store.js";
import {
  esc, pct, chanceText, money, signedMoney, priceText, toast, sheet, explain, skeleton, errorState, installUi,
  setExplainContext, notice, segmented, timeLabel, openExplainer, holdPage,
} from "./ui.js";
import { icon, logo } from "./icons.js";

export {
  slip, priceSlip, priced, onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports, loadSports,
  leagueTitle, sportEmoji, legKey, legFromOffer, session,
};
export const inTelegram = tg.inTelegram;

/** "system" | "dark" | "light" (kept in this browser; inside Telegram, Telegram's theme wins). */
export function applyTheme(choice = prefs.get("theme", "system")) {
  if (tg.inTelegram) return;
  if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
}

// --- routes --------------------------------------------------------------------------------------------

const ROUTES = [
  { name: "today", pattern: /^today$/, tab: "today", load: () => import("./views/today.js") },
  { name: "game", pattern: /^game\/([a-z0-9_]{2,64})\/([A-Za-z0-9_-]{1,64})$/, keys: ["league", "id"], tab: "today", sub: "#/today", title: "Game", load: () => import("./views/game.js") },
  { name: "player", pattern: /^player\/([a-z0-9_]{2,64})\/([0-9]{1,12})$/, keys: ["league", "id"], tab: "today", sub: "#/today", title: "Player", load: () => import("./views/player.js") },
  { name: "build", pattern: /^build(?:\/(rollover))?$/, keys: ["mode"], tab: "build", title: "Rollover planner", load: () => import("./views/build.js") },
  { name: "ask", pattern: /^ask$/, tab: "ask", load: () => import("./views/ask.js") },
  { name: "bets", pattern: /^bets$/, tab: "bets", load: () => import("./views/bets.js") },
  { name: "help", pattern: /^help(?:\/([a-z0-9_-]{1,40}))?$/, keys: ["topic"], tab: "help", title: "Help", load: () => import("./views/help.js") },
  { name: "settings", pattern: /^settings$/, tab: null, sub: "#/today", title: "Settings", load: () => import("./views/settings.js") },
];

// The first dashboard's and the bot's links (#games, #ask, ...) still land somewhere sensible
const OLD_HASHES = {
  games: "today", today: "today", bets: "build", build: "build", calls: "bets?tab=calls", leaders: "bets?tab=leaders",
  record: "bets?tab=record", ask: "ask", help: "help", settings: "settings", "my-bets": "bets",
};

const TABS = [
  { key: "today", label: "Today", href: "#/today", iconName: "today" },
  { key: "build", label: "Build", href: "#/build", iconName: "build" },
  { key: "ask", label: "Ask AI", href: "#/ask", iconName: "ask" },
  { key: "bets", label: "My bets", href: "#/bets", iconName: "bets" },
  { key: "help", label: "Help", href: "#/help", iconName: "help" },
];

/** {path, query} from the hash, without the launch parameters Telegram adds (#tgWebAppData=...). */
function parseHash() {
  let text = location.hash.replace(/^#/, "");
  const q = text.indexOf("?");
  let path = q >= 0 ? text.slice(0, q) : text;
  let query = q >= 0 ? text.slice(q + 1) : "";
  if (path.includes("=")) { // "#tgWebAppData=...&tgWebAppVersion=..."
    query = query ? `${path}&${query}` : path;
    path = "";
  }
  const params = new URLSearchParams(query);
  let cleaned = false;
  for (const key of [...params.keys()]) {
    if (key.startsWith("tgWebApp")) {
      params.delete(key);
      cleaned = true;
    }
  }
  path = path.replace(/^\/+|\/+$/g, "");
  try {
    path = decodeURIComponent(path);
  } catch {
    /* keep it as it came */
  }
  if (!path) cleaned = true; // "/" or "#": show #/today in the address bar
  if (!path.includes("/") && OLD_HASHES[path] && !location.hash.startsWith("#/")) {
    const [newPath, newQuery = ""] = OLD_HASHES[path].split("?");
    for (const [k, v] of new URLSearchParams(newQuery)) params.set(k, v);
    path = newPath;
    cleaned = true;
  }
  if (cleaned) {
    const rest = params.toString();
    replaceAddress(`#/${path || "today"}${rest ? `?${rest}` : ""}`);
  }
  return { path: path || "today", query: params };
}

/**
 * Change the address without a new history entry: the in-app history and the screen's scroll key follow, so
 * Back to this screen later restores where it was scrolled. Screens use this rather than history.replaceState.
 */
export function replaceAddress(hash) {
  if (visited.length && visited[visited.length - 1] === location.hash) visited[visited.length - 1] = hash;
  if (active && active.hash === location.hash) active.hash = hash;
  history.replaceState(history.state, "", hash);
}

function matchRoute(path) {
  for (const route of ROUTES) {
    const m = route.pattern.exec(path);
    if (m) {
      const params = {};
      (route.keys || []).forEach((key, i) => { params[key] = m[i + 1] ?? null; });
      return { route, params };
    }
  }
  return null;
}

// --- navigation ------------------------------------------------------------------------------------------

const visited = []; // in-app history of hashes, to know whether Back stays in the app
const scrolls = new Map(); // scroll position per hash, restored when going back
let goingBack = false;

/** Go to a route ("#/game/basketball_nba/abc"). Closes an open sheet first. */
export async function navigate(hash) {
  if (sheet.isOpen) await sheet.close();
  const target = hash.startsWith("#") ? hash : `#${hash}`;
  if (location.hash === target) rerender();
  else location.hash = target;
}

/** Back within the app, or up to the screen's parent when the app was opened on this screen. */
export function goBack() {
  if (sheet.isOpen) {
    sheet.back(); // an explanation inside a sheet goes back to that sheet; otherwise the sheet closes
    return;
  }
  if (visited.length > 1) {
    history.back();
    return;
  }
  const parent = active?.route?.sub || (active?.route?.name === "help" && active.params.topic ? "#/help" : null)
    || (active?.route?.name === "build" && active.params.mode ? "#/build" : null) || "#/today";
  location.replace(parent);
}

// Set by analyseSlip() and used up by the next render (see "One-shot requests" in the header)
let analysisRequested = false;

/**
 * Open Build and analyse the slip there, once (the slip bar's and slip sheet's "Analyse", Telegram's main
 * button). The request lives in memory, not in the address, so Back or a reload never repeats it.
 */
export async function analyseSlip() {
  if (sheet.isOpen) await sheet.close();
  analysisRequested = true;
  navigate("#/build");
}

function trackVisit() {
  const hash = location.hash;
  if (visited.length > 1 && visited[visited.length - 2] === hash) {
    visited.pop();
    goingBack = true;
  } else if (visited[visited.length - 1] !== hash) {
    visited.push(hash);
    goingBack = false;
  }
}

// --- rendering a screen ----------------------------------------------------------------------------------

let active = null; // {route, params, module, root, controller, hash}
let renderSeq = 0;
const $ = (sel) => document.querySelector(sel);

function isSubPage(route, params) {
  return Boolean(route.sub) || (route.name === "help" && params.topic) || (route.name === "build" && params.mode);
}

/** Draw the screen for the current hash. */
export async function rerender() {
  const seq = ++renderSeq;
  const back = goingBack;
  goingBack = false;
  const wantsAnalysis = analysisRequested; // one render only, whichever screen it is (see the header)
  analysisRequested = false;
  const { path, query } = parseHash();
  const found = matchRoute(path);
  if (!found) {
    location.replace("#/today");
    return;
  }
  const { route, params } = found;
  if (query.has("analyse")) {
    // A link with the flag: run it this once, but leave it out of the address so Back or a reload can't
    query.delete("analyse");
    const rest = query.toString();
    replaceAddress(`#/${path}${rest ? `?${rest}` : ""}`);
    if (route.name === "build" && !params.mode) query.set("analyse", "1");
  } else if (wantsAnalysis && route.name === "build" && !params.mode) {
    query.set("analyse", "1");
  }
  if (active) {
    scrolls.set(active.hash, window.scrollY);
    active.controller.abort();
    try {
      active.module?.onLeave?.();
    } catch (error) {
      console.warn(error);
    }
  }
  tg.setVerticalSwipes(true);
  const controller = new AbortController();
  const root = document.createElement("div");
  root.className = `view view-${route.name}`;
  root.innerHTML = skeleton("cards", 2);
  $("#view").replaceChildren(root);
  active = { route, params: { ...params, query, signal: controller.signal, path }, module: null, root, controller, hash: location.hash };
  updateChrome();
  let module;
  try {
    module = await route.load();
  } catch (error) {
    console.error(error);
    if (seq === renderSeq) root.innerHTML = errorState(new Error("This screen couldn't load. Reload the page to try again."));
    return;
  }
  if (seq !== renderSeq) return;
  active.module = module;
  if (module.title) setTitle(module.title);
  root.innerHTML = "";
  const restore = back ? scrolls.get(location.hash) : 0;
  try {
    const pending = module.render(root, active.params);
    if (!back) window.scrollTo(0, 0);
    await pending;
  } catch (error) {
    if (!isQuiet(error) && seq === renderSeq) {
      console.error(error);
      root.innerHTML = errorState(error);
    }
  }
  if (seq !== renderSeq) return;
  if (back && restore) window.scrollTo(0, restore);
  refreshMainButton();
}

/** Header, tab bar and Telegram's Back button for the current screen. */
function updateChrome() {
  const { route, params } = active;
  const sub = isSubPage(route, params);
  for (const tab of document.querySelectorAll(".tabbar a")) {
    if (tab.dataset.tab === route.tab) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  document.body.dataset.route = route.name;
  $("#back").hidden = !sub || tg.inTelegram;
  setTitle(sub ? route.title || "" : null); // a screen may set a better one once its data is in
  if (route.name === "settings") $(".avatar").setAttribute("aria-current", "page");
  else $(".avatar").removeAttribute("aria-current");
  updateBackButton();
}

function updateBackButton() {
  const sub = active && isSubPage(active.route, active.params);
  tg.setBackButton(sheet.isOpen || sub ? goBack : null);
}

/** The header title on sub-pages (null shows the app's name). */
export function setTitle(text) {
  const title = $("#title");
  if (text === null || text === undefined) {
    title.innerHTML = `${logo(26)}<span class="brand">SportsBet <b>AI</b></span>`;
    document.title = "SportsBet AI";
  } else {
    title.innerHTML = `<span class="page-name">${esc(text)}</span>`;
    document.title = text ? `${text} · SportsBet AI` : "SportsBet AI";
  }
}

/**
 * Ask the screen for its Telegram main button again (after its state changed). A config with
 * ``analyse: true`` is the slip's "Analyse": while Telegram shows that, the slip bar's own Analyse button
 * steps aside (body.tg-analyse), so the screen never offers the same action twice. On a break there is no
 * default "Analyse" (the server refuses it then).
 */
export function refreshMainButton() {
  if (!tg.inTelegram) return;
  let config = null;
  if (!sheet.isOpen) {
    try {
      config = active?.module?.mainButton?.(active.params) || null;
    } catch (error) {
      console.warn(error);
    }
    if (!config && slip.count && active?.route?.name !== "build" && !onBreak()) {
      config = { text: `Analyse slip with AI (${slip.count})`, onClick: analyseSlip, analyse: true };
    }
  }
  const shown = tg.setMainButton(config);
  document.body.classList.toggle("tg-analyse", Boolean(shown && config?.analyse));
}

// --- header: quota pill and avatar -------------------------------------------------------------------------

/** Show the odds feed's requests left ({remaining, used}) in the header. */
export function setQuota(quota) {
  if (!quota || quota.remaining == null) return;
  session.quota = quota;
  const pill = $("#quota");
  pill.hidden = false;
  const low = quota.remaining < 50;
  pill.classList.toggle("low", low);
  pill.innerHTML = `${icon(low ? "alert" : "refresh", { size: 14 })}<span>${Number(quota.remaining).toLocaleString()}</span><span class="pill-word"> odds left</span>`;
  pill.setAttribute("aria-label", `${quota.remaining} odds requests left this month${low ? ", running low" : ""}. What does this mean?`);
}

function updateAvatar() {
  const name = me().name || tg.firstName() || prefs.name();
  $(".avatar").innerHTML = name ? `<span>${esc(name.trim().charAt(0).toUpperCase())}</span>` : icon("user", { size: 20 });
}

function updateBreakBanner() {
  const banner = $("#banner");
  const profile = me();
  if (profile.on_break && profile.cooloff_until) {
    banner.innerHTML = notice(`You're on a break until ${new Date(profile.cooloff_until).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}. Picks, building and plans are switched off until then.`, "info");
    banner.hidden = false;
  } else {
    banner.hidden = true;
    banner.innerHTML = "";
  }
}

// --- the slip bar and slip sheet -------------------------------------------------------------------------

let priceTimer = null;

function schedulePricing() {
  clearTimeout(priceTimer);
  priceTimer = setTimeout(() => priceSlip(), 250);
}

function legsWord(n) {
  return `${n} ${n === 1 ? "pick" : "legs"}`;
}

/** "3 legs · 4.60 · $10 → $46" from the server's pricing (or a placeholder while it prices). */
function slipSummary() {
  const n = slip.count;
  const { busy, result, error } = priced();
  if (onBreak()) return { main: legsWord(n), detail: "on a break" }; // the server prices nothing then
  if (!result) {
    if (busy) return { main: legsWord(n), detail: "pricing…" };
    if (error) return { main: legsWord(n), detail: error.status === 404 ? "tap to review" : "check the slip", warn: true };
    return { main: legsWord(n), detail: "" };
  }
  const total = result.total_stake;
  if (!result.valid || !result.bets) return { main: legsWord(n), detail: "needs a fix", warn: true };
  if (result.type === "multi" || (result.type === "single" && result.legs.length === 1)) {
    return { main: `${legsWord(n)} · ${priceText(result.odds)}`, detail: `${money(total)} → ${money(result.max_payout)}` };
  }
  const kind = result.type === "single" ? `${result.bets} singles` : result.type === "round_robin" ? `Round robin · ${result.bets} bets` : `${result.system || "System"} · ${result.bets} bets`;
  return { main: kind, detail: `${money(total)} → up to ${money(result.max_payout)}` };
}

function renderSlipBar() {
  const bar = $("#slipbar");
  const n = slip.count;
  document.body.classList.toggle("has-slip", n > 0);
  bar.hidden = n === 0;
  // On a break the server refuses an analysis: Analyse is off, and the break banner says why
  const analyse = bar.querySelector(".slip-analyse");
  analyse.disabled = onBreak();
  if (analyse.disabled) analyse.setAttribute("aria-describedby", "banner");
  else analyse.removeAttribute("aria-describedby");
  if (!n) return;
  const s = slipSummary();
  bar.querySelector(".slip-open").innerHTML = `<span class="slip-count" aria-hidden="true">${n}</span>
    <span class="slip-text"><b>${esc(s.main)}</b>${s.detail ? `<span class="${s.warn ? "warn" : ""}">${s.warn ? icon("alert", { size: 14 }) : ""}${esc(s.detail)}</span>` : ""}</span>
    ${icon("up", { size: 18 })}`;
  bar.querySelector(".slip-open").setAttribute("aria-label", `Bet slip: ${s.main}${s.detail ? `, ${s.detail}` : ""}. Open the slip`);
}

function legLine(leg, view, removed) {
  const odds = view?.odds || { decimal: leg.price, american: leg.american };
  const ai = view ? view.ai_prob : leg.ai;
  const market = view ? view.market_prob : leg.fair;
  return `<li class="slip-leg${removed ? " gone" : ""}">
    <div class="grow">
      <div class="row-between"><b>${esc(view?.label || leg.label)}</b><b class="num">${esc(priceText(odds))}</b></div>
      <div class="muted small">${esc(leg.game || view?.game || "")}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""}</div>
      <div class="small">${ai != null ? `AI ${pct(ai)}` : ""}${ai != null && market != null ? " · " : ""}${market != null ? `Market ${pct(market)}` : ""}
        ${removed ? ` ${icon("alert", { size: 14 })}<span class="warn">No longer offered</span>` : ""}${view?.started ? ` <span class="warn">Started</span>` : ""}</div>
    </div>
    <button type="button" class="icon-btn" data-remove-leg="${esc(leg.key)}" aria-label="Remove ${esc(leg.label)}">${icon("trash", { size: 18 })}</button>
  </li>`;
}

// The slip sheet is drawn once, then refreshed part by part (data-part="..."), so a tap on a stake, the
// bet type or a leg's bin doesn't scroll the sheet back to the top, and a stake being typed isn't replaced.

const STAKES = ["5", "10", "25"];
const typeTerm = () => (slip.type === "single" ? "single" : slip.type === "multi" ? "multi" : slip.type === "round_robin" ? "round_robin" : "system");
const stakeLabel = () => (slip.type === "multi" && slip.count > 1 ? "Stake" : "Stake per bet");
const customStake = () => (STAKES.includes(slip.stake) ? "" : slip.stake);

/** The server's view of each leg (by leg key) and the legs it no longer offers. */
function pricedLegs(result) {
  const views = new Map();
  (result?.refs || []).forEach((ref, i) => views.set(legKey(ref), result.legs[i]));
  return { views, removed: new Set((result?.removed || []).map(legKey)) };
}

function slipLegsHtml(result) {
  const { views, removed } = pricedLegs(result);
  return slip.legs.map((leg) => legLine(leg, views.get(leg.key), removed.has(leg.key))).join("");
}

function slipTypeHtml() {
  const typeOptions = [{ value: "single", label: "Singles" }, { value: "multi", label: "Multi" }];
  if (slip.type === "round_robin" || slip.type === "system") typeOptions.push({ value: slip.type, label: slip.type === "round_robin" ? "Round robin" : "System" });
  return `<span class="label">Bet type ${explain(typeTerm())}</span>
      ${segmented("slip-type", typeOptions, slip.type, { label: "Bet type" })}
      ${slip.type === "round_robin" || slip.type === "system" ? '<p class="hint">Round robins and systems are set up in Build.</p>' : ""}`;
}

/** The server's numbers for the slip: chances, payout, what the bookmaker keeps, expected result, warnings. */
function slipNumbersHtml() {
  const { busy, result, error } = priced();
  // On a break the server prices nothing: no numbers from before it, and the line Analyse points to
  if (onBreak()) return `<div id="slip-break-note">${notice("You're on a break: pricing, analysing and saving a slip are off until it ends.", "info")}</div>`;
  let summary = "";
  if (result && result.bets) {
    const c = result.chance || {};
    const e = result.expected || {};
    const ml = result.most_likely;
    const chanceWord = c.headline === "profit" ? "Chance of a profit" : "Chance it wins";
    summary = `<dl class="summary">
      ${result.odds ? `<div><dt>Odds</dt><dd>${esc(priceText(result.odds))}</dd></div>` : ""}
      <div><dt>Costs</dt><dd>${money(result.total_stake, { cents: true })}${result.bets > 1 ? ` <span class="muted">(${result.bets} bets)</span>` : ""}</dd></div>
      <div><dt>Pays ${explain("payout")}</dt><dd>${result.type === "multi" || result.bets === 1 ? "" : "up to "}${money(result.max_payout, { cents: true })}</dd></div>
      <div><dt>${chanceWord} ${explain("one_in")}</dt><dd>${c.ai != null ? `AI ${esc(chanceText(c.ai, c.ai_one_in))}` : ""}${c.ai != null && c.market != null ? "<br>" : ""}${c.market != null ? `Market ${esc(chanceText(c.market, c.market_one_in))}` : ""}${c.ai == null && c.market == null ? "–" : ""}</dd></div>
      <div><dt>Bookmaker keeps ${explain("book_cut")}</dt><dd>${result.book_cut != null ? `about ${pct(result.book_cut)}` : "–"}</dd></div>
      <div><dt>Expected result ${explain("expected_result")}</dt><dd>${e.ai != null ? `${esc(signedMoney(e.ai))} <span class="muted">AI</span>` : ""}${e.ai != null && e.market != null ? "<br>" : ""}${e.market != null ? `${esc(signedMoney(e.market))} <span class="muted">market</span>` : ""}${e.ai == null && e.market == null ? "–" : ""}</dd></div>
      ${ml ? `<div><dt>Most likely ${explain("most_likely")}</dt><dd>${esc(ml.result === "lose all" ? `lose ${money(result.total_stake)}` : ml.result)} (${pct(ml.probability)})</dd></div>` : ""}
    </dl>`;
  }
  const problems = [...(result?.reasons || [])];
  const warnings = [...(result?.warnings || [])];
  return `${busy && !result ? skeleton("lines", 3) : ""}
    ${error && !result ? notice(error.message, error.status === 404 ? "info" : "warn") : ""}
    ${problems.map((r) => notice(r, "danger")).join("")}
    ${summary}
    ${warnings.length ? `<ul class="warnings">${warnings.map((w) => `<li>${icon("alert", { size: 14 })}<span>${esc(w)}</span></li>`).join("")}</ul>` : ""}`;
}

function slipSheetBody() {
  if (!slip.count) {
    return `<div class="empty"><div class="empty-icon">${icon("bets", { size: 28 })}</div><h3>Your slip is empty</h3>
      <p>Tap any price on Today to add a pick, or let the AI build one.</p>
      <div class="empty-actions"><a class="btn btn-primary" href="#/today">Browse games</a><a class="btn btn-secondary" href="#/build">Let AI build one</a></div></div>`;
  }
  return `
    <ul class="slip-legs" data-part="legs">${slipLegsHtml(priced().result)}</ul>
    <div class="field" data-part="type">${slipTypeHtml()}</div>
    <div class="field">
      <label class="label" for="slip-stake" data-part="stake-label">${stakeLabel()}</label>
      <div class="stake-row">
        ${STAKES.map((v) => `<button type="button" class="chip${slip.stake === v ? " on" : ""}" aria-pressed="${slip.stake === v}" data-stake="${v}">$${v}</button>`).join("")}
        <span class="money-input"><span aria-hidden="true">$</span><input id="slip-stake" class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(customStake())}" placeholder="Other" aria-label="Other stake"></span>
      </div>
    </div>
    <div data-part="numbers">${slipNumbersHtml()}</div>
    <p class="muted small">Nothing is bet here: the app prices picks so you can see the chances. Prices are rechecked on the server.</p>`;
}

function slipSheetFooter() {
  if (!slip.count) return "";
  return `<button type="button" class="btn btn-secondary" data-slip="clear">${icon("trash", { size: 18 })}Clear</button>
    <button type="button" class="btn btn-secondary" data-slip="build">Open in Build</button>
    <button type="button" class="btn btn-primary grow" data-slip="analyse"${analyseOff()}>${icon("ask", { size: 18 })}Analyse with AI</button>`;
}

/** The slip sheet's Analyse is off on a break, described by the line in place of the numbers. */
function analyseOff() {
  return onBreak() ? ' disabled aria-describedby="slip-break-note"' : "";
}

let slipSheet = null;

/** The full slip in a bottom sheet: legs, bet type, stake chips and the server's numbers. */
export function openSlipSheet() {
  const s = sheet.open({ title: `Your slip${slip.count ? ` (${slip.count})` : ""}`, body: slipSheetBody(), footer: slipSheetFooter(), size: "tall", onClose: () => { slipSheet = null; } });
  slipSheet = s;
  priceSlip();
  s.panel.addEventListener("click", async (event) => {
    const remove = event.target.closest("[data-remove-leg]");
    if (remove) {
      slip.remove(remove.dataset.removeLeg);
      tg.haptic.select();
      if (!s.panel.contains(document.activeElement)) s.panel.focus({ preventScroll: true }); // its button is gone
      return;
    }
    const stake = event.target.closest("[data-stake]");
    if (stake) {
      slip.set({ stake: stake.dataset.stake });
      return;
    }
    const type = event.target.closest('[data-seg="slip-type"]');
    if (type) {
      slip.set({ type: type.dataset.value });
      return;
    }
    const action = event.target.closest("[data-slip]")?.dataset.slip;
    if (action === "clear") {
      slip.clear();
      toast("Slip cleared");
      s.close();
    } else if (action === "build") {
      navigate("#/build");
    } else if (action === "analyse") {
      analyseSlip();
    }
  });
  s.panel.addEventListener("change", (event) => {
    if (event.target.id !== "slip-stake") return;
    const value = event.target.value.trim().replace(/^\$/, "");
    if (!value) return;
    if (!/^\d{1,6}(\.\d{1,2})?$/.test(value) || Number(value) < 0.01 || Number(value) > 100000) {
      toast("Enter a stake between $0.01 and $100,000", { kind: "warn" });
      return;
    }
    slip.set({ stake: value });
  });
}

/** Bring the open slip sheet up to date in place: same scroll position, the stake being typed left alone. */
function refreshSlipSheet() {
  if (!slipSheet?.open) return;
  const body = slipSheet.body;
  const part = (name) => body.querySelector(`[data-part="${name}"]`);
  sheet.update({ title: `Your slip${slip.count ? ` (${slip.count})` : ""}` });
  if (!slip.count || !part("legs")) { // emptied, or the empty sheet got its first pick: draw it all
    sheet.update({ body: slipSheetBody(), footer: slipSheetFooter() });
    return;
  }
  part("legs").innerHTML = slipLegsHtml(priced().result);
  part("type").innerHTML = slipTypeHtml();
  part("stake-label").textContent = stakeLabel();
  for (const chip of body.querySelectorAll("[data-stake]")) {
    chip.classList.toggle("on", slip.stake === chip.dataset.stake);
    chip.setAttribute("aria-pressed", String(slip.stake === chip.dataset.stake));
  }
  const input = body.querySelector("#slip-stake");
  if (input && document.activeElement !== input) input.value = customStake();
  part("numbers").innerHTML = slipNumbersHtml();
  const analyse = slipSheet.foot?.querySelector('[data-slip="analyse"]');
  if (analyse && analyse.disabled !== onBreak()) { // a break started or ended with the sheet open
    analyse.disabled = onBreak();
    if (analyse.disabled) analyse.setAttribute("aria-describedby", "slip-break-note");
    else analyse.removeAttribute("aria-describedby");
  }
}

// --- sign-in ---------------------------------------------------------------------------------------------

let gateOpen = false;

function showGate(html) {
  const gate = $("#gate");
  gate.innerHTML = `<div class="gate-card" role="dialog" aria-modal="true" aria-labelledby="gate-title" tabindex="-1">
    <div class="gate-brand">${logo(36)}<span class="brand">SportsBet <b>AI</b></span></div>${html}</div>`;
  gate.hidden = false;
  document.body.classList.add("gated");
  holdPage("gate", true, ".sheet-layer"); // Tab stays in the gate: the page (and any sheet) behind it is inert
  gate.querySelector(".gate-card").focus({ preventScroll: true });
  gateOpen = true;
}

function hideGate() {
  $("#gate").hidden = true;
  $("#gate").innerHTML = "";
  document.body.classList.remove("gated");
  holdPage("gate", false);
  gateOpen = false;
}

/** The password form (browsers). Kept as form#sign-in with input[name=password]. */
function showSignIn(note = "") {
  if (gateOpen) return;
  if (sheet.isOpen) sheet.close();
  showGate(`<form id="sign-in" novalidate>
      <h1 id="gate-title">Sign in</h1>
      <p class="muted">Enter the dashboard password from your server setup.</p>
      ${note ? notice(note, "warn") : ""}
      <label class="sr-only" for="password">Dashboard password</label>
      <input class="input" id="password" type="password" name="password" placeholder="Dashboard password" autocomplete="current-password" required>
      <div class="form-error" role="alert"></div>
      <button class="btn btn-primary block" type="submit">Sign in</button>
    </form>
    <p class="muted small">Private app for the owner and friends. Gambling can be addictive · Help 24/7: 1-800-GAMBLER / 0808 8020 133</p>`);
  const form = $("#sign-in");
  form.elements.password.focus();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button");
    const error = form.querySelector(".form-error");
    if (!form.elements.password.value) {
      error.textContent = "Enter the password.";
      return;
    }
    button.disabled = true;
    error.textContent = "";
    try {
      await passwordLogin(form.elements.password.value);
      hideGate();
      start();
    } catch (err) {
      error.textContent = err.message;
      form.elements.password.select();
    } finally {
      button.disabled = false;
    }
  });
}

/** Telegram said no (403): show its message, which includes the person's Telegram id for the owner. */
function showRefused(message) {
  showGate(`<h1 id="gate-title">This app is private</h1>
    ${notice(message, "warn")}
    <p class="muted">Send the id above to the owner so they can add you, then open the app again.</p>
    <button type="button" class="btn btn-secondary block" data-close-app>Close</button>`);
  $("#gate [data-close-app]").addEventListener("click", () => tg.closeApp());
}

/**
 * Inside Telegram a 401 means Telegram's sign-in data has run out: it lasts 24 hours, only Telegram can
 * renew it (by opening the app again), and on Telegram Web the session cookie isn't sent from its frame.
 * The dashboard password is no use to a friend, so this says what to do instead and offers to close the app.
 */
function showTelegramExpired(message = "") {
  if (gateOpen) return;
  if (sheet.isOpen) sheet.close();
  const detail = message && !/^sign in/i.test(message) ? message : "";
  const expired = !detail || /expired/i.test(detail); // else e.g. a wrong signature: the server's words say which
  showGate(`<h1 id="gate-title">Open the app again</h1>
    <p>${expired ? "Your Telegram sign-in has run out: it lasts 24 hours." : "Telegram's sign-in couldn't be checked."} Close the app and open it again from the bot to carry on where you left off${expired ? "" : "; if this keeps happening, tell the owner"}.</p>
    ${detail ? notice(detail, "info") : ""}
    <button type="button" class="btn btn-primary block" data-close-app>Close the app</button>
    <p class="muted small">Gambling can be addictive · Help 24/7: 1-800-GAMBLER / 0808 8020 133</p>`);
  $("#gate [data-close-app]").addEventListener("click", () => tg.closeApp());
}

let telegramRenewed = false; // one quiet sign-in again per visit, then the notice
let renewing = false;

/** A 401 inside Telegram: sign in again once with Telegram's data (it works while that is under a day old). */
async function telegramUnauthorized() {
  if (gateOpen || renewing) return;
  if (telegramRenewed) {
    showTelegramExpired();
    return;
  }
  telegramRenewed = true;
  renewing = true;
  try {
    await telegramLogin(tg.initData());
    renewing = false;
    if (started) rerender(); // signed in again: the screen that was refused draws again
    else start(); // refused while starting up: start again
  } catch (error) {
    renewing = false;
    if (error.status === 403) showRefused(error.message);
    else if (error.status === 503) showSignIn(`Telegram sign-in didn't work: ${error.message}`);
    else showTelegramExpired(error.message);
  }
}

/** Sign out (password sessions) and show the sign-in form again. */
export async function signOut() {
  await logout();
  location.hash = "#/today";
  location.reload();
}

// --- onboarding (first launch) ----------------------------------------------------------------------------

const POPULAR = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "basketball_ncaab",
  "americanfootball_ncaaf", "basketball_wnba", "soccer_epl", "soccer_uefa_champs_league", "soccer_usa_mls"];

/** Age, favourite sports, optional weekly budget, odds format and the AI disclosure, in one sheet. */
export async function openOnboarding() {
  let sports = [];
  try {
    sports = await loadSports();
  } catch (error) {
    if (error.signIn) return;
  }
  const chosen = new Set(favouriteSports());
  let format = oddsFormat();
  let showAll = false;
  // The big leagues first; the rest behind "More sports" so the form stays short
  const popular = sports.filter((s) => POPULAR.includes(s.key) || chosen.has(s.key));
  const sportChips = () => (showAll || popular.length < 3 ? sports : popular).map((s) => `<button type="button" class="chip${chosen.has(s.key) ? " on" : ""}" aria-pressed="${chosen.has(s.key)}" data-fav="${esc(s.key)}">${esc(sportEmoji(s.key))} ${esc(s.title)}${s.ai ? ' <span class="tag">AI</span>' : ""}</button>`).join("")
    + (!showAll && popular.length >= 3 && sports.length > popular.length ? `<button type="button" class="chip add" data-more-sports>${icon("plus", { size: 14 })}More sports (${sports.length - popular.length})</button>` : "");
  const s = sheet.open({
    title: "Welcome to SportsBet AI",
    size: "tall",
    body: `<p class="lead">See each game's real chances, price any slip honestly, and ask an AI analyst about it. Nothing is bet here.</p>
      <div class="field">
        <label class="check"><input type="checkbox" data-age> <span>I'm 18 or older (21+ in the US) and betting is legal where I live.</span></label>
      </div>
      <div class="field"><span class="label">Your sports <span class="muted">(shown first on Today)</span></span>
        <div class="chip-row wrap" data-sports>${sports.length ? sportChips() : '<span class="muted small">Sports load once you\'re signed in.</span>'}</div></div>
      <div class="field"><label class="label" for="onb-budget">Weekly budget <span class="muted">(optional)</span> ${explain("bankroll")}</label>
        <span class="money-input"><span aria-hidden="true">$</span><input id="onb-budget" class="input" type="text" inputmode="decimal" placeholder="e.g. 50" autocomplete="off"></span>
        <p class="hint">Stakes you save count against it. Lowering it applies at once; raising it waits 24 hours.</p></div>
      <div class="field"><span class="label">Odds format</span>
        ${segmented("onb-format", [{ value: "decimal", label: "Decimal 2.50" }, { value: "american", label: "American +150" }], format, { label: "Odds format" })}</div>
      <div class="disclosure">${icon("bot", { size: 20 })}<p><b>You're using an AI analyst.</b> Its numbers are estimates computed in code from odds and stats; Claude only explains them. It can be wrong, and no model reliably beats the betting market. Only bet what you can afford to lose.</p></div>
      <div class="form-error" role="alert" data-onb-error></div>`,
    footer: `<button type="button" class="btn btn-secondary" data-onb="later">Not now</button>
      <button type="button" class="btn btn-primary grow" data-onb="start" disabled>Get started</button>`,
  });
  const startButton = s.panel.querySelector('[data-onb="start"]');
  s.panel.addEventListener("change", (event) => {
    if (event.target.matches("[data-age]")) startButton.disabled = !event.target.checked;
  });
  s.panel.addEventListener("click", async (event) => {
    const fav = event.target.closest("[data-fav]");
    if (fav || event.target.closest("[data-more-sports]")) {
      if (!fav) showAll = true;
      else if (chosen.has(fav.dataset.fav)) chosen.delete(fav.dataset.fav);
      else if (chosen.size < 8) chosen.add(fav.dataset.fav);
      else toast("Pick up to 8 sports", { kind: "warn" });
      s.panel.querySelector("[data-sports]").innerHTML = sportChips();
      return;
    }
    const seg = event.target.closest('[data-seg="onb-format"]');
    if (seg) {
      format = seg.dataset.value;
      for (const b of s.panel.querySelectorAll('[data-seg="onb-format"]')) b.setAttribute("aria-pressed", String(b.dataset.value === format));
      return;
    }
    const action = event.target.closest("[data-onb]")?.dataset.onb;
    if (action === "later") {
      s.close();
    } else if (action === "start") {
      const error = s.panel.querySelector("[data-onb-error]");
      const budget = s.panel.querySelector("#onb-budget").value.trim().replace(/^\$/, "");
      if (budget && !/^\d{1,7}(\.\d{1,2})?$/.test(budget)) {
        error.textContent = "The budget should be an amount like 50 or 25.50.";
        return;
      }
      const changes = { confirm_age: true, odds_format: format, favourite_sports: [...chosen] };
      if (budget) changes.weekly_budget = budget;
      startButton.disabled = true;
      try {
        await updateMe(changes);
        await s.close();
        toast("You're all set", { kind: "good" });
        rerender();
      } catch (err) {
        if (!isQuiet(err)) error.textContent = err.message;
        startButton.disabled = false;
      }
    }
  });
}

/** Pick favourite sports (Today's "+" chip and Settings). */
export async function openSportsSheet({ onDone } = {}) {
  let sports;
  try {
    sports = await loadSports();
  } catch (error) {
    if (!isQuiet(error)) toast(error.message, { kind: "error" });
    return;
  }
  const chosen = new Set(favouriteSports()); // what Today and Settings show, NBA until some are saved
  const groups = [];
  for (const s of sports) {
    let group = groups.find((g) => g.label === s.label);
    if (!group) groups.push(group = { label: s.label, leagues: [] });
    group.leagues.push(s);
  }
  const body = () => groups.map((g) => `<div class="field"><span class="label">${esc(g.label)}</span><div class="chip-row wrap">${g.leagues.map((s) =>
    `<button type="button" class="chip${chosen.has(s.key) ? " on" : ""}" aria-pressed="${chosen.has(s.key)}" data-fav="${esc(s.key)}">${chosen.has(s.key) ? icon("check", { size: 14 }) : icon("plus", { size: 14 })}${esc(s.title)}${s.ai ? ' <span class="tag">AI</span>' : ""}</button>`).join("")}</div></div>`).join("")
    + '<p class="hint">Only the sports you pick are fetched: each one uses some of the odds feed\'s monthly requests.</p>';
  const s = sheet.open({
    title: "Your sports", size: "tall", body: body(),
    footer: '<button type="button" class="btn btn-primary block" data-save-sports>Done</button>',
  });
  s.panel.addEventListener("click", async (event) => {
    const fav = event.target.closest("[data-fav]");
    if (fav) {
      if (chosen.has(fav.dataset.fav)) chosen.delete(fav.dataset.fav);
      else if (chosen.size < 8) chosen.add(fav.dataset.fav);
      else toast("Pick up to 8 sports", { kind: "warn" });
      s.body.innerHTML = body();
      return;
    }
    if (event.target.closest("[data-save-sports]")) {
      try {
        await updateMe({ favourite_sports: [...chosen] });
        await s.close();
        onDone?.();
      } catch (error) {
        if (!isQuiet(error)) toast(error.message, { kind: "error" });
      }
    }
  });
}

// --- reality check ----------------------------------------------------------------------------------------

let realityTimer = null;
let realityMinutes = null; // the interval the running timer was set for

function scheduleRealityCheck() {
  const minutes = me().reality_check_minutes || null;
  // The profile reloads often (Build, Settings, a saved slip): only a new interval restarts the countdown,
  // or every reload would push the next reality check back by a whole interval
  if (minutes === realityMinutes && (realityTimer || !minutes)) return;
  clearInterval(realityTimer);
  realityTimer = null;
  realityMinutes = minutes;
  if (!minutes) return;
  let started = Number(sessionStorage.getItem("sportsbet-session-start")) || Date.now();
  try {
    sessionStorage.setItem("sportsbet-session-start", String(started));
  } catch {
    started = Date.now();
  }
  realityTimer = setInterval(() => {
    const spent = Math.round((Date.now() - started) / 60000);
    if (sheet.isOpen || gateOpen) return; // never a sheet behind the sign-in gate, where Tab could reach it
    sheet.open({
      title: "Reality check",
      body: `<p class="lead">You've had the app open for ${spent} minutes.</p><p>Take a moment: is this still fun? You can set a break in Settings.</p>`,
      footer: `<a class="btn btn-secondary" href="#/settings">Take a break</a><button type="button" class="btn btn-primary grow" data-ack>Keep going</button>`,
    }).panel.addEventListener("click", (event) => { if (event.target.closest("[data-ack]")) sheet.close(); });
  }, minutes * 60000);
}

// --- start -----------------------------------------------------------------------------------------------

function drawShell() {
  $("#tabbar-inner").innerHTML = TABS.map((t) => `<a href="${t.href}" data-tab="${t.key}">${icon(t.iconName, { size: 22 })}<span>${t.label}</span></a>`).join("");
  setTitle(null);
  updateAvatar();
  $("#back").innerHTML = icon("back", { size: 22 });
  $("#back").addEventListener("click", goBack);
  $("#quota").addEventListener("click", () => openExplainer("quota"));
  $("#slipbar .slip-open").addEventListener("click", openSlipSheet);
  $("#slipbar .slip-analyse").addEventListener("click", analyseSlip);
  $("#slipbar .slip-analyse").innerHTML = `${icon("ask", { size: 18 })}<span>Analyse</span>`;
}

// --- the person's time zone ---------------------------------------------------------------------------------

let timezoneChecked = false;

/** This device's IANA time zone ("Europe/London"), or "" when the browser doesn't say. */
function deviceTimezone() {
  try {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return typeof zone === "string" && /^[A-Za-z0-9_+\-/]{1,64}$/.test(zone) ? zone : "";
  } catch {
    return "";
  }
}

/**
 * Once per load: when GET /api/me's "timezone" differs from this device's, PUT the device's, so the
 * server's days, weeks and times follow the person's clock. Quiet: a failure changes nothing on screen,
 * and a server whose profile has no "timezone" field isn't asked.
 */
async function syncTimezone() {
  const profile = session.me;
  if (timezoneChecked || !profile || !("timezone" in profile)) return;
  timezoneChecked = true;
  const zone = deviceTimezone();
  if (!zone || zone === profile.timezone) return;
  try {
    const updated = await api("/api/me", { method: "PUT", body: { timezone: zone }, auth: false });
    if (updated && typeof updated === "object" && "uid" in updated) setMe(updated);
  } catch {
    /* the server keeps the zone it had; nothing to show */
  }
}

let started = false;

async function start() {
  try {
    await loadMe();
  } catch (error) {
    if (error.signIn) return; // the sign-in form is up; it calls start() again
    toast(error.message, { kind: "error" });
  }
  // Before the first screen (whose days may follow the zone), but never holding it up for long
  await Promise.race([syncTimezone(), new Promise((resolve) => { setTimeout(resolve, 1500); })]);
  updateAvatar();
  updateBreakBanner();
  scheduleRealityCheck();
  loadSports().catch(() => null);
  if (slip.count) priceSlip();
  if (!started) {
    started = true;
    parseHash(); // tidy the address first (#/today for "/", no Telegram launch parameters) so Back history is clean
    window.addEventListener("hashchange", () => {
      trackVisit();
      rerender();
    });
    trackVisit();
  }
  await rerender();
  if (session.me && !session.me.age_confirmed) openOnboarding();
}

/** Start the app: Telegram glue, sign-in (Telegram initData or the password), then the first screen. */
export async function boot() {
  applyTheme();
  tg.initTelegram();
  installUi();
  drawShell();
  setExplainContext(() => ({ priced: priced().result }));
  onUnauthorized(() => (tg.inTelegram ? telegramUnauthorized() : showSignIn()));
  slip.onChange(() => {
    renderSlipBar();
    schedulePricing();
    refreshSlipSheet();
    refreshMainButton();
  });
  onPriced(() => {
    renderSlipBar();
    refreshSlipSheet();
  });
  let wasOnBreak = onBreak();
  onMeChange(() => {
    updateAvatar();
    updateBreakBanner();
    scheduleRealityCheck();
    // A break started or ended (here, in another password browser or in the bot): the slip's price from
    // before is out of date: a break drops it (the server prices nothing then), its end asks again; Analyse follows
    if (onBreak() !== wasOnBreak) {
      wasOnBreak = onBreak();
      if (slip.count) priceSlip({ force: true });
      refreshMainButton();
    }
    renderSlipBar(); // prices in the person's odds format, as the screen behind shows them
    refreshSlipSheet();
  });
  sheet.onToggle(() => {
    updateBackButton();
    refreshMainButton();
  });
  // Links clicked inside a sheet close it first, so Back doesn't land on a stale sheet entry
  document.addEventListener("click", (event) => {
    const link = event.target.closest('a[href^="#/"]');
    if (link && sheet.isOpen && link.closest(".sheet-layer")) {
      event.preventDefault();
      navigate(link.getAttribute("href"));
    } else if (event.target.closest('[data-action="retry"]')) {
      rerender();
    }
  });
  renderSlipBar();

  if (tg.inTelegram) {
    setInitData(tg.initData());
    try {
      await telegramLogin(tg.initData());
    } catch (error) {
      if (error.status === 403) {
        showRefused(error.message);
        return;
      }
      // Telegram's data is stale or wrong (401): only opening the app again from the bot renews it. Telegram
      // sign-in isn't set up on this server (503): the owner's password still works.
      if (error.status === 401) showTelegramExpired(error.message);
      else if (error.status !== 0) showSignIn(`Telegram sign-in didn't work: ${error.message}`);
    }
  }
  if (!gateOpen) await start();
  const startParam = tg.startParam();
  if (startParam) {
    const [kind, ...rest] = startParam.split("_");
    if (kind === "player" && rest.length >= 2) navigate(`#/player/${rest.slice(0, -1).join("_")}/${rest[rest.length - 1]}`);
    else if (kind === "game" && rest.length >= 2) navigate(`#/game/${rest.slice(0, -1).join("_")}/${rest[rest.length - 1]}`);
    else if (["ask", "build", "bets", "help", "settings", "today"].includes(kind)) navigate(`#/${kind}`);
  }
}
