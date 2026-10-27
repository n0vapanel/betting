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
//   #/today  #/game/<league>/<id>  #/player/<league>/<id>  #/build  #/build/rollover  #/build/builder  #/ask
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
//                   applyTheme(choice), inTelegram, markPainted(source), and re-exported from store.js: slip, priceSlip,
//                   priced, onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports,
//                   pinnedSports, isAllSports, MAX_PINNED, setSlateLeagues, canPaintCached, loadSports, leagueTitle,
//                   sportEmoji, legKey, legFromOffer, session; from layout.js and cache.js: see "The desk" below
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
//
// ── The desk and the instant Today (v3 Stage 1) ───────────────────────────────────────────────────
// Full details at the top of js/layout.js (panes) and js/cache.js (cache and prefetch). In short, all
// re-exported here, so a screen imports them from "../app.js" (views/today/*.js: "../../app.js"):
//   layout.mode ("phone" <840 | "tablet" 840–1199 | "desk" 1200–1599 | "wide" ≥1600), onLayoutChange(fn)
//   mountTree(el, {signal, title}) -> {unmount(), placement}; openTreeDrawer({returnTo}); closeTreeDrawer();
//     setTreeRail([{key, icon, label, count, live, active}], onPick); treeCollapsed(); setTreeCollapsed(bool)
//   openInspector(hash, {anchor}) -> bool (false: navigate instead); openDetail(hash, {anchor}); closeInspector();
//     inspectorRoute(); inspectorAvailable(); onInspectorChange(fn). Links to #/game/… and #/player/… open there
//     by themselves. The inspector takes its column only while it shows something (body.inspecting).
//   railState() "pinned" | "collapsed" | "peek"; setRailState(s); onRailChange(fn); railShown() (desk/wide, not on
//     Build, the slip's own page); openSlip() (= openSlipSheet(): rail, right drawer or bottom sheet)
//   cache.get/put/del/peek/warm, cache.swr(url, {key, onData, onNotModified, signal, maxAge}) (= swr),
//     SLATE_URL, SLATE_KEY, prefetch(url, {priority}), prefetched(url), gameUrl(league, id),
//     playerUrl(league, id), routeUrl(hash), canPrefetchGame(game, league)
//   markPainted(source)     Today calls it right after its first paint ("cache" or "network"): Telegram's
//                           ready() goes then, so Telegram drops its loading screen on the cached board
//   canPaintCached()        the age and break gates, from the profile (swr checks it before a cached paint)
//   pinnedSports(), isAllSports(), MAX_PINNED, setSlateLeagues(leagues)   "All sports" is the default;
//                           someone's sports are pinned first, never "only these" (store.js)
//   registerLeg(leg) -> key (ui.js)  a price button drawn from the slate: data-leg-key="<key>" toggles it
//   checkTree(...), bindCheckTree(...) (ui.js)  grouped tri-state checkboxes with Select all / Clear
// Hover (170 ms, mouse) or pointer-down (touch) on an element with data-prefetch="#/game/…" (or an
// /api/… path) prefetches it, and loads the page's code (warmView(name, {idle}): Today also loads the game
// page's code once it has painted); player links (#/player/…) prefetch by themselves. Game links need the
// attribute because a game's detail can cost odds credits when its league's odds are old (canPrefetchGame).
// Skeletons stay invisible for 300 ms (styles.css), so only a slow first visit ever shows one.

import { api, setInitData, onUnauthorized, passwordLogin, telegramLogin, logout, isQuiet, prefs, getPlayerId } from "./api.js";
import * as tg from "./telegram.js";
import {
  slip, priceSlip, priced, onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports, loadSports,
  leagueTitle, sportEmoji, legKey, legFromOffer, session, pinnedSports, isAllSports, MAX_PINNED, setSlateLeagues,
  canPaintCached, readCachedMe, clearCachedMe, setMeIdentity,
} from "./store.js";
import {
  esc, pct, chanceText, money, signedMoney, priceText, toast, sheet, explain, skeleton, errorState, installUi,
  setExplainContext, notice, segmented, timeLabel, openExplainer, holdPage, checkTree, bindCheckTree, confirmSheet,
  legChanceWord, ticketChanceWord,
} from "./ui.js";
import { icon, logo } from "./icons.js";
import * as desk from "./layout.js";
import { cache, setPaintCheck, prefetch, routeUrl, SLATE_KEY } from "./cache.js";
import { legBadges } from "./views/context/badges.js";

export {
  slip, priceSlip, priced, onPriced, me, loadMe, setMe, updateMe, onMeChange, onBreak, oddsFormat, favouriteSports, loadSports,
  leagueTitle, sportEmoji, legKey, legFromOffer, session, pinnedSports, isAllSports, MAX_PINNED, setSlateLeagues, canPaintCached,
};
export {
  layout, onLayoutChange, mountTree, openTreeDrawer, closeTreeDrawer, setTreeRail, treeCollapsed, setTreeCollapsed,
  openInspector, openDetail, closeInspector, inspectorRoute, inspectorAvailable, onInspectorChange,
  railState, setRailState, onRailChange, openSlip, railShown,
} from "./layout.js";
export {
  cache, swr, prefetch, prefetched, gameUrl, playerUrl, routeUrl, canPrefetchGame, SLATE_URL, SLATE_KEY,
} from "./cache.js";
export const inTelegram = tg.inTelegram;

/**
 * Today: call right after the first paint of the board ("cache" when drawn from this device's copy,
 * "network" otherwise). Telegram's ready() goes now (once), so its loading screen gives way to the board.
 * Leaves a performance mark "sportsbet:painted" (its detail says which) for the timing checks.
 */
export function markPainted(source = "network") {
  try {
    if (!performance.getEntriesByName("sportsbet:painted").length) performance.mark("sportsbet:painted", { detail: { source } });
  } catch {
    /* old browsers without mark details */
  }
  tg.ready();
}

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
  { name: "build", pattern: /^build(?:\/(rollover|builder))?$/, keys: ["mode"], tab: "build", title: "Rollover planner",
    titles: { rollover: "Rollover planner", builder: "Builder" }, load: () => import("./views/build.js") },
  { name: "ask", pattern: /^ask$/, tab: "ask", load: () => import("./views/ask.js") },
  { name: "bets", pattern: /^bets$/, tab: "bets", load: () => import("./views/bets.js") },
  { name: "help", pattern: /^help(?:\/([a-z0-9_-]{1,40}))?$/, keys: ["topic"], tab: "help", title: "Help", load: () => import("./views/help.js") },
  { name: "settings", pattern: /^settings$/, tab: null, sub: "#/today", title: "Settings", load: () => import("./views/settings.js") },
];

const warmed = new Set();

/**
 * Load a screen's code before it is opened ("game", "player"): with its data on hover or pointer-down, and
 * once Today has painted (``idle``), so the first game opened after a deploy doesn't wait for its modules.
 */
export function warmView(name, { idle = false } = {}) {
  const route = ROUTES.find((r) => r.name === name);
  if (!route || warmed.has(name)) return;
  warmed.add(name);
  const go = () => route.load().catch(() => warmed.delete(name)); // offline: the tap loads it as before
  if (!idle) go();
  else if (typeof requestIdleCallback === "function") requestIdleCallback(go, { timeout: 3000 });
  else setTimeout(go, 300);
}

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
  root.innerHTML = skeleton("cards", 2); // invisible for 300 ms: a module already loaded never shows it
  $("#view").replaceChildren(root);
  active = { route, params: { ...params, query, signal: controller.signal, path }, module: null, root, controller, hash: location.hash };
  desk.routeChanged(); // the inspector belongs to Today; the tree pane to whichever screen mounts one
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
  tg.ready(); // the first screen is drawn (Today calls markPainted() sooner, from its cached copy)
}

/** Header, tab bar and Telegram's Back button for the current screen. */
function updateChrome() {
  const { route, params } = active;
  const sub = isSubPage(route, params);
  for (const tab of document.querySelectorAll(".tabbar a, .toptabs a")) {
    if (tab.dataset.tab === route.tab) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  document.body.dataset.route = route.name;
  document.body.dataset.sub = (route.name === "build" && params.mode) || ""; // "rollover" | "builder" (layout and styles)
  $("#skip").textContent = route.name === "today" ? "Skip to games" : "Skip to content";
  $("#back").hidden = !sub || tg.inTelegram;
  drawTitle(sub ? route.titles?.[params.mode] || route.title || "" : null); // a screen may set a better one once its data is in
  if (route.name === "settings") $(".avatar").setAttribute("aria-current", "page");
  else $(".avatar").removeAttribute("aria-current");
  updateBackButton();
}

function updateBackButton() {
  const sub = active && isSubPage(active.route, active.params);
  tg.setBackButton(sheet.isOpen || sub ? goBack : null);
}

// While the inspector draws a game or player, that screen's setTitle() names the inspector, not the page
let titleSink = null;

/** The header title on sub-pages (null shows the app's name). */
export function setTitle(text) {
  if (titleSink) {
    titleSink(text);
    return;
  }
  drawTitle(text);
}

function drawTitle(text) {
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
  desk.refreshSlipChrome(); // the rail's tab and the header's slip button (desk and tablet)
  if (!n) return;
  const s = slipSummary();
  bar.querySelector(".slip-open").innerHTML = `<span class="slip-count" aria-hidden="true">${n}</span>
    <span class="slip-text"><b>${esc(s.main)}</b>${s.detail ? `<span class="${s.warn ? "warn" : ""}">${s.warn ? icon("alert", { size: 14 }) : ""}${esc(s.detail)}</span>` : ""}</span>
    ${icon("up", { size: 18 })}`;
  bar.querySelector(".slip-open").setAttribute("aria-label", `Bet slip: ${s.main}${s.detail ? `, ${s.detail}` : ""}. Open the slip`);
}

/** The rail tab's and the header button's words: {count, odds (the multi's price, or ""), label}. */
function railSummary() {
  const n = slip.count;
  const { result } = priced();
  const odds = n && result?.valid && result.odds && (result.type === "multi" || result.legs?.length === 1) && !onBreak()
    ? priceText(result.odds) : "";
  const label = n ? `Bet slip: ${legsWord(n)}${odds ? `, odds ${odds}` : ""}` : "Bet slip: empty";
  return { count: n, odds, label };
}

function legLine(leg, view, removed) {
  const odds = view?.odds || { decimal: leg.price, american: leg.american };
  const ai = view ? view.ai_prob : leg.ai;
  const market = view ? view.market_prob : leg.fair;
  // Venue & conditions: the game's context chips (from the slate on this device) and "adj" when the server moved
  // this leg's chance for a forecast or roof change since the odds
  const context = legBadges(leg.ref, { adjusted: view?.adjusted || null });
  return `<li class="slip-leg${removed ? " gone" : ""}">
    <div class="grow">
      <div class="row-between"><b>${esc(view?.label || leg.label)}</b><b class="num">${esc(priceText(odds))}</b></div>
      <div class="muted small">${esc(leg.game || view?.game || "")}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""}</div>
      ${context ? `<div class="slip-ctx">${context}</div>` : ""}
      <div class="small">${ai != null ? `${legChanceWord(view || leg)} ${pct(ai)}` : ""}${ai != null && market != null ? " · " : ""}${market != null ? `Market ${pct(market)}` : ""}
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
function slipNumbersHtml(view) {
  const { busy, result, error } = priced();
  // On a break the server prices nothing: no numbers from before it, and the line Analyse points to
  if (onBreak()) return `<div id="${view.uid}-break-note">${notice("You're on a break: pricing, analysing and saving a slip are off until it ends.", "info")}</div>`;
  let summary = "";
  if (result && result.bets) {
    const c = result.chance || {};
    const e = result.expected || {};
    const ml = result.most_likely;
    const chanceWord = c.headline === "profit" ? "Chance of a profit" : "Chance it wins";
    // "Adjusted" when every leg's own chance was moved by a forecast or roof change (as its price button says), else "AI"
    const own = ticketChanceWord(result);
    summary = `<dl class="summary">
      ${result.odds ? `<div><dt>Odds</dt><dd>${esc(priceText(result.odds))}</dd></div>` : ""}
      <div><dt>Costs</dt><dd>${money(result.total_stake, { cents: true })}${result.bets > 1 ? ` <span class="muted">(${result.bets} bets)</span>` : ""}</dd></div>
      <div><dt>Pays ${explain("payout")}</dt><dd>${result.type === "multi" || result.bets === 1 ? "" : "up to "}${money(result.max_payout, { cents: true })}</dd></div>
      <div><dt>${chanceWord} ${explain("one_in")}</dt><dd>${c.ai != null ? `${own} ${esc(chanceText(c.ai, c.ai_one_in))}` : ""}${c.ai != null && c.market != null ? "<br>" : ""}${c.market != null ? `Market ${esc(chanceText(c.market, c.market_one_in))}` : ""}${c.ai == null && c.market == null ? "–" : ""}</dd></div>
      <div><dt>Bookmaker keeps ${explain("book_cut")}</dt><dd>${result.book_cut != null ? `about ${pct(result.book_cut)}` : "–"}</dd></div>
      <div><dt>Expected result ${explain("expected_result")}</dt><dd>${e.ai != null ? `${esc(signedMoney(e.ai))} <span class="muted">${own === "AI" ? "AI" : "adjusted"}</span>` : ""}${e.ai != null && e.market != null ? "<br>" : ""}${e.market != null ? `${esc(signedMoney(e.market))} <span class="muted">market</span>` : ""}${e.ai == null && e.market == null ? "–" : ""}</dd></div>
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

function slipSheetBody(view) {
  if (!slip.count) {
    return `<div class="empty"><div class="empty-icon">${icon("bets", { size: 28 })}</div><h3>Your slip is empty</h3>
      <p>Tap any price on Today to add a pick, or let the AI build one.</p>
      <div class="empty-actions"><a class="btn btn-primary" href="#/today">Browse games</a><a class="btn btn-secondary" href="#/build/builder">Let AI build one</a></div></div>`;
  }
  return `
    <ul class="slip-legs" data-part="legs">${slipLegsHtml(priced().result)}</ul>
    <div class="field" data-part="type">${slipTypeHtml()}</div>
    <div class="field">
      <label class="label" for="${view.uid}-stake" data-part="stake-label">${stakeLabel()}</label>
      <div class="stake-row">
        ${STAKES.map((v) => `<button type="button" class="chip${slip.stake === v ? " on" : ""}" aria-pressed="${slip.stake === v}" data-stake="${v}">$${v}</button>`).join("")}
        <span class="money-input"><span aria-hidden="true">$</span><input id="${view.uid}-stake" data-slip-stake class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(customStake())}" placeholder="Other" aria-label="Other stake"></span>
      </div>
    </div>
    <div data-part="numbers">${slipNumbersHtml(view)}</div>
    <div class="slip-save" data-part="save">${slipSaveHtml()}</div>
    <p class="muted small">Nothing is bet here: the app prices picks so you can see the chances. Prices are rechecked on the server.</p>`;
}

function slipSheetFooter(view) {
  if (!slip.count) return "";
  return `<button type="button" class="btn btn-secondary" data-slip="clear">${icon("trash", { size: 18 })}Clear</button>
    <button type="button" class="btn btn-secondary" data-slip="build">Open in Build</button>
    <button type="button" class="btn btn-primary grow" data-slip="analyse"${analyseOff(view)}>${icon("ask", { size: 18 })}Analyse with AI</button>`;
}

/** The slip sheet's Analyse is off on a break, described by the line in place of the numbers. */
function analyseOff(view) {
  return onBreak() ? ` disabled aria-describedby="${view.uid}-break-note"` : "";
}

// --- Save and Share (POST /api/slips; a share code belongs to a saved slip) ----------------------------------

let savedSlip = null; // {text, view}: the slip as last saved here, and the server's view of it
let saving = false;
// Each slip view has its own element ids: "slip-…" in the sheet (as before the rail), "rail-…" in the rail

const isSaved = () => Boolean(savedSlip && savedSlip.text === JSON.stringify(slip.body()));

function slipSaveHtml() {
  if (!slip.count) return "";
  const { result } = priced();
  const off = onBreak();
  const canSave = !off && !saving && Boolean(result?.valid && result.bets > 0);
  const saved = isSaved();
  return `<button type="button" class="btn btn-secondary small" data-slip="save"${saved || !canSave ? " disabled" : ""}>${icon(saved ? "check" : "bets", { size: 16 })}${saved ? "Saved" : saving ? "Saving…" : "Save"}</button>
    <button type="button" class="btn btn-secondary small" data-slip="share"${saved || canSave ? "" : " disabled"}>${icon("share", { size: 16 })}Share</button>
    ${saved ? `<span class="small muted">In <a href="#/bets?tab=slips">My bets</a>${savedSlip.view?.share_code ? ` · code <b>${esc(savedSlip.view.share_code)}</b>` : ""}</span>` : ""}`;
}

async function saveSlip() {
  if (saving) return null;
  const body = slip.body();
  const text = JSON.stringify(body);
  saving = true;
  refreshSlipViews();
  try {
    const view = await api("/api/slips", { method: "POST", body });
    savedSlip = { text, view };
    tg.haptic.notify("success");
    toast("Saved to My bets", { kind: "good" });
    if (me().budget) loadMe().catch(() => null); // the weekly budget left has changed
    return view;
  } catch (error) {
    if (!isQuiet(error)) {
      tg.haptic.notify("error");
      toast(error.message, { kind: "warn", ms: 4200 });
    }
    return null;
  } finally {
    saving = false;
    refreshSlipViews();
  }
}

function shareLink(code) {
  const base = String(me().public_url || location.origin).replace(/\/+$/, "");
  return `${base}/#/build?code=${encodeURIComponent(code)}`;
}

async function shareSlip() {
  let view = isSaved() ? savedSlip.view : null;
  if (!view) {
    const total = priced().result?.total_stake;
    const ok = await confirmSheet({
      title: "Save and share?",
      body: `Sharing saves this slip to My bets first: a share code belongs to a saved slip. It's graded when its games end and counts in your profit and loss${me().budget && total ? `, and its ${money(total, { cents: true })} counts toward your weekly budget` : ""}.`,
      confirm: "Save & share",
    });
    if (!ok) return;
    view = await saveSlip();
    if (!view) return;
  }
  const code = view.share_code;
  if (!code) {
    toast("This slip has no share code", { kind: "warn" });
    return;
  }
  const link = shareLink(code);
  const s = sheet.open({
    title: "Share this slip",
    body: `<p class="lead">Anyone using this app can load your picks with this code. Prices are checked again when they open it.</p>
      <div class="share-code" aria-label="Share code ${esc(code.split("").join(" "))}">${esc(code)}</div>
      <p class="hint">${esc(link)}</p>`,
    footer: `<button type="button" class="btn btn-secondary grow" data-share="code">${icon("copy", { size: 18 })}Copy code</button>
      <button type="button" class="btn btn-secondary grow" data-share="link">${icon("copy", { size: 18 })}Copy link</button>
      ${navigator.share ? `<button type="button" class="btn btn-primary grow" data-share="native">${icon("share", { size: 18 })}Share…</button>` : ""}`,
  });
  s.foot.addEventListener("click", async (event) => {
    const how = event.target.closest("[data-share]")?.dataset.share;
    if (!how) return;
    if (how === "native") {
      try {
        await navigator.share({ title: "My SportsBet AI slip", text: `My slip on SportsBet AI: code ${code}`, url: link });
      } catch {
        /* the person closed the share sheet */
      }
      return;
    }
    const { copyText } = await import("./legacy.js"); // only when copying: legacy.js isn't on the start path
    copyText(how === "code" ? code : link, how === "code" ? "Code" : "Link");
  });
}

// --- the slip, drawn in three places ---------------------------------------------------------------------
//
// One slip view, drawn in the phone's bottom sheet, the tablet's right drawer (the same sheet, styled) and
// the desk's rail. Each is drawn once, then refreshed part by part (data-part="..."), so a tap on a stake,
// the bet type or a leg's bin doesn't scroll it back to the top, and a stake being typed isn't replaced.

const slipViews = new Set(); // {body, foot, setTitle(text), home() -> its heading, focusHome(), kind: "sheet" | "rail"}

function drawSlipView(view) {
  view.body.innerHTML = slipSheetBody(view);
  view.foot.innerHTML = slipSheetFooter(view);
  view.foot.hidden = !slip.count;
  view.setTitle(`Your slip${slip.count ? ` (${slip.count})` : ""}`);
}

/** A selector for the same slip control after its part is drawn again (bet type, Save, a stake, a leg's bin). */
function slipControlKey(node) {
  const el = node?.closest?.("[data-seg][data-value], [data-slip], [data-stake], [data-remove-leg]");
  if (!el) return null;
  if (el.dataset.seg != null) return `[data-seg="${CSS.escape(el.dataset.seg)}"][data-value="${CSS.escape(el.dataset.value)}"]`;
  for (const attr of ["data-slip", "data-stake", "data-remove-leg"]) {
    if (el.hasAttribute(attr)) return `[${attr}="${CSS.escape(el.getAttribute(attr))}"]`;
  }
  return null;
}

const SAVE_KEY = '[data-slip="save"]';

/**
 * Focus in a slip view whose control was just drawn again: its new copy; for Save, Share once the slip is
 * saved (Save becomes a disabled "Saved"; while "Saving…" focus waits on the heading); else the view's
 * heading. Never <body>: the rail is a pane, and Tab would start over at the top of the page.
 */
function keepSlipFocus(view, key) {
  const active = document.activeElement;
  const waiting = Boolean(view.pending) && active === view.home();
  if (!waiting && active && active !== document.body && active.isConnected) { // still there, or moved on by the person
    view.pending = null;
    return;
  }
  const want = key || view.pending;
  view.pending = null;
  const usable = (el) => Boolean(el && !el.disabled && el.isConnected);
  const inView = (selector) => view.body.querySelector(selector) || view.foot?.querySelector(selector);
  let target = want ? inView(want) : null;
  if (!usable(target) && want === SAVE_KEY) {
    target = inView('[data-slip="share"]');
    if (!usable(target) && saving) view.pending = SAVE_KEY;
  }
  if (usable(target)) target.focus({ preventScroll: true });
  else if (active !== view.home()) view.focusHome();
}

/** Bring a slip view up to date in place: same scroll position, the stake being typed left alone. */
function refreshSlipView(view) {
  const body = view.body;
  const part = (name) => body.querySelector(`[data-part="${name}"]`);
  const focused = document.activeElement;
  const hadFocus = Boolean(focused && (body.contains(focused) || view.foot?.contains(focused)));
  const key = hadFocus ? slipControlKey(focused) : null;
  view.setTitle(`Your slip${slip.count ? ` (${slip.count})` : ""}`);
  if (!slip.count || !part("legs")) { // emptied, or the empty slip got its first pick: draw it all
    const top = body.scrollTop;
    drawSlipView(view);
    body.scrollTop = top;
    if (hadFocus || view.pending) keepSlipFocus(view, key);
    return;
  }
  part("legs").innerHTML = slipLegsHtml(priced().result);
  part("type").innerHTML = slipTypeHtml();
  part("stake-label").textContent = stakeLabel();
  for (const chip of body.querySelectorAll("[data-stake]")) {
    chip.classList.toggle("on", slip.stake === chip.dataset.stake);
    chip.setAttribute("aria-pressed", String(slip.stake === chip.dataset.stake));
  }
  const input = body.querySelector("[data-slip-stake]");
  if (input && document.activeElement !== input) input.value = customStake();
  part("numbers").innerHTML = slipNumbersHtml(view);
  part("save").innerHTML = slipSaveHtml();
  const analyse = view.foot?.querySelector('[data-slip="analyse"]');
  if (analyse && analyse.disabled !== onBreak()) { // a break started or ended with the slip open
    analyse.disabled = onBreak();
    if (analyse.disabled) analyse.setAttribute("aria-describedby", `${view.uid}-break-note`);
    else analyse.removeAttribute("aria-describedby");
  }
  if (hadFocus || view.pending) keepSlipFocus(view, key);
}

function refreshSlipViews() {
  for (const view of slipViews) refreshSlipView(view);
}

/** The slip's taps and the custom stake, on the element holding a slip view (``onClear`` after Clear). */
function bindSlipView(container, view, { onClear = () => {} } = {}) {
  container.addEventListener("click", async (event) => {
    const remove = event.target.closest("[data-remove-leg]");
    if (remove) {
      slip.remove(remove.dataset.removeLeg);
      tg.haptic.select();
      if (!container.contains(document.activeElement)) view.focusHome(); // its button is gone
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
      onClear();
    } else if (action === "build") {
      navigate("#/build");
    } else if (action === "analyse") {
      analyseSlip();
    } else if (action === "save") {
      saveSlip();
    } else if (action === "share") {
      shareSlip();
    }
  });
  container.addEventListener("change", (event) => {
    if (!event.target.matches("[data-slip-stake]")) return;
    const value = event.target.value.trim().replace(/^\$/, "");
    if (!value) return;
    if (!/^\d{1,6}(\.\d{1,2})?$/.test(value) || Number(value) < 0.01 || Number(value) > 100000) {
      toast("Enter a stake between $0.01 and $100,000", { kind: "warn" });
      return;
    }
    slip.set({ stake: value });
  });
}

let slipSheet = null; // the slip's view while it is open as a sheet

/**
 * Open the slip: in the desk's rail (1200 px and up: focused, or peeking when collapsed), else as the
 * right-hand drawer (tablet) or the bottom sheet (phone): legs, bet type, stake chips and the server's numbers.
 */
export function openSlipSheet() {
  if (desk.railShown()) {
    desk.openSlip();
    priceSlip();
    return;
  }
  const drawer = desk.layout.mode !== "phone";
  const s = sheet.open({
    title: `Your slip${slip.count ? ` (${slip.count})` : ""}`, body: "", footer: "", size: drawer ? "auto" : "tall",
    className: drawer ? "drawer drawer-right slip-sheet" : "slip-sheet",
    onClose: () => {
      if (slipSheet) slipViews.delete(slipSheet);
      slipSheet = null;
    },
  });
  const view = {
    kind: "sheet", uid: "slip", body: s.body, foot: s.foot, // one sheet at a time: #slip-stake, #slip-break-note
    setTitle: (text) => { if (s.open) sheet.update({ title: text }); },
    home: () => s.panel,
    focusHome: () => s.panel.focus({ preventScroll: true }),
  };
  if (slipSheet) slipViews.delete(slipSheet);
  slipSheet = view;
  slipViews.add(view);
  drawSlipView(view);
  priceSlip();
  bindSlipView(s.panel, view, { onClear: () => s.close() });
}

// The desk's rail: its slip is drawn only while the rail is on screen (1200 px and up), so phones and
// tablets carry no hidden second copy of the slip
let railView = null;

/** layout.js hands over the rail's body and foot at start; the slip goes in whenever the rail shows. */
function drawRail(body, foot) {
  railView = {
    kind: "rail", uid: "rail", body, foot,
    setTitle: (text) => { document.getElementById("rail-title").textContent = text; },
    home: () => document.getElementById("rail-title"),
    focusHome: () => document.getElementById("rail-title").focus({ preventScroll: true }),
  };
  // Clear empties the rail in place (the sheet closes instead): focus goes to its heading, not to <body>
  bindSlipView(desk.railParts().panel, railView, { onClear: () => { if (!railView.body.contains(document.activeElement)) railView.focusHome(); } });
  syncRail();
}

function syncRail() {
  if (!railView) return;
  const shown = desk.railShown();
  if (shown && !slipViews.has(railView)) {
    slipViews.add(railView);
    drawSlipView(railView);
  } else if (!shown && slipViews.has(railView)) {
    slipViews.delete(railView);
    railView.body.replaceChildren();
    railView.foot.replaceChildren();
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
  tg.ready(); // inside Telegram the gate must show, not Telegram's loading screen
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
      signedInAgain = true; // whatever drew behind the form was refused: draw it again
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
  clearCachedMe();
  await cache.clear().catch(() => null); // nothing from this session is drawn before the next sign-in
  // The address changes without a hashchange: drawing Today here would only ask for the slate signed out (a 401)
  replaceAddress("#/today");
  location.reload();
}

// --- onboarding (first launch) ----------------------------------------------------------------------------

/** The leagues as check-tree groups, by sport ("⚽ Soccer"), in the order the server lists them. */
function sportGroups(sports) {
  const groups = [];
  for (const s of sports) {
    const label = s.label || s.group || "Other";
    let group = groups.find((g) => g.key === label);
    if (!group) groups.push(group = { key: label, label, items: [] });
    group.items.push({ key: s.key, label: s.title || s.key, tag: s.ai ? "AI" : "" });
  }
  return groups;
}

/**
 * "Your sports" as a check tree: every league ticked means "All sports" (the default, saved as no pins);
 * some ticked means those come first on Today (the rest still follow). Returns {html, bind(panel, onChange)}.
 */
function sportsPicker(sports, name) {
  const all = sports.map((s) => s.key);
  const pinned = pinnedSports().filter((key) => all.includes(key));
  const checked = new Set(pinned.length ? pinned : all);
  const html = sports.length
    ? `${checkTree({ name, groups: sportGroups(sports), checked, allLabel: "All sports", unit: "leagues" })}
      <p class="hint" data-sports-hint></p>`
    : '<p class="muted small">Sports load once you\'re signed in.</p>';
  const hint = (set) => {
    if (!set.size || set.size === all.length) return { text: "All sports: Today lists every league, soonest games first.", ok: true };
    if (set.size > MAX_PINNED) return { text: `Pick up to ${MAX_PINNED} leagues to put first, or tick All sports. Today lists every league either way.`, ok: false };
    return { text: set.size === 1 ? "This one comes first on Today; every other league follows it." : `These ${set.size} come first on Today; every other league follows them.`, ok: true };
  };
  return {
    html,
    /** Wire it in ``panel``; onChange(ok) after every change. Returns {favourites() -> [] | [keys], ok()}. */
    bind(panel, onChange = () => {}) {
      if (!sports.length) return { favourites: () => null, ok: () => true };
      let current = new Set(checked);
      const note = panel.querySelector("[data-sports-hint]");
      const show = () => {
        const h = hint(current);
        note.textContent = h.text;
        note.classList.toggle("warn", !h.ok);
        onChange(h.ok);
      };
      bindCheckTree(panel, { checked, onChange: (set) => { current = set; show(); } });
      show();
      return {
        favourites: () => (!current.size || current.size === all.length ? [] : all.filter((key) => current.has(key))),
        ok: () => hint(current).ok,
      };
    },
  };
}

/**
 * Every league the sheets can offer: GET /api/sports plus the all-sports slate's (from Today, or this
 * device's saved slate when Today hasn't loaded it yet). Before the age is confirmed the slate is gated,
 * so onboarding may only have /api/sports.
 */
async function sheetSports() {
  if (!session.slateLeagues) {
    const saved = await cache.get(SLATE_KEY).catch(() => null);
    if (Array.isArray(saved?.value?.leagues)) setSlateLeagues(saved.value.leagues);
  }
  return loadSports();
}

/** Age, your sports, optional weekly budget, odds format and the AI disclosure, in one sheet. */
export async function openOnboarding() {
  let sports = [];
  try {
    sports = await sheetSports();
  } catch (error) {
    if (error.signIn) return;
  }
  let format = oddsFormat();
  const picker = sportsPicker(sports, "onb-sports");
  const s = sheet.open({
    title: "Welcome to SportsBet AI",
    size: "tall",
    body: `<p class="lead">See each game's real chances, price any slip honestly, and ask an AI analyst about it. Nothing is bet here.</p>
      <div class="field">
        <label class="check"><input type="checkbox" data-age> <span>I'm 18 or older (21+ in the US) and betting is legal where I live.</span></label>
      </div>
      <div class="field"><span class="label">Your sports <span class="muted">(all of them unless you choose)</span></span>
        ${picker.html}</div>
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
  const ageBox = s.panel.querySelector("[data-age]");
  let sportsOk = true;
  const ready = () => { startButton.disabled = !(ageBox.checked && sportsOk); };
  const chosen = picker.bind(s.panel, (ok) => { sportsOk = ok; ready(); });
  s.panel.addEventListener("change", (event) => {
    if (event.target.matches("[data-age]")) ready();
  });
  s.panel.addEventListener("click", async (event) => {
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
      const changes = { confirm_age: true, odds_format: format };
      const favourites = chosen.favourites();
      if (favourites) changes.favourite_sports = favourites;
      if (budget) changes.weekly_budget = budget;
      startButton.disabled = true;
      try {
        await updateMe(changes);
        await s.close();
        toast("You're all set", { kind: "good" });
        rerender();
      } catch (err) {
        if (!isQuiet(err)) error.textContent = err.message;
        ready();
      }
    }
  });
}

/**
 * Choose your sports (Today's sports button, Settings > Your sports): every league ticked is "All sports"
 * (saved as no pins); a few ticked come first on Today. onDone() after saving.
 */
export async function openSportsSheet({ onDone } = {}) {
  let sports;
  try {
    sports = await sheetSports();
  } catch (error) {
    if (!isQuiet(error)) toast(error.message, { kind: "error" });
    return;
  }
  const picker = sportsPicker(sports, "sheet-sports");
  const s = sheet.open({
    title: "Your sports", size: "tall",
    body: `${picker.html}<div class="form-error" role="alert" data-sports-error></div>`,
    footer: '<button type="button" class="btn btn-primary block" data-save-sports>Done</button>',
  });
  const done = s.panel.querySelector("[data-save-sports]");
  const chosen = picker.bind(s.panel, (ok) => { done.disabled = !ok; });
  s.panel.addEventListener("click", async (event) => {
    if (!event.target.closest("[data-save-sports]")) return;
    const favourites = chosen.favourites();
    done.disabled = true;
    try {
      if (favourites) await updateMe({ favourite_sports: favourites });
      await s.close();
      onDone?.();
    } catch (error) {
      if (!isQuiet(error)) s.panel.querySelector("[data-sports-error]").textContent = error.message;
      done.disabled = !chosen.ok();
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
  const tabs = (size) => TABS.map((t) => `<a href="${t.href}" data-tab="${t.key}">${icon(t.iconName, { size })}<span>${t.label}</span></a>`).join("");
  $("#tabbar-inner").innerHTML = tabs(22); // phone: the bottom tab bar
  $("#toptabs").innerHTML = tabs(18); // 840 px and up: the same tabs in the header
  drawTitle(null);
  updateAvatar();
  $("#back").innerHTML = icon("back", { size: 22 });
  $("#back").addEventListener("click", goBack);
  // The first Tab stop: past the header and the league tree, to Today's first game (or the screen's content)
  $("#skip").addEventListener("click", () => {
    const game = active?.route?.name === "today" ? $("#view [data-td-board] :is(.td-open, .gc-link)") : null;
    (game || $("#view")).focus();
  });
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
  if (timezoneChecked || !profile || session.meCached || !("timezone" in profile)) return false;
  timezoneChecked = true;
  const zone = deviceTimezone();
  if (!zone || zone === profile.timezone) return false;
  try {
    const updated = await api("/api/me", { method: "PUT", body: { timezone: zone }, auth: false });
    if (updated && typeof updated === "object" && "uid" in updated) setMe(updated);
    return true;
  } catch {
    return false; /* the server keeps the zone it had; nothing to show */
  }
}

let started = false; // the router is listening (the first screen may have drawn from this device's copy)

function beginRouting() {
  if (started) return;
  started = true;
  parseHash(); // tidy the address first (#/today for "/", no Telegram launch parameters) so Back history is clean
  window.addEventListener("hashchange", () => {
    trackVisit();
    rerender();
  });
  trackVisit();
}

/** What the first screen was drawn with differs from the server's profile in a way it shows. */
function profileShifted(before, after) {
  if (!before || !after) return Boolean(before) !== Boolean(after);
  return before.uid !== after.uid || before.odds_format !== after.odds_format || Boolean(before.age_confirmed) !== Boolean(after.age_confirmed)
    || onBreak(before) !== onBreak(after) || JSON.stringify(pinnedSports(before)) !== JSON.stringify(pinnedSports(after));
}

/**
 * The repeat visit's head start: with this person's profile saved on the device (age confirmed, not on a
 * break), the first screen draws now, while sign-in and GET /api/me run behind it. Today then paints its
 * saved slate at once (cache.swr) and calls markPainted(), which readies Telegram.
 */
function startEarly() {
  const saved = readCachedMe();
  if (!saved || !canPaintCached(saved)) return false;
  setMe(saved, { cached: true });
  updateAvatar();
  beginRouting();
  if (parseHash().path === "today") cache.warm(SLATE_KEY);
  rerender();
  return true;
}

async function start() {
  const drawnWith = started ? { ...me() } : null; // the profile the screen on show was drawn with
  try {
    await loadMe();
  } catch (error) {
    if (error.signIn) { // the sign-in form is up; it calls start() again
      clearCachedMe(); // and the next visit waits for sign-in before drawing anything
      return;
    }
    toast(error.message, { kind: "error" });
  }
  updateAvatar();
  updateBreakBanner();
  scheduleRealityCheck();
  loadSports().catch(() => null);
  if (slip.count) priceSlip();
  if (!started) {
    // Before the first screen (whose days may follow the zone), but never holding it up for long
    await Promise.race([syncTimezone(), new Promise((resolve) => { setTimeout(resolve, 1500); })]);
    beginRouting();
    await rerender();
  } else {
    // Drawn early (or before a sign-in): draw again only when it would now look different
    if (signedInAgain || profileShifted(drawnWith, session.me)) rerender();
    signedInAgain = false;
    syncTimezone().then((zoneChanged) => {
      if (zoneChanged && active?.route?.name !== "today") rerender(); // Today's days are the device's own
    });
  }
  if (session.me && !session.me.age_confirmed) openOnboarding();
}

let signedInAgain = false;

// --- prefetch on hover and pointer-down ---------------------------------------------------------------------

/**
 * Desktop: hovering (or keyboard-focusing) a prefetchable element for 170 ms fetches what it opens; phone:
 * pointer-down does at once. Prefetchable: [data-prefetch] (a route hash such as "#/game/<league>/<id>",
 * an /api/... path, or empty to use the element's href) and player links. At most 2 run at a time (cache.js).
 */
function installPrefetch() {
  const HOVER_MS = 170;
  const target = (el) => (el instanceof Element ? el.closest('[data-prefetch], a[href^="#/player/"]') : null);
  const urlOf = (el) => {
    const value = el.dataset.prefetch || el.getAttribute("href") || "";
    return value.startsWith("/api/") ? value : routeUrl(value);
  };
  const go = (el) => {
    const url = el && urlOf(el);
    if (!url) return;
    prefetch(url, { priority: "high" });
    warmView(url.startsWith("/api/players/") ? "player" : url.startsWith("/api/game") ? "game" : ""); // and its code
  };
  let hovered = null;
  let timer = null;
  const settle = (el) => {
    if (el === hovered) return;
    clearTimeout(timer);
    hovered = el;
    if (el) timer = setTimeout(() => go(el), HOVER_MS);
  };
  document.addEventListener("pointerover", (event) => {
    if (event.pointerType && event.pointerType !== "mouse") return;
    settle(target(event.target));
  }, { passive: true });
  document.addEventListener("pointerout", (event) => {
    if (hovered && !(event.relatedTarget instanceof Node && hovered.contains(event.relatedTarget))) settle(null);
  }, { passive: true });
  document.addEventListener("pointerdown", (event) => go(target(event.target)), { passive: true });
  document.addEventListener("focusin", (event) => settle(target(event.target)));
}

/** Start the app: Telegram glue, sign-in (Telegram initData or the password), then the first screen. */
export async function boot() {
  applyTheme();
  tg.initTelegram();
  installUi();
  // This device's copy of the profile is kept per person: the Telegram user, or this browser's own id
  setMeIdentity(tg.inTelegram ? (tg.userId() ? `tg:${tg.userId()}` : null) : `web:${getPlayerId()}`);
  setPaintCheck(() => canPaintCached());
  drawShell();
  desk.initLayout({
    routeName: () => active?.route?.name || null,
    routeMode: () => active?.params?.mode || null,
    match: (hash) => {
      const text = String(hash).replace(/^#\/?/, "");
      const q = text.indexOf("?");
      const found = matchRoute(q >= 0 ? text.slice(0, q) : text);
      if (!found) return null;
      return { route: found.route, params: { ...found.params, query: new URLSearchParams(q >= 0 ? text.slice(q + 1) : ""), path: q >= 0 ? text.slice(0, q) : text } };
    },
    navigate,
    setTitleSink: (fn) => { titleSink = fn; },
    drawRail,
    railSummary,
    openSlipSheet,
  });
  installPrefetch();
  // The window grew into the rail's layout with the slip open as a drawer: the rail shows it now
  desk.onLayoutChange((mode) => {
    syncRail();
    if (slipSheet && desk.railShown()) sheet.close();
  });
  setExplainContext(() => ({ priced: priced().result }));
  onUnauthorized(() => {
    clearCachedMe(); // signed out: the next visit waits for sign-in before drawing anything
    if (tg.inTelegram) telegramUnauthorized();
    else showSignIn();
  });
  slip.onChange(() => {
    renderSlipBar();
    schedulePricing();
    refreshSlipViews();
    refreshMainButton();
  });
  onPriced(() => {
    renderSlipBar();
    refreshSlipViews();
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
    refreshSlipViews();
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
      const href = link.getAttribute("href");
      // A game or player picked in a sheet (player search) opens beside the board when the inspector is there
      if (/^#\/(game|player)\//.test(href) && desk.inspectorAvailable() && !link.hasAttribute("data-no-inspector")) {
        sheet.close().then(() => desk.openDetail(href));
      } else {
        navigate(href);
      }
    } else if (event.target.closest('[data-action="retry"]')) {
      rerender();
    }
  });
  renderSlipBar();

  if (tg.inTelegram) setInitData(tg.initData()); // every call carries it, so the early screen can load too
  startEarly();
  // Telegram's loading screen never outstays a slow start: ready() at the latest after 3 seconds
  setTimeout(() => tg.ready(), 3000);

  if (tg.inTelegram) {
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
