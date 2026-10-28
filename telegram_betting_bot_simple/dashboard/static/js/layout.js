// The desk: which panes show at which window width, and the hooks screens use to fill them.
//
// ── Layouts (styles.css has the same breakpoints) ─────────────────────────────────────────────────
//   "phone"  < 840 px      bottom tabs, slim slip bar and bottom sheets (unchanged)
//   "tablet" 840–1199 px   tabs in the header; the league tree is a drawer from the left; the slip is a
//                          drawer from the right, opened from the header's slip button
//   "desk"   1200–1599 px  league tree (collapses to a 56 px icon rail) · board · slip rail (pinned, or a
//                          44 px tab showing legs and combined odds). Under 1280 px the tree starts as the
//                          icon rail until the person opens it (the board would be too narrow for names)
//   "wide"   ≥ 1600 px     league tree · board · inspector · slip rail
// The tree pane exists only while a screen has mounted a tree (Today); the inspector only on Today at
// "wide", and it takes its column only while it shows something (body.inspecting): an empty one would
// squeeze the board for nothing. Collapsed/pinned states are remembered in this browser (localStorage
// "sportsbet-desk").
//
// ── API for screens (also re-exported from js/app.js) ─────────────────────────────────────────────
//   layout.mode                        "phone" | "tablet" | "desk" | "wide" (now)
//   onLayoutChange(fn)                 fn(mode, previous) after each breakpoint change; returns unsubscribe
//
//   League tree (Today owns the tree's content; the shell owns where it sits):
//   mountTree(el, {signal, title})     put ``el`` in the left pane (desk/wide); on phone and tablet it is
//                                      kept (alive, hidden) for openTreeDrawer(). Unmounts by itself when
//                                      ``signal`` aborts (pass the screen's params.signal). Returns
//                                      {unmount(), get placement(): "pane" | "drawer" | "hidden"}.
//   openTreeDrawer({returnTo})         tablet: the tree in a drawer from the left; phone: in a tall bottom
//                                      sheet; desk/wide: expands the pane and focuses it. The same element
//                                      moves, so its listeners and state carry over. false without a tree.
//                                      returnTo() -> the element to focus when the drawer closes, should
//                                      the one that opened it be gone (redrawn) by then.
//   closeTreeDrawer()
//   setTreeRail(items, onPick)         what the collapsed 56 px rail shows: items = [{key, icon (an emoji or
//                                      short text), label, count, live (bool), active (bool)}]; a tap calls
//                                      onPick(key). Without it the rail shows only the expand button.
//   treeCollapsed() / setTreeCollapsed(bool)
//
//   Inspector (wide, on Today only):
//   inspectorAvailable()               true when openInspector() would open something
//   openInspector(hash, {anchor})      open "#/game/<league>/<id>" or "#/player/<league>/<id>" (?query
//                                      allowed) in the inspector; the board keeps its scroll, and ``anchor``
//                                      (the row tapped) stays where it is on screen as the board narrows
//                                      for the inspector's column (and widens again when it closes).
//                                      Returns false when it can't (narrower window, another screen, another
//                                      route): then navigate(hash) as usual. openDetail(hash) does both.
//   openDetail(hash, {anchor})         openInspector(hash) || navigate(hash)
//   closeInspector()                   close what it shows (its column goes, the board widens again)
//   inspectorRoute()                   the hash shown in the inspector, or null
//   onInspectorChange(fn)              fn(hash | null) when it opens, changes or closes; returns unsubscribe
//   Links to #/game/… and #/player/… anywhere on the page open in the inspector by themselves when it is
//   available (add data-no-inspector to a link that must navigate). Inside the inspector the views render
//   as on their own page (params.inspector = true); their setTitle() goes to the inspector's heading.
//
//   Slip rail (desk/wide; it steps aside on Build's slip and rollover pages, the slip's own page, and stays
//   on the Builder, #/build/builder):
//   railShown()                        true while the rail is on screen (desk/wide, not on #/build or #/build/rollover)
//   railState()                        "pinned" | "collapsed" (remembered) | "peek" (open over the board
//                                      from the collapsed tab until closed, pinned or clicked away)
//   setRailState(state)
//   onRailChange(fn)                   fn(state); returns unsubscribe
//   openSlip()                         the slip wherever it lives: rail (pinned: focus; collapsed: peek),
//                                      right drawer on tablet, bottom sheet on phone (= openSlipSheet())

import { sheet, esc } from "./ui.js";
import { icon } from "./icons.js";

const MEDIA = {
  tablet: matchMedia("(min-width: 840px)"),
  desk: matchMedia("(min-width: 1200px)"),
  wide: matchMedia("(min-width: 1600px)"),
};
const STATE_KEY = "sportsbet-desk";
const ROOMY = matchMedia("(min-width: 1280px)"); // room for the open tree by default

function currentMode() {
  if (MEDIA.wide.matches) return "wide";
  if (MEDIA.desk.matches) return "desk";
  if (MEDIA.tablet.matches) return "tablet";
  return "phone";
}

let mode = currentMode();
const $ = (sel) => document.querySelector(sel);
const isDesk = () => mode === "desk" || mode === "wide";

function readState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STATE_KEY) || "{}") || {};
    return {
      tree: saved.tree === "collapsed" || saved.tree === "open" ? saved.tree : null, // null: by the window's width
      rail: saved.rail === "collapsed" ? "collapsed" : "pinned",
    };
  } catch {
    return { tree: null, rail: "pinned" };
  }
}

function saveState() {
  try {
    localStorage.setItem(STATE_KEY, JSON.stringify({ tree: state.tree, rail: state.rail === "peek" ? "collapsed" : state.rail }));
  } catch {
    /* private browsing: remembered for this visit only */
  }
}

const state = readState();

/** The tree as the person left it, or by default open from 1280 px and the icon rail below. */
const treeIsCollapsed = () => (state.tree ? state.tree === "collapsed" : !ROOMY.matches);

// What app.js hands over at start (initLayout): the router's pieces and the slip's drawing
let hooks = {
  routeName: () => null, // the current route's name ("today", "game", ...)
  routeMode: () => null, // its sub-page, when it has one ("builder" on #/build/builder)
  match: () => null, // hash -> {route, params} for a route (app.js matchRoute + query)
  navigate: (hash) => { location.hash = hash; },
  setTitleSink: () => {}, // setTitle() goes to fn while the inspector draws (null: back to the header)
  drawRail: () => {}, // (body, foot) -> fill the rail's slip
  railSummary: () => ({ count: 0, odds: "", label: "Bet slip" }),
  openSlipSheet: () => {}, // the slip as a sheet (phone) or right drawer (tablet)
};

const watchers = { layout: new Set(), inspector: new Set(), rail: new Set() };
const emit = (kind, ...args) => watchers[kind].forEach((fn) => { try { fn(...args); } catch (e) { console.warn(e); } });

export const layout = {
  get mode() {
    return mode;
  },
};

export function onLayoutChange(fn) {
  watchers.layout.add(fn);
  return () => watchers.layout.delete(fn);
}

// --- the league tree ------------------------------------------------------------------------------------

let tree = null; // {el, title}
let treeInDrawer = false;
let rail = { items: [], onPick: null };

function placeTree() {
  const pane = $("#tree-pane");
  document.body.classList.toggle("has-tree", Boolean(tree));
  document.body.classList.toggle("tree-collapsed", Boolean(tree) && treeIsCollapsed());
  $("#tree-btn").hidden = !(tree && mode === "tablet");
  if (!tree) {
    pane.hidden = true;
    $("#tree-slot").replaceChildren();
    return;
  }
  pane.querySelector(".pane-title").textContent = tree.title;
  if (!treeInDrawer && tree.el.parentNode !== $("#tree-slot")) $("#tree-slot").replaceChildren(tree.el);
  pane.hidden = !isDesk();
  const toggle = pane.querySelector(".pane-head [data-tree-toggle]");
  toggle.setAttribute("aria-expanded", String(!treeIsCollapsed()));
}

export function mountTree(el, { signal, title = "Sports" } = {}) {
  if (!(el instanceof Element)) throw new TypeError("mountTree needs an element");
  if (tree && tree.el !== el) unmountTree();
  tree = { el, title };
  placeTree();
  const handle = {
    unmount() {
      if (tree?.el === el) unmountTree();
    },
    get placement() {
      if (tree?.el !== el) return "hidden";
      if (treeInDrawer) return "drawer";
      return isDesk() ? "pane" : "hidden";
    },
  };
  signal?.addEventListener("abort", () => handle.unmount(), { once: true });
  return handle;
}

function unmountTree() {
  if (treeInDrawer) sheet.close();
  treeInDrawer = false;
  tree = null;
  rail = { items: [], onPick: null };
  drawTreeRail();
  placeTree();
}

export function treeCollapsed() {
  return treeIsCollapsed();
}

export function setTreeCollapsed(collapsed) {
  state.tree = collapsed ? "collapsed" : "open";
  saveState();
  placeTree();
  emit("layout", mode, mode);
}

export function openTreeDrawer({ returnTo = null } = {}) {
  if (!tree) return false;
  if (isDesk()) {
    setTreeCollapsed(false);
    $("#tree-pane .pane-title").focus({ preventScroll: true });
    return true;
  }
  if (treeInDrawer) return true;
  treeInDrawer = true;
  const node = tree.el;
  sheet.open({
    title: tree.title,
    body: node,
    size: mode === "phone" ? "tall" : "auto",
    className: mode === "tablet" ? "drawer drawer-left" : "",
    onClose: () => {
      treeInDrawer = false;
      if (tree?.el === node) placeTree();
      // The sheet gives focus back to its opener next; when that was redrawn meanwhile, focus goes here
      if (returnTo) queueMicrotask(() => {
        if (document.activeElement && document.activeElement !== document.body) return;
        const back = returnTo();
        if (back?.isConnected) back.focus({ preventScroll: true });
      });
    },
  });
  return true;
}

export function closeTreeDrawer() {
  if (treeInDrawer) sheet.close();
}

export function setTreeRail(items = [], onPick = null) {
  rail = { items: Array.isArray(items) ? items : [], onPick };
  drawTreeRail();
}

function drawTreeRail() {
  const box = $("#tree-rail .rail-items");
  if (!box) return;
  box.innerHTML = rail.items.map((item) => {
    const count = item.count != null ? `${item.count} game${item.count === 1 ? "" : "s"}` : "";
    const words = [item.label, count, item.live ? `${item.live === true ? "" : `${item.live} `}live` : ""].filter(Boolean).join(", ");
    return `<button type="button" class="rail-item" data-rail-key="${esc(item.key)}" aria-pressed="${Boolean(item.active)}" aria-label="${esc(words)}" title="${esc(words)}">
      <span class="rail-icon" aria-hidden="true">${esc(item.icon || String(item.label || "?").slice(0, 2))}</span>
      ${item.count != null ? `<span class="rail-count" aria-hidden="true">${esc(item.count)}</span>` : ""}
      ${item.live ? '<span class="live-dot" aria-hidden="true"></span>' : ""}
    </button>`;
  }).join("");
}

// --- the inspector ----------------------------------------------------------------------------------------

const insp = { stack: [], current: null, seq: 0, returnFocus: null, anchor: null };

/**
 * Give the inspector its column or take it away (body.inspecting), keeping ``anchor`` (the row tapped) at
 * the same place on screen: the board's width changes, and rows above it may change height.
 */
function setInspecting(on, anchor) {
  if (document.body.classList.contains("inspecting") === on) return;
  const keep = anchor?.isConnected ? anchor : null;
  if (!keep) {
    document.body.classList.toggle("inspecting", on);
    return;
  }
  // Rows off screen (content-visibility) take their new height over the next frames: the row is put back
  // after each, for a few frames; the browser's own scroll anchoring stays off meanwhile (no double moves)
  const roots = [document.documentElement, document.body];
  for (const el of roots) el.style.overflowAnchor = "none";
  const top = keep.getBoundingClientRect().top;
  document.body.classList.toggle("inspecting", on);
  const settle = (frames) => {
    if (keep.isConnected) {
      const moved = keep.getBoundingClientRect().top - top;
      if (Math.abs(moved) >= 1) window.scrollBy(0, moved);
    }
    if (frames > 0 && keep.isConnected) requestAnimationFrame(() => settle(frames - 1));
    else for (const el of roots) el.style.overflowAnchor = "";
  };
  settle(6);
}

export function inspectorAvailable() {
  return mode === "wide" && hooks.routeName() === "today";
}

export function inspectorRoute() {
  return insp.current?.hash || null;
}

export function onInspectorChange(fn) {
  watchers.inspector.add(fn);
  return () => watchers.inspector.delete(fn);
}

const normal = (hash) => {
  const text = String(hash || "").trim();
  return text.startsWith("#") ? text : `#${text.startsWith("/") ? "" : "/"}${text}`;
};

/** The inspector's route for a hash (game or player pages only), or null. */
function inspectable(hash) {
  const found = hooks.match(normal(hash));
  return found && (found.route.name === "game" || found.route.name === "player") ? found : null;
}

export function openInspector(hash, { anchor = null } = {}) {
  if (!inspectorAvailable()) return false;
  const target = normal(hash);
  const found = inspectable(target);
  if (!found) return false;
  if (!insp.current) {
    insp.returnFocus = document.activeElement;
    insp.anchor = anchor instanceof Element && !anchor.closest("#inspector") ? anchor : null;
  }
  if (insp.current?.hash === target) {
    $("#inspector .pane-title").focus({ preventScroll: true });
    return true;
  }
  insp.stack.push(target);
  if (insp.stack.length > 12) insp.stack.splice(0, insp.stack.length - 12);
  show(target, found);
  return true;
}

export function openDetail(hash, options = {}) {
  if (openInspector(hash, options)) return true;
  hooks.navigate(normal(hash));
  return false;
}

export function closeInspector({ focus = true } = {}) {
  if (!insp.current) return;
  leaveInspected();
  insp.stack = [];
  drawInspectorEmpty();
  setInspecting(false, insp.anchor);
  insp.anchor = null;
  emit("inspector", null);
  const back = insp.returnFocus;
  insp.returnFocus = null;
  if (focus && back?.isConnected && !back.closest("#inspector")) back.focus({ preventScroll: true });
}

function leaveInspected() {
  const current = insp.current;
  if (!current) return;
  insp.current = null;
  insp.seq += 1;
  hooks.setTitleSink(null);
  current.controller.abort();
  try {
    current.module?.onLeave?.();
  } catch (error) {
    console.warn(error);
  }
}

function inspectorHead({ title = "Details", back = false, full = null } = {}) {
  const pane = $("#inspector");
  pane.querySelector(".pane-title").textContent = title;
  pane.querySelector('[data-insp="back"]').hidden = !back;
  const link = pane.querySelector('[data-insp="full"]');
  link.hidden = !full;
  if (full) link.setAttribute("href", full);
  pane.querySelector('[data-insp="close"]').hidden = !full;
}

function drawInspectorEmpty() {
  inspectorHead({ title: "Details" });
  $("#inspector .inspector-body").innerHTML = `<div class="insp-empty">
    <div class="empty-icon">${icon("today", { size: 28 })}</div>
    <h3>Pick a game</h3>
    <p>Tap a game or a player on the board to see it here. The board keeps its place.</p>
  </div>`;
}

async function show(hash, found) {
  leaveInspected();
  const seq = ++insp.seq;
  const { route, params } = found;
  const controller = new AbortController();
  const root = document.createElement("div");
  root.className = `view view-${route.name} in-inspector`;
  const body = $("#inspector .inspector-body");
  body.replaceChildren(root);
  body.scrollTop = 0;
  insp.current = { hash, controller, root, module: null };
  setInspecting(true, insp.anchor);
  inspectorHead({ title: route.title || "Details", back: insp.stack.length > 1, full: hash });
  emit("inspector", hash);
  $("#inspector .pane-title").focus({ preventScroll: true });
  let module;
  try {
    module = await route.load();
  } catch (error) {
    console.error(error);
    if (seq === insp.seq) root.innerHTML = '<p class="muted">This couldn\'t load. Reload the page to try again.</p>';
    return;
  }
  if (seq !== insp.seq) return;
  insp.current.module = module;
  hooks.setTitleSink((text) => {
    if (seq === insp.seq) $("#inspector .pane-title").textContent = text || route.title || "Details";
  });
  try {
    await module.render(root, { ...params, signal: controller.signal, inspector: true });
  } catch (error) {
    if (seq === insp.seq && error?.name !== "AbortError" && !error?.signIn) {
      console.error(error);
      root.innerHTML = `<p class="muted">${esc(error?.message || "Something went wrong.")}</p>`;
    }
  } finally {
    if (seq === insp.seq) hooks.setTitleSink(null);
  }
}

function inspectorBack() {
  if (insp.stack.length < 2) {
    closeInspector();
    return;
  }
  insp.stack.pop();
  const hash = insp.stack[insp.stack.length - 1];
  const found = inspectable(hash);
  if (found) show(hash, found);
  else closeInspector();
}

/** Show or hide the inspector pane for the current route and width (closing what it showed when it goes). */
function placeInspector() {
  const on = inspectorAvailable();
  document.body.classList.toggle("has-inspector", on);
  $("#inspector").hidden = !on;
  if (!on && insp.current) closeInspector({ focus: false });
}

// --- the slip rail ---------------------------------------------------------------------------------------

export function railState() {
  return state.rail;
}

export function onRailChange(fn) {
  watchers.rail.add(fn);
  return () => watchers.rail.delete(fn);
}

export function setRailState(next) {
  if (!["pinned", "collapsed", "peek"].includes(next) || next === state.rail) return;
  const was = state.rail;
  state.rail = next;
  if (next !== "peek") saveState();
  placeRail();
  if (next === "peek" || (next === "pinned" && was === "peek")) $("#rail-title").focus({ preventScroll: true });
  if (next === "collapsed" && was === "peek") $("#sliprail .rail-tab").focus({ preventScroll: true });
  emit("rail", next);
}

/**
 * The rail shows on desk widths, except on Build's slip and rollover pages: the slip is the page itself there.
 * The Builder (#/build/builder) keeps it, so "Add to slip" lands in view.
 */
export function railShown() {
  return isDesk() && (hooks.routeName() !== "build" || hooks.routeMode() === "builder");
}

function placeRail() {
  const pane = $("#sliprail");
  const shown = railShown();
  if (!shown && state.rail === "peek") state.rail = "collapsed";
  pane.hidden = !shown;
  document.body.classList.toggle("has-rail", shown);
  for (const s of ["pinned", "collapsed", "peek"]) document.body.classList.toggle(`rail-${s}`, shown && state.rail === s);
  const tab = pane.querySelector(".rail-tab");
  tab.setAttribute("aria-expanded", String(state.rail !== "collapsed"));
  pane.querySelector('[data-rail="collapse"]').hidden = state.rail !== "pinned";
  pane.querySelector('[data-rail="pin"]').hidden = state.rail !== "peek";
  pane.querySelector('[data-rail="close"]').hidden = state.rail !== "peek";
  refreshSlipChrome();
}

/** The rail tab's and the header slip button's text: legs and combined odds (app.js's summary). */
export function refreshSlipChrome() {
  const summary = hooks.railSummary();
  const tab = $("#sliprail .rail-tab");
  if (tab) {
    tab.innerHTML = `<span class="slip-count" aria-hidden="true">${esc(summary.count)}</span>
      <span class="rail-tab-text" aria-hidden="true">${esc(summary.odds ? `Slip · ${summary.odds}` : "Slip")}</span>
      ${icon("panel-right", { size: 18 })}`;
    tab.setAttribute("aria-label", `${summary.label}. Open the slip`);
  }
  const button = $("#slip-btn");
  if (button) {
    button.hidden = mode !== "tablet";
    button.innerHTML = `${icon("bets", { size: 20 })}<span class="slip-btn-text">Slip</span>${summary.count ? `<span class="slip-count" aria-hidden="true">${esc(summary.count)}</span>` : ""}${summary.odds ? `<span class="slip-btn-odds">${esc(summary.odds)}</span>` : ""}`;
    button.setAttribute("aria-label", `${summary.label}. Open the slip`);
  }
}

export function openSlip() {
  if (railShown()) {
    if (state.rail === "collapsed") setRailState("peek");
    else $("#rail-title").focus({ preventScroll: true });
    return;
  }
  hooks.openSlipSheet();
}

// --- start and breakpoints ----------------------------------------------------------------------------------

function buildPanes() {
  $("#tree-pane").innerHTML = `<div class="pane-head">
      <h2 class="pane-title" tabindex="-1">Sports</h2>
      <button type="button" class="icon-btn" data-tree-toggle aria-expanded="true" aria-controls="tree-slot" aria-label="Collapse the sports list">${icon("panel-left", { size: 20 })}</button>
    </div>
    <div id="tree-slot" class="pane-body tree-slot"></div>
    <div id="tree-rail" class="tree-rail">
      <button type="button" class="icon-btn" data-tree-toggle aria-expanded="false" aria-controls="tree-slot" aria-label="Show the sports list">${icon("panel-left", { size: 20 })}</button>
      <div class="rail-items"></div>
    </div>`;
  $("#inspector").innerHTML = `<div class="pane-head">
      <button type="button" class="icon-btn" data-insp="back" aria-label="Back" hidden>${icon("back", { size: 20 })}</button>
      <h2 class="pane-title" id="inspector-title" tabindex="-1">Details</h2>
      <a class="icon-btn" data-insp="full" data-no-inspector href="#/today" aria-label="Open as a full page" title="Open as a full page" hidden>${icon("external", { size: 18 })}</a>
      <button type="button" class="icon-btn" data-insp="close" aria-label="Close the details" title="Close" hidden>${icon("close", { size: 20 })}</button>
    </div>
    <div class="pane-body inspector-body"></div>`;
  $("#inspector").setAttribute("aria-labelledby", "inspector-title");
  $("#sliprail").innerHTML = `<button type="button" class="rail-tab" data-rail="peek" aria-controls="rail-panel" aria-expanded="false"></button>
    <div class="rail-panel" id="rail-panel" role="region" aria-labelledby="rail-title">
      <div class="pane-head">
        <h2 class="pane-title" id="rail-title" tabindex="-1">Your slip</h2>
        <button type="button" class="icon-btn" data-rail="pin" aria-label="Pin the slip open" title="Pin the slip open" hidden>${icon("pin", { size: 18 })}</button>
        <button type="button" class="icon-btn" data-rail="close" aria-label="Close the slip" title="Close" hidden>${icon("close", { size: 20 })}</button>
        <button type="button" class="icon-btn" data-rail="collapse" aria-label="Collapse the slip to a tab" title="Collapse">${icon("panel-right", { size: 20 })}</button>
      </div>
      <div class="pane-body rail-body"></div>
      <div class="rail-foot"></div>
    </div>`;
  $("#tree-btn").innerHTML = icon("menu", { size: 22 });
  drawInspectorEmpty();
}

function bindPanes() {
  $("#tree-pane").addEventListener("click", (event) => {
    if (event.target.closest("[data-tree-toggle]")) {
      setTreeCollapsed(!treeIsCollapsed());
      const focusTo = treeIsCollapsed() ? "#tree-rail [data-tree-toggle]" : "#tree-pane .pane-head [data-tree-toggle]";
      $(focusTo)?.focus({ preventScroll: true });
      return;
    }
    const item = event.target.closest("[data-rail-key]");
    if (item && rail.onPick) rail.onPick(item.dataset.railKey);
  });
  $("#tree-btn").addEventListener("click", () => openTreeDrawer());
  $("#inspector").addEventListener("click", (event) => {
    const act = event.target.closest("[data-insp]")?.dataset.insp;
    if (act === "close") closeInspector();
    else if (act === "back") inspectorBack();
    else if (act === "full") closeInspector({ focus: false }); // the link itself navigates
  });
  $("#inspector").addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !sheet.isOpen && insp.current) {
      event.preventDefault();
      closeInspector();
    }
  });
  $("#sliprail").addEventListener("click", (event) => {
    const act = event.target.closest("[data-rail]")?.dataset.rail;
    if (act === "peek") setRailState(state.rail === "collapsed" ? "peek" : "collapsed");
    else if (act === "pin") setRailState("pinned");
    else if (act === "close" || act === "collapse") setRailState("collapsed");
  });
  $("#sliprail").addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.rail === "peek" && !sheet.isOpen) {
      event.preventDefault();
      setRailState("collapsed");
    }
  });
  // A peeking rail closes when the person goes back to the page (not for clicks in a sheet it opened)
  document.addEventListener("pointerdown", (event) => {
    if (state.rail !== "peek" || !isDesk()) return;
    if (event.target.closest("#sliprail, .sheet-layer, #toasts, #gate")) return;
    setRailState("collapsed");
  });
  $("#slip-btn").addEventListener("click", () => openSlip());
  // Links to games and players open in the inspector when it's there
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest('a[href^="#/game/"], a[href^="#/player/"]');
    if (!link || link.hasAttribute("data-no-inspector") || link.closest(".sheet-layer")) return;
    if (openInspector(link.getAttribute("href"), { anchor: link.closest("[data-game]") || link })) event.preventDefault();
  });
}

function changed() {
  const previous = mode;
  mode = currentMode();
  if (mode === previous) return;
  document.documentElement.dataset.layout = mode;
  if (treeInDrawer && (isDesk() || (previous === "phone") !== (mode === "phone"))) sheet.close(); // the drawer's shape changed
  if (!isDesk() && state.rail === "peek") state.rail = "collapsed";
  placeTree();
  placeRail();
  placeInspector();
  emit("layout", mode, previous);
}

/** app.js, once at start: the router's and the slip's pieces (see ``hooks``). */
export function initLayout(given) {
  hooks = { ...hooks, ...given };
  document.documentElement.dataset.layout = mode;
  buildPanes();
  bindPanes();
  for (const query of Object.values(MEDIA)) query.addEventListener("change", changed);
  ROOMY.addEventListener("change", () => { // the default tree follows the width until the person chooses
    if (state.tree) return;
    placeTree();
    emit("layout", mode, mode);
  });
  hooks.drawRail($("#sliprail .rail-body"), $("#sliprail .rail-foot"));
  placeTree();
  placeRail();
  placeInspector();
}

/** app.js, on every route change, before the new screen draws. */
export function routeChanged() {
  placeInspector();
  placeTree();
  const before = $("#sliprail").hidden;
  placeRail();
  if (before !== $("#sliprail").hidden) emit("layout", mode, mode); // the rail came or went (Build)
}

/** The rail's body and foot (app.js refreshes the slip in them). */
export function railParts() {
  return { body: $("#sliprail .rail-body"), foot: $("#sliprail .rail-foot"), panel: $("#rail-panel") };
}
