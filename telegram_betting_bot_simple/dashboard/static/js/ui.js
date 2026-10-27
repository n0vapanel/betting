// Shared components. Most return HTML strings (escape every outside value with esc()); interactive
// ones (sheet, toast) work on the DOM. Clicks are handled once, by delegation (installUi):
//   [data-leg-key]  an odds button: toggles that pick in the slip and lights up every copy of it
//   [data-explain]  an ⓘ button: opens the glossary entry in a bottom sheet
// Numbers shown here are the server's: these helpers format them and never work out odds or chances.

import { icon } from "./icons.js";
import { haptic, openLink } from "./telegram.js";
import { slip, legKey, oddsFormat, legFromOffer, leagueTitle, sportEmoji, MAX_LEGS } from "./store.js";

// --- text and numbers --------------------------------------------------------------------------------

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

/** 64% (from 0.64); "<1%" and ">99%" at the ends; "–" when unknown. */
export function pct(p) {
  if (p === null || p === undefined || Number.isNaN(Number(p))) return "–";
  const value = Number(p);
  if (value > 0 && value < 0.005) return "<1%";
  if (value < 1 && value > 0.995) return ">99%";
  return `${Math.round(value * 100)}%`;
}

/** "1 in 6 (16%)" for a long shot (with the server's one_in), "62%" for something likely. */
export function chanceText(p, oneIn) {
  if (p === null || p === undefined) return "–";
  if (p >= 0.5 || !oneIn) return pct(p);
  return `1 in ${Number(oneIn).toLocaleString()} (${pct(p)})`;
}

/** "$46" or "$46.50" from the server's money strings ("46.00") or numbers. */
export function money(value, { cents = false } = {}) {
  if (value === null || value === undefined || value === "") return "–";
  const n = Number(value);
  if (!Number.isFinite(n)) return "–";
  const abs = Math.abs(n);
  const whole = Math.abs(abs - Math.round(abs)) < 0.005;
  const text = whole && !cents ? Math.round(abs).toLocaleString() : abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${n < -0.004 ? "−" : ""}$${text}`;
}

/** "+$4.20" / "−$1.30" (never colour alone: the sign is always written). */
export function signedMoney(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "–";
  return `${n > 0.004 ? "+" : ""}${money(n, { cents: true })}`;
}

/** A price in the person's format: 2.50 or +150. Takes {decimal, american} or a card offer {price, american}. */
export function priceText(odds) {
  if (!odds) return "–";
  if (oddsFormat() === "american" && odds.american) return odds.american;
  const decimal = Number(odds.decimal ?? odds.price);
  return Number.isFinite(decimal) ? decimal.toFixed(2) : "–";
}

/** "+6.5" for a spread, "O 224.5" / "U 224.5" for a total, "" for a moneyline. */
export function pointText(market, side, point) {
  if (point === null || point === undefined) return "";
  if (market === "spreads") return `${point > 0 ? "+" : ""}${point}`;
  if (market === "totals") return `${side === "over" ? "O" : "U"} ${point}`;
  return "";
}

const TWO_WORD = ["Trail Blazers", "Red Sox", "White Sox", "Blue Jays", "Maple Leafs", "Golden Knights", "Red Wings",
  "Blue Jackets", "Golden State Warriors"];
/** Team nickname for tight spaces ("Oklahoma City Thunder" -> "Thunder"); soccer and others keep their name. */
export function shortName(name, league = "") {
  const text = String(name ?? "");
  if (!/^(basketball_nba|basketball_wnba|americanfootball_nfl|baseball_mlb|icehockey_nhl)$/.test(league)) return text;
  const two = TWO_WORD.find((nick) => text.endsWith(` ${nick}`));
  if (two) return two === "Golden State Warriors" ? "Warriors" : two;
  const words = text.split(" ");
  return words.length > 1 ? words[words.length - 1] : text;
}

function sameDay(a, b) {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

function dayOffset(date) {
  const today = new Date();
  const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const then = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  return Math.round((then - start) / 86400000);
}

const clock = (d) => d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

/** "Tonight 7:30 PM", "Tomorrow 1:00 PM", "Wed 8:10 PM", "Oct 30, 8:10 PM" in local time. */
export function timeLabel(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const offset = dayOffset(d);
  if (offset === 0) return `${d.getHours() >= 17 ? "Tonight" : "Today"} ${clock(d)}`;
  if (offset === 1) return `Tomorrow ${clock(d)}`;
  if (offset > 1 && offset < 7) return `${d.toLocaleDateString([], { weekday: "short" })} ${clock(d)}`;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${clock(d)}`;
}

/** "Today", "Tomorrow", "Wednesday, Oct 28". */
export function dayHeading(iso) {
  const d = new Date(iso);
  const offset = dayOffset(d);
  if (offset === 0) return "Today";
  if (offset === 1) return "Tomorrow";
  if (offset === -1) return "Yesterday";
  return d.toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
}

/** "Oct 25" (short date) */
export function shortDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** "5 min ago", "3 h ago", "2 days ago" from an ISO time or epoch seconds. */
export function ago(when) {
  const ms = typeof when === "number" ? when * 1000 : new Date(when).getTime();
  if (!Number.isFinite(ms)) return "";
  const minutes = Math.max(0, Math.round((Date.now() - ms) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)} h ago`;
  return `${Math.round(minutes / 1440)} days ago`;
}

export { sameDay };

// --- small pieces ------------------------------------------------------------------------------------

/** A pill button: chip("NBA", {active: true, attrs: 'data-league="basketball_nba"'}). */
export function chip(label, { active = false, attrs = "", iconName = "", className = "", html = false } = {}) {
  return `<button type="button" class="chip${active ? " on" : ""}${className ? ` ${className}` : ""}" aria-pressed="${active}" ${attrs}>${iconName ? icon(iconName, { size: 16 }) : ""}${html ? label : esc(label)}</button>`;
}

/** A link styled as a chip (navigation, not a toggle). */
export function linkChip(label, href, { iconName = "", className = "" } = {}) {
  return `<a class="chip${className ? ` ${className}` : ""}" href="${esc(href)}">${iconName ? icon(iconName, { size: 16 }) : ""}${esc(label)}</a>`;
}

/**
 * A segmented control: segmented("mode", [{value: "single", label: "Single"}, ...], "multi").
 * Buttons carry data-seg="mode" data-value="single"; the screen listens for their clicks.
 */
export function segmented(name, options, value, { label = "" } = {}) {
  return `<div class="seg" role="group"${label ? ` aria-label="${esc(label)}"` : ""}>${options.map((o) =>
    `<button type="button" data-seg="${esc(name)}" data-value="${esc(o.value)}" aria-pressed="${String(o.value) === String(value)}">${esc(o.label)}</button>`).join("")}</div>`;
}

/** A status badge with an icon and words, e.g. badge("Out", "danger", "injury"). Kinds: info, good, warn, danger, muted. */
export function badge(text, kind = "muted", iconName = "") {
  return `<span class="badge ${kind}">${iconName ? icon(iconName, { size: 14 }) : ""}${esc(text)}</span>`;
}

/** ESPN injury status as a badge: Out / Doubtful in red, Questionable / Day-to-day in amber. */
export function injuryBadge(status) {
  if (!status) return "";
  const s = String(status).toLowerCase();
  if (s === "active") return "";
  const kind = /out|injured|il\b|-il|suspend|doubtful/.test(s) ? "danger" : /question|day|probable|gtd/.test(s) ? "warn" : "muted";
  return badge(status, kind, "injury");
}

/** A titled card section: section("Who wins", body, {info: "ai_chance"}). */
export function section(title, body, { info = "", id = "", aside = "", className = "" } = {}) {
  return `<section class="card section${className ? ` ${className}` : ""}"${id ? ` id="${esc(id)}"` : ""}>
    <div class="section-head"><h2>${esc(title)}${info ? explain(info) : ""}</h2>${aside}</div>
    ${body}
  </section>`;
}

/** A short note in a tinted box: notice("Odds are cached for 2 hours", "info"). Kinds: info, warn, danger, good. */
export function notice(text, kind = "info", { html = false } = {}) {
  const iconName = { info: "info", warn: "alert", danger: "alert", good: "check" }[kind] || "info";
  return `<div class="notice ${kind}" role="${kind === "danger" ? "alert" : "note"}">${icon(iconName, { size: 18 })}<div>${html ? text : esc(text)}</div></div>`;
}

// --- loading, empty and error states -------------------------------------------------------------------

/**
 * Placeholders while something loads: skeleton("cards", 3) | skeleton("lines", 4) | skeleton("chart") | skeleton("header").
 * They stay invisible for their first 300 ms (styles.css .skel-wrap), so a quick answer never flashes them;
 * on a repeat visit Today paints from the cache and shows none at all.
 */
export function skeleton(kind = "lines", count = 3) {
  const line = (w) => `<div class="skel skel-line" style="width:${w}%"></div>`;
  const wrap = (html, cls = "", label = "Loading") => `<div class="skel-wrap${cls ? ` ${cls}` : ""}" aria-busy="true" aria-label="${label}">${html}</div>`;
  if (kind === "cards") {
    return wrap(Array.from({ length: count }, () => `
      <div class="card skel-card">${line(38)}<div class="skel skel-bar"></div>
        <div class="skel-grid">${Array.from({ length: 6 }, () => '<div class="skel skel-odds"></div>').join("")}</div></div>`).join(""), "stack");
  }
  if (kind === "chart") return wrap('<div class="skel skel-chart"></div>', "", "Loading chart");
  if (kind === "header") {
    return wrap(`<div class="skel-header"><div class="skel skel-avatar"></div><div class="grow">${line(60)}${line(40)}</div></div>`);
  }
  return wrap(Array.from({ length: count }, (_, i) => line([92, 76, 84, 60, 70][i % 5])).join(""), "stack-sm");
}

/**
 * What to do when there's nothing to show:
 *   emptyState({iconName: "today", title: "No games right now", body: "Try another sport", actions: [{label, href} | {label, attrs, primary}]})
 */
export function emptyState({ iconName = "info", title = "", body = "", actions = [], html = false } = {}) {
  return `<div class="empty">
    <div class="empty-icon">${icon(iconName, { size: 28 })}</div>
    ${title ? `<h3>${esc(title)}</h3>` : ""}
    ${body ? `<p>${html ? body : esc(body)}</p>` : ""}
    ${actions.length ? `<div class="empty-actions">${actions.map(actionButton).join("")}</div>` : ""}
  </div>`;
}

function actionButton(a) {
  const cls = `btn ${a.primary ? "btn-primary" : "btn-secondary"}`;
  return a.href ? `<a class="${cls}" href="${esc(a.href)}">${esc(a.label)}</a>`
    : `<button type="button" class="${cls}" ${a.attrs || ""}>${esc(a.label)}</button>`;
}

/**
 * An error in place of a screen's content, with "Try again" (data-action="retry": app.js re-renders the
 * screen). Sign-in errors return "" because the sign-in form already covers the page.
 */
export function errorState(error, { retry = true, title = "" } = {}) {
  if (error?.signIn) return "";
  const message = error?.message || String(error || "Something went wrong.");
  const notBuilt = error?.status === 404;
  return emptyState({
    iconName: notBuilt ? "clock" : "alert",
    title: title || (notBuilt ? "Coming soon" : "Couldn't load this"),
    body: message,
    actions: retry && !notBuilt ? [{ label: "Try again", attrs: 'data-action="retry"' }] : [],
  });
}

// --- toast -------------------------------------------------------------------------------------------

/** A short message above the tab bar. kind: "info" | "good" | "warn" | "error". */
export function toast(text, { kind = "info", ms = 2600 } = {}) {
  let host = document.getElementById("toasts");
  if (!host) {
    host = document.createElement("div");
    host.id = "toasts";
    host.setAttribute("aria-live", "polite");
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  const iconName = { good: "check", warn: "alert", error: "alert" }[kind] || "info";
  el.innerHTML = `${icon(iconName, { size: 18 })}<span>${esc(text)}</span>`;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 200);
  }, ms);
}

// --- bottom sheet (one at a time; Back and Escape close it) ----------------------------------------------
//
// Sheets never stack. Opening one while another is open replaces it: the new one gets a fresh panel and
// state, so the old opener's handle reads closed (its .open is false, its close() does nothing) and the
// listeners it put on its panel go away with that panel. An explanation (an ⓘ) opened inside a sheet is
// different: sheet.info() shows it in place of the sheet's content, and ✕, Back or Escape bring the sheet
// back exactly as it was (a half-made call, the onboarding form), its handle still open throughout.

let current = null; // {root, id, panel, body, foot, titleEl, close, onClose, dismissible, pushed, returnFocus, peek}
let ignorePops = 0;
let sheetSeq = 0;
const sheetWatchers = new Set();
const notifySheet = () => sheetWatchers.forEach((fn) => { try { fn(Boolean(current)); } catch (e) { console.warn(e); } });

function setContent(el, content) {
  el.replaceChildren();
  if (content instanceof Node) el.appendChild(content);
  else el.innerHTML = content;
}

/** A sheet panel (head, body, foot) with no content yet. */
function makePanel(size, className) {
  const panel = document.createElement("div");
  panel.className = `sheet ${size}${className ? ` ${className}` : ""}`;
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.tabIndex = -1;
  panel.innerHTML = `<div class="sheet-grip" aria-hidden="true"></div>
    <div class="sheet-head"><h2 class="sheet-title"></h2>
      <button type="button" class="icon-btn sheet-close" aria-label="Close">${icon("close")}</button></div>
    <div class="sheet-body"></div>
    <div class="sheet-foot"></div>`;
  return panel;
}

/** The state for one sheet: its panel's parts plus how it was opened. */
function sheetState(root, panel, fields) {
  const state = {
    root, panel, peek: null,
    body: panel.querySelector(".sheet-body"),
    foot: panel.querySelector(".sheet-foot"),
    titleEl: panel.querySelector(".sheet-title"),
    close: panel.querySelector(".sheet-close"),
    ...fields,
  };
  state.close.addEventListener("click", dismiss);
  return state;
}

function fillSheet(state, { title = "", body = "", footer = "", label = "" }) {
  state.titleEl.textContent = title;
  state.panel.setAttribute("aria-label", label || title || "Details");
  state.close.hidden = !state.dismissible;
  setContent(state.body, body);
  setContent(state.foot, footer);
  state.foot.hidden = !state.foot.childNodes.length;
  state.body.scrollTop = 0;
}

// --- the page behind a dialog --------------------------------------------------------------------------
//
// While a sheet or the sign-in gate is open, the page behind it is `inert`: Tab and a screen reader stay in
// the dialog (aria-modal alone doesn't keep keyboard focus in). Each dialog holds the page under its own
// name, so closing a sheet while the gate is up keeps it inert.

const BEHIND = "#skip, .app-header, .desk, #slipbar, .tabbar"; // .desk holds the panes: tree, board (.shell), inspector, slip rail
const holders = new Map(); // who holds the page inert -> extra selector for that holder (the gate also covers sheets)

/** Make the page behind a dialog inert (on) or let it go (off). ``also`` adds a selector for this holder. */
export function holdPage(holder, on, also = "") {
  if (on) holders.set(holder, also);
  else holders.delete(holder);
  const selector = [BEHIND, ...holders.values()].filter(Boolean).join(", ");
  const wanted = new Set(holders.size ? document.querySelectorAll(selector) : []);
  for (const el of document.querySelectorAll("[data-held]")) {
    if (!wanted.has(el)) {
      el.inert = false;
      delete el.dataset.held;
    }
  }
  for (const el of wanted) {
    el.inert = true;
    el.dataset.held = "";
  }
}

function teardown() {
  const closing = current;
  current = null;
  closing.root.classList.remove("open");
  document.body.classList.remove("sheet-open");
  holdPage("sheet", false); // before focus goes back to the page
  const done = () => closing.root.remove();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) done();
  else setTimeout(done, 180);
  try {
    closing.onClose?.();
  } catch (e) {
    console.warn(e);
  }
  closing.returnFocus?.focus?.({ preventScroll: true });
  notifySheet();
  return closing;
}

function pushEntry(state) {
  try {
    history.pushState({ ...(history.state || {}), sheet: state.id }, "", location.href);
    state.pushed = true;
  } catch {
    state.pushed = false;
  }
}

/** Show an explanation in place of the open sheet's content (the content stays, hidden, untouched). */
function showPeek(title, html) {
  const c = current;
  if (!c.peek) {
    const el = document.createElement("div");
    el.className = "sheet-peek";
    c.peek = {
      el, title: c.titleEl.textContent, label: c.panel.getAttribute("aria-label"), scroll: c.body.scrollTop,
      footHidden: c.foot.hidden, closeHidden: c.close.hidden, returnFocus: document.activeElement,
    };
    c.body.hidden = true;
    c.foot.hidden = true;
    c.close.hidden = false;
    c.body.after(el);
    el.addEventListener("click", (event) => {
      if (event.target.closest("[data-peek-back]")) hidePeek();
    });
  }
  const back = c.peek.title || "the previous screen";
  c.peek.el.innerHTML = `${html}
    <button type="button" class="btn btn-secondary block peek-back" data-peek-back>${icon("back", { size: 18 })}Back to ${esc(back)}</button>`;
  c.peek.el.scrollTop = 0;
  c.titleEl.textContent = title;
  c.panel.setAttribute("aria-label", title);
  c.close.setAttribute("aria-label", `Close and go back to ${back}`);
  c.panel.focus({ preventScroll: true });
}

/** Back from an explanation to the sheet it was opened in, as the person left it. */
function hidePeek() {
  const c = current;
  const p = c?.peek;
  if (!p) return;
  c.peek = null;
  p.el.remove();
  c.body.hidden = false;
  c.foot.hidden = p.footHidden;
  c.close.hidden = p.closeHidden;
  c.close.setAttribute("aria-label", "Close");
  c.titleEl.textContent = p.title;
  c.panel.setAttribute("aria-label", p.label || p.title || "Details");
  c.body.scrollTop = p.scroll;
  const focus = p.returnFocus?.isConnected && c.panel.contains(p.returnFocus) ? p.returnFocus : c.panel;
  focus.focus?.({ preventScroll: true });
}

/** ✕, the backdrop and Escape: leave an explanation first, then close the sheet if it may be closed. */
function dismiss() {
  if (!current) return;
  if (current.peek) hidePeek();
  else if (current.dismissible) sheet.close();
}

window.addEventListener("popstate", () => {
  if (ignorePops > 0) {
    ignorePops -= 1;
    return;
  }
  if (!current || history.state?.sheet === current.id) return;
  if (current.peek && current.pushed) {
    // Back while an explanation shows: back to the sheet under it, which takes its history entry again
    hidePeek();
    pushEntry(current);
    return;
  }
  // Back pressed while a sheet is open: the sheet's history entry is gone, so just close it
  teardown();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && current) dismiss();
});

export const sheet = {
  /**
   * Open a bottom sheet (replacing any open one: sheets never stack).
   *   const s = sheet.open({ title, body: html | Node, footer: html | Node, onClose, dismissible: true, size: "auto" | "tall" })
   *   s.body / s.foot are the elements to fill or listen on; s.close() closes it; s.open turns false
   *   once it's closed or replaced by another sheet.
   * It has a visible close button; Escape, the backdrop, the browser's Back and Telegram's Back close it.
   * For a short explanation use sheet.info(): inside an open sheet it doesn't replace it.
   */
  open({ title = "", body = "", footer = "", onClose = null, dismissible = true, size = "auto", label = "", className = "" } = {}) {
    if (current) {
      // Replace the open sheet with a fresh panel and state (the layer and its history entry stay)
      const old = current;
      try {
        old.onClose?.();
      } catch (e) {
        console.warn(e);
      }
      const panel = makePanel(size, className);
      old.panel.replaceWith(panel);
      current = sheetState(old.root, panel, { id: old.id, pushed: old.pushed, returnFocus: old.returnFocus, onClose, dismissible });
      fillSheet(current, { title, body, footer, label });
      if (old.root.classList.contains("open")) panel.focus({ preventScroll: true });
      notifySheet();
      return handle();
    }
    const root = document.createElement("div");
    root.className = "sheet-layer";
    root.innerHTML = '<div class="sheet-scrim" data-sheet-scrim></div>';
    const panel = makePanel(size, className);
    root.appendChild(panel);
    document.body.appendChild(root);
    current = sheetState(root, panel, { id: `s${++sheetSeq}`, onClose, dismissible, returnFocus: document.activeElement, pushed: false });
    fillSheet(current, { title, body, footer, label });
    root.querySelector("[data-sheet-scrim]").addEventListener("click", dismiss);
    // A history entry of its own, so the browser's (or Android's) Back closes the sheet, not the page
    pushEntry(current);
    document.body.classList.add("sheet-open");
    holdPage("sheet", true);
    requestAnimationFrame(() => {
      root.classList.add("open");
      current?.panel.focus({ preventScroll: true }); // the dialog itself, so screen readers start at its title
    });
    notifySheet();
    return handle();
  },

  /**
   * A short explanation (the ⓘ sheets). With a sheet open it shows inside that sheet instead of replacing
   * it, with "Back to <sheet>"; ✕, Back and Escape return to the sheet as it was. Otherwise it opens as
   * a sheet of its own.
   */
  info({ title = "", body = "" } = {}) {
    if (!current) {
      sheet.open({ title, body });
      return;
    }
    showPeek(title, body);
  },

  /** Close the open sheet (explanation included); resolves once its history entry is gone (so navigating after it is safe). */
  close() {
    if (!current) return Promise.resolve();
    const closing = teardown();
    if (closing.pushed && history.state?.sheet === closing.id) {
      return new Promise((resolve) => {
        ignorePops += 1;
        const finish = () => {
          window.removeEventListener("popstate", finish);
          clearTimeout(timer);
          resolve();
        };
        const timer = setTimeout(() => {
          ignorePops = Math.max(0, ignorePops - 1);
          finish();
        }, 400);
        window.addEventListener("popstate", finish);
        history.back();
      });
    }
    return Promise.resolve();
  },

  /** One step back (Telegram's Back button): from an explanation to its sheet, else close the sheet. */
  back() {
    if (current?.peek) {
      hidePeek();
      return Promise.resolve();
    }
    return sheet.close();
  },

  get isOpen() {
    return Boolean(current);
  },

  /**
   * Change the open sheet's title, body or footer: only the parts given. The body keeps its scroll
   * position (opening a sheet starts at the top; refreshing one shouldn't jump there).
   */
  update({ title, body, footer, label } = {}) {
    const c = current;
    if (!c) return;
    if (title !== undefined || label !== undefined) {
      const text = title !== undefined ? title : c.peek ? c.peek.title : c.titleEl.textContent;
      if (c.peek) { // an explanation is showing: the sheet's own title comes back with it
        c.peek.title = text;
        c.peek.label = label || text;
      } else {
        c.titleEl.textContent = text;
        c.panel.setAttribute("aria-label", label || text || "Details");
      }
    }
    if (body !== undefined) {
      const top = c.body.scrollTop;
      setContent(c.body, body);
      if (!c.peek) c.body.scrollTop = top;
    }
    if (footer !== undefined) {
      setContent(c.foot, footer);
      const empty = !c.foot.childNodes.length;
      if (c.peek) c.peek.footHidden = empty;
      else c.foot.hidden = empty;
    }
  },

  /** Called with true/false whenever a sheet opens or closes (app.js keeps Telegram's buttons in step). */
  onToggle(fn) {
    sheetWatchers.add(fn);
    return () => sheetWatchers.delete(fn);
  },
};

function handle() {
  const mine = current;
  return { body: mine.body, foot: mine.foot, panel: mine.panel, close: () => (current === mine ? sheet.close() : Promise.resolve()),
    get open() { return current === mine; } };
}

/** A yes/no question in a sheet; resolves true when confirmed. */
export function confirmSheet({ title, body, confirm = "Confirm", cancel = "Cancel", danger = false }) {
  return new Promise((resolve) => {
    let answered = false;
    const s = sheet.open({
      title, body: `<p class="lead">${esc(body)}</p>`,
      footer: `<button type="button" class="btn btn-secondary grow" data-answer="no">${esc(cancel)}</button>
        <button type="button" class="btn ${danger ? "btn-danger" : "btn-primary"} grow" data-answer="yes">${esc(confirm)}</button>`,
      onClose: () => { if (!answered) resolve(false); },
    });
    s.foot.addEventListener("click", (event) => {
      const button = event.target.closest("[data-answer]");
      if (!button) return;
      answered = true;
      resolve(button.dataset.answer === "yes");
      s.close();
    });
  });
}

// --- glossary and ⓘ explainers ------------------------------------------------------------------------

/**
 * The shared glossary: every ⓘ button and the Help screen read it. Each entry:
 * {title, short (one line), body, example, see?: "#/route" ("See it live"), live?: (ctx) => string|null}
 * live() may use ctx.priced (the slip's last server pricing) to give an example from the person's own slip.
 */
export const GLOSSARY = {
  decimal_odds: {
    title: "Decimal odds",
    short: "What each $1 staked returns if it wins, stake included.",
    body: "Multiply your stake by the odds to get the payout. Profit is the payout minus the stake.",
    example: "$10 at 2.50 returns $25: your $10 back plus $15 profit.",
    see: "#/today",
  },
  american_odds: {
    title: "American odds",
    short: "+150 means a $100 bet wins $150; −150 means you stake $150 to win $100.",
    body: "Plus numbers are underdogs, minus numbers favourites. Switch between American and decimal in Settings.",
    example: "+150 is the same price as 2.50 decimal; −150 is 1.67.",
    see: "#/settings",
  },
  market_chance: {
    title: "Market chance",
    short: "The bookmakers' chance with their margin taken out.",
    body: "Prices across bookmakers are averaged and the bookmaker's cut removed, so both sides add up to 100%. It's the hardest view to beat: the closing market picks about 68% of NBA winners.",
    example: "Prices of 1.91 on both sides imply 52% each (about 105% in all); without the margin it's 50% each.",
    see: "#/today",
  },
  markets_say: {
    title: "Markets say (the blend)",
    short: "One chance per outcome, blended from every source that prices the game: the app's best estimate.",
    body: "Sharp bookmakers and exchanges, the US bookmakers' median, Kalshi and Polymarket, ESPN's predictor and ratings, and the app's own model each give a chance with any margin removed. They're averaged in log-odds with starting weights of 50% sharp, 20% US books, 20% prediction markets, 5% ratings and 5% model, shared out over the sources a game has. The market usually wins, so the aim is to match it and explain it, not to beat it; the weights are to be refitted per sport once the graded record is big enough.",
    example: "Sharp 64%, US books 62%, Kalshi 63%: the blend says about 63%.",
    see: "#/today",
  },
  sharp_book: {
    title: "Sharp book",
    short: "A bookmaker or exchange whose prices follow informed money: Pinnacle, Betfair, Matchbook.",
    body: "They take big bets from professionals on thin margins, so their prices are the best single guide to the real chance. The app uses them only to anchor the blend (half its starting weight). Their prices are never offered to you, because most people can't bet there: best prices and every ticket come from bookmakers you can use.",
    example: "Pinnacle 1.65 and 2.38 imply 60.6% + 42.0% = 102.6%; without the margin, 59% and 41%.",
    see: "#/today",
  },
  prediction_market: {
    title: "Prediction market",
    short: "An exchange where people trade yes/no contracts on an outcome: Kalshi and Polymarket.",
    body: "A contract pays $1 if it happens, so its price reads as the crowd's chance. The app takes the middle of the best bid and ask, scales a game's outcomes to add up to 100% and links to the market page. They share 20% of the blend's starting weight and count half in the last 15 minutes before the start, when their prices drift. Multiplying their leg prices never gives a real parlay price.",
    example: "Best bid 61¢, best ask 65¢: the mid is 63¢, a 63% chance.",
    see: "#/today",
  },
  thin_market: {
    title: "Thin market",
    short: "A market with too little money waiting, or too wide a gap between buyers and sellers, to trust.",
    body: "A prediction market counts as thin when its bid-ask spread is over 6 cents or under $1,000 is waiting to trade. Its chip says \"thin\" and the blend leaves it out: one small order can move such a price a long way.",
    example: "Bid 3¢, ask 97¢ with $22 waiting: a last trade at 50¢ says almost nothing.",
    see: "#/today",
  },
  disagreement: {
    title: "Disagreement (⚖)",
    short: "The sources counted in the blend are 5 points or more apart on some outcome.",
    body: "The badge shows how far apart the highest and lowest counted sources are, in percentage points. It often means news some prices haven't caught (a late injury), a stale rating or a slow market. It's a reason to look closer, not a signal to bet.",
    example: "Sharp 64% and Kalshi 57% on the same team: 7 points apart, so the game shows ⚖ 7.",
    see: "#/today",
  },
  ai_chance: {
    title: "AI chance",
    short: "The brain's own probability for an outcome, computed in code.",
    body: "For the NBA the model blends the bookmakers' consensus, ESPN's matchup predictor and a team model (form, rest, injuries). Other sports have no model yet, so their screens show the market's chance and say so. It's an estimate: no model has been shown to beat the market reliably.",
    example: "\"AI 64%\" on a team means the model expects it to win about 64 times in 100 such games.",
    see: "#/today",
  },
  fair_odds: {
    title: "Fair odds",
    short: "1 ÷ the chance: the price at which a bet would break even over time.",
    body: "When a bookmaker's price is below the fair odds, the bet loses money on average. Fair odds by the AI's chance and by the market's can differ.",
    example: "A 40% chance has fair odds of 2.50. A price of 2.30 is below fair.",
  },
  book_cut: {
    title: "What the bookmaker keeps",
    short: "The margin built into the prices, as a share of the money staked.",
    body: "On a typical spread bet at 1.91 a side the bookmaker keeps about 4.5% of stakes on average. In a multi the cut is paid again on every leg, so it grows with each pick.",
    example: "5 legs at a 4.5% cut each: the bookmaker keeps about 21%.",
    live: ({ priced }) => (priced?.book_cut != null
      ? `On your slip the bookmaker keeps about ${pct(priced.book_cut)} of the stake on average.` : null),
    see: "#/build",
  },
  expected_result: {
    title: "Expected result",
    short: "What a bet returns on average, in money, if it could be placed many times.",
    body: "Negative means an expected loss. At normal prices almost every bet has one: that's the bookmaker's cut. It's shown by the AI's chances and by the market's.",
    example: "$10 on a 50% chance at 1.91 returns $9.55 on average: an expected result of −$0.45.",
    live: ({ priced }) => {
      const e = priced?.expected;
      if (!e || (e.ai == null && e.market == null)) return null;
      const parts = [e.ai != null ? `${signedMoney(e.ai)} by the AI's chances` : "", e.market != null ? `${signedMoney(e.market)} by the market's` : ""].filter(Boolean);
      return `Your slip's expected result: ${parts.join(", ")}.`;
    },
  },
  one_in: {
    title: "\"1 in X\"",
    short: "A chance written as a count, which is easier to feel for long shots.",
    body: "1 in 6 is about 17%. A ticket that wins 1 in 50 times loses the other 49.",
    example: "A 4-leg multi of coin flips wins about 1 in 16 times (6%).",
    live: ({ priced }) => (priced?.chance?.ai_one_in || priced?.chance?.market_one_in
      ? `Your slip: ${priced.chance.ai_one_in ? `AI 1 in ${priced.chance.ai_one_in}` : ""}${priced.chance.ai_one_in && priced.chance.market_one_in ? " · " : ""}${priced.chance.market_one_in ? `market 1 in ${priced.chance.market_one_in}` : ""}.` : null),
  },
  single: {
    title: "Singles",
    short: "Each pick is its own bet, with its own stake.",
    body: "One pick losing doesn't sink the others. The total cost is the stake times the number of picks.",
    example: "3 singles at $10 cost $30.",
    see: "#/build",
  },
  multi: {
    title: "Multi (parlay)",
    short: "One bet on several picks: every pick has to win.",
    body: "The odds multiply, so the payout grows fast, and so does the chance of losing. Each leg adds the bookmaker's cut again.",
    example: "3 legs at 1.91 each = 6.97 odds; three coin flips all win about 1 in 8 times.",
    live: ({ priced }) => {
      if (!priced || priced.type !== "multi" || !priced.odds) return null;
      const legs = priced.legs.map((l) => Number(l.odds?.decimal).toFixed(2)).join(" × ");
      const chance = priced.chance?.ai ?? priced.chance?.market;
      return `Your ${priced.legs.length} legs: ${legs} = ${Number(priced.odds.decimal).toFixed(2)}${chance != null ? `; chance ${pct(chance)}` : ""}.`;
    },
    see: "#/build",
  },
  round_robin: {
    title: "Round robin",
    short: "Every combination of k picks from your slip, each its own small multi.",
    body: "Some picks can lose and you still get something back, but each combination is a separate stake, so it costs more.",
    example: "4 picks \"by 2s\" make 6 doubles: $5 each costs $30.",
    see: "#/build",
  },
  system: {
    title: "System bets",
    short: "Named sets of combinations, e.g. Trixie, Yankee, Lucky 15.",
    body: "Trixie: 3 picks, 3 doubles and a treble (4 bets). Patent adds the 3 singles (7). Yankee: 4 picks, 11 bets. Lucky 15 adds the singles (15). The stake is per bet.",
    example: "A $1 Lucky 15 costs $15.",
    see: "#/build",
  },
  moneyline: {
    title: "Winner (moneyline)",
    short: "Pick who wins the game.",
    body: "In soccer there's also a draw. Some sports settle on the result after extra time; the bookmaker's rules decide.",
    example: "Thunder to win at 1.48: $10 returns $14.80 if they win.",
  },
  spread: {
    title: "Spread",
    short: "A head start: the favourite must win by more than the line.",
    body: "−6.5 means the team must win by 7 or more. +6.5 means the team covers if it loses by 6 or fewer, or wins.",
    example: "Thunder −6.5 wins if they win 110–100, loses if they win 105–100.",
  },
  total: {
    title: "Total (over/under)",
    short: "Both teams' points together, over or under the line.",
    body: "A whole-number line can land exactly on the total: that's a push and the stake comes back.",
    example: "Over 224.5 wins on a 115–112 final (227).",
  },
  push: {
    title: "Push",
    short: "The result lands exactly on the line, so the stake comes back.",
    body: "In a multi a pushed leg drops out and the payout shrinks.",
    example: "Over 7 on a 4–3 game is a push.",
  },
  line: {
    title: "Line",
    short: "The number a stat or score has to beat.",
    body: "For a player prop, \"over 24.5 points\" wins with 25 or more. Lines move as money comes in and news breaks.",
    example: "If the line moves from 24.5 to 26.5 after you picked it, the app flags the change.",
  },
  hit_rate: {
    title: "Hit rate",
    short: "How often a player went over a line, with the sample size.",
    body: "\"7/10\" means 7 of the last 10 games went over. Small samples mislead, especially when a player's minutes change, and research finds recent hit rates add only a little on top of the line itself.",
    example: "Last 5: 4/5 · Last 10: 6/10 · Season: 12/20.",
  },
  projection: {
    title: "AI projection",
    short: "The NBA model's expected stat line for today's game.",
    body: "It uses minutes, usage, pace and injuries. Low–high is the 10%–90% range: about one game in five lands outside it. It's made on game day only.",
    example: "Mean 24.1 points, range 15–33, 55% over 23.5.",
  },
  rollover: {
    title: "Rollover",
    short: "Bet everything each day on new picks, rolling the winnings forward. One loss ends it.",
    body: "Chances multiply day after day, so long rollovers almost always fail. The planner shows safer ways (bank half, pot-and-ladder, flat stakes) side by side.",
    example: "A 10-day rollover at 2.00 finishes about 1 in 1,900 times by the market's chances; most runs end on day 1.",
    see: "#/build/rollover",
  },
  bankroll: {
    title: "Bankroll",
    short: "Money set aside for betting: money you can afford to lose.",
    body: "Keeping stakes a small share of it (say 1–5%) makes a losing run survivable. Set a weekly budget in Settings.",
    example: "A $200 bankroll with 2% stakes means $4 a bet.",
    see: "#/settings",
  },
  stake: {
    title: "Stake",
    short: "How much you bet.",
    body: "For round robins and systems the stake is per bet; the total cost is stake × number of bets.",
    example: "$2 on a Yankee (11 bets) costs $22.",
  },
  payout: {
    title: "Payout",
    short: "What comes back if the bet wins, your stake included.",
    body: "Profit is the payout minus the stake.",
    example: "$10 at 4.60 pays $46: $36 profit.",
  },
  most_likely: {
    title: "Most likely outcome",
    short: "The single result with the biggest chance.",
    body: "For most multis it's losing the whole stake, even when the payout looks tempting.",
    example: "\"Lose $10 (84%)\".",
  },
  value_mark: {
    title: "▲ and ! on prices",
    short: "▲: the AI rates the pick a little above what its price implies. !: the AI is far from the bookmakers.",
    body: "▲ shows when the AI's chance is 3 to 8 points above the market's and, by the AI's numbers, the price pays more than it costs on average. It's an opinion, not a guarantee: the bookmakers' prices are hard to beat, and these picks still lose often. ! shows when the gap is 8 points or more, which usually means news the model hasn't caught (an injury, a rested star): check it before trusting the AI. Only NBA games have the AI's chance, so only they get marks.",
    example: "\"AI 55% ▲\" on a price of 2.00 with the market at 48%: the AI expects a small profit on average, and the pick still loses about 45 times in 100.",
    see: "#/today",
  },
  analysts: {
    title: "AI analysts",
    short: "Claude and DeepSeek give their own forecast for a game.",
    body: "They read the game's odds, the AI's numbers, injuries and news. Their forecasts are graded after each game, and they only move the AI's numbers once their record shows they help (at least 40 graded games). Each call costs a little on the server's API account.",
    example: "\"Claude: Thunder 66%\" next to \"AI 64%\".",
  },
  tiers: {
    title: "Safe · Risky · Crazy",
    short: "Ticket tiers by chance, not promises.",
    body: "Safe tickets win about 60% or more of the time by the AI's numbers and still lose the rest. Risky are near coin flips with a real payout; Crazy are long shots. Nothing here is safe money.",
    example: "A Safe ticket at 62% still loses about 4 times in 10.",
    see: "#/build",
  },
  same_game: {
    title: "Same-game picks",
    short: "Picks from one game are linked, so they can't share a multi here.",
    body: "A team winning big makes the over likelier, so multiplying their chances as if unrelated would overstate the chance. Play them as singles instead.",
    example: "Thunder −6.5 and Over 224.5 in one multi: not supported.",
  },
  quota: {
    title: "Odds requests left",
    short: "The odds feed allows about 500 requests a month.",
    body: "Each league's odds cost 3 requests to fetch, so they're kept for up to 2 hours and only the sports you pick are fetched. When few are left the app shows the last saved odds.",
    example: "Keeping NBA and NFL odds fresh for 12 hours uses about 36 requests.",
  },
  brier: {
    title: "Brier score",
    short: "The average squared miss of a win chance: lower is better.",
    body: "Always saying 50% scores 0.250. The closing market scores about 0.20 on the NBA; a sound model lands at 0.20–0.21.",
    example: "Saying 70% for a team that wins misses by 0.3: 0.09 for that game.",
    see: "#/bets",
  },
  calls: {
    title: "Your calls",
    short: "Predict a game and get scored against the AI and the bookmakers.",
    body: "Winner calls score 25 − 100 × (1 − p)², p being the chance you gave the team that won. 50% always scores 0; a wrong 90% call costs −56. Bolder calls cost more when they miss.",
    example: "A right 70% call scores +16; a wrong one −24.",
    see: "#/bets",
  },
};

let explainContext = () => ({});
/** app.js passes a function returning {priced} so explainers can use the person's own slip. */
export function setExplainContext(fn) {
  explainContext = fn;
}

/** An ⓘ button that opens a glossary entry: explain("book_cut"). */
export function explain(term, { label = "" } = {}) {
  const entry = GLOSSARY[term];
  if (!entry) return "";
  return `<button type="button" class="info-btn" data-explain="${esc(term)}" aria-label="${esc(label || `What is ${entry.title}?`)}">${icon("info", { size: 16 })}</button>`;
}

/** Open a glossary entry (one short sheet: definition, example, and the slip's own numbers). */
export function openExplainer(term) {
  const entry = GLOSSARY[term];
  if (!entry) return;
  let live = null;
  try {
    live = entry.live?.(explainContext() || {}) || null;
  } catch {
    live = null;
  }
  // Inside an open sheet (the slip, a call, onboarding) it shows in place and Back returns to that sheet
  sheet.info({
    title: entry.title,
    body: `<p class="lead">${esc(entry.short)}</p>
      <p>${esc(entry.body)}</p>
      <div class="example"><span class="eyebrow">Example</span>${esc(entry.example)}</div>
      ${live ? `<div class="example mine"><span class="eyebrow">Your slip</span>${esc(live)}</div>` : ""}
      <div class="explain-links">
        <a class="text-link" href="#/help/${esc(term)}">More in Help ${icon("chevron", { size: 16 })}</a>
        ${entry.see ? `<a class="text-link" href="${esc(entry.see)}">See it live ${icon("chevron", { size: 16 })}</a>` : ""}
      </div>`,
  });
}

// --- odds buttons and game cards --------------------------------------------------------------------------

const legRegistry = new Map(); // leg key -> the slip leg an odds button adds

/** What ▲ and ! on a price mean, in words (the button's label and the mark's tooltip). */
export const MARK_WORDS = {
  value: "The AI rates this pick above its price: an opinion, not a guarantee",
  far: "The AI is far from the bookmakers here: check the injury news",
};

/**
 * Register a slip leg for a price button drawn by a screen itself (Today's board from the slate); returns
 * the key to put on the button as data-leg-key="<key>" (with aria-pressed). The delegated click handler
 * then toggles that leg in the slip like any oddsButton(), and keeps every copy of the button lit.
 *   leg = {ref: {league, id (the odds event id), market: "h2h"|"spreads"|"totals", side, point}, label, game,
 *          start, league, price, american?, ai?, fair?}
 */
export function registerLeg(leg) {
  const key = legKey(leg.ref);
  legRegistry.set(key, leg);
  return key;
}

/**
 * A price button that toggles a pick in the slip:
 *   oddsButton(card, "spreads", "home")  (card from /api/today or /api/game)
 * Shows the line, the price in the person's format and the AI's chance (or the market's, marked "Mkt",
 * when the league has no model). A started game's buttons are disabled.
 */
export function oddsButton(card, market, side, { wide = false } = {}) {
  const offer = card.markets?.[market]?.[side];
  if (!offer || !offer.price) {
    return `<span class="odds empty${wide ? " wide" : ""}" aria-hidden="true">–</span>`;
  }
  const leg = legFromOffer(card, market, side, offer);
  const key = legKey(leg.ref);
  legRegistry.set(key, leg);
  const modelled = Boolean(card.prediction);
  const chance = modelled ? offer.ai : offer.fair;
  const point = pointText(market, side, offer.point);
  // Like with like: on a line that can push (a whole-number spread or total) the AI's chance is an outright
  // one, while the market's margin-free chance is a chance given no push, so the AI's is put on that footing
  // first (engine/tickets.py's ai_given_no_push). Without a push chance the two compare as they are.
  const noPush = 1 - (Number(offer.push) || 0);
  const aiGivenNoPush = offer.ai != null && noPush > 0 ? offer.ai / noPush : offer.ai;
  const edge = modelled && offer.fair != null && aiGivenNoPush != null ? aiGivenNoPush - offer.fair : 0;
  const value = modelled && !card.live && offer.ev != null && offer.ev > 0.03 && edge >= 0.03 && edge < 0.08;
  const far = modelled && !card.live && edge >= 0.08;
  const selected = slip.has(key);
  const name = offer.label || `${side} ${market}`;
  const chanceWords = chance != null ? `${modelled ? "AI" : "market"} chance ${pct(chance)}` : "";
  // The marks say in words what they mean (screen readers, and phones that have no hover for the title);
  // Today and Game explain them in their legend too (GLOSSARY.value_mark)
  const markWords = value ? MARK_WORDS.value : far ? MARK_WORDS.far : "";
  const mark = value ? `<span class="o-mark" title="${esc(MARK_WORDS.value)}" aria-hidden="true">▲</span>`
    : far ? `<span class="o-mark warn" title="${esc(MARK_WORDS.far)}" aria-hidden="true">!</span>` : "";
  return `<button type="button" class="odds${selected ? " on" : ""}${wide ? " wide" : ""}" data-leg-key="${esc(key)}" aria-pressed="${selected}"
      aria-label="${esc(`${name}, odds ${priceText(offer)}${chanceWords ? `, ${chanceWords}` : ""}${markWords ? `. ${markWords}` : ""}`)}"${card.live ? " disabled" : ""}>
    ${point ? `<span class="o-point">${esc(point)}</span>` : ""}
    <span class="o-price">${esc(priceText(offer))}</span>
    ${chance != null ? `<span class="o-chance">${modelled ? "AI" : "Mkt"} ${pct(chance)}${mark}</span>` : ""}
    <span class="o-check" aria-hidden="true">${icon("check", { size: 12 })}</span>
  </button>`;
}

/** Each team's chance to win: the AI's when the league has a model, else the market's (margin removed). */
export function winChances(card) {
  if (card.prediction) {
    return { away: 1 - card.prediction.home_win, home: card.prediction.home_win, draw: null, source: "AI" };
  }
  const h2h = card.markets?.h2h || {};
  if (h2h.home?.fair != null && h2h.away?.fair != null) {
    return { away: h2h.away.fair, home: h2h.home.fair, draw: h2h.draw?.fair ?? null, source: "Market" };
  }
  return null;
}

/** A two-sided bar: away chance on the left, home on the right; the favourite is green and named. */
export function splitBar(awayName, away, homeName, home, { source = "", draw = null } = {}) {
  if (away == null || home == null) return "";
  const favHome = home >= away;
  const drawPart = draw != null ? draw : 0;
  const total = away + home + drawPart || 1;
  const w = (p) => `${Math.max(2, (p / total) * 100).toFixed(1)}%`;
  // A team's name gives way (…) before its chance does; a draw gets its own line under the bar's middle,
  // so three outcomes fit a phone-width card
  return `<div class="split" role="img" aria-label="${esc(`${source ? `${source} chance: ` : ""}${awayName} ${pct(away)}${draw != null ? `, draw ${pct(draw)}` : ""}, ${homeName} ${pct(home)}`)}">
    <div class="split-labels"><span class="split-team${favHome ? "" : " fav"}"><span class="split-name">${esc(awayName)}</span> <b>${pct(away)}</b></span>
      ${source ? `<span class="split-source">${esc(source)}</span>` : ""}
      <span class="split-team home${favHome ? " fav" : ""}"><b>${pct(home)}</b> <span class="split-name">${esc(homeName)}</span></span></div>
    <div class="split-bar"><span class="${favHome ? "" : "fav"}" style="width:${w(away)}"></span>${draw != null ? `<span class="draw" style="width:${w(draw)}"></span>` : ""}<span class="${favHome ? "fav" : ""}" style="width:${w(home)}"></span></div>
    ${draw != null ? `<div class="split-draw">Draw <b>${pct(draw)}</b></div>` : ""}
  </div>`;
}

const MARKET_NAMES = { h2h: "Winner", spreads: "Spread", totals: "Total" };

/** The odds grid for a game: a row per team, a column per market (Winner · Spread · Total). */
export function oddsGrid(card, { header = true } = {}) {
  const markets = ["h2h", "spreads", "totals"].filter((m) => card.markets?.[m] && Object.keys(card.markets[m]).length);
  if (!markets.length) return `<p class="muted small">No prices for this game right now.</p>`;
  const league = card.league;
  const totalSide = { away: "over", home: "under" };
  const cell = (market, side) => {
    if (market === "totals") return oddsButton(card, "totals", totalSide[side]);
    return oddsButton(card, market, side);
  };
  const row = (side) => `<div class="og-team">${esc(shortName(card[side], league))}${side === "home" ? '<span class="og-home">home</span>' : ""}</div>
    ${markets.map((m) => cell(m, side)).join("")}`;
  const draw = card.markets?.h2h?.draw
    ? `<div class="og-team">Draw</div>${markets.map((m) => (m === "h2h" ? oddsButton(card, "h2h", "draw") : "<span></span>")).join("")}`
    : "";
  return `<div class="odds-grid cols-${markets.length}">
    ${header ? `<div></div>${markets.map((m) => `<div class="og-head">${MARKET_NAMES[m]}</div>`).join("")}` : ""}
    ${row("away")}${row("home")}${draw}
  </div>`;
}

/** The analysts' latest picks on a card, one chip each (they open the game's analysts section). */
export function analystChips(card) {
  const list = Array.isArray(card.analysts) ? card.analysts : [];
  return list.map((a) => {
    const home = a.home_win >= 0.5;
    const team = shortName(home ? card.home : card.away, card.league);
    return `<a class="chip analyst" href="#/game/${esc(card.league)}/${esc(card.id)}?section=analysts" title="${esc(a.label)}'s forecast">${icon("bot", { size: 16 })}${esc(a.label)}: ${esc(team)} ${pct(home ? a.home_win : 1 - a.home_win)}</a>`;
  }).join("");
}

/**
 * A game on the Today list: time and league, the win-chance bar, the odds grid (every price toggles a
 * slip pick), the analysts' chips, "Ask AI" and a link to the game page.
 */
export function gameCard(card, { showLeague = true } = {}) {
  const href = `#/game/${encodeURIComponent(card.league)}/${encodeURIComponent(card.id)}`;
  const chances = winChances(card);
  const p = card.prediction;
  const marketHome = p?.sources?.market?.home_win_prob;
  const caution = p && marketHome != null && Math.abs(p.home_win - marketHome) >= 0.08;
  const away = shortName(card.away, card.league);
  const home = shortName(card.home, card.league);
  const pick = p ? `<span class="badge good">${icon("bot", { size: 14 })}AI: ${esc(shortName(p.pick, card.league))} ${pct(p.pick_prob)}</span>` : "";
  const warn = caution ? badge("Far from market", "warn", "alert") : "";
  const ask = `#/ask?game=${encodeURIComponent(`${card.league}:${card.id}`)}&name=${encodeURIComponent(`${away} @ ${home}`)}`;
  return `<article class="card game-card" data-game="${esc(card.id)}">
    <a class="gc-link" href="${href}" aria-label="${esc(`${card.away} at ${card.home}, ${timeLabel(card.start)}: game details`)}">
      <div class="gc-top">
        <span class="gc-when">${card.live ? `<span class="live-dot" aria-hidden="true"></span><b class="live">LIVE</b>` : icon("clock", { size: 14 })}
          ${card.live ? "" : esc(timeLabel(card.start))}${showLeague ? ` · ${esc(sportEmoji(card.league))} ${esc(card.league_title || leagueTitle(card.league))}` : ""}</span>
        <span class="gc-badges">${warn}${pick}</span>
      </div>
      <div class="gc-teams"><span class="gc-team">${esc(away)}</span><span class="gc-at">@</span><span class="gc-team">${esc(home)}</span></div>
      ${chances ? splitBar(away, chances.away, home, chances.home, { source: chances.source === "AI" ? "AI chance" : "Market chance", draw: chances.draw }) : ""}
    </a>
    ${oddsGrid(card)}
    <div class="gc-foot">
      <div class="chip-row wrap">${analystChips(card)}<a class="chip ask" href="${esc(ask)}">${icon("ask", { size: 16 })}Ask AI</a></div>
      <a class="text-link" href="${href}">Details ${icon("chevron", { size: 16 })}</a>
    </div>
  </article>`;
}

// --- bar chart: last games against a line -----------------------------------------------------------

let chartSeq = 0;

/**
 * Last-N bars against a line, in plain SVG:
 *   barChart({ bars: [{value, label: "@IND", sub: "33m", title: "Oct 25 vs DET: 18"}], line: 24.5, name: "points" })
 * Oldest on the left. Over the line: solid green with the value on top; under: red and hatched; on the
 * line: grey. The dashed line carries its value. Bars without a value are left as gaps.
 */
export function barChart({ bars = [], line = null, name = "", height = 200 } = {}) {
  const id = `c${++chartSeq}`;
  const slot = 34;
  const left = 6;
  const right = 40;
  const top = 22;
  const bottom = 38;
  const width = left + right + slot * Math.max(bars.length, 1);
  const values = bars.map((b) => b.value).filter((v) => v != null);
  const max = Math.max(1, ...values, line ?? 0) * 1.12;
  const plot = height - top - bottom;
  const y = (v) => top + plot - (v / max) * plot;
  let over = 0;
  let under = 0;
  let level = 0;
  const parts = bars.map((b, i) => {
    const x = left + i * slot + slot * 0.19;
    const w = slot * 0.62;
    const cx = left + i * slot + slot / 2;
    // An opponent label wider than its slot ("@UTAH") is squeezed to fit rather than run into the next one
    const text = String(b.label || "");
    const squeeze = text.length * 6.4 + (text.startsWith("@") ? 3 : 0) > slot - 6 ? ` textLength="${slot - 6}" lengthAdjust="spacingAndGlyphs"` : "";
    const labels = `<text x="${cx}" y="${height - bottom + 15}" class="c-label" text-anchor="middle"${squeeze}>${esc(b.label || "")}</text>
      <text x="${cx}" y="${height - bottom + 29}" class="c-sub" text-anchor="middle">${esc(b.sub || "")}</text>`;
    if (b.value == null) return labels;
    const kind = line == null ? "neutral" : b.value > line ? "over" : b.value < line ? "under" : "level";
    if (kind === "over") over += 1;
    else if (kind === "under") under += 1;
    else if (kind === "level") level += 1;
    const h = Math.max(2, (b.value / max) * plot);
    const fill = kind === "under" ? `url(#${id}-hatch)` : "";
    return `<g class="c-bar ${kind}"><title>${esc(b.title || `${b.label}: ${b.value}`)}</title>
      <rect x="${x.toFixed(1)}" y="${(top + plot - h).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="3"${fill ? ` fill="${fill}"` : ""}/>
      <text x="${cx}" y="${(top + plot - h - 5).toFixed(1)}" class="c-value" text-anchor="middle">${esc(formatStat(b.value))}</text>
    </g>${labels}`;
  }).join("");
  const lineY = line != null ? y(line) : null;
  const summary = line != null
    ? `${over} of ${values.length} games over ${line}${level ? `, ${level} exactly on it` : ""}`
    : `${values.length} games`;
  return `<figure class="chart">
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-t" preserveAspectRatio="xMidYMid meet">
      <title id="${id}-t">${esc(`Last ${bars.length} games, ${name}${line != null ? ` against the line ${line}` : ""}: ${summary}`)}</title>
      <defs><pattern id="${id}-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" class="c-hatch-bg"/><line x1="0" y1="0" x2="0" y2="6" class="c-hatch-line"/></pattern></defs>
      <line x1="${left}" x2="${width - right + 4}" y1="${top + plot}" y2="${top + plot}" class="c-axis"/>
      ${parts}
      ${lineY != null ? `<line x1="${left}" x2="${width - right + 4}" y1="${lineY.toFixed(1)}" y2="${lineY.toFixed(1)}" class="c-line"/>
        <text x="${width - right + 7}" y="${(lineY + 4).toFixed(1)}" class="c-line-label">${esc(String(line))}</text>` : ""}
    </svg>
    <figcaption class="chart-key">${line != null ? `<span class="key over">Over the line</span><span class="key under">Under</span>` : ""}<span class="muted">${esc(summary)}</span></figcaption>
  </figure>`;
}

export function formatStat(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "–";
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

// --- grouped checkboxes (the sports sheet and onboarding) --------------------------------------------------

/**
 * Grouped checkboxes: an "All" box, a box per group that is checked, unchecked or mixed (some of its items
 * checked), and a box per item, with "Select all" and "Clear". Groups fold behind a button (closed unless
 * listed in ``open``). Native checkboxes, so the keyboard and screen readers get "mixed" for free.
 *   html = checkTree({ name: "sports", groups: [{key, label, items: [{key, label, tag}]}], checked: Set,
 *                      allLabel: "All sports", open: ["Soccer"] })
 *   const tree = bindCheckTree(container, { checked: Set, onChange(set) })   // after inserting html
 *   tree.checked -> Set of item keys; tree.set(keys) replaces it
 */
export function checkTree({ name, groups = [], checked = new Set(), allLabel = "All", open = [], unit = "leagues" } = {}) {
  const total = groups.reduce((n, g) => n + g.items.length, 0);
  const id = (i) => `ct-${String(name).replace(/[^a-z0-9_-]/gi, "")}-${i}`;
  return `<div class="ctree" data-ctree="${esc(name)}" data-unit="${esc(unit)}">
    <div class="ctree-tools">
      <button type="button" class="btn btn-secondary small" data-ctree-act="all">${icon("check", { size: 16 })}Select all</button>
      <button type="button" class="btn btn-secondary small" data-ctree-act="none">Clear</button>
      <span class="ctree-count muted small" aria-live="polite">${checked.size} of ${total} ${esc(unit)}</span>
    </div>
    <label class="ctree-row ctree-all"><input type="checkbox" data-ctree-box="all"> <span class="grow"><b>${esc(allLabel)}</b></span><span class="ctree-n">${total}</span></label>
    <ul class="ctree-groups">${groups.map((g, i) => {
      const isOpen = open.includes(g.key);
      return `<li class="ctree-group">
        <div class="ctree-grow">
          <label class="ctree-row"><input type="checkbox" data-ctree-box="group" data-group="${esc(g.key)}"> <span class="grow">${esc(g.label)}</span><span class="ctree-n">${g.items.length}</span></label>
          <button type="button" class="icon-btn ctree-fold" aria-expanded="${isOpen}" aria-controls="${id(i)}" aria-label="${esc(`${isOpen ? "Hide" : "Show"} ${g.label} ${unit}`)}" data-ctree-fold>${icon("down", { size: 18 })}</button>
        </div>
        <ul class="ctree-items" id="${id(i)}"${isOpen ? "" : " hidden"}>${g.items.map((it) => `<li><label class="ctree-row ctree-item">
          <input type="checkbox" data-ctree-box="item" data-group="${esc(g.key)}" value="${esc(it.key)}"${checked.has(it.key) ? " checked" : ""}>
          <span class="grow">${esc(it.label)}${it.tag ? ` <span class="tag">${esc(it.tag)}</span>` : ""}</span></label></li>`).join("")}</ul>
      </li>`;
    }).join("")}</ul>
  </div>`;
}

/** Wire a checkTree() up (see there). Every change ends in onChange(set) with the checked item keys. */
export function bindCheckTree(container, { checked = new Set(), onChange = () => {} } = {}) {
  const root = container.matches?.("[data-ctree]") ? container : container.querySelector("[data-ctree]");
  const state = { checked: new Set(checked) };
  const items = () => [...root.querySelectorAll('[data-ctree-box="item"]')];
  const sync = () => {
    const all = items();
    for (const box of all) box.checked = state.checked.has(box.value);
    const mark = (box, list) => {
      const on = list.filter((b) => b.checked).length;
      box.checked = list.length > 0 && on === list.length;
      box.indeterminate = on > 0 && on < list.length;
    };
    for (const box of root.querySelectorAll('[data-ctree-box="group"]')) {
      mark(box, all.filter((b) => b.dataset.group === box.dataset.group));
    }
    mark(root.querySelector('[data-ctree-box="all"]'), all);
    const count = root.querySelector(".ctree-count");
    if (count) count.textContent = `${all.filter((b) => b.checked).length} of ${all.length} ${root.dataset.unit || "leagues"}`;
  };
  const changed = () => {
    sync();
    onChange(new Set(state.checked));
  };
  root.addEventListener("change", (event) => {
    const box = event.target.closest("[data-ctree-box]");
    if (!box) return;
    const kind = box.dataset.ctreeBox;
    const keys = kind === "all" ? items().map((b) => b.value)
      : kind === "group" ? items().filter((b) => b.dataset.group === box.dataset.group).map((b) => b.value) : [box.value];
    for (const key of keys) {
      if (box.checked) state.checked.add(key);
      else state.checked.delete(key);
    }
    changed();
  });
  root.addEventListener("click", (event) => {
    const act = event.target.closest("[data-ctree-act]")?.dataset.ctreeAct;
    if (act) {
      state.checked = new Set(act === "all" ? items().map((b) => b.value) : []);
      changed();
      return;
    }
    const fold = event.target.closest("[data-ctree-fold]");
    if (fold) {
      const list = root.querySelector(`#${CSS.escape(fold.getAttribute("aria-controls"))}`);
      const open = fold.getAttribute("aria-expanded") !== "true";
      fold.setAttribute("aria-expanded", String(open));
      fold.setAttribute("aria-label", fold.getAttribute("aria-label").replace(/^(Show|Hide)/, open ? "Hide" : "Show"));
      if (list) list.hidden = !open;
    }
  });
  sync();
  return {
    get checked() {
      return new Set(state.checked);
    },
    set(keys) {
      state.checked = new Set(keys);
      sync();
    },
  };
}

// --- delegated clicks ---------------------------------------------------------------------------------

function syncOddsButtons() {
  for (const button of document.querySelectorAll("[data-leg-key]")) {
    const on = slip.has(button.dataset.legKey);
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", String(on));
  }
}

/** Wire up the shared click handlers once (app.js calls this at start). */
export function installUi() {
  slip.onChange(syncOddsButtons);
  document.addEventListener("click", (event) => {
    const odds = event.target.closest("[data-leg-key]");
    if (odds && !odds.disabled) {
      event.preventDefault();
      const leg = legRegistry.get(odds.dataset.legKey);
      if (!leg) return;
      const added = slip.toggle(leg);
      haptic.select();
      if (added === null) toast(`A slip holds up to ${MAX_LEGS} picks`, { kind: "warn" });
      return;
    }
    const info = event.target.closest("[data-explain]");
    if (info) {
      event.preventDefault();
      openExplainer(info.dataset.explain);
      return;
    }
    // Outside links open in Telegram's browser inside Telegram
    const link = event.target.closest('a[href^="http"]');
    if (link && link.target === "_blank") {
      event.preventDefault();
      openLink(link.href);
    }
  });
}
