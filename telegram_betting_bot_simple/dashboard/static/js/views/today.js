// Today: every game in every league the app can see, from one all-sports slate (GET /api/slate), filtered
// in the browser. "All sports" is the default; the person's sports are pinned first, never "only these".
//
// Data: cache.swr paints the slate saved on this device at once (Telegram's ready() follows that paint),
// then asks the server with If-None-Match: a 304 keeps the board and refreshes "updated N min ago"; a 200
// redraws it in place: only the rows that changed are drawn again (board.js patchItems), so the page never
// shrinks under the reader and the row being read stays where it was. While Today is open it checks again
// every 30 s when games are live, every minute on a game day and every 5 minutes otherwise (only while the
// page is visible). Refresh asks again at once and never asks for odds. A long board is drawn a screenful
// first, then the rest in slices of SLICE rows, one task each, so a tap is never held up for long.
//
// Odds: the slate carries each league's saved odds and never costs Odds API credits. A league "opened" (on
// its own, or with one or two others chosen by hand) whose odds are over two hours old or missing gets one
// POST /api/slate/odds per visit (the server loads the board in the background, at most once per league per
// 2 hours, never below the reserve); the slate is checked again shortly after so the prices appear. With
// every sport, a sport group or a bigger selection showing nothing is asked for (one tap must not spend the
// day's boards): rows say "Odds load when you open a league" and open that league. The person's pinned
// sports (All sports showing) are asked for too, but at most MAX_AUTO_ASKS boards per visit.
//
// Parts (views/today/): model.js (filter, sort, edges: no DOM), tree.js (league tree and presets),
// filters.js (toolbar), board.js (desktop table and phone cards), row.js and card.js (one game), styles.js.
// Screens around it: the shell's panes (mountTree, the inspector, the slip rail) are in js/layout.js.

import { isQuiet, qs, prefs, api } from "../api.js";
import {
  me, pinnedSports, setSlateLeagues, openOnboarding, markPainted, layout, onLayoutChange, mountTree, openTreeDrawer,
  closeTreeDrawer, setTreeRail, openDetail, inspectorRoute, onInspectorChange, cache, swr, SLATE_URL, SLATE_KEY, prefetch,
  gameUrl, canPrefetchGame, openSportsSheet, rerender, setQuota, warmView,
} from "../app.js";
import { esc, emptyState, errorState, notice, skeleton, sheet, ago, toast, explain } from "../ui.js";
import { icon } from "../icons.js";
import { ensureStyles } from "./today/styles.js";
import { prepare, view, dayStrip, dayKey, leaguesNeedingOdds, SORTS, STATUSES } from "./today/model.js";
import { createTree, railItems } from "./today/tree.js";
import { boardParts, frameHtml, patchItems, keyOf, bindBoard, syncHeader } from "./today/board.js";
import { toolbarHtml, daysHtml, statusHtml, chipsHtml } from "./today/filters.js";

const VIEW_KEY = "todayView"; // {leagues: [keys] | null, sort: {key, dir}}: this browser remembers the choice
const PRESETS_KEY = "todayPresets"; // [{name, leagues: [keys] | null, status}]
const ASKED_KEY = "sportsbet-odds-asked"; // sessionStorage: {league: {code, queued, at, auto}} (once per visit)
const MAX_PRESETS = 8;
const HAND_PICKED = 3; // a selection of this many leagues or fewer asks for their odds by itself
const MAX_AUTO_ASKS = 4; // pinned leagues' boards asked for by themselves per visit (a board costs 3 credits)
const SLICE = 25; // rows (or cards) appended per task after the first screenful
const POLL = { live: 30000, gameDay: 60000, idle: 300000 };
const QUICK = [5000, 12000, 25000, 45000]; // after odds were asked for: look for them sooner

// The board as the person left it this visit (Back to Today finds the same day, filter and search)
let memory = null;
const asked = loadAsked();

function loadAsked() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(ASKED_KEY) || "{}");
    return new Map(Object.entries(saved && typeof saved === "object" ? saved : {}));
  } catch {
    return new Map();
  }
}

function saveAsked() {
  try {
    sessionStorage.setItem(ASKED_KEY, JSON.stringify(Object.fromEntries(asked)));
  } catch {
    /* private mode: remembered for this page only */
  }
}

function savedView() {
  const saved = prefs.get(VIEW_KEY, null) || {};
  const leagues = Array.isArray(saved.leagues) ? saved.leagues.filter((k) => typeof k === "string" && /^[a-z0-9_]{2,64}$/.test(k)) : null;
  const sort = saved.sort && SORTS.includes(saved.sort.key) ? { key: saved.sort.key, dir: saved.sort.dir === -1 ? -1 : 1 } : { key: "time", dir: 1 };
  return { leagues: leagues ? new Set(leagues) : null, sort };
}

function savePresets(list) {
  prefs.set(PRESETS_KEY, list.slice(0, MAX_PRESETS));
}

function loadPresets() {
  const list = prefs.get(PRESETS_KEY, []);
  return Array.isArray(list) ? list.filter((p) => p && typeof p.name === "string").map((p) => ({
    name: p.name.slice(0, 24),
    leagues: Array.isArray(p.leagues) ? p.leagues.filter((k) => typeof k === "string") : null,
    status: STATUSES.includes(p.status) ? p.status : "all",
  })).slice(0, MAX_PRESETS) : [];
}

const sameSet = (a, b) => a.size === b.size && [...a].every((k) => b.has(k));

/**
 * Rows drawn before the first paint: about a screenful (a phone's cards are ~200 px tall), the rest after.
 * The screen's height, not the window's: reading innerHeight in the middle of drawing forces a layout.
 */
const firstChunk = (mode) => (mode === "phone" ? Math.max(4, Math.ceil(Math.min(screen?.height || 800, 1400) / 200) + 1) : 30);

/**
 * The columns the table shows at the board's width (today/styles.js: Context from 700 px, Notes from 820 px, League
 * and "Odds as of" from 1180 px; AI and Edge when a game has the AI's chance): a separator's or fold-out's
 * aria-colspan.
 */
const columnsAt = (width, ai) => (width >= 1180 ? 10 : width >= 820 ? 8 : width >= 700 ? 7 : 6) + (ai ? 2 : 0);

// What each redrawn part of the toolbar was last drawn with: an unchanged part isn't touched (focus stays)
const drawnHtml = new WeakMap();

/** A selector for the same toolbar control after a redraw. */
function controlKey(node) {
  for (const attr of ["data-day", "data-status", "data-chip", "data-td-act", "data-td-day"]) {
    const value = node.closest?.(`[${attr}]`)?.getAttribute(attr);
    if (value != null) return `[${attr}="${CSS.escape(value)}"]`;
  }
  return node.closest?.("[data-chip-tree]") ? "[data-chip-tree]" : null;
}

const focusLost = () => !document.activeElement || document.activeElement === document.body;

/** Set ``el``'s HTML when it changed; focus on a control inside goes to its new copy. */
function setHtml(el, html) {
  if (drawnHtml.get(el) === html) return;
  const key = el.contains(document.activeElement) ? controlKey(document.activeElement) : null;
  el.innerHTML = html;
  drawnHtml.set(el, html);
  if (key && focusLost()) el.querySelector(key)?.focus({ preventScroll: true });
}

/** Player search in a sheet (ESPN's search, any league; results open the player page or the inspector). */
function openSearch() {
  const s = sheet.open({
    title: "Find a player",
    size: "tall",
    body: `<div class="search-row"><label class="sr-only" for="player-q">Player name</label>
        <input id="player-q" class="input" type="search" placeholder="e.g. LeBron James" autocomplete="off" enterkeyhint="search"></div>
      <ul class="list search-results" data-results></ul>`,
  });
  const input = s.body.querySelector("#player-q");
  const results = s.body.querySelector("[data-results]");
  let timer = null;
  let controller = null;
  setTimeout(() => input.focus(), 50);
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) {
      results.innerHTML = "";
      return;
    }
    timer = setTimeout(async () => {
      controller?.abort();
      controller = new AbortController();
      results.innerHTML = `<li>${skeleton("lines", 3)}</li>`;
      try {
        const found = await api(`/api/players/search${qs({ q })}`, { signal: controller.signal });
        results.innerHTML = found.length ? found.map((p) => `<li><a class="list-item" href="#/player/${esc(p.league)}/${esc(p.id)}">
            ${p.headshot ? `<img class="headshot" style="width:40px;height:40px" src="${esc(p.headshot)}" alt="" loading="lazy">` : `<span class="headshot blank" style="width:40px;height:40px">${icon("user", { size: 18 })}</span>`}
            <span class="grow"><b>${esc(p.name)}</b><span>${esc([p.team, p.position].filter(Boolean).join(" · "))}</span></span>${icon("chevron", { size: 18 })}</a></li>`).join("")
          : `<li class="muted small">No players match “${esc(q)}”.</li>`;
      } catch (error) {
        if (!isQuiet(error)) results.innerHTML = `<li>${notice(error.message, "warn")}</li>`;
      }
    }, 300);
  });
}

function ageGate() {
  return emptyState({
    iconName: "shield", title: "Confirm your age first",
    body: "Games, prices and picks are for adults: confirm you're 18+ (21+ in the US) to continue.",
    actions: [{ label: "Confirm age", attrs: "data-onboard", primary: true }],
  });
}

/** A layout-matched placeholder (invisible for its first 300 ms: only a slow first visit ever shows it). */
function skeletonBoard(mode) {
  if (mode === "phone") return skeleton("cards", 3);
  const row = '<div class="td-skel-row"><div class="skel"></div><div class="skel tall"></div><div class="skel tall"></div><div class="skel tall"></div><div class="skel tall"></div><div class="skel"></div></div>';
  return `<div class="skel-wrap tb" aria-busy="true" aria-label="Loading games">${row.repeat(6)}</div>`;
}

export async function render(root, params) {
  ensureStyles();
  const signal = params.signal;
  const pinned = pinnedSports();
  const now = new Date();
  const strip = dayStrip(now);
  const saved = savedView();
  const kept = memory && strip.some((d) => d.key === memory.day) ? memory : null;
  // A link may choose the sort (#/today?sort=dis, from Help's "Sort Today by disagreement"); it is remembered
  const linkSort = params.query?.get?.("sort");
  const state = {
    leagues: saved.leagues,
    sort: SORTS.includes(linkSort) ? { key: linkSort, dir: 1 } : saved.sort,
    day: kept?.day || dayKey(now),
    status: kept?.status || "all",
    search: kept?.search || "",
    expanded: kept?.expanded || new Set(),
  };
  const remember = () => {
    memory = { day: state.day, status: state.status, search: state.search, expanded: state.expanded };
    prefs.set(VIEW_KEY, { leagues: state.leagues ? [...state.leagues] : null, sort: state.sort });
  };
  if (SORTS.includes(linkSort)) remember();
  let presets = loadPresets();

  if (!me().age_confirmed && me().uid) {
    root.innerHTML = `<div class="td-head"><div class="grow"><h1>Today</h1></div></div>${ageGate()}`;
    root.addEventListener("click", (event) => {
      if (event.target.closest("[data-onboard]")) openOnboarding();
    });
    return;
  }

  // The shell started reading this device's slate (cache.warm) while this module loaded: wait for that read
  // (cache.get gives up after a moment), so the toolbar and the first rows go out in one frame
  if (!cache.peek(SLATE_KEY)) await cache.get(SLATE_KEY);
  if (signal.aborted) return;

  const uid = Math.random().toString(36).slice(2, 8);
  root.innerHTML = `${toolbarHtml(state, { idSearch: `td-q-${uid}`, idSort: `td-s-${uid}` })}
    <div class="td-board" data-td-board></div>
    <div class="td-legend" data-td-legend></div>`;
  const $ = (sel) => root.querySelector(sel);
  const els = {
    sub: $("[data-td-sub]"), updated: $("[data-td-updated]"), chips: $("[data-td-chips]"), days: $("[data-td-days]"),
    status: $("[data-td-status]"), search: $("[data-td-search]"), sort: $("[data-td-sort]"), notes: $("[data-td-notes]"),
    board: $("[data-td-board]"), legend: $("[data-td-legend]"), bar: $("[data-td-bar]"), refresh: $("[data-td-refresh]"),
  };

  // A logo that doesn't load gives way to a blank of the same size (one listener for the whole board)
  els.board.addEventListener("error", (event) => {
    const img = event.target;
    if (!(img instanceof HTMLImageElement) || !img.classList.contains("td-logo")) return;
    const blank = document.createElement("span");
    blank.className = "td-logo blank";
    blank.setAttribute("aria-hidden", "true");
    img.replaceWith(blank);
  }, true);

  let shown = null; // the slate on screen
  let meta = null; // {source, at, checked, etag, status} from swr
  let painted = false;
  let failure = null; // the last refresh's error while a copy is on screen
  let prepared = null;
  let current = null; // the last view()
  const cleanups = [];
  signal.addEventListener("abort", () => cleanups.splice(0).forEach((fn) => { try { fn(); } catch { /* gone */ } }), { once: true });

  // --- the league tree (the shell decides where it sits) ---------------------------------------------
  const tree = createTree({
    onSelect(selection, { only }) {
      choose(selection);
      if (only && layout.mode !== "desk" && layout.mode !== "wide") closeTreeDrawer();
    },
    onPreset(key) {
      applyPreset(key);
      if (layout.mode !== "desk" && layout.mode !== "wide") closeTreeDrawer();
    },
    onSavePreset(name) {
      if (presets.length >= MAX_PRESETS) {
        toast(`You can keep ${MAX_PRESETS} views: delete one first`, { kind: "warn" });
        return;
      }
      presets = [...presets.filter((p) => p.name !== name), { name, leagues: state.leagues ? [...state.leagues] : null, status: state.status }];
      savePresets(presets);
      toast(`Saved “${name}”`, { kind: "good" });
      draw();
    },
    onDeletePreset(index) {
      presets = presets.filter((_, i) => i !== index);
      savePresets(presets);
      draw();
    },
    onPin() {
      openSportsSheet({ onDone: () => rerender() }); // pins change the order: Today draws again with them
    },
  });
  // Unmounted while there is nothing to filter (a break, the age gate, a first load that failed); onData mounts it again
  let treeHandle = mountTree(tree.el, { signal, title: "Sports" });
  const showFilters = (on) => {
    els.bar.hidden = !on;
    if (on && !treeHandle) treeHandle = mountTree(tree.el, { signal, title: "Sports" });
    if (!on && treeHandle) {
      treeHandle.unmount();
      treeHandle = null;
    }
  };

  function choose(selection, { status = null, day = null } = {}) {
    state.leagues = selection;
    if (status) state.status = status;
    if (day) state.day = day;
    state.expanded.clear();
    remember();
    draw({ moved: true });
  }

  function applyPreset(key) {
    if (!prepared) return;
    if (key === "all") choose(null, { status: "all" });
    else if (key === "live") choose(null, { status: "live", day: dayKey(new Date()) });
    else if (key === "mine") choose(normal(new Set(pinned)), { status: "all" });
    else if (key.startsWith("group:")) choose(normal(new Set(prepared.groups.find((g) => g.key === key.slice(6))?.leagues || [])), { status: "all" });
    else if (key.startsWith("saved:")) {
      const p = presets[Number(key.slice(6))];
      if (p) choose(p.leagues ? normal(new Set(p.leagues)) : null, { status: p.status || "all" });
    }
  }

  /** A selection of every league is "All sports" (null), so leagues added later are in it too. */
  function normal(set) {
    if (!prepared || !set) return set;
    return prepared.order.length && prepared.order.every((k) => set.has(k)) ? null : set;
  }

  function activePreset() {
    const sel = state.leagues;
    if (!sel && state.status === "all") return "all";
    if (!sel && state.status === "live") return "live";
    const i = presets.findIndex((p) => p.status === state.status && (p.leagues ? sel && sameSet(new Set(p.leagues), sel) : !sel));
    if (i >= 0) return `saved:${i}`;
    if (sel && state.status === "all" && pinned.length && sameSet(sel, new Set(pinned))) return "mine";
    const soccer = prepared?.groups.find((g) => g.key === "Soccer");
    if (sel && soccer && state.status === "all" && sameSet(sel, new Set(soccer.leagues))) return "group:Soccer";
    return null;
  }

  // --- drawing ------------------------------------------------------------------------------------------

  function updatedText() {
    if (!meta) return;
    const when = meta.checked || meta.at;
    const words = failure ? `Couldn't refresh · games from ${ago(new Date(meta.at).toISOString())}` : `Updated ${ago(new Date(when).toISOString())}`;
    els.updated.textContent = `· ${words}`;
    els.updated.classList.toggle("warn", Boolean(failure));
  }

  function selectionWords() {
    if (!state.leagues) return "All sports";
    if (!state.leagues.size) return "No leagues";
    const names = [...state.leagues].map((k) => prepared?.leagues.get(k)?.title || k);
    return names.length <= 3 ? names.join(", ") : `${names.length} leagues`;
  }

  function notesHtml() {
    const out = [];
    if (failure) {
      const offline = typeof navigator !== "undefined" && navigator.onLine === false;
      out.push(notice(offline ? "You're offline: these are the games saved on this device. They refresh when you're back online."
        : `Couldn't refresh the games (${failure.message || "network error"}). Showing the copy from ${ago(new Date(meta.at).toISOString())}.`, "warn"));
    }
    const loading = [...asked.entries()].filter(([, a]) => a && (a.code === "queued" || a.code === "loading" || a.code === "asking") && Date.now() - (a.at || 0) < 120000)
      .map(([k]) => prepared?.leagues.get(k)).filter((e) => e && e.odds !== "fresh").map((e) => e.title);
    if (loading.length) out.push(`<p class="td-ai-note">${icon("refresh", { size: 14, className: "spin" })} Loading the latest odds for ${esc(loading.join(", "))}…</p>`);
    return out.join("");
  }

  function legendHtml(ai) {
    return `<p class="note">“Markets” is one chance blended from the sources that price the game: sharp bookmakers and exchanges, US bookmakers, prediction markets and ratings, each with its margin removed ${explain("markets_say")}.
      ⚖ marks a game whose sources disagree by 5 points or more (the number is how far apart they are) ${explain("disagreement")}; “Low” means no market price can be counted (none, or only thin or out-of-date ones), so it rests on ratings alone, pulled a quarter of the way toward the sport's usual home-win rate.
      ${ai ? `“AI” is the model's chance (NBA for now) ${explain("ai_chance")}; “Edge” is AI minus the bookmakers' chance (“Mkt”, margin removed) in points: ▲ 3 or more, ! 8 or more, which usually means news the model hasn't caught.` : "The AI's own chances are NBA-only for now."}
      “Ctx” chips are the venue, weather, rest and travel (wind, cold, heat, rain, roof closed, altitude, a team that flew east, a short week): they explain the line, and “Adj” marks a game whose total moved for a forecast or roof change since the odds ${explain("explains_vs_moves", { label: "Explains vs moves the number" })}.
      Prices are the best at bookmakers you can bet with; tap one to add it to your slip.</p>`;
  }

  function emptyHtml(v) {
    const dayInfo = strip.find((d) => d.key === state.day);
    const dayWords = dayInfo ? (dayInfo.offset < 2 ? dayInfo.label.toLowerCase() : `on ${dayInfo.long}`) : "that day";
    if (state.leagues && !state.leagues.size) {
      return emptyState({ iconName: "today", title: "No leagues chosen", body: "Pick some sports in the list, or show them all.",
        actions: [{ label: "Show all sports", attrs: 'data-td-act="all"', primary: true }] });
    }
    if (v.searchAll) {
      return emptyState({ iconName: "search", title: `No teams match “${state.search.trim()}”`, body: "Search looks through every league chosen, today and the next six days.",
        actions: [{ label: "Clear the search", attrs: 'data-td-act="clear-search"', primary: true }, ...(state.leagues ? [{ label: "Search all sports", attrs: 'data-td-act="all"' }] : [])] });
    }
    if (v.total > 0) {
      const title = state.status === "live" ? "Nothing live right now"
        : `No games ${state.status === "upcoming" ? "still to start" : "finished yet"} ${dayWords}`;
      return emptyState({ iconName: state.status === "live" ? "clock" : "today", title,
        body: `${v.total} game${v.total === 1 ? "" : "s"} ${dayWords} in ${selectionWords()}.`,
        actions: [{ label: "Show every game", attrs: 'data-td-act="status-all"', primary: true }] });
    }
    const next = strip.find((d) => d.key > state.day && (v.days.get(d.key) || 0) > 0);
    const actions = [];
    if (next) actions.push({ label: `${next.label}: ${v.days.get(next.key)} game${v.days.get(next.key) === 1 ? "" : "s"}`, attrs: `data-td-day="${esc(next.key)}"`, primary: true });
    if (state.leagues) actions.push({ label: "All sports", attrs: 'data-td-act="all"', primary: !next });
    const none = !prepared.games.length;
    return emptyState({
      iconName: "today",
      title: none ? "No games in the next seven days" : `No ${selectionWords()} games ${dayWords}`,
      body: none ? "Every league the app follows is quiet for now. Check back later." : next ? "The next ones are a tap away." : "Nothing in the next seven days for these leagues.",
      actions,
    });
  }

  // The board on screen: {kind: "table" | "cards", body (the element its rows are in), head, ai, items (each
  // row's key and the HTML it was drawn with)}, or null (empty, an error, a skeleton)
  let drawn = null;
  let cols = null; // the columns the table shows at its width (a ResizeObserver keeps it), for aria-colspan
  let redrawWhenSettled = false;

  /**
   * Draw the board from ``current``. A board already on screen (same kind) is brought up to date in place:
   * unchanged rows keep their nodes, so the page never shrinks and focus stays; ``fresh`` (the filters
   * changed) draws it anew instead: a screenful at once, then the rest SLICE rows per task. Returns true when
   * it patched the board in place.
   */
  function drawBoard({ fresh = false } = {}) {
    const mode = layout.mode;
    const v = current;
    if (!v.games.length) {
      boardToken += 1; // a long board's rest, still on its way, is dropped
      drawn = null;
      redrawWhenSettled = false;
      els.board.innerHTML = `<div class="td-empty">${emptyHtml(v)}</div>`;
      setHtml(els.legend, "");
      delete els.board.dataset.settling;
      settle();
      return false;
    }
    const ai = v.games.some((g) => g.ai);
    const focused = els.board.contains(document.activeElement) ? focusKey(document.activeElement) : null;
    const refocus = () => { if (focused && focusLost()) els.board.querySelector(focused)?.focus({ preventScroll: true }); };
    const parts = boardParts({
      mode, games: v.games, prepared, state, ai, asked, pinned, inspected: inspectorRoute(),
      canPrefetch: (game, entry) => canPrefetchGame(game, entry), cols: cols === null ? undefined : columnsAt(cols, ai),
    });
    const kind = parts.table ? "table" : "cards";
    if (!fresh && drawn && drawn.kind === kind && drawn.body.isConnected) {
      if (els.board.hasAttribute("data-settling")) { // the rest of the last drawing is still going in: after it
        redrawWhenSettled = true;
        return false;
      }
      if (parts.table && drawn.head !== parts.head) els.board.querySelector(".tb-headwrap").innerHTML = parts.head;
      if (parts.table && drawn.ai !== ai) els.board.querySelector(".tb").classList.toggle("has-ai", ai);
      patchItems(drawn.body, parts.items, new Map(drawn.items.map((item) => [item.key, item.html])));
      drawn.items = parts.items;
      drawn.head = parts.head;
      drawn.ai = ai;
      setHtml(els.legend, legendHtml(ai));
      refocus();
      settle();
      return true;
    }
    const token = ++boardToken;
    redrawWhenSettled = false;
    const first = firstChunk(mode);
    const rest = parts.items.slice(first);
    els.board.innerHTML = frameHtml(parts, parts.items.slice(0, first).map((item) => item.html).join(""));
    setHtml(els.legend, legendHtml(ai));
    const body = els.board.querySelector(parts.body);
    drawn = { kind, body, head: parts.head, ai, items: parts.items };
    if (parts.table) syncHeader(els.board);
    if (!rest.length) {
      delete els.board.dataset.settling;
      refocus();
      settle();
      return false;
    }
    // A long board: the rest once the first rows have painted, a slice per task (a newer drawing stops it)
    els.board.dataset.settling = "";
    const step = (from) => {
      if (token !== boardToken || signal.aborted) return;
      body.insertAdjacentHTML("beforeend", rest.slice(from, from + SLICE).map((item) => item.html).join(""));
      if (from + SLICE < rest.length) {
        setTimeout(() => step(from + SLICE), 0);
        return;
      }
      delete els.board.dataset.settling;
      refocus();
      settle();
    };
    requestAnimationFrame(() => setTimeout(() => step(0), 0));
    return false;
  }

  /** The first row on screen and where it is (null while the board's top is in view): see keepPlace. */
  function rowOnScreen() {
    const rows = drawn?.body?.isConnected ? drawn.body.children : null;
    if (!rows?.length || els.board.getBoundingClientRect().top >= 0) return null;
    let lo = 0;
    let hi = rows.length - 1;
    while (lo < hi) { // the first row whose bottom is below the window's top
      const mid = (lo + hi) >> 1;
      if (rows[mid].getBoundingClientRect().bottom > 0) hi = mid;
      else lo = mid + 1;
    }
    const key = keyOf(rows[lo]);
    return key ? { key, top: rows[lo].getBoundingClientRect().top } : null;
  }

  /** Scroll so the row read before a refresh is back where it was (rows above it may have come or gone). */
  function keepPlace(anchor) {
    if (!anchor || !drawn?.body?.isConnected) return;
    const value = CSS.escape(anchor.key.slice(2));
    const node = drawn.body.querySelector(anchor.key.startsWith("g:") ? `[data-game="${value}"]` : `[data-sep="${value}"]`);
    if (!node) return;
    const moved = node.getBoundingClientRect().top - anchor.top;
    if (Math.abs(moved) >= 1) window.scrollBy(0, moved);
  }

  /** A redraw for new data (a refresh, odds arriving): the board updates in place and the reader keeps their row. */
  function redrawInPlace() {
    const anchor = rowOnScreen();
    if (drawBoard()) keepPlace(anchor);
  }

  // The board is whole (every slice in): Back's scroll restore waits for it (render resolves after)
  let boardToken = 0;
  let settleWaiters = [];
  const settle = () => {
    if (redrawWhenSettled && !signal.aborted) {
      redrawWhenSettled = false;
      redrawInPlace(); // newer data arrived while the last drawing was going in
      return; // that drawing settles in turn
    }
    const waiting = settleWaiters;
    settleWaiters = [];
    waiting.forEach((fn) => fn());
  };
  const settled = () => (els.board.hasAttribute("data-settling") ? new Promise((resolve) => settleWaiters.push(resolve)) : Promise.resolve());

  /** A selector for the same control after a redraw (a price, a row's link, a fold). */
  function focusKey(node) {
    const leg = node.closest?.("[data-leg-key]");
    if (leg) return `[data-leg-key="${CSS.escape(leg.dataset.legKey)}"]`;
    const expand = node.closest?.("[data-expand]");
    if (expand) return `[data-expand="${CSS.escape(expand.dataset.expand)}"]`;
    const sort = node.closest?.("[data-sort]");
    if (sort) return `[data-sort="${CSS.escape(sort.dataset.sort)}"]`;
    const row = node.closest?.("[data-game]");
    if (row) return `[data-game="${CSS.escape(row.dataset.game)}"] .td-open, [data-game="${CSS.escape(row.dataset.game)}"] .gc-link`;
    return null;
  }

  /**
   * Redraw everything from ``shown`` and ``state``. ``moved``: the filters changed, so the board is drawn
   * anew from its top. ``refresh``: new data, so the row being read stays where it is.
   */
  function draw({ moved = false, refresh = false } = {}) {
    if (!shown) return;
    const anchor = refresh && !moved ? rowOnScreen() : null; // before the toolbar above the board changes
    prepared = prepare(shown);
    if (state.leagues) {
      const known = new Set([...state.leagues].filter((k) => prepared.leagues.has(k)));
      if (!known.size && state.leagues.size) state.leagues = null; // its leagues are gone (out of season): every sport
      else if (known.size !== state.leagues.size) state.leagues = known;
      state.leagues = normal(state.leagues);
    }
    if (!strip.some((d) => d.key === state.day)) state.day = strip[0].key;
    current = view(prepared, { leagues: state.leagues, day: state.day, status: state.status, search: state.search, sort: state.sort, pinned });
    const dayInfo = strip.find((d) => d.key === state.day);
    els.sub.textContent = `${current.searchAll ? "Next 7 days" : dayInfo?.long || ""} · ${selectionWords()} · ${current.games.length} game${current.games.length === 1 ? "" : "s"}`;
    setHtml(els.days, daysHtml(strip, state.day, current.days));
    setHtml(els.status, statusHtml(state.status, current.live));
    setHtml(els.chips, layout.mode === "phone" ? chipsHtml(prepared, current.counts, state.leagues, pinned) : "");
    setHtml(els.notes, notesHtml());
    if (els.sort.value !== state.sort.key) els.sort.value = state.sort.key;
    tree.update({ prepared, counts: current.counts, selection: state.leagues, pinned, presets, activePreset: activePreset(), liveCount: [...current.counts.values()].reduce((n, c) => n + c.live, 0) });
    setTreeRail(railItems(prepared, current.counts, state.leagues), (key) => {
      if (key === "*") choose(null);
      else {
        const group = prepared.groups.find((g) => g.key === key);
        if (group) choose(normal(new Set(group.leagues)));
      }
    });
    if (drawBoard({ fresh: moved })) keepPlace(anchor);
    updatedText();
    if (moved) {
      const top = els.bar.getBoundingClientRect().top;
      if (top < 0) window.scrollTo({ top: Math.max(0, window.scrollY + root.getBoundingClientRect().top - 8) });
    }
    askOdds();
  }

  // --- odds for the leagues opened ------------------------------------------------------------------------

  let quick = 0;
  let quickTimer = null;
  function lookSoon() {
    quick = 0;
    clearTimeout(quickTimer);
    const next = () => {
      if (signal.aborted || quick >= QUICK.length) return;
      quickTimer = setTimeout(() => {
        quick += 1;
        load().finally(() => {
          const waiting = [...asked.values()].some((a) => a.code === "queued" || a.code === "loading");
          const stillStale = [...asked.entries()].some(([k, a]) => (a.code === "queued" || a.code === "loading") && prepared?.leagues.get(k)?.odds !== "fresh");
          if (waiting && stillStale) next();
        });
      }, QUICK[quick]);
    };
    next();
  }
  cleanups.push(() => clearTimeout(quickTimer));

  /**
   * Ask for the odds of the leagues opened (see the header): a selection of up to HAND_PICKED leagues, each
   * once per visit; with All sports showing, the person's pinned leagues, MAX_AUTO_ASKS boards per visit at
   * most. A sport group or a bigger selection asks for nothing (its rows open one league).
   */
  function askOdds() {
    if (!prepared || !current) return;
    let scope = null;
    let room = Infinity;
    if (state.leagues) {
      if (state.leagues.size > HAND_PICKED) return;
      scope = state.leagues;
    } else if (pinned.length) {
      scope = new Set(pinned);
      room = MAX_AUTO_ASKS - [...asked.values()].filter((a) => a?.auto).length;
    }
    if (!scope || !scope.size || room <= 0) return;
    const want = leaguesNeedingOdds(prepared, current.games).filter((k) => scope.has(k) && !asked.has(k)).slice(0, Math.min(room, HAND_PICKED));
    const auto = !state.leagues;
    for (const league of want) {
      asked.set(league, { code: "asking", queued: false, at: Date.now(), auto });
      api("/api/slate/odds", { method: "POST", body: { league } })
        .then((answer) => {
          asked.set(league, { code: String(answer?.code || (answer?.queued ? "queued" : "refused")), queued: Boolean(answer?.queued), at: Date.now(), auto });
          saveAsked();
          if (answer?.queued) lookSoon();
        })
        .catch((error) => {
          asked.set(league, { code: "error", queued: false, at: Date.now(), auto });
          saveAsked();
          if (!isQuiet(error)) console.warn("Odds request", league, error.message);
        })
        .finally(() => {
          if (!signal.aborted && root.isConnected && shown) {
            setHtml(els.notes, notesHtml());
            redrawInPlace();
          }
        });
    }
    if (want.length) setHtml(els.notes, notesHtml());
  }

  // --- loading -------------------------------------------------------------------------------------------

  let loading = null;
  let pollTimer = null;
  let warmups = 0;
  let warmTimer = null;
  cleanups.push(() => clearTimeout(warmTimer));
  let prefetched = false;

  function onData(value, m) {
    if (signal.aborted || !root.isConnected) return;
    meta = m;
    failure = null;
    if (value === shown) {
      updatedText();
      return;
    }
    const first = !shown;
    shown = value;
    setSlateLeagues(value?.leagues || []);
    setQuota(value?.quota); // the header's "N odds left" (the slate carries the month's Odds API quota)
    if (first) showFilters(true);
    draw({ refresh: !first });
    if (!painted) {
      painted = true;
      markPainted(m.source === "cache" ? "cache" : "network");
    }
  }

  function showError(error) {
    if (error?.signIn || signal.aborted || !root.isConnected) return;
    const message = String(error?.message || "");
    drawn = null;
    boardToken += 1;
    showFilters(false); // nothing to filter: no empty tree, search or sort beside the message
    if (/18\+|confirm.{0,20}age/i.test(message) && error.status && error.status < 500) {
      els.board.innerHTML = ageGate();
    } else if (/break/i.test(message) && error.status && error.status < 500) {
      els.board.innerHTML = emptyState({ iconName: "pause", title: "You're on a break", body: message,
        actions: [{ label: "Responsible play", href: "#/help/responsible" }] });
    } else if (typeof navigator !== "undefined" && navigator.onLine === false) {
      els.board.innerHTML = emptyState({ iconName: "alert", title: "You're offline", body: "Today's games load when you're back online.",
        actions: [{ label: "Try again", attrs: 'data-td-act="refresh"', primary: true }] });
    } else {
      els.board.innerHTML = errorState(error, { title: "Couldn't load today's games" });
    }
    painted = true;
    markPainted("network");
  }

  function load({ manual = false } = {}) {
    if (loading) return loading;
    if (manual) els.refresh.classList.add("busy");
    loading = swr(SLATE_URL, { key: SLATE_KEY, signal, onData, onNotModified: (m) => {
      meta = m;
      failure = null;
      updatedText();
      setHtml(els.notes, notesHtml());
    } })
      .catch((error) => {
        if (error?.name === "AbortError" || signal.aborted) return;
        if (shown) {
          failure = error;
          updatedText();
          setHtml(els.notes, notesHtml());
        } else {
          showError(error);
          // The server is still building its first slate (a restart): try again by itself, a few times
          if (error?.status === 503 && warmups < 3) {
            warmups += 1;
            clearTimeout(warmTimer);
            warmTimer = setTimeout(() => { if (!signal.aborted && !shown) load(); }, 3000);
          }
        }
      })
      .finally(() => {
        loading = null;
        els.refresh.classList.remove("busy");
        if (!prefetched && shown && !signal.aborted) {
          prefetched = true;
          prefetchNext();
        }
      });
    return loading;
  }

  /**
   * The next three games' detail, quietly, when it costs no odds credits (cache.canPrefetchGame), and the
   * game page's code, so the first game opened after a deploy waits for neither.
   */
  function prefetchNext() {
    if (!current || !prepared) return;
    const next = current.games.filter((g) => g.status === "pre" && canPrefetchGame(g, prepared.leagues.get(g.league))).slice(0, 3);
    for (const g of next) prefetch(gameUrl(g.league, g.odds_id), { priority: "low" });
    warmView("game", { idle: true });
  }

  function schedule() {
    clearTimeout(pollTimer);
    if (signal.aborted) return;
    const live = prepared?.games.some((g) => g.status === "live");
    const today = dayKey(new Date());
    const gameDay = prepared?.games.some((g) => g._day === today && g.status === "pre");
    pollTimer = setTimeout(() => {
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      load().finally(schedule);
    }, live ? POLL.live : gameDay ? POLL.gameDay : POLL.idle);
  }
  cleanups.push(() => clearTimeout(pollTimer));

  const onVisible = () => {
    if (document.visibilityState === "visible" && meta && Date.now() - (meta.checked || 0) > 60000) load().finally(schedule);
  };
  document.addEventListener("visibilitychange", onVisible);
  cleanups.push(() => document.removeEventListener("visibilitychange", onVisible));
  const ticker = setInterval(updatedText, 30000);
  cleanups.push(() => clearInterval(ticker));
  const online = () => load();
  window.addEventListener("online", online);
  cleanups.push(() => window.removeEventListener("online", online));

  // --- events -----------------------------------------------------------------------------------------------

  cleanups.push(bindBoard(els.board, {
    onSort(key) {
      state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: 1 };
      remember();
      draw();
    },
    onOpenLeague(league) {
      choose(new Set([league]));
    },
    onToggle(id) {
      if (state.expanded.has(id)) state.expanded.delete(id);
      else state.expanded.add(id);
      drawBoard();
      els.board.querySelector(`[data-expand="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
    },
    onOpenDetail(hash, row) {
      openDetail(hash, { anchor: row }); // the row stays put as the board narrows for the inspector
    },
  }));

  root.addEventListener("click", (event) => {
    const t = event.target;
    if (!(t instanceof Element)) return;
    const day = t.closest("[data-day], [data-td-day]");
    if (day) {
      state.day = day.dataset.day || day.dataset.tdDay;
      state.expanded.clear();
      remember();
      draw({ moved: true });
      return;
    }
    const status = t.closest("[data-status]");
    if (status && STATUSES.includes(status.dataset.status)) {
      state.status = status.dataset.status;
      remember();
      draw({ moved: true });
      return;
    }
    const chipEl = t.closest("[data-chip]");
    if (chipEl) {
      choose(chipEl.dataset.chip === "all" ? null : new Set([chipEl.dataset.chip]));
      return;
    }
    if (t.closest("[data-chip-tree]")) {
      // A change in the sheet redraws the chips: focus comes back to the Sports chip, which stays put
      openTreeDrawer({ returnTo: () => root.querySelector(".td-chips [data-chip-tree]") });
      return;
    }
    if (t.closest("[data-td-refresh]")) {
      load({ manual: true }).then(() => { if (!failure) updatedText(); });
      return;
    }
    if (t.closest("[data-td-players]")) {
      openSearch();
      return;
    }
    if (t.closest("[data-onboard]")) {
      openOnboarding();
      return;
    }
    const act = t.closest("[data-td-act]")?.dataset.tdAct;
    if (act === "all") choose(null, { status: "all" });
    else if (act === "status-all") {
      state.status = "all";
      remember();
      draw();
    } else if (act === "clear-search") {
      state.search = "";
      els.search.value = "";
      remember();
      draw();
      els.search.focus();
    } else if (act === "refresh") {
      drawn = null;
      els.board.innerHTML = skeletonBoard(layout.mode);
      load({ manual: true });
    }
  });

  let typing = null;
  cleanups.push(() => clearTimeout(typing));
  els.search.addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(() => {
      state.search = els.search.value.slice(0, 60);
      remember();
      draw();
    }, 120);
  });
  els.search.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && els.search.value) {
      event.preventDefault();
      event.stopPropagation();
      els.search.value = "";
      state.search = "";
      remember();
      draw();
    }
  });
  els.sort.addEventListener("change", () => {
    if (!SORTS.includes(els.sort.value)) return;
    state.sort = { key: els.sort.value, dir: 1 };
    remember();
    draw();
  });

  // The sticky table header sits under the sticky toolbar: keep its offset as the toolbar wraps
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver(() => {
      const sticky = getComputedStyle(els.bar).position === "sticky";
      root.style.setProperty("--td-bar-h", sticky ? `${els.bar.offsetHeight}px` : "0px");
    });
    ro.observe(els.bar);
    cleanups.push(() => ro.disconnect());
    // The board's width shows or hides columns: a separator's or fold-out's one cell spans what is shown
    const widthRo = new ResizeObserver((entries) => {
      const width = entries[entries.length - 1].contentRect.width;
      const before = cols === null ? null : columnsAt(cols, false);
      cols = width;
      if (before === columnsAt(width, false) || !drawn || drawn.kind !== "table") return;
      const span = String(columnsAt(width, drawn.ai));
      for (const cell of els.board.querySelectorAll(".tb-sep-c, .tb-detail-in")) cell.setAttribute("aria-colspan", span);
    });
    widthRo.observe(root);
    cleanups.push(() => widthRo.disconnect());
  }
  cleanups.push(onLayoutChange((mode, previous) => {
    if (!shown) return;
    if ((mode === "phone") !== (previous === "phone")) draw(); // cards <-> table: drawn anew (another kind)
  }));
  cleanups.push(onInspectorChange((hash) => {
    for (const row of els.board.querySelectorAll(".tb-row.is-open")) row.classList.remove("is-open");
    if (!hash) return;
    for (const row of els.board.querySelectorAll(".tb-row[data-route]")) if (row.dataset.route === hash) row.classList.add("is-open");
  }));

  // --- first paint ------------------------------------------------------------------------------------------

  if (!cache.peek(SLATE_KEY)) els.board.innerHTML = skeletonBoard(layout.mode);
  await load();
  if (!signal.aborted) schedule();
  await Promise.race([settled(), new Promise((resolve) => { setTimeout(resolve, 1000); })]);
}

export function onLeave() {
  /* timers and listeners go with the screen's signal (render's cleanups) */
}
