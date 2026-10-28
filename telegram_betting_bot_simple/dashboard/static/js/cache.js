// The client's data cache: what lets Today paint before the network answers.
//
// ── Stale-while-revalidate (IndexedDB) ────────────────────────────────────────────────────────────
//   cache.get(key)                 -> Promise<{value, etag, at, checked} | null>   (never throws; calls made
//                                     while a read of the key is on its way share it: one read, one copy)
//   cache.put(key, value, etag)    -> Promise<void>                                (never throws)
//   cache.del(key) / cache.clear() -> Promise<void>
//   cache.peek(key)                -> the entry already in memory this visit, or null (synchronous)
//   cache.warm(...keys)            -> start reading keys into memory (the shell warms SLATE_KEY at boot)
//   SLATE_URL = "/api/slate", SLATE_KEY = "slate": Today's swr(SLATE_URL, {key: SLATE_KEY, ...})
//   cache.swr(url, {key, onData, onNotModified, signal, maxAge, paint}) -> Promise<{value, meta}>
//     1. Paints at once from the cache when there is a copy (onData(value, meta) with meta.source "cache"),
//        unless paint() says no (default: the person has confirmed their age and isn't on a break).
//     2. Then asks the server with If-None-Match: on 200 it calls onData again (meta.source "network") and
//        stores the new copy after that paint (in memory at once, in IndexedDB when the page is idle); on
//        304 it keeps what is shown and calls onNotModified(meta) with the new meta.checked time ("updated
//        N min ago" should read meta.checked), saving only that time, not the whole copy again.
//     meta = {source: "cache"|"network", at, checked, etag, status}: ``at`` is when this copy was fetched,
//     ``checked`` when the server last said it is current (both ms since the epoch).
//     Resolves to the final {value, meta}; rejects with the ApiError (or AbortError) when the network
//     fails, with error.painted = true when the cached copy is already on screen (show "couldn't refresh"
//     rather than an error page). Copies older than maxAge (default 24 h) are not painted.
//   IndexedDB can be missing or throw (private windows, Safari's 7-day eviction, a blocked open): every
//   call then falls back to an in-memory copy for this visit and the network, and a slow open never holds
//   the first paint up for more than READ_TIMEOUT ms.
//
// ── Prefetch ──────────────────────────────────────────────────────────────────────────────────────
//   prefetch(url, {priority: "high"|"low"}) -> Promise<data|null>   at most 2 requests in flight; the rest
//       wait (high goes first: a hover or a tap; low for "the next 3 games" after Today paints)
//   prefetched(url)                        -> the fresh copy (a clone) or undefined, synchronously
//   api() answers a GET for a prefetched path from here (api.js setGetHook), so game.js and player.js get
//   the copy without knowing; copies live PREFETCH_TTL ms in a small LRU of PREFETCH_MAX entries.
//   gameUrl(league, id) / playerUrl(league, id): the exact paths game.js and player.js request.
//   routeUrl("#/game/<league>/<id>") -> the API path for a route (game or player), or null.
//   canPrefetchGame(game, league): only games whose league's cached odds are under ~2 h old. GET /api/game
//   fetches a fresh odds board (Odds API credits) when the cached one is older, so a hover must never
//   cause that: prefetch a game only when this says yes.
//
// Nothing personal goes under a shared key: the slate is the same for everyone; the profile is cached
// by store.js in localStorage under the person's own id.

import { request, setGetHook } from "./api.js";

const DB_NAME = "sportsbet-cache";
const DB_STORE = "kv";
const DB_VERSION = 1;
const ENTRY_VERSION = 1;
const READ_TIMEOUT = 120; // ms a cache read may take before the page stops waiting for it
const DAY = 24 * 3600 * 1000;

const memory = new Map(); // key -> entry, this visit's copies (always written, read first)
const reads = new Map(); // key -> the IndexedDB read on its way (warm() and swr() share it)
const CHECKED = "\u0000checked"; // key + CHECKED: when the server last said the copy is current (a 304)
let dbPromise = null;
let dbBroken = false;

/** window.indexedDB, or null: reading it can itself throw (sandboxed frames, some private modes). */
function idb() {
  try {
    return typeof indexedDB === "undefined" ? null : indexedDB;
  } catch {
    return null;
  }
}

function openDb() {
  const factory = idb();
  if (dbBroken || !factory) return Promise.resolve(null);
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      let settled = false;
      const done = (db) => {
        if (settled) return;
        settled = true;
        resolve(db);
      };
      // An open that never answers (seen on some WebViews) counts as no IndexedDB for this visit
      const timer = setTimeout(() => {
        dbBroken = true;
        done(null);
      }, 2000);
      try {
        const req = factory.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          try {
            if (!req.result.objectStoreNames.contains(DB_STORE)) req.result.createObjectStore(DB_STORE);
          } catch {
            /* the open fails below */
          }
        };
        req.onsuccess = () => {
          clearTimeout(timer);
          const db = req.result;
          db.onversionchange = () => db.close();
          done(db);
        };
        req.onerror = () => {
          clearTimeout(timer);
          dbBroken = true;
          done(null);
        };
        req.onblocked = () => {
          clearTimeout(timer);
          done(null);
        };
      } catch {
        clearTimeout(timer);
        dbBroken = true;
        done(null);
      }
    });
  }
  return dbPromise;
}

/** Work after the paint: when the page is idle (or soon, where there is no idle callback). */
const idle = (fn) => (typeof requestIdleCallback === "function" ? requestIdleCallback(fn, { timeout: 2000 }) : setTimeout(fn, 50));

/** One request on the store; resolves to its result or null (errors are swallowed: the cache is optional). */
async function withStore(mode, action) {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(DB_STORE, mode);
      const req = action(tx.objectStore(DB_STORE));
      tx.oncomplete = () => resolve(req ? req.result ?? null : null);
      tx.onerror = () => resolve(null);
      tx.onabort = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function valid(entry) {
  return entry && typeof entry === "object" && entry.v === ENTRY_VERSION && "value" in entry ? entry : null;
}

export const cache = {
  /** The cached entry {value, etag, at, checked}, or null. Never throws; waits READ_TIMEOUT ms at most. */
  async get(key) {
    const held = memory.get(key);
    if (held) return held;
    let read = reads.get(key);
    if (!read) {
      // The copy and the time of the last 304 in one transaction (see touch())
      read = withStore("readonly", (store) => {
        const main = store.get(key);
        const checked = store.get(key + CHECKED);
        return {
          get result() {
            const entry = valid(main.result);
            const when = checked.result;
            return entry && Number.isFinite(when) && when > (entry.checked || 0) ? { ...entry, checked: when } : entry;
          },
        };
      }).then((entry) => {
        if (entry && !memory.has(key)) memory.set(key, entry);
        return entry;
      }).finally(() => reads.delete(key));
      reads.set(key, read);
    }
    const entry = await Promise.race([read, new Promise((resolve) => { setTimeout(() => resolve(undefined), READ_TIMEOUT); })]);
    if (entry === undefined) return null; // too slow this time; the network answers instead
    return memory.get(key) || null;
  },

  /**
   * Store a copy: in memory now, in IndexedDB when it can (``later``: once the page is idle, so a big copy's
   * write never stands between new data and its paint).
   */
  async put(key, value, etag = null, { at = Date.now(), later = false } = {}) {
    const entry = { v: ENTRY_VERSION, value, etag: etag || null, at, checked: Date.now() };
    memory.set(key, entry);
    if (later) await new Promise((resolve) => { idle(resolve); });
    const latest = memory.get(key);
    if (!latest || latest.value !== value) return; // a newer copy came meanwhile (it writes itself), or a clear()
    await withStore("readwrite", (store) => store.put(latest, key));
  },

  /** The server said the copy is current (a 304): note when (only the time is written, not the copy). */
  async touch(key) {
    const entry = memory.get(key);
    if (!entry) return null;
    const updated = { ...entry, checked: Date.now() };
    memory.set(key, updated);
    await withStore("readwrite", (store) => store.put(updated.checked, key + CHECKED));
    return updated;
  },

  async del(key) {
    memory.delete(key);
    await withStore("readwrite", (store) => {
      store.delete(key + CHECKED);
      return store.delete(key);
    });
  },

  /** Forget everything (signing out). */
  async clear() {
    memory.clear();
    lru.clear();
    await withStore("readwrite", (store) => store.clear());
  },

  /** This visit's copy, synchronously (null until get() or put() has run for the key). */
  peek(key) {
    return memory.get(key) || null;
  },

  swr: (url, options) => swr(url, options),

  /** Start reading keys into memory now (the shell warms the slate while Today's module loads). */
  warm(...keys) {
    for (const key of keys) if (!memory.has(key)) cache.get(key).catch(() => null);
  },
};

/** The all-sports slate: GET /api/slate, cached under this key (Today and the shell agree on it). */
export const SLATE_URL = "/api/slate";
export const SLATE_KEY = "slate";

// Open the database while the rest of the app loads, so the first read doesn't pay for the open
openDb();

// --- stale-while-revalidate ---------------------------------------------------------------------------

let paintCheck = () => true;

/** app.js: whether a cached copy may be painted (the age and break gates, from the cached profile). */
export function setPaintCheck(fn) {
  paintCheck = fn;
}

/** See the header. */
export async function swr(url, { key = url, onData = null, onNotModified = null, signal, maxAge = DAY, paint = null } = {}) {
  const cached = await cache.get(key);
  let painted = false;
  const mayPaint = paint || paintCheck;
  if (cached && Date.now() - cached.at <= maxAge && mayPaint()) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    painted = true;
    onData?.(cached.value, { source: "cache", at: cached.at, checked: cached.checked, etag: cached.etag, status: 0 });
  }
  let result;
  try {
    // Ask with the copy's ETag only when that copy may be used: a copy we didn't paint must not come back as a 304
    result = await request(url, { signal, headers: cached?.etag && painted ? { "If-None-Match": cached.etag } : null });
  } catch (error) {
    if (error && typeof error === "object") error.painted = painted;
    throw error;
  }
  if (result.status === 304 && painted) {
    const entry = (await cache.touch(key)) || cached;
    const meta = { source: "cache", at: entry.at, checked: entry.checked, etag: entry.etag, status: 304 };
    onNotModified?.(meta);
    return { value: entry.value, meta };
  }
  if (result.status === 304) { // a 304 for a copy we didn't paint (a proxy's): ask again without the ETag
    result = await request(url, { signal, headers: { "Cache-Control": "no-cache" } });
  }
  const etag = result.headers?.get?.("ETag") || null;
  const now = Date.now();
  const stored = cache.put(key, result.data, etag, { at: now, later: true }); // in memory now; on disk after the paint
  const meta = { source: "network", at: now, checked: now, etag, status: result.status };
  if (!signal?.aborted) onData?.(result.data, meta);
  stored.catch(() => null);
  return { value: result.data, meta };
}

// --- prefetch --------------------------------------------------------------------------------------------

const MAX_IN_FLIGHT = 2;
const PREFETCH_MAX = 24;
const PREFETCH_TTL = 90 * 1000;
const QUEUE_MAX = 12;

const lru = new Map(); // url -> {data, at}; Map order is the LRU order (oldest first)
const inFlight = new Map(); // url -> Promise<data|null>
const queue = []; // [{url, resolve}], front first
let running = 0;

function fresh(url) {
  const entry = lru.get(url);
  if (!entry) return null;
  if (Date.now() - entry.at > PREFETCH_TTL) {
    lru.delete(url);
    return null;
  }
  lru.delete(url); // touch: most recently used goes last
  lru.set(url, entry);
  return entry;
}

function remember(url, data) {
  lru.delete(url);
  lru.set(url, { data, at: Date.now() });
  while (lru.size > PREFETCH_MAX) lru.delete(lru.keys().next().value);
}

const clone = (data) => {
  try {
    return structuredClone(data);
  } catch {
    return JSON.parse(JSON.stringify(data));
  }
};

function pump() {
  while (running < MAX_IN_FLIGHT && queue.length) {
    const job = queue.shift();
    running += 1;
    request(job.url, { auth: false })
      .then(({ data }) => {
        remember(job.url, data);
        job.resolve(data);
      })
      .catch(() => job.resolve(null)) // a failed prefetch is silent: the click asks again
      .finally(() => {
        running -= 1;
        inFlight.delete(job.url);
        pump();
      });
  }
}

/** Fetch ``url`` ahead of a tap (see the header). Resolves to the data, or null when it failed. */
export function prefetch(url, { priority = "high" } = {}) {
  if (typeof url !== "string" || !url.startsWith("/api/")) return Promise.resolve(null);
  const held = fresh(url);
  if (held) return Promise.resolve(held.data);
  if (inFlight.has(url)) {
    if (priority === "high") { // already waiting: move it to the front
      const at = queue.findIndex((job) => job.url === url);
      if (at > 0) queue.unshift(...queue.splice(at, 1));
    }
    return inFlight.get(url);
  }
  const promise = new Promise((resolve) => {
    const job = { url, resolve };
    if (priority === "high") queue.unshift(job);
    else queue.push(job);
  });
  inFlight.set(url, promise);
  // The oldest low-priority waits give way when too many pile up (they resolve null, nothing is fetched)
  while (queue.length > QUEUE_MAX) {
    const dropped = queue.pop();
    inFlight.delete(dropped.url);
    dropped.resolve(null);
  }
  pump();
  return promise;
}

/** A fresh prefetched copy of ``url`` (a clone), or undefined. */
export function prefetched(url) {
  const held = fresh(url);
  return held ? clone(held.data) : undefined;
}

// api() asks here first for a GET: a fresh copy, or the prefetch already on its way
setGetHook((path) => {
  const held = fresh(path);
  if (held) return clone(held.data);
  const pending = inFlight.get(path);
  if (pending) return pending.then((data) => (data == null ? undefined : clone(data)));
  return undefined;
});

/** GET /api/game for a game, exactly as game.js asks for it. */
export function gameUrl(league, id) {
  return `/api/game?league=${encodeURIComponent(league)}&id=${encodeURIComponent(id)}`;
}

/** GET /api/players/<league>/<id>, exactly as player.js asks for it. */
export function playerUrl(league, id) {
  return `/api/players/${encodeURIComponent(league)}/${encodeURIComponent(id)}`;
}

/** The API path behind a route hash (#/game/<league>/<id> or #/player/<league>/<id>), or null. */
export function routeUrl(hash) {
  const m = /^#?\/?(game|player)\/([a-z0-9_]{2,64})\/([A-Za-z0-9_-]{1,64})(?:\?.*)?$/.exec(String(hash || ""));
  if (!m) return null;
  if (m[1] === "player") return /^[0-9]{1,12}$/.test(m[3]) ? playerUrl(m[2], m[3]) : null;
  return gameUrl(m[2], m[3]);
}

const ODDS_SAFE_MS = 110 * 60 * 1000; // the server re-fetches odds older than 2 h: keep 10 minutes' margin

/**
 * May a game's detail be prefetched without costing Odds API credits? Only when it has an odds id and its
 * league's cached odds (slate league.odds_as_of, or the game's own odds_as_of) are under ~110 minutes old.
 */
export function canPrefetchGame(game, league = null) {
  if (!game || !game.odds_id) return false;
  const asOf = Date.parse(game.odds_as_of || league?.odds_as_of || "");
  if (!Number.isFinite(asOf)) return false;
  if (league && league.odds && league.odds !== "fresh") return false;
  return Date.now() - asOf < ODDS_SAFE_MS;
}
