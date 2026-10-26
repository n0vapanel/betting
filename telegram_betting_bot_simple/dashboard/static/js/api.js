// The one way the app talks to the server.
//
// Every call carries X-Player, the random id this browser keeps (the dashboard password is shared,
// so the server tells people apart by it: "web:<id>"), and inside Telegram also the Mini App's
// signed initData (Telegram Web shows the app in an iframe, where the session cookie can be dropped).
// A 401 hands over to the sign-in form; any other failure becomes an ApiError with the server's own
// words, so screens can show them as they are.

const PREFS_KEY = "sportsbet-ai"; // the first dashboard's key: {league, format, me: {id, name}, betsSource}
const PLAYER_ID = /^[A-Za-z0-9-]{8,64}$/;

export class ApiError extends Error {
  constructor(message, { status = 0, signIn = false, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status; // 0 when the server couldn't be reached
    this.signIn = signIn; // true on a 401: the sign-in form is already showing
    this.detail = detail;
  }
}

// --- who this browser is ----------------------------------------------------------------------------

function readPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

function writePrefs(prefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* private browsing: the id just lasts for this visit */
  }
}

function newPlayerId() {
  const bytes = crypto.getRandomValues(new Uint8Array(12)); // crypto.randomUUID needs https
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// Kept where the first dashboard kept it, so calls made there still belong to this browser
let playerId = (() => {
  const prefs = readPrefs();
  const known = prefs.me?.id;
  if (typeof known === "string" && PLAYER_ID.test(known)) return known;
  const id = newPlayerId();
  writePrefs({ ...prefs, me: { ...(prefs.me || {}), id } });
  return id;
})();

/** This browser's random id (sent as X-Player; the old calls API takes it as ?player=). */
export function getPlayerId() {
  return playerId;
}

/** Small per-browser preferences in the shared localStorage entry (name for the leaderboard, odds format…). */
export const prefs = {
  get(key, fallback = null) {
    const value = readPrefs()[key];
    return value === undefined ? fallback : value;
  },
  set(key, value) {
    writePrefs({ ...readPrefs(), [key]: value });
  },
  name() {
    return readPrefs().me?.name || "";
  },
  setName(name) {
    const all = readPrefs();
    writePrefs({ ...all, me: { ...(all.me || {}), id: playerId, name } });
  },
};

// --- Telegram and sign-in hooks ------------------------------------------------------------------

let initData = "";
let unauthorized = () => {};

/** Inside Telegram: send the Mini App's initData with every call from now on. */
export function setInitData(value) {
  initData = value || "";
}

/** What to do on a 401 (app.js shows the sign-in form; inside Telegram it signs in again or asks to reopen the app). */
export function onUnauthorized(handler) {
  unauthorized = handler;
}

// --- requests -----------------------------------------------------------------------------------

function headers(hasBody) {
  const out = { "X-Player": playerId, Accept: "application/json" };
  if (hasBody) out["Content-Type"] = "application/json";
  if (initData) out["X-Telegram-Init-Data"] = initData;
  return out;
}

/** The server's reason for a refusal: FastAPI's {detail: "..."} or its list of validation errors. */
function reason(body, status) {
  const detail = body?.detail;
  if (Array.isArray(detail)) {
    const text = detail.map((d) => String(d?.msg || "").replace(/^Value error, /, "")).filter(Boolean).join("; ");
    if (text) return text;
  } else if (typeof detail === "string" && detail) {
    return detail;
  }
  if (status === 404) return "Not available yet on this server.";
  if (status >= 500) return `The server had a problem (${status}). Try again in a minute.`;
  return `Request failed (${status}).`;
}

/**
 * Call the API and return its JSON.
 *   api("/api/today?leagues=basketball_nba")
 *   api("/api/tickets/price", { method: "POST", body: {...}, signal })
 * Options: method, body (an object, sent as JSON), signal (AbortSignal), auth (false: no sign-in form on 401).
 * Throws ApiError (status 0 = network) or the AbortError of a cancelled request.
 */
export async function api(path, { method = "GET", body, signal, auth = true } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: headers(body !== undefined),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      credentials: "same-origin",
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("Couldn't reach the server. Check your connection and try again.");
  }
  const data = await response.json().catch(() => null);
  if (response.status === 401) {
    if (auth) unauthorized();
    throw new ApiError(reason(data, 401) === "Sign in required" ? "Sign in to continue." : reason(data, 401),
      { status: 401, signIn: auth, detail: data?.detail });
  }
  if (!response.ok) throw new ApiError(reason(data, response.status), { status: response.status, detail: data?.detail });
  return data;
}

export const get = (path, options = {}) => api(path, options);
export const post = (path, body = {}, options = {}) => api(path, { ...options, method: "POST", body });
export const put = (path, body = {}, options = {}) => api(path, { ...options, method: "PUT", body });
export const del = (path, options = {}) => api(path, { ...options, method: "DELETE" });

/** A query string from an object, skipping empty values: qs({league: "x", id: 3}) -> "?league=x&id=3". */
export function qs(params) {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(Array.isArray(v) ? v.join(",") : v)}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

// --- signing in ---------------------------------------------------------------------------------

/** The dashboard password (browsers). Throws ApiError with a readable message on failure. */
export async function passwordLogin(password) {
  try {
    return await api("/api/login", { method: "POST", body: { password }, auth: false });
  } catch (error) {
    if (error.status === 401) throw new ApiError("Wrong password, try again.", { status: 401 });
    throw error;
  }
}

/**
 * Sign in with Telegram's initData. A 403 carries the server's message (it includes the person's own
 * Telegram id so they can send it to the owner); show it as it is.
 */
export function telegramLogin(data) {
  return api("/api/tg-login", { method: "POST", body: { init_data: data }, auth: false });
}

export function logout() {
  return api("/api/logout", { method: "POST", auth: false }).catch(() => null);
}

/** True for errors a screen should stay quiet about (the sign-in form or a cancelled request covers them). */
export function isQuiet(error) {
  return error?.signIn || error?.name === "AbortError";
}
