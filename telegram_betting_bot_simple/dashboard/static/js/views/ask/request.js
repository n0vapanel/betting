// Ask AI's request bar: "build me a ticket" in taps (sports, stake, how much back, legs, risk tier).
//
// It is sent as POST /api/chat {request: {leagues, stake, target, max_legs, tier}}: the server builds the
// tickets in code (the target builder) and the assistant explains them. The bar only collects what the
// person chose; it never proposes a bigger stake. Its choices are kept in this browser between visits.

import { prefs } from "../../api.js";
import { me, favouriteSports, leagueTitle, sportEmoji } from "../../app.js";
import { esc, chip, segmented, explain } from "../../ui.js";
import { icon } from "../../icons.js";
import { amountText } from "./format.js";

const KEY = "askRequest";
export const MAX_SPORTS = 4; // the assistant fetches at most this many leagues' odds per request
export const STAKES = ["5", "10", "25"];
const TARGET_TIMES = [2, 5, 10];
const TIERS = ["safe", "risky", "crazy"];
const MONEY = /^\d{1,7}(\.\d{1,2})?$/;

/** The bar's saved choices, checked (a broken or old copy falls back to the defaults). */
export function loadRequest() {
  const saved = prefs.get(KEY, null) || {};
  const favourites = favouriteSports();
  const leagues = Array.isArray(saved.leagues) ? saved.leagues.filter((k) => /^[a-z0-9_]{2,64}$/.test(k)).slice(0, MAX_SPORTS) : [];
  const legs = Number(saved.max_legs);
  const tier = TIERS.includes(saved.tier) ? saved.tier : TIERS.includes(me().risk_tier) ? me().risk_tier : null;
  return {
    leagues: leagues.length ? leagues : favourites.slice(0, 1),
    stake: MONEY.test(String(saved.stake ?? "")) ? String(saved.stake) : "10",
    target: MONEY.test(String(saved.target ?? "")) ? String(saved.target) : "50",
    max_legs: Number.isInteger(legs) && legs >= 1 && legs <= 6 ? legs : 3,
    tier,
  };
}

export function saveRequest(request) {
  prefs.set(KEY, request);
}

/** "$20", "$50", "$100" for a $10 stake: totals back to pick from (whole cents, no rounding surprises). */
export function targetChoices(stake) {
  const cents = Math.round(Number(stake) * 100);
  if (!Number.isFinite(cents) || cents <= 0) return [];
  return TARGET_TIMES.map((times) => {
    const total = cents * times;
    return total % 100 ? (total / 100).toFixed(2) : String(total / 100);
  });
}

/** The request body for /api/chat, or {error} saying what to fix. */
export function checkRequest(request) {
  if (!request.leagues.length) return { error: "Pick at least one sport." };
  const stake = String(request.stake || "").trim();
  if (!MONEY.test(stake) || Number(stake) < 0.01 || Number(stake) > 100000) return { error: "Enter a stake between $0.01 and $100,000." };
  const target = String(request.target || "").trim();
  if (!MONEY.test(target) || Number(target) > 10000000) return { error: "Enter how much you'd like back, e.g. 50." };
  if (Number(target) <= Number(stake)) return { error: "“Want back” is the total if it wins, so it should be more than the stake." };
  const body = { leagues: request.leagues, stake, target, max_legs: request.max_legs };
  if (request.tier) body.tier = request.tier;
  return { body };
}

const TIER_WORDS = { safe: "Safe", risky: "Risky", crazy: "Crazy" };

/** One line for the collapsed bar: "NBA · $10 → $50 back · up to 3 legs · any risk". */
export function requestSummary(request) {
  const sports = request.leagues.map(leagueTitle).join(" + ") || "No sport";
  return `${sports} · ${amountText(request.stake)} → ${amountText(request.target)} back · up to ${request.max_legs} leg${request.max_legs === 1 ? "" : "s"} · ${request.tier ? TIER_WORDS[request.tier] : "any risk"}`;
}

function sportChips(request) {
  const favourites = favouriteSports();
  const shown = [...new Set([...favourites, ...request.leagues])];
  return shown.map((key) => chip(`${sportEmoji(key)} ${leagueTitle(key)}`, {
    active: request.leagues.includes(key), attrs: `data-req-sport="${esc(key)}"`,
  })).join("") + `<button type="button" class="chip add" data-req-more aria-label="More sports">${icon("plus", { size: 14 })}More</button>`;
}

/** The quick-pick chips of an amount row ("stake" or "target"); redrawn alone while the person types. */
export function amountChips(name, choices, value) {
  return choices.map((v) => `<button type="button" class="chip${String(value) === v ? " on" : ""}" aria-pressed="${String(value) === v}" data-req-${name}="${esc(v)}">${esc(amountText(v))}</button>`).join("");
}

function amountRow(name, choices, value, label) {
  const custom = !choices.includes(String(value));
  return `<div class="stake-row">
    <span class="ask-chips" data-req-chips="${name}">${amountChips(name, choices, value)}</span>
    <span class="money-input"><span aria-hidden="true">$</span><input class="input" type="text" inputmode="decimal" autocomplete="off" data-req-input="${name}"
      value="${custom ? esc(value) : ""}" placeholder="Other" aria-label="${esc(label)}"></span>
  </div>`;
}

/**
 * The bar. open: the full form, else one summary line with an "Edit" toggle. telegram: Telegram's
 * main button sends it, so the in-page button is left out. disabled: shown but switched off.
 */
export function requestHtml(request, { open = true, telegram = false, disabled = false } = {}) {
  const tiers = [{ value: "", label: "Any" }, ...TIERS.map((t) => ({ value: t, label: TIER_WORDS[t] }))];
  return `<section class="card ask-request${open ? " open" : ""}" aria-labelledby="ask-req-title">
    <button type="button" class="ask-request-toggle" data-req-toggle aria-expanded="${open}" aria-controls="ask-req-body">
      <span class="ask-req-icon">${icon("target", { size: 20 })}</span>
      <span class="grow"><b id="ask-req-title">Build me a ticket</b>
        <span class="ask-req-summary">${esc(requestSummary(request))}</span></span>
      <span class="ask-req-edit">${open ? "Hide" : "Edit"}${icon(open ? "up" : "down", { size: 18 })}</span>
    </button>
    <fieldset id="ask-req-body" class="ask-request-body"${open ? "" : " hidden"}${disabled ? " disabled" : ""}>
      <legend class="sr-only">Build me a ticket</legend>
      <div class="field"><span class="label">Sports <span class="muted small">(up to ${MAX_SPORTS})</span></span>
        <div class="chip-row wrap" data-req-sports>${sportChips(request)}</div></div>
      <div class="field"><span class="label">Stake ${explain("stake")}</span>${amountRow("stake", STAKES, request.stake, "Other stake")}</div>
      <div class="field"><span class="label">Want back in total ${explain("payout")}</span>
        ${amountRow("target", targetChoices(request.stake), request.target, "Other amount back")}
        <p class="hint">What comes back if it wins, your stake included.</p></div>
      <div class="ask-req-row">
        <div class="field"><span class="label" id="ask-legs-label">Max legs</span>
          <div class="stepper" role="group" aria-labelledby="ask-legs-label">
            <button type="button" data-req-legs="-1" aria-label="One leg fewer"${request.max_legs <= 1 ? " disabled" : ""}>${icon("minus", { size: 18 })}</button>
            <output aria-live="polite">${request.max_legs}</output>
            <button type="button" data-req-legs="1" aria-label="One leg more"${request.max_legs >= 6 ? " disabled" : ""}>${icon("plus", { size: 18 })}</button>
          </div></div>
        <div class="field grow"><span class="label">Risk ${explain("tiers")}</span>
          ${segmented("req-tier", tiers, request.tier || "", { label: "Risk tier" })}</div>
      </div>
      <div class="form-error" role="alert" data-req-error></div>
      ${telegram ? "" : `<button type="button" class="btn btn-primary block" data-req-send>${icon("target", { size: 18 })}Find tickets</button>`}
      <p class="hint">The app's code builds and prices every ticket; the AI explains them. Each request uses some of your daily AI allowance, and nothing goes on your slip until you tap.</p>
    </fieldset>
  </section>`;
}
