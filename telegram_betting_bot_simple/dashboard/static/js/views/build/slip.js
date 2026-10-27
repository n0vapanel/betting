// The slip half of the Build screen (#/build): the picks, the bet type (Single · Multi · Round robin by
// k · a named System), the stake, the server's numbers for the ticket, and Analyse / Save / Share / Clear,
// then "Let AI build one" below.
//
// The slip itself lives in store.js (localStorage) and the shell re-prices it on every change through
// POST /api/tickets/price; this screen listens for both and redraws only the parts that changed, so the
// stake box keeps its focus while the numbers update. Nothing is worked out here: chances, payouts, the
// bookmaker's cut and the expected result are the server's.
//
// Deep links: #/build?analyse=1 runs the analysis (the slip bar's "Analyse"), #/build?code=<share code>
// loads a shared slip (asking first when it would replace picks already on the slip).
//
// On a break the server prices, analyses and saves nothing (Lab.ready), so the numbers make way for one line
// saying until when, and Analyse and Save are off (Telegram's main button goes; the page's own Analyse
// shows, switched off). In the 24-hour picks pause all of that stays open: only the builder below is off.
//
// Undo: Clear and every replace (a share code, the builder's "Use as my slip") keep the slip from before.
// After a replace an Undo sits under the slip's buttons until the person changes the slip; after a Clear
// it is on the empty slip. Clearing straight after a replace keeps the older slip, since the replaced one
// can always be loaded or added again.

import { api, isQuiet } from "../../api.js";
import { slip, priceSlip, priced, onPriced, me, loadMe, onMeChange, legKey, openOnboarding, inTelegram, replaceAddress } from "../../app.js";
import { esc, pct, money, priceText, timeLabel, notice, skeleton, explain, badge, chip, toast, sheet, confirmSheet, ago, legChanceWord } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic, setSecondaryButton } from "../../telegram.js";
import { copyText } from "../../legacy.js";
import {
  STAKES, systemsFor, systemInfo, cleanStake, chanceTiles, summaryRows, outcomesTable, warningsList, redraw, reveal, gateHint,
  gameShort, picksBlock, blockNote,
} from "./common.js";
import { createAnalysis } from "./analyse.js";
import { createBuilder } from "./target.js";

let saved = null; // {text, view}: the slip as last saved (POST /api/slips) and the request it was saved from
// The slip before "Clear" or a replace, so it can be put back: {legs, type, stake, k, system, reason:
// "clear"|"replace", after (a replace: the slip it left, as JSON), what ("code 7KD2QX")}
let cleared = null;
let replacing = false; // a replace is filling the slip: those changes aren't the person's edits
let undoShown = null; // the mounted screen's redraw for the Undo row
const BLOCK_ID = "bx-block-slip"; // the line that says why Analyse and Save are off

/** The break, when there is one: the server refuses pricing, analysing and saving the slip then. */
function breakBlock() {
  const block = picksBlock();
  return block?.kind === "break" ? block : null;
}

const MODES = [
  { value: "single", label: "Single" },
  { value: "multi", label: "Multi" },
  { value: "round_robin", label: "Round robin" },
  { value: "system", label: "System" },
];

/** The mode switch: the four bet types, then the rollover planner (a page of its own). */
export function modesHtml(current) {
  return `<div class="seg bx-modes" role="group" aria-label="Bet type">
    ${MODES.map((m) => `<button type="button" data-mode="${m.value}" aria-pressed="${current === m.value}">${esc(m.label)}</button>`).join("")}
    <a href="#/build/rollover"${current === "rollover" ? ' aria-current="page"' : ""} aria-label="Rollover plan">Rollover</a>
  </div>`;
}

/** Switch the slip's bet type, choosing a round robin size or a named system that fits its picks. */
export function chooseMode(type) {
  const n = slip.count;
  const changes = { type };
  if (type === "round_robin") {
    const k = slip.k;
    changes.k = Number.isInteger(k) && k >= 2 && k <= Math.min(7, n - 1) ? k : 2;
  } else if (type === "system") {
    const current = systemInfo(slip.system);
    const fits = systemsFor(n);
    changes.system = current && current.legs === n ? current.key : fits[0]?.key || (n < 3 ? "trixie" : "goliath");
  }
  slip.set(changes);
  haptic.select();
}

function snapshot() {
  return { legs: slip.legs.map((leg) => ({ ...leg })), type: slip.type, stake: slip.stake, k: slip.k, system: slip.system };
}

/** The last replace can still be undone: the slip is exactly what it left. */
function replacePending() {
  return cleared?.reason === "replace" && cleared.after === JSON.stringify(slip.body());
}

/**
 * Replace the slip with what ``fill()`` puts on it (after emptying it), keeping the old slip for Undo.
 * ``what`` names the new slip in the Undo row ("code 7KD2QX").
 */
function replaceSlip(fill, what = "") {
  const before = slip.count ? snapshot() : null;
  const beforeText = JSON.stringify(slip.body());
  replacing = true;
  try {
    slip.clear();
    fill();
  } finally {
    replacing = false;
  }
  const after = JSON.stringify(slip.body());
  if (before && after !== beforeText) cleared = { ...before, reason: "replace", after, what };
  undoShown?.();
}

function restore() {
  if (!cleared) return;
  const before = cleared;
  cleared = null;
  slip.clear();
  for (const leg of before.legs) slip.add(leg);
  slip.set({ type: before.type, stake: before.stake, k: before.k, system: before.system });
  toast(before.reason === "replace" ? "Your earlier picks are back" : "Slip restored", { kind: "good" });
}

function clearSlip() {
  if (!slip.count) return;
  if (replacePending()) cleared = { ...cleared, reason: "clear", after: null };
  else cleared = { ...snapshot(), reason: "clear" };
  slip.clear();
  haptic.tap();
  toast("Slip cleared · Undo is on the empty slip");
}

function picks(n) {
  return `${n} pick${n === 1 ? "" : "s"}`;
}

// --- pieces ----------------------------------------------------------------------------------------

function subHtml() {
  const n = slip.count;
  const type = slip.type;
  if (type === "single") {
    return `<p class="hint">Each pick is its own bet at the stake below. ${explain("single")}</p>`;
  }
  if (type === "multi") {
    return `<p class="hint">One bet on every pick: they all have to win. ${explain("multi")}</p>`;
  }
  if (type === "round_robin") {
    if (n < 3) {
      return notice(`A round robin needs at least 3 picks; your slip has ${n}. Add more from Today, or choose another bet type.`, "info");
    }
    const sizes = [];
    for (let k = 2; k <= Math.min(7, n - 1); k += 1) sizes.push(k);
    const current = slip.k || 2;
    return `<p class="hint">Every combination of picks, each its own small multi: some can lose and you still get something back. ${explain("round_robin")}</p>
      <div class="chip-row wrap" role="group" aria-label="Picks per bet">${sizes.map((k) => chip(`By ${k}s`, { active: current === k, attrs: `data-k="${k}"` })).join("")}</div>`;
  }
  const fits = systemsFor(n);
  const current = systemInfo(slip.system);
  if (!fits.length) {
    return notice(`System bets take 3 to 8 picks; your slip has ${n}.`, "info") + `<p class="hint">Trixie, Yankee, Lucky 15 and the rest ${explain("system")}</p>`;
  }
  return `<p class="hint">A named set of combinations; the stake is per bet. ${explain("system")}</p>
    <div class="chip-row wrap" role="group" aria-label="System">${fits.map((s) => chip(`${s.title} · ${s.bets} bets`, { active: current?.key === s.key, attrs: `data-system="${s.key}"` })).join("")}</div>
    ${current ? `<p class="hint">${esc(current.title)}: ${esc(current.what)}.</p>` : ""}`;
}

function legRow(leg, view, { gone = false, move = null, weak = false } = {}) {
  const odds = view?.odds || { decimal: leg.price, american: leg.american };
  const ai = view ? view.ai_prob : leg.ai;
  const market = view ? view.market_prob : leg.fair;
  const label = view?.label || leg.label || "Pick";
  const league = view?.league || leg.league || leg.ref?.league || "";
  const game = gameShort(view?.game || leg.game, league);
  const start = view?.start || leg.start;
  const where = [game, start ? timeLabel(start) : ""].filter(Boolean).join(" · ");
  const chances = [ai != null ? `${legChanceWord(view || leg)} ${pct(ai)}` : "", market != null ? `Market ${pct(market)}` : "", view?.chance_label || "", view?.book ? `at ${view.book}` : ""].filter(Boolean).join(" · ");
  const flags = [
    gone ? badge("No longer offered", "warn", "alert") : "",
    view?.started ? badge("Started", "warn", "clock") : "",
    move ? badge(`Line moved from ${move.was}`, "warn", "alert") : "",
    weak ? badge("Weakest pick", "muted", "alert") : "",
  ].filter(Boolean);
  return `<li class="slip-leg${gone ? " gone" : ""}${weak ? " weak" : ""}">
    <div class="grow">
      <div class="row-between"><b>${esc(label)}</b><b class="num">${esc(gone ? "–" : priceText(odds))}</b></div>
      ${where ? `<div class="muted small">${esc(where)}</div>` : ""}
      ${chances ? `<div class="small">${esc(chances)}</div>` : ""}
      ${flags.length ? `<div class="bx-flags">${flags.join("")}</div>` : ""}
    </div>
    <button type="button" class="icon-btn" data-remove-leg="${esc(leg.key)}" aria-label="Remove ${esc(label)}">${icon("trash", { size: 18 })}</button>
  </li>`;
}

function legsHtml() {
  if (!slip.count) return emptyHtml();
  const { result } = priced();
  const views = new Map();
  (result?.refs || []).forEach((ref, i) => views.set(legKey(ref), result.legs[i]));
  const removed = new Set((result?.removed || []).map(legKey));
  const moved = new Map((result?.moved || []).map((m) => [legKey(m.ref), m]));
  const weakRef = result && result.weakest_leg != null && result.legs.length > 1 ? result.refs?.[result.weakest_leg] : null;
  const weakKey = weakRef ? legKey(weakRef) : null;
  return `<ul class="slip-legs">${slip.legs.map((leg) => legRow(leg, views.get(leg.key), {
    gone: removed.has(leg.key), move: moved.get(leg.key), weak: leg.key === weakKey,
  })).join("")}</ul>
    ${removed.size ? `<button type="button" class="btn btn-secondary small" style="margin-top:8px" data-act="drop-gone">${icon("trash", { size: 16 })}Remove the picks no longer offered</button>` : ""}`;
}

function emptyHtml() {
  // On a break or in the picks pause the builder below is off (and says why), so its shortcuts go
  const offerBuilder = !picksBlock();
  return `<div class="empty">
    <div class="empty-icon">${icon("bets", { size: 28 })}</div>
    <h3>Your slip is empty</h3>
    <p>Tap any price on Today to add a pick${offerBuilder ? ", or let the AI build a ticket for you" : ""}.</p>
    <div class="empty-actions">
      <a class="btn btn-primary" href="#/today">${icon("today", { size: 18 })}Browse games</a>
      ${cleared ? `<button type="button" class="btn btn-secondary" data-act="undo-clear" aria-label="Undo: put back the ${picks(cleared.legs.length)} you had">${icon("refresh", { size: 18 })}Put back ${picks(cleared.legs.length)}</button>` : ""}
    </div>
    ${offerBuilder ? `<p class="small muted" style="margin-top:16px">Or let the AI build one ${explain("tiers")}</p>
    <div class="chip-row wrap" style="justify-content:center;margin-top:8px">
      ${chip("Safe", { iconName: "shield", attrs: 'data-prefill-tier="safe"' })}${chip("Risky", { iconName: "scale", attrs: 'data-prefill-tier="risky"' })}${chip("Crazy", { iconName: "zap", attrs: 'data-prefill-tier="crazy"' })}
    </div>` : ""}
    <div class="bx-linkrow" style="justify-content:center"><button type="button" class="btn btn-ghost small" data-act="load-code">${icon("arrow", { size: 16 })}Load a share code</button></div>
  </div>`;
}

function stakeWord() {
  if (slip.type === "single") return slip.count > 1 ? "Stake per pick" : "Stake";
  if (slip.type === "multi") return "Stake";
  return "Stake per bet";
}

function stakeHtml() {
  return `<div class="field" style="margin-top:16px">
    <label class="label" for="bx-stake"><span data-stake-word>${esc(stakeWord())}</span> ${explain("stake")}</label>
    <div class="stake-row">
      ${STAKES.map((v) => `<button type="button" class="chip" data-stake="${v}" aria-pressed="false">$${v}</button>`).join("")}
      <span class="money-input"><span aria-hidden="true">$</span><input id="bx-stake" class="input" type="text" inputmode="decimal" autocomplete="off" placeholder="Other" aria-label="Other stake" aria-describedby="bx-total"></span>
    </div>
    <div class="form-error" role="alert" data-stake-error></div>
    <div class="bx-total" id="bx-total" data-total aria-live="polite"></div>
  </div>`;
}

function totalHtml() {
  const { result } = priced();
  const budget = me().budget;
  const parts = [];
  if (result?.bets > 1) parts.push(`${result.bets} bets × ${money(result.unit_stake, { cents: true })} = <b>${money(result.total_stake, { cents: true })}</b> in all`);
  else if (result?.bets === 1) parts.push(`Costs <b>${money(result.total_stake, { cents: true })}</b>`);
  if (budget?.left != null) parts.push(`${money(budget.left, { cents: true })} left of your weekly budget ${explain("bankroll")}`);
  return parts.join(" · ");
}

function asOfText(result) {
  const times = Object.values(result?.odds_as_of || {}).filter(Boolean);
  if (!times.length) return "";
  const oldest = times.reduce((a, b) => (new Date(a) < new Date(b) ? a : b));
  return `<p class="method">${icon("clock", { size: 14 })} Prices from the odds feed, ${esc(ago(oldest))}. Nothing is bet here. ${explain("quota", { label: "How fresh are the odds?" })}</p>`;
}

function summaryHtml() {
  if (!slip.count) return "";
  const block = breakBlock();
  if (block) return blockNote({ ...block, text: `${block.text} Pricing, analysing and saving a slip are off until then.` }, BLOCK_ID);
  const { busy, result, error } = priced();
  if (!result) {
    if (error) {
      const hint = gateHint(error);
      if (hint?.kind === "age") {
        return `${notice(error.message, "warn")}<button type="button" class="btn btn-primary small" style="margin-top:8px" data-act="confirm-age">Confirm age</button>`;
      }
      return `${notice(error.message, hint ? "info" : "warn")}${hint ? "" : `<button type="button" class="btn btn-secondary small" style="margin-top:8px" data-act="reprice">${icon("refresh", { size: 16 })}Price again</button>`}`;
    }
    return `<div style="margin-top:14px">${skeleton("lines", 4)}</div>`;
  }
  return `<div class="${busy ? "bx-stale" : ""}"${busy ? ' aria-busy="true"' : ""}>
    ${(result.reasons || []).map((r) => notice(r, "danger")).join("")}
    ${result.bets ? `<div style="margin-top:14px">${chanceTiles(result)}</div>${summaryRows(result)}${outcomesTable(result)}` : ""}
    ${warningsList(result.warnings)}
    ${asOfText(result)}
  </div>`;
}

function undoHtml() {
  if (!replacePending()) return "";
  return `<div class="bx-saved bx-undo"><span>Replaced your ${picks(cleared.legs.length)}${cleared.what ? ` with ${esc(cleared.what)}` : ""}.</span>
    <button type="button" class="btn btn-ghost small" data-act="undo-clear" aria-label="Undo: put back the ${picks(cleared.legs.length)} you had">${icon("refresh", { size: 16 })}Undo</button></div>`;
}

function actionsHtml(analysing) {
  if (!slip.count) return "";
  const { result } = priced();
  const isSaved = saved && saved.text === JSON.stringify(slip.body());
  const off = breakBlock() ? ` disabled aria-describedby="${BLOCK_ID}"` : ""; // the line in place of the numbers
  const canSave = !off && Boolean(result?.valid && result.bets > 0);
  // Inside Telegram the main button is Analyse, except on a break: then this one shows, switched off
  const analyse = off ? `<button type="button" class="btn btn-primary bx-main" data-act="analyse"${off}>${icon("ask", { size: 18 })}Analyse with AI</button>`
    : inTelegram ? "" : `<button type="button" class="btn btn-primary bx-main" data-act="analyse"${analysing ? " disabled" : ""}>${icon("ask", { size: 18 })}${analysing ? "Analysing…" : "Analyse with AI"}</button>`;
  return `${undoHtml()}<div class="bx-actions">
      ${analyse}
      <button type="button" class="btn btn-secondary" data-act="save"${isSaved ? " disabled" : off || (canSave ? "" : " disabled")}>${icon(isSaved ? "check" : "bets", { size: 18 })}${isSaved ? "Saved" : "Save"}</button>
      <button type="button" class="btn btn-secondary" data-act="share"${isSaved ? "" : off || (canSave ? "" : " disabled")}>${icon("share", { size: 18 })}Share</button>
      <button type="button" class="btn btn-secondary" data-act="clear">${icon("trash", { size: 18 })}Clear</button>
    </div>
    ${isSaved ? `<div class="bx-saved">${icon("check", { size: 16 })}<span>Saved to <a href="#/bets">My bets</a> · share code <b>${esc(saved.view.share_code || "")}</b></span>
      <button type="button" class="btn btn-ghost small" data-act="copy-code">${icon("copy", { size: 16 })}Copy</button></div>` : ""}
    <div class="bx-linkrow"><button type="button" class="btn btn-ghost small" data-act="load-code">${icon("arrow", { size: 16 })}Load a share code</button></div>`;
}

// --- share codes ---------------------------------------------------------------------------------------

function shareLink(code) {
  const base = String(me().public_url || location.origin).replace(/\/+$/, "");
  return `${base}/#/build?code=${encodeURIComponent(code)}`;
}

function openShare(view, { justSaved = false } = {}) {
  const code = view.share_code;
  if (!code) {
    toast("This bet has no share code", { kind: "warn" });
    return;
  }
  const link = shareLink(code);
  const s = sheet.open({
    title: "Share this slip",
    className: "bx-sheet",
    body: `${justSaved ? `<p class="hint" style="margin-top:0">${icon("check", { size: 14 })} Saved to My bets, where it is graded when the games end.</p>` : ""}
      <p class="lead">Anyone using this app can load your picks with this code. Prices are checked again when they open it.</p>
      <div class="bx-code" aria-label="Share code ${esc(code.split("").join(" "))}">${esc(code)}</div>
      <p class="hint">${esc(link)}</p>`,
    footer: `<button type="button" class="btn btn-secondary grow" data-share="code">${icon("copy", { size: 18 })}Copy code</button>
      <button type="button" class="btn btn-secondary grow" data-share="link">${icon("copy", { size: 18 })}Copy link</button>
      ${navigator.share ? `<button type="button" class="btn btn-primary grow" data-share="native">${icon("share", { size: 18 })}Share…</button>` : ""}`,
  });
  s.foot.addEventListener("click", async (event) => {
    const how = event.target.closest("[data-share]")?.dataset.share;
    if (how === "code") copyText(code, "Code");
    else if (how === "link") copyText(link, "Link");
    else if (how === "native") {
      try {
        await navigator.share({ title: "My SportsBet AI slip", text: `My slip on SportsBet AI: code ${code}`, url: link });
      } catch {
        /* the person closed the share sheet */
      }
    }
  });
}

/** Load a shared slip into this one (GET /api/slips/share/<code>). Resolves true when loaded. */
async function loadCode(code, { signal } = {}) {
  const clean = String(code || "").trim().toUpperCase();
  if (!/^[A-Z0-9]{4,16}$/.test(clean)) throw new Error("A share code is 4 to 16 letters and numbers.");
  const data = await api(`/api/slips/share/${encodeURIComponent(clean)}`, { signal });
  const legs = (data.legs || []).map((ref, i) => ({ ref, label: data.labels?.[i] || "Pick", league: ref.league }));
  if (!legs.length) throw new Error("That slip has no picks.");
  const same = slip.count === legs.length && legs.every((leg) => slip.has(leg.ref));
  const had = slip.count;
  if (had && !same) {
    const ok = await confirmSheet({
      title: "Replace your slip?",
      body: `Code ${clean} has ${picks(legs.length)}. Loading it replaces the ${had} on your slip (Undo puts them back).`,
      confirm: "Replace",
    });
    if (!ok) return false;
  }
  replaceSlip(() => {
    for (const leg of legs) slip.add(leg);
    slip.set({ type: data.type, stake: String(data.stake), k: data.k ?? null, system: data.system ?? null });
  }, `code ${clean}`);
  haptic.notify("success");
  toast(`Loaded slip ${clean}${data.mine ? " (one of yours)" : ""}${replacePending() ? " · Undo is on the slip" : ""}`, { kind: "good" });
  return true;
}

function openLoadCode() {
  const s = sheet.open({
    title: "Load a share code",
    className: "bx-sheet",
    body: `<form data-load-form novalidate>
        <label class="label" for="bx-code-input">Share code</label>
        <input id="bx-code-input" class="input" type="text" maxlength="16" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="e.g. 7KD2QX" enterkeyhint="go">
        <div class="form-error" role="alert" data-load-error></div>
        <p class="hint">Someone shared a slip with you? Its picks replace what's on your slip, and prices are checked again.</p>
      </form>`,
    footer: '<button type="button" class="btn btn-primary block" data-load-go>Load slip</button>',
  });
  const input = s.body.querySelector("#bx-code-input");
  const error = s.body.querySelector("[data-load-error]");
  setTimeout(() => input.focus(), 60);
  const go = async () => {
    const button = s.foot.querySelector("[data-load-go]");
    button.disabled = true;
    error.textContent = "";
    try {
      await loadCode(input.value);
      if (s.open) s.close();
    } catch (err) {
      if (isQuiet(err)) return;
      if (s.open) error.textContent = err.message;
      else toast(err.message, { kind: "error" });
    } finally {
      button.disabled = false;
    }
  };
  s.body.querySelector("[data-load-form]").addEventListener("submit", (event) => {
    event.preventDefault();
    go();
  });
  s.foot.addEventListener("click", (event) => {
    if (event.target.closest("[data-load-go]")) go();
  });
}

// --- the screen ----------------------------------------------------------------------------------------

/**
 * Draw the slip screen in ``root``. Returns {mainButton(), stop(), ready} for build.js:
 * mainButton() is Telegram's "Analyse with AI"; stop() unsubscribes everything when the person leaves.
 */
export function mountSlip(root, { params, refresh }) {
  const signal = params.signal;
  root.innerHTML = `<div class="page-head"><h1>Build</h1><p>Price any slip honestly: the real chance, the payout, what the bookmaker keeps. Nothing is bet here.</p></div>
    <div data-modes></div>
    <div class="bx-sub" data-sub></div>
    <section class="card section bx-slip" id="slip" aria-labelledby="bx-slip-title">
      <div class="section-head"><h2 id="bx-slip-title" data-slip-title>Your slip</h2><span data-slip-odds></span></div>
      <div data-legs></div>
      <div data-stake-box>${stakeHtml()}</div>
      <div data-summary aria-live="polite"></div>
      <div data-actions></div>
      <div data-action-note></div>
    </section>
    <div data-analysis></div>
    <div data-builder></div>`;
  const $ = (sel) => root.querySelector(sel);
  const stakeInput = $("#bx-stake");
  let stakeTimer = null;

  const analysis = createAnalysis($("[data-analysis]"), {
    signal,
    onBusy: () => {
      drawActions();
      refresh();
    },
    onApply: (alt) => {
      toast(`Applied: ${alt.title}`, { kind: "good" });
      reveal($("#slip"));
    },
  });
  const builder = createBuilder($("[data-builder]"), { signal, replaceSlip });

  function drawModes() {
    redraw($("[data-modes]"), modesHtml(slip.type));
    redraw($("[data-sub]"), subHtml());
  }
  function drawHead() {
    const { result } = priced();
    $("[data-slip-title]").textContent = slip.count ? `Your slip (${slip.count})` : "Your slip";
    $("[data-slip-odds]").innerHTML = slip.count && result?.odds && result.valid && !breakBlock() ? `<b class="num">${esc(priceText(result.odds))}</b>` : "";
  }
  function syncStake() {
    const box = $("[data-stake-box]");
    box.hidden = !slip.count;
    $("[data-stake-word]").textContent = stakeWord();
    for (const b of root.querySelectorAll("[data-stake]")) {
      const on = slip.stake === b.dataset.stake;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
    if (document.activeElement !== stakeInput) stakeInput.value = STAKES.includes(slip.stake) ? "" : slip.stake;
    $("[data-total]").innerHTML = totalHtml();
  }
  function drawActions() {
    redraw($("[data-actions]"), actionsHtml(analysis.busy));
  }
  function drawPriced() {
    drawHead();
    redraw($("[data-legs]"), legsHtml());
    redraw($("[data-summary]"), summaryHtml());
    $("[data-total]").innerHTML = totalHtml();
    drawActions();
  }
  function drawAll() {
    drawModes();
    syncStake();
    drawPriced();
    analysis.draw();
    builder.draw();
  }
  const note = (html) => {
    $("[data-action-note]").innerHTML = html;
  };

  // Telegram's second bottom button clears the slip (never shown over a sheet)
  const syncSecondary = () => {
    if (!inTelegram) return;
    setSecondaryButton(slip.count && !sheet.isOpen ? { text: "Clear", onClick: clearSlip } : null);
  };

  const setStake = (value, { quiet = false } = {}) => {
    const error = $("[data-stake-error]");
    if (!value.trim()) {
      error.textContent = "";
      return;
    }
    const clean = cleanStake(value);
    if (!clean) {
      if (!quiet) error.textContent = "Enter a stake between $0.01 and $100,000 (cents at most).";
      return;
    }
    error.textContent = "";
    if (clean !== slip.stake) slip.set({ stake: clean });
  };

  const save = async () => {
    const body = slip.body();
    const text = JSON.stringify(body);
    note("");
    const button = $('[data-act="save"]');
    if (button) button.disabled = true;
    try {
      const view = await api("/api/slips", { method: "POST", body, signal });
      saved = { text, view };
      haptic.notify("success");
      toast("Saved to My bets", { kind: "good" });
      if (root.isConnected) drawActions();
      if (me().budget) loadMe().catch(() => null); // the weekly budget left has changed
      return view;
    } catch (error) {
      if (isQuiet(error) || !root.isConnected) return null;
      haptic.notify("error");
      const hint = gateHint(error);
      note(`${notice(error.message, "warn")}${hint?.kind === "age" ? '<button type="button" class="btn btn-primary small" style="margin-top:8px" data-act="confirm-age">Confirm age</button>' : ""}`);
      drawActions();
      return null;
    }
  };

  // A share code belongs to a saved slip, so sharing an unsaved one saves it: ask first, since a saved
  // slip is graded into the person's profit and loss and its stake counts toward their weekly budget
  const share = async () => {
    if (saved && saved.text === JSON.stringify(slip.body())) {
      openShare(saved.view);
      return;
    }
    const total = priced().result?.total_stake;
    const ok = await confirmSheet({
      title: "Save and share?",
      body: `Sharing saves this slip to My bets first: a share code belongs to a saved slip. It's graded when its games end and counts in your profit and loss${me().budget && total ? `, and its ${money(total, { cents: true })} counts toward your weekly budget` : ""}.`,
      confirm: "Save & share",
    });
    if (!ok || !root.isConnected) return;
    const view = await save();
    if (view && root.isConnected) openShare(view, { justSaved: true });
  };

  root.addEventListener("click", (event) => {
    const mode = event.target.closest("[data-mode]");
    if (mode) {
      chooseMode(mode.dataset.mode);
      return;
    }
    const k = event.target.closest("[data-k]");
    if (k) {
      slip.set({ k: Number(k.dataset.k) });
      haptic.select();
      return;
    }
    const system = event.target.closest("[data-system]");
    if (system) {
      slip.set({ system: system.dataset.system });
      haptic.select();
      return;
    }
    const stake = event.target.closest("[data-stake]");
    if (stake) {
      $("[data-stake-error]").textContent = "";
      slip.set({ stake: stake.dataset.stake });
      haptic.select();
      return;
    }
    const remove = event.target.closest("[data-remove-leg]");
    if (remove) {
      slip.remove(remove.dataset.removeLeg);
      haptic.select();
      return;
    }
    const tier = event.target.closest("[data-prefill-tier]");
    if (tier && !picksBlock()) {
      builder.prefill({ tier: tier.dataset.prefillTier });
      return;
    }
    // The analysis card handles its own buttons
    const actor = event.target.closest("[data-act]");
    if (!actor || actor.closest("[data-analysis]")) return;
    const act = actor.dataset.act;
    if (act === "analyse") analysis.run();
    else if (act === "save") save();
    else if (act === "share") share();
    else if (act === "clear") clearSlip();
    else if (act === "undo-clear") restore();
    else if (act === "load-code") openLoadCode();
    else if (act === "copy-code" && saved?.view?.share_code) copyText(saved.view.share_code, "Code");
    else if (act === "confirm-age") openOnboarding();
    else if (act === "reprice") priceSlip({ force: true });
    else if (act === "drop-gone") {
      for (const ref of priced().result?.removed || []) slip.remove(legKey(ref));
      haptic.select();
    }
  });
  stakeInput.addEventListener("input", () => {
    clearTimeout(stakeTimer);
    stakeTimer = setTimeout(() => setStake(stakeInput.value, { quiet: true }), 600);
  });
  stakeInput.addEventListener("change", () => {
    clearTimeout(stakeTimer);
    setStake(stakeInput.value);
  });
  stakeInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      clearTimeout(stakeTimer);
      setStake(stakeInput.value);
    }
  });

  undoShown = () => {
    if (root.isConnected) drawActions();
  };
  let breakText = breakBlock()?.text || "";
  const stops = [
    slip.onChange(() => {
      // The person changed the slip after a replace: the Undo for it goes
      if (!replacing && cleared?.reason === "replace" && !replacePending()) cleared = null;
      drawModes();
      syncStake();
      drawPriced();
      analysis.draw();
      builder.draw();
      syncSecondary();
      refresh();
    }),
    onPriced(() => {
      drawPriced();
      redraw($("[data-sub]"), subHtml());
    }),
    onMeChange(() => {
      $("[data-total]").innerHTML = totalHtml();
      if (!slip.count) redraw($("[data-legs]"), legsHtml()); // the empty slip's builder shortcuts
      builder.sync(); // a break or the picks pause started or ended
      // A break started, ended or moved: the numbers or the line, Analyse and Save follow (the shell prices
      // the slip again, so a price from before the break doesn't come back)
      const text = breakBlock()?.text || "";
      if (text !== breakText) {
        breakText = text;
        drawPriced();
        analysis.draw();
        refresh();
      }
    }),
    sheet.onToggle(syncSecondary),
  ];

  drawAll();
  syncSecondary();
  priceSlip();

  // Deep links: ?code= loads a shared slip, ?analyse=1 runs the analysis; both once (the address is
  // tidied so a reload or Back doesn't load or spend AI credits again)
  const code = params.query.get("code");
  const wantAnalysis = params.query.get("analyse");
  if (code || wantAnalysis) replaceAddress("#/build");
  const ready = (async () => {
    if (code) {
      try {
        await loadCode(code, { signal });
      } catch (error) {
        if (!isQuiet(error) && root.isConnected) toast(error.message, { kind: "error" });
      }
    }
    if (wantAnalysis && slip.count) {
      await priceSlip();
      if (root.isConnected) analysis.run();
    }
  })();

  return {
    ready,
    mainButton() {
      if (!slip.count || breakBlock()) return null; // on a break the page's own Analyse shows, switched off
      return { text: analysis.busy ? "Analysing…" : "Analyse with AI", onClick: () => analysis.run(), disabled: analysis.busy, progress: analysis.busy, analyse: true };
    },
    stop() {
      stops.forEach((fn) => fn());
      undoShown = null;
      clearTimeout(stakeTimer);
      analysis.stop();
      if (inTelegram) setSecondaryButton(null);
    },
  };
}
