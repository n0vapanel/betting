// Ask AI's "pasted" card: a bet slip the person pasted, as the app read, matched and priced it.
//
// The server sends {type: "pasted", source, pasted: PastedView, analysis: PastedAnalysis} (dashboard/lab.py:
// read_pasted, analyse_pasted). Every number here is the server's: this file lays them out and, when the
// person switches a leg off or changes the stake, asks POST /api/pasted/<id>/analyse again (code only,
// free, debounced) and shows that answer. Nothing goes on the Build slip without the person's tap, the
// stake field never suggests an amount, and every pasted or server word is escaped.
//
// A read the server doubts (needs_confirmation) shows what was read and why, with "Yes, that's right" and
// "Let me fix it", before any numbers.

import { esc, pct, chanceText, money, signedMoney, priceText, timeLabel, shortName, badge, notice, explain } from "../../ui.js";
import { api } from "../../api.js";
import { slip, legKey } from "../../app.js";
import { icon } from "../../icons.js";

export const CONFIRM_CHIP = "Yes, that's right";
export const FIX_CHIP = "Let me fix it";
const LOADABLE = ["h2h", "spreads", "totals"]; // what Build's slip prices
const STAKE = /^\d{1,6}(\.\d{1,2})?$/;
const TOGGLE_MS = 250;
const STAKE_MS = 650;
const MARKETS = { moneyline: "Moneyline", spread: "Spread", total: "Total", player_prop: "Player prop", other: "Other market" };
const SOURCES = { text: "Read by the AI from your text", image: "Read by the AI from your screenshot", basic: "Read by the free reader (code only, no AI)" };

// --- money in the slip's own currency -----------------------------------------------------------------

/** The slip's currency as a prefix: "£", "€", "KSh " (a word gets a space); "$" when none was read. */
function symbol(currency) {
  const c = String(currency || "").trim();
  if (!c || c === "$") return "$";
  return c.length <= 1 ? c : `${c} `;
}

/** "£5.00", "KSh 200.00", "$10.00" (money() then the slip's own currency). */
export function cash(value, currency, { cents = true } = {}) {
  const text = money(value, { cents });
  return text === "–" ? text : text.replace("$", symbol(currency));
}

function signedCash(value, currency) {
  const text = signedMoney(value);
  return text === "–" ? text : text.replace("$", symbol(currency));
}

// --- the card's own state (kept on the card object for this visit) --------------------------------------

/** What the person changed on a card: legs left out, a stake or bet type of their own, the latest analysis. */
function uiOf(card) {
  if (!card.ui) {
    const a = card.analysis || {};
    // A card from the AI's re-check already carries the person's changes (legs left out, stake, bet type)
    const tool = card.source === "pasted_slip_analyse";
    card.ui = {
      drop: (a.drop || []).map(Number).filter(Number.isInteger),
      custom: tool ? { stake: a.stake || null, type: a.ticket_type || null, k: a.k ?? null, system: a.system ?? null } : {},
      stakeText: null, stakeError: "", analysis: null, busy: false, error: "", seq: 0, timer: null, controller: null,
      confirmed: false, answered: false, note: "", altsOpen: false,
    };
  }
  return card.ui;
}

function analysisOf(card) {
  return card.ui?.analysis || card.analysis || null;
}

/** The slip as read: the latest analysis's view (current prices), else the card's own. */
function viewOf(card) {
  return analysisOf(card)?.pasted || card.pasted || {};
}

function pastedId(card) {
  return viewOf(card).id ?? card.pasted?.id ?? null;
}

/** The read is doubtful and not confirmed yet (in the pause the numbers show with the doubts). */
function asking(card) {
  const original = card.pasted || {};
  const a = analysisOf(card);
  return Boolean(original.needs_confirmation && !original.confirmed && !card.ui?.confirmed && !a?.paused_until);
}

// --- words ----------------------------------------------------------------------------------------------

function gameText(game, league) {
  const [away, home] = String(game || "").split(" @ ");
  return home ? `${shortName(away, league)} @ ${shortName(home, league)}` : String(game || "");
}

/** "3-leg multi", "Single", "Round robin by 2s (4 legs)", "Trixie (3 legs)", "4 legs, bet type not stated". */
function kindWords(view) {
  const n = (view.legs || []).length;
  const legs = `${n} leg${n === 1 ? "" : "s"}`;
  switch (view.bet_type) {
    case "single": return n === 1 ? "Single" : `${n} singles`;
    case "multi": return `${n}-leg multi`;
    case "round_robin": return `Round robin${view.k ? ` by ${view.k}s` : ""} (${legs})`;
    case "system": {
      const name = String(view.system || "System bet").replace(/_/g, " ");
      return `${name.charAt(0).toUpperCase()}${name.slice(1)} (${legs})`;
    }
    default: return `${legs}, bet type not stated`;
  }
}

/** How the analysis prices it: "3-leg multi · £5.00", "4 singles · £2.50 each", "Round robin by 2s · 6 bets of £1.66". */
function pricedAs(ticket, currency) {
  if (!ticket?.type) return "";
  const legs = (ticket.legs || []).length;
  const stake = cash(ticket.unit_stake, currency);
  if (ticket.type === "multi") return legs === 1 ? `Single · ${stake}` : `${legs}-leg multi · ${stake}`;
  if (ticket.type === "single") return legs === 1 ? `Single · ${stake}` : `${legs} singles · ${stake} each`;
  const name = ticket.type === "round_robin" ? `Round robin by ${ticket.k || 2}s` : String(ticket.system || "System").replace(/_/g, " ");
  return `${name} · ${ticket.bets || "?"} bets of ${stake}`;
}

function stakeWords(view) {
  if (view.stake) return `${cash(view.stake, view.currency)} stake`;
  if (view.total_stake) return `${cash(view.total_stake, view.currency)} in all`;
  return "no stake read";
}

/** The slip's line as the card's header says it: "Your slip · Bet365 · 3-leg multi · £5.00 stake". */
export function pastedTitle(view) {
  return ["Your slip", view.bookmaker, kindWords(view), stakeWords(view)].filter(Boolean).join(" · ");
}

// --- one leg ----------------------------------------------------------------------------------------------

function status(kind, mark, text) {
  return `<span class="ask-match ${kind}">${mark ? `<span aria-hidden="true">${mark}</span>` : ""}${esc(text)}</span>`;
}

function statusHtml(leg) {
  const m = leg.match || {};
  const game = m.game ? gameText(m.game, m.league) : "";
  const when = m.start ? timeLabel(m.start) : "";
  let html;
  if (m.status === "matched") html = status("good", "✓", [game, when].filter(Boolean).join(" · ") || "Found on the board");
  else if (m.status === "line_differs") html = status("warn", "≈", `Line differs${game ? ` · ${game}` : ""}`);
  else if (m.status === "prop") html = status("info", "", `Player prop${game ? ` · ${game}` : ""}${game && when ? ` · ${when}` : ""}`);
  else if (m.status === "unsupported") html = status("muted", "?", `Market the app doesn't price${game ? ` · ${game}` : ""}`);
  else if (m.status === "no_odds") html = status("danger", "!", "Odds couldn't be read");
  else html = status("muted", "?", "Not found on the app's boards");
  return `${html}${m.live ? ` ${badge("Started", "danger", "clock")}` : ""}`;
}

function oddsHtml(leg) {
  if (!leg.odds) return `<span class="ask-pleg-odds"><b class="num ask-leg-odds">–</b></span>`;
  const shown = priceText(leg.odds);
  const written = String(leg.odds_text || "").trim();
  const plain = written.replace(/^@\s*/, "");
  const same = !written || plain === shown || (Number.isFinite(Number(plain)) && Number(plain) === Number(shown));
  const unchecked = leg.verified === false && leg.source === "text";
  return `<span class="ask-pleg-odds"><b class="num ask-leg-odds">${esc(shown)}</b>${same ? "" : `<span class="small muted">as written ${esc(written)}</span>`}${unchecked ? badge("Check", "warn", "alert") : ""}</span>`;
}

function chanceHtml(leg) {
  if (!leg.odds) return "";
  if (leg.chance_source === "estimated") {
    return leg.market_chance == null ? "" : `<div class="small ask-leg-chance">Estimated ${esc(pct(leg.market_chance))} <span class="muted">(your odds less a typical margin)</span></div>`;
  }
  const parts = [leg.ai_chance != null ? `AI ${pct(leg.ai_chance)}` : "", leg.market_chance != null ? `Market ${pct(leg.market_chance)}` : ""].filter(Boolean);
  return parts.length ? `<div class="small ask-leg-chance">${esc(parts.join(" · "))}</div>` : "";
}

function notesHtml(notes) {
  if (!notes?.length) return "";
  const first = notes.slice(0, 2);
  const rest = notes.slice(2);
  return `<ul class="ask-notes">${first.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    ${rest.length ? `<details class="ask-more"><summary>${rest.length} more note${rest.length === 1 ? "" : "s"}</summary>
      <ul class="ask-notes">${rest.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></details>` : ""}`;
}

function legsHtml(card, key, { reading }) {
  const view = viewOf(card);
  const a = analysisOf(card);
  const ui = uiOf(card);
  const notes = new Map((a?.legs || []).map((n) => [Number(n.index), n.notes || []]));
  const ticket = a?.ticket || {};
  const weakIndex = (ticket.legs || []).length > 1 && ticket.weakest_leg != null ? ticket.legs[ticket.weakest_leg]?.slip_index : null;
  const included = (view.legs || []).filter((leg) => leg.odds && !ui.drop.includes(leg.index)).length;
  return `<ol class="ask-legs ask-plegs">${(view.legs || []).map((raw) => {
    const leg = { ...raw, source: view.source };
    const on = Boolean(leg.odds) && !ui.drop.includes(leg.index);
    const market = MARKETS[leg.market] || "Bet";
    const toggle = reading ? "" : `<label class="ask-pleg-toggle">
        <input type="checkbox" role="switch" data-pleg-toggle="${esc(key)}:${esc(leg.index)}"${on ? " checked" : ""}${!leg.odds ? " disabled" : ""}
          aria-label="${esc(`Leg ${leg.index}, ${leg.label || leg.selection}: ${on ? "in the bet" : "left out"}`)}">
        <span aria-hidden="true">${!leg.odds ? "Left out: odds not read" : on ? "In the bet" : "Left out"}</span></label>`;
    const best = leg.better_price && leg.best_price
      ? `<p class="ask-better">${icon("trend", { size: 14 })}Better price: ${esc(leg.best_price.decimal)}${leg.best_price.book ? ` at ${esc(leg.best_price.book)}` : ""}</p>` : "";
    return `<li class="ask-leg ask-pleg${on || reading ? "" : " off"}${!reading && leg.index === weakIndex && on ? " weak" : ""}" data-pleg="${esc(leg.index)}">
      <div class="ask-pleg-top">
        <span class="ask-pleg-n" aria-hidden="true">${esc(leg.index)}</span>
        <div class="grow"><b class="ask-leg-label">${esc(leg.label || leg.selection || leg.raw)}</b>
          <span class="muted small ask-pleg-market">${esc(market)}${leg.raw && leg.raw !== leg.label ? ` · “${esc(String(leg.raw).slice(0, 120))}”` : ""}</span></div>
        ${oddsHtml(leg)}
      </div>
      <div class="ask-pleg-status">${statusHtml(leg)}${!reading && leg.index === weakIndex && on ? ` ${badge("Weakest leg", "warn", "alert")}` : ""}</div>
      ${reading ? "" : chanceHtml(leg)}
      ${reading ? "" : best}
      ${reading ? (leg.match?.status && leg.match.status !== "matched" && leg.match.note ? `<ul class="ask-notes"><li>${esc(leg.match.note)}</li></ul>` : "") : notesHtml(notes.get(leg.index))}
      ${toggle}
    </li>`;
  }).join("")}</ol>
  ${!reading && included <= 1 && (view.legs || []).length > 1 ? `<p class="small muted ask-pleg-last">At least one leg stays in the bet.</p>` : ""}`;
}

// --- the numbers ------------------------------------------------------------------------------------------

function totalsHtml(ticket, currency) {
  if (!ticket?.bets) return "";
  const c = ticket.chance || {};
  const e = ticket.expected || {};
  const ml = ticket.most_likely;
  const both = (ai, market) => [ai, market].filter(Boolean).join("<br>") || "–";
  const profit = c.headline === "profit";
  const fixed = ticket.type === "multi" || ticket.bets === 1;
  const every = c.every_leg || {};
  const likely = ml ? (ml.result === "lose all" ? `Lose ${cash(ticket.total_stake, currency)}` : ml.result.charAt(0).toUpperCase() + ml.result.slice(1)) : "";
  return `<dl class="summary ask-summary">
    ${ticket.odds ? `<div><dt>Combined odds</dt><dd class="num">${esc(priceText(ticket.odds))}</dd></div>` : ""}
    <div><dt>Total stake</dt><dd class="num">${esc(cash(ticket.total_stake, currency))}${ticket.bets > 1 ? ` <span class="muted">(${esc(ticket.bets)} bets)</span>` : ""}</dd></div>
    <div><dt>Most it pays ${explain("payout")}</dt><dd class="num">${fixed ? "" : "up to "}${esc(cash(ticket.max_payout, currency))}</dd></div>
    <div><dt>${profit ? "Chance of a profit" : "Chance it all wins"} ${explain("one_in")}</dt><dd class="num">${both(
      c.ai != null ? `AI ${esc(chanceText(c.ai, c.ai_one_in))}` : "", c.market != null ? `Market ${esc(chanceText(c.market, c.market_one_in))}` : "")}</dd></div>
    ${profit && (every.ai != null || every.market != null) ? `<div><dt>Every leg wins</dt><dd class="num">${both(
      every.ai != null ? `AI ${esc(pct(every.ai))}` : "", every.market != null ? `Market ${esc(pct(every.market))}` : "")}</dd></div>` : ""}
    <div><dt>Expected result ${explain("expected_result")}</dt><dd class="num">${both(
      e.ai != null ? `${esc(signedCash(e.ai, currency))} <span class="muted">AI</span>` : "", e.market != null ? `${esc(signedCash(e.market, currency))} <span class="muted">market</span>` : "")}</dd></div>
    <div><dt>Bookmaker keeps ${explain("book_cut")}</dt><dd class="num">${ticket.book_cut != null ? `about ${esc(pct(ticket.book_cut))}` : "–"}</dd></div>
    ${ml ? `<div><dt>Most likely ${explain("most_likely")}</dt><dd>${esc(likely)} <span class="muted">(${esc(pct(ml.probability))})</span></dd></div>` : ""}
  </dl>`;
}

/**
 * The bookmaker's stated return against the code's, for the slip as pasted. changed: the person changed the
 * slip on the card (a leg out, another stake or bet type), so the line says it is about the slip as pasted.
 */
function returnHtml(view, changed = false) {
  const currency = view.currency;
  const profit = view.return_basis === "profit";
  const lead = (words) => (changed ? `As pasted: ${words.charAt(0).toLowerCase()}${words.slice(1)}` : words);
  if (view.return_check === "matches" && view.stated_return) {
    return `<p class="ask-return good">${icon("check", { size: 16 })}<span>${esc(lead(`Matches your bookmaker's ${profit ? "“to win” of " : ""}${cash(view.stated_return, currency)}`))}</span></p>`;
  }
  if (view.return_check === "differs" && view.stated_return) {
    const ours = view.computed_return ? `; we get ${cash(view.computed_return, currency)}` : ", which the odds read don't give";
    return `<p class="ask-return warn">${icon("alert", { size: 16 })}<span>${esc(lead(`Your bookmaker says ${cash(view.stated_return, currency)}${ours}: check the odds`))}</span></p>`;
  }
  return "";
}

/** The person changed the slip on the card: a leg left out, a stake or a bet type of their own. */
function changedOf(ui) {
  return Boolean(ui.drop.length || ui.custom.stake || ui.custom.type);
}

/** A note on the slip as read that names a leg now left out ("Leg 3 isn't matched…", "Legs 2 and 3…"). */
function namesDropped(note, drop) {
  const m = /\bLegs? (\d+(?:(?:, | and )\d+)*)/i.exec(String(note));
  return Boolean(m) && m[1].match(/\d+/g).map(Number).some((index) => drop.includes(index));
}

function listHtml(items) {
  if (!items.length) return "";
  const item = (w) => `<li>${icon("alert", { size: 14 })}<span>${esc(w)}</span></li>`;
  const first = items.slice(0, 2);
  const rest = items.slice(2);
  return `<ul class="warnings">${first.map(item).join("")}</ul>
    ${rest.length ? `<details class="ask-more"><summary>${rest.length} more note${rest.length === 1 ? "" : "s"}</summary><ul class="warnings">${rest.map(item).join("")}</ul></details>` : ""}`;
}

function alternativesHtml(analysis, key, currency, open) {
  const alts = (analysis?.alternatives || []).filter((alt) => alt?.ticket);
  if (!alts.length) return "";
  return `<details class="ask-alts" data-palts="${esc(key)}"${open ? " open" : ""}><summary>${icon("scale", { size: 16 })}Other ways to play it (${alts.length})</summary>
    <ul class="ask-alt-list">${alts.map((alt, i) => {
      const t = alt.ticket;
      const c = t.chance || {};
      const p = c.ai ?? c.market;
      const oneIn = c.ai != null ? c.ai_one_in : c.market_one_in;
      const exp = t.expected?.ai ?? t.expected?.market;
      const title = alt.title || pricedAs(t, currency);
      return `<li class="ask-alt">
        <div class="grow"><b>${esc(title)}</b>
          <span class="muted small">${esc(pricedAs(t, currency))}${p != null ? ` · ${c.headline === "profit" ? "profit" : "wins"} ${esc(chanceText(p, oneIn))}` : ""}${exp != null ? ` · expected ${esc(signedCash(exp, currency))}` : ""}</span>
          ${alt.summary ? `<span class="small">${esc(alt.summary)}</span>` : ""}</div>
        <button type="button" class="btn btn-secondary small" data-palt="${esc(key)}:${i}" aria-label="${esc(`Try it: ${title}`)}">Try it</button>
      </li>`;
    }).join("")}</ul>
    <p class="small muted">“Try it” re-prices your slip that way here (free). It never raises your stake.</p></details>`;
}

function liveHtml(card, key) {
  const ui = uiOf(card);
  const a = analysisOf(card);
  const view = viewOf(card);
  if (!a) return returnHtml(card.pasted || view);
  const ticket = a.ticket || {};
  const currency = a.currency ?? view.currency;
  const changed = changedOf(ui);
  // The slip-as-read notes about a leg the person left out no longer describe the bet (each leg's own notes do)
  const read = (view.warnings || []).filter((w) => !namesDropped(w, ui.drop));
  const notes = [...new Set([...read, ...(ticket.warnings || [])].filter(Boolean))];
  return `${ui.busy ? `<p class="ask-pbusy small muted" role="status">${icon("refresh", { size: 14 })}Updating the numbers…</p>` : ""}
    ${ui.error ? notice(ui.error, "warn") : ""}
    ${a.pause_note ? notice(a.pause_note, "info") : ""}
    ${(ticket.reasons || []).map((r) => notice(r, "danger")).join("")}
    ${totalsHtml(ticket, currency)}
    ${returnHtml(card.pasted || view, changed)}
    ${listHtml(notes)}
    ${alternativesHtml(a, key, currency, ui.altsOpen)}`;
}

function stakeHtml(card, key) {
  const ui = uiOf(card);
  const a = analysisOf(card);
  if (!a) return "";
  const view = viewOf(card);
  const currency = a.currency ?? view.currency;
  const id = `pstake-${String(key).replace(/[^0-9a-z]/gi, "-")}`;
  const value = ui.stakeText ?? (a.stake || "");
  const changed = changedOf(ui);
  return `<div class="ask-pstake">
    <label class="ask-pstake-label" for="${esc(id)}">Stake per bet</label>
    <div class="ask-pstake-box"><span aria-hidden="true">${esc(symbol(currency).trim())}</span>
      <input id="${esc(id)}" class="input num" type="text" inputmode="decimal" autocomplete="off" enterkeyhint="done" maxlength="9"
        value="${esc(value)}" data-pasted-stake="${esc(key)}" aria-describedby="${esc(id)}-hint"${ui.stakeError ? ' aria-invalid="true"' : ""}></div>
    <span class="small muted ask-pstake-hint" id="${esc(id)}-hint" data-pasted-hint="${esc(key)}">${ui.stakeError ? `<span class="ask-perror">${esc(ui.stakeError)}</span>` : esc(`Priced as ${pricedAs(a.ticket, currency) || "–"} at your odds`)}</span>
    ${changed ? `<button type="button" class="btn btn-ghost small" data-pasted-reset="${esc(key)}">${icon("refresh", { size: 16 })}As pasted</button>` : ""}
  </div>`;
}

// --- the slip on Build ------------------------------------------------------------------------------------

/** Why a leg can't go on Build's slip, or "" when it can. */
function notLoadable(leg, dropped) {
  const m = leg.match || {};
  if (!leg.odds) return "its odds couldn't be read";
  if (dropped) return "you left it out";
  if (m.status === "matched" && m.live) return "its game has started";
  if (m.status === "matched" && m.ref && LOADABLE.includes(m.ref.market)) return "";
  if (m.status === "line_differs") return "the app's line for it is different, so Build would price another bet";
  if (m.status === "prop") return "player props can't go on Build's slip yet";
  if (m.status === "unsupported") return "Build prices moneylines, spreads and totals only";
  return "it isn't among the games the app has odds for";
}

/**
 * The legs a card can load into Build's slip (the matched ones still in the bet), as slip legs, and the
 * others with why: {legs: [{ref, label, game, start, league, price, ai, fair}], skipped: [{index, label, why}]}.
 */
export function loadPlan(card) {
  const view = viewOf(card);
  const ui = uiOf(card);
  const refs = new Set((view.matched_refs || []).map((ref) => legKey(ref)));
  const legs = [];
  const skipped = [];
  for (const leg of view.legs || []) {
    let why = notLoadable(leg, ui.drop.includes(leg.index));
    const ref = leg.match?.ref;
    if (!why && !refs.has(legKey(ref))) why = "it isn't among the games the app has odds for";
    if (why) {
      skipped.push({ index: leg.index, label: leg.label || leg.selection, why });
      continue;
    }
    legs.push({
      ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
      label: leg.label || leg.selection, game: leg.match.game, start: leg.match.start, league: ref.league,
      price: leg.best_price?.decimal ?? leg.odds?.decimal, american: leg.best_price ? null : leg.odds?.american,
      ai: leg.ai_chance ?? null, fair: leg.market_chance ?? null,
    });
  }
  return { legs, skipped };
}

function loadButton(card, key) {
  const { legs } = loadPlan(card);
  if (legs.length && legs.every((leg) => slip.has(leg.ref))) {
    return `<a class="btn btn-secondary small" href="#/build">${icon("check", { size: 16 })}In your slip · Open Build</a>`;
  }
  const label = legs.length ? `Load ${legs.length} matched leg${legs.length === 1 ? "" : "s"} into Build` : "Load matched legs into Build";
  return `<button type="button" class="btn btn-primary small" data-pasted-load="${esc(key)}">${icon("build", { size: 16 })}${esc(label)}</button>`;
}

function actionsHtml(card, key, options) {
  if (asking(card)) return "";
  const view = viewOf(card);
  const ask = options.canAsk ? options.canAsk(view, card) : "";
  return `${loadButton(card, key)}
    <button type="button" class="btn btn-secondary small" data-pasted-fix="${esc(key)}">${icon("edit", { size: 16 })}Fix the reading</button>
    ${ask ? `<button type="button" class="btn btn-secondary small" data-pasted-ask="${esc(key)}">${icon("chat", { size: 16 })}${esc(ask)}</button>` : ""}`;
}

function confirmHtml(card, key, options) {
  const view = card.pasted || {};
  const reasons = (view.confirm_reasons || []).filter(Boolean);
  const answered = card.ui?.answered || options.later?.has?.(view.id);
  return `<div class="ask-confirm" role="group" aria-label="Check what was read">
    <p><b>Is this what your slip says?</b> The numbers come once you've checked.</p>
    ${reasons.length ? `<ul class="warnings">${reasons.map((r) => `<li>${icon("alert", { size: 14 })}<span>${esc(r)}</span></li>`).join("")}</ul>` : ""}
    ${answered ? `<p class="small muted">${icon("check", { size: 14 })} Answered below.</p>`
      : `<div class="ask-actions-row">
        <button type="button" class="btn btn-primary small" data-pasted-yes="${esc(key)}">${icon("check", { size: 16 })}${esc(CONFIRM_CHIP)}</button>
        <button type="button" class="btn btn-secondary small" data-pasted-fix="${esc(key)}">${icon("edit", { size: 16 })}${esc(FIX_CHIP)}</button></div>`}
  </div>`;
}

/**
 * The pasted card. options: {stale, later: Set of pasted ids shown again in later messages,
 * canAsk(view, card) -> the label of an "ask the AI" button, or ""}.
 */
export function pastedCardHtml(card, key, options = {}) {
  const view = viewOf(card);
  const reading = asking(card);
  const head = pastedTitle(card.pasted || view);
  const source = card.source === "pasted_slip_analyse" ? "Your slip, re-checked by the app's code" : SOURCES[view.source] || "Your slip";
  return `<article class="card ask-card ask-pasted" data-card-key="${esc(key)}" aria-label="${esc(head)}">
    <header class="ask-card-head"><div class="grow"><span class="eyebrow">${esc(source)}</span>
      <h3 tabindex="-1" data-pasted-heading>${esc(head)}</h3>
      ${view.notes ? `<span class="muted small">Not placed: “${esc(String(view.notes).slice(0, 300))}”</span>` : ""}</div></header>
    ${reading ? confirmHtml(card, key, options) : ""}
    <div data-pasted-legs="${esc(key)}">${legsHtml(card, key, { reading })}</div>
    ${reading ? "" : stakeHtml(card, key)}
    <div class="ask-plive" data-pasted-live="${esc(key)}"${uiOf(card).busy ? ' aria-busy="true"' : ""}>${reading ? "" : liveHtml(card, key)}</div>
    ${options.stale && !reading ? `<p class="ask-stale">${icon("clock", { size: 14 })}${esc(options.stale)}</p>` : ""}
    <div class="ask-card-actions" data-pasted-actions="${esc(key)}">${actionsHtml(card, key, options)}</div>
  </article>`;
}

/** The parts of a card on screen that change with a new analysis (the stake box keeps its focus and text). */
function redrawParts(root, card, key, options) {
  const box = root?.querySelector(`[data-card-key="${CSS.escape(key)}"]`);
  if (!box) return;
  const active = document.activeElement;
  const focus = active && box.contains(active)
    ? ["data-pleg-toggle", "data-palt", "data-pasted-reset", "data-pasted-load"].find((name) => active.hasAttribute(name)) : null;
  const focusValue = focus ? active.getAttribute(focus) : null;
  const legs = box.querySelector("[data-pasted-legs]");
  if (legs) legs.innerHTML = legsHtml(card, key, { reading: asking(card) });
  const live = box.querySelector("[data-pasted-live]");
  if (live) {
    live.innerHTML = liveHtml(card, key);
    if (uiOf(card).busy) live.setAttribute("aria-busy", "true");
    else live.removeAttribute("aria-busy");
  }
  const actions = box.querySelector("[data-pasted-actions]");
  if (actions) actions.innerHTML = actionsHtml(card, key, options);
  const hint = box.querySelector("[data-pasted-hint]");
  const stakeBox = box.querySelector(".ask-pstake");
  if (stakeBox && hint) {
    // Only the hint and the reset button: the input keeps what is being typed
    const fresh = document.createElement("div");
    fresh.innerHTML = stakeHtml(card, key);
    const next = fresh.querySelector("[data-pasted-hint]");
    if (next) hint.replaceWith(next);
    stakeBox.querySelector("[data-pasted-reset]")?.remove();
    const reset = fresh.querySelector("[data-pasted-reset]");
    if (reset) stakeBox.appendChild(reset);
    const input = stakeBox.querySelector("[data-pasted-stake]");
    if (input) {
      if (uiOf(card).stakeError) input.setAttribute("aria-invalid", "true");
      else input.removeAttribute("aria-invalid");
      // Nothing typed since the last change (or "As pasted" / "Try it"): the box shows the stake priced
      if (uiOf(card).stakeText == null) input.value = analysisOf(card)?.stake || "";
    }
  }
  if (focus && focusValue && !box.contains(document.activeElement)) {
    const again = box.querySelector(`[${focus}="${CSS.escape(focusValue)}"]`);
    if (again && !again.disabled) again.focus({ preventScroll: true });
    else refocus(box, focus);
  }
}

/**
 * The control that had focus went with the redraw ("As pasted" once nothing is changed, an alternative that
 * no longer applies): focus the alternatives' heading, else the stake box on a keyboard, else the card's
 * heading (a phone's keyboard doesn't pop up), so focus never drops to the page.
 */
function refocus(box, focus) {
  const summary = focus === "data-palt" ? box.querySelector("details.ask-alts[open] > summary") : null;
  const stake = matchMedia("(pointer: fine)").matches ? box.querySelector("[data-pasted-stake]") : null;
  (summary || stake || box.querySelector("[data-pasted-heading]"))?.focus({ preventScroll: true });
}

// --- re-pricing (POST /api/pasted/<id>/analyse) -------------------------------------------------------------

/** What changed, in words for a screen reader (the server's numbers). */
function updatedWords(analysis) {
  const t = analysis?.ticket || {};
  const c = t.chance || {};
  const n = (t.legs || []).length;
  const currency = analysis?.currency;
  const chance = c.ai != null ? `${chanceText(c.ai, c.ai_one_in)} by the AI` : c.market != null ? `${chanceText(c.market, c.market_one_in)} by the market` : "";
  const word = c.headline === "profit" ? "chance of a profit" : "chance it all wins";
  return `Updated: ${n} leg${n === 1 ? "" : "s"} in the bet${chance ? `, ${word} ${chance}` : ""}${t.max_payout ? `, pays up to ${cash(t.max_payout, currency)}` : ""}.`;
}

function requestBody(ui) {
  const body = {};
  if (ui.drop.length) body.drop = [...ui.drop].sort((a, b) => a - b);
  if (ui.custom.stake) body.stake = String(ui.custom.stake);
  if (ui.custom.type) {
    body.type = ui.custom.type;
    if (ui.custom.type === "round_robin" && ui.custom.k) body.k = Number(ui.custom.k);
    if (ui.custom.type === "system" && ui.custom.system) body.system = ui.custom.system;
  }
  return body;
}

/** Ask for the numbers again after a change (debounced: one request for a burst of taps or keys). */
function schedule(card, key, ctx, delay) {
  const ui = uiOf(card);
  clearTimeout(ui.timer);
  ui.timer = setTimeout(() => recompute(card, key, ctx), delay);
}

async function recompute(card, key, ctx) {
  const ui = uiOf(card);
  ui.timer = null;
  const id = pastedId(card);
  if (id == null) return;
  ui.busy = true;
  ui.error = "";
  const seq = ++ui.seq;
  ui.controller?.abort();
  const controller = new AbortController();
  ui.controller = controller;
  redrawParts(ctx.root(), card, key, ctx.options(key));
  try {
    const analysis = await api(`/api/pasted/${encodeURIComponent(id)}/analyse`, { method: "POST", body: requestBody(ui), signal: controller.signal });
    if (seq !== ui.seq) return;
    ui.analysis = analysis;
    ui.busy = false;
    ui.controller = null;
    redrawParts(ctx.root(), card, key, ctx.options(key));
    ctx.announce(updatedWords(analysis));
  } catch (error) {
    if (error?.name === "AbortError" || seq !== ui.seq) return;
    ui.busy = false;
    ui.controller = null;
    ui.error = error?.message || "The numbers couldn't be updated. Try again.";
    redrawParts(ctx.root(), card, key, ctx.options(key));
    ctx.announce(`The numbers couldn't be updated: ${ui.error}`);
  }
}

// --- events (ask.js passes them on) ---------------------------------------------------------------------------

/**
 * ctx: {root(), findCard(key), options(key), announce(text), confirm(card, key), fix(card), ask(card, key),
 * load(card, key), redrawCard(key)}. Returns true when the click was the pasted card's.
 */
export function pastedClick(event, ctx) {
  const target = event.target;
  const alt = target.closest("[data-palt]");
  if (alt) {
    const parts = alt.dataset.palt.split(":");
    const index = Number(parts.pop());
    const key = parts.join(":");
    const card = ctx.findCard(key);
    const option = analysisOf(card || {})?.alternatives?.[index];
    if (!card || !option?.ticket) return true;
    const ui = uiOf(card);
    const t = option.ticket;
    ui.altsOpen = true;
    ui.drop = (option.drop || []).map(Number).filter(Number.isInteger);
    ui.custom = { stake: t.unit_stake || null, type: t.type || null, k: t.k ?? null, system: t.system ?? null };
    ui.stakeText = null;
    ui.stakeError = "";
    const input = ctx.root()?.querySelector(`[data-pasted-stake="${CSS.escape(key)}"]`);
    if (input && t.unit_stake) input.value = t.unit_stake;
    ctx.announce(`Trying it: ${option.title || pricedAs(t, analysisOf(card)?.currency)}`);
    recompute(card, key, ctx);
    return true;
  }
  const reset = target.closest("[data-pasted-reset]");
  if (reset) {
    const key = reset.dataset.pastedReset;
    const card = ctx.findCard(key);
    if (!card) return true;
    const ui = uiOf(card);
    ui.drop = [];
    ui.custom = {};
    ui.stakeText = null;
    ui.stakeError = "";
    ctx.announce("Back to your slip as pasted");
    recompute(card, key, ctx);
    return true;
  }
  const actions = [["data-pasted-yes", "confirm"], ["data-pasted-fix", "fix"], ["data-pasted-ask", "ask"], ["data-pasted-load", "load"]];
  for (const [attr, name] of actions) {
    const button = target.closest(`[${attr}]`);
    if (!button) continue;
    const key = button.getAttribute(attr);
    const card = ctx.findCard(key);
    if (card) ctx[name](card, key);
    return true;
  }
  return false;
}

/** A leg switched on or off: the toggle shows it at once, the numbers follow. */
export function pastedChange(event, ctx) {
  const toggle = event.target.closest?.("[data-pleg-toggle]");
  if (!toggle) return false;
  const parts = toggle.dataset.plegToggle.split(":");
  const index = Number(parts.pop());
  const key = parts.join(":");
  const card = ctx.findCard(key);
  if (!card) return true;
  const ui = uiOf(card);
  const view = viewOf(card);
  const leg = (view.legs || []).find((l) => l.index === index);
  const priced = (view.legs || []).filter((l) => l.odds && !ui.drop.includes(l.index));
  if (!toggle.checked && priced.length <= 1) {
    toggle.checked = true; // the server prices at least one leg
    ctx.announce("At least one leg stays in the bet.");
    return true;
  }
  ui.drop = toggle.checked ? ui.drop.filter((i) => i !== index) : [...new Set([...ui.drop, index])];
  const row = toggle.closest(".ask-pleg");
  row?.classList.toggle("off", !toggle.checked);
  const words = toggle.parentElement?.querySelector("span");
  if (words) words.textContent = toggle.checked ? "In the bet" : "Left out";
  toggle.setAttribute("aria-label", `Leg ${index}, ${leg?.label || leg?.selection || ""}: ${toggle.checked ? "in the bet" : "left out"}`);
  ctx.announce(`Leg ${index} ${toggle.checked ? "back in the bet" : "left out"}. Updating the numbers…`);
  schedule(card, key, ctx, TOGGLE_MS);
  return true;
}

/** "Other ways to play it" opened or closed (a toggle event, caught on the way down): redraws keep it so. */
export function pastedToggle(event, ctx) {
  const details = event.target;
  if (!(details instanceof HTMLDetailsElement) || !details.matches("[data-palts]")) return false;
  const card = ctx.findCard(details.dataset.palts);
  if (card) uiOf(card).altsOpen = details.open;
  return true;
}

/** The stake typed: checked here, then priced once the typing stops. */
export function pastedInput(event, ctx) {
  const input = event.target.closest?.("[data-pasted-stake]");
  if (!input) return false;
  const key = input.dataset.pastedStake;
  const card = ctx.findCard(key);
  if (!card) return true;
  const ui = uiOf(card);
  const value = input.value.trim().replace(/^[^\d]+/, "").replace(/,/g, "");
  ui.stakeText = input.value;
  const amount = Number(value);
  ui.stakeError = !value ? "Type a stake per bet, e.g. 10" : !STAKE.test(value) || !(amount >= 0.01) ? "A stake looks like 10 or 2.50" : "";
  const hint = ctx.root()?.querySelector(`[data-pasted-hint="${CSS.escape(key)}"]`);
  if (ui.stakeError) {
    clearTimeout(ui.timer);
    input.setAttribute("aria-invalid", "true");
    if (hint) hint.innerHTML = `<span class="ask-perror">${esc(ui.stakeError)}</span>`;
    return true;
  }
  input.removeAttribute("aria-invalid");
  ui.custom = { ...ui.custom, stake: value };
  schedule(card, key, ctx, STAKE_MS);
  return true;
}

/** After the slip changed: the cards' "Load into Build" buttons say what's already there. */
export function syncPastedButtons(root, findCard, options) {
  for (const box of root.querySelectorAll("[data-pasted-actions]")) {
    const key = box.dataset.pastedActions;
    const card = findCard(key);
    if (card) box.innerHTML = actionsHtml(card, key, options(key));
  }
}

/** Show a doubtful read's numbers on the card itself (no chat turn: the free reader's cards, older cards). */
export function revealPasted(card) {
  uiOf(card).confirmed = true;
}

/** The chat turn answering "Is that right?" is under way: the question card says it was answered. */
export function markAnswered(card) {
  uiOf(card).answered = true;
}

/** The pasted text a card was read from (to fix it), or "". */
export function pastedText(card) {
  return String(viewOf(card).text || card.pasted?.text || "");
}
