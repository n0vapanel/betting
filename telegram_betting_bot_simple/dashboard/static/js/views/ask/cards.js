// Ask AI's cards: the tickets, rollover plans, players and likeliest winners a reply carries (a pasted slip's
// card is ask/pasted.js).
//
// The server builds every card in code from its tool results (ai/tools.py: ticket_card, plan_card,
// player_card, builder_card, picks_card), so each number here is the brain's: these functions only lay them out. A
// ticket card's "Add to slip" adds its legs only when the person taps it; "Tweak" asks the AI for a change.
//
// Closest effort (v3): a Builder ticket an option picked is titled by it ("Likeliest at your odds"), and shows its
// fit line ("3.52x · 27% (you asked 3.5x · 80%)") and a muted "Relaxed: …" line (card.option, card.fit,
// card.relaxed). brain_top_picks' card (type "picks") is the likeliest winners, ranked, each with its chance,
// price, book, a short why and its own "Add to slip".
//
// The player scan (v3): players_scan's card (type "player_scan", ai/tools.py scan_card) ranks players for a stat and a
// number ("Projected for 20+ points · NBA · tonight"): each row's name, team v opponent and tip time, an injury chip
// (Q, GTD), the projection, the chance with a small bar, the hits ("9/10" L10, "30/45" season), the posted line and
// price, the model ("Recent form" for the form model), "View player" (and "Add to slip" only where the row allows
// it). "Also close" is collapsed; a board still loading says "8 of 10 games loaded" with a Refresh chip that runs
// the scan again (ask.js: GET /api/players/scan, no AI allowance) and redraws the card in place.
// P0: the first five rows, then the rest that qualify under "Also:" and "N more qualify: open the Player finder" (as
// the chat's answer says them); a yes/no stat shows only its chance; rows ranked by edge (posted lines, "best edges"
// across every stat, each row with its own stat) show the edge beside the chance; hot and cold carry the trend chip
// ("L5 28.4 vs 24.1 season", or "… last season"); a name filter says whether each player qualifies and has a note for
// each name it couldn't rank (ambiguous_names, started_names, missing_names); and the board's state is said (games
// still missing data, games already started, ESPN's scoreboard down). The card's own ``sort`` and ``names`` (when the
// server sends them) keep Refresh and "Open in Player finder" on the same scan.

import { esc, pct, chanceText, money, signedMoney, priceText, shortName, timeLabel, ago, badge, injuryBadge, notice, explain } from "../../ui.js";
import { slip } from "../../app.js";
import { icon } from "../../icons.js";
import { pastedCardHtml } from "./pasted.js";
import { optionOf, fitParts, relaxedText } from "../build/builder/fit.js";
import { unitsFor } from "../context/units.js";
import {
  rowItemHtml, rowLeg, ensureScanStyles, finderHash, scanLeague, statOf, askText, whenText, rowListsHtml, alsoLabel, moreText,
  nameNotes, boardNotes, allStarted, notesHtml, statusText, cleanNames, TOP, TREND_SORTS, EDGES,
} from "../players/scan.js";

// --- tickets: legs and bodies -------------------------------------------------------------------------

function refOfLeg(leg) {
  return { league: leg.league, id: String(leg.event_id ?? ""), market: leg.market, side: leg.side, point: leg.point ?? null };
}

/** A ticket's legs as slip legs ({ref, label, game, ...}), refs from the card where it has them. */
export function ticketLegs(ticket, refs = []) {
  return (ticket?.legs || []).map((leg, i) => {
    const ref = refs[i] || ticket.refs?.[i] || refOfLeg(leg);
    return {
      ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
      label: leg.label, game: leg.game, start: leg.start, league: ref.league,
      price: leg.odds?.decimal, american: leg.odds?.american, ai: leg.ai_prob ?? null, fair: leg.market_prob ?? null,
    };
  }).filter((leg) => leg.ref.league && leg.ref.id && ["h2h", "spreads", "totals"].includes(leg.ref.market));
}

/** "10" from "10.00" (the slip's stake chips compare plain amounts). */
export function plainStake(value) {
  const text = String(value ?? "").trim();
  if (!/^\d+(\.\d{1,2})?$/.test(text)) return null;
  return text.replace(/\.00?$/, "").replace(/(\.\d)0$/, "$1");
}

/** The ticket as a /api/chat "slip" body {type, stake, legs, k?, system?} (to ask the AI about it). */
export function ticketBody(ticket, refs) {
  const legs = ticketLegs(ticket, refs);
  if (!legs.length) return null;
  const body = { type: ticket.type || "multi", stake: plainStake(ticket.unit_stake) || "10", legs: legs.map((leg) => leg.ref) };
  if (body.type === "round_robin" && ticket.k) body.k = ticket.k;
  if (body.type === "system" && ticket.system) body.system = ticket.system;
  return body;
}

/**
 * How a ticket's legs sit in the slip: "exact" (the slip is this ticket), "switch" (the same picks played
 * another way, e.g. singles instead of a multi, or a round robin by 3s instead of by 2s), "within" (every
 * leg is there, with other picks besides), "some" or "none". The stake isn't compared: playing a ticket
 * another way must never put a bigger stake back on the slip.
 */
export function slipState(legs, ticket = null) {
  const inSlip = legs.filter((leg) => slip.has(leg.ref)).length;
  if (!inSlip) return "none";
  if (inSlip < legs.length) return "some";
  if (slip.count !== legs.length) return "within";
  const type = ticket?.type;
  if (!type || legs.length === 1) return "exact";
  const sameShape = type === slip.type
    && (type !== "round_robin" || Number(ticket.k || 2) === Number(slip.k || 2))
    && (type !== "system" || String(ticket.system || "") === String(slip.system || ""));
  return sameShape ? "exact" : "switch";
}

/** Why a ticket can't go on the slip as it is, or "" (a started or withdrawn leg). */
function blocked(ticket) {
  if ((ticket.legs || []).some((leg) => leg.started)) return "A game on this ticket has started.";
  if ((ticket.removed || []).length) return "Some picks are no longer offered.";
  return "";
}

// --- tickets: the card ------------------------------------------------------------------------------

const SOURCES = {
  ticket_build: "Built for your request",
  ticket_analyse: "Your ticket, analysed",
  ticket_price: "Priced by the app",
  slip_propose: "Suggested by the AI · check it first",
};

/** "2-leg multi · $10.00 stake", "Round robin by 2s · 6 bets of $1.00". */
export function kindLine(ticket) {
  const legs = (ticket.legs || []).length;
  const stake = money(ticket.unit_stake, { cents: true });
  if (ticket.type === "multi") return `${legs}-leg multi · ${stake} stake`;
  if (ticket.type === "single") return legs === 1 ? `Single · ${stake} stake` : `${legs} singles · ${stake} each`;
  const name = ticket.type === "round_robin" ? `Round robin by ${ticket.k || 2}s` : String(ticket.system || "System").replace(/_/g, " ");
  return `${name} · ${ticket.bets || "?"} bets of ${stake}`;
}

function tierBadge(tier) {
  if (!tier?.key) return "";
  const kind = { safe: "info", risky: "warn", crazy: "danger" }[tier.key] || "muted";
  const loses = tier.loses != null ? ` · loses ${pct(tier.loses)}` : "";
  return `<span class="ask-tier" title="A tier by chance, not a promise">${badge(`${tier.title || tier.key}${loses}`, kind, "shield")}${explain("tiers", { label: "What do the tiers mean?" })}</span>`;
}

function gameText(game, league) {
  const [away, home] = String(game || "").split(" @ ");
  return home ? `${shortName(away, league)} @ ${shortName(home, league)}` : String(game || "");
}

function legNotes(notes) {
  if (!notes?.length) return "";
  const first = notes.slice(0, 2);
  const rest = notes.slice(2);
  return `<ul class="ask-notes">${first.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    ${rest.length ? `<details class="ask-more"><summary>${rest.length} more note${rest.length === 1 ? "" : "s"}</summary>
      <ul class="ask-notes">${rest.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></details>` : ""}`;
}

// A Builder leg's chance is the brain's: "Markets say" (the blend), the books' margin-free chance or the goals model's
const BUILDER_CHANCE = { blend: "Markets say", books: "Books", model: "Goals model", adjusted: "Adj" };

function legsHtml(ticket, notesByIndex, builder = false) {
  const weakest = (ticket.legs || []).length > 1 ? ticket.weakest_leg : null;
  return `<ol class="ask-legs">${(ticket.legs || []).map((leg, i) => {
    const mine = builder && BUILDER_CHANCE[leg.chance_source];
    const chances = [leg.ai_prob != null ? `${mine || "AI"} ${pct(mine ? leg.chance ?? leg.ai_prob : leg.ai_prob)}` : "",
      leg.market_prob != null && !(mine && leg.chance_source === "books") ? `Market ${pct(leg.market_prob)}` : "",
      leg.chance_label || ""].filter(Boolean).join(" · ");
    return `<li class="ask-leg${i === weakest ? " weak" : ""}">
      <div class="row-between"><b class="ask-leg-label">${esc(leg.label)}</b><b class="num ask-leg-odds">${esc(priceText(leg.odds))}</b></div>
      <div class="muted small">${esc(gameText(leg.game, leg.league))}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""}${leg.book ? ` · ${esc(leg.book)}` : ""}</div>
      <div class="small ask-leg-chance">${esc(chances)}${i === weakest ? ` ${badge("Weakest leg", "warn", "alert")}` : ""}${leg.started ? ` ${badge("Started", "danger", "clock")}` : ""}</div>
      ${legNotes(notesByIndex.get(i))}
    </li>`;
  }).join("")}</ol>`;
}

function summaryHtml(ticket) {
  if (!ticket.bets) return "";
  const c = ticket.chance || {};
  const e = ticket.expected || {};
  const ml = ticket.most_likely;
  const both = (ai, market) => [ai, market].filter(Boolean).join("<br>") || "–";
  const chanceWord = c.headline === "profit" ? "Chance of a profit" : "Chance it wins";
  const fixed = ticket.type === "multi" || ticket.bets === 1;
  const likely = ml ? (ml.result === "lose all" ? `Lose ${money(ticket.total_stake)}` : ml.result.charAt(0).toUpperCase() + ml.result.slice(1)) : "";
  return `<dl class="summary ask-summary">
    ${ticket.odds ? `<div><dt>Odds</dt><dd class="num">${esc(priceText(ticket.odds))}</dd></div>` : ""}
    <div><dt>Costs</dt><dd class="num">${money(ticket.total_stake, { cents: true })}${ticket.bets > 1 ? ` <span class="muted">(${ticket.bets} bets)</span>` : ""}</dd></div>
    <div><dt>Pays ${explain("payout")}</dt><dd class="num">${fixed ? "" : "up to "}${money(ticket.max_payout, { cents: true })}</dd></div>
    <div><dt>${chanceWord} ${explain("one_in")}</dt><dd class="num">${both(c.ai != null ? `AI ${esc(chanceText(c.ai, c.ai_one_in))}` : "", c.market != null ? `Market ${esc(chanceText(c.market, c.market_one_in))}` : "")}</dd></div>
    <div><dt>Bookmaker keeps ${explain("book_cut")}</dt><dd class="num">${ticket.book_cut != null ? `about ${pct(ticket.book_cut)}` : "–"}</dd></div>
    <div><dt>Expected result ${explain("expected_result")}</dt><dd class="num">${both(e.ai != null ? `${esc(signedMoney(e.ai))} <span class="muted">AI</span>` : "", e.market != null ? `${esc(signedMoney(e.market))} <span class="muted">market</span>` : "")}</dd></div>
    ${ml ? `<div><dt>Most likely ${explain("most_likely")}</dt><dd>${esc(likely)} <span class="muted">(${pct(ml.probability)})</span></dd></div>` : ""}
  </dl>`;
}

function warningsHtml(warnings) {
  if (!warnings?.length) return "";
  const item = (w) => `<li>${icon("alert", { size: 14 })}<span>${esc(w)}</span></li>`;
  const first = warnings.slice(0, 2);
  const rest = warnings.slice(2);
  return `<ul class="warnings">${first.map(item).join("")}</ul>
    ${rest.length ? `<details class="ask-more"><summary>${rest.length} more warning${rest.length === 1 ? "" : "s"}</summary><ul class="warnings">${rest.map(item).join("")}</ul></details>` : ""}`;
}

function addButton(legs, key, attr, { ticket = null, disabled = false, label = "Add to slip" } = {}) {
  const state = slipState(legs, ticket);
  if (state === "exact") {
    return `<button type="button" class="btn btn-secondary small" data-open-slip ${attr}="${esc(key)}">${icon("check", { size: 16 })}In your slip</button>`;
  }
  if (state === "within" || state === "switch") {
    const words = state === "switch" ? "Play it this way" : label === "Add" ? "Use these" : "Use just these";
    return `<button type="button" class="btn btn-secondary small" ${attr}="${esc(key)}"${disabled ? " disabled" : ""}>${icon("check", { size: 16 })}${words}</button>`;
  }
  return `<button type="button" class="btn btn-primary small" ${attr}="${esc(key)}"${disabled || !legs.length ? " disabled" : ""}>${icon("plus", { size: 16 })}${esc(label)}</button>`;
}

function alternativesHtml(card, key) {
  const alts = (card.alternatives || []).filter((alt) => alt?.ticket?.legs?.length);
  if (!alts.length) return "";
  return `<details class="ask-alts"><summary>${icon("scale", { size: 16 })}Other ways to play it (${alts.length})</summary>
    <ul class="ask-alt-list">${alts.map((alt, i) => {
      const t = alt.ticket;
      const c = t.chance || {};
      const p = c.ai ?? c.market;
      const oneIn = c.ai != null ? c.ai_one_in : c.market_one_in;
      const exp = t.expected?.ai ?? t.expected?.market;
      return `<li class="ask-alt">
        <div class="grow"><b>${esc(alt.title || kindLine(t))}</b>
          <span class="muted small">${esc(kindLine(t))}${p != null ? ` · ${c.headline === "profit" ? "profit" : "wins"} ${esc(chanceText(p, oneIn))}` : ""}${exp != null ? ` · expected ${esc(signedMoney(exp))}` : ""}</span>
          ${alt.summary ? `<span class="small">${esc(alt.summary)}</span>` : ""}</div>
        ${addButton(ticketLegs(t, alt.refs), `${key}:${i}`, "data-alt-add", { ticket: t, disabled: Boolean(blocked(t)), label: "Add" })}
      </li>`;
    }).join("")}</ul></details>`;
}

/** A priced ticket: legs with each one's chances, the honest summary, warnings, then Add to slip / Tweak. */
export function ticketCardHtml(card, key, { stale = "" } = {}) {
  const ticket = card.ticket || {};
  const legs = ticketLegs(ticket, card.refs);
  const notesByIndex = new Map((card.notes || []).map((n) => [n.index, n.notes || []]));
  const problem = blocked(ticket);
  const source = card.proposal ? SOURCES.slip_propose : SOURCES[card.source] || "Ticket";
  // Closest effort: the option that picked it (its title when the server named it so, else a chip above)
  const option = optionOf(card.option || ticket.option);
  const titled = option && card.title === option.label;
  return `<article class="card ask-card ask-ticket" data-card-key="${esc(key)}" aria-label="${esc(`Ticket: ${card.title || kindLine(ticket)}`)}">
    ${option && !titled ? `<div class="ask-opts"><span class="ask-opt k-${esc(option.key || "other")}">${icon(option.iconName, { size: 14 })}${esc(option.label)}</span></div>` : ""}
    <header class="ask-card-head">
      <div class="grow"><span class="eyebrow">${esc(source)}</span>
        <h3${titled ? ` class="ask-opt-title k-${esc(option.key || "other")}"` : ""}>${titled ? icon(option.iconName, { size: 16 }) : ""}${esc(card.title || kindLine(ticket))}</h3>
        <span class="muted small">${esc(kindLine(ticket))}</span></div>
      ${tierBadge(ticket.tier)}
    </header>
    ${fitHtml(card.fit !== undefined ? card.fit : ticket.fit, card.relaxed !== undefined ? card.relaxed : ticket.relaxed)}
    ${legsHtml(ticket, notesByIndex, Boolean(card.builder))}
    ${(ticket.reasons || []).map((r) => notice(r, "danger")).join("")}
    ${summaryHtml(ticket)}
    ${card.verdict ? `<p class="verdict">${esc(card.verdict)}</p>` : ""}
    ${warningsHtml(card.builder ? [...new Set([...(ticket.notes || []), ...(ticket.warnings || [])])] : ticket.warnings)}
    ${alternativesHtml(card, key)}
    ${stale ? `<p class="ask-stale">${icon("clock", { size: 14 })}${esc(stale)}</p>` : ""}
    ${problem ? `<p class="ask-stale warn">${icon("alert", { size: 14 })}${esc(problem)}</p>` : ""}
    <div class="ask-card-actions" data-card-actions="${esc(key)}">
      ${addButton(legs, key, "data-card-add", { ticket, disabled: Boolean(problem) })}
      ${legs.length ? `<button type="button" class="btn btn-secondary small" data-card-tweak="${esc(key)}">${icon("edit", { size: 16 })}Tweak</button>` : ""}
    </div>
  </article>`;
}

/** "3.52x · 27% (you asked 3.5x · 80%)" and the muted "Relaxed: …" line (closest effort), or "". */
function fitHtml(fit, relaxed) {
  const parts = fitParts(fit);
  const words = relaxedText(relaxed);
  return `${parts ? `<p class="ask-fit">${icon("target", { size: 14 })}<span><b class="num">${esc(parts.got)}</b> <span class="muted">(you asked ${esc(parts.asked)})</span></span>${parts.onTarget ? badge("On target", "good", "check") : ""}</p>` : ""}
    ${words ? `<p class="ask-relaxed">${icon("sliders", { size: 14 })}<span>${esc(words)}</span></p>` : ""}`;
}

// --- the likeliest winners (brain_top_picks) --------------------------------------------------------------

const PICK_SOURCES = { blend: "Markets say", books: "books, margin out", model: "goals model", adjusted: "books, adjusted" };

/** A pick as a slip leg ({ref, label, game, ...}), or null when the app's slip can't take it. */
export function pickLeg(pick) {
  const ref = pick?.ref || pick?.refs?.[0];
  if (!ref?.league || ref.id == null || !["h2h", "spreads", "totals"].includes(ref.market)) return null;
  const price = Number(pick.price);
  return {
    ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
    label: pick.label, game: pick.game, start: pick.start, league: ref.league,
    price: Number.isFinite(price) ? price : undefined,
  };
}

/** An edge in points with its sign: "+1.2 pts", "−2.1 pts" (the server's share). */
function edgePts(edge) {
  const v = Number(edge) * 100;
  if (edge == null || !Number.isFinite(v)) return "";
  const text = Math.abs(v) < 0.05 ? "0.0" : Math.abs(v).toFixed(1);
  return `${v > 0.04 ? "+" : v < -0.04 ? "−" : ""}${text} pts`;
}

function pickButton(pick, key, i) {
  const leg = pickLeg(pick);
  const label = String(pick?.label || "this pick");
  if (!leg) return `<button type="button" class="btn btn-secondary small" disabled>${icon("plus", { size: 16 })}Add to slip</button>`;
  const on = slip.has(leg.ref);
  return on
    ? `<button type="button" class="btn btn-secondary small" data-pick-add="${esc(key)}:${i}" aria-pressed="true" aria-label="${esc(`${label}: in your slip. Tap to take it out`)}">${icon("check", { size: 16 })}In your slip</button>`
    : `<button type="button" class="btn btn-primary small" data-pick-add="${esc(key)}:${i}" aria-pressed="false" aria-label="${esc(`Add ${label} to your slip`)}">${icon("plus", { size: 16 })}Add to slip</button>`;
}

/** The likeliest winners: ranked, each with its chance, price, book, a short why and "Add to slip". */
export function picksCardHtml(card, key, { stale = "" } = {}) {
  const picks = (card.picks || []).filter((p) => p && typeof p === "object");
  return `<article class="card ask-card ask-picks" data-card-key="${esc(key)}" aria-label="${esc(card.title || "Most likely winners")}">
    <header class="ask-card-head"><div class="grow"><span class="eyebrow">The brain's likeliest winners</span>
      <h3>${esc(card.title || "Most likely winners")}</h3>
      <span class="muted small">Highest chance first, one pick a game</span></div></header>
    <ol class="ask-pick-list">${picks.map((pick, i) => {
      const league = pick.ref?.league || pick.league;
      const why = ((pick.why_units && pick.why_units[unitsFor(league)]) || pick.why || []);
      const lines = (Array.isArray(why) ? why : [why]).filter(Boolean).slice(0, 2);
      const source = PICK_SOURCES[pick.chance_source] || "";
      const edge = edgePts(pick.edge);
      const price = Number(pick.price);
      return `<li class="ask-pick">
        <span class="ask-pick-rank" aria-label="${esc(`Number ${pick.rank || i + 1}`)}">${esc(pick.rank || i + 1)}</span>
        <div class="ask-pick-body">
          <div class="row-between"><b class="ask-leg-label">${esc(pick.label)}</b><b class="num ask-leg-odds">${esc(Number.isFinite(price) ? priceText({ decimal: price }) : "–")}</b></div>
          <div class="muted small">${esc(gameText(pick.game, league))}${pick.start ? ` · ${esc(timeLabel(pick.start))}` : ""}${pick.book ? ` · ${esc(pick.book)}` : ""}</div>
          <div class="small ask-pick-chance">Wins <b class="num">${esc(pct(pick.chance))}</b>${source ? ` <span class="muted">${esc(source)}</span>` : ""}${edge ? ` · edge <b class="num">${esc(edge)}</b>` : ""}</div>
          ${lines.length ? `<p class="small ask-pick-why">${lines.map(esc).join(" · ")}</p>` : ""}
        </div>
        <div class="ask-pick-add" data-pick-slot="${esc(key)}:${i}">${pickButton(pick, key, i)}</div>
      </li>`;
    }).join("")}</ol>
    ${stale ? `<p class="ask-stale">${icon("clock", { size: 14 })}${esc(stale)}</p>` : ""}
    <p class="small muted">Chances are the brain's; prices are the best at bookmakers you can bet at, as last saved. Check the price where you bet.</p>
  </article>`;
}

/** Refresh the Add buttons of the cards on screen after the slip changed. */
export function syncCardButtons(root, findCard) {
  for (const box of root.querySelectorAll("[data-card-actions]")) {
    const card = findCard(box.dataset.cardActions);
    if (!card?.ticket) continue;
    const first = box.querySelector("[data-card-add]");
    if (first) first.outerHTML = addButton(ticketLegs(card.ticket, card.refs), box.dataset.cardActions, "data-card-add", { ticket: card.ticket, disabled: Boolean(blocked(card.ticket)) });
  }
  for (const slot of root.querySelectorAll("[data-pick-slot]")) {
    const [mi, ci, pi] = slot.dataset.pickSlot.split(":");
    const pick = findCard(`${mi}:${ci}`)?.picks?.[Number(pi)];
    if (!pick) continue;
    const had = slot.contains(document.activeElement);
    slot.innerHTML = pickButton(pick, `${mi}:${ci}`, Number(pi));
    if (had) slot.querySelector("button")?.focus({ preventScroll: true });
  }
  for (const button of root.querySelectorAll("[data-scan-add]")) {
    const [mi, ci, ri] = button.dataset.scanAdd.split(":");
    const row = scanRowAt(findCard(`${mi}:${ci}`), ri);
    if (row) button.outerHTML = scanAddButton(row, `${mi}:${ci}`, Number(ri));
  }
  for (const button of root.querySelectorAll("[data-alt-add]")) {
    const [mi, ci, ai] = button.dataset.altAdd.split(":");
    const alt = findCard(`${mi}:${ci}`)?.alternatives?.[Number(ai)];
    if (alt?.ticket) button.outerHTML = addButton(ticketLegs(alt.ticket, alt.refs), button.dataset.altAdd, "data-alt-add", { ticket: alt.ticket, disabled: Boolean(blocked(alt.ticket)), label: "Add" });
  }
}

// --- rollover plans -----------------------------------------------------------------------------------

const STRATEGY_ORDER = ["classic", "bank_half", "ladder", "flat"];
const STRATEGY_SHORT = { classic: "Classic", bank_half: "Bank half", ladder: "Pot & ladder", flat: "Flat 5%" };

function survivalChart(curve) {
  const n = curve.length;
  if (!n) return "";
  const width = 320;
  const height = 104;
  const top = 16;
  const bottom = 20;
  const slot = width / n;
  const plot = height - top - bottom;
  const labelEvery = n <= 12 ? 1 : Math.ceil(n / 8);
  const bars = curve.map((p, i) => {
    const h = Math.max(1.5, p * plot);
    const x = i * slot + slot * 0.18;
    const cx = i * slot + slot / 2;
    const day = i + 1;
    const label = day % labelEvery === 0 || day === 1 ? `<text x="${cx.toFixed(1)}" y="${height - 5}" class="c-sub" text-anchor="middle">${day}</text>` : "";
    const value = n <= 10 ? `<text x="${cx.toFixed(1)}" y="${(top + plot - h - 4).toFixed(1)}" class="c-value" text-anchor="middle">${esc(pct(p))}</text>` : "";
    return `<rect x="${x.toFixed(1)}" y="${(top + plot - h).toFixed(1)}" width="${(slot * 0.64).toFixed(1)}" height="${h.toFixed(1)}" rx="2"/>${value}${label}`;
  }).join("");
  const words = curve.map((p, i) => `day ${i + 1} ${pct(p)}`).join(", ");
  return `<figure class="ask-survival">
    <figcaption class="small muted">Chance a classic run is still going after each day</figcaption>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(`Chance a classic rollover is still going: ${words}`)}">
      <line x1="0" x2="${width}" y1="${top + plot}" y2="${top + plot}" class="c-axis"/>${bars}</svg>
  </figure>`;
}

/** A rollover comparison: the honest sentence, the four strategies side by side and how fast runs end. */
export function planCardHtml(card) {
  const compare = card.compare || {};
  const body = card.body || {};
  const strategies = compare.strategies || {};
  const rows = STRATEGY_ORDER.filter((name) => strategies[name]?.market).map((name) => {
    const s = strategies[name];
    const m = s.market;
    return `<tr${s.default ? ' class="default"' : ""}>
      <th scope="row">${esc(STRATEGY_SHORT[name] || s.title || name)}${s.default ? ` <span class="badge info">default</span>` : ""}</th>
      <td>${esc(pct(m.chance_complete))}</td><td>${esc(pct(m.chance_profit))}</td>
      <td>${esc(money(m.median_end))}</td><td>${esc(money(m.expected_cost, { cents: true }))}</td></tr>`;
  }).join("");
  const classic = strategies.classic?.market;
  const query = new URLSearchParams({ start: String(body.start ?? compare.start ?? ""), days: String(body.days ?? compare.days ?? ""),
    min_odds: String(body.min_odds ?? compare.min_odds ?? ""), legs_per_day: String(body.legs_per_day ?? compare.legs_per_day ?? 1) });
  const aiLine = compare.p_ai != null && strategies.classic?.ai
    ? `<p class="small muted">By the AI's daily chance (${esc(pct(compare.p_ai))}) a classic run finishes ${esc(pct(strategies.classic.ai.chance_complete))}; the market's is the safer guide.</p>` : "";
  return `<article class="card ask-card ask-plan" aria-label="${esc(`Rollover plan: ${card.title || ""}`)}">
    <header class="ask-card-head"><div class="grow"><span class="eyebrow">Rollover planner · paper only ${explain("rollover")}</span>
      <h3>${esc(card.title || "Rollover")}</h3>
      <span class="muted small">Start ${esc(money(body.start ?? compare.start, { cents: true }))} · ${esc(String(compare.legs_per_day || body.legs_per_day || 1))} leg${(compare.legs_per_day || 1) > 1 ? "s" : ""} a day</span></div></header>
    ${compare.summary ? `<p class="ask-plan-summary">${esc(compare.summary)}</p>` : ""}
    ${rows ? `<div class="table-wrap"><table class="ask-plan-table">
      <thead><tr><th scope="col">Way to play</th><th scope="col">Finishes</th><th scope="col">Profit</th><th scope="col">Typical end</th><th scope="col">Exp. cost</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="small muted">By the market's chances. “Typical end” is the middle result of all runs; “Exp. cost” is what a run loses on average.</p>` : ""}
    ${aiLine}
    ${classic?.survival?.length ? survivalChart(classic.survival) : ""}
    ${compare.requirement ? `<p class="small">${esc(compare.requirement)}</p>` : ""}
    ${warningsHtml(compare.warnings)}
    <div class="ask-card-actions"><a class="btn btn-secondary small" href="#/build/rollover?${esc(query.toString())}">${icon("chart", { size: 16 })}Open the planner</a></div>
  </article>`;
}

// --- players ------------------------------------------------------------------------------------------

const STAT_NAMES = { pts: "PTS", reb: "REB", ast: "AST", fg3m: "3PM", min: "MIN", stl: "STL", blk: "BLK", tov: "TOV" };
const BASKETBALL = ["pts", "reb", "ast", "fg3m", "min"];

function statValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? (Number.isInteger(n) ? String(n) : n.toFixed(1)) : "–";
}

/** A player's card: team and status, next game, and recent averages with their sample sizes. */
export function playerCardHtml(card) {
  const season = card.season || {};
  const last5 = card.last5 || {};
  const last10 = card.last10 || {};
  const keys = Object.keys({ ...season, ...last10 });
  const stats = (keys.some((k) => BASKETBALL.includes(k)) ? BASKETBALL.filter((k) => keys.includes(k)) : keys).slice(0, 5);
  const sample = card.sample || {};
  const status = card.injury?.status || (card.status && card.status !== "Active" ? card.status : "");
  const next = card.next_game;
  const nextText = next ? `Next: ${next.home ? "vs" : "@"} ${next.opponent || next.opponent_name || ""} · ${timeLabel(next.date)}` : "";
  const numeric = /^[0-9]{1,12}$/.test(String(card.id ?? ""));
  const head = (label, n) => `<th scope="col">${esc(label)}${n ? `<span class="ask-sample">${n} g</span>` : ""}</th>`;
  return `<article class="card ask-card ask-player" aria-label="${esc(`Player: ${card.name || card.title || ""}`)}">
    <header class="ask-player-head">
      ${card.headshot ? `<img class="headshot" src="${esc(card.headshot)}" alt="" loading="lazy" width="56" height="56">` : `<span class="headshot blank">${icon("user", { size: 24 })}</span>`}
      <div class="grow"><span class="eyebrow">Player</span><h3>${esc(card.name || card.title || "Player")}</h3>
        <span class="muted small">${esc([card.team, card.position].filter(Boolean).join(" · "))}</span>
        <div class="ask-badges">${status ? injuryBadge(status) : badge("Active", "good", "check")}${card.season_label ? badge(card.season_label, "muted") : ""}</div></div>
    </header>
    ${card.injury?.detail ? `<p class="small muted">${esc(card.injury.detail)}</p>` : ""}
    ${nextText ? `<p class="ask-next">${icon("clock", { size: 16 })}${esc(nextText)}</p>` : ""}
    ${stats.length ? `<div class="table-wrap"><table class="ask-stats">
      <thead><tr><th scope="col">Per game</th>${head("Last 5", sample.last5)}${head("Last 10", sample.last10)}${head("Season", sample.season)}</tr></thead>
      <tbody>${stats.map((k) => `<tr><th scope="row">${esc(STAT_NAMES[k] || k)}</th><td>${statValue(last5[k])}</td><td>${statValue(last10[k])}</td><td>${statValue(season[k])}</td></tr>`).join("")}</tbody>
    </table></div>` : ""}
    ${card.as_of ? `<p class="ask-stale">${icon("clock", { size: 14 })}ESPN data from ${esc(ago(card.as_of))}</p>` : ""}
    ${numeric && card.league ? `<div class="ask-card-actions"><a class="btn btn-secondary small" href="#/player/${esc(encodeURIComponent(card.league))}/${esc(card.id)}">${icon("chart", { size: 16 })}Lines, chart and news</a></div>` : ""}
  </article>`;
}

// --- the player scan (players_scan) -------------------------------------------------------------------------

/** The names a card's scan was filtered by (the server's ``names``, or the tool's ``player``); [] for a ranked scan. */
export function cardNames(card) {
  if (Array.isArray(card?.names) || typeof card?.names === "string") return cleanNames(card.names);
  return typeof card?.player === "string" && card.player.trim() ? cleanNames([card.player]) : [];
}

/**
 * The card's ask for GET /api/players/scan and the Player finder: {league, stat, threshold, side, day, n, sort, names,
 * edgeSide} (``sort`` "hot" or "cold" as the card was ranked, "edges" for a best-edges card, else ""). A best-edges
 * card keeps its players' names ("best props for Luka": his lines only) and the side it was asked for (``edgeSide``:
 * "any good overs" is the overs only; "" is both), so Refresh runs the same scan.
 */
export function scanAsk(card) {
  const league = (card?.leagues || [])[0] || (card?.players || [])[0]?.league || null;
  const threshold = card?.threshold == null || card?.threshold === "" ? null : Number(card.threshold);
  const edges = card?.mode === "edges";
  return {
    league, stat: card?.stat || "pts", threshold: !edges && Number.isFinite(threshold) ? threshold : null,
    side: card?.side === "under" ? "under" : "over", day: card?.day === "tomorrow" ? "tomorrow" : "today",
    n: Math.max(10, (card?.players || []).length),
    sort: edges ? EDGES : TREND_SORTS.includes(card?.sort) ? card.sort : "",
    names: cardNames(card),
    edgeSide: edges && (card?.side === "over" || card?.side === "under") ? card.side : "",
  };
}

// What a fresh scan replaces on the card, and what it takes away when it no longer says it
const SCAN_KEYS = ["ask", "stat", "stat_label", "threshold", "side", "day", "leagues", "players", "close", "as_of",
  "leagues_without_games", "note", "mode", "more", "games", "games_started", "error", "missing_names", "ambiguous_names",
  "started_names"];
const SCAN_GONE = ["loading", "leagues_without_games", "error", "missing_names", "ambiguous_names", "started_names"];

/** A fresh GET /api/players/scan result laid over the card (its title, source, sort and names stay). */
export function refreshedScan(card, fresh) {
  const out = { ...card };
  for (const key of SCAN_KEYS) {
    if (fresh?.[key] !== undefined) out[key] = fresh[key];
  }
  if (fresh?.loading) out.loading = fresh.loading;
  for (const key of SCAN_GONE) {
    if (!fresh?.[key]) delete out[key];
  }
  return out;
}

/** A scan row's own "Add to slip" (only a row the server marks addable, with a game pick the slip takes). */
function scanAddButton(row, key, i) {
  const leg = rowLeg(row);
  if (!leg) return "";
  const label = String(row.label || row.name || "this pick");
  return slip.has(leg.ref)
    ? `<button type="button" class="btn btn-secondary small" data-scan-add="${esc(key)}:${i}" aria-pressed="true" aria-label="${esc(`${label}: in your slip. Tap to take it out`)}">${icon("check", { size: 16 })}In your slip</button>`
    : `<button type="button" class="btn btn-primary small" data-scan-add="${esc(key)}:${i}" aria-pressed="false" aria-label="${esc(`Add ${label} to your slip`)}">${icon("plus", { size: 16 })}Add to slip</button>`;
}

/** The scan row behind "key:i" (players first, then the close ones). */
export function scanRowAt(card, i) {
  const rows = [...(card?.players || []), ...(card?.close || [])];
  return rows[Number(i)] || null;
}

function scanLoadingHtml(card, key, refreshing) {
  const words = statusText(card);
  if (!words) return "";
  return `<p class="ps-loading" role="status">${icon("clock", { size: 14 })}<span>${esc(words)}</span>
    <button type="button" class="chip" data-scan-refresh="${esc(key)}"${refreshing ? " disabled aria-busy=\"true\"" : ""}>${icon("refresh", { size: 16 })}${refreshing ? "Refreshing…" : "Refresh"}</button></p>`;
}

/** The line under the card's title: how the rows are ranked and what each one shows. */
function scanSubtitle(view) {
  if (view.edges) return "Every stat's posted lines, biggest edge first: the edge, chance, hits and the line";
  if (view.trend) return `${view.trend === "hot" ? "Hottest" : "Coldest"} first: the last 5 games' average against the season's`;
  if (view.names) return "Your players: whether each qualifies, the chance, hits and the posted line";
  if (view.posted) return "Biggest edge first: the edge, chance, hits and each player's posted line";
  return view.yesNo ? "Strongest first: chance, hits and the posted line" : "Strongest first: projection, chance, hits and the posted line";
}

/**
 * Players ranked for a stat and a number: the strongest five first, the rest that qualify under "Also:", how many
 * more qualify (the result's ``more``: open the Player finder for them), "Also close" collapsed, a loading line with
 * Refresh, and the notes: each name it couldn't rank (not found, more than one match, game already started), games
 * already started and ESPN's scoreboard down, worded as the chat's answer and the bot word them.
 */
export function scanCardHtml(card, key, { stale = "", refreshing = false } = {}) {
  ensureScanStyles();
  const side = card.side === "under" ? "under" : "over";
  const players = (card.players || []).filter((row) => row && typeof row === "object");
  const close = (card.close || []).filter((row) => row && typeof row === "object");
  const where = (card.leagues || []).map((l) => scanLeague(l)?.title || String(l).split("_").pop().toUpperCase()).join(" + ");
  const ask = scanAsk(card);
  const edges = ask.sort === EDGES;
  const when = whenText(card);
  const title = card.title || (edges ? `Best edges (all stats) · ${when}` : `Projected for ${askText(card)} · ${when}`);
  const without = (card.leagues_without_games || []).filter(Boolean);
  const trend = TREND_SORTS.includes(ask.sort) ? ask.sort : "";
  const view = {
    // each player's posted line ranks by edge (hot and cold rank by the trend, whatever the number)
    edges, posted: !edges && !trend && ask.threshold == null, trend,
    names: ask.names.length > 0, league: ask.league, yesNo: !edges && Boolean(statOf(ask.league, ask.stat)?.yes),
  };
  const item = (row, i) => rowItemHtml({ league: ask.league, ...row }, i, { side, addButton: (r, n) => scanAddButton(r, key, n), view });
  const names = nameNotes(card, when);
  // No games today in the league asked: the scan moved to tomorrow's (or found none tomorrow either)
  const moved = (Array.isArray(card.no_games_today) ? card.no_games_today : []).filter(Boolean).map(String);
  const notes = [...(moved.length ? [{ kind: "info", text: `No games today in ${moved.join(", ")}, so these are tomorrow's.` }] : []),
    ...names, ...boardNotes(card, { when })];
  // The card's own more: how many beyond the rows the result returned, as the chat's answer and the bot say it
  const more = moreText(card, { sort: ask.sort });
  // An empty card's notes may say why already (ESPN down, every game started, the names it couldn't rank)
  const explained = Boolean(card.error) || allStarted(card) || (view.names && names.length > 0);
  const empty = close.length ? "Nobody is projected to get there yet: the closest are below."
    : explained ? "" : edges ? "No posted line has an edge right now." : trend ? "No hot or cold reads yet." : "No player qualifies yet.";
  return `<article class="card ask-card ask-scan" data-card-key="${esc(key)}" aria-label="${esc(title)}">
    <header class="ask-card-head"><div class="grow"><span class="eyebrow">Player scan${where ? ` · ${esc(where)}` : ""}</span>
      <h3>${esc(title)}</h3>
      <span class="muted small">${esc(scanSubtitle(view))}</span></div></header>
    ${scanLoadingHtml(card, key, refreshing)}
    ${without.length ? `<p class="ps-note">No games ${esc(when)} in ${esc(without.join(", "))}${card.checked_tomorrow ? " (or tomorrow)" : ""}.</p>` : ""}
    ${notesHtml(notes)}
    ${players.length ? rowListsHtml(players, item, { label: `Players for ${askText(card)}`, also: alsoLabel(players.length - TOP, { sort: ask.sort, edges }), split: !view.names })
      : empty ? `<p class="small">${esc(empty)}</p>` : ""}
    ${close.length ? `<details class="ps-close"${players.length ? "" : " open"}><summary>${icon("chevron", { size: 16 })}Also close (${close.length})</summary>
      <ol class="ps-list" aria-label="Also close">${close.map((row, i) => item(row, players.length + i)).join("")}</ol></details>` : ""}
    ${stale ? `<p class="ask-stale">${icon("clock", { size: 14 })}${esc(stale)}</p>` : ""}
    <p class="ps-note">${esc(card.slip_note || "Player props are for reading here; the slip takes game picks.")} L5, L10 and Season count the games at ${edges ? "each line" : "the number"}.${
      view.trend ? " L5 against season is the stat's average over this season's last 5 games against the rest of the season (or against last season's, early in a season)." : ""}${
      [...players, ...close].some((r) => r.model === "form") ? " \"Recent form\" rows are projected from the game log." : ""} Lines: DraftKings via ESPN.</p>
    ${more ? `<p class="ps-note ps-more">${esc(more)}</p>` : ""}
    ${ask.league && scanLeague(ask.league) ? `<div class="ask-card-actions"><a class="btn btn-secondary small" href="${esc(finderHash(ask))}">${icon("search", { size: 16 })}Open in Player finder</a></div>` : ""}
  </article>`;
}

/** Any card by its type ("ticket", "plan", "player", "pasted", "picks", "player_scan"); unknown types show their title only. */
export function cardHtml(card, key, options = {}) {
  if (card?.type === "ticket" && card.ticket) return ticketCardHtml(card, key, options);
  if (card?.type === "picks" && Array.isArray(card.picks)) return picksCardHtml(card, key, options);
  if (card?.type === "player_scan" && (Array.isArray(card.players) || Array.isArray(card.close))) return scanCardHtml(card, key, options);
  if (card?.type === "pasted" && card.pasted) return pastedCardHtml(card, key, options);
  if (card?.type === "plan" && card.compare) return planCardHtml(card);
  if (card?.type === "player") return playerCardHtml(card);
  return `<div class="card ask-card"><b>${esc(card?.title || "Card")}</b></div>`;
}
