// Pieces the Build screen's parts share: the named systems, turning a priced leg back into a slip pick,
// and the summary of a priced ticket (the same rows on the slip, the AI-built tickets and the analysis's
// alternatives). Every number shown comes from the server's pricing (engine/tickets.py via
// /api/tickets/*); these helpers only format it.

import { esc, pct, chanceText, money, signedMoney, priceText, shortName, explain, badge } from "../../ui.js";
import { icon } from "../../icons.js";
import { oddsFormat, me, onBreak } from "../../app.js";

export const STAKES = ["5", "10", "25"];
export const MAX_STAKE = 100000;
const STAKE_TEXT = /^\d{1,6}(\.\d{1,2})?$/;

/**
 * The named system bets, by the number of picks each needs. The bet counts are part of each system's
 * definition (a Yankee is always 11 bets); the server prices them.
 */
export const SYSTEMS = [
  { key: "trixie", title: "Trixie", legs: 3, bets: 4, what: "3 doubles and a treble" },
  { key: "patent", title: "Patent", legs: 3, bets: 7, what: "a Trixie plus the 3 singles" },
  { key: "yankee", title: "Yankee", legs: 4, bets: 11, what: "6 doubles, 4 trebles and a four-fold" },
  { key: "lucky15", title: "Lucky 15", legs: 4, bets: 15, what: "a Yankee plus the 4 singles" },
  { key: "super_yankee", title: "Super Yankee", legs: 5, bets: 26, what: "every double, treble, four- and five-fold (also called a Canadian)" },
  { key: "lucky31", title: "Lucky 31", legs: 5, bets: 31, what: "a Super Yankee plus the 5 singles" },
  { key: "heinz", title: "Heinz", legs: 6, bets: 57, what: "every combination of 2 to 6 picks" },
  { key: "lucky63", title: "Lucky 63", legs: 6, bets: 63, what: "a Heinz plus the 6 singles" },
  { key: "super_heinz", title: "Super Heinz", legs: 7, bets: 120, what: "every combination of 2 to 7 picks" },
  { key: "goliath", title: "Goliath", legs: 8, bets: 247, what: "every combination of 2 to 8 picks" },
];
const SYSTEM_ALIASES = { canadian: "super_yankee" };

/** The systems a slip of ``n`` picks can be played as. */
export function systemsFor(n) {
  return SYSTEMS.filter((s) => s.legs === n);
}

export function systemInfo(key) {
  return SYSTEMS.find((s) => s.key === (SYSTEM_ALIASES[key] || key)) || null;
}

export const TYPE_NAMES = { single: "Singles", multi: "Multi", round_robin: "Round robin", system: "System bet" };

/** "Multi", "Round robin by 2s", "Lucky 15" for a priced ticket. */
export function ticketName(t) {
  if (!t) return "";
  if (t.type === "round_robin") return `Round robin by ${t.k || 2}s`;
  if (t.type === "system") return systemInfo(t.system)?.title || "System bet";
  if (t.type === "single") return (t.legs?.length || 0) > 1 ? `${t.legs.length} singles` : "Single";
  return `${t.legs?.length || ""}-leg multi`.replace(/^-/, "");
}

/** A stake typed by the person, as the API takes it ("12.5"), or null when it isn't one. */
export function cleanStake(value) {
  const text = String(value ?? "").trim().replace(/^\$/, "").replace(/,/g, "");
  if (!STAKE_TEXT.test(text)) return null;
  const n = Number(text);
  return n >= 0.01 && n <= MAX_STAKE ? text : null;
}

/** A slip pick from a priced leg (the server's view of it) and its reference. */
export function slipLeg(view, ref) {
  return {
    ref: { league: ref.league, id: String(ref.id), market: ref.market, side: ref.side, point: ref.point ?? null },
    label: view.label,
    game: view.game,
    start: view.start,
    league: view.league || ref.league,
    price: view.odds?.decimal,
    american: view.odds?.american,
    ai: view.ai_prob,
    fair: view.market_prob,
  };
}

const capital = (text) => (text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : "");

/** "Chance it wins" for a ticket that needs every pick, "Chance of a profit" for spread tickets. */
export function chanceWord(chance) {
  return chance?.headline === "profit" ? "Chance of a profit" : "Chance it wins";
}

/** The two chance tiles (the AI's and the market's), "1 in X" for anything below even chances. */
export function chanceTiles(t, { info = true } = {}) {
  const c = t.chance || {};
  const tile = (who, p, oneIn, missing) => `<div class="bx-tile">
      <span>${who}</span>
      ${p == null ? `<b>–</b><small>${esc(missing)}</small>`
        : `<b class="num">${esc(oneIn && p < 0.5 ? `1 in ${Number(oneIn).toLocaleString()}` : pct(p))}</b><small>${esc(oneIn && p < 0.5 ? pct(p) : `about ${pct(p)} of the time`)}</small>`}
    </div>`;
  return `<div class="row-between"><span class="label" style="margin:0">${esc(chanceWord(c))} ${info ? explain("one_in") : ""}</span>
      ${c.label ? badge(capital(c.label), "muted") : ""}</div>
    <div class="bx-tiles">
      ${tile("AI", c.ai, c.ai_one_in, "No AI model for every pick")}
      ${tile("Market", c.market, c.market_one_in, "No market chance for every pick")}
    </div>`;
}

/** "Lose $10", "Profit of $35.21", "Some of the stake back" for the most likely result. */
export function mostLikelyText(ml, total) {
  if (!ml) return "–";
  let what;
  if (ml.result === "lose all") what = `Lose ${money(total)}`;
  else if (ml.result === "profit") what = ml.profit != null ? `A profit of ${money(ml.profit, { cents: true })}` : "A profit";
  else if (ml.result === "stake back") what = "Your stake back";
  else what = ml.profit != null ? `Some back (${signedMoney(ml.profit)})` : "Some of the stake back";
  return `${what} (${pct(ml.probability)})`;
}

function both(ai, market, format) {
  const parts = [];
  if (ai != null) parts.push(`${format(ai)} <span class="muted">AI</span>`);
  if (market != null) parts.push(`${format(market)} <span class="muted">market</span>`);
  return parts.length ? parts.join("<br>") : "–";
}

/** What the bookmaker keeps as a percentage, with a decimal below 10% ("0.5%", "8.3%", "21%"). */
export function cutPct(cut) {
  if (cut == null) return "–";
  if (cut <= 0) return "0%";
  return cut < 0.095 ? `${(cut * 100).toFixed(1)}%` : pct(cut);
}

function cutText(cut) {
  if (cut == null) return "–";
  if (cut <= 0) return "nothing on average <span class=\"muted\">(these prices beat the consensus)</span>";
  return `about ${cutPct(cut)}`;
}

/** "Wizards @ Hornets" from "Washington Wizards @ Charlotte Hornets" (team nicknames where the league has them). */
export function gameShort(game, league) {
  return String(game || "").split(" @ ").map((team) => shortName(team, league || "")).join(" @ ");
}

/**
 * The money rows of a priced ticket: odds, cost, payout, what the bookmaker keeps, the expected result
 * and the most likely outcome. ``compact`` also folds the chances in (for cards without chance tiles).
 */
export function summaryRows(t, { compact = false } = {}) {
  const c = t.chance || {};
  const e = t.expected || {};
  const spread = c.headline === "profit" || t.bets > 1;
  const rows = [];
  if (t.odds) rows.push([`Odds ${explain(oddsFormat() === "american" ? "american_odds" : "decimal_odds")}`, esc(priceText(t.odds))]);
  rows.push(["Costs", `${money(t.total_stake, { cents: true })}${t.bets > 1 ? ` <span class="muted">(${t.bets} bets × ${money(t.unit_stake, { cents: true })})</span>` : ""}`]);
  rows.push([`Pays${spread ? " up to" : ""} ${explain("payout")}`, money(t.max_payout, { cents: true })]);
  if (compact) {
    const parts = [];
    if (c.ai != null) parts.push(`${esc(chanceText(c.ai, c.ai_one_in))} <span class="muted">AI</span>`);
    if (c.market != null) parts.push(`${esc(chanceText(c.market, c.market_one_in))} <span class="muted">market</span>`);
    rows.push([`${esc(chanceWord(c))} ${explain("one_in")}`, parts.join("<br>") || "–"]);
  } else if (c.headline === "profit") {
    rows.push(["Any money back", both(c.any_return?.ai, c.any_return?.market, (p) => pct(p))]);
    rows.push(["Every pick wins", both(c.every_leg?.ai, c.every_leg?.market, (p) => pct(p))]);
  }
  rows.push([`Bookmaker keeps ${explain("book_cut")}`, cutText(t.book_cut)]);
  rows.push([`Expected result ${explain("expected_result")}`, both(e.ai, e.market, (v) => esc(signedMoney(v)))]);
  if (t.most_likely) rows.push([`Most likely ${explain("most_likely")}`, esc(mostLikelyText(t.most_likely, t.total_stake))]);
  return `<dl class="summary">${rows.map(([dt, dd]) => `<div><dt>${dt}</dt><dd>${dd}</dd></div>`).join("")}</dl>`;
}

const RESULT_NAMES = { "lose all": "Lose it all", "some back": "Some back", "stake back": "Stake back", profit: "A profit" };

/** Every result bucket with its chance and what it pays (for tickets whose outcomes spread out). */
export function outcomesTable(t) {
  const rows = t.outcomes || [];
  if (rows.length < 2 || t.chance?.headline !== "profit") return "";
  const pays = (o) => (o.min_return === o.max_return ? money(o.min_return, { cents: true })
    : `${money(o.min_return, { cents: true })}–${money(o.max_return, { cents: true })}`);
  return `<details class="bx-outcomes"><summary>${icon("chart", { size: 16 })}Spread of outcomes</summary>
    <div class="table-wrap"><table>
      <thead><tr><th>Result</th><th>Pays back</th><th>AI</th><th>Market</th></tr></thead>
      <tbody>${rows.map((o) => `<tr><td>${esc(RESULT_NAMES[o.result] || o.result)}</td><td class="num">${pays(o)}</td><td class="num">${pct(o.ai)}</td><td class="num">${pct(o.market)}</td></tr>`).join("")}</tbody>
    </table></div></details>`;
}

/** The server's warnings as a list (never colour alone: each has an icon). */
export function warningsList(list) {
  const items = [...new Set(list || [])];
  return items.length ? `<ul class="warnings">${items.map((w) => `<li>${icon("alert", { size: 14 })}<span>${esc(w)}</span></li>`).join("")}</ul>` : "";
}

/**
 * Replace a container's HTML and give focus back to the same control (a mode, a chip, a button), so a
 * redraw after the server answers never throws a keyboard or screen-reader user back to the top.
 */
export function redraw(el, html) {
  if (!el) return;
  const active = document.activeElement;
  let selector = null;
  if (active && active !== el && el.contains(active)) {
    for (const attr of ["data-mode", "data-k", "data-system", "data-act", "data-remove-leg", "data-stake", "data-seg", "id"]) {
      if (active.hasAttribute(attr)) {
        const value = active.getAttribute(attr);
        selector = `[${attr}="${CSS.escape(value)}"]${attr === "data-seg" ? `[data-value="${CSS.escape(active.dataset.value || "")}"]` : ""}`;
        break;
      }
    }
  }
  el.innerHTML = html;
  if (selector) el.querySelector(selector)?.focus({ preventScroll: true });
}

/** Scroll an element into view below the sticky header (smoothly unless reduced motion is asked for). */
export function reveal(el) {
  if (!el?.isConnected) return;
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  const top = el.getBoundingClientRect().top + window.scrollY - 72;
  window.scrollTo({ top: Math.max(0, top), behavior: smooth ? "smooth" : "auto" });
}

/** Age, cool-off or picks-pause refusals from the server get a way forward instead of a bare error. */
export function gateHint(error) {
  const message = error?.message || "";
  if (/18\+|21\+|confirm/i.test(message)) return { kind: "age", message };
  if (/break until|on a break/i.test(message)) return { kind: "break", message };
  if (/picks are paused/i.test(message)) return { kind: "paused", message };
  return null;
}

/** "Tue, Oct 27, 9:41 PM": when a break or pause ends, as the break banner says it. */
function untilText(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

const later = (iso) => {
  const t = Date.parse(iso || "");
  return Number.isFinite(t) && t > Date.now();
};

/**
 * Why AI picks are off right now, from GET /api/me: a break (on_break, cooloff_until) or the 24-hour pause
 * after a warning sign in the chat (picks_paused_until). {kind: "break" | "paused", until, text}, or null.
 * The server refuses the builder, the Safe / Risky / Crazy tickets and new plans in both, so Build says so
 * up front. In the picks pause pricing, analysing and saving your own slip stay open; a break refuses those
 * too (slip.js switches them off). The rollover comparison (just maths) stays open in both.
 */
export function picksBlock(profile = me()) {
  if (onBreak(profile)) {
    const when = profile.cooloff_until ? untilText(profile.cooloff_until) : "";
    return { kind: "break", until: profile.cooloff_until || null, text: when ? `You're on a break until ${when}.` : "You're on a break." };
  }
  if (later(profile.picks_paused_until)) {
    return { kind: "paused", until: profile.picks_paused_until,
      text: `Picks are paused until ${untilText(profile.picks_paused_until)} after an earlier message about losses or money worries.` };
  }
  return null;
}

/** The one plain line under a control ``picksBlock()`` switches off, with the way to Help > Responsible play. */
export function blockNote(block, id) {
  return `<div class="notice info bx-block" role="note" id="${esc(id)}">${icon("pause", { size: 18 })}<div>
    <span>${esc(block.text)}</span> <a class="text-link" href="#/help/responsible">Responsible play ${icon("chevron", { size: 16 })}</a></div></div>`;
}
