// The Builder's results as HTML: each ticket card (odds, the joint chance as "1 in N" with its fair odds, the
// chance of reaching the target, the expected result in money, what the books keep, one row per leg with its
// price, book, chance, edge, data age, "Why" and Swap), the near misses with the rule each failed, the plain
// "what blocked it" when nothing could be built, and the ideas when correlation ideas were asked for.
//
// Every number is the server's (POST /api/builder/run, /api/builder/price after a swap): this file only lays
// them out. Every outside text is escaped.
//
//   resultHtml(view)        view = {result, cards: [card], stake, at, busy}
//   card = {t (the ticket shown: the server's, or the server's re-pricing after a swap), swapped: {index} | null,
//           explain: {busy, data, error} | null, open: index of the leg whose swaps are open | null, inSlip}

import { esc, pct, money, signedMoney, priceText, timeLabel, badge, explain, notice } from "../../../ui.js";
import { icon } from "../../../icons.js";
import { leagueTitle, me } from "../../../app.js";
import { gameShort, cutPct, mostLikelyText } from "../common.js";

const FLAG_BADGES = {
  disagree: ["Sources disagree", "warn", "scale"],
  thin: ["Thin market", "muted", "alert"],
  low_conf: ["Low confidence", "warn", "alert"],
  model_price: ["Fair odds: no book price", "info", "info"],
  stale_price: ["Old price", "warn", "clock"],
  live: ["Under way", "danger", "clock"],
};
const SOURCES = { blend: "Markets say", books: "books, margin out", model: "goals model" };

/** "1 in 3" below even chances, "62%" from there up (the server's one_in). */
function oneIn(p, n) {
  if (p == null) return "–";
  return n && p < 0.5 ? `1 in ${Number(n).toLocaleString()}` : pct(p);
}

/** An edge in points with its sign: "+1.2 pts", "−2.1 pts". */
function pts(edge) {
  if (edge == null || !Number.isFinite(Number(edge))) return "–";
  const v = Number(edge) * 100;
  const text = Math.abs(v) < 0.05 ? "0.0" : Math.abs(v).toFixed(1);
  return `${v > 0.04 ? "+" : v < -0.04 ? "−" : ""}${text} pts`;
}

function cutLine(cut) {
  if (cut == null) return { value: "–", sub: "" };
  if (cut <= 0) return { value: "0%", sub: "these prices beat the books' consensus" };
  return { value: cutPct(cut), sub: "of the stake on average, every leg's cut compounded" };
}

function ticketKind(t) {
  const n = (t.legs || []).length;
  return n === 1 ? "Single" : `${n}-leg multi`;
}

const LABEL_KIND = { Safe: "good", Risky: "warn", Crazy: "info", Custom: "muted" };

/** One stat tile. ``sub`` is HTML (callers escape it): pieces joined by subLine(). */
function stat(label, value, sub, info = "") {
  return `<div class="bd-stat"><span>${esc(label)}${info ? ` ${explain(info)}` : ""}</span><b class="num">${esc(value)}</b>${sub ? `<small>${sub}</small>` : ""}</div>`;
}

/** Escaped pieces of a tile's small line, joined with " · " (empty pieces left out). */
const subLine = (...parts) => parts.filter(Boolean).join(" · ");
/** An amount that never breaks across lines (its minus sign stays with the number). */
const amount = (text) => `<span class="nowrap">${esc(text)}</span>`;

/** Whether any leg can push (a draw on draw no bet, a whole-number line): a push gives that leg's stake back. */
function canPush(t) {
  return (t.legs || []).some((leg) => Number(leg.push_prob) > 0 || Number(leg.inputs?.push) > 0);
}

/** The ticket's chance of winning money (the server's: price_ticket's chance.profit, or the joint block's). */
function profitChance(t) {
  const p = t.joint?.profit;
  if (p && typeof p === "object") return { p: p.chance ?? p.ai ?? null, n: p.one_in ?? null };
  if (p != null) return { p: Number(p), n: t.joint?.profit_one_in ?? null };
  return { p: t.chance?.profit?.ai ?? null, n: null };
}

/**
 * Whose chances the expected result uses, from the legs' chance sources: "Markets say" (the blend), the books'
 * margin-free chance, the goals model, or a mix. ``books`` is true when every leg is the books' (the second
 * "by the books" number would repeat the first).
 */
function expectedBasis(t) {
  const sources = new Set((t.legs || []).map((leg) => leg.chance_source || "unknown"));
  const only = sources.size === 1 ? [...sources][0] : null;
  if (only === "blend") return { words: "by Markets say", books: false };
  if (only === "books") return { words: "by the books, margin out", books: true };
  if (only === "model") return { words: "by the goals model", books: false };
  const names = [sources.has("blend") ? "Markets say where loaded" : "", sources.has("books") ? "the books" : "", sources.has("model") ? "the goals model" : ""].filter(Boolean);
  if (!names.length) return { words: "by the Builder's chances", books: false };
  const [first, ...rest] = names;
  const tail = rest.length ? `${first.startsWith("Markets") ? ", else " : " and "}${rest.join(" or ")}` : "";
  return { words: `by the Builder's chances (${first}${tail})`, books: false };
}

function legFlags(leg) {
  return (leg.flags || []).filter((f) => FLAG_BADGES[f]).map((f) => badge(...FLAG_BADGES[f])).join("");
}

function swapList(card, ti, li, leg) {
  const options = leg.swaps || [];
  const listId = `bd-sw-${ti}-${li}`;
  const back = card.swapped?.index === li ? card.original.legs[li] : null;
  const items = options.filter((o) => o.key !== leg.key).map((o, k) => {
    const flags = (o.flags || []).filter((f) => FLAG_BADGES[f]).map((f) => FLAG_BADGES[f][0]).join(", ");
    return `<li><button type="button" class="bd-swap-opt" data-bd-pick="${ti}:${li}:${options.indexOf(o)}">
      <span class="bd-so-top"><b>${esc(o.label)}</b><b class="num">${esc(Number(o.price).toFixed(2))}</b></span>
      <span class="bd-so-meta">${esc(gameShort(o.game, o.ref?.league))} · ${esc(o.book || "")} · chance ${esc(pct(o.chance))} · edge ${esc(pts(o.edge))} · price ${esc(o.age || "")} old${flags ? ` · ${esc(flags)}` : ""}</span>
      <span class="bd-so-meta">Ticket ${esc(Number(o.odds).toFixed(2))}x · every leg wins ${esc(pct(o.joint))}</span>
    </button></li>`;
  }).join("");
  const undo = back ? `<li><button type="button" class="bd-swap-opt back" data-bd-undo="${ti}">
      <span class="bd-so-top"><b>${icon("refresh", { size: 14 })} Back to ${esc(back.label)}</b><b class="num">${esc(priceText(back.odds))}</b></span>
      <span class="bd-so-meta">The ticket as it was built</span></button></li>` : "";
  return `<div class="bd-swaps" id="${listId}" role="region" aria-label="${esc(`Swaps for ${leg.label}`)}">
    ${items || undo ? `<p class="bd-hint">Each keeps every rule you set; tap one to re-price the ticket.</p><ul>${undo}${items}</ul>`
      : '<p class="bd-hint">No other leg keeps every rule here.</p>'}
  </div>`;
}

function legHtml(card, ti, leg, li) {
  const t = card.t;
  const locked = card.swapped && card.swapped.index !== li; // swaps were worked out for the ticket as built
  const n = (leg.swaps || []).filter((o) => o.key !== leg.key).length + (card.swapped?.index === li ? 1 : 0);
  const open = card.open === li;
  const why = (leg.why || []).filter(Boolean);
  const chance = leg.chance ?? leg.ai_prob;
  const source = SOURCES[leg.chance_source] || (leg.chance_source ? leg.chance_source : "");
  const age = leg.age ? `${leg.age} old` : "";
  const weak = t.weakest_leg === li && (t.legs || []).length > 1;
  return `<li class="bd-leg${weak ? " weak" : ""}">
    <div class="bd-leg-top"><b class="bd-pick">${esc(leg.label)}</b>
      <span class="bd-price"><b class="num">${esc(priceText(leg.odds || { decimal: Number(leg.price) }))}</b><small>${esc(leg.book || "")}</small></span></div>
    <div class="bd-leg-meta">${esc(gameShort(leg.game, leg.league))}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""} · ${esc(leagueTitle(leg.league))}</div>
    <div class="bd-leg-nums">
      <span>Chance <b class="num">${esc(pct(chance))}</b>${source ? ` <small>${esc(source)}</small>` : ""}</span>
      <span>Edge <b class="num">${esc(pts(leg.edge))}</b></span>
      ${age ? `<span class="${(leg.flags || []).includes("stale_price") ? "warn" : ""}">${icon("clock", { size: 12 })} <b>${esc(age)}</b></span>` : ""}
      ${weak ? badge("Weakest leg", "warn", "alert") : ""}${legFlags(leg)}
    </div>
    ${why.length || leg.risk ? `<p class="bd-why">${why.length ? `<b>Why:</b> ${why.map(esc).join(" · ")}` : ""}${leg.risk ? `${why.length ? " " : ""}<span class="bd-risk"><b>Risk:</b> ${esc(leg.risk)}</span>` : ""}</p>` : ""}
    ${card.swapped?.index === li && !why.length ? '<p class="bd-why muted">Swapped in: its reasons come with the next build.</p>' : ""}
    <div class="bd-leg-actions">
      ${locked ? `<span class="bd-hint">${icon("swap", { size: 14 })} Swaps here come back with the next build</span>`
        : `<button type="button" class="btn btn-ghost small" data-bd-swap="${ti}:${li}" aria-expanded="${open}" aria-controls="bd-sw-${ti}-${li}"${n ? "" : " disabled"}>${icon("swap", { size: 16 })}${n ? `Swap (${n})` : "No swaps"}</button>`}
    </div>
    ${open && !locked ? swapList(card, ti, li, leg) : ""}
  </li>`;
}

/**
 * When the explanation priced the ticket differently from the card (the run was limited to some bookmakers and
 * the explanation used the best price across all of them, or a price moved since the build), say so.
 */
function priceGapHtml(card, data, view) {
  const shown = Number(card.t.odds?.decimal);
  const got = Number(data.ticket?.odds?.decimal);
  if (!Number.isFinite(shown) || !Number.isFinite(got) || Math.abs(shown - got) < 0.005) return "";
  const books = view.result?.controls?.books;
  const why = Array.isArray(books) && books.length ? "the best price now across all bookmakers, not only the ones you chose" : "the price now";
  return `<p class="bd-hint">${icon("info", { size: 14 })} ${esc(`Explained at ${priceText(data.ticket.odds)}, ${why}; this card shows ${priceText(card.t.odds)}.`)}</p>`;
}

function explainHtml(card, ti, view) {
  const x = card.explain;
  if (!x || x.hidden) return "";
  if (x.busy) return `<div class="bd-explain" aria-busy="true"><p class="muted small">${icon("clock", { size: 14 })} Explaining this ticket…</p></div>`;
  if (x.error) return `<div class="bd-explain">${notice(x.error, "warn")}</div>`;
  const data = x.data || {};
  const e = data.explanation;
  const per = new Map((e?.per_leg || []).map((p) => [p.index, p.text]));
  const legs = data.ticket?.legs || [];
  const credits = data.credits;
  const left = credits ? ` AI allowance left today: ${Math.max(0, Math.round(credits.left_cents))}¢ of ${credits.daily_cents}¢.` : "";
  return `<div class="bd-explain" id="bd-x-${ti}" aria-live="polite">
    <h4>${icon("ask", { size: 16 })} ${e?.source === "claude" ? "The AI's explanation" : "The brain's facts"}</h4>
    ${priceGapHtml(card, data, view)}
    ${e?.verdict ? `<p class="lead">${esc(e.verdict)}</p>` : `<p class="small muted">${esc(data.note || (me().ai_enabled ? "The AI's write-up isn't available right now: here are the brain's facts." : "The AI's write-up isn't switched on for this server: here are the brain's facts."))}</p>`}
    ${legs.length ? `<ul class="bullets">${legs.map((leg) => {
      const text = per.get(leg.index) || (data.legs || []).find((l) => l.index === leg.index)?.notes?.slice(0, 2).join(" ");
      return text ? `<li><b>${esc(leg.label)}:</b> ${esc(text)}</li>` : "";
    }).join("")}</ul>` : ""}
    ${e?.case_against ? `<p class="small"><b>The case against:</b> ${esc(e.case_against)}</p>` : ""}
    <p class="method">${e?.source === "claude" ? `Written by Claude around the numbers above, which the app's code worked out. It can still be wrong.` : "Every number and fact here is computed in code."}${esc(left)}</p>
  </div>`;
}

function cardHtml(card, ti, view) {
  const t = card.t;
  const label = t.label || view.result.label || "Custom";
  const target = view.result.target;
  const joint = t.joint || {};
  const reach = t.reach_target;
  const cut = cutLine(t.book_cut);
  const e = t.expected || {};
  const jointP = joint.chance ?? t.chance?.ai;
  const jointN = joint.one_in ?? t.chance?.ai_one_in;
  const fair = joint.fair_odds;
  const fairText = fair ? esc(`fair odds ${Number(fair).toFixed(2)}`) : "";
  const below = (p) => (p != null && p < 0.5 ? esc(pct(p)) : "");
  const stats = [];
  if (canPush(t)) {
    // A pushed leg gives its stake back: every leg winning outright is not the only way to win money
    const profit = profitChance(t);
    stats.push(stat("Every leg wins outright", oneIn(jointP, jointN), below(jointP), "one_in"));
    stats.push(stat("Wins money", oneIn(profit.p, profit.n), subLine(profit.n ? below(profit.p) : "", fairText, esc("a pushed leg gives its stake back")), "fair_odds"));
  } else {
    stats.push(stat(target ? "Every leg wins" : "Chance it wins", oneIn(jointP, jointN), subLine(below(jointP), fairText), "one_in"));
  }
  if (target) {
    const p = reach?.chance;
    stats.push(stat(`Reaches ${Number(reach?.odds ?? target.low ?? target.odds).toFixed(2)}x or more`, p == null ? "–" : oneIn(p, reach?.one_in),
      esc(p == null ? "comes with the next build" : `the low end of your target ${Number(target.odds).toLocaleString()}x ± ${target.tolerance_pct}%`), "target_odds"));
  }
  const basis = expectedBasis(t);
  const main = e.ai ?? e.market;
  const byBooks = e.ai != null && e.market != null && !basis.books ? `${amount(signedMoney(e.market))} ${esc("by the books")}` : "";
  stats.push(stat("Expected result", main != null ? signedMoney(main) : "–",
    subLine(e.ai != null ? esc(basis.words) : e.market != null ? esc("by the books") : "", byBooks), "expected_result"));
  stats.push(stat("Books keep", cut.value, esc(cut.sub), "book_cut"));
  const notes = [...new Set([...(t.notes || []), ...(t.warnings || [])])];
  const inSlip = card.inSlip;
  return `<article class="card bd-ticket t-${esc(String(label).toLowerCase())}" data-bd-card="${ti}" aria-labelledby="bd-t${ti}-h">
    <header class="bd-thead">
      <div class="bd-ttitle"><h3 id="bd-t${ti}-h" tabindex="-1">Ticket ${ti + 1} <span>· ${esc(ticketKind(t))}</span></h3>${badge(label, LABEL_KIND[label] || "muted")}${card.swapped ? badge("Swapped", "info", "swap") : ""}</div>
      <div class="bd-odds"><b class="num">${esc(priceText(t.odds))}</b><small>${money(t.total_stake, { cents: true })} → ${money(t.max_payout, { cents: true })}</small></div>
    </header>
    ${(t.reasons || []).map((r) => notice(r, "danger")).join("")}
    <div class="bd-stats">${stats.join("")}</div>
    <ol class="bd-legs">${(t.legs || []).map((leg, li) => legHtml(card, ti, leg, li)).join("")}</ol>
    ${t.most_likely ? `<p class="bd-hint">Most likely: ${esc(mostLikelyText(t.most_likely, t.total_stake))} ${explain("most_likely")}</p>` : ""}
    ${notes.length ? `<ul class="warnings">${notes.map((n) => `<li>${icon("alert", { size: 14 })}<span>${esc(n)}</span></li>`).join("")}</ul>` : ""}
    <div class="bd-actions">
      ${inSlip ? `<button type="button" class="btn btn-secondary" data-bd-added="${ti}" aria-disabled="true">${icon("check", { size: 18 })}In your slip</button>`
        : `<button type="button" class="btn btn-primary" data-bd-add="${ti}">${icon("plus", { size: 18 })}Add to slip</button>`}
      <button type="button" class="btn btn-secondary" data-bd-explain="${ti}"${card.explain?.busy ? ' aria-disabled="true"' : ""}${card.explain?.data ? ` aria-expanded="${!card.explain.hidden}" aria-controls="bd-x-${ti}"` : ""}>${icon("bot", { size: 18 })}${card.explain?.data ? (card.explain.hidden ? "Show the explanation" : "Hide the explanation") : me().ai_enabled ? "Explain with AI" : "Explain"}</button>
      <button type="button" class="btn btn-secondary" data-bd-ask="${ti}">${icon("ask", { size: 18 })}Ask AI</button>
    </div>
    ${me().ai_enabled ? '<p class="bd-hint bd-cost">Explain and Ask use your daily AI allowance; building, swapping and adding are free.</p>' : ""}
    ${explainHtml(card, ti, view)}
  </article>`;
}

// --- near misses, blocked, ideas ---------------------------------------------------------------------------

function missesHtml(result) {
  const list = (result.near_misses || []).filter((m) => m.rule !== "no_prices");
  if (!list.length) return "";
  return `<details class="card section bd-misses"><summary>${icon("target", { size: 16 })}<span>Near misses <span class="muted">(${list.length})</span></span>${icon("down", { size: 18 })}</summary>
    <p class="bd-hint">Legs that nearly made it, each with the rule it failed ${explain("near_miss")}</p>
    <ul class="bd-miss-list">${list.map((m) => `<li>
      <span class="bd-miss-top"><b>${esc(m.label || "")}</b>${m.price ? `<b class="num">@ ${esc(Number(m.price).toFixed(2))}</b>` : ""}</span>
      <span class="bd-miss-why">${icon("alert", { size: 14 })}${esc(m.reason || m.text || "")}${(m.rules || []).length > 1 ? ` <small>(+${m.rules.length - 1} more rule${m.rules.length > 2 ? "s" : ""})</small>` : ""}</span>
      ${m.game ? `<span class="bd-hint">${esc(gameShort(m.game, m.ref?.league))}${m.book ? ` · ${esc(m.book)}` : ""}${m.chance != null ? ` · chance ${esc(pct(m.chance))}` : ""}</span>` : ""}
    </li>`).join("")}</ul>
  </details>`;
}

function unpricedHtml(result) {
  const rows = result.unpriced?.length ? result.unpriced.map((u) => u.reason || `${u.title}: no prices loaded`)
    : (result.near_misses || []).filter((m) => m.rule === "no_prices").map((m) => m.text || m.reason);
  if (!rows.length) return "";
  return `<details class="bd-unpriced"><summary>${icon("clock", { size: 14 })}<span>${rows.length} league${rows.length === 1 ? "" : "s"} in scope without prices yet</span>${icon("down", { size: 16 })}</summary>
    <ul>${rows.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    <p class="bd-hint">Nothing is fetched for a build: the latest odds load in the background (within the monthly allowance), then build again.</p>
  </details>`;
}

function blockedHtml(result) {
  const b = result.blocked || {};
  const rules = [...(b.rules || []).map((r) => ({ ...r, n: r.removed, verb: "ruled out" })), ...(b.ticket_rules || []).map((r) => ({ ...r, n: r.rejected, verb: "rejected" }))];
  const scope = b.scope || [];
  return `<section class="card section bd-blocked" aria-live="polite">
    <div class="section-head"><h2>${icon("alert", { size: 18 })} No ticket fits these controls</h2></div>
    <p class="lead">${esc(b.message || "Nothing fits every rule.")}</p>
    ${b.passed ? `<p class="bd-hint">${esc(`${b.passed.legs} leg${b.passed.legs === 1 ? "" : "s"} from ${b.passed.games} game${b.passed.games === 1 ? "" : "s"} passed every leg rule.`)}</p>` : ""}
    ${rules.length ? `<h3 class="subhead">What ruled them out</h3><ul class="bd-rules">${rules.map((r) => `<li><span>${esc(r.label)}</span><b class="num">${esc(Number(r.n).toLocaleString())} ${esc(r.verb)}</b></li>`).join("")}</ul>` : ""}
    ${scope.length ? `<p class="bd-hint">Outside your scope: ${esc(scope.map((r) => `${r.label} (${r.removed})`).join(", "))}.</p>` : ""}
    ${(b.relax || []).length ? `<h3 class="subhead">Try relaxing</h3><div class="chip-row wrap">${b.relax.map((r) => `<span class="chip">${icon("sliders", { size: 14 })}${esc(r)}</span>`).join("")}</div>` : ""}
    <div class="empty-actions" style="justify-content:flex-start;margin-top:14px"><button type="button" class="btn btn-secondary" data-bd-open-controls>${icon("sliders", { size: 18 })}Change the controls</button></div>
  </section>`;
}

function ideasHtml(result) {
  const ideas = result.ideas || [];
  if (!ideas.length) return "";
  return `<section class="card section bd-ideas"><div class="section-head"><h2>${icon("swap", { size: 18 })} Ideas only</h2>${badge("Not built or priced", "muted")}</div>
    <ul class="bullets">${ideas.map((i) => `<li>${esc(i.text)}</li>`).join("")}</ul></section>`;
}

function statsLine(view) {
  const s = view.result.stats || {};
  const parts = [];
  if (s.candidates != null) parts.push(`${Number(s.passed || 0).toLocaleString()} of ${Number(s.candidates).toLocaleString()} legs passed the rules (${Number(s.games || 0).toLocaleString()} games)`);
  if (s.combos != null) parts.push(`${Number(s.combos).toLocaleString()} combinations tried`);
  if (s.ms != null) parts.push(`${Math.round(s.ms)} ms`);
  return `<p class="method">${esc(parts.join(" · "))}${parts.length ? ". " : ""}Built ${esc(timeLabel(new Date(view.at).toISOString()))} from the prices already saved (nothing fetched for it). Prices: bookmakers you can bet at only.</p>`;
}

export function resultHtml(view) {
  const result = view.result;
  const cards = view.cards || [];
  const notes = result.notes || [];
  const head = cards.length
    ? `<div class="bd-rhead"><h2>${esc(`${cards.length} ticket${cards.length === 1 ? "" : "s"}`)} ${badge(result.label || "Custom", LABEL_KIND[result.label] || "muted")}</h2>
        <span class="muted small">${result.stake_given ? `at your ${money(result.stake, { cents: true })} stake` : `priced at ${money(result.stake, { cents: true })}`}</span></div>`
    : "";
  return `<div class="bd-results${view.busy ? " bd-stale" : ""}"${view.busy ? ' aria-busy="true"' : ""}>
    ${head}
    ${notes.length ? `<ul class="warnings bd-notes">${notes.map((n) => `<li>${icon(/expect to lose/i.test(n) ? "zap" : "info", { size: 14 })}<span>${esc(n)}</span></li>`).join("")}</ul>` : ""}
    ${unpricedHtml(result)}
    ${cards.length ? `<div class="bd-cards">${cards.map((card, i) => cardHtml(card, i, view)).join("")}</div>` : blockedHtml(result)}
    ${ideasHtml(result)}
    ${missesHtml(result)}
    ${statsLine(view)}
  </div>`;
}
