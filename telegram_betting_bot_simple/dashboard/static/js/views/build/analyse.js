// "Analyse with AI" on the Build screen (POST /api/tickets/analyse).
//
// The server prices the slip again, lists plain facts about each pick (injuries, the AI against the
// market, ESPN's view, the analysts'), works out other ways to play the same picks, and, when Claude is
// set up and the person has AI credits, has Claude write a verdict, a line per pick and the honest case
// against around those numbers (explanation is null otherwise, and the card shows the brain's facts on
// their own). The card opens with four numbers, as the product blueprint asks: the market's chance, the
// AI's, what the bookmaker keeps and the expected result in money. Applying an alternative only ever
// drops picks or changes the bet type and stake; it never adds a pick. Alternatives are worked out for the
// slip as it was analysed, so once the slip changes they can't be applied (the card asks for a fresh
// analysis), and the card steps aside when none of its picks are left on the slip. On a break the server
// refuses an analysis, so none runs and the last one steps aside until the break ends.

import { api, isQuiet } from "../../api.js";
import { slip, me, legKey, openOnboarding, onBreak } from "../../app.js";
import { esc, pct, chanceText, money, signedMoney, priceText, notice, skeleton, errorState, explain, badge } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { ticketName, mostLikelyText, chanceWord, redraw, reveal, gateHint, warningsList, cutPct, gameShort } from "./common.js";

let last = null; // {text, keys, data}: the last analysis this visit, so coming back to Build costs no more credits

const STEPS = [
  "Checking each pick's latest price",
  "Reading the injury report",
  "Comparing the AI with the market",
  "Pricing other ways to play it",
];

function tile(label, value, sub, info = "") {
  return `<div class="bx-tile"><span>${esc(label)} ${info ? explain(info) : ""}</span><b class="num">${esc(value)}</b>${sub ? `<small>${sub}</small>` : ""}</div>`;
}

function oneInValue(p, oneIn) {
  if (p == null) return "–";
  return oneIn && p < 0.5 ? `1 in ${Number(oneIn).toLocaleString()}` : pct(p);
}

/** The four numbers the analysis opens with. */
function fourNumbers(t) {
  const c = t.chance || {};
  const e = t.expected || {};
  const word = chanceWord(c).replace("Chance", "chance");
  const expected = e.ai ?? e.market;
  const expectedSub = [e.ai != null ? "by the AI's chances" : "by the market's chances",
    e.ai != null && e.market != null ? `${esc(signedMoney(e.market))} by the market's` : ""].filter(Boolean).join("<br>");
  return `<div class="bx-tiles four">
    ${tile(`Market ${word}`, oneInValue(c.market, c.market_one_in), c.market != null ? `${pct(c.market)}, margin removed` : "no market chance", "market_chance")}
    ${tile(`AI ${word}`, oneInValue(c.ai, c.ai_one_in), c.ai != null ? `${pct(c.ai)}${c.label ? ` · ${esc(c.label)}` : ""}` : "no AI model for every pick", "ai_chance")}
    ${tile("Bookmaker keeps", cutPct(t.book_cut), "of the stake, on average", "book_cut")}
    ${tile("Expected result", expected == null ? "–" : signedMoney(expected), expectedSub, "expected_result")}
  </div>`;
}

function legsHtml(data) {
  const t = data.ticket;
  const written = new Map((data.explanation?.per_leg || []).map((p) => [p.index, p.text]));
  const facts = new Map((data.legs || []).map((leg) => [leg.index, leg.notes || []]));
  return `<ol class="bx-legnotes">${(t.legs || []).map((leg) => {
    const notes = facts.get(leg.index) || [];
    const text = written.get(leg.index);
    const chances = [leg.ai_prob != null ? `AI ${pct(leg.ai_prob)}` : "", leg.market_prob != null ? `Market ${pct(leg.market_prob)}` : ""].filter(Boolean).join(" · ");
    const bullets = (items) => `<ul class="bullets">${items.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
    // Without a write-up the first facts lead; the rest fold away so a long slip stays readable
    const list = !notes.length ? "" : text || notes.length <= 4 ? bullets(notes)
      : `${bullets(notes.slice(0, 3))}<details class="bx-more"><summary>${icon("info", { size: 16 })}${notes.length - 3} more facts</summary>${bullets(notes.slice(3))}</details>`;
    return `<li>
      <div class="row-between"><b>${esc(leg.label)}</b><b class="num">${esc(priceText(leg.odds))}</b></div>
      <div class="small muted">${esc(gameShort(leg.game, leg.league))}${chances ? ` · ${esc(chances)}` : ""}</div>
      ${t.weakest_leg === leg.index && (t.legs || []).length > 1 ? `<div class="bx-flags">${badge("Weakest pick", "warn", "alert")}</div>` : ""}
      ${text ? `<p>${esc(text)}</p>` : ""}
      ${list ? (text ? `<details class="bx-more"><summary>${icon("info", { size: 16 })}Facts from the brain (${notes.length})</summary>${list}</details>` : list) : ""}
    </li>`;
  }).join("")}</ol>`;
}

function caseAgainst(data) {
  const x = data.explanation;
  if (x?.case_against) return `<p>${esc(x.case_against)}</p>`;
  // No write-up: the same points, from the ticket's own numbers
  const t = data.ticket;
  const points = [];
  const weak = (t.legs || [])[t.weakest_leg];
  if (weak && (t.legs || []).length > 1) {
    const p = weak.ai_prob ?? weak.market_prob;
    points.push(`The weakest pick is ${weak.label}${p != null ? ` (${pct(p)})` : ""}.`);
  }
  if (t.most_likely) points.push(`The most likely result: ${mostLikelyText(t.most_likely, t.total_stake)}.`);
  if (t.expected?.market != null) points.push(`By the market's chances the expected result is ${signedMoney(t.expected.market)}.`);
  return points.length ? `<ul class="bullets">${points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "";
}

function alternativesHtml(data, isStale) {
  const alts = data.alternatives || [];
  if (!alts.length) return "";
  return `<h3 class="subhead">Other ways to play it</h3>
    ${isStale ? '<p class="hint" style="margin-top:0">These were worked out for the slip as it was analysed. Analyse again to compare them with your slip now.</p>' : ""}
    <div class="bx-alts">${alts.map((alt, i) => {
      const t = alt.ticket;
      const c = t.chance || {};
      const chance = c.ai != null ? `AI ${chanceText(c.ai, c.ai_one_in)}` : c.market != null ? `Market ${chanceText(c.market, c.market_one_in)}` : "";
      const e = t.expected?.ai ?? t.expected?.market;
      return `<div class="bx-alt">
        <b>${esc(alt.title)}</b>
        ${alt.summary ? `<p>${esc(alt.summary)}</p>` : ""}
        <div class="small">${esc(ticketName(t))} · costs ${money(t.total_stake, { cents: true })}${chance ? ` · ${esc(chanceWord(c).toLowerCase())} ${esc(chance)}` : ""}${e != null ? ` · expected ${esc(signedMoney(e))}` : ""}</div>
        ${isStale ? `<button type="button" class="btn btn-secondary small" disabled>${icon("refresh", { size: 16 })}Analyse again to use this</button>`
          : `<button type="button" class="btn btn-secondary small" data-apply-alt="${i}">${icon("check", { size: 16 })}Use this instead</button>`}
      </div>`;
    }).join("")}</div>`;
}

function footer(data) {
  const x = data.explanation;
  const credits = data.credits;
  const left = credits ? ` AI credits left today: ${Math.max(0, Math.round(credits.left_cents))}¢ of ${credits.daily_cents}¢.` : "";
  if (x?.source === "claude") {
    return `<p class="method">${icon("bot", { size: 14 })} Written by Claude${x.model ? ` (${esc(x.model)})` : ""} around the numbers above, which the app's code worked out. It can still be wrong.${esc(left)}</p>`;
  }
  return `<p class="method">Every number and fact here is computed in code from the odds, the model and ESPN.${esc(left)}</p>`;
}

function cardHtml(data, isStale) {
  const t = data.ticket;
  const x = data.explanation;
  let intro;
  if (x?.verdict) {
    intro = `<p class="lead">${esc(x.verdict)}</p>`;
  } else if (!t.valid) {
    intro = notice("Fix the slip first: the AI writes its verdict once every pick can be priced together.", "info");
  } else {
    intro = notice(data.note || (me().ai_enabled
      ? "The AI's write-up isn't available right now, so here are the brain's facts on their own."
      : "Claude's write-up isn't switched on for this server (it needs ANTHROPIC_API_KEY), so here are the brain's facts on their own."), "info");
  }
  const source = x?.source === "claude" ? badge("Claude", "info", "bot") : x ? badge("Template", "muted") : badge("Brain only", "muted");
  return `<section class="card section bx-analysis" id="analysis" aria-live="polite">
    <div class="section-head"><h2>${icon("ask", { size: 18 })} Analysis · ${esc(ticketName(t))}</h2>${source}</div>
    ${isStale ? `${notice("Your slip has changed since this analysis.", "warn")}<button type="button" class="btn btn-secondary small" data-act="reanalyse" style="margin-top:8px">${icon("refresh", { size: 16 })}Analyse again</button>` : ""}
    ${(t.reasons || []).map((r) => notice(r, "danger")).join("")}
    ${fourNumbers(t)}
    ${intro}
    <h3 class="subhead">Pick by pick</h3>
    ${legsHtml(data)}
    <h3 class="subhead">The honest case against</h3>
    ${caseAgainst(data)}
    ${alternativesHtml(data, isStale)}
    ${warningsList(t.warnings)}
    ${footer(data)}
  </section>`;
}

function loadingHtml(step) {
  const steps = me().ai_enabled ? [...STEPS, "Writing it up"] : STEPS;
  return `<section class="card section bx-analysis" id="analysis" aria-busy="true">
    <div class="section-head"><h2>${icon("ask", { size: 18 })} Analysing your slip…</h2></div>
    <ol class="bx-steps" aria-live="polite">${steps.map((s, i) => {
      const state = i < step ? "done" : i === step ? "now" : "";
      return `<li class="${state}">${icon(i < step ? "check" : i === step ? "clock" : "minus", { size: 16 })}<span>${esc(s)}${i === step ? "…" : ""}</span></li>`;
    }).join("")}</ol>
    <div style="margin-top:12px">${skeleton("lines", 3)}</div>
  </section>`;
}

/**
 * The analysis card in ``box``. Returns {run, draw, busy, stop}.
 *   run()   analyse the current slip (ignored while one is running)
 *   draw()  redraw (marks the card as stale once the slip changes)
 * ``onBusy(busy)`` runs when an analysis starts or ends; ``onApply(alt)`` after an alternative is used.
 */
export function createAnalysis(box, { signal, onBusy = () => {}, onApply = () => {} } = {}) {
  let busy = false;
  let timer = null;

  const isStale = () => Boolean(last) && last.text !== JSON.stringify(slip.body());
  // None of the analysed picks are on the slip any more: the card is about another slip
  const unrelated = () => !last.keys.some((key) => slip.has(key));
  const draw = () => {
    if (busy) return;
    if (!last || !slip.count || unrelated() || onBreak()) {
      box.innerHTML = "";
      return;
    }
    redraw(box, cardHtml(last.data, isStale()));
  };

  const run = async ({ force = false } = {}) => {
    if (busy || !slip.count || !box.isConnected || onBreak()) return;
    if (!force && last && !isStale()) {
      // The same slip was analysed already: show that again rather than spend more AI credits
      draw();
      reveal(box);
      return;
    }
    const body = slip.body();
    const text = JSON.stringify(body);
    busy = true;
    onBusy(true);
    let step = 0;
    box.innerHTML = loadingHtml(step);
    reveal(box);
    const steps = me().ai_enabled ? STEPS.length + 1 : STEPS.length;
    timer = setInterval(() => {
      step = Math.min(step + 1, steps - 1);
      if (box.isConnected) box.innerHTML = loadingHtml(step);
    }, 900);
    try {
      const data = await api("/api/tickets/analyse", { method: "POST", body, signal });
      last = { text, keys: body.legs.map(legKey), data };
      haptic.notify("success");
    } catch (error) {
      if (isQuiet(error) || !box.isConnected) return;
      haptic.notify("error");
      const hint = gateHint(error);
      box.innerHTML = `<section class="card section" id="analysis">${errorState(error, { retry: false, title: "Couldn't analyse the slip" })}
        <div class="empty-actions">${hint?.kind === "age" ? '<button type="button" class="btn btn-primary" data-act="confirm-age">Confirm age</button>'
          : hint ? "" : `<button type="button" class="btn btn-secondary" data-act="reanalyse">${icon("refresh", { size: 16 })}Try again</button>`}</div></section>`;
      return;
    } finally {
      clearInterval(timer);
      busy = false;
      onBusy(false);
    }
    if (!box.isConnected) return;
    draw();
    reveal(box);
  };

  box.addEventListener("click", (event) => {
    const act = event.target.closest("[data-act]")?.dataset.act;
    if (act === "reanalyse") {
      run({ force: true });
      return;
    }
    if (act === "confirm-age") {
      openOnboarding();
      return;
    }
    const use = event.target.closest("[data-apply-alt]");
    if (use && last) {
      const alt = (last.data.alternatives || [])[Number(use.dataset.applyAlt)];
      // Only onto the slip it was worked out for: on a changed slip it would price a ticket nobody saw
      if (!alt || !alt.refs?.length || isStale() || !alt.refs.every((ref) => slip.has(ref))) {
        draw();
        return;
      }
      const keep = new Set(alt.refs.map(legKey));
      for (const leg of [...slip.legs]) {
        if (!keep.has(leg.key)) slip.remove(leg.key);
      }
      const t = alt.ticket;
      slip.set({
        type: t.type,
        stake: String(Number(t.unit_stake)),
        k: t.type === "round_robin" ? t.k : slip.k,
        system: t.type === "system" ? t.system : slip.system,
      });
      haptic.select();
      onApply(alt);
    }
  });

  draw();
  return {
    run,
    draw,
    get busy() {
      return busy;
    },
    stop() {
      clearInterval(timer);
    },
  };
}
