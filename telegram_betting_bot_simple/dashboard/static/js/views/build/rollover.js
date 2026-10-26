// The rollover planner (#/build/rollover): four ways to play the same run of days side by side, with the
// real chances first (POST /api/rollover/compare), then "Start paper plan" (POST /api/plans).
//
// The owner's rule: a classic all-in rollover is never shown alone. Classic, bank half, pot-and-ladder
// and flat stakes sit next to each other with the chance each lasts every day, ends in profit, its typical
// end, the chance of ending at half the start or less, and the expected cost in money, under the honest
// sentence ("To turn $10 into $10,240 … about 1 in 1,600"). Every number is engine/rollover.py's. Plans
// start on paper and never roll on by themselves: the person chooses each day's picks in My bets.
// A plan starts only from the comparison on screen for the inputs on screen: while an input is invalid,
// or a new comparison is due or loading, the table is dimmed and Start waits. On a break or in the 24-hour
// picks pause the server refuses new plans: Start is off up front with one plain line saying until when,
// and the comparison (just maths) stays open.

import { api, isQuiet, prefs } from "../../api.js";
import { navigate, openOnboarding, replaceAddress, loadMe, onMeChange } from "../../app.js";
import { esc, pct, chanceText, money, notice, skeleton, errorState, explain, segmented, confirmSheet, toast, badge } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { modesHtml, chooseMode } from "./slip.js";
import { groupCurves, survivalChart, bindSurvivalChart } from "./chart.js";
import { cleanStake, warningsList, redraw, reveal, gateHint, picksBlock, blockNote } from "./common.js";

const FORM_KEY = "rolloverForm";
const ORDER = ["classic", "bank_half", "ladder", "flat"];
const SHORT = { classic: "Classic", bank_half: "Bank half", ladder: "Pot & ladder", flat: "Flat 5%" };
const STARTS = ["10", "25", "50"];
const BLOCK_ID = "bx-block-plan"; // the line that says why Start is off
const ODDS = ["1.50", "2.00", "3.00"];

function cleanOdds(value) {
  const text = String(value ?? "").trim();
  if (!/^\d{1,2}(\.\d{1,2})?$/.test(text)) return null;
  const n = Number(text);
  return n >= 1.2 && n <= 20 ? text : null;
}

function cleanDays(value) {
  const n = Number(String(value ?? "").trim());
  return Number.isInteger(n) && n >= 1 && n <= 60 ? n : null;
}

function loadForm() {
  const saved = prefs.get(FORM_KEY, {}) || {};
  const legs = Number(saved.legs_per_day);
  return {
    start: cleanStake(saved.start) || "10",
    days: cleanDays(saved.days) || 10,
    min_odds: cleanOdds(saved.min_odds) || "2.00",
    legs_per_day: [1, 2, 3].includes(legs) ? legs : 1,
    strategy: ORDER.includes(saved.strategy) ? saved.strategy : null,
  };
}

/**
 * The inputs a link brings for this visit (the chat's plan card opens #/build/rollover?start=10&days=5&
 * min_odds=2.00&legs_per_day=1), each checked like a typed one: only the valid ones override the saved form.
 */
function formFromQuery(query) {
  const out = {};
  const start = cleanStake(query.get("start"));
  const days = cleanDays(query.get("days"));
  const odds = cleanOdds(query.get("min_odds"));
  const legs = Number(query.get("legs_per_day") ?? query.get("legs"));
  if (start) out.start = Number.isInteger(Number(start)) ? String(Number(start)) : start; // "10.00" lights the $10 chip
  if (days) out.days = days;
  if (odds) out.min_odds = Number(odds).toFixed(2);
  if ([1, 2, 3].includes(legs)) out.legs_per_day = legs;
  return out;
}

function chip(name, value, current, label) {
  const on = String(current) === String(value);
  return `<button type="button" class="chip${on ? " on" : ""}" aria-pressed="${on}" data-${name}="${esc(value)}">${esc(label)}</button>`;
}

function formHtml(form) {
  return `<form class="card bx-form" data-r-form novalidate>
    <div class="field"><label class="label" for="r-start">Start with</label>
      <div class="stake-row">${STARTS.map((v) => chip("r-start", v, form.start, `$${v}`)).join("")}
        <span class="money-input"><span aria-hidden="true">$</span><input id="r-start" class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(form.start)}"></span></div></div>
    <div class="field"><label class="label" for="r-odds">Odds each day ${explain("decimal_odds")}</label>
      <div class="stake-row">${ODDS.map((v) => chip("r-odds", v, form.min_odds, v)).join("")}
        <span class="money-input"><input id="r-odds" class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(form.min_odds)}" style="padding-left:12px"></span></div></div>
    <div class="field line-control" style="margin-top:18px">
      <span><label class="label" for="r-days" style="margin:0">Days</label><span class="hint" style="display:block;margin-top:2px">1 to 60</span></span>
      <span class="stepper bx-stepper" role="group" aria-label="Days">
        <button type="button" data-r-days="-1" aria-label="One day fewer"${form.days <= 1 ? " disabled" : ""}>${icon("minus", { size: 18 })}</button>
        <input id="r-days" type="text" inputmode="numeric" autocomplete="off" value="${form.days}" aria-label="Days">
        <button type="button" data-r-days="1" aria-label="One day more"${form.days >= 60 ? " disabled" : ""}>${icon("plus", { size: 18 })}</button>
      </span>
    </div>
    <div class="field"><span class="label">Picks each day</span>
      ${segmented("r-legs", [1, 2, 3].map((n) => ({ value: n, label: n === 1 ? "1 pick" : `${n} picks` })), form.legs_per_day, { label: "Picks each day" })}
      <p class="hint">Building a day from more picks pays the bookmaker's cut on each one.</p></div>
    <div class="form-error" role="alert" data-r-error></div>
  </form>`;
}

/** "$1,143" from the server's "1143.30": whole dollars where the table is tight (the day tables keep cents). */
function whole(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `$${Math.round(n).toLocaleString()}` : "–";
}

function cell(text, sub = "") {
  return `<b>${esc(text)}</b>${sub ? `<small>${esc(sub)}</small>` : ""}`;
}

function mostLikely(m) {
  const end = m.most_likely_end;
  if (!end) return "–";
  return end.completes ? cell("Lasts", pct(end.probability)) : cell(`Ends day ${end.day}`, pct(end.probability));
}

function tableHtml(data, chosen) {
  const s = data.strategies || {};
  const keys = ORDER.filter((k) => s[k]?.market);
  const days = data.days;
  const rows = [
    [`Lasts all ${days} day${days === 1 ? "" : "s"}`, (m) => (m.complete_one_in && m.chance_complete < 0.5
      ? cell(`1 in ${Number(m.complete_one_in).toLocaleString()}`, pct(m.chance_complete)) : cell(pct(m.chance_complete)))],
    ["Ends in profit", (m) => cell(chanceText(m.chance_profit))],
    ["Typical end (median)", (m) => cell(whole(m.median_end))],
    ["Ends at half or less", (m) => cell(pct(m.chance_half_or_less))],
    [`Expected cost ${explain("expected_result")}`, (m) => cell(money(m.expected_cost, { cents: true }), `ends ${money(m.expected_end, { cents: true })}`)],
    ["Most likely", mostLikely],
    ["Best case", (m) => cell(whole(m.best_end))],
  ];
  return `<div class="table-wrap bx-compare"><table>
    <caption class="sr-only">The four plans side by side, by the market's chances</caption>
    <thead><tr><th scope="col">By the market's chances</th>${keys.map((k) => `<th scope="col" class="${k === chosen ? "on" : ""}">${esc(SHORT[k])}
      ${k === "classic" ? `<br>${badge("High risk", "warn", "alert")}` : ""}${s[k].default ? '<br><span class="tag">default</span>' : ""}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([label, fn]) => `<tr><th scope="row">${label}</th>${keys.map((k) => `<td class="num${k === chosen ? " on" : ""}">${fn(s[k].market)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>
  <p class="hint">“Lasts”: classic and bank half need every day to win; pot-and-ladder must stay above half the start; flat stakes must not run out.</p>`;
}

function dayTables(data) {
  const ladder = data.ladder_table || [];
  const bank = data.bank_half_table || [];
  const chance = (row) => (row.market_one_in && row.market_chance < 0.5 ? `1 in ${Number(row.market_one_in).toLocaleString()}` : pct(row.market_chance));
  return `<details class="bx-more"><summary>${icon("chart", { size: 16 })}Classic, day by day, if every day wins</summary>
      <div class="table-wrap"><table><thead><tr><th>Day</th><th>Stake</th><th>Returns</th><th>Chance of getting here</th></tr></thead>
      <tbody>${ladder.map((r) => `<tr><td>${r.day}</td><td class="num">${money(r.stake, { cents: true })}</td><td class="num">${money(r.returns, { cents: true })}</td><td class="num">${esc(chance(r))}</td></tr>`).join("")}</tbody></table></div>
    </details>
    <details class="bx-more"><summary>${icon("chart", { size: 16 })}Bank half, day by day, if every day wins</summary>
      <div class="table-wrap"><table><thead><tr><th>Day</th><th>Stake</th><th>Returns</th><th>Banked</th><th>Banked so far</th><th>Next stake</th></tr></thead>
      <tbody>${bank.map((r) => `<tr><td>${r.day}</td><td class="num">${money(r.stake, { cents: true })}</td><td class="num">${money(r.returns, { cents: true })}</td><td class="num">${money(r.bank, { cents: true })}</td><td class="num">${money(r.banked_total, { cents: true })}</td><td class="num">${money(r.roll, { cents: true })}</td></tr>`).join("")}</tbody></table></div>
    </details>`;
}

/** Start's label: why it waits, or what it starts. */
function goLabel(chosen, { starting, blocked }) {
  if (starting) return "Starting…";
  if (blocked === "invalid") return "Fix the inputs to start";
  if (blocked === "waiting") return "Updating the numbers…";
  return `Start paper plan · ${SHORT[chosen] || ""}`;
}

function goHtml(chosen, { starting = false, active = null, blocked = null, block = null } = {}) {
  if (block) { // a break or the picks pause: off, and the line under it says until when
    return `<button type="button" class="btn btn-primary block" style="margin-top:14px" data-r-go disabled aria-describedby="${BLOCK_ID}">${icon("pause", { size: 18 })}Start paper plan</button>
      ${blockNote(block, BLOCK_ID)}`;
  }
  const mark = blocked === "invalid" ? "alert" : blocked === "waiting" ? "clock" : "check";
  return `<button type="button" class="btn btn-primary block" style="margin-top:14px" data-r-go${starting || active || blocked ? " disabled" : ""}>${icon(mark, { size: 18 })}${esc(goLabel(chosen, { starting, blocked }))}</button>`;
}

function resultHtml(data, chosen, { started = null, active = null, starting = false, blocked = null, block = null, inTelegram = false, width } = {}) {
  const s = data.strategies || {};
  const pick = s[chosen];
  const series = groupCurves(ORDER.filter((k) => s[k]?.market).map((k) => ({ name: SHORT[k], curve: s[k].market.survival })));
  const cost = data.sentences?.cost?.[chosen];
  return `<section class="card section${blocked ? " bx-stale" : ""}" id="rollover-result" aria-live="polite">
    <div class="section-head"><h2>Side by side ${explain("rollover")}</h2><span class="muted small">daily chance ${esc(pct(data.p_market))}</span></div>
    <p class="bx-sentence">${esc(data.summary)}</p>
    <p class="hint">${esc(data.requirement)}</p>
    ${tableHtml(data, chosen)}
    <h3 class="subhead">Still going, day by day</h3>
    <div data-chart>${survivalChart(series, { width, caption: "By the market's chances. Most rollover runs end in the first few days." })}</div>
    <div class="bx-strategy">
      <span class="label">Plan to start</span>
      ${segmented("r-strategy", ORDER.filter((k) => s[k]).map((k) => ({ value: k, label: SHORT[k] })), chosen, { label: "Plan to start" })}
      ${pick ? `<p class="bx-rule"><b>${esc(pick.title)}.</b> ${esc(pick.rule)}</p>` : ""}
      ${cost ? `<p class="hint">${esc(cost)}</p>` : ""}
    </div>
    ${warningsList(data.warnings)}
    ${dayTables(data)}
    <div data-plan-note>${planNote({ started, active })}</div>
    ${inTelegram && !block ? "" : `<div data-r-go-box>${goHtml(chosen, { starting, active, blocked, block })}</div>`}
    <p class="method">Paper mode: nothing is bet and nothing rolls on by itself. Each day you choose the picks in My bets, the plan works out the stake, and the result is graded when the games end.</p>
  </section>`;
}

function planNote({ started, active }) {
  if (started) {
    return notice(`Paper plan started: ${started.title}. ${started.message || ""}`, "good")
      + '<a class="btn btn-secondary block" style="margin-top:8px" href="#/bets?tab=plans">Open it in My bets</a>';
  }
  if (active) {
    return notice(`You have a plan running: ${active.title}, day ${active.day} of ${active.days}. Finish or stop it in My bets before starting another.`, "info")
      + '<a class="btn btn-secondary block" style="margin-top:8px" href="#/bets?tab=plans">Open My bets</a>';
  }
  return "";
}

/**
 * Draw the planner in ``root``. Returns {mainButton(), stop(), ready} for build.js; Telegram's main
 * button is "Start paper plan" (none while picks are off: the page's own Start shows, switched off, with
 * the line saying why).
 */
export function mountRollover(root, { params, refresh, inTelegram }) {
  let form = { ...loadForm(), ...formFromQuery(params.query) };
  // The link's numbers are used once: a reload or Back shows the planner as last left, not the link again
  if ([...params.query.keys()].length) replaceAddress("#/build/rollover");
  let data = null;
  let chosen = form.strategy;
  let started = null;
  let active = null;
  let starting = false;
  let controller = null;
  let timer = null;
  let waiting = false; // a comparison is due (the typing pause) or on its way: the one shown is out of date
  let invalid = false; // an input on screen can't be compared: the one shown is for earlier inputs
  let block = picksBlock(); // a break or the picks pause: the server would refuse a new plan
  const blocked = () => (invalid ? "invalid" : waiting ? "waiting" : null);

  root.innerHTML = `${modesHtml("rollover")}
    <div class="page-head" style="margin-top:14px"><h1>Rollover planner</h1>
      <p>Four ways to play the same days, side by side, with the real chances first. Plans start on paper. ${explain("rollover")}</p></div>
    <div data-r-form-box>${formHtml(form)}</div>
    <div data-r-result></div>`;
  const out = root.querySelector("[data-r-result]");
  const save = () => prefs.set(FORM_KEY, { ...form, strategy: chosen });

  const draw = () => {
    if (!data) return;
    const strategies = data.strategies || {};
    if (!chosen || !strategies[chosen]) chosen = data.default && strategies[data.default] ? data.default : ORDER.find((k) => strategies[k]);
    // The chart is drawn at the card's inner width (the card has 14 px of padding and a 1 px border a side)
    const width = out.getBoundingClientRect().width - 30;
    redraw(out, resultHtml(data, chosen, { started, active, starting, blocked: blocked(), block, inTelegram, width }));
    const s = data.strategies;
    bindSurvivalChart(out.querySelector(".bx-chart"), groupCurves(ORDER.filter((k) => s[k]?.market).map((k) => ({ name: SHORT[k], curve: s[k].market.survival }))));
    refresh();
  };

  // Dim the table and hold Start (and Telegram's main button) while the inputs and the numbers disagree
  const syncGo = () => {
    if (data) {
      out.querySelector("#rollover-result")?.classList.toggle("bx-stale", Boolean(blocked()));
      const go = out.querySelector("[data-r-go-box]");
      if (go) go.innerHTML = goHtml(chosen, { starting, active, blocked: blocked(), block });
    }
    refresh();
  };

  const read = () => {
    const error = root.querySelector("[data-r-error]");
    const start = cleanStake(root.querySelector("#r-start").value);
    const odds = cleanOdds(root.querySelector("#r-odds").value);
    const days = cleanDays(root.querySelector("#r-days").value);
    if (!start) error.textContent = "Start with an amount between $0.01 and $100,000.";
    else if (!odds) error.textContent = "Odds each day are between 1.20 and 20.00 (decimal).";
    else if (!days) error.textContent = "Days are a whole number from 1 to 60.";
    else error.textContent = "";
    if (!start || !odds || !days) return null;
    return { start, min_odds: odds, days, legs_per_day: form.legs_per_day };
  };

  const compare = async () => {
    clearTimeout(timer);
    const values = read();
    controller?.abort(); // whatever was on its way is for other inputs
    if (!values) {
      invalid = true;
      waiting = false;
      syncGo();
      return;
    }
    invalid = false;
    waiting = true;
    form = { ...form, ...values };
    save();
    controller = new AbortController();
    const mine = controller;
    params.signal.addEventListener("abort", () => mine.abort(), { once: true });
    if (data) syncGo();
    else out.innerHTML = `<div class="card section">${skeleton("lines", 6)}</div>`;
    try {
      const body = { start: form.start, days: form.days, min_odds: form.min_odds, legs_per_day: form.legs_per_day };
      const result = await api("/api/rollover/compare", { method: "POST", body, signal: mine.signal });
      if (mine !== controller || !root.isConnected) return;
      waiting = false;
      data = result;
      started = null;
      draw();
    } catch (error) {
      if (isQuiet(error) || mine !== controller || !root.isConnected) return;
      waiting = false;
      data = null;
      out.innerHTML = `<div class="section">${errorState(error, { retry: false, title: "Couldn't compare the plans" })}</div>`;
      refresh();
    }
  };
  const later = () => {
    clearTimeout(timer);
    waiting = true;
    syncGo();
    timer = setTimeout(compare, 400);
  };

  /** The comparison on screen is for the inputs on screen. */
  const current = () => {
    const values = read();
    return Boolean(values && data) && Number(values.start) === Number(data.start) && values.days === Number(data.days)
      && Number(values.min_odds) === Number(data.min_odds) && values.legs_per_day === Number(data.legs_per_day);
  };

  const startPlan = async () => {
    if (!data || starting || active || block || blocked()) return;
    if (!current()) {
      compare(); // the numbers first: Start comes back once they match the inputs
      return;
    }
    const strategy = chosen;
    const m = data.strategies?.[strategy]?.market;
    if (strategy === "classic" && m) {
      const odds = m.complete_one_in && m.chance_complete < 0.5 ? `about 1 in ${Number(m.complete_one_in).toLocaleString()}` : pct(m.chance_complete);
      const end = m.most_likely_end;
      const ok = await confirmSheet({
        title: "Start a classic rollover?",
        body: `By the market's chances it lasts all ${data.days} days ${odds}${end && !end.completes ? `, and most likely it ends on day ${end.day} (${pct(end.probability)})` : ""}. It runs on paper: nothing is bet.`,
        confirm: "Start on paper",
      });
      if (!ok || !root.isConnected || block || blocked() || !current()) return;
    }
    starting = true;
    draw();
    try {
      const body = { start: data.start, days: data.days, min_odds: data.min_odds, legs_per_day: data.legs_per_day, strategy, mode: "paper" };
      started = await api("/api/plans", { method: "POST", body, signal: params.signal });
      active = started;
      haptic.notify("success");
      toast("Paper plan started", { kind: "good" });
    } catch (error) {
      if (isQuiet(error) || !root.isConnected) return;
      haptic.notify("error");
      const hint = gateHint(error);
      toast(error.message, { kind: "error", ms: 5000 });
      if (hint?.kind === "age") openOnboarding();
      // A break or pause this page didn't know of (taken in the bot's chat, say): fetch it, and Start says so
      else if (hint?.kind === "break" || hint?.kind === "paused") loadMe().catch(() => null);
    } finally {
      starting = false;
      if (root.isConnected) draw();
    }
    reveal(out.querySelector("[data-plan-note]"));
  };

  root.addEventListener("click", (event) => {
    const mode = event.target.closest("[data-mode]");
    if (mode) {
      chooseMode(mode.dataset.mode);
      navigate("#/build");
      return;
    }
    if (event.target.closest("[data-r-go]")) {
      startPlan();
      return;
    }
    const start = event.target.closest("[data-r-start]");
    if (start) {
      root.querySelector("#r-start").value = start.dataset.rStart;
      for (const b of root.querySelectorAll("[data-r-start]")) {
        b.classList.toggle("on", b === start);
        b.setAttribute("aria-pressed", String(b === start));
      }
      later();
      return;
    }
    const odds = event.target.closest("[data-r-odds]");
    if (odds) {
      root.querySelector("#r-odds").value = odds.dataset.rOdds;
      for (const b of root.querySelectorAll("[data-r-odds]")) {
        b.classList.toggle("on", b === odds);
        b.setAttribute("aria-pressed", String(b === odds));
      }
      later();
      return;
    }
    const days = event.target.closest("[data-r-days]");
    if (days) {
      const input = root.querySelector("#r-days");
      const next = Math.min(60, Math.max(1, (cleanDays(input.value) || form.days) + Number(days.dataset.rDays)));
      input.value = String(next);
      for (const b of root.querySelectorAll("[data-r-days]")) b.disabled = (b.dataset.rDays === "-1" && next <= 1) || (b.dataset.rDays === "1" && next >= 60);
      later();
      return;
    }
    const legs = event.target.closest('[data-seg="r-legs"]');
    if (legs) {
      form = { ...form, legs_per_day: Number(legs.dataset.value) };
      for (const b of root.querySelectorAll('[data-seg="r-legs"]')) b.setAttribute("aria-pressed", String(b === legs));
      later();
      return;
    }
    const strategy = event.target.closest('[data-seg="r-strategy"]');
    if (strategy) {
      chosen = strategy.dataset.value;
      save();
      haptic.select();
      draw();
    }
  });
  root.addEventListener("input", (event) => {
    if (!event.target.closest("[data-r-form]")) return;
    const id = event.target.id;
    if (id === "r-start" || id === "r-odds") {
      const name = id === "r-start" ? "r-start" : "r-odds";
      const value = event.target.value.trim();
      for (const b of root.querySelectorAll(`[data-${name}]`)) {
        const on = b.getAttribute(`data-${name}`) === value;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", String(on));
      }
    }
    later();
  });
  root.addEventListener("submit", (event) => {
    event.preventDefault();
    clearTimeout(timer);
    compare();
  });

  const ready = (async () => {
    // One plan runs at a time: say so up front rather than after the person presses Start
    api("/api/plans?status=active", { signal: params.signal }).then((plans) => {
      active = Array.isArray(plans) && plans.length ? plans[0] : null;
      if (active && data && root.isConnected) draw();
    }).catch(() => null);
    await compare();
  })();

  // A break or the picks pause starting or ending (GET /api/me again) switches Start off or back on
  const offMe = onMeChange(() => {
    const next = picksBlock();
    if (next?.text === block?.text) return;
    block = next;
    if (data && root.isConnected) draw();
    else refresh();
  });

  return {
    ready,
    mainButton() {
      if (!data || active || block) return null;
      const wait = blocked();
      return { text: goLabel(chosen, { starting, blocked: wait }), onClick: startPlan, disabled: starting || Boolean(wait), progress: starting || wait === "waiting" };
    },
    stop() {
      clearTimeout(timer);
      controller?.abort();
      offMe();
    },
  };
}
