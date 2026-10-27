// The Builder's controls as HTML, and reading one changed input back. No state of its own: builder.js keeps
// the controls and draws these into the desktop column, the phone's top strip and the "More controls" sheet.
//
//   presetChipsHtml(s)            the Safe · Risky · Crazy chips (drawn once; presetPressed() sets which is on)
//   presetStatusHtml(s)           "Custom · from Safe" with "Back to Safe" once a control differs, and the line
//                                 under the chips (Crazy: "expect to lose most of these")
//   keyHtml(s, pfx)               the phone's key controls: sports, legs, target odds (± tolerance)
//   groupsHtml(s, pfx, {open})    every group (Scope, Shape, Markets, Correlation, Data rules, Output) as
//                                 collapsible <details>; ``open`` lists the groups shown open
//   readChange(target, s)         {controls?, ui?, error?, field} from the input that changed, or null
//
// ``s`` = {c: controls, ui, presets, described, markets, books, counts, total, label, changed, errors}.
// Inputs carry data-bd="<field>" and data-kind (how to read them). ``pfx`` keeps ids unique when the same
// control is drawn twice (the phone's strip and its sheet). Every outside text is escaped.

import { esc, explain } from "../../../ui.js";
import { icon } from "../../../icons.js";
import { leagueTitle } from "../../../app.js";
import {
  PRESET_KEYS, PRESET_TITLES, PRESET_ICONS, WINDOWS, AGES, MARKET_ORDER, MAX_LEGS, MAX_TICKETS, ageWords, numText,
  pointsText, parseNumber, bandText, cleanMoney, windowProblem,
} from "./model.js";

export const GROUPS = [
  { key: "scope", title: "Scope", iconName: "today" },
  { key: "shape", title: "Shape", iconName: "target" },
  { key: "markets", title: "Markets", iconName: "build" },
  { key: "correlation", title: "Correlation", iconName: "swap" },
  { key: "data", title: "Data rules", iconName: "shield" },
  { key: "output", title: "Output", iconName: "bets" },
];

// --- presets ---------------------------------------------------------------------------------------------

/** "60%+" and "1.15–1.80x" from a preset's own band (no target): what the chip promises, one short line each. */
function presetSub(p) {
  if (!p) return "";
  return [bandText(p.joint_min, p.joint_max, "chance"), bandText(p.odds_min, p.odds_max, "odds")].filter(Boolean)
    .map((line) => `<small>${esc(line)}</small>`).join("");
}

// engine/presets.py SUBTITLES, for a server that doesn't describe its presets
const SUBTITLES = {
  safe: "More likely, small payout. It still loses some of the time.",
  risky: "Balanced: a real chance and a real payout.",
  crazy: "Long shot, big payout: expect to lose most of these.",
};

/** The three preset chips (drawn once; presetPressed() keeps their state, so a click is never lost to a redraw). */
export function presetChipsHtml(s) {
  return `<div class="bd-presets" role="group" aria-label="Presets: each fills every control">${PRESET_KEYS.filter((k) => s.presets?.[k]).map((key) =>
    `<button type="button" class="bd-preset t-${key}" data-bd-preset="${key}" aria-pressed="false">
      ${icon(PRESET_ICONS[key], { size: 18 })}<span><b>${esc(PRESET_TITLES[key])}</b>${presetSub(s.presets[key])}</span></button>`).join("")}</div>`;
}

/** Which chip is pressed: the controls' preset while nothing that shapes the risk has changed, else none. */
export function presetPressed(container, s) {
  const custom = s.label === "Custom";
  for (const b of container.querySelectorAll("[data-bd-preset]")) b.setAttribute("aria-pressed", String(!custom && s.c.preset === b.dataset.bdPreset));
}

/** Under the chips: "Custom · from Safe · 2 changes" with "Back to Safe", and the preset's one line (Crazy's warning). */
export function presetStatusHtml(s) {
  const custom = s.label === "Custom";
  const base = s.c.preset;
  const n = s.changed?.length || 0;
  const from = base ? `from ${PRESET_TITLES[base] || base}${n ? ` · ${n} change${n === 1 ? "" : "s"}` : ""}` : "your own controls";
  const customChip = custom ? `<div class="bd-preset bd-custom" role="status">${icon("sliders", { size: 18 })}<span><b>Custom</b><small>${esc(from)}</small></span>
    ${base ? `<button type="button" class="text-link bd-reset" data-bd-reset>${icon("refresh", { size: 14 })}Back to ${esc(PRESET_TITLES[base])}</button>` : ""}</div>` : "";
  let sub = (base && (s.described?.[base]?.subtitle || SUBTITLES[base])) || "Safe, Risky and Crazy fill every control; change any and it's yours.";
  // Crazy always says it, whatever the server's words are
  if (base === "crazy" && !/expect to lose most of these/i.test(sub)) sub = `${sub} Expect to lose most of these.`;
  return `${customChip}<p class="bd-preset-note${base === "crazy" ? " crazy" : ""}">${base === "crazy" ? icon("alert", { size: 14 }) : ""}<span>${esc(sub)}</span></p>`;
}

// --- small pieces ------------------------------------------------------------------------------------------

const id = (pfx, name) => `${pfx}-${name}`;
const err = (s, field) => s.errors?.[field] || "";

function errorLine(s, pfx, field) {
  const text = err(s, field);
  return `<span class="bd-err" id="${id(pfx, `${field}-err`)}" role="alert">${esc(text)}</span>`;
}

function invalid(s, pfx, field) {
  return err(s, field) ? ` aria-invalid="true" aria-describedby="${id(pfx, `${field}-err`)}"` : "";
}

/**
 * A number box: data-kind says how to read it ("num", "pct" for %, "pts" for points, "int"). ``inputmode`` is
 * "decimal" unless the value can be below zero: an iPhone's decimal keypad has no minus key.
 */
function numberBox(s, pfx, field, value, { label, srLabel = "", suffix = "", placeholder = "", kind = "num", min = "", max = "", info = "", hint = "", inputmode = "decimal" } = {}) {
  return `<div class="bd-num">
    <label class="bd-mini" for="${id(pfx, field)}">${srLabel ? `<span aria-hidden="true">${esc(label)}</span><span class="sr-only">${esc(srLabel)}</span>` : esc(label)}${info ? ` ${explain(info)}` : ""}</label>
    <span class="bd-box${suffix ? " has-suffix" : ""}"><input id="${id(pfx, field)}" class="input" type="text" inputmode="${inputmode}" autocomplete="off"
      data-bd="${field}" data-kind="${kind}" data-min="${min}" data-max="${max}" value="${esc(value)}" placeholder="${esc(placeholder)}"${invalid(s, pfx, field)}>${suffix ? `<span aria-hidden="true">${esc(suffix)}</span>` : ""}</span>
    ${hint ? `<span class="bd-hint">${esc(hint)}</span>` : ""}${errorLine(s, pfx, field)}
  </div>`;
}

function moneyBox(s, pfx, field, value, { label, placeholder = "" } = {}) {
  return `<div class="bd-num">
    <label class="bd-mini" for="${id(pfx, field)}">${esc(label)}</label>
    <span class="money-input"><span aria-hidden="true">$</span><input id="${id(pfx, field)}" class="input" type="text" inputmode="decimal" autocomplete="off"
      data-bd="${field}" data-kind="money" value="${esc(value || "")}" placeholder="${esc(placeholder)}"${invalid(s, pfx, field)}></span>
    ${errorLine(s, pfx, field)}
  </div>`;
}

function check(pfx, field, on, label, { hint = "", info = "", disabled = false } = {}) {
  return `<label class="bd-check${disabled ? " off" : ""}" for="${id(pfx, field)}">
    <input type="checkbox" id="${id(pfx, field)}" data-bd="${field}" data-kind="bool"${on ? " checked" : ""}${disabled ? " disabled" : ""}${hint ? ` aria-describedby="${id(pfx, `${field}-hint`)}"` : ""}>
    <span><b>${esc(label)}</b>${hint ? `<small id="${id(pfx, `${field}-hint`)}">${esc(hint)}</small>` : ""}</span>${info ? explain(info) : ""}
  </label>`;
}

function select(pfx, field, value, options, { label, kind = "int", info = "", hint = "", disabled = false, describedby = "" } = {}) {
  return `<div class="bd-num">
    <label class="bd-mini" for="${id(pfx, field)}">${esc(label)}${info ? ` ${explain(info)}` : ""}</label>
    <select id="${id(pfx, field)}" class="input" data-bd="${field}" data-kind="${kind}"${disabled ? " disabled" : ""}${describedby ? ` aria-describedby="${describedby}"` : ""}>${options.map((o) =>
      `<option value="${esc(o.value)}"${String(o.value) === String(value ?? "") ? " selected" : ""}>${esc(o.label)}</option>`).join("")}</select>
    ${hint ? `<span class="bd-hint">${esc(hint)}</span>` : ""}
  </div>`;
}

const range = (lo, hi) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

function legsOptions(c) {
  return [{ value: "", label: `Auto (1 to ${c.legs_max || 4})` }, ...range(1, MAX_LEGS).map((n) => ({ value: n, label: `${n} leg${n === 1 ? "" : "s"}` }))];
}

/** "All sports · 41 games" / "NBA, EPL · 12 games" for the sports control. */
export function sportsSummary(s) {
  const keys = s.c.leagues;
  const counts = s.counts || new Map();
  const total = keys ? keys.reduce((n, k) => n + (counts.get(k)?.count || 0), 0) : [...counts.values()].reduce((n, c) => n + c.count, 0);
  const games = `${total} game${total === 1 ? "" : "s"}`;
  if (!keys) return { text: "All sports", games };
  const names = keys.map(leagueTitle);
  const text = names.length <= 2 ? names.join(", ") : `${names.slice(0, 2).join(", ")} +${names.length - 2}`;
  return { text: text || "No sports", games };
}

function sportsButton(s, pfx) {
  const sum = sportsSummary(s);
  return `<div class="bd-num bd-sports">
    <span class="bd-mini" id="${id(pfx, "sports-label")}">Sports</span>
    <button type="button" class="input bd-pick" data-bd-sports aria-describedby="${id(pfx, "sports-label")}" aria-haspopup="dialog">
      ${icon("today", { size: 16 })}<span class="grow">${esc(sum.text)}</span><small>${esc(sum.games)}</small>${icon("down", { size: 16 })}</button>
    ${errorLine(s, pfx, "leagues")}
  </div>`;
}

function targetBoxes(s, pfx) {
  const c = s.c;
  const band = bandText(c.odds_min, c.odds_max, "odds");
  return `<div class="bd-pair bd-target">
    ${numberBox(s, pfx, "target_odds", numText(c.target_odds), { label: "Target odds", placeholder: "e.g. 10", suffix: "x", info: "target_odds" })}
    ${numberBox(s, pfx, "tolerance_pct", numText(c.tolerance_pct, 1), { label: "±", srLabel: "Tolerance, plus or minus", suffix: "%", placeholder: "10", info: "tolerance" })}
  </div>
  <p class="bd-hint" data-bd-target-hint>${esc(c.target_odds ? `Tickets paying ${targetRange(c)}.` : band ? `No target: the preset's band, ${band}.` : "No target: any odds.")}</p>`;
}

/** "9.00x to 11.00x" for a target ± tolerance (display of the person's own numbers). */
function targetRange(c) {
  const t = Number(c.target_odds);
  const tol = Number(c.tolerance_pct) / 100;
  const f = (v) => (v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2));
  return `${f(t * (1 - tol))}x to ${f(t * (1 + tol))}x`;
}

// --- the phone's key controls ---------------------------------------------------------------------------------

export function keyHtml(s, pfx = "bdk") {
  return `<div class="bd-key">
    ${sportsButton(s, pfx)}
    <div class="bd-pair">
      ${select(pfx, "legs", s.c.legs ?? "", legsOptions(s.c), { label: "Legs", kind: "legs" })}
    </div>
    ${targetBoxes(s, pfx)}
  </div>`;
}

// --- the groups ------------------------------------------------------------------------------------------------

function summaryFor(key, s) {
  const c = s.c;
  if (key === "scope") {
    const sum = sportsSummary(s);
    const win = WINDOWS.find((w) => w.value === s.ui.win)?.label || "Today";
    return `${sum.text} · ${win} · ${sum.games}`;
  }
  if (key === "shape") {
    const legs = c.legs ? `${c.legs} legs` : "Auto legs";
    const target = c.target_odds ? `target ${numText(c.target_odds)}x ±${numText(c.tolerance_pct, 1)}%` : "no target";
    return `${legs} · ${target}`;
  }
  if (key === "markets") {
    const on = (c.markets || []).map((m) => s.markets?.[m]?.title || m);
    return on.length <= 3 ? on.join(", ") : `${on.length} markets`;
  }
  if (key === "correlation") return c.prefer_correlation ? "One leg per game · same-game ideas on" : "One leg per game";
  if (key === "data") return `Prices up to ${ageWords(c.max_price_age_s)} old · ${c.books ? `${c.books.length} book${c.books.length === 1 ? "" : "s"}` : "all books"}`;
  if (key === "output") return `${c.tickets} ticket${c.tickets === 1 ? "" : "s"}`;
  return "";
}

function scopeBody(s, pfx, { tree = true } = {}) {
  const c = s.c;
  const ui = s.ui;
  const problem = windowProblem(ui);
  const local = (iso) => {
    const t = Date.parse(iso || "");
    if (!Number.isFinite(t)) return "";
    const d = new Date(t);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
  };
  return `<div class="field"><span class="label" id="${id(pfx, "tree-label")}">Sports and leagues</span>
      ${tree ? `<div class="bd-tree" data-bd-tree role="group" aria-labelledby="${id(pfx, "tree-label")}"></div>` : sportsButton(s, pfx)}
      ${tree ? errorLine(s, pfx, "leagues") : ""}</div>
    <div class="field"><span class="label" id="${id(pfx, "win-label")}">Time window</span>
      <div class="seg small bd-wins" role="group" aria-labelledby="${id(pfx, "win-label")}">${WINDOWS.map((w) =>
        `<button type="button" data-bd-win="${w.value}" aria-pressed="${ui.win === w.value}">${esc(w.label)}</button>`).join("")}</div>
      ${ui.win === "custom" ? `<div class="bd-pair" style="margin-top:10px">
        <div class="bd-num"><label class="bd-mini" for="${id(pfx, "from")}">From</label><input id="${id(pfx, "from")}" class="input" type="datetime-local" data-bd="from" data-kind="when" value="${esc(local(ui.from))}"${problem ? ` aria-invalid="true" aria-describedby="${id(pfx, "win-err")}"` : ""}></div>
        <div class="bd-num"><label class="bd-mini" for="${id(pfx, "to")}">To</label><input id="${id(pfx, "to")}" class="input" type="datetime-local" data-bd="to" data-kind="when" value="${esc(local(ui.to))}"${problem ? ` aria-invalid="true" aria-describedby="${id(pfx, "win-err")}"` : ""}></div>
      </div>` : ""}
      <span class="bd-err" id="${id(pfx, "win-err")}" role="alert">${esc(problem)}</span>
    </div>
    <div class="field"><span class="label" id="${id(pfx, "ko-label")}">Kick-off between <span class="muted small">(your time; empty: any)</span></span>
      <div class="bd-pair" role="group" aria-labelledby="${id(pfx, "ko-label")}">
        <div class="bd-num"><label class="bd-mini" for="${id(pfx, "earliest")}">Earliest</label><input id="${id(pfx, "earliest")}" class="input" type="time" data-bd="earliest" data-kind="clock" value="${esc(c.kickoff?.earliest || "")}"></div>
        <div class="bd-num"><label class="bd-mini" for="${id(pfx, "latest")}">Latest</label><input id="${id(pfx, "latest")}" class="input" type="time" data-bd="latest" data-kind="clock" value="${esc(c.kickoff?.latest || "")}"></div>
      </div></div>
    <div class="field">${check(pfx, "exclude_started", true, "Leave out games already started", { hint: "Always on: their prices are from before the start, so they never go into a ticket.", disabled: true })}</div>`;
}

function shapeBody(s, pfx) {
  const c = s.c;
  const ui = s.ui;
  const conv = ui.stake && ui.back ? `$${ui.stake} to win about $${ui.back}: target ${numText(c.target_odds)}x` : ui.stake ? "Add “win about” to set the target from your stake." : "";
  return `<div class="field"><div class="bd-pair">${select(pfx, "legs", c.legs ?? "", legsOptions(c), { label: "Legs", kind: "legs" })}</div>
      <p class="bd-hint">Auto picks how many; a number you set lifts the Whelan cap ${explain("multi", { label: "Why do more legs cost more?" })}</p></div>
    <div class="field">${targetBoxes(s, pfx)}</div>
    <div class="field"><span class="label">Or from your stake</span>
      <div class="bd-pair">${moneyBox(s, pfx, "stake", ui.stake, { label: "Stake", placeholder: "10" })}${moneyBox(s, pfx, "back", ui.back, { label: "Win about (total back)", placeholder: "50" })}</div>
      <p class="bd-hint bd-conv" data-bd-conv aria-live="polite">${esc(conv)}</p></div>
    <div class="field"><span class="label">Each leg</span>
      <div class="bd-pair">
        ${numberBox(s, pfx, "leg_odds_min", numText(c.leg_odds_min), { label: "Odds from", placeholder: "any", min: "1.01" })}
        ${numberBox(s, pfx, "leg_odds_max", numText(c.leg_odds_max), { label: "Odds up to", placeholder: "any", min: "1.01" })}
      </div>
      <div class="bd-pair" style="margin-top:10px">
        ${numberBox(s, pfx, "min_chance", pointsText(c.min_chance), { label: "Chance at least", suffix: "%", kind: "pct", placeholder: "any", info: "market_chance" })}
        ${numberBox(s, pfx, "min_edge", pointsText(c.min_edge), { label: "Edge at least", suffix: "pts", kind: "pts", placeholder: "any", info: "edge", inputmode: "text" })}
      </div></div>
    <div class="field"><span class="label">Most legs per</span>
      <div class="bd-trio">
        ${select(pfx, "max_per_game", 1, [{ value: 1, label: "1" }], { label: "Game", disabled: true, describedby: id(pfx, "per-game") })}
        ${select(pfx, "max_per_team", c.max_per_team, range(1, 4).map((n) => ({ value: n, label: String(n) })), { label: "Team" })}
        ${select(pfx, "max_per_league", c.max_per_league ?? "", [{ value: "", label: "Any" }, ...range(1, MAX_LEGS).map((n) => ({ value: n, label: String(n) }))], { label: "League", kind: "intnull" })}
      </div>
      <p class="bd-hint" id="${id(pfx, "per-game")}">Always one leg per game: same-game legs are linked, and no book's same-game price is loaded ${explain("same_game")}</p></div>
    <details class="bd-adv"><summary>${icon("sliders", { size: 16 })}The preset's own rules</summary>
      <div class="bd-pair">
        ${select(pfx, "legs_max", c.legs_max, range(1, MAX_LEGS).map((n) => ({ value: n, label: String(n) })), { label: "Auto: up to … legs" })}
        ${numberBox(s, pfx, "caution_pp", numText(c.caution_pp, 1), { label: "Caution", suffix: "pts", placeholder: "2", hint: "Points off each chance when ranking" })}
      </div>
      <div class="bd-pair" style="margin-top:10px">
        ${numberBox(s, pfx, "odds_min", numText(c.odds_min), { label: "Ticket odds from", placeholder: "any", suffix: "x" })}
        ${numberBox(s, pfx, "odds_max", numText(c.odds_max), { label: "up to", placeholder: "any", suffix: "x" })}
      </div>
      <div class="bd-pair" style="margin-top:10px">
        ${numberBox(s, pfx, "joint_min", pointsText(c.joint_min), { label: "Whole-ticket chance from", suffix: "%", kind: "pct", placeholder: "any" })}
        ${numberBox(s, pfx, "joint_max", pointsText(c.joint_max), { label: "under", suffix: "%", kind: "pct", placeholder: "any" })}
      </div>
      <p class="bd-hint">The band applies when there's no target; a target replaces it.</p>
      <div class="bd-pair" style="margin-top:10px">
        ${numberBox(s, pfx, "multi_leg_min", pointsText(c.multi_leg_min), { label: "Each leg of a multi at least", suffix: "%", kind: "pct", placeholder: "any" })}
      </div>
      ${check(pfx, "whelan", c.whelan, "Whelan leg cap", { hint: "Auto stops adding legs once more aren't worth combining at these chances." })}
    </details>`;
}

function marketsBody(s, pfx) {
  const c = s.c;
  const catalog = s.markets || {};
  const keys = MARKET_ORDER.filter((m) => catalog[m]);
  const soccer = keys.filter((m) => catalog[m].available && catalog[m].reason);
  const items = keys.map((m) => {
    const entry = catalog[m];
    const on = (c.markets || []).includes(m);
    const off = !entry.available;
    const tag = off ? "" : entry.reason ? '<small>soccer · fair odds</small>' : "";
    return `<label class="bd-mkt${off ? " off" : ""}"${off ? ` title="${esc(entry.reason || "")}"` : ""}>
      <input type="checkbox" data-bd="markets" data-kind="market" value="${esc(m)}"${on && !off ? " checked" : ""}${off ? ` disabled aria-describedby="${id(pfx, "mk-off")}"` : ""}>
      <span>${esc(entry.title || m)}${tag}</span></label>`;
  }).join("");
  const offReasons = [...new Set(keys.filter((m) => !catalog[m].available).map((m) => catalog[m].reason).filter(Boolean))];
  return `<div class="bd-mkts" role="group" aria-label="Markets">${items}</div>
    ${errorLine(s, pfx, "markets")}
    ${soccer.length ? `<p class="bd-hint">${esc(catalog[soccer[0]].reason)} ${explain("dixon_coles")}</p>` : ""}
    ${offReasons.length ? `<p class="bd-hint" id="${id(pfx, "mk-off")}">${icon("clock", { size: 14 })} ${esc(offReasons.join(" "))}</p>` : ""}`;
}

function correlationBody(s, pfx) {
  const c = s.c;
  return `${check(pfx, "avoid_same_game", true, "Avoid the same game", { hint: "Always on: legs from one game rise and fall together, and no book's same-game price is loaded to price them.", info: "same_game", disabled: true })}
    ${check(pfx, "prefer_correlation", c.prefer_correlation, "Prefer positive correlation (ideas only)", { hint: "Suggests same-game pairs that tend to land together. It never builds a same-game ticket: that needs a book's own same-game price, which isn't loaded." })}`;
}

function dataBody(s, pfx) {
  const c = s.c;
  const books = s.books || [];
  const ages = [...new Set([...AGES, c.max_price_age_s])].sort((a, b) => a - b);
  const all = !c.books;
  return `<div class="field"><div class="bd-pair">${select(pfx, "max_price_age_s", c.max_price_age_s, ages.map((a) => ({ value: a, label: ageWords(a) })), { label: "Oldest price allowed", info: "data_age" })}</div></div>
    <div class="field">
      ${check(pfx, "exclude_disagree", c.exclude_disagree, "Leave out games whose sources disagree", { info: "disagreement" })}
      ${check(pfx, "exclude_thin", c.exclude_thin, "Leave out legs priced only by thin markets", { info: "thin_market" })}
      ${check(pfx, "exclude_low_conf", c.exclude_low_conf, "Leave out low-confidence legs", { hint: "A chance from ratings only, with no market price behind it." })}
      ${check(pfx, "exclude_model_price", c.exclude_model_price, "Leave out legs priced at fair odds", { hint: "The goals model's soccer markets, which no bookmaker here prices.", info: "fair_odds" })}
    </div>
    ${books.length ? `<div class="field"><span class="label" id="${id(pfx, "books-label")}">Bookmakers <span class="muted small">(prices from these only)</span></span>
      <div class="bd-books" role="group" aria-labelledby="${id(pfx, "books-label")}">
        <label class="bd-mkt"><input type="checkbox" data-bd="books" data-kind="allbooks"${all ? " checked" : ""}><span>All</span></label>
        ${books.map((b) => `<label class="bd-mkt"><input type="checkbox" data-bd="books" data-kind="book" value="${esc(b.key)}"${all || c.books.includes(b.key) ? " checked" : ""}><span>${esc(b.title)}</span></label>`).join("")}
      </div>
      <p class="bd-hint">Never Pinnacle or an exchange: they only feed the blend ${explain("sharp_book")}</p>${errorLine(s, pfx, "books")}</div>` : ""}`;
}

function outputBody(s, pfx) {
  return `<div class="bd-pair">${select(pfx, "tickets", s.c.tickets, range(1, MAX_TICKETS).map((n) => ({ value: n, label: `${n} ticket${n === 1 ? "" : "s"}` })), { label: "Tickets" })}</div>
    <p class="bd-hint">Tickets share no leg, so each is a different idea.</p>`;
}

const BODIES = { scope: scopeBody, shape: shapeBody, markets: marketsBody, correlation: correlationBody, data: dataBody, output: outputBody };

export function groupsHtml(s, pfx = "bdd", { open = ["scope", "shape"], tree = true } = {}) {
  return GROUPS.map((g) => `<details class="bd-group" data-bd-group="${g.key}"${open.includes(g.key) ? " open" : ""}>
      <summary><span class="bd-gicon" aria-hidden="true">${icon(g.iconName, { size: 16 })}</span><span class="bd-gt"><b>${esc(g.title)}</b><small data-bd-sum="${g.key}">${esc(summaryFor(g.key, s))}</small></span>${icon("down", { size: 18 })}</summary>
      <div class="bd-gbody">${g.key === "scope" ? scopeBody(s, pfx, { tree }) : BODIES[g.key](s, pfx)}</div>
    </details>`).join("");
}

/** The groups' one-line summaries, updated in place (without redrawing inputs being typed in). */
export function syncSummaries(container, s) {
  for (const el of container.querySelectorAll("[data-bd-sum]")) el.textContent = summaryFor(el.dataset.bdSum, s);
  const conv = container.querySelector("[data-bd-conv]");
  if (conv) {
    const ui = s.ui;
    conv.textContent = ui.stake && ui.back && s.c.target_odds ? `$${ui.stake} to win about $${ui.back}: target ${numText(s.c.target_odds)}x` : ui.stake && !ui.back ? "Add “win about” to set the target from your stake." : "";
  }
  for (const hint of container.querySelectorAll("[data-bd-target-hint]")) {
    const c = s.c;
    const band = bandText(c.odds_min, c.odds_max, "odds");
    hint.textContent = c.target_odds ? `Tickets paying ${targetRange(c)}.` : band ? `No target: the preset's band, ${band}.` : "No target: any odds.";
  }
  // The custom window's problem (the inputs being typed in aren't redrawn)
  const problem = windowProblem(s.ui);
  for (const el of container.querySelectorAll('[id$="-win-err"]')) el.textContent = problem;
  for (const input of container.querySelectorAll('[data-kind="when"]')) {
    if (problem) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", input.id.replace(/-(from|to)$/, "-win-err"));
    } else {
      input.removeAttribute("aria-invalid");
    }
  }
  for (const b of container.querySelectorAll("[data-bd-sports]")) {
    const sum = sportsSummary(s);
    b.querySelector(".grow").textContent = sum.text;
    b.querySelector("small").textContent = sum.games;
  }
}

// --- reading a change ------------------------------------------------------------------------------------------

const LIMITS = {
  target_odds: { lo: 1.01, hi: 1e6, words: "Target odds are 1.01 or more (e.g. 10 for 10x)." },
  tolerance_pct: { lo: 0, hi: 90, optional: false, words: "The tolerance is 0 to 90%." },
  leg_odds_min: { lo: 1.01, hi: 1000, words: "Leg odds are 1.01 to 1,000." },
  leg_odds_max: { lo: 1.01, hi: 1000, words: "Leg odds are 1.01 to 1,000." },
  min_chance: { lo: 0, hi: 100, scale: 100, words: "A chance is 0 to 100%." },
  min_edge: { lo: -100, hi: 100, scale: 100, words: "The edge is −100 to +100 points." },
  caution_pp: { lo: 0, hi: 20, optional: false, words: "Caution is 0 to 20 points." },
  odds_min: { lo: 1, hi: 1e6, words: "Ticket odds are 1 or more." },
  odds_max: { lo: 1.01, hi: 1e6, words: "Ticket odds are 1.01 or more." },
  joint_min: { lo: 0, hi: 100, scale: 100, words: "A chance is 0 to 100%." },
  joint_max: { lo: 0, hi: 100, scale: 100, words: "A chance is 0 to 100%." },
  multi_leg_min: { lo: 0, hi: 100, scale: 100, words: "A chance is 0 to 100%." },
};

/**
 * What one changed input means: {field, controls: {...changes}, ui: {...changes}, error} (error "" when fine),
 * or null for anything that isn't a control. ``s`` is the current state (for pairs: min/max, stake/back).
 */
export function readChange(target, s) {
  const field = target?.dataset?.bd;
  if (!field) return null;
  const kind = target.dataset.kind;
  const c = s.c;
  if (kind === "bool") return { field, controls: { [field]: target.checked }, error: "" };
  if (kind === "legs") return { field, controls: { legs: target.value === "" ? null : Number(target.value) }, error: "" };
  if (kind === "int") return { field, controls: { [field]: Number(target.value) }, error: "" };
  if (kind === "intnull") return { field, controls: { [field]: target.value === "" ? null : Number(target.value) }, error: "" };
  if (kind === "market") {
    const set = new Set(c.markets || []);
    if (target.checked) set.add(target.value);
    else set.delete(target.value);
    if (!set.size) return { field: "markets", controls: {}, error: "Pick at least one market." };
    return { field: "markets", controls: { markets: [...set] }, error: "" };
  }
  if (kind === "allbooks") return { field: "books", controls: { books: null }, error: "" };
  if (kind === "book") {
    const all = (s.books || []).map((b) => b.key);
    const set = new Set(c.books || all);
    if (target.checked) set.add(target.value);
    else set.delete(target.value);
    if (!set.size) return { field: "books", controls: {}, error: "Pick at least one bookmaker." };
    return { field: "books", controls: { books: set.size === all.length ? null : all.filter((k) => set.has(k)) }, error: "" };
  }
  if (kind === "clock") {
    const value = /^([01]\d|2[0-3]):[0-5]\d$/.test(target.value) ? target.value : null;
    const kickoff = { earliest: c.kickoff?.earliest || null, latest: c.kickoff?.latest || null, [field]: value };
    return { field, controls: { kickoff: kickoff.earliest || kickoff.latest ? kickoff : null }, error: "" };
  }
  if (kind === "when") {
    const t = Date.parse(target.value || "");
    return { field, ui: { [field]: Number.isFinite(t) ? new Date(t).toISOString() : null }, error: "" };
  }
  if (kind === "money") {
    const text = target.value.trim();
    if (!text) return { field, ui: { [field]: null }, error: "" };
    const clean = cleanMoney(text);
    if (!clean) return { field, ui: {}, error: field === "stake" ? "A stake is $0.01 to $100,000." : "An amount is $0.01 to $100,000." };
    return { field, ui: { [field]: clean }, error: "" };
  }
  const limit = LIMITS[field];
  if (!limit) return null;
  const parsed = parseNumber(target.value, { lo: limit.lo, hi: limit.hi, optional: limit.optional !== false, scale: limit.scale || 1 });
  if (!parsed.ok) return { field, controls: {}, error: limit.words };
  let value = parsed.value;
  if (value === null && limit.optional === false) value = field === "tolerance_pct" ? 10 : 2;
  const changes = { [field]: value };
  // Pairs must stay in order
  const pairs = { leg_odds_min: ["leg_odds_max", 1], leg_odds_max: ["leg_odds_min", -1], odds_min: ["odds_max", 1], odds_max: ["odds_min", -1],
    joint_min: ["joint_max", 1], joint_max: ["joint_min", -1] };
  if (pairs[field] && value != null) {
    const [other, dir] = pairs[field];
    const o = c[other];
    if (o != null && (dir > 0 ? value > o : value < o)) return { field, controls: {}, error: dir > 0 ? "The lower end is above the upper end." : "The upper end is below the lower end." };
  }
  return { field, controls: changes, error: "" };
}
