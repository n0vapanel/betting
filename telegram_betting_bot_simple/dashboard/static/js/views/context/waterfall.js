// The factor waterfall (v3 Stage 4): any Projection.to_dict() (engine/factors.py) drawn as a labelled baseline,
// floating bars that step from it to the number shown, and a sum line. Used for "Why the line is here" (the
// context attribution of the market's margin and total), the NBA model's "Why the numbers" (views/markets/
// factors.js) and the inspector (the same game page). "What moves it" (adjustments) is drawn here too, as its own
// block: the two are never mixed.
//
// Data: Projection = {key, label, baseline_label, baseline, value, unit, factors: [Factor], source?}, Factor =
// {key, label, delta, unit, direction, evidence, source, as_of, confidence}. The server rounds the deltas so
// baseline + deltas = value at the precision it sends (check_sum); this file only lays them out:
//   - the baseline row, then the factors sorted by size, the biggest SHOWN (6); the rest fold into "Other (n
//     factors) ±x" when two or more would fold (one leftover is shown by name), added at the same precision;
//   - the remainder an attribution ends with (key "other": "Team strength and other") stays last, unfolded: it is
//     what the listed factors leave of the market's number, not a factor of its own;
//   - each bar floats from the running total before it to the one after, on one scale per projection;
//   - the total row and a sum line ("0.0 + 1.2 + 0.5 − 4.2 = −2.5 pts") equal to the value shown;
//   - "small, noisy" on weak factors and "untested" where the factor's evidence says so.
//
//   normalizeProjection(raw, key?)   -> the checked projection, or null (no baseline, value or factors)
//   projectionsOf(prediction)        -> the NBA model's projections, checked, in a fixed order
//   waterfallRows(p)                 -> {digits, rows, other: {n, delta, noisy} | null, rest: factor | null}
//   waterfallHtml(p, {title, sub, label}) -> the list (ol.fx-list) with its bars and the sum line
//   adjustmentHtml(adj, {units, league}) / movesHtml(adjustments, {units, league, note}) -> "What moves it"
// Every text goes through esc().

import { esc, ago } from "../../ui.js";
import { icon } from "../../icons.js";
import { ensureMarketStyles } from "../markets/styles.js";
import { ensureContextStyles } from "./styles.js";

export const SHOWN = 6;
export const NOISY = "small, noisy";
export const UNTESTED = "untested";
const REMAINDER = "other"; // the attribution's remainder (engine/context.py): kept last, never folded
const ORDER = ["home_win", "margin", "total", "model_margin", "model_total"];

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const text = (v, n = 160) => (typeof v === "string" ? v.slice(0, n) : "");

/** A projection checked: numbers where numbers belong, text cut to size; null when it can't be drawn. */
export function normalizeProjection(p, key = "", units = "") {
  if (!p || typeof p !== "object" || num(p.baseline) === null || num(p.value) === null) return null;
  const factors = (Array.isArray(p.factors) ? p.factors : []).filter((f) => f && num(f.delta) !== null && typeof f.label === "string").map((f) => ({
    key: text(f.key, 40), label: text(f.label, 80), delta: f.delta, unit: text(f.unit, 12), evidence: text((units && f.evidence_units && typeof f.evidence_units[units] === "string" ? f.evidence_units[units] : f.evidence), 400),
    source: text(f.source, 200), asOf: typeof f.as_of === "string" ? f.as_of : null, noisy: f.confidence === NOISY,
    untested: typeof f.evidence === "string" && f.evidence.toLowerCase().includes(UNTESTED),
  }));
  if (!factors.length) return null;
  return {
    key: text(p.key, 40) || key, label: text(p.label, 100), baseLabel: text(p.baseline_label, 100) || "Starting point",
    baseline: p.baseline, value: p.value, unit: text(p.unit, 12), factors, source: text(p.source, 160),
  };
}

/** The NBA model's projections (GET /api/game prediction.projections), checked, in a fixed order. */
export function projectionsOf(prediction) {
  const raw = prediction?.projections;
  if (!raw || typeof raw !== "object") return [];
  const list = Object.entries(raw).map(([key, p]) => normalizeProjection(p, key)).filter(Boolean);
  const rank = (p) => { const i = ORDER.indexOf(p.key); return i < 0 ? ORDER.length : i; };
  return list.sort((a, b) => rank(a) - rank(b));
}

/** The decimals the server used (at most 2, at least 1), so sums are shown at the same precision. */
export function decimals(values) {
  let d = 0;
  for (const v of values) {
    const s = String(v);
    const i = s.indexOf(".");
    if (i >= 0 && !s.includes("e")) d = Math.max(d, Math.min(2, s.length - i - 1));
  }
  return Math.max(1, d);
}

const round = (v, digits) => Math.round(v * 10 ** digits) / 10 ** digits;

export function fmt(value, digits) {
  const v = round(value, digits);
  return (Object.is(v, -0) ? 0 : v).toFixed(digits).replace(/^-/, "−");
}

/** "+0.7" / "−0.4" / "0.0" (a real minus sign). */
export function signed(value, digits) {
  const t = fmt(Math.abs(value), digits);
  if (Number(t) === 0) return fmt(0, digits);
  return `${value > 0 ? "+" : "−"}${t}`;
}

/** The unit after a delta: percentage points for a chance ("pp"), else the stat's own unit ("pts"). */
const deltaUnit = (unit) => (unit === "%" ? "pp" : unit);
const UNIT_WORDS = { pp: "percentage points", pts: "points", runs: "runs", goals: "goals" };

export function amount(value, unit, digits) {
  return unit === "%" ? `${fmt(value, digits)}%` : `${fmt(value, digits)}${unit ? ` ${unit}` : ""}`;
}

/**
 * The rows of one projection: the biggest SHOWN factors (the remainder aside), then the rest folded into one
 * "Other" row, but only when at least two would fold; the remainder last.
 */
export function waterfallRows(p) {
  const digits = decimals([p.baseline, p.value, ...p.factors.map((f) => f.delta)]);
  const rest = p.factors.find((f) => f.key === REMAINDER) || null;
  const listed = p.factors.filter((f) => f !== rest).map((f, i) => ({ ...f, i }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.i - b.i);
  const fold = listed.length > SHOWN + 1;
  const rows = fold ? listed.slice(0, SHOWN) : listed;
  const folded = fold ? listed.slice(SHOWN) : [];
  const other = folded.length
    ? { n: folded.length, delta: round(folded.reduce((t, f) => t + f.delta, 0), digits), noisy: folded.every((f) => f.noisy), untested: folded.some((f) => f.untested) }
    : null;
  return { digits, rows, other, rest };
}

function tags(f) {
  return `${f.noisy ? ` <span class="fx-noisy">${NOISY}</span>` : ""}${f.untested ? ` <span class="fx-untested">${UNTESTED}</span>` : ""}`;
}

/**
 * The waterfall of one projection: ``title`` (a heading over it, optional), ``sub`` (words after the title, e.g.
 * "Broncos by 2.5"), ``label`` (the list's accessible name; default: the projection's label and value).
 */
export function waterfallHtml(p, { title = "", sub = "", label = "" } = {}) {
  ensureMarketStyles();
  ensureContextStyles();
  const { digits, rows, other, rest } = waterfallRows(p);
  const unit = deltaUnit(p.unit);
  const unitWords = UNIT_WORDS[unit] || unit;
  const steps = [...rows, ...(other ? [{ ...other, label: `Other (${other.n} factor${other.n === 1 ? "" : "s"})`, kind: "other" }] : []),
    ...(rest ? [{ ...rest, kind: "rest" }] : [])];
  // The running total each bar floats between, on one scale from the lowest to the highest point reached
  let run = p.baseline;
  const spans = steps.map((s) => {
    const from = run;
    run = round(run + s.delta, digits + 2);
    return [from, run];
  });
  const points = [p.baseline, p.value, ...spans.flat()];
  let lo = Math.min(...points);
  let hi = Math.max(...points);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.04;
  lo -= pad;
  hi += pad;
  const at = (v) => ((v - lo) / (hi - lo)) * 100;
  const bar = ([from, to], d) => {
    const left = Math.min(at(from), at(to));
    const width = Math.max(1.2, Math.abs(at(to) - at(from)));
    return `<span class="fx-bar wf-track" aria-hidden="true"><i class="${d > 0 ? "up" : d < 0 ? "down" : ""}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"></i></span>`;
  };
  const pin = (v) => `<span class="fx-bar wf-track" aria-hidden="true"><i class="pin" style="left:${at(v).toFixed(2)}%"></i></span>`;
  const delta = (d) => `<span class="fx-d num ${d > 0 ? "up" : d < 0 ? "down" : ""}"><span aria-hidden="true">${esc(signed(d, digits))}${unit ? ` ${esc(unit)}` : ""}</span><span class="sr-only">${esc(`${d > 0 ? "plus" : d < 0 ? "minus" : ""} ${fmt(Math.abs(d), digits)} ${unitWords}`)}</span></span>`;
  const items = steps.map((s, i) => {
    const meta = s.kind === "other" ? "" : [s.evidence, s.source && s.source !== s.evidence ? s.source : "", s.asOf ? `as of ${ago(s.asOf)}` : ""].filter(Boolean).join(" · ");
    const cls = `fx-row${s.kind ? ` ${s.kind}` : ""}${s.noisy ? " noisy" : ""}`;
    return `<li class="${cls}"><span class="fx-l"><span class="fx-name">${esc(s.label)}${tags(s)}</span>${meta ? `<span class="fx-meta">${esc(meta)}</span>` : ""}</span>${bar(spans[i], s.delta)}${delta(s.delta)}</li>`;
  }).join("");
  const name = p.label || p.key;
  const listLabel = label || `${name}: ${amount(p.value, p.unit, digits)}, from ${p.baseLabel}`;
  const sum = `${fmt(p.baseline, digits)} ${steps.map((s) => {
    const t = signed(s.delta, digits);
    return t.startsWith("+") || t.startsWith("−") ? `${t[0]} ${t.slice(1)}` : `+ ${t}`;
  }).join(" ")} = ${amount(p.value, p.unit, digits)}`;
  return `<div class="wf" data-fx="${esc(p.key)}">
    ${title ? `<h3 class="wf-h">${esc(title)}${sub ? ` <small>${esc(sub)}</small>` : ""}</h3>` : ""}
    <ol class="fx-list" aria-label="${esc(listLabel)}">
      <li class="fx-row base"><span class="fx-l"><span class="fx-name">${esc(p.baseLabel)}</span><span class="fx-meta">Starting point</span></span>${pin(p.baseline)}<span class="fx-d num">${esc(amount(p.baseline, p.unit, digits))}</span></li>
      ${items}
      <li class="fx-row total"><span class="fx-l"><span class="fx-name">${esc(name)}</span></span>${pin(p.value)}<b class="fx-d num"><span class="sr-only">equals </span>${esc(amount(p.value, p.unit, digits))}</b></li>
    </ol>
    <p class="wf-sum"><span class="sr-only">The sum: </span>${esc(sum)}</p>
  </div>`;
}

// --- "What moves it" ----------------------------------------------------------------------------------------

const KIND_WORDS = { total: "Total", side: "Margin", win: "Win chance" };

/** "−2.0 pts" for an adjustment's figure in its unit (2 decimals for goals and runs, 1 for points). */
function adjAmount(v, unit, { sign = true } = {}) {
  if (num(v) === null) return "–";
  const digits = unit === "pts" || unit === "pp" || unit === "%" ? 1 : 2;
  const t = sign ? signed(v, digits) : fmt(v, digits);
  return `${t}${unit ? ` ${unit === "%" ? "pp" : unit}` : ""}`;
}

const pctText = (p) => (num(p) === null ? "–" : `${Math.round(p * 100)}%`);

/**
 * One adjustment: what changed (the reason in the person's units when the server sent both), the raw change, the
 * weight, the cap and what was applied, and the chance before and after.
 */
export function adjustmentHtml(adj, { units = "metric" } = {}) {
  if (!adj || typeof adj !== "object" || num(adj.applied) === null) return "";
  const unit = text(adj.unit, 12);
  const reasons = adj.reasons && typeof adj.reasons === "object" ? adj.reasons : {};
  const reason = text(reasons[units] || adj.reason, 400);
  const kind = KIND_WORDS[adj.kind] || "Number";
  const full = adj.full === true;
  const weight = num(adj.weight);
  const cap = num(adj.cap);
  const math = [
    num(adj.effect_at_odds) !== null && num(adj.effect_now) !== null
      ? ["Effect", `${adjAmount(adj.effect_at_odds, unit)} → ${adjAmount(adj.effect_now, unit)}`, "the effect when the odds were fetched, and now"] : null,
    ["Raw change", adjAmount(adj.raw, unit), "Δ: the effect now minus at the odds' time"],
    ["Weight", full ? "1 (in full)" : weight === null ? "–" : String(weight), full ? "no market line to anchor to" : "how much of the change counts"],
    ["Cap", full || cap === null ? "none" : `±${adjAmount(cap, unit, { sign: false })}`, "the most it can move"],
    ["Applied", `${adjAmount(adj.applied, unit)}${adj.capped ? " (capped)" : ""}`, "clamp(weight × raw, −cap, +cap)"],
  ].filter(Boolean);
  const p = adj.p_over && typeof adj.p_over === "object" ? adj.p_over : null;
  const chance = p && num(p.before) !== null && num(p.after) !== null
    ? `<p class="mv-chance">Chance of Over ${esc(num(p.line) ?? "")}: <b>${esc(pctText(p.before))}</b> before → <b>${esc(pctText(p.after))}</b> after</p>` : "";
  return `<li class="mv-item">
    <div class="mv-head">${icon("sliders", { size: 16 })}<span>${esc(kind)}</span> <b class="num">${esc(adjAmount(adj.applied, unit))}</b>${adj.capped ? ' <span class="wf-tag">capped</span>' : ""}${full ? ' <span class="wf-tag">full change</span>' : ""}</div>
    ${reason ? `<p class="mv-reason">${esc(reason)}</p>` : ""}
    <dl class="mv-math">${math.map(([k, v, title]) => `<div title="${esc(title)}"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
    ${chance}
    ${full ? '<p class="mv-reason">There is no market line to anchor to, so the full change applies.</p>' : ""}
  </li>`;
}

/**
 * The "What moves it" block's body: each adjustment, or why nothing moves (``note``: e.g. the forecast at the odds'
 * time is unknown). Its own block, apart from "Why the line is here".
 */
export function movesHtml(adjustments, { units = "metric", note = "" } = {}) {
  ensureContextStyles();
  const list = (Array.isArray(adjustments) ? adjustments : []).map((a) => adjustmentHtml(a, { units })).filter(Boolean);
  if (!list.length) {
    return `<p class="mv-none">${esc(note || "Nothing has changed since the odds were fetched, so nothing here moves a number.")}</p>`;
  }
  return `<ul class="mv-list">${list.join("")}</ul>`;
}
