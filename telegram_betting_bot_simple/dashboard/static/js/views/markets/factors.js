// The model's explanation as typed factors (engine/factors.py; Contract B): each number the NBA model shows is a
// labelled baseline plus sized factors that add up to it. On the game page: a baseline row, the factors sorted
// by size with signed deltas and units (the top 6; the rest fold into "Other (n factors) ±x" when there are two
// or more of them: one left over is shown by name, as folding it saves no room and hides it), "small, noisy"
// on weak ones, and the number they add up to. With no typed factors from the server the old text lines show,
// as before.
//
// Data: GET /api/game's ``prediction.projections`` = {key: Projection}, Projection = {key, label, baseline_label,
// baseline, value, unit, factors: [{key, label, delta, unit, direction, evidence, source, as_of, confidence}]}
// (keys "home_win" (unit "%"), "margin", "total", and the team model's own "model_margin", "model_total"). The
// server rounds the deltas so they still add up to the value shown; this file only lays them out, and the
// "Other" row adds up the folded deltas at the same precision.
//
//   projectionsOf(prediction) -> [Projection] (checked, in a fixed order; only those with factors)
//   factorsSection(game, {key}) -> the section's HTML: the typed list, else the old lines, else ""
//   factorRows(projection)      -> {base, rows: [{label, delta, ...}], other: {n, delta} | null, value}

import { esc, ago, section } from "../../ui.js";
import { ensureMarketStyles } from "./styles.js";

export const SHOWN = 6;
const ORDER = ["home_win", "margin", "total", "model_margin", "model_total"];
const NAMES = { home_win: "Win chance", margin: "Margin", total: "Total", model_margin: "Team model margin", model_total: "Team model total" };
const NOISY = "small, noisy";

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const text = (v, n = 160) => (typeof v === "string" ? v.slice(0, n) : "");

/** The decimals the server used (at most 2), so sums are shown at the same precision. */
function decimals(values) {
  let d = 0;
  for (const v of values) {
    const s = String(v);
    const i = s.indexOf(".");
    if (i >= 0) d = Math.max(d, Math.min(2, s.length - i - 1));
  }
  return Math.max(1, d);
}

/** Checked projections, in a fixed order (unknown keys after), only those with at least one factor. */
export function projectionsOf(prediction) {
  const raw = prediction?.projections;
  if (!raw || typeof raw !== "object") return [];
  const list = [];
  for (const [key, p] of Object.entries(raw)) {
    if (!p || typeof p !== "object" || num(p.baseline) === null || num(p.value) === null) continue;
    const factors = (Array.isArray(p.factors) ? p.factors : []).filter((f) => f && num(f.delta) !== null && typeof f.label === "string").map((f) => ({
      key: text(f.key, 40), label: text(f.label, 80), delta: f.delta, unit: text(f.unit, 12), evidence: text(f.evidence),
      source: text(f.source, 80), asOf: typeof f.as_of === "string" ? f.as_of : null, noisy: f.confidence === NOISY,
    }));
    if (!factors.length) continue;
    list.push({
      key: text(p.key, 40) || key, label: text(p.label, 100), baseLabel: text(p.baseline_label, 100) || "Starting point",
      baseline: p.baseline, value: p.value, unit: text(p.unit, 12), factors,
    });
  }
  const rank = (p) => { const i = ORDER.indexOf(p.key); return i < 0 ? ORDER.length : i; };
  return list.sort((a, b) => rank(a) - rank(b));
}

/** The unit after a delta: percentage points for a chance ("pp"), else the stat's own unit ("pts"). */
const deltaUnit = (unit) => (unit === "%" ? "pp" : unit);
const UNIT_WORDS = { pp: "percentage points", pts: "points" };

function fmt(value, digits) {
  const v = Math.round(value * 10 ** digits) / 10 ** digits;
  return (Object.is(v, -0) ? 0 : v).toFixed(digits);
}

/** "+0.7" / "−0.4" / "0.0" (a real minus sign). */
function signed(value, digits) {
  const t = fmt(Math.abs(value), digits);
  if (Number(t) === 0) return fmt(0, digits);
  return `${value > 0 ? "+" : "−"}${t}`;
}

function amount(value, unit, digits) {
  return unit === "%" ? `${fmt(value, digits)}%` : `${fmt(value, digits)}${unit ? ` ${unit}` : ""}`;
}

/**
 * The rows of one projection: the biggest SHOWN factors, then the rest folded into one "Other" row, but only
 * when at least two would fold (a lone leftover is shown by name: an "Other (1 factor)" row saves no room and
 * would hide, say, one half of a pair such as both teams' back-to-backs).
 */
export function factorRows(p) {
  const digits = decimals([p.baseline, p.value, ...p.factors.map((f) => f.delta)]);
  const sorted = p.factors.map((f, i) => ({ ...f, i })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.i - b.i);
  const fold = sorted.length > SHOWN + 1;
  const rows = fold ? sorted.slice(0, SHOWN) : sorted;
  const rest = fold ? sorted.slice(SHOWN) : [];
  const other = rest.length ? { n: rest.length, delta: Math.round(rest.reduce((t, f) => t + f.delta, 0) * 10 ** digits) / 10 ** digits, noisy: rest.every((f) => f.noisy) } : null;
  return { digits, rows, other };
}

function projectionHtml(p) {
  const { digits, rows, other } = factorRows(p);
  const unit = deltaUnit(p.unit);
  const unitWords = UNIT_WORDS[unit] || unit;
  const biggest = Math.max(...rows.map((f) => Math.abs(f.delta)), other ? Math.abs(other.delta) : 0) || 1;
  const bar = (d) => `<span class="fx-bar" aria-hidden="true"><i class="${d > 0 ? "up" : d < 0 ? "down" : ""}" style="width:${Math.max(2, Math.round((Math.abs(d) / biggest) * 100))}%"></i></span>`;
  const delta = (d) => `<span class="fx-d num ${d > 0 ? "up" : d < 0 ? "down" : ""}"><span aria-hidden="true">${esc(signed(d, digits))}${unit ? ` ${esc(unit)}` : ""}</span><span class="sr-only">${esc(`${d > 0 ? "plus" : d < 0 ? "minus" : ""} ${fmt(Math.abs(d), digits)} ${unitWords}`)}</span></span>`;
  const items = rows.map((f) => {
    const meta = [f.evidence, f.source, f.asOf ? `as of ${ago(f.asOf)}` : ""].filter(Boolean).join(" · ");
    return `<li class="fx-row${f.noisy ? " noisy" : ""}"><span class="fx-l"><span class="fx-name">${esc(f.label)}${f.noisy ? ' <span class="fx-noisy">small, noisy</span>' : ""}</span>${meta ? `<span class="fx-meta">${esc(meta)}</span>` : ""}</span>${bar(f.delta)}${delta(f.delta)}</li>`;
  }).join("");
  const otherRow = other ? `<li class="fx-row other"><span class="fx-l"><span class="fx-name">Other (${esc(other.n)} factor${other.n === 1 ? "" : "s"})${other.noisy ? ' <span class="fx-noisy">small, noisy</span>' : ""}</span></span>${bar(other.delta)}${delta(other.delta)}</li>` : "";
  return `<div class="fx" data-fx="${esc(p.key)}">
    <ol class="fx-list" aria-label="${esc(`${p.label || NAMES[p.key] || p.key}: ${amount(p.value, p.unit, digits)}, from ${p.baseLabel}`)}">
      <li class="fx-row base"><span class="fx-l"><span class="fx-name">${esc(p.baseLabel)}</span><span class="fx-meta">Starting point</span></span><span class="fx-bar" aria-hidden="true"></span><span class="fx-d num">${esc(amount(p.baseline, p.unit, digits))}</span></li>
      ${items}${otherRow}
      <li class="fx-row total"><span class="fx-l"><span class="fx-name">${esc(p.label || NAMES[p.key] || p.key)}</span></span><span class="fx-bar" aria-hidden="true"></span><b class="fx-d num"><span class="sr-only">equals </span>${esc(amount(p.value, p.unit, digits))}</b></li>
    </ol>
  </div>`;
}

/**
 * The game page's "Why the numbers" section: buttons to choose a number (when there are several), then its
 * baseline, factors and value. ``key``: the projection to show (default: the first). Falls back to the old
 * text lines (prediction.factors) when the server sends no typed factors; "" when there is nothing at all.
 */
export function factorsSection(game, { key = null } = {}) {
  const list = projectionsOf(game?.prediction);
  if (!list.length) {
    const lines = (Array.isArray(game?.prediction?.factors) ? game.prediction.factors : []).filter((f) => typeof f === "string");
    if (!lines.length) return "";
    return section("What moved the AI", `<ul class="bullets">${lines.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`, { id: "factors" });
  }
  ensureMarketStyles();
  const current = list.find((p) => p.key === key) || list[0];
  const picker = list.length > 1
    ? `<div class="fx-pick" role="group" aria-label="Which number">${list.map((p) => `<button type="button" class="chip${p.key === current.key ? " on" : ""}" data-fx-key="${esc(p.key)}" aria-pressed="${p.key === current.key}">${esc(NAMES[p.key] || p.label || p.key)}</button>`).join("")}</div>`
    : "";
  return section("Why the numbers", `${picker}<div data-fx-box>${projectionHtml(current)}</div>
    <p class="note">Each number starts from its baseline and every factor adds or takes away a sized amount, so the rows add up to the number shown. “Small, noisy” marks factors the evidence says are weak.</p>`, { id: "factors" });
}

/**
 * Switch the number shown (the buttons in the section); ``onPick(key)`` lets the page keep the choice across a
 * redraw. Returns a function that removes the listener.
 */
export function bindFactors(root, game, { onPick = null } = {}) {
  const click = (event) => {
    const t = event.target instanceof Element ? event.target.closest("[data-fx-key]") : null;
    if (!t) return;
    const list = projectionsOf(game?.prediction);
    const p = list.find((x) => x.key === t.dataset.fxKey);
    const box = root.querySelector("[data-fx-box]");
    if (!p || !box) return;
    box.innerHTML = projectionHtml(p);
    onPick?.(p.key);
    for (const b of root.querySelectorAll("[data-fx-key]")) {
      const on = b === t;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
  };
  root.addEventListener("click", click);
  return () => root.removeEventListener("click", click);
}
