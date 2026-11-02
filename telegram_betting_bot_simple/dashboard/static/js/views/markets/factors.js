// The model's explanation as typed factors (engine/factors.py; Contract B): each number the NBA model shows is a
// labelled baseline plus sized factors that add up to it. On the game page, drawn as a factor waterfall
// (views/context/waterfall.js, v3 Stage 4: the same drawing as "Why the line is here"): a baseline row, the factors
// sorted by size with signed deltas and units (the top 6; the rest fold into "Other (n factors) ±x" when there are
// two or more of them: one left over is shown by name, as folding it saves no room and hides it), bars stepping
// from the baseline to the number, "small, noisy" on weak ones, and the number they add up to with its sum line.
// With no typed factors from the server the old text lines show, as before.
//
// Data: GET /api/game's ``prediction.projections`` = {key: Projection}, Projection = {key, label, baseline_label,
// baseline, value, unit, factors: [{key, label, delta, unit, direction, evidence, source, as_of, confidence}]}
// (keys "home_win" (unit "%"), "margin", "total", and the team model's own "model_margin", "model_total"). The
// server rounds the deltas so they still add up to the value shown; the waterfall only lays them out, and the
// "Other" row adds up the folded deltas at the same precision.
//
//   projectionsOf(prediction) -> [Projection] (checked, in a fixed order; only those with factors)
//   factorsSection(game, {key}) -> the section's HTML: the waterfall, else the old lines, else ""
//   factorRows(projection)      -> {digits, rows, other: {n, delta} | null, rest} (the waterfall's rows)

import { esc, section } from "../../ui.js";
import { ensureMarketStyles } from "./styles.js";
import { SHOWN, projectionsOf, waterfallRows, waterfallHtml } from "../context/waterfall.js";

export { SHOWN, projectionsOf };
const NAMES = { home_win: "Win chance", margin: "Margin", total: "Total", model_margin: "Team model margin", model_total: "Team model total" };

/** The rows of one projection (see waterfall.js waterfallRows). */
export const factorRows = waterfallRows;

function projectionHtml(p) {
  return waterfallHtml({ ...p, label: p.label || NAMES[p.key] || p.key });
}

/**
 * The game page's "Why the numbers" section: buttons to choose a number (when there are several), then its
 * waterfall. ``key``: the projection to show (default: the first). Falls back to the old text lines
 * (prediction.factors) when the server sends no typed factors; "" when there is nothing at all.
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
