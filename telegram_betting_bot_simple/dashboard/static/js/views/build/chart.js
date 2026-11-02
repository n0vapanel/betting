// The rollover planner's survival chart: for each strategy, the chance its run is still going after each
// day (engine/rollover.py's survival curves, by the market's chances), as thin lines in plain SVG.
//
// Strategies with the same curve share one line (classic and bank-half both end at the first loss).
// Identity never rests on colour alone: each line has its own dash pattern, the legend uses the same
// line keys, the last value is written at the line's end, and a table of every day sits underneath.
// Hover, touch-drag or the arrow keys move a crosshair that reads every line at that day.

import { esc, pct } from "../../ui.js";

const WIDTH = 340; // drawn at the width it is shown, so its text stays at 11 px on a phone and a desktop
const H = 176;
const PAD = { left: 34, right: 40, top: 12, bottom: 26 };
const SERIES_MAX = 4;

const key = (n) => `<svg width="24" height="8" viewBox="0 0 24 8" aria-hidden="true"><line x1="1" y1="4" x2="23" y2="4" class="s s${n}"/></svg>`;

function sameCurve(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/**
 * Group strategies that share a curve: [{label, curve: [P(going after day 1), ...]}], at most four.
 * ``entries`` is [{name, curve}] in display order.
 */
export function groupCurves(entries) {
  const groups = [];
  for (const entry of entries) {
    if (!Array.isArray(entry.curve) || !entry.curve.length) continue;
    const same = groups.find((g) => sameCurve(g.curve, entry.curve));
    if (same) same.names.push(entry.name);
    else if (groups.length < SERIES_MAX) groups.push({ names: [entry.name], curve: entry.curve });
  }
  return groups.map((g) => ({ label: g.names.join(" & "), curve: g.curve }));
}

/** The chart's HTML (a <figure> to pass to bindSurvivalChart once it is in the page). */
export function survivalChart(series, { caption = "", width = WIDTH } = {}) {
  if (!series.length) return "";
  const W = Math.max(280, Math.min(680, Math.round(width)));
  const days = Math.max(...series.map((s) => s.curve.length));
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (d) => PAD.left + (days ? (d / days) * plotW : 0);
  const y = (p) => PAD.top + (1 - p) * plotH;
  const grid = [0, 0.5, 1].map((p) => `<line x1="${PAD.left}" x2="${W - PAD.right}" y1="${y(p).toFixed(1)}" y2="${y(p).toFixed(1)}"/>
    <text x="${PAD.left - 6}" y="${(y(p) + 4).toFixed(1)}" text-anchor="end">${p * 100}%</text>`).join("");
  const ticks = [...new Set([0, Math.round(days / 2), days])].map((d) => `<text x="${x(d).toFixed(1)}" y="${H - 8}" text-anchor="middle">${d === 0 ? "Start" : `Day ${d}`}</text>`).join("");
  // Every line starts at 100% (nobody has lost before day 1)
  const paths = series.map((s, i) => {
    const points = [1, ...s.curve].map((p, d) => `${d ? "L" : "M"}${x(d).toFixed(1)} ${y(p).toFixed(1)}`).join(" ");
    return `<path class="s s${i + 1}" d="${points}"/>`;
  }).join("");
  // End values, skipping any that would sit on top of one already written (the legend and table carry them)
  const written = [];
  const ends = series.map((s, i) => {
    const last = s.curve[s.curve.length - 1];
    const ly = y(last) + 4;
    if (written.some((w) => Math.abs(w - ly) < 12)) return "";
    written.push(ly);
    return `<circle class="dot d${i + 1}" cx="${x(s.curve.length).toFixed(1)}" cy="${y(last).toFixed(1)}" r="4"/>
      <text class="g-end" x="${(x(s.curve.length) + 8).toFixed(1)}" y="${ly.toFixed(1)}">${esc(pct(last))}</text>`;
  }).join("");
  const label = `Chance each plan is still going, day by day: ${series.map((s) => `${s.label} ${pct(s.curve[s.curve.length - 1])} after day ${s.curve.length}`).join("; ")}.`;
  const rows = Array.from({ length: days }, (_, d) => `<tr><td>Day ${d + 1}</td>${series.map((s) => `<td class="num">${s.curve[d] != null ? pct(s.curve[d]) : "–"}</td>`).join("")}</tr>`).join("");
  return `<figure class="bx-chart" data-days="${days}" data-w="${W}">
    <div class="bx-legend">${series.map((s, i) => `<span>${key(i + 1)}${esc(s.label)}</span>`).join("")}</div>
    <svg class="g-plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" tabindex="0">
      <g class="g-grid">${grid}</g>
      <line class="g-axis" x1="${PAD.left}" x2="${W - PAD.right}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"/>
      ${ticks}
      ${paths}
      ${ends}
      <g class="g-hover" style="display:none">
        <line class="g-cross" y1="${PAD.top}" y2="${PAD.top + plotH}"/>
        ${series.map((_, i) => `<circle class="dot d${i + 1}" r="4"/>`).join("")}
      </g>
      <rect class="g-hit" x="${PAD.left}" y="0" width="${plotW}" height="${H}" fill="transparent"/>
    </svg>
    <div class="bx-tip" hidden></div>
    ${caption ? `<figcaption class="hint">${esc(caption)}</figcaption>` : ""}
    <details class="bx-more"><summary>The chart as a table</summary>
      <div class="table-wrap"><table><thead><tr><th>After</th>${series.map((s) => `<th>${esc(s.label)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
    </details>
  </figure>`;
}

/** Hover, touch and keyboard reading for a chart drawn by survivalChart(); ``series`` is the same list. */
export function bindSurvivalChart(figure, series) {
  if (!figure || !series.length) return;
  const svg = figure.querySelector("svg.g-plot");
  const hover = svg.querySelector(".g-hover");
  const cross = hover.querySelector(".g-cross");
  const dots = [...hover.querySelectorAll("circle")];
  const tip = figure.querySelector(".bx-tip");
  const days = Number(figure.dataset.days);
  const W = Number(figure.dataset.w) || WIDTH;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  let day = days;

  const show = (d) => {
    day = Math.max(1, Math.min(days, d));
    const cx = PAD.left + (day / days) * plotW;
    cross.setAttribute("x1", cx);
    cross.setAttribute("x2", cx);
    series.forEach((s, i) => {
      const p = s.curve[day - 1];
      dots[i].setAttribute("cx", cx);
      dots[i].setAttribute("cy", PAD.top + (1 - (p ?? 0)) * plotH);
      dots[i].style.display = p == null ? "none" : "";
    });
    hover.style.display = "";
    // Labels go in with textContent: values lead, the line's name follows
    tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "t-day";
    head.textContent = `Still going after day ${day}`;
    tip.appendChild(head);
    series.forEach((s, i) => {
      const row = document.createElement("div");
      row.innerHTML = key(i + 1);
      const value = document.createElement("b");
      value.textContent = s.curve[day - 1] != null ? pct(s.curve[day - 1]) : "–";
      const name = document.createElement("span");
      name.textContent = s.label;
      row.append(value, name);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const box = svg.getBoundingClientRect();
    const px = (cx / W) * box.width;
    const width = tip.offsetWidth;
    const left = px + 12 + width > box.width ? px - 12 - width : px + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${box.top - figure.getBoundingClientRect().top}px`;
  };
  const hide = () => {
    hover.style.display = "none";
    tip.hidden = true;
  };
  const fromPointer = (event) => {
    const box = svg.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * W;
    show(Math.round(((sx - PAD.left) / plotW) * days));
  };
  svg.addEventListener("pointermove", fromPointer);
  svg.addEventListener("pointerdown", fromPointer);
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(day));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -1, ArrowRight: 1, Home: -days, End: days }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    show(day + step);
  });
}
