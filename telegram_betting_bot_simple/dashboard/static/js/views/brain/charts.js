// Brain progress charts, in plain SVG like the rollover planner's (views/build/chart.js): one hue (--bp-c1, the brain;
// validated for contrast on the app's and Telegram's light and dark cards), a 95% band as a light wash of the same hue,
// hairline grid and axes, one scale per chart with labelled axes, a crosshair or per-bar readout on hover, touch and
// the arrow keys, and every value again in a table underneath (the tooltip never holds a value alone).
//
//   gapChartHtml(sport, changes, {width})   the weekly log-loss gap (brain minus sharp market) with its 95% band, a
//                                           zero line ("level with the market"), and markers where a change was promoted
//                                           (solid) or was rolled back or undone (dashed)
//   bindGapChart(figure, sport, changes, {onPick})   readout; Enter or a second tap on a marked week calls onPick(change)
//   reliabilityHtml(cal, {width}) / bindReliability(figure, cal)   the CORP reliability curve over its consistency band
//   pitHtml(hist, {width}) / bindPit(figure, hist)                  a PIT histogram against flat (1) and its band
//   markersFor(sport, changes) -> [{key, week, kind: "live"|"back", decision, label, change}] (the server's markers)
// The figures are the server's; the only arithmetic here is placing them on the scale. Text goes in through esc() or
// textContent.

import { esc, pct } from "../../ui.js";
import { ll, range, weekStart, weekLabel, isoWeek, moveText } from "./model.js";

const DAY = 86400000;
const clampW = (w, lo = 280, hi = 760) => Math.max(lo, Math.min(hi, Math.round(Number(w) || 340)));
const f1 = (v) => v.toFixed(1);
const lineKey = (cls) => `<svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" class="${cls}"/></svg>`;
const bandKey = '<svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><rect x="1" y="1" width="20" height="8" rx="2" class="g-band"/></svg>';
const markKey = (cls) => `<svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true"><line x1="6" y1="1" x2="6" y2="13" class="g-mark ${cls}"/></svg>`;

/** Clean ticks over [lo, hi] (about ``count`` of them) and the decimals they need. */
export function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) { const pad = Math.abs(lo) * 0.5 || 0.001; lo -= pad; hi += pad; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step);
  let digits = Math.max(0, -Math.floor(Math.log10(step)));
  if (Math.abs(Math.round(step * 10 ** digits) - step * 10 ** digits) > 1e-9) digits += 1;
  return { ticks, digits, lo: ticks[0], hi: ticks[ticks.length - 1] };
}

const weekOrdinal = (week) => { const d = weekStart(week); return d ? Math.round(d.getTime() / (7 * DAY)) : null; };

/**
 * The sport's markers (the server's: each promotion, rollback and undo, by week) with the change card each one opens:
 * the ledger entry with the same version and decision, else the promotion it ended.
 */
export function markersFor(sport, changes) {
  const mine = (changes || []).filter((c) => sport && c.sport === sport.sport);
  return (sport?.markers || []).map((m) => {
    const same = mine.filter((c) => c.version === m.version);
    const change = same.find((c) => c.decision === m.decision) || same.find((c) => c.decision === "promoted") || null;
    return { key: weekOrdinal(m.week), week: m.week, kind: m.kind, decision: m.decision, label: m.label, change };
  }).filter((m) => m.key !== null);
}

/** What a marker means, in the gate's (or the owner's) own word. */
export function markerWords(m) {
  return { promoted: "Promoted", rolled_back: "Rolled back", undone: "Undone by the owner" }[m.decision] || "Changed";
}

const markerNote = (m) => {
  const c = m.change;
  const move = c && c.old !== null && c.new !== null && m.decision === "promoted" ? ` (${moveText(c.old, c.new)})` : "";
  return `${markerWords(m)}: ${m.label}${move}`;
};

function figureShell({ cls, label, svg, legend = "", caption = "", table = "", w, extra = "" }) {
  return `<figure class="bp-chart ${cls}" data-w="${w}"${extra}>
    ${legend ? `<div class="bp-legend">${legend}</div>` : ""}
    ${svg.replace("<svg ", `<svg role="img" aria-label="${esc(label)}" tabindex="0" `)}
    <div class="bp-tip" hidden></div>
    ${caption ? `<figcaption class="hint">${caption}</figcaption>` : ""}
    ${table ? `<details class="bp-more"><summary>The chart as a table</summary><div class="table-wrap">${table}</div></details>` : ""}
  </figure>`;
}

// --- the weekly gap -------------------------------------------------------------------------------------------------

const GAP = { H: 210, left: 52, right: 58, top: 12, bottom: 28 };

function gapGeometry(points, W) {
  const keys = points.map((p) => weekOrdinal(p.week));
  const k0 = keys[0];
  const k1 = keys[keys.length - 1];
  const plotW = W - GAP.left - GAP.right;
  const plotH = GAP.H - GAP.top - GAP.bottom;
  const values = [0, ...points.flatMap((p) => [p.gap, p.lo, p.hi])].filter((v) => v !== null);
  const scale = niceTicks(Math.min(...values), Math.max(...values), 4);
  const x = (k) => GAP.left + (k1 === k0 ? plotW / 2 : ((k - k0) / (k1 - k0)) * plotW);
  const y = (v) => GAP.top + (1 - (v - scale.lo) / (scale.hi - scale.lo)) * plotH;
  return { keys, k0, k1, plotW, plotH, scale, x, y };
}

/** The weekly gap chart's HTML (a <figure> for bindGapChart once it is in the page); "" without weekly points. */
export function gapChartHtml(sport, changes, { width = 340 } = {}) {
  const points = sport?.series || [];
  if (!points.length) return "";
  const W = clampW(width, 280, 1000);
  const g = gapGeometry(points, W);
  const { x, y, scale } = g;
  const grid = scale.ticks.map((t) => `<line x1="${GAP.left}" x2="${W - GAP.right}" y1="${f1(y(t))}" y2="${f1(y(t))}"/>
    <text x="${GAP.left - 6}" y="${f1(y(t) + 4)}" text-anchor="end">${esc(ll(t, scale.digits))}</text>`).join("");
  // The 95% band, one run of consecutive weeks with both ends known at a time
  const runs = [];
  let run = [];
  points.forEach((p, i) => {
    if (p.lo !== null && p.hi !== null) run.push([g.keys[i], p]);
    else if (run.length) { runs.push(run); run = []; }
  });
  if (run.length) runs.push(run);
  const band = runs.map((r) => {
    if (r.length === 1) {
      const [k, p] = r[0];
      return `<rect class="g-band" x="${f1(x(k) - 3)}" y="${f1(y(p.hi))}" width="6" height="${f1(Math.max(1, y(p.lo) - y(p.hi)))}" rx="2"/>`;
    }
    const top = r.map(([k, p], i) => `${i ? "L" : "M"}${f1(x(k))} ${f1(y(p.hi))}`).join(" ");
    const bottom = r.slice().reverse().map(([k, p]) => `L${f1(x(k))} ${f1(y(p.lo))}`).join(" ");
    return `<path class="g-band" d="${top} ${bottom} Z"/>`;
  }).join("");
  const line = points.length > 1
    ? `<path class="g-line" d="${points.map((p, i) => `${i ? "L" : "M"}${f1(x(g.keys[i]))} ${f1(y(p.gap))}`).join(" ")}"/>` : "";
  const last = points[points.length - 1];
  const lastY = y(last.gap);
  const zeroY = y(0);
  // The zero line's word sits in the right margin unless the last value's label would sit on it
  const zeroWord = Math.abs(lastY - zeroY) >= 13 ? `<text class="g-zero-t" x="${W - GAP.right + 8}" y="${f1(zeroY + 4)}">level</text>` : "";
  const marks = markersFor(sport, changes).filter((m) => m.key >= g.k0 && m.key <= g.k1);
  const markLines = marks.map((m) => `<line class="g-mark ${m.kind === "live" ? "live" : "back"}" x1="${f1(x(m.key))}" x2="${f1(x(m.key))}" y1="${GAP.top}" y2="${GAP.top + g.plotH}"/>`).join("");
  const tickIdx = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];
  const xTicks = tickIdx.map((i) => {
    const anchor = points.length === 1 ? "middle" : i === 0 ? "start" : i === points.length - 1 ? "end" : "middle";
    return `<text x="${f1(x(g.keys[i]))}" y="${GAP.H - 8}" text-anchor="${anchor}">${esc(weekLabel(points[i].week))}</text>`;
  }).join("");
  const svg = `<svg class="g-plot" viewBox="0 0 ${W} ${GAP.H}">
      <g class="g-grid">${grid}</g>
      ${band}
      <line class="g-zero" x1="${GAP.left}" x2="${W - GAP.right}" y1="${f1(zeroY)}" y2="${f1(zeroY)}"/>
      ${zeroWord}
      ${markLines}
      ${line}
      <circle class="g-dot" cx="${f1(x(g.keys[points.length - 1]))}" cy="${f1(lastY)}" r="4"/>
      <text class="g-end" x="${f1(x(g.keys[points.length - 1]) + 8)}" y="${f1(lastY + 4)}">${esc(ll(last.gap))}</text>
      <line class="g-axis" x1="${GAP.left}" x2="${W - GAP.right}" y1="${GAP.top + g.plotH}" y2="${GAP.top + g.plotH}"/>
      ${xTicks}
      <g class="g-hover" style="display:none"><line class="g-cross" y1="${GAP.top}" y2="${GAP.top + g.plotH}"/><circle class="g-dot" r="4"/></g>
    </svg>`;
  const hasLive = marks.some((m) => m.kind === "live");
  const hasBack = marks.some((m) => m.kind === "back");
  const legend = [
    `<span>${lineKey("g-line")}Brain minus market, each week</span>`,
    `<span>${bandKey}95% range</span>`,
    hasLive ? `<span>${markKey("live")}Promoted</span>` : "",
    hasBack ? `<span>${markKey("back")}Rolled back or undone</span>` : "",
  ].join("");
  const label = `${sport.label}: the brain's weekly log loss minus the sharp market's, ${points.length} week${points.length === 1 ? "" : "s"}, latest ${ll(last.gap)}${last.lo !== null && last.hi !== null ? ` (95% range ${range(last.lo, last.hi)})` : ""}. Below 0 the brain did better than the market.`;
  const table = `<table><caption class="sr-only">${esc(`${sport.label}: weekly log-loss gap`)}</caption>
    <thead><tr><th scope="col">Week of</th><th scope="col">Games</th><th scope="col">Gap</th><th scope="col">95% range</th><th scope="col">Changes</th></tr></thead>
    <tbody>${points.map((p, i) => {
      const here = marks.filter((m) => m.key === g.keys[i]);
      return `<tr><th scope="row">${esc(weekLabel(p.week))}</th><td class="num">${esc(p.n ?? "–")}</td><td class="num">${esc(ll(p.gap))}</td>
        <td class="num">${esc(range(p.lo, p.hi) || "–")}</td><td>${esc(here.map(markerNote).join("; ") || "–")}</td></tr>`;
    }).join("")}</tbody></table>`;
  return figureShell({
    cls: "bp-gap", label, svg, legend, table, w: W,
    caption: `${esc("Log loss per game, brain minus the sharp market, on the same games. Below 0: the brain did better that week; above 0: worse. Weeks run Monday to Sunday.")}`,
  });
}

// --- the readout (hover, touch, keys) --------------------------------------------------------------------------------

/**
 * Hover, touch and keyboard reading for a figure: ``xs`` are the readable positions (viewBox x), ``show(i, hover)``
 * places the hover marks and returns {head, rows: [[value, label]], foot}; ``pick(i)`` (optional) runs on Enter, a
 * click after hovering, or a second tap on the same position.
 */
function bindReadout(figure, { xs, show, pick = null, start = xs.length - 1 }) {
  const svg = figure?.querySelector("svg.g-plot");
  if (!svg || !xs.length) return;
  const tip = figure.querySelector(".bp-tip");
  const hover = svg.querySelector(".g-hover");
  const W = Number(figure.dataset.w) || 340;
  let at = start;
  let shown = -1;
  let shownBeforeDown = -1;
  const nearest = (sx) => {
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(v - sx) < Math.abs(xs[best] - sx)) best = i; });
    return best;
  };
  const draw = (i) => {
    at = Math.max(0, Math.min(xs.length - 1, i));
    const out = show(at, hover) || {};
    if (hover) hover.style.display = "";
    tip.replaceChildren();
    if (out.head) {
      const head = document.createElement("div");
      head.className = "t-head";
      head.textContent = out.head;
      tip.appendChild(head);
    }
    for (const [value, label] of out.rows || []) {
      const row = document.createElement("div");
      const v = document.createElement("b");
      v.textContent = value;
      const k = document.createElement("span");
      k.textContent = label;
      row.append(v, k);
      tip.appendChild(row);
    }
    for (const line of out.notes || []) {
      const note = document.createElement("div");
      note.className = "t-note";
      note.textContent = line;
      tip.appendChild(note);
    }
    if (out.foot) {
      const foot = document.createElement("div");
      foot.className = "t-foot";
      foot.textContent = out.foot;
      tip.appendChild(foot);
    }
    tip.hidden = false;
    const box = svg.getBoundingClientRect();
    const fig = figure.getBoundingClientRect();
    const px = (xs[at] / W) * box.width;
    const width = tip.offsetWidth;
    const left = px + 14 + width > box.width ? px - 14 - width : px + 14;
    tip.style.left = `${Math.max(0, Math.min(box.width - width, left)) + (box.left - fig.left)}px`;
    tip.style.top = `${box.top - fig.top + 4}px`;
    shown = at;
  };
  const hide = () => {
    if (hover) hover.style.display = "none";
    tip.hidden = true;
    shown = -1;
  };
  const fromPointer = (event) => {
    const box = svg.getBoundingClientRect();
    return nearest(((event.clientX - box.left) / box.width) * W);
  };
  svg.addEventListener("pointermove", (event) => { if (event.pointerType === "mouse") draw(fromPointer(event)); });
  svg.addEventListener("pointerdown", (event) => {
    const i = fromPointer(event);
    shownBeforeDown = shown === i ? i : -1;
    draw(i);
  });
  svg.addEventListener("click", (event) => {
    const i = fromPointer(event);
    if (pick && shownBeforeDown === i) pick(i);
    shownBeforeDown = shown;
  });
  svg.addEventListener("pointerleave", (event) => { if (event.pointerType === "mouse") hide(); });
  svg.addEventListener("focus", () => draw(at));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && pick) {
      event.preventDefault();
      pick(at);
      return;
    }
    if (event.key === "Escape") { hide(); return; }
    const step = { ArrowLeft: -1, ArrowRight: 1, Home: -xs.length, End: xs.length }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    draw(at + step);
  });
}

/** The gap chart's readout: the week, its gap and range, and the changes marked there (Enter: open the first). */
export function bindGapChart(figure, sport, changes, { onPick = null } = {}) {
  const points = sport?.series || [];
  if (!figure || !points.length) return;
  const W = Number(figure.dataset.w) || 340;
  const g = gapGeometry(points, W);
  const marks = markersFor(sport, changes);
  const xs = g.keys.map((k) => g.x(k));
  const at = (i) => marks.filter((m) => m.key === g.keys[i]);
  bindReadout(figure, {
    xs,
    show(i, hover) {
      const p = points[i];
      const cx = f1(xs[i]);
      hover.querySelector(".g-cross").setAttribute("x1", cx);
      hover.querySelector(".g-cross").setAttribute("x2", cx);
      const dot = hover.querySelector(".g-dot");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", f1(g.y(p.gap)));
      const here = at(i);
      return {
        head: `Week of ${weekLabel(p.week)}${p.n !== null ? ` · ${p.n} games` : ""}`,
        rows: [[ll(p.gap), "brain minus market"], ...(p.lo !== null && p.hi !== null ? [[range(p.lo, p.hi), "95% range"]] : [])],
        notes: here.map(markerNote),
        foot: here.some((m) => m.change) && onPick ? "Select again (or press Enter) to see the change" : "",
      };
    },
    pick(i) {
      const first = at(i).find((m) => m.change);
      if (first && onPick) onPick(first.change);
    },
  });
}

// --- calibration -----------------------------------------------------------------------------------------------------

const CAL = { left: 44, right: 14, top: 10, bottom: 40 };

function calGeometry(W) {
  const H = Math.round(Math.min(W, 420) * 0.8);
  const plotW = W - CAL.left - CAL.right;
  const plotH = H - CAL.top - CAL.bottom;
  return { H, plotW, plotH, x: (p) => CAL.left + p * plotW, y: (o) => CAL.top + (1 - o) * plotH };
}

/** The reliability curve over its consistency band, with the diagonal ("perfectly calibrated"); "" without points. */
export function reliabilityHtml(cal, { width = 340, label: name = "" } = {}) {
  const curve = cal?.curve || [];
  if (!curve.length) return "";
  const W = clampW(width, 280, 520);
  const g = calGeometry(W);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const grid = ticks.map((t) => `<line x1="${CAL.left}" x2="${W - CAL.right}" y1="${f1(g.y(t))}" y2="${f1(g.y(t))}"/>
      <text x="${CAL.left - 6}" y="${f1(g.y(t) + 4)}" text-anchor="end">${t * 100}%</text>
      <text x="${f1(g.x(t))}" y="${f1(CAL.top + g.plotH + 15)}" text-anchor="${t === 0 ? "start" : t === 1 ? "end" : "middle"}">${t * 100}%</text>`).join("");
  const band = cal.band.length > 1
    ? `<path class="g-band" d="${cal.band.map((b, i) => `${i ? "L" : "M"}${f1(g.x(b.p))} ${f1(g.y(b.hi))}`).join(" ")} ${cal.band.slice().reverse().map((b) => `L${f1(g.x(b.p))} ${f1(g.y(b.lo))}`).join(" ")} Z"/>` : "";
  const line = curve.length > 1 ? `<path class="g-line" d="${curve.map((c, i) => `${i ? "L" : "M"}${f1(g.x(c.p))} ${f1(g.y(c.obs))}`).join(" ")}"/>` : "";
  // A dot per point while they stay apart; a dense CORP curve (21 steps) is the line alone, its dot on hover
  const dots = curve.length <= 12 ? curve.map((c) => `<circle class="g-dot" cx="${f1(g.x(c.p))}" cy="${f1(g.y(c.obs))}" r="4"/>`).join("") : "";
  const svg = `<svg class="g-plot" viewBox="0 0 ${W} ${g.H}">
      <g class="g-grid">${grid}</g>
      ${band}
      <line class="g-diag" x1="${f1(g.x(0))}" y1="${f1(g.y(0))}" x2="${f1(g.x(1))}" y2="${f1(g.y(1))}"/>
      ${line}${dots}
      <line class="g-axis" x1="${CAL.left}" x2="${W - CAL.right}" y1="${f1(CAL.top + g.plotH)}" y2="${f1(CAL.top + g.plotH)}"/>
      <text class="g-title" x="${f1(CAL.left + g.plotW / 2)}" y="${g.H - 4}" text-anchor="middle">The brain said</text>
      <text class="g-title" transform="translate(11 ${f1(CAL.top + g.plotH / 2)}) rotate(-90)" text-anchor="middle">It happened</text>
      <g class="g-hover" style="display:none"><line class="g-cross" y1="${CAL.top}" y2="${CAL.top + g.plotH}"/><circle class="g-dot" r="5"/></g>
    </svg>`;
  const legend = [
    `<span>${lineKey("g-line")}The brain</span>`,
    cal.band.length > 1 ? `<span>${bandKey}Where a well-calibrated brain lands at this many games</span>` : "",
    `<span>${lineKey("g-diag")}Perfectly calibrated</span>`,
  ].join("");
  const bandAt = (p) => cal.band.find((b) => Math.abs(b.p - p) < 1e-6) || null;
  const table = `<table><caption class="sr-only">${esc(`${name} calibration`)}</caption>
    <thead><tr><th scope="col">Brain said</th><th scope="col">Happened</th><th scope="col">Well-calibrated range</th></tr></thead>
    <tbody>${curve.map((c) => {
      const b = bandAt(c.p);
      return `<tr><th scope="row">${esc(pct(c.p))}</th><td class="num">${esc(pct(c.obs))}</td><td class="num">${b ? `${esc(pct(b.lo))} to ${esc(pct(b.hi))}` : "–"}</td></tr>`;
    }).join("")}</tbody></table>`;
  const label = `${name} calibration over ${cal.n ?? "–"} graded games: ${curve.map((c) => `said ${pct(c.p)}, happened ${pct(c.obs)}`).join("; ")}.`;
  return figureShell({ cls: "bp-cal", label, svg, legend, table, w: W });
}

export function bindReliability(figure, cal) {
  const curve = cal?.curve || [];
  if (!figure || !curve.length) return;
  const W = Number(figure.dataset.w) || 340;
  const g = calGeometry(W);
  const bandAt = (p) => cal.band.find((b) => Math.abs(b.p - p) < 1e-6) || null;
  bindReadout(figure, {
    xs: curve.map((c) => g.x(c.p)),
    start: Math.floor(curve.length / 2),
    show(i, hover) {
      const c = curve[i];
      const cx = f1(g.x(c.p));
      hover.querySelector(".g-cross").setAttribute("x1", cx);
      hover.querySelector(".g-cross").setAttribute("x2", cx);
      const dot = hover.querySelector(".g-dot");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", f1(g.y(c.obs)));
      const b = bandAt(c.p);
      return {
        head: `When the brain said ${pct(c.p)}`,
        rows: [[pct(c.obs), "it happened"], ...(b ? [[`${pct(b.lo)} to ${pct(b.hi)}`, "well-calibrated range"]] : [])],
      };
    },
  });
}

// --- PIT histogram ---------------------------------------------------------------------------------------------------

const PIT = { H: 160, left: 36, right: 40, top: 10, bottom: 40 };

function pitGeometry(hist, W) {
  const k = hist.bins.length;
  const plotW = W - PIT.left - PIT.right;
  const plotH = PIT.H - PIT.top - PIT.bottom;
  const top = Math.max(1.5, ...hist.bins, hist.hi ?? 0) * 1.08;
  const scale = niceTicks(0, top, 3);
  const slot = plotW / k;
  const bar = Math.max(4, Math.min(24, slot - 2));
  return { k, plotW, plotH, slot, bar, scale, x: (i) => PIT.left + slot * (i + 0.5), y: (v) => PIT.top + (1 - v / scale.hi) * plotH };
}

const binRange = (i, k) => `${(i / k).toFixed(1)}–${((i + 1) / k).toFixed(1)}`;

/** A PIT histogram: one bar per slice of the predicted distribution, against flat (1) and the band noise allows. */
export function pitHtml(hist, { width = 340 } = {}) {
  if (!hist?.bins?.length) return "";
  const W = clampW(width, 280, 560);
  const g = pitGeometry(hist, W);
  const base = PIT.top + g.plotH;
  const grid = g.scale.ticks.map((t) => `<line x1="${PIT.left}" x2="${W - PIT.right}" y1="${f1(g.y(t))}" y2="${f1(g.y(t))}"/>
      <text x="${PIT.left - 6}" y="${f1(g.y(t) + 4)}" text-anchor="end">${t.toFixed(g.scale.digits)}</text>`).join("");
  const band = hist.lo !== null && hist.hi !== null
    ? `<rect class="g-band" x="${PIT.left}" width="${f1(g.plotW)}" y="${f1(g.y(hist.hi))}" height="${f1(Math.max(1, g.y(hist.lo) - g.y(hist.hi)))}"/>` : "";
  // Bars grow from the baseline, 4 px rounded at the data end, square at the base
  const bars = hist.bins.map((v, i) => {
    const h = Math.max(0, base - g.y(v));
    const r = Math.min(4, h, g.bar / 2);
    const x0 = g.x(i) - g.bar / 2;
    const x1 = x0 + g.bar;
    const yt = base - h;
    return `<path class="g-bar" data-i="${i}" d="M${f1(x0)} ${f1(base)} L${f1(x0)} ${f1(yt + r)} Q${f1(x0)} ${f1(yt)} ${f1(x0 + r)} ${f1(yt)} L${f1(x1 - r)} ${f1(yt)} Q${f1(x1)} ${f1(yt)} ${f1(x1)} ${f1(yt + r)} L${f1(x1)} ${f1(base)} Z"/>`;
  }).join("");
  const svg = `<svg class="g-plot" viewBox="0 0 ${W} ${PIT.H}">
      <g class="g-grid">${grid}</g>
      ${band}
      ${bars}
      <line class="g-zero" x1="${PIT.left}" x2="${W - PIT.right}" y1="${f1(g.y(1))}" y2="${f1(g.y(1))}"/>
      <text class="g-zero-t" x="${W - PIT.right + 6}" y="${f1(g.y(1) + 4)}">flat</text>
      <line class="g-axis" x1="${PIT.left}" x2="${W - PIT.right}" y1="${f1(base)}" y2="${f1(base)}"/>
      <text x="${PIT.left}" y="${f1(base + 15)}" text-anchor="start">0</text>
      <text x="${f1(PIT.left + g.plotW / 2)}" y="${f1(base + 15)}" text-anchor="middle">0.5</text>
      <text x="${f1(W - PIT.right)}" y="${f1(base + 15)}" text-anchor="end">1</text>
      <text class="g-title" x="${f1(PIT.left + g.plotW / 2)}" y="${PIT.H - 4}" text-anchor="middle">Where the result fell in the forecast (PIT)</text>
      <g class="g-hover" style="display:none"><line class="g-cross" y1="${PIT.top}" y2="${f1(base)}"/></g>
    </svg>`;
  const legend = [
    `<span><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="1" y="1" width="10" height="10" rx="2" class="g-bar"/></svg>Share of games, against flat</span>`,
    band ? `<span>${bandKey}Where flat bars land by chance at this many games</span>` : "",
  ].join("");
  const table = `<table><caption class="sr-only">${esc(`${hist.label}: PIT histogram`)}</caption>
    <thead><tr><th scope="col">Slice</th><th scope="col">Height (flat = 1)</th></tr></thead>
    <tbody>${hist.bins.map((v, i) => `<tr><th scope="row">${esc(binRange(i, g.k))}</th><td class="num">${esc(v.toFixed(2))}</td></tr>`).join("")}</tbody></table>`;
  const label = `${hist.label}: PIT histogram over ${hist.n ?? "–"} games, bar heights ${hist.bins.map((v) => v.toFixed(2)).join(", ")}; flat bars (1) mean the spread is right.`;
  return figureShell({ cls: "bp-pit", label, svg, legend, table, w: W });
}

export function bindPit(figure, hist) {
  if (!figure || !hist?.bins?.length) return;
  const W = Number(figure.dataset.w) || 340;
  const g = pitGeometry(hist, W);
  const bars = [...figure.querySelectorAll(".g-bar[data-i]")];
  bindReadout(figure, {
    xs: hist.bins.map((_, i) => g.x(i)),
    start: 0,
    show(i, hover) {
      const cx = f1(g.x(i));
      hover.querySelector(".g-cross").setAttribute("x1", cx);
      hover.querySelector(".g-cross").setAttribute("x2", cx);
      bars.forEach((b, j) => b.classList.toggle("on", j === i));
      return { head: `${hist.label}, slice ${binRange(i, g.k)}`, rows: [[hist.bins[i].toFixed(2), "height (flat is 1)"]] };
    },
  });
  figure.querySelector("svg.g-plot")?.addEventListener("blur", () => bars.forEach((b) => b.classList.remove("on")));
}
