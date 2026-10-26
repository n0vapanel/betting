// Styles only the Build screen uses. styles.css holds the shared design system (tokens, cards, chips,
// sheets); these few layout rules are added once, when Build first opens, and every one is scoped to
// .view-build so nothing here can change another screen. Colours are the shared tokens, except the
// survival chart's series colours (validated for colour-blind separation on both card surfaces).

const CSS = `
.view-build {
  --bx-c1: #3987e5; --bx-c2: #d95926; --bx-c3: #199e70; --bx-c4: #c98500;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) .view-build { --bx-c1: #2a78d6; --bx-c2: #eb6834; --bx-c3: #1baf7a; --bx-c4: #eda100; }
}
:root[data-theme="light"] .view-build { --bx-c1: #2a78d6; --bx-c2: #eb6834; --bx-c3: #1baf7a; --bx-c4: #eda100; }

.view-build .bx-modes { overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; }
.view-build .bx-modes::-webkit-scrollbar { display: none; }
.view-build .bx-modes > button, .view-build .bx-modes > a { flex: 1 0 auto; padding: 0 8px; font-size: 13.5px; min-height: 40px; }
/* Phones of 380 px and less: all five fit without scrolling sideways (at 360 px "Rollover" was cut off) */
@media (max-width: 380px) {
  .view-build .bx-modes { gap: 2px; }
  .view-build .bx-modes > button, .view-build .bx-modes > a { padding: 0 4px; font-size: 13px; }
}
.view-build .bx-sub { margin-top: 10px; display: grid; gap: 8px; }
.view-build .stake-row .money-input { max-width: 240px; }
.view-build .bx-sub .hint { margin-top: 0; }
.view-build .bx-slip { margin-top: 12px; }
.view-build .bx-slip > .section-head { margin-bottom: 8px; }
.view-build .slip-leg.weak { border-color: color-mix(in srgb, var(--warn) 55%, transparent); }
.view-build .bx-flags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.view-build .bx-total { margin-top: 8px; font-size: 13.5px; color: var(--text-2); }
.view-build .bx-total b { color: var(--text); }
.view-build .bx-stale { opacity: 0.55; transition: opacity var(--speed); }
.view-build .bx-tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 14px; }
.view-build .bx-tile { background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; min-width: 0; }
.view-build .bx-tile > span { display: flex; align-items: center; gap: 2px; color: var(--text-2); font-size: 12.5px; }
.view-build .bx-tile > b { display: block; font-size: 20px; font-weight: 800; line-height: 1.25; margin-top: 2px; }
.view-build .bx-tile > small { display: block; color: var(--text-2); font-size: 12.5px; margin-top: 2px; }
.view-build .bx-tile .badge { margin-top: 6px; }
.view-build .bx-outcomes { margin-top: 12px; }
.view-build .bx-outcomes summary, .view-build details.bx-more > summary {
  cursor: pointer; min-height: var(--tap); display: flex; align-items: center; gap: 6px; font-weight: 650; font-size: 14px; color: var(--info);
}
.view-build details.bx-more { margin-top: 12px; }
/* Picks off (a break, the picks pause): the builder's form and the Safe / Risky / Crazy panel, dimmed and
   inert, under one line that says until when (WCAG exempts inactive controls from contrast minimums) */
.view-build .bx-fieldset { border: 0; margin: 0; padding: 0; min-width: 0; }
.view-build .bx-fieldset:disabled { opacity: 0.55; }
.view-build .bx-fieldset:disabled :is(button, input, .chip) { cursor: not-allowed; }
.view-build details.bx-more.off > summary { color: var(--text-2); cursor: not-allowed; flex-wrap: wrap; }
.view-build details.bx-more.off > summary [data-b-off] { margin-left: auto; }
.view-build .bx-block { margin-top: 12px; }
.view-build .bx-block > div { display: flex; flex-wrap: wrap; align-items: center; column-gap: 8px; }
.view-build .bx-block .text-link { margin: -10px 0; }
.view-build .bx-actions { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 14px; }
.view-build .bx-actions .btn { min-width: 0; padding: 0 10px; }
.view-build .bx-actions .bx-main { grid-column: 1 / -1; }
.view-build .bx-saved { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; margin-top: 10px; font-size: 13.5px; color: var(--text-2); }
.view-build .bx-saved b { color: var(--text); letter-spacing: 0.06em; }
.view-build .bx-undo { justify-content: space-between; margin-top: 12px; padding: 4px 4px 4px 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); }
.view-build .bx-undo > span { flex: 1 1 12em; }
.view-build .bx-linkrow { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; }
.view-build .bx-steps { list-style: none; display: grid; gap: 6px; margin-top: 12px; font-size: 14px; color: var(--text-3); }
.view-build .bx-steps li { display: flex; gap: 8px; align-items: center; }
.view-build .bx-steps li.done { color: var(--text-2); }
.view-build .bx-steps li.now { color: var(--text); font-weight: 650; }
.view-build .bx-analysis .lead { margin-top: 12px; }
.view-build .bx-legnotes { list-style: none; display: grid; gap: 10px; margin-top: 4px; padding: 0; }
.view-build .bx-legnotes > li { padding: 10px 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); }
.view-build .bx-legnotes p { font-size: 14px; margin-top: 6px; }
.view-build .bx-legnotes .bullets { margin-top: 6px; font-size: 13px; color: var(--text-2); }
.view-build .bx-alts { display: grid; gap: 10px; }
.view-build .bx-alt { padding: 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); display: grid; gap: 8px; }
.view-build .bx-alt p { font-size: 13.5px; color: var(--text-2); }
.view-build .bx-tickets { display: grid; gap: 12px; margin-top: 12px; }
.view-build .bx-ticket { border-top: 3px solid var(--line-strong); display: grid; gap: 10px; }
.view-build .bx-ticket.t-safe { border-top-color: var(--accent); }
.view-build .bx-ticket.t-risky { border-top-color: var(--warn); }
.view-build .bx-ticket.t-crazy { border-top-color: var(--violet); }
.view-build .bx-ticket .summary { margin-top: 0; }
.view-build .bx-ticket-legs { list-style: none; display: grid; gap: 6px; }
.view-build .bx-ticket-legs li { border-top: 1px solid var(--line); padding-top: 6px; }
.view-build .bx-ticket-legs li:first-child { border-top: 0; padding-top: 0; }
.view-build .bx-form .chip-row .chip { padding: 0 11px; gap: 5px; font-size: 13.5px; }
.view-build .bx-two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.view-build .bx-two > .field { margin-top: 0; min-width: 0; }
.view-build .bx-form .field + .bx-two, .view-build .bx-form .bx-two + .field, .view-build .bx-form .bx-two + .bx-two { margin-top: 18px; }
.view-build .bx-stepper input {
  width: 56px; min-height: var(--tap); border: 0; background: transparent; color: var(--text); text-align: center;
  font-weight: 800; font-size: 16px; font-variant-numeric: tabular-nums;
}
.view-build .bx-stepper input:focus { outline: none; box-shadow: inset 0 0 0 2px var(--accent-soft); }
.view-build .bx-stepper button:disabled { opacity: 0.35; }
.bx-sheet .bx-code {
  font: 800 30px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: 0.14em; text-align: center;
  padding: 16px 12px; border-radius: 12px; background: var(--card-2); border: 1px dashed var(--line-strong); word-break: break-all;
}
.bx-sheet .bx-code + .hint { word-break: break-all; text-align: center; }
.view-build .bx-sentence { margin-top: 12px; font-size: 15px; line-height: 1.45; background: var(--card-2); border-radius: 12px;
  padding: 12px 14px; border-left: 3px solid var(--warn); }
.view-build .bx-compare { margin-top: 12px; }
.view-build .bx-compare table { font-size: 12.5px; table-layout: fixed; min-width: 340px; }
.view-build .bx-compare th, .view-build .bx-compare td { padding: 8px 4px; white-space: normal; vertical-align: top; overflow-wrap: anywhere; }
.view-build .bx-compare th { text-transform: none; letter-spacing: 0; font-size: 12px; color: var(--text-2); line-height: 1.3; }
.view-build .bx-compare th:first-child, .view-build .bx-compare td:first-child {
  position: sticky; left: 0; z-index: 1; background: var(--card); width: 27%; text-align: left; color: var(--text-2); font-weight: 600;
}
.view-build .bx-compare td b { display: block; color: var(--text); }
.view-build .bx-compare td small { display: block; color: var(--text-3); font-size: 11.5px; }
.view-build .bx-compare .on { background: var(--accent-soft); }
.view-build .bx-compare td.on small { color: var(--text-2); } /* --text-3 on the green tint is 4.3:1 in dark */
.view-build .bx-compare th.on { color: var(--text); }
.view-build .bx-compare .tag, .view-build .bx-compare .badge { margin-top: 4px; display: inline-flex; white-space: nowrap; }
.view-build .bx-compare .badge { font-size: 11px; padding: 1px 6px; gap: 3px; }
.view-build .bx-strategy { margin-top: 14px; }
.view-build .bx-strategy .seg { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.view-build .bx-rule { margin-top: 10px; font-size: 14px; }
.view-build .bx-chart { margin: 8px 0 0; position: relative; touch-action: pan-y; }
.view-build .bx-chart svg.g-plot { display: block; width: 100%; height: auto; overflow: visible; }
.view-build .bx-chart .g-grid line { stroke: var(--line); stroke-width: 1; }
.view-build .bx-chart .g-axis { stroke: var(--line-strong); stroke-width: 1; }
.view-build .bx-chart text { fill: var(--text-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.view-build .bx-chart .g-end { fill: var(--text-2); font-size: 11.5px; font-weight: 650; }
.view-build .bx-chart .s { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.view-build .bx-chart .s1 { stroke: var(--bx-c1); }
.view-build .bx-chart .s2 { stroke: var(--bx-c2); stroke-dasharray: 6 4; }
.view-build .bx-chart .s3 { stroke: var(--bx-c3); stroke-dasharray: 2 4; }
.view-build .bx-chart .s4 { stroke: var(--bx-c4); stroke-dasharray: 10 3 2 3; }
.view-build .bx-chart .dot { stroke: var(--card); stroke-width: 2; }
.view-build .bx-chart .d1 { fill: var(--bx-c1); }
.view-build .bx-chart .d2 { fill: var(--bx-c2); }
.view-build .bx-chart .d3 { fill: var(--bx-c3); }
.view-build .bx-chart .d4 { fill: var(--bx-c4); }
.view-build .bx-chart .g-cross { stroke: var(--text-3); stroke-width: 1; }
.view-build .bx-chart svg.g-plot:focus { outline: none; }
.view-build .bx-chart svg.g-plot:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 6px; }
.view-build .bx-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; color: var(--text-2); margin-top: 8px; }
.view-build .bx-legend span { display: inline-flex; align-items: center; gap: 6px; }
.view-build .bx-legend svg { flex: none; }
.view-build .bx-tip {
  position: absolute; top: 0; pointer-events: none; background: var(--card); border: 1px solid var(--line-strong); border-radius: 10px;
  box-shadow: var(--shadow-sm); padding: 6px 9px; font-size: 12.5px; min-width: 132px; z-index: 2;
}
.view-build .bx-tip .t-day { color: var(--text-2); font-size: 12px; margin-bottom: 2px; }
.view-build .bx-tip div { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.view-build .bx-tip b { font-variant-numeric: tabular-nums; }
.view-build .bx-tip span { color: var(--text-2); }
@media (min-width: 900px) {
  .view-build .bx-tiles.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .view-build .bx-tickets { grid-template-columns: repeat(auto-fit, minmax(0, 1fr)); align-items: start; }
  .view-build .bx-actions { grid-template-columns: 2fr 1fr 1fr 1fr; }
  .view-build .bx-actions .bx-main { grid-column: auto; }
  .view-build .bx-compare table { font-size: 13.5px; }
  .view-build .bx-compare th, .view-build .bx-compare td { padding: 8px 8px; }
  .view-build .bx-strategy .seg { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
@media (max-width: 360px) {
  .view-build .bx-two { grid-template-columns: minmax(0, 1fr); }
}
`;

/** Add the Build screen's styles to the page (once). */
export function ensureStyles() {
  if (document.getElementById("build-styles")) return;
  const style = document.createElement("style");
  style.id = "build-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
