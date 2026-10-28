// Styles only the Brain progress screen uses (My bets → AI record → Brain progress), added once when it first opens.
// Every rule is scoped to .bp so nothing here can change another screen. Colours are the shared tokens (so Telegram's
// own theme carries through), except the brain's one chart hue, --bp-c1: slot-1 blue, #2a78d6 on light cards and
// #3987e5 on dark ones (checked with the dataviz validator against #ffffff, #121a2b, Telegram's #2c2c2e and #000000:
// lightness, chroma and 3:1 contrast all pass). Status always carries a word or an icon as well as a colour.

const CSS = `
.bp { --bp-c1: #3987e5; }
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) .bp { --bp-c1: #2a78d6; } }
:root[data-theme="light"] .bp { --bp-c1: #2a78d6; }

.bp .bp-intro { display: grid; gap: 8px; }
.bp .bp-intro p { color: var(--text-2); font-size: 13.5px; }
.bp .bp-status { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.bp .bp-status .grow { flex: 1 1 160px; min-width: 0; font-size: 13.5px; color: var(--text-2); }
.bp .bp-status .seg { flex: 0 0 auto; }
.bp .bp-status .seg button { min-width: 84px; }

.bp .bp-heads { list-style: none; display: grid; gap: 8px; }
.bp .bp-head { display: grid; gap: 4px; width: 100%; text-align: left; padding: 10px 12px; min-height: var(--tap); border-radius: 12px;
  background: var(--card-2); border: 1px solid var(--line); color: var(--text); }
.bp button.bp-head:hover { border-color: var(--line-strong); }
.bp .bp-head[aria-pressed="true"] { border-color: var(--info); box-shadow: inset 0 0 0 1px var(--info); }
.bp .bp-head-top { display: flex; align-items: baseline; justify-content: space-between; gap: 2px 8px; min-width: 0; flex-wrap: wrap; }
.bp .bp-head-top b { font-size: 15px; }
.bp .bp-head p { font-size: 13.5px; line-height: 1.45; overflow-wrap: anywhere; }
.bp .bp-head .bp-ver { font-size: 12.5px; color: var(--text-2); min-width: 0; overflow-wrap: anywhere; text-align: right; }

.bp .bp-sports { overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; margin: 0 calc(-1 * var(--gutter)); padding: 0 var(--gutter); }
.bp .bp-sports::-webkit-scrollbar { display: none; }
.bp .bp-sports .seg { width: max-content; min-width: 100%; }
.bp .bp-sports .seg button { flex: 1 0 auto; padding: 0 12px; }
.bp .bp-sport-box { display: grid; gap: 0; }

.bp .bp-chart { margin: 4px 0 0; position: relative; touch-action: pan-y; min-width: 0; }
.bp .bp-chart svg.g-plot { display: block; width: 100%; height: auto; overflow: visible; }
.bp .bp-chart.bp-cal svg.g-plot { max-width: 520px; }
.bp .bp-chart.bp-pit svg.g-plot { max-width: 560px; }
.bp .bp-chart .g-grid line { stroke: var(--line); stroke-width: 1; }
.bp .bp-chart .g-axis { stroke: var(--line-strong); stroke-width: 1; }
.bp .bp-chart .g-zero, .bp .bp-chart .g-diag { stroke: var(--text-3); stroke-width: 1; }
.bp .bp-legend .g-diag { stroke: var(--text-3); stroke-width: 1.5; }
.bp .bp-chart text { fill: var(--text-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.bp .bp-chart .g-title { fill: var(--text-2); font-size: 11.5px; font-weight: 600; }
.bp .bp-chart .g-zero-t { fill: var(--text-2); font-size: 11px; }
.bp .bp-chart .g-end { fill: var(--text); font-size: 11.5px; font-weight: 650; }
.bp .g-line { fill: none; stroke: var(--bp-c1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.bp .g-band { fill: var(--bp-c1); fill-opacity: 0.14; stroke: none; }
.bp .g-dot { fill: var(--bp-c1); stroke: var(--card); stroke-width: 2; }
.bp .g-bar { fill: var(--bp-c1); }
.bp .g-bar.on { fill-opacity: 0.75; }
.bp .g-mark { stroke: var(--text-3); stroke-width: 1.5; }
.bp .g-mark.back { stroke-dasharray: 4 3; }
.bp .bp-chart .g-cross { stroke: var(--text-3); stroke-width: 1; }
.bp .bp-chart svg.g-plot:focus { outline: none; }
.bp .bp-chart svg.g-plot:focus-visible { outline: 2px solid var(--info); outline-offset: 4px; border-radius: 6px; }
.bp .bp-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; color: var(--text-2); margin: 2px 0 6px; }
.bp .bp-legend span { display: inline-flex; align-items: center; gap: 6px; }
.bp .bp-axis-y { font-size: 12.5px; color: var(--text-2); margin: 4px 0 2px; }
.bp .bp-tip { position: absolute; z-index: 2; min-width: 150px; max-width: min(280px, 86vw); padding: 8px 10px; border-radius: 10px; background: var(--card);
  border: 1px solid var(--line-strong); box-shadow: var(--shadow-sm); font-size: 12.5px; pointer-events: none; display: grid; gap: 3px; }
.bp .bp-tip .t-head { color: var(--text-2); font-weight: 600; }
.bp .bp-tip b { margin-right: 6px; font-size: 13.5px; }
.bp .bp-tip span { color: var(--text-2); }
.bp .bp-tip .t-note { color: var(--text); border-top: 1px solid var(--line); padding-top: 3px; overflow-wrap: anywhere; }
.bp .bp-tip .t-foot { color: var(--text-3); font-size: 11.5px; }
.bp details.bp-more > summary { list-style: none; cursor: pointer; min-height: var(--tap); display: flex; align-items: center; gap: 6px;
  font-weight: 600; font-size: 13.5px; color: var(--info); }
.bp details.bp-more > summary::-webkit-details-marker { display: none; }
.bp details.bp-more table { font-size: 13px; }
.bp details.bp-older[open] > summary .icon { transform: rotate(180deg); }
.bp details.bp-older > .bp-list { margin-top: 4px; }

.bp .bp-words { font-size: 13.5px; line-height: 1.5; margin-top: 6px; }
.bp .bp-sub { font-size: 14px; font-weight: 700; margin: 14px 0 4px; }
.bp .bp-nums { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 8px; margin: 8px 0 0; }
.bp .bp-nums > div { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; min-width: 0; }
.bp .bp-nums dt { color: var(--text-2); font-size: 12px; }
.bp .bp-nums dd { margin: 0; font-weight: 700; font-size: 14px; overflow-wrap: anywhere; }
.bp .bp-nums dd small { display: block; font-weight: 500; font-size: 12px; color: var(--text-2); }

.bp .bp-list { list-style: none; display: grid; gap: 10px; }
.bp .bp-item { border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; display: grid; gap: 6px; min-width: 0; background: var(--card); }
.bp .bp-item.flash { animation: bp-flash 1.6s ease-out 1; }
@keyframes bp-flash { 0% { box-shadow: 0 0 0 3px var(--info); } 100% { box-shadow: 0 0 0 3px transparent; } }
@media (prefers-reduced-motion: reduce) { .bp .bp-item.flash { animation: none; box-shadow: 0 0 0 2px var(--info); } }
.bp .bp-item-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; min-width: 0; }
.bp .bp-item-top h3 { font-size: 15px; font-weight: 700; line-height: 1.3; min-width: 0; overflow-wrap: anywhere; }
.bp .bp-item-top .badge { flex: none; }
.bp .bp-item .bp-move { font-size: 15px; font-weight: 700; }
.bp .bp-item .bp-meta { font-size: 12.5px; color: var(--text-2); overflow-wrap: anywhere; }
.bp .bp-item p { font-size: 13.5px; line-height: 1.5; overflow-wrap: anywhere; }
.bp .bp-by { display: block; font-size: 12px; color: var(--text-3); overflow-wrap: anywhere; }
.bp .bp-back { font-size: 13px; padding: 7px 10px; border-radius: 10px; background: var(--warn-soft); border: 1px solid color-mix(in srgb, var(--warn) 40%, transparent); }
.bp .bp-back.owner { background: var(--card-2); border-color: var(--line); }
.bp .bp-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.bp .bp-actions .btn { padding: 0 14px; }
.bp .bp-progress { display: grid; gap: 4px; font-size: 13px; }
.bp .bp-progress .meter > span { background: var(--bp-c1); }
.bp .bp-progress .row-between span { color: var(--text-2); }
.bp .bp-note { white-space: pre-line; font-size: 14px; line-height: 1.55; padding: 10px 12px; border-radius: 10px; background: var(--card-2); border: 1px solid var(--line); overflow-wrap: anywhere; }
.bp .bp-kv { list-style: none; display: grid; gap: 0; }
.bp .bp-kv li { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; padding: 8px 0; border-top: 1px solid var(--line); font-size: 13.5px; min-width: 0; }
.bp .bp-kv li:first-child { border-top: 0; }
.bp .bp-kv li > span:first-child { color: var(--text-2); min-width: 0; flex: 1 1 auto; }
.bp .bp-kv li > b { text-align: right; overflow-wrap: anywhere; min-width: 0; flex: 0 1 auto; max-width: 72%; }
.bp .bp-empty { color: var(--text-2); font-size: 13.5px; line-height: 1.5; }
.bp .bp-held-sum { font-size: 14px; font-weight: 650; }
.bp .bp-cell-note { display: block; font-size: 11.5px; font-weight: 500; color: var(--text-2); text-transform: none; letter-spacing: 0; white-space: normal; }
.bp .bp-hyp { margin: 8px 0 0 18px; display: grid; gap: 4px; font-size: 13.5px; }
.bp tbody th[scope="row"] { min-width: 120px; text-transform: none; letter-spacing: 0; font-size: 13.5px; font-weight: 600; color: var(--text); white-space: normal; }
.bp .bp-two { display: grid; gap: 12px; }
@media (min-width: 1000px) {
  .bp .bp-two { grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }
}
`;

export function ensureBrainStyles() {
  if (document.getElementById("brain-styles")) return;
  const style = document.createElement("style");
  style.id = "brain-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
