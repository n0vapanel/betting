// Styles only My bets uses. styles.css holds the shared design system (tokens, cards, chips, sheets,
// badges); these layout rules are added once, when My bets first opens, and every one is scoped to
// .view-bets (or to the screen's own sheets, .mb-sheet) so nothing here can change another screen.
// Colours are the shared tokens; results always carry a word or an icon as well as a colour.

const CSS = `
.view-bets .mb-tabs { overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; margin: 0 calc(-1 * var(--gutter)); padding: 0 var(--gutter); }
.view-bets .mb-tabs::-webkit-scrollbar { display: none; }
.view-bets .mb-tabs .seg { width: max-content; min-width: 100%; }
.view-bets .mb-tabs .seg button { flex: 1 0 auto; padding: 0 12px; }

.view-bets .mb-results-head { flex-wrap: wrap; row-gap: 8px; }
.view-bets .mb-results-head h2 { font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 4px; }
/* 36 px buttons; the shared ::after reaches 4 px above and below, so each period is a 44 px tap target */
.view-bets .mb-results .seg button { min-height: 36px; font-size: 13px; padding: 0 10px; }
.view-bets .mb-net { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 10px; margin-top: 10px; }
.view-bets .mb-net-v { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.view-bets .mb-net-v.up { color: var(--accent); }
.view-bets .mb-net-v.down { color: var(--danger); }
.view-bets .mb-kpis { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.view-bets .mb-kpis > div { background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 8px 10px; min-width: 0; }
.view-bets .mb-kpis b { display: block; font-size: 17px; font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.view-bets .mb-kpis span { display: flex; align-items: center; gap: 2px; color: var(--text-2); font-size: 12px; }
.view-bets .mb-lines { display: grid; gap: 4px; margin-top: 10px; font-size: 13px; color: var(--text-2); }
.view-bets .mb-lines b { color: var(--text); }

.view-bets .mb-slip { display: grid; gap: 10px; }
.view-bets .mb-head { display: flex; align-items: flex-start; gap: 10px; min-width: 0; }
.view-bets .mb-head h3 { font-size: 16px; font-weight: 700; line-height: 1.3; }
.view-bets .mb-head .muted { display: block; font-size: 12.5px; }
.view-bets .mb-head .badge { flex: none; margin-top: 1px; }
.view-bets .mb-badges { display: flex; flex-wrap: wrap; gap: 6px; }
.view-bets .mb-result { font-size: 14px; padding: 8px 10px; border-radius: 10px; background: var(--card-2); border: 1px solid var(--line); }
.view-bets .mb-result b { font-weight: 750; }
.view-bets .mb-legs { list-style: none; display: grid; gap: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.view-bets .mb-leg { padding: 9px 12px; border-top: 1px solid var(--line); display: grid; gap: 2px; min-width: 0; }
.view-bets .mb-leg:first-child { border-top: 0; }
.view-bets .mb-leg .row-between { align-items: flex-start; }
.view-bets .mb-leg-label { min-width: 0; overflow-wrap: anywhere; }
.view-bets .mb-leg-game { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.view-bets .mb-mark { display: inline-flex; align-items: center; gap: 3px; font-weight: 650; font-size: 12.5px; white-space: nowrap; flex: none; }
.view-bets .mb-mark.good { color: var(--accent); }
.view-bets .mb-mark.bad { color: var(--danger); }
.view-bets .mb-mark.warn { color: var(--warn); }
.view-bets .mb-mark.muted { color: var(--text-2); }
.view-bets .mb-nums, .mb-sheet .mb-nums { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; }
.view-bets .mb-nums > div, .mb-sheet .mb-nums > div { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; min-width: 0; }
.view-bets .mb-nums dt, .mb-sheet .mb-nums dt { color: var(--text-2); font-size: 12px; display: flex; align-items: center; gap: 2px; min-height: 20px; }
.view-bets .mb-nums dd, .mb-sheet .mb-nums dd { margin: 0; font-weight: 700; font-size: 14px; overflow-wrap: anywhere; }
.view-bets .mb-nums dd .muted, .mb-sheet .mb-nums dd .muted { font-weight: 500; font-size: 12px; }
.view-bets details.mb-more > summary, .mb-sheet details.mb-more > summary {
  list-style: none; cursor: pointer; min-height: var(--tap); display: flex; align-items: center; gap: 6px; font-weight: 650; font-size: 14px; color: var(--info);
}
.view-bets details.mb-more > summary::-webkit-details-marker { display: none; }
.view-bets details.mb-more > summary .icon { transition: transform var(--speed); }
.view-bets details.mb-more[open] > summary .icon { transform: rotate(180deg); }
.view-bets details.mb-more > div { padding-top: 4px; }
.view-bets .mb-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.view-bets .mb-actions .grow { flex: 1; }

.view-bets .mb-plan { display: grid; gap: 12px; scroll-margin-top: calc(var(--header-h) + var(--safe-top) + 12px); }
.view-bets .mb-days { list-style: none; display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0; }
.view-bets .mb-day {
  width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; font-size: 12px; font-weight: 700;
  background: var(--card-2); border: 1px solid var(--line); color: var(--text-3);
}
.view-bets .mb-day.won { background: var(--accent-soft); border-color: var(--accent-line); color: var(--accent); }
.view-bets .mb-day.lost { background: var(--danger-soft); border-color: color-mix(in srgb, var(--danger) 45%, transparent); color: var(--danger); }
.view-bets .mb-day.void, .view-bets .mb-day.partial { background: var(--card-3); color: var(--text-2); }
.view-bets .mb-day.live { background: var(--warn-soft); border-color: color-mix(in srgb, var(--warn) 45%, transparent); color: var(--warn); }
.view-bets .mb-day.next { border: 2px dashed var(--info); color: var(--info); background: var(--info-soft); }
.view-bets .mb-day.skip { opacity: 0.45; }
.view-bets .mb-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.view-bets .mb-tiles > div { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; min-width: 0; }
.view-bets .mb-tiles span { display: block; color: var(--text-2); font-size: 12px; }
.view-bets .mb-tiles b { display: block; font-size: 15px; font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.view-bets .mb-plan .notice { margin-top: 0; }
.view-bets .mb-daylist { list-style: none; display: grid; gap: 0; margin: 0; padding: 0; }
.view-bets .mb-daylist li { display: grid; grid-template-columns: 54px minmax(0, 1fr); gap: 10px; padding: 9px 0; border-top: 1px solid var(--line); font-size: 13.5px; }
.view-bets .mb-daylist li:first-child { border-top: 0; }
.view-bets .mb-daylist li > b { font-size: 13px; }
.view-bets .mb-daylist .muted { font-size: 12.5px; }
.view-bets .mb-compare { list-style: none; display: grid; gap: 6px; margin: 8px 0 0; padding: 0; font-size: 13px; }
.view-bets .mb-compare li { display: flex; justify-content: space-between; gap: 10px; padding: 7px 10px; border-radius: 10px; background: var(--card-2); }
.view-bets .mb-compare li.mine { outline: 1px solid var(--accent-line); }
.view-bets .mb-compare li span:last-child { text-align: right; color: var(--text-2); }

.mb-sheet .mb-picks { list-style: none; display: grid; gap: 8px; margin: 0; padding: 0; }
.mb-sheet .mb-picks .check { align-items: center; }
.mb-sheet .mb-picks .check .grow { display: grid; gap: 1px; }
.mb-sheet .mb-picks .check .muted { font-size: 12.5px; }
.mb-sheet .mb-picks .check.off { opacity: 0.55; }
.mb-sheet .mb-priced { margin-top: 12px; }
.mb-sheet .mb-code {
  margin: 14px auto 8px; padding: 12px 16px; border-radius: 14px; background: var(--card-2); border: 1px dashed var(--line-strong);
  font: 800 26px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: 0.14em; text-align: center; user-select: all;
}
.mb-sheet .warnings { margin-top: 10px; }

.view-bets .mb-statgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.view-bets .mb-statgrid .stat { padding: 10px 12px; }
.view-bets .mb-statgrid .stat-v { font-size: 19px; }
.view-bets .mb-brier { display: grid; gap: 8px; margin-top: 12px; }
.view-bets .mb-brier-row { display: grid; grid-template-columns: 112px minmax(0, 1fr) 46px; gap: 8px; align-items: center; font-size: 13px; }
.view-bets .mb-brier-row b { text-align: right; }
.view-bets .mb-track { height: 10px; border-radius: 999px; background: var(--card-3); overflow: hidden; }
.view-bets .mb-track > span { display: block; height: 100%; border-radius: 999px; background: var(--info); }
.view-bets .mb-track.ai > span { background: var(--accent); }
.view-bets .mb-track.base > span { background: var(--text-3); }
.view-bets .mb-calib { list-style: none; display: grid; gap: 10px; margin: 0; padding: 0; }
.view-bets .mb-calib li { display: grid; gap: 4px; }
.view-bets .mb-calib .row-between { font-size: 13px; }
.view-bets .mb-pair { display: grid; gap: 3px; }
.view-bets .mb-pair .mb-track { height: 8px; }
.view-bets .mb-tier { display: grid; grid-template-columns: 36px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 0; border-top: 1px solid var(--line); }
.view-bets .mb-tier:first-child { border-top: 0; }
.view-bets .mb-tier-icon { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; background: var(--card-2); }
.view-bets .mb-tier.safe .mb-tier-icon { color: var(--accent); }
.view-bets .mb-tier.risky .mb-tier-icon { color: var(--warn); }
.view-bets .mb-tier.crazy .mb-tier-icon { color: var(--violet); }
.view-bets .mb-tier .small { color: var(--text-2); }
.view-bets .mb-tier > b { text-align: right; font-size: 15px; }
.view-bets .mb-tier > b small { display: block; font-size: 11.5px; font-weight: 600; color: var(--text-2); }
.view-bets .mb-analyst { padding: 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); display: grid; gap: 8px; }
.view-bets .mb-analyst .mb-nums > div { background: var(--card); }
.view-bets .mb-games { list-style: none; display: grid; margin: 0; padding: 0; }
.view-bets .mb-games li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 10px; padding: 9px 0; border-top: 1px solid var(--line); font-size: 13.5px; }
.view-bets .mb-games li:first-child { border-top: 0; }
.view-bets .mb-games .muted { font-size: 12.5px; }
.view-bets .mb-ranks { list-style: none; display: grid; margin: 0; padding: 0; }
.view-bets .mb-rank { display: grid; grid-template-columns: 34px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 0; border-top: 1px solid var(--line); }
.view-bets .mb-rank:first-child { border-top: 0; }
.view-bets .mb-rank.me { background: var(--accent-soft); border-radius: 10px; padding-left: 8px; padding-right: 8px; margin: 2px -8px; border-top-color: transparent; }
.view-bets .mb-rank > span:first-child { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; background: var(--card-2); font-weight: 750; font-size: 13px; }
.view-bets .mb-rank .muted { font-size: 12.5px; display: block; }
.view-bets .mb-rank > b { text-align: right; }
.view-bets .mb-rank > b small { display: block; font-size: 11.5px; font-weight: 600; color: var(--text-2); }

@media (max-width: 420px) {
  .view-bets .mb-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 359px) {
  .view-bets .mb-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .view-bets .mb-nums { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) {
  .view-bets details.mb-more > summary .icon { transition: none; }
}
`;

/** Add My bets' styles to the page (once). */
export function ensureStyles() {
  if (document.getElementById("bets-styles")) return;
  const style = document.createElement("style");
  style.id = "bets-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
