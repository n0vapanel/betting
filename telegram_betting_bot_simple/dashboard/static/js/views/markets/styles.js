// Styles for "Markets say" on the game page, the typed factor list, the explainer sheet and the AI record's
// Markets tables. Added once, when one of them is first drawn (a module in the page's import map, so no extra
// request). Every rule is scoped to .mk-say, .fx*, .mk-sheet or .mk-record-box / .mk-sport. Colours are the
// shared tokens (Telegram's theme flows through them); meaning is never carried by colour alone: statuses have
// words, deltas have signs, "vs blend" says better or worse.

const CSS = `
/* --- "Markets say" ---------------------------------------------------------------------------------------- */
.mk-say .section-head { flex-wrap: wrap; row-gap: 2px; }
.mk-headline { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
.mk-big { display: flex; align-items: baseline; gap: 8px; min-width: 0; margin: 0; }
.mk-big b { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.mk-who { font-size: 16px; font-weight: 700; min-width: 0; overflow-wrap: anywhere; }
.mk-others { margin: 0; color: var(--text-2); font-size: 13.5px; font-variant-numeric: tabular-nums; }
.mk-badges { flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.mk-badges .badge { white-space: normal; }
.mk-dis { gap: 4px; }
.mk-low-note { margin-top: 6px; color: var(--text-2); }
.mk-caption { margin: 12px 0 6px; font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-3); }
.mk-chips { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 8px; }
.mk-chips > li { min-width: 0; display: grid; }
.mk-chip {
  position: relative; display: grid; gap: 1px; align-content: start; min-height: 64px; min-width: 0; padding: 8px 10px; border-radius: 12px;
  background: var(--card-2); border: 1px solid var(--line); color: var(--text);
}
.mk-chip.link { border-color: color-mix(in srgb, var(--info) 40%, transparent); }
.mk-chip.link:hover { text-decoration: none; background: var(--info-soft); }
.mk-chip.off { border-style: dashed; }
.mk-chip.off .mk-chip-p { color: var(--text-2); }
.mk-chip-n { display: flex; align-items: center; gap: 4px; min-width: 0; font-size: 12.5px; font-weight: 650; color: var(--text-2); overflow-wrap: anywhere; }
.mk-chip.link .mk-chip-n { color: var(--info-ink); }
.mk-chip-n .icon { flex: none; }
.mk-chip-p { font-size: 20px; font-weight: 800; line-height: 1.2; }
.mk-chip-t { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; min-width: 0; font-size: 11.5px; color: var(--text-2); }
.mk-chip-s { font-size: 10.5px; font-weight: 750; letter-spacing: 0.03em; text-transform: uppercase; padding: 0 5px; border-radius: 999px; border: 1px solid var(--line-strong); }
.mk-chip-s.thin, .mk-chip-s.stale { color: var(--warn-ink); border-color: color-mix(in srgb, var(--warn) 45%, transparent); background: var(--warn-soft); }
.mk-links { margin-top: 8px; color: var(--text-2); }
.mk-links a { display: inline-flex; align-items: center; gap: 2px; min-height: var(--tap); }
.mk-how { display: flex; margin-top: 2px; }
.mk-more { margin-top: 10px; }
.mk-more > summary {
  list-style: none; cursor: pointer; min-height: var(--tap); display: flex; align-items: center; gap: 6px; font-weight: 650; font-size: 14px; color: var(--info);
}
.mk-more > summary::-webkit-details-marker { display: none; }
.mk-more > summary .icon { transition: transform var(--speed); }
.mk-more[open] > summary .icon { transform: rotate(180deg); }
.mk-table th[scope="row"] { text-align: left; font-weight: 650; color: var(--text); text-transform: none; letter-spacing: 0; font-size: 13.5px; }
.mk-td-note { display: block; color: var(--text-2); font-size: 12px; font-weight: 500; white-space: normal; text-align: left; min-width: 10em; }
td.mk-td-note { display: table-cell; }

/* --- the explainer sheet ------------------------------------------------------------------------------------ */
.mk-sheet .mk-h, .mk-record-box .mk-h { font-size: 15px; font-weight: 750; margin: 16px 0 6px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; }
.mk-sheet p { margin: 0 0 8px; line-height: 1.5; }
.mk-sheet .mk-weights th[scope="row"] { white-space: normal; }
.mk-sheet .mk-weights .mk-td-note { min-width: 0; }

/* --- the factor list ---------------------------------------------------------------------------------------- */
.fx-pick { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.fx-pick .chip { min-height: 36px; }
@media (pointer: coarse) { .fx-pick .chip { min-height: 44px; } }
.fx-list { list-style: none; margin: 0; padding: 0; display: grid; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.fx-row {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(32px, 22%) auto; align-items: center; gap: 4px 10px;
  padding: 8px 12px; border-top: 1px solid var(--line); min-width: 0;
}
.fx-row:first-child { border-top: 0; }
.fx-row.base { background: var(--card-2); }
.fx-row.total { background: var(--card-2); border-top: 1px solid var(--line-strong); }
.fx-row.total .fx-name { font-weight: 750; }
.fx-l { display: grid; gap: 1px; min-width: 0; }
.fx-name { font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
.fx-meta { font-size: 12px; color: var(--text-2); overflow-wrap: anywhere; }
.fx-noisy { display: inline-block; font-size: 10.5px; font-weight: 750; letter-spacing: 0.03em; text-transform: uppercase; color: var(--text-2); border: 1px dashed var(--line-strong); border-radius: 999px; padding: 0 5px; vertical-align: 1px; }
.fx-row.noisy .fx-bar i { opacity: 0.55; }
.fx-bar { display: block; height: 8px; border-radius: 999px; background: transparent; min-width: 0; }
.fx-bar i { display: block; height: 100%; border-radius: 999px; background: var(--card-3); }
.fx-bar i.up { background: var(--accent); }
.fx-bar i.down { background: var(--danger); }
.fx-d { font-size: 14px; font-weight: 750; white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
.fx-d.up { color: var(--good-ink); }
.fx-d.down { color: var(--danger-ink); }
@media (max-width: 400px) { .fx-row { grid-template-columns: minmax(0, 1fr) auto; } .fx-bar { display: none; } }

/* --- the AI record's Markets tables ------------------------------------------------------------------------ */
.mk-record-box .mk-sport + .mk-sport { margin-top: 6px; }
.mk-record td, .mk-record th { padding-top: 7px; padding-bottom: 7px; }
.mk-record th[scope="row"] small, .mk-th-sub { display: block; font-size: 11px; color: var(--text-2); font-weight: 500; text-transform: none; letter-spacing: 0; }
.mk-blend-row th, .mk-blend-row td { font-weight: 750; background: var(--card-2); }
.mk-record { font-size: 13px; }
.mk-record th, .mk-record td { padding-left: 4px; padding-right: 4px; }
.mk-record th:last-child, .mk-record td:last-child { padding-right: 0; }
.mk-record th[scope="row"] { white-space: normal; overflow-wrap: break-word; }
.mk-sc > span { display: block; white-space: nowrap; font-variant-numeric: tabular-nums; }
.mk-score { font-weight: 650; }
.mk-sc small { color: var(--text-2); font-size: 11px; font-weight: 500; }
.mk-sc [class^="mk-vs-"] { font-size: 11.5px; }
.mk-sc small { font-size: 10.5px; }
.mk-vs-better { color: var(--good-ink); }
.mk-sc .mk-vs-base { font-size: 10.5px; color: var(--text-2); }
.mk-record .mk-of { display: block; font-size: 10.5px; color: var(--text-2); font-weight: 500; white-space: nowrap; }
.mk-vs-worse { color: var(--text); }
.mk-empty { color: var(--text-2); font-size: 14px; margin-top: 8px; }

/* --- the soccer "Goals model" panel (markets/goals.js) ------------------------------------------------------ */
.gm-xg { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.gm-xg > div { display: grid; gap: 2px; padding: 8px 10px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); min-width: 0; }
.gm-xg b { font-size: 22px; font-weight: 800; line-height: 1.1; }
.gm-xg span { font-size: 13px; color: var(--text-2); overflow-wrap: anywhere; }
.gm-goals .gm-lead { font-size: 12.5px; color: var(--text-2); margin-top: 6px; }
.gm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); gap: 10px; margin-top: 12px; }
.gm-block { min-width: 0; border: 1px solid var(--line); border-radius: 12px; background: var(--card-2); padding: 8px 10px; }
.gm-table { width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; }
.gm-table thead th:first-child { width: 42%; }
.gm-tt thead th:first-child { width: 22%; }
.gm-table caption { text-align: left; font-weight: 700; font-size: 13.5px; padding: 0 0 6px; color: var(--text); }
.gm-table th, .gm-table td {
  padding: 5px 4px; border-top: 1px solid var(--line); border-bottom: 0; text-align: right; vertical-align: top; white-space: normal;
  text-transform: none; letter-spacing: 0;
}
.gm-table thead th { border-top: 0; font-size: 11.5px; font-weight: 650; color: var(--text-2); }
.gm-table thead th abbr { text-decoration: none; }
.gm-table th[scope="row"] { text-align: left; font-weight: 600; font-size: 13px; color: var(--text); overflow-wrap: anywhere; padding-left: 0; }
.gm-table td { font-variant-numeric: tabular-nums; }
.gm-table th[scope="row"] small { display: block; font-weight: 500; font-size: 11.5px; color: var(--text-2); }
.gm-table thead th:first-child { text-align: left; }
.gm-table td b { font-weight: 750; }
.gm-table td small { color: var(--text-2); }
.gm-tt td b { margin-left: 4px; }
.gm-foot { font-size: 11.5px; color: var(--text-2); margin-top: 4px; }
`;

export function ensureMarketStyles() {
  if (typeof document === "undefined" || document.getElementById("markets-styles")) return;
  const style = document.createElement("style");
  style.id = "markets-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
