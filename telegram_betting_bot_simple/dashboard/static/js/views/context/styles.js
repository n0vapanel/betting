// Styles for Venue & conditions (v3 Stage 4): the context badges (board cell, phone cards, slip and builder legs),
// the Venue & conditions panel and its phone one-liner and sheet, the factor waterfall ("Why the line is here", the
// NBA model's "Why the numbers") and "What moves it". Added once, when one of them is first drawn (a module in the
// page's import map, so no extra request). Rules are scoped to .cx-*, .wf-*, .mv-* and the .fx-* list the waterfall
// shares with the old factor list. Colours are the shared tokens (Telegram's theme flows through them); meaning is
// never carried by colour alone: chips have words in their label, deltas have signs, "small, noisy" and
// "untested" are written out.

const CSS = `
/* --- badges ------------------------------------------------------------------------------------------------- */
.cx-badges { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px; min-width: 0; max-width: 100%; }
.cx-b {
  display: inline-flex; align-items: center; gap: 3px; flex: none; max-width: 100%; min-height: 20px; padding: 0 6px; border-radius: 999px;
  border: 1px solid var(--line-strong); background: var(--card-2); color: var(--text-2); font-size: 11.5px; font-weight: 650; line-height: 1;
  white-space: nowrap; font-variant-numeric: tabular-nums;
}
.cx-b .icon { flex: none; }
.cx-b.cx-adj { color: var(--info-ink); border-color: color-mix(in srgb, var(--info) 45%, transparent); background: var(--info-soft); }
.cx-b.cx-wind .icon, .cx-b.cx-rain .icon, .cx-b.cx-cold .icon { color: var(--info-ink); }
.cx-b.cx-hot .icon { color: var(--warn-ink); }
.cx-s { display: none; }
.cx-l { overflow: hidden; text-overflow: ellipsis; }
/* The board's Context cell: short text ("16", "1.6k"); the long text ("16 mph", "1,580 m") from 1,180 px */
.cxv-cell { display: flex; flex-direction: column; align-items: flex-start; gap: 3px; }
.cxv-cell .cx-b { min-height: 18px; padding: 0 5px; font-size: 11px; gap: 2px; }
.cxv-cell .cx-s { display: inline; }
.cxv-cell .cx-l { display: none; }
@container today (min-width: 1180px) {
  .cxv-cell .cx-s { display: none; }
  .cxv-cell .cx-l { display: inline; }
}
/* Under the time on a narrow board (no Context column): short text, a little smaller */
.cxv-in { gap: 3px; }
.cxv-in .cx-b { min-height: 16px; padding: 0 4px; font-size: 10.5px; gap: 2px; }
.cxv-in .cx-s { display: inline; }
.cxv-in .cx-l { display: none; }
.cxv-leg .cx-b { font-size: 11px; min-height: 18px; }
.cxv-card { margin: 2px 14px 6px; display: flex; }
.cx-panel-badges { margin: 2px 0 8px; }
.cxv-panel .cx-b { min-height: 24px; font-size: 12.5px; padding: 0 8px; gap: 4px; }
.slip-ctx { margin-top: 4px; }

/* --- the Venue & conditions panel ------------------------------------------------------------------------- */
.cx-panel { min-width: 0; }
.cx-panel .section-head { flex-wrap: wrap; row-gap: 2px; }
.cx-where { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 10px; min-width: 0; }
.cx-where b { font-size: 16px; font-weight: 750; overflow-wrap: anywhere; }
.cx-where span { color: var(--text-2); font-size: 13.5px; }
.cx-map { display: inline-flex; align-items: center; gap: 4px; min-height: var(--tap, 44px); font-size: 13.5px; font-weight: 650; }
.cx-facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); gap: 6px 14px; margin: 6px 0 2px; font-size: 13.5px; }
.cx-facts > div { min-width: 0; }
.cx-facts dt { font-size: 11px; font-weight: 750; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-3); }
.cx-facts dd { margin: 0; overflow-wrap: anywhere; }
.cx-sub { margin: 14px 0 6px; font-size: 12px; font-weight: 750; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-3); }
.cx-trips { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.cx-trip { display: grid; gap: 2px; padding: 8px 10px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); min-width: 0; font-size: 13.5px; }
.cx-trip-head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; }
.cx-trip-head b { overflow-wrap: anywhere; }
.cx-trip-role { color: var(--text-2); font-size: 12.5px; }
.cx-trip-facts { color: var(--text-2); overflow-wrap: anywhere; }
.cx-now { margin: 0 0 8px; font-size: 14px; overflow-wrap: anywhere; }
.cx-hours { list-style: none; margin: 0; padding: 0 0 2px; display: grid; grid-auto-flow: column; grid-auto-columns: minmax(58px, 1fr); gap: 6px; overflow-x: auto; overscroll-behavior-x: contain; }
.cx-hour { display: grid; justify-items: center; gap: 2px; padding: 6px 4px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); font-size: 12.5px; font-variant-numeric: tabular-nums; min-width: 0; }
.cx-hour.on { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.cx-hour-t { font-weight: 700; color: var(--text-2); white-space: nowrap; }
.cx-hour.on .cx-hour-t { color: var(--text); }
.cx-hour-temp { font-size: 15px; font-weight: 750; }
.cx-hour-wind { display: inline-flex; align-items: center; gap: 2px; white-space: nowrap; }
.cx-arrow { display: inline-grid; place-items: center; color: var(--info-ink); }
.cx-arrow .icon { transition: none; }
.cx-hour-rain { color: var(--text-2); white-space: nowrap; }
.cx-announced { margin-top: 8px; font-size: 13.5px; padding: 8px 10px; border-radius: 12px; border: 1px dashed var(--line-strong); overflow-wrap: anywhere; }
.cx-note { margin-top: 8px; font-size: 12.5px; color: var(--text-2); overflow-wrap: anywhere; }
.cx-attr { display: inline-flex; align-items: center; gap: 4px; min-height: var(--tap, 44px); font-size: 12.5px; }
.cx-src { font-size: 12px; color: var(--text-3); margin-top: 6px; overflow-wrap: anywhere; }
.cx-facts .cx-src { display: block; margin-top: 2px; font-size: 11.5px; }
.cx-units { font-size: 12.5px; color: var(--text-2); }
.cx-units a { display: inline-flex; align-items: center; min-height: 32px; }
@media (pointer: coarse) { .cx-units a { min-height: 44px; } }
.cx-sheet .cx-units { display: block; margin-top: 8px; }
.cx-loading { color: var(--text-2); font-size: 13.5px; display: flex; align-items: center; gap: 8px; }
.cx-loading .skel { flex: 1 1 auto; height: 12px; border-radius: 6px; }

/* the phone one-liner (opens the full panel in a bottom sheet) */
.cx-line {
  display: flex; align-items: center; gap: 8px; width: 100%; min-height: 48px; padding: 8px 12px; margin: 12px 0 0; border-radius: var(--radius, 14px);
  border: 1px solid var(--line); background: var(--card); color: var(--text); text-align: left; font-size: 14px;
}
.cx-line:hover { background: var(--card-2); }
.cx-line > .icon:first-child { flex: none; color: var(--text-2); }
.cx-line-t { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cx-line-t b { font-weight: 700; }
.cx-line > .icon:last-child { flex: none; color: var(--text-3); }
.cx-sheet .cx-panel { border: 0; padding: 0; background: none; box-shadow: none; margin: 0; }
.cx-sheet .cx-panel > .section-head { display: none; }

/* the game page: the panel beside the teams on a wide page */
.cx-wrap { container: gamepage / inline-size; min-width: 0; }
.cx-top { display: grid; gap: 0; min-width: 0; }
.cx-top > * { min-width: 0; }
.cx-slot:empty, [data-cx]:empty { display: none; }
@container gamepage (min-width: 860px) {
  .cx-top.has-panel { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); column-gap: 14px; align-items: start; }
  .cx-top.has-panel .cx-slot > .section { margin-top: 0; }
}

/* --- the waterfall (shares .fx-* with the factor list) ---------------------------------------------------- */
.wf { margin-top: 2px; }
.wf + .wf { margin-top: 14px; }
.wf-h { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 8px; margin: 0 0 6px; font-size: 14.5px; font-weight: 750; }
.wf-h small { font-size: 12.5px; font-weight: 500; color: var(--text-2); }
.fx-bar.wf-track { position: relative; display: block; height: 10px; border-radius: 3px; background: color-mix(in srgb, var(--card-3) 45%, transparent); overflow: hidden; }
.fx-bar.wf-track i { position: absolute; top: 0; bottom: 0; height: auto; border-radius: 3px; background: var(--card-3); }
.fx-bar.wf-track i.up { background: var(--accent); }
.fx-bar.wf-track i.down { background: var(--danger); }
.fx-bar.wf-track i.pin { width: 3px; margin-left: -1px; background: var(--text-2); border-radius: 2px; }
.fx-row.noisy .fx-bar.wf-track i { opacity: 0.55; }
/* A phone: the bar gets a line of its own under the words, the full width of the row */
@media (max-width: 400px) {
  .wf .fx-row { grid-template-columns: minmax(0, 1fr) auto; }
  .wf .fx-row .fx-bar.wf-track { display: block; grid-column: 1 / -1; grid-row: 2; height: 8px; }
}
.fx-row.rest { background: color-mix(in srgb, var(--card-2) 55%, transparent); }
.fx-untested { display: inline-block; font-size: 10.5px; font-weight: 750; letter-spacing: 0.03em; text-transform: uppercase; color: var(--warn-ink); border: 1px dashed color-mix(in srgb, var(--warn) 55%, transparent); border-radius: 999px; padding: 0 5px; vertical-align: 1px; }
.wf-sum { margin-top: 6px; font-size: 12.5px; color: var(--text-2); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.wf-sum b { color: var(--text); }
.wf-tag { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 750; letter-spacing: 0.04em; text-transform: uppercase; padding: 1px 7px; border-radius: 999px; border: 1px solid var(--line-strong); color: var(--text-2); background: var(--card-2); white-space: nowrap; }
.wf-tag.moves { color: var(--info-ink); border-color: color-mix(in srgb, var(--info) 45%, transparent); background: var(--info-soft); }
.cx-why .section-head, .mv .section-head { flex-wrap: wrap; row-gap: 4px; }

/* --- "What moves it" ---------------------------------------------------------------------------------------- */
.mv-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.mv-item { display: grid; gap: 6px; padding: 10px 12px; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--info) 40%, transparent); background: var(--info-soft); min-width: 0; }
.mv-head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; font-weight: 700; }
.mv-head b { font-size: 16px; }
.mv-reason { margin: 0; font-size: 13.5px; overflow-wrap: anywhere; }
.mv-math { display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 4px 12px; margin: 0; font-size: 13px; }
.mv-math > div { min-width: 0; }
.mv-math dt { font-size: 10.5px; font-weight: 750; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-2); }
.mv-math dd { margin: 0; font-weight: 700; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.mv-chance { margin: 0; font-size: 13.5px; font-variant-numeric: tabular-nums; }
.mv-none { margin: 0; font-size: 13.5px; color: var(--text-2); }

/* the totals' adjustment under the game page's prices */
.cx-adj-note { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; margin-top: 8px; font-size: 12.5px; color: var(--text-2); overflow-wrap: anywhere; }
`;

export function ensureContextStyles() {
  if (typeof document === "undefined" || document.getElementById("context-styles")) return;
  const style = document.createElement("style");
  style.id = "context-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
