// Styles only Today uses: the toolbar, the league tree, the desktop board and the phone cards. styles.css
// holds the shared design system (tokens, cards, chips, odds buttons, sheets) and the shell's panes; these
// rules are added once, when Today first loads (a module in the page's import map, so no extra request),
// and every one is scoped to .view-today or to the tree (.lt), which the shell moves between its left pane,
// a drawer and a sheet. Colours are the shared tokens (Telegram's theme flows through them); meaning is
// never carried by colour alone (live has a word, odds dots have a legend and labels, edges have signs).

const CSS = `
.view-today { container: today / inline-size; --lh: 28px; }
@media (pointer: coarse) { .view-today { --lh: 44px; } }

/* --- toolbar ------------------------------------------------------------------------------------------- */
.view-today .td-head { display: flex; align-items: center; gap: 4px; margin-bottom: 6px; }
.view-today .td-head h1 { font-size: 22px; font-weight: 750; letter-spacing: -0.01em; }
.view-today .td-sub { color: var(--text-2); font-size: 13px; margin-top: 2px; }
.view-today .td-updated { white-space: nowrap; }
.view-today .td-updated.warn { color: var(--warn-ink); }
.view-today .td-tree-btn { display: none; }
.view-today .td-chips { margin-bottom: 4px; }
.view-today .td-chip-list { display: contents; } /* the redrawn chips; the Sports chip after them stays put */
.view-today .td-chip-n { color: var(--text-2); font-weight: 600; font-size: 12.5px; }
.view-today .td-chips .chip .live-dot { width: 7px; height: 7px; }
.view-today .td-bar { display: grid; gap: 8px; padding: 6px 0 8px; }
.view-today .td-days { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; padding: 4px var(--gutter); margin: 0 calc(-1 * var(--gutter)); }
.view-today .td-days::-webkit-scrollbar { display: none; }
.view-today .td-day {
  position: relative; flex: none; display: inline-flex; align-items: center; gap: 6px; min-height: 38px; padding: 0 12px;
  border-radius: 999px; border: 1px solid var(--line-strong); background: var(--card); color: var(--text); font-weight: 650; font-size: 13.5px; white-space: nowrap;
}
.view-today .td-day::after { content: ""; position: absolute; inset: -3px 0; }
.view-today .td-day:hover { border-color: var(--text-3); }
.view-today .td-day.on { background: var(--accent-soft); border-color: var(--accent); }
.view-today .td-day.zero { color: var(--text-3); }
.view-today .td-day-n { font-size: 12px; font-weight: 700; color: var(--text-2); min-width: 1ch; }
.view-today .td-day.on .td-day-n { color: var(--good-ink); }
.view-today .td-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0; }
.view-today .td-controls > [data-td-status] { flex: 1 1 100%; min-width: 0; }
.view-today .td-status { width: 100%; }
.view-today .td-status button { padding: 0 6px; display: inline-flex; align-items: center; justify-content: center; gap: 4px; }
.view-today .td-status .live-dot { width: 7px; height: 7px; }
.view-today .td-live-n { font-size: 12px; color: var(--danger-ink); font-weight: 750; }
.view-today .td-search { position: relative; flex: 1 1 160px; min-width: 0; }
.view-today .td-search .input { min-height: 40px; padding: 8px 10px 8px 34px; }
.view-today .td-search-icon { position: absolute; left: 11px; top: 50%; transform: translateY(-50%); color: var(--text-3); display: grid; pointer-events: none; }
.view-today .td-sort { flex: 0 1 auto; min-width: 0; }
.view-today .td-sort .input { min-height: 40px; padding: 8px 10px; width: 10.5em; max-width: 100%; text-overflow: ellipsis; }
@media (pointer: coarse), (max-width: 839px) { .view-today .td-search .input, .view-today .td-sort .input { min-height: 44px; } }
.view-today .td-notes:empty { display: none; }
.view-today .td-notes { display: grid; gap: 6px; margin: 2px 0 8px; }
.view-today .td-notes .notice { padding: 8px 10px; font-size: 13px; }
.view-today .td-ai-note { color: var(--text-2); font-size: 12.5px; }

@media (min-width: 840px) {
  .view-today .td-chips { display: none; }
  .view-today .td-bar {
    position: sticky; top: var(--pane-top, 0px); z-index: 4; background: var(--bg); /* under the shell's panes (5): a peeking slip covers it */
    grid-template-columns: minmax(0, 1fr); padding: 8px 0; margin: 0; border-bottom: 1px solid var(--line);
  }
  .view-today .td-days { margin: 0; padding: 2px 0; }
  .view-today .td-controls > [data-td-status] { flex: 0 1 auto; }
}
@media (min-width: 840px) and (max-width: 1199px) {
  .view-today .td-tree-btn { display: inline-flex; margin-left: 4px; }
}
@container today (min-width: 980px) {
  .view-today .td-bar { grid-template-columns: auto minmax(0, 1fr); align-items: center; column-gap: 16px; }
  .view-today .td-controls { justify-content: flex-end; flex-wrap: nowrap; }
  .view-today .td-search { flex: 0 1 220px; }
}

/* --- the league tree (in the left pane, a drawer or a sheet) ------------------------------------------- */
.lt { display: grid; grid-template-columns: minmax(0, 1fr); gap: 4px; --row: 36px; min-width: 0; }
@media (pointer: coarse) { .lt { --row: 44px; } }
.lt-presets { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 0 8px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.lt-saved { display: inline-flex; align-items: center; border-radius: 999px; }
.lt-preset {
  position: relative; display: inline-flex; align-items: center; gap: 5px; min-height: 32px; padding: 0 10px; border-radius: 999px;
  border: 1px solid var(--line-strong); background: var(--card); color: var(--text); font-size: 13px; font-weight: 600; white-space: nowrap;
  max-width: 100%; overflow: hidden; text-overflow: ellipsis;
}
.lt-preset::after { content: ""; position: absolute; inset: -6px -2px; }
.lt-preset:hover { border-color: var(--text-3); }
.lt-preset[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); }
.lt-preset .live-dot { width: 7px; height: 7px; }
.lt-pn { color: var(--danger-ink); font-weight: 750; font-size: 12px; }
.lt-add { color: var(--text-2); border-style: dashed; }
.lt-saved .lt-preset { border-top-right-radius: 0; border-bottom-right-radius: 0; }
.lt-del {
  position: relative; display: inline-grid; place-items: center; width: 30px; min-height: 32px; border: 1px solid var(--line-strong); border-left: 0;
  border-radius: 0 999px 999px 0; background: var(--card); color: var(--text-2);
}
.lt-del::after { content: ""; position: absolute; inset: -6px -4px -6px 0; }
.lt-del:hover { color: var(--danger-ink); }
.lt-save { display: flex; gap: 6px; align-items: center; padding: 4px 0 8px; }
.lt-save .input { min-height: 38px; padding: 6px 10px; }
.lt-row { display: flex; align-items: center; min-height: var(--row); min-width: 0; border-radius: 10px; }
.lt-hit { flex: none; display: grid; place-items: center; width: 32px; min-height: var(--row); cursor: pointer; }
.lt-box { width: 18px; height: 18px; margin: 0; accent-color: var(--accent); cursor: pointer; }
.lt-name {
  flex: 1 1 auto; display: flex; align-items: center; gap: 6px; min-width: 0; min-height: var(--row); padding: 0 8px 0 2px;
  border-radius: 8px; color: var(--text); text-align: left; font-size: 14px;
}
.lt-name:hover { background: var(--card-2); }
.lt-text { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lt-pin { color: var(--warn); vertical-align: -1px; }
.lt-icon { flex: none; width: 20px; text-align: center; }
.lt-dots { flex: none; display: inline-flex; align-items: center; gap: 5px; }
.lt-dots .live-dot { width: 7px; height: 7px; }
.lt-live { display: inline-flex; align-items: center; gap: 3px; font-size: 11.5px; font-weight: 750; color: var(--danger-ink); font-variant-numeric: tabular-nums; }
.lt-pins { margin-top: 4px; justify-self: start; padding: 0 6px; white-space: normal; text-align: left; max-width: 100%; }
.lt-n { flex: none; min-width: 2ch; text-align: right; font-size: 12.5px; font-weight: 650; color: var(--text-2); font-variant-numeric: tabular-nums; }
.lt-all { background: var(--card-2); border: 1px solid var(--line); }
.lt-groups, .lt-items { list-style: none; margin: 0; padding: 0; display: grid; gap: 1px; min-width: 0; }
.lt-grow .lt-name { font-weight: 650; }
.lt-items { padding-left: 22px; }
.lt-items[hidden] { display: none; }
.lt-zero .lt-name { color: var(--text-3); }
.lt-fold { width: 36px; height: var(--row); }
@media (pointer: coarse) { .lt-hit, .lt-fold { width: 44px; } } /* a finger's target: a near miss mustn't hit "Show only" */
.lt-fold .icon { transition: transform var(--speed); }
.lt-fold[aria-expanded="true"] .icon { transform: rotate(180deg); }
.lt-key { margin-top: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; line-height: 1.6; }
.odds-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.odds-dot.fresh { background: var(--accent); }
.odds-dot.stale { background: var(--warn); }
.odds-dot.none { background: transparent; box-shadow: inset 0 0 0 1.5px var(--text-3); }
.sheet .lt { padding-bottom: 8px; }

/* --- the desktop board ---------------------------------------------------------------------------------- */
/* Under 820 px (a desk with the tree and the slip open, or the inspector beside it) there is no Notes column:
   the badges and "as of" go under the time (.tb-notes-in), so the names keep the room. today.js columnsAt()
   follows these breakpoints (6 columns, 7 from 820, 9 from 1180; AI and Edge add 2). The League and "Odds as
   of" columns wait for 1180 px, so the names keep about 330 px beside them. */
.view-today .tb {
  --c-when: 84px; --c-lg: 0px; --c-match: minmax(130px, 1fr); --c-h2h: 60px; --c-sp: 80px; --c-tot: 92px; --c-mkt: 72px;
  --c-ai: 46px; --c-edge: 56px; --c-notes: 76px; --tb-min: 518px;
  --cols: var(--c-when) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt);
  border: 1px solid var(--line); border-radius: var(--radius); background: var(--card); overflow: clip; margin-top: 4px;
}
.view-today .tb.has-ai { --cols: var(--c-when) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt) var(--c-ai) var(--c-edge); --tb-min: 620px; }
.view-today .tb-c.tb-badges { display: none; }
.view-today .tb-notes-in { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; min-width: 0; font-size: 11.5px; color: var(--text-2); }
.view-today .tb-notes-in .badge { font-size: 10.5px; line-height: 1.3; padding: 1px 5px; max-width: 100%; white-space: normal; } /* "Far from market" wraps rather than being cut */
@container today (min-width: 820px) {
  .view-today .tb {
    --c-when: 100px; --c-match: minmax(180px, 1fr); --c-h2h: 70px; --c-sp: 88px; --c-tot: 100px; --c-mkt: 76px; --c-ai: 52px; --c-edge: 62px; --c-notes: 104px; --tb-min: 718px;
    --cols: var(--c-when) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt) var(--c-notes);
  }
  .view-today .tb.has-ai { --tb-min: 832px; --cols: var(--c-when) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt) var(--c-ai) var(--c-edge) var(--c-notes); }
  .view-today .tb-c.tb-badges { display: flex; }
  .view-today .tb-notes-in { display: none; }
}
@container today (min-width: 1180px) {
  .view-today .tb {
    --c-when: 84px; --c-lg: 108px; --c-match: minmax(200px, 1fr); --c-h2h: 72px; --c-sp: 90px; --c-tot: 100px; --c-mkt: 88px; --c-ai: 54px;
    --c-edge: 64px; --c-notes: 120px; --c-asof: 84px; --tb-min: 946px;
    --cols: var(--c-when) var(--c-lg) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt) var(--c-notes) var(--c-asof);
  }
  .view-today .tb.has-ai {
    --tb-min: 1064px;
    --cols: var(--c-when) var(--c-lg) var(--c-match) var(--c-h2h) var(--c-sp) var(--c-tot) var(--c-mkt) var(--c-ai) var(--c-edge) var(--c-notes) var(--c-asof);
  }
}
.view-today .tb-headwrap {
  position: sticky; top: calc(var(--pane-top, 0px) + var(--td-bar-h, 0px)); z-index: 3; overflow: hidden;
  background: var(--card-2); border-bottom: 1px solid var(--line-strong);
}
.view-today .tb-bodywrap { overflow-x: auto; overscroll-behavior-x: contain; }
.view-today .tb-head, .view-today .tb-body { min-width: var(--tb-min); }
.view-today .tb-hrow, .view-today .tb-row { display: grid; grid-template-columns: var(--cols); }
.view-today .tb-c.tb-lg, .view-today .tb-c.tb-asof { display: none; }
@container today (min-width: 1180px) {
  .view-today .tb-c.tb-lg, .view-today .tb-c.tb-asof { display: grid; }
  .view-today .tb-c .tb-lg-in, .view-today .tb-c .tb-asof-in { display: none; }
}
.view-today .tb-c { min-width: 0; padding: 6px 6px; display: grid; grid-auto-rows: var(--lh); align-content: center; row-gap: 2px; }
.view-today .tb-hrow .tb-c {
  grid-auto-rows: auto; align-items: center; grid-auto-flow: column; justify-content: start; gap: 0; min-height: 38px; padding: 4px 4px;
  font-size: 11px; font-weight: 750; letter-spacing: 0.02em; text-transform: uppercase; color: var(--text-2); white-space: nowrap; overflow: hidden;
}
@container today (min-width: 820px) { .view-today .tb-hrow .tb-c { padding: 4px 6px; font-size: 11.5px; letter-spacing: 0.04em; } }
.view-today .tb-hrow .tb-c > span { overflow: hidden; text-overflow: ellipsis; }
.view-today .tb-sort { position: relative; display: inline-flex; align-items: center; gap: 2px; min-height: 30px; color: inherit; font: inherit; letter-spacing: inherit; text-transform: inherit; padding: 0 2px; border-radius: 6px; }
.view-today .tb-sort::after { content: ""; position: absolute; inset: -6px -2px; }
.view-today .tb-sort:hover { color: var(--text); }
.view-today [aria-sort="ascending"] .tb-sort, .view-today [aria-sort="descending"] .tb-sort { color: var(--text); }
.view-today .tb-arrow { font-size: 9px; min-width: 9px; }
.view-today .tb-hrow .h-full { display: none; }
@container today (min-width: 1180px) { .view-today .tb-hrow .h-full { display: inline; } .view-today .tb-hrow .h-short { display: none; } }
.view-today .tb-when { position: sticky; left: 0; z-index: 1; background: var(--card); border-right: 1px solid var(--line); }
.view-today .tb-hrow .tb-when { background: var(--card-2); }
.view-today .tb-row {
  position: relative; border-bottom: 1px solid var(--line); background: var(--card); cursor: pointer;
  content-visibility: auto; contain-intrinsic-size: auto calc(var(--lh) * 2 + 16px);
}
.view-today .tb-row.l3 { contain-intrinsic-size: auto calc(var(--lh) * 3 + 18px); }
.view-today .tb-row:hover, .view-today .tb-row:hover .tb-when { background: var(--card-2); }
.view-today .tb-row.is-open, .view-today .tb-row.is-open .tb-when { background: var(--accent-soft); }
.view-today .tb-row.is-open { box-shadow: inset 3px 0 0 var(--accent); }
.view-today .tb-row.is-open .tb-when { box-shadow: inset 3px 0 0 var(--accent); }
.view-today .tb-row.is-final { color: var(--text-2); }
.view-today .tb-when { font-size: 13px; align-content: center; grid-auto-rows: auto; gap: 2px; }
.view-today .tb-lg, .view-today .tb-lg-in { font-size: 12.5px; color: var(--text-2); min-width: 0; overflow: hidden; }
.view-today .td-league { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.view-today .tb-lg { align-content: center; grid-auto-rows: auto; }
.view-today .td-live { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; }
.view-today .td-clock { font-size: 12px; color: var(--danger-ink); font-weight: 650; font-variant-numeric: tabular-nums; }
.view-today .td-final { font-weight: 650; color: var(--text-2); }
.view-today .td-post { display: inline-flex; align-items: center; gap: 3px; color: var(--warn-ink); font-weight: 650; font-size: 12px; }
.view-today .td-time { font-weight: 650; }
.view-today .td-open { display: block; width: 100%; min-width: 0; text-align: left; color: var(--text); border-radius: 6px; }
.view-today .td-open:hover { text-decoration: none; }
.view-today .tb-match { grid-auto-rows: auto; }
.view-today .td-lines { display: grid; grid-auto-rows: var(--lh); row-gap: 2px; min-width: 0; }
.view-today .td-team { display: flex; align-items: center; gap: 6px; min-width: 0; font-size: 14px; }
.view-today .td-tname { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.view-today .td-team.won .td-tname, .view-today .td-team.won .td-score { font-weight: 800; color: var(--text); }
.view-today .td-drawline .td-tname { color: var(--text-2); font-weight: 500; padding-left: 26px; }
.view-today .tm-short { display: none; }
@container today (max-width: 819px) { .view-today .tb .tm-full { display: none; } .view-today .tb .tm-short { display: inline; } }
.view-today .td-logo { flex: none; width: 20px; height: 20px; object-fit: contain; border-radius: 4px; }
.view-today .td-logo.blank { display: inline-block; background: var(--card-3); border-radius: 50%; }
.view-today .td-rec { flex: none; font-size: 11.5px; color: var(--text-3); }
@container today (max-width: 719px) { .view-today .tb .td-rec { display: none; } } /* a narrow board keeps the names readable */
.view-today .td-b2b { flex: none; font-size: 10px; font-weight: 800; letter-spacing: 0.03em; color: var(--warn-ink); background: var(--warn-soft); border-radius: 4px; padding: 1px 4px; }
.view-today .td-score { flex: none; min-width: 2.2ch; text-align: right; font-size: 14.5px; }
/* Price cells: tighter padding, and columns wide enough for "U 224.5 1.91" (a price must never spill out of its button) */
.view-today .tb-row .tb-h2h, .view-today .tb-row .tb-spreads, .view-today .tb-row .tb-totals { padding-inline: 4px; }
.view-today .tb-row .odds.tb-odds, .view-today .tb-row .odds.empty.tb-odds {
  min-height: 0; height: var(--lh); flex-direction: row; gap: 3px; padding: 0 4px; border-radius: 8px; justify-content: center;
}
.view-today .tb-odds .o-point { font-size: 11px; }
.view-today .tb-odds .o-price { font-size: 13px; }
.view-today .tb-odds .o-check { display: none !important; }
.view-today .tb-gap { height: var(--lh); }
.view-today .tb-none { grid-column: span 3; display: flex; align-items: center; justify-content: center; padding: 6px 10px; }
.view-today .td-noodds { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; text-align: center; line-height: 1.35; }
.view-today .td-noodds.link { color: var(--info-ink); font-weight: 600; min-height: 32px; padding: 0 8px; border-radius: 8px; border: 1px dashed color-mix(in srgb, var(--info) 45%, transparent); }
.view-today .td-noodds.link:hover { background: var(--info-soft); }
@media (pointer: coarse) { .view-today .td-noodds.link { min-height: 44px; } }
.view-today .td-v { display: flex; align-items: center; justify-content: flex-end; font-size: 13px; color: var(--text-2); white-space: nowrap; }
.view-today .td-v.fav { color: var(--text); font-weight: 750; }
.view-today .td-edge.up { color: var(--good-ink); font-weight: 750; }
.view-today .td-edge.far { color: var(--warn-ink); font-weight: 750; }
.view-today .td-edge.down { color: var(--text-3); }
.view-today .td-mark { margin-left: 3px; font-size: 10px; color: var(--good-ink); }
.view-today .td-mark.warn { color: var(--warn-ink); font-weight: 800; font-size: 12px; }
.view-today .tb-mkt, .view-today .tb-ai, .view-today .tb-edge { padding-right: 10px; }
/* "Markets say": ⚖ (sources 5 points or more apart, with how far) and "Low" (ratings only) sit at the start of a
   line, the blend's value at its end; the cell's label and tooltip list every source */
.view-today .tb-row .tb-mkt { padding-left: 4px; padding-right: 8px; }
.view-today .tb-mkt .td-v { gap: 3px; }
.view-today .tb-mkt .mk-flag { margin-right: auto; }
.view-today .mk-flag {
  display: inline-flex; align-items: center; gap: 2px; flex: none; padding: 0 4px; min-height: 18px; border-radius: 999px;
  font-size: 10.5px; font-weight: 750; line-height: 1; white-space: nowrap; border: 1px solid; font-variant-numeric: tabular-nums;
}
.view-today .mk-flag.dis { color: var(--warn-ink); border-color: color-mix(in srgb, var(--warn) 45%, transparent); background: var(--warn-soft); }
.view-today .mk-flag.low { color: var(--text-2); border-color: var(--line-strong); background: var(--card-2); font-weight: 650; }
.view-today .mk-scale { font-size: 12px; }
.view-today .tb-mkt.is-low .mk-v { color: var(--text-2); font-style: italic; }
.view-today .tb-hrow .tb-mkt, .view-today .tb-hrow .tb-ai, .view-today .tb-hrow .tb-edge { justify-content: end; }
.view-today .tb-badges { display: flex; flex-wrap: wrap; align-content: center; align-items: center; gap: 4px; }
.view-today .tb-badges .badge { font-size: 11px; padding: 1px 6px; white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.view-today .tb-asof, .view-today .tb-asof-in { font-size: 12px; color: var(--text-2); }
.view-today .tb-asof-in { display: block; width: 100%; }
.view-today .tb-asof { align-content: center; grid-auto-rows: auto; }
.view-today .tb-sep { background: var(--bg); border-bottom: 1px solid var(--line); }
.view-today .tb-sep-c { position: sticky; left: 0; display: inline-block; padding: 12px 10px 6px; font-size: 12px; font-weight: 750; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-2); }
.view-today .tb-detail { background: var(--card-2); border-bottom: 1px solid var(--line); }
.view-today .tb-detail-in { position: sticky; left: 0; max-width: min(100%, 680px); padding: 10px 14px 12px; display: grid; gap: 6px; }
.view-today .td-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 6px 16px; font-size: 13px; }
.view-today .td-facts dt { font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-3); }
.view-today .td-facts dd { margin: 0; }
.view-today .tb-foot { padding: 10px 2px 0; }

/* --- phone cards ------------------------------------------------------------------------------------------ */
.view-today .td-cards { gap: 10px; }
.view-today .td-card { content-visibility: auto; contain-intrinsic-size: auto 250px; }
.view-today .td-card .gc-link { display: block; width: 100%; text-align: left; color: var(--text); }
.view-today .td-card .gc-when { gap: 4px; }
.view-today .td-card .td-dot { color: var(--text-3); }
.view-today .td-card .td-league { display: inline; }
.view-today .td-card-teams { margin: 6px 0 4px; --lh: 30px; }
.view-today .td-card-teams .td-team { font-size: 15.5px; }
.view-today .td-card-teams .td-tname { font-weight: 700; }
.view-today .td-card-teams .td-logo { width: 22px; height: 22px; }
.view-today .td-card-teams .td-score { font-size: 17px; }
.view-today .td-nums { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; padding: 0 14px 8px; margin: 0; }
.view-today .td-num { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 5px 8px; min-width: 0; }
.view-today .td-num dt { font-size: 10.5px; font-weight: 750; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-3); }
.view-today .td-num dd { margin: 0; font-size: 14px; font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.view-today .td-num.up dd { color: var(--good-ink); }
@media (max-width: 400px) { .view-today .td-num { padding: 5px 6px; } .view-today .td-num dd { font-size: 13px; } .view-today .td-nums .td-mark { margin-left: 1px; } }
.view-today .td-num.far dd { color: var(--warn-ink); }
.view-today .mk-strip {
  display: flex; align-items: center; gap: 6px; min-width: 0; margin: 0 14px 8px; padding: 5px 8px; border-radius: 10px;
  background: var(--card-2); border: 1px solid var(--line); font-size: 12.5px; color: var(--text-2); white-space: nowrap;
}
.view-today .mk-k { flex: none; font-size: 10.5px; font-weight: 750; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-3); }
@media (max-width: 399px) { .view-today .mk-k { display: none; } } /* the tile above says "Markets": the sources get the room */
.view-today .mk-flags { flex: none; display: inline-flex; gap: 4px; }
.view-today .mk-flags:empty { display: none; }
.view-today .mk-src { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
/* Sources that disagree (⚖): the whole list shows, on a line of its own under the flags when it doesn't fit beside them */
.view-today .mk-strip.wrap { flex-wrap: wrap; row-gap: 3px; }
.view-today .mk-strip.wrap .mk-src { flex: 1 1 240px; white-space: normal; overflow: visible; overflow-wrap: anywhere; }
.view-today .mk-src b { color: var(--text); font-weight: 700; }
.view-today .td-card-prices .odds-grid { padding: 2px 14px 0; }
.view-today .td-card-none { padding: 2px 14px 6px; color: var(--text-2); }
.view-today .td-card .td-noodds.link { min-height: 44px; width: 100%; justify-content: center; }
.view-today .td-card .tb-detail-in { position: static; padding: 8px 14px; }
.view-today .td-card .gc-foot { min-height: 48px; }
.view-today .td-card-asof { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.view-today .td-sep { margin: 14px 0 4px; }

/* --- empty, skeleton ---------------------------------------------------------------------------------------- */
.view-today .td-empty { margin-top: 8px; }
.view-today .td-skel-row { display: grid; grid-template-columns: 90px minmax(0, 1fr) repeat(3, 80px) 60px; gap: 10px; align-items: center; padding: 12px; border-bottom: 1px solid var(--line); }
.view-today .td-skel-row .skel { height: 14px; }
.view-today .td-skel-row .skel.tall { height: 46px; }
@media (prefers-reduced-motion: reduce) { .view-today .spin { animation: none; } }
.view-today .spin, .view-today [data-td-refresh].busy .icon { animation: td-spin 1s linear infinite; }
@keyframes td-spin { to { transform: rotate(360deg); } }
`;

export function ensureStyles() {
  if (document.getElementById("today-styles")) return;
  const style = document.createElement("style");
  style.id = "today-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
