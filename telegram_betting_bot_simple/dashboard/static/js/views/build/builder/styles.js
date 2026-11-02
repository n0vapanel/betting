// Styles only the Builder uses (#/build/builder and its sheets). Added once, when the Builder first opens.
// Every rule is scoped to .view-build or to the Builder's sheets (.bd-sheet), except the one that lets the
// Builder's two columns use more of a wide window. Colours are the shared tokens, so the dark, light and
// Telegram themes all flow through; meaning is never carried by colour alone (labels have words, edges
// have signs, flags have names).

const CSS = `
/* the Builder may use more of a wide window than the other screens' 920 px column */
@media (min-width: 840px) {
  body[data-route="build"][data-sub="builder"] .shell > * { max-width: 1320px; }
}

:is(.view-build, .bd-sheet) .bd-head h1 { display: flex; align-items: center; gap: 8px; }

/* --- layout ----------------------------------------------------------------------------------------- */
.view-build .bd-layout { display: grid; gap: 14px; min-width: 0; }
.view-build .bd-layout.is-wide { grid-template-columns: minmax(290px, 350px) minmax(0, 1fr); align-items: start; gap: 18px; }
.view-build .is-wide .bd-preset { padding: 8px 10px; }
.view-build .is-wide .bd-preset:not(.bd-custom) .icon { display: none; } /* the column is narrow: the bands need the room */
.view-build .bd-controls, .view-build .bd-scroll { min-width: 0; display: grid; gap: 10px; align-content: start; }
.view-build .bd-controls.off { opacity: 0.55; }
.view-build [data-bd-pstatus] { margin-top: 8px; }
/* Desktop: a sticky column whose controls scroll inside it while the run bar (Build and its error line) stays at
   its foot, outside the scrolling part. The column is as tall as the window below where it starts (--bd-top,
   kept by builder.js while the page head is still in view; the sticky top once it sticks), so the run bar is
   always on screen. */
.view-build .is-wide .bd-controls {
  --bd-top-at: max(var(--bd-top, 0px), calc(var(--header-h) + var(--safe-top) + 12px));
  position: sticky; top: calc(var(--header-h) + var(--safe-top) + 12px);
  max-height: calc(100vh - var(--bd-top-at) - 12px); max-height: calc(100dvh - var(--bd-top-at) - 12px);
  display: flex; flex-direction: column; gap: 0; overflow: hidden;
  border: 1px solid var(--line); border-radius: var(--radius); background: var(--card);
}
.view-build .is-wide .bd-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 12px; scrollbar-width: thin; }
.view-build .is-wide .bd-run { flex: none; padding: 10px 12px 12px; background: var(--card); border-top: 1px solid var(--line); }
.view-build .bd-out { min-width: 0; container: bdout / inline-size; }
.view-build .bd-run { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.view-build .bd-run .btn { flex: 1 1 140px; }
.view-build .bd-run .form-error { flex: 1 1 100%; margin-top: 0; }
.view-build .bd-stale-note { flex: 1 1 100%; display: flex; align-items: center; gap: 4px; margin: 0; }

/* --- presets ---------------------------------------------------------------------------------------- */
.view-build .bd-presets { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.view-build .bd-preset {
  position: relative; display: flex; align-items: center; gap: 8px; min-height: 52px; min-width: 0; padding: 8px 10px; border-radius: 12px; text-align: left;
  background: var(--card-2); border: 1px solid var(--line-strong); color: var(--text); transition: background var(--speed), border-color var(--speed);
}
.view-build .bd-preset:hover { border-color: var(--text-3); }
.view-build .bd-preset > span { display: grid; min-width: 0; }
.view-build .bd-preset b { font-size: 14.5px; line-height: 1.2; }
.view-build .bd-preset small { font-size: 11.5px; line-height: 1.3; color: var(--text-2); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.view-build .bd-preset .icon { flex: none; color: var(--text-2); }
.view-build .bd-preset[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); }
.view-build .bd-preset.t-safe[aria-pressed="true"] .icon { color: var(--good-ink); }
.view-build .bd-preset.t-risky[aria-pressed="true"] { background: var(--warn-soft); border-color: color-mix(in srgb, var(--warn) 70%, transparent); }
.view-build .bd-preset.t-risky[aria-pressed="true"] .icon { color: var(--warn-ink); }
.view-build .bd-preset.t-crazy[aria-pressed="true"] { background: var(--violet-soft); border-color: color-mix(in srgb, var(--violet) 70%, transparent); }
.view-build .bd-preset.t-crazy[aria-pressed="true"] .icon { color: var(--violet); }
.view-build .bd-custom { background: var(--info-soft); border-color: color-mix(in srgb, var(--info) 60%, transparent); min-height: 44px; margin-bottom: 8px; padding-right: 4px; }
.view-build .bd-custom > .icon { color: var(--info-ink); }
.view-build .bd-custom > span { flex: 1 1 auto; }
.view-build .bd-custom .bd-reset { flex: none; margin-left: auto; padding: 0 8px; position: relative; }
.view-build .bd-custom .bd-reset::after { content: ""; position: absolute; inset: -6px -2px; } /* a 44 px hit area inside the chip */
.view-build .bd-preset-note { display: flex; align-items: flex-start; gap: 6px; margin: 0; font-size: 13px; line-height: 1.4; color: var(--text-2); }
.view-build .bd-preset-note > .icon { flex: none; color: var(--violet); margin-top: 2px; }
.view-build .bd-preset-note > span { min-width: 0; }

.view-build .bd-preset-note.crazy > span { color: var(--text); }
.view-build .bd-reset { min-height: 32px; font-size: 13px; gap: 4px; }
.view-build .nowrap { white-space: nowrap; }
/* 360 px phones: three presets still fit, the band under each name drops to one short line */
@media (max-width: 380px) {
  .view-build .bd-preset { padding: 6px 7px; gap: 6px; }
  .view-build .bd-preset .icon { display: none; }
}

/* --- groups and fields (the column, the phone strip and the sheet) --------------------------------------- */
:is(.view-build, .bd-sheet) .bd-group { border: 1px solid var(--line); border-radius: 12px; background: var(--card); min-width: 0; }
:is(.view-build, .bd-sheet) .bd-group + .bd-group { margin-top: 8px; }
:is(.view-build, .bd-sheet) .bd-group > summary {
  list-style: none; cursor: pointer; display: flex; align-items: center; gap: 10px; min-height: 52px; padding: 6px 10px 6px 12px; border-radius: 12px;
}
:is(.view-build, .bd-sheet) .bd-group > summary::-webkit-details-marker { display: none; }
:is(.view-build, .bd-sheet) .bd-group > summary:hover { background: var(--card-2); }
:is(.view-build, .bd-sheet) .bd-group > summary > .icon { flex: none; margin-left: auto; color: var(--text-2); transition: transform var(--speed); }
:is(.view-build, .bd-sheet) .bd-group[open] > summary > .icon { transform: rotate(180deg); }
:is(.view-build, .bd-sheet) .bd-gicon { flex: none; width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center; background: var(--card-2); color: var(--text-2); }
:is(.view-build, .bd-sheet) .bd-gt { display: grid; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-gt b { font-size: 14.5px; }
:is(.view-build, .bd-sheet) .bd-gt small { font-size: 12px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
:is(.view-build, .bd-sheet) .bd-gbody { padding: 4px 12px 14px; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-gbody > .field:first-child { margin-top: 6px; }
:is(.view-build, .bd-sheet) .field { margin-top: 14px; }
:is(.view-build, .bd-sheet) .label { font-size: 13.5px; margin-bottom: 6px; }
:is(.view-build, .bd-sheet) .bd-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-items: start; }
:is(.view-build, .bd-sheet) .bd-pair.bd-target { grid-template-columns: minmax(0, 1fr) var(--w-tol, 96px); }
:is(.view-build, .bd-sheet) .bd-trio { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
:is(.view-build, .bd-sheet) .bd-num { display: grid; gap: 4px; min-width: 0; align-content: start; }
:is(.view-build, .bd-sheet) .bd-mini { display: flex; align-items: center; gap: 2px; font-size: 12.5px; font-weight: 650; color: var(--text-2); min-height: 22px; }
:is(.view-build, .bd-sheet) .bd-num .input { min-height: var(--tap); padding: 8px 10px; font-variant-numeric: tabular-nums; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-num select.input { padding: 6px 8px; }
:is(.view-build, .bd-sheet) .bd-num .money-input .input { padding-left: 26px; }
:is(.view-build, .bd-sheet) .bd-num input[type="time"], :is(.view-build, .bd-sheet) .bd-num input[type="datetime-local"] { padding: 6px 8px; font-size: 14px; }
:is(.view-build, .bd-sheet) .bd-box { position: relative; display: flex; align-items: center; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-box.has-suffix .input { padding-right: 34px; }
:is(.view-build, .bd-sheet) .bd-box > span { position: absolute; right: 10px; color: var(--text-2); font-size: 13px; pointer-events: none; }
:is(.view-build, .bd-sheet) .input[aria-invalid="true"] { border-color: var(--danger); }
:is(.view-build, .bd-sheet) .bd-err { color: var(--danger-ink); font-size: 12.5px; }
:is(.view-build, .bd-sheet) .bd-err:empty { display: none; }
:is(.view-build, .bd-sheet) .bd-hint { display: block; color: var(--text-2); font-size: 12.5px; line-height: 1.45; margin: 6px 0 0; }
:is(.view-build, .bd-sheet) .bd-hint .icon { vertical-align: -2px; }
:is(.view-build, .bd-sheet) .bd-hint .info-btn { margin: -8px 0; }
:is(.view-build, .bd-sheet) .bd-conv:empty { display: none; }
:is(.view-build, .bd-sheet) .bd-check {
  display: flex; align-items: flex-start; gap: 10px; min-height: var(--tap); padding: 8px 4px; cursor: pointer; border-radius: 10px;
}
:is(.view-build, .bd-sheet) .bd-check + .bd-check { border-top: 1px solid var(--line); border-radius: 0; }
:is(.view-build, .bd-sheet) .bd-check input { flex: none; width: 20px; height: 20px; margin: 2px 0 0; accent-color: var(--accent); }
:is(.view-build, .bd-sheet) .bd-check > span { flex: 1 1 auto; display: grid; gap: 2px; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-check b { font-size: 14px; font-weight: 600; }
:is(.view-build, .bd-sheet) .bd-check small { font-size: 12.5px; color: var(--text-2); line-height: 1.4; }
:is(.view-build, .bd-sheet) .bd-check.off { cursor: not-allowed; }
:is(.view-build, .bd-sheet) .bd-check.off input { opacity: 0.6; }
:is(.view-build, .bd-sheet) .bd-check.off b { color: var(--text-2); }
:is(.view-build, .bd-sheet) .bd-num select.input:disabled { opacity: 1; color: var(--text-2); background: var(--card-2); border-style: dashed; cursor: not-allowed; }
:is(.view-build, .bd-sheet) .bd-check .info-btn { margin: -2px 0 0; flex: none; }
:is(.view-build, .bd-sheet) .bd-mkts, :is(.view-build, .bd-sheet) .bd-books { display: flex; flex-wrap: wrap; gap: 8px; }
:is(.view-build, .bd-sheet) .bd-mkt {
  position: relative; display: inline-flex; align-items: center; gap: 6px; min-height: 40px; padding: 4px 12px 4px 8px; border-radius: 999px; cursor: pointer;
  background: var(--card-2); border: 1px solid var(--line-strong); font-size: 13.5px; font-weight: 600; max-width: 100%;
}
:is(.view-build, .bd-sheet) .bd-mkt input { width: 18px; height: 18px; margin: 0; accent-color: var(--accent); flex: none; }
:is(.view-build, .bd-sheet) .bd-mkt > span { display: grid; min-width: 0; line-height: 1.2; }
:is(.view-build, .bd-sheet) .bd-mkt small { font-size: 11px; font-weight: 500; color: var(--text-2); }
:is(.view-build, .bd-sheet) .bd-mkt:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); }
:is(.view-build, .bd-sheet) .bd-mkt.off { cursor: not-allowed; border-style: dashed; color: var(--text-2); background: transparent; }
:is(.view-build, .bd-sheet) .bd-adv { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 4px; }
:is(.view-build, .bd-sheet) .bd-adv > summary {
  cursor: pointer; min-height: var(--tap); display: flex; align-items: center; gap: 6px; font-weight: 650; font-size: 13.5px; color: var(--info);
}
:is(.view-build, .bd-sheet) .bd-wins button { padding: 0 6px; font-size: 13px; }
:is(.view-build, .bd-sheet) .bd-tree { min-width: 0; }
/* Today's tree rows are 36 px for a mouse and 44 px on a touch screen; with a mouse the Builder's are a little taller */
@media (pointer: fine) { :is(.view-build, .bd-sheet) .bd-tree .lt { --row: 40px; } }
:is(.view-build, .bd-sheet) .bd-tree-wait { padding: 8px 0; }
:is(.view-build, .bd-sheet) .bd-pick { display: flex; align-items: center; gap: 8px; text-align: left; cursor: pointer; font-weight: 600; }
:is(.view-build, .bd-sheet) .bd-pick .grow { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
:is(.view-build, .bd-sheet) .bd-pick small { flex: none; color: var(--text-2); font-weight: 500; font-size: 12.5px; }
:is(.view-build, .bd-sheet) .bd-pick > .icon { flex: none; color: var(--text-2); }
/* Closest effort's switch: a checkbox drawn as a switch (role="switch"), the whole row its 44 px target */
:is(.view-build, .bd-sheet) .bd-switch input[role="switch"] {
  appearance: none; -webkit-appearance: none; position: relative; width: 40px; height: 24px; margin: 0; border-radius: 999px; cursor: pointer;
  background: var(--card-2); border: 1px solid var(--line-strong); transition: background var(--speed), border-color var(--speed);
}
:is(.view-build, .bd-sheet) .bd-switch input[role="switch"]::after {
  content: ""; position: absolute; top: 2px; left: 2px; width: 18px; height: 18px; border-radius: 50%; background: var(--text-2);
  transition: transform var(--speed), background var(--speed);
}
:is(.view-build, .bd-sheet) .bd-switch input[role="switch"]:checked { background: var(--accent); border-color: var(--accent); }
:is(.view-build, .bd-sheet) .bd-switch input[role="switch"]:checked::after { transform: translateX(16px); background: var(--accent-ink, #fff); }
:is(.view-build, .bd-sheet) .bd-switch input[role="switch"]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
:is(.view-build, .bd-sheet) .bd-switch { align-items: center; }
:is(.view-build, .bd-sheet) .bd-switch .info-btn { margin: -4px 0; }
:is(.view-build, .bd-sheet) .bd-pair.bd-chance { margin-top: 10px; }
.view-build .bd-key { display: grid; gap: 10px; padding: 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--card); min-width: 0; }
.view-build .bd-key .bd-hint { margin-top: -4px; }
@media (max-width: 360px) {
  :is(.view-build, .bd-sheet) .bd-trio { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
  :is(.view-build, .bd-sheet) .bd-pair { gap: 8px; }
}

/* --- results ---------------------------------------------------------------------------------------------- */
.view-build .bd-results { display: grid; gap: 12px; min-width: 0; }
.view-build .bd-stale { opacity: 0.55; transition: opacity var(--speed); }
.view-build .bd-rhead { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 10px; }
.view-build .bd-rhead h2 { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 750; }
.view-build .bd-notes { margin-top: 0; }
.view-build .bd-notes .icon { color: var(--violet); }
.view-build .bd-cards { display: grid; gap: 12px; min-width: 0; }
.view-build .bd-ticket { display: grid; gap: 12px; border-top: 3px solid var(--line-strong); min-width: 0; container: bdcard / inline-size; }
/* a very wide column: two tickets side by side */
@container bdout (min-width: 940px) { .view-build .bd-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; } }
.view-build .bd-ticket.t-safe { border-top-color: var(--accent); }
.view-build .bd-ticket.t-risky { border-top-color: var(--warn); }
.view-build .bd-ticket.t-crazy { border-top-color: var(--violet); }
.view-build .bd-ticket.t-custom { border-top-color: var(--info); }
.view-build .bd-thead { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; min-width: 0; }
.view-build .bd-ttitle { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; min-width: 0; }
.view-build .bd-ttitle h3 { font-size: 16px; font-weight: 750; }
.view-build .bd-ttitle h3:focus { outline: none; }
.view-build .bd-ttitle h3:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.view-build .bd-ttitle h3 span { font-weight: 600; color: var(--text-2); }
.view-build .bd-odds { display: grid; justify-items: end; flex: none; }
.view-build .bd-odds b { font-size: 24px; font-weight: 800; line-height: 1.1; letter-spacing: -0.01em; }
.view-build .bd-odds small { font-size: 12.5px; color: var(--text-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.view-build .bd-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
@container bdcard (min-width: 420px) { .view-build .bd-stats { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); } }
.view-build .bd-stat { background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 8px 10px; min-width: 0; }
.view-build .bd-stat > span { display: flex; align-items: center; gap: 2px; color: var(--text-2); font-size: 12px; line-height: 1.3; }
.view-build .bd-stat > b { display: block; font-size: 18px; font-weight: 800; line-height: 1.3; margin-top: 2px; }
.view-build .bd-stat > small { display: block; color: var(--text-2); font-size: 11.5px; line-height: 1.35; margin-top: 2px; overflow-wrap: break-word; }
.view-build .bd-legs { list-style: none; display: grid; gap: 0; padding: 0; margin: 0; }
.view-build .bd-leg { display: grid; gap: 4px; padding: 10px 0; border-top: 1px solid var(--line); min-width: 0; }
.view-build .bd-leg:first-child { border-top: 0; padding-top: 0; }
.view-build .bd-leg.weak .bd-pick::after { content: ""; }
.view-build .bd-leg-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; min-width: 0; }
.view-build .bd-leg-top .bd-pick { font-size: 15px; min-width: 0; overflow-wrap: anywhere; }
.view-build .bd-price { display: grid; justify-items: end; flex: none; }
.view-build .bd-price b { font-size: 16px; }
.view-build .bd-price small { font-size: 11.5px; color: var(--text-2); }
.view-build .bd-leg-meta { font-size: 12.5px; color: var(--text-2); overflow-wrap: anywhere; }
.view-build .bd-leg-nums { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; font-size: 12.5px; color: var(--text-2); }
.view-build .bd-leg-nums b { color: var(--text); font-variant-numeric: tabular-nums; }
.view-build .bd-leg-nums small { color: var(--text-3); }
.view-build .bd-leg-nums .warn b { color: var(--warn-ink); }
.view-build .bd-leg-nums .icon { vertical-align: -2px; }
.view-build .bd-leg-nums .badge { font-size: 11px; padding: 1px 6px; }
.view-build .bd-why { font-size: 13px; color: var(--text-2); line-height: 1.45; margin: 2px 0 0; overflow-wrap: anywhere; }
.view-build .bd-why b { color: var(--text); font-weight: 650; }
.view-build .bd-risk b { color: var(--warn-ink); }
.view-build .bd-leg-actions { display: flex; align-items: center; gap: 8px; min-height: 36px; }
.view-build .bd-leg-actions .btn { padding: 0 6px; margin-left: -6px; }
.view-build .bd-leg-actions .bd-hint { margin: 0; display: inline-flex; align-items: center; gap: 4px; }
.view-build .bd-swaps { border: 1px solid var(--line); border-radius: 12px; padding: 8px; background: var(--card-2); }
.view-build .bd-swaps > .bd-hint { margin: 0 4px 6px; }
.view-build .bd-swaps ul { list-style: none; display: grid; gap: 6px; padding: 0; margin: 0; }
.view-build .bd-swap-opt {
  display: grid; gap: 2px; width: 100%; text-align: left; padding: 8px 10px; border-radius: 10px; background: var(--card); border: 1px solid var(--line-strong);
  min-height: var(--tap); color: var(--text);
}
.view-build .bd-swap-opt:hover:not(:disabled) { border-color: var(--accent); }
.view-build .bd-swap-opt:disabled { opacity: 0.6; }
.view-build .bd-swap-opt.back { border-style: dashed; }
.view-build .bd-so-top { display: flex; justify-content: space-between; gap: 8px; font-size: 14px; }
.view-build .bd-so-top .icon { vertical-align: -2px; }
.view-build .bd-so-meta { font-size: 12px; color: var(--text-2); overflow-wrap: anywhere; }
.view-build .bd-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.view-build .bd-actions .btn { flex: 1 1 150px; min-width: 0; padding: 0 10px; }
.view-build .bd-actions .btn:first-child { flex: 2 1 100%; }
.view-build .bd-actions .btn[aria-disabled="true"] { opacity: 0.6; cursor: default; }
@container bdcard (min-width: 560px) { .view-build .bd-actions .btn:first-child { flex-basis: 200px; } }
.view-build .bd-cost { margin-top: -4px; }
.view-build .bd-explain { border-top: 1px solid var(--line); padding-top: 10px; display: grid; gap: 8px; }
.view-build .bd-explain h4 { display: flex; align-items: center; gap: 6px; font-size: 14.5px; }
.view-build .bd-explain .bullets { font-size: 13.5px; }
.view-build .bd-misses > summary, .view-build .bd-unpriced > summary {
  list-style: none; cursor: pointer; display: flex; align-items: center; gap: 8px; min-height: var(--tap); font-weight: 650; font-size: 14.5px;
}
.view-build .bd-misses > summary::-webkit-details-marker, .view-build .bd-unpriced > summary::-webkit-details-marker { display: none; }
.view-build .bd-misses > summary > .icon:last-child, .view-build .bd-unpriced > summary > .icon:last-child { margin-left: auto; color: var(--text-2); transition: transform var(--speed); }
.view-build .bd-misses[open] > summary > .icon:last-child, .view-build .bd-unpriced[open] > summary > .icon:last-child { transform: rotate(180deg); }
.view-build .bd-miss-list { list-style: none; display: grid; gap: 0; padding: 0; margin: 8px 0 0; }
.view-build .bd-miss-list li { display: grid; gap: 2px; padding: 8px 0; border-top: 1px solid var(--line); }
.view-build .bd-miss-top { display: flex; justify-content: space-between; gap: 8px; font-size: 14px; min-width: 0; overflow-wrap: anywhere; }
.view-build .bd-miss-why { display: flex; align-items: flex-start; gap: 6px; font-size: 13px; color: var(--text); }
.view-build .bd-miss-why .icon { flex: none; color: var(--warn); margin-top: 2px; }
.view-build .bd-miss-list .bd-hint { margin: 0; }
.view-build .bd-unpriced { padding: 2px 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); font-size: 13px; }
.view-build .bd-unpriced > summary { font-size: 13.5px; font-weight: 600; color: var(--text-2); min-height: var(--tap); }
.view-build .bd-unpriced ul { margin: 0 0 4px 18px; display: grid; gap: 4px; color: var(--text-2); }
.view-build .bd-unpriced .bd-hint { margin-bottom: 8px; }
.view-build .bd-blocked .lead { font-size: 15px; }
.view-build .bd-rules { list-style: none; display: grid; gap: 0; padding: 0; margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.view-build .bd-rules li { display: flex; justify-content: space-between; gap: 12px; padding: 8px 12px; border-top: 1px solid var(--line); font-size: 13.5px; }
.view-build .bd-rules li:first-child { border-top: 0; }
.view-build .bd-rules span { min-width: 0; overflow-wrap: anywhere; }
.view-build .bd-blocked .chip { cursor: default; white-space: normal; text-align: left; }
.view-build .bd-ideas .bullets { font-size: 13.5px; }
/* closest effort: the option chips above a ticket, its fit line and what was relaxed */
.view-build .bd-opts { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: -4px; }
.view-build .bd-opt {
  display: inline-flex; align-items: center; gap: 4px; min-height: 26px; padding: 2px 10px 2px 8px; border-radius: 999px; max-width: 100%;
  font-size: 12.5px; font-weight: 700; line-height: 1.25; background: var(--accent-soft); color: var(--text); border: 1px solid color-mix(in srgb, var(--accent) 55%, transparent);
}
.view-build .bd-opt .icon { flex: none; color: var(--accent); }
.view-build .bd-opt.k-your_chance { background: var(--info-soft); border-color: color-mix(in srgb, var(--info) 55%, transparent); }
.view-build .bd-opt.k-your_chance .icon { color: var(--info-ink); }
.view-build .bd-opt.k-closest { background: var(--violet-soft); border-color: color-mix(in srgb, var(--violet) 55%, transparent); }
.view-build .bd-opt.k-closest .icon { color: var(--violet); }
.view-build .bd-fit { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; margin: 0; font-size: 13.5px; line-height: 1.4; }
.view-build .bd-fit > .icon { flex: none; color: var(--text-2); }
.view-build .bd-fit > span { min-width: 0; overflow-wrap: anywhere; }
.view-build .bd-fit b { font-weight: 750; }
.view-build .bd-fit .bd-asked { color: var(--text-2); }
.view-build .bd-fit .badge { font-size: 11px; padding: 1px 6px; }
.view-build .bd-relaxed { display: flex; align-items: flex-start; gap: 6px; margin: -4px 0 0; font-size: 12.5px; line-height: 1.45; color: var(--text-2); }
.view-build .bd-relaxed > .icon { flex: none; margin-top: 2px; }
.view-build .bd-relaxed > span { min-width: 0; overflow-wrap: anywhere; }
.view-build .bd-nothing .lead { font-size: 15px; }

/* the sheets: the groups fill the sheet's width; Telegram's main button is hidden while a sheet is open, so
   the sheet has its own Build */
.bd-sheet .sheet-body { padding-top: 4px; }
.bd-sheet .bd-group { background: var(--card); }
.bd-sheet .sheet-foot .btn { flex: 1 1 0; }
`;

/** Add the Builder's styles to the page (once). */
export function ensureBuilderStyles() {
  if (document.getElementById("builder-styles")) return;
  const style = document.createElement("style");
  style.id = "builder-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
