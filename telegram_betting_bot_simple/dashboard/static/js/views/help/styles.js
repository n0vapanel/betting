// Styles only the Help screen uses. styles.css holds the shared design system (tokens, cards, chips,
// sheets); these layout rules are added once, when Help first opens, and every one is scoped to
// .view-help so nothing here can change another screen. Colours are the shared tokens.

const CSS = `
.view-help .hp-search { position: relative; margin: 4px 0 4px; }
.view-help .hp-search .icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: var(--text-3); pointer-events: none; }
.view-help .hp-search .input { padding-left: 40px; }
.view-help .hp-search .input::-webkit-search-cancel-button { cursor: pointer; }
.view-help .hp-count { color: var(--text-2); font-size: 13px; margin-top: 10px; }

.view-help .hp-guides { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.view-help .hp-guide {
  display: flex; flex-direction: column; gap: 6px; min-height: 96px; padding: 12px; border-radius: var(--radius);
  background: var(--card); border: 1px solid var(--line); color: var(--text); min-width: 0;
}
.view-help .hp-guide:hover { text-decoration: none; border-color: var(--line-strong); }
.view-help .hp-guide .icon { color: var(--info); }
.view-help .hp-guide b { font-size: 15px; line-height: 1.25; }
.view-help .hp-guide span { color: var(--text-2); font-size: 13px; line-height: 1.35; }
.view-help .hp-guide.first { grid-column: 1 / -1; flex-direction: row; align-items: center; min-height: 64px; background: var(--accent-soft); border-color: var(--accent-line); }
.view-help .hp-guide.first .icon { color: var(--accent); }
.view-help .hp-guide.first > div { display: grid; gap: 2px; min-width: 0; flex: 1; }
.view-help .hp-guide.rg .icon { color: var(--warn); }

.view-help .hp-group + .hp-group { margin-top: 18px; }
.view-help .hp-group h3 { font-size: 12px; font-weight: 750; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-3); margin-bottom: 2px; }
.view-help .list-item .grow span { display: block; }
.view-help .list-item .hp-also { color: var(--text-3); font-size: 12px; }

.view-help .hp-lead { font-size: 16px; }
.view-help .hp-body p + p { margin-top: 10px; }
.view-help .hp-worked { counter-reset: step; list-style: none; display: grid; gap: 10px; margin: 0; padding: 0; }
.view-help .hp-worked li { counter-increment: step; display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 10px; align-items: start; }
.view-help .hp-worked li::before {
  content: counter(step); display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%;
  background: var(--info-soft); color: var(--info); font-weight: 750; font-size: 13px; margin-top: 1px;
}
.view-help .hp-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.view-help .hp-actions .btn { flex: 1 1 auto; }
.view-help .hp-related { margin-top: 4px; }

.view-help .hp-steps { list-style: none; display: grid; gap: 10px; margin: 0; padding: 0; counter-reset: step; }
.view-help .hp-step { display: grid; grid-template-columns: 40px minmax(0, 1fr); gap: 12px; align-items: start; }
.view-help .hp-step-icon {
  position: relative; display: grid; place-items: center; width: 40px; height: 40px; border-radius: 12px;
  background: var(--accent-soft); color: var(--accent);
}
.view-help .hp-step-icon b {
  position: absolute; right: -5px; top: -5px; width: 18px; height: 18px; border-radius: 50%; display: grid; place-items: center;
  background: var(--card); border: 1px solid var(--line-strong); color: var(--text); font-size: 11px;
}
.view-help .hp-step h3 { font-size: 15px; }
.view-help .hp-step p { color: var(--text-2); font-size: 14px; margin-top: 2px; }
.view-help .hp-step .text-link { position: relative; min-height: 36px; font-size: 14px; }
.view-help .hp-step .text-link::after { content: ""; position: absolute; inset: -4px 0; } /* a 44 px tap target */

.view-help .hp-points { display: grid; gap: 8px; margin: 10px 0 0; padding: 0; list-style: none; }
.view-help .hp-points li { display: grid; grid-template-columns: 20px minmax(0, 1fr); gap: 8px; align-items: start; font-size: 14px; }
.view-help .hp-points li > .icon { margin-top: 2px; color: var(--accent); }
.view-help .hp-points.is-warn li > .icon { color: var(--warn); }
.view-help .hp-points.is-info li > .icon { color: var(--info); }

.view-help .hp-quote { margin-top: 12px; padding: 12px 14px; border-radius: 12px; background: var(--card-2); border-left: 3px solid var(--accent); font-size: 14px; }

.view-help .hp-scens { display: grid; gap: 10px; }
.view-help .hp-scen { padding: 10px 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); }
.view-help .hp-scen-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: 2px 8px; }
.view-help .hp-scen-head span { color: var(--text-2); font-size: 13px; }
.view-help .hp-scen-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 8px 0 0; }
.view-help .hp-scen-grid > div { min-width: 0; }
.view-help .hp-scen-grid dt { color: var(--text-2); font-size: 12px; }
.view-help .hp-scen-grid dt span { color: var(--text-3); }
.view-help .hp-scen-grid dd { margin: 0; font-weight: 750; font-size: 14px; }
.view-help .hp-scen-grid dd.hp-ev { font-weight: 500; font-size: 12px; color: var(--text-2); }
@media (max-width: 540px) {
  .view-help .hp-scen-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

.view-help .hp-bars { display: grid; gap: 8px; margin-top: 10px; }
.view-help .hp-bar { display: grid; grid-template-columns: 52px minmax(0, 1fr) 48px; gap: 8px; align-items: center; font-size: 13px; }
.view-help .hp-bar > span:first-child { color: var(--text-2); }
.view-help .hp-bar > b { text-align: right; }
.view-help .hp-track { height: 10px; border-radius: 999px; background: var(--card-3); overflow: hidden; }
.view-help .hp-track > span { display: block; height: 100%; border-radius: 999px; background: var(--info); }

.view-help .hp-faq { display: grid; gap: 0; }
.view-help .hp-faq details { border-top: 1px solid var(--line); }
.view-help .hp-faq details:first-child { border-top: 0; }
.view-help .hp-faq summary {
  list-style: none; cursor: pointer; display: flex; align-items: center; gap: 10px; min-height: 52px; padding: 8px 0; font-weight: 650;
}
.view-help .hp-faq summary::-webkit-details-marker { display: none; }
.view-help .hp-faq summary .icon { margin-left: auto; color: var(--text-3); transition: transform var(--speed); }
.view-help .hp-faq details[open] summary .icon { transform: rotate(180deg); }
.view-help .hp-faq .hp-answer { padding: 0 0 14px; color: var(--text-2); font-size: 14px; }
.view-help .hp-faq .hp-answer a { font-weight: 600; }

.view-help .hp-lines { display: grid; gap: 10px; }
.view-help .hp-line { padding: 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); }
.view-help .hp-line .row { flex-wrap: wrap; gap: 6px 10px; margin-top: 6px; }
.view-help .hp-line .btn { min-height: 40px; }
.view-help .hp-limit { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 10px; align-items: start; padding: 10px 0; border-top: 1px solid var(--line); }
.view-help .hp-limit:first-child { border-top: 0; padding-top: 0; }
.view-help .hp-limit > .icon { color: var(--info); margin-top: 2px; }
.view-help .hp-limit p { color: var(--text-2); font-size: 14px; margin-top: 2px; }

.view-help .hp-empty { padding: 20px 4px; color: var(--text-2); text-align: center; }

@media (max-width: 359px) {
  .view-help .hp-guides { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) {
  .view-help .hp-faq summary .icon { transition: none; }
}
`;

/** Add the Help screen's styles to the page (once). */
export function ensureStyles() {
  if (document.getElementById("help-styles")) return;
  const style = document.createElement("style");
  style.id = "help-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
