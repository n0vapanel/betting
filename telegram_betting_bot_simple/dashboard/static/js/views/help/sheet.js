// A ⓘ button for the terms only Help has so far (ROI, Kelly, implied probability, AI vs market): it
// opens the same kind of short sheet as ui.js's shared explainer (one definition, one example, "More in
// Help"), so a screen can explain them in place. Terms the shared GLOSSARY has use its explain() as usual.

import { esc, sheet, explain, GLOSSARY } from "../../ui.js";
import { icon } from "../../icons.js";
import { term } from "./content.js";

/** An ⓘ button for any glossary term: the shared one when ui.js has it, else this one. */
export function termInfo(key, { label = "" } = {}) {
  if (GLOSSARY[key]) return explain(key, { label });
  const t = term(key);
  if (!t) return "";
  return `<button type="button" class="info-btn" data-help-term="${esc(key)}" aria-label="${esc(label || `What is ${t.title}?`)}">${icon("info", { size: 16 })}</button>`;
}

/** Open a Help-only term in a bottom sheet. */
export function openTerm(key) {
  const t = term(key);
  if (!t) return;
  sheet.open({
    title: t.title,
    body: `<p class="lead">${esc(t.short)}</p>
      <p>${esc(t.body)}</p>
      <div class="example"><span class="eyebrow">Example</span>${esc(t.example)}</div>
      <div class="explain-links">
        <a class="text-link" href="#/help/${esc(key)}">More in Help ${icon("chevron", { size: 16 })}</a>
        ${t.see ? `<a class="text-link" href="${esc(t.see)}">${esc(t.seeLabel)} ${icon("chevron", { size: 16 })}</a>` : ""}
      </div>`,
  });
}

/** Open Help-only terms from [data-help-term] buttons inside ``container``. */
export function bindTermInfo(container) {
  container.addEventListener("click", (event) => {
    const button = event.target.closest("[data-help-term]");
    if (!button) return;
    event.preventDefault();
    openTerm(button.dataset.helpTerm);
  });
}
