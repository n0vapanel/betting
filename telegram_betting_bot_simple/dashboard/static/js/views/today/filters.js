// Today's toolbar: the title line ("updated N min ago", refresh, find a player), the phone's league chips,
// the date strip (today, tomorrow and the next five days, with game counts), the status filter, the team
// search and the sort. HTML strings for the parts that redraw (counts change with every slate); the search
// box and sort menu are drawn once so typing and focus are never interrupted.

import { esc, chip } from "../../ui.js";
import { icon } from "../../icons.js";
import { sportEmoji } from "../../app.js";

export const STATUS_LABELS = [
  { value: "all", label: "All" },
  { value: "upcoming", label: "Upcoming" },
  { value: "live", label: "Live" },
  { value: "finished", label: "Finished" },
];

export const SORT_LABELS = [
  { value: "time", label: "Time" },
  { value: "league", label: "League" },
  { value: "gap", label: "Biggest AI vs market gap" },
  { value: "market", label: "Market favourite" },
  { value: "ai", label: "AI favourite" },
];

/** The date strip: [{key, label, long}] with each day's count in the leagues chosen. */
export function daysHtml(strip, current, counts) {
  return strip.map((d) => {
    const n = counts.get(d.key) || 0;
    const on = d.key === current;
    return `<button type="button" class="td-day${on ? " on" : ""}${n ? "" : " zero"}" data-day="${esc(d.key)}" aria-pressed="${on}" aria-label="${esc(`${d.offset < 2 ? `${d.label}, ` : ""}${d.long}: ${n} game${n === 1 ? "" : "s"}`)}">
      <span class="td-day-l">${esc(d.label)}</span><span class="td-day-n num">${esc(n)}</span></button>`;
  }).join("");
}

/** All / Upcoming / Live (n) / Finished, as the shared segmented control. */
export function statusHtml(current, live) {
  return `<div class="seg small td-status" role="group" aria-label="Show games">${STATUS_LABELS.map((s) => {
    const on = s.value === current;
    const n = s.value === "live" && live ? ` <span class="td-live-n">${esc(live)}</span>` : "";
    return `<button type="button" data-status="${s.value}" aria-pressed="${on}"${s.value === "live" && live ? ` aria-label="${esc(`Live, ${live} now`)}"` : ""}>${s.value === "live" && live ? '<span class="live-dot" aria-hidden="true"></span>' : ""}${esc(s.label)}${n}</button>`;
  }).join("")}</div>`;
}

/**
 * The phone's league chips: All sports, the pinned leagues, then the busiest. The "Sports" chip after them
 * (it opens the tree) is drawn once by toolbarHtml, so it is still there to take focus back when the sheet
 * it opened closes after a change redrew these.
 */
export function chipsHtml(prepared, counts, selection, pinned, { max = 10 } = {}) {
  const only = selection && selection.size === 1 ? [...selection][0] : null;
  const keys = [];
  for (const key of pinned) if (prepared.leagues.has(key) && !keys.includes(key)) keys.push(key);
  const busy = [...prepared.order].filter((k) => !keys.includes(k) && (counts.get(k)?.count || 0) > 0)
    .sort((a, b) => (counts.get(b)?.live || 0) - (counts.get(a)?.live || 0) || (counts.get(b)?.count || 0) - (counts.get(a)?.count || 0));
  for (const key of busy) if (keys.length < max) keys.push(key);
  if (only && !keys.includes(only)) keys.unshift(only);
  const custom = selection && selection.size !== 1;
  return `${chip("All sports", { active: !selection, attrs: 'data-chip="all"' })}
    ${custom ? chip(`${selection.size} league${selection.size === 1 ? "" : "s"}`, { active: true, attrs: "data-chip-tree" }) : ""}
    ${keys.map((key) => {
      const entry = prepared.leagues.get(key);
      const c = counts.get(key) || { count: 0, live: 0 };
      const label = `${esc(sportEmoji(key))} ${esc(entry?.title || key)} <span class="td-chip-n">${esc(c.count)}</span>${c.live ? '<span class="live-dot" aria-hidden="true"></span>' : ""}`;
      return chip(label, { active: only === key, html: true, attrs: `data-chip="${esc(key)}" aria-label="${esc(`${entry?.title || key}: ${c.count} game${c.count === 1 ? "" : "s"}${c.live ? `, ${c.live} live` : ""}`)}"` });
    }).join("")}`;
}

/** The toolbar's frame; the parts marked data-* are redrawn as the slate and filters change. */
export function toolbarHtml(state, { idSearch, idSort }) {
  return `<div class="td-head">
      <div class="grow"><h1>Today</h1><p class="td-sub"><span data-td-sub></span> <span class="td-updated" data-td-updated aria-live="polite"></span></p></div>
      <button type="button" class="icon-btn" data-td-refresh aria-label="Refresh the games">${icon("refresh", { size: 20 })}</button>
      <button type="button" class="icon-btn" data-td-players aria-label="Find a player">${icon("user", { size: 20 })}</button>
      <button type="button" class="btn btn-secondary small td-tree-btn" data-chip-tree>${icon("menu", { size: 16 })}Sports</button>
    </div>
    <div class="chip-scroll td-chips" role="toolbar" aria-label="Leagues"><div class="td-chip-list" data-td-chips></div>
      <button type="button" class="chip add" data-chip-tree aria-label="All sports and leagues">${icon("menu", { size: 16 })}Sports</button></div>
    <div class="td-bar" data-td-bar>
      <div class="td-days" role="group" aria-label="Day" data-td-days></div>
      <div class="td-controls">
        <div data-td-status></div>
        <div class="td-search">
          <label class="sr-only" for="${idSearch}">Search teams</label>
          <span class="td-search-icon" aria-hidden="true">${icon("search", { size: 16 })}</span>
          <input id="${idSearch}" class="input" type="search" placeholder="Search teams" autocomplete="off" enterkeyhint="search" value="${esc(state.search)}" data-td-search>
        </div>
        <div class="td-sort">
          <label class="sr-only" for="${idSort}">Sort games by</label>
          <select id="${idSort}" class="input" data-td-sort>${SORT_LABELS.map((s) => `<option value="${s.value}"${state.sort.key === s.value ? " selected" : ""}>Sort: ${esc(s.label)}</option>`).join("")}</select>
        </div>
      </div>
    </div>
    <div class="td-notes" data-td-notes></div>`;
}
