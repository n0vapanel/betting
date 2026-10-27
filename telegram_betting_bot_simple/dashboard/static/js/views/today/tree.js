// The league tree: "All sports", a group per sport and its leagues, each with a tri-state checkbox, the day's
// game count, a live dot and an odds-freshness dot; saved views ("presets") on top. On desktop it sits in
// the shell's left pane (mountTree); on tablet it is a drawer and on phone a tall sheet (openTreeDrawer),
// the same element each time, so its state and listeners carry over.
//
//   const tree = createTree({onSelect(selection, {only}), onPreset(preset), onSavePreset(name), onDeletePreset(i), onPin()})
//   tree.el                                     the element to mount
//   tree.update({prepared, counts, selection, pinned, presets, activePreset, liveCount})   redraw
//   selection: null for every league, else a Set of league keys (an empty Set: none)
//
// A checkbox adds or removes; the name beside it shows only that league (or sport): that is "opening" a
// league, which is when its odds are asked for. Native checkboxes, so keyboards and screen readers get the
// mixed state for free. Every outside text is escaped.

import { esc } from "../../ui.js";
import { icon } from "../../icons.js";
import { triState, groupName, groupIcon } from "./model.js";

const ODDS_WORDS = {
  fresh: "Odds up to date",
  stale: "Odds over 2 hours old",
  none: "No odds yet: they load when you open the league",
};

let seq = 0;

/** The live count beside a row's name: a pulsing dot and the number (words are in the row's label). */
const liveHtml = (n) => (n ? `<span class="lt-live" title="${esc(`${n} live now`)}" aria-hidden="true"><span class="live-dot"></span>${esc(n)}</span>` : "");

export function createTree({ onSelect, onPreset, onSavePreset, onDeletePreset, onPin }) {
  const id = `lt${++seq}`;
  const el = document.createElement("div");
  el.className = "lt";
  el.setAttribute("data-today-tree", "");
  const open = new Set(); // groups unfolded by the person
  let folded = new Set(); // groups folded by the person (the rest follow the default)
  let last = null;

  const allKeys = () => (last ? last.prepared.order : []);
  const groupKeys = (name) => (last?.prepared.groups.find((g) => g.key === name)?.leagues || []);

  function presetHtml({ presets, activePreset, pinned, liveCount, prepared }) {
    const builtIn = [{ key: "all", label: "All sports" }];
    if (pinned.length) builtIn.push({ key: "mine", label: "My sports" });
    builtIn.push({ key: "live", label: "Live now", count: liveCount });
    if (prepared.groups.some((g) => g.key === "Soccer")) builtIn.push({ key: "group:Soccer", label: "Soccer" });
    const button = (p) => `<button type="button" class="lt-preset" data-preset="${esc(p.key)}" aria-pressed="${activePreset === p.key}">
      ${p.key === "live" ? '<span class="live-dot" aria-hidden="true"></span>' : ""}${esc(p.label)}${p.count ? ` <span class="lt-pn">${esc(p.count)}</span>` : ""}</button>`;
    const saved = presets.map((p, i) => `<span class="lt-saved">${button({ key: `saved:${i}`, label: p.name })}<button type="button" class="lt-del" data-preset-del="${i}" aria-label="${esc(`Delete the saved view ${p.name}`)}" title="Delete">${icon("close", { size: 14 })}</button></span>`).join("");
    return `<div class="lt-presets" role="group" aria-label="Views">${builtIn.map(button).join("")}${saved}
        <button type="button" class="lt-preset lt-add" data-preset-save aria-expanded="false" aria-controls="${id}-save">${icon("plus", { size: 14 })}Save view</button></div>
      <form class="lt-save" id="${id}-save" data-preset-form hidden>
        <label class="sr-only" for="${id}-name">Name for this view</label>
        <input id="${id}-name" class="input" name="name" maxlength="24" placeholder="Name this view" autocomplete="off" enterkeyhint="done" required>
        <button type="submit" class="btn btn-primary small">Save</button>
      </form>`;
  }

  function dots(entry, counts) {
    const c = counts.get(entry.key) || { count: 0, live: 0 };
    const odds = entry.odds === "fresh" || entry.odds === "stale" ? entry.odds : "none";
    return `${liveHtml(c.live)}<span class="odds-dot ${odds}" title="${esc(ODDS_WORDS[odds])}" aria-hidden="true"></span>`;
  }

  function leagueRow(entry, counts, selection, pinned) {
    const c = counts.get(entry.key) || { count: 0, live: 0 };
    const on = !selection || selection.has(entry.key);
    const week = last?.prepared.week.get(entry.key) || 0;
    const words = [`${c.count} game${c.count === 1 ? "" : "s"} on the day shown`, c.live ? `${c.live} live` : "", `${week} in the next 7 days`, ODDS_WORDS[entry.odds] || ODDS_WORDS.none].filter(Boolean).join(", ");
    return `<li class="lt-row lt-league${c.count ? "" : " lt-zero"}" data-league="${esc(entry.key)}">
      <label class="lt-hit"><input type="checkbox" class="lt-box" data-lt-box="league" value="${esc(entry.key)}"${on ? " checked" : ""} aria-label="${esc(`Include ${entry.title}`)}"></label>
      <button type="button" class="lt-name" data-only="${esc(entry.key)}" aria-label="${esc(`Show only ${entry.title}: ${words}`)}" title="${esc(words)}">
        <span class="lt-text">${esc(entry.title)}${pinned.includes(entry.key) ? ` ${icon("star", { size: 12, className: "lt-pin", label: "pinned" })}` : ""}</span>
        <span class="lt-dots">${dots(entry, counts)}</span><span class="lt-n">${esc(c.count)}</span>
      </button>
    </li>`;
  }

  function draw(next) {
    last = next;
    const { prepared, counts, selection, pinned } = next;
    const pinnedFirst = (keys) => [...keys].sort((a, b) => (pinned.includes(a) ? 0 : 1) - (pinned.includes(b) ? 0 : 1));
    const groups = [...prepared.groups].sort((a, b) => (a.leagues.some((k) => pinned.includes(k)) ? 0 : 1) - (b.leagues.some((k) => pinned.includes(k)) ? 0 : 1));
    const total = [...counts.values()].reduce((n, c) => n + c.count, 0);
    const totalLive = [...counts.values()].reduce((n, c) => n + c.live, 0);
    const allState = triState(prepared.order, selection);
    const body = groups.map((g) => {
      const keys = g.leagues;
      const count = keys.reduce((n, k) => n + (counts.get(k)?.count || 0), 0);
      const live = keys.reduce((n, k) => n + (counts.get(k)?.live || 0), 0);
      const state = triState(keys, selection);
      // Open: groups the person unfolded, groups partly chosen, and small sports; folded on request
      const isOpen = !folded.has(g.key) && (open.has(g.key) || state === "mixed" || keys.length <= 3);
      const name = groupName(g.label);
      return `<li class="lt-group" data-group="${esc(g.key)}">
        <div class="lt-row lt-grow">
          <label class="lt-hit"><input type="checkbox" class="lt-box" data-lt-box="group" data-group="${esc(g.key)}"${state === "on" ? " checked" : ""} data-mixed="${state === "mixed"}" aria-label="${esc(`Include all ${name}`)}"></label>
          <button type="button" class="lt-name" data-only-group="${esc(g.key)}" aria-label="${esc(`Show only ${name}: ${count} game${count === 1 ? "" : "s"}${live ? `, ${live} live` : ""}`)}">
            <span class="lt-icon" aria-hidden="true">${esc(groupIcon(g.label))}</span><span class="lt-text">${esc(name)}</span>
            <span class="lt-dots">${liveHtml(live)}</span><span class="lt-n">${esc(count)}</span>
          </button>
          <button type="button" class="icon-btn lt-fold" data-fold="${esc(g.key)}" aria-expanded="${isOpen}" aria-controls="${id}-g-${esc(g.key.replace(/[^A-Za-z0-9_-]/g, "_"))}" aria-label="${esc(`${isOpen ? "Hide" : "Show"} ${name} leagues`)}">${icon("down", { size: 18 })}</button>
        </div>
        <ul class="lt-items" id="${id}-g-${esc(g.key.replace(/[^A-Za-z0-9_-]/g, "_"))}"${isOpen ? "" : " hidden"}>
          ${pinnedFirst(keys).map((k) => leagueRow(prepared.leagues.get(k), counts, selection, pinned)).join("")}
        </ul>
      </li>`;
    }).join("");
    const focusKey = el.contains(document.activeElement) ? focusId(document.activeElement) : null;
    const form = el.querySelector("[data-preset-form]");
    const formOpen = form && !form.hidden;
    const typed = form?.querySelector("input")?.value || "";
    el.innerHTML = `${presetHtml(next)}
      <div class="lt-row lt-all">
        <label class="lt-hit"><input type="checkbox" class="lt-box" data-lt-box="all"${allState === "on" ? " checked" : ""} data-mixed="${allState === "mixed"}" aria-label="Include every sport"></label>
        <button type="button" class="lt-name" data-only-all aria-label="${esc(`Show all sports: ${total} game${total === 1 ? "" : "s"}${totalLive ? `, ${totalLive} live` : ""}`)}">
          <span class="lt-text"><b>All sports</b></span><span class="lt-dots">${liveHtml(totalLive)}</span><span class="lt-n">${esc(total)}</span>
        </button>
      </div>
      <ul class="lt-groups">${body}</ul>
      <p class="lt-key small muted"><span class="odds-dot fresh" aria-hidden="true"></span> odds fresh · <span class="odds-dot stale" aria-hidden="true"></span> over 2 h · <span class="odds-dot none" aria-hidden="true"></span> load when opened</p>
      <button type="button" class="btn btn-ghost small lt-pins" data-pin-sports>${icon("star", { size: 16 })}${pinned.length ? "Change your pinned sports" : "Pin your sports to the top"}</button>`;
    for (const box of el.querySelectorAll('input[data-mixed="true"]')) box.indeterminate = true;
    if (formOpen) {
      const f = el.querySelector("[data-preset-form]");
      f.hidden = false;
      f.querySelector("input").value = typed;
      el.querySelector("[data-preset-save]").setAttribute("aria-expanded", "true");
    }
    if (focusKey) el.querySelector(focusKey)?.focus({ preventScroll: true });
  }

  /** A selector that finds the same control after a redraw. */
  function focusId(node) {
    const pick = (attr) => node.getAttribute?.(attr);
    for (const attr of ["data-only", "data-only-group", "data-fold", "data-preset", "data-preset-del"]) {
      const v = pick(attr);
      if (v != null) return `[${attr}="${CSS.escape(v)}"]`;
    }
    if (node.matches?.("[data-only-all]")) return "[data-only-all]";
    if (node.matches?.("[data-preset-save]")) return "[data-preset-save]";
    if (node.matches?.('[data-lt-box="all"]')) return '[data-lt-box="all"]';
    if (node.matches?.('[data-lt-box="group"]')) return `[data-lt-box="group"][data-group="${CSS.escape(node.dataset.group)}"]`;
    if (node.matches?.('[data-lt-box="league"]')) return `[data-lt-box="league"][value="${CSS.escape(node.value)}"]`;
    if (node.matches?.("[data-preset-form] input")) return "[data-preset-form] input";
    return null;
  }

  const current = () => (last?.selection ? new Set(last.selection) : new Set(allKeys()));
  const normal = (set) => (set.size === allKeys().length && allKeys().every((k) => set.has(k)) ? null : set);

  el.addEventListener("change", (event) => {
    const box = event.target.closest("[data-lt-box]");
    if (!box || !last) return;
    const kind = box.dataset.ltBox;
    let next;
    if (kind === "all") {
      next = box.checked ? null : new Set();
    } else {
      next = current();
      const keys = kind === "group" ? groupKeys(box.dataset.group) : [box.value];
      for (const key of keys) {
        if (box.checked) next.add(key);
        else next.delete(key);
      }
      next = normal(next);
    }
    onSelect(next, { only: false });
  });

  el.addEventListener("click", (event) => {
    const t = event.target;
    const only = t.closest("[data-only]");
    if (only) {
      onSelect(new Set([only.dataset.only]), { only: true });
      return;
    }
    const group = t.closest("[data-only-group]");
    if (group) {
      onSelect(normal(new Set(groupKeys(group.dataset.onlyGroup))), { only: true, group: group.dataset.onlyGroup });
      return;
    }
    if (t.closest("[data-only-all]")) {
      onSelect(null, { only: false });
      return;
    }
    const fold = t.closest("[data-fold]");
    if (fold) {
      const key = fold.dataset.fold;
      const willOpen = fold.getAttribute("aria-expanded") !== "true";
      if (willOpen) {
        open.add(key);
        folded.delete(key);
      } else {
        open.delete(key);
        folded.add(key);
      }
      fold.setAttribute("aria-expanded", String(willOpen));
      fold.setAttribute("aria-label", fold.getAttribute("aria-label").replace(/^(Show|Hide)/, willOpen ? "Hide" : "Show"));
      const list = el.querySelector(`#${CSS.escape(fold.getAttribute("aria-controls"))}`);
      if (list) list.hidden = !willOpen;
      return;
    }
    if (t.closest("[data-pin-sports]")) {
      onPin();
      return;
    }
    const del = t.closest("[data-preset-del]");
    if (del) {
      onDeletePreset(Number(del.dataset.presetDel));
      return;
    }
    const save = t.closest("[data-preset-save]");
    if (save) {
      const form = el.querySelector("[data-preset-form]");
      form.hidden = !form.hidden;
      save.setAttribute("aria-expanded", String(!form.hidden));
      if (!form.hidden) form.querySelector("input").focus();
      return;
    }
    const preset = t.closest("[data-preset]");
    if (preset) onPreset(preset.dataset.preset);
  });

  el.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-preset-form]");
    if (!form) return;
    event.preventDefault();
    const name = form.querySelector("input").value.trim().slice(0, 24);
    if (!name) return;
    form.querySelector("input").value = "";
    form.hidden = true;
    onSavePreset(name);
  });

  el.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && event.target.closest("[data-preset-form]")) {
      const form = el.querySelector("[data-preset-form]");
      form.hidden = true;
      el.querySelector("[data-preset-save]")?.setAttribute("aria-expanded", "false");
      el.querySelector("[data-preset-save]")?.focus();
      event.stopPropagation();
    }
  });

  return {
    el,
    update: draw,
    /** Forget the folds (a new person, or a reset). */
    resetFolds() {
      open.clear();
      folded = new Set();
    },
  };
}

/**
 * The collapsed 56 px rail: one icon per sport (and "All"), with its count and a live dot (layout.setTreeRail).
 */
export function railItems(prepared, counts, selection) {
  const items = [{
    key: "*", icon: "★", label: "All sports",
    count: [...counts.values()].reduce((n, c) => n + c.count, 0),
    live: [...counts.values()].reduce((n, c) => n + c.live, 0) || false,
    active: !selection,
  }];
  for (const g of prepared.groups) {
    const count = g.leagues.reduce((n, k) => n + (counts.get(k)?.count || 0), 0);
    const live = g.leagues.reduce((n, k) => n + (counts.get(k)?.live || 0), 0);
    const only = selection && selection.size === g.leagues.length && g.leagues.every((k) => selection.has(k));
    items.push({ key: g.key, icon: groupIcon(g.label) || groupName(g.label).slice(0, 2), label: groupName(g.label), count, live: live || false, active: Boolean(only) });
  }
  return items;
}
