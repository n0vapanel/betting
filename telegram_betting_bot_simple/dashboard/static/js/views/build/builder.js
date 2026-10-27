// The Builder (#/build/builder): Safe, Risky and Crazy as presets that fill every control, the controls
// themselves, and up to five tickets that share no leg, each saying why (v3 Stage 3).
//
// Server (dashboard/lab.py, engine/builder.py):
//   GET  /api/builder/presets       {presets: {safe, risky, crazy: controls}, described, markets: {key: {title,
//                                    available, reason}}, defaults: {preset, books, scope_fields, saved, ...}}
//   POST /api/builder/run           {controls, stake?} -> the engine's Result: tickets (priced at the stake, each
//                                    leg with why, risk, swaps), near_misses, blocked, stats, notes, ideas, target,
//                                    unpriced. Shared by everyone asking the same thing for a minute: no fetch.
//   POST /api/builder/price         {legs: [ref], stake?, books?} -> a ticket re-priced after a Swap (no rerun)
//   PUT  /api/me {builder}          the person's last controls (also kept on this device: prefs "builder")
//   POST /api/tickets/analyse       "Explain with AI" (the slip's analysis, paid from the AI allowance)
// Every route here runs the gates and the picks pause; on a break or in the pause the controls are off up
// front, with one line saying until when (common.js picksBlock).
//
// Layouts: from 840 px the controls are a left column of collapsible groups (sticky, with Build at its
// foot) and the tickets fill the right; the slip rail stays on at 1200 px and up (layout.js railShown).
// On a phone the preset chips and three key controls (sports, legs, target odds) sit on top, every other
// control is in the "More controls" sheet, and the tickets follow below. Inside Telegram the main button
// is "Build tickets".
//
// "Custom": any control that changes the risk (not the scope: sports, time, books, tickets) and differs from
// the preset the controls came from relabels the run Custom, as engine/presets.py label_for does.
// Swap: a leg's alternatives were worked out by the server with the other legs fixed, so after one swap only
// that leg's alternatives (and going back) stay on offer until the next build.

import { api, prefs, isQuiet } from "../../api.js";
import {
  me, loadMe, setMe, slip, layout, onLayoutChange, onMeChange, navigate, openOnboarding, pinnedSports, swr, SLATE_URL,
  SLATE_KEY, setSlateLeagues, inTelegram, replaceAddress, onPriced, priced, legKey,
} from "../../app.js";
import { esc, skeleton, errorState, toast, sheet, emptyState, priceText } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { prepare } from "../today/model.js";
import { createTree } from "../today/tree.js";
import { ensureStyles as ensureTreeStyles } from "../today/styles.js";
import { slipLeg, gateHint, picksBlock, blockNote, reveal } from "./common.js";
import {
  DEFAULTS, SCOPE_FIELDS, PRESET_KEYS, PRESET_TITLES, cleanControls, cleanUi, labelFor, changedFields, withPreset,
  windowFor, windowProblem, runBody, savedBody, restore, targetFrom, backFrom, windowCounts, numText,
} from "./builder/model.js";
import { presetChipsHtml, presetPressed, presetStatusHtml, keyHtml, groupsHtml, syncSummaries, readChange } from "./builder/form.js";
import { resultHtml } from "./builder/results.js";
import { ensureBuilderStyles } from "./builder/styles.js";

const LOCAL_KEY = "builder";
const BLOCK_ID = "bd-block";
const SAVE_DELAY = 1500;

// Kept for the visit: coming back to the Builder shows the last tickets at once
let memory = { setup: null, run: null };

export function mountBuilder(root, { params, refresh }) {
  ensureBuilderStyles();
  ensureTreeStyles();
  const signal = params.signal;
  let setup = memory.setup; // {presets, described, markets, books, scope}
  let s = null; // {c, ui, errors}
  let run = memory.run; // {data, at, body, cards}
  let busy = false;
  let controller = null;
  let block = picksBlock();
  let phone = layout.mode === "phone";
  let prepared = null;
  let more = null; // the "More controls" sheet's handle
  let saveTimer = null;
  let dirty = false; // controls changed since the server last kept them
  let touched = false; // the person changed the controls on this visit (a newer copy from another device then waits)
  let ranWith = run?.key ?? null; // the controls the tickets shown were built with (currentKey())
  const stops = [];

  root.innerHTML = `<div class="page-head bd-head"><h1>${icon("build", { size: 22 })}Builder</h1>
      <p>Safe, Risky and Crazy fill every control; change any and the run is yours (Custom). Every number comes from the brain, and nothing is bet here.</p></div>
    <div data-bd-block></div>
    <div class="bd-layout" data-bd-layout>${skeleton("lines", 4)}</div>`;
  const $ = (sel) => root.querySelector(sel);

  // --- the league tree (Today's component, in the Scope group or a sheet) -------------------------------
  const tree = createTree({
    views: false, pins: false, countWords: "in your time window", only: "Only",
    onSelect(selection) {
      // The container holding the tree isn't redrawn (the tree redraws itself and keeps the focus)
      const home = tree.el.closest("[data-bd-ctl]") || (more?.open && more.body.contains(tree.el) ? more.body : null);
      apply({ leagues: selection ? [...selection] : null }, null, home);
      if (selection && !selection.size) setError("leagues", "Pick at least one sport.");
      else setError("leagues", "");
    },
  });
  const treeHost = document.createElement("div");
  treeHost.className = "bd-tree-wait muted small";
  treeHost.textContent = "Loading the sports…";

  const counts = () => windowCounts(prepared, windowFor(s?.ui));
  const drawTree = () => {
    if (!prepared || !s) return;
    tree.update({ prepared, counts: counts(), selection: s.c.leagues ? new Set(s.c.leagues) : null, pinned: pinnedSports(), presets: [], activePreset: null, liveCount: 0 });
  };
  const placeTree = (container) => {
    const host = container?.querySelector("[data-bd-tree]");
    if (host) host.replaceChildren(prepared ? tree.el : treeHost);
  };

  // --- state ----------------------------------------------------------------------------------------------
  const scope = () => setup?.scope || SCOPE_FIELDS;
  const view = () => ({
    c: s.c, ui: s.ui, presets: setup.presets, described: setup.described, markets: setup.markets, books: setup.books,
    counts: counts(), label: labelFor(s.c, setup.presets, scope()), changed: changedFields(s.c, setup.presets, scope()),
    errors: s.errors,
  });

  function saveLocal() {
    prefs.set(LOCAL_KEY, { controls: s.c, ui: { ...s.ui, at: new Date().toISOString() } });
  }

  let lastSaved = null; // what the server last kept (without its time), so an unchanged run saves nothing
  function saveServer({ now = false } = {}) {
    clearTimeout(saveTimer);
    if (!dirty || block || !s) return;
    const go = () => {
      dirty = false;
      const body = savedBody(s.c, s.ui);
      const key = JSON.stringify({ ...body, window: null, ui: { ...body.ui, at: null } });
      if (key === lastSaved) return;
      api("/api/me", { method: "PUT", body: { builder: body } })
        .then((profile) => {
          lastSaved = key;
          if (profile?.uid) setMe(profile);
        })
        .catch(() => { dirty = true; }); // kept on this device anyway; tried again with the next change
    };
    if (now) go();
    else saveTimer = setTimeout(go, SAVE_DELAY);
  }

  // --- drawing ------------------------------------------------------------------------------------------------
  function controlsParts() {
    return { presets: $("[data-bd-presets]"), ctl: $("[data-bd-ctl]"), sheetBody: more?.open ? more.body : null };
  }

  /**
   * Replace a container's HTML, keeping focus on the same control (by id, or by its data attribute). The same
   * HTML again leaves the elements alone (a tap landing on them mid-redraw isn't lost).
   */
  const drawn = new WeakMap();
  function redrawBox(el, html) {
    if (!el) return;
    if (drawn.get(el) === html && el.firstChild) {
      placeTree(el);
      return;
    }
    drawn.set(el, html);
    const active = document.activeElement;
    let sel = null;
    if (active && el.contains(active)) {
      if (active.id) sel = `#${CSS.escape(active.id)}`;
      else for (const attr of ["data-bd-preset", "data-bd-win", "data-bd-reset", "data-bd-sports"]) {
        if (active.hasAttribute(attr)) {
          sel = `[${attr}="${CSS.escape(active.getAttribute(attr))}"]`;
          break;
        }
      }
    }
    // Groups keep the person's open and closed state across a redraw
    const had = el.querySelector("details[data-bd-group]");
    const open = new Set([...el.querySelectorAll("details[data-bd-group][open], details.bd-adv[open]")].map((d) => d.dataset.bdGroup || "adv"));
    el.innerHTML = html;
    if (had) for (const d of el.querySelectorAll("details[data-bd-group], details.bd-adv")) d.open = open.has(d.dataset.bdGroup || "adv");
    placeTree(el);
    if (sel) el.querySelector(sel)?.focus({ preventScroll: true });
  }

  function ctlHtml(v) {
    return phone ? keyHtml(v, "bdk") : groupsHtml(v, "bdd", { open: ["scope", "shape"] });
  }

  function drawControls({ except = null } = {}) {
    if (!s || !setup) return;
    const v = view();
    const parts = controlsParts();
    // The chips and the Build button stay the same elements: a change read on blur (the person typed, then
    // tapped a chip or Build) mustn't swap the element under the pointer and lose the tap
    if (parts.presets) {
      if (!parts.presets.querySelector("[data-bd-preset]")) parts.presets.innerHTML = `${presetChipsHtml(v)}<div data-bd-pstatus></div>`;
      presetPressed(parts.presets, v);
      redrawBox(parts.presets.querySelector("[data-bd-pstatus]"), presetStatusHtml(v));
    }
    // The container being typed in keeps its inputs (only its summaries follow); what it shows is no longer
    // what was last drawn there, so its next redraw always happens
    if (parts.ctl && parts.ctl !== except) redrawBox(parts.ctl, ctlHtml(v));
    else if (parts.ctl) {
      drawn.delete(parts.ctl);
      syncSummaries(parts.ctl, v);
    }
    if (parts.sheetBody && parts.sheetBody !== except) redrawBox(parts.sheetBody, groupsHtml(v, "bds", { open: [] }));
    else if (parts.sheetBody) {
      drawn.delete(parts.sheetBody);
      syncSummaries(parts.sheetBody, v);
    }
    drawTree();
    drawRunBar();
  }

  /** The run bar: drawn once per layout, then updated in place (see drawControls on lost taps). */
  function drawRunBar() {
    const bar = $("[data-bd-run]");
    if (!bar) return;
    if (!bar.querySelector("[data-bd-error]")) {
      bar.innerHTML = `${phone ? `<button type="button" class="btn btn-secondary" data-bd-more aria-haspopup="dialog">${icon("sliders", { size: 18 })}More controls</button>` : ""}
        ${inTelegram && phone ? "" : '<button type="button" class="btn btn-primary" data-bd-build></button>'}
        <div class="form-error" role="alert" data-bd-error></div>
        <p class="bd-hint bd-stale-note" data-bd-stale hidden>${icon("info", { size: 14 })} The tickets below are for your controls before these changes.</p>`;
    }
    const buildButton = bar.querySelector("[data-bd-build]");
    if (buildButton) {
      buildButton.disabled = Boolean(block || busy);
      if (block) buildButton.setAttribute("aria-describedby", BLOCK_ID);
      else buildButton.removeAttribute("aria-describedby");
      const state = block ? "off" : busy ? "busy" : "ready";
      if (buildButton.dataset.state !== state) {
        buildButton.innerHTML = `${icon(block ? "pause" : busy ? "clock" : "build", { size: 18 })}${busy ? "Building…" : "Build tickets"}`;
        buildButton.dataset.state = state;
      }
    }
    const moreButton = bar.querySelector("[data-bd-more]");
    if (moreButton) {
      moreButton.disabled = Boolean(block);
      if (block) moreButton.setAttribute("aria-describedby", BLOCK_ID);
      else moreButton.removeAttribute("aria-describedby");
    }
    const error = bar.querySelector("[data-bd-error]");
    if (error.textContent !== (s?.formError || "")) {
      error.textContent = s?.formError || "";
      if (!run && !busy) drawResults(); // with no tickets yet, the results say what to fix too
    }
    bar.querySelector("[data-bd-stale]").hidden = !(run && ranWith !== null && ranWith !== currentKey() && !busy && !block);
  }

  function cardsFor(data) {
    return (data.tickets || []).map((t) => ({ original: t, t, swapped: null, explain: null, open: null }));
  }

  function inSlipNow(card) {
    const refs = card.t.refs || (card.t.legs || []).map((l) => l.ref);
    return refs.length > 0 && refs.every((ref) => ref && slip.has(ref));
  }

  function drawResults({ focus = null } = {}) {
    const out = $("[data-bd-out]");
    if (!out) return;
    if (block) {
      out.innerHTML = emptyState({ iconName: "pause", title: "The Builder is off for now", body: block.text,
        actions: [{ label: "Responsible play", href: "#/help/responsible" }] });
      return;
    }
    if (!run) {
      out.innerHTML = busy ? `<div aria-busy="true">${skeleton("cards", 2)}<p class="muted small" style="margin-top:8px">Searching the saved prices for tickets that fit…</p></div>`
        : s?.formError ? emptyState({ iconName: "alert", title: "Fix the controls first", body: s.formError.replace(/^Fix first: /, "") })
          : emptyState({ iconName: "build", title: "Pick a preset or set the controls", body: "Then Build tickets. Up to five tickets that share no leg, each with the reasons behind every leg." });
      return;
    }
    for (const card of run.cards) card.inSlip = inSlipNow(card);
    const html = resultHtml({ result: run.data, cards: run.cards, at: run.at, busy });
    // Focus stays on the same control, else on what took its place in the same card (Add to slip becomes
    // "In your slip", a picked swap gives way to that leg's Swap button), else on the card's heading: never on
    // another card's button, where a second Enter would act on a different ticket
    const active = document.activeElement;
    let sel = focus;
    let attr = null;
    let cardIndex = null;
    if (!sel && active && out.contains(active)) {
      cardIndex = active.closest("[data-bd-card]")?.dataset.bdCard ?? null;
      for (const name of ["data-bd-swap", "data-bd-add", "data-bd-added", "data-bd-explain", "data-bd-ask", "data-bd-pick", "data-bd-undo"]) {
        if (active.hasAttribute(name)) {
          attr = name;
          sel = `[${name}="${CSS.escape(active.getAttribute(name))}"]`;
          break;
        }
      }
      if (!sel && active.matches(".bd-misses > summary, .bd-unpriced > summary")) sel = active.parentElement.matches(".bd-misses") ? ".bd-misses > summary" : ".bd-unpriced > summary";
    }
    const misses = out.querySelector(".bd-misses")?.open;
    const unpriced = out.querySelector(".bd-unpriced")?.open;
    out.innerHTML = html;
    if (misses) out.querySelector(".bd-misses").open = true;
    if (unpriced) out.querySelector(".bd-unpriced").open = true;
    if (!sel && cardIndex === null) return;
    const card = cardIndex !== null ? out.querySelector(`[data-bd-card="${CSS.escape(cardIndex)}"]`) : null;
    const target = (sel && out.querySelector(sel))
      || (card && attr === "data-bd-add" && card.querySelector("[data-bd-added]"))
      || (card && (attr === "data-bd-pick" || attr === "data-bd-undo") && card.querySelector('[data-bd-swap][aria-expanded="true"]'))
      || card?.querySelector(".bd-ttitle h3");
    target?.focus({ preventScroll: true });
  }

  function drawLayout() {
    phone = layout.mode === "phone";
    const box = $("[data-bd-layout]");
    if (!setup || !s) return;
    box.className = `bd-layout ${phone ? "is-phone" : "is-wide"}`;
    box.innerHTML = `<aside class="bd-controls" aria-label="Builder controls">
        <div class="bd-scroll"><div data-bd-presets></div><div data-bd-ctl></div></div>
        <div class="bd-run" data-bd-run></div>
      </aside>
      <section class="bd-out" data-bd-out aria-label="Tickets"></section>
      <p class="sr-only" role="status" data-bd-say></p>`;
    if (!phone && more?.open) more.close();
    drawControls();
    drawResults();
    fitColumn();
  }

  /**
   * Desktop: the controls column is as tall as the window below where it starts, so its run bar (Build and its
   * error line) is on screen from the first view, not only once the page has scrolled the column up to stick.
   */
  let fitFrame = 0;
  function fitColumn() {
    const aside = root.querySelector(".is-wide > .bd-controls");
    if (!aside) return;
    const top = `${Math.round(aside.getBoundingClientRect().top)}px`;
    if (aside.style.getPropertyValue("--bd-top") !== top) aside.style.setProperty("--bd-top", top);
  }
  const fitSoon = () => {
    if (fitFrame || phone) return;
    fitFrame = requestAnimationFrame(() => {
      fitFrame = 0;
      if (root.isConnected) fitColumn();
    });
  };

  function setError(field, text) {
    if (!s) return;
    const errors = { ...s.errors };
    if (text) errors[field] = text;
    else delete errors[field];
    const changed = (s.errors[field] || "") !== (text || "");
    s.errors = errors;
    if (!changed) return;
    for (const box of [$("[data-bd-ctl]"), more?.open ? more.body : null]) if (box) drawn.delete(box);
    // Only the error lines (inputs keep what the person is typing)
    for (const el of root.querySelectorAll(`[id$="-${CSS.escape(field)}-err"]`)) el.textContent = text || "";
    if (more?.open) for (const el of more.body.querySelectorAll(`[id$="-${CSS.escape(field)}-err"]`)) el.textContent = text || "";
    for (const input of [...root.querySelectorAll(`[data-bd="${field}"]`), ...(more?.open ? more.body.querySelectorAll(`[data-bd="${field}"]`) : [])]) {
      if (input.type === "checkbox") continue;
      if (text) {
        input.setAttribute("aria-invalid", "true");
        const errId = input.id ? `${input.id}-err` : "";
        if (errId) input.setAttribute("aria-describedby", errId);
      } else {
        input.removeAttribute("aria-invalid");
      }
    }
  }

  // --- changing the controls ---------------------------------------------------------------------------------
  /**
   * Apply changes to the controls and/or the ui state. ``source`` is the container the person is typing in:
   * it isn't redrawn (only its summaries and the inputs that follow from the change).
   */
  function apply(controls, ui, source, { field = null } = {}) {
    if (!s) return;
    if (controls) s.c = { ...s.c, ...controls };
    if (ui) s.ui = { ...s.ui, ...ui };
    // Stake and "win about" set the target; a target typed with a stake set moves "win about"
    let follow = null;
    if (ui && ("stake" in ui || "back" in ui)) {
      const t = s.ui.stake && s.ui.back ? targetFrom(s.ui.stake, s.ui.back) : null;
      if (t) {
        s.c = { ...s.c, target_odds: t };
        follow = "target_odds";
      }
    } else if (controls && "target_odds" in controls && s.ui.stake) {
      s.ui = { ...s.ui, back: s.c.target_odds ? backFrom(s.ui.stake, s.c.target_odds) : null };
      follow = "back";
    }
    if (field !== "markets" && field !== "books") s.formError = "";
    dirty = true;
    touched = true;
    saveLocal();
    saveServer();
    drawControls({ except: source });
    if (source && follow) {
      const input = source.querySelector(`[data-bd="${follow}"]`);
      if (input && document.activeElement !== input) input.value = follow === "target_odds" ? numText(s.c.target_odds) : s.ui.back || "";
    }
    refresh();
  }

  function onControlEvent(event, container) {
    const target = event.target;
    if (!target?.dataset?.bd) return;
    drawn.delete(container); // what the person typed is no longer what was drawn: the next redraw must happen
    const typing = event.type === "input";
    if (typing && (target.type === "checkbox" || target.tagName === "SELECT")) return; // "change" handles those
    const r = readChange(target, view());
    if (!r) return;
    if (r.error) {
      if (!typing) setError(r.field, r.error);
      if (r.field === "markets" || r.field === "books") target.checked = true; // the last one can't go
      return;
    }
    setError(r.field, "");
    apply(r.controls || null, r.ui || null, container, { field: r.field });
  }

  function onControlClick(event) {
    const t = event.target;
    const preset = t.closest("[data-bd-preset]");
    if (preset) {
      choosePreset(preset.dataset.bdPreset, { run: true });
      return true;
    }
    if (t.closest("[data-bd-reset]")) {
      choosePreset(s.c.preset, { run: false });
      return true;
    }
    const win = t.closest("[data-bd-win]");
    if (win) {
      const mode = win.dataset.bdWin;
      const ui = { win: mode };
      if (mode === "custom" && !(s.ui.from && s.ui.to)) {
        const w = windowFor({ win: "24h" });
        ui.from = w.from;
        ui.to = w.to;
      }
      s.ui = { ...s.ui, ...ui };
      dirty = true;
      touched = true;
      saveLocal();
      saveServer();
      drawControls();
      refresh();
      return true;
    }
    if (t.closest("[data-bd-sports]")) {
      openSports();
      return true;
    }
    if (t.closest("[data-bd-more]")) {
      openMore();
      return true;
    }
    if (t.closest("[data-bd-build]")) {
      if (more?.open) more.close();
      runBuild();
      return true;
    }
    return false;
  }

  function choosePreset(name, { run: go = false } = {}) {
    if (!setup.presets[name]) return;
    s.c = withPreset(name, s.c, setup.presets, scope());
    s.errors = {};
    s.formError = "";
    // A target from the stake belongs to the old controls
    s.ui = { ...s.ui, back: null };
    dirty = true;
    touched = true;
    haptic.select();
    saveLocal();
    saveServer();
    drawControls();
    refresh();
    if (go) runBuild();
  }

  // --- sheets (phone) ---------------------------------------------------------------------------------------
  function openMore() {
    if (block) return;
    const v = view();
    more = sheet.open({
      title: "More controls",
      size: "tall",
      className: "bd-sheet",
      body: groupsHtml(v, "bds", { open: [] }), // six short rows: the person opens what they need
      footer: `<button type="button" class="btn btn-secondary grow" data-bd-done>Done</button>
        <button type="button" class="btn btn-primary grow" data-bd-sheet-build>${icon("build", { size: 18 })}Build tickets</button>`,
      onClose: () => {
        more = null;
        if (root.isConnected) {
          drawControls();
          refresh();
        }
      },
    });
    placeTree(more.body);
    drawTree();
    const handle = more;
    handle.body.addEventListener("change", (e) => onControlEvent(e, handle.body));
    handle.body.addEventListener("input", (e) => onControlEvent(e, handle.body));
    handle.body.addEventListener("click", (e) => { onControlClick(e); });
    handle.foot.addEventListener("click", (e) => {
      if (e.target.closest("[data-bd-done]")) handle.close();
      else if (e.target.closest("[data-bd-sheet-build]")) {
        handle.close().then(() => runBuild());
      }
    });
    refresh();
  }

  function openSports() {
    if (block) return;
    const body = document.createElement("div");
    body.className = "bd-tree bd-tree-sheet";
    body.appendChild(prepared ? tree.el : treeHost);
    drawTree();
    const handle = sheet.open({
      title: "Sports and leagues",
      size: "tall",
      className: "bd-sheet",
      body,
      footer: `<button type="button" class="btn btn-primary grow" data-bd-done>Done</button>`,
      onClose: () => {
        if (root.isConnected) drawControls();
      },
    });
    handle.foot.addEventListener("click", (e) => {
      if (e.target.closest("[data-bd-done]")) handle.close();
    });
  }

  // --- building ------------------------------------------------------------------------------------------------
  const currentKey = () => (s ? JSON.stringify({ ...runBody(s.c, s.ui), window: null, tz: null, win: s.ui.win, from: s.ui.win === "custom" ? s.ui.from : null, to: s.ui.win === "custom" ? s.ui.to : null, stake: s.ui.stake }) : null);

  function problems() {
    const out = [];
    if (Array.isArray(s.c.leagues) && !s.c.leagues.length) out.push("Pick at least one sport.");
    const w = windowProblem(s.ui);
    if (w) out.push(w);
    for (const text of Object.values(s.errors || {})) if (text) out.push(text);
    return [...new Set(out)];
  }

  async function runBuild({ show = true } = {}) {
    if (!s || !setup || block) return;
    const issues = problems();
    if (issues.length) {
      s.formError = `Fix first: ${issues.join(" ")}`;
      drawRunBar();
      if (phone || run) toast(issues[0], { kind: "warn" }); // with no tickets yet, the results pane says it too
      return;
    }
    s.formError = "";
    const body = { controls: runBody(s.c, s.ui) };
    if (s.ui.stake) body.stake = s.ui.stake;
    controller?.abort();
    controller = new AbortController();
    const mine = controller;
    signal.addEventListener("abort", () => mine.abort(), { once: true });
    busy = true;
    drawRunBar();
    drawResults();
    refresh();
    const key = currentKey();
    try {
      const data = await api("/api/builder/run", { method: "POST", body, signal: mine.signal });
      if (mine !== controller) return;
      run = { data, at: Date.now(), body, key, cards: cardsFor(data) };
      memory.run = run;
      ranWith = key;
      haptic.notify((data.tickets || []).length ? "success" : "warning");
      saveServer({ now: true }); // the controls just used are the person's last (when they changed)
    } catch (error) {
      if (isQuiet(error) || mine !== controller || !root.isConnected) return;
      haptic.notify("error");
      showRunError(error);
      return;
    } finally {
      if (mine === controller) {
        busy = false;
        controller = null;
        if (root.isConnected) {
          drawRunBar();
          refresh();
        }
      }
    }
    if (!root.isConnected) return;
    drawResults();
    const n = (run.data.tickets || []).length;
    const say = $("[data-bd-say]");
    if (say) say.textContent = n ? `${n} ticket${n === 1 ? "" : "s"} built, ${run.data.label || "Custom"}.` : `No ticket fits: ${run.data.blocked?.message || "see the rules to relax"}`;
    if (phone && show) reveal($("[data-bd-out]"));
  }

  function showRunError(error) {
    const out = $("[data-bd-out]");
    const hint = gateHint(error);
    if (hint?.kind === "break" || hint?.kind === "paused") {
      loadMe().catch(() => null); // sync() below switches everything off and says until when
    }
    if (!out) return;
    if (error.status === 404) {
      out.innerHTML = emptyState({ iconName: "clock", title: "The Builder isn't on this server yet", body: "It needs the server's latest update." });
      return;
    }
    out.innerHTML = `${errorState(error, { retry: false, title: "Couldn't build tickets" })}
      <div class="empty-actions">${hint?.kind === "age" ? '<button type="button" class="btn btn-primary" data-bd-age>Confirm age</button>'
        : hint ? "" : `<button type="button" class="btn btn-secondary" data-bd-build>${icon("refresh", { size: 16 })}Try again</button>`}</div>`;
  }

  // --- results: swaps, slip, explain, ask ---------------------------------------------------------------------
  const plainStake = (v) => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? String(Math.round(n * 100) / 100) : "10";
  };

  async function pickSwap(ti, li, k) {
    const card = run?.cards[ti];
    const base = card?.original;
    const option = base?.legs?.[li]?.swaps?.[k];
    if (!card || !option || card.repricing) return;
    const refs = (base.legs || []).map((leg, i) => (i === li ? option.ref : leg.ref));
    const body = { legs: refs, stake: plainStake(base.unit_stake) };
    // The run's own books and target (the controls on screen may have changed since): the server adds the joint
    // chance, its fair odds and the chance of reaching the target, as the run's tickets carry them
    const ran = run.data.controls || {};
    if (Array.isArray(ran.books) && ran.books.length) body.books = ran.books;
    if (run.data.target?.odds) {
      body.target_odds = run.data.target.odds;
      if (run.data.target.tolerance_pct != null) body.tolerance_pct = run.data.target.tolerance_pct;
    }
    card.repricing = true;
    try {
      const priced = await api("/api/builder/price", { method: "POST", body, signal });
      if (!root.isConnected || run?.cards[ti] !== card) return;
      if ((priced.removed || []).length) {
        toast("A leg is no longer on offer: build again for fresh tickets", { kind: "warn" });
        return;
      }
      card.t = swappedTicket(base, priced, li, option);
      card.swapped = { index: li, key: option.key };
      card.open = li;
      card.explain = null;
      haptic.select();
      toast(`Swapped in ${option.label}`, { kind: "good" });
    } catch (error) {
      if (isQuiet(error) || !root.isConnected) return;
      toast(error.message || "Couldn't re-price the ticket", { kind: "error" });
      if (gateHint(error)) loadMe().catch(() => null);
    } finally {
      card.repricing = false;
      // The option picked is gone from the list (or the list stays as it was): focus goes to that leg's Swap
      // button, not to the top of the page
      const out = $("[data-bd-out]");
      const here = out && (out.contains(document.activeElement) || document.activeElement === document.body);
      if (root.isConnected) drawResults(here ? { focus: `[data-bd-swap="${ti}:${li}"]` } : {});
    }
  }

  /** The ticket after a swap: the server's re-pricing, with each kept leg's reasons and the new leg's swap data. */
  function swappedTicket(base, priced, li, option) {
    const legs = (priced.legs || []).map((pl, i) => {
      const old = base.legs.find((l) => sameRef(l.ref, pl.ref || refOf(pl)));
      if (old && i !== li) return { ...old, ...pl, edge: old.edge, why: old.why, why_units: old.why_units, risk: old.risk, swaps: old.swaps, chance: old.chance, chance_source: old.chance_source, age: old.age, flags: old.flags };
      return { ...pl, ref: option.ref, chance: option.chance, edge: option.edge, age: option.age, flags: option.flags, why: option.why || [], why_units: option.why_units || null, risk: option.risk || null, swaps: base.legs[li].swaps, chance_source: option.chance_source || null };
    });
    return {
      ...priced, legs, refs: priced.refs || legs.map((l) => l.ref), label: "Custom",
      joint: priced.joint || { chance: priced.chance?.ai ?? option.joint, one_in: priced.chance?.ai_one_in, fair_odds: null },
      reach_target: priced.reach_target || (run.data.target ? null : { chance: priced.chance?.ai ?? option.joint, one_in: priced.chance?.ai_one_in }),
      notes: Array.isArray(priced.notes) ? priced.notes : priced.model_priced ? ["Priced at fair odds; check your bookmaker's price."] : [],
    };
  }

  const refOf = (v) => ({ league: v.league, id: String(v.event_id ?? v.id), market: v.market, side: v.side, point: v.point ?? null });
  const sameRef = (a, b) => a && b && a.league === b.league && String(a.id) === String(b.id) && a.market === b.market && a.side === b.side
    && (a.point ?? null) === (b.point ?? null);

  function addToSlip(ti) {
    const card = run?.cards[ti];
    if (!card) return;
    const t = card.t;
    const refs = t.refs || t.legs.map((l) => l.ref);
    const legs = (t.legs || []).map((view, i) => slipLeg(view, refs[i] || view.ref));
    const wasEmpty = !slip.count;
    let added = 0;
    for (const leg of legs) if (slip.add(leg)) added += 1;
    if (wasEmpty && added) slip.set({ type: added === 1 ? "single" : "multi", stake: plainStake(t.unit_stake) });
    haptic.select();
    if (added < legs.length) toast(`Added ${added} of ${legs.length} picks: a slip holds up to 12`, { kind: "warn" });
    else if (wasEmpty) toast(`Added ${added} pick${added === 1 ? "" : "s"} to your slip`, { kind: "good" });
    else toast(`Added ${added} pick${added === 1 ? "" : "s"}: your slip now has ${slip.count}. Open it to choose the bet type.`, { kind: "good" });
    if (added && runBooks()) watchSlipPrices(t.legs || [], refs);
  }

  /**
   * The slip prices each pick at the best price across every bookmaker you can bet at; a run limited to some
   * bookmakers may show another price. Once the slip is priced (the server's numbers), say which picks moved.
   */
  function watchSlipPrices(legs, refs) {
    const keys = refs.map((ref) => legKey(ref));
    let off = null;
    const timer = setTimeout(() => off?.(), 15000);
    const check = ({ result } = priced()) => {
      if (!result || !Array.isArray(result.refs)) return false;
      const views = new Map(result.refs.map((ref, i) => [legKey(ref), result.legs?.[i]]));
      if (!keys.every((key) => views.has(key))) return false;
      const moved = legs.map((leg, i) => ({ leg, view: views.get(keys[i]) }))
        .filter(({ leg, view }) => view && (Math.abs(Number(view.price) - Number(leg.price ?? leg.odds?.decimal)) > 0.004 || (view.book && leg.book && view.book !== leg.book)));
      if (moved.length) {
        const { leg, view } = moved[0];
        const more = moved.length > 1 ? ` (and ${moved.length - 1} more)` : "";
        toast(`Your slip uses the best price across all bookmakers: ${leg.label} is ${priceText(view.odds || { decimal: Number(view.price) })} at ${view.book || "another bookmaker"} there, ${priceText(leg.odds || { decimal: Number(leg.price) })} at ${leg.book} on this card${more}.`, { kind: "warn", ms: 6000 });
      }
      return true;
    };
    off = onPriced(() => {
      if (check()) {
        off();
        clearTimeout(timer);
      }
    });
    stops.push(() => {
      off();
      clearTimeout(timer);
    });
  }

  /** The bookmakers the run was limited to (its controls'), or null for all of them. */
  const runBooks = () => {
    const books = run?.data?.controls?.books;
    return Array.isArray(books) && books.length ? books : null;
  };

  async function explainTicket(ti) {
    const card = run?.cards[ti];
    if (!card || card.explain?.busy) return;
    if (card.explain?.data) { // already explained: show or hide it, without spending the allowance again
      card.explain.hidden = !card.explain.hidden;
      drawResults();
      return;
    }
    const t = card.t;
    card.explain = { busy: true };
    drawResults();
    try {
      // The run's bookmakers go along, so the explanation prices each leg where the card does
      const body = { type: t.type === "single" ? "single" : "multi", stake: plainStake(t.unit_stake), legs: t.refs || t.legs.map((l) => l.ref) };
      if (runBooks()) body.books = runBooks();
      const data = await api("/api/tickets/analyse", { method: "POST", signal, body });
      card.explain = { data };
      if (data.credits) loadMe().catch(() => null); // the allowance left has changed
    } catch (error) {
      if (isQuiet(error)) return;
      card.explain = { error: error.message || "Couldn't explain this ticket." };
      if (gateHint(error)) loadMe().catch(() => null);
    }
    if (root.isConnected) drawResults();
  }

  async function askAbout(ti) {
    const card = run?.cards[ti];
    if (!card) return;
    const t = card.t;
    const legs = t.refs || t.legs.map((l) => l.ref);
    const label = `${t.label || run.data.label || "Builder"} ticket · ${legs.length === 1 ? "single" : `${legs.length}-leg multi`} at ${priceText(t.odds)}`;
    try {
      const ask = await import("../ask.js");
      const body = { type: legs.length === 1 ? "single" : "multi", stake: plainStake(t.unit_stake), legs };
      if (runBooks()) body.books = runBooks();
      ask.attachTicket?.({ body, label,
        draft: "Why these legs? What's the main risk, and what would make you leave one out?" });
    } catch {
      /* Ask's code couldn't load: the screen opens without the ticket */
    }
    navigate("#/ask");
  }

  function onResultsClick(event) {
    const t = event.target;
    const swap = t.closest("[data-bd-swap]");
    if (swap) {
      const [ti, li] = swap.dataset.bdSwap.split(":").map(Number);
      const card = run?.cards[ti];
      if (!card) return;
      card.open = card.open === li ? null : li;
      drawResults();
      return;
    }
    const pick = t.closest("[data-bd-pick]");
    if (pick) {
      const [ti, li, k] = pick.dataset.bdPick.split(":").map(Number);
      // aria-disabled, not disabled: a disabled button drops the keyboard focus to the page
      if (pick.getAttribute("aria-disabled") === "true") return;
      pick.setAttribute("aria-disabled", "true");
      pickSwap(ti, li, k);
      return;
    }
    const undo = t.closest("[data-bd-undo]");
    if (undo) {
      const card = run?.cards[Number(undo.dataset.bdUndo)];
      if (card) {
        card.t = card.original;
        card.swapped = null;
        card.explain = null;
        drawResults();
      }
      return;
    }
    const add = t.closest("[data-bd-add]");
    if (add) {
      addToSlip(Number(add.dataset.bdAdd));
      return;
    }
    const ex = t.closest("[data-bd-explain]");
    if (ex) {
      explainTicket(Number(ex.dataset.bdExplain));
      return;
    }
    const ask = t.closest("[data-bd-ask]");
    if (ask) {
      askAbout(Number(ask.dataset.bdAsk));
      return;
    }
    if (t.closest("[data-bd-open-controls]")) {
      if (phone) openMore();
      else {
        const first = root.querySelector(".bd-controls details[data-bd-group]");
        if (first) first.open = true;
        root.querySelector(".bd-scroll")?.scrollTo({ top: 0 }); // the column's own scroll: back to the presets
        root.querySelector(".bd-controls")?.scrollIntoView({ block: "start", behavior: "smooth" });
        root.querySelector(".bd-controls summary")?.focus({ preventScroll: true });
      }
      return;
    }
    if (t.closest("[data-bd-age]")) openOnboarding();
    if (t.closest("[data-bd-build]")) runBuild();
  }

  // --- wiring --------------------------------------------------------------------------------------------------
  root.addEventListener("change", (e) => {
    const box = e.target.closest("[data-bd-ctl]");
    if (box) onControlEvent(e, box);
  });
  root.addEventListener("input", (e) => {
    const box = e.target.closest("[data-bd-ctl]");
    if (box) onControlEvent(e, box);
  });
  root.addEventListener("click", (e) => {
    if (e.target.closest(".bd-controls") && onControlClick(e)) return;
    if (e.target.closest("[data-bd-out]")) onResultsClick(e);
  });
  // Enter in a number box builds (desktop habit), after the change is read
  root.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.matches?.(".bd-controls input[type=text]")) {
      e.preventDefault();
      e.target.dispatchEvent(new Event("change", { bubbles: true }));
      runBuild();
    }
  });

  function syncBlock() {
    const next = picksBlock();
    const changed = Boolean(next) !== Boolean(block) || next?.text !== block?.text;
    block = next;
    $("[data-bd-block]").innerHTML = block ? blockNote(block, BLOCK_ID) : "";
    if (block) {
      controller?.abort();
      busy = false;
      if (more?.open) more.close();
    }
    root.querySelector(".bd-controls")?.toggleAttribute("inert", Boolean(block));
    root.querySelector(".bd-controls")?.classList.toggle("off", Boolean(block));
    if (changed) {
      drawRunBar();
      drawResults();
      refresh();
      fitSoon(); // the line above the columns came or went
    }
  }

  // The desktop column's height follows where it starts on screen (fitColumn)
  window.addEventListener("scroll", fitSoon, { passive: true });
  window.addEventListener("resize", fitSoon, { passive: true });
  stops.push(() => {
    window.removeEventListener("scroll", fitSoon);
    window.removeEventListener("resize", fitSoon);
    cancelAnimationFrame(fitFrame);
  });
  if (typeof ResizeObserver === "function") { // the page head's height (fonts, a narrower window)
    const watch = new ResizeObserver(fitSoon);
    watch.observe(root.querySelector(".bd-head"));
    watch.observe(root.querySelector("[data-bd-block]"));
    stops.push(() => watch.disconnect());
  }

  stops.push(onLayoutChange(() => {
    if ((layout.mode === "phone") !== phone) drawLayout();
  }));
  stops.push(onMeChange(() => {
    if (!root.isConnected) return;
    syncBlock();
    adoptNewer();
  }));

  /**
   * The server's copy of the controls is newer than the one on screen (saved from another device) and the
   * person hasn't changed anything on this visit: show it, and build with it (free: shared results, no fetch).
   */
  function adoptNewer() {
    if (!s || !setup || touched || block) return;
    const server = me().builder;
    const at = Date.parse(server?.ui?.at || "");
    if (!Number.isFinite(at) || at <= (Date.parse(s.ui.at || "") || 0)) return;
    const kept = restore(prefs.get(LOCAL_KEY, null), server, setup.presets);
    if (!kept || (Date.parse(kept.ui.at || "") || 0) <= (Date.parse(s.ui.at || "") || 0)) return;
    s = { c: kept.controls, ui: kept.ui, errors: {} };
    s.formError = "";
    prefs.set(LOCAL_KEY, { controls: s.c, ui: s.ui }); // this device follows (with the server copy's time)
    for (const box of [$("[data-bd-ctl]"), more?.open ? more.body : null]) if (box) drawn.delete(box);
    drawControls();
    refresh();
    if (!(run && ranWith === currentKey())) runBuild({ show: false });
  }
  stops.push(slip.onChange(() => {
    if (run && root.isConnected) drawResults();
  }));

  // --- start ----------------------------------------------------------------------------------------------------
  const ready = (async () => {
    syncBlock();
    try {
      if (!setup) {
        const data = await api("/api/builder/presets", { signal });
        const presets = {};
        for (const key of PRESET_KEYS) if (data.presets?.[key]) presets[key] = cleanControls(data.presets[key], DEFAULTS);
        setup = {
          presets,
          described: data.described || {},
          markets: data.markets || {},
          books: Array.isArray(data.defaults?.books) ? data.defaults.books.filter((b) => b && typeof b.key === "string") : [],
          scope: Array.isArray(data.defaults?.scope_fields) ? data.defaults.scope_fields : SCOPE_FIELDS,
          defaultPreset: PRESET_KEYS.includes(data.defaults?.preset) ? data.defaults.preset : "risky",
          saved: data.defaults?.saved || null,
        };
        memory.setup = setup;
      }
    } catch (error) {
      if (isQuiet(error) || !root.isConnected) return;
      const hint = gateHint(error);
      if (hint?.kind === "break" || hint?.kind === "paused") {
        // The presets are behind the same gates: the line above says until when
        await loadMe().catch(() => null);
        if (!root.isConnected) return;
        syncBlock();
        $("[data-bd-layout]").innerHTML = emptyState({ iconName: "pause", title: "The Builder is off for now",
          body: picksBlock()?.text || error.message, actions: [{ label: "Responsible play", href: "#/help/responsible" }] });
        return;
      }
      $("[data-bd-layout]").innerHTML = error.status === 404
        ? emptyState({ iconName: "clock", title: "The Builder isn't on this server yet", body: "It needs the server's latest update. Your slip and the rollover planner still work." })
        : `${errorState(error, { title: "Couldn't open the Builder", retry: true })}${hint?.kind === "age" ? '<div class="empty-actions"><button type="button" class="btn btn-primary" data-bd-age>Confirm age</button></div>' : ""}`;
      $("[data-bd-layout]").querySelector("[data-bd-age]")?.addEventListener("click", () => openOnboarding());
      return;
    }
    if (!root.isConnected) return;
    // The person's last controls (this device's or the server's, whichever is newer), else the default preset
    const local = prefs.get(LOCAL_KEY, null);
    const kept = restore(local, me().builder || setup.saved, setup.presets);
    s = kept ? { c: kept.controls, ui: kept.ui, errors: {} } : { c: cleanControls(setup.presets[setup.defaultPreset] || DEFAULTS, DEFAULTS), ui: cleanUi(null), errors: {} };
    if (!s.c.preset && !kept) s.c.preset = setup.defaultPreset;
    const asked = params.query?.get("preset");
    if (asked && setup.presets[asked]) s.c = withPreset(asked, s.c, setup.presets, scope());
    if (asked) {
      touched = true; // a preset tapped on the Build page wins over a copy from another device
      replaceQuery();
    }
    drawLayout();
    syncBlock();
    // Today's slate gives the tree its leagues and counts (cached on this device; one small request)
    swr(SLATE_URL, { key: SLATE_KEY, signal, onData: (value) => {
      if (!root.isConnected) return;
      setSlateLeagues(value?.leagues || []);
      prepared = prepare(value);
      for (const box of [$("[data-bd-ctl]"), more?.open ? more.body : null]) placeTree(box);
      drawControls();
    } }).catch(() => {
      treeHost.textContent = "The sports list couldn't load: every league with prices is searched.";
    });
    // Built straight away: a preset tapped elsewhere, or a first visit (shared results, no fetch, no credits)
    const same = run && ranWith === currentKey();
    if (!block && (asked || !same)) await runBuild({ show: false });
  })();

  function replaceQuery() {
    try {
      replaceAddress("#/build/builder"); // through the shell: Back and a reload don't apply the preset again
    } catch {
      /* the address keeps its ?preset; a reload applies it again */
    }
  }

  return {
    ready,
    mainButton() {
      if (!s || !setup || block || (more?.open)) return null;
      return { text: busy ? "Building…" : "Build tickets", onClick: () => runBuild(), progress: busy, disabled: busy };
    },
    stop() {
      stops.forEach((fn) => fn());
      controller?.abort();
      if (dirty) saveServer({ now: true });
      clearTimeout(saveTimer);
      if (more?.open) more.close();
      if (run) memory.run = run;
    },
  };
}

/** The Build screen's "Builder" card (on #/build): a short pitch and the three presets, each opening the Builder. */
export function builderLinkHtml() {
  return PRESET_KEYS.map((k) => `<a class="chip" href="#/build/builder?preset=${k}">${icon({ safe: "shield", risky: "scale", crazy: "zap" }[k], { size: 16 })}${esc(PRESET_TITLES[k])}</a>`).join("");
}
