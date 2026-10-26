// "Let AI build one" on the Build screen: stake X, want Y back (POST /api/tickets/build).
//
// The brain (engine/target.py) fixes the odds a target needs before any pick is chosen, shows the honest
// ceiling for those odds first (with no real edge nothing beats 1 in M for M times the stake), then
// searches the chosen sports' prices for up to three tickets, each priced the same way as the slip.
// The stake is never raised. Nothing goes on the slip until the person taps "Add to slip".
// Below the form sit the first dashboard's Safe / Risky / Crazy tickets (legacy.js), opened on demand.
// On a break, or in the 24-hour pause after a warning sign in the chat, the server refuses both: the form
// and the panel are switched off up front, with one plain line saying until when (common.js picksBlock).

import { api, isQuiet, prefs } from "../../api.js";
import { slip, me, loadMe, favouriteSports, leagueTitle, sportEmoji, openSportsSheet, openOnboarding, loadSports } from "../../app.js";
import { esc, pct, chanceText, money, priceText, notice, skeleton, errorState, explain, badge, segmented, toast, timeLabel } from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { renderAiBets } from "../../legacy.js";
import { STAKES, cleanStake, slipLeg, summaryRows, redraw, reveal, gateHint, gameShort, picksBlock, blockNote } from "./common.js";

const FORM_KEY = "buildForm";
const TARGETS = ["25", "50", "100"];
const TIER_OPTIONS = [{ value: "", label: "Any" }, { value: "safe", label: "Safe" }, { value: "risky", label: "Risky" }, { value: "crazy", label: "Crazy" }];
const TIER_ICONS = { safe: "shield", risky: "scale", crazy: "zap" };
const MAX_TARGET = 10000000;
const MAX_SPORTS = 8;
const BLOCK_ID = "bx-block-builder"; // the line that says why the form and the panel are off

let lastBuild = null; // {data, at}: the last result this visit, shown again when Build reopens

function cleanTarget(value) {
  const text = String(value ?? "").trim().replace(/^\$/, "").replace(/,/g, "");
  if (!/^\d{1,8}(\.\d{1,2})?$/.test(text)) return null;
  const n = Number(text);
  return n >= 0.01 && n <= MAX_TARGET ? text : null;
}

const favourites = () => favouriteSports();

function loadForm() {
  const saved = prefs.get(FORM_KEY, {}) || {};
  const favs = favourites();
  const leagues = Array.isArray(saved.leagues) ? saved.leagues.filter((k) => /^[a-z0-9_]{2,64}$/.test(k)).slice(0, MAX_SPORTS) : [];
  const legs = Number(saved.max_legs);
  const tier = TIER_OPTIONS.some((t) => t.value === saved.tier) ? saved.tier : (me().risk_tier || "");
  return {
    leagues: leagues.length ? leagues : favs.slice(0, MAX_SPORTS),
    stake: cleanStake(saved.stake) || "10",
    target: cleanTarget(saved.target) || "50",
    max_legs: Number.isInteger(legs) && legs >= 1 && legs <= 6 ? legs : 3,
    tier,
  };
}

function sportChips(form) {
  const keys = [...new Set([...favourites(), ...form.leagues])];
  return keys.map((key) => {
    const on = form.leagues.includes(key);
    return `<button type="button" class="chip${on ? " on" : ""}" aria-pressed="${on}" data-b-sport="${esc(key)}">${esc(sportEmoji(key))} ${esc(leagueTitle(key))}</button>`;
  }).join("") + `<button type="button" class="chip add" data-b-more aria-label="Choose your sports">${icon("plus", { size: 14 })}Sports</button>`;
}

function chipRow(name, values, current) {
  return values.map((v) => `<button type="button" class="chip${current === v ? " on" : ""}" aria-pressed="${current === v}" data-${name}="${v}">$${v}</button>`).join("");
}

function formHtml(form, block = null) {
  // A disabled fieldset switches every control off at once, and screen readers announce each as unavailable
  const off = block ? ` disabled aria-describedby="${BLOCK_ID}"` : "";
  return `<form class="bx-form" data-b-form novalidate><fieldset class="bx-fieldset"${off}>
    <legend class="sr-only">Let AI build one</legend>
    <div class="field"><span class="label">Sports</span><div class="chip-row wrap" data-b-sports>${sportChips(form)}</div></div>
    <div class="field"><label class="label" for="b-stake">Stake ${explain("stake")}</label>
      <div class="stake-row" data-b-stakes>${chipRow("b-stake", STAKES, form.stake)}
        <span class="money-input"><span aria-hidden="true">$</span><input id="b-stake" class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(form.stake)}"></span></div></div>
    <div class="field"><label class="label" for="b-target">Want to win</label>
      <div class="stake-row" data-b-targets>${chipRow("b-target", TARGETS, form.target)}
        <span class="money-input"><span aria-hidden="true">$</span><input id="b-target" class="input" type="text" inputmode="decimal" autocomplete="off" value="${esc(form.target)}" aria-describedby="b-target-hint"></span></div></div>
    <p class="hint" id="b-target-hint">“Want to win” is the total paid back if it wins, your stake included.</p>
    <div class="field"><span class="label">Tier ${explain("tiers")}</span>${segmented("b-tier", TIER_OPTIONS, form.tier, { label: "Tier" })}</div>
    <div class="field line-control" style="margin-top:18px">
      <span><span class="label" style="margin:0">Most picks per ticket</span><span class="hint" style="display:block;margin-top:2px">Each extra pick adds the bookmaker's cut again.</span></span>
      <span class="stepper" role="group" aria-label="Most picks per ticket">
        <button type="button" data-b-legs="-1" aria-label="Fewer picks"${form.max_legs <= 1 ? " disabled" : ""}>${icon("minus", { size: 18 })}</button>
        <output aria-live="polite" data-b-legs-out>${form.max_legs}</output>
        <button type="button" data-b-legs="1" aria-label="More picks"${form.max_legs >= 6 ? " disabled" : ""}>${icon("plus", { size: 18 })}</button>
      </span>
    </div>
    <div class="form-error" role="alert" data-b-error></div>
    <button class="btn btn-primary block" type="submit" style="margin-top:12px"${block ? ` aria-describedby="${BLOCK_ID}"` : ""}>${icon(block ? "pause" : "bot", { size: 18 })}Build tickets</button>
  </fieldset></form>`;
}

function tierBadge(tier) {
  if (!tier) return "";
  const loses = tier.loses != null ? `loses ${pct(tier.loses)}` : "";
  if (!tier.title) return loses ? badge(`Loses ${pct(tier.loses)}`, "muted") : "";
  return badge(`${tier.title}${loses ? ` · ${loses}` : ""}`, "muted", TIER_ICONS[tier.key] || "");
}

function gapText(gap) {
  if (gap == null) return "";
  const text = String(gap);
  if (text.startsWith("-")) return `${money(text.slice(1), { cents: true })} short of your target`;
  if (Number(text) === 0) return "exactly your target";
  return `${money(text, { cents: true })} more than your target`;
}

function ticketCard(t, i) {
  const inSlip = (t.refs || []).length > 0 && (t.refs || []).every((ref) => slip.has(ref));
  const cautious = t.cautious || {};
  const legs = (t.legs || []).map((leg) => {
    const chances = [leg.ai_prob != null ? `AI ${pct(leg.ai_prob)}` : "", leg.market_prob != null ? `Market ${pct(leg.market_prob)}` : ""].filter(Boolean).join(" · ");
    return `<li><div class="row-between"><b>${esc(leg.label)}</b><b class="num">${esc(priceText(leg.odds))}</b></div>
      <div class="small muted">${esc(gameShort(leg.game, leg.league))}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""}${chances ? ` · ${esc(chances)}` : ""}</div></li>`;
  }).join("");
  return `<article class="card bx-ticket${t.tier?.key ? ` t-${esc(t.tier.key)}` : ""}">
    <div class="row-between"><b>${esc(t.name || "Ticket")} · ${esc(priceText(t.odds))}</b>${tierBadge(t.tier)}</div>
    <ul class="bx-ticket-legs">${legs}</ul>
    ${(t.reasons || []).map((r) => notice(r, "danger")).join("")}
    ${summaryRows(t, { compact: true })}
    <p class="small muted">${cautious.chance != null ? `Cautious chance (the AI's pulled halfway back to the market's): ${esc(chanceText(cautious.chance, cautious.one_in))}. ` : ""}${t.target_gap != null ? `Pays ${esc(gapText(t.target_gap))}.` : ""}</p>
    <div class="action-bar" style="margin-top:0">
      ${inSlip ? `<button type="button" class="btn btn-secondary" disabled>${icon("check", { size: 18 })}In your slip</button>`
        : `<button type="button" class="btn btn-primary" data-b-add="${i}">${icon("plus", { size: 18 })}Add to slip</button>
           ${slip.count ? `<button type="button" class="btn btn-secondary" data-b-replace="${i}">Use as my slip</button>` : ""}`}
    </div>
  </article>`;
}

/**
 * The honest ceiling's tile: "1 in X" below even chances and a percentage from there up, as everywhere
 * on Build ("1 in 1" would read as a certainty). A target at or under the stake needs no bet at all.
 */
function ceilingTile(data) {
  const p = data.ceiling_chance;
  let value = "–";
  let sub = "";
  if (p != null && p >= 1) sub = "your target is no more than your stake";
  else if (p != null && data.ceiling_one_in && p < 0.5) {
    value = `1 in ${Number(data.ceiling_one_in).toLocaleString()}`;
    sub = `${pct(p)}, whichever picks are chosen`;
  } else if (p != null) {
    value = pct(p);
    sub = "at most, whichever picks are chosen";
  }
  return `<div class="bx-tile"><span>Best chance with no edge ${explain("one_in")}</span><b class="num">${esc(value)}</b>
    ${sub ? `<small>${esc(sub)}</small>` : ""}</div>`;
}

function resultHtml(data, at) {
  const errors = Object.entries(data.errors || {});
  const target = data.target_odds || {};
  const asOf = Object.values(data.odds_as_of || {}).filter(Boolean);
  return `<div class="section" aria-live="polite">
    ${errors.map(([league, message]) => notice(`${leagueTitle(league)}: ${message}`, "info")).join("")}
    <div class="bx-tiles">
      <div class="bx-tile"><span>Odds needed ${explain(me().odds_format === "american" ? "american_odds" : "decimal_odds")}</span><b class="num">${esc(priceText({ decimal: target.decimal, american: target.american }))}</b>
        <small>${money(data.stake, { cents: true })} → ${money(data.target_return, { cents: true })}</small></div>
      ${ceilingTile(data)}
    </div>
    <div class="verdict">${esc(data.verdict)}</div>
    ${(data.notes || []).length ? `<ul class="bullets" style="margin-top:10px;color:var(--text-2)">${data.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    ${(data.tickets || []).length ? `<div class="bx-tickets">${data.tickets.map(ticketCard).join("")}</div>` : ""}
    <p class="method">${data.searched ? `Searched ${Number(data.searched.combinations || 0).toLocaleString()} combinations of ${Number(data.searched.pool || 0).toLocaleString()} picks. ` : ""}${asOf.length ? `Odds as of ${esc(asOf.join(", "))}. ` : ""}Built ${esc(timeLabel(new Date(at).toISOString()))}.</p>
  </div>`;
}

/**
 * The builder card in ``root``. Returns {prefill(changes), draw(), sync()}.
 *   prefill({tier})  set a form field and bring the form into view (the empty slip's tier chips)
 *   draw()           redraw the tickets (after the slip changes, "Add to slip" becomes "In your slip")
 *   sync()           after GET /api/me changed: switch the form and the panel off (a break, the picks pause)
 *                    or back on
 * ``replaceSlip(fill, what)`` does "Use as my slip": it empties the slip, runs ``fill()`` and keeps the
 * old slip so the person can undo it (slip.js).
 */
export function createBuilder(root, { signal, replaceSlip = (fill) => { slip.clear(); fill(); } } = {}) {
  let form = loadForm();
  let busy = false;
  let block = picksBlock(); // a break or the picks pause: the server would refuse the form and the panel
  root.innerHTML = `<section class="card section bx-builder" id="builder">
      <div class="section-head"><h2>${icon("bot", { size: 18 })} Let AI build one</h2></div>
      <p class="small muted">Say what you'd stake and what you'd like back. The brain shows the honest ceiling first, then up to three tickets from the prices on offer. Nothing goes on your slip until you tap.</p>
      <div data-b-block>${block ? blockNote(block, BLOCK_ID) : ""}</div>
      <div style="margin-top:14px" data-b-form-box>${formHtml(form, block)}</div>
      <div data-built></div>
    </section>
    <details class="card section bx-more" data-b-tiers>
      <summary>${icon("shield", { size: 16 })}Today's NBA Safe · Risky · Crazy tickets</summary>
      <div data-ai-bets style="margin-top:8px"></div>
    </details>`;
  const box = root.querySelector("[data-b-form-box]");
  const built = root.querySelector("[data-built]");
  const tiers = root.querySelector("[data-b-tiers]");
  const save = () => prefs.set(FORM_KEY, form);
  const errorBox = () => root.querySelector("[data-b-error]");

  const drawBuilt = () => {
    if (block) { // the AI's tickets wait with the form; the line above says until when
      built.innerHTML = "";
      return;
    }
    if (!lastBuild) return;
    redraw(built, resultHtml(lastBuild.data, lastBuild.at));
  };
  // The Safe / Risky / Crazy panel: closed and announced as unavailable while picks are off
  const syncPanel = () => {
    const summary = tiers.querySelector("summary");
    tiers.classList.toggle("off", Boolean(block));
    summary.querySelector("[data-b-off]")?.remove();
    if (block) {
      tiers.open = false;
      summary.setAttribute("aria-disabled", "true");
      summary.setAttribute("aria-describedby", BLOCK_ID);
      summary.insertAdjacentHTML("beforeend", `<span data-b-off>${badge("Off for now", "muted", "pause")}</span>`);
    } else {
      summary.removeAttribute("aria-disabled");
      summary.removeAttribute("aria-describedby");
    }
  };
  const syncChips = (name, current) => {
    for (const b of root.querySelectorAll(`[data-${name}]`)) {
      const on = b.getAttribute(`data-${name}`) === current;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
  };
  const redrawForm = () => {
    redraw(box, formHtml(form, block));
  };
  const sync = () => {
    const next = picksBlock();
    const changed = Boolean(next) !== Boolean(block);
    block = next;
    root.querySelector("[data-b-block]").innerHTML = block ? blockNote(block, BLOCK_ID) : "";
    if (changed) {
      redrawForm();
      if (block || lastBuild) drawBuilt(); // a refusal's error goes too: the line says why
    }
    syncPanel();
  };

  const submit = async () => {
    if (busy || block) return;
    const error = errorBox();
    const stake = cleanStake(root.querySelector("#b-stake")?.value);
    const target = cleanTarget(root.querySelector("#b-target")?.value);
    if (!form.leagues.length) {
      error.textContent = "Pick at least one sport.";
      return;
    }
    if (!stake) {
      error.textContent = "Enter a stake between $0.01 and $100,000.";
      return;
    }
    if (!target) {
      error.textContent = "Enter what you'd like back, e.g. 50.";
      return;
    }
    error.textContent = "";
    form = { ...form, stake, target };
    save();
    busy = true;
    const button = root.querySelector('[data-b-form] [type="submit"]');
    if (button) button.disabled = true;
    built.innerHTML = `<div class="section">${skeleton("cards", 1)}<p class="muted small" style="margin-top:8px">Searching ${esc(form.leagues.map(leagueTitle).join(", "))} prices for tickets near your target…</p></div>`;
    try {
      const body = { leagues: form.leagues, stake, target, max_legs: form.max_legs };
      if (form.tier) body.tier = form.tier;
      const data = await api("/api/tickets/build", { method: "POST", body, signal });
      lastBuild = { data, at: Date.now() };
      if (!root.isConnected) return;
      drawBuilt();
      haptic.notify("success");
      reveal(built);
    } catch (err) {
      if (isQuiet(err) || !root.isConnected) return;
      const hint = gateHint(err);
      built.innerHTML = `<div class="section">${errorState(err, { retry: false, title: "Couldn't build tickets" })}
        ${hint?.kind === "age" ? '<div class="empty-actions"><button type="button" class="btn btn-primary" data-b-age>Confirm age</button></div>' : ""}</div>`;
      // A break or pause this page didn't know of (taken in the bot's chat, say): fetch it, and sync() says so
      if (hint?.kind === "break" || hint?.kind === "paused") loadMe().catch(() => null);
    } finally {
      busy = false;
      const again = root.querySelector('[data-b-form] [type="submit"]');
      if (again) again.disabled = false;
    }
  };

  const addTicket = (t, replace) => {
    const legs = (t.legs || []).map((view, i) => slipLeg(view, (t.refs || [])[i] || {}));
    if (!legs.length) return;
    let added = 0;
    const fill = () => {
      const wasEmpty = !slip.count;
      for (const leg of legs) {
        if (slip.add(leg)) added += 1;
      }
      if (wasEmpty) slip.set({ type: legs.length === 1 ? "single" : "multi", stake: String(Number(t.unit_stake)) });
    };
    if (replace) replaceSlip(fill, t.name ? `the AI's ${t.name.toLowerCase()}` : "the AI's ticket");
    else fill();
    haptic.select();
    if (added < legs.length) toast(`Added ${added} of ${legs.length} picks: a slip holds up to 12`, { kind: "warn" });
    else toast(replace ? "Your slip is now this ticket · Undo is on the slip" : `Added ${added} pick${added === 1 ? "" : "s"} to your slip`, { kind: "good" });
    reveal(document.getElementById("slip"));
  };

  root.addEventListener("click", (event) => {
    const sport = event.target.closest("[data-b-sport]");
    if (sport) {
      const key = sport.dataset.bSport;
      if (form.leagues.includes(key)) form = { ...form, leagues: form.leagues.filter((k) => k !== key) };
      else if (form.leagues.length < MAX_SPORTS) form = { ...form, leagues: [...form.leagues, key] };
      else toast(`Pick up to ${MAX_SPORTS} sports`, { kind: "warn" });
      save();
      redraw(root.querySelector("[data-b-sports]"), sportChips(form));
      return;
    }
    if (event.target.closest("[data-b-more]")) {
      openSportsSheet({ onDone: () => {
        if (!root.isConnected) return;
        const favs = favourites();
        form = { ...form, leagues: form.leagues.filter((k) => favs.includes(k)) };
        if (!form.leagues.length) form.leagues = favs.slice(0, MAX_SPORTS);
        save();
        redraw(root.querySelector("[data-b-sports]"), sportChips(form));
      } });
      return;
    }
    const stake = event.target.closest("[data-b-stake]");
    if (stake) {
      form = { ...form, stake: stake.dataset.bStake };
      root.querySelector("#b-stake").value = form.stake;
      syncChips("b-stake", form.stake);
      save();
      return;
    }
    const target = event.target.closest("[data-b-target]");
    if (target) {
      form = { ...form, target: target.dataset.bTarget };
      root.querySelector("#b-target").value = form.target;
      syncChips("b-target", form.target);
      save();
      return;
    }
    const tier = event.target.closest('[data-seg="b-tier"]');
    if (tier) {
      form = { ...form, tier: tier.dataset.value };
      for (const b of root.querySelectorAll('[data-seg="b-tier"]')) b.setAttribute("aria-pressed", String(b.dataset.value === form.tier));
      save();
      return;
    }
    const legs = event.target.closest("[data-b-legs]");
    if (legs) {
      const next = Math.min(6, Math.max(1, form.max_legs + Number(legs.dataset.bLegs)));
      form = { ...form, max_legs: next };
      root.querySelector("[data-b-legs-out]").textContent = String(next);
      for (const b of root.querySelectorAll("[data-b-legs]")) b.disabled = (b.dataset.bLegs === "-1" && next <= 1) || (b.dataset.bLegs === "1" && next >= 6);
      save();
      return;
    }
    const add = event.target.closest("[data-b-add]");
    const replace = event.target.closest("[data-b-replace]");
    if ((add || replace) && lastBuild) {
      const t = lastBuild.data.tickets?.[Number(add ? add.dataset.bAdd : replace.dataset.bReplace)];
      if (t) addTicket(t, Boolean(replace));
      return;
    }
    if (event.target.closest("[data-b-age]")) openOnboarding();
  });

  root.addEventListener("input", (event) => {
    if (event.target.id === "b-stake") {
      const value = cleanStake(event.target.value);
      if (value) {
        form = { ...form, stake: value };
        syncChips("b-stake", value);
      }
    } else if (event.target.id === "b-target") {
      const value = cleanTarget(event.target.value);
      if (value) {
        form = { ...form, target: value };
        syncChips("b-target", value);
      }
    }
  });

  root.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  tiers.querySelector("summary").addEventListener("click", (event) => {
    if (block) event.preventDefault(); // stays closed: the server would refuse these picks
  });
  tiers.addEventListener("toggle", () => {
    const host = tiers.querySelector("[data-ai-bets]");
    if (tiers.open && block) {
      tiers.open = false;
      return;
    }
    if (tiers.open && !host.dataset.loaded) {
      host.dataset.loaded = "1";
      renderAiBets(host, { league: "basketball_nba" });
    }
  });

  // League names arrive with /api/sports
  loadSports().then(() => {
    if (root.isConnected) redraw(root.querySelector("[data-b-sports]"), sportChips(form));
  }).catch(() => null);
  drawBuilt();
  syncPanel();

  return {
    sync,
    prefill(changes) {
      form = { ...form, ...changes };
      save();
      redrawForm();
      reveal(root.querySelector("#builder"));
    },
    draw: drawBuilt,
  };
}
