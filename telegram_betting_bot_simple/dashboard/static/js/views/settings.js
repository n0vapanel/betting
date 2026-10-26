// Settings: who you are, how numbers look, your sports, and your limits.
//   Profile      name (shown on the leaderboard), how you're signed in, age confirmation
//   Display      odds format (decimal or American), theme (browser only: Telegram uses its own)
//   Your sports  the favourites Today shows first (only these are fetched from the odds feed)
//   Limits       weekly budget (lowering applies now, raising waits 24 hours), reality check, and a break
//                of 24 hours, 3 days, 1 week, 1 month or 3 months, confirmed first and never cut short
//   AI           the disclosure and today's AI allowance
//   Account      sign out (password sessions), the welcome screen again, the helplines
// Everything saves through PUT /api/me (POST /api/me/cooloff for a break). The server applies the
// rules; this screen says what they are before anyone taps, and shows the server's answer afterwards.
// Deep links: #/settings?focus=limits | budget | break scroll to that part.

import { api, isQuiet, prefs } from "../api.js";
import {
  me, loadMe, updateMe, setMe, onMeChange, openSportsSheet, openOnboarding, signOut, leagueTitle, sportEmoji, loadSports,
  inTelegram, applyTheme,
} from "../app.js";
import { esc, money, notice, segmented, toast, confirmSheet, explain, section } from "../ui.js";
import { icon } from "../icons.js";
import { haptic } from "../telegram.js";

const BREAKS = [
  { hours: 24, label: "24 hours" }, { hours: 72, label: "3 days" }, { hours: 168, label: "1 week" },
  { hours: 720, label: "1 month" }, { hours: 2160, label: "3 months" },
];
const REALITY = [{ value: "", label: "Off" }, { value: "15", label: "15 min" }, { value: "30", label: "30 min" }, { value: "60", label: "60 min" }];
const MONEY = /^\d{1,7}(\.\d{1,2})?$/;

const CSS = `
.view-settings .st-row { display: flex; align-items: center; gap: 10px; min-height: var(--tap); }
.view-settings .st-row + .st-row { border-top: 1px solid var(--line); }
.view-settings .st-row > .icon { color: var(--text-2); }
.view-settings .st-budget { padding: 12px; border-radius: 12px; background: var(--card-2); border: 1px solid var(--line); display: grid; gap: 8px; }
.view-settings .st-budget .row-between b { font-size: 20px; font-weight: 800; }
.view-settings .st-breaks { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 8px; }
.view-settings .st-breaks .btn { padding: 0 10px; }
.view-settings .st-example { margin-top: 8px; font-size: 13px; color: var(--text-2); }
.view-settings .st-age { border-color: color-mix(in srgb, var(--warn) 45%, transparent); }
.view-settings [id^="st-"] { scroll-margin-top: calc(var(--header-h) + var(--safe-top) + 12px); }
.view-settings .st-focus { animation: st-focus 1.6s ease-out 1; border-radius: var(--radius); }
@keyframes st-focus { 0% { box-shadow: 0 0 0 3px var(--info); } 100% { box-shadow: 0 0 0 3px transparent; } }
@media (prefers-reduced-motion: reduce) { .view-settings .st-focus { animation: none; } }
`;

function ensureStyles() {
  if (document.getElementById("settings-styles")) return;
  const style = document.createElement("style");
  style.id = "settings-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}

const longDate = (iso) => new Date(iso).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const cents = (value) => money(Number(value || 0) / 100, { cents: true }); // the AI allowance comes in cents

// --- the parts ------------------------------------------------------------------------------------

function ageHtml(p) {
  if (p.age_confirmed) return "";
  return `<section class="card section st-age">
    <h2 class="row" style="font-size:16px">${icon("shield", { size: 20 })}Confirm your age</h2>
    <p class="small muted" style="margin-top:6px">Games, prices, slips, plans and the AI are for adults only.</p>
    <label class="check" style="margin-top:10px"><input type="checkbox" data-age> <span>I'm 18 or older (21+ in the US) and betting is legal where I live.</span></label>
    <button type="button" class="btn btn-primary block" style="margin-top:10px" data-confirm-age disabled>Confirm</button>
  </section>`;
}

function profileHtml(p) {
  const how = p.is_telegram ? "Signed in with Telegram" : "Signed in with the dashboard password on this browser";
  return section("Profile", `<div class="field"><label class="label" for="set-name">Name <span class="muted">(shown on the leaderboard)</span></label>
      <div class="search-row"><input id="set-name" class="input" maxlength="24" autocomplete="nickname" value="${esc(p.name || prefs.name() || "")}" placeholder="e.g. Sam">
        <button type="button" class="btn btn-secondary" data-save-name>Save</button></div></div>
    <div class="st-row" style="margin-top:8px">${icon(p.is_telegram ? "send" : "user", { size: 18 })}<span class="grow small">${esc(how)}</span></div>
    ${p.age_confirmed ? `<div class="st-row">${icon("check", { size: 18 })}<span class="grow small">Age confirmed: 18+ (21+ in the US)</span></div>` : ""}`);
}

function displayHtml(p) {
  const american = p.odds_format === "american";
  return section("Display", `<div class="field"><span class="label">Odds format ${explain(american ? "american_odds" : "decimal_odds")}</span>
      ${segmented("format", [{ value: "decimal", label: "Decimal 2.50" }, { value: "american", label: "American +150" }], american ? "american" : "decimal", { label: "Odds format" })}
      <p class="st-example">A $10 bet at ${american ? "+150" : "2.50"} returns $25: your $10 back plus $15 profit.</p></div>
    ${inTelegram ? "" : `<div class="field"><span class="label">Theme</span>${segmented("theme", [{ value: "system", label: "Phone setting" }, { value: "dark", label: "Dark" }, { value: "light", label: "Light" }], prefs.get("theme", "system"), { label: "Theme" })}</div>`}`);
}

function sportsHtml(p) {
  const list = p.favourite_sports?.length ? p.favourite_sports : ["basketball_nba"];
  return section("Your sports", `<p class="small muted">Shown first on Today. Only these are fetched, because each one uses some of the odds feed's monthly requests.</p>
    <div class="chip-row wrap" style="margin-top:10px">${list.map((key) => `<span class="chip">${esc(sportEmoji(key))} ${esc(leagueTitle(key))}</span>`).join("")}
      <button type="button" class="chip add" data-sports>${icon("edit", { size: 14 })}Change</button></div>`, { info: "quota" });
}

function budgetHtml(p) {
  const b = p.budget;
  let status = '<p class="small muted">No budget set: the stakes you save aren\'t limited.</p>';
  if (b) {
    const weekly = Number(b.weekly);
    const share = weekly > 0 ? Math.min(100, (Number(b.used) / weekly) * 100) : 100; // bar width only
    status = `<div class="st-budget">
      <div class="row-between"><b>${money(b.left, { cents: true })} left</b><span class="muted small">of ${money(b.weekly, { cents: true })} this week</span></div>
      <span class="meter" role="img" aria-label="${esc(`${money(b.used, { cents: true })} of ${money(b.weekly, { cents: true })} used`)}"><span style="width:${share.toFixed(1)}%"></span></span>
      <span class="small muted">${money(b.used, { cents: true })} staked so far${b.resets ? ` · starts again ${esc(longDate(b.resets))}` : ""}</span>
    </div>
    ${b.pending ? notice(`Raising to ${money(b.pending.amount, { cents: true })} takes effect ${longDate(b.pending.effective_at)}. Until then your budget stays ${money(b.weekly, { cents: true })}.`, "info") : ""}`;
  }
  return `<div class="field" id="st-budget"><span class="label">Weekly budget ${explain("bankroll")}</span>
    ${status}
    <div class="search-row" style="margin-top:10px"><span class="money-input grow"><span aria-hidden="true">$</span>
      <input id="set-budget" class="input" type="text" inputmode="decimal" autocomplete="off" placeholder="${b ? esc(Number(b.weekly).toFixed(0)) : "e.g. 50"}" aria-label="Weekly budget in dollars"></span>
      <button type="button" class="btn btn-secondary" data-save-budget>${b ? "Change" : "Set"}</button></div>
    <p class="hint" data-budget-hint>Stakes you save count against it, and anything past it is refused. Lowering applies at once; raising waits 24 hours.</p>
  </div>`;
}

function breakHtml(p) {
  if (p.on_break && p.cooloff_until) {
    const until = new Date(p.cooloff_until).getTime();
    const longer = BREAKS.filter((x) => Date.now() + x.hours * 3600 * 1000 > until + 3600 * 1000); // at least an hour longer
    return `<div class="field" id="st-break"><span class="label">Take a break</span>
      ${notice(`You're on a break until ${longDate(p.cooloff_until)}. Picks, building and plans are off until then, and it can't be cut short.`, "good")}
      ${longer.length ? `<p class="hint">Want it longer? A break can be extended, never shortened.</p>
        <div class="st-breaks" style="margin-top:8px">${longer.map((x) => `<button type="button" class="btn btn-secondary small" data-break="${x.hours}">${icon("pause", { size: 16 })}${esc(x.label)}</button>`).join("")}</div>` : ""}
    </div>`;
  }
  return `<div class="field" id="st-break"><span class="label">Take a break</span>
    <p class="small muted">Picks, building, plans and the AI's picks switch off for as long as you choose. It can't be cut short, and nothing starts again by itself.</p>
    <div class="st-breaks" style="margin-top:10px">${BREAKS.map((x) => `<button type="button" class="btn btn-secondary" data-break="${x.hours}">${icon("pause", { size: 18 })}${esc(x.label)}</button>`).join("")}</div>
  </div>`;
}

function limitsHtml(p) {
  return `<section class="card section" id="st-limits">
    <div class="section-head"><h2>${icon("shield", { size: 18 })} Limits</h2><a class="text-link" href="#/help/responsible">Why ${icon("chevron", { size: 16 })}</a></div>
    ${budgetHtml(p)}
    <div class="field"><span class="label">Reality check</span>
      ${segmented("reality", REALITY, p.reality_check_minutes ? String(p.reality_check_minutes) : "", { label: "Reality check" })}
      <p class="hint">A reminder of how long the app has been open, every so many minutes, until you close it.</p></div>
    ${breakHtml(p)}
  </section>`;
}

function aiHtml(p) {
  const c = p.credits;
  const meter = c && c.daily_cents ? `<div class="field"><span class="label">Today's AI allowance</span>
      <div class="row-between small"><span><b>${cents(c.left_cents)}</b> left</span><span class="muted">of ${cents(c.daily_cents)} a day</span></div>
      <span class="meter" style="margin-top:6px" role="img" aria-label="${esc(`${cents(c.left_cents)} of ${cents(c.daily_cents)} left today`)}"><span style="width:${Math.max(0, Math.min(100, (c.left_cents / c.daily_cents) * 100)).toFixed(1)}%"></span></span>
      <p class="hint">Chats and written analyses use it; prices, chances, slips and plans don't. It refills every day.</p></div>` : "";
  return section("About the AI", `<div class="disclosure" style="margin-top:0">${icon("bot", { size: 20 })}<p><b>You're using an AI analyst.</b>
      Its numbers are computed in code from odds and stats; Claude only explains them. It can be wrong, and no model reliably beats the betting market.
      Only bet what you can afford to lose.</p></div>
    ${p.ai_enabled ? meter : `<p class="hint" style="margin-top:10px">The AI chat isn't switched on for this server; everything else works without it.</p>`}
    <a class="text-link" href="#/help/ai">How the AI decides ${icon("chevron", { size: 16 })}</a>`);
}

function accountHtml(p) {
  return section("Help and account", `<div class="st-row">${icon("lifebuoy", { size: 18 })}<span class="grow small">Free, confidential help 24/7: <a href="tel:18004262537">1-800-GAMBLER</a> · <a href="tel:08088020133">0808 8020 133</a></span></div>
    <div class="st-row">${icon("book", { size: 18 })}<a class="grow small" href="#/help/start">How the app works, in 60 seconds</a></div>
    <div class="st-row">${icon("refresh", { size: 18 })}<button type="button" class="grow small" style="text-align:left;color:var(--info);min-height:var(--tap)" data-onboard>Show the welcome screen again</button></div>
    ${p.is_telegram
      ? `<p class="hint">You're signed in through Telegram. Your bets and settings belong to this Telegram account.</p>`
      : `<button type="button" class="btn btn-secondary block" style="margin-top:10px" data-sign-out>${icon("logout", { size: 18 })}Sign out</button>
        <p class="hint">Your bets and settings belong to this browser's sign-in.</p>`}`);
}

// --- the screen -----------------------------------------------------------------------------------

let stop = null;

export function onLeave() {
  stop?.();
  stop = null;
}

export async function render(root, params) {
  ensureStyles();
  const draw = () => {
    const p = me();
    root.innerHTML = `<div class="page-head"><h1>Settings</h1><p>Your profile, how numbers look, and your limits.</p></div>
      ${ageHtml(p)}${profileHtml(p)}${displayHtml(p)}${sportsHtml(p)}${limitsHtml(p)}${aiHtml(p)}${accountHtml(p)}`;
  };
  draw();
  // Changes made elsewhere (the onboarding sheet, a budget refresh) redraw this screen, unless the person is typing
  stop?.();
  stop = onMeChange(() => {
    if (root.isConnected && !root.contains(document.activeElement?.closest?.("input") || null)) draw();
  });
  // Fresh sport names, budget used and AI allowance (they change all the time), then the deep-linked part
  const target = { limits: "#st-limits", budget: "#st-budget", break: "#st-break" }[params.query.get("focus")];
  const reveal = () => {
    const el = target && root.isConnected ? root.querySelector(target) : null;
    if (!el) return;
    el.scrollIntoView({ block: "start" });
    el.classList.add("st-focus");
  };
  requestAnimationFrame(reveal);
  Promise.allSettled([loadSports(), loadMe()]).then(() => {
    if (!root.isConnected) return;
    draw();
    reveal();
  });

  const save = async (changes, message) => {
    try {
      const p = await updateMe(changes);
      haptic.notify("success");
      if (message) toast(typeof message === "function" ? message(p) : message, { kind: "good", ms: 3600 });
      draw();
      return p;
    } catch (error) {
      if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
      return null;
    }
  };

  const saveBudget = () => {
    const input = root.querySelector("#set-budget");
    const value = input.value.trim().replace(/^\$/, "").replace(/,/g, "");
    if (!MONEY.test(value)) {
      toast("Enter an amount like 50 or 25.50", { kind: "warn" });
      input.focus();
      return;
    }
    const before = me().budget;
    save({ weekly_budget: value }, (p) => {
      if (p.budget?.pending) return `Raise saved: ${money(p.budget.pending.amount, { cents: true })} a week from ${longDate(p.budget.pending.effective_at)}`;
      if (before && Number(value) < Number(before.weekly)) return `Budget lowered to ${money(p.budget.weekly, { cents: true })} a week, starting now`;
      return `Weekly budget: ${money(p.budget?.weekly ?? value, { cents: true })}`;
    });
  };

  const startBreak = async (hours) => {
    const choice = BREAKS.find((x) => x.hours === hours);
    if (!choice) return;
    const ends = longDate(new Date(Date.now() + hours * 3600 * 1000).toISOString());
    const extending = me().on_break;
    const ok = await confirmSheet({
      title: extending ? `Extend your break to ${choice.label}?` : `Take a break for ${choice.label}?`,
      body: `Until about ${ends}, picks, building, plans and the AI's picks are switched off. It can't be cut short or undone, and nothing starts again by itself.`,
      confirm: extending ? "Extend my break" : "Start my break",
    });
    if (!ok) return;
    try {
      setMe(await api("/api/me/cooloff", { method: "POST", body: { hours } }));
      haptic.notify("success");
      toast("Your break has started. Take care.", { kind: "good", ms: 4000 });
      draw();
    } catch (error) {
      if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
    }
  };

  root.addEventListener("input", (event) => {
    if (event.target.id === "set-budget") {
      const hint = root.querySelector("[data-budget-hint]");
      const b = me().budget;
      const value = event.target.value.trim().replace(/^\$/, "").replace(/,/g, "");
      if (!hint) return;
      if (!value || !MONEY.test(value) || !b) {
        hint.textContent = "Stakes you save count against it, and anything past it is refused. Lowering applies at once; raising waits 24 hours.";
      } else if (Number(value) > Number(b.weekly)) {
        hint.textContent = `That's a raise from ${money(b.weekly, { cents: true })}: it takes effect 24 hours after you save it.`;
      } else {
        hint.textContent = "That's lower: it applies straight away.";
      }
    }
  });
  root.addEventListener("change", (event) => {
    if (event.target.matches("[data-age]")) root.querySelector("[data-confirm-age]").disabled = !event.target.checked;
  });
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    if (event.target.id === "set-budget") saveBudget();
    else if (event.target.id === "set-name") root.querySelector("[data-save-name]")?.click();
  });
  root.addEventListener("click", (event) => {
    const seg = event.target.closest("[data-seg]");
    if (seg) {
      if (seg.getAttribute("aria-pressed") === "true") return;
      if (seg.dataset.seg === "format") save({ odds_format: seg.dataset.value }, "Odds format saved");
      else if (seg.dataset.seg === "theme") {
        prefs.set("theme", seg.dataset.value);
        applyTheme(seg.dataset.value);
        draw();
      } else if (seg.dataset.seg === "reality") {
        const minutes = seg.dataset.value ? Number(seg.dataset.value) : null;
        save({ reality_check_minutes: minutes }, minutes ? `Reality check every ${minutes} minutes` : "Reality check off");
      }
      return;
    }
    if (event.target.closest("[data-save-name]")) {
      const name = root.querySelector("#set-name").value.trim().slice(0, 24);
      prefs.setName(name); // the calls leaderboard reads the name kept in this browser
      save({ name: name || null }, name ? "Name saved" : "Name cleared");
    } else if (event.target.closest("[data-save-budget]")) {
      saveBudget();
    } else if (event.target.closest("[data-confirm-age]")) {
      save({ confirm_age: true }, "Thanks: age confirmed");
    } else if (event.target.closest("[data-sports]")) {
      openSportsSheet({ onDone: draw });
    } else if (event.target.closest("[data-onboard]")) {
      openOnboarding();
    } else if (event.target.closest("[data-sign-out]")) {
      signOut();
    } else {
      const brk = event.target.closest("[data-break]");
      if (brk) startBreak(Number(brk.dataset.break));
    }
  });
}

