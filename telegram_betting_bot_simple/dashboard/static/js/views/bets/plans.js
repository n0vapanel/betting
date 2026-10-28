// My bets → Plans: rollover plans day by day (GET /api/plans), with the next day played on the person's
// own picks and Stop.
//
// The server decides everything about a plan: each day's stake (from the strategy and the money the plan
// actually has), when the next day opens (only once the previous day has a result: nothing moves on by
// itself), when a run has failed, and the break suggested after one. This screen shows that, lets the
// person choose the day's picks from their slip (ticked by them, never added for them), shows the
// server's pricing of those picks at the plan's stake, and sends them with POST /api/plans/<id>/day.

import { api, isQuiet } from "../../api.js";
import { me, loadMe, slip } from "../../app.js";
import {
  esc, pct, chanceText, money, signedMoney, priceText, timeLabel, shortDate, badge, notice, emptyState, errorState,
  skeleton, sheet, toast, confirmSheet, explain,
} from "../../ui.js";
import { icon } from "../../icons.js";
import { haptic } from "../../telegram.js";
import { numbersHtml } from "./slips.js";

const STATUS = {
  active: { text: "Active", kind: "good", iconName: "trend" },
  done: { text: "Finished", kind: "info", iconName: "check" },
  failed: { text: "Run ended", kind: "danger", iconName: "close" },
  stopped: { text: "Stopped", kind: "muted", iconName: "pause" },
};
const DAY_RESULT = {
  won: ["won", "check", "won"], lost: ["lost", "close", "lost"], void: ["void", "minus", "pushed or void"],
  partial: ["partial", "minus", "partly won"],
};

const picksWord = (n) => (n === 1 ? "1 pick" : `1 to ${n} picks`);

// --- a plan as a card ---------------------------------------------------------------------------------

function dayPills(p) {
  const played = new Map((p.days_played || []).map((d) => [d.day, d]));
  const pills = [];
  for (let day = 1; day <= p.days; day += 1) {
    const d = played.get(day);
    let cls = "future";
    let iconName = "";
    let label = `Day ${day}: to come`;
    if (d && !d.result) {
      [cls, iconName, label] = ["live", "clock", `Day ${day}: in play`];
    } else if (d) {
      const [kind, name, words] = DAY_RESULT[d.result.status] || DAY_RESULT.void;
      [cls, iconName, label] = [kind, name, `Day ${day}: ${words}`];
    } else if (p.next?.day === day) {
      [cls, label] = ["next", `Day ${day}: next, ready to play`];
    } else if (p.status !== "active") {
      [cls, label] = ["skip", `Day ${day}: not played`];
    }
    pills.push(`<li class="mb-day ${cls}" title="${esc(label)}"><span class="sr-only">${esc(label)}</span>${iconName ? icon(iconName, { size: 14 }) : `<span aria-hidden="true">${day}</span>`}</li>`);
  }
  return `<ol class="mb-days" aria-label="Days of the plan">${pills.join("")}</ol>`;
}

function planCard(p) {
  const status = STATUS[p.status] || STATUS.stopped;
  const played = (p.days_played || []).length;
  const waiting = played && !(p.days_played[played - 1].result);
  const actions = [];
  const onBreak = me().on_break && me().cooloff_until;
  if (p.status === "active" && p.next && onBreak) {
    actions.push(`<button type="button" class="btn btn-secondary" disabled>${icon("pause", { size: 18 })}On a break until ${esc(timeLabel(me().cooloff_until))}</button>`);
  } else if (p.status === "active" && p.next) {
    actions.push(`<button type="button" class="btn btn-primary" data-play-day="${esc(p.id)}">${icon("arrow", { size: 18 })}Play day ${p.next.day} · ${money(p.next.stake, { cents: true })}</button>`);
  } else if (p.status === "active" && waiting) {
    actions.push(`<button type="button" class="btn btn-secondary" disabled>${icon("clock", { size: 18 })}Day ${p.day} in play</button>`);
  }
  if (p.status === "active") actions.push(`<button type="button" class="btn btn-ghost" data-stop-plan="${esc(p.id)}">${icon("pause", { size: 18 })}Stop plan</button>`);
  return `<article class="card mb-plan" id="plan-${esc(p.id)}" data-plan-id="${esc(p.id)}">
    <div class="mb-head">
      <div class="grow"><h3>${esc(p.title)}</h3>
        <span class="muted">Started ${esc(shortDate(p.created_at))} · ${p.days} days at odds of ${esc(p.min_odds)}+ · ${picksWord(p.legs_per_day)} a day</span></div>
      ${badge(status.text, status.kind, status.iconName)}
    </div>
    <div class="mb-badges">${p.mode === "paper" ? badge("Paper: no money", "muted", "book") : badge("Tracked: counts against your budget", "warn", "wallet")}
      <span class="badge muted">${p.day} of ${p.days} days played</span></div>
    ${dayPills(p)}
    <div class="mb-tiles">
      <div><span>Started with</span><b>${money(p.start, { cents: true })}</b></div>
      <div><span>Has now</span><b>${money(p.bankroll, { cents: true })}</b></div>
      <div><span>Banked</span><b>${money(p.banked, { cents: true })}</b></div>
      <div><span>Net</span><b>${esc(signedMoney(p.profit))}</b></div>
    </div>
    ${p.message ? notice(p.message, p.status === "failed" ? "warn" : p.status === "active" ? "info" : "good") : ""}
    ${p.suggestion ? `${notice(p.suggestion, "warn")}<div class="mb-actions"><a class="btn btn-secondary small" href="#/settings?focus=break">${icon("pause", { size: 16 })}Take a break</a>
      <a class="btn btn-ghost small" href="#/help/responsible">${icon("lifebuoy", { size: 16 })}Responsible play</a></div>` : ""}
    ${p.rule ? `<p class="small muted">${esc(p.rule)}</p>` : ""}
    ${actions.length ? `<div class="mb-actions">${actions.join("")}</div>` : ""}
    <details class="mb-more" data-plan-detail="${esc(p.id)}"><summary>${icon("down", { size: 16 })}Day by day, and the chances when it started</summary>
      <div data-detail-body>${skeleton("lines", 3)}</div></details>
  </article>`;
}

function dayLine(d) {
  const s = d.slip;
  const picks = s ? s.legs.map((leg) => `${leg.label} ${priceText(leg.odds)}`).join(" + ") : "";
  const r = d.result;
  let outcome = `<span class="mb-mark warn">${icon("clock", { size: 14 })}In play</span>`;
  if (r) {
    const [, name, words] = DAY_RESULT[r.status] || DAY_RESULT.void;
    const kind = r.status === "won" ? "good" : r.status === "lost" ? "bad" : "muted";
    outcome = `<span class="mb-mark ${kind}">${icon(name, { size: 14 })}${esc(words[0].toUpperCase() + words.slice(1))}</span> · ${money(r.returns, { cents: true })} back`;
  }
  const kept = r && (Number(d.banked) > 0 || Number(d.rolled) > 0)
    ? ` · banked ${money(d.banked, { cents: true })}, rolled on ${money(d.rolled, { cents: true })}` : "";
  return `<li><b>Day ${d.day}</b><div><div>${outcome}</div>
    <div class="muted">Stake ${money(d.stake, { cents: true })}${kept}</div>
    ${picks ? `<div class="muted">${esc(picks)}${s?.legs?.[0]?.start ? ` · ${esc(timeLabel(s.legs[0].start))}` : ""}</div>` : ""}</div></li>`;
}

/** What the planner said when the plan started (stored with the plan), for the chosen strategy and the rest. */
function comparisonHtml(p) {
  const c = p.comparison;
  if (!c?.strategies) return "";
  const mine = c.strategies[p.strategy]?.market;
  const rows = Object.entries(c.strategies).map(([key, entry]) => {
    const m = entry.market;
    if (!m) return "";
    return `<li class="${key === p.strategy ? "mine" : ""}"><span>${esc(entry.title)}${key === p.strategy ? " (yours)" : ""}</span>
      <span>${pct(m.chance_profit)} chance of a profit · expected cost ${money(m.expected_cost, { cents: true })}</span></li>`;
  }).join("");
  return `<h4 class="subhead" style="margin-top:14px">When it started ${explain("rollover")}</h4>
    ${c.summary ? `<p class="small">${esc(c.summary)}</p>` : ""}
    ${mine ? `<dl class="mb-nums" style="margin-top:10px">
      <div><dt>Chance of finishing</dt><dd>${esc(chanceText(mine.chance_complete, mine.complete_one_in))}</dd></div>
      <div><dt>Chance of a profit</dt><dd>${pct(mine.chance_profit)}</dd></div>
      <div><dt>Typical end</dt><dd>${money(mine.median_end, { cents: true })}</dd></div>
      <div><dt>Ends at half or less</dt><dd>${pct(mine.chance_half_or_less)}</dd></div>
    </dl>` : ""}
    ${c.sentences?.cost?.[p.strategy] ? `<p class="small muted" style="margin-top:8px">${esc(c.sentences.cost[p.strategy])}</p>` : ""}
    <ul class="mb-compare" aria-label="The four strategies by the market's chances">${rows}</ul>
    <p class="note">By the market's chances, as the planner worked them out when the plan started.</p>`;
}

async function fillDetail(details, id, signal) {
  const body = details.querySelector("[data-detail-body]");
  if (details.dataset.loaded) return;
  details.dataset.loaded = "1";
  try {
    const p = await api(`/api/plans/${encodeURIComponent(id)}`, { signal });
    if (!body.isConnected) return;
    const days = p.days_played || [];
    body.innerHTML = `${days.length ? `<ul class="mb-daylist">${days.map(dayLine).join("")}</ul>` : '<p class="muted small">No days played yet.</p>'}
      ${comparisonHtml(p)}`;
  } catch (error) {
    delete details.dataset.loaded;
    if (!isQuiet(error) && body.isConnected) body.innerHTML = notice(error.message, "warn");
  }
}

// --- playing the next day -----------------------------------------------------------------------------

/**
 * The sheet for the plan's next day: tick 1 to N of the slip's picks, see the server's pricing of them at
 * the plan's stake, then play the day. Resolves when the sheet closes; calls onPlayed(plan) on success.
 */
function openPlayDay(p, { onPlayed }) {
  const next = p.next;
  const max = next.max_legs || p.legs_per_day || 1;
  // The person ticks the day's picks themselves; a slip of exactly one pick starts ticked
  const chosen = new Set(slip.count === 1 ? [slip.legs[0].key] : []);
  let pricing = { busy: false, result: null, error: null };
  let sending = false;
  let problem = "";
  let controller = null;
  let timer = null;

  const refs = () => slip.legs.filter((leg) => chosen.has(leg.key)).map((leg) => ({ ...leg.ref }));

  const pickList = () => `<ul class="mb-picks">${slip.legs.map((leg) => {
    const on = chosen.has(leg.key);
    const full = !on && chosen.size >= max;
    return `<li><label class="check${full ? " off" : ""}"><input type="checkbox" data-pick="${esc(leg.key)}"${on ? " checked" : ""}>
      <span class="grow"><b>${esc(leg.label)}</b><span class="muted">${esc(leg.game || "")}${leg.start ? ` · ${esc(timeLabel(leg.start))}` : ""}</span></span>
      <b class="num">${esc(priceText({ decimal: leg.price, american: leg.american }))}</b></label></li>`;
  }).join("")}</ul>`;

  const pricedHtml = () => {
    if (!chosen.size) return `<p class="hint">Tick ${picksWord(max)} for day ${next.day}.</p>`;
    if (pricing.busy && !pricing.result) return skeleton("lines", 3);
    if (pricing.error) return notice(pricing.error.message, "warn");
    const r = pricing.result;
    if (!r) return "";
    const parts = [];
    for (const reason of r.reasons || []) parts.push(notice(reason, "danger"));
    if ((r.removed || []).length) parts.push(notice("A pick is no longer offered: untick it or remove it from your slip.", "warn"));
    if (r.odds && Number(r.odds.decimal) < Number(next.min_odds)) {
      parts.push(notice(`These picks pay ${Number(r.odds.decimal).toFixed(2)} in all; this plan needs ${next.min_odds} or more. Choose a pick at ${next.min_odds} or above.`, "warn"));
    }
    if (r.bets) parts.push(numbersHtml({ ...r, stake: r.total_stake }, { info: false }));
    if ((r.warnings || []).length) {
      parts.push(`<ul class="warnings">${r.warnings.map((w) => `<li>${icon("alert", { size: 14 })}<span>${esc(w)}</span></li>`).join("")}</ul>`);
    }
    return parts.join("");
  };

  const body = () => {
    const intro = notice(`${p.title}: today's stake is ${money(next.stake, { cents: true })}, set by the plan from the money it has. Choose ${picksWord(max)} paying ${next.min_odds} or more in all.${p.mode === "paper" ? " Paper mode: nothing is bet." : ""}`, "info");
    if (!slip.count) {
      return `${intro}${emptyState({
        iconName: "bets", title: "Your slip is empty",
        body: `Add a pick at odds of ${next.min_odds} or more from Today, then come back here (My bets → Plans).`,
        actions: [{ label: "Browse Today", href: "#/today", primary: true }, { label: "Let AI build one", href: "#/build" }],
      })}`;
    }
    return `${intro}
      <div class="field"><span class="label">Your slip's picks <span class="muted">(${picksWord(max)})</span></span>${pickList()}</div>
      <div class="mb-priced" data-priced aria-live="polite">${pricedHtml()}</div>
      <div class="form-error" role="alert">${esc(problem)}</div>
      <p class="hint">${max > 1 ? "One pick near the minimum odds costs less than two: each leg adds the bookmaker's cut. " : ""}The result is graded when the games end, and the next day opens only after that.</p>`;
  };

  const footer = () => (slip.count
    ? `<button type="button" class="btn btn-secondary" data-cancel>Cancel</button>
      <button type="button" class="btn btn-primary grow" data-confirm${!chosen.size || sending ? " disabled" : ""}>${sending ? "Saving…" : `Play day ${next.day} · ${money(next.stake, { cents: true })}`}</button>`
    : "");

  const view = sheet.open({ title: `Day ${next.day} of ${p.days}`, size: "tall", className: "mb-sheet", body: body(), footer: footer(),
    onClose: () => { controller?.abort(); clearTimeout(timer); } });

  const redraw = ({ all = false } = {}) => {
    if (!view.open) return;
    if (all) {
      sheet.update({ body: body(), footer: footer() });
      return;
    }
    const box = view.body.querySelector("[data-priced]");
    if (box) box.innerHTML = pricedHtml();
    view.foot.innerHTML = footer();
    view.foot.hidden = !view.foot.childNodes.length;
  };

  const price = () => {
    clearTimeout(timer);
    controller?.abort();
    const legs = refs();
    if (!legs.length) {
      pricing = { busy: false, result: null, error: null };
      redraw();
      return;
    }
    pricing = { ...pricing, busy: true, error: null };
    redraw();
    timer = setTimeout(async () => {
      controller = new AbortController();
      try {
        const result = await api("/api/tickets/price", {
          method: "POST", signal: controller.signal,
          body: { type: legs.length === 1 ? "single" : "multi", stake: next.stake, legs },
        });
        pricing = { busy: false, result, error: null };
      } catch (error) {
        if (error.name === "AbortError") return;
        pricing = { busy: false, result: null, error: isQuiet(error) ? null : error };
      }
      redraw();
    }, 200);
  };

  view.panel.addEventListener("change", (event) => {
    const box = event.target.closest("[data-pick]");
    if (!box) return;
    const key = box.dataset.pick;
    if (box.checked && chosen.size >= max) {
      box.checked = false;
      toast(`This plan takes ${picksWord(max)} a day`, { kind: "warn" });
      return;
    }
    if (box.checked) chosen.add(key);
    else chosen.delete(key);
    haptic.select();
    problem = "";
    view.body.querySelector(".mb-picks").outerHTML = pickList();
    price();
  });

  view.panel.addEventListener("click", async (event) => {
    if (event.target.closest("[data-cancel]")) {
      view.close();
      return;
    }
    if (!event.target.closest("[data-confirm]") || sending) return;
    const legs = refs();
    if (!legs.length) return;
    sending = true;
    problem = "";
    redraw({ all: true });
    try {
      const plan = await api(`/api/plans/${encodeURIComponent(p.id)}/day`, { method: "POST", body: { legs } });
      haptic.notify("success");
      // The picks are on the plan's day now: take them off the slip so they aren't played twice
      for (const leg of slip.legs.filter((l) => chosen.has(l.key))) slip.remove(leg.key);
      if (me().budget) loadMe().catch(() => null);
      toast(`Day ${next.day} is in play. Its picks moved from your slip to the plan.`, { kind: "good", ms: 4000 });
      await view.close();
      onPlayed?.(plan);
    } catch (error) {
      sending = false;
      if (isQuiet(error)) return;
      haptic.notify("error");
      problem = error.message;
      redraw({ all: true });
    }
  });

  price();
  return view;
}

async function stopPlan(p, button, reload) {
  const ok = await confirmSheet({
    title: "Stop this plan?",
    body: `${p.title}: it stops where it is${(p.days_played || []).some((d) => !d.result) ? ", and the day in play still settles and is recorded" : ""}. You can start a new plan in Build.`,
    confirm: "Stop plan", danger: true,
  });
  if (!ok) return;
  button.disabled = true;
  try {
    await api(`/api/plans/${encodeURIComponent(p.id)}/stop`, { method: "POST" });
    toast("Plan stopped");
    reload();
  } catch (error) {
    button.disabled = false;
    if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
  }
}

// --- the tab ------------------------------------------------------------------------------------------

/**
 * Draw the Plans tab into ``box``.
 *   ctx = {signal, query: URLSearchParams (plan=<id> opens that plan's details, play=1 its next day),
 *          onReady(plan|null, play)}   onReady tells bets.js which plan Telegram's main button can play
 */
export async function renderPlans(box, ctx) {
  box.innerHTML = skeleton("cards", 2);
  let plans;
  try {
    plans = await api("/api/plans", { signal: ctx.signal });
  } catch (error) {
    if (!isQuiet(error) && box.isConnected) box.innerHTML = errorState(error);
    return;
  }
  if (!box.isConnected) return;
  const reload = () => renderPlans(box, { ...ctx, query: new URLSearchParams() });
  const play = (p) => openPlayDay(p, { onPlayed: reload });
  const ready = me().on_break ? null : plans.find((p) => p.status === "active" && p.next) || null;
  ctx.onReady?.(ready, () => ready && play(ready));
  if (!plans.length) {
    box.innerHTML = emptyState({
      iconName: "trend", title: "No plans yet",
      body: "The rollover planner compares four ways to play a run of days, real chances first, then starts a paper plan you play one day at a time.",
      actions: [{ label: "Open the planner", href: "#/build/rollover", primary: true }, { label: "Rollover explained", href: "#/help/rollover" }],
    });
    return;
  }
  const active = plans.filter((p) => p.status === "active");
  const past = plans.filter((p) => p.status !== "active");
  const failed = past.filter((p) => p.status === "failed").length;
  box.innerHTML = `${active.length ? `<div class="stack">${active.map(planCard).join("")}</div>`
      : `${notice("No plan is running. Every attempt stays here, failures included.", "info")}
        <a class="btn btn-secondary block" style="margin-top:10px" href="#/build/rollover">${icon("trend", { size: 18 })}Open the rollover planner</a>`}
    ${past.length ? `<h2 class="subhead">Earlier plans · ${past.length}${failed ? ` (${failed} ended early)` : ""}</h2>
      <div class="stack">${past.map(planCard).join("")}</div>` : ""}
    <p class="note">Plans never move on by themselves: each day needs your tap, and only after the previous day's result is in. ${explain("rollover", { label: "What is a rollover?" })}</p>`;

  box.onclick = (event) => {
    const playButton = event.target.closest("[data-play-day]");
    const stopButton = event.target.closest("[data-stop-plan]");
    const id = playButton?.dataset.playDay || stopButton?.dataset.stopPlan;
    const p = id ? plans.find((x) => String(x.id) === id) : null;
    if (!p) return;
    if (playButton && p.next) play(p);
    else if (stopButton) stopPlan(p, stopButton, reload);
  };
  for (const details of box.querySelectorAll("[data-plan-detail]")) {
    details.addEventListener("toggle", () => {
      if (details.open) fillDetail(details, details.dataset.planDetail, ctx.signal);
    });
  }
  // Deep link: #/bets?tab=plans&plan=<id>[&play=1]
  const wanted = ctx.query?.get("plan");
  const card = wanted ? box.querySelector(`[data-plan-id="${CSS.escape(wanted)}"]`) : null;
  if (card) {
    const details = card.querySelector("[data-plan-detail]");
    details.open = true;
    requestAnimationFrame(() => card.scrollIntoView({ block: "start" }));
    const p = plans.find((x) => String(x.id) === wanted);
    if (ctx.query.get("play") === "1" && p?.next) play(p);
  }
}

