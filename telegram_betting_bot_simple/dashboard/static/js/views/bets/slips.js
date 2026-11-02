// My bets → Upcoming and Settled: the person's saved bets (GET /api/slips) as cards, the results
// summary above them, and Share / Delete on each card.
//
// Every price, chance and payout on a card is the server's pricing from when the bet was saved, and
// every result is the server's grade. The results summary is the one sum made here: it adds up the
// server's own settled amounts in whole cents (exact, no floating-point drift) to show the net profit or
// loss (what came back − what was staked) and the ROI (net ÷ staked). It never shows "total won" alone.
// Paper plan days are play money, so the summary leaves them out and says so. Void bets gave their whole
// stake back and were never at risk, so (like the server's budget) it counts them but leaves them out of
// "staked", "came back" and the ROI; counting them would pull the ROI towards 0.

import { api, isQuiet, prefs } from "../../api.js";
import { me, loadMe } from "../../app.js";
import {
  esc, pct, chanceText, money, signedMoney, priceText, timeLabel, shortDate, badge, emptyState, errorState, skeleton,
  segmented, sheet, toast, confirmSheet, explain as explainButton,
} from "../../ui.js";
import { icon } from "../../icons.js";
import { copyText } from "../../legacy.js";
import { haptic } from "../../telegram.js";
import { termInfo } from "../help/sheet.js";

const DAY = 24 * 3600 * 1000;
const PERIODS = [{ value: "7d", label: "7 days" }, { value: "30d", label: "30 days" }, { value: "all", label: "All" }];
const SYSTEM_NAMES = {
  trixie: "Trixie", patent: "Patent", yankee: "Yankee", lucky15: "Lucky 15", canadian: "Canadian",
  super_yankee: "Super Yankee", lucky31: "Lucky 31", heinz: "Heinz", lucky63: "Lucky 63", super_heinz: "Super Heinz",
  goliath: "Goliath",
};
// A ticket's result, in words and an icon as well as a colour
const RESULTS = {
  won: { text: "Won", kind: "good", iconName: "check" },
  lost: { text: "Lost", kind: "danger", iconName: "close" },
  partial: { text: "Partly won", kind: "warn", iconName: "minus" },
  void: { text: "Void", kind: "muted", iconName: "refresh" },
};
const LEG_RESULTS = {
  won: ["Won", "good", "check"], lost: ["Lost", "bad", "close"], push: ["Push", "muted", "minus"],
  void: ["Void", "muted", "minus"], cancelled: ["Void", "muted", "minus"], dead_heat: ["Dead heat", "warn", "minus"],
};

// --- money in whole cents ----------------------------------------------------------------------------

/** Whole cents from the server's money strings ("12.34", "-2.35"); null when it isn't one. */
export function toCents(value) {
  const match = /^(-)?(\d+)(?:\.(\d{1,2}))?$/.exec(String(value ?? "").trim());
  if (!match) return null;
  const cents = Number(match[2]) * 100 + Number((match[3] || "").padEnd(2, "0"));
  return match[1] ? -cents : cents;
}

export const fromCents = (cents) => (cents / 100).toFixed(2);

// --- what a saved bet is ------------------------------------------------------------------------------

/** "3-leg multi", "2 singles", "Round robin by 2 · 6 bets", "Lucky 15 · 15 bets". */
export function ticketName(s) {
  const n = s.legs?.length || 0;
  if (s.type === "multi") return n === 1 ? "Single" : `${n}-leg multi`;
  if (s.type === "single") return n === 1 ? "Single" : `${n} singles`;
  if (s.type === "round_robin") return `Round robin by ${s.k || 2}${s.bets ? ` · ${s.bets} bets` : ""}`;
  if (s.type === "system") return `${SYSTEM_NAMES[s.system] || "System bet"}${s.bets ? ` · ${s.bets} bets` : ""}`;
  return s.title || "Bet";
}

const startOf = (leg) => (leg.start ? new Date(leg.start).getTime() : NaN);

/** "upcoming" (no game started), "in_play" (one has; or a start is unknown), "settled" or "void". */
export function slipPhase(s, now = Date.now()) {
  if (s.status === "settled") return "settled";
  if (s.status === "void") return "void";
  const started = (s.legs || []).some((leg) => !(startOf(leg) > now));
  return started ? "in_play" : "upcoming";
}

const firstStart = (s) => Math.min(...(s.legs || []).map(startOf).filter(Number.isFinite), Infinity);
const isPaper = (s) => s.plan?.mode === "paper";

function statusBadge(s, phase) {
  if (phase === "upcoming") return badge("Upcoming", "info", "clock");
  if (phase === "in_play") return badge("In play", "warn", "clock");
  const r = RESULTS[phase === "void" ? "void" : s.result?.status] || RESULTS.void;
  return badge(r.text, r.kind, r.iconName);
}

/** "lose $10 (86%)", "a profit of $4.70 (49%)": the single most likely outcome, by the server. */
function mostLikelyText(ml, total) {
  if (!ml) return "";
  const words = {
    "lose all": `lose ${money(total, { cents: true })}`,
    "some back": ml.profit != null ? `get some back (${signedMoney(ml.profit)})` : "get some back",
    "stake back": "stake back",
    profit: ml.profit != null ? `a profit of ${money(ml.profit, { cents: true })}` : "a profit",
  }[ml.result] || ml.result;
  return `${words} (${pct(ml.probability)})`;
}

/**
 * The numbers the server priced when the bet was saved: stake, odds, payout, chances, cut, expected result.
 * ``info: false`` leaves out the ⓘ buttons (inside a sheet, where an explainer would replace the sheet).
 */
export function numbersHtml(s, { info = true } = {}) {
  const explain = info ? explainButton : () => "";
  const c = s.chance || {};
  const e = s.expected || {};
  const several = (s.bets || 1) > 1 && s.type !== "multi";
  const chance = [c.ai != null ? `AI ${esc(chanceText(c.ai, c.ai_one_in))}` : "", c.market != null ? `Market ${esc(chanceText(c.market, c.market_one_in))}` : ""]
    .filter(Boolean).join("<br>");
  const expected = [e.ai != null ? `${esc(signedMoney(e.ai))} <span class="muted">AI</span>` : "", e.market != null ? `${esc(signedMoney(e.market))} <span class="muted">market</span>` : ""]
    .filter(Boolean).join("<br>");
  const items = [
    ["Staked", `${money(s.stake, { cents: true })}${(s.bets || 1) > 1 ? ` <span class="muted">(${s.bets} × ${money(s.unit_stake, { cents: true })})</span>` : ""}`],
    s.odds ? ["Odds", esc(priceText(s.odds))] : null,
    s.max_payout != null ? [`${several ? "Pays up to" : "Pays if it wins"} ${explain("payout")}`, money(s.max_payout, { cents: true })] : null,
    chance ? [`${c.headline === "profit" ? "Chance of a profit" : "Chance it wins"} ${explain("one_in")}`, chance] : null,
    s.book_cut != null ? [`Bookmaker keeps ${explain("book_cut")}`, `about ${pct(s.book_cut)}`] : null,
    expected ? [`Expected result ${explain("expected_result")}`, expected] : null,
    s.most_likely ? [`Most likely ${explain("most_likely")}`, esc(mostLikelyText(s.most_likely, s.stake))] : null,
  ].filter(Boolean);
  return `<dl class="mb-nums">${items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
}

function resultLine(s, phase) {
  const r = s.result || {};
  const stake = money(r.total_stake ?? s.stake, { cents: true });
  if (phase === "void" || r.status === "void") {
    return `<p class="mb-result">${icon("refresh", { size: 16 })} <b>Void:</b> every pick pushed or was called off, so the ${stake} stake came back.</p>`;
  }
  const net = `net <b>${esc(signedMoney(r.profit))}</b>`;
  if (r.status === "won") return `<p class="mb-result"><b>Won:</b> ${money(r.returns, { cents: true })} back on ${stake} staked · ${net}</p>`;
  if (r.status === "lost") return `<p class="mb-result"><b>Lost:</b> ${stake} staked, nothing back · ${net}</p>`;
  if (r.status === "partial") {
    return `<p class="mb-result"><b>Partly won:</b> ${r.paid_bets ?? "some"} of ${r.bets ?? s.bets} bets paid, ${money(r.returns, { cents: true })} back on ${stake} · ${net}</p>`;
  }
  return "";
}

function legRow(leg, phase, now) {
  const mark = LEG_RESULTS[leg.result?.result];
  const open = phase === "upcoming" || phase === "in_play";
  let status = "";
  if (mark) status = `<span class="mb-mark ${mark[1]}">${icon(mark[2], { size: 14 })}${mark[0]}</span>`;
  else if (open && !(startOf(leg) > now)) status = `<span class="mb-mark warn">${icon("clock", { size: 14 })}In play</span>`;
  const chances = open ? [leg.ai_prob != null ? `AI ${pct(leg.ai_prob)}` : "", leg.market_prob != null ? `market ${pct(leg.market_prob)}` : ""].filter(Boolean).join(" · ") : "";
  return `<li class="mb-leg">
    <div class="row-between"><b class="mb-leg-label">${esc(leg.label)}</b><b class="num">${esc(priceText(leg.odds))}</b></div>
    <div class="row-between small"><span class="muted mb-leg-game">${leg.start ? `${esc(timeLabel(leg.start))} · ` : ""}${esc(leg.game || "")}</span>${status}</div>
    ${chances ? `<div class="tiny muted">${esc(chances)}</div>` : ""}
  </li>`;
}

/** One saved bet as a card. */
export function slipCard(s, now = Date.now()) {
  const phase = slipPhase(s, now);
  const open = phase === "upcoming" || phase === "in_play";
  const n = s.legs?.length || 0;
  const canDelete = phase === "upcoming" && s.status === "saved" && !s.plan;
  const plan = s.plan ? badge(`Plan day ${s.plan.day}${s.plan.mode === "paper" ? " · paper" : ""}`, "info", "trend") : "";
  return `<article class="card mb-slip" data-slip-id="${esc(s.id)}">
    <div class="mb-head">
      <div class="grow"><h3>${esc(ticketName(s))}</h3>
        <span class="muted">Saved ${esc(shortDate(s.created_at))} · ${n} pick${n === 1 ? "" : "s"}</span></div>
      ${statusBadge(s, phase)}
    </div>
    ${plan ? `<div class="mb-badges">${plan}</div>` : ""}
    ${open ? "" : resultLine(s, phase)}
    <ul class="mb-legs">${(s.legs || []).map((leg) => legRow(leg, phase, now)).join("")}</ul>
    ${open ? numbersHtml(s) : `<details class="mb-more"><summary>${icon("down", { size: 16 })}The numbers when it was saved</summary><div>${numbersHtml(s)}</div></details>`}
    <div class="mb-actions">
      ${s.share_code ? `<button type="button" class="btn btn-secondary small" data-share-slip="${esc(s.id)}">${icon("share", { size: 16 })}Share</button>` : ""}
      ${s.plan ? `<a class="btn btn-ghost small" href="#/bets?tab=plans&plan=${esc(s.plan.id)}">${icon("trend", { size: 16 })}Open the plan</a>` : ""}
      <span class="grow"></span>
      ${canDelete ? `<button type="button" class="btn btn-ghost small" data-delete-slip="${esc(s.id)}">${icon("trash", { size: 16 })}Delete</button>` : ""}
    </div>
  </article>`;
}

// --- the results summary ------------------------------------------------------------------------------

function summarize(slips, period, now = Date.now()) {
  const since = period === "7d" ? now - 7 * DAY : period === "30d" ? now - 30 * DAY : 0;
  const sum = { staked: 0, back: 0, count: 0, won: 0, lost: 0, partial: 0, void: 0, openCount: 0, openStaked: 0, paper: 0 };
  for (const s of slips) {
    if (isPaper(s)) {
      sum.paper += 1;
      continue;
    }
    if (s.status === "saved") {
      sum.openCount += 1;
      sum.openStaked += toCents(s.stake) || 0;
      continue;
    }
    if (s.status !== "settled" && s.status !== "void") continue;
    if (new Date(s.updated_at || s.created_at).getTime() < since) continue; // by when it settled
    if (s.status === "void" || s.result?.status === "void") {
      sum.void += 1; // the stake came back: shown in the record line, not in the money
      continue;
    }
    const staked = toCents(s.result?.total_stake ?? s.stake);
    const back = toCents(s.result?.returns);
    if (staked === null || back === null) continue;
    sum.staked += staked;
    sum.back += back;
    sum.count += 1;
    if (s.result?.status in RESULTS) sum[s.result.status] += 1;
  }
  return sum;
}

/** "−12.4%": net ÷ staked, one decimal, the sign always written. */
function roiText(net, staked) {
  if (!staked) return "–";
  const value = (net / staked) * 100;
  const text = Math.abs(value).toFixed(1);
  return `${value > 0.04 ? "+" : value < -0.04 ? "−" : ""}${text}%`;
}

/** The card above the lists: net profit or loss and ROI for a period, then open bets and the budget. */
export function resultsHtml(slips, period) {
  const sum = summarize(slips, period);
  const net = sum.back - sum.staked;
  const when = { "7d": "in the last 7 days", "30d": "in the last 30 days", all: "yet" }[period];
  const record = ["won", "lost", "partial"].filter((k) => sum[k]).map((k) => `${sum[k]} ${RESULTS[k].text.toLowerCase()}`);
  if (sum.void) record.push(`${sum.void} void (stake back, not counted)`);
  const budget = me().budget;
  const lines = [];
  if (sum.openCount) lines.push(`<span><b>${sum.openCount}</b> open bet${sum.openCount === 1 ? "" : "s"} · ${money(fromCents(sum.openStaked), { cents: true })} staked, waiting for results</span>`);
  if (budget) lines.push(`<span>Weekly budget: <b>${money(budget.left, { cents: true })}</b> left of ${money(budget.weekly, { cents: true })} · <a href="#/settings?focus=limits">Limits</a></span>`);
  if (sum.paper) lines.push(`<span>Paper plan days are play money, so they aren't counted here.</span>`);
  return `<section class="card mb-results" aria-labelledby="mb-results-h">
    <div class="row-between mb-results-head"><h2 id="mb-results-h">Your results ${termInfo("roi", { label: "What are net profit and ROI?" })}</h2>
      ${segmented("mb-period", PERIODS, period, { label: "Period" })}</div>
    ${sum.count ? `<div class="mb-net"><span class="mb-net-v ${net > 0 ? "up" : net < 0 ? "down" : ""}">${esc(signedMoney(fromCents(net)))}</span>
        <span class="muted">net ${net > 0 ? "profit" : net < 0 ? "loss" : "result: even"} on ${sum.count} settled bet${sum.count === 1 ? "" : "s"}</span></div>
      <div class="mb-kpis">
        <div><b>${esc(roiText(net, sum.staked))}</b><span>ROI ${termInfo("roi")}</span></div>
        <div><b>${money(fromCents(sum.staked), { cents: true })}</b><span>staked</span></div>
        <div><b>${money(fromCents(sum.back), { cents: true })}</b><span>came back</span></div>
      </div>
      ${record.length ? `<div class="mb-lines"><span>${esc(record.join(" · "))}</span></div>` : ""}`
    : `<p class="muted" style="margin-top:8px">${sum.void
      ? `No bets won or lost ${when}: ${sum.void === 1 ? "the void one gave its" : `the ${sum.void} void ones gave their`} stake back, so there's no profit or loss to show.`
      : `No settled bets ${when}. Once bets settle, this shows your net profit or loss and your ROI.`}</p>`}
    ${lines.length ? `<div class="mb-lines">${lines.join("")}</div>` : ""}
  </section>`;
}

// --- the two lists ------------------------------------------------------------------------------------

/** Upcoming (and in play) soonest first; settled newest first. */
export function listHtml(slips, which, now = Date.now()) {
  if (which === "upcoming") {
    const open = slips.filter((s) => s.status === "saved").sort((a, b) => firstStart(a) - firstStart(b));
    if (!open.length) {
      return emptyState({
        iconName: "bets", title: "No open bets",
        body: "Tap prices on Today to build a slip, then save it in Build to track it here. It settles by itself when the games end.",
        actions: [{ label: "Browse Today", href: "#/today", primary: true }, { label: "Open Build", href: "#/build" }],
      });
    }
    return `<div class="stack">${open.map((s) => slipCard(s, now)).join("")}</div>`;
  }
  const done = slips.filter((s) => s.status === "settled" || s.status === "void")
    .sort((a, b) => new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at));
  if (!done.length) {
    return emptyState({
      iconName: "check", title: "Nothing settled yet",
      body: "Saved bets settle by themselves from the final scores, usually within an hour of the last game ending.",
      actions: [{ label: "See open bets", href: "#/bets?tab=upcoming" }],
    });
  }
  return `<div class="stack">${done.map((s) => slipCard(s, now)).join("")}</div>`;
}

// --- share and delete ---------------------------------------------------------------------------------

function shareLink(code) {
  const base = String(me().public_url || location.origin).replace(/\/+$/, "");
  return `${base}/#/build?code=${encodeURIComponent(code)}`;
}

function openShare(s) {
  const code = s.share_code;
  const link = shareLink(code);
  const view = sheet.open({
    title: "Share this bet",
    className: "mb-sheet",
    body: `<p class="lead">Anyone using this app can load these picks and the stake into their own slip with the code. Prices are checked again when they open it; your results and other bets stay private.</p>
      <div class="mb-code" aria-label="Share code ${esc(code.split("").join(" "))}">${esc(code)}</div>
      <p class="hint" style="overflow-wrap:anywhere">${esc(link)}</p>`,
    footer: `<button type="button" class="btn btn-secondary grow" data-share="code">${icon("copy", { size: 18 })}Copy code</button>
      <button type="button" class="btn btn-secondary grow" data-share="link">${icon("copy", { size: 18 })}Copy link</button>
      ${navigator.share ? `<button type="button" class="btn btn-primary grow" data-share="native">${icon("share", { size: 18 })}Share…</button>` : ""}`,
  });
  view.foot.addEventListener("click", async (event) => {
    const how = event.target.closest("[data-share]")?.dataset.share;
    if (how === "code") copyText(code, "Code");
    else if (how === "link") copyText(link, "Link");
    else if (how === "native") {
      try {
        await navigator.share({ title: "My SportsBet AI bet", text: `My picks on SportsBet AI: code ${code}`, url: link });
      } catch {
        /* the person closed the share sheet */
      }
    }
  });
}

async function deleteSlip(s, button, reload) {
  const ok = await confirmSheet({
    title: "Delete this bet?",
    body: `${ticketName(s)}, ${money(s.stake, { cents: true })} staked. It comes off My bets and stops counting against your weekly budget. This can't be undone.`,
    confirm: "Delete", danger: true,
  });
  if (!ok) return;
  button.disabled = true;
  try {
    await api(`/api/slips/${encodeURIComponent(s.id)}`, { method: "DELETE" });
    haptic.notify("success");
    toast("Bet deleted");
    if (me().budget) loadMe().catch(() => null);
    reload();
  } catch (error) {
    button.disabled = false;
    if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
  }
}

/**
 * Draw Upcoming or Settled into ``box``: the results card, then the list.
 *   ctx = {signal, slips: (fresh?) => Promise<[slip]>, onChange()}
 */
export async function renderSlips(box, which, ctx) {
  box.innerHTML = skeleton("cards", 2);
  let slips;
  try {
    slips = await ctx.slips();
  } catch (error) {
    if (!isQuiet(error) && box.isConnected) box.innerHTML = errorState(error);
    return;
  }
  if (!box.isConnected) return;
  let period = PERIODS.some((p) => p.value === prefs.get("betsPeriod")) ? prefs.get("betsPeriod") : "30d";
  const draw = () => {
    box.innerHTML = `${resultsHtml(slips, period)}<div class="section">${listHtml(slips, which)}</div>`;
  };
  draw();
  box.onclick = (event) => {
    const seg = event.target.closest('[data-seg="mb-period"]');
    if (seg) {
      period = seg.dataset.value;
      prefs.set("betsPeriod", period);
      draw();
      return;
    }
    const share = event.target.closest("[data-share-slip]");
    const del = event.target.closest("[data-delete-slip]");
    const id = (share || del)?.dataset.shareSlip || (share || del)?.dataset.deleteSlip;
    const s = id ? slips.find((x) => String(x.id) === id) : null;
    if (!s) return;
    if (share) openShare(s);
    else deleteSlip(s, del, () => renderSlips(box, which, { ...ctx, slips: () => ctx.slips(true) }));
  };
}
