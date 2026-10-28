// Brain progress (My bets → AI record → Brain progress; also #/bets?tab=brain, and linked from Help → How the brain
// learns): how the brain is scoring itself against the sharp market and simple baselines, what it changed and why, what
// it tested and held, what is running in shadow, what the connected AI is doing and what it costs. One request,
// GET /api/brain/progress (it reads the store only); the owner also gets Undo on each live change (POST /api/brain/undo
// {version}), the Learning On / Paused switch (POST /api/brain/pause {paused}) and, while the tripwire holds the
// brain in safe mode, Clear safe mode (POST /api/brain/clear-safe-mode); the server refuses these to anyone else with
// a 403 and says who the owner is in the payload ({owner}).
//
// Blocks, top to bottom (method.md section 8, as the spec simplifies it):
//   the honest promise and the learning switch · 1 headline per sport (a button: it picks the sport below) ·
//   the sport row (the one filter: it scopes 2, 3 and 7) with 2 score over time, 3 calibration (+ PIT, new targets)
//   and 7 adjustments · 4 what changed and why · 5 held steady · 6 experiments · 8 AI · 9 health.
//
//   renderBrain(box, {signal, query}) -> draws into ``box``; query: sport=<key> picks the sport first
// Every figure is the server's; status words come from its fields (model.js); text goes in through esc().

import { api, isQuiet } from "../../api.js";
import { replaceAddress, leagueTitle } from "../../app.js";
import { esc, section, skeleton, errorState, emptyState, notice, badge, segmented, toast, confirmSheet, ago } from "../../ui.js";
import { icon } from "../../icons.js";
import { ensureBrainStyles } from "./styles.js";
import {
  normalizeProgress, ll, range, fixed, dayLabel, timeLabel, weekLabel, moveText, sourceWords, providerName, games,
} from "./model.js";
import { gapChartHtml, bindGapChart, reliabilityHtml, bindReliability, pitHtml, bindPit } from "./charts.js";
import { learningSwitchHtml, setLearningPaused } from "./learning.js";

// The badge of a ledger entry: the gate's decision, or what later happened to a promotion (its rollback or undo)
const DECISION_BADGE = {
  promoted: ["Promoted", "good", "check"],
  shadow: ["In shadow", "info", "clock"],
  rolled_back: ["Rolled back", "warn", "refresh"],
  undone: ["Undone by the owner", "muted", "refresh"],
  safe_mode: ["Safe mode", "warn", "alert"],
  cleared: ["Safe mode cleared", "muted", "check"],
};
const RECENT_CHANGES = 3;
const GATE_WORDS = { nudge: "small nudge", structural: "bigger change, needed proof", rollback: "rollback", owner: "the owner", tripwire: "tripwire" };
const GATE_TESTS = {
  nudge: "Non-inferiority on held-out games: the top of the 95% range must stay within a small margin of 0",
  structural: "Superiority: Diebold-Mariano (HLN) on daily totals, Holm-adjusted within the week, then 2 weeks in shadow",
};

const pValue = (p) => (p === null ? "" : p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
const dollars = (cents) => (cents === null ? "–" : `$${(cents / 100).toFixed(2)}`);
const share = (a, b) => (a === null || !b ? 0 : Math.max(0, Math.min(100, (a / b) * 100)));
const meter = (part, whole, label) => `<span class="meter" role="img" aria-label="${esc(label)}"><span style="width:${share(part, whole).toFixed(1)}%"></span></span>`;
const kv = (rows) => `<ul class="bp-kv">${rows.filter(Boolean).map(([k, v]) => `<li><span>${esc(k)}</span><b>${esc(v)}</b></li>`).join("")}</ul>`;
const nums = (rows) => `<dl class="bp-nums">${rows.filter(Boolean).map(([k, v, small]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}${small ? `<small>${esc(small)}</small>` : ""}</dd></div>`).join("")}</dl>`;
const withRange = (iv, fmt = ll) => (iv ? `${fmt(iv.value)}${iv.lo !== null && iv.hi !== null ? ` (95% range ${range(iv.lo, iv.hi, fmt)})` : ""}` : "–");
const byWords = (source, model) => {
  if (!source || source === "template") return "Written from a template";
  const who = providerName(source.startsWith("ai:") ? source.slice(3) : source) || "the AI";
  return `Written by ${who}${model ? ` (${model})` : ""} and checked against the gate's numbers`;
};

// --- 0 · the promise and the switch -----------------------------------------------------------------------------------

function introHtml(d) {
  const l = d.learning;
  const state = l.enabled ? badge("Learning on", "good", "check") : badge("Learning paused", "warn", "pause");
  const control = d.owner
    ? `<div class="bp-status">${learningSwitchHtml(l, { name: "bp-learning" })}</div>`
    : `<div class="bp-status">${state}<span class="grow">${esc(l.enabled ? "The brain scores itself every night and decides on changes once a week." : "Learning is paused: the brain keeps its current values.")}</span></div>`;
  const safe = d.safeMode ? notice(`Safe mode${d.safeMode.since ? ` since ${timeLabel(d.safeMode.since)}` : ""}: every sport is on its starting values until the owner clears it.${d.safeMode.reason ? ` ${d.safeMode.reason}` : ""}`, "warn")
    + (d.owner ? `<div class="bp-status"><button type="button" class="btn btn-secondary" data-bp-clear-safe>${icon("check", { size: 16 })}Clear safe mode</button></div>` : "") : "";
  return `<section class="card section bp-intro">
    <div class="section-head" style="margin-bottom:0"><h2>${icon("trend", { size: 18 })}Is the brain getting better?</h2></div>
    <p>The brain leans mostly on the market, which is very hard to beat. Getting better means its chances are well calibrated (when it says 60%, it happens about 60% of the time), its log loss on games it hasn't seen moves toward or below the sharp market's, and new targets beat simple baselines. <b>Most weeks nothing changes</b>, and the page says why.</p>
    <p>The AI never forecasts old games (it may already know how they ended): it reads the misses and proposes changes, judged only on games played after the proposal. <a href="#/help/brain">How the brain learns</a></p>
    ${safe}
    ${control}
    ${d.asOf ? `<p class="hint" style="margin-top:0">${esc(`Numbers as of ${timeLabel(d.asOf)}`)}</p>` : ""}
  </section>`;
}

// --- 1 · headline per sport -------------------------------------------------------------------------------------------

function versionLine(s) {
  const v = s.version;
  const base = v.number === 0 ? "the starting values (version 0)" : `version ${v.number}`;
  const date = v.since ? `, since ${dayLabel(v.since)}` : "";
  const src = v.source === "owner" ? " · set by the owner" : v.source && v.source !== "default" && v.number ? ` · ${sourceWords(v.source)}` : "";
  return `Live: ${base}${date}${src}`;
}

export function headlineText(s) {
  const h = s.headline;
  if (h.text) return h.text;
  if (!h.n) return `${s.label}: no graded games with a market price yet.`;
  if (h.gap === null) return `${s.label}, last ${games(h.n)}: ${h.word || "too few days to say"}.`;
  const r = range(h.lo, h.hi);
  return `${s.label}, last ${games(h.n)}: ${h.word || "compared with the sharp market"} (log loss ${ll(h.gap)}${r ? `, 95% range ${r}` : ""}).`;
}

function headlinesHtml(d, selected) {
  if (!d.sports.length) {
    return section("How it's doing", emptyState({
      iconName: "chart", title: "Nothing graded yet",
      body: "The brain scores itself on games after they finish, against the sharp market's price on the same games. The first scores show here after the first night with graded games; the first weekly decisions need a few hundred games per sport.",
    }));
  }
  return section("How it's doing", `<ul class="bp-heads">${d.sports.map((s) => `<li>
      <button type="button" class="bp-head" data-bp-sport="${esc(s.sport)}" aria-pressed="${s.sport === selected}" aria-controls="bp-sport">
        <span class="bp-head-top"><b>${esc(s.label)}</b><span class="bp-ver">${esc(versionLine(s))}</span></span>
        <p>${esc(headlineText(s))}</p>
      </button></li>`).join("")}</ul>
    <p class="hint">Log loss is the score the brain is judged on: lower is better, and 0 here means level with the sharp market on the same games. A few ten-thousandths either way is normal noise.</p>`, { info: "log_loss" });
}

// --- 2, 3, 7 · one sport ----------------------------------------------------------------------------------------------

function calibrationWords(cal) {
  const rows = [
    cal.slope ? ["Calibration slope", withRange(cal.slope, (v) => fixed(v, 2)), "1 is ideal: below 1 overconfident, above 1 too timid"] : null,
    cal.intercept ? ["Calibration intercept", withRange(cal.intercept, (v) => fixed(v, 2)), "0 is ideal"] : null,
    cal.n !== null ? ["Games", (cal.n ?? 0).toLocaleString(), "the latest graded games"] : null,
  ];
  return `${cal.words ? `<p class="bp-words">${esc(cal.words)}</p>` : ""}${nums(rows)}`;
}

/** Every target the brain is scored on this week, each against its own baseline (the sharp market, climatology...). */
function targetsHtml(s) {
  if (!s.targets.length) return "";
  const week = s.targets.find((t) => t.week)?.week;
  return `<h3 class="bp-sub">By target${week ? `, week of ${esc(weekLabel(week))}` : ""}</h3>
    <p class="hint" style="margin-top:0">Each target's log loss (counts: log score) minus its baseline's on the same games: below 0 the brain did better. Corners and cards show on games, as "model only, low confidence", only once they beat climatology through the gate.</p>
    <div class="table-wrap" style="margin-top:6px"><table class="bp-targets"><caption class="sr-only">${esc(`${s.label}: each target against its baseline`)}</caption>
      <thead><tr><th scope="col">Target</th><th scope="col">Games</th><th scope="col">Against</th><th scope="col">Difference</th></tr></thead>
      <tbody>${s.targets.map((t) => `<tr><th scope="row">${esc(t.label)}${t.status ? `<small class="bp-cell-note">${esc(t.status)}</small>` : ""}</th>
        <td class="num">${esc(t.n ?? "–")}</td><td>${esc(t.vs || "–")}</td>
        <td class="num">${esc(ll(t.gap))}${t.lo !== null && t.hi !== null ? `<small class="bp-cell-note">${esc(range(t.lo, t.hi))}</small>` : ""}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

function adjustmentsHtml(a) {
  if (!a) return "";
  const body = a.n
    ? `${nums([
        ["Adjusted games", a.n.toLocaleString(), a.games !== null ? `of ${a.games.toLocaleString()} with a total line${a.week ? `, week of ${weekLabel(a.week)}` : ""}` : ""],
        ["Change in log loss", a.dll ? ll(a.dll.value) : "–", a.dll && a.dll.lo !== null ? `95% range ${range(a.dll.lo, a.dll.hi)}` : "adjusted minus unadjusted, same games"],
        ["Change in Brier", a.dbrier ? ll(a.dbrier.value) : "–", a.dbrier && a.dbrier.lo !== null ? `95% range ${range(a.dbrier.lo, a.dbrier.hi)}` : ""],
        ["Slope β", a.beta ? fixed(a.beta.value, 2) : "–", a.beta && a.beta.lo !== null ? `95% range ${range(a.beta.lo, a.beta.hi, (v) => fixed(v, 2))}` : "too few games for a slope"],
      ])}
      ${a.byCause.length ? `<p class="bp-meta" style="margin-top:6px">${esc(`By cause: ${a.byCause.map((c) => `${c.label} ${c.n}`).join(" · ")}`)}</p>` : ""}`
    : '<p class="bp-empty">No adjusted games graded this week.</p>';
  return `<h3 class="bp-sub">Adjustments (weather and roof)</h3>
    <p class="hint" style="margin-top:0">The same games scored with and without the late weather or roof adjustment. β is the share of the applied change that showed up in the final score: 1 means the size was right, 0 means it was noise.</p>
    ${body}`;
}

function sportBodyHtml(s) {
  const pits = s.targets.filter((t) => t.pit.length);
  return `<h3 class="bp-sub" style="margin-top:6px">Score over time</h3>
    ${s.series.length ? '<p class="bp-axis-y">Brain minus sharp market, log loss per game (lower is better)</p><div data-bp-gap></div>' : '<p class="bp-empty">No weekly scores yet: the first one comes after a week of graded games.</p>'}
    <div class="bp-two">
      <div><h3 class="bp-sub">Calibration</h3>
        ${s.calibration ? `${s.calibration.curve.length ? "<div data-bp-cal></div>" : ""}${calibrationWords(s.calibration)}` : '<p class="bp-empty">Not enough graded games for a calibration curve yet.</p>'}</div>
      ${pits.length ? `<div><h3 class="bp-sub">Spread of the counts (PIT)</h3>
        <p class="hint" style="margin-top:0">Flat bars mean the predicted spread is right; a U shape means too confident, a hump too cautious.</p>
        ${pits.map((t) => `<div data-bp-pit="${esc(t.target)}"></div>`).join("")}</div>` : ""}
    </div>
    ${targetsHtml(s)}
    ${adjustmentsHtml(s.adjustments)}`;
}

// --- 4 · what changed and why -----------------------------------------------------------------------------------------

function changeHtml(c, owner, shadowNow = new Set()) {
  const later = c.rolledBack ? (c.rolledBack.decision === "undone" ? "undone" : "rolled_back") : null;
  const [word, kind, iconName] = DECISION_BADGE[later || c.decision] || ["", "muted", ""];
  const e = c.evidence;
  const moved = (c.decision === "promoted" || c.decision === "shadow") && c.old !== null && c.new !== null;
  const tiles = e.n !== null || e.dll !== null ? nums([
    ["Judged on", e.n === null ? "–" : games(e.n), e.trainN ? `held out; fitted on ${e.trainN.toLocaleString()}` : "held out of the fit"],
    ["Change in log loss", ll(e.dll), e.lo !== null && e.hi !== null ? `95% range ${range(e.lo, e.hi)}` : ""],
  ]) : "";
  // The gate's test is named only when it left evidence: an AI proposal frozen as a challenger has none yet (it is
  // judged on games after it was made; "not tested yet" only while it is still among the experiments), and the
  // week's candidate count belongs to a tested change only
  const frozen = moved && c.decision === "shadow" && e.n === null && e.p === null;
  const test = frozen
    ? (!c.rolledBack && shadowNow.has(c.version) ? `Not tested yet: scored in shadow on games after ${c.at ? dayLabel(c.at) : "it was made"}; it is judged once it has enough games` : "")
    : [moved && GATE_TESTS[c.gate] ? GATE_TESTS[c.gate] : "", pValue(e.p), e.holmP !== null ? `Holm-adjusted ${e.holmP < 0.001 ? "< 0.001" : e.holmP.toFixed(3)}` : "",
      moved && e.tried ? `${e.tried} candidate${e.tried === 1 ? "" : "s"} tried that week` : ""].filter(Boolean).join(" · ");
  const words = c.text || c.reason;
  const gate = c.reason && c.text && !c.text.includes(c.reason) ? `<p class="bp-meta">${esc(`The gate: ${c.reason}`)}</p>` : "";
  const back = c.rolledBack
    ? `<div class="bp-back${c.rolledBack.decision === "undone" ? " owner" : ""}">${icon("refresh", { size: 14 })} ${esc(`${c.rolledBack.decision === "undone" ? "Undone by the owner" : "Rolled back"}${c.rolledBack.at ? ` on ${dayLabel(c.rolledBack.at)}` : ""}${c.rolledBack.reason ? `: ${c.rolledBack.reason}` : "."}`)}</div>` : "";
  return `<li class="bp-item" id="bp-change-${esc(c.id)}" tabindex="-1">
    <div class="bp-item-top"><h3>${esc(c.label)}</h3>${word ? badge(word, kind, iconName) : ""}</div>
    ${moved ? `<div class="bp-move">${esc(moveText(c.old, c.new))}</div>` : ""}
    <div class="bp-meta">${esc([c.sportLabel, c.at ? dayLabel(c.at) : "", c.version !== null ? `version ${c.version}` : "", GATE_WORDS[c.gate] || c.gate, sourceWords(c.source)]
      .filter((w, i, all) => w && all.findIndex((x) => String(x).toLowerCase() === String(w).toLowerCase()) === i).join(" · "))}</div>
    ${tiles}
    ${test ? `<p class="bp-meta">${esc(test)}</p>` : ""}
    ${words ? `<p>${esc(words)}</p><span class="bp-by">${esc(c.text ? byWords(c.textSource, c.textModel) : "The gate's own words")}</span>` : ""}
    ${gate}
    ${back}
    ${owner && c.canUndo && c.version !== null ? `<div class="bp-actions"><button type="button" class="btn btn-secondary" data-bp-undo="${esc(c.version)}" data-bp-label="${esc(c.label)}" data-bp-sport-label="${esc(c.sportLabel)}" data-bp-shadow="${c.decision === "shadow"}">${icon("refresh", { size: 16 })}${c.decision === "shadow" ? "Withdraw" : "Undo"}</button></div>` : ""}
  </li>`;
}

function changesHtml(d) {
  // The newest few in full; older ones folded (the changelog only grows), opened by a marker that points into them
  const recent = d.changes.slice(0, RECENT_CHANGES);
  const older = d.changes.slice(RECENT_CHANGES);
  const shadowNow = new Set(d.experiments.map((x) => x.version).filter((v) => v !== null));
  const body = d.changes.length
    ? `<ul class="bp-list">${recent.map((c) => changeHtml(c, d.owner, shadowNow)).join("")}</ul>
      ${older.length ? `<details class="bp-more bp-older"><summary>${icon("down", { size: 16 })}${esc(`${older.length} older change${older.length === 1 ? "" : "s"}`)}</summary>
        <ul class="bp-list">${older.map((c) => changeHtml(c, d.owner, shadowNow)).join("")}</ul></details>` : ""}`
    : `<p class="bp-empty">No changes yet. A change goes live only when the gate's evidence says it's safe, and most weeks nothing changes.</p>`;
  return section("What changed and why", `${body}
    <p class="note">A change applies only to games logged after it; nothing already shown changes. The previous version keeps being scored in the background, so a change that turns out worse is reversed automatically.</p>`, { id: "bp-changes" });
}

// --- 5 · held steady --------------------------------------------------------------------------------------------------

function heldHtml(d) {
  const h = d.held;
  if (!h || !h.tested) {
    return section("Held steady", `<p class="bp-empty">${esc(h?.words || "No weekly test has run yet.")} Each Monday the brain tests a short list of candidate changes; most weeks none of them passes, and this is where it says why.</p>`);
  }
  const items = h.items.map((i) => `<li class="bp-item">
      <div class="bp-item-top"><h3>${esc(i.label)}</h3>${i.decision === "rejected" ? badge("Didn't pass", "muted", "close") : badge("Held", "info", "pause")}</div>
      <div class="bp-meta">${esc([i.sportLabel, i.n !== null ? `${games(i.n)} judged` : "", i.needs ? `about ${i.needs.toLocaleString()} more needed` : ""].filter(Boolean).join(" · "))}</div>
      ${i.reason ? `<p>${esc(i.reason)}</p>` : ""}
    </li>`).join("");
  return section("Held steady", `<p class="bp-held-sum">${esc(h.words || `${h.tested} candidates tested, ${h.passed || "none"} passed.`)}</p>
    ${h.week ? `<p class="bp-meta">${esc(`The weekly decision for the week of ${weekLabel(h.week)}`)}</p>` : ""}
    ${items ? `<ul class="bp-list" style="margin-top:10px">${items}</ul>` : ""}
    <p class="note">Edges this small need thousands of games to prove, so "not yet" is the usual answer. That is the honest version of constantly improving when the brain is already close to the market.</p>`);
}

// --- 6 · experiments --------------------------------------------------------------------------------------------------

function experimentsHtml(d) {
  if (!d.experiments.length) return section("Experiments", '<p class="bp-empty">Nothing is running in shadow right now. A bigger change, or any AI proposal, is scored in the background on new games first and goes live only if it proves itself.</p>');
  return section("Experiments", `<ul class="bp-list">${d.experiments.map((e) => `<li class="bp-item">
      <div class="bp-item-top"><h3>${esc(e.label)}</h3>${badge("In shadow", "info", "clock")}</div>
      <div class="bp-meta">${esc([e.sportLabel, sourceWords(e.source), e.since ? `since ${dayLabel(e.since)}` : "", e.version !== null ? `version ${e.version}` : ""].filter(Boolean).join(" · "))}</div>
      <div class="bp-progress"><div class="row-between"><b>${esc(e.needed ? `${e.n.toLocaleString()} of ${e.needed.toLocaleString()} games` : games(e.n))}</b><span>scored in shadow</span></div>
        ${e.needed ? meter(e.n, e.needed, `${e.n} of ${e.needed} games scored in shadow`) : ""}</div>
      <p>${esc(e.dll !== null ? `Running change in log loss: ${ll(e.dll)}${e.lo !== null && e.hi !== null ? ` (95% range ${range(e.lo, e.hi)})` : ""}. Below 0 is better; it decides nothing until the minimum is reached.` : "Too few games for a running score yet.")}</p>
    </li>`).join("")}</ul>`);
}

// --- 8 · AI -----------------------------------------------------------------------------------------------------------

function aiHtml(d) {
  const a = d.ai;
  const who = a.provider ? `${providerName(a.provider)}${a.model ? ` (${a.model})` : ""}` : a.reason || "None: notes come from templates";
  const note = a.note
    ? `<div class="bp-note">${esc(a.note.text)}</div>
       ${a.note.hypotheses.length ? `<ul class="bp-hyp">${a.note.hypotheses.map((h) => `<li>${esc(h)}</li>`).join("")}</ul>` : ""}
       <p class="bp-by" style="margin-top:6px">${esc([byWords(a.note.source, a.note.model), a.note.at ? timeLabel(a.note.at) : ""].filter(Boolean).join(" · "))}</p>`
    : '<p class="bp-empty">No nightly note yet.</p>';
  const memo = a.memo
    ? `<h3 class="bp-sub">Monthly review${a.memo.month ? ` (${esc(a.memo.month)})` : ""}</h3><div class="bp-note">${esc(a.memo.text)}</div>
       ${a.memo.ideas.length ? `<ul class="bp-hyp">${a.memo.ideas.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : ""}
       <p class="bp-by" style="margin-top:6px">${esc([byWords(a.memo.source, a.memo.model), a.memo.at ? timeLabel(a.memo.at) : ""].filter(Boolean).join(" · "))}. Ideas only: nothing here changes the brain by itself.</p>`
    : "";
  const p = a.proposals;
  const c = a.cost;
  return section("The AI's part", `${kv([["In use", who], a.provider && a.reason ? ["Why", a.reason] : null, a.pending ? ["Calls waiting for an answer", String(a.pending)] : null])}
    <h3 class="bp-sub">Latest nightly note</h3>${note}
    <h3 class="bp-sub">Proposals</h3>
    ${nums([["Made", String(p.made)], ["Passed", String(p.passed)], ["Failed", String(p.failed)], ["Waiting for later games", String(p.pending)],
      p.refused ? ["Refused by the code's checks", String(p.refused)] : null])}
    <h3 class="bp-sub">Spend this month</h3>
    <div class="bp-progress"><div class="row-between"><b>${esc(dollars(c.spentCents))}</b><span>${esc(c.capCents !== null ? `of the ${dollars(c.capCents)} monthly cap` : "no cap set")}</span></div>
      ${c.capCents ? meter(c.spentCents ?? 0, c.capCents, `${dollars(c.spentCents)} of ${dollars(c.capCents)} spent this month`) : ""}
      ${c.reservedCents ? `<span class="bp-meta">${esc(`plus ${dollars(c.reservedCents)} set aside for calls still out`)}</span>` : ""}</div>
    ${c.capCents && c.spentCents !== null && c.spentCents >= c.capCents ? notice("This month's cap is reached: notes and changelogs come from templates until next month.", "info") : ""}
    ${kv([["Texts rejected by the numbers check", String(a.rejected)]])}
    ${memo}
    <p class="note">The AI never forecasts old games, never sets a value that ships, and its proposals are judged only on games played after they were made. Its spend never touches people's chat allowance.</p>`, { id: "bp-ai" });
}

// --- 9 · health -------------------------------------------------------------------------------------------------------

function healthHtml(d) {
  const h = d.health;
  const graded = h.sports.length
    ? kv(h.sports.map((s) => [s.label, `${s.graded.toLocaleString()} graded · ${s.voided.toLocaleString()} voided${s.reasons.length ? ` (${s.reasons.map(([k, v]) => `${k} ${v}`).join(", ")})` : ""}`]))
    : '<p class="bp-empty">No games graded yet.</p>';
  const backfill = h.backfill.length ? `<ul class="bp-list">${h.backfill.map((b) => `<li class="bp-progress">
      <div class="row-between"><b>${esc(b.label)}</b><span>${esc(b.units ? (b.todo ? "Filling" : "Done") : "Waiting")}</span></div>
      ${b.units ? meter(b.done, b.units, `${b.done} of ${b.units} parts loaded`) : ""}
      <span class="bp-meta">${esc([b.units ? `${b.done.toLocaleString()} of ${b.units.toLocaleString()} parts loaded` : "Not started",
        b.games ? `${b.games.toLocaleString()} games stored` : "", b.errors ? `${b.errors} error${b.errors === 1 ? "" : "s"}` : "",
        b.updated ? `last ${ago(b.updated)}` : ""].filter(Boolean).join(" · "))}</span>
    </li>`).join("")}</ul>` : `<p class="bp-empty">${h.history.length ? "The nightly backfill hasn't run yet; the games below were stored as they were graded." : "Nothing loaded yet: old games load a little each night."}</p>`;
  const history = h.history.length ? `<div class="table-wrap" style="margin-top:8px"><table><caption class="sr-only">Old games stored, per league</caption>
      <thead><tr><th scope="col">League</th><th scope="col">Games</th><th scope="col">With closing lines</th><th scope="col">With stats</th></tr></thead>
      <tbody>${h.history.slice(0, 12).map((x) => `<tr><th scope="row">${esc(leagueTitle(x.league) || x.league)}</th><td class="num">${esc(x.games.toLocaleString())}</td>
        <td class="num">${esc(x.close.toLocaleString())}</td><td class="num">${esc(x.stats.toLocaleString())}</td></tr>`).join("")}</tbody></table></div>` : "";
  const run = (t) => (t ? `${timeLabel(t)} (${ago(t)})` : "not yet");
  const t = h.trials;
  return section("Health", `<h3 class="bp-sub" style="margin-top:0">Graded and voided games</h3>${graded}
    <h3 class="bp-sub">Old games being loaded</h3><p class="hint" style="margin-top:0">Results and closing lines from free public sources, a little each night, to test the parts that can be rebuilt from closing prices.</p>
    ${backfill}${history}
    <h3 class="bp-sub">Runs</h3>
    ${kv([["Nightly scoring", run(h.runs.night)], ["Weekly decision", run(h.runs.week)], ["Monthly review", run(h.runs.month)],
      ["Last 31 days", t.month ? `${t.month} candidate${t.month === 1 ? "" : "s"} tried, ${t.promoted} promoted, ${t.rolledBack} rolled back` : "No candidates tried yet"],
      h.backlog !== null ? ["Games waiting over 2 days for a grade", String(h.backlog)] : null])}`);
}

// --- the screen -------------------------------------------------------------------------------------------------------

export async function renderBrain(box, { signal, query = new URLSearchParams() } = {}) {
  ensureBrainStyles();
  box.innerHTML = `<div class="bp">${skeleton("lines", 6)}</div>`;
  let data;
  const load = async () => normalizeProgress(await api("/api/brain/progress", { signal }));
  try {
    data = await load();
  } catch (error) {
    if (!isQuiet(error) && box.isConnected) box.innerHTML = `<div class="bp">${errorState(error)}</div>`;
    return;
  }
  if (!box.isConnected) return;

  let selected = null;
  let drawnWidth = 0;
  const pick = (key) => (data.sports.some((s) => s.sport === key) ? key : data.sports[0]?.sport || null);
  selected = pick(query.get("sport"));

  const sportSection = () => {
    if (!data.sports.length) return "";
    const s = data.sports.find((x) => x.sport === selected);
    const picker = data.sports.length > 1
      ? `<div class="bp-sports" role="navigation" aria-label="Sport">${segmented("bp-sport", data.sports.map((x) => ({ value: x.sport, label: x.label })), selected, { label: "Sport" })}</div>` : "";
    return `<section class="card section" id="bp-sport" aria-label="${esc(`${s.label}: scores, calibration and adjustments`)}">
      <div class="section-head"><h2>${esc(s.label)}</h2><span class="muted small">${esc(versionLine(s))}</span></div>
      ${picker}
      <div class="bp-sport-box" data-bp-box>${sportBodyHtml(s)}</div>
    </section>`;
  };

  // Charts are drawn at the width they're shown, so their text stays 11 px on a phone and a desk
  const drawCharts = () => {
    const s = data.sports.find((x) => x.sport === selected);
    const holder = box.querySelector("[data-bp-box]");
    if (!s || !holder) return;
    drawnWidth = holder.clientWidth;
    const gap = holder.querySelector("[data-bp-gap]");
    if (gap) {
      gap.innerHTML = gapChartHtml(s, data.changes, { width: gap.clientWidth });
      bindGapChart(gap.querySelector("figure"), s, data.changes, { onPick: openChange });
    }
    const cal = holder.querySelector("[data-bp-cal]");
    if (cal && s.calibration) {
      cal.innerHTML = reliabilityHtml(s.calibration, { width: cal.clientWidth, label: s.label });
      bindReliability(cal.querySelector("figure"), s.calibration);
    }
    for (const slot of holder.querySelectorAll("[data-bp-pit]")) {
      const t = s.targets.find((x) => x.target === slot.dataset.bpPit);
      if (!t) continue;
      const hist = { label: t.label, n: t.n, bins: t.pit, lo: null, hi: null };
      slot.innerHTML = `<p class="bp-meta" style="margin:6px 0 0">${esc(`${hist.label}${hist.n !== null ? ` · ${games(hist.n)}` : ""}`)}</p>${pitHtml(hist, { width: slot.clientWidth })}`;
      bindPit(slot.querySelector("figure"), hist);
    }
  };

  const draw = () => {
    box.innerHTML = `<div class="bp">${introHtml(data)}${headlinesHtml(data, selected)}${sportSection()}${changesHtml(data)}${heldHtml(data)}${experimentsHtml(data)}${aiHtml(data)}${healthHtml(data)}</div>`;
    drawCharts();
  };

  function openChange(change) {
    const card = box.querySelector(`#bp-change-${CSS.escape(String(change.id))}`);
    if (!card) return;
    const fold = card.closest("details");
    if (fold) fold.open = true;
    card.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    card.classList.remove("flash");
    void card.offsetWidth;
    card.classList.add("flash");
    card.focus({ preventScroll: true });
  }

  const choose = (key, { scroll = false } = {}) => {
    if (!key || key === selected) {
      if (scroll) box.querySelector("#bp-sport")?.scrollIntoView({ block: "start" });
      return;
    }
    selected = key;
    for (const b of box.querySelectorAll("[data-bp-sport]")) b.setAttribute("aria-pressed", String(b.dataset.bpSport === key));
    const holder = box.querySelector("#bp-sport");
    if (holder) holder.outerHTML = sportSection();
    drawCharts();
    const hash = `#/bets?tab=record&view=brain&sport=${encodeURIComponent(key)}`;
    if (location.hash !== hash && location.hash.startsWith("#/bets")) replaceAddress(hash);
    if (scroll) box.querySelector("#bp-sport")?.scrollIntoView({ block: "start" });
  };

  // The redraw replaces the button the confirm sheet handed focus back to: ``then`` puts focus on what changed, so
  // keyboard and screen-reader users keep their place
  const focusOn = (el, { center = false } = {}) => {
    if (!el) return;
    const fold = el.closest("details");
    if (fold) fold.open = true;
    if (!el.matches("button, a[href], input, select, textarea, [tabindex]")) el.setAttribute("tabindex", "-1");
    el.focus({ preventScroll: center });
    if (center) el.scrollIntoView({ block: "center" });
  };
  const learningFocus = () => box.querySelector('[data-seg="bp-learning"][aria-pressed="true"]') || box.querySelector(".bp-intro h2");

  const reload = async (then) => {
    try {
      const fresh = await load();
      if (!box.isConnected) return;
      data = fresh;
      selected = pick(selected);
      draw();
      then?.();
    } catch (error) {
      if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
    }
  };

  draw();

  box.addEventListener("click", async (event) => {
    const head = event.target.closest("[data-bp-sport]");
    if (head) {
      choose(head.dataset.bpSport, { scroll: true });
      return;
    }
    const seg = event.target.closest("[data-seg]");
    if (seg?.dataset.seg === "bp-sport") {
      choose(seg.dataset.value);
      return;
    }
    if (seg?.dataset.seg === "bp-learning") {
      if (seg.getAttribute("aria-pressed") === "true" || seg.disabled) return;
      const paused = seg.dataset.value === "paused";
      if (paused) {
        const ok = await confirmSheet({
          title: "Pause learning?",
          body: "The brain keeps its current values: nothing is tuned, promoted or rolled back, and no AI calls are made, until you switch learning on again. Scores keep being recorded.",
          confirm: "Pause learning",
        });
        if (!ok) return;
      }
      try {
        await setLearningPaused(paused);
        toast(paused ? "Learning paused" : "Learning is on again", { kind: "good" });
      } catch (error) {
        if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
      }
      await reload(() => focusOn(learningFocus()));
      return;
    }
    const clear = event.target.closest("[data-bp-clear-safe]");
    if (clear) {
      const ok = await confirmSheet({
        title: "Clear safe mode?",
        body: "The live versions serve again for games from now on. Clear it only once you know why the tripwire fired.",
        confirm: "Clear safe mode",
      });
      if (!ok) return;
      clear.disabled = true;
      try {
        const out = await api("/api/brain/clear-safe-mode", { method: "POST", body: {} });
        toast(out?.cleared ? "Safe mode cleared" : "Safe mode was already off", { kind: "good" });
      } catch (error) {
        if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
      }
      // Still in safe mode (the clear failed): its button again; otherwise the learning switch
      await reload(() => focusOn(box.querySelector("[data-bp-clear-safe]") || learningFocus()));
      return;
    }
    const undo = event.target.closest("[data-bp-undo]");
    if (undo) {
      const version = Number(undo.dataset.bpUndo);
      const shadow = undo.dataset.bpShadow === "true";
      const ok = await confirmSheet(shadow ? {
        title: "Withdraw this experiment?",
        body: `${undo.dataset.bpSportLabel}: "${undo.dataset.bpLabel}" stops being scored in shadow. Nothing live changes.`,
        confirm: "Withdraw it",
      } : {
        title: "Undo this change?",
        body: `${undo.dataset.bpSportLabel}: "${undo.dataset.bpLabel}" goes back to the version before it, for games from now on. Games already logged keep the numbers they were shown with.`,
        confirm: "Undo the change",
      });
      if (!ok) return;
      undo.disabled = true;
      const card = undo.closest(".bp-item")?.id || "";
      try {
        const out = await api("/api/brain/undo", { method: "POST", body: { version } });
        // A withdrawn challenger was never live: now_live is then the champion that served all along, not a revert.
        // The server's own word wins when it sends one (a challenger promoted since the page loaded is undone instead)
        const withdrawn = typeof out?.withdrawn === "boolean" ? out.withdrawn : shadow;
        const now = typeof out?.now_live === "number" ? out.now_live : null;
        toast(withdrawn ? "Experiment withdrawn: nothing live changed"
          : now === null ? "Change undone" : now === 0 ? "Undone: back on the starting values" : `Undone: back on version ${now}`, { kind: "good", ms: 3600 });
      } catch (error) {
        if (!isQuiet(error)) toast(error.message, { kind: "error", ms: 4000 });
      }
      // Back to the card that was acted on (it now says what happened), else the newest entry for that version
      await reload(() => {
        const sameVersion = data.changes.find((c) => c.version === version);
        focusOn((card && box.querySelector(`#${CSS.escape(card)}`))
          || (sameVersion && box.querySelector(`#bp-change-${CSS.escape(String(sameVersion.id))}`))
          || box.querySelector("#bp-changes h2"), { center: true });
      });
    }
  });

  // Redraw the charts when the column's width changes enough to matter (rotation, the desk's panes)
  if (typeof ResizeObserver === "function") {
    let timer = 0;
    const watch = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!box.isConnected) { watch.disconnect(); return; }
        const holder = box.querySelector("[data-bp-box]");
        if (holder && Math.abs(holder.clientWidth - drawnWidth) > 24) drawCharts();
      }, 120);
    });
    watch.observe(box);
    signal?.addEventListener("abort", () => watch.disconnect());
  }
}
