// My bets → Leaders and AI record, laid out for a phone (the first dashboard showed these as wide tables).
//   Leaders:   GET /api/leaderboard: people's calls scored against the AI and the bookmakers
//   AI record: GET /api/track-record: every NBA game the AI predicted before tip-off, graded after it
//              (winners, Brier scores next to the bookmakers', calibration, the spread and total), the
//              daily Safe / Risky / Crazy AI bets' record, and the AI analysts (Claude, DeepSeek) with
//              the trust they've earned. Misses included.
// Every figure is the server's; this file only lays them out.

import { api, isQuiet } from "../../api.js";
import { esc, pct, shortName, badge, emptyState, errorState, skeleton, section, explain, shortDate } from "../../ui.js";
import { icon } from "../../icons.js";

const LEAGUE = "basketball_nba"; // the one league with an AI model so far
const signed = (n, digits = 1) => (n == null ? "–" : `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(Number(n)).toFixed(digits)}`);
const brier = (x) => (x == null ? "–" : Number(x).toFixed(3));
const TIER_ICONS = { safe: "shield", risky: "scale", crazy: "zap" };

// --- leaderboard --------------------------------------------------------------------------------------

export async function renderLeaders(box, { signal }) {
  box.innerHTML = skeleton("lines", 5);
  let rows;
  try {
    rows = await api("/api/leaderboard", { signal });
  } catch (error) {
    if (!isQuiet(error) && box.isConnected) box.innerHTML = errorState(error);
    return;
  }
  if (!box.isConnected) return;
  const list = rows.length ? `<ol class="mb-ranks">${rows.map((r, i) => {
    const you = r.mine === true; // the server marks the asker's own rows
    return `<li class="mb-rank${you ? " me" : ""}">
      <span aria-label="Rank ${i + 1}">${i + 1}</span>
      <div class="grow"><b>${esc(r.name)}${you ? " (you)" : ""}</b>
        <span class="muted">${r.calls} call${r.calls === 1 ? "" : "s"} · beat the AI ${r.beat_ai}/${r.calls}${r.market_calls ? ` · vs bookmakers ${signed(r.vs_market)}` : ""}</span></div>
      <b>${signed(r.points)} pts<small>vs AI ${signed(r.vs_ai)}</small></b>
    </li>`;
  }).join("")}</ol>` : emptyState({
    iconName: "trophy", title: "No graded calls yet",
    body: "Calls are scored after each game finishes. Make one on any NBA game page.",
    actions: [{ label: "Go to Today", href: "#/today", primary: true }],
  });
  box.innerHTML = `${section("Leaderboard", `${list}`, { info: "calls", aside: `<a class="text-link" href="#/bets?tab=calls">Your calls ${icon("chevron", { size: 16 })}</a>` })}
    <p class="note">Points from scored calls. “vs AI” is your points minus the AI's on the same games: copying the AI scores 0,
      so above 0 means you read those games better. “vs bookmakers” compares winner calls with the market.</p>`;
}

// --- the AI's record ----------------------------------------------------------------------------------

const rec = (x) => (x ? `${x.won}–${x.lost}${x.push ? `–${x.push}` : ""}` : "–");
const rate = (x) => (x && x.won + x.lost ? pct(x.won / (x.won + x.lost)) : "–"); // a record's win share, as the old dashboard showed it

function stat(value, label) {
  return `<div class="stat"><div class="stat-v">${value}</div><div class="stat-k">${label}</div></div>`;
}

/** Brier scores as bars (shorter is better), the AI next to the bookmakers and "always the home team". */
function brierBars(r) {
  const rows = [["AI", r.brier, "ai"], ["Bookmakers", r.market_brier, ""], ["Always home", r.baseline_brier, "base"]].filter(([, v]) => v != null);
  const top = Math.max(0.3, ...rows.map(([, v]) => v));
  return `<div class="mb-brier" role="img" aria-label="${esc(`Brier scores, lower is better: ${rows.map(([k, v]) => `${k} ${brier(v)}`).join(", ")}`)}">
    ${rows.map(([label, value, cls]) => `<div class="mb-brier-row" aria-hidden="true"><span>${esc(label)}</span>
      <span class="mb-track ${cls}"><span style="width:${((value / top) * 100).toFixed(1)}%"></span></span><b>${brier(value)}</b></div>`).join("")}
  </div>
  <p class="hint">Lower is better. Always saying 50% scores 0.250; the closing market is near 0.20 on the NBA.</p>`;
}

function calibrationHtml(buckets) {
  const used = (buckets || []).filter((b) => b.games);
  if (!used.length) return "";
  return section("Calibration", `<p class="small muted">When the AI says X%, does it happen X% of the time? Small samples swing a lot.</p>
    <ul class="mb-calib" style="margin-top:10px">${used.map((b) => `<li>
      <div class="row-between"><b>${esc(b.range)}</b><span class="muted">${b.games} game${b.games === 1 ? "" : "s"}</span></div>
      <div class="mb-pair" role="img" aria-label="${esc(`AI said ${pct(b.predicted)}, happened ${pct(b.actual)}`)}">
        <div class="row-between tiny" aria-hidden="true"><span>AI said ${pct(b.predicted)}</span><span>happened ${pct(b.actual)}</span></div>
        <span class="mb-track ai" aria-hidden="true"><span style="width:${Math.round(b.predicted * 100)}%"></span></span>
        <span class="mb-track" aria-hidden="true"><span style="width:${Math.round(b.actual * 100)}%"></span></span>
      </div></li>`).join("")}</ul>`);
}

function tiersHtml(tiers) {
  if (!tiers?.length) return "";
  return section("AI bets record", `<p class="small muted">The top Safe, Risky and Crazy bet each day, saved before its first game and graded at the price shown. Expect a return a little below zero: the bookmaker's cut.</p>
    <div style="margin-top:6px">${tiers.map((t) => {
      const ret = t.roi == null ? "–" : `${signed(t.roi * 100)}%`;
      return `<div class="mb-tier ${esc(t.tier)}"><span class="mb-tier-icon">${icon(TIER_ICONS[t.tier] || "target", { size: 18 })}</span>
        <div class="grow"><b>${esc(t.title)}</b><div class="small">${t.tickets ? `${t.won} of ${t.tickets} won · AI said ${pct(t.predicted)}, happened ${pct(t.actual)}${t.void ? ` · ${t.void} void` : ""}` : "No bets graded yet"}</div></div>
        <b>${t.tickets ? esc(ret) : "–"}<small>return</small></b></div>`;
    }).join("")}</div>`, { info: "tiers" });
}

function analystsHtml(analysts) {
  if (!analysts?.length) return "";
  return section("AI analysts", `<p class="small muted">Claude and DeepSeek forecast games too, and are graded like everyone else. They move the AI's numbers only once their record shows it helps.</p>
    <div class="stack-sm" style="margin-top:10px">${analysts.map((a) => {
      const vsAi = a.points != null && a.ai_points != null ? a.points - a.ai_points : null;
      const trust = a.games < a.min_games ? `0% <span class="muted">(${a.games}/${a.min_games} games)</span>` : pct(a.trust);
      return `<div class="mb-analyst">
        <div class="row-between"><b class="row">${icon("bot", { size: 18 })}${esc(a.label)}</b>${a.enabled ? badge("On", "good", "check") : badge("Off", "muted")}</div>
        <dl class="mb-nums">
          <div><dt>Graded games</dt><dd>${a.games}</dd></div>
          <div><dt>Winners right</dt><dd>${a.games ? pct(a.accuracy) : "–"}</dd></div>
          <div><dt>Brier ${explain("brier")}</dt><dd>${brier(a.brier)} <span class="muted">AI ${brier(a.brier_ai)} · books ${brier(a.brier_market)}</span></dd></div>
          <div><dt>Points vs the AI</dt><dd>${vsAi == null ? "–" : signed(vsAi)}</dd></div>
          <div><dt>Trust earned</dt><dd>${trust}</dd></div>
          <div><dt>Calls in 24 h</dt><dd>${a.calls_24h}/${a.daily_limit} <span class="muted">$${Number(a.cost_24h || 0).toFixed(2)} ($${Number(a.cost || 0).toFixed(2)} total)</span></dd></div>
        </dl></div>`;
    }).join("")}</div>`, { info: "analysts" });
}

function recentHtml(recent) {
  if (!recent?.length) return "";
  return section("Latest games", `<ul class="mb-games">${recent.map((g) => {
    const homeFav = g.home_win >= 0.5;
    const team = shortName(homeFav ? g.home : g.away, LEAGUE);
    const right = g.hit
      ? `<span class="mb-mark good">${icon("check", { size: 14 })}Right</span>`
      : `<span class="mb-mark bad">${icon("close", { size: 14 })}Wrong</span>`;
    return `<li><div><b>${esc(shortName(g.away, LEAGUE))} @ ${esc(shortName(g.home, LEAGUE))}</b>
        <div class="muted">${g.start ? `${esc(shortDate(g.start))} · ` : ""}AI: ${esc(team)} ${pct(Math.max(g.home_win, 1 - g.home_win))} · final ${g.final_away}–${g.final_home}</div></div>
      ${right}</li>`;
  }).join("")}</ul>`);
}

export async function renderRecord(box, { signal }) {
  box.innerHTML = skeleton("lines", 6);
  let r;
  try {
    r = await api(`/api/track-record?league=${LEAGUE}`, { signal });
  } catch (error) {
    if (!isQuiet(error) && box.isConnected) box.innerHTML = errorState(error);
    return;
  }
  if (!box.isConnected) return;
  const head = `<p class="muted">Every NBA game the AI predicted before tip-off, graded after the final buzzer. Honest numbers, misses included.
    <a href="#/help/ai">How the AI decides</a></p>`;
  if (!r.games) {
    box.innerHTML = `${head}${emptyState({ iconName: "chart", title: "No graded games yet", body: "The record starts with the games the AI predicts from now on: check back after tonight's games." })}
      ${tiersHtml(r.tiers)}${analystsHtml(r.analysts)}`;
    return;
  }
  box.innerHTML = `${head}
    <section class="card section">
      <div class="row-between"><h2 style="font-size:16px">Winners right</h2><span class="muted small">${r.games} games graded</span></div>
      <div class="mb-net"><span class="mb-net-v">${pct(r.winner_accuracy)}</span><span class="muted">of games, the AI's favourite won${r.games < 100 ? ` (only ${r.games} games: a small sample)` : ""}</span></div>
      ${brierBars(r)}
      <div class="mb-statgrid">
        ${stat(rec(r.ats), `vs the closing spread (${rate(r.ats)})`)}
        ${stat(rec(r.totals), `vs the closing total (${rate(r.totals)})`)}
        ${stat(Number(r.margin_error).toFixed(1), "average margin miss (pts)")}
        ${stat(Number(r.total_error).toFixed(1), "average total miss (pts)")}
        ${stat(Number(r.log_loss).toFixed(3), "log loss (lower is better)")}
        ${stat(brier(r.brier), `AI Brier score ${explain("brier")}`)}
      </div>
    </section>
    ${calibrationHtml(r.calibration)}
    ${tiersHtml(r.tiers)}
    ${analystsHtml(r.analysts)}
    ${recentHtml(r.recent)}
    <p class="note">What good looks like: nobody gets near 100%. The closing betting market picks about 68% of NBA winners
      (65.7–69.8% by season) with a Brier score near 0.20; a sound model lands at 66–69% and 0.20–0.21. Against the
      closing spread, 50–53% is realistic and 52.4% is needed to break even at standard prices.</p>`;
}
