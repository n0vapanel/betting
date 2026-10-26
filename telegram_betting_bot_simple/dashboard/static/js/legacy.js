// The first dashboard's features, re-homed for the new screens. Nothing here is lost from the old
// app.js; each piece is now a function a screen can call:
//   openCallSheet(game, {onSaved})   "Make your call" for an NBA game, judged live by POST /api/judge
//   analystsHtml(game) + bindAnalysts(container, game, rerender)   Claude/DeepSeek forecasts with "Ask now"
//   renderMyCalls(root), renderLeaderboard(root), renderTrackRecord(root, league)   (My bets screen)
//   renderAiBets(root, {league, source})   the Safe / Risky / Crazy AI bets for a league (Build screen)
//   loadMyCalls(), callSummary(call, game), copyText(text, what)
// The calls API predates per-person accounts: it takes this browser's id as ?player= (same as X-Player).

import { api, getPlayerId, prefs, isQuiet } from "./api.js";
import { me } from "./store.js";
import {
  esc, pct, money, priceText, shortName, sheet, toast, errorState, emptyState, skeleton, segmented, explain,
  timeLabel, ago, badge,
} from "./ui.js";
import { icon } from "./icons.js";

export const STATS = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "Threes", stl: "Steals", blk: "Blocks", tov: "Turnovers", pra: "Pts+Reb+Ast" };
// FiveThirtyEight's forecasting-game points for the chance given to the team that won
export const forecastPoints = (p, weight = 1) => Math.round(10 * weight * (25 - 100 * (1 - Math.round(p * 100) / 100) ** 2)) / 10;
const PROP_WEIGHT = 0.4;
export const CAUTION_GAP = 0.08; // the AI this far from the bookmakers gets a "check the news" warning
const signed = (n, digits = 1) => `${n > 0 ? "+" : ""}${Number(n).toFixed(digits)}`;
const agreement = (p) => (p >= 0.6 ? "AI agrees" : p >= 0.4 ? "Coin flip for the AI" : "AI disagrees");
const compareConfidence = (yours, ai) => (ai < 0.5 ? "against the AI" : Math.abs(yours - ai) <= 0.05 ? "same as the AI"
  : yours > ai ? "bolder than the AI" : "more cautious than the AI");
const marginText = (margin, game) => {
  if (margin == null) return "–";
  const team = margin >= 0 ? game.home : game.away;
  return `${shortName(team, game.league)} by ${Math.abs(margin).toFixed(1)}`;
};
const player = () => encodeURIComponent(getPlayerId());

// --- copying ---------------------------------------------------------------------------------------

/** Copy text; the clipboard API needs https, so fall back to the older copy command, then to a sheet. */
export async function copyText(text, what = "Text") {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${what} copied`, { kind: "good" });
    return;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.className = "offscreen";
    document.body.appendChild(area);
    area.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch {
      copied = false;
    }
    area.remove();
    if (copied) {
      toast(`${what} copied`, { kind: "good" });
      return;
    }
  }
  const s = sheet.open({
    title: what,
    body: `<p class="muted">Select the text and copy it.</p><textarea class="input" readonly rows="8">${esc(text)}</textarea>`,
  });
  s.body.querySelector("textarea").select();
}

// --- my calls --------------------------------------------------------------------------------------

let callsCache = null;

/** This browser's calls (GET /api/calls?player=...), cached until a call is saved or deleted. */
export async function loadMyCalls({ fresh = false } = {}) {
  if (!callsCache || fresh) callsCache = await api(`/api/calls?player=${player()}`);
  return callsCache;
}

export function callSummary(call, game) {
  const parts = [];
  if (call.winner) parts.push(`${shortName(call.winner === "home" ? game.home : game.away, game.league)} ${pct(call.confidence)}`);
  if (call.home_score != null) parts.push(`${call.away_score}–${call.home_score}`);
  if (call.props?.length) parts.push(`${call.props.length} player call${call.props.length > 1 ? "s" : ""}`);
  return parts.join(" · ");
}

/**
 * "Make your call" on a game with an AI prediction (NBA): winner and confidence, an optional final
 * score and up to 8 player lines, with the AI's view of each part as it's made (POST /api/judge).
 * ``game`` is the /api/game response (its players fill the player picker).
 */
export async function openCallSheet(game, { onSaved } = {}) {
  const p = game.prediction;
  if (!p) return;
  let existing = null;
  try {
    existing = (await loadMyCalls()).find((c) => c.event_id === game.id) || null;
  } catch (error) {
    if (error.signIn) return;
  }
  const draft = {
    game,
    verdict: null,
    players: ["away", "home"].flatMap((side) =>
      (game.players?.[side] || []).filter((pl) => !pl.out).map((pl) => ({ ...pl, team: shortName(game[side], game.league) }))),
    call: existing && existing.status === "pending"
      ? JSON.parse(JSON.stringify(existing.call))
      : { winner: p.home_win >= 0.5 ? "home" : "away", confidence: 0.6, home_score: null, away_score: null, props: [] },
  };
  const away = shortName(game.away, game.league);
  const home = shortName(game.home, game.league);
  const s = sheet.open({
    title: `Your call: ${away} @ ${home}`,
    size: "tall",
    body: callForm(draft),
    footer: `<button type="button" class="btn btn-secondary" data-call-cancel>Cancel</button>
      <button type="button" class="btn btn-primary grow" data-call-save>${existing?.status === "pending" ? "Update call" : "Save call"}</button>`,
  });
  const $ = (sel) => s.panel.querySelector(sel);
  let judgeTimer = null;

  const renderProps = () => {
    const options = (selected) => draft.players.map((pl) =>
      `<option value="${esc(pl.id)}"${String(pl.id) === String(selected) ? " selected" : ""}>${esc(pl.name)} (${esc(pl.team)})</option>`).join("");
    const judged = draft.verdict?.props || [];
    $("[data-props]").innerHTML = draft.call.props.map((prop, i) => {
      const ai = judged.find((j) => String(j.player_id) === String(prop.player_id) && j.stat === prop.stat && Number(j.line) === Number(prop.line) && j.side === prop.side);
      const conf = prop.confidence ?? 0.6;
      return `<div class="prop-row" data-prop="${i}">
        <select class="input" data-k="player_id" aria-label="Player">${draft.players.length ? options(prop.player_id) : '<option value="">No players listed</option>'}</select>
        <select class="input" data-k="stat" aria-label="Stat">${Object.entries(STATS).map(([k, v]) => `<option value="${k}"${k === prop.stat ? " selected" : ""}>${v}</option>`).join("")}</select>
        <input class="input" type="number" inputmode="decimal" data-k="line" min="0" max="150" step="0.5" value="${prop.line}" aria-label="Line">
        <div class="seg small">${["over", "under"].map((side) => `<button type="button" data-side="${side}" aria-pressed="${prop.side === side}">${side === "over" ? "Over" : "Under"}</button>`).join("")}</div>
        <select class="input" data-k="confidence" aria-label="How sure are you?">${[55, 60, 65, 70, 75, 80, 90, 100].map((c) =>
          `<option value="${c / 100}"${Math.round(conf * 100) === c ? " selected" : ""}>${c}%</option>`).join("")}</select>
        <span class="prop-ai">${ai ? `<b class="${ai.ai_prob >= 0.6 ? "good" : ai.ai_prob < 0.4 ? "bad" : ""}">AI ${pct(ai.ai_prob)}</b> <span class="muted">avg ${ai.ai_mean.toFixed(1)}</span> · ` : ""}
          <span class="muted" title="Your points if this call is right / wrong">${signed(forecastPoints(conf, PROP_WEIGHT))} / ${signed(forecastPoints(1 - conf, PROP_WEIGHT))} pts</span></span>
        <button type="button" class="icon-btn" data-remove-prop="${i}" aria-label="Remove this player call">${icon("close", { size: 18 })}</button>
      </div>`;
    }).join("");
    $("[data-add-prop]").hidden = draft.call.props.length >= 8 || !draft.players.length;
  };

  const updateHints = () => {
    const { call, verdict } = draft;
    const aiWinner = call.winner === "home" ? p.home_win : 1 - p.home_win;
    const team = shortName(call.winner === "home" ? game.home : game.away, game.league);
    $("[data-conf-label]").textContent = pct(call.confidence);
    const marketHome = p.sources?.market?.home_win_prob;
    const market = marketHome != null ? (call.winner === "home" ? marketHome : 1 - marketHome) : null;
    const caution = marketHome != null && Math.abs(p.home_win - marketHome) >= CAUTION_GAP;
    $("[data-winner-hint]").innerHTML = `<b class="${aiWinner >= 0.6 ? "good" : aiWinner < 0.4 ? "bad" : ""}">${agreement(aiWinner)}</b>:
      it gives ${esc(team)} <b>${pct(aiWinner)}</b>${market != null ? ` (bookmakers ${pct(market)})` : ""}; you're ${compareConfidence(call.confidence, aiWinner)}.
      Points: <b class="good">${signed(forecastPoints(call.confidence))}</b> if ${esc(team)} win,
      <b class="bad">${signed(forecastPoints(1 - call.confidence))}</b> if they lose.
      ${caution ? `<br><span class="warn">${icon("alert", { size: 14 })} The AI is far from the bookmakers on this game: check the injury news.</span>` : ""}`;
    const score = verdict?.score;
    $("[data-score-hint]").innerHTML = score && call.home_score != null
      ? `${esc(shortName(score.winner_side === "home" ? game.home : game.away, game.league))} by ${Math.abs(score.margin)}+: AI <b>${pct(score.wins_by_at_least)}</b> ·
         total over ${score.total}: AI <b>${pct(score.total_over_yours)}</b>`
      : "Leave blank to call just the winner.";
    $("[data-verdict]").textContent = verdict?.verdict || "";
    $("[data-verdict]").hidden = !verdict?.verdict;
    for (const b of s.panel.querySelectorAll("[data-winner]")) b.setAttribute("aria-pressed", String(b.dataset.winner === call.winner));
  };

  const payload = () => {
    const call = draft.call;
    const out = { winner: call.winner, confidence: call.confidence, props: call.props.filter((x) => x.player_id) };
    if (call.home_score != null && call.away_score != null) {
      out.home_score = call.home_score;
      out.away_score = call.away_score;
    }
    return out;
  };

  const judge = () => {
    clearTimeout(judgeTimer);
    updateHints();
    judgeTimer = setTimeout(async () => {
      if (!s.open) return;
      try {
        const verdict = await api("/api/judge", { method: "POST", body: { league: game.league, id: game.id, call: payload() } });
        if (!s.open) return;
        draft.verdict = verdict;
        $("[data-call-error]").textContent = "";
      } catch (error) {
        if (s.open && !isQuiet(error)) $("[data-call-error]").textContent = error.message;
        return;
      }
      renderProps();
      updateHints();
    }, 250);
  };

  // Lines default to just under the player's projection, so changing player or stat resets them
  const suggestLine = (prop) => {
    const pl = draft.players.find((x) => String(x.id) === String(prop.player_id));
    if (!pl) return prop.line;
    const mean = prop.stat === "pra" ? pl.pts + pl.reb + pl.ast : pl[prop.stat] ?? 0;
    return Math.max(0.5, Math.round(mean) - 0.5);
  };

  const addProp = () => {
    const used = new Set(draft.call.props.map((x) => String(x.player_id)));
    const next = draft.players.find((pl) => !used.has(String(pl.id))) || draft.players[0];
    if (!next) return;
    const prop = { player_id: String(next.id), stat: "pts", line: 0.5, side: "over", confidence: 0.6 };
    prop.line = suggestLine(prop);
    draft.call.props.push(prop);
    renderProps();
    judge();
  };

  const save = async (button) => {
    const name = $("[data-name]").value.trim();
    if (!name) {
      $("[data-call-error]").textContent = "Add your name for the leaderboard.";
      $("[data-name]").focus();
      return;
    }
    prefs.setName(name);
    button.disabled = true;
    try {
      await api("/api/calls", { method: "POST", body: { league: game.league, id: game.id, player: getPlayerId(), name, call: payload() } });
      callsCache = null;
      await s.close();
      toast("Call saved: it's scored after the game", { kind: "good" });
      onSaved?.();
    } catch (error) {
      if (!isQuiet(error)) $("[data-call-error]").textContent = error.message;
    } finally {
      button.disabled = false;
    }
  };

  s.panel.addEventListener("click", (event) => {
    const t = event.target.closest("button");
    if (!t) return;
    if (t.dataset.winner) {
      draft.call.winner = t.dataset.winner;
      const call = draft.call;
      // A winner that contradicts the score clears the score
      if (call.home_score != null && call.away_score != null && (call.home_score > call.away_score) !== (call.winner === "home")) {
        call.home_score = call.away_score = null;
        $("[data-home-score]").value = $("[data-away-score]").value = "";
      }
      judge();
    } else if (t.dataset.side) {
      draft.call.props[Number(t.closest(".prop-row").dataset.prop)].side = t.dataset.side;
      renderProps();
      judge();
    } else if (t.dataset.removeProp != null) {
      draft.call.props.splice(Number(t.dataset.removeProp), 1);
      renderProps();
      judge();
    } else if (t.hasAttribute("data-add-prop")) {
      addProp();
    } else if (t.hasAttribute("data-call-cancel")) {
      s.close();
    } else if (t.hasAttribute("data-call-save")) {
      save(t);
    }
  });
  s.panel.addEventListener("input", (event) => {
    const input = event.target;
    if (input.matches("[data-confidence]")) {
      draft.call.confidence = Number(input.value) / 100;
      updateHints();
    } else if (input.matches("[data-home-score], [data-away-score]")) {
      const homeValue = $("[data-home-score]").value;
      const awayValue = $("[data-away-score]").value;
      draft.call.home_score = homeValue === "" ? null : Math.round(Number(homeValue));
      draft.call.away_score = awayValue === "" ? null : Math.round(Number(awayValue));
      if (draft.call.home_score != null && draft.call.away_score != null && draft.call.home_score !== draft.call.away_score) {
        draft.call.winner = draft.call.home_score > draft.call.away_score ? "home" : "away";
      }
      judge();
    }
  });
  s.panel.addEventListener("submit", (event) => event.preventDefault());
  s.panel.addEventListener("change", (event) => {
    const input = event.target;
    if (input.matches("[data-confidence]")) judge();
    const row = input.closest(".prop-row");
    if (!row) return;
    const prop = draft.call.props[Number(row.dataset.prop)];
    const key = input.dataset.k;
    if (key === "line") prop.line = Math.max(0, Math.round(Number(input.value) * 2) / 2);
    else if (key === "confidence") prop.confidence = Number(input.value);
    else if (key) prop[key] = input.value;
    if (key === "player_id" || key === "stat") prop.line = suggestLine(prop);
    renderProps();
    judge();
  });
  renderProps();
  judge();
}

function callForm(draft) {
  const { game, call } = draft;
  const p = game.prediction;
  const away = shortName(game.away, game.league);
  const home = shortName(game.home, game.league);
  return `<form class="call-form" autocomplete="off">
    <p class="muted small">${esc(timeLabel(game.start))} · the AI says ${esc(away)} ${Math.round(p.away_score)}, ${esc(home)} ${Math.round(p.home_score)}</p>
    <div class="field">
      <span class="label">Who wins? ${explain("calls")}</span>
      <div class="seg big">${["away", "home"].map((side) => `<button type="button" data-winner="${side}" aria-pressed="${call.winner === side}">${esc(side === "home" ? home : away)}</button>`).join("")}</div>
    </div>
    <div class="field">
      <label class="label" for="call-confidence">How sure are you? <b data-conf-label>${pct(call.confidence)}</b></label>
      <input class="range" id="call-confidence" type="range" data-confidence min="50" max="100" step="1" value="${Math.round(call.confidence * 100)}">
      <div class="hint" data-winner-hint></div>
    </div>
    <div class="field">
      <span class="label">Final score <span class="muted">(optional)</span></span>
      <div class="score-row">
        <label class="muted" for="call-away">${esc(away)}</label><input class="input" id="call-away" type="number" inputmode="numeric" data-away-score min="40" max="250" placeholder="${Math.round(p.away_score)}" value="${call.away_score ?? ""}">
        <label class="muted" for="call-home">${esc(home)}</label><input class="input" id="call-home" type="number" inputmode="numeric" data-home-score min="40" max="250" placeholder="${Math.round(p.home_score)}" value="${call.home_score ?? ""}">
      </div>
      <div class="hint" data-score-hint></div>
      <div class="hint">Scored as 15 minus your margin miss, plus 20 minus your total miss, plus 10 if both scores are within 2.</div>
    </div>
    <div class="field">
      <span class="label">Player calls <span class="muted">(optional, up to 8; each counts 40% of a winner call)</span></span>
      <div data-props class="stack-sm"></div>
      <button type="button" class="btn btn-ghost" data-add-prop>${icon("plus", { size: 18 })}Add a player call</button>
    </div>
    <div class="field">
      <label class="label" for="call-name">Your name <span class="muted">(for the leaderboard)</span></label>
      <input class="input" id="call-name" type="text" data-name maxlength="24" value="${esc(prefs.name() || me().name || "")}" placeholder="e.g. Sam">
    </div>
    <div class="verdict" data-verdict hidden></div>
    <div class="form-error" role="alert" data-call-error></div>
  </form>`;
}

/** The person's calls list (My bets → Calls). */
export async function renderMyCalls(root) {
  root.innerHTML = skeleton("lines", 4);
  let calls;
  try {
    calls = await loadMyCalls({ fresh: true });
  } catch (error) {
    root.innerHTML = errorState(error);
    return;
  }
  if (!root.isConnected) return;
  const graded = calls.filter((c) => c.status === "graded");
  const points = graded.reduce((sum, c) => sum + c.user_points, 0);
  const aiPoints = graded.reduce((sum, c) => sum + c.ai_points, 0);
  const vsAi = points - aiPoints;
  const winnerParts = graded.map((c) => c.result.parts.find((part) => part.part === "winner" && part.market_points != null)).filter(Boolean);
  const vsMarket = winnerParts.reduce((sum, part) => sum + part.points - part.market_points, 0);
  const headline = !graded.length ? "Your calls are scored after each game, next to the AI's score on the same games."
    : vsAi >= 0 ? `You beat the AI by ${vsAi.toFixed(1)} points on the ${graded.length} game${graded.length > 1 ? "s" : ""} you called.`
    : `The AI is ahead of you by ${(-vsAi).toFixed(1)} points on the ${graded.length} game${graded.length > 1 ? "s" : ""} you called.`;
  root.innerHTML = `
    <p class="muted">${esc(headline)} ${explain("calls")}</p>
    <div class="stats">
      <div class="stat"><div class="stat-v">${graded.length}/${calls.length}</div><div class="stat-k">calls scored</div></div>
      <div class="stat"><div class="stat-v">${signed(points)}</div><div class="stat-k">your points</div></div>
      <div class="stat"><div class="stat-v ${vsAi >= 0 ? "good" : "bad"}">${signed(vsAi)}</div><div class="stat-k">vs the AI (same games)</div></div>
      <div class="stat"><div class="stat-v ${vsMarket >= 0 ? "good" : "bad"}">${signed(vsMarket)}</div><div class="stat-k">vs the bookmakers (winner calls)</div></div>
    </div>
    ${calls.length ? `<div class="stack">${calls.map(callCard).join("")}</div>` : emptyState({
      iconName: "target", title: "No calls yet",
      body: "Open an NBA game on Today and tap “Make your call”. It's scored after the game against the AI.",
      actions: [{ label: "Go to Today", href: "#/today", primary: true }],
    })}
    <p class="note">Winner calls: 25 − 100 × (1 − p)², p = the chance you gave the team that won. 50% always scores 0; a
      calibrated 70% call averages only about +4 a game, and a wrong 90% call costs −56.</p>`;
  root.onclick = async (event) => {
    const button = event.target.closest("[data-delete-call]");
    if (!button) return;
    button.disabled = true;
    try {
      await api(`/api/calls/${encodeURIComponent(button.dataset.deleteCall)}?player=${player()}`, { method: "DELETE" });
      callsCache = null;
      toast("Call deleted");
      renderMyCalls(root);
    } catch (error) {
      button.disabled = false;
      if (!isQuiet(error)) toast(error.message, { kind: "error" });
    }
  };
}

function callCard(row) {
  const game = { home: row.home, away: row.away, league: row.league };
  const call = row.call;
  const ai = row.ai || {};
  const lines = [];
  if (call.winner) {
    const team = call.winner === "home" ? row.home : row.away;
    lines.push(`Winner: <b>${esc(team)}</b> at ${pct(call.confidence)} · AI ${pct(ai.winner?.ai_prob)}${ai.winner?.market_prob != null ? ` · bookmakers ${pct(ai.winner.market_prob)}` : ""}`);
  }
  if (call.home_score != null) {
    lines.push(`Score: ${esc(shortName(row.away, row.league))} ${call.away_score}–${call.home_score} ${esc(shortName(row.home, row.league))} · AI ${Math.round(ai.away_score)}–${Math.round(ai.home_score)}`);
  }
  for (const prop of ai.props || []) {
    lines.push(`${esc(prop.player)} ${prop.side} ${prop.line} ${STATS[prop.stat]?.toLowerCase() || prop.stat} at ${pct(prop.confidence)} · AI ${pct(prop.ai_prob)}`);
  }
  let status = badge("Pending", "info", "clock");
  let result = "";
  if (row.status === "graded") {
    const r = row.result;
    const diff = row.user_points - row.ai_points;
    status = badge(`${diff > 0 ? "Beat the AI" : diff === 0 ? "Tied the AI" : "AI won"} ${signed(diff)}`, diff >= 0 ? "good" : "danger", diff >= 0 ? "check" : "close");
    result = `<div class="result">Final: ${esc(shortName(row.away, row.league))} ${r.away}–${r.home} ${esc(shortName(row.home, row.league))} ·
      you <b class="${row.user_points >= 0 ? "good" : "bad"}">${signed(row.user_points)}</b> · AI <b>${signed(row.ai_points)}</b>${r.market_points != null ? ` · bookmakers ${signed(r.market_points)} <span class="muted">(winner)</span>` : ""}
      <div class="chip-row wrap">${r.parts.map((part) => `<span class="part">${part.part === "prop" ? esc(part.player || "") : part.part}${part.void ? " void" : `: ${signed(part.points)}`}</span>`).join("")}</div></div>`;
  } else if (row.status === "void") {
    status = badge("Void", "muted");
  }
  const actions = row.status === "pending" && new Date(row.start) > new Date()
    ? `<button type="button" class="btn btn-ghost small" data-delete-call="${esc(row.id)}">${icon("trash", { size: 16 })}Delete</button>` : "";
  return `<div class="card call-card">
    <div class="row-between"><span class="muted small">${esc(timeLabel(row.start))} · ${esc(row.away)} @ ${esc(row.home)}</span>${status}</div>
    ${lines.map((l) => `<div class="call-line">${l}</div>`).join("")}
    ${result}
    ${ai.verdict || actions ? `<div class="row-between"><span class="muted small">${esc(ai.verdict || "")}</span>${actions}</div>` : ""}
  </div>`;
}

// --- leaderboard and the AI's record -------------------------------------------------------------------

export async function renderLeaderboard(root) {
  root.innerHTML = skeleton("lines", 5);
  let rows;
  try {
    rows = await api("/api/leaderboard");
  } catch (error) {
    root.innerHTML = errorState(error);
    return;
  }
  if (!root.isConnected) return;
  // The server marks the asker's own rows ("mine": true) and blanks everyone else's id: on the shared
  // password a browser's id is all that tells people apart, so anyone holding it could act as them.
  const isMe = (r) => r.mine === true;
  root.innerHTML = `
    <p class="muted">Points from scored calls. "vs AI" is your points minus the AI's on the same games: copying the AI
      scores 0, so above 0 means you read those games better. "vs bookmakers" compares winner calls with the market.</p>
    ${rows.length ? `<div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Name</th><th>Calls</th><th>Points</th><th>vs AI</th><th>vs books</th><th>Beat AI</th></tr></thead>
      <tbody>${rows.map((r, i) => `<tr${isMe(r) ? ' class="me"' : ""}>
        <td>${i + 1}</td><td>${esc(r.name)}${isMe(r) ? " (you)" : ""}</td><td>${r.calls}</td><td><b>${signed(r.points)}</b></td>
        <td class="${r.vs_ai >= 0 ? "good" : "bad"}">${signed(r.vs_ai)}</td><td>${r.market_calls ? signed(r.vs_market) : "–"}</td><td>${r.beat_ai}/${r.calls}</td></tr>`).join("")}
      </tbody></table></div>` : emptyState({ iconName: "trophy", title: "No graded calls yet", body: "Calls are scored after each game finishes." })}`;
}

export async function renderTrackRecord(root, league = "basketball_nba") {
  root.innerHTML = skeleton("lines", 6);
  let r;
  try {
    r = await api(`/api/track-record?league=${encodeURIComponent(league)}`);
  } catch (error) {
    root.innerHTML = errorState(error);
    return;
  }
  if (!root.isConnected) return;
  const rec = (x) => `${x.won}-${x.lost}${x.push ? `-${x.push}` : ""}`;
  const rate = (x) => (x.won + x.lost ? ` (${pct(x.won / (x.won + x.lost))})` : "");
  root.innerHTML = `
    <p class="muted">Every game the AI predicted before tip-off, graded after the final buzzer. Honest numbers, including the misses.</p>
    ${!r.games ? emptyState({ iconName: "chart", title: "No graded games yet", body: "The record starts with the games the AI predicts from now on: check back after tonight's games." }) : `
    <div class="stats">
      <div class="stat"><div class="stat-v">${r.games}</div><div class="stat-k">games graded</div></div>
      <div class="stat"><div class="stat-v">${pct(r.winner_accuracy)}</div><div class="stat-k">winners right</div></div>
      <div class="stat"><div class="stat-v">${r.brier.toFixed(3)}</div><div class="stat-k">AI Brier score ${explain("brier")}</div></div>
      <div class="stat"><div class="stat-v">${r.market_brier != null ? r.market_brier.toFixed(3) : "–"}</div><div class="stat-k">bookmakers' Brier</div></div>
      <div class="stat"><div class="stat-v">${r.baseline_brier.toFixed(3)}</div><div class="stat-k">"always the home team"</div></div>
      <div class="stat"><div class="stat-v">${r.log_loss.toFixed(3)}</div><div class="stat-k">AI log loss (lower is better)</div></div>
      <div class="stat"><div class="stat-v">${rec(r.ats)}${rate(r.ats)}</div><div class="stat-k">vs the closing spread</div></div>
      <div class="stat"><div class="stat-v">${rec(r.totals)}${rate(r.totals)}</div><div class="stat-k">vs the closing total</div></div>
      <div class="stat"><div class="stat-v">${r.margin_error.toFixed(1)}</div><div class="stat-k">avg margin miss (pts)</div></div>
      <div class="stat"><div class="stat-v">${r.total_error.toFixed(1)}</div><div class="stat-k">avg total miss (pts)</div></div>
    </div>
    <h3 class="subhead">Calibration: when the AI says X%, does it happen X% of the time?</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>AI's favourite at</th><th>Games</th><th>AI said</th><th>Happened</th><th></th></tr></thead>
      <tbody>${r.calibration.map((b) => `<tr><td>${esc(b.range)}</td><td>${b.games}</td><td>${pct(b.predicted)}</td><td>${pct(b.actual)}</td>
        <td class="calib">${b.games ? `<span class="meter"><span style="width:${Math.round(b.actual * 100)}%"></span></span>` : ""}</td></tr>`).join("")}</tbody>
    </table></div>
    ${tierRecord(r.tiers)}
    <h3 class="subhead">Latest games</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Game</th><th>AI</th><th>Final</th><th>Right?</th></tr></thead>
      <tbody>${r.recent.map((g) => `<tr><td>${esc(g.away)} @ ${esc(g.home)}</td>
        <td>${esc(shortName(g.home_win >= 0.5 ? g.home : g.away, league))} ${pct(Math.max(g.home_win, 1 - g.home_win))}</td>
        <td>${g.final_away}–${g.final_home}</td><td>${g.hit ? `<span class="good">${icon("check", { size: 16, label: "Right" })}</span>` : `<span class="bad">${icon("close", { size: 16, label: "Wrong" })}</span>`}</td></tr>`).join("")}</tbody>
    </table></div>`}
    ${!r.games ? tierRecord(r.tiers) : ""}
    ${analystRecord(r.analysts)}
    <p class="note">What good looks like: nobody gets near 100%. The closing betting market picks about 68% of NBA winners
      (65.7–69.8% by season) with a Brier score near 0.20; a sound model lands at 66–69% and 0.20–0.21. Against the
      closing spread, 50–53% is realistic and 52.4% is needed to break even at standard prices.</p>`;
}

// The AI analysts (Claude, DeepSeek): graded next to the AI and the bookmakers, with the trust they've earned
function analystRecord(analysts) {
  if (!analysts?.length) return "";
  const brier = (x) => (x == null ? "–" : x.toFixed(3));
  const rows = analysts.map((a) => `<tr><td>${esc(a.label)}${a.enabled ? "" : ' <span class="muted small">(off)</span>'}</td>
    <td>${a.games}</td><td>${a.games ? pct(a.accuracy) : "–"}</td><td>${brier(a.brier)}</td><td>${brier(a.brier_ai)}</td>
    <td>${brier(a.brier_market)}</td>
    <td class="${a.points == null || a.ai_points == null ? "" : a.points >= a.ai_points ? "good" : "bad"}">${a.points != null && a.ai_points != null ? signed(a.points - a.ai_points) : "–"}</td>
    <td>${a.games < a.min_games ? `0% <span class="muted small">(${a.games}/${a.min_games} games)</span>` : pct(a.trust)}</td>
    <td>${a.calls_24h}/${a.daily_limit} · $${a.cost_24h.toFixed(2)} <span class="muted small">($${a.cost.toFixed(2)} total)</span></td></tr>`).join("");
  return `
    <h3 class="subhead">AI analysts: Claude and DeepSeek, graded like everyone else ${explain("analysts")}</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Analyst</th><th>Graded</th><th>Winners right</th><th>Brier</th><th>AI's Brier</th><th>Books' Brier</th>
        <th>Points vs AI</th><th>Trust earned</th><th>Calls (24 h) · spend</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

// How the AI's own daily bets did, per risk level: is Safe really safe?
function tierRecord(tiers) {
  if (!tiers) return "";
  const rows = tiers.map((t) => `<tr><td>${esc(t.title)}</td><td>${t.tickets}</td><td>${t.won}</td>
    <td>${pct(t.predicted)}</td><td>${pct(t.actual)}</td>
    <td class="${t.roi == null ? "" : t.roi >= 0 ? "good" : "bad"}">${t.roi == null ? "–" : `${signed(t.roi * 100)}%`}</td></tr>`).join("");
  return `
    <h3 class="subhead">AI bets record: the top Safe, Risky and Crazy bet each day ${explain("tiers")}</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Level</th><th>Bets graded</th><th>Won</th><th>AI said</th><th>Happened</th><th>Return</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="note">Each level's top bet is saved before its first game starts and graded at the price shown. Expect
      returns a little below zero: the bookmakers' cut.</p>`;
}

// --- AI analysts on a game ----------------------------------------------------------------------------

let asking = null; // "gameId:analyst" while an analyst is being asked

/** The analysts' cards for a game (``game.analysts`` from /api/game: {analysts: [...], setup: [...]}), or a setup note. */
export function analystsHtml(game) {
  const view = game.analysts;
  if (!view || Array.isArray(view)) {
    return `<p class="muted small">No AI analysts are switched on. To add Claude and/or DeepSeek, put
      <code>ANTHROPIC_API_KEY=...</code> and/or <code>DEEPSEEK_API_KEY=...</code> in front of the install command on the
      server and run it again. They then forecast each NBA game a few hours before tip-off.</p>`;
  }
  const setup = Object.fromEntries((view.setup || []).map((entry) => [entry.key, entry]));
  return `<div class="stack-sm">${view.analysts.map((a) => analystCard(game, a, setup[a.key])).join("")}</div>
    <p class="note">Each analyst reads this game's odds, the AI's numbers, injuries, rest and player projections
      (Claude can also search the web for late news), then gives its own forecast, graded after the game. An analyst
      only moves the AI's numbers once its record shows it helps. Each call costs a little on the server's API account.</p>`;
}

// Links found by Claude's web search: only plain web addresses are shown as links
const webLink = (source) => /^https?:\/\//i.test(source.url || "")
  ? `<a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.title || source.url)}</a>` : "";

function analystCard(game, a, setup) {
  const f = a.forecast;
  const busy = asking === `${game.id}:${a.key}`;
  const usage = setup ? `${setup.calls_24h}/${setup.daily_limit} calls in 24 h` : "";
  const ask = a.enabled && !game.live
    ? `<button type="button" class="btn btn-secondary small" data-ask="${esc(a.key)}"${busy || asking ? " disabled" : ""}>${busy ? `Asking ${esc(a.label)}… (up to 2 min)` : f ? "Ask again" : `Ask ${esc(a.label)} now`}</button>`
    : "";
  const head = `<div class="analyst-head"><b>${icon("bot", { size: 16 })} ${esc(a.label)}</b>
    <span class="muted small">${esc(a.model || "")}${f ? ` · ${ago(a.created_at)}` : ""}${f && a.cost != null ? ` · about $${a.cost.toFixed(3)}` : ""}${usage ? ` · ${usage}` : ""}</span>${ask}</div>`;
  if (!f) {
    return `<div class="analyst">${head}<div class="muted small">${a.enabled
      ? "No forecast yet: it runs by itself a few hours before tip-off, or ask now."
      : "Switched off on the server (no API key)."}</div></div>`;
  }
  const home = f.home_win_prob >= 0.5;
  const side = (p) => (p == null ? null : home ? p : 1 - p);
  const team = shortName(home ? game.home : game.away, game.league);
  const gap = a.market_home_win != null ? side(f.home_win_prob) - side(a.market_home_win) : null;
  return `<div class="analyst">
    ${head}
    <div class="analyst-call">${esc(team)} <b>${pct(side(f.home_win_prob))}</b> · ${esc(marginText(f.home_margin, game))} ·
      total ${Number(f.total).toFixed(0)} · ${badge(`${f.confidence} confidence`, "muted")}</div>
    <div class="muted small">Same team: bookmakers ${pct(side(a.market_home_win))}, AI ${pct(side(a.base_home_win))}${gap != null && Math.abs(gap) >= 0.005 ? ` · ${gap > 0 ? "above" : "below"} the bookmakers by ${Math.abs(gap * 100).toFixed(0)} pts` : ""}</div>
    <p>${esc(f.summary)}</p>
    ${f.factors?.length ? `<ul class="bullets">${f.factors.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    <div class="small"><b>Where it sees an edge:</b> ${esc(f.edge)}</div>
    ${a.news ? `<details class="news-details"><summary>Late news it used</summary><p class="small">${esc(a.news)}</p>
      <div class="small">${(a.sources || []).map(webLink).filter(Boolean).join(" · ")}</div></details>` : ""}
    <div class="muted small">Trust earned: ${pct(a.trust)}${a.trust > 0
      ? " of its move away from the AI is in the AI's numbers"
      : " (its forecasts don't move the AI until its graded record shows they help)"}</div>
  </div>`;
}

/** Handle "Ask <analyst> now" inside ``container``; ``rerender()`` redraws the section after each change. */
export function bindAnalysts(container, game, rerender) {
  container.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-ask]");
    if (!button || asking) return;
    asking = `${game.id}:${button.dataset.ask}`;
    rerender();
    try {
      game.analysts = await api("/api/analysts/ask", { method: "POST", body: { league: game.league, id: game.id, analyst: button.dataset.ask } });
      toast("New forecast in", { kind: "good" });
    } catch (error) {
      if (!isQuiet(error)) toast(error.message, { kind: "error" });
    } finally {
      asking = null;
      rerender();
    }
  });
}

// --- AI bets: Safe, Risky, Crazy ------------------------------------------------------------------------

const TIERS = [
  { key: "safe", title: "Safe", iconName: "shield", blurb: "More likely, small payout." },
  { key: "risky", title: "Risky", iconName: "scale", blurb: "Balanced: a real chance and a real payout." },
  { key: "crazy", title: "Crazy", iconName: "zap", blurb: "Long shot, big payout." },
];

/** The AI's Safe / Risky / Crazy bets for one league (GET /api/ai-bets), with a source switch. */
export async function renderAiBets(root, { league = "basketball_nba", source = prefs.get("betsSource") === "mine" ? "mine" : "all" } = {}) {
  root.innerHTML = skeleton("cards", 2);
  let data;
  try {
    data = await api(`/api/ai-bets?league=${encodeURIComponent(league)}&source=${source}&player=${player()}`);
  } catch (error) {
    root.innerHTML = errorState(error);
    return;
  }
  if (!root.isConnected) return;
  root.innerHTML = `
    <div class="row-between wrap">
      <p class="muted small grow">${esc(data?.note || "Built from the AI's probabilities and the best current prices.")} ${explain("tiers")}</p>
      ${segmented("bets-source", [{ value: "all", label: "All games" }, { value: "mine", label: "From my calls" }], source, { label: "Build from" })}
    </div>
    <div class="tiers">${TIERS.map((tier) => tierColumn(tier, data.tiers?.[tier.key])).join("")}</div>`;
  root.onclick = (event) => {
    const seg = event.target.closest('[data-seg="bets-source"]');
    if (seg) {
      prefs.set("betsSource", seg.dataset.value);
      renderAiBets(root, { league, source: seg.dataset.value });
      return;
    }
    const copy = event.target.closest("[data-copy-bet]");
    if (copy) {
      const [tier, index] = copy.dataset.copyBet.split(":");
      const bet = data.tiers?.[tier]?.bets?.[Number(index)];
      if (bet) copyText(betText(bet), "Bet");
    }
  };
}

function tierColumn(tier, data) {
  const bets = data?.bets || [];
  return `<div class="tier tier-${tier.key}">
    <div class="tier-head"><h3>${icon(tier.iconName, { size: 18 })} ${tier.title}</h3>
      <div class="small">${esc(data?.rule || tier.blurb)}</div>
      ${data?.detail ? `<div class="muted small">${esc(data.detail)}</div>` : ""}</div>
    ${bets.length ? bets.map((bet, i) => betCard(bet, tier.key, i)).join("") : `<div class="empty-slip">${esc(data?.empty || "Nothing fits this level right now: the AI lays off.")}</div>`}
  </div>`;
}

function betCard(bet, tier, index) {
  const legs = bet.legs.map((leg) => `
    <div class="bet-leg">
      <div class="row-between"><b>${esc(leg.label)}</b><b class="num">${esc(priceText(leg.odds))}</b></div>
      <div class="muted small">${esc(leg.game)} · AI ${pct(leg.probability)}${leg.fair != null ? ` · bookmakers ${pct(leg.fair)}` : ""}
        ${leg.check_news ? `<span class="warn"> · ${icon("alert", { size: 14 })} check the injury news</span>` : ""}</div>
    </div>`).join("");
  const kind = bet.badge === "Value" ? "good" : bet.badge === "Close to fair" ? "info" : "muted";
  return `<div class="bet">
    <div class="row-between"><b>${bet.legs.length === 1 ? "Single" : `${bet.legs.length}-leg multi`}${bet.book ? ` <span class="muted small">at ${esc(bet.book)}</span>` : ""}</b>
      ${badge(bet.badge, kind)}</div>
    ${legs}
    <div class="sum">
      <div class="sum-row"><span>Odds</span><b>${esc(priceText(bet.odds))}</b></div>
      <div class="sum-row"><span>$10 returns</span><b>${money(10 * bet.odds.decimal, { cents: true })}</b></div>
      <div class="sum-row"><span>Chance it wins</span><b>AI ${pct(bet.probability)}${bet.fair_probability != null ? ` · books ${pct(bet.fair_probability)}` : ""}</b></div>
      <div class="sum-row muted"><span>Fair odds</span><span>AI ${esc(bet.fair_odds.ai)}${bet.fair_odds.market ? ` · books ${esc(bet.fair_odds.market)}` : ""}</span></div>
      <div class="sum-row"><span class="muted">Expected result per $10</span><b class="${bet.ev > 0 ? "good" : "bad"}">${bet.ev >= 0 ? "+" : "−"}${money(Math.abs(bet.ev) * 10, { cents: true })}</b></div>
      ${bet.book_edge != null ? `<div class="sum-row muted"><span>Bookmaker's edge on this bet</span><span>${(bet.book_edge * 100).toFixed(1)}%</span></div>` : ""}
    </div>
    ${bet.why ? `<div class="muted small">${esc(bet.why)}</div>` : ""}
    <button type="button" class="btn btn-secondary small block" data-copy-bet="${tier}:${index}">${icon("copy", { size: 16 })}Copy</button>
  </div>`;
}

export function betText(bet) {
  const lines = bet.legs.map((leg) => `• ${leg.label} (${priceText(leg.odds)}) — ${leg.game}`);
  lines.push(`${bet.legs.length === 1 ? "Single" : `${bet.legs.length}-leg multi`} at ${priceText(bet.odds)}${bet.book ? ` (${bet.book})` : ""}: AI chance ${pct(bet.probability)}`);
  return `SportsBet AI\n${lines.join("\n")}`;
}
