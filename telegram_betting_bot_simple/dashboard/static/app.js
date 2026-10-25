"use strict";

// --- small helpers -----------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (p) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const money = (n) => `$${(n ?? 0).toFixed(2)}`;
const signed = (n, digits = 1) => `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
// Team nickname for tight spaces ("Oklahoma City Thunder" -> "Thunder"); soccer clubs keep their full name
const TWO_WORD_NICKNAMES = ["Trail Blazers", "Red Sox", "White Sox", "Blue Jays", "Maple Leafs", "Golden Knights", "Red Wings", "Blue Jackets"];
const shortName = (name, league = state.league) => {
  name = String(name);
  if ((league || "").startsWith("soccer_")) return name;
  const twoWord = TWO_WORD_NICKNAMES.find((nickname) => name.endsWith(` ${nickname}`));
  if (twoWord) return twoWord;
  const words = name.split(" ");
  return words.length > 1 ? words[words.length - 1] : name;
};
const STATS = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "Threes", stl: "Steals", blk: "Blocks", tov: "Turnovers", pra: "Pts+Reb+Ast" };
// FiveThirtyEight's forecasting-game points for the chance given to the team that won
const forecastPoints = (p, weight = 1) => Math.round(10 * weight * (25 - 100 * (1 - Math.round(p * 100) / 100) ** 2)) / 10;
const winnerPoints = (p) => forecastPoints(p);
const PROP_WEIGHT = 0.4;
const CAUTION_GAP = 0.08; // the AI this far from the bookmakers gets a "check the news" warning
const agreement = (p) => (p >= 0.6 ? "AI agrees" : p >= 0.4 ? "Coin flip for the AI" : "AI disagrees");
const compareConfidence = (yours, ai) => (ai < 0.5 ? "against the AI" : Math.abs(yours - ai) <= 0.05 ? "same as the AI"
  : yours > ai ? "bolder than the AI" : "more cautious than the AI");

function toast(text) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = text;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (response.status === 401) {
    showSignIn();
    throw Object.assign(new Error("Sign in to see games and predictions"), { signIn: true });
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => String(d.msg || "").replace(/^Value error, /, "")).join("; ") : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

// --- modals (in place of the browser's own pop-ups) ----------------------------------

function removeModal() {
  document.querySelector(".modal-backdrop")?.remove();
  document.body.classList.remove("modal-open");
}

function openModal(html, { dismissible = true, wide = false } = {}) {
  removeModal();
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  if (dismissible) backdrop.dataset.dismissible = "1";
  backdrop.innerHTML = `<div class="modal${wide ? " wide" : ""}" role="dialog" aria-modal="true">${html}</div>`;
  document.body.appendChild(backdrop);
  document.body.classList.add("modal-open");
  backdrop.querySelector("input, textarea, button")?.focus();
  return backdrop.firstElementChild;
}

function closeModal() {
  removeModal();
  state.draft = null; // a call being made is dropped with its modal
}

function showSignIn() {
  if ($("#sign-in")) return; // already showing
  state.draft = null;
  openModal(`
    <form id="sign-in">
      <div class="brand"><span class="logo">◆</span>SportsBet <b>AI</b></div>
      <h2>Sign in</h2>
      <p class="muted">Enter the dashboard password from your server setup.</p>
      <input type="password" name="password" placeholder="Dashboard password" autocomplete="current-password" required>
      <div class="form-error" role="alert"></div>
      <button class="primary wide" type="submit">Sign in</button>
    </form>`, { dismissible: false });
}

async function signIn(form) {
  const button = form.querySelector("button");
  const error = form.querySelector(".form-error");
  button.disabled = true;
  error.textContent = "";
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: form.elements.password.value }),
    });
    if (response.status === 401) throw new Error("Wrong password, try again.");
    if (!response.ok) throw new Error(`Couldn't sign in (${response.status}). Try again.`);
    closeModal();
    start();
  } catch (err) {
    error.textContent = err instanceof TypeError ? "Couldn't reach the dashboard. Check your connection." : err.message;
    form.elements.password.select();
  } finally {
    button.disabled = false;
  }
}

// Copy text: the clipboard API only works on https pages, so fall back to the older copy
// command, and if that fails too show the text in a modal to copy by hand
async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${what} copied`);
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
      toast(`${what} copied`);
      return;
    }
  }
  openModal(`
    <h2>${esc(what)}</h2>
    <p class="muted">Select the text and copy it (Ctrl+C, or ⌘C on a Mac).</p>
    <textarea readonly rows="8">${esc(text)}</textarea>
    <div class="modal-actions"><button class="primary" data-close>Done</button></div>`).querySelector("textarea").select();
}

// --- state (settings and who you are survive a reload in this browser) ----------------

const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem("sportsbet-ai") || "{}");
  } catch {
    return {};
  }
})();

function newPlayerId() {
  const bytes = crypto.getRandomValues(new Uint8Array(12)); // crypto.randomUUID needs https
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

const VIEWS = ["games", "bets", "calls", "leaders", "record"];
const state = {
  leagues: [],
  league: saved.league || null,
  view: VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "games",
  format: saved.format === "decimal" ? "decimal" : "american",
  me: { id: saved.me?.id || newPlayerId(), name: saved.me?.name || "" },
  data: null, // /api/games response for the league on screen
  details: {}, // /api/game responses by game id
  open: new Set(), // game ids with details expanded
  tabs: {}, // selected details tab by game id
  calls: [], // my calls (/api/calls)
  betsSource: saved.betsSource === "mine" ? "mine" : "all",
  draft: null, // the call being made in the modal
};

function persist() {
  try {
    localStorage.setItem("sportsbet-ai", JSON.stringify({
      league: state.league, format: state.format, me: state.me, betsSource: state.betsSource,
    }));
  } catch {
    /* private browsing: settings just won't persist */
  }
}
persist(); // keep the new player id

// --- formatting ------------------------------------------------------------------

function priceText(odds) {
  if (!odds) return "–";
  if (state.format === "decimal") return Number(odds.decimal ?? odds.price).toFixed(2);
  return odds.american ?? "–";
}

function pointText(market, side, point) {
  if (point == null) return "";
  if (market === "spreads") return `${point > 0 ? "+" : ""}${point}`;
  if (market === "totals") return `${side === "over" ? "O" : "U"} ${point}`;
  return "";
}

function marginText(margin, game) {
  if (margin == null) return "–";
  const team = margin >= 0 ? game.home : game.away;
  return `${shortName(team)} by ${Math.abs(margin).toFixed(1)}`;
}

function when(iso) {
  const d = new Date(iso);
  return d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

// --- views and sidebar --------------------------------------------------------------

function setView(view) {
  state.view = view;
  if (location.hash.slice(1) !== view) history.replaceState(null, "", `#${view}`);
  for (const button of $("#views").querySelectorAll("button")) button.classList.toggle("on", button.dataset.view === view);
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== view;
  // Leagues matter for games and bets only
  document.body.classList.toggle("no-sidebar", !["games", "bets"].includes(view));
  render();
}

function render() {
  if (state.view === "games") loadGames();
  else if (state.view === "bets") loadBets();
  else if (state.view === "calls") loadCalls();
  else if (state.view === "leaders") loadLeaders();
  else if (state.view === "record") loadRecord();
}

async function loadLeagues() {
  try {
    state.leagues = await api("/api/leagues");
  } catch (error) {
    if (!error.signIn) $("#view-games").innerHTML = `<div class="error-box">${esc(error.message)}</div>`;
    return false;
  }
  const all = state.leagues.flatMap((group) => group.leagues);
  if (!all.some((league) => league.key === state.league)) {
    state.league = (all.find((league) => league.key === "basketball_nba") || all[0] || {}).key || null;
  }
  renderSidebar();
  return true;
}

function renderSidebar() {
  $("#sidebar").innerHTML = state.leagues.map((group) => `
    <div class="sport">
      <div class="sport-name">${esc(group.label)}</div>
      ${group.leagues.map((league) => `
        <button class="league-btn${league.key === state.league ? " on" : ""}" data-league="${esc(league.key)}">
          <span>${esc(league.title)}</span>${league.ai ? '<span class="badge-ai">AI</span>' : ""}
        </button>`).join("")}
    </div>`).join("");
}

function leagueTitle(key = state.league) {
  return state.leagues.flatMap((g) => g.leagues).find((l) => l.key === key)?.title || key || "";
}

function renderQuota(quota) {
  const el = $("#quota");
  if (!quota || quota.remaining == null) return;
  el.textContent = `Odds API: ${quota.remaining} left`;
  el.classList.toggle("low", quota.remaining < 50);
}

// --- games: the AI's predictions first --------------------------------------------------

async function loadGames(refresh = false) {
  const view = $("#view-games");
  const league = state.league;
  if (!league) return;
  if (!refresh || !state.data) {
    view.innerHTML = `${gamesHeader()}<div class="loading">Loading games and the AI's predictions…</div>`;
  }
  try {
    const data = await api(`/api/games?league=${encodeURIComponent(league)}${refresh ? "&refresh=1" : ""}`);
    if (league !== state.league || state.view !== "games") return;
    const before = state.data?.league === league ? state.data.odds_as_of : null;
    state.data = data;
    renderGames();
    renderQuota(data.quota);
    if (refresh) toast(data.odds_as_of === before ? "Odds are fresh (less than 5 minutes old)" : `Odds updated ${data.odds_as_of}`);
  } catch (error) {
    if (!error.signIn) view.innerHTML = `${gamesHeader()}<div class="error-box">${esc(error.message)}</div>`;
  }
}

function gamesHeader() {
  const data = state.data?.league === state.league ? state.data : null;
  const sub = data ? [
    data.ai ? `AI model ${data.model}: predictions, your calls and player projections` : "Market view: the bookmakers' chances (AI model for this league is coming)",
    data.odds_as_of ? `odds as of ${data.odds_as_of}` : null,
    `${data.games.length} games`,
  ].filter(Boolean).join(" · ") : "";
  return `
    <div class="league-head">
      <div><h1>${esc(leagueTitle())}</h1><div class="sub">${esc(sub)}</div></div>
      <button class="ghost" id="refresh" title="Fetch new odds (at most every 5 minutes)">↻ Refresh odds</button>
    </div>`;
}

function renderGames() {
  const data = state.data;
  const view = $("#view-games");
  if (!data.games.length) {
    view.innerHTML = `${gamesHeader()}<div class="loading">No upcoming games with odds right now.</div>`;
    return;
  }
  view.innerHTML = `${gamesHeader()}<div class="games">${data.games.map(renderCard).join("")}</div>`;
}

function myCall(gameId) {
  return state.calls.find((call) => call.event_id === gameId);
}

// The chance each team wins: the AI's for AI leagues, the bookmakers' margin-free one otherwise
function winChances(game) {
  if (game.prediction) return { away: 1 - game.prediction.home_win, home: game.prediction.home_win, source: "AI" };
  const h2h = game.markets.h2h || {};
  if (h2h.home?.fair != null && h2h.away?.fair != null) {
    return { away: h2h.away.fair, home: h2h.home.fair, draw: h2h.draw?.fair, source: "Market" };
  }
  return null;
}

function leanChip(game, market) {
  const sides = game.markets[market];
  if (!sides) return "";
  const entries = Object.entries(sides).filter(([side]) => side !== "draw" && sides[side].ai != null);
  if (!entries.length) return "";
  const [side, offer] = entries.reduce((best, entry) => (entry[1].ai > best[1].ai ? entry : best));
  const label = market === "h2h" ? `${shortName(side === "home" ? game.home : game.away)} win`
    : market === "spreads" ? `${shortName(side === "home" ? game.home : game.away)} ${pointText(market, side, offer.point)}`
    : `${side === "over" ? "Over" : "Under"} ${offer.point}`;
  const edge = offer.fair != null ? offer.ai - offer.fair : 0;
  const caution = !game.live && state.data.ai && edge >= CAUTION_GAP;
  const value = !game.live && !caution && offer.ev != null && offer.ev > 0.03 && edge >= 0.03;
  const name = { h2h: "Winner", spreads: "Spread", totals: "Total" }[market];
  const note = caution ? " · ⚠ the AI is far from the bookmakers here: check the injury news before trusting it"
    : value ? " · the AI rates this above the price (an opinion, not a guarantee)" : "";
  return `<span class="lean${value ? " value" : ""}${caution ? " caution" : ""}" title="Best price ${esc(priceText({ american: offer.american, price: offer.price }))} at ${esc(offer.book)} · bookmakers ${pct(offer.fair)}${note}">
    <span class="lean-k">${name}</span> ${esc(label)} <b>${pct(offer.ai)}</b></span>`;
}

function renderCard(game) {
  const p = game.prediction;
  const chances = winChances(game);
  const open = state.open.has(game.id);
  const mine = myCall(game.id);
  const team = (side) => {
    const chance = chances ? chances[side] : null;
    const favourite = chances && chance >= Math.max(chances.home, chances.away);
    return `
      <div class="team-row">
        <span class="tname"><span class="full">${esc(game[side])}</span><span class="short">${esc(shortName(game[side]))}</span>${side === "home" ? '<span class="muted small"> (home)</span>' : ""}</span>
        <span class="tscore">${p ? Math.round(side === "home" ? p.home_score : p.away_score) : ""}</span>
        <span class="bar"><span class="fill${favourite ? " fav" : ""}" style="width:${chance != null ? Math.round(chance * 100) : 0}%"></span></span>
        <span class="tpct${favourite ? " fav" : ""}">${pct(chance)}</span>
      </div>`;
  };
  const marketHome = p?.sources?.market?.home_win_prob;
  const caution = p && marketHome != null && Math.abs(p.home_win - marketHome) >= CAUTION_GAP;
  const chip = p
    ? `${caution ? '<span class="chip caution" title="The AI is far from the bookmakers on this game: check the injury news">⚠ far from market</span> ' : ""}<span class="chip ai">${game.live ? "Pre-game AI" : "AI pick"}: ${esc(shortName(p.pick))} ${pct(p.pick_prob)}</span>`
    : chances ? `<span class="chip">Market favourite: ${esc(shortName(chances.home >= chances.away ? game.home : game.away))}</span>` : "";
  const draw = chances?.draw != null ? `<div class="muted small draw">Draw ${pct(chances.draw)}</div>` : "";
  const [winSide, loseSide] = p && p.margin < 0 ? ["away", "home"] : ["home", "away"];
  const summary = p
    ? `AI: ${esc(shortName(game[winSide]))} ${Math.round(p[`${winSide}_score`])}–${Math.round(p[`${loseSide}_score`])} ${esc(shortName(game[loseSide]))} · ${esc(marginText(p.margin, game))} · total ${p.total.toFixed(0)}`
    : `${chances ? "Chances from the bookmakers' odds, margin removed" : "No odds yet"}`;
  const callButton = p && !game.live
    ? `<button class="primary small call-btn" data-call="${esc(game.id)}">${mine ? "✎ Edit your call" : "🔮 Make your call"}</button>`
    : "";
  const yours = mine ? `<span class="mycall">Your call: ${esc(callSummary(mine.call, game))}${mine.status === "graded" ? ` · ${signed(mine.user_points)} pts` : ""}</span>` : "";
  return `
    <article class="card" data-id="${esc(game.id)}">
      <div class="card-top">
        <span class="${game.live ? "live" : "when"}">${game.live ? "● LIVE" : esc(game.when)}</span>${chip}
      </div>
      <div class="matchup">${team("away")}${team("home")}${draw}</div>
      <div class="sumline">${summary}</div>
      <div class="leans">${leanChip(game, "h2h")}${leanChip(game, "spreads")}${leanChip(game, "totals")}</div>
      <div class="card-foot">
        ${yours || '<span></span>'}
        <span class="foot-actions">${callButton}
          <button class="link details-btn" data-id="${esc(game.id)}">${open ? "Hide ▴" : p ? "Prediction &amp; players ▾" : "All odds ▾"}</button>
        </span>
      </div>
      ${open ? `<div class="details">${renderDetails(game)}</div>` : ""}
    </article>`;
}

function callSummary(call, game) {
  const parts = [];
  if (call.winner) parts.push(`${shortName(call.winner === "home" ? game.home : game.away)} ${pct(call.confidence)}`);
  if (call.home_score != null) parts.push(`${call.home_score}–${call.away_score}`);
  if (call.props?.length) parts.push(`${call.props.length} player call${call.props.length > 1 ? "s" : ""}`);
  return parts.join(" · ");
}

// --- game details ----------------------------------------------------------------------

async function loadDetail(id) {
  if (!state.details[id]) {
    try {
      state.details[id] = await api(`/api/game?league=${encodeURIComponent(state.league)}&id=${encodeURIComponent(id)}`);
    } catch (error) {
      if (error.signIn) throw error;
      state.details[id] = { error: error.message };
    }
  }
  return state.details[id];
}

async function toggleDetails(id) {
  if (state.open.has(id)) {
    state.open.delete(id);
    renderGames();
    return;
  }
  state.open.add(id);
  renderGames();
  try {
    await loadDetail(id);
  } catch {
    state.open.delete(id);
  }
  if (state.view === "games") renderGames();
}

function renderDetails(game) {
  const detail = state.details[game.id];
  if (!detail) return '<div class="loading">Loading prediction, players and every bookmaker…</div>';
  if (detail.error) return `<div class="error-box">${esc(detail.error)}</div>`;
  const tabs = detail.prediction ? ["prediction", "players", "odds"] : ["odds"];
  const tab = tabs.includes(state.tabs[game.id]) ? state.tabs[game.id] : tabs[0];
  const names = { prediction: "Prediction", players: "Players", odds: "All odds" };
  const body = tab === "prediction" ? predictionTab(detail) : tab === "players" ? playersTab(detail) : oddsTab(detail);
  return `
    <div class="tabs">${tabs.map((t) => `<button class="${t === tab ? "on" : ""}" data-tab="${t}" data-id="${esc(game.id)}">${names[t]}</button>`).join("")}</div>
    ${body}`;
}

function predictionTab(game) {
  const p = game.prediction;
  const s = p.sources;
  const row = (name, source) => source ? `
    <tr><td>${name}</td><td>${pct(source.home_win_prob)}</td><td>${marginText(source.margin, game)}</td>
    <td>${source.total != null ? source.total.toFixed(1) : "–"}</td></tr>` : "";
  const injuries = ["away", "home"].map((side) => {
    const list = game.injuries?.[side] || [];
    return `<div><b>${esc(game[side])}:</b> ${list.length ? list.map(esc).join(", ") : "no injuries reported"}</div>`;
  }).join("");
  return `
    <div class="table-wrap"><table>
      <thead><tr><th>Source</th><th>${esc(shortName(game.home))} win</th><th>Margin</th><th>Total</th></tr></thead>
      <tbody>
        ${row("Betting market (no margin)", s.market)}${row("ESPN predictor", s.espn)}${row("Our team model", s.model)}
        <tr class="final"><td>AI prediction</td><td>${pct(p.home_win)}</td><td>${marginText(p.margin, game)}</td><td>${p.total.toFixed(1)}</td></tr>
      </tbody>
    </table></div>
    ${p.factors.length ? `<h4>What moved our model</h4><ul class="factors">${p.factors.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
    <h4>Injury report</h4>${injuries}
    <p class="note">${esc(p.method || "")}</p>`;
}

function playersTab(game) {
  if (!game.players) return '<div class="loading">Player projections are not available for this game yet.</div>';
  const table = (side) => {
    const players = game.players[side] || [];
    return `
      <h4>${esc(game[side])} (projected ${Math.round(side === "home" ? game.prediction.home_score : game.prediction.away_score)})</h4>
      <div class="table-wrap"><table>
        <thead><tr><th>Player</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th><th>3PM</th><th>Stl</th><th>Blk</th><th>TO</th></tr></thead>
        <tbody>${players.map((pl) => pl.out ? `
          <tr class="out"><td>${esc(pl.name)} <span class="muted">${esc(pl.position)}</span></td><td colspan="7"></td><td><span class="tag">${esc(pl.status)}</span></td></tr>` : `
          <tr><td>${esc(pl.name)} <span class="muted">${esc(pl.position)}</span>${pl.status ? ` <span class="tag">${esc(pl.status)}</span>` : ""}</td>
            <td>${pl.minutes.toFixed(0)}</td><td><b>${pl.pts.toFixed(1)}</b></td><td>${pl.reb.toFixed(1)}</td><td>${pl.ast.toFixed(1)}</td>
            <td>${pl.fg3m.toFixed(1)}</td><td>${pl.stl.toFixed(1)}</td><td>${pl.blk.toFixed(1)}</td><td>${pl.tov.toFixed(1)}</td></tr>`).join("")}
        </tbody>
      </table></div>`;
  };
  return `${table("away")}${table("home")}
    <p class="note">Projections: minutes and per-minute production from this season blended with last season,
      injured players' minutes shared across the rotation, points scaled to the team's predicted score and other
      stats to the predicted pace. Passes made need NBA tracking data (planned).</p>`;
}

function oddsTab(game) {
  const books = game.books || [];
  if (!books.length) return '<div class="loading">No bookmaker odds for this game.</div>';
  const price = (offer, market, side) => offer
    ? `${market !== "h2h" ? `${esc(pointText(market, side, offer.point))} ` : ""}${esc(priceText({ american: offer.american, price: offer.price }))}`
    : "–";
  return `
    <div class="table-wrap"><table>
      <thead><tr><th>Bookmaker</th><th>${esc(shortName(game.away))} win</th><th>${esc(shortName(game.home))} win</th>
        <th>${esc(shortName(game.away))} spread</th><th>${esc(shortName(game.home))} spread</th><th>Over</th><th>Under</th></tr></thead>
      <tbody>${books.map((b) => `
        <tr><td>${esc(b.title)}</td>
          <td>${price(b.markets.h2h?.away, "h2h", "away")}</td><td>${price(b.markets.h2h?.home, "h2h", "home")}</td>
          <td>${price(b.markets.spreads?.away, "spreads", "away")}</td><td>${price(b.markets.spreads?.home, "spreads", "home")}</td>
          <td>${price(b.markets.totals?.over, "totals", "over")}</td><td>${price(b.markets.totals?.under, "totals", "under")}</td></tr>`).join("")}
      </tbody>
    </table></div>`;
}

// --- making a call -----------------------------------------------------------------------

async function openCall(gameId) {
  const game = state.data.games.find((g) => g.id === gameId);
  if (!game?.prediction) return;
  const existing = myCall(gameId);
  const p = game.prediction;
  state.draft = {
    game,
    players: [],
    verdict: null,
    call: existing && existing.status === "pending"
      ? JSON.parse(JSON.stringify(existing.call))
      : { winner: p.home_win >= 0.5 ? "home" : "away", confidence: 0.6, home_score: null, away_score: null, props: [] },
  };
  openModal(callForm(), { wide: true });
  try {
    const detail = await loadDetail(gameId);
    if (!state.draft || state.draft.game.id !== gameId) return;
    state.draft.players = ["away", "home"].flatMap((side) =>
      (detail.players?.[side] || []).filter((pl) => !pl.out).map((pl) => ({ ...pl, team: shortName(game[side]) })));
    fillProps();
    renderProps();
  } catch {
    /* sign-in modal took over */
  }
  judgeDraft();
}

function callForm() {
  const { game, call } = state.draft;
  const p = game.prediction;
  const side = (s) => `<button type="button" class="${call.winner === s ? "on" : ""}" data-winner="${s}">${esc(shortName(game[s]))}</button>`;
  return `
    <form id="call-form" autocomplete="off">
      <h2>Your call: ${esc(shortName(game.away))} @ ${esc(shortName(game.home))}</h2>
      <p class="muted small">${esc(game.when)} · the AI says ${esc(shortName(game.away))} ${Math.round(p.away_score)}, ${esc(shortName(game.home))} ${Math.round(p.home_score)}</p>

      <div class="field">
        <label>Who wins?</label>
        <div class="seg big" id="winner-seg">${side("away")}${side("home")}</div>
      </div>
      <div class="field">
        <label for="confidence">How sure are you? <b id="conf-label">${pct(call.confidence)}</b></label>
        <input type="range" id="confidence" min="50" max="100" step="1" value="${Math.round(call.confidence * 100)}">
        <div class="hint" id="winner-hint"></div>
      </div>

      <div class="field">
        <label>Final score <span class="muted">(optional)</span></label>
        <div class="score-row">
          <span>${esc(shortName(game.away))}</span><input type="number" id="away-score" min="40" max="250" placeholder="${Math.round(p.away_score)}" value="${call.away_score ?? ""}">
          <span>${esc(shortName(game.home))}</span><input type="number" id="home-score" min="40" max="250" placeholder="${Math.round(p.home_score)}" value="${call.home_score ?? ""}">
        </div>
        <div class="hint" id="score-hint"></div>
        <div class="hint">Scored as 15 minus your margin miss, plus 20 minus your total miss, plus 10 if both scores are within 2.</div>
      </div>

      <div class="field">
        <label>Player calls <span class="muted">(optional, up to 8; each counts 40% of a winner call)</span></label>
        <div id="props"></div>
        <button type="button" class="link" id="add-prop">+ Add a player call</button>
      </div>

      <div class="field">
        <label for="my-name">Your name <span class="muted">(for the leaderboard)</span></label>
        <input type="text" id="my-name" maxlength="24" value="${esc(state.me.name)}" placeholder="e.g. Sam">
      </div>
      <div class="verdict" id="verdict"></div>
      <div class="form-error" role="alert" id="call-error"></div>
      <div class="modal-actions"><button type="button" class="ghost" data-close>Cancel</button><button type="submit" class="primary">Save call</button></div>
    </form>`;
}

function renderProps() {
  const draft = state.draft;
  if (!draft) return;
  const options = (selected) => draft.players.map((pl) =>
    `<option value="${esc(pl.id)}"${String(pl.id) === String(selected) ? " selected" : ""}>${esc(pl.name)} (${esc(pl.team)})</option>`).join("");
  const props = draft.call.props;
  const judged = draft.verdict?.props || [];
  $("#props").innerHTML = props.map((prop, i) => {
    const ai = judged.find((j) => String(j.player_id) === String(prop.player_id) && j.stat === prop.stat && Number(j.line) === Number(prop.line) && j.side === prop.side);
    return `
      <div class="prop-row" data-prop="${i}">
        <select data-k="player_id">${draft.players.length ? options(prop.player_id) : '<option value="">Loading players…</option>'}</select>
        <select data-k="stat">${Object.entries(STATS).map(([k, v]) => `<option value="${k}"${k === prop.stat ? " selected" : ""}>${v}</option>`).join("")}</select>
        <input type="number" data-k="line" min="0" max="150" step="0.5" value="${prop.line}">
        <div class="seg"><button type="button" data-side="over" class="${prop.side === "over" ? "on" : ""}">Over</button><button type="button" data-side="under" class="${prop.side === "under" ? "on" : ""}">Under</button></div>
        <select data-k="confidence" title="How sure are you?">${[55, 60, 65, 70, 75, 80, 90, 100].map((c) =>
          `<option value="${c / 100}"${Math.round((prop.confidence ?? 0.6) * 100) === c ? " selected" : ""}>${c}%</option>`).join("")}</select>
        <span class="prop-ai">${ai ? `<span class="${ai.ai_prob >= 0.6 ? "good" : ai.ai_prob < 0.4 ? "bad" : ""}">AI <b>${pct(ai.ai_prob)}</b></span> <span class="muted">avg ${ai.ai_mean.toFixed(1)}</span>` : ""}
          <span class="muted small" title="Your points if this call is right / wrong">${signed(forecastPoints(prop.confidence ?? 0.6, PROP_WEIGHT))} / ${signed(forecastPoints(1 - (prop.confidence ?? 0.6), PROP_WEIGHT))} pts</span></span>
        <button type="button" class="x" data-remove-prop="${i}" title="Remove">×</button>
      </div>`;
  }).join("");
  $("#add-prop").hidden = props.length >= 8;
}

function addProp() {
  state.draft.call.props.push({ player_id: "", stat: "pts", line: 0.5, side: "over", confidence: 0.6 });
  fillProps();
  renderProps();
  judgeDraft();
}

// New rows get the next player down the list with no call yet (players come busiest first);
// rows added while the players are still loading are filled in when they arrive
function fillProps() {
  const draft = state.draft;
  if (!draft.players.length) return;
  const used = new Set(draft.call.props.map((prop) => String(prop.player_id)));
  for (const prop of draft.call.props.filter((x) => !x.player_id)) {
    const next = draft.players.find((pl) => !used.has(String(pl.id))) || draft.players[0];
    prop.player_id = String(next.id);
    prop.line = suggestLine(prop);
    used.add(prop.player_id);
  }
}

// The line defaults to just under the player's projection, so changing player or stat resets it
function suggestLine(prop) {
  const pl = state.draft.players.find((x) => String(x.id) === String(prop.player_id));
  if (!pl) return prop.line;
  const mean = prop.stat === "pra" ? pl.pts + pl.reb + pl.ast : pl[prop.stat] ?? 0;
  return Math.max(0.5, Math.round(mean) - 0.5);
}

function draftPayload() {
  const call = state.draft.call;
  const out = { winner: call.winner, confidence: call.confidence, props: call.props.filter((x) => x.player_id) };
  if (call.home_score != null && call.away_score != null) {
    out.home_score = call.home_score;
    out.away_score = call.away_score;
  }
  return out;
}

let judgeTimer = null;
function judgeDraft() {
  clearTimeout(judgeTimer);
  updateHints();
  judgeTimer = setTimeout(async () => {
    const draft = state.draft;
    if (!draft) return;
    try {
      const verdict = await api("/api/judge", {
        method: "POST",
        body: JSON.stringify({ league: draft.game.league, id: draft.game.id, call: draftPayload() }),
      });
      if (state.draft !== draft) return;
      draft.verdict = verdict;
      $("#call-error").textContent = "";
    } catch (error) {
      if (state.draft === draft && !error.signIn) $("#call-error").textContent = error.message;
      return;
    }
    renderProps();
    updateHints();
  }, 250);
}

function updateHints() {
  const draft = state.draft;
  if (!draft || !$("#call-form")) return;
  const { call, game, verdict } = draft;
  const p = game.prediction;
  const aiWinner = call.winner === "home" ? p.home_win : 1 - p.home_win;
  const team = shortName(call.winner === "home" ? game.home : game.away);
  $("#conf-label").textContent = pct(call.confidence);
  const marketHome = p.sources?.market?.home_win_prob;
  const market = marketHome != null ? (call.winner === "home" ? marketHome : 1 - marketHome) : null;
  const caution = marketHome != null && Math.abs(p.home_win - marketHome) >= CAUTION_GAP;
  $("#winner-hint").innerHTML = `<b class="${aiWinner >= 0.6 ? "good" : aiWinner < 0.4 ? "bad" : ""}">${agreement(aiWinner)}</b>:
    it gives ${esc(team)} <b>${pct(aiWinner)}</b>${market != null ? ` (bookmakers ${pct(market)})` : ""}; you're ${compareConfidence(call.confidence, aiWinner)}.
    Points: <span class="good">${signed(winnerPoints(call.confidence))}</span> if ${esc(team)} win,
    <span class="bad">${signed(winnerPoints(1 - call.confidence))}</span> if they lose.
    ${caution ? '<br><span class="warn">⚠ The AI is far from the bookmakers on this game: check the injury news.</span>' : ""}`;
  const score = verdict?.score;
  $("#score-hint").innerHTML = score && call.home_score != null
    ? `${esc(shortName(score.winner_side === "home" ? game.home : game.away))} by ${Math.abs(score.margin)}+: AI <b>${pct(score.wins_by_at_least)}</b> ·
       total over ${score.total}: AI <b>${pct(score.total_over_yours)}</b>`
    : "Leave blank to call just the winner.";
  $("#verdict").textContent = verdict?.verdict || "";
}

async function saveCall(form) {
  const draft = state.draft;
  const name = $("#my-name").value.trim();
  if (!name) {
    $("#call-error").textContent = "Add your name for the leaderboard.";
    $("#my-name").focus();
    return;
  }
  state.me.name = name;
  persist();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    await api("/api/calls", {
      method: "POST",
      body: JSON.stringify({ league: draft.game.league, id: draft.game.id, player: state.me.id, name, call: draftPayload() }),
    });
    closeModal();
    toast("Call saved: it's scored after the game");
    await refreshCalls();
    if (state.view === "games") renderGames();
    else render();
  } catch (error) {
    if (!error.signIn) $("#call-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function refreshCalls() {
  try {
    state.calls = await api(`/api/calls?player=${encodeURIComponent(state.me.id)}`);
  } catch (error) {
    if (error.signIn) throw error;
  }
}

// --- AI bets: safe, risky, crazy ----------------------------------------------------------

const TIERS = [
  { key: "safe", title: "Safe", icon: "🛡️", blurb: "More likely, small payout." },
  { key: "risky", title: "Risky", icon: "⚖️", blurb: "Balanced: a real chance and a real payout." },
  { key: "crazy", title: "Crazy", icon: "🚀", blurb: "Long shot, big payout." },
];

async function loadBets() {
  const view = $("#view-bets");
  const league = state.league;
  if (!league) return;
  view.innerHTML = `${betsHeader()}<div class="loading">The AI is building bets from every game…</div>`;
  try {
    const query = `league=${encodeURIComponent(league)}&source=${state.betsSource}&player=${encodeURIComponent(state.me.id)}`;
    const data = await api(`/api/ai-bets?${query}`);
    if (league !== state.league || state.view !== "bets") return;
    view.innerHTML = `${betsHeader(data)}<div class="tiers">${TIERS.map((tier) => tierColumn(tier, data.tiers?.[tier.key])).join("")}</div>`;
    state.bets = data;
  } catch (error) {
    if (!error.signIn) view.innerHTML = `${betsHeader()}<div class="error-box">${esc(error.message)}</div>`;
  }
}

function betsHeader(data) {
  return `
    <div class="league-head">
      <div><h1>AI Bets · ${esc(leagueTitle())}</h1>
        <div class="sub">${esc(data?.note || "Built from the AI's probabilities and the best current prices.")}</div></div>
      <div class="seg" id="bets-source">
        <button data-source="all" class="${state.betsSource === "all" ? "on" : ""}">All games</button>
        <button data-source="mine" class="${state.betsSource === "mine" ? "on" : ""}">From my calls</button>
      </div>
    </div>`;
}

function tierColumn(tier, data) {
  const bets = data?.bets || [];
  return `
    <div class="tier tier-${tier.key}">
      <div class="tier-head"><h2>${tier.icon} ${tier.title}</h2>
        <div class="small">${esc(data?.rule || tier.blurb)}</div>
        ${data?.detail ? `<div class="muted small">${esc(data.detail)}</div>` : ""}</div>
      ${bets.length ? bets.map((bet, i) => betCard(bet, tier.key, i)).join("") : `<div class="empty-slip">${esc(data?.empty || "Nothing fits this level right now: the AI lays off.")}</div>`}
    </div>`;
}

function betCard(bet, tier, index) {
  const legs = bet.legs.map((leg) => `
    <div class="bet-leg">
      <div class="leg-top"><span class="leg-label">${esc(leg.label)}</span><span class="leg-price">${esc(priceText(leg.odds))}</span></div>
      <div class="leg-game">${esc(leg.game)} · AI ${pct(leg.probability)}${leg.fair != null ? ` · bookmakers ${pct(leg.fair)}` : ""}
        ${leg.check_news ? '<span class="warn"> · ⚠ check the injury news</span>' : ""}</div>
    </div>`).join("");
  const badgeClass = bet.badge === "Value" ? "win" : bet.badge === "Close to fair" ? "pending" : "";
  return `
    <div class="bet">
      <div class="bet-title"><span>${bet.legs.length === 1 ? "Single" : `${bet.legs.length}-leg multi`}${bet.book ? ` <span class="muted small">at ${esc(bet.book)}</span>` : ""}</span>
        <span class="badge ${badgeClass}">${esc(bet.badge)}</span></div>
      ${legs}
      <div class="bet-sum">
        <div class="sum-row"><span>Odds</span><b>${esc(priceText(bet.odds))}</b></div>
        <div class="sum-row"><span>$10 returns</span><b>${money(10 * bet.odds.decimal)}</b></div>
        <div class="sum-row"><span>Chance it wins</span><b>AI ${pct(bet.probability)}${bet.fair_probability != null ? ` · bookmakers ${pct(bet.fair_probability)}` : ""}</b></div>
        <div class="sum-row muted"><span>Fair odds</span><span>AI ${esc(bet.fair_odds.ai)}${bet.fair_odds.market ? ` · bookmakers ${esc(bet.fair_odds.market)}` : ""}</span></div>
        <div class="sum-row"><span class="muted">Expected result per $10</span><span class="${bet.ev > 0 ? "good" : "bad"}">${bet.ev >= 0 ? "+" : "−"}${money(Math.abs(bet.ev) * 10)}</span></div>
        ${bet.book_edge != null ? `<div class="sum-row muted"><span>Bookmaker's edge on this bet</span><span>${(bet.book_edge * 100).toFixed(1)}%</span></div>` : ""}
      </div>
      ${bet.why ? `<div class="why">${esc(bet.why)}</div>` : ""}
      <button class="ghost small wide" data-copy-bet="${tier}:${index}">Copy</button>
    </div>`;
}

function betText(bet) {
  const lines = bet.legs.map((leg) => `• ${leg.label} (${priceText(leg.odds)}) — ${leg.game}`);
  lines.push(`${bet.legs.length === 1 ? "Single" : `${bet.legs.length}-leg multi`} at ${priceText(bet.odds)}${bet.book ? ` (${bet.book})` : ""}: AI chance ${pct(bet.probability)}`);
  return `SportsBet AI\n${lines.join("\n")}`;
}

// --- my calls, leaderboard, AI record ---------------------------------------------------------

async function loadCalls() {
  const view = $("#view-calls");
  view.innerHTML = '<div class="loading">Loading your calls…</div>';
  try {
    await refreshCalls();
  } catch {
    return;
  }
  if (state.view !== "calls") return;
  const graded = state.calls.filter((c) => c.status === "graded");
  const points = graded.reduce((sum, c) => sum + c.user_points, 0);
  const aiPoints = graded.reduce((sum, c) => sum + c.ai_points, 0);
  const vsAi = points - aiPoints;
  const winnerParts = graded.map((c) => c.result.parts.find((part) => part.part === "winner" && part.market_points != null)).filter(Boolean);
  const vsMarket = winnerParts.reduce((sum, part) => sum + part.points - part.market_points, 0);
  const headline = !graded.length ? "Your calls are scored after each game, next to the AI's score on the same games."
    : vsAi >= 0 ? `You beat the AI by ${vsAi.toFixed(1)} points on the ${graded.length} game${graded.length > 1 ? "s" : ""} you called.`
    : `The AI is ahead of you by ${(-vsAi).toFixed(1)} points on the ${graded.length} game${graded.length > 1 ? "s" : ""} you called.`;
  view.innerHTML = `
    <div class="league-head"><div><h1>My calls</h1>
      <div class="sub">${state.me.name ? `As ${esc(state.me.name)} · ` : ""}${esc(headline)}</div></div></div>
    <div class="stats">
      <div class="stat"><div class="stat-v">${graded.length}/${state.calls.length}</div><div class="stat-k">calls scored</div></div>
      <div class="stat"><div class="stat-v">${signed(points)}</div><div class="stat-k">your points</div></div>
      <div class="stat"><div class="stat-v ${vsAi >= 0 ? "good" : "bad"}">${signed(vsAi)}</div><div class="stat-k">vs the AI (same games)</div></div>
      <div class="stat"><div class="stat-v ${vsMarket >= 0 ? "good" : "bad"}">${signed(vsMarket)}</div><div class="stat-k">vs the bookmakers (winner calls)</div></div>
    </div>
    <p class="note">Winner calls: 25 − 100 × (1 − p)², p = the chance you gave the team that won. 50% always scores 0; a
      calibrated 70% call averages only about +4 a game, and a wrong 90% call costs −56. Bolder calls cost more when they miss.</p>
    ${state.calls.length ? state.calls.map(callCard).join("") : `<div class="empty-slip">No calls yet. Open <b>Games</b> and press <b>🔮 Make your call</b> on an NBA game.</div>`}`;
}

function callCard(row) {
  const game = { home: row.home, away: row.away };
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
  let status = '<span class="badge pending">Pending</span>';
  let result = "";
  if (row.status === "graded") {
    const r = row.result;
    const diff = row.user_points - row.ai_points;
    status = `<span class="badge ${diff >= 0 ? "win" : "loss"}">${diff > 0 ? "Beat the AI" : diff === 0 ? "Tied the AI" : "AI won"} ${signed(diff)}</span>`;
    result = `<div class="result">Final: ${esc(shortName(row.away, row.league))} ${r.away}–${r.home} ${esc(shortName(row.home, row.league))} ·
      you <b class="${row.user_points >= 0 ? "good" : "bad"}">${signed(row.user_points)}</b> · AI <b>${signed(row.ai_points)}</b>${r.market_points != null ? ` · bookmakers ${signed(r.market_points)} <span class="muted">(winner)</span>` : ""}
      ${r.parts.map((part) => `<span class="part">${part.part === "prop" ? esc(part.player || "") : part.part}${part.void ? " void" : `: ${signed(part.points)}`}</span>`).join("")}</div>`;
  } else if (row.status === "void") {
    status = '<span class="badge">Void</span>';
  }
  const actions = row.status === "pending" && new Date(row.start) > new Date()
    ? `<button class="link" data-delete-call="${row.id}">Delete</button>` : "";
  return `
    <div class="card call-card">
      <div class="card-top"><span class="when">${esc(when(row.start))} · ${esc(row.away)} @ ${esc(row.home)}</span>${status}</div>
      ${lines.map((l) => `<div class="call-line">${l}</div>`).join("")}
      ${result}
      <div class="card-foot"><span class="muted small">${esc(ai.verdict || "")}</span>${actions}</div>
    </div>`;
}

async function loadLeaders() {
  const view = $("#view-leaders");
  view.innerHTML = '<div class="loading">Loading the leaderboard…</div>';
  let rows;
  try {
    rows = await api("/api/leaderboard");
  } catch (error) {
    if (!error.signIn) view.innerHTML = `<div class="error-box">${esc(error.message)}</div>`;
    return;
  }
  if (state.view !== "leaders") return;
  view.innerHTML = `
    <div class="league-head"><div><h1>Leaderboard</h1>
      <div class="sub">Points from scored calls. "vs AI" is your points minus the AI's on the same games: copying the AI scores 0,
        so anything above 0 means you read those games better. "vs bookmakers" compares winner calls with the betting market, the hardest opponent.</div></div></div>
    ${rows.length ? `<div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Name</th><th>Calls</th><th>Points</th><th>vs AI</th><th>vs bookmakers</th><th>Beat the AI</th></tr></thead>
      <tbody>${rows.map((r, i) => `<tr${r.player === `web:${state.me.id}` ? ' class="me"' : ""}>
        <td>${i + 1}</td><td>${esc(r.name)}</td><td>${r.calls}</td><td><b>${signed(r.points)}</b></td>
        <td class="${r.vs_ai >= 0 ? "good" : "bad"}">${signed(r.vs_ai)}</td><td>${r.market_calls ? signed(r.vs_market) : "–"}</td><td>${r.beat_ai}/${r.calls}</td></tr>`).join("")}
      </tbody></table></div>` : '<div class="empty-slip">No graded calls yet. Calls are scored after each game finishes.</div>'}`;
}

async function loadRecord() {
  const view = $("#view-record");
  view.innerHTML = '<div class="loading">Loading the AI\'s record…</div>';
  let r;
  try {
    r = await api("/api/track-record?league=basketball_nba");
  } catch (error) {
    if (!error.signIn) view.innerHTML = `<div class="error-box">${esc(error.message)}</div>`;
    return;
  }
  if (state.view !== "record") return;
  const rec = (x) => `${x.won}-${x.lost}${x.push ? `-${x.push}` : ""}`;
  const rate = (x) => (x.won + x.lost ? ` (${pct(x.won / (x.won + x.lost))})` : "");
  view.innerHTML = `
    <div class="league-head"><div><h1>AI record · NBA</h1>
      <div class="sub">Every game the AI predicted before tip-off, graded after the final buzzer. Honest numbers, including the misses.</div></div></div>
    ${!r.games ? '<div class="empty-slip">No graded games yet. The record starts with the games the AI predicts from now on: check back after tonight\'s games.</div>' : `
    <div class="stats">
      <div class="stat"><div class="stat-v">${r.games}</div><div class="stat-k">games graded</div></div>
      <div class="stat"><div class="stat-v">${pct(r.winner_accuracy)}</div><div class="stat-k">winners right</div></div>
      <div class="stat"><div class="stat-v">${r.brier.toFixed(3)}</div><div class="stat-k">AI Brier score (lower is better)</div></div>
      <div class="stat"><div class="stat-v">${r.market_brier != null ? r.market_brier.toFixed(3) : "–"}</div><div class="stat-k">bookmakers' Brier score</div></div>
      <div class="stat"><div class="stat-v">${r.baseline_brier.toFixed(3)}</div><div class="stat-k">"always the home team" Brier</div></div>
      <div class="stat"><div class="stat-v">${r.log_loss.toFixed(3)}</div><div class="stat-k">AI log loss (lower is better)</div></div>
      <div class="stat"><div class="stat-v">${rec(r.ats)}${rate(r.ats)}</div><div class="stat-k">vs the closing spread</div></div>
      <div class="stat"><div class="stat-v">${rec(r.totals)}${rate(r.totals)}</div><div class="stat-k">vs the closing total</div></div>
      <div class="stat"><div class="stat-v">${r.margin_error.toFixed(1)}</div><div class="stat-k">avg margin miss (pts)</div></div>
      <div class="stat"><div class="stat-v">${r.total_error.toFixed(1)}</div><div class="stat-k">avg total miss (pts)</div></div>
    </div>
    <h3 class="section">Calibration: when the AI says X%, does it happen X% of the time?</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>AI's favourite at</th><th>Games</th><th>AI said</th><th>Happened</th><th></th></tr></thead>
      <tbody>${r.calibration.map((b) => `<tr><td>${b.range}</td><td>${b.games}</td><td>${pct(b.predicted)}</td><td>${pct(b.actual)}</td>
        <td class="calib">${b.games ? `<span class="bar"><span class="fill fav" style="width:${Math.round(b.actual * 100)}%"></span></span>` : ""}</td></tr>`).join("")}</tbody>
    </table></div>
    ${tierRecord(r.tiers)}
    <h3 class="section">Latest games</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Game</th><th>AI</th><th>Final</th><th></th></tr></thead>
      <tbody>${r.recent.map((g) => `<tr><td>${esc(g.away)} @ ${esc(g.home)}</td>
        <td>${esc(shortName(g.home_win >= 0.5 ? g.home : g.away, "basketball_nba"))} ${pct(Math.max(g.home_win, 1 - g.home_win))}</td>
        <td>${g.final_away}–${g.final_home}</td><td>${g.hit ? '<span class="good">✓</span>' : '<span class="bad">✗</span>'}</td></tr>`).join("")}</tbody>
    </table></div>`}
    ${!r.games ? tierRecord(r.tiers) : ""}
    <p class="note">What good looks like: nobody gets near 100%. The closing betting market picks about 68% of NBA winners
      (65.7-69.8% by season) with a Brier score near 0.20; a sound model lands at 66-69% and 0.20-0.21. Brier score is the
      average squared miss of the win chance: always saying 50% scores 0.250. Against the closing spread, 50-53% is
      realistic and 52.4% is needed to break even at standard prices.</p>`;
}

// How the AI's own daily bets did, per risk level: is Safe really safe?
function tierRecord(tiers) {
  if (!tiers) return "";
  const rows = tiers.map((t) => `<tr><td>${esc(t.title)}</td><td>${t.tickets}</td><td>${t.won}</td>
    <td>${pct(t.predicted)}</td><td>${pct(t.actual)}</td>
    <td class="${t.roi == null ? "" : t.roi >= 0 ? "good" : "bad"}">${t.roi == null ? "–" : `${signed(t.roi * 100)}%`}</td></tr>`).join("");
  return `
    <h3 class="section">AI bets record: the top Safe, Risky and Crazy bet each day</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Level</th><th>Bets graded</th><th>Won</th><th>AI said</th><th>Happened</th><th>Return</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="note">Each level's top bet is saved before its first game starts and graded at the price shown. A level
      earns trust only if "Happened" keeps up with "AI said" over many bets; expect returns a little below zero,
      the bookmakers' cut.</p>`;
}

// --- events ----------------------------------------------------------------------------

document.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.view) {
    setView(target.dataset.view);
  } else if (target.dataset.league && target.classList.contains("league-btn")) {
    state.league = target.dataset.league;
    state.open.clear();
    persist();
    renderSidebar();
    if (state.view !== "games" && state.view !== "bets") setView("games");
    else render();
  } else if (target.dataset.call) {
    openCall(target.dataset.call);
  } else if (target.classList.contains("details-btn")) {
    toggleDetails(target.dataset.id);
  } else if (target.dataset.tab) {
    state.tabs[target.dataset.id] = target.dataset.tab;
    renderGames();
  } else if (target.dataset.winner && state.draft) {
    state.draft.call.winner = target.dataset.winner;
    const call = state.draft.call;
    // A winner that contradicts the score clears the score
    if (call.home_score != null && call.away_score != null && (call.home_score > call.away_score) !== (call.winner === "home")) {
      call.home_score = call.away_score = null;
      $("#home-score").value = $("#away-score").value = "";
    }
    for (const b of $("#winner-seg").querySelectorAll("button")) b.classList.toggle("on", b.dataset.winner === call.winner);
    judgeDraft();
  } else if (target.dataset.side && state.draft) {
    const row = target.closest(".prop-row");
    state.draft.call.props[Number(row.dataset.prop)].side = target.dataset.side;
    renderProps();
    judgeDraft();
  } else if (target.dataset.removeProp != null && state.draft) {
    state.draft.call.props.splice(Number(target.dataset.removeProp), 1);
    renderProps();
    judgeDraft();
  } else if (target.id === "add-prop") {
    addProp();
  } else if (target.dataset.format) {
    state.format = target.dataset.format;
    persist();
    for (const b of $("#format-seg").querySelectorAll("button")) b.classList.toggle("on", b.dataset.format === state.format);
    if (state.view === "games" && state.data) renderGames();
    else if (state.view === "bets") loadBets();
  } else if (target.dataset.source) {
    state.betsSource = target.dataset.source;
    persist();
    loadBets();
  } else if (target.dataset.copyBet) {
    const [tier, index] = target.dataset.copyBet.split(":");
    const bet = state.bets?.tiers?.[tier]?.bets?.[Number(index)];
    if (bet) copyText(betText(bet), "Bet");
  } else if (target.dataset.deleteCall) {
    api(`/api/calls/${target.dataset.deleteCall}?player=${encodeURIComponent(state.me.id)}`, { method: "DELETE" })
      .then(() => { toast("Call deleted"); loadCalls(); })
      .catch((error) => { if (!error.signIn) toast(error.message); });
  } else if (target.id === "refresh") {
    if (state.league) loadGames(true);
  } else if (target.id === "sign-out") {
    fetch("/api/logout", { method: "POST" }).finally(() => location.reload());
  } else if (target.dataset.close != null) {
    closeModal();
  }
});

document.addEventListener("input", (event) => {
  const input = event.target;
  const draft = state.draft;
  if (!draft) return;
  if (input.id === "confidence") {
    draft.call.confidence = Number(input.value) / 100;
    updateHints();
    return;
  }
  if (input.id === "home-score" || input.id === "away-score") {
    const home = $("#home-score").value, away = $("#away-score").value;
    draft.call.home_score = home === "" ? null : Math.round(Number(home));
    draft.call.away_score = away === "" ? null : Math.round(Number(away));
    if (draft.call.home_score != null && draft.call.away_score != null && draft.call.home_score !== draft.call.away_score) {
      draft.call.winner = draft.call.home_score > draft.call.away_score ? "home" : "away";
      for (const b of $("#winner-seg").querySelectorAll("button")) b.classList.toggle("on", b.dataset.winner === draft.call.winner);
    }
    judgeDraft();
  }
});

document.addEventListener("change", (event) => {
  const input = event.target;
  const draft = state.draft;
  if (input.id === "confidence" && draft) judgeDraft();
  const row = input.closest(".prop-row");
  if (!row || !draft) return;
  const prop = draft.call.props[Number(row.dataset.prop)];
  const key = input.dataset.k;
  if (key === "line") prop.line = Math.max(0, Math.round(Number(input.value) * 2) / 2);
  else if (key === "confidence") prop.confidence = Number(input.value);
  else prop[key] = input.value;
  if (key === "player_id" || key === "stat") prop.line = suggestLine(prop);
  renderProps();
  judgeDraft();
});

document.addEventListener("submit", (event) => {
  event.preventDefault();
  if (event.target.id === "sign-in") signIn(event.target);
  else if (event.target.id === "call-form") saveCall(event.target);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $('.modal-backdrop[data-dismissible="1"]')) closeModal();
});

// Clicking outside a modal closes it, except the sign-in form
document.addEventListener("mousedown", (event) => {
  if (event.target.matches('.modal-backdrop[data-dismissible="1"]')) closeModal();
});

window.addEventListener("hashchange", () => {
  const view = location.hash.slice(1);
  if (VIEWS.includes(view) && view !== state.view) setView(view);
});

// --- start -------------------------------------------------------------------------------

async function start() {
  for (const b of $("#format-seg").querySelectorAll("button")) b.classList.toggle("on", b.dataset.format === state.format);
  if (!(await loadLeagues())) return;
  try {
    await refreshCalls();
  } catch {
    return;
  }
  setView(state.view);
}

start();
