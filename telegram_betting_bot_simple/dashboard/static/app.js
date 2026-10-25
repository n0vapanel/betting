"use strict";

// --- small helpers -----------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (p) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const money = (n) => `$${(n ?? 0).toFixed(2)}`;
const signed = (n, digits = 1) => `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
// Team nickname for tight spaces ("Oklahoma City Thunder" -> "Thunder", "Novak Djokovic" -> "Djokovic");
// soccer clubs keep their full name ("Manchester United" isn't "United")
const TWO_WORD_NICKNAMES = ["Trail Blazers", "Red Sox", "White Sox", "Blue Jays", "Maple Leafs", "Golden Knights", "Red Wings", "Blue Jackets"];
const shortName = (name) => {
  name = String(name);
  if ((state.league || "").startsWith("soccer_")) return name;
  const twoWord = TWO_WORD_NICKNAMES.find((nickname) => name.endsWith(` ${nickname}`));
  if (twoWord) return twoWord;
  const words = name.split(" ");
  return words.length > 1 ? words[words.length - 1] : name;
};

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
    throw Object.assign(new Error("Sign in to see games and odds"), { signIn: true });
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

// --- modals (in place of the browser's own pop-ups) ----------------------------------

function openModal(html, { dismissible = true } = {}) {
  closeModal();
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  if (dismissible) backdrop.dataset.dismissible = "1";
  backdrop.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${html}</div>`;
  document.body.appendChild(backdrop);
  document.body.classList.add("modal-open");
  backdrop.querySelector("input, textarea, button")?.focus();
  return backdrop.firstElementChild;
}

function closeModal() {
  document.querySelector(".modal-backdrop")?.remove();
  document.body.classList.remove("modal-open");
}

function showSignIn() {
  if ($("#sign-in")) return; // already showing
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
    loadLeagues();
    priceSlip();
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

// --- state (picks and settings survive a reload in this browser) ----------------

const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem("sportsbet-ai") || "{}");
  } catch {
    return {};
  }
})();

const state = {
  leagues: [],
  league: saved.league || null,
  data: null, // /api/games response for the league on screen
  picks: Array.isArray(saved.picks) ? saved.picks : [], // {key, league, id, market, side, label, game}
  mode: saved.mode === "single" ? "single" : "multi",
  format: saved.format === "decimal" ? "decimal" : "american",
  stake: typeof saved.stake === "number" ? saved.stake : 10,
  stakes: saved.stakes || {}, // single stakes by pick key
  details: {}, // /api/game responses by game id
  open: new Set(), // game ids with details expanded
  tabs: {}, // selected details tab by game id
  slip: null, // last /api/slip response
};

function persist() {
  try {
    localStorage.setItem("sportsbet-ai", JSON.stringify({
      league: state.league, picks: state.picks, mode: state.mode,
      format: state.format, stake: state.stake, stakes: state.stakes,
    }));
  } catch {
    /* private browsing: settings just won't persist */
  }
}

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

// --- sidebar -----------------------------------------------------------------------

async function loadLeagues() {
  try {
    state.leagues = await api("/api/leagues");
  } catch (error) {
    $("#games").innerHTML = error.signIn ? "" : `<div class="error-box">${esc(error.message)}</div>`;
    $("#league-title").textContent = error.signIn ? "" : "Couldn't load leagues";
    return;
  }
  const all = state.leagues.flatMap((group) => group.leagues);
  if (!all.some((league) => league.key === state.league)) {
    state.league = (all.find((league) => league.key === "basketball_nba") || all[0] || {}).key || null;
  }
  renderSidebar();
  if (state.league) loadGames();
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

// --- games ---------------------------------------------------------------------------

async function loadGames(refresh = false) {
  const league = state.league;
  const title = state.leagues.flatMap((g) => g.leagues).find((l) => l.key === league)?.title || league;
  $("#league-title").textContent = title;
  if (!refresh) $("#games").innerHTML = '<div class="loading">Loading games, odds and predictions…</div>';
  try {
    const data = await api(`/api/games?league=${encodeURIComponent(league)}${refresh ? "&refresh=1" : ""}`);
    if (league !== state.league) return; // the user switched leagues meanwhile
    const before = state.data?.odds_as_of;
    state.data = data;
    renderGames();
    renderQuota(data.quota);
    if (refresh) toast(data.odds_as_of === before ? "Odds are fresh (less than 5 minutes old)" : `Odds updated ${data.odds_as_of}`);
  } catch (error) {
    if (!error.signIn) $("#games").innerHTML = `<div class="error-box">${esc(error.message)}</div>`;
  }
}

function renderQuota(quota) {
  const el = $("#quota");
  if (!quota || quota.remaining == null) return;
  el.textContent = `Odds API: ${quota.remaining} left`;
  el.classList.toggle("low", quota.remaining < 50);
}

function renderGames() {
  const data = state.data;
  $("#league-sub").textContent = [
    data.ai ? `AI model ${data.model}` : "Market view (AI model for this league is coming)",
    data.odds_as_of ? `odds as of ${data.odds_as_of}` : null,
    `${data.games.length} games`,
  ].filter(Boolean).join(" · ");
  if (!data.games.length) {
    $("#games").innerHTML = '<div class="loading">No upcoming games with odds right now.</div>';
    return;
  }
  $("#games").innerHTML = data.games.map(renderCard).join("");
}

function cell(game, market, side) {
  const offer = game.markets[market]?.[side];
  if (!offer) return '<div class="odds empty">–</div>';
  const key = `${game.id}|${market}|${side}`;
  const selected = state.picks.some((pick) => pick.key === key);
  // Live odds move with the score; pre-game probabilities would be misleading next to them
  const value = !game.live && offer.ev != null && offer.ev > 0.05;
  const label = state.data.ai ? "AI" : "Mkt";
  const chance = game.live ? "live" : `${label} ${pct(offer.ai)}`;
  return `
    <button class="odds${selected ? " sel" : ""}" data-key="${esc(key)}" data-market="${market}" data-side="${side}"
      data-id="${esc(game.id)}" title="Best price at ${esc(offer.book)} (${offer.books} bookmakers)${value ? " · our model rates this 5%+ above the price (an opinion, not a guarantee)" : ""}">
      ${market !== "h2h" ? `<span class="pt">${esc(pointText(market, side, offer.point))}</span>` : ""}
      <span class="pr">${esc(priceText({ american: offer.american, price: offer.price }))}</span>
      <span class="aip${value ? " val" : ""}">${chance}</span>
    </button>`;
}

function renderCard(game) {
  const p = game.prediction;
  const open = state.open.has(game.id);
  const chip = p
    ? `<span class="chip ai">${game.live ? "Pre-game AI" : "AI pick"}: ${esc(shortName(p.pick))} ${pct(p.pick_prob)}</span>`
    : game.espn_home_win != null
      ? `<span class="chip">ESPN: ${esc(shortName(game.espn_home_win >= 0.5 ? game.home : game.away))} ${pct(Math.max(game.espn_home_win, 1 - game.espn_home_win))}</span>`
      : "";
  const row = (side) => `
    <div class="team" title="${esc(game[side])}"><span class="full">${esc(game[side])}</span><span class="short">${esc(shortName(game[side]))}</span>${p ? `<span class="proj">${Math.round(side === "home" ? p.home_score : p.away_score)}</span>` : ""}</div>
    ${cell(game, "spreads", side)}${cell(game, "totals", side === "away" ? "over" : "under")}${cell(game, "h2h", side)}`;
  const draw = game.markets.h2h?.draw
    ? `<div class="team">Draw</div><div></div><div></div>${cell(game, "h2h", "draw")}`
    : "";
  const foot = p
    ? `<span>AI: ${esc(shortName(game.away))} ${Math.round(p.away_score)} – ${Math.round(p.home_score)} ${esc(shortName(game.home))} · total ${p.total}</span>`
    : '<span class="muted">Market prices · tap for every bookmaker</span>';
  return `
    <article class="card" data-id="${esc(game.id)}">
      <div class="card-top">
        <span class="${game.live ? "live" : "when"}">${game.live ? "● LIVE" : esc(game.when)}</span>${chip}
      </div>
      <div class="market-grid">
        <div></div><div class="mh">Spread</div><div class="mh">Total</div><div class="mh">Moneyline</div>
        ${row("away")}${row("home")}${draw}
      </div>
      <div class="card-foot">${foot}
        <button class="link details-btn" data-id="${esc(game.id)}">${open ? "Hide ▴" : p
          ? '<span class="full">Prediction &amp; players ▾</span><span class="short">Details ▾</span>'
          : "All odds ▾"}</button>
      </div>
      ${open ? `<div class="details">${renderDetails(game)}</div>` : ""}
    </article>`;
}

// --- game details ----------------------------------------------------------------------

async function toggleDetails(id) {
  if (state.open.has(id)) {
    state.open.delete(id);
    renderGames();
    return;
  }
  state.open.add(id);
  renderGames();
  if (!state.details[id]) {
    try {
      state.details[id] = await api(`/api/game?league=${encodeURIComponent(state.league)}&id=${encodeURIComponent(id)}`);
    } catch (error) {
      if (error.signIn) state.open.delete(id);
      else state.details[id] = { error: error.message };
    }
    renderGames();
  }
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
    <div class="scoreline">
      <div><div class="tn">${esc(game.away)}</div><div class="sc">${p.away_score.toFixed(0)}</div></div>
      <div class="vs">predicted</div>
      <div><div class="tn">${esc(game.home)}</div><div class="sc">${p.home_score.toFixed(0)}</div></div>
    </div>
    <div class="wpbar">
      <div class="${p.home_win < 0.5 ? "fav" : "dog"}" style="width:${(1 - p.home_win) * 100}%">${esc(shortName(game.away))} ${pct(1 - p.home_win)}</div>
      <div class="${p.home_win >= 0.5 ? "fav" : "dog"} right" style="width:${p.home_win * 100}%">${pct(p.home_win)} ${esc(shortName(game.home))}</div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Source</th><th>${esc(shortName(game.home))} win</th><th>Margin</th><th>Total</th></tr></thead>
      <tbody>
        ${row("Betting market (no margin)", s.market)}${row("ESPN predictor", s.espn)}${row("Our team model", s.model)}
        <tr class="final"><td>AI prediction</td><td>${pct(p.home_win)}</td><td>${marginText(p.margin, game)}</td><td>${p.total.toFixed(1)}</td></tr>
      </tbody>
    </table></div>
    ${p.factors.length ? `<h4>What moved our model</h4><ul class="factors">${p.factors.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
    <h4>Injury report</h4>${injuries}
    <p class="note">Model v1 blends the bookmakers' consensus (65%), ESPN's matchup predictor (10%) and our team model
      (25%: points scored and allowed this season and last, home court, back-to-backs, injured players' scoring).
      Win, cover and over/under chances assume the final margin and total vary around the prediction by about
      12.5 and 18 points. The closing betting market is the best public NBA predictor; where our model disagrees
      with it, treat that as an opinion, not an edge.</p>`;
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
      <thead><tr><th>Bookmaker</th><th>${esc(shortName(game.away))} ML</th><th>${esc(shortName(game.home))} ML</th>
        <th>${esc(shortName(game.away))} spread</th><th>${esc(shortName(game.home))} spread</th><th>Over</th><th>Under</th></tr></thead>
      <tbody>${books.map((b) => `
        <tr><td>${esc(b.title)}</td>
          <td>${price(b.markets.h2h?.away, "h2h", "away")}</td><td>${price(b.markets.h2h?.home, "h2h", "home")}</td>
          <td>${price(b.markets.spreads?.away, "spreads", "away")}</td><td>${price(b.markets.spreads?.home, "spreads", "home")}</td>
          <td>${price(b.markets.totals?.over, "totals", "over")}</td><td>${price(b.markets.totals?.under, "totals", "under")}</td></tr>`).join("")}
      </tbody>
    </table></div>`;
}

// --- bet slip ---------------------------------------------------------------------------

function togglePick(button) {
  const key = button.dataset.key;
  const index = state.picks.findIndex((pick) => pick.key === key);
  if (index >= 0) {
    state.picks.splice(index, 1);
    delete state.stakes[key];
  } else {
    const game = state.data.games.find((g) => g.id === button.dataset.id);
    const offer = game.markets[button.dataset.market][button.dataset.side];
    state.picks.push({
      key, league: state.league, id: game.id, market: button.dataset.market, side: button.dataset.side,
      label: offer.label, game: `${game.away} @ ${game.home}`,
    });
    if (state.mode === "single" && state.stakes[key] == null) state.stakes[key] = state.stake;
  }
  persist();
  renderGames();
  priceSlip();
}

let slipTimer = null;
let slipRequest = 0;
function priceSlip() {
  clearTimeout(slipTimer);
  renderSlip();
  slipTimer = setTimeout(async () => {
    if (!state.picks.length) {
      state.slip = null;
      renderSlip();
      return;
    }
    const request = ++slipRequest;
    const stakes = {};
    state.picks.forEach((pick, i) => { stakes[i] = Number(state.stakes[pick.key] ?? state.stake) || 0; });
    try {
      const result = await api("/api/slip", {
        method: "POST",
        body: JSON.stringify({
          mode: state.mode,
          stake: Number(state.stake) || 0,
          stakes,
          picks: state.picks.map(({ league, id, market, side }) => ({ league, id, market, side })),
        }),
      });
      if (request !== slipRequest) return; // a newer request is on its way
      if (result.removed?.length) {
        const gone = new Set(result.removed.map((r) => `${r.id}|${r.market}|${r.side}`));
        state.picks = state.picks.filter((pick) => !gone.has(pick.key));
        persist();
        toast(`${gone.size} pick(s) are no longer offered and were removed`);
      }
      state.slip = result;
    } catch (error) {
      state.slip = error.signIn ? null : { error: error.message };
    }
    renderSlip();
  }, 150);
}

function renderSlip() {
  $("#slip-count").textContent = state.picks.length;
  $("#fab-count").textContent = state.picks.length;
  for (const seg of ["#mode-seg", "#slip-mode"]) {
    for (const button of $(seg).querySelectorAll("button")) button.classList.toggle("on", button.dataset.mode === state.mode);
  }
  for (const button of $("#format-seg").querySelectorAll("button")) button.classList.toggle("on", button.dataset.format === state.format);

  const body = $("#slip-body");
  if (!state.picks.length) {
    body.innerHTML = `<div class="empty-slip">Tap any odds to add a pick.<br><br>
      <b>Singles</b>: each pick is its own bet.<br><b>Multi</b>: combine picks from different games into one parlay.</div>`;
    return;
  }
  const slip = state.slip && !state.slip.error && state.slip.mode === state.mode ? state.slip : null;
  const legs = state.picks.map((pick, i) => {
    const leg = slip?.legs?.[i];
    const ev = leg?.ev;
    const stake = state.stakes[pick.key] ?? state.stake;
    return `
      <div class="leg${leg?.started ? " bad" : ""}">
        <div class="leg-top">
          <span class="leg-label">${esc(leg?.label || pick.label)}</span>
          <span><span class="leg-price">${leg ? esc(priceText(leg.odds)) : "…"}</span>
          <button class="x" data-remove="${esc(pick.key)}" title="Remove">×</button></span>
        </div>
        <div class="leg-game">${esc(pick.game)}</div>
        <div class="leg-meta">
          <span>AI chance ${pct(leg?.probability)}</span>
          ${ev != null ? `<span class="${ev > 0 ? "good" : ""}" style="color:${ev > 0 ? "var(--accent)" : "var(--muted)"}">EV ${signed(ev * 100)}%</span>` : ""}
        </div>
        ${state.mode === "single" ? `
          <div class="stake-row"><span class="muted">Stake $</span>
            <input type="number" min="0" step="1" inputmode="decimal" value="${esc(stake)}" data-stake="${esc(pick.key)}">
            <span class="muted">Returns <b style="color:var(--text)">${leg ? money(leg.returns) : "…"}</b></span></div>` : ""}
      </div>`;
  }).join("");

  let summary = "";
  if (state.slip?.error) {
    summary = `<div class="reasons">${esc(state.slip.error)}</div>`;
  } else if (state.mode === "multi") {
    const odds = slip?.odds;
    const others = (slip?.books || []).slice(1, 6);
    summary = `
      <div class="summary">
        <div class="stake-row" style="margin-top:0"><span class="muted">Stake $</span>
          <input type="number" min="0" step="1" inputmode="decimal" value="${esc(state.stake)}" id="multi-stake"></div>
        <div class="quick">${[5, 10, 25, 50, 100].map((v) => `<button data-quick="${v}">$${v}</button>`).join("")}</div>
        <div class="sum-row"><span>${state.picks.length}-leg multi odds</span><b>${odds ? esc(priceText(odds)) : "…"}</b></div>
        ${slip?.book ? `<div class="sum-row muted"><span>Best price at</span><span>${esc(slip.book)}</span></div>` : ""}
        <div class="sum-row big"><span>Potential payout</span><span>${slip ? money(slip.returns) : "…"}</span></div>
        <div class="sum-row"><span class="muted">Profit</span><span>${slip ? money(slip.profit) : "…"}</span></div>
        <div class="sum-row"><span class="muted">AI chance all legs win</span><span>${pct(slip?.probability)}</span></div>
        ${slip?.ev != null ? `<div class="sum-row"><span class="muted">Expected value</span><span class="${slip.ev > 0 ? "good" : "bad"}">${signed(slip.ev * 100)}%</span></div>` : ""}
        ${slip?.book_edge != null ? `<div class="sum-row"><span class="muted">Bookmaker edge on this multi</span><span>${(slip.book_edge * 100).toFixed(1)}%</span></div>` : ""}
        ${others.length ? `<div class="books-list">Other bookmakers: ${others.map((b) => `<span>${esc(b.book)} ${esc(state.format === "decimal" ? b.decimal.toFixed(2) : b.american)}</span>`).join("")}</div>` : ""}
      </div>`;
  } else if (slip) {
    summary = `
      <div class="summary">
        <div class="sum-row"><span class="muted">Total stake</span><span>${money(slip.stake)}</span></div>
        <div class="sum-row big"><span>Total potential returns</span><span>${money(slip.returns)}</span></div>
      </div>`;
  }
  const reasons = slip?.reasons?.length ? `<ul class="reasons">${slip.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";
  body.innerHTML = `${legs}${summary}${reasons}
    <div class="slip-actions"><button class="primary" id="copy-slip">Copy slip</button></div>`;
}

function slipText() {
  const slip = state.slip;
  if (!slip || slip.error) return "";
  const lines = state.picks.map((pick, i) => `• ${slip.legs[i]?.label || pick.label} (${priceText(slip.legs[i]?.odds)}) — ${pick.game}`);
  if (state.mode === "multi") {
    lines.push(`${state.picks.length}-leg multi at ${priceText(slip.odds)}${slip.book ? ` (${slip.book})` : ""}: ${money(slip.stake)} returns ${money(slip.returns)}`);
    if (slip.probability != null) lines.push(`AI chance all legs win: ${pct(slip.probability)}`);
  } else {
    lines.push(`Singles: ${money(slip.stake)} staked returns ${money(slip.returns)} if all win`);
  }
  return `SportsBet AI\n${lines.join("\n")}`;
}

// --- events ----------------------------------------------------------------------------

document.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.league && target.classList.contains("league-btn")) {
    state.league = target.dataset.league;
    state.open.clear();
    persist();
    renderSidebar();
    loadGames();
  } else if (target.classList.contains("odds") && target.dataset.key) {
    togglePick(target);
  } else if (target.classList.contains("details-btn")) {
    toggleDetails(target.dataset.id);
  } else if (target.dataset.tab) {
    state.tabs[target.dataset.id] = target.dataset.tab;
    renderGames();
  } else if (target.dataset.mode) {
    state.mode = target.dataset.mode;
    persist();
    priceSlip();
  } else if (target.dataset.format) {
    state.format = target.dataset.format;
    persist();
    if (state.data) renderGames();
    renderSlip();
  } else if (target.dataset.remove) {
    state.picks = state.picks.filter((pick) => pick.key !== target.dataset.remove);
    delete state.stakes[target.dataset.remove];
    persist();
    if (state.data) renderGames();
    priceSlip();
  } else if (target.dataset.quick) {
    state.stake = Number(target.dataset.quick);
    persist();
    priceSlip();
  } else if (target.id === "slip-clear") {
    state.picks = [];
    state.stakes = {};
    persist();
    if (state.data) renderGames();
    priceSlip();
  } else if (target.id === "copy-slip") {
    const text = slipText();
    if (text) copyText(text, "Bet slip");
  } else if (target.id === "refresh") {
    if (state.league) loadGames(true);
  } else if (target.id === "slip-fab") {
    document.body.classList.toggle("slip-open");
  } else if (target.id === "sign-out") {
    fetch("/api/logout", { method: "POST" }).finally(() => location.reload());
  } else if (target.dataset.close != null) {
    closeModal();
  }
});

document.addEventListener("submit", (event) => {
  if (event.target.id !== "sign-in") return;
  event.preventDefault();
  signIn(event.target);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $('.modal-backdrop[data-dismissible="1"]')) closeModal();
});

// Clicking outside a modal closes it, except the sign-in form
document.addEventListener("mousedown", (event) => {
  if (event.target.matches('.modal-backdrop[data-dismissible="1"]')) closeModal();
});

document.addEventListener("input", (event) => {
  const input = event.target;
  if (input.id === "multi-stake") {
    state.stake = Math.max(0, Number(input.value) || 0);
  } else if (input.dataset.stake) {
    state.stakes[input.dataset.stake] = Math.max(0, Number(input.value) || 0);
  } else {
    return;
  }
  persist();
  // Re-price without re-rendering the input being typed in
  clearTimeout(slipTimer);
  slipTimer = setTimeout(() => {
    const focused = document.activeElement?.id || document.activeElement?.dataset?.stake;
    priceSlip();
    setTimeout(() => {
      const again = focused === "multi-stake" ? $("#multi-stake") : $(`[data-stake="${CSS.escape(focused || "")}"]`);
      if (again) { again.focus(); again.setSelectionRange?.(again.value.length, again.value.length); }
    }, 200);
  }, 400);
});

renderSlip();
loadLeagues();
priceSlip();
