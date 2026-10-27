// Game: one game in full (GET /api/game). "Markets say" (the markets brain's blend with a chip per source,
// views/markets/say.js), the AI's view next to the market's and ESPN's, the prices (every one toggles a slip
// pick), for soccer the "Goals model" (markets/goals.js: both teams to score, draw no bet, double chance, team
// totals and the likeliest scores at fair odds), why the model's numbers are what they are (typed factors that
// add up to each number, views/markets/factors.js; the old text lines when the server sends none), injuries,
// the AI analysts with "Ask now", the players (each opens a player page), every bookmaker's odds, and "Make
// your call".

import { api, isQuiet, qs } from "../api.js";
import { setTitle, refreshMainButton, leagueTitle, sportEmoji } from "../app.js";
import {
  esc, pct, shortName, timeLabel, section, oddsGrid, splitBar, winChances, badge, explain, skeleton, errorState,
  notice, priceText, pointText, injuryBadge, segmented, emptyState,
} from "../ui.js";
import { icon } from "../icons.js";
import { analystsHtml, bindAnalysts, openCallSheet, loadMyCalls, callSummary, CAUTION_GAP } from "../legacy.js";
import { marketsSection, bindMarkets, blendChances, consensusOf } from "./markets/say.js";
import { factorsSection, bindFactors } from "./markets/factors.js";
import { goalsSection } from "./markets/goals.js";
import { ensureMarketStyles } from "./markets/styles.js";

let current = null; // {game, params} for the Telegram main button

export function mainButton() {
  const game = current?.game;
  if (!game?.prediction || game.live) return null;
  return { text: "Make your call", onClick: () => openCallSheet(game, { onSaved: () => current?.redraw?.() }) };
}

export function onLeave() {
  current = null;
}

function marginText(margin, game) {
  if (margin == null) return "–";
  const team = margin >= 0 ? game.home : game.away;
  return `${shortName(team, game.league)} by ${Math.abs(margin).toFixed(1)}`;
}

function hero(game) {
  const p = game.prediction;
  // The blend is the app's best estimate: it fills the bar when the markets brain has one
  const blend = blendChances(game);
  const chances = blend ? { ...blend, source: "Markets say" } : winChances(game);
  const away = shortName(game.away, game.league);
  const home = shortName(game.home, game.league);
  const marketHome = p?.sources?.market?.home_win_prob;
  const caution = p && marketHome != null && Math.abs(p.home_win - marketHome) >= CAUTION_GAP;
  return `<section class="card game-hero">
    <div class="hero-meta">
      <span>${esc(sportEmoji(game.league))} ${esc(leagueTitle(game.league))}</span>
      <span>${game.live ? `<span class="live-dot" aria-hidden="true"></span> <b class="live">LIVE</b>` : `${icon("clock", { size: 14 })} ${esc(timeLabel(game.start))}`}</span>
      ${p ? badge(`AI: ${shortName(p.pick, game.league)} ${pct(p.pick_prob)}`, "good", "bot") : ""}
      ${caution ? badge("Far from market: check the news", "warn", "alert") : ""}
    </div>
    <div class="hero-teams">
      <div class="hero-team"><b>${esc(game.away)}</b><span>Away</span></div>
      ${p ? `<div class="hero-score" aria-label="AI's predicted score">${Math.round(p.away_score)}<span class="hero-vs"> – </span>${Math.round(p.home_score)}</div>` : '<div class="hero-vs">@</div>'}
      <div class="hero-team"><b>${esc(game.home)}</b><span>Home</span></div>
    </div>
    ${p ? `<p class="muted small" style="text-align:center">AI's predicted score · ${esc(marginText(p.margin, game))} · total ${p.total.toFixed(0)}</p>` : ""}
    ${chances ? `<div style="margin-top:12px">${splitBar(away, chances.away, home, chances.home, { source: chances.source === "AI" ? "AI chance" : chances.source === "Markets say" ? "Markets say" : "Market chance", draw: chances.draw })}</div>` : ""}
  </section>`;
}

function compareRow(label, sub, homeWin, { final = false } = {}) {
  if (homeWin == null) return "";
  const away = 1 - homeWin;
  const favHome = homeWin >= 0.5;
  return `<div class="compare-row${final ? " final" : ""}" role="row">
    <span class="src" role="rowheader"><b>${esc(label)}</b>${esc(sub)}</span>
    <span class="v${favHome ? "" : " good"}" role="cell">${pct(away)}</span>
    <div class="split-bar" aria-hidden="true"><span class="${favHome ? "" : "fav"}" style="width:${(away * 100).toFixed(1)}%"></span><span class="${favHome ? "fav" : ""}" style="width:${(homeWin * 100).toFixed(1)}%"></span></div>
    <span class="v${favHome ? " good" : ""}" role="cell">${pct(homeWin)}</span>
  </div>`;
}

function whoWins(game) {
  const p = game.prediction;
  // Without the AI's own model, "Markets say" already shows the market's and ESPN's views (and more)
  if (!p && consensusOf(game)) return "";
  const s = p?.sources || {};
  const marketHome = s.market?.home_win_prob ?? (game.markets?.h2h?.draw ? null : game.markets?.h2h?.home?.fair);
  const espnHome = s.espn?.home_win_prob ?? game.espn_home_win;
  const rows = [
    p ? compareRow("AI", "NBA model", p.home_win, { final: true }) : "",
    compareRow("Market", "no margin", marketHome),
    compareRow("ESPN", "predictor", espnHome),
    p ? compareRow("Team model", "form, rest", s.model?.home_win_prob) : "",
  ].join("");
  if (!rows) return "";
  const away = shortName(game.away, game.league);
  const home = shortName(game.home, game.league);
  const note = p ? `The AI leans on the bookmakers' consensus and adds ESPN's predictor and its own team model.`
    : `No AI model for ${leagueTitle(game.league)} yet: these are the market's and ESPN's views.`;
  return section("Who wins", `<div class="compare" role="table" aria-label="Win chances by source">
      <div class="compare-head" role="row"><span></span><span class="teams"><span>${esc(away)}</span><span>${esc(home)}</span></span></div>
      ${rows}
    </div>
    ${p ? `<p class="small" style="margin-top:10px">AI: <b>${esc(marginText(p.margin, game))}</b> · total <b>${p.total.toFixed(1)}</b>${s.market?.total != null ? ` <span class="muted">(market ${s.market.total.toFixed(1)})</span>` : ""}</p>` : ""}
    <p class="note">${esc(note)}</p>`, { info: p ? "ai_chance" : "market_chance" });
}

function prices(game) {
  const books = Object.values(game.markets || {}).flatMap((sides) => Object.values(sides)).map((o) => o.books || 0);
  const most = books.length ? Math.max(...books) : 0;
  return section("Prices", `${oddsGrid(game)}
    <p class="note">Best price for each pick across ${most} bookmaker${most === 1 ? "" : "s"}; ${game.prediction ? "“AI” is the brain's chance" : "“Mkt” is the market's chance"}.
      Tap a price to add it to your slip ${explain("multi", { label: "How do slips work?" })}</p>
    ${game.prediction ? `<p class="note">▲ next to a chance: the AI rates that pick above its price (an opinion, not a guarantee). “!”: the AI is far from the bookmakers, so check the injury news ${explain("value_mark", { label: "What do ▲ and ! mean?" })}</p>` : ""}`,
  { info: "spread", aside: "" });
}

function injuries(game) {
  const sides = ["away", "home"].map((side) => {
    const list = game.injuries?.[side] || [];
    return `<div class="injury-team"><b>${esc(game[side])}</b>${list.length
      ? `<div class="chip-row wrap">${list.map((entry) => {
        const m = /^(.*) \(([^)]*)\)$/.exec(entry);
        return m ? `<span class="row" style="gap:6px">${esc(m[1])} ${injuryBadge(m[2])}</span>` : `<span>${esc(entry)}</span>`;
      }).join("")}</div>` : '<span class="muted small">No injuries reported</span>'}</div>`;
  }).join("");
  return section("Injury report", `<div class="injury-list">${sides}</div>`);
}

function playerRow(league, pl) {
  const proj = pl.out ? injuryBadge(pl.status || "Out")
    : `<span class="proj"><span><b>${pl.pts.toFixed(1)}</b> pts</span><span><b>${pl.reb.toFixed(1)}</b> reb</span><span><b>${pl.ast.toFixed(1)}</b> ast</span></span>`;
  return `<li><a class="player-row${pl.out ? " out" : ""}" href="#/player/${esc(league)}/${esc(pl.id)}">
    <span class="who"><b>${esc(pl.name)}</b><span>${esc(pl.position || "")}${pl.out ? "" : ` · ${Math.round(pl.minutes)} min`}${!pl.out && pl.status ? ` · ${esc(pl.status)}` : ""}</span></span>
    ${proj}${icon("chevron", { size: 18 })}</a></li>`;
}

function playersHtml(game, side) {
  if (!game.players) {
    return `<p class="muted small">Find a player to see their last 10 games, hit rates and news.</p>
      <div class="search-row" style="margin-top:8px"><label class="sr-only" for="game-player-q">Player name</label>
        <input id="game-player-q" class="input" type="search" placeholder="Search ${esc(leagueTitle(game.league))} players" autocomplete="off" data-player-search></div>
      <ul class="list search-results" data-player-results></ul>`;
  }
  const list = game.players[side] || [];
  return `${segmented("players-side", [{ value: "away", label: shortName(game.away, game.league) }, { value: "home", label: shortName(game.home, game.league) }], side, { label: "Team" })}
    <p class="muted small" style="margin:10px 0 2px">Projected for this game. Tap a player for their last 10 games and hit rates.</p>
    <ul class="player-list">${list.length ? list.map((pl) => playerRow(game.league, pl)).join("") : '<li class="muted small">No roster from ESPN right now.</li>'}</ul>`;
}

function booksTable(game) {
  const books = game.books || [];
  if (!books.length) return "";
  const away = shortName(game.away, game.league);
  const home = shortName(game.home, game.league);
  const draw = books.some((b) => b.markets?.h2h?.draw);
  const cell = (offer, market, side) => offer
    ? `${market !== "h2h" ? `<span class="muted">${esc(pointText(market, side, offer.point))}</span> ` : ""}${esc(priceText(offer))}` : "–";
  const sharp = books.some((b) => b.role === "sharp");
  return section("Every bookmaker", `<div class="table-wrap"><table>
    <thead><tr><th>Bookmaker</th><th>${esc(away)}</th>${draw ? "<th>Draw</th>" : ""}<th>${esc(home)}</th><th>${esc(away)} spr.</th><th>${esc(home)} spr.</th><th>Over</th><th>Under</th></tr></thead>
    <tbody>${books.map((b) => `<tr><td>${esc(b.title)}${b.role === "sharp" ? ' <span class="badge muted" title="A sharp book or exchange: it feeds Markets say only; its prices are never offered as yours to take">consensus only</span>' : ""}</td>
      <td>${cell(b.markets.h2h?.away, "h2h", "away")}</td>${draw ? `<td>${cell(b.markets.h2h?.draw, "h2h", "draw")}</td>` : ""}<td>${cell(b.markets.h2h?.home, "h2h", "home")}</td>
      <td>${cell(b.markets.spreads?.away, "spreads", "away")}</td><td>${cell(b.markets.spreads?.home, "spreads", "home")}</td>
      <td>${cell(b.markets.totals?.over, "totals", "over")}</td><td>${cell(b.markets.totals?.under, "totals", "under")}</td></tr>`).join("")}
    </tbody></table></div>
    ${sharp ? `<p class="note">“Consensus only”: sharp books and exchanges feed Markets say ${explain("sharp_book")}; best prices and tickets come only from bookmakers you can bet at.</p>` : ""}`);
}

function actions(game, myCall) {
  const ask = `#/ask?game=${encodeURIComponent(`${game.league}:${game.id}`)}&name=${encodeURIComponent(`${shortName(game.away, game.league)} @ ${shortName(game.home, game.league)}`)}`;
  const canCall = game.prediction && !game.live;
  return `<div class="action-bar">
    ${canCall ? `<button type="button" class="btn btn-primary" data-make-call>${icon("target", { size: 18 })}${myCall ? "Edit your call" : "Make your call"}</button>` : ""}
    <a class="btn btn-secondary" href="${esc(ask)}">${icon("ask", { size: 18 })}Ask AI about this game</a>
  </div>
  ${myCall ? `<p class="small" style="margin-top:8px">${icon("target", { size: 14 })} Your call: ${esc(callSummary(myCall.call, game))}</p>` : ""}`;
}

export async function render(root, params) {
  const { league, id } = params;
  setTitle(leagueTitle(league));
  root.innerHTML = `${skeleton("header")}<div class="section">${skeleton("cards", 2)}</div>`;
  let game;
  try {
    game = await api(`/api/game${qs({ league, id })}`, { signal: params.signal });
  } catch (error) {
    if (isQuiet(error) || !root.isConnected) return;
    root.innerHTML = error.status === 404
      ? emptyState({ iconName: "today", title: "This game isn't on the board", body: "It may have finished or the bookmakers took it down.", actions: [{ label: "Back to Today", href: "#/today", primary: true }] })
      : errorState(error);
    return;
  }
  if (!root.isConnected) return;
  setTitle(`${shortName(game.away, league)} @ ${shortName(game.home, league)}`);
  let side = "away";
  let myCall = null;
  let fxKey = null; // the number the factor list shows (kept across a redraw)
  const short = (name) => shortName(name, league);
  const state = { game, params, redraw: null };
  current = state;

  if (game.soccer) ensureMarketStyles(); // the Goals model's table styles (markets/styles.js)
  const draw = () => {
    root.innerHTML = `${hero(game)}
      ${actions(game, myCall)}
      ${marketsSection(game, { short })}
      ${whoWins(game)}
      ${prices(game)}
      ${goalsSection(game, { short })}
      ${factorsSection(game, { key: fxKey })}
      ${injuries(game)}
      ${game.prediction || (game.analysts && !Array.isArray(game.analysts)) ? section("AI analysts", `<div data-analysts>${analystsHtml(game)}</div>`, { info: "analysts", id: "analysts" }) : ""}
      ${section("Players", `<div data-players>${playersHtml(game, side)}</div>`, { id: "players" })}
      ${booksTable(game)}
      ${game.prediction?.method ? `<p class="method">${esc(game.prediction.method)}</p>` : ""}`;
  };
  state.redraw = async () => {
    try {
      myCall = (await loadMyCalls({ fresh: true })).find((c) => c.event_id === game.id && c.status === "pending") || null;
    } catch {
      myCall = null;
    }
    if (root.isConnected) draw();
  };
  draw();
  refreshMainButton();
  if (game.prediction) state.redraw();

  const query = params.query.get("section");
  if (query) root.querySelector(`#${CSS.escape(query)}`)?.scrollIntoView({ block: "start" });

  // "Markets say": the explainer sheet; the factor list: which number it explains
  bindMarkets(root, game);
  bindFactors(root, game, { onPick: (key) => { fxKey = key; } });

  // Analysts: "Ask now" redraws only their section
  bindAnalysts(root, game, () => {
    const box = root.querySelector("[data-analysts]");
    if (box) box.innerHTML = analystsHtml(game);
  });

  let searchTimer = null;
  root.addEventListener("click", (event) => {
    const seg = event.target.closest('[data-seg="players-side"]');
    if (seg) {
      side = seg.dataset.value;
      root.querySelector("[data-players]").innerHTML = playersHtml(game, side);
      return;
    }
    if (event.target.closest("[data-make-call]")) openCallSheet(game, { onSaved: () => state.redraw() });
  });
  root.addEventListener("input", (event) => {
    if (!event.target.matches("[data-player-search]")) return;
    clearTimeout(searchTimer);
    const q = event.target.value.trim();
    const results = root.querySelector("[data-player-results]");
    if (q.length < 2) {
      results.innerHTML = "";
      return;
    }
    searchTimer = setTimeout(async () => {
      try {
        const found = await api(`/api/players/search${qs({ q, league })}`, { signal: params.signal });
        if (!root.isConnected) return;
        results.innerHTML = found.length ? found.map((p) => `<li><a class="list-item" href="#/player/${esc(p.league)}/${esc(p.id)}">
          <span class="grow"><b>${esc(p.name)}</b><span>${esc([p.team, p.position].filter(Boolean).join(" · "))}</span></span>${icon("chevron", { size: 18 })}</a></li>`).join("")
          : `<li class="muted small">No ${esc(leagueTitle(league))} players match “${esc(q)}”.</li>`;
      } catch (error) {
        if (!isQuiet(error) && root.isConnected) results.innerHTML = `<li>${notice(error.message, "warn")}</li>`;
      }
    }, 300);
  });
}
