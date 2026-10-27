// One game as a phone card: when and where, both teams (logo, record, score), the top three numbers
// (market chance, the AI's chance, the edge between them, for the favourite or the AI's pick) and the
// price buttons (Winner · Spread · Total, as the game page has them). Fed by the slate; HTML only.

import { esc, pct } from "../../ui.js";
import { icon } from "../../icons.js";
import { favourite, bestEdge, teamShort } from "./model.js";
import {
  MARKETS, detailHash, statusHtml, statusWords, teamLine, leagueHtml, priceButton, lineSides, badgesHtml, asOfHtml,
  noOddsHtml, points, CAUTION, NOTABLE, detailHtml,
} from "./row.js";

function sideName(game, side) {
  if (side === "draw") return "Draw";
  return game[side]?.abbr || teamShort(game[side], game.league) || side;
}

/** Up to three numbers: the market's favourite, the AI's pick and the edge (AI minus market) on it. */
export function topNumbers(game) {
  const out = [];
  const fav = favourite(game, "chance");
  if (fav) out.push({ label: "Market", value: `${pct(fav.chance)} ${sideName(game, fav.side)}`, words: `Market chance ${pct(fav.chance)} for ${fav.side === "draw" ? "a draw" : game[fav.side]?.name || fav.side}` });
  const ai = favourite(game, "ai");
  if (ai) out.push({ label: "AI", value: `${pct(ai.chance)} ${sideName(game, ai.side)}`, words: `AI chance ${pct(ai.chance)} for ${game[ai.side]?.name || ai.side}` });
  const edge = bestEdge(game);
  if (edge) {
    const kind = edge.edge >= CAUTION ? "far" : edge.edge >= NOTABLE ? "up" : "";
    out.push({
      label: "Edge", value: `${points(edge.edge)} ${sideName(game, edge.side)}`, kind,
      words: `AI ${points(edge.edge)} points against the market on ${edge.side === "draw" ? "the draw" : game[edge.side]?.name || edge.side}`,
    });
  }
  return out;
}

export function cardHtml(game, { entry, withDay, asked, prefetchable, expanded, showLeague = true }) {
  const href = detailHash(game);
  const sides = lineSides(game);
  const away = game.away?.name || "Away";
  const home = game.home?.name || "Home";
  const label = `${away} at ${home}, ${statusWords(game, { withDay })}`;
  const head = `<div class="gc-top"><span class="gc-when">${statusHtml(game, { withDay })}${showLeague ? ` <span class="td-dot" aria-hidden="true">·</span> ${leagueHtml(game, entry)}` : ""}</span>
      <span class="gc-badges">${badgesHtml(game)}</span></div>
    <div class="td-lines td-card-teams">${teamLine(game, "away")}${teamLine(game, "home")}</div>`;
  const top = href
    ? `<a class="gc-link" href="${esc(href)}"${prefetchable ? ` data-prefetch="${esc(href)}"` : ""} aria-label="${esc(`${label}: game details`)}">${head}</a>`
    : `<button type="button" class="gc-link td-expand" data-expand="${esc(game.id)}" aria-expanded="${Boolean(expanded)}" aria-label="${esc(`${label}: more about this game`)}">${head}</button>`;
  const nums = topNumbers(game);
  const numbers = nums.length ? `<dl class="td-nums">${nums.map((n) => `<div class="td-num ${n.kind || ""}" aria-label="${esc(n.words)}"><dt>${esc(n.label)}</dt><dd class="num">${esc(n.value)}${n.kind === "far" ? ' <span class="td-mark warn" aria-hidden="true">!</span>' : n.kind === "up" ? ' <span class="td-mark" aria-hidden="true">▲</span>' : ""}</dd></div>`).join("")}</dl>` : "";
  let prices;
  if (game.odds && game.status !== "final") {
    const markets = MARKETS.filter((m) => game.odds[m.key]);
    const row = (s) => `<div class="og-team">${esc(s === "draw" ? "Draw" : teamShort(game[s], game.league))}${s === "home" ? '<span class="og-home">home</span>' : ""}</div>
      ${markets.map((m) => {
        const side = m.key === "totals" ? { away: "over", home: "under" }[s] : s;
        if (!side || (m.key !== "h2h" && s === "draw")) return "<span></span>";
        return priceButton(game, m.key, side);
      }).join("")}`;
    prices = `<div class="odds-grid cols-${markets.length}"><div></div>${markets.map((m) => `<div class="og-head">${esc(m.label)}</div>`).join("")}
      ${sides.map(row).join("")}</div>`;
  } else {
    prices = `<p class="td-card-none small">${noOddsHtml(game, entry, asked)}</p>`;
  }
  const ask = `#/ask?game=${encodeURIComponent(`${game.league}:${game.odds_id || ""}`)}&name=${encodeURIComponent(`${teamShort(game.away, game.league)} @ ${teamShort(game.home, game.league)}`)}`;
  const foot = game.odds_id ? `<div class="gc-foot"><span class="small muted td-card-asof">${game.odds_as_of ? `Odds ${asOfHtml(game)}` : ""}</span>
    <span class="chip-row"><a class="chip ask" href="${esc(ask)}">${icon("ask", { size: 16 })}Ask AI</a>
    <a class="text-link" href="${esc(href)}" tabindex="-1" aria-hidden="true">Details ${icon("chevron", { size: 16 })}</a></span></div>` : "";
  return `<article class="card game-card td-card${game.status === "live" ? " is-live" : ""}" data-game="${esc(game.id)}">
    ${top}${numbers}<div class="td-card-prices">${prices}</div>${expanded && !href ? detailHtml(game, entry, asked).replace('role="row"', "").replace(/ role="cell" aria-colspan="\d+"/, "") : ""}${foot}
  </article>`;
}
