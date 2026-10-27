// The game page's "Goals model" panel for soccer (v3 Stage 3): the markets the odds board doesn't carry, from
// the goal rates the market's own 1X2 and over/under 2.5 imply (engine/soccer.py: a Dixon–Coles grid with
// rho −0.13 and a +1.5-point nudge to both teams to score). GET /api/game sends them as ``soccer``:
//   {rates: {home, away}, rho, btts_nudge, inputs: {home, draw, away, over_2_5}, fit_ok, model_1x2,
//    markets: {btts: {yes, no}, draw_no_bet: {home, away}, double_chance: {1X, 12, X2},
//              team_totals: {home: [{line, over, under}], away: [...]}},   each outcome {p, fair, book, push?}
//    correct_score: [{score, p, fair}], teams: {home, away}, note}
// Each outcome shows the model's chance and its fair odds, beside a bookmaker's price when one exists (none do
// on this app's boards yet: they only carry moneylines, spreads and totals). Nothing here is tappable: a fair
// price is no bookmaker's offer. The numbers are the server's; this file lays them out (American odds are the
// same price written the other way, for people who read them).

import { esc, pct, section, badge, notice } from "../../ui.js";
import { oddsFormat } from "../../app.js";
import { american } from "../today/model.js";

/** A price in the person's format (fair odds come as decimals). */
function priceWords(decimal) {
  const d = Number(decimal);
  if (!Number.isFinite(d) || d <= 1) return "–";
  return oddsFormat() === "american" ? american(d) : d.toFixed(2);
}

function bookCell(o) {
  return o?.book ? `<td class="num">${esc(priceWords(o.book.price ?? o.book))}${o.book.title ? ` <small>${esc(o.book.title)}</small>` : ""}</td>` : '<td class="num muted">–</td>';
}

function rows(list) {
  return list.map(([name, o, extra]) => (o ? `<tr><th scope="row">${esc(name)}${extra ? `<small>${esc(extra)}</small>` : ""}</th>
    <td class="num">${esc(pct(o.p))}</td><td class="num"><b>${esc(priceWords(o.fair))}</b></td>${bookCell(o)}</tr>` : "")).join("");
}

function table(caption, body) {
  return `<div class="gm-block"><table class="gm-table">
    <caption>${esc(caption)}</caption>
    <thead><tr><th scope="col"><span class="sr-only">Outcome</span></th><th scope="col">Chance</th><th scope="col"><abbr title="Fair odds">Fair</abbr></th><th scope="col">Book</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function teamTotals(team, lines) {
  if (!Array.isArray(lines) || !lines.length) return "";
  return `<div class="gm-block"><table class="gm-table gm-tt">
    <caption>${esc(team)}: goals</caption>
    <thead><tr><th scope="col">Line</th><th scope="col">Over</th><th scope="col">Under</th></tr></thead>
    <tbody>${lines.map((l) => `<tr><th scope="row" class="num">${esc(Number(l.line).toFixed(1))}</th>
      <td class="num">${esc(pct(l.over?.p))} <b>${esc(priceWords(l.over?.fair))}</b></td>
      <td class="num">${esc(pct(l.under?.p))} <b>${esc(priceWords(l.under?.fair))}</b></td></tr>`).join("")}</tbody></table>
    <p class="gm-foot">Chance, then fair odds. No book prices these here.</p></div>`;
}

/** The "Goals model" section, or "" when the game has no soccer block. ``short`` shortens team names. */
export function goalsSection(game, { short = (name) => String(name || "") } = {}) {
  const g = game?.soccer;
  if (!g || !g.markets) return "";
  const home = g.teams?.home || game.home;
  const away = g.teams?.away || game.away;
  const h = short(home);
  const a = short(away);
  const m = g.markets;
  const rates = g.rates || g.expected_goals || {};
  const dnbDraw = m.draw_no_bet?.home?.push ?? m.draw_no_bet?.away?.push;
  const fitNote = g.fit_ok === false && g.model_1x2 && g.inputs
    ? notice(`The goal rates can't match the market exactly here (model: ${h} ${pct(g.model_1x2.home)}, draw ${pct(g.model_1x2.draw)}, ${a} ${pct(g.model_1x2.away)}; market: ${pct(g.inputs.home)}, ${pct(g.inputs.draw)}, ${pct(g.inputs.away)}), so treat these as rough.`, "warn")
    : "";
  const body = `<div class="gm-xg" role="group" aria-label="Expected goals">
      <div><b class="num">${esc(Number(rates.home).toFixed(2))}</b><span>${esc(h)}</span></div>
      <div><b class="num">${esc(Number(rates.away).toFixed(2))}</b><span>${esc(a)}</span></div>
    </div>
    <p class="gm-lead">Expected goals, from the market's home, draw and away chances${g.inputs?.over_2_5 != null ? ` and over 2.5 goals (${esc(pct(g.inputs.over_2_5))}${g.adjusted && g.inputs.over_2_5_before != null ? `, adjusted from the market's ${esc(pct(g.inputs.over_2_5_before))} for a weather or roof change since the odds were fetched` : ""})` : ""}.</p>
    ${fitNote}
    <div class="gm-grid">
      ${table("Both teams to score", rows([["Yes", m.btts?.yes], ["No", m.btts?.no]]))}
      ${table("Draw no bet", rows([[h, m.draw_no_bet?.home, dnbDraw != null ? `a draw (${pct(dnbDraw)}) gives the stake back` : ""], [a, m.draw_no_bet?.away]]))}
      ${table("Double chance", rows([[`${h} or draw`, m.double_chance?.["1X"]], [`${h} or ${a}`, m.double_chance?.["12"]], [`Draw or ${a}`, m.double_chance?.X2]]))}
      ${teamTotals(h, m.team_totals?.home)}
      ${teamTotals(a, m.team_totals?.away)}
      ${(g.correct_score || []).length ? `<div class="gm-block"><table class="gm-table">
        <caption>Likeliest scores</caption>
        <thead><tr><th scope="col">${esc(h)}–${esc(a)}</th><th scope="col">Chance</th><th scope="col"><abbr title="Fair odds">Fair</abbr></th></tr></thead>
        <tbody>${g.correct_score.map((r) => `<tr><th scope="row" class="num">${esc(r.score)}</th><td class="num">${esc(pct(r.p))}</td><td class="num"><b>${esc(priceWords(r.fair))}</b></td></tr>`).join("")}</tbody></table></div>` : ""}
    </div>
    <p class="note">${esc(g.note || "Goal rates fitted to the market's prices, then a Dixon–Coles grid. Priced at fair odds; check your bookmaker's price.")}</p>`;
  return section("Goals model", body, { info: "dixon_coles", id: "goals", className: "gm-goals", aside: badge("Fair odds", "info") });
}
