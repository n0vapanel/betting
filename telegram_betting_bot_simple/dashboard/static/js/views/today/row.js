// One game as a desktop board row (a CSS-grid row in an ARIA table), plus the pieces cards share: the
// price button, the time and status, the team lines and the chance cells. HTML strings only; every outside
// text goes through esc(). The numbers are the slate's (the server's); edges are the AI's chance minus the
// bookmakers' margin-free chance for the same outcome (model.js), shown in percentage points.

import { esc, pct, priceText, ago, badge, registerLeg } from "../../ui.js";
import { slip, sportEmoji, leagueTitle } from "../../app.js";
import { icon } from "../../icons.js";
import { offer, legFor, edges, teamShort, marketsOf } from "./model.js";

export const MARKETS = [
  { key: "h2h", label: "Winner", sides: ["away", "home", "draw"] },
  { key: "spreads", label: "Spread", sides: ["away", "home"] },
  { key: "totals", label: "Total", sides: ["over", "under"] },
];
const CHECK = `<span class="o-check" aria-hidden="true">${icon("check", { size: 12 })}</span>`; // a picked price's tick (cards)
export const CAUTION = 0.08; // a gap this big usually means news the model hasn't caught
export const NOTABLE = 0.03; // an edge worth drawing attention to

/** The game's detail route, or null for a game the odds feed doesn't know (no detail page yet). */
export function detailHash(game) {
  return game.odds_id ? `#/game/${encodeURIComponent(game.league)}/${encodeURIComponent(game.odds_id)}` : null;
}

/** "+3.2" (points) from 0.032. */
export function points(edge) {
  const v = Math.round(edge * 1000) / 10;
  if (Object.is(v, -0) || v === 0) return "0.0";
  return `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
}

/** A logo (ESPN's image hosts only, as the server checked) or a blank of the same size. */
export function logo(team, size = 20) {
  const url = team?.logo;
  if (typeof url !== "string" || !/^https:\/\/[a-z0-9.-]*espncdn\.com\//i.test(url)) {
    return `<span class="td-logo blank" style="width:${size}px;height:${size}px" aria-hidden="true"></span>`;
  }
  return `<img class="td-logo" src="${esc(url)}" width="${size}" height="${size}" alt="" loading="lazy" decoding="async" referrerpolicy="no-referrer">`;
}

// Formatters made once: toLocale*String builds a new one on every call, which a 300-game board feels
const CLOCK = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const WEEKDAY = new Intl.DateTimeFormat(undefined, { weekday: "short" });
const STAMP = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const LONG = new Intl.DateTimeFormat(undefined, { weekday: "long", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** Clock time ("7:00 PM"), with the weekday when the game isn't on the day shown. */
export function startText(game, { withDay = false } = {}) {
  const d = new Date(game._start ?? game.start);
  if (Number.isNaN(d.getTime())) return "";
  const time = CLOCK.format(d);
  return withDay ? `${WEEKDAY.format(d)} ${time}` : time;
}

/** The status as words and markup: LIVE with the clock, Final, Postponed, or the start time. */
export function statusHtml(game, { withDay = false } = {}) {
  if (game.status === "live") {
    return `<span class="td-live"><span class="live-dot" aria-hidden="true"></span><b class="live">LIVE</b></span>${game.clock ? `<span class="td-clock">${esc(game.clock)}</span>` : ""}`;
  }
  if (game.status === "final") return '<span class="td-final">Final</span>';
  if (game.status === "postponed") return `<span class="td-post">${icon("pause", { size: 13 })}Postponed</span>`;
  return `<span class="td-time num">${esc(startText(game, { withDay }))}</span>`;
}

/** Status in words for labels ("live, Q3 4:12" / "final" / "starts 7:00 PM"). */
export function statusWords(game, { withDay = false } = {}) {
  if (game.status === "live") return `live${game.clock ? `, ${game.clock}` : ""}`;
  if (game.status === "final") return "final";
  if (game.status === "postponed") return "postponed";
  return `starts ${startText(game, { withDay })}`;
}

/** One team's line: logo, name (full and short; CSS picks), record, score. */
export function teamLine(game, side) {
  const team = game[side] || {};
  const other = game[side === "home" ? "away" : "home"] || {};
  const scored = Number.isFinite(team.score) && game.status !== "pre";
  const won = game.status === "final" && Number.isFinite(team.score) && Number.isFinite(other.score) && team.score > other.score;
  const b2b = (game.badges || []).includes(`b2b-${side}`);
  return `<span class="td-team${won ? " won" : ""}">
    ${logo(team)}
    <span class="td-tname" title="${esc(team.name || "")}"><span class="tm-full">${esc(team.name || "")}</span><span class="tm-short">${esc(teamShort(team, game.league))}</span></span>
    ${team.record ? `<span class="td-rec num">${esc(team.record)}</span>` : ""}
    ${b2b ? `<span class="td-b2b" title="${esc(`${team.name || "This team"} played yesterday (back-to-back)`)}">B2B</span>` : ""}
    ${scored ? `<b class="td-score num">${esc(team.score)}</b>` : ""}
  </span>`;
}

/** The leagues' names for a row ("🏀 NBA"). */
export function leagueHtml(game, entry) {
  return `<span class="td-league"><span aria-hidden="true">${esc(sportEmoji(game.league))}</span> ${esc(entry?.title || leagueTitle(game.league))}</span>`;
}

/**
 * A price that toggles a pick in the slip through the shell's one delegated handler (ui.js registerLeg).
 * Started and finished games keep their prices on show but can't be added: they were pre-game prices.
 */
export function priceButton(game, market, side, { compact = false, chance = true } = {}) {
  const leg = legFor(game, market, side);
  if (!leg) return `<span class="odds empty${compact ? " tb-odds" : ""}" aria-hidden="true">–</span>`;
  const o = offer(game, market, side);
  const key = registerLeg(leg);
  const on = slip.has(key);
  const point = market === "spreads" ? `${o.point > 0 ? "+" : ""}${o.point}` : market === "totals" ? `${side === "over" ? "O" : "U"} ${o.point}` : "";
  const shown = { price: o.price, american: leg.american };
  const closed = game.status !== "pre";
  const aiChance = leg.ai;
  const mkt = leg.fair;
  const chanceText = !chance ? "" : aiChance != null ? `AI ${pct(aiChance)}` : mkt != null ? `Mkt ${pct(mkt)}` : "";
  const words = `${leg.label}, odds ${priceText(shown)}${o.book ? ` at ${o.book}` : ""}${aiChance != null ? `, AI chance ${pct(aiChance)}` : ""}${mkt != null ? `, market chance ${pct(mkt)}` : ""}${closed ? " (started: can't be added)" : ""}`;
  return `<button type="button" class="odds${compact ? " tb-odds" : ""}${on ? " on" : ""}" data-leg-key="${esc(key)}" aria-pressed="${on}" aria-label="${esc(words)}"${o.book ? ` title="${esc(`Best price: ${o.book}`)}"` : ""}${closed ? " disabled" : ""}>
    ${point ? `<span class="o-point">${esc(point)}</span>` : ""}<span class="o-price">${esc(priceText(shown))}</span>${chanceText && !compact ? `<span class="o-chance">${esc(chanceText)}</span>` : ""}${compact ? "" : CHECK}</button>`;
}

/** Sides on the lines of a row: away, home, and a draw line for three-way markets. */
export function lineSides(game) {
  return game.odds?.h2h?.draw || game.chance?.draw != null ? ["away", "home", "draw"] : ["away", "home"];
}

/** A chance cell: one value per line (away, home, draw), the favourite in bold. */
export function chanceCell(game, source, sides) {
  const c = game[source];
  if (!c) return sides.map(() => '<span class="td-v muted">–</span>').join("");
  const values = sides.map((s) => (typeof c[s] === "number" ? c[s] : null));
  const top = Math.max(...values.filter((v) => v !== null));
  return values.map((v) => (v === null ? '<span class="td-v muted">–</span>'
    : `<span class="td-v num${v === top ? " fav" : ""}">${esc(pct(v))}</span>`)).join("");
}

// A source's name in words for labels (the slate sends short names: "US", "Poly"); "Model" is the NBA team
// model on its own, named as the game page's "Who wins" names it (the AI's chance is another number)
const SOURCE_WORDS = { US: "US books", Poly: "Polymarket", Model: "Team model" };
// The slate's short names of the market sources (sharp books, US books, prediction markets)
const MARKET_SOURCES = new Set(["Sharp", "US", "Kalshi", "Poly"]);
const STATUS_WORDS = { thin: "thin market", stale: "out of date", missing: "no price" };

/** "Celtics" / "the draw" for a side, in words. */
function sideWords(game, side) {
  if (side === "draw") return "the draw";
  return game[side]?.name || teamShort(game[side], game.league) || side;
}

/** A source's chance in words: "Kalshi 63%" / "Polymarket 61% (thin market)". */
function sourceWords(s) {
  return `${SOURCE_WORDS[s.name] || s.name} ${pct(s.p)}${s.status !== "ok" ? ` (${STATUS_WORDS[s.status] || s.status})` : ""}`;
}

/**
 * Why a "Markets say" is low confidence, in words: no market price can be counted, either because none prices
 * the game or because the ones listed are thin or out of date (shown, but not counted).
 */
export function lowWords(m) {
  const listed = (m?.sources || []).filter((s) => MARKET_SOURCES.has(s.name) && s.status !== "ok");
  if (!listed.length) return "Low confidence: no market prices this game, ratings only";
  const names = listed.map((s) => SOURCE_WORDS[s.name] || s.name).join(", ");
  if (listed.length === 1) {
    const why = { thin: "thin", stale: "out of date" }[listed[0].status] || "not usable";
    return `Low confidence: ratings only, as the only market price (${names}) is ${why} and not counted`;
  }
  return `Low confidence: ratings only, as the market prices (${names}) are thin or out of date and not counted`;
}

/**
 * "Markets say" in words, for a label or a tooltip, as sentences: the blend's favourite, how sure, how far
 * the sources are apart and each source's chance for that side. [] without a "Markets say".
 */
export function marketsParts(game) {
  const m = marketsOf(game);
  if (!m) return [];
  const parts = [`Markets say ${sideWords(game, m.fav)} ${pct(m.p[m.fav])}`];
  if (m.conf === "low") parts.push(lowWords(m));
  if (m.dis) parts.push(`The sources disagree by ${m.spread != null ? `${Math.round(m.spread)} points` : "5 points or more"}`);
  if (m.sources.length) parts.push(`${sideWords(game, m.fav)} by source: ${m.sources.map(sourceWords).join(", ")}`);
  return parts;
}

/** marketsParts as one label ("" without a "Markets say"). */
export function marketsWords(game) {
  return marketsParts(game).join(". ");
}

/**
 * The ⚖ badge: "⚖ 7" on the board (points apart; the words are in the cell's label and tooltip), "⚖ 7 pts
 * apart" on a phone card (``long``).
 */
export function disBadge(m, { long = false } = {}) {
  if (!m?.dis) return "";
  const n = m.spread != null ? Math.round(m.spread) : null;
  const text = long ? (n != null ? `${n} pts apart` : "Sources differ") : n != null ? String(n) : "";
  return `<span class="mk-flag dis"><span class="mk-scale" aria-hidden="true">\u2696\uFE0E</span>${text ? `<span class="mk-flag-t">${esc(text)}</span>` : ""}</span>`;
}

/** The low-confidence badge (no market price that can be counted: ratings only). */
export function lowBadge(m, { long = false } = {}) {
  return m?.conf === "low" ? `<span class="mk-flag low">${long ? "Low confidence" : "Low"}</span>` : "";
}

/**
 * The Markets column: the blend per line (away, home, draw; the favourite in bold), with ⚖ when the sources
 * disagree by 5 points or more and "Low" when no market price can be counted. The tooltip and label list the
 * sources. Older slates (no "Markets say") show the market's chance as before.
 */
export function marketsCell(game, sides) {
  const m = marketsOf(game);
  const c = game.chance;
  if (!c) return `<div class="tb-c tb-mkt" role="cell">${sides.map(() => '<span class="td-v muted">–</span>').join("")}</div>`;
  const values = sides.map((s) => (typeof c[s] === "number" ? c[s] : null));
  const top = Math.max(...values.filter((v) => v !== null));
  const flags = [disBadge(m), lowBadge(m)].filter(Boolean);
  const parts = marketsParts(game);
  const lines = values.map((v, i) => `<span class="td-v num${v !== null && v === top ? " fav" : ""}${v === null ? " muted" : ""}">${flags[i] || ""}<span class="mk-v">${v === null ? "–" : esc(pct(v))}</span></span>`).join("");
  return `<div class="tb-c tb-mkt${m?.dis ? " is-dis" : ""}${m?.conf === "low" ? " is-low" : ""}" role="cell"${parts.length ? ` aria-label="${esc(parts.join(". "))}" title="${parts.map(esc).join("&#10;")}"` : ""}>${lines}</div>`;
}

/** The edge cell: AI minus market per line, in points; ▲ marks a notable edge, ! a suspiciously big one. */
export function edgeCell(game, sides) {
  const e = edges(game);
  if (!e) return sides.map(() => '<span class="td-v muted">–</span>').join("");
  return sides.map((s) => {
    const v = e[s];
    if (v === undefined) return '<span class="td-v muted">–</span>';
    const kind = v >= CAUTION ? "far" : v >= NOTABLE ? "up" : v <= -NOTABLE ? "down" : "";
    const mark = kind === "far" ? '<span class="td-mark warn" aria-hidden="true">!</span>' : kind === "up" ? '<span class="td-mark" aria-hidden="true">▲</span>' : "";
    return `<span class="td-v num td-edge ${kind}">${esc(points(v))}${mark}</span>`;
  }).join("");
}

/** Edge words for a label ("AI 3.2 points above the market on Boston Celtics"). */
export function edgeWords(game) {
  const e = edges(game);
  if (!e) return "";
  let best = null;
  for (const [side, v] of Object.entries(e)) if (!best || v > best[1]) best = [side, v];
  if (!best || Math.abs(best[1]) < 0.0005) return "AI and market agree";
  const name = best[0] === "draw" ? "the draw" : game[best[0]]?.name || best[0];
  return `AI ${points(best[1]).replace(/^[+−]/, "")} points above the market on ${name}`;
}

/** Badges a row shows: back-to-back, neutral site, a caution when the AI is far from the market. */
export function badgesHtml(game) {
  const out = [];
  const list = Array.isArray(game.badges) ? game.badges : [];
  for (const b of list) {
    if (b === "live" || /^b2b-(home|away)$/.test(b)) continue; // live: the status says it; b2b: by the team
    if (b === "neutral") out.push(badge("Neutral site", "muted"));
    else if (typeof b === "string") out.push(badge(b.replace(/[-_]/g, " "), "muted"));
  }
  const b2b = list.filter((b) => /^b2b-(home|away)$/.test(b)).map((b) => game[b.slice(4)]?.abbr || teamShort(game[b.slice(4)], game.league));
  if (b2b.length) out.unshift(`<span class="badge warn" title="${esc("Played yesterday: back-to-back")}">B2B ${esc(b2b.join(" · "))}</span>`);
  const e = edges(game);
  if (e && Math.max(...Object.values(e).map(Math.abs)) >= CAUTION && game.status === "pre") {
    out.push(`<span class="badge warn" title="${esc("The AI is far from the bookmakers: check the injury news")}">${icon("alert", { size: 12 })}Far from market</span>`);
  }
  if (game.venue && game.venue.indoor === false && game.status === "pre") out.push(badge("Outdoors", "muted"));
  return out.join("");
}

/** "12 min ago" for the game's odds, or "–". */
export function asOfHtml(game, { blank = false } = {}) {
  if (!game.odds_as_of) return blank ? "" : '<span class="muted" title="No odds yet">–</span>';
  const when = new Date(game.odds_as_of);
  const stamp = Number.isNaN(when.getTime()) ? "" : STAMP.format(when);
  return `<time class="td-asof" datetime="${esc(game.odds_as_of)}" title="${esc(`Odds as of ${stamp}`)}">${esc(ago(game.odds_as_of))}</time>`;
}

/**
 * Where a game without prices says why: odds not loaded for its league yet (a button opens the league,
 * which asks for them), on their way, no market for it, or a finished game.
 */
export function noOddsHtml(game, entry, asked) {
  if (game.status === "final") return '<span class="td-noodds muted">No prices: the game is over</span>';
  if (game.status === "postponed") return '<span class="td-noodds muted">No prices: postponed</span>';
  if (game.status === "live") return '<span class="td-noodds muted">In play: no prices here</span>';
  const answer = asked?.get?.(game.league);
  if (entry && entry.odds === "fresh") return '<span class="td-noodds muted">No prices for this game yet</span>';
  if (answer?.code === "queued" || answer?.code === "loading" || answer?.code === "asking") {
    if (Date.now() - (answer.at || 0) > 120000) return '<span class="td-noodds muted">The odds didn\'t arrive: try again later</span>';
    return `<span class="td-noodds">${icon("refresh", { size: 14, className: "spin" })}Loading odds…</span>`;
  }
  if (answer && !answer.queued && ["not_in_season", "no_games", "no_listing"].includes(answer.code)) {
    return '<span class="td-noodds muted">No odds market for this game</span>';
  }
  if (answer && !answer.queued && ["reserve", "daily_cap"].includes(answer.code)) {
    return '<span class="td-noodds muted">Odds are saved for later this month</span>';
  }
  return `<button type="button" class="td-noodds link" data-open-league="${esc(game.league)}" aria-label="${esc(`Odds load when you open a league: open ${entry?.title || "this league"}`)}">Odds load when you open a league</button>`;
}

/**
 * A desktop board row. ``ai`` says whether the AI and Edge columns are shown; ``withDay`` adds the weekday to
 * the time (a search across days).
 */
export function rowHtml(game, { entry, ai, withDay, open, asked, prefetchable, expanded, cols = ai ? 11 : 9 }) {
  const href = detailHash(game);
  const sides = lineSides(game);
  const lines = sides.length;
  const away = game.away?.name || "Away";
  const home = game.home?.name || "Home";
  const label = `${away} at ${home}, ${statusWords(game, { withDay })}${game.status !== "pre" && Number.isFinite(game.away?.score) ? `, ${game.away.score}–${game.home?.score}` : ""}`;
  const matchup = `<span class="td-lines">${teamLine(game, "away")}${teamLine(game, "home")}${lines === 3 ? '<span class="td-team td-drawline"><span class="td-tname">Draw</span></span>' : ""}</span>`;
  const opener = href
    ? `<a class="td-open" href="${esc(href)}" aria-label="${esc(`${label}: game details`)}">${matchup}</a>`
    : `<button type="button" class="td-open" data-expand="${esc(game.id)}" aria-expanded="${Boolean(expanded)}" aria-label="${esc(`${label}: more about this game`)}">${matchup}</button>`;
  const odds = game.odds && game.status !== "final"
    ? MARKETS.map((m) => `<div class="tb-c tb-${m.key}" role="cell">${sides.map((s) => {
      const side = m.key === "totals" ? { away: "over", home: "under" }[s] : s;
      if (!side || (m.key !== "h2h" && s === "draw")) return '<span class="tb-gap" aria-hidden="true"></span>';
      return priceButton(game, m.key, side, { compact: true, chance: false });
    }).join("")}</div>`).join("")
    : `<div class="tb-c tb-none" role="cell" aria-colspan="3">${noOddsHtml(game, entry, asked)}</div>`; // Winner, Spread and Total
  const aiCells = ai ? `<div class="tb-c tb-ai" role="cell">${chanceCell(game, "ai", sides)}</div>
    <div class="tb-c tb-edge" role="cell"${edges(game) ? ` aria-label="${esc(edgeWords(game))}"` : ""}>${edgeCell(game, sides)}</div>` : "";
  const notes = badgesHtml(game);
  const asOf = asOfHtml(game, { blank: true });
  // A narrow board (under 820 px) has no Notes column: its badges and "as of" go under the time instead
  return `<div class="tb-row l${lines}${open ? " is-open" : ""}${game.status === "live" ? " is-live" : ""}${game.status === "final" ? " is-final" : ""}" role="row" data-game="${esc(game.id)}"${href ? ` data-route="${esc(href)}"` : ""}${prefetchable && href ? ` data-prefetch="${esc(href)}"` : ""}>
    <div class="tb-c tb-when" role="rowheader">${statusHtml(game, { withDay })}<span class="tb-lg-in">${leagueHtml(game, entry)}</span>${notes || asOf ? `<span class="tb-notes-in">${notes}${asOf}</span>` : ""}</div>
    <div class="tb-c tb-lg" role="cell">${leagueHtml(game, entry)}</div>
    <div class="tb-c tb-match" role="cell">${opener}</div>
    ${odds}
    ${marketsCell(game, sides)}
    ${aiCells}
    <div class="tb-c tb-badges" role="cell">${notes}<span class="tb-asof-in">${asOf}</span></div>
    <div class="tb-c tb-asof" role="cell">${asOfHtml(game)}</div>
  </div>${expanded && !href ? detailHtml(game, entry, asked, { cols }) : ""}`;
}

/**
 * The fold-out under a game that has no detail page yet: venue, records, start, "Markets say" in full (each
 * source, and why a blend is low confidence: the cell's tooltip isn't reachable by touch or keyboard) and why
 * there are no prices. Its one cell spans the row (``cols``: the columns the board shows; Today keeps
 * aria-colspan up to date).
 */
export function detailHtml(game, entry, asked, { cols = 11 } = {}) {
  const when = new Date(game._start ?? game.start);
  const venue = game.venue ? [game.venue.name, game.venue.city].filter(Boolean).join(", ") : "";
  const facts = [
    ["Starts", Number.isNaN(when.getTime()) ? "" : LONG.format(when)],
    ["Where", venue ? `${venue}${game.venue.indoor === true ? " (indoors)" : game.venue.indoor === false ? " (outdoors)" : ""}` : ""],
    ["Records", game.away?.record || game.home?.record ? `${game.away?.name || "Away"} ${game.away?.record || "–"} · ${game.home?.name || "Home"} ${game.home?.record || "–"}` : ""],
    ["Status", game.status === "live" ? `Live${game.clock ? ` (${game.clock})` : ""}` : game.status === "final" ? "Final" : game.status === "postponed" ? "Postponed" : "Not started"],
  ].filter(([, v]) => v);
  const parts = marketsParts(game);
  const m = parts.length ? marketsOf(game) : null;
  const markets = parts.length
    ? `<p class="small td-mk-detail"><b>${esc(parts[0])}.</b> ${esc(parts.slice(1).join(". "))}${parts.length > 1 ? "." : ""}${m?.conf === "low" ? " The ratings are pulled a quarter of the way toward the sport's usual home-win rate (an even split where there is none)." : ""}</p>`
    : "";
  return `<div class="tb-detail" role="row" data-detail="${esc(game.id)}"><div class="tb-detail-in" role="cell" aria-colspan="${esc(cols)}">
    <dl class="td-facts">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
    ${markets}
    <p class="small">${noOddsHtml(game, entry, asked)}</p>
  </div></div>`;
}
