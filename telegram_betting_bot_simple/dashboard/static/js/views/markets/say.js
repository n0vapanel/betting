// "Markets say" on the game page: the markets brain's blend (one chance per outcome, the app's best estimate)
// as a headline with its confidence, a chip per source (Sharp · US books · Kalshi · Polymarket · ESPN · ratings ·
// Model), each with its chance, "as of" and a "thin" / "out of date" label, Kalshi and Polymarket opening their
// market pages, the ⚖ badge when the counted sources are 5 points or more apart, and an explainer sheet.
//
// Data: GET /api/game's ``consensus`` = {blend: Blend | null, sources: [Source], links: [{name, url}]}
// (engine/consensus.py; Contract B). The spec named it ``markets``, but the game card's ``markets`` already holds
// the prices, so the server sends ``consensus``; a ``markets`` block that has a blend is read too.
//   Blend  = {p: {home, away, draw?}, used: [names], weights: {name: w}, confidence: "normal"|"low",
//             spread_pp, disagree, as_of, n_sources}
//   Source = {name, kind, p: {home, away, draw?}, as_of, status: "ok"|"thin"|"stale"|"missing", url, detail}
// Every number is the server's; this file only lays them out (and escapes every outside text).
//
//   consensusOf(game)          -> {blend, sources, links} checked, or null
//   blendChances(game)         -> {home, away, draw} from the blend (for the hero's bar), or null
//   marketsSection(game)       -> the section's HTML ("" without a consensus)
//   bindMarkets(root, game)    -> wires the explainer buttons (data-mk-explain); returns an unbind function
//   openBlendSheet(consensus)  -> the explainer: how the blend works, its starting weights (and this game's),
//                                 why the market usually wins and how weights are to be refitted from the record
//   sourceLabel(name, kind)    -> a source's name as the app shows it ("Model" is "Team model")
//   weightPct(w)               -> a weight as a percentage, with one decimal when it isn't whole ("2.5%")

import { esc, pct, ago, sheet, badge } from "../../ui.js";
import { icon } from "../../icons.js";
import { ensureMarketStyles } from "./styles.js";

const OUTCOMES = ["home", "away", "draw"];
const STATUSES = ["ok", "thin", "stale", "missing"];
const KIND_ORDER = { sharp: 0, us_books: 1, prediction_market: 2, model_rating: 3, team_model: 4 };
const STATUS_WORDS = { thin: "thin", stale: "out of date", missing: "no price" };
const STATUS_LONG = {
  thin: "thin market: too little money or too wide a spread, so it isn't counted",
  stale: "out of date, so it isn't counted",
  missing: "no price right now",
};
// The starting weights (engine/consensus.py DEFAULT_WEIGHTS): fixed teaching text for the explainer; a game's own
// weights always come from the server's blend
export const STARTING_WEIGHTS = [
  { kind: "sharp", label: "Sharp books and exchanges", who: "Pinnacle, Betfair, Matchbook (margin removed)", w: 0.5 },
  { kind: "us_books", label: "US bookmakers", who: "the median margin-free price of the books you can bet at", w: 0.2 },
  { kind: "prediction_market", label: "Prediction markets", who: "Kalshi and Polymarket mid prices, sharing the weight", w: 0.2 },
  { kind: "model_rating", label: "Ratings", who: "ESPN's predictor, BPI/FPI, Elo, Torvik", w: 0.05 },
  { kind: "team_model", label: "Team model", who: "the app's NBA team model on its own (form, rest, injuries)", w: 0.05 },
];
const MARKET_KINDS = new Set(["sharp", "us_books", "prediction_market"]);

/**
 * A source's name as the app shows it. The server calls the NBA team model on its own "Model"; the game page's
 * "Who wins" and Help call it the team model, and the AI's chance (which leans on the market) is another number.
 */
export function sourceLabel(name, kind = "") {
  return kind === "team_model" || name === "Model" ? "Team model" : name;
}

/** A weight as a percentage: whole ("50%") or with one decimal ("2.5%"), so a game's weights add up to 100%. */
export function weightPct(w) {
  const v = Number(w) * 100;
  if (!Number.isFinite(v)) return "–";
  return Math.abs(v - Math.round(v)) < 0.05 ? `${Math.round(v)}%` : `${v.toFixed(1)}%`;
}

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const https = (url) => (typeof url === "string" && /^https:\/\/[^\s"'<>\\]+$/i.test(url) && url.length <= 300 ? url : null);

function chances(p) {
  if (!p || typeof p !== "object") return null;
  const out = {};
  for (const o of OUTCOMES) if (num(p[o]) !== null) out[o] = p[o];
  return out.home !== undefined && out.away !== undefined ? out : null;
}

/** The game's "Markets say" detail, checked, or null. */
export function consensusOf(game) {
  // The server sends ``consensus``; a ``markets`` block with a blend (the spec's name) is read too
  const raw = game?.consensus || (game?.markets && ("blend" in game.markets || "sources" in game.markets) ? game.markets : null);
  if (!raw || typeof raw !== "object") return null;
  const sources = (Array.isArray(raw.sources) ? raw.sources : []).filter((s) => s && typeof s.name === "string").map((s) => ({
    name: s.name.slice(0, 40),
    kind: typeof s.kind === "string" ? s.kind : "",
    p: chances(s.p),
    asOf: typeof s.as_of === "string" ? s.as_of : null,
    status: STATUSES.includes(s.status) ? s.status : "ok",
    url: https(s.url),
    detail: typeof s.detail === "string" ? s.detail.slice(0, 200) : "",
  })).sort((a, b) => (KIND_ORDER[a.kind] ?? 9) - (KIND_ORDER[b.kind] ?? 9));
  const b = raw.blend && typeof raw.blend === "object" ? raw.blend : null;
  const blend = b && chances(b.p) ? {
    p: chances(b.p),
    used: Array.isArray(b.used) ? b.used.filter((n) => typeof n === "string") : [],
    weights: b.weights && typeof b.weights === "object" ? Object.fromEntries(Object.entries(b.weights).filter(([, w]) => num(w) !== null)) : {},
    confidence: b.confidence === "low" ? "low" : "normal",
    spread: num(b.spread_pp),
    disagree: b.disagree === true,
    asOf: typeof b.as_of === "string" ? b.as_of : null,
    n: num(b.n_sources),
    base: chances(b.base), // the base rate a low-confidence blend was pulled toward (an even split when unknown)
  } : null;
  const named = new Set(sources.filter((s) => s.url).map((s) => s.url));
  const links = (Array.isArray(raw.links) ? raw.links : []).filter((l) => l && typeof l.name === "string" && https(l.url) && !named.has(l.url))
    .map((l) => ({ name: l.name.slice(0, 40), url: https(l.url) }));
  if (!blend && !sources.length) return null;
  return { blend, sources, links };
}

/** The blend's chances {home, away, draw} for the hero's bar, or null. */
export function blendChances(game) {
  const c = consensusOf(game);
  if (!c?.blend) return null;
  return { home: c.blend.p.home, away: c.blend.p.away, draw: c.blend.p.draw ?? null };
}

/** The favourite of the blend (or of the first source when there is no blend). */
function favouriteOf(c) {
  const p = c.blend?.p || c.sources.find((s) => s.p)?.p;
  if (!p) return null;
  let fav = null;
  for (const o of OUTCOMES) if (p[o] !== undefined && (!fav || p[o] > p[fav])) fav = o;
  return fav;
}

/** What a low-confidence blend was pulled toward, in words: the sport's home-win rate, or an even split. */
function baseWords(base) {
  const values = base ? Object.values(base) : [];
  if (!values.length || values.every((v) => Math.abs(v - values[0]) < 1e-9)) return "an even split";
  return base.draw !== undefined
    ? `the sport's usual split (home ${pct(base.home)}, draw ${pct(base.draw)}, away ${pct(base.away)})`
    : `the sport's usual home-win rate (${pct(base.home)})`;
}

/**
 * Why a blend is low confidence: no market price can be counted. Either none prices the game, or the ones shown
 * (a chip below) are thin or out of date, so they aren't counted.
 */
function lowNote(c) {
  const shown = c.sources.filter((s) => MARKET_KINDS.has(s.kind) && s.status !== "ok" && s.p);
  if (!shown.length) return "No market prices this game";
  const names = shown.map((s) => `${sourceLabel(s.name, s.kind)} (${STATUS_WORDS[s.status]})`).join(", ");
  return shown.length === 1
    ? `The only market price, ${names}, isn't counted`
    : `The market prices here, ${names}, aren't counted`;
}

/** A side's name: the game route sends the teams as names. */
function sideName(game, side) {
  if (side === "draw") return "Draw";
  return String(game[side] || side);
}

function shortSide(game, side, short) {
  return side === "draw" ? "the draw" : short(game[side]);
}

function weightOf(c, name) {
  const w = c.blend?.weights?.[name];
  return num(w);
}

function chipHtml(c, s, fav, who) {
  const p = s.p?.[fav] ?? null;
  const counted = s.status === "ok" && (!c.blend || !c.blend.used.length || c.blend.used.includes(s.name));
  const when = s.asOf ? ago(s.asOf) : "";
  const w = weightOf(c, s.name);
  const name = sourceLabel(s.name, s.kind);
  const words = [`${name}: ${who} ${p === null ? "no chance given" : pct(p)}`];
  if (when) words.push(`as of ${when}`);
  if (s.status !== "ok") words.push(STATUS_LONG[s.status]);
  else if (w !== null) words.push(`${weightPct(w)} of the blend`);
  if (s.detail) words.push(s.detail);
  if (s.url) words.push(`opens ${name}'s market page`);
  const inner = `<span class="mk-chip-n">${esc(name)}${s.url ? icon("external", { size: 13 }) : ""}</span>
    <b class="mk-chip-p num">${p === null ? "–" : esc(pct(p))}</b>
    <span class="mk-chip-t">${when ? `<time datetime="${esc(s.asOf)}">${esc(when)}</time>` : ""}${s.status !== "ok" ? `<span class="mk-chip-s ${esc(s.status)}">${esc(STATUS_WORDS[s.status])}</span>` : ""}</span>`;
  const cls = `mk-chip${counted ? "" : " off"}${s.url ? " link" : ""}`;
  const label = esc(words.join(". "));
  const title = s.detail ? ` title="${esc(s.detail)}"` : "";
  return s.url
    ? `<li><a class="${cls}" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer" aria-label="${label}"${title}>${inner}</a></li>`
    : `<li><div class="${cls}" role="group" aria-label="${label}"${title}>${inner}</div></li>`;
}

/**
 * The "Markets say" section of the game page ("" without a consensus). ``short`` turns a team's name into the
 * short form the page uses ("Celtics").
 */
export function marketsSection(game, { short = (name) => String(name || "") } = {}) {
  const c = consensusOf(game);
  if (!c) return "";
  ensureMarketStyles();
  const fav = favouriteOf(c);
  const b = c.blend;
  const started = game.live || (Date.parse(game.start) || Infinity) <= Date.now();
  let head;
  if (b && fav) {
    const others = OUTCOMES.filter((o) => o !== fav && b.p[o] !== undefined)
      .map((o) => `${esc(o === "draw" ? "Draw" : short(game[o]))} ${esc(pct(b.p[o]))}`).join(" · ");
    const n = b.n ?? b.used.length;
    const confidence = b.confidence === "low"
      ? badge("Low confidence: ratings only", "warn")
      : badge(`Blend of ${n} source${n === 1 ? "" : "s"}`, "info");
    // Whole points, as the board, the phone card, the tooltip, the bot and Help have it
    const dis = b.disagree
      ? `<span class="badge warn mk-dis"><span aria-hidden="true">⚖︎</span>Sources differ${b.spread !== null ? ` by ${esc(Math.round(b.spread))} pts` : ""}</span>` : "";
    head = `<div class="mk-headline">
        <p class="mk-big"><span class="sr-only">Markets say: </span><b class="num">${esc(pct(b.p[fav]))}</b> <span class="mk-who">${esc(fav === "draw" ? "Draw" : sideName(game, fav))}</span></p>
        ${others ? `<p class="mk-others">${others}</p>` : ""}
        <div class="mk-badges">${confidence}${dis}</div>
      </div>
      ${b.confidence === "low" ? `<p class="small mk-low-note">${esc(lowNote(c))}, so the chance rests on ratings alone, pulled a quarter of the way toward ${esc(baseWords(b.base))}.</p>` : ""}`;
  } else {
    head = `<p class="muted small">No blend yet: none of the sources below can be counted right now.</p>`;
  }
  const who = fav ? shortSide(game, fav, short) : "";
  const chips = c.sources.length && fav
    ? `<p class="mk-caption">${esc(fav === "draw" ? "A draw" : `${short(game[fav])} to win`)}, by source</p>
      <ul class="mk-chips" role="list">${c.sources.map((s) => chipHtml(c, s, fav, who)).join("")}</ul>` : "";
  const extra = c.links.length
    ? `<p class="small mk-links">More: ${c.links.map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener noreferrer">${esc(l.name)} ${icon("external", { size: 13 })}</a>`).join(" · ")}</p>` : "";
  const asOf = b?.asOf ? `<span class="small muted">as of <time datetime="${esc(b.asOf)}">${esc(ago(b.asOf))}</time></span>` : "";
  return `<section class="card section mk-say" id="markets" aria-labelledby="mk-title">
    <div class="section-head"><h2 id="mk-title">Markets say<button type="button" class="info-btn" data-mk-explain aria-label="How the blend works">${icon("info", { size: 16 })}</button></h2>${asOf}</div>
    ${head}
    ${chips}
    ${extra}
    ${details(game, c, fav, short)}
    <p class="note">${started ? "As it stood at the start: the last view before the game began. " : ""}The blend is the app's best estimate; the market usually wins, so the aim is to match it and explain it, not to beat it.</p>
    <button type="button" class="text-link mk-how" data-mk-explain>How the blend works ${icon("chevron", { size: 16 })}</button>
  </section>`;
}

/** "How this blend was worked out": every source with its chance, its weight in this blend and its status. */
function details(game, c, fav, short) {
  if (!c.sources.length || !fav) return "";
  const rows = c.sources.map((s) => {
    const w = weightOf(c, s.name);
    const p = s.p?.[fav] ?? null;
    return `<tr><th scope="row">${esc(sourceLabel(s.name, s.kind))}</th><td class="num">${p === null ? "–" : esc(pct(p))}</td>
      <td class="num">${s.status === "ok" && w !== null ? esc(weightPct(w)) : "–"}</td>
      <td>${s.status === "ok" ? "counted" : esc(STATUS_WORDS[s.status])}</td>
      <td class="mk-td-note">${esc([s.detail, s.asOf ? ago(s.asOf) : ""].filter(Boolean).join(" · "))}</td></tr>`;
  }).join("");
  return `<details class="mk-more">
    <summary>${icon("down", { size: 16 })}How this blend was worked out</summary>
    <div class="table-wrap"><table class="mk-table">
      <caption class="sr-only">Each source's chance for ${esc(fav === "draw" ? "a draw" : short(game[fav]))} and its weight in the blend</caption>
      <thead><tr><th scope="col">Source</th><th scope="col">${esc(fav === "draw" ? "Draw" : short(game[fav]))}</th><th scope="col">Weight</th><th scope="col">Status</th><th scope="col">Note</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="small muted">Weights start at the values in “How the blend works” and are shared out over the sources this game has; a thin or out-of-date source is shown but not counted.</p>
  </details>`;
}

/** The explainer sheet (also from Help). ``c``: this game's consensus, for its own weights (optional). */
export function openBlendSheet(c = null) {
  ensureMarketStyles();
  const used = c?.blend ? Object.entries(c.blend.weights).filter(([, w]) => w > 0) : [];
  const weights = STARTING_WEIGHTS.map((k) => `<tr><th scope="row">${esc(k.label)}<span class="mk-td-note">${esc(k.who)}</span></th><td class="num">${esc(weightPct(k.w))}</td></tr>`).join("");
  sheet.open({
    title: "How the blend works",
    size: "tall",
    className: "mk-sheet",
    body: `<p class="lead">“Markets say” is one chance per outcome, blended from every source that prices the game: the app's best estimate.</p>
      <h3 class="mk-h">How it's worked out</h3>
      <p>Each source gives its chance with any bookmaker margin taken out. The chances are averaged in log-odds (a logarithmic opinion pool), each kind of source with its own weight. Only sources marked OK count: a thin, out-of-date or missing one is shown but left out. Kalshi and Polymarket share the prediction-market weight, and within 15 minutes of the start they count half, because their prices drift near expiry.</p>
      <h3 class="mk-h">Starting weights</h3>
      <div class="table-wrap"><table class="mk-table mk-weights"><caption class="sr-only">Starting weight of each kind of source</caption>
        <thead><tr><th scope="col">Source</th><th scope="col">Weight</th></tr></thead><tbody>${weights}</tbody></table></div>
      <p class="small muted">A game without some of these shares their weight out over the ones it has.</p>
      ${used.length ? `<div class="example"><span class="eyebrow">This game</span>${esc(used.map(([name, w]) => `${sourceLabel(name)} ${weightPct(w)}`).join(" · "))}</div>` : ""}
      <h3 class="mk-h">Why the market usually wins</h3>
      <p>Sharp prices carry the money and the news of many people who bet for a living, so a fresh closing price is very hard to beat. Vegas closing lines beat a well-known stats model (538's RAPTOR) against the spread, and the best fitted blends gave the market 60–74% of the weight. So the app anchors on the market and explains it; it doesn't claim to beat it.</p>
      <h3 class="mk-h">When the sources disagree, or there's no market</h3>
      <p>⚖ appears when the counted sources are 5 points or more apart on any outcome: often news some prices haven't caught, a thin market or stale ratings. A reason to look closer, not a signal to bet. When no sharp, US-book or prediction-market price can be counted (there is none, or the ones shown are thin or out of date), the ratings are averaged equally, pulled a quarter of the way toward the sport's usual home-win rate (an even split at a neutral site or where there is none), and labelled “Low confidence”.</p>
      <h3 class="mk-h">Weights are earned</h3>
      <p>These are starting weights. After each game every source's forecast is graded (Brier score and log loss), next to the blend's. The weights stay at these starting values for now; once a sport has enough graded games they are to be refitted from that record, so a source earns weight only by forecasting better.</p>
      <div class="explain-links">
        <a class="text-link" href="#/bets?tab=record">See the record ${icon("chevron", { size: 16 })}</a>
        <a class="text-link" href="#/help/markets_say">More in Help ${icon("chevron", { size: 16 })}</a>
      </div>`,
  });
}

/** The explainer buttons in ``root`` (data-mk-explain). Returns a function that removes the listener. */
export function bindMarkets(root, game) {
  const click = (event) => {
    const t = event.target instanceof Element ? event.target.closest("[data-mk-explain]") : null;
    if (!t) return;
    event.preventDefault();
    openBlendSheet(consensusOf(game));
  };
  root.addEventListener("click", click);
  return () => root.removeEventListener("click", click);
}
