// The AI record's "Markets" tables: for each sport, every source of the markets brain (Sharp, US books, Kalshi,
// Polymarket, ESPN, ratings, the model) graded on finished games, next to the blend on the same games, so the
// record shows which source has earned its weight. Lower is better for both scores.
//
// Data: GET /api/record/markets = {sports: [{league, title, n, low_confidence, blend: {n, brier, log_loss},
//   sources: [{name, kind, n, brier, log_loss, blend_brier, blend_log_loss}]}], games, note}. Every figure is
//   the server's; under each score, "vs blend" is the source's score minus the blend's on the same games,
//   written with its sign and in words ("better" / "worse"), never by colour alone. A source graded on fewer
//   games than the blend row is compared with the blend on its own games only, so that score is shown under the
//   difference ("vs blend 0.236") and the games read "1 of 2": the blend row's own score is never the base of
//   a difference it didn't make. Four columns (source, games, Brier, log loss), so a phone shows them all.
//
//   renderMarketsRecord(box, {signal}) -> draws the tables (or an empty state) into ``box``, then the Venue &
//                                         conditions line under them (GET /api/record/context, its own request)
//   marketsRecordHtml(data)            -> the HTML, for tests

import { api, isQuiet } from "../../api.js";
import { esc, section, skeleton, notice } from "../../ui.js";
import { leagueTitle, sportEmoji } from "../../app.js";
import { ensureMarketStyles } from "./styles.js";
import { sourceLabel } from "./say.js";
import { contextRecordHtml } from "../context/conditions.js";

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const score = (v) => (num(v) === null ? "–" : v.toFixed(3));

/** "−0.004 better" / "+0.012 worse" / "level": a source's score minus the blend's (lower is better). */
function versus(mine, blend) {
  const a = num(mine);
  const b = num(blend);
  if (a === null || b === null) return { text: "–", words: "", kind: "" };
  const d = Math.round((a - b) * 1000) / 1000;
  if (d === 0) return { text: "0.000", words: "level", kind: "level" };
  return { text: `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}`, words: d > 0 ? "worse" : "better", kind: d > 0 ? "worse" : "better" };
}

/**
 * A score with the source's difference from the blend under it ("0.201" / "−0.001 better"). ``own``: the
 * source has fewer games than the blend row, so the blend's score on its games (the difference's base) is
 * shown too ("vs blend 0.236").
 */
function scoreCell(mine, blend, { own = false } = {}) {
  const v = versus(mine, blend);
  const base = own && v.words ? `<span class="mk-vs-base">vs blend ${esc(score(blend))}</span>` : "";
  return `<td class="num mk-sc"><span class="mk-score">${esc(score(mine))}</span>${v.words ? `<span class="mk-vs-${v.kind}">${esc(v.text)} <small>${esc(v.words)}</small></span>` : ""}${base}</td>`;
}

function sportHtml(sport) {
  const title = typeof sport.title === "string" && sport.title ? sport.title : leagueTitle(sport.league);
  const blend = sport.blend || {};
  const sources = (Array.isArray(sport.sources) ? sport.sources : []).filter((s) => s && typeof s.name === "string");
  const all = num(blend.n);
  let fewer = false;
  const rows = sources.map((s) => {
    const n = num(s.n);
    const own = n !== null && all !== null && n !== all; // graded on other games than the blend row's
    fewer ||= own;
    return `<tr><th scope="row">${esc(sourceLabel(s.name.slice(0, 40), s.kind))}</th><td class="num">${esc(n ?? "–")}${own ? ` <small class="mk-of">of ${esc(all)}</small>` : ""}</td>
      ${scoreCell(s.brier, s.blend_brier, { own })}${scoreCell(s.log_loss, s.blend_log_loss, { own })}</tr>`;
  }).join("");
  const own = fewer ? `<p class="small muted">A source with fewer games than the blend is compared with the blend on its own games only: that score is under its difference (“vs blend”), not the Blend row's.</p>` : "";
  const low = num(sport.low_confidence) ? `<p class="small muted">${esc(sport.low_confidence)} of these games had low confidence (no market price could be counted: ratings only).</p>` : "";
  return `<div class="mk-sport">
    <h3 class="mk-h"><span aria-hidden="true">${esc(sportEmoji(sport.league || ""))}</span> ${esc(title)} <span class="muted small">${esc(num(sport.n) ?? 0)} game${sport.n === 1 ? "" : "s"} graded</span></h3>
    <div class="table-wrap"><table class="mk-table mk-record">
      <caption class="sr-only">${esc(`${title}: each source's Brier score and log loss on graded games, each with its difference from the blend on the same games (lower is better)`)}</caption>
      <thead><tr><th scope="col">Source</th><th scope="col">Games</th><th scope="col">Brier <span class="mk-th-sub">vs blend</span></th><th scope="col">Log loss <span class="mk-th-sub">vs blend</span></th></tr></thead>
      <tbody>
        <tr class="mk-blend-row"><th scope="row">Blend <small>the app's estimate</small></th><td class="num">${esc(num(blend.n) ?? "–")}</td>
          <td class="num mk-sc"><span class="mk-score">${esc(score(blend.brier))}</span></td><td class="num mk-sc"><span class="mk-score">${esc(score(blend.log_loss))}</span></td></tr>
        ${rows}
      </tbody></table></div>
    ${own}${low}
  </div>`;
}

/** The Markets section's HTML from the server's answer. */
export function marketsRecordHtml(data) {
  const sports = (Array.isArray(data?.sports) ? data.sports : []).filter((s) => s && typeof s.league === "string");
  const intro = `<p class="small muted">Each source's last chance before the start, graded after the final, next to the blend's on the same games. Lower is better for both scores: a coin flip scores 0.250 Brier on a two-way game. The weights stay at their starting values for now; once a sport has enough graded games they are to be refitted from this record, so a source earns weight only by forecasting better.</p>`;
  if (!sports.length) {
    return section("Markets", `${intro}<p class="mk-empty">No graded games yet: each game is scored once it finishes, so this fills in from the next results.</p>`, { info: "markets_say", id: "markets-record" });
  }
  return section("Markets", `${intro}${sports.map(sportHtml).join("")}
    ${typeof data.note === "string" ? `<p class="note">${esc(data.note.slice(0, 400))}</p>` : ""}`, { info: "markets_say", id: "markets-record", className: "mk-record-box" });
}

/** Draw the Markets record into ``box`` (its own request: the rest of the AI record never waits for it). */
export async function renderMarketsRecord(box, { signal } = {}) {
  ensureMarketStyles();
  box.innerHTML = skeleton("lines", 3);
  const context = api("/api/record/context", { signal }).catch(() => null);
  let data;
  try {
    data = await api("/api/record/markets", { signal });
  } catch (error) {
    if (isQuiet(error) || !box.isConnected) return;
    box.innerHTML = section("Markets", error.status === 404
      ? '<p class="small muted">The markets record isn\'t on this server yet.</p>'
      : notice(`Couldn't load the markets record: ${error.message || "network error"}`, "warn"), { id: "markets-record" });
    return;
  }
  if (!box.isConnected) return;
  box.innerHTML = marketsRecordHtml(data);
  const found = await context;
  if (!found || !box.isConnected) return;
  const html = contextRecordHtml(found);
  if (html) (box.querySelector("#markets-record") || box).insertAdjacentHTML("beforeend", html);
}

