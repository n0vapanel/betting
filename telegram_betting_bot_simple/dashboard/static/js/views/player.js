// Player: who and what's next (team, position, injury badge, next game), market chips, the last 10
// games as bars against the line, hit rates with their sample sizes, the AI's projection card (NBA, game
// day), dated news with sources, the posted props, and "Ask AI about <player>".
//
// All figures come from GET /api/players/{league}/{id}: ESPN's game log and news, DraftKings' lines and
// hit rates, and the model's projection and P(over) computed on the server. Moving the line with the
// − / + buttons only recounts the games drawn on the chart (the AI's chance needs a posted line).

import { api, isQuiet } from "../api.js";
import { setTitle, leagueTitle, sportEmoji } from "../app.js";
import {
  esc, pct, chanceText, priceText, timeLabel, shortDate, ago, section, chip, badge, injuryBadge, explain,
  skeleton, errorState, emptyState, notice, barChart, formatStat,
} from "../ui.js";
import { icon } from "../icons.js";
import { setVerticalSwipes } from "../telegram.js";

const LABELS = {
  pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "Threes", stl: "Steals", blk: "Blocks", tov: "Turnovers",
  "pts+reb+ast": "Pts+Reb+Ast", "pts+reb": "Pts+Reb", "pts+ast": "Pts+Ast", "reb+ast": "Reb+Ast", "stl+blk": "Stl+Blk",
  "PASS YDS": "Passing yds", "RUSH YDS": "Rushing yds", "REC YDS": "Receiving yds", REC: "Receptions", "PASS TD": "Passing TDs",
  "RUSH TD": "Rushing TDs", "REC TD": "Receiving TDs", CMP: "Completions", "PASS ATT": "Pass attempts", CAR: "Carries",
  H: "Hits", R: "Runs", RBI: "RBIs", HR: "Home runs", K: "Strikeouts", BB: "Walks", SB: "Stolen bases", ER: "Earned runs",
  "H+R+RBI": "Hits+Runs+RBIs", "H+2B+3B+3B+HR+HR+HR": "Total bases",
  G: "Goals", A: "Assists", PTS: "Points", SOG: "Shots on goal", SV: "Saves",
};
const PREFERRED = {
  basketball: ["pts", "reb", "ast", "fg3m", "pts+reb+ast"],
  americanfootball: ["PASS YDS", "RUSH YDS", "REC YDS", "REC", "PASS TD"],
  baseball: ["H", "R", "RBI", "HR", "K", "H+R+RBI"],
  icehockey: ["G", "A", "PTS", "SOG", "SV"],
};
const PROJECTION_KEY = { "pts+reb+ast": "pra" }; // the model's key for a stat, where it differs

let leaving = null;

export function onLeave() {
  setVerticalSwipes(true);
  leaving?.();
  leaving = null;
}

/** A row's value for a stat key or a sum ("pts+reb+ast"), as the server's stat_value does. */
function statValue(row, stat) {
  const stats = row.stats || {};
  if (stat in stats) return stats[stat];
  let total = 0;
  for (const part of String(stat).split("+")) {
    const key = part.trim();
    let value = stats[key];
    if (value == null) value = Object.entries(stats).find(([k]) => k.toLowerCase() === key.toLowerCase())?.[1];
    if (value == null) return null;
    total += value;
  }
  return total;
}

function averageOf(averages, stat) {
  if (!averages) return null;
  if (stat in averages) return averages[stat];
  let total = 0;
  for (const part of stat.split("+")) {
    if (averages[part] == null) return null;
    total += averages[part];
  }
  return total;
}

function label(stat) {
  return LABELS[stat] || stat;
}

/** The market chips: the sport's usual stats that this player's log has, then any other posted prop. */
function marketsFor(card) {
  const sport = String(card.league || "").split("_")[0];
  const rows = card.games || [];
  const has = (stat) => rows.some((row) => statValue(row, stat) != null);
  const nonzero = (stat) => rows.some((row) => (statValue(row, stat) || 0) > 0);
  const propStats = (card.props || []).map((p) => p.stat).filter(Boolean);
  let list = (PREFERRED[sport] || []).filter((stat) => has(stat) && (sport === "basketball" || nonzero(stat) || propStats.includes(stat)));
  if (!list.length) list = Object.keys(card.last10 || {}).filter((k) => k !== "min" && k !== "MIN").slice(0, 5);
  for (const stat of propStats) if (!list.includes(stat)) list.push(stat);
  return list;
}

/** The posted line for a stat: an over/under line first, else the first milestone. */
function postedProp(card, stat) {
  const props = (card.props || []).filter((p) => p.stat === stat && p.line != null);
  return props.find((p) => p.kind === "over_under") || props[0] || null;
}

/** A posted line for a stat at exactly ``line`` (one with an over price first: milestones like "20+" have one). */
function propAt(card, stat, line) {
  const props = (card.props || []).filter((p) => p.stat === stat && p.line != null && Number(p.line) === Number(line));
  return props.find((p) => p.over) || props[0] || null;
}

/** With no posted line, a reference line just under the 10-game average (always x.5, so no pushes). */
function referenceLine(card, stat) {
  const avg = averageOf(card.last10, stat);
  if (avg == null) return 0.5;
  return Math.max(0.5, Math.round(avg) - 0.5);
}

function lineStep(line) {
  return line >= 60 ? 5 : line >= 12 ? 1 : 0.5;
}

function headerHtml(card) {
  const injury = card.injury;
  const status = injury?.status || (card.status && card.status !== "Active" ? card.status : "");
  const next = card.next_game;
  const nextText = next ? `${next.home ? "vs" : "@"} ${next.opponent_name || next.opponent || "TBD"} · ${timeLabel(next.date)}${next.status === "in" ? " · live" : ""}` : "";
  return `<section class="card">
    <div class="player-hero">
      ${card.headshot ? `<img class="headshot" src="${esc(card.headshot)}" alt="" width="72" height="72" loading="lazy" data-headshot>`
        : `<span class="headshot blank">${icon("user", { size: 30 })}</span>`}
      <div class="grow">
        <h1>${esc(card.name)}</h1>
        <div class="meta">${esc([card.team, card.position, `${sportEmoji(card.league)} ${leagueTitle(card.league)}`].filter(Boolean).join(" · "))}</div>
        <div class="badges">${status ? injuryBadge(status) : badge("Active", "good", "check")}${card.season_label ? badge(card.season_label, "muted") : ""}</div>
      </div>
    </div>
    ${injury?.detail ? notice(`${injury.detail}${injury.return_date ? ` Expected back around ${shortDate(injury.return_date)}.` : ""}`, "warn") : ""}
    ${next ? `<div class="next-game">${icon("today", { size: 18 })}<span class="grow"><b>Next game</b> · ${esc(nextText)}</span></div>` : '<p class="muted small" style="margin-top:10px">No upcoming game listed on ESPN.</p>'}
    ${card.note?.headline ? `<p class="small" style="margin-top:10px">${icon("news", { size: 14 })} ${esc(card.note.headline)} <span class="faint">${esc(card.note.source || "")}${card.note.published ? ` · ${esc(ago(card.note.published))}` : ""}</span></p>` : ""}
  </section>`;
}

function chartBars(card, stat) {
  const rows = (card.games || []).slice(0, 10).reverse(); // oldest on the left
  return rows.map((row) => {
    const value = statValue(row, stat);
    const minutes = row.minutes != null ? `${Math.round(row.minutes)}m` : "";
    return {
      value,
      label: `${row.home ? "" : "@"}${row.opponent || ""}`,
      sub: minutes || shortDate(row.date),
      title: `${shortDate(row.date)} ${row.home ? "vs" : "@"} ${row.opponent || ""}: ${value == null ? "–" : formatStat(value)} ${label(stat).toLowerCase()}${minutes ? `, ${minutes}in` : ""}${row.result ? ` (${row.result} ${row.score || ""})` : ""}`,
    };
  });
}

function hitRatesHtml(card, stat, line) {
  const prop = propAt(card, stat, line);
  const rows = card.games || [];
  const count = (sample) => {
    const values = sample.map((row) => statValue(row, stat)).filter((v) => v != null);
    return { hits: values.filter((v) => v > line).length, games: values.length, pushes: values.filter((v) => v === line).length };
  };
  const windows = [["Last 5", count(rows.slice(0, 5))], ["Last 10", count(rows.slice(0, 10))]];
  const server = prop?.hit_rates || null;
  if (server?.games?.season) {
    windows.push(["Season", { hits: server.hits.season, games: server.games.season, pushes: server.pushes?.season || 0 }]);
  }
  const chips = windows.map(([name, w]) => {
    if (!w.games) return `<div class="hit"><b>–</b><span>${name}: no games</span></div>`;
    const rate = w.hits / w.games;
    const kind = rate >= 0.6 ? "hot" : rate <= 0.4 ? "cold" : "";
    return `<div class="hit ${kind}"><b>${w.hits}/${w.games}</b><span>${name} over${w.pushes ? ` · ${w.pushes} push` : ""}</span></div>`;
  }).join("");
  return `<div class="hit-rates" aria-label="How often ${esc(card.name)} went over ${line}">${chips}</div>`;
}

function aiCardHtml(card, stat, line) {
  const projection = card.projection;
  const key = PROJECTION_KEY[stat] || stat;
  const stats = projection?.stats?.[key];
  const prop = propAt(card, stat, line);
  const ai = prop?.ai || null;
  const others = (card.props || []).some((p) => p.stat === stat && p.ai);
  if (!projection || !stats) {
    const avg = (name, averages, n) => {
      const value = averageOf(averages, stat);
      return value == null ? "" : `<span class="avg"><b>${formatStat(value)}</b> ${name} <span class="faint">(${n} games)</span></span>`;
    };
    const sample = card.sample || {};
    const averages = [avg("last 5", card.last5, sample.last5), avg("last 10", card.last10, sample.last10), avg("season", card.season, sample.season)].join("");
    return section("AI projection", `<p class="muted small">${esc(card.projection_note || "No projection for this player right now.")}</p>
      ${averages ? `<p class="small" style="margin:10px 0 6px">${esc(label(stat))} per game, from ESPN's game log:</p><div class="avg-row">${averages}</div>` : ""}`,
    { info: "projection", className: "ai-card" });
  }
  const over = ai ? `<div class="ai-stat"><span>Chance of over ${esc(String(line))}</span><b>${esc(chanceText(ai.over, ai.over_one_in))}</b></div>
      <div class="ai-stat"><span>Under ${esc(String(line))}</span><b>${pct(ai.under)}</b>${ai.push ? ` <span class="faint small">push ${pct(ai.push)}</span>` : ""}</div>` : "";
  const fair = ai && prop ? `<div class="ai-stat"><span>Fair odds (over) ${explain("fair_odds")}</span><b>${ai.fair_over != null ? Number(ai.fair_over).toFixed(2) : "–"}</b>
      ${prop.over ? ` <span class="small muted">book ${esc(priceText(prop.over))}</span>` : ""}</div>` : "";
  return section("AI projection", `<div class="ai-grid">
      <div class="ai-stat"><span>Expected ${esc(label(stat).toLowerCase())}</span><b>${formatStat(stats.mean)}</b></div>
      <div class="ai-stat"><span>Likely range (10–90%)</span><b>${formatStat(stats.low)}–${formatStat(stats.high)}</b></div>
      ${over}${fair}
    </div>
    ${ai ? (ai.edge_over == null ? ""
      : ai.edge_over > 0 ? `<p class="small" style="margin-top:10px">${icon("trend", { size: 14 })} The AI rates the over above the bookmaker's price: an opinion, not a guarantee.</p>`
      : `<p class="small muted" style="margin-top:10px">The price is below the AI's fair odds: on its numbers the over loses money on average.</p>`)
      : `<p class="muted small" style="margin-top:10px">${others ? `No line is posted at ${esc(String(line))}: the AI's chance is given for posted lines (tap one under “Posted lines”).` : "No posted line for this stat, so no chance of going over is given."}</p>`}
    <p class="note">${esc(projection.game)} · ${projection.minutes != null ? `${Math.round(projection.minutes)} minutes projected · ` : ""}${esc(projection.method || "")}</p>`,
  { info: "projection", className: "ai-card" });
}

function newsHtml(card) {
  const items = card.news || [];
  if (!items.length) return section("News", '<p class="muted small">No news about this player in the last 30 days.</p>');
  return section("News", `<ul class="news-list">${items.map((item) => `<li class="news-item">
      ${item.link && /^https?:/i.test(item.link) ? `<a href="${esc(item.link)}" target="_blank" rel="noopener noreferrer">${esc(item.headline)} ${icon("external", { size: 14 })}</a>` : `<b>${esc(item.headline)}</b>`}
      <div class="meta">${icon("clock", { size: 12 })}<span>${esc(item.published ? `${shortDate(item.published)} · ${ago(item.published)}` : "")}</span>${item.source ? `<span>· ${esc(item.source)}</span>` : ""}</div>
      ${item.summary ? `<p>${esc(item.summary)}</p>` : ""}
    </li>`).join("")}</ul>`);
}

function propsHtml(card) {
  const props = card.props || [];
  const info = card.props_info || {};
  if (!props.length) {
    return section("Posted lines", `<p class="muted small">${esc(info.reason || "No lines posted for this player yet.")}</p>`);
  }
  const groups = new Map();
  for (const p of props) {
    const key = p.stat || p.name || p.market;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }
  const row = (p) => {
    const what = p.kind === "milestone" ? `${p.label} (over ${p.line})` : p.kind === "yes" ? p.label || p.name : `Over/under ${p.line}`;
    const prices = p.kind === "yes" ? (p.yes ? `Yes ${priceText(p.yes)}` : "")
      : [p.over ? `Over ${priceText(p.over)}` : "", p.under ? `Under ${priceText(p.under)}` : ""].filter(Boolean).join(" · ") || (p.sides_known === false ? "line only" : "");
    const facts = [p.ai ? `AI ${pct(p.ai.over)} over` : "", p.hit_rates?.games?.last10 ? `last 10: ${p.hit_rates.last10}` : ""].filter(Boolean).join(" · ");
    const body = `<span><b>${esc(what)}</b>${p.open_line != null && p.open_line !== p.line ? ` <span class="faint small">(opened ${esc(String(p.open_line))})</span>` : ""}
        ${facts ? `<span class="small muted" style="display:block">${esc(facts)}</span>` : ""}</span>
      <span class="small num">${esc(prices)}</span>`;
    return p.stat && p.line != null
      ? `<li><button type="button" class="prop-line" data-set-line="${esc(String(p.line))}" data-set-stat="${esc(p.stat)}">${body}</button></li>`
      : `<li class="prop-line">${body}</li>`;
  };
  const group = ([key, list]) => `<h3 class="subhead">${esc(LABELS[key] || list[0].name || key)}</h3>
    <ul class="props-list">${list.slice().sort((a, b) => (a.line ?? 0) - (b.line ?? 0)).map(row).join("")}</ul>`;
  const all = [...groups.entries()];
  const rest = all.slice(3);
  // Baseball and football post dozens of markets: the first few show, the rest fold away
  return section(`Posted lines${info.book ? ` · ${info.book}` : ""}`, `${all.slice(0, 3).map(group).join("")}
  ${rest.length ? `<details class="more-lines"><summary>${rest.length} more market${rest.length === 1 ? "" : "s"} (${rest.reduce((n, [, list]) => n + list.length, 0)} lines)</summary>${rest.map(group).join("")}</details>` : ""}
  <p class="note">${info.as_of ? `As of ${esc(timeLabel(info.as_of))}${info.stale ? " (couldn't refresh: may be out of date)" : ""}. ` : ""}Player props are for reading here; the slip takes game picks.</p>`);
}

export async function render(root, params) {
  const { league, id } = params;
  setTitle("Player");
  setVerticalSwipes(false); // the chart scrolls sideways on small phones
  root.innerHTML = `<section class="card">${skeleton("header")}</section><div class="section card">${skeleton("chart")}</div>`;
  let card;
  try {
    card = await api(`/api/players/${encodeURIComponent(league)}/${encodeURIComponent(id)}`, { signal: params.signal });
  } catch (error) {
    if (isQuiet(error) || !root.isConnected) return;
    if (error.status === 404) {
      root.innerHTML = emptyState({ iconName: "user", title: "Player not found", body: error.message, actions: [{ label: "Back to Today", href: "#/today", primary: true }] });
    } else if (/18\+|age/i.test(error.message)) {
      root.innerHTML = emptyState({ iconName: "shield", title: "Confirm your age first", body: error.message, actions: [{ label: "Open settings", href: "#/settings", primary: true }] });
    } else {
      root.innerHTML = errorState(error);
    }
    return;
  }
  if (!root.isConnected) return;
  setTitle(card.name || "Player");
  const markets = marketsFor(card);
  let stat = markets.includes(params.query.get("stat")) ? params.query.get("stat") : markets[0];
  let line = null;

  const lineFor = (s) => {
    const prop = postedProp(card, s);
    return prop ? Number(prop.line) : referenceLine(card, s);
  };
  line = stat ? lineFor(stat) : null;

  const statSection = () => {
    if (!stat) return section("Last 10 games", '<p class="muted small">No game log for this player yet.</p>');
    const prop = postedProp(card, stat);
    const at = propAt(card, stat, line);
    const posted = Boolean(at);
    const book = card.props_info?.book || "Bookmaker";
    const source = at ? `${book} line${at.kind === "milestone" ? ` (${at.label})` : ""}` : prop ? `Your line (posted: ${prop.line})`
      : "Reference line: no line posted (10-game average)";
    const bars = chartBars(card, stat);
    return section(`Last ${bars.length} games · ${label(stat)}`, `
      <div class="line-control">
        <span class="small muted">${esc(source)} ${explain("line")}</span>
        <span class="stepper" role="group" aria-label="Line">
          <button type="button" data-line="-1" aria-label="Lower the line">${icon("minus", { size: 18 })}</button>
          <output aria-live="polite">${esc(String(line))}</output>
          <button type="button" data-line="1" aria-label="Raise the line">${icon("plus", { size: 18 })}</button>
        </span>
      </div>
      ${bars.length ? barChart({ bars, line, name: label(stat).toLowerCase() }) : '<p class="muted small">No games logged yet.</p>'}
      <h3 class="subhead">Hit rate ${explain("hit_rate")}</h3>
      ${hitRatesHtml(card, stat, line)}
      ${posted || !prop ? "" : `<button type="button" class="btn btn-ghost small" data-reset-line>${icon("refresh", { size: 16 })}Back to the posted line (${esc(String(prop.line))})</button>`}`,
    { id: "chart" });
  };

  const draw = () => {
    const ask = `#/ask?player=${encodeURIComponent(`${league}:${id}`)}&name=${encodeURIComponent(card.name || "")}`;
    root.innerHTML = `${headerHtml(card)}
      ${markets.length ? `<div class="chip-scroll section" role="toolbar" aria-label="Stat">${markets.map((m) => {
        const prop = postedProp(card, m);
        return chip(`${label(m)}${prop ? ` ${prop.line}` : ""}`, { active: m === stat, attrs: `data-stat="${esc(m)}"` });
      }).join("")}</div>` : ""}
      <div data-stat-section>${statSection()}</div>
      <div data-ai>${stat ? aiCardHtml(card, stat, line) : ""}</div>
      <div class="action-bar"><a class="btn btn-primary" href="${esc(ask)}">${icon("ask", { size: 18 })}Ask AI about ${esc(card.name || "this player")}</a></div>
      ${newsHtml(card)}
      ${propsHtml(card)}
      ${card.missing?.length ? `<p class="note">Missing from ESPN right now: ${esc(card.missing.join("; "))}.</p>` : ""}
      <p class="method">Game log, news and lines from ESPN (DraftKings props), as of ${esc(card.as_of ? timeLabel(card.as_of) : "now")}. Hit rates count games strictly over the line.</p>`;
  };
  const redrawStat = () => {
    root.querySelector("[data-stat-section]").innerHTML = statSection();
    root.querySelector("[data-ai]").innerHTML = stat ? aiCardHtml(card, stat, line) : "";
  };
  draw();

  const onClick = (event) => {
    const chosen = event.target.closest("[data-stat]");
    if (chosen) {
      stat = chosen.dataset.stat;
      line = lineFor(stat);
      for (const b of root.querySelectorAll("[data-stat]")) {
        const on = b.dataset.stat === stat;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", String(on));
      }
      redrawStat();
      return;
    }
    const step = event.target.closest("[data-line]");
    if (step) {
      const delta = Number(step.dataset.line) * lineStep(line);
      line = Math.max(0.5, Math.round((line + delta) * 2) / 2);
      redrawStat();
      return;
    }
    if (event.target.closest("[data-reset-line]")) {
      line = lineFor(stat);
      redrawStat();
      return;
    }
    const posted = event.target.closest("[data-set-line]");
    if (posted) {
      stat = posted.dataset.setStat;
      line = Number(posted.dataset.setLine);
      if (!markets.includes(stat)) markets.push(stat);
      for (const b of root.querySelectorAll("[data-stat]")) {
        const on = b.dataset.stat === stat;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", String(on));
      }
      redrawStat();
      root.querySelector("#chart")?.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  };
  root.addEventListener("click", onClick);
  // A missing headshot becomes a plain circle (error events don't bubble, so listen while capturing)
  root.addEventListener("error", (event) => {
    if (event.target.matches?.("[data-headshot]")) event.target.replaceWith(Object.assign(document.createElement("span"), { className: "headshot blank" }));
  }, true);
  leaving = () => root.removeEventListener("click", onClick);
}
