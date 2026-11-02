// Venue & conditions (v3 Stage 4): where the game is, the weather at the start, each team's rest and travel, and the
// two layers that never mix: "Why the line is here" (the context's attribution of the market's own numbers; it
// explains and moves nothing) and "What moves it" (an adjustment, only for a forecast or roof change after the odds
// were fetched).
//
// Data: GET /api/game's ``context`` (data/context.py): {"status": "loading"} until the background worker has built
// it (the page re-checks once after 20 s), {"status": "none"} for a sport without context, or the ready block:
//   {status: "ready", as_of, start, odds_as_of, league, sport, unit,
//    venue: {id, name, city, country, lat, lon, map, elevation_m, roof, roof_state, surface, capacity, park, approx,
//            source, tz} | null,
//    teams: {home|away: {name, rest_days, back_to_back, last: {date, opponent, venue, city}, travel_km, tz_shift_h}},
//    weather: {source, attribution: {text, url}, fetched_at, start: {t, temp_c, wind_kmh, wind_dir, gust_kmh,
//              precip_prob, precip_mm}, hours: [...], announced: {condition, temp_f, wind_text} | null,
//              effect: {total, unit} | null, at_odds: {known, fetched_at} | null, stale?} | null,
//    attribution: {margin: Projection | null, total: Projection | null},
//    adjustments: [Adjustment + {reasons: {metric, imperial}, p_over: {line, before, after} | null}],
//    badges: [code], line: {metric, imperial}}
// Figures arrive metric; units.js writes them in the person's units. Text the server writes is plain and escaped
// here.
//
//   contextOf(game)                         -> the ready block, or null (loading, none, missing)
//   panelHtml(block, game, {units, inSheet}) -> the "Venue & conditions" section
//   oneLine(block, {units})                 -> "Coors Field · 5,184 ft · 16 mph out to RF · 81°F" (plain text)
//   lineHtml(block, game, {units})          -> the phone one-liner (a button that opens the panel in a bottom sheet)
//   loadingHtml()                           -> the light placeholder while the block is on its way
//   whyHtml(block, game, {units, pick})     -> "Why the line is here" (attribution waterfalls; explains only)
//   movesSection(block, {units})            -> "What moves it" (adjustments, with raw, weight, cap and applied)
//   gameContext(game, {phone, pick})        -> {top, line, why} for the game page (see views/game.js)
//   adjustedNote(game, {units})             -> the line under the game page's prices when the totals' chances moved
//   bindContext(root, getGame)              -> the one-liner's sheet and the waterfall picker; returns unbind
//   openConditionsSheet(block, game)
//   contextRecordHtml(data)                 -> the AI record's one line (GET /api/record/context)

import { esc, section, explain, sheet, badge } from "../../ui.js";
import { icon } from "../../icons.js";
import { badgesHtml } from "./badges.js";
import { unitsFor, wind, temp, height, distance, announcedWind, announcedTemp, wordsFor, windValue, speedUnit } from "./units.js";
import { normalizeProjection, waterfallHtml, movesHtml, signed, fmt } from "./waterfall.js";
import { ensureContextStyles } from "./styles.js";

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v, n = 120) => (typeof v === "string" ? v.slice(0, n) : "");
const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
const COMPASS_WORDS = { N: "north", E: "east", S: "south", W: "west" };
export const RECHECK_MS = 20000;

const HOUR = new Intl.DateTimeFormat(undefined, { hour: "numeric" });
const DAY = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
const STAMP = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });

/** The game's ready context block, or null. */
export function contextOf(game) {
  const c = game?.context;
  return c && typeof c === "object" && c.status === "ready" ? c : null;
}

function leagueOf(block, game) {
  return str(block?.league, 64) || str(game?.league, 64);
}

/** "W" for 270°: where the wind blows from. */
function compass(deg) {
  if (num(deg) === null) return "";
  return COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
}

function compassWords(code) {
  return code.split("").map((c) => COMPASS_WORDS[c] || c).join("-");
}

/** An arrow pointing where the wind blows to (it is reported by where it comes from). */
function windArrow(deg) {
  if (num(deg) === null) return "";
  const to = (deg + 180) % 360;
  return `<span class="cx-arrow" aria-hidden="true" style="transform:rotate(${to.toFixed(0)}deg)">${icon("wind-arrow", { size: 12 })}</span>`;
}

function roofWords(venue) {
  const roof = str(venue?.roof, 20);
  const state = str(venue?.roof_state, 20);
  if (roof === "dome" || state === "dome") return "Indoors (dome)";
  if (roof === "retractable") return state === "closed" ? "Retractable: closed" : state === "open" ? "Retractable: open" : "Retractable: state not announced";
  if (roof === "open") return "Open air";
  return "";
}

const SURFACES = { grass: "Grass", turf: "Artificial turf", hardwood: "Hardwood", ice: "Ice" };

/** Plain words for the one-liner and the sheet's title: the venue, its elevation from 1,000 m, the conditions. */
export function oneLine(block, { units = "metric" } = {}) {
  if (!block) return "";
  const venue = block.venue || {};
  const parts = [];
  if (str(venue.name)) parts.push(str(venue.name));
  const elevation = num(venue.elevation_m);
  if (elevation !== null && elevation >= 1000) parts.push(height(elevation, units));
  const line = block.line && typeof block.line === "object" ? str(block.line[units], 200) : "";
  if (line) parts.push(line);
  return parts.join(" · ");
}

function factsHtml(venue, units) {
  const park = venue.park && typeof venue.park === "object" ? venue.park : null;
  const facts = [
    ["Elevation", num(venue.elevation_m) !== null ? height(venue.elevation_m, units) : ""],
    ["Roof", roofWords(venue)],
    ["Surface", SURFACES[venue.surface] || str(venue.surface, 30)],
    ["Capacity", num(venue.capacity) !== null ? `${Math.round(venue.capacity).toLocaleString("en-US")} seats` : ""],
    ["Park factors", park && (num(park.runs) !== null || num(park.hr) !== null)
      ? [num(park.runs) !== null ? `Runs ${park.runs}` : "", num(park.hr) !== null ? `HR ${park.hr}` : ""].filter(Boolean).join(" · ") : ""],
  ].filter(([, v]) => v);
  if (!facts.length) return "";
  return `<dl class="cx-facts">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}${k === "Park factors" && str(park?.source) ? `<span class="cx-src">${esc(str(park.source, 120))} (100 = average; HR shown, not added)</span>` : ""}</dd></div>`).join("")}</dl>`;
}

function mapLink(venue) {
  const lat = num(venue.lat);
  const lon = num(venue.lon);
  if (lat === null || lon === null || Math.abs(lat) > 90 || Math.abs(lon) > 180) return "";
  const given = str(venue.map, 300);
  const url = /^https:\/\/www\.openstreetmap\.org\/\?mlat=-?[0-9.]+&mlon=-?[0-9.]+#map=\d+\/-?[0-9.]+\/-?[0-9.]+$/.test(given)
    ? given : `https://www.openstreetmap.org/?mlat=${lat.toFixed(4)}&mlon=${lon.toFixed(4)}#map=16/${lat.toFixed(4)}/${lon.toFixed(4)}`;
  return `<a class="cx-map" href="${esc(url)}" target="_blank" rel="noopener noreferrer" aria-label="${esc(`${str(venue.name) || "The venue"} on OpenStreetMap (opens in a new tab)`)}">${icon("map-pin", { size: 16 })}Map</a>`;
}

function lastDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}/.test(value)) return "";
  const d = new Date(`${value.slice(0, 10)}T12:00:00Z`);
  return Number.isNaN(d.getTime()) ? "" : DAY.format(d);
}

function shiftWords(h) {
  if (num(h) === null) return "";
  const v = Math.round(Math.abs(h) * 10) / 10;
  if (v === 0) return "same time zone";
  return `${v} h ${h > 0 ? "east" : "west"}`;
}

function restWords(trip) {
  if (trip.back_to_back) return "no rest (back-to-back)";
  const d = num(trip.rest_days);
  if (d === null) return "";
  return `${d} day${d === 1 ? "" : "s"}' rest`;
}

function tripHtml(trip, side, game, units) {
  if (!trip || typeof trip !== "object") return "";
  const name = str(trip.name, 80) || str(typeof game?.[side] === "string" ? game[side] : game?.[side]?.name, 80) || (side === "home" ? "Home team" : "Away team");
  const last = trip.last && typeof trip.last === "object" ? trip.last : null;
  const where = last ? [str(last.venue, 80), str(last.city, 60)].filter(Boolean) : [];
  const lastText = last
    ? `Last played ${[lastDate(last.date), str(last.opponent, 60) ? `vs ${str(last.opponent, 60)}` : "", where.length ? `at ${where.join(", ")}` : ""].filter(Boolean).join(" ")}`
    : "No game in the last few days on record";
  const facts = [
    lastText,
    num(trip.travel_km) !== null ? `${distance(trip.travel_km, units)} travelled` : "",
    shiftWords(trip.tz_shift_h),
    restWords(trip),
  ].filter(Boolean);
  return `<li class="cx-trip">
    <div class="cx-trip-head"><b>${esc(name)}</b><span class="cx-trip-role">${side === "home" ? "Home" : "Away"}</span>${trip.back_to_back ? badge("Back-to-back", "warn") : ""}</div>
    <span class="cx-trip-facts">${esc(facts.join(" · "))}</span>
  </li>`;
}

function hourHtml(hour, units, startT) {
  const t = new Date(str(hour.t, 40));
  const label = Number.isNaN(t.getTime()) ? "" : HOUR.format(t);
  const on = startT && hour.t === startT;
  const w = num(hour.wind_kmh);
  const dir = compass(hour.wind_dir);
  const words = [
    label ? `${on ? "At the start, " : ""}${label}` : "",
    num(hour.temp_c) !== null ? temp(hour.temp_c, units) : "",
    w !== null ? `wind ${wind(w, units)}${dir ? ` from the ${compassWords(dir)}` : ""}${num(hour.gust_kmh) !== null ? `, gusts ${wind(hour.gust_kmh, units)}` : ""}` : "",
    num(hour.precip_prob) !== null ? `${Math.round(hour.precip_prob)}% chance of rain` : "",
  ].filter(Boolean).join(", ");
  return `<li class="cx-hour${on ? " on" : ""}" aria-label="${esc(words)}">
    <span class="cx-hour-t" aria-hidden="true">${esc(on ? `${label} ★` : label)}</span>
    <span class="cx-hour-temp" aria-hidden="true">${esc(num(hour.temp_c) !== null ? temp(hour.temp_c, units) : "–")}</span>
    <span class="cx-hour-wind" aria-hidden="true">${windArrow(hour.wind_dir)}${esc(w !== null ? `${windValue(w, units)}` : "–")}</span>
    <span class="cx-hour-rain" aria-hidden="true">${icon("droplet", { size: 11 })}${esc(num(hour.precip_prob) !== null ? `${Math.round(hour.precip_prob)}%` : "–")}</span>
  </li>`;
}

function atStartWords(start, units, league) {
  if (!start || typeof start !== "object") return "";
  const at = /^(americanfootball|soccer)_/.test(league) ? "At kick-off" : /^baseball_/.test(league) ? "At first pitch" : "At the start";
  const dir = compass(start.wind_dir);
  const parts = [
    num(start.temp_c) !== null ? temp(start.temp_c, units) : "",
    num(start.wind_kmh) !== null ? `wind ${wind(start.wind_kmh, units)}${dir ? ` from the ${dir}` : ""}${num(start.gust_kmh) !== null ? ` (gusts ${wind(start.gust_kmh, units)})` : ""}` : "",
    num(start.precip_prob) !== null ? `${Math.round(start.precip_prob)}% chance of rain` : "",
    num(start.precip_mm) !== null && start.precip_mm > 0 ? `${fmt(start.precip_mm, 1)} mm/h` : "",
  ].filter(Boolean);
  return parts.length ? `${at}: ${parts.join(", ")}` : "";
}

function weatherHtml(block, units) {
  const venue = block.venue || {};
  const w = block.weather && typeof block.weather === "object" ? block.weather : null;
  const league = leagueOf(block);
  if (!w) {
    const state = str(venue.roof_state, 20);
    const roof = str(venue.roof, 20);
    if (roof === "dome" || state === "dome") return `<p class="cx-note">${icon("roof", { size: 14 })} Indoors: the weather doesn't count here.</p>`;
    if (state === "closed") return `<p class="cx-note">${icon("roof", { size: 14 })} Roof closed: the weather doesn't count.</p>`;
    if (!venue.name) return "";
    return '<p class="cx-note">No forecast for this game yet: forecasts start 7 days before the start.</p>';
  }
  const start = w.start && typeof w.start === "object" ? w.start : null;
  const hours = (Array.isArray(w.hours) ? w.hours : []).filter((h) => h && typeof h === "object").slice(0, 12);
  const announced = w.announced && typeof w.announced === "object" ? w.announced : null;
  const effect = w.effect && typeof w.effect === "object" && num(w.effect.total) !== null ? w.effect : null;
  const effectUnit = str(effect?.unit, 12);
  const atOdds = w.at_odds && typeof w.at_odds === "object" ? w.at_odds : null;
  const fetched = str(w.fetched_at, 40);
  const fetchedDate = fetched ? new Date(fetched) : null;
  const attribution = w.attribution && typeof w.attribution === "object" ? w.attribution : null;
  const attrUrl = attribution && /^https:\/\/open-meteo\.com\/?/.test(str(attribution.url, 200)) ? str(attribution.url, 200) : "https://open-meteo.com/";
  const roofClosed = str(venue.roof_state, 20) === "closed";
  return `${start ? `<p class="cx-now">${esc(atStartWords(start, units, league))}</p>` : ""}
    ${roofClosed ? `<p class="cx-note">${icon("roof", { size: 14 })} Roof closed: the weather doesn't count.</p>` : ""}
    ${str(venue.roof, 20) === "retractable" && !str(venue.roof_state, 20) ? `<p class="cx-note">${icon("roof", { size: 14 })} The roof's state isn't announced, so the weather doesn't count here: the forecast is shown for reference only.</p>` : ""}
    ${hours.length ? `<ul class="cx-hours" aria-label="${esc(`Hour by hour around the start, in ${wordsFor("temp", units)} and ${wordsFor("speed", units)}`)}">${hours.map((h) => hourHtml(h, units, start?.t)).join("")}</ul>
      <p class="cx-note">★ the start · wind in ${esc(speedUnit(units))}, the arrow pointing where it blows · chance of rain</p>` : ""}
    ${announced ? `<p class="cx-announced"><b>MLB's announcement:</b> ${esc([str(announced.condition, 60), announcedTemp(announced.temp_f, units), announcedWind(announced.wind_text, units)].filter(Boolean).join(" · "))}</p>` : ""}
    ${effect && Math.abs(effect.total) >= 0.005 ? `<p class="cx-note">The weather's effect on the total now: <b>${esc(signed(effect.total, effectUnit === "pts" ? 1 : 2))} ${esc(effectUnit)}</b>. The line already holds the forecast from when the odds were fetched: only a change since then moves a number (What moves it).</p>` : ""}
    ${atOdds && atOdds.known === false ? '<p class="cx-note">The forecast at the odds\' time isn\'t known, so nothing is adjusted: the weather only explains the line.</p>' : ""}
    ${w.stale ? '<p class="cx-note">This forecast is older than usual: the weather service didn\'t answer the last check.</p>' : ""}
    <p class="cx-note">${fetchedDate && !Number.isNaN(fetchedDate.getTime()) ? `Forecast fetched at ${esc(STAMP.format(fetchedDate))}. ` : ""}${w.source === "MLB" ? "Weather as MLB announced it." : `<a class="cx-attr" href="${esc(attrUrl)}" target="_blank" rel="noopener noreferrer">${esc(str(attribution?.text, 80) || "Weather data by Open-Meteo.com")} ${icon("external", { size: 12 })}</a>`}</p>`;
}

/**
 * The "Venue & conditions" section: name, city and map link; elevation, roof and its state, surface, capacity and
 * park factors; each team's travel line; the hourly weather; MLB's announcement; the Open-Meteo attribution.
 * ``inSheet``: drawn inside the phone's bottom sheet (no section head of its own).
 */
export function panelHtml(block, game, { units = unitsFor(leagueOf(block, game)), inSheet = false } = {}) {
  if (!block) return "";
  ensureContextStyles();
  const venue = block.venue && typeof block.venue === "object" ? block.venue : {};
  const teams = block.teams && typeof block.teams === "object" ? block.teams : {};
  const league = leagueOf(block, game);
  const city = [str(venue.city, 60), str(venue.country, 40)].filter(Boolean).join(", ");
  const badges = badgesHtml(block.badges, { league, game: { home: teams.home?.name || game?.home, away: teams.away?.name || game?.away }, units, variant: "panel" });
  const where = venue.name
    ? `<div class="cx-where"><b>${esc(str(venue.name))}</b>${city ? `<span>${esc(city)}</span>` : ""}${mapLink(venue)}</div>
      ${venue.approx ? '<p class="cx-note">Approximate location: this venue isn\'t in the app\'s table yet, so it was placed by its city.</p>' : ""}`
    : '<p class="cx-note">This venue isn\'t in the app\'s table yet: rest and travel only.</p>';
  const trips = ["away", "home"].map((side) => tripHtml(teams[side], side, game, units)).filter(Boolean).join("");
  const body = `${where}
    ${badges ? `<div class="cx-panel-badges">${badges}</div>` : ""}
    ${factsHtml(venue, units)}
    ${trips ? `<h3 class="cx-sub">Rest and travel</h3><ul class="cx-trips">${trips}</ul>` : ""}
    <h3 class="cx-sub">Weather</h3>
    ${weatherHtml(block, units)}
    ${str(venue.source) ? `<p class="cx-src">Venue: ${esc(str(venue.source, 240))}</p>` : ""}`;
  const aside = `<span class="cx-units">${esc(units === "imperial" ? "Imperial units" : "Metric units")} · <a href="#/settings?focus=units" aria-label="${esc(`${units === "imperial" ? "Imperial" : "Metric"} units: change them in Settings`)}">change</a></span>`;
  // In the phone's sheet the section head is the sheet's own title: the units line goes under the panel instead
  return section("Venue & conditions", `${body}${inSheet ? aside : ""}`, { info: "explains_vs_moves", id: "conditions", className: `cx-panel${inSheet ? " in-sheet" : ""}`, aside: inSheet ? "" : aside });
}

/** The light placeholder while the block is on its way (the page checks again once, after 20 s). */
export function loadingHtml() {
  ensureContextStyles();
  return `<section class="card section cx-panel is-loading" id="conditions" aria-busy="true">
    <div class="cx-loading">${icon("stadium", { size: 18 })}<span>Venue &amp; conditions are on their way</span><span class="skel" aria-hidden="true"></span></div>
  </section>`;
}

/** The phone one-liner: a button with the venue and conditions that opens the full panel in a bottom sheet. */
export function lineHtml(block, game, { units = unitsFor(leagueOf(block, game)) } = {}) {
  ensureContextStyles();
  const text = oneLine(block, { units }) || "Venue & conditions";
  return `<button type="button" class="cx-line" data-cx-open aria-haspopup="dialog" aria-label="${esc(`Venue and conditions: ${text}. Open the details`)}">
    ${icon("stadium", { size: 18 })}<span class="cx-line-t">${esc(text)}</span>${icon("chevron", { size: 18 })}</button>`;
}

/** The full panel in a bottom sheet (the phone one-liner opens it). */
export function openConditionsSheet(block, game) {
  if (!block) return;
  const units = unitsFor(leagueOf(block, game));
  sheet.open({
    title: str(block.venue?.name) || "Venue & conditions",
    body: `<div class="cx-sheet">${panelHtml(block, game, { units, inSheet: true })}</div>`,
    size: "tall",
    label: "Venue and conditions",
  });
}

// --- "Why the line is here" and "What moves it" ---------------------------------------------------------------

function teamName(game, side, block) {
  const t = block?.teams?.[side]?.name || (typeof game?.[side] === "string" ? game[side] : game?.[side]?.name);
  return str(t, 60) || (side === "home" ? "Home" : "Away");
}

/** "Broncos by 2.5" / "Even" for a margin; "44.5 pts" for a total. */
function valueWords(p, game, block) {
  const digits = p.unit === "goals" ? 2 : 1;
  if (p.key === "margin") {
    if (Math.abs(p.value) < 0.05) return "Even";
    return `${teamName(game, p.value > 0 ? "home" : "away", block)} by ${fmt(Math.abs(p.value), digits)} ${p.unit}`;
  }
  return `${fmt(p.value, digits)} ${p.unit}`;
}

/** The attribution's projections, checked: [{key, p, title}] (margin, total). */
export function attributionsOf(block) {
  const a = block?.attribution && typeof block.attribution === "object" ? block.attribution : {};
  const out = [];
  for (const [key, title] of [["margin", "Margin"], ["total", "Total"]]) {
    const p = normalizeProjection(a[key], key, unitsFor(leagueOf(block)));
    if (p) out.push({ key, p, title });
  }
  return out;
}

/** The projection shown first: the one with more listed factors of any size (the margin on a tie). */
function defaultPick(list) {
  const score = (p) => p.factors.filter((f) => f.key !== "other" && Math.abs(f.delta) >= 0.05).length;
  let best = list[0];
  for (const item of list) if (score(item.p) > score(best.p)) best = item;
  return best?.key || null;
}

function oneWaterfall(item, game, block) {
  const sub = valueWords(item.p, game, block);
  return waterfallHtml(item.p, {
    title: `${item.title}: ${sub}`, sub: item.p.source ? `from ${item.p.source.charAt(0).toLowerCase()}${item.p.source.slice(1)}` : "",
    label: `Why the ${item.title.toLowerCase()} is ${sub}: from ${item.p.baseLabel}`,
  });
}

/**
 * "Why the line is here": the market's margin and total split into baseline, home field, altitude, rest and
 * travel, park, weather at the odds' time and the remainder. It explains and moves nothing. ``pick``: the
 * projection to show first ("margin" | "total").
 */
export function whyHtml(block, game, { pick = null } = {}) {
  const list = attributionsOf(block);
  if (!list.length) return "";
  const current = list.find((x) => x.key === pick) || list.find((x) => x.key === defaultPick(list)) || list[0];
  const picker = list.length > 1
    ? `<div class="fx-pick" role="group" aria-label="Which line">${list.map((x) => `<button type="button" class="chip${x.key === current.key ? " on" : ""}" data-wf-key="${esc(x.key)}" aria-pressed="${x.key === current.key}">${esc(x.title)}</button>`).join("")}</div>`
    : "";
  const tag = '<span class="wf-tag">explains, doesn\'t move the number</span>';
  return section("Why the line is here", `${picker}<div data-wf-box>${oneWaterfall(current, game, block)}</div>
    <p class="note">The market's own number, split into what it is made of. The remainder (“Team strength and other”) is what the listed effects leave, so the rows add up exactly. Nothing here changes a chance: ${explain("explains_vs_moves", { label: "Explains vs moves the number" })}</p>`,
  { id: "context-why", className: "cx-why", aside: tag });
}

/** "What moves it": the adjustments (a forecast or roof change after the odds were fetched), or why nothing moves. */
export function movesSection(block, { units = "metric" } = {}) {
  if (!block) return "";
  const atOdds = block.weather?.at_odds;
  const adjustments = Array.isArray(block.adjustments) ? block.adjustments : [];
  // Why nothing moves, in words that hold whoever's chances the page shows (the market's, or the NBA model's):
  // this block moves none of them
  const indoors = str(block.venue?.roof, 20) === "dome" || str(block.venue?.roof_state, 20) === "dome";
  const note = !adjustments.length && atOdds && atOdds.known === false
    ? "The forecast at the odds' time isn't known, so nothing moves: the weather only explains the line."
    : !adjustments.length && !block.weather
      ? indoors
        ? "Indoors: there's no weather to change, so nothing here moves a number."
        : "No forecast or roof change since the odds were fetched, so nothing here moves a number."
      // A retractable roof: the server moves nothing unless the roof's state is on record both then and now, so
      // "nothing has changed" may not hold (a roof announced closed since, its state at the odds' time unknown)
      : !adjustments.length && str(block.venue?.roof, 20) === "retractable"
        ? "No adjustment applies, so nothing here moves a number."
        : "";
  return section("What moves it", `${movesHtml(adjustments, { units, note })}
    <p class="note">Only news the odds couldn't hold moves a number: a forecast or roof that changed after they were fetched. Part of the change counts (weight 0.5 to start; the brain retunes it per sport only through its gate), up to a cap: design choices rather than published values. <a href="#/help/conditions">How it works</a></p>`,
  { id: "context-moves", className: "mv", aside: '<span class="wf-tag moves">moves the number</span>' });
}

/**
 * The game page's context parts: {top: the panel (wide pages) or "", line: the phone one-liner or "", why: "Why
 * the line is here" + "What moves it"}; a light placeholder while the block is loading; all "" without context.
 */
export function gameContext(game, { phone = false, pick = null } = {}) {
  const c = game?.context;
  if (!c || typeof c !== "object") return { top: "", line: "", why: "" };
  if (c.status === "loading") return phone ? { top: "", line: loadingHtml(), why: "" } : { top: loadingHtml(), line: "", why: "" };
  const block = contextOf(game);
  if (!block) return { top: "", line: "", why: "" };
  const units = unitsFor(leagueOf(block, game));
  const why = `${whyHtml(block, game, { pick })}${movesSection(block, { units })}`;
  return phone ? { top: "", line: lineHtml(block, game, { units }), why } : { top: panelHtml(block, game, { units }), line: "", why };
}

/** Under the game page's prices: the totals' chances an adjustment moved, with why and by how much. */
export function adjustedNote(game) {
  const totals = game?.markets?.totals;
  if (!totals || typeof totals !== "object") return "";
  const over = totals.over?.adjusted ? totals.over : totals.under?.adjusted ? totals.under : null;
  if (!over) return "";
  ensureContextStyles();
  const adj = over.adjusted;
  const block = contextOf(game);
  const units = unitsFor(leagueOf(block, game));
  const full = Array.isArray(block?.adjustments) ? block.adjustments.find((a) => a?.kind === "total") : null;
  const reason = str(full?.reasons?.[units] || adj.reason, 300);
  const sideName = over === totals.over ? "Over" : "Under";
  const before = num(adj.before);
  const after = num(over.ai);
  return `<p class="cx-adj-note">${badgesHtml(["adj"], { league: game.league, variant: "leg" })}<span>${esc(`${sideName} ${over.point ?? ""}: chance ${before !== null ? `${Math.round(before * 100)}%` : "–"} → ${after !== null ? `${Math.round(after * 100)}%` : "–"}. ${reason}`)}</span></p>`;
}

/** The one-liner's sheet and the waterfall picker. ``getGame()`` returns the game shown now. Returns unbind. */
export function bindContext(root, getGame, { onPick = null } = {}) {
  const click = (event) => {
    const t = event.target instanceof Element ? event.target : null;
    if (!t) return;
    if (t.closest("[data-cx-open]")) {
      const game = getGame();
      openConditionsSheet(contextOf(game), game);
      return;
    }
    const pick = t.closest("[data-wf-key]");
    if (pick) {
      const game = getGame();
      const block = contextOf(game);
      const item = attributionsOf(block).find((x) => x.key === pick.dataset.wfKey);
      const box = root.querySelector("[data-wf-box]");
      if (!item || !box) return;
      box.innerHTML = oneWaterfall(item, game, block);
      onPick?.(item.key);
      for (const b of root.querySelectorAll("[data-wf-key]")) {
        const on = b === pick;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", String(on));
      }
    }
  };
  root.addEventListener("click", click);
  return () => root.removeEventListener("click", click);
}

// --- the AI record's line ---------------------------------------------------------------------------------------

/** "Venue & conditions: 42 games logged, 30 graded · Brier 0.245 before adjustments, 0.244 after (5 adjusted)". */
export function contextRecordHtml(data) {
  if (!data || typeof data !== "object") return "";
  const games = num(data.games) ?? 0;
  const graded = num(data.graded) ?? 0;
  const b = (v) => (num(v) === null ? "–" : v.toFixed(3));
  const adjusted = num(data.adjusted_games) ?? 0;
  const parts = [`${games} game${games === 1 ? "" : "s"} with a total line logged, ${graded} graded`];
  if (graded) parts.push(`Brier ${b(data.brier_before)} for the over before any adjustment, ${b(data.brier_after)} after`);
  parts.push(adjusted ? `${adjusted} game${adjusted === 1 ? "" : "s"} adjusted` : "no game adjusted yet");
  return `<p class="small cx-record" data-ctx-record>${icon("stadium", { size: 14 })} <b>Venue &amp; conditions:</b> ${esc(parts.join(" · "))}. The adjustment's weight (0.5 to start) is retuned only when the brain's weekly check on graded games shows the adjustments were too big or too small, and the change passes the gate. <a href="#/help/conditions">How it works</a> · <a href="#/bets?tab=brain">Brain progress</a></p>`;
}

