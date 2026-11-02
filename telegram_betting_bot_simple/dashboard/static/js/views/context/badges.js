// Context badges (v3 Stage 4): the compact codes a game carries for its venue, weather, rest and travel, drawn as
// small chips with an icon, short text in the person's units, an aria-label and a title in words.
//
// The codes come from the server (engine/context.py badges(), on slate rows and in /api/game's context block), at
// most 3 per game, most important first; each is a name with an optional number after a colon (metric):
//   wind:<km/h>  open air, 24+ km/h at the start      cold:<°C>  0 °C or below, open air
//   hot:<°C>     30 °C or above, open air             rain       60%+ chance and 0.5+ mm/h
//   roof         a retractable roof announced closed  alt:<m>    1,500 m or higher
//   tz-home:<h> / tz-away:<h>   2 h+ eastward within 3 days
//   rest-home:<days> / rest-away:<days>   an unusual rest gap (NFL: a Thursday game); <days> since the last game,
//                shown as the days off between (one fewer), as the Rest and travel panel says
//   adj          an adjustment moves a number (a forecast or roof change after the odds were fetched)
// The slate's own live, neutral and b2b-home / b2b-away are not context codes (the board draws those itself).
//
//   parseCode(code)                  -> {code, kind, side, value} | null (null for anything that isn't a context code)
//   contextCodes(list)               -> the context codes in a badge list, in order, each once
//   describe(code, {league, game, units}) -> {kind, iconName, short, long, label} (label: the words for both
//                                       title and aria-label, e.g. "Wind 16 mph at kick-off")
//   badgesHtml(list, {league, game, units, max = 3, variant = "card", labels}) -> the chips' HTML ("" when none)
//       variant "cell" (the board's Context cell: short text, long from 1,180 px), "in" (under the time on a narrow
//       board), "card" (phone cards), "leg" (slip legs, builder legs), "panel" (the Venue & conditions panel);
//       ``labels``: {code: words} to say more than the code can (an adjusted leg's reason)
//   legBadges(ref, {adjusted, max = 3}) -> a slip or builder leg's chips: "adj" (with its reason) when the leg's
//       own chance moved, then its game's context codes from the cached slate (their "adj" left out: an adjustment
//       moves only the totals)
//   slateBadges(league, oddsId)      -> the context codes of a game in the cached slate ([] when unknown): the slip's
//                                       legs show their game's badges without a request
// Every text goes through esc(); a code that doesn't parse is dropped, never shown raw.

import { esc } from "../../ui.js";
import { icon } from "../../icons.js";
import { cache, SLATE_KEY } from "../../cache.js";
import { unitsFor, wind, windValue, speedUnit, temp, tempValue, height, heightShort } from "./units.js";
import { ensureContextStyles } from "./styles.js";

export const MAX_BADGES = 3;
const CODE = /^(wind|cold|hot|rain|roof|alt|tz-home|tz-away|rest-home|rest-away|adj)(?::(-?\d{1,5}(?:\.\d{1,2})?))?$/;
const NEEDS_VALUE = new Set(["wind", "cold", "hot", "alt", "tz-home", "tz-away", "rest-home", "rest-away"]);
const ICONS = {
  wind: "wind", cold: "snow", hot: "thermometer", rain: "rain", roof: "roof", alt: "mountain", tz: "plane", rest: "bed", adj: "sliders",
};

/** {code, kind, side, value} for a context code, else null. */
export function parseCode(code) {
  if (typeof code !== "string" || code.length > 24) return null;
  const m = CODE.exec(code);
  if (!m) return null;
  const name = m[1];
  const value = m[2] === undefined ? null : Number(m[2]);
  if (NEEDS_VALUE.has(name) && value === null) return null;
  const [kind, side] = name.includes("-") ? name.split("-") : [name, null];
  return { code, kind, side, value };
}

/** The context codes in a badge list (the slate's own live / neutral / b2b-* left out), in order, each once. */
export function contextCodes(list) {
  const out = [];
  for (const code of Array.isArray(list) ? list : []) if (parseCode(code) && !out.includes(code)) out.push(code);
  return out;
}

/** "kick-off" (football, soccer), "first pitch" (baseball), "the start" (anything else). */
function momentWords(league = "") {
  if (/^(americanfootball|soccer|rugby|aussierules)_/.test(league)) return "kick-off";
  if (/^baseball_/.test(league)) return "first pitch";
  return "the start";
}

function teamWords(game, side) {
  const team = game?.[side];
  const name = typeof team === "string" ? team : team?.name || team?.short || "";
  const role = side === "home" ? "Home team" : "Away team";
  return name ? `${role} (${String(name).slice(0, 60)})` : role;
}

function teamAbbr(game, side) {
  const team = game?.[side];
  if (!team || typeof team !== "object") return side === "home" ? "Home" : "Away";
  return String(team.abbr || team.short || (side === "home" ? "Home" : "Away")).slice(0, 12);
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const hoursText = (h) => (Number.isInteger(h) ? String(h) : h.toFixed(1));

/**
 * One code in words and short text: {kind, iconName, short (a tight cell), long (a card or leg), label (title and
 * aria-label)}; null for a code that doesn't parse.
 */
export function describe(code, { league = "", game = null, units = unitsFor(league) } = {}) {
  const c = parseCode(code);
  if (!c) return null;
  const at = momentWords(league);
  const v = c.value;
  switch (c.kind) {
    case "wind": {
      const shown = windValue(v, units);
      return { kind: "wind", iconName: ICONS.wind, short: String(shown), long: wind(v, units), label: `Wind ${wind(v, units)} at ${at}`, unit: speedUnit(units) };
    }
    case "cold":
      return { kind: "cold", iconName: ICONS.cold, short: `${tempValue(v, units)}°`.replace(/^-/, "−"), long: temp(v, units), label: `Freezing: ${temp(v, units)} at ${at}` };
    case "hot":
      return { kind: "hot", iconName: ICONS.hot, short: `${tempValue(v, units)}°`, long: temp(v, units), label: `Hot: ${temp(v, units)} at ${at}` };
    case "rain":
      return { kind: "rain", iconName: ICONS.rain, short: "Rain", long: "Rain", label: `Rain likely at ${at} (60% chance or more)` };
    case "roof":
      return { kind: "roof", iconName: ICONS.roof, short: "Roof", long: "Roof closed", label: "Roof closed: the weather doesn't count" };
    case "alt":
      return { kind: "alt", iconName: ICONS.alt, short: heightShort(v, units), long: height(v, units), label: `Altitude ${height(v, units)}` };
    case "tz": {
      const h = Math.abs(v);
      return {
        kind: "tz", side: c.side, iconName: ICONS.tz, short: `+${hoursText(h)}h`, long: `${teamAbbr(game, c.side)} +${hoursText(h)}h`,
        label: `${teamWords(game, c.side)} travelled ${h === 1 ? "1 time zone" : `${hoursText(h)} time zones`} east in the last 3 days`,
      };
    }
    case "rest": {
      // The code counts the days since the last game (Sunday -> Wednesday is 3); the chip shows the days off
      // between them (2), the same number as the Rest and travel panel's "2 days' rest" (rest_days)
      const d = Math.round(v);
      const off = Math.max(0, d - 1);
      return {
        kind: "rest", side: c.side, iconName: ICONS.rest, short: `${off}d`, long: `${teamAbbr(game, c.side)} ${off}d rest`,
        label: `${teamWords(game, c.side)}: ${off ? `${plural(off, "day")}' rest` : "no rest"}, ${plural(d, "day")} since its last game (a short week)`,
      };
    }
    case "adj":
      return { kind: "adj", iconName: ICONS.adj, short: "Adj", long: "Adjusted", label: "Adjusted for a forecast or roof change after the odds were fetched" };
    default:
      return null;
  }
}

/**
 * The chips for a game's badge list: its context codes only, at most ``max``. ``variant``: "cell" | "card" | "leg"
 * | "panel" (see the header). "" when there are none.
 */
export function badgesHtml(list, { league = "", game = null, units = unitsFor(league), max = MAX_BADGES, variant = "card", labels = null } = {}) {
  const codes = contextCodes(list).slice(0, Math.max(0, max));
  if (!codes.length) return "";
  ensureContextStyles();
  const chips = codes.map((code) => {
    const d = describe(code, { league, game, units });
    if (!d) return "";
    const own = labels && typeof labels[code] === "string" ? labels[code].slice(0, 300) : "";
    const label = esc(own || d.label);
    return `<span class="cx-b cx-${d.kind}" role="img" aria-label="${label}" title="${label}">${icon(d.iconName, { size: 12 })}<span class="cx-s" aria-hidden="true">${esc(d.short)}</span><span class="cx-l" aria-hidden="true">${esc(d.long)}</span></span>`;
  }).join("");
  return `<span class="cx-badges cxv-${esc(variant)}">${chips}</span>`;
}

// --- the cached slate's badges, for slip legs -----------------------------------------------------------------

const indexes = new WeakMap(); // slate value -> Map("<league>:<odds id>" -> badges)

/** The context codes of the slate's game ``<league>:<odds id>`` ([] when the slate on this device doesn't have it). */
export function slateBadges(league, oddsId) {
  if (typeof league !== "string" || oddsId === undefined || oddsId === null) return [];
  const slate = cache.peek(SLATE_KEY)?.value;
  if (!slate || typeof slate !== "object" || !Array.isArray(slate.games)) return [];
  let index = indexes.get(slate);
  if (!index) {
    index = new Map();
    for (const game of slate.games) {
      if (!game || typeof game.id !== "string") continue;
      const codes = contextCodes(game.badges);
      if (codes.length) index.set(game.id, { codes, game });
    }
    indexes.set(slate, index);
  }
  return index.get(`${league}:${oddsId}`)?.codes || [];
}

/**
 * A slip or builder leg's chips: "adj" first when the leg's own chance was adjusted (``adjusted``: {applied, reason},
 * its reason in the chip's words), then its game's context codes from the cached slate, without the game's "adj"
 * (an adjustment moves only the totals' chances). ``ref``: {league, id (the odds event id)}.
 */
export function legBadges(ref, { adjusted = null, max = MAX_BADGES, league = ref?.league || "" } = {}) {
  const codes = slateBadges(ref?.league, ref?.id).filter((code) => code !== "adj");
  const own = adjusted && typeof adjusted === "object" && adjusted.reasons && typeof adjusted.reasons === "object" ? adjusted.reasons[unitsFor(league)] : null;
  const reason = typeof own === "string" ? own : adjusted && typeof adjusted === "object" && typeof adjusted.reason === "string" ? adjusted.reason : "";
  const list = adjusted ? ["adj", ...codes] : codes;
  return badgesHtml(list, {
    league, game: slateGame(ref?.league, ref?.id), max, variant: "leg",
    labels: adjusted ? { adj: reason ? `Adjusted: ${reason}` : "Adjusted for a forecast or roof change after the odds were fetched" } : null,
  });
}

/** The slate's game for a leg (its names, for the badges' words), or null. */
export function slateGame(league, oddsId) {
  if (!slateBadges(league, oddsId).length) return null;
  return indexes.get(cache.peek(SLATE_KEY)?.value)?.get(`${league}:${oddsId}`)?.game || null;
}
