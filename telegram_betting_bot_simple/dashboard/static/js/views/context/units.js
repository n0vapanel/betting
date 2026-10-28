// Units for Venue & conditions (v3 Stage 4): the person's setting (Settings → Units: Auto · Metric · Imperial,
// PUT /api/me {units}) and the few conversions that write the server's metric figures in it. The server sends
// km/h, °C, metres and kilometres; this file only changes how they read (formatting, never a model number).
//
//   unitsSetting(profile?)          -> "auto" | "metric" | "imperial" (the profile's, "auto" when unset)
//   unitsFor(league, setting?)      -> "metric" | "imperial": "auto" is imperial in the app's US leagues
//                                      (as data/context.py's US_LEAGUES), metric elsewhere
//   wind(kmh, u) "16 mph" / "26 km/h" · windValue(kmh, u) 16 · speedUnit(u) "mph"
//   temp(c, u) "81°F" / "27°C" · tempValue(c, u) 81
//   height(m, u) "5,184 ft" / "1,580 m" · heightShort(m, u) "5.2k" / "1.6k"
//   distance(km, u) "190 mi" / "306 km"
//   wordsFor(kind, u)               -> the unit in words for screen readers ("miles an hour", "degrees Celsius")

import { me } from "../../store.js";

export const US_LEAGUES = new Set([
  "americanfootball_nfl", "americanfootball_ncaaf", "basketball_nba", "basketball_wnba", "basketball_ncaab",
  "icehockey_nhl", "baseball_mlb", "soccer_usa_mls",
]);
export const UNIT_SETTINGS = ["auto", "metric", "imperial"];

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const whole = (v) => Math.round(v).toLocaleString("en-US");
// A real minus sign for negative figures ("−2°C"), as the server writes them
const minus = (text) => text.replace(/^-/, "−");

/** The person's setting: "auto" | "metric" | "imperial". */
export function unitsSetting(profile = me()) {
  const value = profile?.units;
  return UNIT_SETTINGS.includes(value) ? value : "auto";
}

/** The units a game's figures are shown in: the setting, or for "auto" imperial in US leagues. */
export function unitsFor(league, setting = unitsSetting()) {
  if (setting === "metric" || setting === "imperial") return setting;
  return US_LEAGUES.has(league) ? "imperial" : "metric";
}

export const windValue = (kmh, u) => (num(kmh) === null ? null : Math.round(u === "imperial" ? kmh / 1.609344 : kmh));
export const speedUnit = (u) => (u === "imperial" ? "mph" : "km/h");
export function wind(kmh, u) {
  const v = windValue(kmh, u);
  return v === null ? "" : `${whole(v)} ${speedUnit(u)}`;
}

export const tempValue = (c, u) => (num(c) === null ? null : Math.round(u === "imperial" ? c * 9 / 5 + 32 : c));
export function temp(c, u) {
  const v = tempValue(c, u);
  return v === null ? "" : minus(`${v}°${u === "imperial" ? "F" : "C"}`);
}

export function height(m, u) {
  if (num(m) === null) return "";
  return u === "imperial" ? `${whole(m / 0.3048)} ft` : `${whole(m)} m`;
}

/** A height in thousands for a tight cell: "1.6k" (m) / "5.2k" (ft); under 1,000 as it is. */
export function heightShort(m, u) {
  if (num(m) === null) return "";
  const v = u === "imperial" ? m / 0.3048 : m;
  return v >= 1000 ? `${(Math.round(v / 100) / 10).toFixed(1)}k` : whole(v);
}

export function distance(km, u) {
  if (num(km) === null) return "";
  return u === "imperial" ? `${whole(km / 1.609344)} mi` : `${whole(km)} km`;
}

const WORDS = {
  speed: { metric: "kilometres an hour", imperial: "miles an hour" },
  temp: { metric: "degrees Celsius", imperial: "degrees Fahrenheit" },
  height: { metric: "metres", imperial: "feet" },
  distance: { metric: "kilometres", imperial: "miles" },
};

export function wordsFor(kind, u) {
  return WORDS[kind]?.[u === "imperial" ? "imperial" : "metric"] || "";
}

/** MLB's announced wind ("14 mph, Out To RF") in the person's units ("23 km/h, Out To RF"). */
export function announcedWind(text, u) {
  if (typeof text !== "string") return "";
  const t = text.slice(0, 80);
  if (u === "imperial") return t;
  return t.replace(/(\d+(?:\.\d+)?)\s*mph/i, (_, mph) => `${whole(Number(mph) * 1.609344)} km/h`);
}

/** MLB's announced temperature (°F) in the person's units. */
export function announcedTemp(f, u) {
  if (num(f) === null) return "";
  return u === "imperial" ? minus(`${Math.round(f)}°F`) : minus(`${Math.round((f - 32) * 5 / 9)}°C`);
}
