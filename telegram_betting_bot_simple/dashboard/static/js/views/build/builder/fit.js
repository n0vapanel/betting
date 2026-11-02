// Closest effort's words (v3): which option picked a ticket, how close it gets to what was asked and the rules the
// Builder relaxed for it, as the Builder's results and Ask AI's ticket cards show them. No DOM and no maths: every
// number is the server's (engine/builder.py ticket_fit, result.options, ticket.relaxed), only formatted here. These
// return plain text; callers escape it.
//
//   OPTION_TITLES                 {your_odds: "Likeliest at your odds", your_chance: "Closest to your chance",
//                                  closest: "Best balance"} (the engine's labels, for a server that sends none)
//   optionsFor(result, index)     the options that picked ticket ``index``: [{key, label, iconName}], first label first
//   optionOf(option)              one {key, label} as an option chip ({key, label, iconName}), or null
//   fitParts(fit)                 {got: "3.52x · 31%", asked: "3.5x · 80%", text: "3.52x · 31% (you asked 3.5x · 80%)",
//                                  onTarget} or null without a target
//   relaxedWords(relaxed)         the relaxed rules' words, once each: ["per-leg chance lowered from 60% to 55%", ...]
//   relaxedText(relaxed)          "Relaxed: per-leg chance lowered from 60% to 55%; odds window widened to ±20%." or ""

import { pct } from "../../../ui.js";

export const OPTION_TITLES = { your_odds: "Likeliest at your odds", your_chance: "Closest to your chance", closest: "Best balance" };
const OPTION_ICONS = { your_odds: "target", your_chance: "shield", closest: "scale" };
const MAX_WORDS = 160;

const clip = (text) => {
  const s = String(text ?? "").replace(/\s+/g, " ").trim();
  return s.length > MAX_WORDS ? `${s.slice(0, MAX_WORDS - 1)}…` : s;
};

/** One option ({key, label}) as a chip: the server's label, else the engine's for its key. */
export function optionOf(option) {
  if (!option || typeof option !== "object") return null;
  const key = typeof option.key === "string" ? option.key : "";
  const label = clip(option.label || OPTION_TITLES[key] || "");
  if (!label) return null;
  return { key, label, iconName: OPTION_ICONS[key] || "target" };
}

/**
 * The options that picked ticket ``index`` of a run (result.options: [{key, label, ticket_index}]), the ticket's own
 * ``option`` first when the server set one; merged options (the same ticket picked twice) keep their first label.
 */
export function optionsFor(result, index, ticket = null) {
  const out = [];
  const seen = new Set();
  const push = (option) => {
    const chip = optionOf(option);
    if (!chip || seen.has(chip.label)) return;
    seen.add(chip.label);
    out.push(chip);
  };
  if (ticket?.option) push(ticket.option);
  for (const option of Array.isArray(result?.options) ? result.options : []) {
    if (option && Number.isInteger(option.ticket_index) && option.ticket_index === index) push(option);
  }
  return out;
}

const isNum = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

/** "3.5" for 3.5, "10" for 10.00, "1,250" for 1250: the person's own target odds. */
function targetOdds(v) {
  const n = Number(v);
  return n >= 1000 ? Math.round(n).toLocaleString() : String(Number(n.toFixed(2)));
}

/** "80%" for 0.8, "75.5%" for 0.755: the person's own target chance. */
function targetChance(v) {
  const points = Number((Number(v) * 100).toFixed(1));
  return `${points}%`;
}

/**
 * How close a ticket gets (engine/builder.py ticket_fit: {odds, chance, target_odds, target_chance, odds_gap_pct,
 * chance_gap_pp, meets: {odds, chance}}): "3.52x · 31% (you asked 3.5x · 80%)". null without a target.
 */
export function fitParts(fit) {
  if (!fit || typeof fit !== "object") return null;
  const askedOdds = isNum(fit.target_odds);
  const askedChance = isNum(fit.target_chance);
  if (!askedOdds && !askedChance) return null;
  const got = [isNum(fit.odds) ? `${Number(fit.odds).toFixed(2)}x` : "", isNum(fit.chance) ? pct(fit.chance) : ""].filter(Boolean).join(" · ");
  const asked = [askedOdds ? `${targetOdds(fit.target_odds)}x` : "", askedChance ? targetChance(fit.target_chance) : ""].filter(Boolean).join(" · ");
  const meets = fit.meets && typeof fit.meets === "object" ? fit.meets : {};
  const onTarget = (!askedOdds || meets.odds === true) && (!askedChance || meets.chance === true);
  if (!got) return null;
  return { got, asked, text: `${got} (you asked ${asked})`, onTarget };
}

/** The relaxed rules' words ([{rule, from, to, words}] or plain strings), once each, in the server's order. */
export function relaxedWords(relaxed) {
  if (!Array.isArray(relaxed)) return [];
  const out = [];
  for (const entry of relaxed) {
    const words = clip(typeof entry === "string" ? entry : entry?.words).replace(/[\s.;]+$/, "");
    if (words && !out.includes(words)) out.push(words);
  }
  return out.slice(0, 8);
}

/** "Relaxed: per-leg chance lowered from 60% to 55%; odds window widened to ±20%." ("" when nothing was). */
export function relaxedText(relaxed) {
  const words = relaxedWords(relaxed);
  return words.length ? `Relaxed: ${words.join("; ")}.` : "";
}
