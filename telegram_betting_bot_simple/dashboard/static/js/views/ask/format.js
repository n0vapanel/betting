// Ask AI: turning the assistant's words and the person's own requests into display text.
//
// Replies are plain text from the server (short paragraphs and "- " lists). They are escaped first and
// only then given paragraphs, lists and bold, so nothing the model writes can inject markup. Numbers in a
// reply are the server's: they passed its check against the brain's tool results before being sent.

import { esc, money } from "../../ui.js";
import { leagueTitle } from "../../app.js";

const HELPLINES = [
  // The helplines in support messages become tap-to-call links
  [/1-800-GAMBLER/g, "tel:18004262537"],
  [/0808 ?8020 ?133/g, "tel:08088020133"],
  [/\b(?:call or text|text or call|call) 988\b/g, "tel:988"],
  [/116 ?123/g, "tel:116123"],
];

function inline(line, { links = false } = {}) {
  let html = esc(line).replace(/\*\*([^*\n]{1,200})\*\*/g, "<b>$1</b>");
  if (links) {
    for (const [pattern, href] of HELPLINES) {
      html = html.replace(pattern, (match) => {
        const number = match.replace(/^(?:call or text|text or call|call) /, "");
        return match.replace(number, `<a href="${href}">${number}</a>`);
      });
    }
  }
  return html;
}

const BULLET = /^\s*[-•*]\s+/;
// "1. ..." lines keep their numbers: a pasted slip's legs are read back by number ("drop leg 3")
const NUMBERED = /^\s*(\d{1,2})[.)]\s+/;

/**
 * A reply as safe HTML: blank lines split paragraphs, "- " lines become a list and "1. " lines a numbered
 * one (each item keeps its own number), **bold** stays bold.
 * formatReply(text, {links: true}) also turns the helpline numbers into tap-to-call links.
 */
export function formatReply(text, options = {}) {
  const blocks = String(text ?? "").replace(/\r/g, "").trim().split(/\n{2,}/);
  return blocks.map((block) => {
    const out = [];
    let list = [];
    let ordered = false;
    let para = [];
    const flushList = () => {
      if (list.length) {
        const items = list.map((item) => `<li${ordered ? ` value="${item.n}"` : ""}>${inline(item.text, options)}</li>`).join("");
        out.push(ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`);
      }
      list = [];
    };
    const flushPara = () => {
      if (para.length) out.push(`<p>${para.map((line) => inline(line, options)).join("<br>")}</p>`);
      para = [];
    };
    for (const line of block.split("\n")) {
      if (!line.trim()) continue;
      const numbered = line.match(NUMBERED);
      if (numbered || BULLET.test(line)) {
        flushPara();
        if (list.length && Boolean(numbered) !== ordered) flushList();
        ordered = Boolean(numbered);
        list.push(numbered ? { n: Number(numbered[1]), text: line.replace(NUMBERED, "") } : { text: line.replace(BULLET, "") });
      } else {
        flushList();
        para.push(line);
      }
    }
    flushPara();
    flushList();
    return out.join("");
  }).join("");
}

/** "$10" / "$12.50" from what the person typed ("10", "12.5"). */
export function amountText(value) {
  return money(value, { cents: !/^\d+(\.0{1,2})?$/.test(String(value ?? "")) });
}

/** The request bar as the thread shows it (worded like the server's own description). */
export function describeRequest(request) {
  const parts = [(request.leagues || []).map(leagueTitle).join(" + ") || "Any sport"];
  if (request.stake && request.target) parts.push(`${amountText(request.stake)} to get about ${amountText(request.target)} back`);
  else if (request.stake) parts.push(`${amountText(request.stake)} stake`);
  if (request.max_legs) parts.push(`max ${request.max_legs} legs`);
  if (request.tier) parts.push(request.tier.charAt(0).toUpperCase() + request.tier.slice(1));
  return parts.join(", ");
}

const KIND_WORDS = { single: "singles", multi: "multi", round_robin: "round robin", system: "system bet" };

/** "3-leg multi, $10 stake" for a ticket body {type, stake, legs}. */
export function describeTicket(body) {
  const legs = (body?.legs || []).length;
  const kind = legs === 1 ? "single" : KIND_WORDS[body?.type] || "multi";
  return `${legs}-leg ${kind}${body?.stake ? `, ${amountText(body.stake)} stake` : ""}`;
}

/** What a turn is doing while the server works on it (the server's own step names, shown in turn). */
export function stepsFor(body) {
  if (body.slip) {
    return ["Reading your ticket…", "Checking the latest odds…", "Analysing each leg…", "Double-checking the numbers…"];
  }
  if (body.request) {
    const sports = (body.request.leagues || []).map(leagueTitle).join("/") || "today's";
    return ["Reading your request…", `Finding ${sports} games…`, "Building tickets that fit…", "Pricing every leg…",
      "Double-checking the numbers…"];
  }
  return ["Reading your message…", "Thinking…", "Looking it up with the brain's tools…", "Double-checking the numbers…"];
}
