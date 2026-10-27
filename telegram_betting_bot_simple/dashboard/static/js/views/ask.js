// Ask AI: a chat with the app's AI analyst (POST /api/chat), plus a request bar to have tickets built.
//
// What the screen holds:
//   - the request bar ("Build me a ticket": sports, stake, want back, max legs 1–6, Safe/Risky/Crazy),
//     sent as {request}; the server builds the tickets in code and the assistant explains them;
//   - the thread: the person's messages, the AI's replies (paragraphs and lists, escaped), the cards
//     the server built from the brain's results (tickets with Add to slip / Tweak, rollover plans,
//     players), follow-up chips, the AI disclosure at the top and a typing state with step lines;
//   - the composer dock: free text, optionally with the slip (or a ticket from a card) attached;
//   - "📋 Paste a slip" (ask/paste.js): a bookmaker's slip as text or a screenshot, read by the AI
//     (POST /api/chat/slip, a chat turn) or by the free code-only reader (POST /api/pasted, no AI: also
//     the way in when this server has no assistant or the AI says it isn't set up); pasting slip-like
//     text into the composer offers the same in one tap. Its card (ask/pasted.js) shows what was read,
//     each leg's match and chances, leg toggles and a stake that re-price for free, Load into Build,
//     Fix the reading, and "Is this right?" first when the read is doubtful;
//   - the daily AI allowance meter, the list of past chats (GET /api/chat/threads) and plain states
//     for "no assistant on this server", "confirm your age", "on a break" and "allowance used up".
// Deep links: #/ask?game=<league>:<id>&name=..  #/ask?player=<league>:<id>&name=..  #/ask?slip=1
//   (attaches the slip)  #/ask?thread=<id>  #/ask?q=<text>  #/ask?new=1  #/ask?paste=1 (the paste sheet)
// From another screen: attachTicket({body, label, draft}) then navigate("#/ask") (the Builder's "Ask AI").
//
// Rules kept here: every number shown is the server's (the cards only lay them out); nothing goes on
// the slip without the person's own tap; chips never suggest staking more; a chat turn keeps running
// when the person leaves the screen, and its answer is waiting when they come back.

import { api, prefs, isQuiet } from "../api.js";
import {
  me, slip, session, loadSports, leagueTitle, sportEmoji, refreshMainButton, openSlipSheet, openOnboarding, inTelegram,
  replaceAddress, loadMe, legKey,
} from "../app.js";
import { esc, pct, ago, chip, notice, toast, sheet, confirmSheet, emptyState, skeleton, errorState } from "../ui.js";
import { icon } from "../icons.js";
import { haptic } from "../telegram.js";
import { formatReply, describeRequest, describeTicket, stepsFor } from "./ask/format.js";
import {
  loadRequest, saveRequest, checkRequest, requestHtml, requestSummary, amountChips, targetChoices, MAX_SPORTS, STAKES,
} from "./ask/request.js";
import { cardHtml, ticketLegs, ticketBody, slipState, syncCardButtons, plainStake, kindLine } from "./ask/cards.js";
import { openPasteSheet, looksLikeSlip, shotBlocked, MAX_PASTE } from "./ask/paste.js";
import {
  pastedClick, pastedChange, pastedInput, pastedToggle, syncPastedButtons, revealPasted, markAnswered, pastedText, loadPlan,
  CONFIRM_CHIP, FIX_CHIP,
} from "./ask/pasted.js";

const THREAD_KEY = "askThread";
const STEP_MS = 2400; // how long each step line shows while the server works
const SLOW_MS = 15000; // after this, say that tickets can take up to a minute
const STALE_MS = 30 * 60 * 1000; // cards older than this say their prices will be rechecked
const MAX_MESSAGE = 2000;
// Reply kinds whose words the app's code wrote, not Claude (ai/assistant.py's Turn kinds): labelled honestly
const CODE_KINDS = new Set(["support", "fix", "gate", "unread", "confirm", "free", "paused", "template", "error", "clarify"]);
// The least an AI turn needs from today's allowance (ai/assistant.py MIN_TURN_CENTS): under it the server
// makes no AI call at all (no answer, no screenshot read), so the screen says so instead of offering one
const MIN_TURN_CENTS = 5;
const DEFAULT_DISCLOSURE = "You're chatting with an AI analyst (Claude, made by Anthropic). The app's code works out "
  + "every number; the AI explains them and can still be wrong. Nothing here is a tip, and only you add bets to your slip.";
const SUGGESTIONS = ["Who does the AI favour tonight?", "Explain what the bookmaker keeps", "Why do rollovers end early?",
  "How does the AI work out its chances?"];
const PASTE_STEPS = ["Reading your slip…", "Matching your slip to our games…", "Pricing it at your odds…", "Double-checking the numbers…"];
const FREE_STEPS = ["Reading your slip (code only)…", "Matching your slip to our games…", "Pricing it at your odds…"];
const SHOWN_PASTE = 500; // characters of a pasted slip shown as the person's message (the server keeps the same)
// The free reader's own lines (no AI): what the card below holds, no numbers
const FREE_DONE = "Read by the app's free reader (code only, no AI). The card shows what it read, how each leg matched and the code's numbers at your odds.";
const FREE_DOUBT = "Read by the app's free reader (code only, no AI). Before the numbers, check what it read: the card says what looks off.";
const FREE_PAUSED = "Read by the app's free reader (code only, no AI). Picks are paused, so there are no suggestions: the card shows the code's numbers for your slip.";

// What the chat holds for this visit (kept when the person leaves the screen and comes back)
const state = {
  threadId: null,
  messages: [], // {role: "user"|"assistant", text, note?, noteIcon?, cards?, chips?, kind?, at, retry?}
  disclosure: null,
  credits: null,
  request: null, // the request bar's choices
  requestOpen: null, // null: open while the thread is empty
  draft: "",
  attach: null, // {kind: "slip"} | {kind: "ticket", body, label}
  pasteDraft: { text: "", image: null }, // the paste sheet as last left, until it is sent (screenshots stay in memory only)
  lastPaste: null, // the last paste sent ({text, image, id}: id once its card came back): "Fix the reading" and a refused paste start from it
  pasteHint: false, // slip-like text was just pasted into the composer: offer to analyse it
  busy: null, // {steps, step, started} while a turn is with the server
  loading: false, // a past chat is loading: nothing can be sent until it's on screen
  noAssistant: false,
  focus: false, // put the cursor in the composer after drawing
  freshTopic: false, // another screen started a new topic (attachTicket): don't resume the last chat
};

let view = null; // {root, stepTimer, offSlip} while the screen is showing
// Counts thread switches (open, new chat, a new topic): a chat that finishes loading after the person
// already moved on is dropped instead of replacing what they chose
let threadSwitches = 0;

// --- styles ---------------------------------------------------------------------------------------------

let stylesReady = null;

/** The screen's own stylesheet (js/views/ask/ask.css), loaded once before the first draw. */
function ensureStyles() {
  if (!stylesReady) {
    stylesReady = new Promise((resolve) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = new URL("./ask/ask.css", import.meta.url).href;
      link.addEventListener("load", resolve);
      link.addEventListener("error", resolve);
      document.head.appendChild(link);
      setTimeout(resolve, 1500);
    });
  }
  return stylesReady;
}

// --- small helpers ---------------------------------------------------------------------------------------

const nowIso = () => new Date().toISOString();
const $ = (selector) => view?.root.querySelector(selector) || null;
const finePointer = () => matchMedia("(pointer: fine)").matches;
const smooth = () => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

/** Why the chat can't be used right now: "off" (no assistant), "age", "break", or "". */
function blockedReason() {
  const profile = me();
  if (state.noAssistant || (profile.uid && profile.ai_enabled === false)) return "off";
  if (profile.uid && !profile.age_confirmed) return "age";
  if (profile.on_break) return "break";
  return "";
}

function credits() {
  return state.credits || me().credits || null;
}

/** Why a slip can't be pasted right now (the server refuses pasted slips too), or "". */
function pasteBlocked() {
  const profile = me();
  if (profile.uid && !profile.age_confirmed) return "Confirm you're 18 or older (21+ in the US) first, then paste your slip.";
  if (profile.on_break) return "Pasting a slip is off during your break.";
  if (busy()) return state.loading ? "Wait for your chat to load first." : "Wait for the AI's answer first.";
  return "";
}

/** The AI can read a paste here (else only the free reader is offered). */
function aiReads() {
  return !state.noAssistant && me().ai_enabled !== false;
}

/** Too little of today's AI allowance is left for any AI call (the server's floor, not just zero). */
function creditsOut() {
  const c = credits();
  return Boolean(c && c.daily_cents && Number(c.left_cents) < MIN_TURN_CENTS);
}

/** Why a screenshot can't be read now (words for the person), or "" when the AI can read one. */
function shotRefusal() {
  return shotBlocked({ ai: aiReads(), paused: pausedNow(), creditsOut: creditsOut() });
}

function pausedNow() {
  const until = Date.parse(me().picks_paused_until || "");
  return Number.isFinite(until) && until > Date.now();
}

/** The request bar starts open in an empty chat; a chat that is still loading keeps it closed. */
function requestOpen() {
  return state.requestOpen ?? (!state.loading && !state.messages.length && !state.busy);
}

/** A card of a message by its key "message:card". */
function findCard(key) {
  const [mi, ci] = String(key).split(":").map(Number);
  return state.messages[mi]?.cards?.[ci] || null;
}

function announce(text) {
  const live = $("[data-announce]");
  if (live) live.textContent = text;
}

// --- drawing ---------------------------------------------------------------------------------------------

function creditsHtml() {
  const c = credits();
  if (!c || !c.daily_cents || blockedReason() === "off") return "";
  const left = Math.max(0, Math.min(1, Number(c.left_cents) / Number(c.daily_cents)));
  const low = left < 0.2;
  return `<button type="button" class="ask-credits${low ? " low" : ""}" data-credits-info aria-label="${esc(`AI allowance today: ${pct(left)} left${low ? ", running low" : ""}. What is this?`)}">
    ${icon("zap", { size: 16 })}<span>AI allowance today</span>
    <span class="meter" aria-hidden="true"><span style="width:${(left * 100).toFixed(1)}%"></span></span>
    <b>${pct(left)} left</b>${low ? icon("alert", { size: 16 }) : ""}${icon("info", { size: 16 })}
  </button>`;
}

function statusHtml() {
  const reason = blockedReason();
  if (reason === "off") {
    return `<div class="ask-off">${emptyState({
      iconName: "bot",
      title: "AI chat isn't switched on here yet",
      body: "The server needs an Anthropic API key (ANTHROPIC_API_KEY) for the chat. Everything else works without it: each game's chances on Today, honest prices for any slip (paste one from your bookmaker below: the free reader checks it with code only), and tickets built by the app's own code in Build.",
      actions: [{ label: "📋 Paste a slip (free reader)", attrs: "data-paste-open", primary: true },
        { label: "Let the app build a ticket", href: "#/build" }, { label: "Browse games", href: "#/today" }],
    })}</div>`;
  }
  if (reason === "age") {
    return `${notice("Confirm you're 18 or older (21+ in the US) to chat with the AI analyst.", "warn")}
      <button type="button" class="btn btn-primary" data-confirm-age>Confirm my age</button>`;
  }
  if (reason === "break") {
    return notice("The AI chat is off during your break. You can still read your past chats.", "info");
  }
  if (creditsOut()) {
    const gone = Number(credits().left_cents) <= 0;
    return notice(`${gone ? "You've used today's AI allowance" : "What's left of today's AI allowance isn't enough for an answer"}. It resets at midnight (US Eastern); games, odds and slip pricing still work without it.`, "info");
  }
  return "";
}

function disclosureHtml() {
  return `<div class="disclosure ask-disclosure">${icon("bot", { size: 20 })}<p>${esc(state.disclosure || DEFAULT_DISCLOSURE)}</p></div>`;
}

function welcomeHtml() {
  const slipChip = slip.count
    ? chip(`Is my slip any good? (${slip.count} pick${slip.count === 1 ? "" : "s"})`, { attrs: "data-suggest-slip", iconName: "bets" })
    : "";
  return `<div class="ask-welcome">
    <h2>What would you like to know?</h2>
    <p>Ask about tonight's games, a player or your slip, or use “Build me a ticket” above.</p>
    <div class="ask-welcome-paste">
      <button type="button" class="btn btn-secondary" data-paste-open><span aria-hidden="true">📋</span>Paste a slip</button>
      <span class="small muted">From any bookmaker, as text or a screenshot: the app checks it at your odds.</span>
    </div>
    <div class="chip-row wrap">${slipChip}${SUGGESTIONS.map((s) => chip(s, { attrs: `data-chip="${esc(s)}"` })).join("")}</div>
  </div>`;
}

function staleText(at) {
  const time = new Date(at).getTime();
  if (!Number.isFinite(time) || Date.now() - time < STALE_MS) return "";
  return `Prices from ${ago(at)}: the slip rechecks them when you add these picks.`;
}

function kindExtras(m, i) {
  if (m.kind === "support" || m.kind === "paused") {
    return `<div class="ask-actions-row">
      <a class="btn btn-secondary small" href="tel:18004262537">${icon("lifebuoy", { size: 16 })}1-800-GAMBLER</a>
      <a class="btn btn-secondary small" href="tel:08088020133">${icon("lifebuoy", { size: 16 })}0808 8020 133</a>
      <a class="btn btn-secondary small" href="#/settings">${icon("shield", { size: 16 })}Budget and breaks</a></div>`;
  }
  if (m.kind === "gate" && /18 or older/.test(m.text) && !me().age_confirmed) {
    return `<div class="ask-actions-row"><button type="button" class="btn btn-primary small" data-confirm-age>Confirm my age</button></div>`;
  }
  if (m.kind === "template") {
    return `<p class="ask-meta">${icon("shield", { size: 14 })}Written by the app's code: the AI's wording didn't pass the number check.</p>`;
  }
  if (m.kind === "free") {
    return `<p class="ask-meta">${icon("shield", { size: 14 })}Read and worked out by the app's code: no AI used.</p>`;
  }
  if (m.kind === "unread" || m.kind === "fix") {
    return `<div class="ask-actions-row"><button type="button" class="btn btn-secondary small" data-paste-again><span aria-hidden="true">📋</span>Paste the slip again</button></div>`;
  }
  if (m.kind === "error" && m.retry && m.retryPath === "/api/chat/slip") {
    return `<div class="ask-actions-row"><button type="button" class="btn btn-secondary small" data-retry="${i}">${icon("refresh", { size: 16 })}Try again</button></div>`;
  }
  if (m.kind === "answer" && (/\d/.test(m.text) || m.cards?.length)) {
    return `<p class="ask-meta">${icon("check", { size: 14 })}Numbers checked against the app's own results</p>`;
  }
  return "";
}

/** A pasted card that still asks "Is this what your slip says?" (its own buttons stand in for the chips). */
function asksToConfirm(card) {
  return card?.type === "pasted" && card.pasted?.needs_confirmation && !card.pasted?.confirmed && !card.ui?.confirmed
    && !(card.ui?.analysis || card.analysis)?.paused_until;
}

/**
 * The label of a pasted card's "ask the AI" button, or "" when the AI can't take a question now: during the
 * picks pause the server answers slip questions with the pause line only (the card's own toggles and stake
 * box still answer "what if I drop leg 2?" for free), and without enough allowance it can't answer at all.
 */
function askLabel(pasted, card = null) {
  if (!aiReads() || blockedReason() || busy() || pausedNow() || creditsOut()) return "";
  const until = Date.parse((card?.ui?.analysis || card?.analysis)?.paused_until || "");
  if (Number.isFinite(until) && until > Date.now()) return "";
  if (pasted?.thread_id && Number(pasted.thread_id) === Number(state.threadId)) return "Ask about this slip";
  return pasted?.text ? "Ask the AI about it" : "";
}

/** The ids of pasted slips shown in messages after message ``index`` (an earlier "Is this right?" was answered). */
function laterPasted(index) {
  const ids = new Set();
  state.messages.slice(index + 1).forEach((m) => (m.cards || []).forEach((card) => {
    if (card?.type === "pasted" && card.pasted?.id != null) ids.add(card.pasted.id);
  }));
  return ids;
}

/** What a card in message ``index`` needs to draw itself. */
function cardOptions(index) {
  const m = state.messages[index];
  return { stale: m ? staleText(m.at) : "", later: laterPasted(index), canAsk: askLabel };
}

function messageHtml(m, i, last) {
  if (m.role === "user") {
    return `<div class="ask-msg me">
      <div class="bubble me${m.paste ? " ask-paste-bubble" : ""}">${m.thumb ? `<img class="ask-thumb" src="${esc(m.thumb)}" alt="Your screenshot (not kept)">` : ""}${esc(m.text)}</div>
      ${m.note ? `<span class="ask-note">${icon(m.noteIcon || "bets", { size: 14 })}${esc(m.note)}</span>` : ""}
    </div>`;
  }
  if (m.kind === "failed") {
    return `<div class="ask-msg ai kind-failed" data-msg="${i}">
      <div class="bubble ai">${icon("alert", { size: 18 })}<div>${esc(m.text)}</div></div>
      ${m.retry || m.freeText || m.fix ? `<div class="ask-actions-row">
        ${m.freeText ? `<button type="button" class="btn btn-primary small" data-free-read="${i}">${icon("shield", { size: 16 })}Read without AI (free)</button>` : ""}
        ${m.fix ? `<button type="button" class="btn btn-secondary small" data-paste-open><span aria-hidden="true">📋</span>Fix the paste</button>` : ""}
        ${m.retry ? `<button type="button" class="btn btn-secondary small" data-retry="${i}">${icon("refresh", { size: 16 })}Try again</button>` : ""}
      </div>` : ""}
    </div>`;
  }
  const support = m.kind === "support" || m.kind === "paused" || m.kind === "gate";
  const options = cardOptions(i);
  let chips = last && !state.busy ? (m.chips || []) : [];
  // A card asking "Is this right?" carries those two answers itself
  if ((m.cards || []).some(asksToConfirm)) chips = chips.filter((c) => c !== CONFIRM_CHIP && c !== FIX_CHIP);
  // The server's "Try again" sends the same question again when we know it; otherwise it goes as a message
  const chipAttrs = (c) => (m.retry && /^try again$/i.test(c) ? `data-retry="${i}"` : `data-chip="${esc(c)}"`);
  return `<div class="ask-msg ai kind-${esc(m.kind || "answer")}" data-msg="${i}">
    <div class="ask-who">${icon("bot", { size: 16 })}<span>${m.local || CODE_KINDS.has(m.kind) ? "App's code" : "AI analyst"}</span>${m.at ? `<time datetime="${esc(m.at)}">${esc(ago(m.at))}</time>` : ""}</div>
    <div class="bubble ai ask-text">${formatReply(m.text, { links: support })}</div>
    ${kindExtras(m, i)}
    ${(m.cards || []).map((card, ci) => cardHtml(card, `${i}:${ci}`, options)).join("")}
    ${chips.length ? `<div class="chip-row wrap ask-chips-row" aria-label="Suggested follow-ups">${chips.map((c) =>
      chip(c, { attrs: chipAttrs(c) })).join("")}</div>` : ""}
  </div>`;
}

function typingHtml() {
  const busy = state.busy;
  const steps = busy.steps.slice(0, busy.step + 1);
  const slow = Date.now() - busy.started > SLOW_MS;
  return `<div class="ask-msg ai typing" data-typing aria-busy="true">
    <div class="ask-who">${icon("bot", { size: 16 })}<span>AI analyst</span></div>
    <div class="bubble ai ask-typing">
      <span class="ask-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <ol class="ask-steps">${steps.map((s, k) => `<li class="${k === steps.length - 1 ? "now" : "done"}">${esc(s)}</li>`).join("")}</ol>
      ${slow ? '<p class="ask-slow">Tickets and analyses can take up to a minute.</p>' : ""}
    </div>
  </div>`;
}

function threadHtml() {
  if (state.loading) return `${disclosureHtml()}${skeleton("lines", 5)}`;
  const parts = [disclosureHtml()];
  if (!state.messages.length && !state.busy && !blockedReason()) parts.push(welcomeHtml());
  const lastAssistant = state.messages.map((m) => m.role).lastIndexOf("assistant");
  state.messages.forEach((m, i) => parts.push(messageHtml(m, i, i === lastAssistant && i === state.messages.length - 1)));
  if (state.busy) parts.push(typingHtml());
  return parts.join("");
}

function attachHtml() {
  if (state.attach?.kind === "slip" && slip.count) {
    return `<div class="ask-attach"><span>${icon("bets", { size: 16 })}<b>Your slip</b><span>· ${esc(describeTicket(slip.body()))}</span></span>
      <button type="button" class="icon-btn" data-detach aria-label="Don't send the slip">${icon("close", { size: 18 })}</button></div>`;
  }
  if (state.attach?.kind === "ticket") {
    return `<div class="ask-attach"><span>${icon("edit", { size: 16 })}<b>About</b><span>${esc(state.attach.label)}</span></span>
      <button type="button" class="icon-btn" data-detach aria-label="Don't send this ticket">${icon("close", { size: 18 })}</button></div>`;
  }
  if (slip.count && !blockedReason()) {
    return `<div class="chip-row">${chip(`Attach my slip (${slip.count})`, { attrs: "data-attach-slip", iconName: "plus", className: "add" })}</div>`;
  }
  return "";
}

function placeholder() {
  if (state.loading) return "Loading your chat…";
  if (state.attach) return "Ask about it, or just send…";
  if (state.messages.length) return "Ask a follow-up…";
  return "Ask about a game, player or slip…";
}

/** Nothing goes out while a turn is with the server or a past chat is still loading (it'd land in the wrong chat). */
function busy() {
  return Boolean(state.busy || state.loading);
}

function canSend() {
  return !busy() && !blockedReason() && Boolean(state.draft.trim() || state.attach);
}

/** "Looks like a bet slip: analyse it?" after slip-like text was pasted into the composer. */
function hintHtml() {
  if (!state.pasteHint || blockedReason() || !looksLikeSlip(state.draft)) return "";
  return `<div class="ask-paste-hint" role="status">
    <span aria-hidden="true">📋</span><span class="grow">Looks like a bet slip: analyse it?</span>
    <button type="button" class="btn btn-primary small" data-paste-hint${busy() ? " disabled" : ""}>Analyse it</button>
    <button type="button" class="icon-btn" data-paste-hint-close aria-label="It's not a bet slip">${icon("close", { size: 18 })}</button>
  </div>`;
}

function dockHtml() {
  if (blockedReason() === "off") return "";
  const off = Boolean(blockedReason());
  return `<div class="ask-hint-area" data-hint>${hintHtml()}</div>
    <div class="ask-attach-area" data-attach>${attachHtml()}</div>
    <form class="ask-composer" data-composer novalidate>
      <button type="button" class="btn btn-secondary ask-paste-btn" data-paste-open aria-label="Paste a slip" title="Paste a bet slip (text or a screenshot)"${off ? " disabled" : ""}><span aria-hidden="true">📋</span><span class="ask-paste-word" aria-hidden="true">Paste a slip</span></button>
      <label class="sr-only" for="ask-text">Message the AI analyst</label>
      <textarea id="ask-text" class="input" rows="1" maxlength="${MAX_MESSAGE}" placeholder="${esc(placeholder())}" enterkeyhint="${finePointer() ? "send" : "enter"}"${off ? " disabled" : ""}>${esc(state.draft)}</textarea>
      <button class="btn btn-primary ask-send" type="submit" aria-label="Send"${canSend() ? "" : " disabled"}>${icon("send", { size: 18 })}</button>
    </form>
    <p class="ask-count" data-count>${countText()}</p>`;
}

function countText() {
  const n = state.draft.length;
  return n > MAX_MESSAGE - 200 ? `${n} / ${MAX_MESSAGE}` : "";
}

function drawCredits() {
  const box = $("[data-credits]");
  if (box) box.innerHTML = creditsHtml();
}

function drawStatus() {
  const box = $("[data-status]");
  if (box) box.innerHTML = statusHtml();
}

function drawRequest() {
  const box = $("[data-request]");
  if (!box) return;
  const reason = blockedReason();
  box.innerHTML = reason === "off" ? "" : requestHtml(state.request, { open: requestOpen(), telegram: inTelegram, disabled: Boolean(reason) || busy() });
}

function drawThread() {
  const box = $("[data-thread]");
  if (box) box.innerHTML = threadHtml();
}

function drawDock() {
  const box = $("[data-dock]");
  if (!box) return;
  box.hidden = blockedReason() === "off";
  box.innerHTML = dockHtml();
  growComposer();
}

/** The dock's buttons and attachment line, without touching the text being typed. */
function updateDock() {
  const hint = $("[data-hint]");
  if (hint) hint.innerHTML = hintHtml();
  const attach = $("[data-attach]");
  if (attach) attach.innerHTML = attachHtml();
  const paste = $(".ask-paste-btn");
  if (paste) paste.disabled = Boolean(blockedReason());
  const send = $(".ask-send");
  if (send) send.disabled = !canSend();
  const text = $("#ask-text");
  if (text) {
    text.placeholder = placeholder();
    text.disabled = Boolean(blockedReason());
  }
  const count = $("[data-count]");
  if (count) count.textContent = countText();
}

function draw() {
  if (!view) return;
  view.root.innerHTML = `<div class="ask">
    <div class="page-head ask-head">
      <div class="grow"><h1>Ask AI</h1><p>The app's code works out every number; the AI explains it.</p></div>
      <div class="ask-head-actions">
        <button type="button" class="btn btn-secondary small" data-threads>${icon("chat", { size: 16 })}Chats</button>
        <button type="button" class="icon-btn" data-new-chat aria-label="New chat">${icon("plus", { size: 20 })}</button>
      </div>
    </div>
    <div data-credits>${creditsHtml()}</div>
    <div class="ask-status" data-status>${statusHtml()}</div>
    <div data-request></div>
    <div class="ask-thread" data-thread aria-label="Chat with the AI analyst">${threadHtml()}</div>
    <div class="ask-dock" data-dock></div>
    <div class="sr-only" aria-live="polite" data-announce></div>
  </div>`;
  drawRequest();
  drawDock();
  if (state.focus) {
    state.focus = false;
    const text = $("#ask-text");
    if (text && !text.disabled && finePointer()) {
      text.focus({ preventScroll: true });
      text.setSelectionRange(text.value.length, text.value.length);
    }
  }
}

function redrawAfterTurn() {
  drawCredits();
  drawStatus();
  drawRequest();
  drawThread();
  if (blockedReason() === "off") drawDock();
  else updateDock();
}

function growComposer() {
  const text = $("#ask-text");
  if (!text) return;
  if (!text.value) {
    text.style.height = ""; // one line (a long placeholder must not make it taller)
    return;
  }
  text.style.height = "auto";
  text.style.height = `${Math.min(text.scrollHeight + 2, 160)}px`;
}

// --- typing steps -----------------------------------------------------------------------------------------

function startSteps() {
  stopSteps();
  if (!view || !state.busy) return;
  view.stepTimer = setInterval(() => {
    if (!state.busy) return stopSteps();
    state.busy.step = Math.min(state.busy.step + 1, state.busy.steps.length - 1);
    const typing = $("[data-typing]");
    if (typing) typing.outerHTML = typingHtml();
  }, STEP_MS);
}

function stopSteps() {
  if (view?.stepTimer) clearInterval(view.stepTimer);
  if (view) view.stepTimer = null;
}

// --- sending ------------------------------------------------------------------------------------------------

/** The person's message goes up at once, then the typing state with ``steps``. */
function startTurn(display, steps) {
  state.messages.push({ role: "user", text: display.shown, note: display.note || "", noteIcon: display.noteIcon || "",
    thumb: display.thumb || null, paste: Boolean(display.paste), at: nowIso() });
  state.busy = { steps, step: 0, started: Date.now() };
  state.requestOpen = false;
  haptic.tap();
  if (view) {
    redrawAfterTurn();
    $("[data-typing]")?.scrollIntoView({ block: "end", behavior: smooth() });
    startSteps();
  }
  refreshMainButton();
}

function endTurn() {
  if (view) {
    redrawAfterTurn();
    const replies = view.root.querySelectorAll(".ask-msg.ai[data-msg]");
    replies[replies.length - 1]?.scrollIntoView({ block: "start", behavior: smooth() });
  }
  refreshMainButton();
}

/**
 * One turn: show the person's message at once, the typing state, then the reply (or what went wrong).
 * path: /api/chat (a message, a request or a slip) or /api/chat/slip (a pasted slip: {text?, image?}).
 */
async function send(body, display, { path = "/api/chat", steps = null } = {}) {
  if (busy() || blockedReason()) return;
  const payload = { ...body };
  if (state.threadId) payload.thread_id = state.threadId;
  startTurn(display, steps || stepsFor(body));

  let data = null;
  let failure = null;
  try {
    // No abort signal: the turn finishes on the server anyway, so its answer is kept for when they're back
    data = await api(path, { method: "POST", body: payload });
  } catch (error) {
    failure = error;
  }
  stopSteps();
  state.busy = null;
  if (data) received(data, body, path);
  else failed(failure, body, path);
  endTurn();
}

function received(data, body, path = "/api/chat") {
  if (data.thread_id) {
    state.threadId = data.thread_id;
    prefs.set(THREAD_KEY, data.thread_id);
  }
  if (data.disclosure) state.disclosure = data.disclosure;
  if (data.credits) {
    state.credits = data.credits;
    if (session.me) session.me.credits = data.credits; // Settings shows the same allowance
  }
  const reply = data.reply || {};
  const kind = reply.kind || "answer";
  state.messages.push({
    role: "assistant", text: reply.text || "", cards: Array.isArray(reply.cards) ? reply.cards : [],
    chips: Array.isArray(reply.chips) ? reply.chips : [], kind, at: nowIso(), retry: kind === "error" ? body : null, retryPath: path,
  });
  // The paste just sent is this pasted slip ("Fix the reading" on its card opens it with its screenshot)
  const read = path === "/api/chat/slip" ? (reply.cards || []).find((card) => card?.type === "pasted" && card.pasted?.id != null) : null;
  if (read && state.lastPaste && state.lastPaste.id == null) state.lastPaste.id = read.pasted.id;
  haptic.notify(kind === "error" ? "error" : ["gate", "support", "paused"].includes(kind) ? "warning" : "success");
  announce(`The AI replied: ${String(reply.text || "").slice(0, 200)}`);
  // A warning sign pauses picks on the server: the profile says until when (the paste sheet and the cards read it)
  if (kind === "support" || kind === "paused") refreshPause();
}

/** Read the profile again after a warning sign, then take "Ask about this slip" off the cards on screen. */
function refreshPause() {
  loadMe().then(() => {
    if (view) syncPastedButtons(view.root, findCard, pastedCtx.options);
  }).catch(() => null);
}

function failed(error, body, path = "/api/chat") {
  let text = error?.message || "Something went wrong. Try again in a minute.";
  const pasted = path !== "/api/chat";
  const extra = {};
  if (error?.status === 400 && /ANTHROPIC_API_KEY|isn't set up/i.test(text)) {
    state.noAssistant = true;
    text = "AI chat isn't switched on for this server yet.";
    if (pasted && body.text) {
      text += " The free reader (code only, no AI) can still read your slip's text and price it at your odds.";
      extra.freeText = body.text;
    } else if (pasted) {
      text += " Paste the slip's text instead of a screenshot and the free reader (code only) reads it.";
      extra.fix = true;
    }
  } else if (error?.status === 404 && state.threadId) {
    state.threadId = null;
    prefs.set(THREAD_KEY, null);
    text = "That chat isn't available any more. Try again to start a new one.";
  } else if (error?.signIn) {
    text = "Sign in again, then try again.";
  } else if (pasted && (error?.status === 400 || error?.status === 413 || error?.status === 422)) {
    // The paste itself was refused (too long, not a picture it can read, no bet found): fix it, don't resend it
    if (error.status === 413) text = "That screenshot is too large to send: crop it to just the slip, or paste its text.";
    extra.fix = true;
  }
  if (pasted && state.lastPaste) state.pasteDraft = { ...state.lastPaste }; // "Fix the paste" opens with what was sent
  const retryable = !state.noAssistant && !extra.fix;
  state.messages.push({ role: "assistant", kind: "failed", text, retry: retryable ? body : null, retryPath: path, at: nowIso(), ...extra });
  haptic.notify("error");
  announce(`${pasted ? "Your slip" : "The message"} didn't go through: ${text}`);
}

/** Send a failed message again (or ask again after the server's own error reply). */
function retry(index) {
  const failedMessage = state.messages[index];
  if (!failedMessage?.retry || busy()) return;
  const asked = state.messages[index - 1];
  const display = asked?.role === "user"
    ? { shown: asked.text, note: asked.note, noteIcon: asked.noteIcon, thumb: asked.thumb, paste: asked.paste } : { shown: "Try again" };
  if (failedMessage.kind === "failed") state.messages.splice(asked?.role === "user" ? index - 1 : index, asked?.role === "user" ? 2 : 1);
  const path = failedMessage.retryPath || "/api/chat";
  if (path === "/api/pasted") readFree(failedMessage.retry.text);
  else send(failedMessage.retry, display, { path, steps: path === "/api/chat/slip" ? PASTE_STEPS : null });
}

// --- pasted slips ------------------------------------------------------------------------------------------

/** "📋 My bet slip (screenshot):\n<the text, cut to SHOWN_PASTE>" as the person's message (as the server keeps it). */
function pasteShown(text, image, free = false) {
  const head = `📋 My bet slip${image ? " (screenshot)" : ""}${free ? " (free reader)" : ""}`;
  if (!text) return head;
  return `${head}:\n${text.length <= SHOWN_PASTE ? text : `${text.slice(0, SHOWN_PASTE).trimEnd()}…`}`;
}

/**
 * The paste sheet (prefilled with the last paste, or what's given). The AI reads when it can, else only the
 * free reader; a screenshot only when the AI can read one now. note: a line at the top of the sheet.
 */
function openPaste({ text, image, file = null, note = "" } = {}) {
  const reason = pasteBlocked();
  if (reason) {
    toast(reason, { kind: "warn" });
    if (me().uid && !me().age_confirmed) openOnboarding();
    return;
  }
  if (text !== undefined) state.pasteDraft = { text, image: image === undefined ? null : image };
  openPasteSheet({
    text: state.pasteDraft.text, image: state.pasteDraft.image, file, note, ai: aiReads(), creditsOut: creditsOut(), paused: pausedNow(),
    onDraft: (draft) => { state.pasteDraft = draft; },
    onAnalyse: (draft) => sendPaste(draft),
    onFree: (pastedTextValue) => readFree(pastedTextValue),
  });
}

/** A pasted slip for the AI to read (POST /api/chat/slip): a chat turn like any other. */
function sendPaste({ text = "", image = null } = {}) {
  if (pasteBlocked()) return;
  const clean = String(text || "").trim().slice(0, MAX_PASTE);
  if (!aiReads()) {
    if (clean) readFree(clean);
    return;
  }
  // The AI doesn't read screenshots in the pause or without enough allowance: the text alone goes (the
  // server's free reader reads it), and the person's message doesn't claim a screenshot was read
  const picture = image && !shotRefusal() ? image : null;
  if (!clean && !picture) return;
  const body = {};
  if (clean) body.text = clean;
  if (picture) body.image = { media_type: picture.media_type || "image/jpeg", data: picture.base64 };
  state.pasteHint = false;
  state.lastPaste = { text: clean, image: picture, id: null };
  state.pasteDraft = { text: "", image: null };
  send(body, {
    shown: pasteShown(clean, picture), note: picture ? "Screenshot · read, then not kept" : "Pasted slip", noteIcon: "bets",
    thumb: picture?.dataUrl || null, paste: true,
  }, { path: "/api/chat/slip", steps: PASTE_STEPS });
}

/** The free, code-only reader (POST /api/pasted): no AI, no allowance. Its card is shown as the app's code's. */
async function readFree(text) {
  const clean = String(text || "").trim().slice(0, MAX_PASTE);
  if (!clean || pasteBlocked()) return;
  const body = { text: clean };
  if (state.threadId) body.thread_id = state.threadId;
  state.pasteHint = false;
  state.lastPaste = { text: clean, image: null, id: null };
  const sent = state.lastPaste;
  state.pasteDraft = { text: "", image: null };
  startTurn({ shown: pasteShown(clean, null, true), note: "Free reader · code only, no AI", noteIcon: "shield", paste: true }, FREE_STEPS);
  let data = null;
  let failure = null;
  try {
    data = await api("/api/pasted", { method: "POST", body });
  } catch (error) {
    failure = error;
  }
  stopSteps();
  state.busy = null;
  if (data?.pasted) {
    const analysis = data.analysis || null;
    const doubt = data.pasted.needs_confirmation && !analysis?.paused_until;
    const kind = data.support ? "support" : analysis?.paused_until ? "paused" : "free";
    if (data.pasted.id != null && sent.id == null) sent.id = data.pasted.id;
    if (kind !== "free") refreshPause();
    const words = data.support || (analysis?.paused_until ? FREE_PAUSED : doubt ? FREE_DOUBT : FREE_DONE);
    state.messages.push({ role: "assistant", kind, text: words, local: true, at: nowIso(), chips: [],
      cards: [{ type: "pasted", source: "free_reader", pasted: data.pasted, analysis }] });
    haptic.notify(kind === "free" ? "success" : "warning");
    announce(doubt ? "The free reader read your slip: check what it read." : "The free reader read your slip: the card shows the numbers.");
  } else {
    let message = failure?.message || "Something went wrong. Try again in a minute.";
    const extra = {};
    if (failure?.status === 404 && state.threadId) {
      state.threadId = null;
      prefs.set(THREAD_KEY, null);
      message = "That chat isn't available any more. Try again to read it in a new one.";
      extra.retry = { text: clean };
    } else if (failure?.status >= 400 && failure?.status < 500 && !failure?.signIn) {
      extra.fix = true; // no bet found, too long...: the person fixes the text
      state.pasteDraft = { text: clean, image: null };
    } else if (!failure?.signIn) {
      extra.retry = { text: clean };
    }
    state.messages.push({ role: "assistant", kind: "failed", text: message, retryPath: "/api/pasted", at: nowIso(), ...extra });
    haptic.notify("error");
    announce(`The free reader couldn't read it: ${message}`);
  }
  endTurn();
}

/** "Yes, that's right" on a card: the AI's chat turn when the card is the last word in this chat, else the card shows its numbers. */
function confirmPasted(card, key) {
  const index = Number(String(key).split(":")[0]);
  const pasted = card.pasted || {};
  const inThread = pasted.thread_id && Number(pasted.thread_id) === Number(state.threadId);
  if (card.source !== "free_reader" && inThread && index === state.messages.length - 1 && aiReads() && !blockedReason() && !busy()) {
    markAnswered(card);
    send({ message: CONFIRM_CHIP }, { shown: CONFIRM_CHIP });
    return;
  }
  revealPasted(card);
  redrawCard(key);
  announce("Thanks: the card now shows the numbers for your slip.");
  view?.root.querySelector(`[data-card-key="${CSS.escape(key)}"] [data-pleg-toggle]`)?.focus({ preventScroll: true });
}

/**
 * "Fix the reading" / "Let me fix it": the sheet again with what this card was read from. The last paste
 * (with its screenshot, kept in this page only) when the card is that paste; else the card's own text; a
 * screenshot from an earlier paste isn't kept anywhere, so the sheet asks for it again.
 */
function fixPasted(card) {
  const last = state.lastPaste;
  if (!card) {
    if (last) openPaste({ text: last.text || "", image: last.image || null });
    else openPaste();
    return;
  }
  const id = card.pasted?.id ?? (card.ui?.analysis || card.analysis)?.pasted?.id;
  if (last && id != null && last.id != null && Number(last.id) === Number(id)) {
    openPaste({ text: last.text || "", image: last.image || null });
    return;
  }
  const text = pastedText(card);
  const shot = (card.pasted?.source || (card.ui?.analysis || card.analysis)?.pasted?.source) === "image";
  const note = shot ? "Screenshots aren't kept, so this slip's screenshot can't be shown again: add it again, or paste the slip's text." : "";
  openPaste({ text, image: null, note });
}

/** "Ask about this slip": the composer (the chat already holds the slip), or the AI reads it into this chat. */
function askAboutPasted(card) {
  const pasted = (card.ui?.analysis || card.analysis)?.pasted || card.pasted || {};
  const label = askLabel(pasted, card);
  if (!label) return;
  if (label === "Ask about this slip") {
    const text = $("#ask-text");
    if (text) {
      text.placeholder = "Ask about this slip, e.g. what if I drop leg 2?";
      text.focus();
      text.scrollIntoView({ block: "nearest", behavior: smooth() });
    }
    return;
  }
  sendPaste({ text: pasted.text });
}

/** "Load matched legs into Build": the person's own matched legs onto the slip, the others explained. */
async function loadIntoBuild(card) {
  const { legs, skipped } = loadPlan(card);
  const reasons = skipped.map((s) => `<li>${icon("alert", { size: 14 })}<span><b>Leg ${esc(s.index)}</b> (${esc(s.label)}): ${esc(s.why)}.</span></li>`).join("");
  if (!legs.length) {
    sheet.info({
      title: "Nothing to load yet",
      body: `<p>None of these legs can go on Build's slip:</p><ul class="warnings">${reasons}</ul>
        <p class="small muted">Build prices moneylines, spreads and totals for the games the app has odds for.</p>`,
    });
    return;
  }
  const fresh = legs.filter((leg) => !slip.has(leg.ref));
  let how = "add";
  const others = slip.legs.filter((have) => !legs.some((leg) => legKey(leg.ref) === have.key)).length;
  if (fresh.length && others) {
    how = await chooseHow(slip.count);
    if (!how) return;
  }
  if (how === "replace") slip.clear();
  let added = 0;
  for (const leg of legs) {
    if (slip.has(leg.ref)) continue;
    if (!slip.add(leg)) {
      toast("Your slip is full: remove a pick first", { kind: "warn" });
      break;
    }
    added += 1;
  }
  if (how === "replace" || !others) {
    // The slip is now just these legs: play them the way the card does, at the person's own stake
    const analysis = card.ui?.analysis || card.analysis || {};
    const type = analysis.ticket_type === "single" ? "single" : "multi";
    const stake = plainStake(analysis.stake);
    slip.set(stake ? { type, stake } : { type });
  }
  haptic.notify("success");
  const words = added ? `Loaded ${added} pick${added === 1 ? "" : "s"} into your slip` : "These picks are already in your slip";
  if (!skipped.length) {
    toast(`${words}: Build prices them at the best odds the app has`, { kind: "good" });
    return;
  }
  const s = sheet.open({
    title: words,
    body: `<p>Build prices them at the best odds the app has, which can differ from your bookmaker's.</p>
      <p class="label" style="margin-top:14px">Not loaded</p><ul class="warnings">${reasons}</ul>`,
    footer: `<button type="button" class="btn btn-secondary grow" data-stay>Stay here</button>
      <a class="btn btn-primary grow" href="#/build" data-go-build>${icon("build", { size: 18 })}Open Build</a>`,
  });
  s.foot.addEventListener("click", (event) => {
    if (event.target.closest("[data-stay]")) s.close();
    else if (event.target.closest("[data-go-build]")) {
      event.preventDefault();
      s.close().then(() => { location.hash = "#/build"; });
    }
  });
}

/** Draw one card again (after it was confirmed on the spot). */
function redrawCard(key) {
  const box = view?.root.querySelector(`[data-card-key="${CSS.escape(key)}"]`);
  const card = findCard(key);
  if (box && card) box.outerHTML = cardHtml(card, key, cardOptions(Number(String(key).split(":")[0])));
}

// What the pasted cards need from the screen (ask/pasted.js handles their own taps and typing)
const pastedCtx = {
  root: () => view?.root || null,
  findCard,
  options: (key) => cardOptions(Number(String(key).split(":")[0])),
  announce,
  confirm: confirmPasted,
  fix: fixPasted,
  ask: askAboutPasted,
  load: loadIntoBuild,
};

function sendComposer() {
  if (!canSend()) return;
  const text = state.draft.trim().slice(0, MAX_MESSAGE);
  const body = {};
  let note = "";
  let noteIcon = "";
  if (text) body.message = text;
  if (state.attach?.kind === "slip" && slip.count) {
    body.slip = slip.body();
    note = `Your slip: ${describeTicket(body.slip)}`;
    noteIcon = "bets";
  } else if (state.attach?.kind === "ticket") {
    body.slip = state.attach.body;
    note = `About: ${state.attach.label}`;
    noteIcon = "edit";
  }
  if (!body.message && !body.slip) return;
  state.draft = "";
  state.attach = null;
  state.pasteHint = false;
  const shown = text || `Analyse my slip: ${describeTicket(body.slip)}`;
  const box = $("#ask-text");
  if (box) {
    box.value = "";
    growComposer();
    if (!finePointer()) box.blur(); // let the phone's keyboard go so the answer has room
  }
  send(body, { shown, note, noteIcon });
}

function sendRequest() {
  if (busy() || blockedReason()) return;
  const checked = checkRequest(state.request);
  const error = $("[data-req-error]");
  if (checked.error) {
    if (error) error.textContent = checked.error;
    haptic.notify("warning");
    return;
  }
  if (error) error.textContent = "";
  saveRequest(state.request);
  send({ request: checked.body }, { shown: describeRequest(checked.body), note: "Build me a ticket", noteIcon: "target" });
}

// Follow-ups about "the" ticket go with that ticket when the reply showed exactly one, so the AI looks
// at the same legs and stake rather than guessing which one was meant
const TICKET_FOLLOW_UPS = ["Explain the risk", "Make it safer", "Swap the weakest leg"];

function sendChip(button) {
  const text = button.dataset.chip;
  const message = state.messages[Number(button.closest("[data-msg]")?.dataset.msg)];
  if (text === FIX_CHIP) { // fixing a read happens in the paste sheet, not in a chat turn
    const card = [...(message?.cards || [])].reverse().find((c) => c?.type === "pasted");
    fixPasted(card || null);
    return;
  }
  const tickets = (message?.cards || []).filter((card) => card?.type === "ticket" && card.ticket);
  const body = tickets.length === 1 && TICKET_FOLLOW_UPS.includes(text) ? ticketBody(tickets[0].ticket, tickets[0].refs) : null;
  if (body) {
    const label = `${tickets[0].title || describeTicket(body)} (${body.legs.length} leg${body.legs.length === 1 ? "" : "s"})`;
    send({ message: text, slip: body }, { shown: text, note: `About: ${label}`, noteIcon: "edit" });
  } else {
    send({ message: text }, { shown: text });
  }
}

function sendSlip(message) {
  if (!slip.count) {
    toast("Your slip is empty: tap prices on Today to add picks", { kind: "warn" });
    return;
  }
  const body = { slip: slip.body() };
  if (message) body.message = message;
  send(body, { shown: message || `Analyse my slip: ${describeTicket(body.slip)}`, note: `Your slip: ${describeTicket(body.slip)}`, noteIcon: "bets" });
}

// --- the slip: adding a card's ticket ---------------------------------------------------------------------

function chooseHow(count) {
  return new Promise((resolve) => {
    let answered = false;
    const s = sheet.open({
      title: "Add to your slip",
      body: `<p class="lead">Your slip already has ${count} pick${count === 1 ? "" : "s"}.</p>
        <p class="muted">Replace them with this ticket, or add its picks to them? Picks from the same market of a game swap over.</p>`,
      footer: `<button type="button" class="btn btn-secondary grow" data-how="add">Add to my picks</button>
        <button type="button" class="btn btn-primary grow" data-how="replace">Replace my slip</button>`,
      onClose: () => { if (!answered) resolve(null); },
    });
    s.foot.addEventListener("click", (event) => {
      const button = event.target.closest("[data-how]");
      if (!button) return;
      answered = true;
      resolve(button.dataset.how);
      s.close();
    });
  });
}

/** The slip's bet type and stake for a ticket (its own: the stake the person asked for, never more). */
function slipSettings(ticket) {
  const settings = { type: ticket.type || "multi" };
  const stake = plainStake(ticket.unit_stake);
  if (stake) settings.stake = stake;
  if (ticket.type === "round_robin") settings.k = ticket.k || 2;
  if (ticket.type === "system") settings.system = ticket.system || null;
  return settings;
}

/** Put a ticket's legs on the slip (the person tapped Add); a fresh slip takes the ticket's type and stake. */
async function addTicket(ticket, refs) {
  const legs = ticketLegs(ticket, refs);
  if (!legs.length) return;
  const where = slipState(legs, ticket);
  if (where === "exact") {
    openSlipSheet();
    return;
  }
  if (where === "switch") {
    const yes = await confirmSheet({
      title: "Play it this way?",
      body: `Keep your ${legs.length} picks and switch the slip to ${kindLine(ticket).toLowerCase().replace(" · ", ", ")}?`,
      confirm: "Switch",
    });
    if (yes) {
      slip.set(slipSettings(ticket));
      haptic.notify("success");
      toast("Your slip now plays it this way", { kind: "good" });
    }
    return;
  }
  let how = "replace";
  if (where === "within") {
    const others = slip.count - legs.length;
    const yes = await confirmSheet({
      title: "Use just this ticket?",
      body: `Its picks are already on your slip, with ${others} other pick${others === 1 ? "" : "s"}. Remove ${others === 1 ? "it" : "them"} and use this ticket's bet type and stake?`,
      confirm: "Use just these",
    });
    if (!yes) return;
  } else if (slip.count) {
    how = await chooseHow(slip.count);
    if (!how) return;
  }
  if (how === "replace") slip.clear();
  let added = 0;
  for (const leg of legs) {
    if (slip.has(leg.ref)) continue;
    if (!slip.add(leg)) {
      toast("Your slip is full: remove a pick first", { kind: "warn" });
      break;
    }
    added += 1;
  }
  if (how === "replace") slip.set(slipSettings(ticket));
  haptic.notify("success");
  toast(added ? `Added ${added} pick${added === 1 ? "" : "s"} to your slip` : "Your slip is now this ticket", { kind: "good" });
}

// --- tweaking a ticket ---------------------------------------------------------------------------------------

function openTweak(card) {
  const ticket = card.ticket || {};
  const body = ticketBody(ticket, card.refs);
  if (!body) return;
  const title = card.title || describeTicket(body);
  const label = `${title} (${body.legs.length} leg${body.legs.length === 1 ? "" : "s"})`;
  const weakest = body.legs.length > 1 && ticket.weakest_leg != null ? ticket.legs?.[ticket.weakest_leg]?.label : null;
  const options = [
    { key: "safer", text: "Make it safer", hint: "Likelier to win, smaller payout" },
    weakest ? { key: "swap", text: `Swap the weakest leg (${weakest})`, hint: "A different pick in its place" } : null,
    { key: "risk", text: "Explain the risk", hint: "What could go wrong, in plain words" },
    { key: "ask", text: "Ask something else about it…", hint: "Type your own question" },
    card.source === "ticket_build" ? { key: "request", text: "Change my request", hint: "Different sports, amounts or legs" } : null,
  ].filter(Boolean);
  const s = sheet.open({
    title: `Tweak “${title}”`,
    body: `<p class="muted small">The AI looks at this exact ticket again. Each answer uses some of your daily AI allowance.</p>
      <ul class="list">${options.map((o) => `<li><button type="button" class="list-item ask-tweak" data-tweak="${o.key}">
        <span class="grow"><b>${esc(o.text)}</b><span>${esc(o.hint)}</span></span>${icon("chevron", { size: 18 })}</button></li>`).join("")}</ul>`,
  });
  s.body.addEventListener("click", async (event) => {
    const choice = event.target.closest("[data-tweak]")?.dataset.tweak;
    if (!choice) return;
    await s.close();
    if (choice === "ask") {
      state.attach = { kind: "ticket", body, label };
      updateDock();
      const text = $("#ask-text");
      text?.focus();
      text?.scrollIntoView({ block: "nearest", behavior: smooth() });
      refreshMainButton();
    } else if (choice === "request") {
      state.request = { ...state.request, stake: plainStake(ticket.unit_stake) || state.request.stake, max_legs: Math.min(6, Math.max(1, body.legs.length)) };
      state.requestOpen = true;
      drawRequest();
      $("[data-request]")?.scrollIntoView({ block: "start", behavior: smooth() });
      refreshMainButton();
    } else {
      const message = choice === "safer" ? "Make it safer" : choice === "swap" ? `Swap the weakest leg (${weakest})` : "Explain the risk";
      send({ message, slip: body }, { shown: message, note: `About: ${label}`, noteIcon: "edit" });
    }
  });
}

// --- past chats ------------------------------------------------------------------------------------------------

/**
 * What a stored question sent, as a /api/chat body, so a "Try again" under its reply can send it again:
 * {request} from the request bar, {slip, message?} for a slip, else {message}.
 */
function askedBody(data) {
  if (data.paste) return null; // a pasted slip is kept cut short (and a screenshot not at all): paste it again instead
  if (data.request && typeof data.request === "object") return { request: data.request };
  const text = String(data.text || "").trim().slice(0, MAX_MESSAGE);
  if (data.slip && typeof data.slip === "object") {
    // A slip sent without words is stored with the server's own description, which isn't a message
    return /^Analyse my slip:/.test(text) || !text ? { slip: data.slip } : { slip: data.slip, message: text };
  }
  return text ? { message: text } : null;
}

function fromStored(row) {
  const content = row.content;
  const data = content && typeof content === "object" ? content : { text: String(content ?? "") };
  if (row.role === "user") {
    if (data.paste) {
      return { role: "user", text: data.text || pasteShown("", data.paste === "image"), paste: true, at: row.at, asked: null,
        note: data.paste === "image" ? "Screenshot · read, then not kept" : "Pasted slip", noteIcon: "bets" };
    }
    const note = data.slip ? `Your slip: ${describeTicket(data.slip)}` : data.request ? "Build me a ticket" : "";
    return { role: "user", text: data.text || (data.slip ? `Analyse my slip: ${describeTicket(data.slip)}` : data.request ? describeRequest(data.request) : ""),
      note, noteIcon: data.slip ? "bets" : data.request ? "target" : "", at: row.at, asked: askedBody(data) };
  }
  if (row.role === "assistant") {
    return { role: "assistant", text: data.text || "", cards: Array.isArray(data.cards) ? data.cards : [],
      chips: Array.isArray(data.chips) ? data.chips : [], kind: data.kind || "answer", at: row.at, disclosure: data.disclosure || null };
  }
  return null;
}

/** A stored chat's rows as messages; the server's error replies can send their question again. */
function storedMessages(rows) {
  const messages = (Array.isArray(rows) ? rows : []).map(fromStored).filter(Boolean);
  messages.forEach((m, i) => {
    const asked = messages[i - 1];
    if (m.kind === "error" && asked?.role === "user") m.retry = asked.asked;
  });
  return messages;
}

function resetThread() {
  threadSwitches += 1; // a chat still loading is no longer wanted
  state.loading = false;
  state.threadId = null;
  state.messages = [];
  state.disclosure = null;
  state.requestOpen = null;
  prefs.set(THREAD_KEY, null);
}

/** Open a past chat (GET /api/chat/<id>). quiet: no toast when it's gone (resuming after a reload). */
async function openThread(id, { quiet = false } = {}) {
  if (state.busy) {
    toast("Wait for the AI's answer first", { kind: "warn" });
    return;
  }
  resetThread();
  const switches = threadSwitches;
  state.loading = true; // sending waits (busy()), so nothing is asked in a chat that's about to be replaced
  if (view) {
    drawRequest();
    drawThread();
    updateDock();
  }
  refreshMainButton();
  let rows = null;
  let failure = null;
  try {
    rows = await api(`/api/chat/${encodeURIComponent(id)}`);
  } catch (error) {
    failure = error;
  }
  // Another chat, a new chat or a new topic was chosen while this one loaded: that choice stands
  if (switches !== threadSwitches) return;
  state.loading = false;
  if (failure) {
    resetThread();
    if (!quiet && !isQuiet(failure)) toast(failure.status === 404 ? "That chat isn't available any more" : failure.message, { kind: "warn" });
  } else if (!state.busy && !state.messages.length) {
    state.messages = storedMessages(rows);
    state.threadId = Number(id);
    state.disclosure = state.messages.find((m) => m.disclosure)?.disclosure || null;
    prefs.set(THREAD_KEY, state.threadId);
  }
  if (!view) return;
  drawRequest();
  drawThread();
  updateDock();
  refreshMainButton();
  const messages = view.root.querySelectorAll(".ask-msg");
  if (messages.length) messages[messages.length - 1].scrollIntoView({ block: "start" });
}

function newChat() {
  if (state.busy) {
    toast("Wait for the AI's answer first", { kind: "warn" });
    return;
  }
  resetThread();
  state.attach = null;
  if (!view) return;
  drawRequest();
  drawThread();
  updateDock();
  refreshMainButton();
  window.scrollTo({ top: 0, behavior: smooth() });
}

function threadListHtml(threads) {
  if (!threads.length) {
    return emptyState({ iconName: "chat", title: "No chats yet", body: "Your conversations with the AI show up here, including the ones you have with the Telegram bot." });
  }
  return `<ul class="list">${threads.map((t) => {
    const count = Number(t.messages) || 0;
    const current = Number(t.id) === Number(state.threadId);
    return `<li><button type="button" class="list-item ask-thread-item" data-open-thread="${esc(t.id)}"${current ? ' aria-current="true"' : ""}>
      ${icon("chat", { size: 20 })}
      <span class="grow"><b>${esc(t.title || "Untitled chat")}</b><span>${count} message${count === 1 ? "" : "s"}${t.last_at ? ` · ${esc(ago(t.last_at))}` : ""}${current ? " · open now" : ""}</span></span>
      ${icon("chevron", { size: 18 })}</button></li>`;
  }).join("")}</ul>`;
}

async function openThreads() {
  const s = sheet.open({
    title: "Your chats", size: "tall", body: skeleton("lines", 5),
    footer: `<button type="button" class="btn btn-primary block" data-sheet-new>${icon("plus", { size: 18 })}New chat</button>`,
  });
  s.panel.addEventListener("click", async (event) => {
    const item = event.target.closest("[data-open-thread]");
    if (item) {
      await s.close();
      openThread(item.dataset.openThread);
    } else if (event.target.closest("[data-sheet-new]")) {
      await s.close();
      newChat();
    }
  });
  try {
    const threads = await api("/api/chat/threads");
    if (s.open) s.body.innerHTML = threadListHtml(Array.isArray(threads) ? threads : []);
  } catch (error) {
    if (s.open) s.body.innerHTML = isQuiet(error) ? "" : errorState(error, { retry: false, title: "Couldn't load your chats" });
  }
}

// --- sheets: allowance, sports for the request --------------------------------------------------------------

function openCreditsInfo() {
  const c = credits();
  const left = c && c.daily_cents ? Math.max(0, Math.min(1, Number(c.left_cents) / Number(c.daily_cents))) : null;
  sheet.open({
    title: "Your daily AI allowance",
    body: `<p class="lead">${left != null ? `${esc(pct(left))} of today's allowance is left.` : "Each answer uses a little of a daily allowance."}</p>
      <p>Every answer costs the server a little on Anthropic's API, so each person gets a daily allowance. Longer jobs (building or analysing tickets) use more than a quick question.</p>
      <p>It resets at midnight (US Eastern). Games, odds, slip pricing and the code-only analysis in Build don't use it.</p>`,
  });
}

async function openRequestSports() {
  let sports = [];
  try {
    sports = await loadSports();
  } catch (error) {
    if (!isQuiet(error)) toast(error.message, { kind: "error" });
    return;
  }
  const chosen = new Set(state.request.leagues);
  const groups = [];
  for (const s of sports) {
    const label = s.label || s.group || "Other";
    let group = groups.find((g) => g.label === label);
    if (!group) groups.push(group = { label, leagues: [] });
    group.leagues.push(s);
  }
  const body = () => groups.map((g) => `<div class="field"><span class="label">${esc(g.label)}</span><div class="chip-row wrap">${g.leagues.map((s) =>
    chip(`${sportEmoji(s.key)} ${s.title}`, { active: chosen.has(s.key), attrs: `data-pick="${esc(s.key)}"` })).join("")}</div></div>`).join("")
    + `<p class="hint">Up to ${MAX_SPORTS} sports per request: each one's odds use some of the odds feed's monthly requests.</p>`;
  const s = sheet.open({
    title: "Sports for this request", size: "tall", body: body(),
    footer: '<button type="button" class="btn btn-primary block" data-picked>Done</button>',
  });
  s.panel.addEventListener("click", async (event) => {
    const pick = event.target.closest("[data-pick]");
    if (pick) {
      const key = pick.dataset.pick;
      if (chosen.has(key)) chosen.delete(key);
      else if (chosen.size < MAX_SPORTS) chosen.add(key);
      else toast(`Pick up to ${MAX_SPORTS} sports`, { kind: "warn" });
      s.body.innerHTML = body();
      return;
    }
    if (event.target.closest("[data-picked]")) {
      state.request = { ...state.request, leagues: [...chosen] };
      saveRequest(state.request);
      await s.close();
      drawRequest();
    }
  });
}

// --- the request bar's controls --------------------------------------------------------------------------------

function updateRequestBits(changed) {
  const summary = $(".ask-req-summary");
  if (summary) summary.textContent = requestSummary(state.request);
  if (changed === "stake" || changed === "all") {
    const box = $('[data-req-chips="stake"]');
    if (box) box.innerHTML = amountChips("stake", STAKES, state.request.stake);
    const targets = $('[data-req-chips="target"]');
    if (targets) targets.innerHTML = amountChips("target", targetChoices(state.request.stake), state.request.target);
  }
  if (changed === "target") {
    const targets = $('[data-req-chips="target"]');
    if (targets) targets.innerHTML = amountChips("target", targetChoices(state.request.stake), state.request.target);
  }
  const error = $("[data-req-error]");
  if (error) error.textContent = "";
  saveRequest(state.request);
}

function onRequestClick(event) {
  const target = event.target;
  if (target.closest("[data-req-toggle]")) {
    state.requestOpen = !requestOpen();
    drawRequest();
    refreshMainButton();
    return true;
  }
  const sport = target.closest("[data-req-sport]");
  if (sport) {
    const key = sport.dataset.reqSport;
    const leagues = state.request.leagues.includes(key) ? state.request.leagues.filter((k) => k !== key) : [...state.request.leagues, key];
    if (leagues.length > MAX_SPORTS) {
      toast(`Pick up to ${MAX_SPORTS} sports`, { kind: "warn" });
      return true;
    }
    state.request = { ...state.request, leagues };
    sport.classList.toggle("on", leagues.includes(key));
    sport.setAttribute("aria-pressed", String(leagues.includes(key)));
    updateRequestBits("sports");
    return true;
  }
  if (target.closest("[data-req-more]")) {
    openRequestSports();
    return true;
  }
  for (const name of ["stake", "target"]) {
    const choice = target.closest(`[data-req-${name}]`);
    if (choice) {
      state.request = { ...state.request, [name]: choice.getAttribute(`data-req-${name}`) };
      const input = $(`[data-req-input="${name}"]`);
      if (input) input.value = "";
      updateRequestBits(name);
      return true;
    }
  }
  const legs = target.closest("[data-req-legs]");
  if (legs) {
    const next = Math.min(6, Math.max(1, state.request.max_legs + Number(legs.dataset.reqLegs)));
    state.request = { ...state.request, max_legs: next };
    const stepper = legs.closest(".stepper");
    stepper.querySelector("output").textContent = String(next);
    stepper.querySelector('[data-req-legs="-1"]').disabled = next <= 1;
    stepper.querySelector('[data-req-legs="1"]').disabled = next >= 6;
    updateRequestBits("legs");
    return true;
  }
  const tier = target.closest('[data-seg="req-tier"]');
  if (tier) {
    state.request = { ...state.request, tier: tier.dataset.value || null };
    for (const b of view.root.querySelectorAll('[data-seg="req-tier"]')) b.setAttribute("aria-pressed", String(b === tier));
    updateRequestBits("tier");
    return true;
  }
  if (target.closest("[data-req-send]")) {
    sendRequest();
    return true;
  }
  return false;
}

function onRequestInput(input) {
  const name = input.dataset.reqInput;
  const value = input.value.trim().replace(/^\$/, "").replace(/,/g, "");
  if (!value) return;
  state.request = { ...state.request, [name]: value };
  updateRequestBits(name === "stake" ? "stake" : "target");
}

// --- events ------------------------------------------------------------------------------------------------------

function onClick(event) {
  const target = event.target;
  if (onRequestClick(event)) return;
  if (pastedClick(event, pastedCtx)) return;
  const simple = [["[data-threads]", openThreads], ["[data-new-chat]", newChat], ["[data-credits-info]", openCreditsInfo],
    ["[data-confirm-age]", openOnboarding], ["[data-open-slip]", openSlipSheet], ["[data-paste-open]", () => openPaste()], ["[data-paste-again]", () => fixPasted(null)],
    ["[data-paste-hint]", analyseDraft], ["[data-paste-hint-close]", dismissHint]];
  for (const [selector, action] of simple) {
    if (target.closest(selector)) {
      action();
      return;
    }
  }
  const chipButton = target.closest("[data-chip]");
  if (chipButton) {
    sendChip(chipButton);
    return;
  }
  if (target.closest("[data-suggest-slip]")) {
    sendSlip("Is my slip any good?");
    return;
  }
  const again = target.closest("[data-retry]");
  if (again) {
    retry(Number(again.dataset.retry));
    return;
  }
  const free = target.closest("[data-free-read]");
  if (free) {
    const failedMessage = state.messages[Number(free.dataset.freeRead)];
    if (failedMessage?.freeText && !busy()) readFree(failedMessage.freeText);
    return;
  }
  if (target.closest("[data-attach-slip]")) {
    state.attach = { kind: "slip" };
    updateDock();
    refreshMainButton();
    return;
  }
  if (target.closest("[data-detach]")) {
    state.attach = null;
    updateDock();
    refreshMainButton();
    return;
  }
  const add = target.closest("[data-card-add]");
  if (add) {
    const card = findCard(add.dataset.cardAdd);
    if (card?.ticket) addTicket(card.ticket, card.refs);
    return;
  }
  const altAdd = target.closest("[data-alt-add]");
  if (altAdd) {
    const [mi, ci, ai] = altAdd.dataset.altAdd.split(":");
    const alt = findCard(`${mi}:${ci}`)?.alternatives?.[Number(ai)];
    if (alt?.ticket) addTicket(alt.ticket, alt.refs);
    return;
  }
  const tweak = target.closest("[data-card-tweak]");
  if (tweak) {
    const card = findCard(tweak.dataset.cardTweak);
    if (card?.ticket) openTweak(card);
  }
}

function onInput(event) {
  const target = event.target;
  if (target.id === "ask-text") {
    const had = Boolean(state.draft.trim());
    const hinted = Boolean(hintHtml());
    state.draft = target.value;
    if (state.pasteHint && !looksLikeSlip(state.draft)) state.pasteHint = false;
    growComposer();
    const button = $(".ask-send");
    if (button) button.disabled = !canSend();
    const count = $("[data-count]");
    if (count) count.textContent = countText();
    if (hinted !== Boolean(hintHtml())) {
      const hint = $("[data-hint]");
      if (hint) hint.innerHTML = hintHtml();
    }
    if (had !== Boolean(state.draft.trim())) refreshMainButton();
    return;
  }
  if (pastedInput(event, pastedCtx)) return;
  if (target.matches("[data-req-input]")) onRequestInput(target);
}

function onChange(event) {
  pastedChange(event, pastedCtx);
}

/** A <details> opened or closed (toggle doesn't bubble: caught on the way down). */
function onToggle(event) {
  pastedToggle(event, pastedCtx);
}

/**
 * Something pasted into the composer: a picture opens the paste sheet with it; text that looks like a bet
 * slip offers "Analyse it" (and goes to the sheet when it's longer than a chat message can be).
 */
function onPaste(event) {
  if (event.target.id !== "ask-text" || blockedReason()) return;
  const data = event.clipboardData;
  if (!data) return;
  const picture = [...(data.files || [])].find((f) => /^image\//.test(f.type));
  if (picture) {
    event.preventDefault();
    const refusal = shotRefusal();
    if (!refusal) openPaste({ text: "", image: null, file: picture });
    else toast(refusal, { kind: "warn" });
    return;
  }
  const text = data.getData("text/plain") || "";
  if (!looksLikeSlip(text)) return;
  const box = event.target;
  const room = MAX_MESSAGE - (box.value.length - Math.abs((box.selectionEnd ?? 0) - (box.selectionStart ?? 0)));
  if (text.length > room) {
    event.preventDefault();
    openPaste({ text: text.slice(0, MAX_PASTE), image: null });
    return;
  }
  state.pasteHint = true; // drawn by the input event that follows the paste
}

/** "Analyse it" on the composer's hint: the composer's text goes as a pasted slip. */
function analyseDraft() {
  const text = state.draft.trim();
  if (!text || busy()) return;
  state.draft = "";
  state.pasteHint = false;
  const box = $("#ask-text");
  if (box) {
    box.value = "";
    growComposer();
    if (!finePointer()) box.blur();
  }
  sendPaste({ text });
}

function dismissHint() {
  state.pasteHint = false;
  const hint = $("[data-hint]");
  if (hint) hint.innerHTML = "";
  $("#ask-text")?.focus({ preventScroll: true });
}

function onKeydown(event) {
  // Enter sends on a keyboard (Shift+Enter for a new line); phones keep Enter for new lines
  if (event.target.id === "ask-text" && event.key === "Enter" && !event.shiftKey && !event.isComposing && finePointer()) {
    event.preventDefault();
    sendComposer();
  }
}

function onSubmit(event) {
  if (event.target.matches("[data-composer]")) {
    event.preventDefault();
    sendComposer();
  }
}

function onSlipChange() {
  if (!view) return;
  if (state.attach?.kind === "slip" && !slip.count) state.attach = null;
  syncCardButtons(view.root, findCard);
  syncPastedButtons(view.root, findCard, pastedCtx.options);
  updateDock();
  if (!state.messages.length && !state.busy) drawThread(); // the "Is my slip any good?" suggestion
  refreshMainButton();
}

// --- deep links ----------------------------------------------------------------------------------------------------

/**
 * Apply #/ask?... once, then tidy the address so a later visit doesn't apply it again.
 * Returns {open: a thread id to open, fresh: true when the link starts a new topic, paste: open the paste sheet}.
 */
function applyQuery(query) {
  if (!query || ![...query.keys()].length) return { open: null, fresh: false, paste: false };
  let open = null;
  let fresh = false;
  const paste = query.get("paste") === "1";
  const thread = query.get("thread");
  const game = query.get("game");
  const player = query.get("player");
  const name = (query.get("name") || "").slice(0, 80);
  if (thread && /^\d{1,9}$/.test(thread)) {
    open = Number(thread);
  } else if (game || player) {
    const league = String(game || player).split(":")[0];
    const sport = /^[a-z0-9_]{2,64}$/.test(league) ? leagueTitle(league) : "";
    if (!state.busy && (state.messages.length || state.loading)) resetThread(); // a new topic gets its own chat
    fresh = true;
    state.draft = game
      ? `${name || "This game"}${sport ? ` (${sport})` : ""}: who does the AI favour, and is there any value in the prices?`
      : `What should I know about ${name || "this player"}${sport ? ` (${sport})` : ""} before the next game?`;
    state.focus = true;
  } else if (query.get("q")) {
    state.draft = query.get("q").slice(0, MAX_MESSAGE);
    state.focus = true;
  }
  if (query.get("new") === "1" && !state.busy && !game && !player) {
    resetThread();
    fresh = true;
  }
  if (query.get("slip") === "1") {
    if (slip.count) state.attach = { kind: "slip" };
    else toast("Your slip is empty: tap prices on Today to add picks", { kind: "warn" });
    state.focus = true;
  }
  try {
    if (location.hash !== "#/ask") replaceAddress("#/ask"); // through the shell: Back still finds this screen
  } catch {
    /* the address keeps its query; nothing else depends on it */
  }
  return { open, fresh, paste };
}

// --- the screen ----------------------------------------------------------------------------------------------------

export async function render(root, params) {
  await ensureStyles();
  if (!root.isConnected) return;
  if (!state.request) state.request = loadRequest();
  view = { root, stepTimer: null, offSlip: null };
  const applied = applyQuery(params.query);
  const { open, paste } = applied;
  const fresh = applied.fresh || state.freshTopic;
  state.freshTopic = false;
  draw();
  root.addEventListener("click", onClick);
  root.addEventListener("input", onInput);
  root.addEventListener("change", onChange);
  root.addEventListener("toggle", onToggle, true);
  root.addEventListener("paste", onPaste);
  root.addEventListener("keydown", onKeydown);
  root.addEventListener("submit", onSubmit);
  view.offSlip = slip.onChange(onSlipChange);
  if (state.busy) startSteps();
  refreshMainButton();
  // Sport names for the request bar (cached after the first load)
  loadSports().then(() => { if (view?.root === root) drawRequest(); }).catch(() => null);
  if (paste) openPaste(); // #/ask?paste=1 (Help's "See it live"): the sheet over this chat
  const resume = Number(prefs.get(THREAD_KEY, null));
  if (open && open !== state.threadId) await openThread(open);
  else if (!open && !fresh && !paste && !state.threadId && !state.messages.length && !busy() && Number.isInteger(resume) && resume > 0) {
    await openThread(resume, { quiet: true });
  }
}

/**
 * Another screen's ticket to ask about (the Builder's "Ask AI"): attached to the composer as "About: <label>"
 * with ``draft`` in the box, in a new chat. Nothing is sent until the person sends it (it uses the allowance).
 *   attachTicket({body: {type, stake, legs: [ref]}, label, draft})
 */
export function attachTicket({ body, label, draft = "" } = {}) {
  if (!body || !Array.isArray(body.legs) || !body.legs.length) return;
  if (!state.busy && (state.messages.length || state.loading || state.threadId)) resetThread();
  state.attach = { kind: "ticket", body, label: String(label || "A ticket").slice(0, 120) };
  if (draft && !state.draft.trim()) state.draft = String(draft).slice(0, MAX_MESSAGE);
  state.freshTopic = true;
  state.focus = true;
}

export function onLeave() {
  stopSteps();
  view?.offSlip?.();
  view = null;
}

/** Telegram's main button: "Find tickets" while the request bar is open, "Send" with something to send. */
export function mainButton() {
  if (!view || blockedReason()) return null;
  if (state.busy) return { text: "The AI is answering…", disabled: true, progress: true, onClick: () => {} };
  if (state.loading) return { text: "Loading your chat…", disabled: true, progress: true, onClick: () => {} };
  if (requestOpen()) return { text: "Find tickets", onClick: sendRequest };
  if (canSend()) return { text: "Send", onClick: sendComposer };
  if (slip.count && !state.attach) {
    return {
      text: `Ask about my slip (${slip.count})`,
      onClick: () => {
        state.attach = { kind: "slip" };
        updateDock();
        refreshMainButton();
      },
    };
  }
  return null;
}
