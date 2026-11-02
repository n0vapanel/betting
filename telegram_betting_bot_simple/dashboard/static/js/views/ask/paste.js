// Ask AI: pasting a bet slip. The "📋 Paste a slip" sheet (the slip's text and/or a screenshot), the
// screenshot's downscaling in the browser, and the check that tells pasted chat text from a slip.
//
// What leaves the browser: the text as pasted (up to MAX_PASTE characters) and, for a screenshot, a JPEG
// redrawn on a canvas at most MAX_SIDE px on its long side and MAX_IMAGE_BYTES in size (the server takes
// 1.8 MB and keeps none of it: only what was read). Nothing here reads the slip: the AI or the server's
// free reader does, and the server's code checks and prices it.

import { esc, sheet, notice } from "../../ui.js";
import { icon } from "../../icons.js";

export const MAX_PASTE = 6000; // engine/pasted.py MAX_PASTE_CHARS
export const MAX_SIDE = 1600; // px on the long side
export const MAX_IMAGE_BYTES = 1_500_000;
const MAX_FILE_BYTES = 40 * 1024 * 1024; // a photo bigger than this isn't a screenshot: refused before decoding
// Long side and JPEG quality, tried in turn until the screenshot fits MAX_IMAGE_BYTES
const ATTEMPTS = [[1600, 0.85], [1600, 0.75], [1400, 0.7], [1200, 0.65], [1000, 0.6]];

// --- does this text look like a slip? ---------------------------------------------------------------

// An odds-like token, as engine/pasted.py's _ODDS_TOKEN (without a look-behind, which older iOS
// webviews can't parse): American (+150, -110), decimal with 2-3 places (1.85, 1,85), fractional (5/2)
// or evens. A line (-4.5, 224.5) or an amount after a currency sign isn't odds.
const ODDS = /(^|[^\w.,/:$£€₦₹₵+-])([+-]\d{3,5}(?![\d.,/])|\d{1,3}\.\d{2,3}(?![\d.,/%])|\d{1,3},\d{2}(?![\d.,/%])|\d{1,3}\/\d{1,3}(?![\d/.,])|evs|evens)(?![A-Za-z])/gi;

function oddsLike(token) {
  const t = token.toLowerCase();
  if (t === "evs" || t === "evens") return true;
  if (/^[+-]/.test(t)) return Math.abs(Number(t)) >= 100;
  if (t.includes("/")) {
    const [a, b] = t.split("/").map(Number);
    return a > 0 && b > 0;
  }
  const n = Number(t.replace(",", "."));
  return n > 1 && n <= 1001;
}

/** How many odds-like tokens a line holds. */
function oddsCount(line) {
  let count = 0;
  for (const match of line.matchAll(ODDS)) if (oddsLike(match[2])) count += 1;
  return count;
}

/** Pasted text that looks like a bet slip: two or more odds-like tokens over two or more lines. */
export function looksLikeSlip(text) {
  const lines = String(text || "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return false;
  const counts = lines.map(oddsCount);
  return counts.reduce((a, b) => a + b, 0) >= 2 && counts.filter(Boolean).length >= 2;
}

// --- screenshots ------------------------------------------------------------------------------------

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function canvasBlob(canvas, quality) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}

/** The picture as something a canvas can draw: an ImageBitmap (turned upright), else an <img>. */
async function decode(file) {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      /* some browsers take no options, or can't decode this type: try an <img> */
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * A screenshot redrawn as a JPEG of at most MAX_SIDE px on its long side and MAX_IMAGE_BYTES:
 * {base64, dataUrl, width, height, bytes, media_type}. Throws an Error with words for the person.
 */
export async function downscale(file) {
  if (!file || !/^image\//.test(file.type || "")) throw new Error("That file isn't a picture: add a screenshot of the slip (PNG or JPEG).");
  if (file.size > MAX_FILE_BYTES) throw new Error("That picture is too big to send: take a screenshot of just the slip.");
  let picture;
  try {
    picture = await decode(file);
  } catch {
    throw new Error("That picture couldn't be opened here: take a screenshot (PNG or JPEG) and add that.");
  }
  const width = picture.width || picture.naturalWidth;
  const height = picture.height || picture.naturalHeight;
  if (!width || !height) throw new Error("That picture couldn't be opened here: take a screenshot (PNG or JPEG) and add that.");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  try {
    for (const [side, quality] of ATTEMPTS) {
      const scale = Math.min(1, side / Math.max(width, height));
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      ctx.fillStyle = "#ffffff"; // a see-through PNG gets a white page, not a black one
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(picture, 0, 0, canvas.width, canvas.height);
      const blob = await canvasBlob(canvas, quality);
      if (blob && blob.size <= MAX_IMAGE_BYTES) {
        const dataUrl = await blobToDataUrl(blob);
        return { dataUrl, base64: dataUrl.slice(dataUrl.indexOf(",") + 1), width: canvas.width, height: canvas.height,
          bytes: blob.size, media_type: "image/jpeg" };
      }
    }
  } finally {
    picture.close?.();
    canvas.width = 0;
    canvas.height = 0;
  }
  throw new Error("That screenshot has too much detail to send: crop it to just the bet slip and add it again.");
}

function sizeText(bytes) {
  return bytes >= 1_000_000 ? `${(bytes / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1000))} KB`;
}

// --- the sheet ---------------------------------------------------------------------------------------

/**
 * Why the AI can't read a screenshot now, in words for the person, or "" when it can: no AI on this server,
 * the picks pause (no AI reads then) or too little of today's allowance (no AI call at all).
 */
export function shotBlocked({ ai = true, paused = false, creditsOut = false } = {}) {
  if (!ai) return "The free reader reads text only: paste the slip's text.";
  if (paused) return "While picks are paused the AI doesn't read screenshots: paste the slip's text and the app's code reads it.";
  if (creditsOut) return "There isn't enough of today's AI allowance left to read a screenshot: paste the slip's text and the free reader (code only) reads it.";
  return "";
}

const EXAMPLE = "e.g.\nBoston Celtics -3.5  -110\nDenver Nuggets ML  -400\nStake $10   To win $13.86";

/**
 * The paste sheet. options:
 *   text, image ({dataUrl, base64, ...} from downscale), file (a picture to add straight away)
 *   ai: true when the AI can read it ("Analyse"); false offers only the free reader (code only, text only)
 *   creditsOut: too little of today's AI allowance is left for any AI call (no screenshot can be read then)
 *   paused: picks are paused (the AI doesn't read screenshots then; the free reader reads text)
 *   Without the AI, in the pause or out of allowance the sheet offers no screenshot (nothing would read it)
 *   note: a line at the top (e.g. that an earlier screenshot wasn't kept)
 *   onAnalyse({text, image}) and onFree(text): called after the sheet closes; return nothing
 *   onDraft({text, image}): every change, so a closed sheet opens again as it was
 * The pasted text and the screenshot are only shown back to the person here, escaped.
 */
export function openPasteSheet({ text = "", image = null, file = null, ai = true, creditsOut = false, paused = false,
  note = "", onAnalyse, onFree, onDraft = () => {} } = {}) {
  const refusal = shotBlocked({ ai, paused, creditsOut }); // "" when the AI can read a screenshot now
  const shots = !refusal;
  const draft = { text: String(text || "").slice(0, MAX_PASTE), image: shots ? image : null };
  let working = false;

  const allowanceNote = !ai ? "" : paused
    ? "While picks are paused the AI doesn't read slips: Analyse reads pasted text with the app's code (no AI), and screenshots can't be read."
    : creditsOut
      ? "There isn't enough of today's AI allowance left for the AI: Analyse reads pasted text with the free reader (code only), and screenshots can't be read until it resets at midnight (US Eastern)."
      : "Analyse uses a little of your daily AI allowance. The free reader costs nothing and reads plain text.";

  const shotHtml = () => {
    if (!shots) return "";
    if (draft.image) {
      return `<figure class="ask-shot">
        <img src="${esc(draft.image.dataUrl)}" alt="Your screenshot of the slip" width="${esc(draft.image.width)}" height="${esc(draft.image.height)}">
        <figcaption class="small muted">${esc(`${draft.image.width} × ${draft.image.height} px · ${sizeText(draft.image.bytes)}`)}</figcaption>
        <button type="button" class="btn btn-secondary small" data-shot-remove>${icon("trash", { size: 16 })}Remove</button>
      </figure>`;
    }
    return `<button type="button" class="btn btn-secondary ask-shot-add" data-shot-add${working ? " disabled" : ""}><span aria-hidden="true">📷</span>Add a screenshot</button>`;
  };

  const body = document.createElement("form");
  body.className = "ask-paste";
  body.noValidate = true;
  body.innerHTML = `${note ? `<div class="ask-paste-lead" data-paste-lead>${notice(note, "info")}</div>` : ""}
    <p class="muted small">${shots ? "The AI copies what your slip says; then the app's code checks every number, finds each leg among the games it has odds for and prices it at your odds."
    : "The free reader (code only, no AI) reads one bet per line with its odds at the end, then the app's code finds each leg among the games it has odds for and prices it at your odds."}</p>
    <div class="field">
      <label class="label" for="paste-text">Paste the text of your bet slip from your bookmaker's app or site</label>
      <textarea id="paste-text" class="input ask-paste-text" rows="7" maxlength="${MAX_PASTE}" placeholder="${esc(EXAMPLE)}" spellcheck="false" autocomplete="off">${esc(draft.text)}</textarea>
      <p class="ask-count" data-paste-count></p>
    </div>
    ${shots ? `<div class="field ask-shot-field">
      <span class="label" id="paste-shot-label">Or a screenshot of it</span>
      <div data-shot>${shotHtml()}</div>
      <input type="file" accept="image/*" class="ask-file" data-shot-input tabindex="-1" aria-hidden="true">
      <p class="hint">${icon("shield", { size: 14 })} Screenshots aren't kept: only what's read from them, in your own history.</p>
    </div>` : ""}
    <p class="hint" data-paste-note>${esc(allowanceNote)}</p>
    <p class="form-error" data-paste-error role="alert"></p>`;

  const footer = `${ai ? `<button type="button" class="btn btn-secondary grow" data-paste-free>Read without AI (free)</button>
      <button type="button" class="btn btn-primary grow" data-paste-analyse>${icon("ask", { size: 18 })}Analyse</button>`
    : `<button type="button" class="btn btn-primary block" data-paste-free>Read without AI (free)</button>`}`;

  const s = sheet.open({ title: "📋 Paste a slip", body, footer, size: "tall", className: "ask-paste-sheet" });
  const area = body.querySelector("#paste-text");
  const fileInput = body.querySelector("[data-shot-input]");
  const error = body.querySelector("[data-paste-error]");

  const setError = (message) => {
    error.textContent = message || "";
  };
  const drawShot = () => {
    const box = body.querySelector("[data-shot]");
    if (box) box.innerHTML = shotHtml();
  };
  const drawButtons = () => {
    const text = draft.text.trim();
    const free = s.foot.querySelector("[data-paste-free]");
    if (free) free.disabled = working || !text;
    const analyse = s.foot.querySelector("[data-paste-analyse]");
    if (analyse) analyse.disabled = working || (!text && !draft.image);
    const count = body.querySelector("[data-paste-count]");
    if (count) count.textContent = draft.text.length > MAX_PASTE - 500 ? `${draft.text.length.toLocaleString()} / ${MAX_PASTE.toLocaleString()}` : "";
  };

  async function addPicture(picture) {
    if (!shots) {
      setError(refusal);
      return;
    }
    if (working) return;
    working = true;
    setError("");
    const box = body.querySelector("[data-shot]");
    if (box) box.innerHTML = `<p class="small muted" role="status">${icon("clock", { size: 14 })} Getting the screenshot ready…</p>`;
    drawButtons();
    try {
      draft.image = await downscale(picture);
      onDraft({ ...draft });
    } catch (e) {
      setError(e?.message || "That picture couldn't be added.");
    }
    working = false;
    if (!s.open) return;
    drawShot();
    drawButtons();
    body.querySelector(draft.image ? "[data-shot-remove]" : "[data-shot-add]")?.focus({ preventScroll: true });
  }

  area.addEventListener("input", () => {
    draft.text = area.value.slice(0, MAX_PASTE);
    setError("");
    onDraft({ ...draft });
    drawButtons();
  });
  // A picture pasted into the box (a copied screenshot) is added as the screenshot
  area.addEventListener("paste", (event) => {
    const picture = [...(event.clipboardData?.files || [])].find((f) => /^image\//.test(f.type));
    if (!picture) return;
    event.preventDefault();
    addPicture(picture); // says why when no screenshot can be read now
  });
  body.addEventListener("click", (event) => {
    if (event.target.closest("[data-shot-add]")) {
      fileInput?.click();
    } else if (event.target.closest("[data-shot-remove]")) {
      draft.image = null;
      onDraft({ ...draft });
      drawShot();
      drawButtons();
      body.querySelector("[data-shot-add]")?.focus({ preventScroll: true });
    }
  });
  fileInput?.addEventListener("change", () => {
    const picture = fileInput.files?.[0];
    fileInput.value = "";
    if (picture) addPicture(picture);
  });
  // A picture dropped on the sheet (a computer)
  body.addEventListener("dragover", (event) => {
    if ([...(event.dataTransfer?.items || [])].some((i) => i.kind === "file")) event.preventDefault();
  });
  body.addEventListener("drop", (event) => {
    const files = [...(event.dataTransfer?.files || [])];
    if (!files.length) return;
    event.preventDefault(); // a dropped file never replaces the app in the tab
    const picture = files.find((f) => /^image\//.test(f.type));
    if (picture) addPicture(picture); // says why when no screenshot can be read now
    else setError("That file isn't a picture: add a screenshot of the slip (PNG or JPEG), or paste its text.");
  });
  body.addEventListener("submit", (event) => event.preventDefault());

  s.foot.addEventListener("click", async (event) => {
    if (working) return;
    const text = draft.text.trim();
    if (event.target.closest("[data-paste-free]")) {
      if (!text) {
        setError(draft.image ? "The free reader reads text only: paste the slip's text, or use Analyse for the screenshot."
          : "Paste the text of your bet slip first.");
        return;
      }
      await s.close();
      onFree?.(text);
      return;
    }
    if (event.target.closest("[data-paste-analyse]")) {
      if (!text && !draft.image) {
        setError("Paste the text of your bet slip or add a screenshot of it.");
        return;
      }
      if (!text && !shots) {
        setError(refusal);
        return;
      }
      await s.close();
      onAnalyse?.({ text, image: shots ? draft.image : null });
    }
  });

  drawButtons();
  if (file) addPicture(file);
  else requestAnimationFrame(() => {
    if (!s.open) return;
    if (matchMedia("(pointer: fine)").matches || !draft.text) area.focus({ preventScroll: true });
  });
  return s;
}
