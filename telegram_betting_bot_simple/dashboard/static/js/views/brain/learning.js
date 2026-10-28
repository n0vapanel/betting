// The "Learning: On / Paused" switch, shared by the Brain progress page and Settings (owner only: the server says who
// the owner is, as GET /api/brain/progress {owner}, and refuses anyone else's POST /api/brain/pause with a 403).
// Pausing freezes the brain's current values: nothing is tuned, promoted or rolled back until it is switched on again.
// LEARN_ENABLED=0 on the server pauses it there (learning.paused_by "env"), and then the switch can't turn it on.
//
//   fetchLearning({signal}) -> {owner, learning} from GET /api/brain/progress (null when the route isn't there)
//   setLearningPaused(paused) -> POST /api/brain/pause {paused}; resolves to the server's answer, throws its ApiError
//   learningSwitchHtml(learning, {name}) -> the segmented On / Paused control (data-seg=name) with its words
//   learningWords(learning) -> one line on the current state

import { api } from "../../api.js";
import { esc, segmented } from "../../ui.js";
import { normalizeProgress, timeLabel } from "./model.js";

export const LEARNING_OPTIONS = [{ value: "on", label: "On" }, { value: "paused", label: "Paused" }];

export async function fetchLearning({ signal } = {}) {
  const raw = await api("/api/brain/progress", { signal });
  const data = normalizeProgress(raw);
  return { owner: data.owner, learning: data.learning };
}

export function setLearningPaused(paused) {
  return api("/api/brain/pause", { method: "POST", body: { paused: Boolean(paused) } });
}

/** "On: …" / "Paused by the owner on Oct 28, 09:30: …" / "Paused on the server …". */
export function learningWords(learning) {
  if (!learning) return "";
  if (learning.enabled) return "On: the brain scores itself every night and decides on changes once a week.";
  if (learning.pausedBy === "env") return "Paused on the server (LEARN_ENABLED=0): the brain keeps its current values, and it can't be switched on from here.";
  const when = learning.pausedAt ? ` on ${timeLabel(learning.pausedAt)}` : "";
  return `Paused${learning.pausedBy === "owner" ? " by the owner" : ""}${when}: the brain keeps its current values; nothing is tuned or rolled back until it's switched on again.`;
}

/** The On / Paused control and its words; disabled when the server holds it paused (env). */
export function learningSwitchHtml(learning, { name = "learning" } = {}) {
  const value = learning?.enabled === false ? "paused" : "on";
  const locked = learning?.pausedBy === "env";
  let seg = segmented(name, LEARNING_OPTIONS, value, { label: "Learning" });
  if (locked) seg = seg.replace(/<button /g, "<button disabled ");
  return `${seg}<p class="hint" data-learning-words>${esc(learningWords(learning))}</p>`;
}
