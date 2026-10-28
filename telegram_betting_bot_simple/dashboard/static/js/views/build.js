// Build: the slip and everything done with it, in three routes.
//   #/build           the slip (build/slip.js): picks, Single · Multi · Round robin by k · a named
//                     System, stake chips, the server's numbers, "Analyse with AI" (build/analyse.js),
//                     Save, Share code, Clear, then the Builder's card and "Find players" (build/target.js: the
//                     way into the Player finder, #/players)
//   #/build/builder   the Builder (build/builder.js): Safe · Risky · Crazy presets filling every control,
//                     up to five tickets that share no leg, each leg with Why and Swap, near misses
//   #/build/rollover  the rollover planner (build/rollover.js): classic, bank half, pot-and-ladder and
//                     flat stakes side by side, the survival chart (build/chart.js), "Start paper plan"
// Deep links: #/build?analyse=1 (the slip bar's "Analyse"), #/build?code=<share code>,
// #/build/builder?preset=safe|risky|crazy (that preset, built at once).
// Inside Telegram the bottom main button is "Analyse with AI" on the slip, "Build tickets" in the Builder
// and "Start paper plan" on the planner, and the second bottom button clears the slip. Every number shown
// comes from the server.
// On a break or in the 24-hour picks pause the Builder and "Start paper plan" are off up front, with one
// line saying until when (build/common.js picksBlock); on a
// break the slip's Analyse and Save are off too, since the server then refuses pricing, analysing and saving
// a slip. The profile is fetched again on each visit, since a break or the pause can start elsewhere (the
// bot's chat, another device).

import { refreshMainButton, inTelegram, loadMe } from "../app.js";
import { ensureStyles } from "./build/styles.js";
import { mountSlip } from "./build/slip.js";
import { mountRollover } from "./build/rollover.js";
import { mountBuilder } from "./build/builder.js";

let screen = null; // the mounted part: {mainButton(), stop(), ready}

export function mainButton() {
  return screen?.mainButton?.() || null;
}

export function onLeave() {
  screen?.stop?.();
  screen = null;
}

export async function render(root, params) {
  ensureStyles();
  const options = { params, refresh: refreshMainButton, inTelegram };
  screen = params.mode === "rollover" ? mountRollover(root, options)
    : params.mode === "builder" ? mountBuilder(root, options) : mountSlip(root, options);
  // A break or a picks pause started elsewhere (the bot's chat, another device) shows once this is back
  loadMe().catch(() => null);
  await screen.ready;
}
