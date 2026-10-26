// Build: the slip and everything done with it, in two routes.
//   #/build           the slip (build/slip.js): picks, Single · Multi · Round robin by k · a named
//                     System, stake chips, the server's numbers, "Analyse with AI" (build/analyse.js),
//                     Save, Share code, Clear, then "Let AI build one" (build/target.js)
//   #/build/rollover  the rollover planner (build/rollover.js): classic, bank half, pot-and-ladder and
//                     flat stakes side by side, the survival chart (build/chart.js), "Start paper plan"
// Deep links: #/build?analyse=1 (the slip bar's "Analyse"), #/build?code=<share code>.
// Inside Telegram the bottom main button is "Analyse with AI" on the slip and "Start paper plan" on the
// planner, and the second bottom button clears the slip. Every number shown comes from the server.

import { refreshMainButton, inTelegram } from "../app.js";
import { ensureStyles } from "./build/styles.js";
import { mountSlip } from "./build/slip.js";
import { mountRollover } from "./build/rollover.js";

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
  screen = params.mode === "rollover" ? mountRollover(root, options) : mountSlip(root, options);
  await screen.ready;
}
