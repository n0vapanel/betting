// My bets: everything the person has saved, graded honestly, plus how the AI itself is doing.
//   Upcoming   saved bets whose games haven't finished (GET /api/slips), soonest first
//   Settled    graded bets, newest first; both lists sit under "Your results": net profit or loss and
//              ROI for 7 days, 30 days or all time (never "total won"), open stakes and the budget left
//   Plans      rollover plans day by day; "Play day N" builds the day from the person's own picks, and
//              only once the previous day has its result (GET /api/plans, POST /api/plans/<id>/day)
//   Calls      the person's calls against the AI (legacy.js, unchanged)
//   Leaders    the calls leaderboard
//   AI record  the AI's graded predictions, its Safe / Risky / Crazy bets' record and the analysts
// Deep links: #/bets?tab=upcoming|settled|plans|calls|leaders|record (tab=slips is Upcoming, as before),
//   #/bets?tab=plans&plan=<id> opens a plan, &play=1 also opens its next day.
// Inside Telegram the bottom main button is "Play day N" while a plan's next day is ready on Plans.
//
// The pieces live in bets/: slips.js (cards, results, share, delete), plans.js, record.js, styles.js.

import { api } from "../api.js";
import { refreshMainButton, replaceAddress } from "../app.js";
import { segmented } from "../ui.js";
import { renderMyCalls } from "../legacy.js";
import { ensureStyles } from "./bets/styles.js";
import { renderSlips } from "./bets/slips.js";
import { renderPlans } from "./bets/plans.js";
import { renderLeaders, renderRecord } from "./bets/record.js";
import { bindTermInfo } from "./help/sheet.js";

const TABS = [
  { value: "upcoming", label: "Upcoming" },
  { value: "settled", label: "Settled" },
  { value: "plans", label: "Plans" },
  { value: "calls", label: "Calls" },
  { value: "leaders", label: "Leaders" },
  { value: "record", label: "AI record" },
];
const OLD_TABS = { slips: "upcoming", open: "upcoming", history: "settled", leaderboard: "leaders" };

let screen = null; // {tab, ready: plan whose next day can be played, play()} while My bets is showing

export function mainButton() {
  const plan = screen?.tab === "plans" ? screen.ready : null;
  if (!plan?.next) return null;
  return { text: `Play day ${plan.next.day} of ${plan.days}`, onClick: () => screen?.play?.() };
}

export function onLeave() {
  screen = null;
}

/**
 * Change the address without a new history entry, through the shell, so its in-app history (and the
 * scroll position Back restores) follows the new address.
 */
function setAddress(hash) {
  if (location.hash !== hash) replaceAddress(hash);
}

function pickTab(query) {
  const asked = query.get("tab") || "";
  const tab = OLD_TABS[asked] || asked;
  return TABS.some((t) => t.value === tab) ? tab : "upcoming";
}

export async function render(root, params) {
  ensureStyles();
  screen = { tab: pickTab(params.query), ready: null, play: null };
  const { signal } = params;
  root.innerHTML = `<div class="page-head"><h1>My bets</h1>
      <p>Your bets and plans, graded honestly; your calls against the AI; and the AI's own record.</p></div>
    <div class="mb-tabs" role="navigation" aria-label="My bets sections">${segmented("bets-tab", TABS, screen.tab, { label: "Section" })}</div>
    <div class="section" data-box></div>`;
  const box = root.querySelector("[data-box]");
  bindTermInfo(root);

  // One load of the saved bets serves both Upcoming and Settled; a change (delete) loads them again
  let slipsLoad = null;
  const slips = (fresh = false) => {
    if (fresh || !slipsLoad) {
      slipsLoad = api("/api/slips", { signal });
      slipsLoad.catch(() => { slipsLoad = null; });
    }
    return slipsLoad;
  };

  // Each tab draws into a fresh pane of its own. A tab still loading when the person moves on finds its
  // pane gone from the page (its isConnected check fails) and stops, instead of drawing over the new tab
  // or claiming Telegram's main button; so do its later reloads (after a delete, a play or a stop).
  // Resolves once the tab has drawn.
  let first = true;
  const show = async () => {
    const tab = screen.tab;
    const pane = document.createElement("div");
    box.replaceChildren(pane);
    screen.ready = null;
    screen.play = null;
    refreshMainButton();
    const query = first ? params.query : new URLSearchParams();
    first = false;
    if (tab === "upcoming" || tab === "settled") return renderSlips(pane, tab, { signal, slips });
    if (tab === "plans") {
      return renderPlans(pane, {
        signal, query,
        onReady: (plan, play) => {
          if (!screen || screen.tab !== "plans" || !pane.isConnected) return;
          screen.ready = plan;
          screen.play = play;
          refreshMainButton();
        },
      });
    }
    if (tab === "calls") return renderMyCalls(pane);
    if (tab === "leaders") return renderLeaders(pane, { signal });
    return renderRecord(pane, { signal });
  };

  const tabs = root.querySelector(".mb-tabs");
  // Bring the chosen tab into the strip's view sideways only: scrollIntoView would also scroll the page,
  // undoing the scroll position the shell restores on Back
  const reveal = () => {
    const on = tabs.querySelector('[aria-pressed="true"]');
    if (!on) return;
    const tab = on.getBoundingClientRect();
    const strip = tabs.getBoundingClientRect();
    if (tab.left < strip.left) tabs.scrollLeft -= strip.left - tab.left + 8;
    else if (tab.right > strip.right) tabs.scrollLeft += tab.right - strip.right + 8;
  };
  root.addEventListener("click", (event) => {
    const seg = event.target.closest('[data-seg="bets-tab"]');
    if (!seg || seg.dataset.value === screen?.tab) return;
    screen.tab = seg.dataset.value;
    for (const b of tabs.querySelectorAll('[data-seg="bets-tab"]')) b.setAttribute("aria-pressed", String(b === seg));
    setAddress(`#/bets?tab=${screen.tab}`);
    reveal();
    show();
  });
  requestAnimationFrame(reveal);
  // Wait for the first tab, so the shell restores the scroll position (Back) on the drawn page
  await show();
}
