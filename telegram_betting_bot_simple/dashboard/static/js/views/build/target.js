// "Let AI build one" on the Build screen (#/build): the way into the Builder (#/build/builder, builder.js).
//
// The Builder replaced the old stake-and-target form and the first dashboard's NBA-only Safe / Risky / Crazy
// panel: its presets fill every control (sports, time, legs, target odds or a stake and "win about", markets,
// data rules), and any edit makes the run Custom. This card says so in two lines and offers the three presets,
// each opening the Builder with that preset built at once. Nothing goes on the slip from here.
// On a break, or in the 24-hour pause after a warning sign in the chat, the server refuses the Builder: the
// card's buttons are off up front, with one plain line saying until when (common.js picksBlock).

import { navigate } from "../../app.js";
import { esc } from "../../ui.js";
import { icon } from "../../icons.js";
import { picksBlock, blockNote } from "./common.js";

const BLOCK_ID = "bx-block-builder"; // the line that says why the Builder is off
const PRESETS = [
  { key: "safe", title: "Safe", sub: "60%+", iconName: "shield" },
  { key: "risky", title: "Risky", sub: "25–60%", iconName: "scale" },
  { key: "crazy", title: "Crazy", sub: "3–25%", iconName: "zap" },
];

function cardHtml(block) {
  const off = block ? ` aria-disabled="true" aria-describedby="${BLOCK_ID}" tabindex="-1"` : "";
  return `<div class="section-head"><h2>${icon("build", { size: 18 })} Let AI build one</h2></div>
    <p class="small muted">The Builder: Safe, Risky and Crazy fill every control (sports, time, legs, target odds, markets, data rules), and you can change any of them. Each ticket says why it was picked, what it costs on average and what nearly made it.</p>
    <div data-b-block>${block ? blockNote(block, BLOCK_ID) : ""}</div>
    <div class="bx-presets${block ? " off" : ""}" role="group" aria-label="Open the Builder with a preset">
      ${PRESETS.map((p) => `<a class="bx-preset t-${p.key}" href="#/build/builder?preset=${p.key}"${off}>${icon(p.iconName, { size: 18 })}<span><b>${esc(p.title)}</b><small>${esc(p.sub)}</small></span></a>`).join("")}
    </div>
    <a class="btn btn-primary block${block ? " is-off" : ""}" href="#/build/builder" style="margin-top:12px"${off}>${icon(block ? "pause" : "sliders", { size: 18 })}Open the Builder</a>`;
}

/**
 * The Builder's card in ``root``. Returns {prefill({tier}), draw(), sync()}:
 *   prefill({tier})  open the Builder with that preset (the empty slip's Safe / Risky / Crazy chips)
 *   draw()           nothing to redraw here (kept for slip.js)
 *   sync()           after GET /api/me changed: switch the card off (a break, the picks pause) or back on
 */
export function createBuilder(root) {
  let block = picksBlock();
  root.innerHTML = `<section class="card section bx-builder" id="builder">${cardHtml(block)}</section>`;
  const card = root.querySelector("#builder");
  card.addEventListener("click", (event) => {
    if (block && event.target.closest("a")) event.preventDefault(); // off: the line above says until when
  });
  return {
    sync() {
      const next = picksBlock();
      if (Boolean(next) === Boolean(block) && next?.text === block?.text) return;
      block = next;
      card.innerHTML = cardHtml(block);
    },
    prefill({ tier } = {}) {
      if (picksBlock()) return;
      navigate(`#/build/builder${["safe", "risky", "crazy"].includes(tier) ? `?preset=${tier}` : ""}`);
    },
    draw() {},
  };
}
