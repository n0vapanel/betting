// Help: how the app works, as static pages (nothing here calls the AI or costs anything).
//   #/help              search, the guides and the whole glossary by group
//   #/help/glossary     the glossary on its own, with the same search
//   #/help/<guide>      start (the 60-second tour) · ai (how the AI decides) · rollover (rollover
//                       explained, with the blueprint's real numbers) · responsible · faq
//                       (#/help/faq?item=3 opens one answer)
//   #/help/<term>       one glossary entry: the definition, a worked example step by step, the person's
//                       own slip where the term applies, "See it live" and related terms. Every ⓘ sheet's
//                       "More in Help" link lands here (the keys are ui.js's GLOSSARY keys).
// ?q=<words> on #/help or #/help/glossary starts with a search (it matches titles, other names such as
// "vig" or "parlay", definitions, the guides and the FAQ).
//
// The words live in help/content.js. Numbers there are fixed teaching examples worked out by hand; the
// only live number is the "Your slip" line, which is the server's pricing of the person's own slip.

import { me, priced, setTitle, replaceAddress } from "../app.js";
import { esc, section, notice, emptyState } from "../ui.js";
import { icon } from "../icons.js";
import { ensureStyles } from "./help/styles.js";
import {
  terms, term, GROUPS, TOPICS, topic, START_STEPS, ROLLOVER_TABLE, SURVIVAL, STRATEGIES, HELPLINES, WARNING_SIGNS, FAQ,
} from "./help/content.js";

const chevron = () => icon("chevron", { size: 18 });

/**
 * Change the address without a new history entry, through the shell, so its in-app history (and the
 * scroll position Back restores) follows the new address.
 */
function setAddress(hash) {
  if (location.hash !== hash) replaceAddress(hash);
}

// --- search -------------------------------------------------------------------------------------------

/** Lower case, accents and punctuation out, so "Vig", "vig." and "víg" all match. */
function norm(text) {
  return String(text ?? "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[−–—]/g, "-").replace(/[^a-z0-9%+.\- ]+/g, " ").replace(/\s+/g, " ").trim();
}

const stripTags = (html) => String(html).replace(/<[^>]+>/g, " ");

/**
 * Score one item against the query's words: every word must appear somewhere; a hit in the title or
 * another name counts more than one in the definition. 0 means no match.
 */
function score(fields, words, phrase) {
  let total = 0;
  for (const word of words) {
    let best = 0;
    for (const [text, weight] of fields) {
      if (!text) continue;
      const at = text.indexOf(word);
      if (at < 0) continue;
      const startsWord = at === 0 || text[at - 1] === " ";
      best = Math.max(best, weight * (startsWord ? 2 : 1));
    }
    if (!best) return 0;
    total += best;
  }
  if (phrase.length > 2 && fields.slice(0, 2).some(([text]) => text && text.includes(phrase))) total += 20;
  return total;
}

function search(query) {
  const phrase = norm(query);
  const words = phrase.split(" ").filter(Boolean);
  if (!words.length) return null;
  const rank = (list) => list.filter((x) => x.score > 0).sort((a, b) => b.score - a.score);
  const foundTerms = rank(terms().map((t) => ({
    item: t,
    score: score([[norm(t.title), 6], [norm([...t.aliases, ...t.aka].join(" ")), 5], [norm(t.short), 2],
      [norm(`${t.body} ${t.example} ${t.worked.join(" ")}`), 1]], words, phrase),
  })));
  const foundTopics = rank(TOPICS.map((t) => ({
    item: t, score: score([[norm(t.title), 6], [norm(t.words), 4], [norm(t.blurb), 2]], words, phrase),
  })));
  const foundFaq = rank(FAQ.map((f, index) => ({
    item: { ...f, index }, score: score([[norm(f.q), 4], [norm(stripTags(f.a)), 1]], words, phrase),
  })));
  return { terms: foundTerms.map((x) => x.item), topics: foundTopics.map((x) => x.item), faq: foundFaq.map((x) => x.item) };
}

function searchBox(value) {
  return `<div class="hp-search" role="search">
    <label class="sr-only" for="help-q">Search help</label>
    ${icon("search", { size: 18 })}
    <input id="help-q" class="input" type="search" value="${esc(value)}" placeholder="Search: parlay, vig, push, rollover…"
      autocomplete="off" autocapitalize="off" spellcheck="false" enterkeyhint="search">
  </div>`;
}

function resultsHtml(found, query) {
  const count = found.terms.length + found.topics.length + found.faq.length;
  if (!count) {
    return `<div class="hp-empty"><p>Nothing matches “${esc(query)}”.</p>
      <p class="small">Try another word (for example “parlay”, “juice” or “cash out”), or browse the <a href="#/help/glossary">glossary</a>.</p></div>`;
  }
  const parts = [`<p class="hp-count" role="status">${count} result${count === 1 ? "" : "s"} for “${esc(query)}”</p>`];
  if (found.topics.length) {
    parts.push(section("Guides", `<ul class="list">${found.topics.map((t) => `<li><a class="list-item" href="#/help/${t.key}">
      ${icon(t.iconName, { size: 20 })}<span class="grow"><b>${esc(t.title)}</b><span>${esc(t.blurb)}</span></span>${chevron()}</a></li>`).join("")}</ul>`));
  }
  if (found.terms.length) parts.push(section("Glossary", termList(found.terms, { showAka: true })));
  if (found.faq.length) {
    parts.push(section("Questions", `<ul class="list">${found.faq.map((f) => `<li><a class="list-item" href="#/help/faq?item=${f.index}">
      ${icon("help", { size: 20 })}<span class="grow"><b>${esc(f.q)}</b></span>${chevron()}</a></li>`).join("")}</ul>`));
  }
  return parts.join("");
}

/** Wire the search box: results replace the page's usual content while there are words in it. */
function bindSearch(root, baseHash) {
  const input = root.querySelector("#help-q");
  const results = root.querySelector("[data-results]");
  const usual = root.querySelector("[data-usual]");
  const show = () => {
    const query = input.value.trim();
    const found = search(query);
    results.hidden = !found;
    usual.hidden = Boolean(found);
    results.innerHTML = found ? resultsHtml(found, query) : "";
    // Keep the words in the address, so Back from an entry comes back to the same results
    const hash = query ? `${baseHash}?q=${encodeURIComponent(query)}` : baseHash;
    if (location.hash !== hash) setAddress(hash);
  };
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(show, 120);
  });
  input.addEventListener("search", show);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      input.blur(); // closes the phone keyboard so the results can be read
    }
  });
  if (input.value.trim()) show();
}

// --- the glossary -------------------------------------------------------------------------------------

function termList(list, { showAka = false } = {}) {
  return `<ul class="list">${list.map((t) => `<li><a class="list-item" href="#/help/${esc(t.key)}">
    <span class="grow"><b>${esc(t.title)}</b><span>${esc(t.short)}</span>${showAka && t.aka.length ? `<span class="hp-also">Also: ${esc(t.aka.join(", "))}</span>` : ""}</span>${chevron()}</a></li>`).join("")}</ul>`;
}

function glossaryGroups() {
  const all = terms();
  return GROUPS.map((g) => {
    const list = all.filter((t) => t.group === g.key);
    return list.length ? `<div class="hp-group"><h3>${esc(g.title)}</h3>${termList(list)}</div>` : "";
  }).join("");
}

function helplineCard() {
  return `<section class="card section">
    <div class="row-between"><h2 class="row" style="font-size:16px">${icon("lifebuoy", { size: 20 })}Need to talk to someone?</h2></div>
    <p class="small muted" style="margin-top:6px">Free, confidential and open 24/7: US <a href="tel:18004262537">1-800-GAMBLER</a> · UK <a href="tel:08088020133">0808 8020 133</a>.</p>
    <a class="text-link" href="#/help/responsible">Responsible play ${icon("chevron", { size: 16 })}</a>
  </section>`;
}

function guidesHtml() {
  return `<div class="hp-guides">${TOPICS.map((t, i) => (i === 0
    ? `<a class="hp-guide first" href="#/help/${t.key}">${icon(t.iconName, { size: 24 })}<div><b>${esc(t.title)}</b><span>${esc(t.blurb)}</span></div>${chevron()}</a>`
    : `<a class="hp-guide${t.key === "responsible" ? " rg" : ""}" href="#/help/${t.key}">${icon(t.iconName, { size: 22 })}<b>${esc(t.title)}</b><span>${esc(t.blurb)}</span></a>`)).join("")}</div>`;
}

function renderHome(root, params) {
  const q = params.query.get("q") || "";
  root.innerHTML = `<div class="page-head"><h1>Help</h1><p>How the app works, in plain words. These pages never call the AI.</p></div>
    ${searchBox(q)}
    <div data-results hidden></div>
    <div data-usual>
      <div class="section">${guidesHtml()}</div>
      ${section("Glossary", glossaryGroups(), { aside: `<span class="muted small">${terms().length} terms</span>` })}
      ${helplineCard()}
    </div>`;
  bindSearch(root, "#/help");
}

function renderGlossary(root, params) {
  setTitle("Glossary");
  const q = params.query.get("q") || "";
  root.innerHTML = `<div class="page-head"><h1>Glossary</h1><p>Every term in the app, each with a worked example and a link to see it live.</p></div>
    ${searchBox(q)}
    <div data-results hidden></div>
    <div data-usual><div class="card section">${glossaryGroups()}</div></div>`;
  bindSearch(root, "#/help/glossary");
}

function renderTerm(root, t) {
  setTitle(t.title);
  const group = GROUPS.find((g) => g.key === t.group);
  let live = null;
  try {
    live = t.live?.({ priced: priced().result }) || null; // the server's pricing of the person's own slip
  } catch {
    live = null;
  }
  const see = t.see || (t.topic ? `#/help/${t.topic}` : null);
  const guide = t.topic ? topic(t.topic) : null;
  const ask = `#/ask?q=${encodeURIComponent(`Explain "${t.title}" with an example from tonight's games`)}`;
  const related = t.related.map((key) => term(key)).filter(Boolean);
  root.innerHTML = `<div class="page-head"><span class="eyebrow">Glossary · ${esc(group?.title || "")}</span><h1>${esc(t.title)}</h1></div>
    <section class="card">
      <p class="hp-lead">${esc(t.short)}</p>
      <div class="hp-body" style="margin-top:10px"><p>${esc(t.body)}</p></div>
      ${t.aka.length ? `<p class="hint">Also called: ${esc(t.aka.join(", "))}</p>` : ""}
    </section>
    ${t.worked.length ? section("Worked example", `<ol class="hp-worked">${t.worked.map((step) => `<li><span>${esc(step)}</span></li>`).join("")}</ol>`)
      : `<div class="example"><span class="eyebrow">Example</span>${esc(t.example)}</div>`}
    ${live ? `<div class="example mine"><span class="eyebrow">Your slip, priced by the server</span>${esc(live)}</div>` : ""}
    <div class="hp-actions">
      ${see ? `<a class="btn btn-primary" href="${esc(see)}">${icon("arrow", { size: 18 })}${esc(t.seeLabel)}</a>` : ""}
      ${guide && see !== `#/help/${guide.key}` ? `<a class="btn btn-secondary" href="#/help/${guide.key}">${icon(guide.iconName, { size: 18 })}${esc(guide.title)}</a>` : ""}
      ${me().ai_enabled ? `<a class="btn btn-secondary" href="${esc(ask)}">${icon("ask", { size: 18 })}Ask AI about it</a>` : ""}
    </div>
    ${related.length ? section("Related", `<div class="chip-row wrap hp-related">${related.map((r) => `<a class="chip" href="#/help/${esc(r.key)}">${esc(r.title)}</a>`).join("")}</div>`) : ""}
    <a class="text-link" href="#/help/glossary">${icon("book", { size: 16 })} All terms</a>`;
}

// --- the guides ---------------------------------------------------------------------------------------

function points(items, kind = "") {
  return `<ul class="hp-points${kind ? ` is-${kind}` : ""}">${items.map(([iconName, html]) => `<li>${icon(iconName, { size: 18 })}<span>${html}</span></li>`).join("")}</ul>`;
}

function startHtml() {
  return `<div class="page-head"><span class="eyebrow">Start here · about 60 seconds</span><h1>How SportsBet AI works</h1>
      <p>Real chances for games, honest prices for any slip, and an AI analyst to ask. Nothing is bet here.</p></div>
    <section class="card"><ol class="hp-steps">${START_STEPS.map((s, i) => `<li class="hp-step">
      <span class="hp-step-icon" aria-hidden="true">${icon(s.iconName, { size: 20 })}<b>${i + 1}</b></span>
      <div><h3>${esc(s.title)}</h3><p>${esc(s.text)}</p><a class="text-link" href="${esc(s.href)}">${esc(s.link)} ${icon("chevron", { size: 16 })}</a></div>
    </li>`).join("")}</ol></section>
    ${section("Three things to know", points([
      ["info", "<b>Tap the ⓘ next to a word</b> for one short explanation with an example, often from your own slip."],
      ["scale", "<b>Every ticket shows the real chance</b> as “1 in X”, what the bookmaker keeps and the expected result in money. Most bets lose a little on average: that's the bookmaker's cut."],
      ["shield", "<b>Your limits are yours to set</b> in Settings: a weekly budget, a reality check and breaks from 24 hours to 3 months."],
    ]))}
    <div class="hp-actions"><a class="btn btn-primary" href="#/today">${icon("today", { size: 18 })}Start on Today</a>
      <a class="btn btn-secondary" href="#/help/glossary">${icon("book", { size: 18 })}Browse the glossary</a></div>`;
}

function aiHtml() {
  return `<div class="page-head"><span class="eyebrow">Guide</span><h1>How the AI decides</h1>
      <p>The brain counts, Claude explains, and the analysts have to earn their say.</p></div>
    ${section("1 · The brain computes every number", `<p class="small muted">The “brain” is the app's own code. Every chance, price, payout, cut and expected result you see comes from it.</p>
      ${points([
        ["scale", "<b>The market</b>: the best prices from several bookmakers, with their margin taken out, as the baseline in every sport."],
        ["chart", "<b>The NBA model</b> blends the bookmakers' consensus, ESPN's matchup predictor and a team model (form, rest, travel, injuries), plus player projections for props."],
        ["build", "<b>Ticket maths</b>: one pricing function for singles, multis, round robins and system bets, in exact money rounded once. It knows each leg's push chance and blocks same-game legs in a multi."],
        ["info", "<b>Other sports</b> have no model yet, so they show the market's chance and label it “Mkt”, never dressed up as the AI's."],
      ], "info")}`)}
    ${section("2 · Claude explains, it doesn't count", points([
      ["bot", "Claude (Anthropic's model) chooses which of the brain's tools to call (games, odds, players, pricing, the rollover planner) and writes the words around their results."],
      ["shield", "A check reads every reply before you do: a number that didn't come from this turn's tools, or words like “lock” or “guaranteed”, get it rejected. After a second miss you get a plain template instead."],
      ["check", "It never adds a pick to your slip, never suggests a bigger stake, and says when it doesn't know."],
      ["lifebuoy", "A small, fast model first sorts each message (what you're asking, and any sign of chasing losses or distress). A strong sign switches picks off for 24 hours and shows support instead."],
    ]))}
    ${section("3 · Analysts are graded and earn trust", `${points([
        ["target", "Claude and DeepSeek can each forecast an NBA game a few hours before tip-off. Every forecast is frozen before the game and scored after it, next to the AI's and the bookmakers'."],
        ["trophy", "An analyst moves nothing until it has 40 graded games. After that it earns weight only if its record beats the AI's, and loses it again if it slips."],
      ])}
      <a class="text-link" href="#/bets?tab=record">See the AI's and the analysts' record ${icon("chevron", { size: 16 })}</a>`, { info: "analysts" })}
    ${section("Fixed words for chances", `<p class="small muted">Nobody, Claude included, gets to pick flattering words: code maps each chance to a label.</p>
      <ul class="list">
        <li class="list-item"><span class="grow"><b>Favourite</b><span>60% or more</span></span></li>
        <li class="list-item"><span class="grow"><b>Coin flip</b><span>40% to 60%</span></span></li>
        <li class="list-item"><span class="grow"><b>Underdog</b><span>20% to 40%</span></span></li>
        <li class="list-item"><span class="grow"><b>Long shot</b><span>5% to 20%</span></span></li>
        <li class="list-item"><span class="grow"><b>Lottery ticket</b><span>under 5%</span></span></li>
      </ul>`)}
    ${section("What it can't do", `<p>No model has been shown to beat the betting market reliably. The bookmakers' closing prices pick about 68% of NBA winners; a sound model lands near that, not far above it. The AI's record, misses included, is public in My bets.</p>`)}
    <div class="hp-actions"><a class="btn btn-secondary" href="#/help/ai_vs_market">${icon("scale", { size: 18 })}AI chance vs market chance</a>
      <a class="btn btn-secondary" href="#/help/brier">${icon("chart", { size: 18 })}How forecasts are scored</a></div>`;
}

function rolloverHtml() {
  const t = ROLLOVER_TABLE;
  const table = `<div class="hp-scens">${t.rows.map((r) => `<div class="hp-scen">
      <div class="hp-scen-head"><b>${esc(r.label)} a day</b><span>${esc(r.note)}</span></div>
      <dl class="hp-scen-grid">${r.cells.map(([chance, ev], i) => `<div><dt>${esc(t.columns[i][0])} <span>(${esc(t.columns[i][1])})</span></dt>
        <dd>${esc(chance)}</dd><dd class="hp-ev">${esc(ev)} on average</dd></div>`).join("")}</dl>
    </div>`).join("")}</div>
  <p class="note">The chance of winning every day at odds of 2.00, and the average result as a share of the stake. In the edge rows the average is positive only because of rare complete runs; the typical run still loses its stake.</p>`;
  const bars = `<div class="hp-bars" role="img" aria-label="At 47% a day: 47% of runs survive day 1, 22.1% day 2, 10.4% day 3, 4.9% day 4, 2.3% day 5">
    ${SURVIVAL.map((d) => `<div class="hp-bar" aria-hidden="true"><span>Day ${d.day}</span><span class="hp-track"><span style="width:${(d.share * 100).toFixed(1)}%"></span></span><b>${esc(d.text)}</b></div>`).join("")}
  </div>`;
  const strategies = `<div class="hp-lines">${STRATEGIES.map((s) => `<div class="hp-line">
      <b>${esc(s.title)}</b>
      <div class="small" style="margin-top:6px"><span class="muted">At market odds (47% a day):</span> ${esc(s.market[0])} chance of a profit · typical end ${esc(s.market[1])} · ${esc(s.market[2])} end at half or less</div>
      <div class="small"><span class="muted">With a real edge (53%):</span> ${esc(s.edge[0])} · ${esc(s.edge[1])} · ${esc(s.edge[2])}</div>
    </div>`).join("")}</div>
    <p class="note">30-day runs, one bet a day at 2.00, worked out exactly with the rollover planner's own maths; the bankroll starts at 1, so “typical end 0.90” means 90% of the start. The pot-and-ladder never wipes the bankroll out, and flat stakes do the least damage of all.</p>`;
  return `<div class="page-head"><span class="eyebrow">Guide</span><h1>Rollover explained</h1>
      <p>Stake an amount, then re-stake everything it wins each day. One loss ends it.</p></div>
    <section class="card">
      <p class="hp-lead">A rollover is a multi spread over days: the chances multiply, day after day.</p>
      <div class="hp-quote">To turn $10 into $10,240 in 10 days you need 10 wins in a row at 2.00. The market gives that about 1 in 1,900. Most likely the run ends on day 1 (53%).</div>
      <p class="small muted" style="margin-top:10px">At odds of 2.00 the market's own chance of winning a day is about 46–48% once the bookmaker's cut is out. The chance of finishing N days is that chance multiplied N times.</p>
    </section>
    ${section("The chance of finishing", table, { info: "one_in" })}
    ${section("Most runs end on day 1 or 2", `<p class="small muted">Share of runs still alive after each day, at the market's 47% a day.</p>${bars}
      ${points([
        ["alert", "About <b>78% of rollovers are over by day 2</b>, and the average run wins fewer than one day (0.89) before its first loss."],
        ["alert", "Building each day's 2.00 from <b>two short-priced legs</b> makes it worse: the cut is taken twice a day. At a 4% cut per leg, a 10-day plan completes about 1 in 2,317 and loses 55.8% on average, against 33.5% with one leg a day."],
        ["alert", "Restarting until a run lands takes about <b>1,901 starting stakes</b> per completed 10-day run at 47%, losing 46% of everything put in."],
        ["alert", "A coin-flip chance of finishing 10 days needs <b>93.3% a day</b>: at 2.00 that's an 87% edge per bet. No priced market offers that."],
      ], "warn")}`)}
    ${section("Safer ways to play a run", strategies)}
    ${section("Why the AI's chance must be cautious", `<p>Small overconfidence balloons over a run. An AI that says 52% when the truth is 47% overstates a 10-day completion chance 2.75 times, and a 30-day one 20.8 times. And staking everything at 2.00 with a real 2–5 point edge is 10 to 25 times the <a href="#/help/kelly">Kelly stake</a>. So the planner uses the market's chance, and the AI's only alongside it.</p>`)}
    ${section("How plans work in this app", points([
      ["check", "The planner shows classic, bank-half, pot-and-ladder and flat stakes <b>side by side</b>, with the chance of finishing, of a profit and the expected cost in money, before you start."],
      ["check", "Plans start in <b>paper mode</b>: tracked, but no money implied."],
      ["check", "<b>Nothing moves on by itself</b>: each day needs your tap, and only after the previous day's result is in."],
      ["check", "A failed run ends there, with a <b>24-hour pause</b> before a new plan. Every attempt stays in My bets, failures included."],
    ]))}
    <div class="hp-actions"><a class="btn btn-primary" href="#/build/rollover">${icon("trend", { size: 18 })}Open the rollover planner</a>
      <a class="btn btn-secondary" href="#/bets?tab=plans">${icon("bets", { size: 18 })}My plans</a></div>`;
}

function responsibleHtml() {
  return `<div class="page-head"><span class="eyebrow">Guide</span><h1>Responsible play</h1>
      <p>Betting should be entertainment that costs money, like a ticket to a game. If it stops being fun, stop.</p></div>
    ${section("Your limits", `<div>
        <div class="hp-limit">${icon("wallet", { size: 20 })}<div><b>Weekly budget</b><p>Stakes you save count against it, and anything past it is refused. Lowering it applies at once; raising it waits 24 hours.</p></div></div>
        <div class="hp-limit">${icon("clock", { size: 20 })}<div><b>Reality check</b><p>A reminder every 15, 30 or 60 minutes of how long the app has been open.</p></div></div>
        <div class="hp-limit">${icon("pause", { size: 20 })}<div><b>Take a break</b><p>24 hours, 3 days, a week, a month or 3 months. Picks, building and plans switch off until it ends, and it can't be cut short.</p></div></div>
      </div>
      <div class="hp-actions"><a class="btn btn-primary" href="#/settings?focus=limits">${icon("shield", { size: 18 })}Set my limits</a>
        <a class="btn btn-secondary" href="#/settings?focus=break">${icon("pause", { size: 18 })}Take a break</a></div>`)}
    ${section("Signs it's time to stop", `${points(WARNING_SIGNS.map((text) => ["alert", esc(text)]), "warn")}
      <p class="small" style="margin-top:10px">If any of these sound familiar, take a break and talk to someone. It's common, and help works.</p>`)}
    ${section("What this app does", points([
      ["scale", "Shows the real chance and the expected cost on every ticket and plan, and your net result, never just “total won”."],
      ["shield", "Never says “lock”, “guaranteed” or “sure thing”, never suggests a bigger stake, and never adds a pick for you."],
      ["lifebuoy", "If a message sounds like chasing losses or distress, the AI stops giving picks for 24 hours and shows support instead."],
      ["check", "Sends no notifications that nudge you to bet."],
    ]))}
    ${section("Free, confidential help", `<div class="hp-lines">${HELPLINES.map((h) => `<div class="hp-line">
        <span class="eyebrow">${esc(h.where)}</span><b>${esc(h.name)}</b>
        <p class="small muted" style="margin-top:2px">${esc(h.note)}</p>
        <div class="row">${h.tel ? `<a class="btn btn-secondary small" href="tel:${esc(h.tel)}">${icon("lifebuoy", { size: 16 })}Call ${esc(h.phone)}</a>` : ""}
          <a class="btn btn-ghost small" href="${esc(h.url)}" target="_blank" rel="noopener noreferrer">${icon("external", { size: 16 })}Website</a></div>
      </div>`).join("")}</div>
      <p class="note">In an emergency, or if you're thinking about harming yourself, call your local emergency number.</p>`)}`;
}

function faqHtml(open) {
  return `<div class="page-head"><span class="eyebrow">Guide</span><h1>FAQ</h1><p>Short answers to what people ask most.</p></div>
    <section class="card"><div class="hp-faq">${FAQ.map((f, i) => `<details id="faq-${i}"${i === open ? " open" : ""}>
      <summary><span>${esc(f.q)}</span>${icon("down", { size: 18 })}</summary>
      <div class="hp-answer">${f.a}</div>
    </details>`).join("")}</div></section>
    <p class="note">Still stuck? Search the glossary on the <a href="#/help">Help</a> page${me().ai_enabled ? ", or <a href=\"#/ask\">ask the AI</a>" : ""}.</p>`;
}

function renderTopic(root, t, params) {
  setTitle(t.title);
  if (t.key === "start") root.innerHTML = startHtml();
  else if (t.key === "ai") root.innerHTML = aiHtml();
  else if (t.key === "rollover") root.innerHTML = rolloverHtml();
  else if (t.key === "responsible") root.innerHTML = responsibleHtml();
  else {
    const raw = params.query.get("item"); // none (plain #/help/faq) opens no answer: Number(null) would be 0
    const item = raw === null || raw.trim() === "" ? NaN : Number(raw);
    const open = Number.isInteger(item) && item >= 0 && item < FAQ.length ? item : -1;
    root.innerHTML = faqHtml(open);
    if (open >= 0) requestAnimationFrame(() => root.querySelector(`#faq-${open}`)?.scrollIntoView({ block: "center" }));
  }
}

// --- the screen -----------------------------------------------------------------------------------------

export async function render(root, params) {
  ensureStyles();
  const key = params.topic;
  if (!key) {
    renderHome(root, params);
    return;
  }
  if (key === "glossary") {
    renderGlossary(root, params);
    return;
  }
  const guide = topic(key);
  if (guide) {
    renderTopic(root, guide, params);
    return;
  }
  const entry = term(key);
  if (entry) {
    renderTerm(root, entry);
    return;
  }
  root.innerHTML = `${notice("There's no help page by that name.", "info")}${emptyState({
    iconName: "help", title: "Not found", body: "Search the help instead, or start with the 60-second tour.",
    actions: [{ label: "All help", href: "#/help", primary: true }, { label: "Start here", href: "#/help/start" }],
  })}`;
}
