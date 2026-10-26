// Today: sport chips (favourites first, "+" to change them), then every game with odds, soonest first.
// Each card shows the win chances (the AI's for the NBA, the market's elsewhere), price buttons that
// toggle slip picks, the AI analysts' chips and "Ask AI". Tapping the card opens the game.
//
// The odds feed's quota is precious (3 requests per league fetch, cached 2 hours on the server), so only
// the chosen sports are asked for: one sport at a time, or "All" favourites together. Only the latest
// chip tap's answer is drawn: a slow "All" can't land on top of the single sport tapped after it.

import { api, prefs, isQuiet, qs } from "../api.js";
import { me, favouriteSports, loadSports, leagueTitle, sportEmoji, openOnboarding, openSportsSheet, setQuota, rerender } from "../app.js";
import { esc, chip, gameCard, dayHeading, emptyState, errorState, skeleton, notice, explain, sheet } from "../ui.js";
import { icon } from "../icons.js";

const favourites = () => favouriteSports();

function selected(favs) {
  const saved = prefs.get("todaySport", "all");
  if (saved === "all" && favs.length > 1) return "all";
  return favs.includes(saved) ? saved : favs[0];
}

function chipsHtml(favs, current) {
  return `<div class="chip-scroll" role="toolbar" aria-label="Sports">
    ${favs.length > 1 ? chip("All", { active: current === "all", attrs: 'data-sport="all"' }) : ""}
    ${favs.map((key) => chip(`${sportEmoji(key)} ${leagueTitle(key)}`, { active: current === key, attrs: `data-sport="${esc(key)}"` })).join("")}
    <button type="button" class="chip add" data-add-sport aria-label="Choose your sports">${icon("plus", { size: 16 })}Sports</button>
  </div>`;
}

function headHtml(data) {
  const asOf = data ? Object.entries(data.odds_as_of || {}).filter(([, v]) => v) : [];
  const when = asOf.length === 1 ? `Odds as of ${asOf[0][1]}` : asOf.length ? `Odds as of ${asOf.map(([k, v]) => `${leagueTitle(k)} ${v}`).join(", ")}` : "";
  const date = new Date().toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
  return `<div class="today-head">
    <div class="grow"><h1>Today</h1><p>${esc(date)}${when ? ` · ${esc(when)}` : ""}${when ? explain("quota", { label: "How fresh are the odds?" }) : ""}</p></div>
    <button type="button" class="icon-btn" data-search aria-label="Search players">${icon("search", { size: 20 })}</button>
  </div>`;
}

function gamesHtml(data, current) {
  const games = data.games || [];
  const errors = Object.entries(data.errors || {});
  const problems = errors.length
    ? `<div class="league-errors">${errors.map(([league, message]) => notice(`${leagueTitle(league)}: ${message}`, "info")).join("")}</div>`
    : "";
  if (!games.length) {
    const what = current === "all" ? "your sports" : leagueTitle(current);
    return `${problems}<div class="section">${emptyState({
      iconName: "today",
      title: `No ${what} games with odds right now`,
      body: "Bookmakers post lines a day or two ahead. Try another sport, or check back later.",
      actions: [{ label: "Choose sports", attrs: "data-add-sport", primary: true }, { label: "Let AI build a slip", href: "#/build" }],
    })}</div>`;
  }
  const dayOf = (card) => (card.live ? "Live now" : dayHeading(card.start));
  const counts = {};
  for (const card of games) counts[dayOf(card)] = (counts[dayOf(card)] || 0) + 1;
  let lastDay = "";
  const parts = [];
  for (const card of games) {
    const day = dayOf(card);
    if (day !== lastDay) {
      parts.push(`<h2 class="day-head">${esc(day)} · ${counts[day]} game${counts[day] === 1 ? "" : "s"}</h2>`);
      lastDay = day;
    }
    parts.push(gameCard(card, { showLeague: current === "all" }));
  }
  const modelled = games.some((g) => g.prediction);
  const legend = `<p class="note">${modelled ? `“AI” is the brain's chance ${explain("ai_chance")}; ` : ""}“Mkt” is the bookmakers' chance with their margin removed ${explain("market_chance")}. Tap a price to add it to your slip.</p>
    ${modelled ? `<p class="note">▲ next to a chance: the AI rates that pick above its price (an opinion, not a guarantee). “!”: the AI is far from the bookmakers, so check the injury news ${explain("value_mark", { label: "What do ▲ and ! mean?" })}</p>` : ""}`;
  return `${problems}<div class="stack">${parts.join("")}</div>${legend}`;
}

/**
 * Load and draw the games for ``current``. ``signal`` is this load's own: the screen aborts the previous
 * load when a chip is tapped (and every load when the person leaves), so only the latest one draws.
 */
async function loadGames(root, list, current, favs, signal) {
  list.innerHTML = skeleton("cards", 3);
  const leagues = current === "all" ? favs : [current];
  let data;
  try {
    data = await api(`/api/today${qs({ leagues })}`, { signal });
  } catch (error) {
    if (isQuiet(error) || signal.aborted || !root.isConnected) return;
    if (/18\+|age/i.test(error.message)) {
      list.innerHTML = emptyState({
        iconName: "shield", title: "Confirm your age first",
        body: "Games, prices and picks are for adults: confirm you're 18+ (21+ in the US) to continue.",
        actions: [{ label: "Confirm age", attrs: "data-onboard", primary: true }],
      });
    } else if (/break/i.test(error.message)) {
      list.innerHTML = emptyState({ iconName: "pause", title: "You're on a break", body: error.message,
        actions: [{ label: "Responsible play", href: "#/help/responsible" }] });
    } else {
      list.innerHTML = errorState(error);
    }
    return;
  }
  if (signal.aborted || !root.isConnected) return;
  setQuota(data.quota);
  root.querySelector("[data-head]").innerHTML = headHtml(data);
  list.innerHTML = gamesHtml(data, current);
}

/** Player search in a sheet (ESPN's search, any league; results open the player page). */
function openSearch() {
  const s = sheet.open({
    title: "Find a player",
    size: "tall",
    body: `<div class="search-row"><label class="sr-only" for="player-q">Player name</label>
        <input id="player-q" class="input" type="search" placeholder="e.g. LeBron James" autocomplete="off" enterkeyhint="search"></div>
      <ul class="list search-results" data-results></ul>`,
  });
  const input = s.body.querySelector("#player-q");
  const results = s.body.querySelector("[data-results]");
  let timer = null;
  let controller = null;
  setTimeout(() => input.focus(), 50);
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) {
      results.innerHTML = "";
      return;
    }
    timer = setTimeout(async () => {
      controller?.abort();
      controller = new AbortController();
      results.innerHTML = `<li>${skeleton("lines", 3)}</li>`;
      try {
        const found = await api(`/api/players/search${qs({ q })}`, { signal: controller.signal });
        results.innerHTML = found.length ? found.map((p) => `<li><a class="list-item" href="#/player/${esc(p.league)}/${esc(p.id)}">
            ${p.headshot ? `<img class="headshot" style="width:40px;height:40px" src="${esc(p.headshot)}" alt="" loading="lazy">` : `<span class="headshot blank" style="width:40px;height:40px">${icon("user", { size: 18 })}</span>`}
            <span class="grow"><b>${esc(p.name)}</b><span>${esc([p.team, p.position, leagueTitle(p.league)].filter(Boolean).join(" · "))}</span></span>${icon("chevron", { size: 18 })}</a></li>`).join("")
          : `<li class="muted small">No players match “${esc(q)}”.</li>`;
      } catch (error) {
        if (!isQuiet(error)) results.innerHTML = `<li>${notice(error.message, "warn")}</li>`;
      }
    }, 300);
  });
}

export async function render(root, params) {
  const favs = favourites();
  let current = selected(favs);
  loadSports().then(() => {
    // Titles arrive with /api/sports: redraw the chips once they have
    const bar = root.querySelector("[data-chips]");
    if (bar && root.isConnected) bar.innerHTML = chipsHtml(favs, current);
  }).catch(() => null);
  root.innerHTML = `<div data-head>${headHtml(null)}</div>
    <div data-chips>${chipsHtml(favs, current)}</div>
    <div data-list class="section"></div>`;
  const list = root.querySelector("[data-list]");
  // One load at a time: a new chip tap aborts the load before it, and leaving the screen aborts them all
  let loading = null;
  const load = () => {
    loading?.abort();
    const controller = new AbortController();
    loading = controller;
    const stop = () => controller.abort();
    params.signal.addEventListener("abort", stop, { once: true });
    return loadGames(root, list, current, favs, controller.signal)
      .finally(() => params.signal.removeEventListener("abort", stop));
  };
  root.addEventListener("click", (event) => {
    const sport = event.target.closest("[data-sport]");
    if (sport) {
      current = sport.dataset.sport;
      prefs.set("todaySport", current);
      root.querySelector("[data-chips]").innerHTML = chipsHtml(favs, current);
      load();
      return;
    }
    if (event.target.closest("[data-add-sport]")) openSportsSheet({ onDone: () => rerender() });
    else if (event.target.closest("[data-search]")) openSearch();
    else if (event.target.closest("[data-onboard]")) openOnboarding();
  });
  if (!me().age_confirmed && me().uid) {
    list.innerHTML = emptyState({
      iconName: "shield", title: "Confirm your age first",
      body: "Games, prices and picks are for adults: confirm you're 18+ (21+ in the US) to continue.",
      actions: [{ label: "Confirm age", attrs: "data-onboard", primary: true }],
    });
    return;
  }
  await load();
}
